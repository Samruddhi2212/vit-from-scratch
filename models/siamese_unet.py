"""
Siamese U-Net (FC-Siam-diff) for Change Detection — built from scratch.

Implements the FC-Siam-diff architecture from Daudt et al. (2018),
"Fully Convolutional Siamese Networks for Change Detection."

Architecture overview:

    img1 (B, 3, 256, 256) ──┐
                              ├─→ Shared Encoder ─→ feats1 at 4 scales + bottleneck
    img2 (B, 3, 256, 256) ──┘                  └─→ feats2 at 4 scales + bottleneck

    Bottleneck:  |bottleneck1 - bottleneck2|

    Decoder:     4 upsample stages, each receiving |skip1 - skip2| via concat
                                    ↓
                         logits (B, 1, 256, 256)

The absolute difference at each skip connection level provides an explicit
inductive bias for change detection: regions where features differ across
time produce large magnitudes, directly signalling change.

No pretrained weights are used. The entire model is trained from random
initialisation, matching the from-scratch Siamese ViT for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Double convolution: Conv3x3-BN-ReLU × 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SiameseUNet(nn.Module):
    """
    FC-Siam-diff: shared-weight Siamese encoder with absolute feature
    differencing at skip connections and a symmetric U-Net decoder.

    Args:
        in_channels:  Input channels per image (3 for RGB).
        out_channels: Output channels (1 for binary change map).
        features:     Channel widths at each encoder level.
                      Default [64, 128, 256, 512] matches the standard
                      U-Net configuration from Ronneberger et al. (2015).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: list[int] | None = None,
    ) -> None:
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.features = list(features)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Shared encoder (called once per image) ────────────────────────
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in self.features:
            self.encoders.append(_ConvBlock(ch, f))
            ch = f

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = _ConvBlock(self.features[-1], self.features[-1] * 2)

        # ── Decoder ───────────────────────────────────────────────────────
        # Each level: upsample + concat(upsampled, |skip1-skip2|) + conv block
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        decoder_in = self.features[-1] * 2  # bottleneck output channels
        for f in reversed(self.features):
            self.upconvs.append(
                nn.ConvTranspose2d(decoder_in, f, kernel_size=2, stride=2)
            )
            # After concat: f (from upconv) + f (from |skip1 - skip2|)
            self.dec_blocks.append(_ConvBlock(f + f, f))
            decoder_in = f

        # ── Output head ───────────────────────────────────────────────────
        self.head = nn.Conv2d(self.features[0], out_channels, kernel_size=1)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run one image through the shared encoder.

        Returns:
            bottleneck: (B, features[-1]*2, H/16, W/16) — deepest features.
            skips:      list of encoder outputs at each level before pooling,
                        ordered shallow-to-deep.
        """
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        return x, skips

    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            img1: (B, C, H, W) — before image.
            img2: (B, C, H, W) — after  image.

        Returns:
            (B, 1, H, W) raw logits (no sigmoid).
            Pass directly to BCEWithLogitsLoss or BCEDiceLoss.
        """
        b1, skips1 = self._encode(img1)
        b2, skips2 = self._encode(img2)

        # Bottleneck difference
        x = torch.abs(b1 - b2)

        # Decoder with skip-connection differencing
        for upconv, dec_block, s1, s2 in zip(
            self.upconvs,
            self.dec_blocks,
            reversed(skips1),
            reversed(skips2),
        ):
            x = upconv(x)

            # Handle spatial size mismatch from non-power-of-2 inputs
            if x.shape[2:] != s1.shape[2:]:
                x = F.interpolate(
                    x, size=s1.shape[2:], mode="bilinear", align_corners=False
                )

            diff = torch.abs(s1 - s2)
            x = torch.cat([x, diff], dim=1)
            x = dec_block(x)

        return self.head(x)

    def get_param_count(self) -> dict[str, int]:
        """Parameter counts broken down by component.

        Matches the interface of SiameseViTChangeDetection.get_param_count()
        so train.py can log counts uniformly.
        """
        enc_p = sum(p.numel() for p in self.encoders.parameters())
        enc_p += sum(p.numel() for p in self.bottleneck.parameters())
        dec_p = sum(p.numel() for p in self.upconvs.parameters())
        dec_p += sum(p.numel() for p in self.dec_blocks.parameters())
        head_p = sum(p.numel() for p in self.head.parameters())
        return {
            "encoder": enc_p,
            "diff_module": 0,  # differencing is parameter-free (|f1-f2|)
            "decoder": dec_p + head_p,
            "total": enc_p + dec_p + head_p,
        }


# ──────────────────────────────────────────────────────
# TESTS — python models/siamese_unet.py
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    model = SiameseUNet(in_channels=3, out_channels=1).to(device)

    counts = model.get_param_count()
    total = counts["total"]
    print("=" * 56)
    print("SiameseUNet (FC-Siam-diff)  parameter breakdown")
    print("=" * 56)
    for name, n in counts.items():
        pct = f"({100 * n / total:.1f}%)" if name != "total" else ""
        print(f"  {name:<14} : {n:>12,}  {pct}")

    print("\n" + "=" * 56)
    print("Forward pass  (B=2, 256×256)")
    print("=" * 56)

    img1 = torch.randn(2, 3, 256, 256, device=device)
    img2 = torch.randn(2, 3, 256, 256, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(img1, img2)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float()

    print(f"  img1          : {tuple(img1.shape)}")
    print(f"  img2          : {tuple(img2.shape)}")
    print(f"  logits        : {tuple(logits.shape)}")
    print(f"  probs range   : [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"  binary mask   : {tuple(mask.shape)}  values={mask.unique().tolist()}")

    assert logits.shape == (2, 1, 256, 256), f"Logits shape wrong: {logits.shape}"
    assert mask.shape == (2, 1, 256, 256)

    print("\n" + "=" * 56)
    print("Gradient flow")
    print("=" * 56)

    model.train()
    model.zero_grad()
    logits = model(img1, img2)
    target = torch.zeros_like(logits)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    print(f"  BCEWithLogitsLoss : {loss.item():.4f}")
    all_ok = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in model.parameters()
    )
    print(f"  All param grads   : {'valid' if all_ok else 'FAILED'}")

    print("\n All SiameseUNet tests PASSED!")
