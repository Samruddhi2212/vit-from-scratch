"""
Progressive Upsampling Decoder for Siamese ViT Change Detection.

Converts the (B, N, C) patch-token features produced by FeatureDifferenceModule
back into a full-resolution (B, 1, H, W) change-probability map.

Pipeline:

    (B, 256, in_dim)  — N=256 patch tokens, one per 16×16 pixel region
          │
    Linear + GELU     — project each token to hidden_dims[0]*4 channels
          │
    Reshape           — (B, hidden*4, 16, 16)  ← treat patches as 2D spatial grid
          │
    Up1  ×2           — (B, 128, 32, 32)
          │
    Up2  ×2           — (B,  64, 64, 64)
          │
    Up3  ×2           — (B,  32, 128, 128)
          │
    Up4  ×2           — (B,  16, 256, 256)
          │
    Conv2d(1×1)       — (B,   1, 256, 256)  ← raw logits, apply sigmoid for prob

Each upsampling stage is:
    Upsample(×2, bilinear) → Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU

WHY bilinear upsample + conv instead of transposed conv?
  Transposed convolutions can produce "checkerboard" artefacts due to
  uneven overlap when stride > 1 (Odena et al., 2016). The bilinear +
  conv pattern avoids this: bilinear handles the spatial interpolation
  cleanly, then the conv learns the local refinement.

WHY two convs per stage?
  The first conv (after upsample) fuses the interpolated features.
  The second adds an extra non-linear capacity at the current resolution
  before the next upsampling step — important for sharp boundary recovery.
"""

import torch
import torch.nn as nn


class ProgressiveDecoder(nn.Module):
    """
    Progressive upsampling decoder: token features → full-resolution mask.

    Args:
        in_dim          : Feature dim per token from FeatureDifferenceModule. Default 256.
        hidden_dims     : Channel counts at each upsampling stage.
                          Length must equal log2(img_size / num_patches_side).
                          Default [128, 64, 32, 16] for 16→256 (4 doublings).
        num_patches_side: Spatial side length of the token grid.
                          = sqrt(num_patches) = img_size // patch_size.
                          Default 16  (for 256×256 image, patch_size=16).
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dims: list[int] = None,
        num_patches_side: int = 16,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32, 16]

        self.num_patches_side = num_patches_side

        # ── Initial projection ────────────────────────────────────────────
        # Lift each token to hidden_dims[0]*4 channels before spatial reshape.
        # The *4 gives the first upsample block richer per-pixel capacity.
        self.initial_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0] * 4),
            nn.GELU(),
        )

        # ── Upsampling stages ─────────────────────────────────────────────
        dims = [hidden_dims[0] * 4] + hidden_dims
        self.up_stages = nn.ModuleList([
            self._make_upsample_block(dims[i], dims[i + 1])
            for i in range(len(hidden_dims))
        ])

        # ── Output head ───────────────────────────────────────────────────
        # 1×1 conv collapses channels to a single logit per pixel.
        # No sigmoid here — use BCEWithLogitsLoss during training for
        # better numerical stability.
        self.final_conv = nn.Conv2d(hidden_dims[-1], 1, kernel_size=1)

    # ------------------------------------------------------------------
    @staticmethod
    def _make_upsample_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Bilinear upsample ×2 followed by two Conv-BN-ReLU layers."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim) — patch token features, N = num_patches_side²

        Returns:
            (B, 1, H, W) — change logits at full image resolution
                           Apply torch.sigmoid() to get probabilities.

        Shape trace (default config, B=2):
            (2, 256, 256)
          → initial_proj  → (2, 256, 512)
          → reshape        → (2, 512, 16, 16)
          → up1            → (2, 128, 32, 32)
          → up2            → (2,  64, 64, 64)
          → up3            → (2,  32, 128, 128)
          → up4            → (2,  16, 256, 256)
          → final_conv     → (2,   1, 256, 256)
        """
        B, N, D = x.shape
        P = self.num_patches_side

        # 1. Project tokens: (B, N, D) → (B, N, C)
        x = self.initial_proj(x)           # (B, N, hidden[0]*4)

        # 2. Reshape to 2D spatial grid: (B, N, C) → (B, C, P, P)
        x = x.transpose(1, 2).contiguous()  # (B, C, N)
        x = x.reshape(B, -1, P, P)          # (B, C, P, P)

        # 3. Progressive upsampling
        for stage in self.up_stages:
            x = stage(x)

        # 4. Output logits
        return self.final_conv(x)          # (B, 1, H, W)


# ──────────────────────────────────────────────────────
# TESTS — python models/decoder.py
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 56)
    print("ProgressiveDecoder  (in_dim=256, 16→256)")
    print("=" * 56)

    decoder = ProgressiveDecoder(
        in_dim=256,
        hidden_dims=[128, 64, 32, 16],
        num_patches_side=16,
    )

    # ── Parameter breakdown ───────────────────────────────────────────────
    proj_p  = sum(p.numel() for p in decoder.initial_proj.parameters())
    up_p    = sum(p.numel() for p in decoder.up_stages.parameters())
    head_p  = sum(p.numel() for p in decoder.final_conv.parameters())
    total   = sum(p.numel() for p in decoder.parameters())

    print(f"Parameters:")
    print(f"  initial_proj (Linear) : {proj_p:>10,}")
    for i, stage in enumerate(decoder.up_stages):
        sp = sum(p.numel() for p in stage.parameters())
        print(f"  up{i+1}                  : {sp:>10,}")
    print(f"  final_conv (1×1)      : {head_p:>10,}")
    print(f"  {'─'*35}")
    print(f"  Total                 : {total:>10,}")

    # ── Shape trace ───────────────────────────────────────────────────────
    print("\nShape trace (B=2):")
    B = 2
    x = torch.randn(B, 256, 256)

    print(f"  Input                 : {tuple(x.shape)}")

    # manual trace
    _x = decoder.initial_proj(x)
    print(f"  After initial_proj    : {tuple(_x.shape)}")
    _x = _x.transpose(1, 2).reshape(B, -1, 16, 16)
    print(f"  After reshape to 2D   : {tuple(_x.shape)}")
    stage_names = ["16→32", "32→64", "64→128", "128→256"]
    for name, stage in zip(stage_names, decoder.up_stages):
        _x = stage(_x)
        print(f"  After up ({name})   : {tuple(_x.shape)}")
    _x = decoder.final_conv(_x)
    print(f"  After final_conv      : {tuple(_x.shape)}")

    # ── Full forward pass ─────────────────────────────────────────────────
    print("\nFull forward pass:")
    x   = torch.randn(B, 256, 256)
    out = decoder(x)
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    assert out.shape == (B, 1, 256, 256), f"Shape wrong: {out.shape}"

    # ── Gradient flow ─────────────────────────────────────────────────────
    print("\nGradient flow:")
    x_g = torch.randn(B, 256, 256, requires_grad=True)
    decoder(x_g).sum().backward()
    assert x_g.grad is not None and not torch.isnan(x_g.grad).any()
    all_ok = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in decoder.parameters()
    )
    print(f"  Input grad clean  : True")
    print(f"  Param grads clean : {all_ok}")

    print("\nAll ProgressiveDecoder tests PASSED!")
