"""
Siamese ViT Change Detection Model — Full Assembly.

Brings together every component built from scratch:

    SiameseViTEncoder           (shared ViTEncoder backbone)
        └─ ViTEncoder × 2       (same weights, two forward passes)
               └─ PatchEmbedding
               └─ TransformerEncoderBlock × depth
               └─ LayerNorm
    FeatureDifferenceModule     (combines feat1 / feat2 into a change signal)
    ProgressiveDecoder          (16×16 tokens → 256×256 pixel mask)

Full pipeline:

    img1 (B, 3, 256, 256) ──┐
                              ├─→ SiameseViTEncoder ─→ feat1 (B, 256, 768)
    img2 (B, 3, 256, 256) ──┘                      └─→ feat2 (B, 256, 768)
                                                              │
                                                  FeatureDifferenceModule
                                                              │
                                                    diff_feat (B, 256, 256)
                                                              │
                                                   ProgressiveDecoder
                                                              │
                                               change_logits (B, 1, 256, 256)

No pretrained weights are used anywhere. The entire model — patch projection,
positional embeddings, 12 transformer blocks, difference module, and decoder —
is trained from random initialisation on the OSCD dataset.
"""

import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit import SiameseViTEncoder
from models.feature_difference import FeatureDifferenceModule
from models.decoder import ProgressiveDecoder


class SiameseViTChangeDetection(nn.Module):
    """
    Complete Siamese ViT Change Detection model built from scratch.

    Args:
        img_size     : Input image resolution (square).         Default 256.
        patch_size   : ViT patch size.                          Default 16.
        in_channels  : Input channels.                          Default 3.
        embed_dim    : ViT token embedding dimension.           Default 768.
        depth        : Number of Transformer blocks.            Default 12.
        num_heads    : Attention heads per block.               Default 12.
        mlp_ratio    : MLP expansion ratio.                     Default 4.0.
        dropout      : Dropout in encoder and decoder.          Default 0.1.
        attn_dropout : Dropout inside attention weights.        Default 0.0.
        diff_type    : FeatureDifferenceModule strategy.
                       'subtract' | 'concat_project' | 'attention'.
        decoder_dims : Channel counts for each decoder stage.
        diff_out_dim : Output dim of the difference module (= decoder input dim).
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        diff_type: str = "concat_project",
        decoder_dims: list[int] = None,
        diff_out_dim: int = 256,
    ) -> None:
        super().__init__()
        if decoder_dims is None:
            decoder_dims = [128, 64, 32, 16]

        self.img_size   = img_size
        self.patch_size = patch_size

        # ── 1. Siamese encoder (single shared ViTEncoder) ─────────────────
        self.encoder = SiameseViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

        # ── 2. Feature difference module ──────────────────────────────────
        self.diff_module = FeatureDifferenceModule(
            in_dim=embed_dim,
            out_dim=diff_out_dim,
            diff_type=diff_type,
        )

        # ── 3. Progressive decoder ────────────────────────────────────────
        self.decoder = ProgressiveDecoder(
            in_dim=diff_out_dim,
            hidden_dims=decoder_dims,
            num_patches_side=img_size // patch_size,
        )

    # ------------------------------------------------------------------
    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            img1: (B, C, H, W) — before image
            img2: (B, C, H, W) — after  image

        Returns:
            (B, 1, H, W) — change logits (apply sigmoid for probabilities,
                           or pass directly to BCEWithLogitsLoss)
        """
        feat1, feat2     = self.encoder(img1, img2)   # (B, N, embed_dim) each
        diff_feat        = self.diff_module(feat1, feat2)  # (B, N, diff_out_dim)
        change_logits    = self.decoder(diff_feat)    # (B, 1, H, W)
        return change_logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self, img1: torch.Tensor, img2: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Return binary change mask (0/1 float) after sigmoid + threshold.

        Args:
            img1, img2  : Input image pair.
            threshold   : Probability cutoff.  Default 0.5.

        Returns:
            (B, 1, H, W) float tensor of 0s and 1s.
        """
        probs = torch.sigmoid(self.forward(img1, img2))
        return (probs > threshold).float()

    # ------------------------------------------------------------------
    def get_param_count(self) -> dict[str, int]:
        """Parameter counts broken down by component."""
        enc_p  = sum(p.numel() for p in self.encoder.parameters())
        diff_p = sum(p.numel() for p in self.diff_module.parameters())
        dec_p  = sum(p.numel() for p in self.decoder.parameters())
        return {
            "encoder"     : enc_p,
            "diff_module" : diff_p,
            "decoder"     : dec_p,
            "total"       : enc_p + diff_p + dec_p,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_siamese_vit_cd(config: dict | None = None) -> SiameseViTChangeDetection:
    """Build the model from a config dict (or defaults).

    Default config = ViT-Base equivalent encoder (~86M params, ~89M total).

    Example — lighter model for quick experiments:
        model = build_siamese_vit_cd({
            'depth': 6, 'embed_dim': 384, 'num_heads': 6,
        })
    """
    default = dict(
        img_size=256, patch_size=16, in_channels=3,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        dropout=0.1, attn_dropout=0.0,
        diff_type="concat_project",
        decoder_dims=[128, 64, 32, 16],
        diff_out_dim=256,
    )
    if config:
        default.update(config)
    return SiameseViTChangeDetection(**default)


# ──────────────────────────────────────────────────────
# TESTS — python models/siamese_vit.py
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Build model ───────────────────────────────────────────────────────
    model = build_siamese_vit_cd()
    model = model.to(device)

    # ── Parameter breakdown ───────────────────────────────────────────────
    counts = model.get_param_count()
    total  = counts["total"]
    print("=" * 56)
    print("SiameseViTChangeDetection  parameter breakdown")
    print("=" * 56)
    for name, n in counts.items():
        pct = f"({100 * n / total:.1f}%)" if name != "total" else ""
        print(f"  {name:<14} : {n:>12,}  {pct}")

    # ── Forward pass ──────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("Forward pass  (B=2, 256×256)")
    print("=" * 56)

    img1 = torch.randn(2, 3, 256, 256, device=device)
    img2 = torch.randn(2, 3, 256, 256, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(img1, img2)
        probs  = torch.sigmoid(logits)
        mask   = model.predict(img1, img2)

    print(f"  img1          : {tuple(img1.shape)}")
    print(f"  img2          : {tuple(img2.shape)}")
    print(f"  logits        : {tuple(logits.shape)}")
    print(f"  probs range   : [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"  binary mask   : {tuple(mask.shape)}  values={mask.unique().tolist()}")

    assert logits.shape == (2, 1, 256, 256), f"Logits shape wrong: {logits.shape}"
    assert mask.shape   == (2, 1, 256, 256)
    assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    # ── Gradient flow ─────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("Gradient flow")
    print("=" * 56)

    model.train()
    model.zero_grad()
    logits = model(img1, img2)
    target = torch.zeros_like(logits)
    loss   = nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    print(f"  BCEWithLogitsLoss : {loss.item():.4f}")
    all_ok = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in model.parameters()
    )
    print(f"  All param grads   : {'valid' if all_ok else 'FAILED'}")

    print("\n All SiameseViTChangeDetection tests PASSED!")
