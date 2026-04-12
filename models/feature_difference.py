"""
Feature Difference Module for Siamese ViT Change Detection.

After the Siamese encoder produces feat1 and feat2 (one per time step),
this module combines them into a single change-aware feature tensor.

Three strategies are provided:

  subtract        — |feat1 - feat2| → project
                    Simplest baseline. Captures magnitude of change but
                    loses the sign and individual context of each branch.

  concat_project  — [feat1, feat2, feat1-feat2, feat1*feat2] → MLP
                    Four complementary interactions:
                      feat1, feat2     : individual context (what was / what is)
                      feat1 - feat2    : signed change direction
                      feat1 * feat2    : element-wise similarity (high where unchanged)
                    Concatenating all four gives the MLP the richest possible
                    signal to learn change-sensitive representations from.

  attention       — feat1 cross-attends to feat2, then concat with |feat1-feat2|
                    Cross-attention lets each spatial position in feat1 look at
                    the most relevant positions in feat2 (not just the same location).
                    Useful for changes that involve spatial displacement or context.

All three output (B, N, out_dim) where N = num_patches.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class FeatureDifferenceModule(nn.Module):
    """
    Combines Siamese encoder features into change-aware representations.

    Args:
        in_dim   : Input feature dimension from ViTEncoder.    Default 768.
        out_dim  : Output channel dimension.                   Default 256.
        diff_type: Combination strategy.
                   'subtract'       — |f1-f2| → Linear → GELU → Linear
                   'concat_project' — [f1, f2, f1-f2, f1*f2] → 2-layer MLP
                   'attention'      — cross-attention + |f1-f2| → project
    """

    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 256,
        diff_type: str = "concat_project",
    ) -> None:
        super().__init__()
        self.diff_type = diff_type

        if diff_type == "subtract":
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )

        elif diff_type == "concat_project":
            # [feat1 | feat2 | feat1-feat2 | feat1*feat2]  →  in_dim*4
            self.proj = nn.Sequential(
                nn.Linear(in_dim * 4, in_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )

        elif diff_type == "attention":
            # feat1 queries feat2 to find spatially-relevant context
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=in_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(in_dim)
            # concat(cross_attn_out, |feat1-feat2|)  →  in_dim*2
            self.proj = nn.Sequential(
                nn.Linear(in_dim * 2, out_dim),
                nn.GELU(),
            )

        else:
            raise ValueError(
                f"Unknown diff_type '{diff_type}'. "
                "Choose 'subtract', 'concat_project', or 'attention'."
            )

    def forward(
        self, feat1: torch.Tensor, feat2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feat1: (B, N, in_dim) — features from the before image
            feat2: (B, N, in_dim) — features from the after  image

        Returns:
            (B, N, out_dim) — change-aware patch features
        """
        if self.diff_type == "subtract":
            diff = torch.abs(feat1 - feat2)      # (B, N, D)
            return self.proj(diff)

        elif self.diff_type == "concat_project":
            combined = torch.cat(
                [feat1, feat2, feat1 - feat2, feat1 * feat2], dim=-1
            )                                    # (B, N, 4D)
            return self.proj(combined)

        elif self.diff_type == "attention":
            # Cross-attention: each position in feat1 attends over all of feat2
            attn_out, _ = self.cross_attn(feat1, feat2, feat2)  # (B, N, D)
            attn_out = self.norm(attn_out + feat1)               # residual + LN
            diff = torch.abs(feat1 - feat2)                      # (B, N, D)
            combined = torch.cat([attn_out, diff], dim=-1)       # (B, N, 2D)
            return self.proj(combined)


class MultiScaleDiffModule(nn.Module):
    """Multi-scale feature difference for Siamese ViT change detection.

    Applies a FeatureDifferenceModule at each of the 4 encoder scales
    (shallow → deep), then fuses all scale outputs into a single
    change-aware representation fed to the decoder.

    Why per-scale diffs?
      - Shallow scale  (depth/4)  : edges, colour shift, fine texture changes
      - Mid scales     (depth/2, 3*depth/4): structural / shape changes
      - Deep scale     (depth)    : semantic changes ("building appeared")
    Fusing all four lets the decoder use whichever cue is most reliable
    for each spatial location.

    Args:
        in_dim    : Embedding dimension from the ViT encoder.  Default 768.
        out_dim   : Output channel count (= decoder input dim). Default 256.
        n_scales  : Number of encoder scales to fuse.          Default 4.

    Input / output
    --------------
    forward(feats1_list, feats2_list)
        feats1_list, feats2_list : each a list of n_scales tensors,
                                   each (B, N, in_dim)
    returns : (B, N, out_dim)  — fused multi-scale change feature

    Shape trace (defaults, B=2, N=256):
        per scale : [f1, f2] → concat_project → (B, 256, out_dim)
        stack     : 4 × (B, 256, out_dim) → cat dim=-1 → (B, 256, 4*out_dim)
        fusion    : Linear(4*out_dim, 2*out_dim) → GELU → Linear(2*out_dim, out_dim)
                  → (B, 256, out_dim)
    """

    def __init__(
        self,
        in_dim:   int = 768,
        out_dim:  int = 256,
        n_scales: int = 4,
    ) -> None:
        super().__init__()
        self.n_scales = n_scales

        # One lightweight diff module per scale (concat_project strategy)
        self.scale_diffs = nn.ModuleList([
            FeatureDifferenceModule(in_dim, out_dim, diff_type="concat_project")
            for _ in range(n_scales)
        ])

        # Fuse the n_scales change features into a single representation
        self.fusion = nn.Sequential(
            nn.Linear(n_scales * out_dim, 2 * out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * out_dim, out_dim),
            nn.GELU(),
        )

    def forward(
        self,
        feats1_list: list[torch.Tensor],
        feats2_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            feats1_list : list of n_scales tensors (B, N, in_dim)
            feats2_list : list of n_scales tensors (B, N, in_dim)
        Returns:
            (B, N, out_dim)
        """
        scale_outs = [
            diff(f1, f2)
            for diff, f1, f2 in zip(self.scale_diffs, feats1_list, feats2_list)
        ]                                           # n_scales × (B, N, out_dim)
        combined = torch.cat(scale_outs, dim=-1)   # (B, N, n_scales * out_dim)
        return self.fusion(combined)               # (B, N, out_dim)


# ──────────────────────────────────────────────────────
# TESTS — python models/feature_difference.py
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    B, N, D, OUT = 2, 256, 768, 256

    feat1 = torch.randn(B, N, D)
    feat2 = torch.randn(B, N, D)

    for diff_type in ("subtract", "concat_project", "attention"):
        print("=" * 56)
        print(f"FeatureDifferenceModule  diff_type='{diff_type}'")
        print("=" * 56)

        m     = FeatureDifferenceModule(in_dim=D, out_dim=OUT, diff_type=diff_type)
        total = sum(p.numel() for p in m.parameters())
        print(f"Parameters : {total:,}")

        out = m(feat1, feat2)
        print(f"Input      : feat1={tuple(feat1.shape)}, feat2={tuple(feat2.shape)}")
        print(f"Output     : {tuple(out.shape)}")
        assert out.shape == (B, N, OUT), f"Shape wrong: {out.shape}"

        # Gradient flow
        m.zero_grad()
        feat1_g = feat1.detach().requires_grad_(True)
        feat2_g = feat2.detach().requires_grad_(True)
        m(feat1_g, feat2_g).sum().backward()
        assert feat1_g.grad is not None and feat2_g.grad is not None
        print(f"Grad flow  : OK")
        print()

    print("All FeatureDifferenceModule tests PASSED!")
