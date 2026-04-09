"""
Patch Embedding Module for Vision Transformer.

This is the FIRST step in the ViT pipeline. It takes a raw image
and converts it into a sequence of token embeddings that the
transformer can process — exactly like how a language model
converts words into word embeddings.

The process:
  Image [B, 3, H, W]
    → Split into patches and project each one     → [B, N, D]
    → Prepend a learnable [CLS] token              → [B, N+1, D]
    → Add positional embeddings                    → [B, N+1, D]
    → Apply dropout                                → [B, N+1, D]

MATHEMATICAL INSIGHT — Patch Projection ≡ Convolution:
    Splitting an image into P×P patches, flattening each, and multiplying
    by a weight matrix W ∈ ℝ^(P²C × D) is IDENTICAL to running a Conv2d
    with kernel_size=P and stride=P. Here's why:

    Conv2d slides a filter of size P×P across the image with stride P
    (no overlap). At each position, it computes a dot product between
    the filter and the P×P×C patch — which is exactly what flattening
    the patch and multiplying by W does. The conv filter weights ARE
    the rows of the projection matrix, just reshaped.

    We use Conv2d in the implementation because it's cleaner and faster,
    but mathematically they're the same operation.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts images into a sequence of patch embeddings with
    [CLS] token and positional encoding.

    Args:
        img_size   : Input image size (square assumed). Default 256.
        patch_size : Side length of each patch.        Default 16.
        in_channels: Number of input channels.         Default 3 (RGB).
        embed_dim  : Token embedding dimension.        Default 768 (ViT-Base).
        dropout    : Dropout applied after positional embedding. Default 0.0.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        )
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # ── Patch projection ──────────────────────────────────────────────
        # Conv2d(kernel=P, stride=P) ≡ flatten each P×P patch + linear project.
        # Input : (B, C,         H,   W  )
        # Output: (B, embed_dim, H/P, W/P)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # ── [CLS] token ───────────────────────────────────────────────────
        # Shape (1, 1, embed_dim) — broadcast over batch at forward time.
        # After all transformer layers it aggregates global image information
        # and is used for the final classification / change prediction.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Positional embeddings ─────────────────────────────────────────
        # Without these the transformer has no notion of where each patch is.
        # Self-attention is permutation-equivariant — shuffling patches
        # would produce the same output (shuffled). Positional embeddings
        # break this symmetry.
        # Shape: (1, num_patches + 1, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        # Conv2d weight treated as a linear projection matrix (xavier uniform).
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))
        nn.init.zeros_(self.proj.bias)
        # Small normal init for learned tokens — follows the ViT paper (std=0.02).
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)  — e.g. (B, 3, 256, 256)

        Returns:
            (B, num_patches + 1, embed_dim)  — e.g. (B, 257, 768)

        Step-by-step for B=2, img=256, patch=16, embed=768:
            (B, 3,   256, 256)
          → proj →
            (B, 768, 16,  16)    # 256/16 = 16 positions per side
          → flatten(2) →
            (B, 768, 256)        # 16×16 = 256 patches
          → transpose(1,2) →
            (B, 256, 768)        # sequence-first layout
          → prepend CLS →
            (B, 257, 768)
          → + pos_embed →
            (B, 257, 768)        # each token gets its position signal
          → dropout →
            (B, 257, 768)
        """
        B = x.shape[0]

        # 1. Project: (B, C, H, W) → (B, D, H/P, W/P)
        x = self.proj(x)

        # 2. Flatten spatial dims, move embed last: → (B, N, D)
        x = x.flatten(2).transpose(1, 2)

        # 3. Prepend [CLS] token: (B, 1, D) cat (B, N, D) → (B, N+1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 4. Add positional embeddings (broadcast over batch)
        x = x + self.pos_embed

        # 5. Dropout
        x = self.dropout(x)

        return x


# ──────────────────────────────────────────────────────
# TESTS — python models/patch_embedding.py
# ──────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 56)
    print("PatchEmbedding  (img=256, patch=16, embed=768)")
    print("=" * 56)

    pe = PatchEmbedding(img_size=256, patch_size=16, in_channels=3, embed_dim=768)

    total = sum(p.numel() for p in pe.parameters())
    print("Parameters:")
    print(f"  proj (Conv2d)  : {sum(p.numel() for p in pe.proj.parameters()):>10,}")
    print(f"  cls_token      : {pe.cls_token.numel():>10,}")
    print(f"  pos_embed      : {pe.pos_embed.numel():>10,}")
    print(f"  Total          : {total:>10,}")
    print(f"\nnum_patches = {pe.num_patches}  (({pe.img_size}/{pe.patch_size})²)")

    # ── Shape test ───────────────────────────────────────────────────────
    print("\n── Shape test ──")
    x   = torch.randn(2, 3, 256, 256)
    out = pe(x)
    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(out.shape)}")
    assert out.shape == (2, 257, 768), f"Shape wrong: {out.shape}"

    # ── Positional embeddings are active ─────────────────────────────────
    pe_no_drop = PatchEmbedding(img_size=256, patch_size=16, embed_dim=768)
    out1 = pe_no_drop(x)
    with torch.no_grad():
        pe_no_drop.pos_embed.zero_()
    out2 = pe_no_drop(x)
    assert not torch.allclose(out1, out2), "pos_embed should affect output"
    print("Positional embeddings affect output: True")

    # ── Gradient flow ────────────────────────────────────────────────────
    print("\n── Gradient flow ──")
    x_g = torch.randn(2, 3, 256, 256, requires_grad=True)
    pe(x_g).sum().backward()
    assert x_g.grad is not None and not torch.isnan(x_g.grad).any()
    all_ok = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in pe.parameters()
    )
    print(f"  Input grad clean : True")
    print(f"  Param grads clean: {all_ok}")

    print("\nAll PatchEmbedding tests PASSED!")
