"""
Patch Embedding Module for Vision Transformer.

This is the FIRST step in the ViT pipeline. It takes a raw image
and converts it into a sequence of token embeddings that the
transformer can process — exactly like how a language model
converts words into word embeddings.

The process:
  Image [B, 3, 32, 32]
    → Split into patches and project each one     → [B, 64, 128]
    → Prepend a learnable [CLS] token              → [B, 65, 128]
    → Add positional embeddings                    → [B, 65, 128]
    → Apply dropout                                → [B, 65, 128]

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

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import ViTConfig


class PatchEmbedding(nn.Module):
    """
    Converts images into a sequence of patch embeddings with
    [CLS] token and positional encoding.
    """
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.d_model = config.d_model
        self.num_patches = config.num_patches
        
        # ─────────────────────────────────────────────
        # 1. PATCH PROJECTION
        # ─────────────────────────────────────────────
        # This Conv2d splits the image into patches AND projects them
        # in a single operation.
        #
        # - in_channels=3: RGB input
        # - out_channels=d_model: each patch becomes a d_model-dimensional vector
        # - kernel_size=patch_size: the "filter" is exactly one patch in size
        # - stride=patch_size: no overlap between patches
        #
        # For a 32×32 image with patch_size=4:
        #   Input:  [B, 3, 32, 32]
        #   Output: [B, 128, 8, 8]  (128 channels, 8×8 grid of patches)
        #
        self.projection = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # ─────────────────────────────────────────────
        # 2. [CLS] TOKEN
        # ─────────────────────────────────────────────
        # A learnable vector that gets prepended to the patch sequence.
        # After passing through all transformer layers, this token has
        # "attended to" all patches and aggregates global image information.
        # We use it for the final classification.
        #
        # Shape: [1, 1, d_model] — the 1s are for broadcasting over batch
        # and sequence dimensions.
        #
        # WHY a special token instead of just averaging all patches?
        # The [CLS] token is free to learn its own representation that's
        # optimized for classification. Averaging forces all patch tokens
        # to carry classification-relevant info, which can hurt their
        # ability to represent local features. (We'll compare both in ablations.)
        #
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        
        # ─────────────────────────────────────────────
        # 3. POSITIONAL EMBEDDINGS
        # ─────────────────────────────────────────────
        # Without these, the transformer has NO idea where each patch
        # came from in the image. Self-attention is permutation equivariant
        # (we'll prove this in the report), meaning if you shuffle the
        # patches, the output gets shuffled the same way.
        #
        # We use LEARNED positional embeddings (not sinusoidal).
        # The ViT paper found learned embeddings work just as well,
        # and they're simpler to implement.
        #
        # Shape: [1, num_patches + 1, d_model]
        # The +1 is for the [CLS] token's position.
        #
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.d_model)
        )
        
        # ─────────────────────────────────────────────
        # 4. DROPOUT
        # ─────────────────────────────────────────────
        # Applied after adding positional embeddings.
        # Randomly zeros out some elements during training to prevent
        # overfitting (the model can't rely on any single feature).
        #
        self.dropout = nn.Dropout(config.dropout)
        
        # ─────────────────────────────────────────────
        # INITIALIZE WEIGHTS
        # ─────────────────────────────────────────────
        # CLS token: initialize from a normal distribution with small std
        # Positional embeddings: same initialization
        # These initializations come from the original ViT paper.
        #
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image → patch embeddings with [CLS] and positions.
        
        Args:
            x: Input images, shape [B, C, H, W]
               e.g., [4, 3, 32, 32] for a batch of 4 CIFAR-10 images
        
        Returns:
            Patch embeddings, shape [B, num_patches + 1, d_model]
            e.g., [4, 65, 128]
        """
        B = x.shape[0]  # batch size
        
        # STEP 1: Project patches
        # [B, 3, 32, 32] → [B, 128, 8, 8]
        x = self.projection(x)
        
        # STEP 2: Flatten spatial dimensions and transpose
        # [B, 128, 8, 8] → [B, 128, 64] → [B, 64, 128]
        # flatten(2) merges the last two dims: 8×8 → 64
        # transpose swaps dims 1 and 2: [B, d_model, N] → [B, N, d_model]
        x = x.flatten(2).transpose(1, 2)
        
        # STEP 3: Prepend [CLS] token
        # cls_token is [1, 1, 128] — expand to [B, 1, 128] for the batch
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # Concatenate: [B, 1, 128] + [B, 64, 128] → [B, 65, 128]
        x = torch.cat([cls_tokens, x], dim=1)
        
        # STEP 4: Add positional embeddings
        # pos_embed is [1, 65, 128] — broadcasts over batch dimension
        # This is element-wise addition: each token gets position info
        x = x + self.pos_embed
        
        # STEP 5: Dropout
        x = self.dropout(x)
        
        return x


# ──────────────────────────────────────────────────────
# TEST — Run this file directly to verify it works
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    config = ViTConfig()
    
    # Create the module
    patch_embed = PatchEmbedding(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in patch_embed.parameters())
    print(f"PatchEmbedding parameters: {total_params:,}")
    print(f"  - Projection (Conv2d): {sum(p.numel() for p in patch_embed.projection.parameters()):,}")
    print(f"  - CLS token: {patch_embed.cls_token.numel():,}")
    print(f"  - Positional embedding: {patch_embed.pos_embed.numel():,}")
    
    # Create a dummy batch of images
    # 4 images, 3 channels (RGB), 32×32 pixels
    dummy_images = torch.randn(4, 3, 32, 32)
    print(f"\nInput shape:  {dummy_images.shape}")
    
    # Forward pass
    output = patch_embed(dummy_images)
    print(f"Output shape: {output.shape}")
    
    # Verify shape
    expected_shape = (4, config.seq_length, config.d_model)
    assert output.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    
    # Verify [CLS] token is at position 0
    # After expansion, all items in the batch should have the same CLS token
    # (before pos_embed is added — but after, they're still the same since
    # pos_embed is the same for all items)
    print(f"\n[CLS] token position: index 0")
    print(f"Patch tokens: indices 1 to {config.num_patches}")
    
    # Verify positional embeddings are added (not zeros)
    # If we run with no dropout, the output should differ from just projection + CLS
    patch_embed_no_drop = PatchEmbedding(config)
    patch_embed_no_drop.dropout = nn.Dropout(0.0)  # disable dropout for this test
    out1 = patch_embed_no_drop(dummy_images)
    
    # Manually check: are positional embeddings actually doing something?
    # Zero out pos_embed and compare
    with torch.no_grad():
        patch_embed_no_drop.pos_embed.zero_()
    out2 = patch_embed_no_drop(dummy_images)
    
    pos_embed_makes_difference = not torch.allclose(out1, out2)
    print(f"\nPositional embeddings affect output: {pos_embed_makes_difference}")
    assert pos_embed_makes_difference, "Positional embeddings should change the output!"
    
    print("\n All PatchEmbedding tests PASSED!")


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE PATCH EMBEDDING
# Explicit constructor args — no ViTConfig dependency.
# Designed for 256×256 inputs (OSCD / Siamese ViT change detection).
# ──────────────────────────────────────────────────────────────────────────────

class StandalonePatchEmbedding(nn.Module):
    """
    Patch Embedding with explicit constructor arguments.

    Converts (B, C, H, W) images into (B, num_patches + 1, embed_dim) token
    sequences ready for a Transformer encoder.

    Args:
        img_size   : Input image size (square assumed). Default 256.
        patch_size : Side length of each patch.        Default 16.
        in_channels: Number of input channels.         Default 3 (RGB).
        embed_dim  : Token embedding dimension.        Default 768 (ViT-Base).
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        )
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2   # 256 for 256/16

        # ── Patch projection ──────────────────────────────────────────────────
        # Conv2d(kernel=P, stride=P) ≡ flatten each P×P patch + linear project.
        # Input : (B, C,          H,   W  )
        # Output: (B, embed_dim,  H/P, W/P)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # ── Learnable [CLS] token ─────────────────────────────────────────────
        # Shape (1, 1, embed_dim) — broadcast over batch at forward time.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Learnable positional embeddings ───────────────────────────────────
        # One vector per position: num_patches patch tokens + 1 CLS token.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        # Conv2d weight treated as a linear projection matrix (xavier uniform).
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))
        nn.init.zeros_(self.proj.bias)
        # Small normal init for learned tokens — follows the ViT paper (std=0.02).
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — e.g. (B, 3, 256, 256)

        Returns:
            (B, num_patches + 1, embed_dim) — e.g. (B, 257, 768)

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
        """
        B = x.shape[0]

        # 1. Project patches: (B, C, H, W) → (B, embed_dim, H/P, W/P)
        x = self.proj(x)

        # 2. Flatten spatial grid and move embed dim last:
        #    (B, embed_dim, H/P, W/P) → (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        # 3. Prepend [CLS] token: (B, 1, embed_dim) cat (B, N, embed_dim)
        #    → (B, N+1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 4. Add positional embeddings (broadcast over batch)
        x = x + self.pos_embed

        return x


# ──────────────────────────────────────────────────────
# TEST — python models/patch_embedding.py
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("StandalonePatchEmbedding  (256×256, patch=16, d=768)")
    print("=" * 52)

    pe = StandalonePatchEmbedding(img_size=256, patch_size=16, in_channels=3, embed_dim=768)

    proj_params = sum(p.numel() for p in pe.proj.parameters())
    print(f"Parameters:")
    print(f"  proj (Conv2d)    : {proj_params:>10,}")
    print(f"  cls_token        : {pe.cls_token.numel():>10,}")
    print(f"  pos_embed        : {pe.pos_embed.numel():>10,}")
    print(f"  Total            : {sum(p.numel() for p in pe.parameters()):>10,}")
    print(f"\nnum_patches = {pe.num_patches}  "
          f"(({pe.img_size}/{pe.patch_size})² = {pe.num_patches})")

    x = torch.randn(2, 3, 256, 256)
    print(f"\nInput  shape : {tuple(x.shape)}")
    out = pe(x)
    print(f"Output shape : {tuple(out.shape)}")

    assert out.shape == (2, 257, 768), f"Shape mismatch: {out.shape}"
    print("\nAll StandalonePatchEmbedding tests PASSED!")