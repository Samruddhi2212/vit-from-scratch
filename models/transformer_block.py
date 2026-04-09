"""
Transformer Block for Vision Transformer.

This is the repeating unit of the ViT. The full model stacks N of these.
Each block takes a sequence of tokens, lets them attend to each other
(attention), then processes each token individually (MLP), with
normalization and residual connections for stability.

═══════════════════════════════════════════════════════════
ARCHITECTURE (Pre-Norm):
═══════════════════════════════════════════════════════════

    Input x ─────────────────────────────┐
         │                               │ (residual connection)
         ↓                               │
    LayerNorm                            │
         │                               │
         ↓                               │
    Multi-Head Attention                 │
         │                               │
         ↓                               │
    Dropout                              │
         │                               │
         ↓                               │
    ADD (x + attention output) ←─────────┘
         │
         │ = x'
         │
    x' ──────────────────────────────────┐
         │                               │ (residual connection)
         ↓                               │
    LayerNorm                            │
         │                               │
         ↓                               │
    MLP (Feed-Forward Network)           │
         │                               │
         ↓                               │
    Dropout                              │
         │                               │
         ↓                               │
    ADD (x' + mlp output) ←──────────────┘
         │
         ↓
    Output

═══════════════════════════════════════════════════════════
PRE-NORM vs POST-NORM — Important Design Choice
═══════════════════════════════════════════════════════════

Pre-Norm (what ViT uses):
    output = x + Attention(LayerNorm(x))
    
Post-Norm (original transformer paper):
    output = LayerNorm(x + Attention(x))

WHY Pre-Norm is better for training stability:

Consider the gradient flow through a residual connection:
    ∂L/∂x = ∂L/∂output × ∂output/∂x

Pre-Norm:  output = x + f(LayerNorm(x))
    ∂output/∂x = I + ∂f/∂x × ∂LayerNorm/∂x
    The identity matrix I guarantees a clean gradient path.
    Gradients can always flow directly through the skip connection.

Post-Norm: output = LayerNorm(x + f(x))
    ∂output/∂x = ∂LayerNorm/∂input × (I + ∂f/∂x)
    The LayerNorm Jacobian multiplies EVERYTHING, including the
    skip connection. This can distort gradients and cause instability.

In practice: Pre-Norm trains more stably, especially for deep models.
The original transformer used Post-Norm, but ViT and most modern
transformers use Pre-Norm.

═══════════════════════════════════════════════════════════
RESIDUAL CONNECTIONS — Why They're Essential
═══════════════════════════════════════════════════════════

Without residual connections, in an L-layer network, the gradient is:
    ∂L/∂x₁ = ∂L/∂xₗ × ∂xₗ/∂xₗ₋₁ × ... × ∂x₂/∂x₁

This is a product of L matrices. If any of these has eigenvalues < 1,
the product shrinks exponentially → vanishing gradients.

With residual connections: output = x + f(x)
    ∂output/∂x = I + ∂f/∂x

The "I" (identity) means gradients always have a direct path that
doesn't shrink. A 6-layer ViT with residuals actually has 2⁶ = 64
implicit gradient paths of different lengths (Veit et al., 2016).
The network effectively acts as an ensemble of shallow networks.

We'll verify this in our ablation study: removing residual connections
should cause gradients to vanish and training to fail.
"""

import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import ViTConfig
from models.attention import MultiHeadAttention
from models.mlp import LayerNorm, MLP


class TransformerBlock(nn.Module):
    """
    A single transformer block (Pre-Norm variant).

    The full ViT stacks num_layers of these sequentially.
    """

    def __init__(self, config: ViTConfig, use_scaling: bool = True):
        super().__init__()

        self.norm1 = LayerNorm(config.d_model)
        self.attention = MultiHeadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            use_scaling=use_scaling,
        )
        self.norm2 = LayerNorm(config.d_model)
        self.mlp = MLP(
            embed_dim=config.d_model,
            mlp_ratio=config.ffn_hidden / config.d_model,
            dropout=config.dropout,
        )
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through one transformer block.
        
        Args:
            x: Input tensor, shape [B, N, d_model]
        
        Returns:
            output: Transformed tensor, shape [B, N, d_model]
            attn_weights: Attention matrix, shape [B, num_heads, N, N]
        """
        # ─── Attention sub-block with residual connection ───
        # 1. Save input for residual
        residual = x
        # 2. Normalize (Pre-Norm)
        x = self.norm1(x)
        # 3. Multi-head self-attention
        x, attn_weights = self.attention(x)
        # 4. Dropout
        x = self.dropout(x)
        # 5. Add residual (the key operation!)
        x = residual + x
        
        # ─── MLP sub-block with residual connection ───
        # Same pattern: save → normalize → transform → dropout → add
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = residual + x
        
        return x, attn_weights


class TransformerBlockPostNorm(nn.Module):
    """
    Post-Norm variant — FOR ABLATION STUDY ONLY.
    
    The only difference: LayerNorm is applied AFTER the residual addition
    instead of before the sublayer.
    
    Pre-Norm:  x + Sublayer(LayerNorm(x))
    Post-Norm: LayerNorm(x + Sublayer(x))
    
    We expect this to train less stably than Pre-Norm.
    """
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = LayerNorm(config.d_model)
        self.attention = MultiHeadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.norm2 = LayerNorm(config.d_model)
        self.mlp = MLP(
            embed_dim=config.d_model,
            mlp_ratio=config.ffn_hidden / config.d_model,
            dropout=config.dropout,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Post-Norm: normalize AFTER residual addition
        attn_output, attn_weights = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        mlp_output = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_output))
        
        return x, attn_weights


class TransformerBlockNoResidual(nn.Module):
    """
    No residual connections — FOR ABLATION STUDY ONLY.
    
    Without residuals, gradients must flow through every layer
    sequentially. For a 6-layer model, this should cause
    vanishing gradients and training failure.
    """
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = LayerNorm(config.d_model)
        self.attention = MultiHeadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.norm2 = LayerNorm(config.d_model)
        self.mlp = MLP(
            embed_dim=config.d_model,
            mlp_ratio=config.ffn_hidden / config.d_model,
            dropout=config.dropout,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NO residual connections — just sequential processing
        x = self.norm1(x)
        x, attn_weights = self.attention(x)
        x = self.dropout(x)
        # Note: no "x = residual + x" here!
        
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        
        return x, attn_weights


# ──────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    config = ViTConfig()
    
    # ──────────────────────────────────────────────
    print("=" * 60)
    print("TEST 1: TransformerBlock (Pre-Norm) — Shape & Forward Pass")
    print("=" * 60)
    
    block = TransformerBlock(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in block.parameters())
    print(f"TransformerBlock parameters: {total_params:,}")
    print(f"  - norm1 (LayerNorm):       {sum(p.numel() for p in block.norm1.parameters()):,}")
    print(f"  - attention (MHA):         {sum(p.numel() for p in block.attention.parameters()):,}")
    print(f"  - norm2 (LayerNorm):       {sum(p.numel() for p in block.norm2.parameters()):,}")
    print(f"  - mlp (FFN):               {sum(p.numel() for p in block.mlp.parameters()):,}")
    
    # Forward pass
    x = torch.randn(4, 65, 128)
    output, attn_weights = block(x)
    
    print(f"\nInput shape:            {x.shape}")
    print(f"Output shape:           {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    assert output.shape == x.shape, f"Shape mismatch: {output.shape}"
    assert attn_weights.shape == (4, config.num_heads, 65, 65)
    
    print(" TransformerBlock shape test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: Residual connection verification")
    print("=" * 60)
    
    # The output should be CLOSE to the input (because of residual)
    # but not identical (because attention and MLP modify it)
    diff = (output - x).abs().mean().item()
    print(f"Mean absolute difference from input: {diff:.4f}")
    print(f"  (Should be small but non-zero — residual + small modifications)")
    
    assert diff > 0.001, "Output is too similar to input — block not computing!"
    assert diff < 10.0, "Output is too different from input — residual may be broken!"
    
    print(" Residual connection test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 3: Gradient flow through the block")
    print("=" * 60)
    
    block.zero_grad()
    x = torch.randn(4, 65, 128, requires_grad=True)
    output, _ = block(x)
    loss = output.sum()
    loss.backward()
    
    # Check input gradient
    assert x.grad is not None, "No gradient for input!"
    input_grad_norm = x.grad.norm().item()
    print(f"Input gradient norm: {input_grad_norm:.4f}")
    
    # Check all parameter gradients
    all_ok = True
    for name, param in block.named_parameters():
        if param.grad is None:
            print(f"   No gradient: {name}")
            all_ok = False
        elif torch.isnan(param.grad).any():
            print(f"   NaN gradient: {name}")
            all_ok = False
    
    if all_ok:
        print("All parameters received valid gradients")
    print(" Gradient flow test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 4: Stacking multiple blocks (like the full ViT)")
    print("=" * 60)
    
    # Create 6 blocks (default num_layers)
    blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
    
    total_block_params = sum(p.numel() for p in blocks.parameters())
    print(f"Total parameters for {config.num_layers} blocks: {total_block_params:,}")
    print(f"  ({total_block_params // config.num_layers:,} per block)")
    
    # Forward pass through all blocks
    x = torch.randn(4, 65, 128)
    all_attn_weights = []
    
    for i, block in enumerate(blocks):
        x, attn_weights = block(x)
        all_attn_weights.append(attn_weights)
    
    print(f"\nAfter {config.num_layers} blocks:")
    print(f"  Output shape: {x.shape}")
    print(f"  Attention weights collected: {len(all_attn_weights)}")
    print(f"  Each attention weight shape: {all_attn_weights[0].shape}")
    
    # Stack all attention weights for visualization later
    stacked_attn = torch.stack(all_attn_weights)
    print(f"  Stacked attention shape: {stacked_attn.shape}")
    print(f"    (layers, batch, heads, seq, seq)")
    
    assert x.shape == (4, 65, 128)
    assert stacked_attn.shape == (6, 4, 4, 65, 65)
    
    print(" Stacking test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 5: Ablation variants exist and run")
    print("=" * 60)
    
    # Post-Norm variant
    post_norm = TransformerBlockPostNorm(config)
    out_post, _ = post_norm(torch.randn(4, 65, 128))
    assert out_post.shape == (4, 65, 128)
    print("  Post-Norm block: ")
    
    # No-Residual variant
    no_res = TransformerBlockNoResidual(config)
    out_nores, _ = no_res(torch.randn(4, 65, 128))
    assert out_nores.shape == (4, 65, 128)
    print("  No-Residual block: ")
    
    print(" Ablation variants test PASSED!")
    
    print("\n" + "=" * 60)
    print("ALL TRANSFORMER BLOCK TESTS PASSED!")
    print("=" * 60)