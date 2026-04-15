"""
Transformer Encoder Block for Vision Transformer.

This is the repeating unit of the ViT. The full model stacks N of these.
Each block takes a sequence of tokens, lets them attend to each other
(MHSA), then processes each token independently (MLP), with
Pre-LayerNorm and residual connections for stability.

═══════════════════════════════════════════════════════════
ARCHITECTURE (Pre-LN / Pre-Norm):
═══════════════════════════════════════════════════════════

    Input x ─────────────────────────────┐
         │                               │ (residual)
         ↓                               │
    LayerNorm                            │
         │                               │
         ↓                               │
    MHSA                                 │
         │                               │
      Dropout                            │
         │                               │
    x + attn_out  ◄──────────────────────┘
         │
    x' ──────────────────────────────────┐
         │                               │ (residual)
         ↓                               │
    LayerNorm                            │
         │                               │
         ↓                               │
    MLP                                  │
         │                               │
      Dropout                            │
         │                               │
    x' + mlp_out  ◄──────────────────────┘
         │
       Output

═══════════════════════════════════════════════════════════
PRE-NORM vs POST-NORM
═══════════════════════════════════════════════════════════

Pre-Norm (used here):  output = x + Sublayer(LayerNorm(x))
Post-Norm (original):  output = LayerNorm(x + Sublayer(x))

WHY Pre-Norm is better for training stability:
  With Pre-Norm the identity I always exists in the gradient path:
    ∂output/∂x = I + ∂Sublayer/∂LayerNorm · ∂LayerNorm/∂x
  Gradients can always flow directly through the skip connection.

  With Post-Norm the LayerNorm Jacobian multiplies everything,
  including the skip, which can distort gradients and cause
  instability in deep models.

═══════════════════════════════════════════════════════════
RESIDUAL CONNECTIONS — Why They're Essential
═══════════════════════════════════════════════════════════

Without residuals in an L-layer network:
    ∂L/∂x₁ = product of L Jacobians → shrinks exponentially (vanishing grads)

With residuals: output = x + f(x)
    ∂output/∂x = I + ∂f/∂x
    The identity I gives gradients a direct highway that never shrinks.
    A 12-layer ViT with residuals has 2¹² = 4096 implicit gradient paths
    (Veit et al., 2016) — effectively an ensemble of shallow networks.
"""

import torch
import torch.nn as nn

from configs.config import ViTConfig
from models.attention import MultiHeadAttention
from models.mlp import MLP


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block (Pre-LN architecture).

    Combines MHSA and MLP with Pre-LayerNorm and residual connections.

    Args:
        embed_dim   : Token embedding dimension.           Default 768.
        num_heads   : Number of attention heads.           Default 12.
        mlp_ratio   : MLP hidden-dim expansion factor.     Default 4.0.
        dropout     : Dropout after attention and MLP.     Default 0.0.
        attn_dropout: Dropout inside attention weights.    Default 0.0.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            (B, N, embed_dim)
        """
        # Attention sub-block: Pre-LN + residual
        attn_out, _ = self.attn(self.norm1(x))
        x = x + self.drop1(attn_out)

        # MLP sub-block: Pre-LN + residual
        x = x + self.drop2(self.mlp(self.norm2(x)))

        return x

    def get_attention_weights(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (output, attn_weights) — useful for visualization / rollout.

        Args:
            x: (B, N, embed_dim)
        Returns:
            out         : (B, N, embed_dim)
            attn_weights: (B, num_heads, N, N)
        """
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x, attn_weights


# ── Ablation variants (used by ViT for comparative experiments) ───────────────

class TransformerBlockPostNorm(nn.Module):
    """Post-Norm variant — FOR ABLATION STUDY ONLY.

    Pre-Norm:  x + Sublayer(LayerNorm(x))
    Post-Norm: LayerNorm(x + Sublayer(x))   ← applied AFTER residual addition

    We expect this to train less stably than Pre-Norm.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn  = MultiHeadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp   = MLP(
            embed_dim=config.d_model,
            mlp_ratio=config.ffn_hidden / config.d_model,
            dropout=config.dropout,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x, attn_weights


class TransformerBlockNoResidual(nn.Module):
    """No residual connections — FOR ABLATION STUDY ONLY.

    Without residuals, gradients must flow through every layer sequentially.
    For a deep model this should cause vanishing gradients and training failure.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn  = MultiHeadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp   = MLP(
            embed_dim=config.d_model,
            mlp_ratio=config.ffn_hidden / config.d_model,
            dropout=config.dropout,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, attn_weights = self.attn(self.norm1(x))
        x = self.dropout(x)
        x = self.mlp(self.norm2(x))
        x = self.dropout(x)
        return x, attn_weights


# ──────────────────────────────────────────────────────
# TESTS — python models/transformer_block.py
# ──────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── TEST 1: Shape & parameter count ──────────────────────────────────
    print("=" * 60)
    print("TEST 1: TransformerEncoderBlock (embed=768, heads=12)")
    print("=" * 60)

    block = TransformerEncoderBlock(embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1)

    total = sum(p.numel() for p in block.parameters())
    print("Parameters:")
    print(f"  norm1 (LN)  : {sum(p.numel() for p in block.norm1.parameters()):>10,}")
    print(f"  attn  (MHSA): {sum(p.numel() for p in block.attn.parameters()):>10,}")
    print(f"  norm2 (LN)  : {sum(p.numel() for p in block.norm2.parameters()):>10,}")
    print(f"  mlp   (FFN) : {sum(p.numel() for p in block.mlp.parameters()):>10,}")
    print(f"  Total       : {total:>10,}")

    x   = torch.randn(2, 257, 768)
    out = block(x)
    print(f"\nInput  : {tuple(x.shape)}")
    print(f"Output : {tuple(out.shape)}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    print(" Shape test PASSED!")

    # ── TEST 2: Residual connection ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: Residual connection")
    print("=" * 60)

    diff = (out - x).abs().mean().item()
    print(f"Mean |output - input|: {diff:.4f}  (non-zero = block is computing)")
    assert 0.001 < diff < 10.0
    print(" PASSED!")

    # ── TEST 3: Attention weights via get_attention_weights ───────────────
    print("\n" + "=" * 60)
    print("TEST 3: get_attention_weights")
    print("=" * 60)

    out2, attn = block.get_attention_weights(x)
    print(f"Output      : {tuple(out2.shape)}")
    print(f"Attn weights: {tuple(attn.shape)}")
    assert out2.shape  == (2, 257, 768)
    assert attn.shape  == (2, 12, 257, 257)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 12, 257), atol=1e-5)
    print(" PASSED!")

    # ── TEST 4: Gradient flow ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 4: Gradient flow")
    print("=" * 60)

    block.zero_grad()
    x_g = torch.randn(2, 257, 768, requires_grad=True)
    block(x_g).sum().backward()
    assert x_g.grad is not None and not torch.isnan(x_g.grad).any()
    all_ok = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in block.parameters()
    )
    print(f"  Input grad norm : {x_g.grad.norm().item():.4f}")
    print(f"  All param grads : {'valid' if all_ok else 'MISSING'}")
    print(" PASSED!")

    # ── TEST 5: Stack 12 blocks ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 5: Stacking 12 blocks (ViT-Base depth)")
    print("=" * 60)

    blocks = nn.ModuleList([
        TransformerEncoderBlock(embed_dim=768, num_heads=12) for _ in range(12)
    ])
    total_stack = sum(p.numel() for p in blocks.parameters())
    print(f"12 blocks total params : {total_stack:,}  ({total_stack//12:,} per block)")

    x = torch.randn(2, 257, 768)
    for blk in blocks:
        x = blk(x)
    assert x.shape == (2, 257, 768)
    print(f"Output after 12 blocks : {tuple(x.shape)}")
    print(" PASSED!")

    # ── TEST 6: Ablation variants ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 6: Ablation variants (Post-Norm, No-Residual)")
    print("=" * 60)

    from configs.config import ViTConfig
    cfg = ViTConfig()

    pn = TransformerBlockPostNorm(cfg)
    out_pn, _ = pn(torch.randn(2, 65, 128))
    assert out_pn.shape == (2, 65, 128)
    print("  Post-Norm block     : PASSED")

    nr = TransformerBlockNoResidual(cfg)
    out_nr, _ = nr(torch.randn(2, 65, 128))
    assert out_nr.shape == (2, 65, 128)
    print("  No-Residual block   : PASSED")

    print("\n" + "=" * 60)
    print("ALL TRANSFORMER BLOCK TESTS PASSED!")
    print("=" * 60)
