"""
Attention Module for Vision Transformer.

This file contains the two most important pieces of the entire project:
1. Scaled Dot-Product Attention (the core computation)
2. Multi-Head Attention (the wrapper that runs multiple attentions in parallel)

═══════════════════════════════════════════════════════════
THE INTUITION (before the math):
═══════════════════════════════════════════════════════════

Imagine you're looking at a photo of a dog on a beach.
Patch #12 contains part of the dog's ear.
Patch #15 contains part of the dog's tail.
Patch #40 contains sand.

When processing patch #12 (the ear), the model should "pay attention"
to patch #15 (the tail) because they're both part of the dog — knowing
about the tail helps understand the ear is part of an animal.
It should pay LESS attention to patch #40 (sand) because sand isn't
helpful for understanding what the ear is.

Self-attention computes exactly this: for each patch, it figures out
how much to look at every other patch, then creates a new representation
that's a weighted mix of all patches (weighted by "how relevant they are").

═══════════════════════════════════════════════════════════
THE MATH:
═══════════════════════════════════════════════════════════

Given input X (a sequence of token vectors):

1. Create three projections:
   Q = X @ W_Q    (Query: "what am I looking for?")
   K = X @ W_K    (Key: "what do I contain?")
   V = X @ W_V    (Value: "what information do I provide?")

2. Compute attention scores:
   scores = Q @ K^T    (dot product measures similarity between queries and keys)

3. Scale the scores:
   scores = scores / √d_k

   *** THIS IS THE KEY DERIVATION ***
   If q_i, k_i ~ N(0,1) independently, then:
     q·k = Σᵢ qᵢkᵢ  (sum of d_k terms)
     E[q·k] = 0
     Var(q·k) = d_k   (each term has variance 1, and they're independent)

   So for d_k=64, the dot products have std ≈ 8
   Softmax with large inputs saturates → gradients vanish → can't learn
   Dividing by √d_k restores Var = 1 → softmax behaves well

4. Apply softmax:
   attention_weights = softmax(scores, dim=-1)
   (each row sums to 1 — it's a probability distribution over "who to attend to")

5. Weighted sum of values:
   output = attention_weights @ V
   (each token's output is a weighted combination of all value vectors)

═══════════════════════════════════════════════════════════
MULTI-HEAD ATTENTION:
═══════════════════════════════════════════════════════════

Instead of one attention with d_model dimensions, we run h separate
attentions (heads), each with d_k = d_model/h dimensions.

WHY? A single attention matrix has rank ≤ d_k. That means it can
only represent one "type" of relationship at a time. Multiple heads
let the model capture different relationships simultaneously:
  - Head 1 might attend to nearby patches (local texture)
  - Head 2 might attend to patches of similar color
  - Head 3 might attend to distant but semantically related patches

The outputs of all heads are concatenated and projected back to d_model.

FUSED QKV PROJECTION:
  Instead of three separate Linear(D, D) calls for Q, K, V,
  we use one Linear(D, 3D) and split the output. This is ~1.5x
  faster because a single large GEMM is more efficient on GPU
  than three smaller ones.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dropout: nn.Dropout | None = None,
    scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

    Args:
        Q: Query tensor,  shape (B, num_heads, N, d_k)
        K: Key tensor,    shape (B, num_heads, N, d_k)
        V: Value tensor,  shape (B, num_heads, N, d_k)
        dropout: Optional dropout applied to attention weights.
        scale: Whether to scale by √d_k (set False for ablation).

    Returns:
        output:           (B, num_heads, N, d_k)
        attention_weights:(B, num_heads, N, N)
    """
    d_k = Q.shape[-1]

    # Q @ Kᵀ: entry (i,j) = how much token i should attend to token j
    scores = torch.matmul(Q, K.transpose(-2, -1))   # (B, h, N, N)

    if scale:
        scores = scores / math.sqrt(d_k)

    attention_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    output = torch.matmul(attention_weights, V)     # (B, h, N, d_k)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with fused QKV projection.

    Uses a single Linear(D, 3D) to compute Q, K, V in one matmul,
    then splits and reshapes into per-head tensors.

    Args:
        embed_dim  : Token embedding dimension.  Default 768 (ViT-Base).
        num_heads  : Number of attention heads.  Default 12.
        dropout    : Dropout on attention weights and output projection.
        qkv_bias   : Whether to add bias to the QKV projection. Default True.
        use_scaling: Scale attention scores by 1/√d_k. Default True.
                     Set False for ablation study.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.embed_dim   = embed_dim
        self.num_heads   = num_heads
        self.head_dim    = embed_dim // num_heads
        self.scale       = self.head_dim ** -0.5
        self.use_scaling = use_scaling

        self.qkv          = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj         = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, embed_dim)  — e.g. (B, 257, 768)

        Returns:
            out         : (B, N, embed_dim)
            attn_weights: (B, num_heads, N, N)  — pre-dropout softmax weights

        Step-by-step for B=2, N=257, embed_dim=768, heads=12, head_dim=64:

          qkv(x)   : (B, N, 2304)
          reshape  : (B, N, 3, 12, 64)
          permute  : (3, B, 12, N, 64)
          q/k/v    : (B, 12, N, 64) each

          q @ kᵀ   : (B, 12, N, N)  raw scores
          / √64    : (B, 12, N, N)  scaled
          softmax  : (B, 12, N, N)  attention weights
          dropout  : (B, 12, N, N)

          attn @ v : (B, 12, N, 64)
          reshape  : (B, N, 768)    concatenated heads
          proj     : (B, N, 768)    output projection
        """
        B, N, C = x.shape

        # ── 1. Fused QKV ──────────────────────────────────────────────────
        qkv = self.qkv(x)                              # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)              # (3, B, h, N, head_dim)
        q, k, v = qkv.unbind(0)                        # each: (B, h, N, head_dim)

        # ── 2. Scaled dot-product attention ───────────────────────────────
        scale = self.scale if self.use_scaling else 1.0
        attn = (q @ k.transpose(-2, -1)) * scale       # (B, h, N, N)
        attn = attn.softmax(dim=-1)
        attn_weights = attn                             # save before dropout
        attn = self.attn_dropout(attn)

        # ── 3. Weighted sum + concat heads ────────────────────────────────
        x = attn @ v                                   # (B, h, N, head_dim)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)  # (B, N, D)

        # ── 4. Output projection ───────────────────────────────────────────
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x, attn_weights

    @torch.no_grad()
    def get_attention_map(
        self,
        x: torch.Tensor,
        head: int | None = None,
    ) -> torch.Tensor:
        """Return attention weights averaged across heads (or a single head).

        Args:
            x    : (B, N, embed_dim) input tokens.
            head : Head index (0-based). If None, return mean across all heads.

        Returns:
            (B, N, N) attention map.
        """
        was_training = self.training
        self.eval()
        _, attn_weights = self.forward(x)   # (B, h, N, N)
        if was_training:
            self.train()

        if head is not None:
            return attn_weights[:, head]    # (B, N, N)
        return attn_weights.mean(dim=1)     # (B, N, N)


# ──────────────────────────────────────────────────────
# TESTS — python models/attention.py
# ──────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── TEST 1: scaled_dot_product_attention ─────────────────────────────
    print("=" * 60)
    print("TEST 1: Scaled Dot-Product Attention")
    print("=" * 60)

    B, h, N, d_k = 2, 12, 257, 64
    Q = torch.randn(B, h, N, d_k)
    K = torch.randn(B, h, N, d_k)
    V = torch.randn(B, h, N, d_k)

    out, attn = scaled_dot_product_attention(Q, K, V)
    print(f"Output shape : {tuple(out.shape)}")
    print(f"Attn shape   : {tuple(attn.shape)}")
    assert out.shape  == (B, h, N, d_k)
    assert attn.shape == (B, h, N, N)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, h, N), atol=1e-5)
    assert (attn >= 0).all()
    print(" PASSED!")

    # ── TEST 2: √d_k scaling effect ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: Verify √d_k scaling reduces softmax saturation")
    print("=" * 60)

    def entropy(a):
        return -(a * (a + 1e-10).log()).sum(dim=-1).mean().item()

    _, a_scaled   = scaled_dot_product_attention(Q, K, V, scale=True)
    _, a_unscaled = scaled_dot_product_attention(Q, K, V, scale=False)
    print(f"Entropy WITH scaling   : {entropy(a_scaled):.4f}")
    print(f"Entropy WITHOUT scaling: {entropy(a_unscaled):.4f}")
    assert entropy(a_scaled) > entropy(a_unscaled), \
        "Scaling should produce more uniform (higher entropy) attention"
    print(" PASSED!")

    # ── TEST 3: MultiHeadAttention shape & params ─────────────────────────
    print("\n" + "=" * 60)
    print("TEST 3: MultiHeadAttention (embed=768, heads=12)")
    print("=" * 60)

    mha   = MultiHeadAttention(embed_dim=768, num_heads=12, dropout=0.1)
    total = sum(p.numel() for p in mha.parameters())
    print(f"Parameters:")
    print(f"  qkv  (768→2304): {sum(p.numel() for p in mha.qkv.parameters()):>10,}")
    print(f"  proj (768→768) : {sum(p.numel() for p in mha.proj.parameters()):>10,}")
    print(f"  Total          : {total:>10,}")
    print(f"\nhead_dim = {mha.head_dim}  (768 / 12)")
    print(f"scale    = {mha.scale:.6f}  (1 / √{mha.head_dim})")

    x = torch.randn(2, 257, 768)
    out, attn = mha(x)
    print(f"\nInput  : {tuple(x.shape)}")
    print(f"Output : {tuple(out.shape)}")
    print(f"Attn   : {tuple(attn.shape)}")
    assert out.shape  == (2, 257, 768)
    assert attn.shape == (2, 12, 257, 257)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 12, 257), atol=1e-5)
    print(" Shape test PASSED!")

    # ── TEST 4: attention map visualization helper ────────────────────────
    print("\n" + "=" * 60)
    print("TEST 4: get_attention_map")
    print("=" * 60)

    amap_mean  = mha.get_attention_map(x)
    amap_head0 = mha.get_attention_map(x, head=0)
    print(f"Mean-head map : {tuple(amap_mean.shape)}")
    print(f"Head-0 map    : {tuple(amap_head0.shape)}")
    assert amap_mean.shape  == (2, 257, 257)
    assert amap_head0.shape == (2, 257, 257)
    print(" PASSED!")

    # ── TEST 5: gradient flow ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 5: Gradient flow")
    print("=" * 60)

    mha.zero_grad()
    x_g = torch.randn(2, 257, 768, requires_grad=True)
    mha(x_g)[0].sum().backward()
    assert x_g.grad is not None and not torch.isnan(x_g.grad).any()
    all_ok = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in mha.parameters()
    )
    print(f"  Input grad clean : True")
    print(f"  Param grads clean: {all_ok}")
    print(" PASSED!")

    print("\n" + "=" * 60)
    print("ALL ATTENTION TESTS PASSED!")
    print("=" * 60)
