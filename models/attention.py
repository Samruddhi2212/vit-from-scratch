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
   
   So for d_k=32, the dot products have std ≈ 5.7
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
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import ViTConfig


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dropout: nn.Dropout = None,
    scale: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.
    
    This is the CORE computation. Everything else is a wrapper around this.
    
    Args:
        Q: Query tensor,  shape [B, num_heads, N, d_k]
        K: Key tensor,    shape [B, num_heads, N, d_k]
        V: Value tensor,  shape [B, num_heads, N, d_k]
        dropout: Optional dropout to apply to attention weights
        scale: Whether to scale by √d_k (set to False for ablation study)
    
    Returns:
        output: Weighted sum of values, shape [B, num_heads, N, d_k]
        attention_weights: The attention matrix, shape [B, num_heads, N, N]
    """
    d_k = Q.shape[-1]  # dimension per head
    
    # ─── Step 1: Compute raw attention scores ───
    # Q @ K^T: each query dot-producted with each key
    # [B, h, N, d_k] @ [B, h, d_k, N] → [B, h, N, N]
    # Entry (i, j) = how much token i should attend to token j
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # ─── Step 2: Scale ───
    # Without scaling, for d_k=32, scores have std ≈ 5.7
    # This makes softmax saturate (output ≈ one-hot)
    # Dividing by √d_k brings std back to ≈ 1
    if scale:
        scores = scores / math.sqrt(d_k)
    
    # ─── Step 3: Softmax ───
    # Convert scores to probabilities (each row sums to 1)
    # dim=-1 means softmax over the last dimension (the "attending to" dimension)
    # After this: attention_weights[b, h, i, j] = probability that token i
    #             attends to token j, in head h, for batch item b
    attention_weights = F.softmax(scores, dim=-1)
    
    # ─── Step 4: Dropout on attention weights (optional) ───
    # During training, randomly zero out some attention connections
    # This prevents the model from relying too heavily on specific relationships
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # ─── Step 5: Weighted sum of values ───
    # [B, h, N, N] @ [B, h, N, d_k] → [B, h, N, d_k]
    # Each token's output = weighted average of all value vectors
    # Weights come from the attention matrix
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    
    Runs multiple attention heads in parallel, each attending to
    different aspects of the input, then concatenates their outputs.
    
    Parameter count:
      W_Q: d_model × d_model = 128 × 128 = 16,384
      W_K: d_model × d_model = 128 × 128 = 16,384
      W_V: d_model × d_model = 128 × 128 = 16,384
      W_O: d_model × d_model = 128 × 128 = 16,384
      Biases: 4 × d_model = 4 × 128 = 512
      Total: 65,536 + 512 = 66,048
    """
    
    def __init__(self, config: ViTConfig, use_scaling: bool = True):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = config.d_k
        self.use_scaling = use_scaling  # flag for ablation study
        
        # ─── Projection matrices ───
        # Each of these projects the full d_model-dimensional input
        # into d_model dimensions. We then RESHAPE the output to
        # separate the heads: [B, N, d_model] → [B, N, num_heads, d_k]
        #
        # WHY one big linear instead of separate linears per head?
        # Mathematically identical, but one big matrix multiply is
        # much faster on GPU than num_heads small ones.
        #
        self.W_Q = nn.Linear(config.d_model, config.d_model)  # Query projection
        self.W_K = nn.Linear(config.d_model, config.d_model)  # Key projection
        self.W_V = nn.Linear(config.d_model, config.d_model)  # Value projection
        self.W_O = nn.Linear(config.d_model, config.d_model)  # Output projection
        
        self.attn_dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Input tensor, shape [B, N, d_model]
               where N = num_patches + 1 (including [CLS])
        
        Returns:
            output: Transformed tensor, shape [B, N, d_model]
            attention_weights: Attention matrix, shape [B, num_heads, N, N]
        """
        B, N, _ = x.shape
        
        # ─── Step 1: Project input into Q, K, V ───
        # Each: [B, N, d_model] → [B, N, d_model]
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # ─── Step 2: Reshape to separate heads ───
        # [B, N, d_model] → [B, N, num_heads, d_k] → [B, num_heads, N, d_k]
        #
        # The reshape splits the last dimension: d_model → (num_heads × d_k)
        # The transpose moves num_heads before N so each head processes
        # its own sequence independently
        #
        # Example for d_model=128, num_heads=4, d_k=32:
        #   [4, 65, 128] → [4, 65, 4, 32] → [4, 4, 65, 32]
        #   Now we have 4 heads, each seeing a sequence of 65 tokens in 32 dims
        #
        Q = Q.reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = K.reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = V.reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)
        
        # ─── Step 3: Apply scaled dot-product attention ───
        # This is where the actual attention computation happens
        # output: [B, num_heads, N, d_k]
        # attn_weights: [B, num_heads, N, N]
        output, attn_weights = scaled_dot_product_attention(
            Q, K, V,
            dropout=self.attn_dropout if self.training else None,
            scale=self.use_scaling
        )
        
        # ─── Step 4: Concatenate heads ───
        # [B, num_heads, N, d_k] → [B, N, num_heads, d_k] → [B, N, d_model]
        #
        # transpose undoes the earlier transpose
        # reshape merges the last two dims: (num_heads × d_k) = d_model
        #
        # .contiguous() ensures the tensor is stored in contiguous memory
        # (needed after transpose for reshape to work correctly)
        #
        output = output.transpose(1, 2).contiguous().reshape(B, N, self.d_model)
        
        # ─── Step 5: Output projection ───
        # [B, N, d_model] → [B, N, d_model]
        # This allows the model to learn how to combine information
        # from different heads
        output = self.W_O(output)
        
        return output, attn_weights


# ──────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    config = ViTConfig()
    
    print("=" * 60)
    print("TEST 1: Scaled Dot-Product Attention")
    print("=" * 60)
    
    B, h, N, d_k = 2, 4, 65, 32
    Q = torch.randn(B, h, N, d_k)
    K = torch.randn(B, h, N, d_k)
    V = torch.randn(B, h, N, d_k)
    
    output, attn_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"Q shape:               {Q.shape}")
    print(f"K shape:               {K.shape}")
    print(f"V shape:               {V.shape}")
    print(f"Output shape:          {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Check output shape matches V shape
    assert output.shape == (B, h, N, d_k), \
        f"Output shape mismatch: {output.shape}"
    
    # Check attention weights sum to 1 along the last dimension
    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        "Attention weights don't sum to 1!"
    
    # Check all attention weights are non-negative
    assert (attn_weights >= 0).all(), \
        "Found negative attention weights!"
    
    print(" Scaled dot-product attention test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: Verify √d_k scaling effect")
    print("=" * 60)
    
    # Compare attention WITH and WITHOUT scaling
    # Without scaling, attention should be more "peaky" (lower entropy)
    _, attn_scaled = scaled_dot_product_attention(Q, K, V, scale=True)
    _, attn_unscaled = scaled_dot_product_attention(Q, K, V, scale=False)
    
    # Compute entropy: -Σ p * log(p)
    # Higher entropy = more uniform attention (good)
    # Lower entropy = more concentrated/peaky attention (bad)
    def attention_entropy(attn):
        # Add small epsilon to avoid log(0)
        return -(attn * (attn + 1e-10).log()).sum(dim=-1).mean().item()
    
    entropy_scaled = attention_entropy(attn_scaled)
    entropy_unscaled = attention_entropy(attn_unscaled)
    
    print(f"Entropy WITH scaling:    {entropy_scaled:.4f}")
    print(f"Entropy WITHOUT scaling: {entropy_unscaled:.4f}")
    print(f"Scaling increases entropy (more uniform attention): {entropy_scaled > entropy_unscaled}")
    
    # The scaled version should have higher entropy (more uniform, less saturated)
    assert entropy_scaled > entropy_unscaled, \
        "Scaling should increase attention entropy!"
    
    print(" Scaling verification PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 3: Multi-Head Attention")
    print("=" * 60)
    
    mha = MultiHeadAttention(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"MultiHeadAttention parameters: {total_params:,}")
    print(f"  - W_Q: {sum(p.numel() for p in mha.W_Q.parameters()):,}")
    print(f"  - W_K: {sum(p.numel() for p in mha.W_K.parameters()):,}")
    print(f"  - W_V: {sum(p.numel() for p in mha.W_V.parameters()):,}")
    print(f"  - W_O: {sum(p.numel() for p in mha.W_O.parameters()):,}")
    
    # Forward pass
    x = torch.randn(4, 65, 128)  # [batch, seq_len, d_model]
    output, attn_weights = mha(x)
    
    print(f"\nInput shape:           {x.shape}")
    print(f"Output shape:          {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Check shapes
    assert output.shape == (4, 65, 128), \
        f"Output shape mismatch: {output.shape}"
    assert attn_weights.shape == (4, config.num_heads, 65, 65), \
        f"Attention weights shape mismatch: {attn_weights.shape}"
    
    # Check that the output is actually different from the input
    # (the module is computing something, not just passing through)
    assert not torch.allclose(output, x, atol=1e-3), \
        "Output shouldn't be identical to input!"
    
    print(" Multi-head attention test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 4: Gradient flow")
    print("=" * 60)
    
    # Verify that gradients flow through the entire module
    mha.zero_grad()
    x = torch.randn(4, 65, 128, requires_grad=True)
    output, _ = mha(x)
    loss = output.sum()
    loss.backward()
    
    # Check all parameters have gradients
    all_have_grads = True
    for name, param in mha.named_parameters():
        if param.grad is None:
            print(f"   No gradient for: {name}")
            all_have_grads = False
        elif torch.isnan(param.grad).any():
            print(f"   NaN gradient for: {name}")
            all_have_grads = False
    
    # Check input has gradient too
    assert x.grad is not None, "Input should have gradient!"
    assert not torch.isnan(x.grad).any(), "Input gradient has NaN!"
    
    if all_have_grads:
        print("All parameters received valid gradients")
    print(" Gradient flow test PASSED!")
    
    print("\n" + "=" * 60)
    print("ALL ATTENTION TESTS PASSED!")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE MULTI-HEAD SELF-ATTENTION
# Explicit constructor args — no ViTConfig dependency.
# Fused QKV projection for efficiency (standard in modern ViT implementations).
# ──────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with fused QKV projection.

    Key difference from MultiHeadAttention above:
      - Single nn.Linear(D, 3D) computes Q, K, V in one matmul instead of three.
        This is ~1.5x faster in practice because a single large GEMM is more
        efficient on GPU than three smaller ones.
      - Explicit constructor args (no ViTConfig dependency).
      - Returns attention weights for visualization / rollout.

    Args:
        embed_dim : Token embedding dimension.  Default 768 (ViT-Base).
        num_heads : Number of attention heads.  Default 12.
        dropout   : Dropout applied to attention weights and output projection.
        qkv_bias  : Whether to add bias to the QKV projection. Default True.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads          # d_k per head
        self.scale     = self.head_dim ** -0.5           # 1 / √d_k

        # Fused QKV: one matmul produces Q, K, V concatenated.
        # Weight shape: (embed_dim, 3 * embed_dim)
        self.qkv          = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj         = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, embed_dim)  — e.g. (B, 257, 768)

        Returns:
            out        : (B, N, embed_dim)  — attended output
            attn_weights: (B, num_heads, N, N) — softmax weights (for viz)

        Step-by-step for B=2, N=257, embed_dim=768, num_heads=12, head_dim=64:

          x           : (B, N, 768)
          qkv(x)      : (B, N, 2304)   ← 768*3
          reshape     : (B, N, 3, 12, 64)
          permute     : (3, B, 12, N, 64)
          q/k/v each  : (B, 12, N, 64)

          q @ kᵀ      : (B, 12, N, N)  ← raw scores
          / scale     : (B, 12, N, N)  ← prevent softmax saturation
          softmax     : (B, 12, N, N)  ← attention weights (rows sum to 1)
          dropout     : (B, 12, N, N)

          attn @ v    : (B, 12, N, 64)
          transpose   : (B, N, 12, 64)
          reshape     : (B, N, 768)    ← concatenate heads
          proj        : (B, N, 768)    ← output projection
        """
        B, N, C = x.shape

        # ── 1. Fused QKV projection ────────────────────────────────────────
        qkv = self.qkv(x)                              # (B, N, 3*embed_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)              # (3, B, h, N, head_dim)
        q, k, v = qkv.unbind(0)                        # each: (B, h, N, head_dim)

        # ── 2. Scaled dot-product attention ───────────────────────────────
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, N, N)
        attn = attn.softmax(dim=-1)
        attn_weights = attn                             # save before dropout
        attn = self.attn_dropout(attn)

        # ── 3. Weighted sum of values ──────────────────────────────────────
        x = attn @ v                                   # (B, h, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)        # (B, N, embed_dim)

        # ── 4. Output projection ───────────────────────────────────────────
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x, attn_weights

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_attention_map(
        self,
        x: torch.Tensor,
        head: int | None = None,
    ) -> torch.Tensor:
        """Return attention weights averaged across heads (or a single head).

        Useful for visualising which patches the model attends to.

        Args:
            x    : (B, N, embed_dim) input tokens.
            head : If given, return weights for that head only (0-indexed).
                   If None, return the mean across all heads.

        Returns:
            attn_map : (B, N, N) — where attn_map[b, i, j] is how much
                       token i attends to token j in batch item b.
        """
        was_training = self.training
        self.eval()
        _, attn_weights = self.forward(x)   # (B, num_heads, N, N)
        if was_training:
            self.train()

        if head is not None:
            return attn_weights[:, head]    # (B, N, N)
        return attn_weights.mean(dim=1)     # (B, N, N) — mean over heads


# ──────────────────────────────────────────────────────
# TEST — python models/attention.py
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("MultiHeadSelfAttention  (embed=768, heads=12)")
    print("=" * 56)

    mhsa = MultiHeadSelfAttention(embed_dim=768, num_heads=12, dropout=0.1)

    qkv_params  = sum(p.numel() for p in mhsa.qkv.parameters())
    proj_params = sum(p.numel() for p in mhsa.proj.parameters())
    total       = sum(p.numel() for p in mhsa.parameters())
    print(f"Parameters:")
    print(f"  qkv  (Linear 768→2304) : {qkv_params:>10,}")
    print(f"  proj (Linear 768→768)  : {proj_params:>10,}")
    print(f"  Total                  : {total:>10,}")
    print(f"\nhead_dim = {mhsa.head_dim}  (768 / 12)")
    print(f"scale    = {mhsa.scale:.6f}  (1 / √{mhsa.head_dim})")

    # ── shape test ──────────────────────────────────────────────────────
    print("\n── Shape test (B=2, N=257, C=768) ──")
    x = torch.randn(2, 257, 768)
    out, attn = mhsa(x)
    print(f"  Input         : {tuple(x.shape)}")
    print(f"  Output        : {tuple(out.shape)}")
    print(f"  Attn weights  : {tuple(attn.shape)}")
    assert out.shape  == (2, 257, 768),       f"Output shape wrong: {out.shape}"
    assert attn.shape == (2, 12, 257, 257),   f"Attn shape wrong: {attn.shape}"

    # ── attention weights sanity ─────────────────────────────────────────
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        "Attention weights must sum to 1 across last dim"
    assert (attn >= 0).all(), "Attention weights must be non-negative"
    print("  Attn rows sum to 1 : True")
    print("  Attn non-negative  : True")

    # ── visualization helper ─────────────────────────────────────────────
    print("\n── Attention map (mean over heads) ──")
    attn_map = mhsa.get_attention_map(x)           # (B, N, N)
    print(f"  Mean-head map shape : {tuple(attn_map.shape)}")
    attn_head0 = mhsa.get_attention_map(x, head=0) # (B, N, N)
    print(f"  Head-0    map shape : {tuple(attn_head0.shape)}")

    # ── gradient flow ────────────────────────────────────────────────────
    print("\n── Gradient flow ──")
    mhsa.zero_grad()
    x_g = torch.randn(2, 257, 768, requires_grad=True)
    out_g, _ = mhsa(x_g)
    out_g.sum().backward()
    assert x_g.grad is not None and not torch.isnan(x_g.grad).any()
    no_nan = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in mhsa.parameters()
    )
    print(f"  Input grad clean   : True")
    print(f"  Param grads clean  : {no_nan}")

    print("\nAll MultiHeadSelfAttention tests PASSED!")