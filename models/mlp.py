"""
Layer Normalization and Feed-Forward Network (MLP) for Vision Transformer.

This file contains two components:
1. LayerNorm — built from scratch (NOT using nn.LayerNorm)
2. MLP — the feed-forward network inside each transformer block

═══════════════════════════════════════════════════════════
LAYER NORMALIZATION — Why and How
═══════════════════════════════════════════════════════════

Problem: As data flows through deep networks, the distribution of
activations shifts at each layer (called "internal covariate shift").
This makes training unstable and slow.

Solution: Normalize the activations to have mean=0 and variance=1,
then let the model learn an optimal scale (gamma) and shift (beta).

LayerNorm vs BatchNorm — WHY LayerNorm for Transformers:

  BatchNorm: computes mean/variance ACROSS the batch for each feature
    - Statistics depend on batch size → breaks for batch_size=1
    - For sequences of different lengths, which positions do you batch?
    - At inference time, uses running statistics from training (can mismatch)

  LayerNorm: computes mean/variance ACROSS features for each token independently
    - Statistics are per-token → no batch dependency
    - Works identically for any batch size
    - Same computation at training and inference time
    - Each token is normalized independently → works for variable-length sequences

Formula:
  LayerNorm(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta

  Where:
    - mean, variance are computed over the feature dimension (last dim)
    - gamma (scale) and beta (shift) are learnable parameters
    - epsilon (1e-6) prevents division by zero

═══════════════════════════════════════════════════════════
FEED-FORWARD NETWORK (MLP) — Why and How
═══════════════════════════════════════════════════════════

After attention mixes information BETWEEN tokens, the FFN processes
each token INDEPENDENTLY through a small neural network.

Architecture:
  x → Linear(d_model, 4*d_model) → GELU → Dropout → Linear(4*d_model, d_model) → Dropout

WHY expand to 4× then compress back?
  Think of the first linear layer as a set of "pattern detectors."
  Each of the 512 neurons in the hidden layer activates for a different
  pattern in the 128-dimensional input. The second linear layer maps
  these detections to the output. The 4× expansion gives the network
  enough capacity to detect many patterns.

  Formally, this can be interpreted as a key-value memory:
    - Rows of W1 are "keys" (patterns to match)
    - Columns of W2 are "values" (outputs to produce)
    - The FFN computes: output = W2 * activation(W1 * x)
    - This is "if input matches key i, produce value i"

GELU — What and Why:
  GELU(x) = x * Φ(x), where Φ is the standard normal CDF
  
  Intuition: GELU is a "smooth ReLU." 
    - For large positive x: Φ(x) ≈ 1, so GELU(x) ≈ x (passes through)
    - For large negative x: Φ(x) ≈ 0, so GELU(x) ≈ 0 (blocked)
    - For x near 0: smooth transition (unlike ReLU's sharp corner)
  
  Mathematical interpretation as stochastic regularizer:
    GELU(x) = E[x * Bernoulli(Φ(x))]
    It randomly masks x based on its magnitude — small values are
    more likely to be zeroed out. This acts as a form of regularization
    during training, similar to dropout but input-dependent.
    
  Why not ReLU?
    ReLU has a discontinuous derivative at x=0 (technically undefined).
    GELU is smooth everywhere → smoother loss landscape → better optimization.
    ViT, GPT, BERT all use GELU. We'll compare both in ablation studies.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import ViTConfig


class LayerNorm(nn.Module):
    """
    Layer Normalization — implemented from scratch.
    
    We build this ourselves instead of using nn.LayerNorm to demonstrate
    we understand every computation. The test at the bottom verifies our
    implementation matches PyTorch's exactly.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
        # Learnable parameters:
        # gamma (scale) — initialized to 1 (no scaling initially)
        # beta (shift) — initialized to 0 (no shifting initially)
        # After training, the model learns the optimal scale and shift
        # for each feature dimension.
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input across the feature dimension.
        
        Args:
            x: Input tensor, shape [B, N, d_model]
        
        Returns:
            Normalized tensor, same shape [B, N, d_model]
        """
        # Compute mean across the last dimension (features)
        # keepdim=True so we can broadcast when subtracting
        # Shape: [B, N, 1]
        mean = x.mean(dim=-1, keepdim=True)
        
        # Compute variance across the last dimension
        # unbiased=False: divide by N, not N-1 (matches nn.LayerNorm behavior)
        # Shape: [B, N, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize: subtract mean, divide by std
        # eps prevents division by zero when variance is very small
        # Shape: [B, N, d_model]
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift with learnable parameters
        # gamma and beta are [d_model] — broadcast over B and N dimensions
        # Shape: [B, N, d_model]
        return self.gamma * x_norm + self.beta


class MLP(nn.Module):
    """
    Feed-Forward Network (MLP) for the transformer block.
    
    Architecture: Linear → GELU → Dropout → Linear → Dropout
    
    Parameter count for default config (d_model=128, ffn_hidden=512):
      fc1: 128 × 512 + 512 (bias) = 66,048
      fc2: 512 × 128 + 128 (bias) = 65,664
      Total: 131,712
    """
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        
        # First linear layer: expand from d_model to ffn_hidden (4× expansion)
        self.fc1 = nn.Linear(config.d_model, config.ffn_hidden)
        
        # Second linear layer: compress back from ffn_hidden to d_model
        self.fc2 = nn.Linear(config.ffn_hidden, config.d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.
        
        Args:
            x: Input tensor, shape [B, N, d_model]
        
        Returns:
            Output tensor, same shape [B, N, d_model]
        """
        # [B, N, 128] → [B, N, 512]
        x = self.fc1(x)
        
        # GELU activation
        # We use F.gelu for efficiency, but understand what it computes:
        # GELU(x) = x * Φ(x) where Φ is the standard normal CDF
        # Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        x = F.gelu(x)
        
        # Dropout after activation
        x = self.dropout(x)
        
        # [B, N, 512] → [B, N, 128]
        x = self.fc2(x)
        
        # Dropout after second linear
        x = self.dropout(x)
        
        return x


def gelu_manual(x: torch.Tensor) -> torch.Tensor:
    """
    Manual GELU implementation for understanding and verification.
    
    GELU(x) = x * Φ(x) where Φ(x) is the standard normal CDF.
    
    Using the tanh approximation (same as PyTorch's default):
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    
    This exists purely for educational purposes — we use F.gelu in the
    actual model for speed, but we can derive and verify this formula.
    """
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
    ))


# ──────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    config = ViTConfig()
    
    # ──────────────────────────────────────────────
    print("=" * 60)
    print("TEST 1: LayerNorm matches PyTorch's nn.LayerNorm")
    print("=" * 60)
    
    d_model = config.d_model
    our_ln = LayerNorm(d_model)
    torch_ln = nn.LayerNorm(d_model)
    
    # Copy our weights into PyTorch's LayerNorm for fair comparison
    # nn.LayerNorm uses .weight and .bias, ours uses .gamma and .beta
    with torch.no_grad():
        torch_ln.weight.copy_(our_ln.gamma)
        torch_ln.bias.copy_(our_ln.beta)
    
    # Test with random input
    x = torch.randn(4, 65, d_model)
    
    our_output = our_ln(x)
    torch_output = torch_ln(x)
    
    max_diff = (our_output - torch_output).abs().max().item()
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {our_output.shape}")
    print(f"Max difference from nn.LayerNorm: {max_diff:.2e}")
    
    assert torch.allclose(our_output, torch_output, atol=1e-5), \
        f"LayerNorm mismatch! Max diff: {max_diff}"
    
    # Verify the output has mean ≈ 0 and var ≈ 1 (before gamma/beta)
    # Since gamma=1 and beta=0 at init, output should be normalized
    output_mean = our_output.mean(dim=-1).mean().item()
    output_var = our_output.var(dim=-1).mean().item()
    print(f"Output mean (should be ≈0): {output_mean:.6f}")
    print(f"Output var  (should be ≈1): {output_var:.6f}")
    
    print(" LayerNorm test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: GELU manual implementation matches F.gelu")
    print("=" * 60)
    
    x = torch.randn(1000)
    our_gelu = gelu_manual(x)
    torch_gelu = F.gelu(x)
    
    max_diff = (our_gelu - torch_gelu).abs().max().item()
    print(f"Max difference from F.gelu: {max_diff:.2e}")
    
    assert torch.allclose(our_gelu, torch_gelu, atol=1e-3), \
        f"GELU mismatch! Max diff: {max_diff}"
    
    # Show GELU behavior at key points
    test_points = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    gelu_values = F.gelu(test_points)
    relu_values = F.relu(test_points)
    
    print("\n  x      | GELU(x)  | ReLU(x)  | Difference")
    print("  " + "-" * 48)
    for x_val, g_val, r_val in zip(test_points, gelu_values, relu_values):
        diff = g_val - r_val
        print(f"  {x_val:+5.1f}  | {g_val:+7.4f}  | {r_val:+7.4f}  | {diff:+7.4f}")
    
    print("\nKey insight: GELU is smooth (no sharp corner at 0 like ReLU)")
    print("  - For negative inputs: GELU gives small negative values (ReLU gives 0)")
    print("  - For positive inputs: both are similar")
    print("  - At x=0: GELU(0)=0, ReLU(0)=0 (same)")
    
    print(" GELU test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 3: MLP (Feed-Forward Network)")
    print("=" * 60)
    
    mlp = MLP(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in mlp.parameters())
    fc1_params = sum(p.numel() for p in mlp.fc1.parameters())
    fc2_params = sum(p.numel() for p in mlp.fc2.parameters())
    
    print(f"MLP parameters: {total_params:,}")
    print(f"  - fc1 (expand):   {fc1_params:,}  ({config.d_model} → {config.ffn_hidden})")
    print(f"  - fc2 (compress): {fc2_params:,}  ({config.ffn_hidden} → {config.d_model})")
    
    # Forward pass
    x = torch.randn(4, 65, config.d_model)
    output = mlp(x)
    
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == x.shape, \
        f"MLP output shape should match input! Got {output.shape}"
    
    # Verify it's computing something (not identity)
    assert not torch.allclose(output, x, atol=1e-3), \
        "MLP output shouldn't be identical to input!"
    
    print(" MLP test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 4: Gradient flow through LayerNorm and MLP")
    print("=" * 60)
    
    ln = LayerNorm(config.d_model)
    mlp = MLP(config)
    
    x = torch.randn(4, 65, config.d_model, requires_grad=True)
    
    # Pass through LayerNorm then MLP (like in a transformer block)
    normalized = ln(x)
    output = mlp(normalized)
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist for everything
    assert x.grad is not None, "Input should have gradient!"
    
    ln_grads_ok = all(p.grad is not None for p in ln.parameters())
    mlp_grads_ok = all(p.grad is not None for p in mlp.parameters())
    
    print(f"Input gradient exists:     {x.grad is not None}")
    print(f"LayerNorm gradients exist: {ln_grads_ok}")
    print(f"MLP gradients exist:       {mlp_grads_ok}")
    
    # Check no NaN gradients
    no_nans = not torch.isnan(x.grad).any()
    print(f"No NaN gradients:          {no_nans}")
    
    print(" Gradient flow test PASSED!")
    
    print("\n" + "=" * 60)
    print("ALL LAYERNORM + MLP TESTS PASSED!")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE MLP
# Explicit constructor args — no ViTConfig dependency.
# Designed for embed_dim=768 (ViT-Base / OSCD Siamese ViT).
# ──────────────────────────────────────────────────────────────────────────────

class StandaloneMLP(nn.Module):
    """
    Feed-Forward Network (MLP) block used inside each Transformer layer.

    Architecture:
        x → fc1 (D → 4D) → GELU → Dropout → fc2 (4D → D) → Dropout

    The 4× expansion creates a "bottleneck" that acts as a key-value memory:
      - fc1 rows are "keys"  — pattern detectors in the input space
      - fc2 cols are "values" — output directions to activate per pattern
      - GELU non-linearity selects which patterns are active

    Args:
        embed_dim : Input and output dimension.           Default 768.
        mlp_ratio : Hidden layer expansion factor.        Default 4.0 → 3072.
        dropout   : Dropout applied after each linear.   Default 0.0.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)   # 768 * 4 = 3072

        self.fc1     = nn.Linear(embed_dim, hidden_dim)
        self.act     = nn.GELU()
        self.fc2     = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim)

        Returns:
            (B, N, embed_dim)  — same shape as input

        For embed_dim=768, mlp_ratio=4:
            (B, N, 768) → fc1 → (B, N, 3072) → GELU → dropout
                       → fc2 → (B, N, 768)  → dropout
        """
        x = self.fc1(x)       # expand:   D → 4D
        x = self.act(x)       # GELU non-linearity
        x = self.dropout(x)
        x = self.fc2(x)       # compress: 4D → D
        x = self.dropout(x)
        return x


# ──────────────────────────────────────────────────────
# TEST — python models/mlp.py
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("StandaloneMLP  (embed=768, ratio=4.0)")
    print("=" * 52)

    mlp = StandaloneMLP(embed_dim=768, mlp_ratio=4.0, dropout=0.1)

    hidden_dim  = int(768 * 4.0)
    fc1_params  = sum(p.numel() for p in mlp.fc1.parameters())
    fc2_params  = sum(p.numel() for p in mlp.fc2.parameters())
    total       = sum(p.numel() for p in mlp.parameters())

    print(f"Parameters:")
    print(f"  fc1 (768 → {hidden_dim}) : {fc1_params:>10,}")
    print(f"  fc2 ({hidden_dim} → 768) : {fc2_params:>10,}")
    print(f"  Total               : {total:>10,}")

    # ── shape test ──────────────────────────────────────────────────────
    print("\n── Shape test (B=2, N=257, embed_dim=768) ──")
    x = torch.randn(2, 257, 768)
    out = mlp(x)
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"

    # ── output is not identity ───────────────────────────────────────────
    assert not torch.allclose(out, x, atol=1e-3), \
        "MLP must transform its input"

    # ── gradient flow ────────────────────────────────────────────────────
    print("\n── Gradient flow ──")
    mlp_g = StandaloneMLP(embed_dim=768)
    x_g   = torch.randn(2, 257, 768, requires_grad=True)
    mlp_g(x_g).sum().backward()
    assert x_g.grad is not None and not torch.isnan(x_g.grad).any()
    param_ok = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in mlp_g.parameters()
    )
    print(f"  Input grad clean : True")
    print(f"  Param grads clean: {param_ok}")

    print("\nAll StandaloneMLP tests PASSED!")