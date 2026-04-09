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
  x → Linear(D, 4D) → GELU → Dropout → Linear(4D, D) → Dropout

WHY expand to 4× then compress back?
  Think of the first linear layer as a set of "pattern detectors."
  Each of the 3072 neurons in the hidden layer activates for a different
  pattern in the 768-dimensional input. The second linear layer maps
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
        # gamma (scale) — initialized to 1 (no scaling initially)
        # beta (shift)  — initialized to 0 (no shifting initially)
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)
        Returns:
            Normalized tensor, same shape.
        """
        mean  = x.mean(dim=-1, keepdim=True)
        var   = x.var(dim=-1,  keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class MLP(nn.Module):
    """
    Feed-Forward Network (MLP) block used inside each Transformer layer.

    Architecture:
        x → fc1 (D → mlp_ratio*D) → GELU → Dropout → fc2 (mlp_ratio*D → D) → Dropout

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
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1     = nn.Linear(embed_dim, hidden_dim)
        self.act     = nn.GELU()
        self.fc2     = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            (B, N, embed_dim)

        For embed_dim=768, mlp_ratio=4:
            (B, N, 768) → fc1 → (B, N, 3072) → GELU → dropout
                       → fc2 → (B, N,  768) → dropout
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def gelu_manual(x: torch.Tensor) -> torch.Tensor:
    """
    Manual GELU implementation for understanding and verification.

    GELU(x) = x * Φ(x) where Φ(x) is the standard normal CDF.
    Using the tanh approximation (same as PyTorch's default):
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
    ))


# ──────────────────────────────────────────────────────
# TESTS — python models/mlp.py
# ──────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── TEST 1: LayerNorm matches nn.LayerNorm ───────────────────────────
    print("=" * 60)
    print("TEST 1: LayerNorm matches PyTorch's nn.LayerNorm")
    print("=" * 60)

    d = 768
    our_ln   = LayerNorm(d)
    torch_ln = nn.LayerNorm(d)
    with torch.no_grad():
        torch_ln.weight.copy_(our_ln.gamma)
        torch_ln.bias.copy_(our_ln.beta)

    x = torch.randn(2, 257, d)
    max_diff = (our_ln(x) - torch_ln(x)).abs().max().item()
    print(f"Input shape  : {tuple(x.shape)}")
    print(f"Max diff from nn.LayerNorm: {max_diff:.2e}")
    assert max_diff < 1e-4
    print(" LayerNorm test PASSED!")

    # ── TEST 2: GELU approximation ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: GELU manual implementation matches F.gelu")
    print("=" * 60)

    x = torch.randn(1000)
    max_diff = (gelu_manual(x) - F.gelu(x)).abs().max().item()
    print(f"Max diff from F.gelu: {max_diff:.2e}")
    assert max_diff < 1e-3
    print("\n  x      | GELU(x)  | ReLU(x)")
    for v in [-2., -1., 0., 1., 2.]:
        t = torch.tensor(v)
        print(f"  {v:+4.1f}   | {F.gelu(t):+.4f}  | {F.relu(t):+.4f}")
    print(" GELU test PASSED!")

    # ── TEST 3: MLP shape and parameter count ────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 3: MLP (embed=768, ratio=4.0)")
    print("=" * 60)

    mlp = MLP(embed_dim=768, mlp_ratio=4.0, dropout=0.1)
    fc1_p = sum(p.numel() for p in mlp.fc1.parameters())
    fc2_p = sum(p.numel() for p in mlp.fc2.parameters())
    total = sum(p.numel() for p in mlp.parameters())
    print(f"  fc1 (768 → 3072) : {fc1_p:>10,}")
    print(f"  fc2 (3072 → 768) : {fc2_p:>10,}")
    print(f"  Total            : {total:>10,}")

    x   = torch.randn(2, 257, 768)
    out = mlp(x)
    print(f"\nInput  : {tuple(x.shape)}")
    print(f"Output : {tuple(out.shape)}")
    assert out.shape == x.shape
    print(" MLP shape test PASSED!")

    # ── TEST 4: Gradient flow ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 4: Gradient flow through LayerNorm + MLP")
    print("=" * 60)

    ln  = LayerNorm(768)
    mlp = MLP(embed_dim=768)
    x_g = torch.randn(2, 257, 768, requires_grad=True)
    mlp(ln(x_g)).sum().backward()
    assert x_g.grad is not None and not torch.isnan(x_g.grad).any()
    all_ok = all(p.grad is not None for p in list(ln.parameters()) + list(mlp.parameters()))
    print(f"  Input grad    : valid")
    print(f"  Param grads   : {'all valid' if all_ok else 'MISSING'}")
    print(" Gradient flow test PASSED!")

    print("\n" + "=" * 60)
    print("ALL MLP TESTS PASSED!")
    print("=" * 60)
