# Siamese ViT Change Detection — Project Handbook
### Complete Reference for Quiz / Presentation

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset — OSCD](#2-dataset--oscd)
3. [Data Pipeline](#3-data-pipeline)
4. [Full Architecture](#4-full-architecture)
5. [Every Layer Explained](#5-every-layer-explained)
6. [Activation Functions](#6-activation-functions)
7. [Loss Functions](#7-loss-functions)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Optimizer & Learning Rate Schedule](#9-optimizer--learning-rate-schedule)
10. [Training Hyperparameters](#10-training-hyperparameters)
11. [Key Design Decisions & Why](#11-key-design-decisions--why)
12. [Ablation Studies](#12-ablation-studies)
13. [Parameter Count Breakdown](#13-parameter-count-breakdown)
14. [Common Quiz Questions & Answers](#14-common-quiz-questions--answers)
15. [Resources & References](#15-resources--references)
16. [Full Forward Pass Walkthrough](#16-full-forward-pass-walkthrough)
17. [ViT Intuitions & Mental Models](#17-vit-intuitions--mental-models)
18. [Inductive Bias — CNN vs ViT](#18-inductive-bias--cnn-vs-vit)
19. [Common Failure Modes & Debugging](#19-common-failure-modes--debugging)

---

## 1. Project Overview

**Title:** Vision Transformer from Scratch for Urban Change Detection

**Core Task:** Binary semantic segmentation — given two satellite images of the same city taken at different times, predict *which pixels changed* (buildings built/demolished, roads added, land use shifted).

**Output:** A binary mask of the same resolution as the input — `1 = changed`, `0 = no change`.

**What makes this hard:**
- Extreme class imbalance: changed pixels are often < 5% of the image
- No pretrained weights — everything trained from random init on OSCD
- Must compare two images while being invariant to lighting/seasonal differences

**Key novelty:** The entire model — patch projection, positional embeddings, 12 transformer blocks, feature difference module, and decoder — is implemented from scratch without using `nn.TransformerEncoder` or any pretrained checkpoints.

---

## 2. Dataset — OSCD

**Full name:** Onera Satellite Change Detection dataset

**Sensor:** Sentinel-2 multispectral satellite (ESA Copernicus programme)
- 13 spectral bands at various resolutions (10m, 20m, 60m ground sampling distance)
- We use only **RGB (bands 4, 3, 2)** → 3-channel input

**Coverage:** 24 bi-temporal image pairs of urban areas worldwide
- Cities include: Paris, Tokyo, Las Vegas, Mumbai, etc.
- Time separation: months to years apart

**Annotation:** Pixel-level binary change masks (`cm.png` per region)
- `1` = changed (construction, demolition, road work, etc.)
- `0` = no change

**Directory structure after preprocessing:**
```
urban_train/
  images/
    <region>/
      imgs_1_rect/   ← time 1 TIF bands
      imgs_2_rect/   ← time 2 TIF bands
  train_labels/
    <region>/cm/cm.png
```

**Splits:** Official train/test split defined in `train.txt` / `test.txt` (auto-discovered if missing)

---

## 3. Data Pipeline

### Step 1 — `scripts/preprocess_oscd.py`
**Input:** Raw `.tif` Sentinel-2 bands per region  
**Output:** `.npy` arrays (float32, normalized)

What it does:
1. Reads band TIFs for time 1 and time 2
2. Stacks selected bands (RGB = bands 4, 3, 2 → indices 3, 2, 1)
3. Normalizes each band to `[0, 1]` by dividing by 10000 (Sentinel-2 reflectance scale), then clips
4. Saves `before.npy`, `after.npy`, and `mask.npy` per region into the output directory
5. Supports auto-discovery: if `train.txt`/`test.txt` not found, discovers regions from directory structure

### Step 2 — `scripts/extract_patches.py`
**Input:** `.npy` arrays from Step 1  
**Output:** `256×256` PNG image pairs in `processed_oscd/train/`, `processed_oscd/val/`, `processed_oscd/test/`

What it does:
1. Slides a `256×256` window over each region with `stride=128` (50% overlap)
2. Saves patch pairs as `A/0001.png` (before), `B/0001.png` (after), `label/0001.png` (mask)
3. Splits regions into train / val / test sets

**Why 50% overlap (stride=128)?** Increases training data diversity by generating overlapping patches. Boundary regions contribute to more samples.

### Step 3 — `utils/oscd_dataset.py`
DataLoader pipeline for training:
- Loads PNG patch pairs
- Applies data augmentation during training (random flips, rotations)
- Returns `{"image1": Tensor, "image2": Tensor, "mask": Tensor}` per sample

---

## 4. Full Architecture

```
img1 (B, 3, 256, 256) ──┐
                          ├──► SiameseViTEncoder (SHARED WEIGHTS) ──► feat1 (B, 256, 768)
img2 (B, 3, 256, 256) ──┘                                         └──► feat2 (B, 256, 768)
                                                                               │
                                                              FeatureDifferenceModule
                                                            (concat_project strategy)
                                                                               │
                                                                    diff_feat (B, 256, 256)
                                                                               │
                                                                  ProgressiveDecoder
                                                        (4 × bilinear upsample stages)
                                                                               │
                                                         change_logits (B, 1, 256, 256)
                                                                               │
                                                                          Sigmoid
                                                                               │
                                                         change_prob   (B, 1, 256, 256)
                                                                               │
                                                                   threshold @ 0.5
                                                                               │
                                                         binary_mask   (B, 1, 256, 256)  {0, 1}
```

### SiameseViTEncoder (expanded)
```
(B, 3, 256, 256)
  │
  ▼  PatchEmbedding
  │   Conv2d(3, 768, kernel=16, stride=16) ──► (B, 768, 16, 16)
  │   flatten(2) + transpose(1,2)          ──► (B, 256, 768)
  │   prepend [CLS] token                  ──► (B, 257, 768)
  │   + learnable pos_embed (257, 768)     ──► (B, 257, 768)
  │   Dropout(0.1)
  │
  ▼  × 12  TransformerEncoderBlock  (Pre-LN)
  │   ┌─────────────────────────────┐
  │   │ LayerNorm(768)              │
  │   │ MultiHeadAttention          │
  │   │   QKV: Linear(768 → 2304)  │
  │   │   12 heads, head_dim = 64  │
  │   │   scores / √64             │
  │   │   Softmax (attention wts)  │
  │   │   Dropout(attn_dropout)    │
  │   │   weighted sum of V        │
  │   │   proj: Linear(768 → 768) │
  │   │   Dropout(0.1)             │
  │   │ + residual                 │
  │   │                            │
  │   │ LayerNorm(768)             │
  │   │ MLP                        │
  │   │   Linear(768 → 3072) GELU │
  │   │   Dropout(0.1)             │
  │   │   Linear(3072 → 768)      │
  │   │   Dropout(0.1)             │
  │   │ + residual                 │
  │   └─────────────────────────────┘
  │
  ▼  LayerNorm(768)
  │  drop [CLS] → x[:, 1:]
  ▼
(B, 256, 768)   ← patch features (one per 16×16 region)
```

---

## 5. Every Layer Explained

### 5.1 PatchEmbedding
| Component | Detail |
|-----------|--------|
| `proj` | `Conv2d(3, 768, kernel_size=16, stride=16)` — projects each 16×16 patch to 768-dim vector |
| `cls_token` | Learnable parameter `(1, 1, 768)` — global class/representation token |
| `pos_embed` | Learnable parameter `(1, 257, 768)` — one position per patch + CLS |
| `dropout` | `Dropout(0.1)` |
| Init | Conv weight: `xavier_uniform`; pos_embed & CLS: `trunc_normal(std=0.02)` |

**Why Conv2d = patch splitting:** A Conv2d with `kernel_size=P, stride=P` is mathematically identical to splitting into P×P patches, flattening each, and multiplying by a weight matrix. No overlap because stride = kernel size.

**Why learned positional embeddings?** Self-attention is permutation-equivariant — shuffling patches gives the same output. Positional embeddings inject spatial location so the model knows *where* each patch is.

---

### 5.2 MultiHeadAttention
| Component | Detail |
|-----------|--------|
| `qkv` | `Linear(768, 2304, bias=True)` — fused projection for Q, K, V |
| `proj` | `Linear(768, 768)` — output projection after head concatenation |
| `attn_dropout` | `Dropout(0.0)` on attention weights |
| `proj_dropout` | `Dropout(0.1)` on output |
| num_heads | 12 |
| head_dim | 64  (768 / 12) |
| scale | 1 / √64 = 0.125 |

**Computation (per forward pass, B=batch, N=257 tokens):**
```
qkv  = Linear(x)       → (B, N, 2304)
     → reshape          → (B, N, 3, 12, 64)
     → permute          → (3, B, 12, N, 64)
Q, K, V = unbind(0)    each: (B, 12, N, 64)

scores = Q @ Kᵀ / √64  → (B, 12, N, N)
attn   = softmax(scores) → (B, 12, N, N)  ← attention weights (sum to 1 per row)
out    = attn @ V        → (B, 12, N, 64)
       → reshape          → (B, N, 768)
       → proj Linear      → (B, N, 768)
```

**Why scale by √d_k?** Without scaling, dot products grow as √d_k (for random unit vectors with d_k=64, std≈8). This causes softmax saturation (near-one-hot distribution), making gradients vanish. Dividing by √d_k restores unit variance.

**Why fused QKV?** One `Linear(D, 3D)` is faster than three `Linear(D, D)` — a single large GEMM is more GPU-efficient.

---

### 5.3 MLP (Feed-Forward Network inside Transformer Block)
| Component | Detail |
|-----------|--------|
| `fc1` | `Linear(768, 3072)` — expand 4× |
| `act` | `GELU()` |
| `fc2` | `Linear(3072, 768)` — project back |
| `dropout` | `Dropout(0.1)` after each linear |

Shape trace: `(B, N, 768) → (B, N, 3072) → GELU → (B, N, 768)`

**Why 4× expansion?** The wider hidden layer acts as a key-value memory: fc1 rows are "pattern detectors" (keys), fc2 columns are "output directions" (values). 4× gives enough capacity to detect many different feature patterns per token.

**Why GELU not ReLU?** GELU is smooth everywhere (no sharp corner at 0), giving a smoother loss landscape. Also acts as a stochastic regularizer: GELU(x) = x × Φ(x), where Φ is the normal CDF, probabilistically gating values by magnitude.

---

### 5.4 TransformerEncoderBlock (Pre-LN)
```
x  ──────────────────────────────┐
│  LayerNorm → MHSA → Dropout   │ residual
└──────────────────────► + ◄────┘
                          │
x' ──────────────────────────────┐
│  LayerNorm → MLP → Dropout    │ residual
└──────────────────────► + ◄────┘
```

**Pre-LN vs Post-LN:**
- Pre-LN: `output = x + Sublayer(LayerNorm(x))` — identity always in gradient path, more stable
- Post-LN: `output = LayerNorm(x + Sublayer(x))` — LayerNorm Jacobian multiplies everything, can distort gradients in deep models

**Why residual connections?** Without them, gradients must pass through all L Jacobians multiplicatively → vanish for deep networks. With residuals, gradient ∂output/∂x = I + ∂f/∂x — the identity term gives gradients a direct highway. A 12-layer ViT with residuals has 2¹² = 4096 implicit gradient paths (Veit et al., 2016).

---

### 5.5 LayerNorm
Formula: `LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β`

Where μ, σ² are computed over the **feature dimension** (last dim) for each token independently.

| Component | Detail |
|-----------|--------|
| `gamma` (γ) | Learnable scale, init = 1 |
| `beta` (β) | Learnable shift, init = 0 |
| `eps` | 1e-6 (prevents division by zero) |

**Why LayerNorm not BatchNorm for Transformers?**
- BatchNorm statistics depend on batch size → breaks for batch_size=1 or variable-length sequences
- LayerNorm normalizes per-token → batch-size independent, same at train and inference time

---

### 5.6 FeatureDifferenceModule
**Strategy used:** `concat_project`

| Input | Detail |
|-------|--------|
| feat1 | (B, 256, 768) — before image features |
| feat2 | (B, 256, 768) — after image features |

**Operation:**
```
combined = concat([feat1, feat2, feat1−feat2, feat1×feat2], dim=-1)  → (B, 256, 3072)

MLP:
  Linear(3072 → 768)  → GELU → Dropout(0.1)
  Linear(768  → 256)  → GELU → Dropout(0.1)
Output: (B, 256, 256)
```

**Why four concatenated signals?**
- `feat1`: what the scene looked like before
- `feat2`: what the scene looks like after
- `feat1 − feat2`: signed change direction (sign matters: building appears vs disappears)
- `feat1 × feat2`: element-wise similarity (high value = token unchanged, low = changed)

Together they give the MLP the richest possible change signal.

**Other available strategies:**
- `subtract`: `|feat1−feat2|` → MLP (loses sign, loses individual context)
- `attention`: cross-attention of feat1 over feat2 → concat with `|feat1−feat2|` → project (better for spatial displacement but more parameters)

---

### 5.7 ProgressiveDecoder
**Input:** `(B, 256, 256)` patch tokens (one per 16×16 pixel region)  
**Output:** `(B, 1, 256, 256)` full-resolution logits

```
(B, 256, 256)
  ▼  initial_proj: Linear(256 → 512) → GELU
(B, 256, 512)
  ▼  reshape: transpose + view
(B, 512, 16, 16)   ← treats 16×16 token grid as 2D spatial feature map
  ▼  up1: Bilinear×2 → Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU
(B, 128, 32, 32)
  ▼  up2: same pattern
(B, 64, 64, 64)
  ▼  up3: same pattern
(B, 32, 128, 128)
  ▼  up4: same pattern
(B, 16, 256, 256)
  ▼  final_conv: Conv2d(16, 1, kernel_size=1)
(B, 1, 256, 256)   ← raw logits
```

**Why bilinear upsample + conv instead of transposed conv?**  
Transposed convolutions can produce "checkerboard" artifacts due to uneven overlap when stride > 1 (Odena et al., 2016). Bilinear handles spatial interpolation cleanly; the subsequent conv learns local refinement.

**Why two convs per stage?**  
First conv fuses interpolated features. Second conv adds extra non-linear capacity at the current resolution before the next upsampling step — important for sharp boundary recovery.

---

### 5.8 Dropout

Dropout randomly zeroes individual activations with probability `p` at training time, then scales surviving activations by `1/(1-p)` to keep the expected sum constant. At inference, Dropout is a no-op.

| Location | p | Purpose |
|----------|---|---------|
| Embedding (after pos_embed) | 0.1 | Prevents overfitting to specific patch-position combinations |
| MLP (after fc1, after fc2) | 0.1 | Prevents co-adaptation of neurons in the feed-forward path |
| Output projection (after head concat in MHSA) | 0.1 | Regularizes the attention output path |
| Attention weights | 0.0 | Off — allows attention maps to be interpretable; small N=257 means regularization is less critical here |

**Why dropout helps transformers:** With N=257 tokens each attending to all others, the model can over-rely on specific token-to-token attention patterns. Dropout forces it to build redundant, distributed representations.

**Inverted dropout (what PyTorch does):**
```
At train time:  mask = Bernoulli(1-p),  output = (x * mask) / (1-p)
At eval time:   output = x
```
Dividing by `(1-p)` at train time means you never have to change anything at eval — weights see the same expected magnitude during both phases.

**Dropout is NOT applied to:**
- LayerNorm parameters (γ, β)
- The CLS token directly
- Positional embeddings directly

---

### 5.9 Weight Initialization

Initialization matters enormously for transformers — poor init causes attention collapse or gradient explosion from epoch 0.

| Component | Init Strategy | Why |
|-----------|--------------|-----|
| Linear layers (QKV, proj, MLP) | `trunc_normal(std=0.02)` | Small values keep attention logits near zero at epoch 0 (softmax near-uniform), avoid saturation |
| Conv2d (patch projection) | `xavier_uniform` | Keeps variance of activations stable across channels at init |
| LayerNorm γ | 1 | Starts as identity (no scaling) |
| LayerNorm β | 0 | Starts as identity (no shift) |
| CLS token | `trunc_normal(std=0.02)` | Small random init; learned during training |
| Positional embeddings | `trunc_normal(std=0.02)` | Small init allows the model to learn spatial patterns without bias |
| Biases | 0 | Standard; avoids breaking symmetry via bias asymmetry |

**Why `trunc_normal(std=0.02)`?**
- At init, the QKV weight matrix has std=0.02. For a token with std≈1, `QKᵀ` has std ≈ 0.02 × √768 ≈ 0.55 → attention logits are small → softmax outputs near-uniform attention → all tokens contribute equally to start. This is a neutral, safe starting point.
- "Truncated" = values beyond ±2σ are resampled. Avoids rare large-magnitude inits that break attention immediately.

---

### 5.10 What Multi-Head Attention Heads Actually Learn

Different heads in MHSA specialize over training. Common observed patterns in ViT:

| Head Type | What it attends to | Purpose |
|-----------|-------------------|---------|
| **Local heads** | Nearby patches (adjacent 16×16 regions) | Fine-grained spatial structure |
| **Global heads** | Patches anywhere in the image | Semantic relationships (sky-with-buildings) |
| **Diagonal heads** | Self-attention (each token to itself) | Pass-through / identity |
| **[CLS] heads** | [CLS] attends to semantically relevant patches | Summary aggregation |
| **Induction heads** | Pattern completion: if A→B appeared before, attend to B when seeing A again | In-context learning capability |

**Why 12 heads for 768 dimensions?**
768 / 12 = 64 dimensions per head. This is the ViT-Base sweet spot from the original paper — each head has enough capacity (64-dim Q, K, V) to represent a distinct attention pattern while keeping total parameter count manageable. More heads = more diversity; fewer dimensions per head = less capacity per head.

**Attention weight matrix visualization:** At each layer, the (N×N) = (257×257) attention matrix shows which patches attend to which. By epoch ~20, you typically see:
- Sparse diagonal structure (each patch mostly attends to itself + neighbors)
- CLS row attending broadly across the image
- Certain heads becoming "background suppression" heads (no-change regions push their keys apart)

---

## 6. Activation Functions

| Function | Location | Formula | Why Used |
|----------|----------|---------|---------|
| **GELU** | MLP blocks (transformer), FeatureDifferenceModule, initial_proj in decoder | `x × Φ(x)` where Φ = normal CDF | Smooth, differentiable everywhere; better than ReLU for transformers; acts as stochastic regularizer |
| **ReLU** | ProgressiveDecoder upsampling stages | `max(0, x)` | Standard for CNN feature maps; fast; no saturation for positive values |
| **Softmax** | MultiHeadAttention (attention weights) | `exp(xᵢ)/Σexp(xⱼ)` along dim=-1 | Converts raw scores to probability distribution over positions (each row sums to 1) |
| **Sigmoid** | Final prediction (inference) | `1/(1+e^{-x})` | Converts logits to probabilities in [0,1] for binary change map |

**Note:** Sigmoid is NOT applied during training — `BCEWithLogitsLoss` and `BCEDiceLoss` take raw logits for numerical stability (fused log-sum-exp computation avoids overflow).

**GELU vs ReLU detail:**
```
GELU(1.0) = 0.8413    ReLU(1.0) = 1.0
GELU(0.0) = 0.0       ReLU(0.0) = 0.0
GELU(-1.0) = -0.1587  ReLU(-1.0) = 0.0
```
GELU passes negative values slightly (unlike hard ReLU cutoff) → less "dead neuron" problem.

---

## 7. Loss Functions

### 7.1 BCEDiceLoss (default, used in training)

```
Loss = bce_weight × BCE + dice_weight × (1 − Dice)
     = 0.5 × BCE + 0.5 × (1 − Dice)
```

**BCE (Binary Cross-Entropy with Logits):**
```
BCE = −[t × log(σ(x)) + (1−t) × log(1−σ(x))]
```
With `pos_weight = 5.0` to upweight the rare changed class:
```
BCE_weighted = −[5.0 × t × log(σ(x)) + (1−t) × log(1−σ(x))]
```

**Soft Dice Loss:**
```
Dice = (2 × Σ(p×t) + ε) / (Σp + Σt + ε)
Loss_dice = 1 − Dice
```
Where `p = sigmoid(logits)` (soft probabilities, not binary), `t` = ground truth, `ε = 1.0` (Laplace smoothing)

**Why combine BCE + Dice?**
- BCE: per-pixel term, gives dense gradient signal
- Dice: region-level term, directly optimizes overlap F1 (handles imbalance by normalizing by predicted area)
- BCE alone can trivially minimize loss by predicting all 0s on imbalanced data
- Dice alone can be noisy for very small objects
- Together they balance pixel accuracy and region overlap

**Why pos_weight=5.0?** Changed pixels are ~5-20× rarer than unchanged. Multiplying their loss by 5 forces the model to treat false negatives as 5× more costly than false positives.

---

### 7.2 FocalDiceLoss (alternative for severe imbalance)

```
FL(p) = −α(1−p)^γ log(p)     for positives (t=1)
FL(p) = −(1−α)p^γ log(1−p)   for negatives (t=0)

Loss = focal_weight × FL + dice_weight × Dice_loss
```

Parameters: α=0.25, γ=2.0 (default)

**Why Focal?** `(1−p)^γ` is a modulating factor that down-weights easy examples (model already confident). With γ=2, an easy example with p=0.9 contributes only (0.1)² = 1% of the loss. Hard examples dominate training → better performance on rare change pixels.

---

## 8. Evaluation Metrics

All metrics are **pixel-level binary classification** where positive = change.

| Symbol | Meaning |
|--------|---------|
| TP | Predicted change AND actually changed |
| FP | Predicted change but NOT changed |
| FN | Missed change (predicted no-change but actually changed) |
| TN | Predicted no-change AND actually not changed |

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Precision** | TP / (TP + FP) | Of pixels predicted changed, how many actually changed? |
| **Recall** | TP / (TP + FN) | Of all actual changes, how many did we detect? |
| **F1 Score** | 2×P×R / (P+R) | Harmonic mean — balances precision and recall. **Primary metric.** |
| **IoU / Jaccard** | TP / (TP+FP+FN) | Overlap between predicted and ground-truth change mask |
| **Accuracy** | (TP+TN)/(TP+FP+FN+TN) | Pixel-level correctness (misleadingly high with imbalanced data) |
| **Cohen's Kappa** | (p_o − p_e)/(1 − p_e) | Agreement beyond chance (more reliable than accuracy for imbalanced) |

**Best model checkpoint** is saved when val-F1 improves.

**Why F1, not accuracy?**  
If 95% of pixels are "no change", a model that always predicts 0 gets 95% accuracy — but F1=0. F1 requires actually detecting changes.

**IoU vs F1:**  
IoU = TP/(TP+FP+FN) = F1/(2−F1). IoU is harsher than F1 (penalizes FP and FN equally). IoU=0.5 corresponds to F1=0.667.

---

## 9. Optimizer & Learning Rate Schedule

### Optimizer: AdamW

```
m_t = β₁ × m_{t-1} + (1−β₁) × g_t          (first moment, momentum)
v_t = β₂ × v_{t-1} + (1−β₂) × g_t²         (second moment, adaptive LR)
m̂_t = m_t / (1−β₁ᵗ)                         (bias correction)
v̂_t = v_t / (1−β₂ᵗ)
θ_t = θ_{t-1} − lr × m̂_t / (√v̂_t + ε)     (parameter update)
θ_t -= lr × λ × θ_{t-1}                      (decoupled weight decay)
```

| Param | Value |
|-------|-------|
| lr | 1e-3 (peak) |
| β₁ | 0.9 |
| β₂ | 0.999 (default) |
| ε | 1e-8 (default) |
| weight_decay | 0.05 |

**AdamW vs Adam:** Adam applies weight decay through the gradient (L2 reg), which interacts incorrectly with adaptive learning rates (effectively a different decay per parameter). AdamW applies weight decay directly to the weights, decoupled from the gradient step — mathematically cleaner and works better with LR scheduling.

**What is weight-decayed?** All parameters EXCEPT biases, LayerNorm weights, CLS token, and positional embeddings (these are excluded from decay — standard practice).

---

### LR Schedule: Linear Warmup + Cosine Annealing

```
Epoch 0────────────10──────────────────────────────200
LR   1e-6 ──▲── 1e-3 ──────cosine decay──────── 1e-6
             warmup              cosine annealing
```

**Phase 1 — Linear Warmup (epochs 0 to 10):**
```
lr = min_lr + (max_lr − min_lr) × (epoch / warmup_epochs)
```

**Phase 2 — Cosine Annealing (epochs 10 to 200):**
```
progress = (epoch − warmup_epochs) / (total_epochs − warmup_epochs)
lr = min_lr + (1 − min_lr_ratio) × 0.5 × (1 + cos(π × progress)) × base_lr
```

**Why warmup?** At random initialization, attention weights are noisy (near-uniform softmax). A large LR immediately would push the model to a bad region. Warmup lets the model first form meaningful attention patterns with small gradient steps.

**Why cosine decay?** Smoothly decays from max to min, spending more time at lower LRs (fine-tuning regime). More effective than step decay for transformers.

---

## 10. Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `epochs` | 200 | Max training epochs |
| `batch_size` | 8 | Training batch (memory constraint) |
| `eval_batch_size` | 16 | Larger eval batch (no gradients stored) |
| `lr` | 1e-3 | Peak LR (after warmup) |
| `min_lr` | 1e-6 | LR floor at end of cosine decay |
| `warmup_epochs` | 10 | Linear warmup epochs |
| `weight_decay` | 0.05 | AdamW L2 regularization |
| `grad_clip` | 1.0 | Max gradient norm (prevents exploding gradients) |
| `patience` | 30 | Early stopping: epochs without F1 improvement |
| `loss` | bce_dice | BCE + Dice combined |
| `pos_weight` | 5.0 | BCE upweight for change pixels |
| `num_workers` | 8 | SLURM CPUs per task |
| `img_size` | 256 | Patch size (H = W) |
| `patch_size` | 16 | ViT patch size |

**Model hyperparameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `embed_dim` | 768 | Token embedding dimension (ViT-Base standard) |
| `depth` | 12 | Transformer blocks |
| `num_heads` | 12 | Attention heads |
| `mlp_ratio` | 4.0 | MLP hidden = 768 × 4 = 3072 |
| `dropout` | 0.1 | Applied in embedding, MLP, output projection |
| `attn_dropout` | 0.0 | Inside attention weights |
| `diff_type` | concat_project | Feature combination strategy |
| `diff_out_dim` | 256 | FeatureDifferenceModule output dim |
| `decoder_dims` | [128, 64, 32, 16] | Decoder stage channel counts |

---

## 11. Key Design Decisions & Why

### Why Siamese architecture?
Both images share the **exact same encoder weights**. This ensures both branches learn the same feature space — if they had separate weights, "changed" could be explained by the branches learning different representations rather than actual scene changes.

### Why ViT instead of CNN for the encoder?
- CNNs process local patches with fixed receptive fields — attention between distant regions requires many layers
- ViT self-attention directly computes relationships between ALL patch pairs in one layer
- For change detection, the model needs to understand global scene context (e.g., a road patch makes more sense when considering the surrounding neighborhood)

### Why patch_size=16?
- 256/16 = 16 → 16×16 = 256 patches (N=256)
- Attention matrix: 257×257 ≈ 66K entries per head (manageable)
- Smaller patches (8): N=1024, attention matrix 16× larger (too slow)
- Larger patches (32): N=64, too coarse for 256×256 change masks

### Why Pre-LN (Pre-Norm) not Post-LN?
Pre-LN: `x + f(LayerNorm(x))` — the skip connection is always clean, gradients flow directly.  
Post-LN: `LayerNorm(x + f(x))` — gradients of early layers are scaled by the LayerNorm Jacobian, causing instability in deep models (>6 layers). ViT paper actually used Post-LN but later work showed Pre-LN trains more stably from scratch.

### Why concat_project for feature difference?
Simple subtraction `|f1−f2|` loses: the sign of change, the individual context of each branch. Concatenating four signals (f1, f2, f1−f2, f1×f2) preserves all information and lets the MLP learn any combination.

### Why bilinear upsample not transposed convolution in decoder?
Transposed convolutions with stride > 1 produce checkerboard artifacts (Odena et al., 2016) because output pixels overlap unevenly. Bilinear interpolation handles spatial scaling cleanly; the subsequent conv learns spatial refinement.

### Why Mixed Precision Training?
`torch.amp.autocast` + `GradScaler` runs forward pass in float16 (faster, less memory) and backward in float32. GradScaler multiplies loss before backward to prevent float16 underflow, then unscales before optimizer step.

---

## 12. Ablation Studies

The project includes three ablation axes:

### Ablation 1: Feature Difference Strategy (`diff_type`)
| Strategy | Description | Expected Performance |
|----------|-------------|---------------------|
| `subtract` | `|f1−f2|` → 2-layer MLP | Baseline — loses sign and context |
| `concat_project` | `[f1, f2, f1−f2, f1×f2]` → 2-layer MLP | Best — richest signal |
| `attention` | Cross-attention of f1 over f2, then concat with `|f1−f2|` | Better for spatial displacement |

### Ablation 2: Transformer Block Normalization
| Variant | Description | Expected Impact |
|---------|-------------|----------------|
| `pre_norm` (default) | `x + f(LN(x))` | Most stable training |
| `post_norm` | `LN(x + f(x))` | Less stable, may diverge |
| `no_residual` | No skip connections | Vanishing gradients, likely fails |

### Ablation 3: Classification Token (for base ViT on CIFAR-10)
| Variant | Description |
|---------|-------------|
| `[CLS] token` (default) | Last-layer CLS aggregates all patches |
| `Global Average Pooling` | Mean of all patch tokens |

---

## 13. Parameter Count Breakdown

### Full model (ViT-Base equivalent encoder):

| Component | Parameters | % of Total |
|-----------|-----------|-----------|
| SiameseViTEncoder (~ViT-Base) | ~86M | ~96% |
| FeatureDifferenceModule | ~2.4M | ~2.7% |
| ProgressiveDecoder | ~0.9M | ~1% |
| **Total** | **~89M** | 100% |

### Encoder breakdown:

| Sub-component | Parameters |
|---------------|-----------|
| PatchEmbedding (Conv2d + CLS + pos_embed) | ~590K |
| 12 × TransformerEncoderBlock | ~85M |
| &nbsp;&nbsp;&nbsp;per block: | ~7.1M |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MHSA (QKV+proj): 768×2304+2304×768 = 4×768² | ~2.4M |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MLP (fc1+fc2): 768×3072+3072×768 = 2×768×3072 | ~4.7M |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;LayerNorm ×2: 2×768×2 | ~3K |
| Final LayerNorm | ~1.5K |

### Decoder breakdown:

| Component | Parameters |
|-----------|-----------|
| initial_proj Linear(256→512) | ~131K |
| up1 Conv blocks (512→128) | ~590K |
| up2 Conv blocks (128→64) | ~148K |
| up3 Conv blocks (64→32) | ~37K |
| up4 Conv blocks (32→16) | ~9K |
| final_conv Conv2d(16→1, 1×1) | ~17 |

---

## 14. Common Quiz Questions & Answers

**Q: What is the task this model solves?**  
A: Binary pixel-level change detection between two co-registered satellite images. For each pixel, the model predicts whether a change occurred between time 1 and time 2.

**Q: What dataset is used?**  
A: OSCD (Onera Satellite Change Detection) — bi-temporal Sentinel-2 satellite images of 24 urban areas worldwide, with pixel-level binary change annotations.

**Q: Why Siamese architecture?**  
A: Weight sharing between the two branches ensures both images are embedded into the same feature space. Without shared weights, feature differences could be artifacts of the two encoders learning different representations, not actual scene changes.

**Q: What is self-attention doing?**  
A: For each token (patch), it computes how much to attend to every other token. Q (query) asks "what am I looking for?", K (key) says "what do I contain?", V (value) says "what information do I provide?". The output is a weighted sum of all V vectors, weighted by Q·K softmax scores.

**Q: Why do we scale by √d_k in attention?**  
A: With d_k=64, dot products of random unit vectors have std=√64=8. Softmax with large inputs saturates (near-one-hot), making gradients vanish. Dividing by √d_k restores std≈1, keeping softmax in a well-behaved range.

**Q: What does the [CLS] token do?**  
A: It is a learnable token prepended to the sequence. Through 12 layers of self-attention, it attends to all patch tokens and accumulates global image information. For change detection, the CLS token is dropped — we use the per-patch tokens.

**Q: What is the difference between LayerNorm and BatchNorm?**  
A: BatchNorm normalizes across the batch dimension per feature; LayerNorm normalizes across the feature dimension per token. LayerNorm has no batch-size dependency, works identically at train and test time, and handles variable-length sequences — making it ideal for transformers.

**Q: Why GELU instead of ReLU?**  
A: GELU is smooth and differentiable everywhere (no sharp corner at 0). It acts as an input-dependent stochastic gate: GELU(x) = x × Φ(x), probabilistically blocking small values. Used by ViT, GPT, BERT — empirically better than ReLU for transformers.

**Q: What is the loss function?**  
A: BCEDiceLoss = 0.5 × BCE + 0.5 × Dice. BCE provides per-pixel supervision; Dice directly optimizes region overlap (F1). A pos_weight=5.0 upweights the rare changed class in BCE to counter class imbalance.

**Q: Why use Dice loss for change detection?**  
A: Change pixels are rare (often <10% of image). Pure BCE on such imbalanced data has trivially low loss even if the model predicts all zeros. Dice = 2×TP/(2TP+FP+FN) directly measures detection quality and is not affected by the large number of TN pixels.

**Q: Why is accuracy not the primary metric?**  
A: If 95% of pixels are unchanged, a model predicting all zeros gets 95% accuracy but detects nothing. F1 and IoU are invariant to true negatives, making them honest measures for imbalanced detection tasks.

**Q: What is Cohen's Kappa?**  
A: Kappa = (p_observed − p_expected) / (1 − p_expected). It measures agreement beyond what would be expected by chance. Kappa=0 means no better than random; Kappa=1 is perfect. More reliable than accuracy for imbalanced data.

**Q: How does the decoder convert patch tokens to a full-resolution mask?**  
A: The 256 patch tokens (one per 16×16 pixel region) are projected to 512 channels, reshaped to a 16×16 spatial grid, then progressively upsampled 4× (2× each stage) via bilinear interpolation + conv layers until reaching 256×256. A final 1×1 conv produces the single-channel logit map.

**Q: What is warmup and why is it needed?**  
A: Linear LR warmup ramps the learning rate from a small value (1e-6) to the target (1e-3) over the first 10 epochs. At initialization, attention weights are essentially random — large LR would push the model to a bad local minimum. Warmup lets the model form stable attention patterns before increasing the step size.

**Q: What is AdamW?**  
A: Adam optimizer with decoupled weight decay. Standard Adam applies weight decay through the gradient (equivalent to L2 regularization), which interacts incorrectly with the adaptive learning rate — effectively applying different weight decay rates per parameter. AdamW applies it directly to the weights, independent of the gradient, which is mathematically correct and empirically better.

**Q: What is gradient clipping and why is it used?**  
A: `clip_grad_norm_(params, max_norm=1.0)` scales down all gradients if their global L2 norm exceeds 1.0. Prevents "exploding gradients" — sudden large gradient spikes that can destabilize training, common in deep transformers.

**Q: What is mixed precision training?**  
A: Running the forward pass in float16 (faster computation, less GPU memory) and keeping the backward pass / optimizer in float32. `GradScaler` multiplies the loss before backward to prevent float16 underflow, then unscales before the optimizer step.

**Q: How many parameters does the model have?**  
A: Approximately 89 million total. ~86M in the shared ViT-Base encoder, ~2.4M in the feature difference module, ~0.9M in the decoder.

**Q: Is any pretrained model used?**  
A: No. The entire model is trained from random initialization on the OSCD dataset. Weight init follows ViT paper: `trunc_normal(std=0.02)` for linear layers and conv, `zeros` for biases, `ones`/`zeros` for LayerNorm gamma/beta.

**Q: Why extract 256×256 patches with 50% overlap (stride=128)?**  
A: The original Sentinel-2 images are large (thousands of pixels). Patching with overlap (stride=128 = 50% overlap) generates more training samples from each region and ensures boundary regions are well-represented.

---

### Architecture Deep-Dive Questions

**Q: What is "token mixing" vs "channel mixing" in a Transformer block?**  
A: These are two complementary operations:
- **Token mixing (MHSA):** Mixes information *across positions*. Each token's output is a weighted sum of all other tokens' values. It asks: "What context from other patches should I aggregate?"
- **Channel mixing (MLP):** Mixes information *across dimensions within each token*. Applied independently to every token at the same position. It asks: "Given this token's aggregated features, what higher-level representation should I compute?"  
The two operations are interleaved: token mixing → channel mixing → token mixing → …

**Q: What is the computational (memory) complexity of self-attention and why does it matter?**  
A: O(N²·D) in time and O(N²) in memory for the attention matrix, where N = sequence length, D = embedding dim. For our model N=257 (manageable). But for full-resolution images (N could be thousands), this quadratic scaling becomes prohibitive — a key limitation of vanilla ViT.

**Q: What is the receptive field of a ViT compared to a CNN?**  
A: A ViT has a **global receptive field from layer 1** — every patch can attend to every other patch immediately. A CNN builds receptive field gradually: a 3×3 conv has a 3×3 field, two stacked 3×3 convs give 5×5, etc. This is both ViT's strength (global context immediately) and weakness (lacks the local-first inductive bias that helps CNNs learn efficiently with less data).

**Q: Why does ViT use 1D positional embeddings for a 2D image?**  
A: Tokens are flattened to a 1D sequence (left-to-right, top-to-bottom), so 1D positional embeddings can capture the full (row, col) position implicitly — position i in the 1D sequence uniquely maps to row `i//W`, col `i%W`. The ViT paper found 1D learned embeddings slightly outperformed 2D variants because the model can learn any spatial relationship it finds useful, not just row/col factorization.

**Q: What happens to the [CLS] token during the 12 transformer blocks?**  
A: In each block, [CLS] participates in self-attention like any other token — it can attend to all patch tokens and patch tokens can attend to it. Over 12 layers, [CLS] progressively aggregates global information from all patches. In classification ViTs, the final [CLS] embedding is passed to a classification head. In our model we **discard** [CLS] after encoding and use the 256 patch tokens for dense prediction instead.

**Q: Why do we discard the [CLS] token in this project?**  
A: [CLS] aggregates a single global image-level embedding — useful for classification. For change detection we need per-pixel outputs, so we keep the 256 patch tokens (each representing one 16×16 spatial region). [CLS] is kept during the encoder because it contributes via cross-attention: patch tokens attend to [CLS] and accumulate global context through it.

**Q: What is the role of the QKV projection? Why not just use x directly for attention?**  
A: Raw token embeddings x are a single generic representation. Projecting to Q, K, V via separate learned matrices lets the model express three different "views" of each token:
- Q ("what am I looking for?"): specialized for computing compatibility with other tokens
- K ("what do I offer?"): specialized for being matched against
- V ("what information do I carry?"): specialized for what gets aggregated  
Using x for all three would constrain these roles to be identical, severely limiting expressiveness.

**Q: Can you walk through the shapes in the attention forward pass?**  
A: Starting from input `x` of shape `(B, N, D)` = `(B, 257, 768)`:
```
qkv = Linear(768, 2304)(x)          → (B, 257, 2304)
    → reshape (B, 257, 3, 12, 64)
    → permute (3, B, 12, 257, 64)
Q, K, V = unbind(dim=0)             each: (B, 12, 257, 64)

scores = Q @ K.transpose(-2,-1)     → (B, 12, 257, 257)
scores /= sqrt(64)                   → (B, 12, 257, 257)
attn = softmax(scores, dim=-1)      → (B, 12, 257, 257)  ← sums to 1 per row
out  = attn @ V                     → (B, 12, 257, 64)
     → transpose + reshape          → (B, 257, 768)
     → Linear(768, 768)             → (B, 257, 768)
```

**Q: Why is softmax applied along the last dimension (over keys) in attention?**  
A: Each query token computes a score against every key token. We want those scores to form a probability distribution — "how much of each value do I take?". Applying softmax over the key dimension (dim=-1) means each query's attention weights sum to 1, so the output is a convex combination of V vectors. If we applied it over queries, we'd be normalizing in the wrong direction.

**Q: What is the difference between depth (12 blocks) and width (768 dims)?**  
A: **Depth** = how many sequential transformation stages the model applies. Each block refines token representations by mixing information and applying non-linear transforms. More depth = more abstract, hierarchical representations. **Width** = how many dimensions each token carries. Wider tokens can represent more complex information per position. ViT-Base (12×768) uses the original paper's balanced setting. ViT-Large uses 24 blocks × 1024 dims; ViT-Huge uses 32 × 1280.

**Q: What would happen if we used Post-LN instead of Pre-LN?**  
A: Post-LN (`LN(x + f(x))`) was used in the original ViT paper but requires careful warmup and lower learning rates for stability. The LN Jacobian at init can distort gradients, causing some layers to receive very small or very large updates. In practice, models deeper than 6 layers trained from scratch often diverge with Post-LN unless heavily tuned. Pre-LN (`x + f(LN(x))`) maintains a clean gradient highway through the residual connection, making training significantly more stable.

**Q: What would happen without residual connections?**  
A: The gradient of the loss with respect to early layer parameters requires multiplying the Jacobians of all subsequent layers: ∂L/∂θ₁ = ∂L/∂θ₁₂ × J₁₂ × J₁₁ × … × J₁. For 12 layers with small Jacobians (common at init), this product rapidly approaches zero — **vanishing gradient**. With residuals, ∂(x+f(x))/∂x = I + ∂f/∂x. The identity term means gradient always has a direct path, regardless of how small ∂f/∂x is.

**Q: Why does the MLP use a 4× expansion ratio (768→3072)?**  
A: The MLP is interpreted as a differentiable key-value memory. The 3072 intermediate neurons each act as a "pattern detector" (fc1 rows = keys) that fires for specific input patterns. The fc2 columns are "output directions" (values) that add to the token representation when that pattern is detected. More intermediate neurons = more patterns the model can store. 4× is empirically validated across GPT, BERT, ViT — wide enough for rich representations without being too expensive.

---

### Data & Training Questions

**Q: Why is the OSCD dataset hard to train on from scratch?**  
A: Several compounding difficulties: (1) Only 14 training regions — extremely small dataset for a ViT-Base model with 89M parameters. (2) Severe class imbalance: often <5% changed pixels. (3) No pretrained weights, so the model must learn visual features + change semantics from scratch. (4) Sentinel-2 imagery has unique radiometric properties different from ImageNet — pretrained weights would be partially inappropriate anyway.

**Q: What is early stopping and when does it trigger here?**  
A: Training halts if val-F1 does not improve for `patience=30` consecutive epochs. The best checkpoint (highest val-F1 seen) is saved separately from `last_model.pth`. Early stopping prevents overfitting and saves GPU time when the model has plateaued. It also acts as implicit regularization.

**Q: What does it mean when F1=0.0000 throughout training?**  
A: The model is predicting all-zero masks (no change anywhere) or all-one maps. With BCE+Dice loss, a model stuck predicting all-zeros is a classic failure mode under class imbalance — BCE loss decreases simply by confidently predicting 0 everywhere, since the majority of pixels ARE 0. Dice then also shows 0 (2×TP=0 since TP=0). Root causes: missing positive patches, `pos_weight` too low, learning rate too high causing the model to collapse early.

**Q: What is gradient clipping and when does it activate?**  
A: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` computes the global L2 norm of all gradient tensors concatenated: `‖g‖ = √(Σᵢ gᵢ²)`. If this exceeds 1.0, all gradients are scaled down by `1.0 / ‖g‖`. It activates when a batch contains unusually large errors or when attention weights create large second-order effects. In transformers, gradient spikes can happen when attention suddenly shifts to a new pattern — clipping prevents this from destroying previously learned weights.

**Q: Why is `pos_weight` set to 20.0 in the latest run?**  
A: With ~0% positive patches (all-zero masks from the bug), `pos_weight` was irrelevant. But even with correct data, OSCD change masks are heavily imbalanced (~5-15% change). `pos_weight=20.0` tells BCE: "a false negative on a changed pixel is 20× more costly than a false positive on an unchanged pixel." This forces the model to actively predict change rather than defaulting to all-zero. The right value depends on the actual imbalance ratio in your data.

**Q: What is the difference between val loss decreasing and F1 improving?**  
A: They can diverge! Val loss decreasing means the model's predicted probabilities are moving in the right direction (pushing changed pixels toward 1, unchanged toward 0). But F1 only improves when the model's **binary predictions** (prob > 0.5 → 1) are actually correct detections. A model with decreasing loss but F1=0 is making progress on calibration but hasn't crossed the 0.5 decision threshold to start predicting any change pixels.

**Q: What is the purpose of the validation set in this pipeline?**  
A: Val set (2 held-out regions) serves three functions: (1) `best_model.pth` checkpoint is saved at peak val-F1 — prevents saving an overfit model. (2) Early stopping signal — if val-F1 plateaus, halt training. (3) Hyperparameter tuning proxy — loss function, learning rate, pos_weight choices are evaluated on val, not test.

**Q: What data augmentation is applied and why?**  
A: Training augmentations (via `albumentations`): random horizontal/vertical flips, random 90° rotations. **Critical constraint:** the same augmentation must be applied identically to `image1`, `image2`, and `mask` simultaneously — if you flip image1 but not image2, the spatial correspondence breaks and the mask no longer aligns. This is why `albumentations` with multi-image support is used rather than standard `torchvision` transforms.

**Q: Why normalize Sentinel-2 images with 2nd–98th percentile clipping?**  
A: Sentinel-2 reflectance values span a huge range due to clouds, shadows, water bodies, and atmospheric effects. Hard 0–10000 normalization would compress the useful range. Percentile clipping (2nd to 98th) removes extreme outliers (cloud highlights, deep shadows) and maps the bulk of the distribution to [0, 255]. This is robust normalization, similar to histogram stretching in remote sensing.

---

### Change Detection Specific Questions

**Q: How is this different from standard image segmentation?**  
A: In standard segmentation, one image is given and pixels are classified into categories (road, building, sky). In change detection, **two images are given** (bi-temporal) and the task is to identify pixels that changed *between the two dates*. The model must jointly reason about both images and learn what "change" means (new building, demolished structure, land use shift) vs what doesn't count as change (seasonal vegetation, lighting differences).

**Q: What kinds of changes does the model detect vs miss?**  
A: OSCD labels mark **land use changes**: new buildings, demolished structures, road construction. The model is not trained to detect: vegetation change (seasonal cycles appear as change to a naive model but are labeled no-change), illumination differences, registration misalignment (images may not be pixel-perfectly co-registered).

**Q: What is the risk of a Siamese model memorizing images rather than detecting change?**  
A: If the two branches had *different* weights, a model could minimize training loss by learning "branch 1 activations look like X, branch 2 activations look like Y, difference = change" — encoding branch identity rather than actual change. **Shared weights prevent this**: both images are projected into the same feature space, so differences in token embeddings genuinely reflect semantic scene differences, not encoder artifacts.

**Q: Why do we use absolute difference `|f1−f2|` AND signed difference `f1−f2`?**  
A: `|f1−f2|` captures *magnitude* of change but loses direction — a new building (f2 > f1) and demolished building (f1 > f2) look identical. `f1−f2` (signed) preserves this directionality. Together with `f1×f2` (similarity) and the individual embeddings, the MLP can learn to distinguish appearance vs disappearance of structures.

**Q: Could we use cross-attention between the two image features instead of a difference module?**  
A: Yes — this is the `attention` strategy in the `diff_type` ablation. Cross-attention lets feat2 query over feat1 positions (and vice versa), potentially handling spatial misalignment better (a structure slightly shifted between dates would still be matched via attention). The trade-off: cross-attention is O(N²) and adds more parameters. The `concat_project` approach is simpler, more interpretable, and comparable in performance for well-registered images.

---

### Loss & Metrics Deep Dive

**Q: What is the Dice coefficient and where does the formula come from?**  
A: Dice = 2|A∩B| / (|A|+|B|), measuring overlap between two sets A and B. For predictions vs ground truth: A = predicted change pixels, B = actual change pixels. A∩B = TP. So:
```
Dice = 2×TP / (2×TP + FP + FN)
     = F1 score
```
Dice and F1 are identical for binary classification! The Dice *loss* = 1 − Dice, minimized when Dice is maximized (best overlap).

**Q: What is the smooth/epsilon term in Dice loss and why ε=1.0?**  
A: Without ε: if a batch has no positive pixels at all (all background), both numerator and denominator are 0 → undefined (NaN). ε is added to both: `(2ΣTP + ε) / (Σpred + Σgt + ε)`. ε=1.0 (Laplace smoothing) is chosen to be numerically significant — small ε like 1e-6 still causes instability when predictions and targets are both near-zero. ε=1.0 means a batch with zero predictions and zero targets returns Dice=1/(0+0+1)=1.0... actually Dice = (0+1)/(0+0+1) = 1.0, which is fine (empty prediction on empty target = perfect).

**Q: What does Cohen's Kappa measure and when is it better than F1?**  
A: Kappa = (observed agreement − chance agreement) / (1 − chance agreement). It accounts for the probability that a random predictor would agree by chance. For highly imbalanced data, even F1 can be inflated if you get lucky by predicting all-1s on a class-heavy split. Kappa normalizes this. Kappa=0 = random, Kappa=1 = perfect, Kappa<0 = worse than random.

**Q: Why is Focal Loss useful for severe imbalance?**  
A: Standard BCE treats every pixel's loss equally. Focal Loss adds a modulating factor `(1−p)^γ`: easy examples (model already confident, p=0.95) contribute `(0.05)² = 0.25%` of their normal loss. Hard examples dominate. This focuses training on difficult, ambiguous pixels (often the change boundary pixels) rather than the easy background. γ=2 is the standard from the original RetinaNet paper.

---

## 15. Resources & References

### Papers
| Paper | Why Relevant |
|-------|-------------|
| Dosovitskiy et al. (2020) — "An Image is Worth 16×16 Words" | Original ViT paper — architecture basis |
| Lin et al. (2017) — "Focal Loss for Dense Object Detection" | Focal loss theory used in FocalDiceLoss |
| Daudt et al. (2018) — "Fully Convolutional Siamese Networks for Change Detection" | Siamese architecture for change detection context |
| Odena et al. (2016) — "Deconvolution and Checkerboard Artifacts" | Motivation for bilinear upsample in decoder |
| Veit et al. (2016) — "Residual Networks Behave Like Ensembles of Relatively Shallow Networks" | Residual connections as ensemble of 2ᴸ paths |

### Dataset
| Resource | Detail |
|----------|--------|
| OSCD Dataset | Onera Satellite Change Detection — 24 bi-temporal Sentinel-2 city pairs |
| Sentinel-2 | ESA satellite, 13 multispectral bands, 10m resolution (RGB bands) |

### Libraries / Tools
| Library | Use |
|---------|-----|
| PyTorch | Deep learning framework |
| torch.amp | Mixed precision training (autocast + GradScaler) |
| TensorBoard | Training curve logging (`SummaryWriter`) |
| NumPy | Array operations in preprocessing |
| GDAL / rasterio | Reading `.tif` Sentinel-2 files |
| SLURM | HPC job scheduling on Explorer cluster |

### HPC Config
| Detail | Value |
|--------|-------|
| Cluster | Explorer (Cofc / institutional HPC) |
| Partition | `gpu` |
| GPU | 1 GPU per job |
| CPUs | 8 |
| Memory | 64 GB |
| Time limit | 8 hours |
| Module | `anaconda3/2024.06` |
| Conda env | `vit-cd` |

---

---

## 16. Full Forward Pass Walkthrough

A complete trace through the model for one batch `B=2`, two images of size `256×256`, RGB.

```
INPUT
  img1: (2, 3, 256, 256)    ← batch of 2 "before" images, 3 RGB channels
  img2: (2, 3, 256, 256)    ← batch of 2 "after" images

══════════════════════════════════════════════════
  SHARED SIAMESE ENCODER (run twice, same weights)
══════════════════════════════════════════════════

Step 1: PatchEmbedding
  Conv2d(3→768, kernel=16, stride=16)
    img1: (2,3,256,256) → (2,768,16,16)
  flatten spatial: (2,768,256) → transpose: (2,256,768)
  ← 256 patch tokens, each = 768-dim projection of one 16×16 region

Step 2: Prepend [CLS] token
  cls_token: (1,1,768) → expand: (2,1,768)
  concat: (2,257,768)   ← position 0 = CLS, positions 1-256 = patches

Step 3: Add positional embeddings
  pos_embed: (1,257,768) → broadcast add
  x: (2,257,768)         ← tokens now know their spatial position
  Dropout(0.1)

Step 4: × 12 TransformerEncoderBlock
  [Each block — input/output: (2,257,768)]

  4a. Pre-LN 1: LayerNorm(768) over last dim → (2,257,768)

  4b. MultiHeadAttention:
      qkv = Linear(768,2304)   → (2,257,2304)
          → reshape             → (2,257,3,12,64)
          → permute             → (3,2,12,257,64)
      Q,K,V each:               (2,12,257,64)
      scores = Q@Kᵀ / 8.0     → (2,12,257,257)  ← 257×257 attention matrix
      attn = softmax(dim=-1)   → (2,12,257,257)  ← rows sum to 1
      out = attn @ V           → (2,12,257,64)
          → reshape             → (2,257,768)
      proj = Linear(768,768)   → (2,257,768)
      Dropout(0.1)

  4c. Residual add: x = x + out   → (2,257,768)

  4d. Pre-LN 2: LayerNorm(768)    → (2,257,768)

  4e. MLP:
      fc1 = Linear(768,3072)+GELU → (2,257,3072)
      Dropout(0.1)
      fc2 = Linear(3072,768)      → (2,257,768)
      Dropout(0.1)

  4f. Residual add: x = x + MLP  → (2,257,768)

Step 5: Final LayerNorm(768)      → (2,257,768)

Step 6: Drop [CLS], keep patches
  x[:, 1:, :]                    → (2,256,768)

  feat1 = encoder(img1)          → (2,256,768)
  feat2 = encoder(img2)          → (2,256,768)   ← same encoder weights

══════════════════════════════════════════════════
  FEATURE DIFFERENCE MODULE
══════════════════════════════════════════════════

Step 7: Construct 4-part signal
  diff   = feat1 − feat2         → (2,256,768)
  prod   = feat1 × feat2         → (2,256,768)
  combined = cat([feat1, feat2, diff, prod], dim=-1)  → (2,256,3072)

Step 8: MLP projection
  Linear(3072,768) + GELU        → (2,256,768)
  Dropout(0.1)
  Linear(768,256) + GELU         → (2,256,256)
  Dropout(0.1)

  diff_feat:                      (2,256,256)

══════════════════════════════════════════════════
  PROGRESSIVE DECODER
══════════════════════════════════════════════════

Step 9: Initial projection
  Linear(256,512) + GELU         → (2,256,512)

Step 10: Reshape to 2D spatial
  transpose(1,2) → (2,512,256)
  view(-1,512,16,16)             → (2,512,16,16)  ← 16×16 grid of 512-ch features

Step 11: Upsample stage 1
  Bilinear ×2                    → (2,512,32,32)
  Conv3×3→128 + BN + ReLU       → (2,128,32,32)
  Conv3×3→128 + BN + ReLU       → (2,128,32,32)

Step 12: Upsample stage 2
  Bilinear ×2                    → (2,128,64,64)
  Conv3×3→64 + BN + ReLU        → (2,64,64,64)
  Conv3×3→64 + BN + ReLU        → (2,64,64,64)

Step 13: Upsample stage 3
  Bilinear ×2                    → (2,64,128,128)
  Conv3×3→32 + BN + ReLU        → (2,32,128,128)
  Conv3×3→32 + BN + ReLU        → (2,32,128,128)

Step 14: Upsample stage 4
  Bilinear ×2                    → (2,32,256,256)
  Conv3×3→16 + BN + ReLU        → (2,16,256,256)
  Conv3×3→16 + BN + ReLU        → (2,16,256,256)

Step 15: Final prediction
  Conv2d(16→1, kernel=1)         → (2,1,256,256)  ← raw logits

OUTPUT
  logits: (2,1,256,256)    → BCEDiceLoss(logits, mask) during training
  probs = sigmoid(logits)  → (2,1,256,256)         during inference
  pred = probs > 0.5       → (2,1,256,256)  {0,1}  binary change map
```

**Total FLOPs rough estimate:**
- Patch embedding: 256 patches × 16×16×3×768 ≈ 150M
- 12 attention blocks: each 2×257²×768 ≈ 100M → 1.2B total
- 12 MLP blocks: each 2×257×768×3072×2 ≈ 770M total
- Decoder: ~80M
- **Grand total: ~2B FLOPs per forward pass** (per image pair)

---

## 17. ViT Intuitions & Mental Models

### 17.1 "An Image is Worth 16×16 Words"
The core ViT insight: treat an image like a sentence. Just as NLP models process a sentence as a sequence of word tokens, ViT processes an image as a sequence of patch tokens. Self-attention then captures relationships between patches the same way it captures relationships between words — without needing to know in advance that nearby words/patches are more related.

### 17.2 Why Self-Attention is Powerful
Think of self-attention as a learned, input-dependent aggregation. Instead of applying the same convolution filter everywhere (CNN), attention computes a *different* weighted sum of all other positions for each position, and those weights change based on the actual content of the image.

Example: for a change detection task, a patch showing "new concrete" in the after-image should strongly attend to the same spatial position in the before-image AND to neighboring context patches. Attention learns this naturally; a CNN would need many layers to propagate that context.

### 17.3 The Residual Stream View
Think of the transformer as a **residual stream**: a vector of size 768 flows through the network, and each attention head and MLP *adds* to it rather than replacing it. Each block enriches the representation incrementally.

```
x₀  (embedding)
x₁ = x₀ + MHSA₁(LN(x₀)) + MLP₁(LN(x₀ + MHSA₁))
x₂ = x₁ + MHSA₂(LN(x₁)) + MLP₂(...)
...
x₁₂ = final representation
```

The residual stream can be decomposed into additive contributions from each head at each layer — interpretability researchers read off "what did each head write to the stream."

### 17.4 Token Routing in Change Detection
In a change detection model, here is what the residual stream should look like after 12 blocks:

- **Early blocks (1–4):** Local patch features — edges, textures, colors in each 16×16 region
- **Middle blocks (5–8):** Semantic features — this patch is "building", "road", "vegetation"
- **Late blocks (9–12):** Relational features — this patch is in the same location as another region, that region has changed since before

The FeatureDifferenceModule then takes the final-layer patch tokens and extracts the change signal.

### 17.5 The Decoder as a "Spatial Inverse" of the Encoder
The encoder collapses spatial resolution: 256×256 pixels → 16×16 patches. The decoder inverts this: 16×16 patch tokens → 256×256 pixel mask. But it doesn't perfectly invert — it learns to *fill in* the fine-grained spatial details using the coarse patch-level change signal plus the learned convolutional filters. The two convs per upsample stage are the decoder's "inference" about what happened at sub-patch resolution.

---

## 18. Inductive Bias — CNN vs ViT

**Inductive bias** is the set of assumptions a model architecture makes about the data, independent of training.

| Bias Type | CNN | ViT |
|-----------|-----|-----|
| **Translation equivariance** | Yes (convolution is shift-equivariant) | No (learned position embeddings break this) |
| **Locality** | Yes (3×3 conv = only neighbors interact) | No (every patch attends to every other from layer 1) |
| **Parameter sharing** | Yes (same filter applied everywhere) | Only within each linear layer |
| **Spatial hierarchy** | Yes (pooling builds coarse-to-fine) | Implicit (later layers can attend more broadly) |

**Consequences:**
- CNNs are very data-efficient: their inductive biases match natural images well, so they need fewer samples to learn good representations.
- ViTs need more data to learn what CNNs get for free (locality, translation equivariance) — but once they have enough data, they can surpass CNNs because they're not *constrained* to local processing.
- This is why pretrained ViTs (on ImageNet-21k or JFT-3B) dramatically outperform trained-from-scratch ViTs, especially on small datasets like OSCD (24 regions).

**The implication for this project:** Training ViT-Base (~89M params) from scratch on ~130 patches is extremely challenging. A pretrained encoder or a smaller architecture (ViT-Small, ViT-Tiny) would be more practical for real research. The "from scratch" constraint is a pedagogical choice demonstrating that it's *possible*, not that it's optimal.

---

## 19. Common Failure Modes & Debugging

### F1 = 0.0000 throughout training
**Symptoms:** Val F1 never moves from 0, loss slowly decreases, early stopping triggers.  
**Root cause:** Model is predicting all-zeros (no change) everywhere.  
**Checklist:**
1. Are there positive patches? Check `pos_avail` in patch extraction output. If 0, mask.npy is missing or all-zero.
2. Is the labels path correct? Verify `cm.png` exists and is non-trivial (`mask.sum() > 0`).
3. Is `pos_weight` sufficient? With severe imbalance, try 10–30.
4. Is the learning rate too high? Try 1e-4 with longer warmup.

### Loss oscillates, F1 never stabilizes
**Symptoms:** Train loss spikes randomly, val F1 jumps between 0 and small values.  
**Root cause:** LR too high, or batch size too small (noisy gradient estimates).  
**Fix:** Lower LR, increase batch size, increase warmup epochs, add more grad_clip.

### Val loss increases while train loss decreases (overfitting)
**Symptoms:** Train loss ↓ smoothly, val loss ↓ then ↑ after ~50 epochs.  
**Root cause:** 89M parameters, 130 training patches — model memorizes training patches.  
**Fix:** More dropout, stronger weight decay (try 0.1), data augmentation, smaller model.

### CUDA out of memory
**Symptoms:** `RuntimeError: CUDA out of memory` during forward pass.  
**Fix:** Reduce batch_size, use `eval_batch_size` smaller, enable `torch.amp.autocast`, reduce `num_workers`.

### Preprocessing fails with "No such file or directory" for some regions
**Root cause:** Some OSCD regions are missing specific band TIF files (e.g., `rennes/imgs_2_rect/B04.tif`). This is a dataset download issue — some regions only have partial downloads.  
**Fix:** Those regions are automatically skipped by the `try/except` in `preprocess_oscd.py`. Verify with `ls urban_train/images/<region>/imgs_2_rect/`.

### All patches are negative (pos_avail=0 everywhere)
**Root cause:** `labels_root` path doesn't point to the directory containing `<region>/cm/cm.png`.  
**Fix:** Check `ls "$OSCD_LABELS_DIR"` shows region subdirectories (not another wrapper directory). Update `OSCD_LABELS_DIR` in the pipeline script.

---

*This handbook covers the complete implementation as of the current codebase. Keep this file local — do not push to GitHub.*
