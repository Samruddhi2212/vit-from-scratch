# Vision Transformer from Scratch for Urban Change Detection

## 📌 Project Overview

This project focuses on implementing a **Vision Transformer (ViT) from scratch** for the task of **urban growth change detection** using bi-temporal satellite imagery.

Unlike traditional CNN-based change detection models, this project emphasizes:

- Complete mathematical derivation of Vision Transformer
- Implementation of patch embedding and multi-head self-attention from first principles
- Application of transformer-based feature learning to remote sensing imagery
- Performance evaluation for urban change detection

The primary goal is to understand and build the transformer architecture mathematically and programmatically without using pretrained ViT models.

---

## 🌍 Problem Statement

Given two satellite images of the same geographic region captured at different times:

- $\( X\_{T1} \)$
- $\( X\_{T2} \)$

The objective is to predict a binary change map:

$$
\[
Y \in \{0,1\}^{H \times W}
\]
$$

Where:

- 0 → No change
- 1 → Urban growth / structural change

This is formulated as a supervised semantic segmentation problem.

---

## 🏗️ System Architecture

![VIT Architecture](architecture.png)

---

## 🧠 Mathematical Formulation

### 1️⃣ Patch Embedding

Given an image:

$$
\[
X \in \mathbb{R}^{H \times W \times C}
\]
$$

Divide the image into patches of size \( P \times P \).

Number of patches:

$$
\[
N = \frac{HW}{P^2}
\]
$$

Each flattened patch is linearly projected into embedding space:

$$
\[
z_0^i = x_p^i E + p_i
\]
$$

Where:

$- \( E \in \mathbb{R}^{(P^2C) \times D} \) is the learnable projection matrix$
$- \( p_i \) is positional encoding$
$- \( D \) is embedding dimension$

---

### 2️⃣ Multi-Head Self-Attention (MHSA)

For token matrix \( Z \):

$$
\[
Q = ZW_Q, \quad K = ZW_K, \quad V = ZW_V
\]
$$

Self-attention is computed as:

$$
\[
Attention(Q,K,V) =
\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]
$$

Multiple heads allow the model to capture different spatial relationships.

---

### 3️⃣ Transformer Encoder Block

Each encoder block consists of:

- Layer Normalization
- Multi-Head Self-Attention
- Residual Connection
- Feed Forward Network (MLP)

$\[Z' = Z + \text{MHSA}(\text{LN}(Z))\]$

$\[ Z\_{out} = Z' + \text{MLP}(\text{LN}(Z'))\]$

---

## 🏗️ Architecture Overview

- Siamese Input (T1 and T2 images)
- Patch Embedding Layer
- Transformer Encoder Stack
- Feature Fusion Layer
- Segmentation Decoder
- Binary Change Map Output

The architecture is designed specifically for bi-temporal remote sensing imagery.

---

## 📊 Dataset

### About: LEVIR-CD

The **LEVIR Change Detection (LEVIR-CD)** dataset consists of **637 bi-temporal high-resolution Google Earth image pairs** for building-level change detection in urban areas.

- **Resolution:** 0.5m per pixel (very high resolution)
- **Image size:** 1024 × 1024 RGB
- **Change types:** New buildings, demolished structures, construction sites
- **Labels:** Pixel-level binary change masks (0 = no change, 255 = change)
- **Change ratio:** ~3% of pixels per image (imbalanced)

**Pre-defined splits:**

| Split | Image pairs |
|-------|------------|
| Train | 411 |
| Val   | 62 |
| Test  | 124 |

During training, random 256×256 crops are sampled from each 1024×1024 image, effectively multiplying the training data diversity each epoch.

### Obtaining LEVIR-CD

The dataset is **not** stored in this repository (it is large and subject to usage terms). Download it separately, then point `--data_dir` at the folder that contains `train/`, `val/`, and `test/` (see layout below). Uncompressed size is typically on the order of **~2–3 GB**.

**Official release (authors):** [LEVIR-CD — dataset page](https://justchenhao.github.io/LEVIR/)

**Kaggle (community mirrors; search “LEVIR-CD” if links move):**

- [LEVIR-CD](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd) — original-scale splits (637 pairs total; matches the table above).
- [LEVIR-CD+ (change detection)](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd-change-detection) — extended **LEVIR-CD+** variant (more samples than the classic 637-pair split).

After download, arrange or symlink so this repo sees the **STANet / LEVIR layout** (names must align across `A`, `B`, and `label`):

```text
LEVIR CD/   (or any path you pass to --data_dir)
├── train/   A/  B/  label/
├── val/     A/  B/  label/
└── test/    A/  B/  label/
```

Then run training, for example:

```bash
python train.py --data_dir "./LEVIR CD" --output_dir "./outputs/siamese_vit"
```

## 📏 Evaluation Metrics

To evaluate change detection performance:

- Intersection over Union (IoU)
- F1 Score
- Precision
- Recall
- Dice Coefficient

---

## 📈 Training Results

Best validation performance after 173 epochs on LEVIR-CD:

| Metric    | Score  |
|-----------|--------|
| F1        | 0.6655 |
| IoU       | 0.4987 |
| Kappa     | 0.6538 |
| Threshold | 0.35   |

### Training Curves

The figure below shows loss, F1, IoU, Precision/Recall, Cohen's Kappa, learning rate schedule, and Precision-Recall trade-off over training:

![Training Curves](outputs/training_curves.png)

---

## 🔍 Prediction Visualizations

Each row shows 8 randomly sampled val-split images that contain ground-truth change pixels:

| Column | Description |
|--------|-------------|
| Before (T1) | Pre-change satellite image |
| After (T2) | Post-change satellite image |
| Ground Truth | Binary change mask (white = change) |
| Prob Heatmap | Model's raw predicted probability (hot = high) |
| Prediction | Thresholded binary prediction (threshold = 0.35) |
| TP/FP/FN Overlay | Green = correct detection, Red = false alarm, Yellow = missed change |

![Prediction Visualization](outputs/siamese_vit/predictions.png)

---

## 📁 Results and outputs layout

- **`outputs/`** — LEVIR-CD **change-detection** experiments (training logs, curves, prediction grids) from `train.py` and related scripts. Per-model subfolders keep runs organized.
- **`results/`** — **CIFAR-10 classification** ViT artifacts (confusion matrix, attention maps, rollout figures) from `scripts/train_cifar10.py` and `utils/evaluation.py`.

See [docs/RESULTS_LAYOUT.md](docs/RESULTS_LAYOUT.md) for the full convention and regeneration commands.

---

## 🛠️ Tech Stack

- Python
- PyTorch
- NumPy
- OpenCV
- Matplotlib

---

## 🎯 Project Goals

- Understand Vision Transformer mathematically
- Implement transformer architecture from scratch
- Apply transformer-based learning to urban change detection
- Compare performance with baseline CNN-based approaches

---

## 📌 Course Information

Submitted as part of Machine Learning course project.
