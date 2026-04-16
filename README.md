# Vision Transformer from Scratch for Urban Change Detection

## Documentation map (for reviewers)

| Resource | Description |
|----------|-------------|
| [docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md) | **Start here:** folders, scripts, and how pieces connect. |
| [docs/PROJECT_ABSTRACT.md](docs/PROJECT_ABSTRACT.md) | Short formal abstract and scope. |
| [docs/RESULTS_LAYOUT.md](docs/RESULTS_LAYOUT.md) | Where logs, checkpoints, and figures are written. |
| [docs/ABLATIONS_HPC.md](docs/ABLATIONS_HPC.md) | Running CIFAR ablations on a Slurm cluster. |
| [notebooks/README.md](notebooks/README.md) | What each notebook contains. |
| [CHANGELOG.md](CHANGELOG.md) | Notable repo-level changes (incl. organization for submission). |
| [pyproject.toml](pyproject.toml) | Package metadata and `pip install -e .` (see **Environment setup** below). |
| [docs/SECURITY.md](docs/SECURITY.md) | SSH / credentials: what must never be committed. |

---

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

**Editable install (recommended for development and tests):** installs `models`, `utils`, `configs`, and `scripts` as packages so imports work from any working directory.

```bash
pip install -e ".[dev]"            # includes pytest
pytest
```

`pytest` runs `tests/test_smoke.py`: model forwards, merged LEVIR YAML for each CD backbone, CIFAR/ablation CLI parsing, and script `--help` / lightweight checks (no dataset downloads, no training loops).

The editable install also registers the **`train-change-detection`** console command (LEVIR-CD), equivalent to `python scripts/train_change_detection.py`.

---

## Project overview

This repository is a **hands-on study of Vision Transformers (ViTs)** built **from scratch in PyTorch**—patch embedding, multi-head self-attention, transformer blocks, and training loops—**without loading pretrained ViT weights**. The same building blocks are used in two settings:

1. **Bi-temporal change detection (main application)**  
   Two satellite images of the same place at two times are encoded with a **Siamese** backbone; features are compared and decoded into a **pixel-wise change mask**. Training targets the **LEVIR-CD** dataset (and tooling exists for related setups). Backbones include a **ViT-style encoder**, a **CNN U-Net** baseline, and a **Swin Transformer**-style encoder—so you can compare inductive biases and complexity under one training script.

2. **Image classification (pedagogy and ablations)**  
   A compact ViT is trained on **CIFAR-10** (`models/vit.py`, `configs/config.py`) to make attention maps, confusion matrices, and ablation studies tractable on a laptop or Colab. This path is about **understanding** attention and optimization; it is separate from the satellite pipeline.

If you read one idea from this file: a ViT turns an image into a **sequence of tokens**, runs **self-attention** so every token can attend to every other token, and stacks depth so **global context** is built without convolutions. For change detection, we run that idea **twice** (time 1 and time 2) and then ask **where the representations disagree**—that disagreement is turned into a map of “what changed.”

---

## Problem statement (change detection)

We observe two co-registered images of the same region:

- $X_{T1}$ — before (time 1)  
- $X_{T2}$ — after (time 2)

The goal is a binary **change map** at pixel resolution:

$$
Y \in \{0, 1\}^{H \times W}
$$

where $0$ means no change and $1$ means urban growth or other structural change of interest. This is **supervised semantic segmentation** with sparse positives: many pixels are “no change,” so losses and metrics are chosen to handle imbalance (see `utils/losses.py`, `utils/metrics.py`).

---

## Core math (what the code implements)

### Patch embedding

An image is a tensor $X \in \mathbb{R}^{H \times W \times C}$. It is split into non-overlapping patches of size $P \times P$, giving

$$
N = \frac{H}{P} \cdot \frac{W}{P}
$$

patches (for square images with divisible sizes, $N = HW / P^2$). Each patch is flattened to a vector in $\mathbb{R}^{P^2 C}$ and linearly projected to model dimension $D$. A learnable **[CLS]** token and **positional embeddings** are added so the model knows *which patch came from where*. In short: **tokens = projected patches + position information** (see `models/patch_embedding.py`).

### Multi-head self-attention (MHSA)

Stack all token embeddings into rows of a matrix $Z \in \mathbb{R}^{L \times D}$ (with $L = N$ or $N{+}1$ if a class token is used). For each head, linear maps produce

$$
Q = Z W_Q,\quad K = Z W_K,\quad V = Z W_V
$$

(with shapes chosen so each head operates in dimension $d_k = D / H_{\text{heads}}$). **Scaled dot-product attention** is

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.
$$

The factor $\sqrt{d_k}$ keeps dot products from growing too large as dimension increases—without it, softmax would often saturate and gradients would vanish. Multiple heads let the model attend to **different types of relationships** in parallel; outputs are concatenated and projected back to $D$.

### Transformer encoder block (pre-norm, as in common ViT code)

Each block applies **LayerNorm → sublayer → residual** twice:

$$
Z' = Z + \mathrm{MHSA}(\mathrm{LN}(Z)), \qquad
Z_{\mathrm{out}} = Z' + \mathrm{MLP}(\mathrm{LN}(Z')).
$$

This **pre-norm** arrangement stabilizes training in deep stacks (the exact order in code is in `models/transformer_block.py`). Ablation variants (post-norm, no residual) live in the same module for coursework experiments.

---

## Architecture (how pieces map to folders)

**Change detection (`scripts/train_change_detection.py` or `train-change-detection` after editable install, `configs/train_*.yaml`):**

- **Siamese encoders** share weights: same backbone on $X_{T1}$ and $X_{T2}$.
- **Difference / fusion** combines multi-scale features (`models/feature_difference.py`).
- **Decoder** upsamples to full-resolution logits (`models/decoder.py`).
- Entry points: `models/siamese_vit.py`, `models/siamese_unet.py`, `models/siamese_swin.py`.

**Classification (`scripts/train_cifar10.py`, `utils/training.py`):**

- **ViT** → class logits; evaluation and plots in `utils/evaluation.py`, `utils/visualization.py`.
- Optional **ablation runner**: `scripts/run_ablations.py`.

High-level diagram (LEVIR pipeline):

![VIT Architecture](architecture.png)

---

## Repository layout (where to look)

| Path | Role |
|------|------|
| `scripts/train_change_detection.py` | LEVIR-CD training: YAML + CLI; TensorBoard + checkpoints under `--output_dir`. |
| `configs/` | `train_config.yaml` (ViT CD), `train_unet_config.yaml`, `train_swin_config.yaml`; `config.py` holds **CIFAR ViT** hyperparameters (`ViTConfig`). |
| `models/` | Attention, patch embed, transformer blocks, decoders, Siamese assemblies, Swin building blocks. |
| `utils/` | Metrics, losses, **LEVIR-CD** dataloaders (`oscd_dataset.py`), **CIFAR-10** loaders and constants (`dataset.py`), training loop, evaluation, visualization. |
| `scripts/` | CIFAR training, ablations, plotting, prediction visualization helpers. |
| `slurm/` | Example batch scripts for cluster runs. |
| `outputs/` | **Change-detection** runs (logs, curves, figures); checkpoints are usually gitignored. |
| `results_cifar10/` | **CIFAR-10** figures (confusion matrix, attention tests); CLI runs also write under `outputs/train/results_cifar10/` by default. |

More detail: [docs/RESULTS_LAYOUT.md](docs/RESULTS_LAYOUT.md). Full tree: [docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md).

---

## Dataset: LEVIR-CD

The **LEVIR Change Detection (LEVIR-CD)** dataset provides **637** bi-temporal high-resolution Google Earth **pairs** for building-centric change detection.

- **GSD:** about **0.5 m** per pixel (very high resolution for Earth observation).
- **Tile size:** **1024 × 1024** RGB.
- **Labels:** pixel-wise binary masks (e.g., 0 / 255 for no-change / change).
- **Imbalance:** only a small fraction of pixels are “change,” which motivates focal / dice-style objectives and careful metrics.

**Official splits (pair counts):**

| Split | Pairs |
|-------|------|
| Train | 411 |
| Val   | 62 |
| Test  | 124 |

During training, **random 256×256 crops** are drawn from each 1024×1024 tile, which increases diversity each epoch without new downloads.

### Obtaining LEVIR-CD

Data are **not** vendored in this repo. Download separately, then point `--data_dir` at the root that contains `train/`, `val/`, and `test/` with aligned **A** (time 1), **B** (time 2), and **label** folders.

**Author page:** [LEVIR-CD](https://justchenhao.github.io/LEVIR/)

**Example mirrors (links may move):** search Kaggle for “LEVIR-CD”; some mirrors ship the classic 637-pair layout, others extended variants (e.g., LEVIR-CD+).

Expected layout:

```text
LEVIR CD/   (or any path you pass to --data_dir)
├── train/   A/  B/  label/
├── val/     A/  B/  label/
└── test/    A/  B/  label/
```

Example:

```bash
python scripts/train_change_detection.py --data_dir "./LEVIR CD" --output_dir "./outputs/siamese_vit"
```

After `pip install -e .`, you can also run `train-change-detection ...` (same arguments as the script). Use `--model vit|unet|swin` and a matching config file as needed; see `python scripts/train_change_detection.py --help` and the YAML files for defaults (image size, patch size, loss, schedule, etc.).

---

## CIFAR-10 path (classification ViT)

For **instruction and analysis**, the same conceptual ViT stack is trained on **CIFAR-10** (32×32 natural images, 10 classes):

```bash
python scripts/train_cifar10.py --epochs 200 --num-workers 4
```

Checkpoints and plots default to `outputs/train/checkpoints/` and `outputs/train/results_cifar10/` (or set `--output-dir`). Figures tracked under `results_cifar10/` in the repo are **static examples**; regenerate after training. **Ablations** without a notebook:

```bash
python scripts/run_ablations.py --ablation-epochs 50 --num-workers 4
```

---

## Evaluation metrics (change detection)

We report standard **segmentation** statistics: **IoU**, **F1**, **precision**, **recall**, **Dice**, and **Cohen’s kappa**—see `utils/metrics.py` for definitions. During training, validation metrics use the configured **probability threshold** (default **0.5** in `scripts/train_change_detection.py` unless your YAML sets `threshold`). For deployment or visualization you may tune the threshold on the validation set to trade precision for recall on rare positive pixels.

---

## Training results (LEVIR-CD, three backbones)

The table below is **validation** performance at the **best epoch by validation F1** (the same criterion used to save `best_model.pth`). Numbers are read from the training logs shipped under `outputs/`:

| Architecture | Log file | Best epoch (val F1) | F1 | IoU | Cohen’s κ | Precision | Recall |
|--------------|----------|----------------------|-----|-----|-----------|-----------|--------|
| **Siamese U-Net** (CNN baseline) | `outputs/siamese_unet/train.log` | 187 | **0.8843** | 0.7927 | 0.8805 | 0.8769 | 0.8919 |
| **Siamese Swin** (windowed attention) | `outputs/siamese_swin/train.log` | 162 | **0.8613** | 0.7564 | 0.8568 | 0.8671 | 0.8556 |
| **Siamese ViT** (global attention) | `outputs/vit_teammate_train.log` | 195 | **0.8236** | 0.7001 | 0.8176 | 0.8281 | 0.8191 |

On this benchmark run, the **CNN U-Net** achieves the highest validation F1; **Swin** is second; the **plain ViT** encoder is competitive but slightly lower—consistent with the idea that local inductive bias and stable optimization still matter for high-resolution, imbalanced segmentation. Your own runs will differ with seeds, schedules, and data paths.

### Training curves (one figure per model)

Loss, F1, IoU, precision/recall, Cohen’s kappa, learning-rate schedule, and precision–recall curves (as produced by the training script):

**Siamese U-Net**

![Training curves — U-Net](outputs/siamese_unet/training_curves.png)

**Siamese ViT** (matches `vit_teammate_train.log`)

![Training curves — ViT](outputs/vit_teammate_training_curves.png)

**Siamese Swin**

![Training curves — Swin](outputs/siamese_swin/training_curves.png)

The aggregate figure `outputs/training_curves.png` is an additional export kept at the repo root for reports.

---

## Prediction visualizations

**Example — Siamese ViT.** Each row shows **8** randomly sampled validation crops that contain at least one positive (change) pixel:

| Column | Description |
|--------|-------------|
| Before (T1) | Pre-change satellite image |
| After (T2) | Post-change satellite image |
| Ground truth | Binary change mask (white = change) |
| Prob heatmap | Predicted probability of change |
| Prediction | Thresholded binary map |
| TP/FP/FN overlay | Green = correct change, red = false alarm, yellow = missed change |

![Prediction Visualization](outputs/siamese_vit/predictions.png)

---

## Results and outputs layout

- **`outputs/`** — LEVIR-CD **change-detection** experiments from `scripts/train_change_detection.py` (TensorBoard events, `train.log`, curves, prediction grids). Subfolders such as `outputs/siamese_vit/`, `outputs/siamese_unet/`, `outputs/siamese_swin/` keep runs separated.
- **`results_cifar10/`** — **CIFAR-10 classification** artifacts (confusion matrices, attention / rollout figures from `scripts/train_cifar10.py` and `utils/evaluation.py`). The same directory name appears under `outputs/train/` when using the CLI (constant `CIFAR10_RESULTS_DIR` in `utils/cifar_paths.py`).

See [docs/RESULTS_LAYOUT.md](docs/RESULTS_LAYOUT.md) for regeneration commands.

---

## Tech stack

- **Python**, **PyTorch**, **Torchvision**
- **NumPy**, **OpenCV**, **Pillow**
- **Matplotlib**, **Seaborn**
- **scikit-learn** (e.g., t-SNE in visualizations)
- **Einops** (tensor reshapes where used), **Albumentations** (where used in pipelines)
- **PyYAML** (configs), **TensorBoard** (training logs)
- **rasterio** / **tifffile** (geospatial I/O where applicable)

Full pins: `requirements.txt`.

---

## Project goals

- **Understand** the ViT pipeline mathematically: patches → tokens → attention → depth.
- **Implement** attention, MLPs, and normalization **explicitly** (readable modules, not a black-box `nn.TransformerEncoder` for the educational path).
- **Apply** transformers to **remote-sensing change detection**, alongside **strong CNN and Swin baselines** in the same training framework.
- **Document** what each major directory and script is for, so a first-time reader can navigate the codebase with confidence.

---

## Course information

Submitted as part of a **Machine Learning** course project.
