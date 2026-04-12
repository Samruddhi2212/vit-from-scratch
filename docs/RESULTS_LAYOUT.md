# Where results live in this repository

Two folders are used on purpose so classification experiments stay separate from change-detection runs.

## `results_cifar10/` — CIFAR-10 (classification ViT)

Contains **static evaluation figures** produced by the **image-classification** path (`models/vit.py`, `configs/config.py`, `scripts/train_cifar10.py`, `utils/evaluation.py`):

- Confusion matrix and attention / rollout visualizations under `results_cifar10/test_viz/`
- These are **not** LEVIR-CD outputs; they document the from-scratch ViT on CIFAR-10.

Large binaries (`.pt` / `.pth` checkpoints) stay **gitignored**; regenerate with the training script after downloading CIFAR-10.

## `outputs/` — LEVIR-CD change detection

Contains **experiment logs and plots** for **siamese** models trained with `train.py` (ViT, U-Net, Swin), e.g.:

- `outputs/siamese_vit/`, `outputs/siamese_unet/`, `outputs/siamese_swin/`
- Teammate / comparison logs such as `vit_teammate_train.log` and matching `*_training_curves.png` at the `outputs/` root when checked in

Checkpoints (`best_model.pth`, etc.) are typically **local or cluster-only** and remain ignored by `.gitignore`; only selected `.log` / `.png` files may be tracked for reports.

## Scripts

- Regenerate LEVIR training figures from logs: `python scripts/plot_all_training_curves.py`
- CIFAR training: `python scripts/train_cifar10.py` (writes checkpoints under `outputs/train/checkpoints/` and figures under `outputs/train/results_cifar10/` by default unless `--output-dir` is set)
