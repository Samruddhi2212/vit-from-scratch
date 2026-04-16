# Repository layout (navigation guide)

Run all CLI commands from the **repository root** unless a script says otherwise.

## Top level

| Path | Purpose |
|------|---------|
| `scripts/train_change_detection.py` | Main **LEVIR-CD change-detection** entry point: `--model vit\|unet\|swin`, YAML configs, checkpoints under `--output_dir`. After `pip install -e .`, the **`train-change-detection`** command is the same program. |
| `README.md` | Project overview, math, how to train, where outputs go. |
| `requirements.txt` | Pinned Python dependencies (`pip install -r requirements.txt`). |
| `pyproject.toml` | Project metadata and `pip install -e .` (editable install). |
| `configs/` | YAML for CD training; `config.py` defines **CIFAR ViT** `ViTConfig` (used by classification / ablations). |
| `models/` | Attention, embeddings, transformer blocks, decoders, Siamese assemblies, Swin backbone. |
| `utils/` | Metrics, losses, LEVIR dataset (`oscd_dataset.py`), CIFAR loaders (`dataset.py`, `cifar10_standalone.py`), training loop, evaluation, visualization. |
| `scripts/` | Standalone CLIs: CIFAR training, ablations, plotting, HPC helpers, diagnostics. |
| `slurm/` | Example **Slurm** scripts for Northeastern Explorer–style clusters. |
| `notebooks/` | Interactive walkthroughs (Colab-friendly paths in cells may need editing). |
| `docs/` | This file, HPC ablation notes, results layout, project abstract. |
| `outputs/` | **Gitignored** run artifacts (logs, checkpoints, figures); example logs may be present for reports. |

## `scripts/` (CLI tools)

| Script | Role |
|--------|------|
| `train_change_detection.py` | **LEVIR-CD** Siamese training (ViT / U-Net / Swin); primary entry point for change detection. |
| `train_cifar10.py` | CIFAR-10 classification ViT (instruction / analysis). |
| `run_ablations.py` | Full CIFAR ablation sweep; writes `outputs/ablations/` and merged `all_ablation_results.pt`. |
| `plot_ablation_results.py` | Regenerate ablation **figures** from `all_ablation_results.pt` only (no GPU). |
| `export_ablation_json.py` | Export a compact **JSON** summary from `all_ablation_results.pt`. |
| `visualize_training.py` | Plot metrics from a `train.log`. |
| `visualize_predictions.py` | Change-detection prediction grids from a checkpoint. |
| `plot_all_training_curves.py` | Combine multiple `train.log` curves into one figure. |
| `verify_pytorch_stack.py` | Smoke-test `torch` (and optionally `torchvision`) on a cluster. |
| `sync_ablations_from_hpc.sh` | `rsync` ablation outputs from Explorer to a local clone. |
| `submit_ablations_chain.sh` | Submit chained Slurm jobs A → B → C. |
| `inspect_training_progress.py` | Inspect serialized training history checkpoints. |
| `visualize_oscd.py` | OSCD / CD visualization helper. |
| `diagnose_mps.py` | macOS **MPS** backward debugging (Siamese ViT tiny forward/backward). |

## `models/` (notable modules)

- `vit.py` — CIFAR-scale ViT and building blocks used pedagogically.
- `siamese_vit.py`, `siamese_unet.py`, `siamese_swin.py` — **Siamese** change-detection models for `scripts/train_change_detection.py`.
- `swin/` — Swin-style backbone (`SwinBackbone`, blocks, window attention). Import from `models.swin`, not legacy paths.

## Tests

Smoke tests live in `tests/test_smoke.py` (`pytest`). They check imports, small forward passes, merged LEVIR YAML for each change-detection backbone, CIFAR/ablation CLI parsing, and `scripts/* --help`—without downloading datasets or running training loops:

```bash
pip install -e ".[dev]"
pytest
```

For a manual import check (requires dependencies such as `tensorboard` for `scripts/train_change_detection.py`):

```bash
python -c "from models.siamese_vit import SiameseViTChangeDetection; print('OK')"
```

## See also

- [RESULTS_LAYOUT.md](RESULTS_LAYOUT.md) — where logs and figures land.
- [ABLATIONS_HPC.md](ABLATIONS_HPC.md) — running CIFAR ablations on a cluster.
- [PROJECT_ABSTRACT.md](PROJECT_ABSTRACT.md) — short formal abstract.
- [SECURITY.md](SECURITY.md) — credentials, SSH, and what not to commit.
