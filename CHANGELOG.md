# Changelog

All notable changes to this repository are documented here (course submission snapshot).

## 2026-04 — Repository organization

- Added reviewer-facing navigation: `docs/REPOSITORY_LAYOUT.md`, documentation map in `README.md`, `notebooks/README.md`.
- Moved `PROJECT_ABSTRACT.md` → `docs/PROJECT_ABSTRACT.md`.
- Renamed `notebooks/training+ablation_studies.ipynb` → `notebooks/training_and_ablation_studies.ipynb` (shell-friendly name).
- Moved `diagnose_mps.py` → `scripts/diagnose_mps.py`; added `scripts/export_ablation_json.py` (replaces ad-hoc root script).
- Added `.editorconfig` for consistent formatting in editors that support it.
- Moved LEVIR-CD training to `scripts/train_change_detection.py`; `pyproject.toml` registers the **`train-change-detection`** console command (same CLI) after `pip install -e .`.

No training logic or model definitions were intentionally changed as part of this cleanup.

## 2026-04 — Packaging and import hygiene

- Added `pyproject.toml` for `pip install -e .` and optional `dev` extras (`pytest`).
- Added `pytest.ini` and `tests/test_smoke.py`: forward passes, LEVIR config merge for all CD backbones, CIFAR/ablation `parse_args`, CLI `--help`, and utility scripts.
- Removed redundant `sys.path` manipulation from library modules (`models/`, `utils/`); use editable install or ensure repo root is on `sys.path` before imports (see `scripts/_paths.py`).
- Removed unused files: empty `models/positional_encoding.py`, deprecated stub `models/swin_attention.py`, empty `tests/test_components.py`.
- Standardized script path bootstrap (`scripts/visualize_predictions.py`, `scripts/visualize_oscd.py`, `scripts/train_change_detection.py`).

## 2026-04 — Secrets and stray artifacts

- Removed mistaken `ssh-copy-id*` files and a root-level Slurm `train_*.out` from the working tree (never belong in the repo).
- Added `docs/SECURITY.md` and expanded `.gitignore` patterns for keys / PEM / `ssh-copy-id` filenames.
