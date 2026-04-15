# Changelog

All notable changes to this repository are documented here (course submission snapshot).

## 2026-04 — Repository organization

- Added reviewer-facing navigation: `docs/REPOSITORY_LAYOUT.md`, documentation map in `README.md`, `notebooks/README.md`.
- Moved `PROJECT_ABSTRACT.md` → `docs/PROJECT_ABSTRACT.md`.
- Renamed `notebooks/training+ablation_studies.ipynb` → `notebooks/training_and_ablation_studies.ipynb` (shell-friendly name).
- Moved `diagnose_mps.py` → `scripts/diagnose_mps.py`; added `scripts/export_ablation_json.py` (replaces ad-hoc root script).
- Added `.editorconfig` for consistent formatting in editors that support it.

No training logic or model definitions were intentionally changed as part of this cleanup.
