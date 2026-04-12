#!/usr/bin/env python3
"""
Generate training-curve figures for each architecture using the best available logs:

  - ViT:     outputs/vit_teammate_train.log   (teammate-style focal run, ~0.82 val F1)
  - U-Net:   outputs/siamese_unet/train_run1.log  (best val F1 ~0.885)
  - Swin:    outputs/siamese_swin/train.log

Run from repo root:
    python scripts/plot_all_training_curves.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "visualize_training.py"

RUNS: list[tuple[Path, Path, str]] = [
    (
        ROOT / "outputs" / "vit_teammate_train.log",
        ROOT / "outputs" / "vit_teammate_training_curves.png",
        "Siamese ViT (teammate run, LEVIR-CD)",
    ),
    (
        ROOT / "outputs" / "siamese_unet" / "train_run1.log",
        ROOT / "outputs" / "siamese_unet" / "training_curves.png",
        "Siamese U-Net FC-Siam-diff (best run, LEVIR-CD)",
    ),
    (
        ROOT / "outputs" / "siamese_swin" / "train.log",
        ROOT / "outputs" / "siamese_swin" / "training_curves.png",
        "Siamese Swin-Tiny (LEVIR-CD)",
    ),
]


def main() -> int:
    if not SCRIPT.is_file():
        print(f"Missing {SCRIPT}", file=sys.stderr)
        return 1

    for log_path, out_path, title in RUNS:
        if not log_path.is_file():
            print(f"SKIP (missing log): {log_path}", file=sys.stderr)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--log",
            str(log_path),
            "--out",
            str(out_path),
            "--title",
            title,
        ]
        print(" ".join(cmd))
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
