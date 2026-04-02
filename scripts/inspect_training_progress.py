#!/usr/bin/env python3
"""Inspect a saved training history from train() (e.g. after reconnecting to HPC).

Usage:
  python scripts/inspect_training_progress.py outputs/train/checkpoints/vit_cifar10_history.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("history_pt", type=str, help="Path to *_history.pt")
    args = p.parse_args()
    path = Path(args.history_pt)
    if not path.is_file():
        print(f"Not found: {path}")
        sys.exit(1)
    h = torch.load(path, map_location="cpu", weights_only=False)
    keys = ["train_loss", "val_loss", "train_acc", "val_acc"]
    n = len(h.get("train_loss", []))
    print(f"File: {path.resolve()}")
    print(f"Epochs completed (from train_loss length): {n}")
    if n:
        print(f"  Last train loss: {h['train_loss'][-1]:.4f}")
        print(f"  Last val acc:    {h['val_acc'][-1]:.2f}%")
        print(f"  Best val acc:    {max(h['val_acc']):.2f}%")


if __name__ == "__main__":
    main()
