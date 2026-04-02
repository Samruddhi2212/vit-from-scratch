#!/usr/bin/env python3
"""Inspect a saved training history from train() (e.g. after reconnecting to HPC).

On Explorer, run with the same env as training (system python on the login node has no torch):

  module load anaconda3/2024.06 && source activate pytorch_env
  python scripts/inspect_training_progress.py outputs/train/checkpoints/vit_cifar10_history.pt

Or:  conda run -n pytorch_env python scripts/inspect_training_progress.py PATH
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    try:
        import torch
    except ImportError:
        print(
            "PyTorch is not available in this Python.\n"
            "On Explorer:  module load anaconda3/2024.06 && source activate pytorch_env\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(127)

    p = argparse.ArgumentParser()
    p.add_argument("history_pt", type=str, help="Path to *_history.pt")
    args = p.parse_args()
    path = Path(args.history_pt)
    if not path.is_file():
        print(f"Not found: {path}")
        sys.exit(1)
    h = torch.load(path, map_location="cpu", weights_only=False)
    n = len(h.get("train_loss", []))
    print(f"File: {path.resolve()}")
    print(f"Epochs completed (from train_loss length): {n}")
    if n:
        print(f"  Last train loss: {h['train_loss'][-1]:.4f}")
        print(f"  Last val acc:    {h['val_acc'][-1]:.2f}%")
        print(f"  Best val acc:    {max(h['val_acc']):.2f}%")


if __name__ == "__main__":
    main()
