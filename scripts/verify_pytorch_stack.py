#!/usr/bin/env python3
"""
Confirm torch + torchvision import (required by utils/dataset.py for CIFAR loaders).
Run after `source .venv_hpc/bin/activate` on the cluster if jobs fail with
ModuleNotFoundError: No module named 'torchvision'.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
        import torchvision
    except ImportError as e:
        print(
            "ERROR: PyTorch stack is incomplete.\n"
            "  Activate:  source .venv_hpc/bin/activate\n"
            "  Sync deps: pip install -r requirements.txt\n"
            "  Or:        pip install 'torchvision>=0.15.0'\n"
            f"  ({e})",
            file=sys.stderr,
        )
        return 1
    print(f"torch {torch.__version__}  torchvision {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
