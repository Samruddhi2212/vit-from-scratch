#!/usr/bin/env python3
"""
Smoke-test torch import (required for all GPU work).

torchvision is optional: CIFAR ablations use utils/cifar10_standalone (no torchvision).
Satellite / V_Dataset paths still need torchvision — install with pip when using those.

On Explorer login nodes, load CUDA modules before importing torch if you see libnvJitLink errors:
  module load cuda/12.3.0 && module load cuDNN/9.10.2
"""

from __future__ import annotations

import sys


def _hpc_cuda_module_help() -> str:
    return (
        "CUDA toolkit libraries are not on the dynamic linker path (common on login nodes).\n"
        "  Match Slurm: module load cuda/12.3.0\n"
        "               module load cuDNN/9.10.2\n"
        "  Then:        source .venv_hpc/bin/activate\n"
        "               python scripts/verify_pytorch_stack.py\n"
        "  Slurm batch jobs already run `module load`; this is for interactive checks.\n"
        "  If modules fail on the login node, use `salloc` to a GPU node or rely on the job logs."
    )


def _is_nvjitlink_missing(exc: BaseException) -> bool:
    return "nvjitlink" in str(exc).lower()


def main() -> int:
    try:
        import torch
    except (ImportError, OSError) as e:
        if _is_nvjitlink_missing(e):
            print("ERROR: CUDA shared libraries not found.\n" + _hpc_cuda_module_help(), file=sys.stderr)
            print(f"  ({e})", file=sys.stderr)
            return 1
        print(
            "ERROR: torch import failed.\n"
            "  Activate: source .venv_hpc/bin/activate\n"
            "  Install:  pip install torch (see PyTorch website for CUDA wheel)\n"
            f"  ({e})",
            file=sys.stderr,
        )
        return 1

    tv = None
    try:
        import torchvision

        tv = torchvision.__version__
    except ImportError:
        pass
    except OSError as e:
        if _is_nvjitlink_missing(e):
            print("ERROR: CUDA shared libraries not found (while importing torchvision).\n" + _hpc_cuda_module_help(), file=sys.stderr)
            print(f"  ({e})", file=sys.stderr)
            return 1
        raise

    print(f"torch {torch.__version__}  torchvision {tv or '(not installed — OK for CIFAR ablations)'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
