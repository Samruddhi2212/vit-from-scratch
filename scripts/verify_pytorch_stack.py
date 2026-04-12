#!/usr/bin/env python3
"""
Confirm torch + torchvision import (required by utils/dataset.py for CIFAR loaders).

On Explorer, load the same CUDA modules as `slurm/run_ablations_*.sbatch` before
importing torch on a login node; otherwise CUDA 12 libs (e.g. libnvJitLink.so.12)
may be missing from LD_LIBRARY_PATH.
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
        import torchvision
    except (ImportError, OSError) as e:
        if _is_nvjitlink_missing(e):
            print("ERROR: CUDA shared libraries not found.\n" + _hpc_cuda_module_help(), file=sys.stderr)
            print(f"  ({e})", file=sys.stderr)
            return 1
        if isinstance(e, ImportError):
            print(
                "ERROR: PyTorch stack is incomplete.\n"
                "  Activate:  source .venv_hpc/bin/activate\n"
                "  Sync deps: pip install -r requirements.txt\n"
                "  Or:        pip install 'torchvision>=0.15.0'\n"
                f"  ({e})",
                file=sys.stderr,
            )
            return 1
        raise
    print(f"torch {torch.__version__}  torchvision {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
