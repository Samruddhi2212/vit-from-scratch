"""Resolve repository root and prepend to sys.path (works with Slurm/tmux if cwd is wrong)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    """Directory containing configs/, models/, utils/, scripts/."""
    return Path(__file__).resolve().parent.parent


def setup_sys_path() -> Path:
    root = repo_root()
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)
    return root


def default_output_dir(root: Path, name: str) -> Path:
    """outputs/<name> under repo unless VIT_OUTPUT_DIR is set."""
    env = os.environ.get("VIT_OUTPUT_DIR")
    if env:
        p = Path(env) / name
    else:
        p = root / "outputs" / name
    p.mkdir(parents=True, exist_ok=True)
    return p
