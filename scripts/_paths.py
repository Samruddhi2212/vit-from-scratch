"""
Repository paths and ``sys.path`` setup for CLIs under ``scripts/``.

Scripts should insert the repo root **before** ``from models...`` / ``from utils...``:

    from pathlib import Path
    import sys
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

Alternatively, install the project in editable mode (recommended for IDEs and tests)::

    pip install -e .

Then ``models``, ``utils``, and ``configs`` resolve without path hacks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    """Directory containing ``configs/``, ``models/``, ``utils/``, ``scripts/``."""
    return Path(__file__).resolve().parent.parent


def setup_sys_path() -> Path:
    """Prepend ``repo_root()`` to ``sys.path`` if missing. Returns the root path."""
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
