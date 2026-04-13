#!/usr/bin/env python3
"""
Regenerate ablation figures (and CSV summary) from ``all_ablation_results.pt`` only.

Use after rsync from HPC or to refresh plots without retraining:

  python scripts/plot_ablation_results.py
  python scripts/plot_ablation_results.py --input outputs/ablations/all_ablation_results.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from utils.ablation_plots import save_all_ablation_figures


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot CIFAR ablation metrics from merged .pt file")
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to all_ablation_results.pt (default: outputs/ablations/all_ablation_results.pt)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for PNG/CSV (default: same directory as --input)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        pt = Path(args.input)
    else:
        pt = _REPO / "outputs" / "ablations" / "all_ablation_results.pt"
    if not pt.is_file():
        print(f"ERROR: file not found: {pt}", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(args.output_dir) if args.output_dir else pt.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(pt, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        print("ERROR: checkpoint must be a dict of experiment results", file=sys.stderr)
        sys.exit(1)

    paths = save_all_ablation_figures(data, out_dir)
    if not paths:
        print("Nothing to plot (empty results?).")
        sys.exit(1)
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
