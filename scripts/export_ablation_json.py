#!/usr/bin/env python3
"""
Export a compact JSON summary from ``outputs/ablations/all_ablation_results.pt``.

No plotting — for quick inspection, reports, or piping to ``jq``.

Example:
  python scripts/export_ablation_json.py
  python scripts/export_ablation_json.py --input path/to/all_ablation_results.pt --output summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export ablation metrics JSON from merged .pt file")
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to all_ablation_results.pt (default: outputs/ablations/all_ablation_results.pt)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write JSON to this file (default: print to stdout)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pt_path = Path(args.input) if args.input else _REPO / "outputs" / "ablations" / "all_ablation_results.pt"
    if not pt_path.is_file():
        print(f"ERROR: not found: {pt_path}", file=sys.stderr)
        return 1

    results = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(results, dict):
        print("ERROR: checkpoint must be a dict", file=sys.stderr)
        return 1

    out: dict[str, dict] = {}
    for key, val in results.items():
        out[key] = {
            "name": val.get("name", key),
            "description": val.get("description", ""),
            "best_val_acc": round(float(val.get("best_val_acc", 0.0)), 2),
            "test_acc": round(float(val.get("test_acc", 0.0)), 2),
            "params": int(val.get("params", 0)),
        }

    text = json.dumps(out, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
