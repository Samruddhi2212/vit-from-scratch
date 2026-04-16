#!/usr/bin/env python3
"""
Run all ablation experiments sequentially (no Jupyter).

Example:
  python scripts/run_ablations.py --ablation-epochs 50 --num-workers 4

Use tmux or Slurm for long runs so browser disconnects do not stop training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from configs.config import ViTConfig
from models.ablation_variants import ViTNoPosition
from scripts._paths import default_output_dir
from utils.ablation import run_ablation
from utils.ablation_plots import save_all_ablation_figures
from utils.dataset import get_cifar10_loaders


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ViT ablation studies on CIFAR-10")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for ablation checkpoints and plots (default: outputs/ablations)",
    )
    p.add_argument(
        "--ablation-epochs",
        type=int,
        default=50,
        help="Epochs per ablation (notebook default 50)",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset of keys, e.g. baseline,no_scaling,gap",
    )
    p.add_argument(
        "--no-merge",
        action="store_true",
        help="Do not load existing all_ablation_results.pt (default: merge for chained Slurm jobs)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _want(key: str, only: set[str] | None) -> bool:
    if only is None:
        return True
    return key in only


def main() -> None:
    args = parse_args(None)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    only = None
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}

    if args.output_dir:
        ablation_dir = Path(args.output_dir)
    else:
        ablation_dir = default_output_dir(_REPO, "ablations")
    ablation_dir.mkdir(parents=True, exist_ok=True)

    results_path = ablation_dir / "all_ablation_results.pt"
    all_results: dict = {}
    if not args.no_merge and results_path.is_file():
        prev = torch.load(results_path, map_location="cpu", weights_only=False)
        if isinstance(prev, dict):
            all_results = dict(prev)
            print(
                f"Merged {len(all_results)} prior experiment(s) from {results_path.name} "
                "(chained ablation job)"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    base_config = ViTConfig()
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        base_config, num_workers=args.num_workers
    )

    ae = args.ablation_epochs

    if _want("baseline", only):
        all_results["baseline"] = run_ablation(
            experiment_name="baseline_50ep",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            ablation_dir=str(ablation_dir),
            ablation_epochs=ae,
            description="Standard ViT with default config",
        )

    if _want("no_scaling", only):
        all_results["no_scaling"] = run_ablation(
            experiment_name="no_scaling",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            ablation_dir=str(ablation_dir),
            ablation_epochs=ae,
            use_scaling=False,
            description="Remove sqrt(d_k) scaling from attention scores",
        )

    if _want("no_position", only):
        all_results["no_position"] = run_ablation(
            experiment_name="no_position",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            ablation_dir=str(ablation_dir),
            ablation_epochs=ae,
            config=ViTConfig(),
            model_factory=lambda c: ViTNoPosition(c),
            description="No positional encoding (zeroed and frozen)",
        )

    for ps in (2, 8, 16):
        key = f"patch_{ps}"
        if not _want(key, only):
            continue
        cfg_ps = ViTConfig()
        cfg_ps.patch_size = ps
        all_results[key] = run_ablation(
            experiment_name=f"patch_size_{ps}",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            ablation_dir=str(ablation_dir),
            ablation_epochs=ae,
            config=cfg_ps,
            description=f"Patch size = {ps} ({(cfg_ps.image_size // ps) ** 2} patches)",
        )

    for nh in (1, 2, 8):
        key = f"heads_{nh}"
        if not _want(key, only):
            continue
        cfg_nh = ViTConfig()
        cfg_nh.num_heads = nh
        all_results[key] = run_ablation(
            experiment_name=f"heads_{nh}",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            ablation_dir=str(ablation_dir),
            ablation_epochs=ae,
            config=cfg_nh,
            description=f"{nh} attention head(s), d_k = {cfg_nh.d_model // nh}",
        )

    if _want("no_residual", only):
        all_results["no_residual"] = run_ablation(
            experiment_name="no_residual",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            ablation_dir=str(ablation_dir),
            ablation_epochs=ae,
            block_type="no_residual",
            description="Remove residual/skip connections",
        )

    if _want("post_norm", only):
        all_results["post_norm"] = run_ablation(
            experiment_name="post_norm",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            ablation_dir=str(ablation_dir),
            ablation_epochs=ae,
            block_type="post_norm",
            description="Post-Norm transformer blocks",
        )

    if _want("gap", only):
        all_results["gap"] = run_ablation(
            experiment_name="global_avg_pool",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            ablation_dir=str(ablation_dir),
            ablation_epochs=ae,
            use_cls_token=False,
            description="Global average pooling instead of [CLS]",
        )

    if not all_results:
        print("Nothing to run (--only filter excluded everything).")
        return

    baseline_acc = (
        all_results["baseline"]["best_val_acc"]
        if "baseline" in all_results
        else next(iter(all_results.values()))["best_val_acc"]
    )

    print("\n" + "=" * 90)
    print("ABLATION STUDY RESULTS")
    print("=" * 90)
    print(
        f"{'Experiment':<30s} | {'Val Acc':>8s} | {'Test Acc':>8s} | {'Delta':>9s} | {'Params':>10s}"
    )
    print("-" * 90)
    sorted_rows = sorted(all_results.values(), key=lambda x: x["best_val_acc"], reverse=True)
    for r in sorted_rows:
        delta = r["best_val_acc"] - baseline_acc
        delta_str = f"{delta:+.2f}%" if r["name"] != "baseline_50ep" else "—"
        print(
            f"{r['description'][:30]:<30s} | {r['best_val_acc']:>7.2f}% | {r['test_acc']:>7.2f}% | {delta_str:>9s} | {r['params']:>10,}"
        )
    print("=" * 90)

    for p in save_all_ablation_figures(all_results, ablation_dir):
        print(f"Saved {p}")

    torch.save(all_results, ablation_dir / "all_ablation_results.pt")
    print(f"Saved combined results to {ablation_dir / 'all_ablation_results.pt'}")
    print("ALL ABLATION STUDIES COMPLETE")


if __name__ == "__main__":
    main()
