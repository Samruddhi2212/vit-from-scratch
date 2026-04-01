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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.config import ViTConfig
from models.ablation_variants import ViTNoPosition
from scripts._paths import default_output_dir
from utils.ablation import run_ablation
from utils.dataset import get_cifar10_loaders


def parse_args() -> argparse.Namespace:
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
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _want(key: str, only: set[str] | None) -> bool:
    if only is None:
        return True
    return key in only


def plot_ablation_curves(all_results: dict, ablation_dir: Path) -> None:
    colors_by_name = {
        "baseline_50ep": "black",
        "no_scaling": "red",
        "no_position": "orange",
        "patch_size_2": "cyan",
        "patch_size_8": "blue",
        "patch_size_16": "navy",
        "heads_1": "green",
        "heads_2": "lime",
        "heads_8": "darkgreen",
        "no_residual": "magenta",
        "post_norm": "purple",
        "global_avg_pool": "brown",
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for _key, result in all_results.items():
        h = result["history"]
        epochs = range(1, len(h["val_acc"]) + 1)
        exp = result["name"]
        color = colors_by_name.get(exp, "gray")
        label = result["description"][:35]
        axes[0].plot(epochs, h["train_loss"], color=color, alpha=0.7, label=label)
        axes[1].plot(epochs, h["val_acc"], color=color, alpha=0.7, label=label)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss — All Ablations")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy (%)")
    axes[1].set_title("Validation Accuracy — All Ablations")
    axes[1].legend(fontsize=7, loc="lower right")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out = ablation_dir / "all_ablations_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_bar_chart(all_results: dict, ablation_dir: Path, baseline_acc: float) -> None:
    sorted_results = sorted(
        all_results.values(), key=lambda x: x["best_val_acc"], reverse=True
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    names = [r["description"][:25] for r in sorted_results]
    val_accs = [r["best_val_acc"] for r in sorted_results]
    test_accs = [r["test_acc"] for r in sorted_results]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, val_accs, width, label="Val Accuracy", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, test_accs, width, label="Test Accuracy", color="coral", alpha=0.8)
    ax.axhline(
        y=baseline_acc,
        color="black",
        linestyle="--",
        alpha=0.5,
        label=f"Baseline ({baseline_acc:.1f}%)",
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ablation Study: Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = ablation_dir / "ablation_bar_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main() -> None:
    args = parse_args()
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    base_config = ViTConfig()
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        base_config, num_workers=args.num_workers
    )

    all_results: dict = {}
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

    baseline_acc = all_results["baseline"]["best_val_acc"] if "baseline" in all_results else next(
        iter(all_results.values())
    )["best_val_acc"]

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

    if len(all_results) > 1 or "baseline" in all_results:
        plot_ablation_curves(all_results, ablation_dir)
        plot_bar_chart(all_results, ablation_dir, baseline_acc)

    torch.save(all_results, ablation_dir / "all_ablation_results.pt")
    print(f"Saved combined results to {ablation_dir / 'all_ablation_results.pt'}")
    print("ALL ABLATION STUDIES COMPLETE")


if __name__ == "__main__":
    main()
