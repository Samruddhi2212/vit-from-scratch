"""
Figures for CIFAR ViT ablations — used by scripts/run_ablations.py and scripts/plot_ablation_results.py.

The merged checkpoint ``all_ablation_results.pt`` holds per-run history and metrics; plots are
deterministic from that file (regenerate anytime without retraining).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS_BY_EXPERIMENT_NAME: dict[str, str] = {
    "baseline_50ep": "black",
    "no_scaling": "#c0392b",
    "no_position": "#e67e22",
    "patch_size_2": "#1abc9c",
    "patch_size_8": "#2980b9",
    "patch_size_16": "#1a5276",
    "heads_1": "#27ae60",
    "heads_2": "#58d68d",
    "heads_8": "#145a32",
    "no_residual": "#af7ac5",
    "post_norm": "#8e44ad",
    "global_avg_pool": "#795548",
}


def _baseline_val_acc(all_results: dict[str, Any]) -> float:
    if "baseline" in all_results:
        return float(all_results["baseline"]["best_val_acc"])
    return float(next(iter(all_results.values()))["best_val_acc"])


def plot_ablation_curves(all_results: dict[str, Any], ablation_dir: Path) -> Path:
    """Training loss and validation accuracy vs epoch (one curve per experiment)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for _key, result in sorted(all_results.items(), key=lambda kv: kv[1]["name"]):
        h = result["history"]
        epochs = range(1, len(h["val_acc"]) + 1)
        exp = result["name"]
        color = COLORS_BY_EXPERIMENT_NAME.get(exp, "#7f8c8d")
        label = exp.replace("_", " ")
        axes[0].plot(epochs, h["train_loss"], color=color, alpha=0.85, linewidth=1.5, label=label)
        axes[1].plot(epochs, h["val_acc"], color=color, alpha=0.85, linewidth=1.5, label=label)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("Training loss — all ablations")
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation accuracy (%)")
    axes[1].set_title("Validation accuracy — all ablations")
    axes[1].legend(fontsize=8, loc="lower right", ncol=2)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out = ablation_dir / "all_ablations_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_bar_chart(all_results: dict[str, Any], ablation_dir: Path, baseline_acc: float) -> Path:
    """Grouped val / test accuracy by experiment (sorted by best val acc)."""
    sorted_results = sorted(all_results.values(), key=lambda x: x["best_val_acc"], reverse=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    names = [r["name"].replace("_", " ")[:22] for r in sorted_results]
    val_accs = [r["best_val_acc"] for r in sorted_results]
    test_accs = [r["test_acc"] for r in sorted_results]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, val_accs, width, label="Best val acc", color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, test_accs, width, label="Test acc", color="coral", alpha=0.85)
    ax.axhline(
        y=baseline_acc,
        color="black",
        linestyle="--",
        alpha=0.55,
        linewidth=1.5,
        label=f"Baseline val ({baseline_acc:.2f}%)",
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ablation study — accuracy comparison (sorted by val)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(100, max(val_accs + test_accs) * 1.05))
    plt.tight_layout()
    out = ablation_dir / "ablation_bar_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_delta_vs_baseline(all_results: dict[str, Any], ablation_dir: Path, baseline_acc: float) -> Path:
    """
    Horizontal bar: Δ best val acc vs baseline (percentage points).
    Easiest view of which changes help or hurt.
    """
    rows: list[tuple[str, float, str]] = []
    for _k, r in all_results.items():
        if r["name"] == "baseline_50ep":
            continue
        delta = r["best_val_acc"] - baseline_acc
        rows.append((r["name"].replace("_", " "), delta, r["name"]))

    if not rows:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No non-baseline runs to compare.", ha="center", va="center")
        ax.axis("off")
        out = ablation_dir / "ablation_delta_vs_baseline.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    rows.sort(key=lambda t: t[1])
    names = [t[0] for t in rows]
    deltas = [t[1] for t in rows]
    colors = ["#27ae60" if d >= 0 else "#c0392b" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(rows))))
    y = np.arange(len(names))
    ax.barh(y, deltas, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Δ validation accuracy vs baseline (percentage points)")
    ax.set_title("Effect of each ablation (positive = better than baseline)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = ablation_dir / "ablation_delta_vs_baseline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def write_summary_csv(all_results: dict[str, Any], ablation_dir: Path, baseline_acc: float) -> Path:
    """Tab-separated numbers for papers / Excel."""
    out = ablation_dir / "ablation_summary.csv"
    sorted_rows = sorted(all_results.values(), key=lambda x: x["best_val_acc"], reverse=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "experiment_name",
                "best_val_acc",
                "test_acc",
                "delta_val_vs_baseline",
                "params",
                "description",
            ]
        )
        for r in sorted_rows:
            delta = r["best_val_acc"] - baseline_acc if r["name"] != "baseline_50ep" else 0.0
            w.writerow(
                [
                    r["name"],
                    f"{r['best_val_acc']:.4f}",
                    f"{r['test_acc']:.4f}",
                    f"{delta:+.4f}",
                    r["params"],
                    r["description"],
                ]
            )
    return out


def save_all_ablation_figures(all_results: dict[str, Any], ablation_dir: Path) -> list[Path]:
    """Write learning curves, CSV summary; bar + delta when multiple runs or baseline present."""
    if len(all_results) < 1:
        return []
    ablation_dir = Path(ablation_dir)
    ablation_dir.mkdir(parents=True, exist_ok=True)
    baseline_acc = _baseline_val_acc(all_results)
    paths: list[Path] = []
    paths.append(plot_ablation_curves(all_results, ablation_dir))
    paths.append(write_summary_csv(all_results, ablation_dir, baseline_acc))
    if len(all_results) > 1 or "baseline" in all_results:
        paths.append(plot_bar_chart(all_results, ablation_dir, baseline_acc))
        paths.append(plot_delta_vs_baseline(all_results, ablation_dir, baseline_acc))
    return paths
