"""
Visualize training metrics from train.log for Siamese ViT Change Detection.

Parses the log file and produces a multi-panel figure covering:
  - Train / Val Loss
  - F1 Score over epochs
  - IoU over epochs
  - Precision & Recall over epochs
  - Cohen's Kappa over epochs
  - Learning Rate schedule
  - Precision-Recall trade-off scatter
  - F1 / IoU relationship
  - Max predicted probability over epochs

Usage:
    python scripts/visualize_training.py \
        --log outputs/siamese_vit/train.log \
        --out outputs/siamese_vit/training_curves.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Parse log
# ──────────────────────────────────────────────────────────────────────────────

EPOCH_PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+"
    r"train_loss=([0-9.nan]+)\s+"
    r"val_loss=([0-9.nan]+)\s+"
    r"F1=([0-9.nan]+)\s+"
    r"IoU=([0-9.nan]+)\s+"
    r"P=([0-9.nan]+)\s+"
    r"R=([0-9.nan]+)\s+"
    r"Kappa=([0-9.\-nan]+)\s+"
    r"max_prob=([0-9.nan]+)\s+"
    r"lr=([0-9.e\-+nan]+)"
)


def parse_log(log_path: Path) -> dict[str, list]:
    data = {k: [] for k in [
        "epoch", "train_loss", "val_loss", "f1", "iou",
        "precision", "recall", "kappa", "max_prob", "lr"
    ]}

    with open(log_path) as f:
        for line in f:
            m = EPOCH_PATTERN.search(line)
            if not m:
                continue
            vals = [float(v) if v != "nan" else float("nan") for v in m.groups()]
            keys = list(data.keys())
            for k, v in zip(keys, vals):
                data[k].append(v)

    return {k: np.array(v) for k, v in data.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "train":     "#2196F3",
    "val":       "#FF5722",
    "f1":        "#4CAF50",
    "iou":       "#9C27B0",
    "precision": "#FF9800",
    "recall":    "#00BCD4",
    "kappa":     "#E91E63",
    "lr":        "#607D8B",
    "max_prob":  "#795548",
}


def smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average."""
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    padded = np.pad(x, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(x)]


def plot_training(data: dict[str, np.ndarray], out_path: Path) -> None:
    epochs = data["epoch"]

    # mask out NaN epochs for clean lines
    valid = ~np.isnan(data["train_loss"]) & ~np.isnan(data["val_loss"])
    e = epochs[valid]

    fig = plt.figure(figsize=(20, 24))
    fig.patch.set_facecolor("#0F0F0F")
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    def _ax(row, col, colspan=1):
        if colspan == 2:
            ax = fig.add_subplot(gs[row, col:col+colspan])
        else:
            ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#1A1A2E")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333366")
        return ax

    def _style_ax(ax, title, xlabel="Epoch", ylabel=""):
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, color="#AAAAAA", fontsize=10)
        ax.set_ylabel(ylabel, color="#AAAAAA", fontsize=10)
        ax.tick_params(colors="#AAAAAA")
        ax.spines["bottom"].set_color("#444466")
        ax.spines["left"].set_color("#444466")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color="#2A2A4A", linewidth=0.5, alpha=0.7)

    # ── 1. Train / Val Loss ───────────────────────────────────────────────────
    ax1 = _ax(0, 0, colspan=2)
    _style_ax(ax1, "Train & Validation Loss", ylabel="Loss")
    ax1.plot(e, data["train_loss"][valid], color=COLORS["train"], lw=1.2, alpha=0.4, label="_raw")
    ax1.plot(e, smooth(data["train_loss"][valid]), color=COLORS["train"], lw=2.2, label="Train Loss")
    ax1.plot(e, data["val_loss"][valid], color=COLORS["val"], lw=1.2, alpha=0.4, label="_raw")
    ax1.plot(e, smooth(data["val_loss"][valid]), color=COLORS["val"], lw=2.2, label="Val Loss")
    best_epoch = e[np.nanargmax(data["f1"][valid])]
    ax1.axvline(best_epoch, color="#FFD700", lw=1.5, linestyle="--", alpha=0.8, label=f"Best F1 (ep {int(best_epoch)})")
    ax1.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=10)

    # ── 2. Learning Rate ──────────────────────────────────────────────────────
    ax2 = _ax(0, 2)
    _style_ax(ax2, "Learning Rate Schedule", ylabel="LR")
    ax2.plot(e, data["lr"][valid], color=COLORS["lr"], lw=2)
    ax2.set_yscale("log")

    # ── 3. F1 Score ───────────────────────────────────────────────────────────
    ax3 = _ax(1, 0, colspan=2)
    _style_ax(ax3, "Validation F1 Score", ylabel="F1")
    ax3.plot(e, data["f1"][valid], color=COLORS["f1"], lw=1.2, alpha=0.35)
    ax3.plot(e, smooth(data["f1"][valid]), color=COLORS["f1"], lw=2.5, label="Val F1")
    best_f1 = np.nanmax(data["f1"][valid])
    ax3.axhline(best_f1, color="#FFD700", lw=1.2, linestyle="--", alpha=0.7, label=f"Best F1 = {best_f1:.4f}")
    ax3.fill_between(e, 0, smooth(data["f1"][valid]), color=COLORS["f1"], alpha=0.08)
    ax3.set_ylim(0, 1)
    ax3.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=10)

    # ── 4. IoU ────────────────────────────────────────────────────────────────
    ax4 = _ax(1, 2)
    _style_ax(ax4, "Validation IoU", ylabel="IoU")
    ax4.plot(e, data["iou"][valid], color=COLORS["iou"], lw=1.2, alpha=0.35)
    ax4.plot(e, smooth(data["iou"][valid]), color=COLORS["iou"], lw=2.5)
    best_iou = np.nanmax(data["iou"][valid])
    ax4.axhline(best_iou, color="#FFD700", lw=1.2, linestyle="--", alpha=0.7, label=f"Best IoU = {best_iou:.4f}")
    ax4.fill_between(e, 0, smooth(data["iou"][valid]), color=COLORS["iou"], alpha=0.08)
    ax4.set_ylim(0, 1)
    ax4.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)

    # ── 5. Precision & Recall ─────────────────────────────────────────────────
    ax5 = _ax(2, 0, colspan=2)
    _style_ax(ax5, "Precision & Recall over Training", ylabel="Score")
    ax5.plot(e, data["precision"][valid], color=COLORS["precision"], lw=1.2, alpha=0.35)
    ax5.plot(e, smooth(data["precision"][valid]), color=COLORS["precision"], lw=2.2, label="Precision")
    ax5.plot(e, data["recall"][valid], color=COLORS["recall"], lw=1.2, alpha=0.35)
    ax5.plot(e, smooth(data["recall"][valid]), color=COLORS["recall"], lw=2.2, label="Recall")
    ax5.set_ylim(0, 1)
    ax5.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=10)

    # ── 6. Cohen's Kappa ──────────────────────────────────────────────────────
    ax6 = _ax(2, 2)
    _style_ax(ax6, "Cohen's Kappa", ylabel="Kappa")
    ax6.plot(e, data["kappa"][valid], color=COLORS["kappa"], lw=1.2, alpha=0.35)
    ax6.plot(e, smooth(data["kappa"][valid]), color=COLORS["kappa"], lw=2.5)
    ax6.axhline(0, color="#AAAAAA", lw=0.8, linestyle="--", alpha=0.5)
    best_kappa = np.nanmax(data["kappa"][valid])
    ax6.axhline(best_kappa, color="#FFD700", lw=1.2, linestyle="--", alpha=0.7, label=f"Best = {best_kappa:.4f}")
    ax6.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)

    # ── 7. Precision-Recall Scatter (colored by epoch) ────────────────────────
    ax7 = _ax(3, 0)
    _style_ax(ax7, "Precision–Recall Trade-off", xlabel="Recall", ylabel="Precision")
    sc = ax7.scatter(
        data["recall"][valid], data["precision"][valid],
        c=e, cmap="plasma", s=15, alpha=0.7, zorder=3
    )
    # mark best F1 point
    best_idx = np.nanargmax(data["f1"][valid])
    ax7.scatter(data["recall"][valid][best_idx], data["precision"][valid][best_idx],
                color="#FFD700", s=120, zorder=5, marker="*", label=f"Best F1")
    # iso-F1 curves
    for f in [0.3, 0.5, 0.6, 0.65]:
        r_range = np.linspace(0.01, 1.0, 200)
        p_iso   = f * r_range / (2 * r_range - f + 1e-9)
        mask    = (p_iso >= 0) & (p_iso <= 1)
        ax7.plot(r_range[mask], p_iso[mask], "--", color="white", lw=0.6, alpha=0.3)
        ax7.text(r_range[mask][-1] + 0.01, p_iso[mask][-1], f"F1={f}",
                 color="white", fontsize=7, alpha=0.5)
    cbar = fig.colorbar(sc, ax=ax7)
    cbar.set_label("Epoch", color="#AAAAAA", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#AAAAAA")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#AAAAAA")
    ax7.set_xlim(0, 1); ax7.set_ylim(0, 1)
    ax7.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)

    # ── 8. F1 vs IoU relationship ─────────────────────────────────────────────
    ax8 = _ax(3, 1)
    _style_ax(ax8, "F1 vs IoU", xlabel="F1", ylabel="IoU")
    sc2 = ax8.scatter(data["f1"][valid], data["iou"][valid],
                      c=e, cmap="viridis", s=15, alpha=0.7)
    # theoretical F1-IoU curve: IoU = F1 / (2 - F1)
    f_range = np.linspace(0, 1, 200)
    ax8.plot(f_range, f_range / (2 - f_range), color="#FFD700", lw=1.5,
             linestyle="--", label="IoU = F1/(2−F1)", alpha=0.8)
    ax8.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    ax8.set_xlim(0, 1); ax8.set_ylim(0, 1)
    cbar2 = fig.colorbar(sc2, ax=ax8)
    cbar2.set_label("Epoch", color="#AAAAAA", fontsize=8)
    cbar2.ax.yaxis.set_tick_params(color="#AAAAAA")
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="#AAAAAA")

    # ── 9. Max Predicted Probability ─────────────────────────────────────────
    ax9 = _ax(3, 2)
    _style_ax(ax9, "Max Predicted Probability", ylabel="max_prob")
    ax9.plot(e, data["max_prob"][valid], color=COLORS["max_prob"], lw=1.5, alpha=0.6)
    ax9.axhline(0.5, color="#AAAAAA", lw=0.8, linestyle="--", alpha=0.5, label="threshold=0.35")
    ax9.axhline(0.35, color=COLORS["val"], lw=0.8, linestyle=":", alpha=0.7)
    ax9.set_ylim(0, 1.05)
    ax9.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Siamese ViT Change Detection — LEVIR-CD Training\n"
        f"Best Val F1 = {best_f1:.4f}  |  Best IoU = {best_iou:.4f}  |  Best Kappa = {best_kappa:.4f}  |  Total Epochs = {int(e[-1])+1}",
        color="white", fontsize=15, fontweight="bold", y=0.98
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Plot training curves from train.log")
    p.add_argument("--log", default="outputs/siamese_vit/train.log",
                   help="Path to train.log")
    p.add_argument("--out", default="outputs/siamese_vit/training_curves.png",
                   help="Output PNG path")
    args = p.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    data = parse_log(log_path)
    print(f"Parsed {len(data['epoch'])} epoch entries from {log_path}")
    print(f"  Best F1    : {np.nanmax(data['f1']):.4f}  (epoch {int(data['epoch'][np.nanargmax(data['f1'])])})")
    print(f"  Best IoU   : {np.nanmax(data['iou']):.4f}")
    print(f"  Best Kappa : {np.nanmax(data['kappa']):.4f}")

    plot_training(data, Path(args.out))


if __name__ == "__main__":
    main()
