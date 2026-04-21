"""
Generate a multi-panel comparison chart of all three Siamese models
trained on LEVIR-CD:  U-Net · Swin · ViT

Panels:
  1. Validation F1 over epochs          (main metric)
  2. Validation IoU over epochs
  3. Validation Loss over epochs
  4. Training Loss over epochs
  5. Precision vs Recall over epochs (dual-line per model)
  6. Bar chart — best metrics side-by-side

Usage:
    python scripts/plot_model_comparison.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
LOGS = {
    "Siamese U-Net\n(31M params)": ROOT / "outputs/siamese_unet/train.log",
    "Siamese Swin\n(40M params)":  ROOT / "outputs/siamese_swin/train.log",
    "Siamese ViT\n(89M params)":   ROOT / "outputs/siamese_vit/train.log",
}
OUT = ROOT / "outputs/model_comparison.png"

# ── colour palette ─────────────────────────────────────────────────────────────
COLOURS = {
    "Siamese U-Net\n(31M params)": "#00D084",   # emerald green
    "Siamese Swin\n(40M params)":  "#F0A500",   # amber
    "Siamese ViT\n(89M params)":   "#5B9BF0",   # cornflower blue
}
BEST_MARKER = "★"

# ── parse log ─────────────────────────────────────────────────────────────────
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)"
    r"\s+F1=([\d.]+)\s+IoU=([\d.]+)\s+P=([\d.]+)\s+R=([\d.]+)"
    r"\s+Kappa=([\d.]+)"
)

def parse_log(path: Path) -> dict:
    """Return dict of lists; only the LAST run in the log file is used."""
    epochs, train_loss, val_loss = [], [], []
    f1, iou, prec, rec, kappa   = [], [], [], [], []

    # find all run boundaries (lines with "Starting training")
    text  = path.read_text()
    start = 0
    for m in re.finditer(r"Starting training", text):
        start = m.start()          # keep moving to the last occurrence

    for line in text[start:].splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        epochs.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))
        val_loss.append(float(m.group(3)))
        f1.append(float(m.group(4)))
        iou.append(float(m.group(5)))
        prec.append(float(m.group(6)))
        rec.append(float(m.group(7)))
        kappa.append(float(m.group(8)))

    return dict(epochs=epochs, train_loss=train_loss, val_loss=val_loss,
                f1=f1, iou=iou, prec=prec, rec=rec, kappa=kappa)


# ── load all models ────────────────────────────────────────────────────────────
data = {}
for name, path in LOGS.items():
    if not path.exists():
        print(f"[WARN] log not found: {path}")
        continue
    d = parse_log(path)
    if d["epochs"]:
        data[name] = d
        best_idx = int(np.argmax(d["f1"]))
        print(f"{name.replace(chr(10),' ')}: "
              f"{len(d['epochs'])} epochs | "
              f"Best F1={d['f1'][best_idx]:.4f} @ epoch {d['epochs'][best_idx]}")

# ── figure layout ──────────────────────────────────────────────────────────────
BG     = "#0D0D1A"
PANEL  = "#13132A"
GRID   = "#2A2A4A"
WHITE  = "#EAEAF4"
SUBTLE = "#6666AA"

fig = plt.figure(figsize=(22, 18), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = GridSpec(3, 3, figure=fig,
              hspace=0.42, wspace=0.32,
              left=0.06, right=0.97, top=0.91, bottom=0.06)

ax_f1    = fig.add_subplot(gs[0, 0])   # Val F1
ax_iou   = fig.add_subplot(gs[0, 1])   # Val IoU
ax_kappa = fig.add_subplot(gs[0, 2])   # Kappa
ax_vloss = fig.add_subplot(gs[1, 0])   # Val loss
ax_tloss = fig.add_subplot(gs[1, 1])   # Train loss
ax_pr    = fig.add_subplot(gs[1, 2])   # Precision & Recall
ax_bar   = fig.add_subplot(gs[2, :])   # Bar summary (full width)

axes = [ax_f1, ax_iou, ax_kappa, ax_vloss, ax_tloss, ax_pr, ax_bar]

def style_ax(ax, title, xlabel="Epoch", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=SUBTLE, fontsize=9)
    ax.set_ylabel(ylabel, color=SUBTLE, fontsize=9)
    ax.tick_params(colors=SUBTLE, labelsize=8)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

style_ax(ax_f1,    "Validation F1 Score",    ylabel="F1")
style_ax(ax_iou,   "Validation IoU",         ylabel="IoU")
style_ax(ax_kappa, "Cohen's Kappa",           ylabel="κ")
style_ax(ax_vloss, "Validation Loss",         ylabel="Loss")
style_ax(ax_tloss, "Training Loss",           ylabel="Loss")
style_ax(ax_pr,    "Precision & Recall",      ylabel="Score")
style_ax(ax_bar,   "Best Validation Metrics — Side-by-Side Comparison",
         xlabel="", ylabel="Score")

# ── plot curves ────────────────────────────────────────────────────────────────
for name, d in data.items():
    c   = COLOURS[name]
    lbl = name.replace("\n", " ")
    ep  = d["epochs"]

    best_idx = int(np.argmax(d["f1"]))

    # F1
    ax_f1.plot(ep, d["f1"], color=c, linewidth=1.8, label=lbl)
    ax_f1.scatter(ep[best_idx], d["f1"][best_idx],
                  color=c, s=90, zorder=5, marker="*")

    # IoU
    ax_iou.plot(ep, d["iou"], color=c, linewidth=1.8, label=lbl)
    ax_iou.scatter(ep[best_idx], d["iou"][best_idx],
                   color=c, s=90, zorder=5, marker="*")

    # Kappa
    ax_kappa.plot(ep, d["kappa"], color=c, linewidth=1.8, label=lbl)

    # Val loss
    ax_vloss.plot(ep, d["val_loss"], color=c, linewidth=1.8, label=lbl)

    # Train loss
    ax_tloss.plot(ep, d["train_loss"], color=c, linewidth=1.8,
                  linestyle="--", label=lbl)

    # Precision (solid) + Recall (dashed) — same colour
    ax_pr.plot(ep, d["prec"], color=c, linewidth=1.5,
               label=f"{lbl} — Precision")
    ax_pr.plot(ep, d["rec"],  color=c, linewidth=1.5,
               linestyle=":", label=f"{lbl} — Recall")

# ── add best-F1 horizontal reference lines on F1 panel ────────────────────────
for name, d in data.items():
    c = COLOURS[name]
    best_f1 = max(d["f1"])
    ax_f1.axhline(best_f1, color=c, linewidth=0.6, linestyle=":",
                  alpha=0.6)
    ax_f1.text(len(d["epochs"]) * 0.02, best_f1 + 0.005,
               f"{best_f1:.4f}", color=c, fontsize=7.5, alpha=0.9)

# ── legends ────────────────────────────────────────────────────────────────────
legend_kw = dict(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE,
                 fontsize=8, loc="lower right")

ax_f1.legend(**legend_kw)
ax_iou.legend(**legend_kw)
ax_kappa.legend(**legend_kw)
ax_vloss.legend(**legend_kw)
ax_tloss.legend(**legend_kw)

# Custom legend for precision/recall panel
pr_handles = []
for name, d in data.items():
    c   = COLOURS[name]
    lbl = name.replace("\n", " ")
    pr_handles.append(plt.Line2D([0], [0], color=c, lw=1.5,
                                 label=f"{lbl} P"))
    pr_handles.append(plt.Line2D([0], [0], color=c, lw=1.5,
                                 linestyle=":", label=f"{lbl} R"))
ax_pr.legend(handles=pr_handles, facecolor=PANEL, edgecolor=GRID,
             labelcolor=WHITE, fontsize=7, loc="lower right", ncol=2)

# ── bar chart ──────────────────────────────────────────────────────────────────
metrics     = ["F1", "IoU", "Precision", "Recall", "Kappa"]
metric_keys = ["f1", "iou", "prec", "rec", "kappa"]

models      = list(data.keys())
n_models    = len(models)
n_metrics   = len(metrics)
bar_width   = 0.22
x           = np.arange(n_metrics)

for i, name in enumerate(models):
    d      = data[name]
    c      = COLOURS[name]
    best_i = int(np.argmax(d["f1"]))
    values = [d[k][best_i] for k in metric_keys]
    offset = (i - (n_models - 1) / 2) * bar_width
    bars   = ax_bar.bar(x + offset, values, bar_width,
                        color=c, alpha=0.85,
                        label=name.replace("\n", " "),
                        edgecolor=PANEL, linewidth=0.5)
    for bar, val in zip(bars, values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    color=WHITE, fontsize=8, fontweight="bold")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metrics, color=WHITE, fontsize=11, fontweight="bold")
ax_bar.set_ylim(0, 1.08)
ax_bar.yaxis.set_tick_params(labelcolor=SUBTLE)
ax_bar.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE,
              fontsize=9, loc="upper right", ncol=3)

# winner annotation
unet_f1 = max(data.get("Siamese U-Net\n(31M params)", {}).get("f1", [0]))
ax_bar.annotate("← Best overall",
                xy=(0 - bar_width, unet_f1),
                xytext=(0.6, unet_f1 + 0.06),
                color="#00D084", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#00D084", lw=1.2))

# ── super-title ────────────────────────────────────────────────────────────────
fig.suptitle(
    "Siamese Networks for Change Detection — LEVIR-CD\n"
    "Training Dynamics & Best Metric Comparison:  U-Net  ·  Swin  ·  ViT",
    color=WHITE, fontsize=15, fontweight="bold", y=0.97
)

# ── save ───────────────────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved → {OUT}")
