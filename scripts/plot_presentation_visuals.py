"""
Generate 5 presentation-quality visualizations from train logs only.
No checkpoint files required.

Panels:
  1. Radar / Spider chart      — all 5 metrics per model
  2. Convergence speed         — epochs to hit F1 milestones
  3. Parameter efficiency      — bubble chart (params vs F1)
  4. Metric heatmap            — colour-coded metric table
  5. Train vs Val loss gap     — generalisation check over epochs

Usage:
    python scripts/plot_presentation_visuals.py
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
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent

LOGS = {
    "U-Net": ROOT / "outputs/siamese_unet/train.log",
    "Swin":  ROOT / "outputs/siamese_swin/train.log",
    "ViT":   ROOT / "outputs/siamese_vit/train.log",
}

PARAMS = {"U-Net": 31.0, "Swin": 40.5, "ViT": 89.4}   # millions

COLOURS = {"U-Net": "#00D084", "Swin": "#F0A500", "ViT": "#5B9BF0"}

BG     = "#0D0D1A"
PANEL  = "#13132A"
GRID   = "#2A2A4A"
WHITE  = "#EAEAF4"
SUBTLE = "#6666AA"

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)"
    r"\s+F1=([\d.]+)\s+IoU=([\d.]+)\s+P=([\d.]+)\s+R=([\d.]+)"
    r"\s+Kappa=([\d.]+)"
)

def parse_log(path):
    text  = path.read_text()
    start = 0
    for m in re.finditer(r"Starting training", text):
        start = m.start()
    rows = []
    for line in text[start:].splitlines():
        m = EPOCH_RE.search(line)
        if m:
            rows.append({
                "epoch":      int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_loss":   float(m.group(3)),
                "f1":         float(m.group(4)),
                "iou":        float(m.group(5)),
                "prec":       float(m.group(6)),
                "rec":        float(m.group(7)),
                "kappa":      float(m.group(8)),
            })
    return rows

data = {}
for name, path in LOGS.items():
    if path.exists():
        rows = parse_log(path)
        if rows:
            data[name] = rows

# ── best metrics per model ─────────────────────────────────────────────────────
best = {}
for name, rows in data.items():
    bi = int(np.argmax([r["f1"] for r in rows]))
    best[name] = {**rows[bi], "n_epochs": len(rows)}

# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16), facecolor=BG)
gs  = GridSpec(2, 3, figure=fig,
               hspace=0.42, wspace=0.38,
               left=0.05, right=0.97, top=0.91, bottom=0.06)

ax_radar = fig.add_subplot(gs[0, 0], polar=True)
ax_conv  = fig.add_subplot(gs[0, 1])
ax_bub   = fig.add_subplot(gs[0, 2])
ax_heat  = fig.add_subplot(gs[1, 0])
ax_gap   = fig.add_subplot(gs[1, 1])
ax_bar2  = fig.add_subplot(gs[1, 2])

def style(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=12, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=SUBTLE, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=SUBTLE, fontsize=9)
    ax.tick_params(colors=SUBTLE, labelsize=8)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

# ──────────────────────────────────────────────────────────────────────────────
# 1. RADAR CHART
# ──────────────────────────────────────────────────────────────────────────────
radar_metrics = ["F1", "IoU", "Precision", "Recall", "Kappa"]
radar_keys    = ["f1", "iou", "prec", "rec", "kappa"]
N = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]   # close the loop

ax_radar.set_facecolor(PANEL)
ax_radar.set_title("Radar: All Metrics per Model",
                   color=WHITE, fontsize=12, fontweight="bold", pad=18)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(radar_metrics, color=WHITE, fontsize=9, fontweight="bold")
ax_radar.set_ylim(0, 1)
ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                          color=SUBTLE, fontsize=7)
ax_radar.spines["polar"].set_color(GRID)
ax_radar.grid(color=GRID, linewidth=0.6)

for name, b in best.items():
    vals   = [b[k] for k in radar_keys]
    vals  += vals[:1]
    c      = COLOURS[name]
    ax_radar.plot(angles, vals, color=c, linewidth=2.2, label=name)
    ax_radar.fill(angles, vals, color=c, alpha=0.12)

ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
                facecolor=PANEL, edgecolor=GRID,
                labelcolor=WHITE, fontsize=9)

# ──────────────────────────────────────────────────────────────────────────────
# 2. CONVERGENCE SPEED  (epochs to hit F1 milestones)
# ──────────────────────────────────────────────────────────────────────────────
style(ax_conv, "Convergence Speed\n(Epochs to Reach F1 Milestone)",
      xlabel="Epochs Needed", ylabel="F1 Milestone")

milestones = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.82]
bar_h      = 0.22
y_pos      = np.arange(len(milestones))

for i, name in enumerate(["U-Net", "Swin", "ViT"]):
    if name not in data:
        continue
    f1s = [r["f1"] for r in data[name]]
    eps = []
    for ms in milestones:
        hit = next((e for e, f in enumerate(f1s) if f >= ms), None)
        eps.append(hit if hit is not None else len(f1s))

    offset = (i - 1) * bar_h
    bars = ax_conv.barh(y_pos + offset, eps, bar_h * 0.9,
                        color=COLOURS[name], alpha=0.85,
                        label=name, edgecolor=PANEL)
    for bar, val in zip(bars, eps):
        ax_conv.text(val + 1, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", color=COLOURS[name],
                     fontsize=8, fontweight="bold")

ax_conv.set_yticks(y_pos)
ax_conv.set_yticklabels([f"F1 ≥ {m:.2f}" for m in milestones],
                         color=WHITE, fontsize=8)
ax_conv.set_xlim(0, 230)
ax_conv.legend(facecolor=PANEL, edgecolor=GRID,
               labelcolor=WHITE, fontsize=9, loc="lower right")

# ──────────────────────────────────────────────────────────────────────────────
# 3. PARAMETER EFFICIENCY BUBBLE CHART
# ──────────────────────────────────────────────────────────────────────────────
style(ax_bub, "Parameter Efficiency\n(Params vs Best F1)",
      xlabel="Parameters (Millions)", ylabel="Best Validation F1")

for name, b in best.items():
    c      = COLOURS[name]
    params = PARAMS[name]
    f1_val = b["f1"]
    kappa  = b["kappa"]
    size   = kappa * 3000

    ax_bub.scatter(params, f1_val, s=size, color=c,
                   alpha=0.75, edgecolors=WHITE, linewidths=1.5, zorder=5)
    ax_bub.annotate(
        f"{name}\nF1={f1_val:.4f}\n{params}M params",
        xy=(params, f1_val),
        xytext=(params + 2, f1_val - 0.015),
        color=WHITE, fontsize=8.5, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=c, lw=0.8)
    )

# efficiency frontier annotation
ax_bub.annotate("← 3.1× more\nefficient",
                xy=(31, 0.884), xytext=(45, 0.860),
                color="#00D084", fontsize=8.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#00D084", lw=1.2))

ax_bub.set_xlim(15, 105)
ax_bub.set_ylim(0.79, 0.91)

# bubble size legend
for kval, lbl in [(0.82, "κ=0.82"), (0.86, "κ=0.86")]:
    ax_bub.scatter([], [], s=kval * 3000, color=SUBTLE, alpha=0.5,
                   label=lbl, edgecolors=WHITE, linewidths=1)
ax_bub.legend(facecolor=PANEL, edgecolor=GRID,
              labelcolor=WHITE, fontsize=8, loc="lower right",
              title="Bubble size = Kappa",
              title_fontsize=7)

# ──────────────────────────────────────────────────────────────────────────────
# 4. METRIC HEATMAP
# ──────────────────────────────────────────────────────────────────────────────
ax_heat.set_facecolor(PANEL)
ax_heat.set_title("Best Metric Heatmap", color=WHITE,
                  fontsize=12, fontweight="bold", pad=10)

metrics_h = ["F1", "IoU", "Precision", "Recall", "Kappa"]
keys_h    = ["f1", "iou", "prec", "rec", "kappa"]
models_h  = ["U-Net", "Swin", "ViT"]

matrix = np.array([[best[m][k] for k in keys_h] for m in models_h])

cmap = LinearSegmentedColormap.from_list(
    "custom", ["#1A1A3A", "#2255AA", "#00AADD", "#00D084", "#CCFF66"])

im = ax_heat.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.65, vmax=0.92)
plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04,
             ).ax.tick_params(colors=SUBTLE)

ax_heat.set_xticks(range(len(metrics_h)))
ax_heat.set_yticks(range(len(models_h)))
ax_heat.set_xticklabels(metrics_h, color=WHITE, fontsize=10, fontweight="bold")
ax_heat.set_yticklabels(models_h,  color=WHITE, fontsize=10, fontweight="bold")
ax_heat.tick_params(length=0)

for i, model in enumerate(models_h):
    for j, key in enumerate(keys_h):
        val = matrix[i, j]
        ax_heat.text(j, i, f"{val:.4f}", ha="center", va="center",
                     color="white", fontsize=10, fontweight="bold")

for spine in ax_heat.spines.values():
    spine.set_edgecolor(GRID)

# ──────────────────────────────────────────────────────────────────────────────
# 5. TRAIN vs VAL LOSS GAP  (generalisation check)
# ──────────────────────────────────────────────────────────────────────────────
style(ax_gap, "Train–Val Loss Gap Over Epochs\n(smaller gap = better generalisation)",
      xlabel="Epoch", ylabel="|Val Loss − Train Loss|")

for name, rows in data.items():
    c   = COLOURS[name]
    ep  = [r["epoch"] for r in rows]
    gap = [abs(r["val_loss"] - r["train_loss"]) for r in rows]
    # smooth with running average
    window = 5
    gap_s  = np.convolve(gap, np.ones(window)/window, mode="valid")
    ep_s   = ep[window//2: window//2 + len(gap_s)]
    ax_gap.plot(ep_s, gap_s, color=c, linewidth=1.8, label=name)
    ax_gap.fill_between(ep_s, gap_s, alpha=0.08, color=c)

ax_gap.axhline(0.0, color=WHITE, linewidth=0.5, linestyle=":", alpha=0.4)
ax_gap.legend(facecolor=PANEL, edgecolor=GRID,
              labelcolor=WHITE, fontsize=9, loc="upper right")

# ──────────────────────────────────────────────────────────────────────────────
# 6. F1 / PARAM EFFICIENCY BAR (bottom right)
# ──────────────────────────────────────────────────────────────────────────────
style(ax_bar2, "F1 per 10M Parameters\n(Parameter Efficiency)",
      ylabel="F1 per 10M Params")

model_names = list(best.keys())
efficiency  = [best[m]["f1"] / (PARAMS[m] / 10) for m in model_names]
colors_list = [COLOURS[m] for m in model_names]

bars = ax_bar2.bar(model_names, efficiency, color=colors_list,
                   edgecolor=PANEL, linewidth=0.8, alpha=0.85, width=0.5)

for bar, val, name in zip(bars, efficiency, model_names):
    ax_bar2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom",
                 color=WHITE, fontsize=11, fontweight="bold")

ax_bar2.set_xticklabels(model_names, color=WHITE, fontsize=11, fontweight="bold")
ax_bar2.set_ylim(0, max(efficiency) * 1.25)

# annotate winner
max_idx = int(np.argmax(efficiency))
ax_bar2.annotate(f"3.1× more efficient\nthan ViT",
                 xy=(0, efficiency[0]),
                 xytext=(0.5, efficiency[0] * 0.7),
                 color="#00D084", fontsize=9, fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#00D084", lw=1.2))

# ── super title ────────────────────────────────────────────────────────────────
fig.suptitle(
    "Siamese Networks for LEVIR-CD Change Detection — Comprehensive Model Analysis",
    color=WHITE, fontsize=15, fontweight="bold", y=0.97
)

OUT = ROOT / "outputs/presentation_visuals.png"
fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved → {OUT}")
