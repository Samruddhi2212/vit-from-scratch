"""
Precision-Recall curves for all available Siamese models on LEVIR-CD val set.

Sweeps 50 thresholds in [0.05, 0.95], computes P/R for every val image,
then plots the macro-averaged PR curve per model with AUC annotation.

Usage:
    python scripts/plot_pr_curve.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.siamese_vit   import build_siamese_vit_cd
from models.siamese_swin  import build_siamese_swin_cd
from models.siamese_unet  import SiameseUNet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

BG     = "#0D0D1A"
PANEL  = "#13132A"
GRID   = "#2A2A4A"
WHITE  = "#EAEAF4"
SUBTLE = "#6666AA"

CHECKPOINTS = {
    "U-Net": ROOT / "outputs/siamese_unet/best_model.pth",
    "Swin":  ROOT / "outputs/siamese_swin/best_model.pth",
    "ViT":   ROOT / "outputs/siamese_vit/last_model.pth",
}
MODEL_COLOURS = {"U-Net": "#00D084", "Swin": "#F0A500", "ViT": "#5B9BF0"}

DATA_DIR   = ROOT / "LEVIR CD" / "val"
PATCH_SIZE = 256
OUT        = ROOT / "outputs/pr_curves.png"

THRESHOLDS = np.linspace(0.05, 0.95, 50)

device = torch.device("cpu")

val_tfm = A.Compose([
    A.CenterCrop(PATCH_SIZE, PATCH_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
], additional_targets={"image2": "image"})


def load_sample(stem):
    img1 = np.array(Image.open(DATA_DIR/"A"/f"{stem}.png").convert("RGB"))
    img2 = np.array(Image.open(DATA_DIR/"B"/f"{stem}.png").convert("RGB"))
    mask = np.array(Image.open(DATA_DIR/"label"/f"{stem}.png").convert("L"))
    mask = (mask > 127).astype(np.uint8)
    res  = val_tfm(image=img1, image2=img2, mask=mask)
    t1 = torch.from_numpy(res["image"].transpose(2,0,1)).float().unsqueeze(0)
    t2 = torch.from_numpy(res["image2"].transpose(2,0,1)).float().unsqueeze(0)
    gt = res["mask"]
    return t1, t2, gt


def load_vit(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})
    model_cfg = {k: cfg.get(k, v) for k, v in dict(
        img_size=256, patch_size=16, in_channels=3, embed_dim=768,
        depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1,
        attn_dropout=0.0, diff_type="concat_project",
        diff_out_dim=256, decoder_dims=[128,64,32,16]).items()}
    m = build_siamese_vit_cd(model_cfg).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, float(ckpt.get("best_f1", 0))

def load_swin(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})
    m = build_siamese_swin_cd(cfg).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, float(ckpt.get("best_f1", 0))

def load_unet(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})
    m = SiameseUNet(in_channels=cfg.get("in_channels", 3)).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, float(ckpt.get("best_f1", 0))


# ── load available models ──────────────────────────────────────────────────────
loaders = {"U-Net": load_unet, "Swin": load_swin, "ViT": load_vit}
models_info = {}
for name, loader in loaders.items():
    path = CHECKPOINTS[name]
    if not path.exists():
        print(f"  {name}: checkpoint not found — skipping")
        continue
    model, best_f1 = loader(path)
    models_info[name] = {"model": model, "best_f1": best_f1}
    print(f"  {name}: loaded (best_f1={best_f1:.4f})")

stems = sorted([p.stem for p in (DATA_DIR/"A").glob("*.png")])
print(f"Val images: {len(stems)}")


# ── collect per-image probabilities ───────────────────────────────────────────
# all_probs[name] = list of (probs_flat, gt_flat) per image
all_probs: dict[str, list] = {n: [] for n in models_info}

for i, stem in enumerate(stems):
    t1, t2, gt = load_sample(stem)
    gt_flat = gt.flatten()
    for name, info in models_info.items():
        with torch.no_grad():
            logits = info["model"](t1, t2)
            prob   = torch.sigmoid(logits).squeeze().numpy().flatten()
        all_probs[name].append((prob, gt_flat))
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{len(stems)} images...")

print("Sweeping thresholds...")


# ── compute PR curve per model ────────────────────────────────────────────────
def pr_at_thresh(pairs, t):
    tp = fp = fn = 0
    for prob, gt in pairs:
        pred = (prob >= t).astype(np.uint8)
        tp += int(((pred==1) & (gt==1)).sum())
        fp += int(((pred==1) & (gt==0)).sum())
        fn += int(((pred==0) & (gt==1)).sum())
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return prec, rec

pr_curves: dict[str, tuple] = {}
for name, pairs in all_probs.items():
    precs, recs = [], []
    for t in THRESHOLDS:
        p, r = pr_at_thresh(pairs, t)
        precs.append(p)
        recs.append(r)
    # sort by recall for AUC
    order  = np.argsort(recs)
    recs_s = np.array(recs)[order]
    prec_s = np.array(precs)[order]
    auc    = float(np.trapezoid(prec_s, recs_s))
    pr_curves[name] = (recs_s, prec_s, auc)
    print(f"  {name}: AUC-PR = {auc:.4f}")


# ── plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)

# Left: classic PR curve
ax = axes[0]
ax.set_facecolor(PANEL)
ax.set_title("Precision–Recall Curve\n(Threshold sweep on LEVIR-CD val set)",
             color=WHITE, fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Recall", color=SUBTLE, fontsize=10)
ax.set_ylabel("Precision", color=SUBTLE, fontsize=10)
ax.tick_params(colors=SUBTLE, labelsize=9)
ax.grid(color=GRID, linewidth=0.5, linestyle="--")
for sp in ax.spines.values(): sp.set_edgecolor(GRID)

for name, (recs, precs, auc) in pr_curves.items():
    c = MODEL_COLOURS[name]
    ax.plot(recs, precs, color=c, linewidth=2.5, label=f"{name}  (AUC={auc:.4f})")
    # mark the operating threshold (trained threshold)
    ckpt = torch.load(CHECKPOINTS[name], map_location="cpu", weights_only=False)
    op_thresh = ckpt.get("cfg", {}).get("threshold", 0.5)
    pairs = all_probs[name]
    op_p, op_r = pr_at_thresh(pairs, op_thresh)
    ax.scatter(op_r, op_p, color=c, s=120, zorder=6, marker="D",
               edgecolors=WHITE, linewidths=1.2)
    ax.annotate(f"t={op_thresh}", xy=(op_r, op_p),
                xytext=(op_r - 0.06, op_p - 0.03),
                color=c, fontsize=8.5, fontweight="bold")

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE, fontsize=10,
          loc="lower left")

# Right: F1 score vs threshold
ax2 = axes[1]
ax2.set_facecolor(PANEL)
ax2.set_title("F1 Score vs Threshold\n(Helps pick the optimal operating point)",
              color=WHITE, fontsize=13, fontweight="bold", pad=10)
ax2.set_xlabel("Threshold", color=SUBTLE, fontsize=10)
ax2.set_ylabel("F1 Score", color=SUBTLE, fontsize=10)
ax2.tick_params(colors=SUBTLE, labelsize=9)
ax2.grid(color=GRID, linewidth=0.5, linestyle="--")
for sp in ax2.spines.values(): sp.set_edgecolor(GRID)

for name, pairs in all_probs.items():
    c   = MODEL_COLOURS[name]
    f1s = []
    for t in THRESHOLDS:
        p, r = pr_at_thresh(pairs, t)
        f1 = 2*p*r / (p + r + 1e-8)
        f1s.append(f1)
    f1s = np.array(f1s)
    best_t   = THRESHOLDS[int(np.argmax(f1s))]
    best_f1  = float(np.max(f1s))
    ax2.plot(THRESHOLDS, f1s, color=c, linewidth=2.5,
             label=f"{name}  (best F1={best_f1:.4f} @ t={best_t:.2f})")
    ax2.axvline(best_t, color=c, linewidth=0.8, linestyle=":", alpha=0.6)
    ax2.scatter(best_t, best_f1, color=c, s=100, zorder=6,
                edgecolors=WHITE, linewidths=1.2)

ax2.set_xlim(0.05, 0.95)
ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE, fontsize=10,
           loc="upper right")

fig.suptitle(
    "Precision–Recall Analysis — Siamese Change Detection  |  LEVIR-CD Val Set",
    color=WHITE, fontsize=14, fontweight="bold", y=1.01
)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"\nSaved → {OUT}")
