"""
Side-by-side prediction panel: same val images through all 3 models.

For each selected sample, shows 7 columns:
  1. Before (T1)
  2. After  (T2)
  3. Ground Truth mask
  4. U-Net  — TP/FP/FN overlay
  5. Swin   — TP/FP/FN overlay
  6. ViT    — TP/FP/FN overlay
  7. Probability heatmap comparison (3 sub-rows stacked)

Usage:
    python scripts/plot_side_by_side.py
"""

from __future__ import annotations
import sys, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import albumentations as A
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.siamese_vit   import build_siamese_vit_cd
from models.siamese_swin  import build_siamese_swin_cd
from models.siamese_unet  import SiameseUNet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

BG    = "#0D0D1A"
PANEL = "#13132A"
GRID  = "#2A2A4A"
WHITE = "#EAEAF4"

CHECKPOINTS = {
    "U-Net": ROOT / "outputs/siamese_unet/best_model.pth",
    "Swin":  ROOT / "outputs/siamese_swin/best_model.pth",
    "ViT":   ROOT / "outputs/siamese_vit/last_model.pth",
}
MODEL_COLOURS = {"U-Net": "#00D084", "Swin": "#F0A500", "ViT": "#5B9BF0"}

DATA_DIR   = ROOT / "LEVIR CD" / "val"
N_SAMPLES  = 5     # rows in the figure
PATCH_SIZE = 256
SEED       = 42
OUT        = ROOT / "outputs/side_by_side_comparison.png"


# ── helpers ────────────────────────────────────────────────────────────────────

def denorm(t: torch.Tensor) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN).reshape(3,1,1)
    std  = np.array(IMAGENET_STD).reshape(3,1,1)
    img  = np.clip(t.cpu().numpy() * std + mean, 0, 1)
    return (img.transpose(1,2,0) * 255).astype(np.uint8)


def overlay(img_rgb, pred, gt):
    out   = img_rgb.copy().astype(np.float32)
    alpha = 0.6
    tp = (pred==1) & (gt==1)
    fp = (pred==1) & (gt==0)
    fn = (pred==0) & (gt==1)
    out[tp] = out[tp]*(1-alpha) + np.array([0,   220,  80], np.float32)*alpha
    out[fp] = out[fp]*(1-alpha) + np.array([220,  30,  30], np.float32)*alpha
    out[fn] = out[fn]*(1-alpha) + np.array([255, 220,   0], np.float32)*alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def metrics(pred, gt):
    tp = int(((pred==1)&(gt==1)).sum())
    fp = int(((pred==1)&(gt==0)).sum())
    fn = int(((pred==0)&(gt==1)).sum())
    f1  = 2*tp / (2*tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return f1, iou


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
    m  = torch.from_numpy(res["mask"]).float().unsqueeze(0).unsqueeze(0)
    return t1, t2, m, res["image"], res["image2"]


# ── load models ────────────────────────────────────────────────────────────────

device = torch.device("cpu")

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
    return m, float(ckpt.get("best_f1", 0)), cfg.get("threshold", 0.35)

def load_swin(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})
    m = build_siamese_swin_cd(cfg).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, float(ckpt.get("best_f1", 0)), cfg.get("threshold", 0.5)

def load_unet(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})
    m = SiameseUNet(in_channels=cfg.get("in_channels", 3)).to(device)
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, float(ckpt.get("best_f1", 0)), cfg.get("threshold", 0.5)

print("Loading models...")
models_info = {}
loaders = {"U-Net": load_unet, "Swin": load_swin, "ViT": load_vit}
for name, loader in loaders.items():
    path = CHECKPOINTS[name]
    if not path.exists():
        print(f"  {name}: checkpoint not found ({path}) — skipping")
        continue
    model, best_f1, thresh = loader(path)
    models_info[name] = {"model": model, "best_f1": best_f1, "thresh": thresh}
    print(f"  {name}: best_f1={best_f1:.4f}  threshold={thresh}")

available_models = list(models_info.keys())


# ── pick samples with change pixels ───────────────────────────────────────────

stems = sorted([p.stem for p in (DATA_DIR/"A").glob("*.png")])
random.seed(SEED)

change_stems = []
for s in stems:
    mask = np.array(Image.open(DATA_DIR/"label"/f"{s}.png").convert("L"))
    # centre crop and check
    h, w = mask.shape
    y = (h - PATCH_SIZE)//2
    x = (w - PATCH_SIZE)//2
    crop = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
    if (crop > 127).sum() > 200:          # at least 200 changed pixels
        change_stems.append(s)

selected = random.sample(change_stems, min(N_SAMPLES, len(change_stems)))
print(f"\nSelected {len(selected)} samples with change pixels")


# ── figure layout ──────────────────────────────────────────────────────────────
# Columns: Before | After | GT | <model...> | Prob heatmaps (stacked)
# Last column split into len(available_models) sub-rows via inset axes

n_rows  = len(selected)
n_mods  = len(available_models)
n_cols  = 3 + n_mods + 1   # Before, After, GT, models..., heatmaps

fig_w = n_cols * 3.2
fig_h = n_rows * 3.4

fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
fig.patch.set_facecolor(BG)

if n_rows == 1:
    axes = [axes]

model_titles = [
    f"{name}  (F1={models_info[name]['best_f1']:.4f})"
    for name in available_models
]
heat_label = "Prob Heatmaps\n(" + " / ".join(available_models) + ")"
col_titles = ["Before (T1)", "After (T2)", "Ground Truth"] + model_titles + [heat_label]

for j, title in enumerate(col_titles):
    axes[0][j].set_title(title, color=WHITE, fontsize=9,
                          fontweight="bold", pad=6)

def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)


# ── inference + plotting ───────────────────────────────────────────────────────

for row_idx, stem in enumerate(selected):
    t1, t2, gt_t, img1_norm, img2_norm = load_sample(stem)
    img1_rgb = denorm(t1.squeeze(0))
    img2_rgb = denorm(t2.squeeze(0))
    gt       = (gt_t.squeeze().numpy() > 0.5).astype(np.uint8)

    row_axes = axes[row_idx]

    # col 0 — Before
    clean(row_axes[0])
    row_axes[0].imshow(img1_rgb)
    row_axes[0].set_ylabel(f"{stem}", color="#9999BB", fontsize=7)

    # col 1 — After
    clean(row_axes[1])
    row_axes[1].imshow(img2_rgb)

    # col 2 — Ground truth
    clean(row_axes[2])
    gt_disp = np.zeros((*gt.shape, 3), dtype=np.uint8)
    gt_disp[gt==1] = [255,255,255]
    row_axes[2].imshow(gt_disp)
    pct = gt.mean()*100
    row_axes[2].set_xlabel(f"Change: {pct:.1f}%", color="#9999BB", fontsize=7)

    # cols 3..3+n_mods: model overlays
    all_probs = {}
    for col_off, name in enumerate(available_models):
        ax     = row_axes[3 + col_off]
        info   = models_info[name]
        thresh = info["thresh"]
        with torch.no_grad():
            logits = info["model"](t1, t2)
            probs  = torch.sigmoid(logits).squeeze().numpy()
        pred = (probs >= thresh).astype(np.uint8)
        all_probs[name] = probs

        ov = overlay(img2_rgb, pred, gt)
        f1, iou = metrics(pred, gt)

        clean(ax)
        ax.imshow(ov)
        ax.set_xlabel(f"F1={f1:.3f}  IoU={iou:.3f}",
                      color=MODEL_COLOURS[name], fontsize=7.5, fontweight="bold")

    # last col — stacked probability heatmaps via inset axes
    ax_main = row_axes[-1]
    clean(ax_main)
    ax_main.set_visible(False)

    bbox  = ax_main.get_position()
    sub_h = bbox.height / n_mods

    for sub_i, name in enumerate(available_models):
        inset = fig.add_axes([
            bbox.x0,
            bbox.y0 + (n_mods - 1 - sub_i) * sub_h,
            bbox.width,
            sub_h * 0.92
        ])
        inset.set_facecolor(PANEL)
        inset.set_xticks([]); inset.set_yticks([])
        for sp in inset.spines.values(): sp.set_edgecolor(GRID)
        inset.imshow(all_probs[name], cmap="hot", vmin=0, vmax=1, aspect="auto")
        inset.set_ylabel(name, color=MODEL_COLOURS[name],
                         fontsize=7.5, fontweight="bold", rotation=0,
                         labelpad=28, va="center")

    print(f"  Row {row_idx+1}/{len(selected)}: {stem}")

# ── legend ─────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=(0/255, 220/255, 80/255),  label="TP — Correct change"),
    mpatches.Patch(color=(220/255, 30/255, 30/255), label="FP — False alarm"),
    mpatches.Patch(color=(255/255, 220/255, 0/255), label="FN — Missed change"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3,
           facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE,
           fontsize=10, bbox_to_anchor=(0.5, 0.005), framealpha=0.95)

model_str = "  ·  ".join(available_models)
fig.suptitle(
    f"Side-by-Side Change Detection — {model_str}  |  LEVIR-CD Val Set",
    color=WHITE, fontsize=14, fontweight="bold", y=1.002
)

plt.tight_layout(rect=[0, 0.03, 1, 1])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"\nSaved → {OUT}")
