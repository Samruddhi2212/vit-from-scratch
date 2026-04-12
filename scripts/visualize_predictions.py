"""
Visualize change detection predictions from the best model checkpoint.

For each sample shows a 6-panel row:
  1. Before image (T1)
  2. After image  (T2)
  3. Ground truth mask
  4. Predicted probability heatmap
  5. Binary prediction (thresholded)
  6. Overlay — prediction on top of T2 (TP/FP/FN coloured)

Usage:
    python scripts/visualize_predictions.py \
        --checkpoint outputs/siamese_vit/best_model.pth \
        --data_dir   "LEVIR CD" \
        --split      val \
        --n_samples  8 \
        --out        outputs/siamese_vit/predictions.png
"""

from __future__ import annotations

import argparse
import random
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from utils.oscd_dataset import OSCDDataset, IMAGENET_MEAN, IMAGENET_STD
from models.siamese_vit import build_siamese_vit_cd
from models.siamese_swin import build_siamese_swin_cd
from models.siamese_unet import SiameseUNet


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """CHW float tensor (ImageNet-normalised) → HWC uint8."""
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img  = tensor.cpu().numpy() * std + mean
    img  = np.clip(img, 0, 1)
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


def make_overlay(img_rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Colour-coded overlay on the after image:
      TP = green   (correctly detected change)
      FP = red     (false alarm)
      FN = yellow  (missed change)
      TN = original image (no change, correct)
    """
    overlay = img_rgb.copy().astype(np.float32)
    alpha   = 0.55

    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)

    # green for TP
    overlay[tp] = overlay[tp] * (1 - alpha) + np.array([0, 220, 80],   dtype=np.float32) * alpha
    # red for FP
    overlay[fp] = overlay[fp] * (1 - alpha) + np.array([220, 30,  30],  dtype=np.float32) * alpha
    # yellow for FN
    overlay[fn] = overlay[fn] * (1 - alpha) + np.array([255, 220,  0],  dtype=np.float32) * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)
    return {"F1": f1, "IoU": iou, "P": precision, "R": recall}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="outputs/siamese_vit/best_model.pth")
    p.add_argument(
        "--config",
        default=None,
        help="Optional YAML merged under checkpoint cfg (useful if ckpt lacks keys)",
    )
    p.add_argument(
        "--model",
        default=None,
        choices=["vit", "unet", "swin"],
        help="Override architecture; default is ckpt['cfg']['model'] or 'vit'",
    )
    p.add_argument("--data_dir",    default="LEVIR CD")
    p.add_argument("--split",       default="val", choices=["train", "val", "test"])
    p.add_argument("--n_samples",   type=int, default=8)
    p.add_argument("--threshold",   type=float, default=0.35)
    p.add_argument("--patch_size",  type=int, default=256)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out",         default="outputs/siamese_vit/predictions.png")
    p.add_argument("--only_change", action="store_true",
                   help="Only sample images that contain ground-truth change pixels")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = dict(ckpt.get("cfg", {}))
    if args.config:
        cfg = {**_load_yaml(args.config), **cfg}

    model_name = args.model or cfg.get("model", "vit")

    if model_name == "unet":
        model = SiameseUNet(in_channels=cfg.get("in_channels", 3)).to(device)
    elif model_name == "swin":
        model = build_siamese_swin_cd(cfg).to(device)
    else:
        model_cfg = {
            "img_size":    cfg.get("img_size",    args.patch_size),
            "patch_size":  cfg.get("patch_size",  16),
            "in_channels": cfg.get("in_channels", 3),
            "embed_dim":   cfg.get("embed_dim",   768),
            "depth":       cfg.get("depth",       12),
            "num_heads":   cfg.get("num_heads",   12),
            "mlp_ratio":   cfg.get("mlp_ratio",   4.0),
            "dropout":     cfg.get("dropout",     0.1),
            "attn_dropout":cfg.get("attn_dropout",0.0),
            "diff_type":   cfg.get("diff_type",   "concat_project"),
            "diff_out_dim":cfg.get("diff_out_dim", 256),
            "decoder_dims":cfg.get("decoder_dims", [128, 64, 32, 16]),
        }
        model = build_siamese_vit_cd(model_cfg).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"Loaded checkpoint: model={model_name}  "
        f"epoch={ckpt.get('epoch')}  best_f1={ckpt.get('best_f1', 0):.4f}"
    )

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = OSCDDataset(
        Path(args.data_dir) / args.split,
        split=args.split,
        patch_size=args.patch_size,
    )
    if args.only_change:
        change_indices = [
            i for i in range(len(dataset))
            if dataset[i]["mask"].sum() > 0
        ]
        print(f"Found {len(change_indices)} samples with change pixels")
        pool = change_indices
    else:
        pool = list(range(len(dataset)))

    n = min(args.n_samples, len(pool))
    indices = random.sample(pool, n)
    print(f"Visualizing {n} samples from '{args.split}' split")

    # ── build figure ──────────────────────────────────────────────────────────
    cols    = 6
    fig, axes = plt.subplots(n, cols, figsize=(cols * 3.5, n * 3.5))
    fig.patch.set_facecolor("#0D0D1A")

    if n == 1:
        axes = [axes]

    col_titles = [
        "Before (T1)", "After (T2)", "Ground Truth",
        "Prob Heatmap", "Prediction", "TP/FP/FN Overlay"
    ]
    for j, title in enumerate(col_titles):
        axes[0][j].set_title(title, color="white", fontsize=11,
                             fontweight="bold", pad=8)

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        img1_t = sample["image1"].unsqueeze(0).to(device)
        img2_t = sample["image2"].unsqueeze(0).to(device)
        mask_t = sample["mask"]   # (1, H, W)

        with torch.no_grad():
            logits = model(img1_t, img2_t)          # (1, 1, H, W)
            probs  = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)

        pred = (probs >= args.threshold).astype(np.uint8)
        gt   = (mask_t.squeeze().numpy() > 0.5).astype(np.uint8)

        img1_rgb = denormalize(sample["image1"])
        img2_rgb = denormalize(sample["image2"])
        overlay  = make_overlay(img2_rgb, pred, gt)
        metrics  = compute_metrics(pred, gt)

        row_axes = axes[i]

        def _clear(ax):
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#0D0D1A")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

        # col 0 — Before
        _clear(row_axes[0])
        row_axes[0].imshow(img1_rgb)
        row_axes[0].set_ylabel(f"Sample {idx}", color="#AAAAAA", fontsize=9)

        # col 1 — After
        _clear(row_axes[1])
        row_axes[1].imshow(img2_rgb)

        # col 2 — Ground truth
        _clear(row_axes[2])
        gt_disp = np.zeros((*gt.shape, 3), dtype=np.uint8)
        gt_disp[gt == 1] = [255, 255, 255]
        row_axes[2].imshow(gt_disp)
        change_pct = gt.mean() * 100
        row_axes[2].set_xlabel(f"Change: {change_pct:.1f}%", color="#AAAAAA", fontsize=8)

        # col 3 — Probability heatmap
        _clear(row_axes[3])
        im = row_axes[3].imshow(probs, cmap="hot", vmin=0, vmax=1)
        plt.colorbar(im, ax=row_axes[3], fraction=0.046, pad=0.04).ax.tick_params(colors="#AAAAAA")
        row_axes[3].axhline(y=0, color="none")  # spacer

        # col 4 — Binary prediction
        _clear(row_axes[4])
        pred_disp = np.zeros((*pred.shape, 3), dtype=np.uint8)
        pred_disp[pred == 1] = [255, 255, 255]
        row_axes[4].imshow(pred_disp)
        row_axes[4].set_xlabel(
            f"F1={metrics['F1']:.3f}  IoU={metrics['IoU']:.3f}\n"
            f"P={metrics['P']:.3f}  R={metrics['R']:.3f}",
            color="#AAAAAA", fontsize=8
        )

        # col 5 — Overlay
        _clear(row_axes[5])
        row_axes[5].imshow(overlay)

    # legend for overlay column
    legend_patches = [
        mpatches.Patch(color=(0/255, 220/255, 80/255),  label="TP (correct change)"),
        mpatches.Patch(color=(220/255, 30/255, 30/255), label="FP (false alarm)"),
        mpatches.Patch(color=(255/255, 220/255, 0/255), label="FN (missed change)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               facecolor="#1A1A2E", labelcolor="white", fontsize=10,
               bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

    title_arch = {"vit": "Siamese ViT", "unet": "Siamese U-Net", "swin": "Siamese Swin"}[
        model_name
    ]
    fig.suptitle(
        f"{title_arch} Change Detection — LEVIR-CD  |  Split: {args.split}  |  "
        f"Threshold: {args.threshold}",
        color="white", fontsize=14, fontweight="bold", y=1.001,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
