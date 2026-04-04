#!/usr/bin/env python3
"""
End-to-end preprocessing for the OSCD (Onera Satellite Change Detection) dataset.

Loads raw Sentinel-2 GeoTIFFs, stacks RGB bands, extracts overlapping 256×256
patch pairs, filters and balances classes, splits into train/val/test, saves
PNGs, writes metadata.json, and generates sample visualizations — all in one
command.

Usage:
    python preprocess_oscd.py \
        --raw_dir  ./OSCD \
        --output_dir ./processed_oscd \
        --patch_size 256

    python preprocess_oscd.py \
        --raw_dir ./OSCD \
        --output_dir ./processed_oscd \
        --patch_size 256 \
        --overlap 64 \
        --min_change_ratio 0.005 \
        --neg_pos_ratio 1.0 \
        --val_fraction 0.2 \
        --workers 4 \
        --n_vis 4 \
        --seed 42

Expected raw_dir layout (standard OSCD download):
    raw_dir/
    ├── Onera Satellite Change Detection dataset - Images/
    │   ├── train.txt
    │   ├── test.txt
    │   └── {region}/
    │       ├── imgs_1_rect/  B01.tif … B12.tif
    │       └── imgs_2_rect/  B01.tif … B12.tif
    └── Onera Satellite Change Detection dataset - Train Labels/
        └── {region}/cm/cm.png

Alternatively, raw_dir can be the Images directory itself, in which case the
Train Labels directory is looked up as a sibling.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import rasterio
    from rasterio.enums import Resampling
except ImportError:
    sys.exit(
        "ERROR: rasterio is required.  Install with:\n"
        "    pip install rasterio\n"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BAND_ORDER = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B10", "B11", "B12",
]
RGB_INDICES = [3, 2, 1]   # B04 → R, B03 → G, B02 → B (0-based within BAND_ORDER)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

_IMAGES_HINT      = "Onera Satellite Change Detection dataset - Images"
_TRAIN_LABELS_HINT = "Onera Satellite Change Detection dataset - Train Labels"
_TEST_LABELS_HINT  = "Onera Satellite Change Detection dataset - Test Labels"

log = logging.getLogger("preprocess_oscd")

# ─────────────────────────────────────────────────────────────────────────────
# Directory discovery
# ─────────────────────────────────────────────────────────────────────────────

def _find_subdir(parent: Path, hint: str) -> Optional[Path]:
    """Return the first child of *parent* whose name contains *hint* (case-insensitive)."""
    hint_lower = hint.lower()
    for child in sorted(parent.iterdir()):
        if child.is_dir() and hint_lower in child.name.lower():
            return child
    return None


def discover_dirs(raw_dir: Path) -> tuple[Path, Optional[Path], Optional[Path]]:
    """Locate images_dir, train_labels_dir, test_labels_dir.

    Handles two layouts:
    - raw_dir IS the Images directory (contains train.txt / test.txt).
    - raw_dir CONTAINS the standard OSCD sub-directories.

    Returns
    -------
    images_dir        : Path  — always resolved
    train_labels_dir  : Path | None
    test_labels_dir   : Path | None
    """
    raw_dir = raw_dir.resolve()

    # Case A: raw_dir is the Images directory itself
    if (raw_dir / "train.txt").exists():
        images_dir = raw_dir
        parent = raw_dir.parent
        train_labels = _find_subdir(parent, "train label") or _find_subdir(parent, "train labels")
        test_labels  = _find_subdir(parent, "test label")  or _find_subdir(parent, "test labels")
        return images_dir, train_labels, test_labels

    # Case B: raw_dir contains the sub-directories
    images_dir = (
        (raw_dir / _IMAGES_HINT)
        if (raw_dir / _IMAGES_HINT).exists()
        else _find_subdir(raw_dir, "images")
    )
    if images_dir is None or not (images_dir / "train.txt").exists():
        raise FileNotFoundError(
            f"Could not find an Images directory with train.txt under:\n  {raw_dir}\n\n"
            f"Expected either:\n"
            f"  {raw_dir / _IMAGES_HINT}\n"
            f"  or raw_dir itself to be the Images directory."
        )

    train_labels = (
        (raw_dir / _TRAIN_LABELS_HINT)
        if (raw_dir / _TRAIN_LABELS_HINT).exists()
        else _find_subdir(raw_dir, "train label")
    )
    test_labels = (
        (raw_dir / _TEST_LABELS_HINT)
        if (raw_dir / _TEST_LABELS_HINT).exists()
        else _find_subdir(raw_dir, "test label")
    )
    return images_dir, train_labels, test_labels


# ─────────────────────────────────────────────────────────────────────────────
# Band loading
# ─────────────────────────────────────────────────────────────────────────────

def _reference_shape(img_dir: Path) -> tuple[int, int]:
    """Return (H, W) of the 10 m reference band (B04)."""
    with rasterio.open(img_dir / "B04.tif") as src:
        return src.height, src.width


def _load_band(tif_path: Path, target_h: int, target_w: int) -> np.ndarray:
    """Read a single-band GeoTIFF, resampling to (target_h, target_w) if needed.

    Returns (H, W) float32.
    """
    with rasterio.open(tif_path) as src:
        if src.height == target_h and src.width == target_w:
            return src.read(1).astype(np.float32)
        return src.read(
            1,
            out_shape=(target_h, target_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)


def load_rgb(img_dir: Path, target_h: int, target_w: int) -> np.ndarray:
    """Stack R=B04, G=B03, B=B02 into (H, W, 3) float32."""
    rgb_bands = [BAND_ORDER[i] for i in RGB_INDICES]   # ["B04", "B03", "B02"]
    channels = [_load_band(img_dir / f"{b}.tif", target_h, target_w) for b in rgb_bands]
    return np.stack(channels, axis=-1)


def load_mask(mask_path: Path, target_h: int, target_w: int) -> np.ndarray:
    """Load cm.png and return binary (0/1) uint8 mask resized to (target_h, target_w)."""
    img = Image.open(mask_path).convert("L")
    if img.height != target_h or img.width != target_w:
        img = img.resize((target_w, target_h), Image.NEAREST)
    return (np.array(img, dtype=np.uint8) > 127).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Per-channel 2nd–98th percentile clip → [0, 255] uint8.

    Robust to cloud / haze outliers common in Sentinel-2 data.

    Parameters
    ----------
    arr : (H, W, C) float32
    """
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[2]):
        ch = arr[:, :, c]
        lo = float(np.percentile(ch, 2))
        hi = float(np.percentile(ch, 98))
        if hi - lo < 1e-6:
            hi = lo + 1.0          # flat channel (e.g. all-zero band)
        out[:, :, c] = (ch - lo) / (hi - lo)
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Patch extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_patches(
    img1: np.ndarray,
    img2: np.ndarray,
    mask: Optional[np.ndarray],
    patch_size: int,
    stride: int,
    min_change_ratio: float,
    neg_pos_ratio: float,
    rng: random.Random,
) -> tuple[list, list, list, list, int, int]:
    """Extract, filter, and balance patches from a single region.

    Parameters
    ----------
    img1, img2   : (H, W, 3) uint8
    mask         : (H, W) uint8 binary (0/1), or None for test regions
    patch_size   : spatial size of each patch
    stride       : step between patch origins (< patch_size ⟹ overlap)
    min_change_ratio : minimum changed-pixel fraction to count as a positive patch
    neg_pos_ratio    : max negatives kept = n_pos × neg_pos_ratio  (1.0 ⟹ balanced)
    rng          : seeded Random instance

    Returns
    -------
    p1s, p2s, lbls, flags, n_pos_available, n_neg_available
    flags[i] is True if patch i is positive (has change).
    """
    H, W = img1.shape[:2]
    all_pos: list[tuple] = []
    all_neg: list[tuple] = []

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            p1 = img1[r: r + patch_size, c: c + patch_size]
            p2 = img2[r: r + patch_size, c: c + patch_size]

            if mask is None:
                lbl = np.zeros((patch_size, patch_size), dtype=np.uint8)
                all_neg.append((p1, p2, lbl))
                continue

            lbl_bin = mask[r: r + patch_size, c: c + patch_size]
            change_ratio = float(lbl_bin.mean())
            lbl_255 = (lbl_bin * 255).astype(np.uint8)

            if change_ratio >= min_change_ratio:
                all_pos.append((p1, p2, lbl_255))
            else:
                all_neg.append((p1, p2, lbl_255))

    n_pos_avail = len(all_pos)
    n_neg_avail = len(all_neg)

    if mask is None:
        # Test set: keep all, no labels to filter on
        kept = [(p1, p2, lbl, False) for p1, p2, lbl in all_neg]
    else:
        n_neg_keep = max(1, int(n_pos_avail * neg_pos_ratio)) if n_pos_avail > 0 else len(all_neg)
        kept_neg = rng.sample(all_neg, min(n_neg_keep, len(all_neg)))
        kept = (
            [(p1, p2, lbl, True)  for p1, p2, lbl in all_pos] +
            [(p1, p2, lbl, False) for p1, p2, lbl in kept_neg]
        )

    p1s   = [t[0] for t in kept]
    p2s   = [t[1] for t in kept]
    lbls  = [t[2] for t in kept]
    flags = [t[3] for t in kept]
    return p1s, p2s, lbls, flags, n_pos_avail, n_neg_avail


# ─────────────────────────────────────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────────────────────────────────────

def save_patches(
    p1s: list[np.ndarray],
    p2s: list[np.ndarray],
    lbls: list[np.ndarray],
    split_dir: Path,
    region: str,
) -> None:
    """Write patch PNGs to split_dir/{A,B,label}/."""
    for sub in ("A", "B", "label"):
        (split_dir / sub).mkdir(parents=True, exist_ok=True)

    for i, (p1, p2, lbl) in enumerate(zip(p1s, p2s, lbls)):
        name = f"{region}_{i:04d}.png"
        Image.fromarray(p1).save(split_dir / "A"     / name)
        Image.fromarray(p2).save(split_dir / "B"     / name)
        Image.fromarray(lbl, mode="L").save(split_dir / "label" / name)


# ─────────────────────────────────────────────────────────────────────────────
# Worker  (module-level so it is picklable for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _worker(task: dict) -> dict:
    """Process one region end-to-end.  All arguments packed in *task* dict.

    Returns a result dict with stats / error information.
    """
    region     = task["region"]
    split      = task["split"]
    imgs_root  = Path(task["images_root"])
    lbl_root   = Path(task["labels_root"]) if task["labels_root"] else None
    out_root   = Path(task["out_root"])
    patch_size = task["patch_size"]
    stride     = task["stride"]
    min_cr     = task["min_change_ratio"]
    neg_pos    = task["neg_pos_ratio"]
    seed       = task["seed"]
    rng        = random.Random(seed ^ hash(region) & 0xFFFFFFFF)

    result: dict = {"region": region, "split": split}

    try:
        imgs_1_dir = imgs_root / region / "imgs_1_rect"
        imgs_2_dir = imgs_root / region / "imgs_2_rect"

        if not imgs_1_dir.exists():
            raise FileNotFoundError(f"imgs_1_rect not found: {imgs_1_dir}")
        if not imgs_2_dir.exists():
            raise FileNotFoundError(f"imgs_2_rect not found: {imgs_2_dir}")

        # ── reference shape from 10 m B04 ─────────────────────────────
        target_h, target_w = _reference_shape(imgs_1_dir)
        result["image_size"] = [target_h, target_w]

        if target_h < patch_size or target_w < patch_size:
            raise ValueError(
                f"Image {target_h}×{target_w} is smaller than "
                f"patch_size={patch_size}. Skipping."
            )

        # ── load RGB (only 3 bands needed, much faster than all 13) ───
        rgb1 = load_rgb(imgs_1_dir, target_h, target_w)
        rgb2 = load_rgb(imgs_2_dir, target_h, target_w)

        # ── load mask ─────────────────────────────────────────────────
        mask = None
        change_pct = None
        if lbl_root is not None:
            mask_path = lbl_root / region / "cm" / "cm.png"
            if mask_path.exists():
                mask = load_mask(mask_path, target_h, target_w)
                change_pct = float(mask.mean() * 100)
            else:
                result["warning"] = f"cm.png not found at {mask_path}"

        result["change_pct"] = change_pct

        # ── normalise ─────────────────────────────────────────────────
        img1_u8 = normalise_to_uint8(rgb1)
        img2_u8 = normalise_to_uint8(rgb2)

        # ── extract patches ───────────────────────────────────────────
        p1s, p2s, lbls, flags, n_pos_avail, n_neg_avail = extract_patches(
            img1=img1_u8,
            img2=img2_u8,
            mask=mask,
            patch_size=patch_size,
            stride=stride,
            min_change_ratio=min_cr,
            neg_pos_ratio=neg_pos,
            rng=rng,
        )

        if not p1s:
            result["warning"] = result.get("warning", "") + " No patches extracted."
            result.update(n_patches=0, n_pos=0, n_neg=0,
                          n_pos_avail=n_pos_avail, n_neg_avail=n_neg_avail)
            return result

        # ── save ──────────────────────────────────────────────────────
        split_dir = out_root / split
        save_patches(p1s, p2s, lbls, split_dir, region)

        n_pos = sum(flags)
        n_neg = len(flags) - n_pos
        result.update(
            n_patches=len(p1s),
            n_pos=n_pos,
            n_neg=n_neg,
            n_pos_avail=n_pos_avail,
            n_neg_avail=n_neg_avail,
        )

    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def _make_overlay(img_b: np.ndarray, mask_255: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a red overlay on the after-image where mask is non-zero."""
    comp = img_b.astype(np.float32)
    red = np.zeros_like(comp)
    red[..., 0] = 255.0
    changed = (mask_255 > 127)[..., None]
    comp = np.where(changed, (1 - alpha) * comp + alpha * red, comp)
    return np.clip(comp, 0, 255).astype(np.uint8)


def generate_visualizations(
    out_root: Path,
    splits: tuple[str, ...],
    n: int,
    seed: int,
) -> None:
    """Save n-sample grid figures to out_root/visualizations/."""
    vis_dir = out_root / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for split in splits:
        a_dir = out_root / split / "A"
        if not a_dir.exists():
            continue
        stems = sorted(p.stem for p in a_dir.glob("*.png"))
        if not stems:
            continue

        chosen = sorted(rng.sample(stems, min(n, len(stems))))
        n_rows = len(chosen)
        fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        col_titles = ["Before (A)", "After (B)", "GT Change Mask", "Overlay (mask on B)"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=12, fontweight="bold")

        for row, stem in enumerate(chosen):
            fname = stem + ".png"
            img_a   = np.array(Image.open(out_root / split / "A"     / fname).convert("RGB"))
            img_b   = np.array(Image.open(out_root / split / "B"     / fname).convert("RGB"))
            mask_raw = np.array(Image.open(out_root / split / "label" / fname).convert("L"))
            change_pct = 100.0 * (mask_raw > 127).sum() / mask_raw.size

            # parse region name and patch index
            parts = stem.rsplit("_", 1)
            region_name = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
            patch_idx   = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

            axes[row, 0].imshow(img_a)
            axes[row, 0].set_ylabel(
                f"{region_name}\npatch {patch_idx:04d}",
                fontsize=8, rotation=0, labelpad=72, va="center",
            )
            axes[row, 1].imshow(img_b)
            axes[row, 2].imshow(mask_raw, cmap="gray", vmin=0, vmax=255)
            axes[row, 2].text(
                0.02, 0.97, f"{change_pct:.1f}% changed",
                transform=axes[row, 2].transAxes,
                fontsize=8, color="lime", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
            )
            overlay = _make_overlay(img_b, mask_raw)
            axes[row, 3].imshow(overlay)
            axes[row, 3].legend(
                handles=[mpatches.Patch(color="red", alpha=0.55, label="change")],
                loc="lower right", fontsize=7, framealpha=0.6,
            )
            for ax in axes[row]:
                ax.axis("off")

        fig.suptitle(
            f"OSCD — {split} split  ({n_rows} samples)",
            fontsize=14, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        out_path = vis_dir / f"samples_{split}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        tqdm.write(f"  Visualization → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────

def build_metadata(
    region_results: list[dict],
    config: dict,
    images_dir: Path,
    out_root: Path,
) -> dict:
    """Aggregate per-region results into a metadata dict."""
    split_stats: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        regs = [r for r in region_results if r.get("split") == split and "error" not in r]
        n_patches  = sum(r.get("n_patches", 0) for r in regs)
        n_pos      = sum(r.get("n_pos", 0)     for r in regs)
        n_neg      = sum(r.get("n_neg", 0)     for r in regs)
        total_px   = n_patches * config["patch_size"] ** 2
        split_stats[split] = {
            "n_regions":  len(regs),
            "n_patches":  n_patches,
            "n_pos":      n_pos,
            "n_neg":      n_neg,
            "pct_pos_patches": round(100.0 * n_pos / n_patches, 2) if n_patches else 0.0,
            "pos_neg_ratio": f"{n_pos}:{n_neg}",
        }

    failed = [r["region"] for r in region_results if "error" in r]

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "images_dir": str(images_dir),
        "output_dir": str(out_root),
        "n_regions_ok":     len([r for r in region_results if "error" not in r]),
        "n_regions_failed": len(failed),
        "failed_regions":   failed,
        "splits": split_stats,
        "regions": region_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end OSCD preprocessing: GeoTIFFs → RGB PNG patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw_dir",    required=True,
                   help="Root of the raw OSCD download (contains the Images sub-directory).")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for PNG patches (will be created if needed).")
    p.add_argument("--patch_size", type=int, default=256,
                   help="Spatial size of each square patch.")
    p.add_argument("--overlap",    type=int, default=64,
                   help="Pixel overlap between adjacent patches (stride = patch_size - overlap).")
    p.add_argument("--min_change_ratio", type=float, default=0.005,
                   help="Minimum fraction of changed pixels for a patch to be 'positive'.")
    p.add_argument("--neg_pos_ratio",    type=float, default=1.0,
                   help="Max negatives kept per region = n_pos × this value.  "
                        "1.0 = balanced, 0.333 = 3:1 pos:neg.")
    p.add_argument("--val_fraction",     type=float, default=0.2,
                   help="Fraction of labelled train regions reserved for validation.")
    p.add_argument("--workers",   type=int, default=4,
                   help="Number of parallel worker processes (0 = single-process, useful for debugging).")
    p.add_argument("--n_vis",     type=int, default=4,
                   help="Number of visualization samples per split.")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = parse_args()

    raw_dir  = Path(args.raw_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    stride   = args.patch_size - args.overlap

    if stride <= 0:
        sys.exit(f"ERROR: overlap ({args.overlap}) must be less than patch_size ({args.patch_size}).")

    # ── discover directories ──────────────────────────────────────────────────
    print("\nOSCD End-to-End Preprocessing")
    print("=" * 64)
    try:
        images_dir, train_labels_dir, test_labels_dir = discover_dirs(raw_dir)
    except FileNotFoundError as exc:
        sys.exit(f"ERROR: {exc}")

    print(f"  images_dir       : {images_dir}")
    print(f"  train_labels_dir : {train_labels_dir or '(not found)'}")
    print(f"  test_labels_dir  : {test_labels_dir  or '(not found)'}")
    print(f"  output_dir       : {out_root}")
    print(f"  patch_size       : {args.patch_size}  overlap={args.overlap}  stride={stride}")
    print(f"  min_change_ratio : {args.min_change_ratio:.1%}")
    print(f"  neg_pos_ratio    : {args.neg_pos_ratio}  (1.0 = balanced)")
    print(f"  val_fraction     : {args.val_fraction}")
    print(f"  workers          : {args.workers}")
    print(f"  seed             : {args.seed}")

    # ── read split lists ──────────────────────────────────────────────────────
    train_txt = images_dir / "train.txt"
    test_txt  = images_dir / "test.txt"

    if not train_txt.exists():
        sys.exit(f"ERROR: train.txt not found at {train_txt}")

    all_train = [r.strip() for r in train_txt.read_text().strip().split(",") if r.strip()]
    all_test  = [r.strip() for r in test_txt.read_text().strip().split(",")  if r.strip()] \
                if test_txt.exists() else []

    rng_split = random.Random(args.seed)
    shuffled_train = all_train[:]
    # keep sorted for reproducibility, but allow randomisation if desired
    n_val = max(1, round(len(shuffled_train) * args.val_fraction))
    val_regions   = shuffled_train[-n_val:]
    train_regions = shuffled_train[:-n_val]

    print(f"\n  Train regions ({len(train_regions)}): {train_regions}")
    print(f"  Val   regions ({len(val_regions)}):   {val_regions}")
    print(f"  Test  regions ({len(all_test)}):  {all_test}")

    # ── build task list ───────────────────────────────────────────────────────
    tasks: list[dict] = []
    for region in train_regions:
        tasks.append(dict(
            region=region, split="train",
            images_root=str(images_dir),
            labels_root=str(train_labels_dir) if train_labels_dir else None,
            out_root=str(out_root),
            patch_size=args.patch_size, stride=stride,
            min_change_ratio=args.min_change_ratio,
            neg_pos_ratio=args.neg_pos_ratio,
            seed=args.seed,
        ))
    for region in val_regions:
        tasks.append(dict(
            region=region, split="val",
            images_root=str(images_dir),
            labels_root=str(train_labels_dir) if train_labels_dir else None,
            out_root=str(out_root),
            patch_size=args.patch_size, stride=stride,
            min_change_ratio=args.min_change_ratio,
            neg_pos_ratio=args.neg_pos_ratio,
            seed=args.seed,
        ))
    for region in all_test:
        tasks.append(dict(
            region=region, split="test",
            images_root=str(images_dir),
            labels_root=str(test_labels_dir) if test_labels_dir else None,
            out_root=str(out_root),
            patch_size=args.patch_size, stride=stride,
            min_change_ratio=args.min_change_ratio,
            neg_pos_ratio=args.neg_pos_ratio,
            seed=args.seed,
        ))

    # ── process regions ───────────────────────────────────────────────────────
    print(f"\nProcessing {len(tasks)} regions …\n")
    out_root.mkdir(parents=True, exist_ok=True)
    region_results: list[dict] = []

    if args.workers <= 1:
        # Single-process (easier to debug)
        for task in tqdm(tasks, desc="Regions", unit="region"):
            result = _worker(task)
            region_results.append(result)
            _log_result(result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, t): t["region"] for t in tasks}
            with tqdm(total=len(tasks), desc="Regions", unit="region") as pbar:
                for fut in as_completed(futures):
                    result = fut.result()
                    region_results.append(result)
                    _log_result(result)
                    pbar.update(1)

    # ── summary ───────────────────────────────────────────────────────────────
    _print_summary(region_results)

    # ── metadata.json ─────────────────────────────────────────────────────────
    config_dict = {
        "patch_size":         args.patch_size,
        "overlap":            args.overlap,
        "stride":             stride,
        "min_change_ratio":   args.min_change_ratio,
        "neg_pos_ratio":      args.neg_pos_ratio,
        "val_fraction":       args.val_fraction,
        "seed":               args.seed,
    }
    metadata = build_metadata(region_results, config_dict, images_dir, out_root)
    meta_path = out_root / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\n  Metadata → {meta_path}")

    # ── visualizations ────────────────────────────────────────────────────────
    if args.n_vis > 0:
        print(f"\nGenerating {args.n_vis} visualization samples per split …")
        generate_visualizations(out_root, ("train", "val", "test"), args.n_vis, args.seed)

    print("\nDone.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers (called in main process)
# ─────────────────────────────────────────────────────────────────────────────

def _log_result(r: dict) -> None:
    region = r["region"]
    split  = r["split"]
    if "error" in r:
        tqdm.write(f"  [ERROR] [{split:5s}] {region}: {r['error']}")
        return
    warn = f"  WARN: {r['warning']}" if "warning" in r else ""
    n = r.get("n_patches", 0)
    pos = r.get("n_pos", 0)
    neg = r.get("n_neg", 0)
    size = r.get("image_size", ["?", "?"])
    chg  = f"  global_change={r['change_pct']:.1f}%" if r.get("change_pct") is not None else ""
    tqdm.write(
        f"  [OK]    [{split:5s}] {region:20s}  "
        f"size={size[0]}×{size[1]}"
        f"{chg}  "
        f"patches={n} ({pos} pos + {neg} neg)"
        f"{warn}"
    )


def _print_summary(region_results: list[dict]) -> None:
    ok     = [r for r in region_results if "error" not in r]
    failed = [r for r in region_results if "error" in r]

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Regions processed : {len(ok)} / {len(region_results)}")
    if failed:
        print(f"  Failed regions    : {[r['region'] for r in failed]}")

    for split in ("train", "val", "test"):
        regs = [r for r in ok if r["split"] == split]
        if not regs:
            continue
        n_patches = sum(r.get("n_patches", 0) for r in regs)
        n_pos     = sum(r.get("n_pos", 0)     for r in regs)
        n_neg     = sum(r.get("n_neg", 0)     for r in regs)
        ratio_str = f"{n_pos / n_neg:.2f}:1" if n_neg else "∞"
        print(
            f"  {split:5s}  regions={len(regs):2d}  "
            f"patches={n_patches:5d}  "
            f"pos={n_pos:5d}  neg={n_neg:5d}  "
            f"pos:neg={ratio_str}"
        )

    grand = sum(r.get("n_patches", 0) for r in ok)
    print(f"\n  Grand total patches : {grand}")
    print("=" * 64)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
