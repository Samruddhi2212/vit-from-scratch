"""
Visualization and statistics script for the preprocessed OSCD dataset.

Usage:
    python scripts/visualize_oscd.py                          # default: 4 samples/split
    python scripts/visualize_oscd.py --n_samples 6 --seed 99
    python scripts/visualize_oscd.py --data_root /path/to/processed_oscd

Outputs
-------
    visualizations/oscd_samples_{split}.png   — grid figures per split
    visualizations/oscd_stats.txt             — printed statistics also saved here
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image

# ── project root on path ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.oscd_dataset import OSCDDataset, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

SPLITS = ("train", "val", "test")


def _parse_stem(stem: str) -> tuple[str, int]:
    """Return (region_name, patch_index) from a file stem like 'paris_0042'.

    The index is always the last underscore-delimited token and is 4 digits.
    Region names themselves may contain underscores (e.g. 'saclay_e').
    """
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
        return parts[0], int(parts[1])
    # fallback: treat the whole stem as region name, index 0
    return stem, 0


def _load_raw(split_dir: Path, stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw uint8 arrays (no normalisation) for a given stem.

    Returns
    -------
    img_a : (256, 256, 3) uint8 — before image
    img_b : (256, 256, 3) uint8 — after  image
    mask  : (256, 256)    uint8 — binary 0/1 (converted from 0/255)
    """
    fname = stem + ".png"
    img_a = np.array(Image.open(split_dir / "A"     / fname).convert("RGB"))
    img_b = np.array(Image.open(split_dir / "B"     / fname).convert("RGB"))
    mask_raw = np.array(Image.open(split_dir / "label" / fname).convert("L"))
    mask = (mask_raw > 127).astype(np.uint8)
    return img_a, img_b, mask


def _overlay(img_b: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a red change mask over the 'after' image.

    Parameters
    ----------
    img_b : (H, W, 3) uint8
    mask  : (H, W)    binary 0/1
    alpha : transparency of the overlay (0 = invisible, 1 = opaque)

    Returns
    -------
    composite : (H, W, 3) uint8
    """
    composite = img_b.astype(np.float32).copy()
    red_layer = np.zeros_like(composite)
    red_layer[..., 0] = 255.0          # pure red
    composite = np.where(
        mask[..., None] == 1,
        (1 - alpha) * composite + alpha * red_layer,
        composite,
    )
    return np.clip(composite, 0, 255).astype(np.uint8)


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation and return an HWC uint8 array.

    Parameters
    ----------
    tensor : (3, H, W) float32, ImageNet-normalised

    Returns
    -------
    (H, W, 3) uint8 image, values clipped to [0, 255]
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)
    img = tensor.cpu() * std + mean                    # undo normalisation
    img = (img.clamp(0.0, 1.0).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# figure generation
# ──────────────────────────────────────────────────────────────────────────────

def make_sample_figure(
    split_dir: Path,
    stems: list[str],
    split: str,
    out_path: Path,
) -> None:
    """Create and save a grid figure for the selected stems.

    Each row = one sample.  Columns: Before (A) | After (B) | GT Mask | Overlay.
    """
    n = len(stems)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]          # ensure 2-D indexing always works

    col_titles = ["Before (A)", "After (B)", "GT Change Mask", "Overlay (mask on B)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=8)

    for row, stem in enumerate(stems):
        img_a, img_b, mask = _load_raw(split_dir, stem)
        region, patch_idx = _parse_stem(stem)
        change_pct = 100.0 * mask.sum() / mask.size

        row_axes = axes[row]

        # ── before ────────────────────────────────────────────────────────
        row_axes[0].imshow(img_a)
        row_axes[0].set_ylabel(
            f"{region}\npatch {patch_idx:04d}",
            fontsize=9, rotation=0, labelpad=72, va="center",
        )

        # ── after ─────────────────────────────────────────────────────────
        row_axes[1].imshow(img_b)

        # ── mask ──────────────────────────────────────────────────────────
        row_axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
        row_axes[2].text(
            0.02, 0.97,
            f"changed: {change_pct:.1f}%",
            transform=row_axes[2].transAxes,
            fontsize=8, color="lime", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55),
        )

        # ── overlay ───────────────────────────────────────────────────────
        overlay_img = _overlay(img_b, mask)
        row_axes[3].imshow(overlay_img)
        red_patch = mpatches.Patch(color="red", alpha=0.55, label="change")
        row_axes[3].legend(
            handles=[red_patch], loc="lower right",
            fontsize=7, framealpha=0.6,
        )

        for ax in row_axes:
            ax.axis("off")

    fig.suptitle(
        f"OSCD — {split} split   ({n} random samples)",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_statistics(
    data_root: Path,
    rng: random.Random,
    max_stat_samples: int = 500,
) -> dict[str, dict]:
    """Compute per-split statistics.

    Returns a dict keyed by split name, each containing:
        n_patches        : int
        pct_changed      : float   — % of *pixels* that are changed
        pct_unchanged    : float
        image1_mean      : (3,) ndarray   — channel-wise mean of normalised images
        image1_std       : (3,) ndarray
        image2_mean      : (3,) ndarray
        image2_std       : (3,) ndarray
    """
    stats: dict[str, dict] = {}

    for split in SPLITS:
        split_dir = data_root / split
        if not (split_dir / "A").exists():
            print(f"  [skip] {split}: directory not found ({split_dir / 'A'})")
            continue

        # ── load dataset (normalised, for pixel stats) ────────────────────
        ds = OSCDDataset(split_dir, split=split, patch_size=256)
        n = len(ds)

        # sample a subset if large
        indices = list(range(n))
        if n > max_stat_samples:
            rng.shuffle(indices)
            indices = indices[:max_stat_samples]

        # accumulators
        total_pixels   = 0
        changed_pixels = 0
        sum1    = np.zeros(3, dtype=np.float64)
        sumsq1  = np.zeros(3, dtype=np.float64)
        sum2    = np.zeros(3, dtype=np.float64)
        sumsq2  = np.zeros(3, dtype=np.float64)
        n_px    = 0

        for idx in indices:
            sample = ds[idx]
            t1   = sample["image1"].numpy()   # (3, H, W) float32, normalised
            t2   = sample["image2"].numpy()
            mask = sample["mask"].numpy()      # (1, H, W) float32 0/1

            # class distribution — use raw mask file for true counts
            stem  = ds.stems[idx]
            fname = stem + ".png"
            mask_raw = np.array(
                Image.open(split_dir / "label" / fname).convert("L")
            )
            binary = (mask_raw > 127).astype(np.uint8)
            total_pixels   += binary.size
            changed_pixels += int(binary.sum())

            # pixel stats over normalised images
            H, W = t1.shape[1], t1.shape[2]
            px = H * W
            sum1   += t1.reshape(3, -1).sum(axis=1)
            sumsq1 += (t1.reshape(3, -1) ** 2).sum(axis=1)
            sum2   += t2.reshape(3, -1).sum(axis=1)
            sumsq2 += (t2.reshape(3, -1) ** 2).sum(axis=1)
            n_px   += px

        # finalise stats
        mean1 = sum1  / n_px
        mean2 = sum2  / n_px
        std1  = np.sqrt(np.maximum(sumsq1 / n_px - mean1 ** 2, 0))
        std2  = np.sqrt(np.maximum(sumsq2 / n_px - mean2 ** 2, 0))

        pct_changed   = 100.0 * changed_pixels / total_pixels if total_pixels else 0.0
        pct_unchanged = 100.0 - pct_changed

        stats[split] = {
            "n_patches":     n,
            "n_stat_samples": len(indices),
            "pct_changed":   pct_changed,
            "pct_unchanged": pct_unchanged,
            "image1_mean":   mean1,
            "image1_std":    std1,
            "image2_mean":   mean2,
            "image2_std":    std2,
        }

    return stats


def print_and_save_statistics(stats: dict[str, dict], out_path: Path) -> None:
    """Pretty-print statistics to stdout and save to a text file."""
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("OSCD Dataset Statistics")
    lines.append("=" * 70)
    lines.append("")

    for split, s in stats.items():
        lines.append(f"Split: {split.upper()}")
        lines.append(f"  Total patches          : {s['n_patches']}")
        lines.append(
            f"  Stat sample size       : {s['n_stat_samples']} "
            f"(of {s['n_patches']})"
        )
        lines.append("")
        lines.append("  Class distribution (pixels):")
        lines.append(f"    Changed   (1) : {s['pct_changed']:.2f}%")
        lines.append(f"    Unchanged (0) : {s['pct_unchanged']:.2f}%")
        lines.append("")
        lines.append("  Pixel-value statistics (ImageNet-normalised images):")
        lines.append(
            f"    image1 (before)  mean : "
            f"R={s['image1_mean'][0]:.4f}  "
            f"G={s['image1_mean'][1]:.4f}  "
            f"B={s['image1_mean'][2]:.4f}"
        )
        lines.append(
            f"    image1 (before)  std  : "
            f"R={s['image1_std'][0]:.4f}  "
            f"G={s['image1_std'][1]:.4f}  "
            f"B={s['image1_std'][2]:.4f}"
        )
        lines.append(
            f"    image2 (after)   mean : "
            f"R={s['image2_mean'][0]:.4f}  "
            f"G={s['image2_mean'][1]:.4f}  "
            f"B={s['image2_mean'][2]:.4f}"
        )
        lines.append(
            f"    image2 (after)   std  : "
            f"R={s['image2_std'][0]:.4f}  "
            f"G={s['image2_std'][1]:.4f}  "
            f"B={s['image2_std'][2]:.4f}"
        )
        lines.append("")
        lines.append(
            "  Normalization check (expected mean≈0, std≈1 after ImageNet norm):"
        )
        for label, key_m, key_s in [
            ("image1", "image1_mean", "image1_std"),
            ("image2", "image2_mean", "image2_std"),
        ]:
            m = s[key_m]
            sv = s[key_s]
            ok_mean = all(abs(v) < 0.5 for v in m)
            ok_std  = all(0.5 < v < 2.0 for v in sv)
            lines.append(
                f"    {label}: mean in [-0.5, 0.5]? {'YES' if ok_mean else 'NO'}  |  "
                f"std in [0.5, 2.0]? {'YES' if ok_std else 'NO'}"
            )
        lines.append("")
        lines.append("-" * 70)
        lines.append("")

    # ImageNet reference (what correctly-normalised values look like)
    lines.append("Reference: ImageNet mean = (0.485, 0.456, 0.406)")
    lines.append("           ImageNet std  = (0.229, 0.224, 0.225)")
    lines.append(
        "After correct normalisation, channel means should be close to 0"
    )
    lines.append(
        "and stds close to 1 (assuming the dataset has ImageNet-like distribution)."
    )
    lines.append("=" * 70)

    text = "\n".join(lines)
    print(text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"\n  Statistics saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise OSCD preprocessed patches and print dataset statistics."
    )
    p.add_argument(
        "--data_root", default="processed_oscd",
        help="Path to the processed_oscd directory (default: processed_oscd)",
    )
    p.add_argument(
        "--out_dir", default="visualizations",
        help="Output directory for figures and stats (default: visualizations)",
    )
    p.add_argument(
        "--n_samples", type=int, default=4,
        help="Number of random samples to show per split (default: 4)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sample selection (default: 42)",
    )
    p.add_argument(
        "--max_stat_samples", type=int, default=500,
        help="Max patches to use when computing statistics (default: 500)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = ROOT / args.data_root
    out_dir   = ROOT / args.out_dir
    rng = random.Random(args.seed)

    print(f"\nOSCD Visualization Script")
    print(f"  data_root : {data_root}")
    print(f"  out_dir   : {out_dir}")
    print(f"  n_samples : {args.n_samples}")
    print(f"  seed      : {args.seed}")
    print()

    # ── sample figures ─────────────────────────────────────────────────────────
    print("── Sample Figures ──────────────────────────────────────────────────")
    for split in SPLITS:
        split_dir = data_root / split
        a_dir = split_dir / "A"
        if not a_dir.exists():
            print(f"  [skip] {split}: {a_dir} not found")
            continue

        stems = sorted(p.stem for p in a_dir.glob("*.png"))
        if not stems:
            print(f"  [skip] {split}: no PNG files found")
            continue

        n_draw = min(args.n_samples, len(stems))
        chosen = rng.sample(stems, n_draw)
        chosen.sort()      # deterministic row order in figure

        out_fig = out_dir / f"oscd_samples_{split}.png"
        print(f"\n  {split.upper()} split — {len(stems)} patches total, "
              f"drawing {n_draw} samples")
        make_sample_figure(split_dir, chosen, split, out_fig)

    # ── dataset statistics ─────────────────────────────────────────────────────
    print("\n── Dataset Statistics ──────────────────────────────────────────────")
    stats = compute_statistics(data_root, rng, max_stat_samples=args.max_stat_samples)
    if stats:
        out_stats = out_dir / "oscd_stats.txt"
        print()
        print_and_save_statistics(stats, out_stats)
    else:
        print("  No valid splits found — nothing to report.")

    print("\nDone.")


if __name__ == "__main__":
    main()
