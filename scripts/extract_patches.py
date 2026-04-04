"""
Extract 256x256 patch pairs from preprocessed OSCD .npy arrays and save as PNGs.

For each region the script:
  1. Loads img1_rgb.npy, img2_rgb.npy, mask.npy from the preprocessed directory.
  2. Normalises each image to [0, 1] via per-channel 2nd–98th percentile clipping
     (robust to cloud / shadow outliers in Sentinel-2 data) then converts to uint8.
  3. Tiles the image with non-overlapping 256×256 patches.
  4. Keeps:
       - all POSITIVE patches  (change ratio ≥ 1 %)
       - NEGATIVE patches      (change ratio  < 1 %) up to pos // 3  (3:1 ratio)
  5. Writes A/, B/, label/ PNGs under the target split directory.

Split assignment
  - Train regions  → sorted list from train.txt
  - Last 20 % (≈3 regions) become validation; the rest are training.
  - Test regions   → no labels available; label/ PNGs are saved as all-zero masks.

Usage
-----
    python scripts/extract_patches.py \
        --preprocessed_dir outputs/oscd_preprocessed \
        --images_dir "/path/to/.../Onera Satellite Change Detection dataset - Images" \
        --output_dir processed_oscd \
        [--patch_size 256] \
        [--min_change_ratio 0.01] \
        [--neg_pos_ratio 0.333]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────

def normalise_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Per-channel 2nd–98th percentile clip → scale to [0, 255] uint8.

    arr : (H, W, C) float32
    returns: (H, W, C) uint8
    """
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[2]):
        ch = arr[:, :, c]
        lo = float(np.percentile(ch, 2))
        hi = float(np.percentile(ch, 98))
        if hi - lo < 1e-6:          # flat channel (e.g. all-zero band)
            hi = lo + 1.0
        out[:, :, c] = (ch - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def extract_patches(
    img1: np.ndarray,
    img2: np.ndarray,
    mask: np.ndarray | None,
    patch_size: int,
    min_change_ratio: float,
    neg_pos_ratio: float,
    rng: random.Random,
    stride: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Return lists of (patch1, patch2, label_patch) for a single region.

    If mask is None (test set) every patch is kept; label patches are zeros.
    """
    H, W = img1.shape[:2]
    if stride is None:
        stride = patch_size
    all_pos, all_neg = [], []

    for row in range(0, H - patch_size + 1, stride):
        for col in range(0, W - patch_size + 1, stride):
            p1 = img1[row:row + patch_size, col:col + patch_size]
            p2 = img2[row:row + patch_size, col:col + patch_size]

            if mask is None:
                lbl = np.zeros((patch_size, patch_size), dtype=np.uint8)
                all_neg.append((p1, p2, lbl))
                continue

            lbl_raw = mask[row:row + patch_size, col:col + patch_size]
            change_ratio = lbl_raw.mean()           # mask is 0/1
            lbl = (lbl_raw * 255).astype(np.uint8)  # save as 0/255 PNG

            if change_ratio >= min_change_ratio:
                all_pos.append((p1, p2, lbl))
            else:
                all_neg.append((p1, p2, lbl))

    # ── sampling ──────────────────────────────────────────────
    n_pos  = len(all_pos)
    n_neg_target = max(1, int(n_pos * neg_pos_ratio))

    if mask is None:
        # test: keep everything (no label → can't filter)
        kept_pos = []
        kept_neg = all_neg
    else:
        kept_neg = rng.sample(all_neg, min(n_neg_target, len(all_neg)))
        kept_pos = all_pos

    # tag each tuple with its true category so callers can report accurately
    kept = [(p1, p2, lbl, True)  for p1, p2, lbl in kept_pos] + \
           [(p1, p2, lbl, False) for p1, p2, lbl in kept_neg]

    p1s   = [t[0] for t in kept]
    p2s   = [t[1] for t in kept]
    lbls  = [t[2] for t in kept]
    flags = [t[3] for t in kept]   # True = positive patch
    return p1s, p2s, lbls, flags, n_pos, len(all_neg)


def save_patches(
    p1s: list[np.ndarray],
    p2s: list[np.ndarray],
    lbls: list[np.ndarray],
    flags: list[bool],
    split_dir: Path,
    region: str,
    start_idx: int = 0,
) -> int:
    """Write patches to split_dir/{A,B,label}/ and return the count saved."""
    for sub in ("A", "B", "label"):
        (split_dir / sub).mkdir(parents=True, exist_ok=True)

    for i, (p1, p2, lbl) in enumerate(zip(p1s, p2s, lbls)):
        name = f"{region}_{start_idx + i:04d}.png"
        Image.fromarray(p1).save(split_dir / "A"     / name)
        Image.fromarray(p2).save(split_dir / "B"     / name)
        Image.fromarray(lbl).save(split_dir / "label" / name)

    return len(p1s), sum(flags), len(flags) - sum(flags)


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 256×256 patch pairs from OSCD .npy arrays.")
    parser.add_argument("--preprocessed_dir", default="outputs/oscd_preprocessed",
                        help="Root dir of .npy files produced by preprocess_oscd.py")
    parser.add_argument("--images_dir", required=True,
                        help='Path to "Onera Satellite Change Detection dataset - Images"')
    parser.add_argument("--output_dir", default="processed_oscd",
                        help="Output root for PNG patches (default: processed_oscd)")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride between patches (default: patch_size = non-overlapping). "
                             "Use e.g. 128 for 50%% overlap and ~4x more patches.")
    parser.add_argument("--min_change_ratio", type=float, default=0.01,
                        help="Minimum fraction of changed pixels to keep a positive patch (default 0.01)")
    parser.add_argument("--neg_pos_ratio", type=float, default=0.333,
                        help="Max ratio of negative-to-positive patches kept (default 0.333 → 3:1 pos:neg)")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                        help="Fraction of train regions held out for validation (default 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pre_root = Path(args.preprocessed_dir)
    out_root = Path(args.output_dir)
    images_root = Path(args.images_dir)
    stride = args.stride if args.stride is not None else args.patch_size

    # ── split assignment ──────────────────────────────────────
    train_all  = (images_root / "train.txt").read_text().strip().split(",")
    test_regions = (images_root / "test.txt").read_text().strip().split(",")

    n_val = max(1, int(len(train_all) * args.val_fraction))
    val_regions   = train_all[-n_val:]      # last n regions → val
    train_regions = train_all[:-n_val]      # rest → train

    print(f"\nOSCD Patch Extraction  (patch={args.patch_size}×{args.patch_size})")
    print(f"  Preprocessed dir : {pre_root}")
    print(f"  Output dir       : {out_root}")
    print(f"  Train regions    : {train_regions}")
    print(f"  Val   regions    : {val_regions}")
    print(f"  Test  regions    : {test_regions}")
    print(f"  Stride           : {stride}  ({'non-overlapping' if stride == args.patch_size else f'{args.patch_size/stride:.1f}x overlap'})")
    print(f"  Min change ratio : {args.min_change_ratio:.0%}")
    print(f"  Neg/pos ratio    : {args.neg_pos_ratio:.2f}  (≈ 3:1 pos:neg)\n")

    # ── per-split stats ───────────────────────────────────────
    stats: dict[str, dict] = {
        "train": {"pos": 0, "neg": 0, "total": 0},
        "val":   {"pos": 0, "neg": 0, "total": 0},
        "test":  {"pos": 0, "neg": 0, "total": 0},
    }
    region_stats: list[dict] = []

    splits_map = (
        [(r, "train") for r in train_regions] +
        [(r, "val")   for r in val_regions]   +
        [(r, "test")  for r in test_regions]
    )

    for region, split in tqdm(splits_map, desc="Regions", unit="region"):
        tqdm.write(f"  [{split:5s}] {region} ...")

        region_dir = pre_root / region
        img1 = np.load(region_dir / "img1_rgb.npy")
        img2 = np.load(region_dir / "img2_rgb.npy")

        mask_path = region_dir / "mask.npy"
        mask = np.load(mask_path) if mask_path.exists() else None

        # normalise to uint8
        img1_u8 = normalise_to_uint8(img1)
        img2_u8 = normalise_to_uint8(img2)

        H, W = img1.shape[:2]
        if H < args.patch_size or W < args.patch_size:
            tqdm.write(f"         SKIP — image {H}×{W} smaller than patch {args.patch_size}×{args.patch_size}")
            continue

        p1s, p2s, lbls, flags, n_pos_avail, n_neg_avail = extract_patches(
            img1=img1_u8,
            img2=img2_u8,
            mask=mask,
            patch_size=args.patch_size,
            min_change_ratio=args.min_change_ratio,
            neg_pos_ratio=args.neg_pos_ratio,
            rng=rng,
            stride=stride,
        )

        n_saved, n_saved_pos, n_saved_neg = save_patches(
            p1s, p2s, lbls, flags, out_root / split, region
        )

        stats[split]["pos"]   += n_saved_pos
        stats[split]["neg"]   += n_saved_neg
        stats[split]["total"] += n_saved

        region_stats.append({
            "region": region, "split": split,
            "image_size": f"{H}×{W}",
            "pos_avail": n_pos_avail, "neg_avail": n_neg_avail,
            "saved": n_saved, "saved_pos": n_saved_pos, "saved_neg": n_saved_neg,
        })
        tqdm.write(
            f"         size={H}×{W}  "
            f"pos_avail={n_pos_avail}  neg_avail={n_neg_avail}  "
            f"→ saved {n_saved} ({n_saved_pos} pos + {n_saved_neg} neg)"
        )

    # ── summary ──────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    for split, s in stats.items():
        total = s["total"]
        pos   = s["pos"]
        neg   = s["neg"]
        ratio = f"{pos / neg:.1f}:1" if neg else "∞"
        print(f"  {split:5s}  total={total:5d}  pos={pos:5d}  neg={neg:5d}  pos:neg={ratio}")

    grand = sum(s["total"] for s in stats.values())
    print(f"\n  Grand total patches : {grand}")
    print(f"  Output root         : {out_root.resolve()}")
    print(f"\n  Directory layout:")
    print(f"    {out_root}/train/A/       — before patches (RGB, uint8 PNG)")
    print(f"    {out_root}/train/B/       — after  patches (RGB, uint8 PNG)")
    print(f"    {out_root}/train/label/   — binary mask    (0 / 255 PNG)")
    print(f"    {out_root}/val/  ...      (same)")
    print(f"    {out_root}/test/ ...      (label = all-zero, no ground truth)\n")


if __name__ == "__main__":
    main()
