"""
Preprocess the OSCD (Onera Satellite Change Detection) dataset.

For each region:
  - Stack all 13 Sentinel-2 bands from imgs_1_rect/ and imgs_2_rect/
    into (H, W, 13) arrays, resampled to 10m resolution.
  - Extract RGB sub-stack [B04, B03, B02] -> (H, W, 3).
  - Load the binary change mask (cm.png), resize to 10m dims, binarize.
  - Save .npy files under outputs/oscd_preprocessed/<region>/.

Usage:
    python scripts/preprocess_oscd.py \
        --images_dir "/path/to/Onera Satellite Change Detection dataset - Images" \
        --labels_dir "/path/to/Onera Satellite Change Detection dataset - Train Labels" \
        --output_dir outputs/oscd_preprocessed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image
from tqdm import tqdm

# canonical band order and their native Sentinel-2 resolution
BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06",
              "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

# indices of RGB bands within BAND_ORDER (0-based)
# B04=index 3 (Red), B03=index 2 (Green), B02=index 1 (Blue)
RGB_INDICES = [3, 2, 1]  # R, G, B


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_and_resample_band(tif_path: Path, target_height: int, target_width: int) -> np.ndarray:
    """Read a single-band GeoTIFF and resample to (target_height, target_width).

    Returns a float32 array of shape (H, W).
    """
    with rasterio.open(tif_path) as src:
        if src.height == target_height and src.width == target_width:
            data = src.read(1).astype(np.float32)
        else:
            data = src.read(
                1,
                out_shape=(target_height, target_width),
                resampling=Resampling.bilinear,
            ).astype(np.float32)
    return data


def _reference_shape(img_dir: Path) -> tuple[int, int]:
    """Return (H, W) of the 10m reference band (B04)."""
    with rasterio.open(img_dir / "B04.tif") as src:
        return src.height, src.width


def stack_bands(img_dir: Path, target_h: int, target_w: int) -> np.ndarray:
    """Stack all available bands into (H, W, C) float32, resampled to 10m grid.

    Skips missing band files so regions with incomplete downloads still process.
    """
    arrays = []
    for band in BAND_ORDER:
        tif = img_dir / f"{band}.tif"
        if not tif.exists():
            continue
        arrays.append(_load_and_resample_band(tif, target_h, target_w))
    return np.stack(arrays, axis=-1)  # (H, W, C)


def load_mask(mask_path: Path, target_h: int, target_w: int) -> np.ndarray:
    """Load cm.png, resize to (target_h, target_w), return binary uint8 (0/1)."""
    img = Image.open(mask_path).convert("L")  # force grayscale
    if img.height != target_h or img.width != target_w:
        img = img.resize((target_w, target_h), Image.NEAREST)
    mask = np.array(img, dtype=np.uint8)
    mask = (mask > 127).astype(np.uint8)  # 255 -> 1, 0 -> 0
    return mask


# ---------------------------------------------------------------------------
# per-region processing
# ---------------------------------------------------------------------------

def process_region(
    region: str,
    images_root: Path,
    labels_root: Path | None,
    output_root: Path,
    split: str,
) -> dict:
    """Process one region and save .npy artefacts. Returns a metadata dict."""
    region_img_dir = images_root / region
    out_dir = output_root / region
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs_1 = region_img_dir / "imgs_1_rect"
    imgs_2 = region_img_dir / "imgs_2_rect"

    # --- reference shape from 10m B04 band ---
    target_h, target_w = _reference_shape(imgs_1)

    # --- RGB bands only [R=B04, G=B03, B=B02] ---
    # Load the three RGB bands directly — avoids failures on regions
    # where other bands (B07, B8A, etc.) are missing from the download.
    rgb1 = np.stack([
        _load_and_resample_band(imgs_1 / "B04.tif", target_h, target_w),
        _load_and_resample_band(imgs_1 / "B03.tif", target_h, target_w),
        _load_and_resample_band(imgs_1 / "B02.tif", target_h, target_w),
    ], axis=-1)  # (H, W, 3)
    rgb2 = np.stack([
        _load_and_resample_band(imgs_2 / "B04.tif", target_h, target_w),
        _load_and_resample_band(imgs_2 / "B03.tif", target_h, target_w),
        _load_and_resample_band(imgs_2 / "B02.tif", target_h, target_w),
    ], axis=-1)  # (H, W, 3)

    np.save(out_dir / "img1_rgb.npy", rgb1)
    np.save(out_dir / "img2_rgb.npy", rgb2)

    # --- mask (train only) ---
    mask_info = {}
    if labels_root is not None:
        mask_path = labels_root / region / "cm" / "cm.png"
        if mask_path.exists():
            mask = load_mask(mask_path, target_h, target_w)
            np.save(out_dir / "mask.npy", mask)
            change_pct = mask.mean() * 100
            mask_info = {"mask_saved": True, "change_pct": round(change_pct, 2)}
        else:
            mask_info = {"mask_saved": False, "reason": "cm.png not found"}
    else:
        mask_info = {"mask_saved": False, "reason": "no labels dir (test set)"}

    return {
        "region": region,
        "split": split,
        "shape_10m": (target_h, target_w),
        "stack_shape": stack1.shape,
        "rgb_shape": rgb1.shape,
        **mask_info,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess OSCD dataset to .npy files.")
    parser.add_argument(
        "--images_dir",
        required=True,
        help='Path to "Onera Satellite Change Detection dataset - Images"',
    )
    parser.add_argument(
        "--labels_dir",
        default=None,
        help='Path to "Onera Satellite Change Detection dataset - Train Labels" (optional)',
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/oscd_preprocessed",
        help="Root output directory for .npy files (default: outputs/oscd_preprocessed)",
    )
    args = parser.parse_args()

    images_root = Path(args.images_dir)
    labels_root = Path(args.labels_dir) if args.labels_dir else None
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # read official splits if available, otherwise auto-discover from directories
    train_txt = images_root / "train.txt"
    test_txt  = images_root / "test.txt"

    if train_txt.exists() and test_txt.exists():
        train_regions = train_txt.read_text().strip().split(",")
        test_regions  = test_txt.read_text().strip().split(",")
    else:
        # auto-discover: folders that have a matching label are train, rest are test
        all_dirs = sorted(
            d.name for d in images_root.iterdir()
            if d.is_dir() and (d / "imgs_1_rect").exists()
        )
        if labels_root is not None:
            train_regions = [d for d in all_dirs if (labels_root / d / "cm" / "cm.png").exists()]
            test_regions  = [d for d in all_dirs if d not in train_regions]
        else:
            train_regions = all_dirs
            test_regions  = []
        print("  Note: train.txt/test.txt not found — regions auto-discovered from directories.")

    train_regions = [r.strip() for r in train_regions if r.strip()]
    test_regions  = [r.strip() for r in test_regions  if r.strip()]

    all_regions = [(r, "train") for r in train_regions] + \
                  [(r, "test")  for r in test_regions]

    print(f"\nOSCD Preprocessing")
    print(f"  Images root : {images_root}")
    print(f"  Labels root : {labels_root}")
    print(f"  Output root : {output_root}")
    print(f"  Train       : {len(train_regions)} regions")
    print(f"  Test        : {len(test_regions)} regions")
    print(f"  Band order  : {BAND_ORDER}")
    print(f"  RGB bands   : B04 (R), B03 (G), B02 (B)\n")

    results = []
    for region, split in tqdm(all_regions, desc="Regions", unit="region"):
        tqdm.write(f"  [{split:5s}] {region} ...")
        try:
            meta = process_region(
                region=region,
                images_root=images_root,
                labels_root=labels_root if split == "train" else None,
                output_root=output_root,
                split=split,
            )
            results.append(meta)
            mask_str = f"  change={meta['change_pct']:.1f}%" if meta.get("change_pct") is not None else f"  {meta.get('reason','')}"
            tqdm.write(
                f"         shape={meta['shape_10m']}  "
                f"all_bands={meta['stack_shape']}  "
                f"rgb={meta['rgb_shape']}"
                f"{mask_str}"
            )
        except Exception as exc:
            tqdm.write(f"         ERROR: {exc}")
            results.append({"region": region, "split": split, "error": str(exc)})

    # --- summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ok     = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    print(f"  Processed : {len(ok)}/{len(results)} regions")
    if failed:
        print(f"  Failed    : {[r['region'] for r in failed]}")

    train_ok = [r for r in ok if r["split"] == "train"]
    if train_ok:
        heights = [r["shape_10m"][0] for r in train_ok]
        widths  = [r["shape_10m"][1] for r in train_ok]
        changes = [r["change_pct"] for r in train_ok if r.get("change_pct") is not None]
        print(f"  Train H range     : {min(heights)} – {max(heights)}")
        print(f"  Train W range     : {min(widths)} – {max(widths)}")
        if changes:
            print(f"  Change % range    : {min(changes):.1f}% – {max(changes):.1f}%  (mean {np.mean(changes):.1f}%)")

    print(f"\n  Output files per region:")
    print(f"    img1_all_bands.npy  (H, W, 13) float32 — all bands, time 1")
    print(f"    img2_all_bands.npy  (H, W, 13) float32 — all bands, time 2")
    print(f"    img1_rgb.npy        (H, W,  3) float32 — RGB [R,G,B], time 1")
    print(f"    img2_rgb.npy        (H, W,  3) float32 — RGB [R,G,B], time 2")
    print(f"    mask.npy            (H, W)     uint8   — binary change mask (train only)")
    print(f"\n  All saved under: {output_root.resolve()}\n")


if __name__ == "__main__":
    main()
