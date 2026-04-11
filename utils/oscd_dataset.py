"""
PyTorch Dataset and DataLoaders for LEVIR-CD change detection pairs.

Expected directory layout (LEVIR-CD ships pre-split):

    LEVIR CD/
    ├── train/
    │   ├── A/       *.png  — before image (RGB uint8, 1024×1024)
    │   ├── B/       *.png  — after  image (RGB uint8, 1024×1024)
    │   └── label/   *.png  — mask   (0 / 255 grayscale, 1024×1024)
    ├── val/   ...
    └── test/  ...

Each __getitem__ returns:
    {
        "image1": FloatTensor (3, H, W),   # before, ImageNet-normalised
        "image2": FloatTensor (3, H, W),   # after,  ImageNet-normalised
        "mask":   FloatTensor (1, H, W),   # binary  0.0 / 1.0
    }

Augmentation strategy (train only)
-----------------------------------
Albumentations is used so that the *same* spatial transform is applied
to image1, image2, and the mask in lock-step.  Colour / intensity
augmentations are applied only to the two images, never to the mask.

  1. PadIfNeeded(288×288) + RandomCrop(256×256)
       Slight translation: the 256px patch is padded to 288 with
       reflection and a random 256×256 window is cropped back out.
  2. HorizontalFlip  (p=0.5)
  3. VerticalFlip    (p=0.5)
  4. RandomRotate90  (p=0.5)
  5. Normalize with ImageNet mean/std  (applied to images only)

Val / test: only Normalize (no spatial or colour augmentation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image

import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# constants
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ──────────────────────────────────────────────────────────────────────────────
# transforms
# ──────────────────────────────────────────────────────────────────────────────

def _build_train_transform(patch_size: int = 256) -> A.Compose:
    """Spatial + colour augmentation with synchronised A / B / mask transforms.

    Crop strategy — full-image random crop
    ----------------------------------------
    Previous approach padded to patch_size+32 (288) then cropped, limiting
    the crop window to ±32 px around the image centre.  For LEVIR-CD images
    (1024×1024) that wasted 93 % of each image.

    New approach: PadIfNeeded only guarantees the image is at least patch_size
    wide/tall (handles rare cases where source images are smaller).  For the
    normal 1024×1024 case no padding is added and RandomCrop samples freely
    from the entire 1024×1024 spatial extent — 16× more crop positions than
    before.
    """
    return A.Compose(
        [
            # ── spatial (applied to image1, image2, mask) ──────────────────
            # Pad only if the image is smaller than the desired patch size.
            # For 1024×1024 LEVIR-CD images this is a no-op.
            A.PadIfNeeded(
                min_height=patch_size, min_width=patch_size,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
            # Random crop from the full image extent — not just ±32 px.
            A.RandomCrop(height=patch_size, width=patch_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # ── colour (images only; mask skipped via targets) ─────────────
            A.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.1, hue=0.05,
                p=0.5,
            ),
            # ── normalise ──────────────────────────────────────────────────
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
        additional_targets={"image2": "image"},   # sync image2 with image1
    )


def _build_val_transform(patch_size: int = 256) -> A.Compose:
    """Center-crop to patch_size then normalise — no random augmentation."""
    return A.Compose(
        [
            A.CenterCrop(height=patch_size, width=patch_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
        additional_targets={"image2": "image"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# dataset
# ──────────────────────────────────────────────────────────────────────────────

class OSCDDataset(Dataset):
    """Patch-pair dataset for LEVIR-CD change detection.

    Parameters
    ----------
    root : str | Path
        Path to the split directory, e.g. ``LEVIR CD/train``.
    split : 'train' | 'val' | 'test'
        Controls which augmentation pipeline is applied.
    patch_size : int
        Expected spatial size of patches (default 256).
    transform : A.Compose | None
        Override the default Albumentations pipeline.
    n_crops : int
        Number of random crops to sample per image per epoch (train only).
        Each crop is independently augmented, so the same image yields
        n_crops distinct training samples.  Default 1 (original behaviour).
        Val/test always use n_crops=1 (centre crop, no repetition).

        Example: 411 train images × n_crops=4 → 1 644 patches/epoch.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        patch_size: int = 256,
        transform: A.Compose | None = None,
        n_crops: int = 1,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        # Only multiply training data; val/test always use a single crop.
        self.n_crops = n_crops if split == "train" else 1

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = _build_train_transform(patch_size)
        else:
            self.transform = _build_val_transform(patch_size)

        # collect stems present in ALL THREE of A/, B/, label/
        a_dir   = self.root / "A"
        b_dir   = self.root / "B"
        lbl_dir = self.root / "label"
        if not a_dir.exists() or not list(a_dir.glob("*.png")):
            import warnings
            warnings.warn(
                f"No patches found for split '{split}' under {self.root}. "
                "This split will be empty."
            )
            self.stems = []
            return
        a_stems   = set(p.stem for p in a_dir.glob("*.png"))
        b_stems   = set(p.stem for p in b_dir.glob("*.png")) if b_dir.exists() else set()
        lbl_stems = set(p.stem for p in lbl_dir.glob("*.png")) if lbl_dir.exists() else a_stems
        common = a_stems & b_stems & lbl_stems
        dropped = len(a_stems) - len(common)
        if dropped:
            import warnings
            warnings.warn(
                f"Split '{split}': dropped {dropped} stems missing from A/B/label. "
                f"Using {len(common)} complete triplets."
            )
        self.stems = sorted(common)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.stems) * self.n_crops

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Map virtual index back to a real image stem.
        # Indices 0..n_crops-1 all map to stems[0], etc.
        # Since RandomCrop is stochastic, each call naturally yields a
        # different crop location — no extra logic needed.
        stem = self.stems[idx % len(self.stems)]
        fname = stem + ".png"

        # ── load images (RGB uint8 H×W×3) ──────────────────────────────
        img1 = np.array(Image.open(self.root / "A"     / fname).convert("RGB"))
        img2 = np.array(Image.open(self.root / "B"     / fname).convert("RGB"))

        # ── load mask (grayscale H×W, values 0 or 255) ─────────────────
        mask_raw = np.array(Image.open(self.root / "label" / fname).convert("L"))
        # binarise: 255 → 1,  0 → 0
        mask = (mask_raw > 127).astype(np.uint8)

        # ── augment ─────────────────────────────────────────────────────
        result = self.transform(image=img1, image2=img2, mask=mask)
        img1_aug = result["image"]    # (H, W, 3) float32, normalised
        img2_aug = result["image2"]   # (H, W, 3) float32, normalised
        mask_aug = result["mask"]     # (H, W)    uint8

        # ── to tensor ───────────────────────────────────────────────────
        def _to_chw(arr: np.ndarray) -> torch.Tensor:
            """HWC → CHW float32 tensor."""
            return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float()

        t1   = _to_chw(img1_aug)                                    # (3, H, W)
        t2   = _to_chw(img2_aug)                                    # (3, H, W)
        tmsk = torch.from_numpy(mask_aug).unsqueeze(0).float()      # (1, H, W)

        return {"image1": t1, "image2": t2, "mask": tmsk}


# ──────────────────────────────────────────────────────────────────────────────
# dataloaders
# ──────────────────────────────────────────────────────────────────────────────

def get_oscd_dataloaders(
    data_root: str | Path = "LEVIR CD",
    patch_size: int = 256,
    train_batch_size: int = 8,
    eval_batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    n_crops: int = 1,
) -> dict[str, DataLoader]:
    """Return {'train': ..., 'val': ..., 'test': ...} DataLoaders.

    Parameters
    ----------
    data_root : path
        Root of the LEVIR-CD directory tree (contains train/, val/, test/).
    patch_size : int
        Spatial size expected for each patch (default 256).
    train_batch_size : int
        Batch size for the training loader (default 8).
    eval_batch_size : int
        Batch size for val and test loaders (default 16).
    num_workers : int
        Dataloader workers (default 4).
    pin_memory : bool
        Pin memory for faster GPU transfers (default True).
    n_crops : int
        Random crops sampled per training image per epoch (default 1).
        Val/test are unaffected.
    """
    data_root = Path(data_root)

    datasets = {
        "train": OSCDDataset(
            data_root / "train", split="train",
            patch_size=patch_size, n_crops=n_crops,
        ),
        "val":  OSCDDataset(data_root / "val",  split="val",  patch_size=patch_size),
        "test": OSCDDataset(data_root / "test", split="test", patch_size=patch_size),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,       # keep batch size consistent during training
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    # ── print summary ───────────────────────────────────────────────────
    print("LEVIR-CD DataLoaders")
    print(f"  data_root  : {data_root.resolve()}")
    for split, ds in datasets.items():
        bs = train_batch_size if split == "train" else eval_batch_size
        n_batches = len(loaders[split])
        n_imgs = len(ds.stems)
        crops_str = f"  ×{ds.n_crops} crops" if ds.n_crops > 1 else ""
        print(
            f"  {split:5s}  images={n_imgs:4d}{crops_str}  "
            f"patches={len(ds):4d}  batch_size={bs}  batches={n_batches}"
        )

    return loaders
