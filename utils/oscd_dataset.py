"""
PyTorch Dataset and DataLoaders for preprocessed OSCD patch pairs.

Expected directory layout (produced by scripts/extract_patches.py):

    processed_oscd/
    ├── train/
    │   ├── A/       {region}_{idx:04d}.png  — before image (RGB uint8)
    │   ├── B/       {region}_{idx:04d}.png  — after  image (RGB uint8)
    │   └── label/   {region}_{idx:04d}.png  — mask   (0 / 255 grayscale)
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
    """Spatial + colour augmentation with synchronised A / B / mask transforms."""
    pad = patch_size + 32                     # pad to 288 before crop
    return A.Compose(
        [
            # ── spatial (applied to image1, image2, mask) ──────────────────
            A.PadIfNeeded(
                min_height=pad, min_width=pad,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
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


def _build_val_transform() -> A.Compose:
    """Normalise only — no augmentation."""
    return A.Compose(
        [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)],
        additional_targets={"image2": "image"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# dataset
# ──────────────────────────────────────────────────────────────────────────────

class OSCDDataset(Dataset):
    """Patch-pair dataset for OSCD change detection.

    Parameters
    ----------
    root : str | Path
        Path to the split directory, e.g. ``processed_oscd/train``.
    split : 'train' | 'val' | 'test'
        Controls which augmentation pipeline is applied.
    patch_size : int
        Expected spatial size of patches (default 256).
    transform : A.Compose | None
        Override the default Albumentations pipeline.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        patch_size: int = 256,
        transform: A.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = _build_train_transform(patch_size)
        else:
            self.transform = _build_val_transform()

        # collect file stems in sorted order so A / B / label are aligned
        a_dir = self.root / "A"
        if not a_dir.exists():
            raise FileNotFoundError(
                f"A/ directory not found under {self.root}. "
                "Run scripts/extract_patches.py first."
            )
        self.stems = sorted(p.stem for p in a_dir.glob("*.png"))
        if not self.stems:
            raise ValueError(f"No PNG files found in {a_dir}")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.stems)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        stem = self.stems[idx]
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
    data_root: str | Path = "processed_oscd",
    patch_size: int = 256,
    train_batch_size: int = 8,
    eval_batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """Return {'train': ..., 'val': ..., 'test': ...} DataLoaders.

    Parameters
    ----------
    data_root : path
        Root of the processed_oscd directory tree.
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
    """
    data_root = Path(data_root)

    datasets = {
        split: OSCDDataset(data_root / split, split=split, patch_size=patch_size)
        for split in ("train", "val", "test")
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
    print("OSCD DataLoaders")
    print(f"  data_root  : {data_root.resolve()}")
    for split, ds in datasets.items():
        bs = train_batch_size if split == "train" else eval_batch_size
        n_batches = len(loaders[split])
        print(
            f"  {split:5s}  patches={len(ds):4d}  "
            f"batch_size={bs}  batches={n_batches}"
        )

    return loaders
