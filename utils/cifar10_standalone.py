"""
CIFAR-10 train/val/test loaders without torchvision (torch + NumPy + PIL only).

Used when torchvision is unavailable (e.g. HPC compute nodes with torch but no pip).
Download the official tarball once on a machine with HTTPS (typically the login node)
into ``data/cifar10/``; compute jobs then read from shared NFS.

Augmentations mirror the intent of ``get_cifar10_loaders`` (crop, flip, jitter, erase)
but omit torchvision's RandAugment for a lighter substitute (color jitter + small rotation).
"""

from __future__ import annotations

import os
import pickle
import random
import tarfile
import urllib.request
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

# Must match utils/dataset.py (denormalize_cifar10)
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)

_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _unpickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _reshape_batch(data: np.ndarray) -> np.ndarray:
    """(N, 3072) uint8 -> (N, 32, 32, 3) uint8."""
    return data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def ensure_cifar10_downloaded(root: str) -> None:
    """Download and extract official CIFAR-10 if missing."""
    extracted = os.path.join(root, "cifar-10-batches-py")
    if os.path.isdir(extracted) and os.path.isfile(os.path.join(extracted, "data_batch_1")):
        return
    os.makedirs(root, exist_ok=True)
    tar_path = os.path.join(root, "cifar-10-python.tar.gz")
    if not os.path.isfile(tar_path):
        urllib.request.urlretrieve(_CIFAR_URL, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(root)


def _load_train_val_arrays(root: str) -> Tuple[np.ndarray, np.ndarray]:
    """Full training set: (50000, 32, 32, 3) uint8, labels (50000,)."""
    base = os.path.join(root, "cifar-10-batches-py")
    xs, ys = [], []
    for i in range(1, 6):
        d = _unpickle(os.path.join(base, f"data_batch_{i}"))
        xs.append(_reshape_batch(d[b"data"]))
        ys.extend(d[b"labels"])
    x = np.concatenate(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    return x, y


def _load_test_arrays(root: str) -> Tuple[np.ndarray, np.ndarray]:
    base = os.path.join(root, "cifar-10-batches-py")
    d = _unpickle(os.path.join(base, "test_batch"))
    x = _reshape_batch(d[b"data"])
    y = np.asarray(d[b"labels"], dtype=np.int64)
    return x, y


def _to_tensor_chw(img_hwc_uint8: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img_hwc_uint8).permute(2, 0, 1).float() / 255.0
    return t


def _normalize(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(_CIFAR_MEAN, dtype=t.dtype, device=t.device).view(3, 1, 1)
    std = torch.tensor(_CIFAR_STD, dtype=t.dtype, device=t.device).view(3, 1, 1)
    return (t - mean) / std


def _random_erasing(t: torch.Tensor, p: float = 0.25) -> torch.Tensor:
    if random.random() >= p:
        return t
    _, h, w = t.shape
    area = h * w
    for _ in range(10):
        erase_area = random.uniform(0.02, 0.15) * area
        aspect = random.uniform(0.3, 3.3)
        eh = int(round((erase_area * aspect) ** 0.5))
        ew = int(round((erase_area / aspect) ** 0.5))
        if eh < h and ew < w:
            i = random.randint(0, h - eh)
            j = random.randint(0, w - ew)
            t = t.clone()
            t[:, i : i + eh, j : j + ew] = 0.0
            return t
    return t


def _train_augment(
    img_hwc: np.ndarray, image_size: int, train: bool
) -> torch.Tensor:
    """PIL-free augment for training; eval path is deterministic normalize."""
    if not train:
        t = _to_tensor_chw(img_hwc)
        if image_size != 32:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False
            ).squeeze(0)
        return _normalize(t)

    arr = np.asarray(img_hwc, dtype=np.uint8)
    pad = 4
    arr = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    h, w = arr.shape[0], arr.shape[1]
    top = random.randint(0, h - 32)
    left = random.randint(0, w - 32)
    arr = arr[top : top + 32, left : left + 32].copy()
    if random.random() < 0.5:
        arr = np.fliplr(arr).copy()
    pil = Image.fromarray(arr)
    if random.random() < 0.5:
        arr = np.asarray(pil).astype(np.float32)
        for ch in range(3):
            arr[..., ch] = np.clip(
                arr[..., ch] * (0.85 + 0.3 * random.random()), 0, 255
            )
        pil = Image.fromarray(arr.astype(np.uint8))
    if random.random() < 0.3:
        pil = pil.rotate(random.uniform(-15, 15), fillcolor=(128, 128, 128))
    arr = np.asarray(pil, dtype=np.uint8)
    t = _to_tensor_chw(arr)
    if image_size != 32:
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False
        ).squeeze(0)
    t = _normalize(t)
    return _random_erasing(t, p=0.25)


class _CIFAR10Standalone(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        train: bool,
        image_size: int,
    ):
        self.images = images
        self.labels = labels
        self.train = train
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.images[idx]
        y = self.labels[idx]
        t = _train_augment(x, self.image_size, train=self.train)
        return t, torch.tensor(y, dtype=torch.long)


def get_cifar10_loaders_standalone(
    config,
    num_workers: int = 2,
    val_fraction: float = 0.1,
    seed: int = 42,
):
    """Same return type as ``get_cifar10_loaders`` — no torchvision."""
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cifar10")
    os.makedirs(root, exist_ok=True)
    ensure_cifar10_downloaded(root)

    train_x, train_y = _load_train_val_arrays(root)
    test_x, test_y = _load_test_arrays(root)

    n_train = 50000
    n_val = int(n_train * val_fraction)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_train, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    image_size = int(config.image_size)

    train_set = _CIFAR10Standalone(train_x[train_idx], train_y[train_idx], True, image_size)
    val_set = _CIFAR10Standalone(train_x[val_idx], train_y[val_idx], False, image_size)
    test_set = _CIFAR10Standalone(test_x, test_y, False, image_size)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader, test_loader
