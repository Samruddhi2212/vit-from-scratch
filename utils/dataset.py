"""
Dataset loading and augmentation for ViT training.

═══════════════════════════════════════════════════════════
WHY DATA AUGMENTATION IS CRITICAL FOR ViT:
═══════════════════════════════════════════════════════════

CNNs have built-in "inductive biases" — translation equivariance
(weight sharing) and locality (small receptive fields). These biases
act as implicit regularization, helping the model learn from less data.

ViTs have NONE of these biases. Self-attention is global from layer 1,
and there's no weight sharing across spatial positions. This means:
  - ViTs are more flexible (can learn any pattern)
  - But they need MORE data or MORE augmentation to avoid overfitting

Our augmentation strategy:
  1. RandomCrop with padding — slight spatial translation
  2. RandomHorizontalFlip — horizontal symmetry
  3. RandAugment — automated augmentation (rotations, color shifts, etc.)
  4. Normalize — standardize pixel values
  5. RandomErasing — randomly mask out rectangles (similar to dropout)
"""

import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import rasterio
#─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class V_Dataset(Dataset):

    def __init__(self, image_dir_list, image_size=(224, 224), transform=None):
        self.image_dir_list = image_dir_list
        self.image_size     = image_size

        # Default transform mirrors your /255.0 normalization
        # Replace Normalize values with your dataset's actual mean/std if known
        self.transform = transform or T.Compose([
            T.Resize(image_size),
            T.ToTensor(),                          # HWC uint8 → CHW float [0,1]
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),  # ImageNet stats (good for ViT)
        ])

    def _load_tif(self, path):
        """
        Loads a .tif file and returns a float32 tensor (C, H, W) in [0, 1].
        Handles both:
          - 3-band RGB TIFs  (standard satellite RGB)
          - Multi-band TIFs  (e.g. 4-band RGBI, 8-band multispectral)
        Only the first 3 bands are used for ViT (pretrained on RGB).
        """
        with rasterio.open(path) as src:
            # Read first 3 bands → (3, H, W) uint16 or uint8
            bands = src.read([1, 2, 3]) if src.count >= 3 else src.read()
            img = bands.astype(np.float32)
        # Normalize to [0, 1] based on dtype range
        if img.max() > 1.0:
            img = img / (np.info(np.uint16).max if img.max() > 255 else 255.0)

        img = np.clip(img, 0, 1)
        h, w = self.image_size
        # Resize each band independently then stack
        img = np.stack([
            np.array(Image.fromarray((b * 255).astype(np.uint8)).resize((w, h)))
            for b in img
        ], axis=0).astype(np.float32) / 255.0   # (C, H, W)

        return torch.tensor(img)

    def __len__(self):
        return len(self.image_dir_list)

    def __getitem__(self, index):
        image_path, label_path = self.image_dir_list[index]

        # ── Load image ──────────────────────────────────────────────────────
        img = self._load_tif(image_path)        # → tensor (C, H, W)
        img = self.transform(img)

        # ── Load label ──────────────────────────────────────────────────────
        with open(label_path, "r") as f:
            first_line = f.readline().strip().split()
            label = int(first_line[0])

        return img, torch.tensor(label, dtype=torch.long)

# ─────────────────────────────────────────────
# DATALOADERS
# ─────────────────────────────────────────────
def build_loaders(train_file_list, val_file_list, test_file_list,
                  image_size=(224, 224), batch_size=16, num_workers=4):

    # Augmentation for training only
    train_transform = T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = V_Dataset(train_file_list, image_size, transform=train_transform)
    val_dataset   = V_Dataset(val_file_list,   image_size, transform=val_test_transform)
    test_dataset  = V_Dataset(test_file_list,  image_size, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ─── CIFAR-10 (classification ViT: notebooks, scripts/train_cifar10.py, ablations) ───

CIFAR10_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def denormalize_cifar10(tensor: torch.Tensor) -> torch.Tensor:
    """Undo CIFAR-10 Normalize for visualization. Accepts (C, H, W) or (N, C, H, W)."""
    mean = torch.tensor(CIFAR10_MEAN, device=tensor.device, dtype=tensor.dtype)
    std = torch.tensor(CIFAR10_STD, device=tensor.device, dtype=tensor.dtype)
    if tensor.dim() == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


class _CIFARSubset(Dataset):
    """CIFAR-10 train split with per-split transforms."""

    def __init__(self, root, indices, transform, download=True):
        self.ds = torchvision.datasets.CIFAR10(
            root=root, train=True, download=download, transform=None
        )
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, y = self.ds[self.indices[i]]
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def get_cifar10_loaders(config, num_workers=2, val_fraction=0.1, seed=42):
    """
    Train / val split from official CIFAR-10 train (50k); test = official test (10k).

    ``config`` must provide ``batch_size`` and ``image_size`` (e.g. ``ViTConfig`` for 32×32).
    Downloads to ``<repo>/data/cifar10/`` (ignored by git).
    """
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cifar10")
    os.makedirs(root, exist_ok=True)

    mean, std = CIFAR10_MEAN, CIFAR10_STD

    train_tf = T.Compose(
        [
            T.RandomCrop(config.image_size, padding=4),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.25),
        ]
    )
    eval_tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    n_train = 50000
    n_val = int(n_train * val_fraction)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_train, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_set = _CIFARSubset(root, train_idx, train_tf)
    val_set = _CIFARSubset(root, val_idx, eval_tf)
    test_set = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=eval_tf
    )

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
