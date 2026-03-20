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

import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import rasterio
import tifffile
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
