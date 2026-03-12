"""
Dataset loading and augmentation for ViT training.

This file handles:
1. CIFAR-10 data loading with proper train/val/test splits
2. Data augmentation (critical for ViT — explained below)
3. EuroSAT data loading (for the application phase)

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

The DeiT paper (Touvron et al., 2021) showed that with strong
augmentation, ViTs can match CNNs even on small datasets like
CIFAR-10 and ImageNet-1K (without needing ImageNet-21K pretraining).

Our augmentation strategy:
  1. RandomCrop with padding — slight spatial translation
  2. RandomHorizontalFlip — horizontal symmetry
  3. RandAugment — automated augmentation (rotations, color shifts, etc.)
  4. Normalize — standardize pixel values
  5. RandomErasing — randomly mask out rectangles (similar to dropout)
"""

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import ViTConfig, EuroSATConfig


def get_cifar10_loaders(
    config: ViTConfig,
    data_dir: str = "./data",
    val_split: float = 0.1,
    num_workers: int = 2
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create CIFAR-10 train/val/test data loaders.
    
    Args:
        config: ViTConfig with batch_size and image_size
        data_dir: Where to download/find CIFAR-10
        val_split: Fraction of training data to use for validation
        num_workers: Number of parallel data loading workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # ─── Training augmentation ───
    # Strong augmentation is essential for ViT on small datasets
    train_transform = transforms.Compose([
        # RandomCrop: randomly crop a 32×32 region from a 40×40 padded image
        # This gives the model slight translation invariance
        # (ViT doesn't have this built-in like CNNs)
        transforms.RandomCrop(config.image_size, padding=4),
        
        # RandomHorizontalFlip: 50% chance to mirror the image
        # Most objects look similar flipped horizontally
        transforms.RandomHorizontalFlip(p=0.5),
        
        # RandAugment: automatically applies 2 random augmentations
        # from a set of ~14 options (rotation, color, sharpness, etc.)
        # magnitude=9 controls how strong each augmentation is (0-30 scale)
        # This is the key augmentation that makes ViT work on small datasets
        transforms.RandAugment(num_ops=2, magnitude=9),
        
        # Convert PIL image to tensor and scale pixels from [0, 255] to [0, 1]
        transforms.ToTensor(),
        
        # Normalize to mean=0, std=1 using CIFAR-10 channel statistics
        # These numbers are precomputed over the entire CIFAR-10 training set
        # Red channel:   mean=0.4914, std=0.2470
        # Green channel: mean=0.4822, std=0.2435
        # Blue channel:  mean=0.4465, std=0.2616
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
        
        # RandomErasing: randomly erase a rectangle in the image
        # This forces the model to not rely on any single region
        # Similar idea to dropout, but in input space
        # p=0.25 means 25% chance of erasing
        transforms.RandomErasing(p=0.25),
    ])
    
    # ─── Validation/Test transforms ───
    # NO augmentation — we want to evaluate on clean, unmodified images
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])
    
    # ─── Download and load CIFAR-10 ───
    # torchvision handles downloading automatically
    train_dataset_full = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # We need a separate dataset object with eval transforms for validation
    val_dataset_full = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,  # Already downloaded above
        transform=eval_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform
    )
    
    # ─── Split training data into train and validation ───
    # We use a fixed random seed so the split is reproducible
    # (same images in train/val every time you run)
    total_train = len(train_dataset_full)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size
    
    # Generate indices for the split
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(total_train),
        [train_size, val_size],
        generator=generator
    )
    
    # Create subset datasets using the indices
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    # ─── Create DataLoaders ───
    # DataLoader handles batching, shuffling, and parallel loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,        # Shuffle training data every epoch
        num_workers=num_workers,
        pin_memory=True,     # Faster GPU transfer
        drop_last=True       # Drop incomplete last batch (for stable batch norm)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,       # No shuffling for evaluation
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# CIFAR-10 class names for visualization
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Normalization statistics (needed for de-normalizing images for display)
CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)


def denormalize_cifar10(images: torch.Tensor) -> torch.Tensor:
    """
    Reverse the normalization to get images back to [0, 1] range
    for display with matplotlib.
    
    Args:
        images: Normalized images, shape [B, 3, H, W]
    
    Returns:
        De-normalized images, clamped to [0, 1]
    """
    return (images * CIFAR10_STD + CIFAR10_MEAN).clamp(0, 1)


# ──────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    config = ViTConfig()
    
    print("=" * 60)
    print("Downloading CIFAR-10 and creating data loaders...")
    print("=" * 60)
    
    train_loader, val_loader, test_loader = get_cifar10_loaders(config)
    
    print(f"\nDataset sizes:")
    print(f"  Training:   {len(train_loader.dataset):,} images")
    print(f"  Validation: {len(val_loader.dataset):,} images")
    print(f"  Test:       {len(test_loader.dataset):,} images")
    
    print(f"\nBatch configuration:")
    print(f"  Batch size:      {config.batch_size}")
    print(f"  Training batches:   {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches:       {len(test_loader)}")
    
    # ─── Verify a training batch ───
    print("\n" + "=" * 60)
    print("Verifying training batch...")
    print("=" * 60)
    
    images, labels = next(iter(train_loader))
    print(f"  Image batch shape: {images.shape}")
    print(f"  Label batch shape: {labels.shape}")
    print(f"  Image dtype:       {images.dtype}")
    print(f"  Label dtype:       {labels.dtype}")
    print(f"  Pixel range:       [{images.min():.2f}, {images.max():.2f}]")
    print(f"  Label range:       [{labels.min()}, {labels.max()}]")
    
    assert images.shape == (config.batch_size, 3, config.image_size, config.image_size), \
        f"Unexpected image shape: {images.shape}"
    assert labels.shape == (config.batch_size,), \
        f"Unexpected label shape: {labels.shape}"
    assert labels.min() >= 0 and labels.max() <= 9, \
        f"Labels out of range: [{labels.min()}, {labels.max()}]"
    
    # ─── Verify a validation batch (should have no augmentation) ───
    print("\n" + "=" * 60)
    print("Verifying validation batch...")
    print("=" * 60)
    
    val_images, val_labels = next(iter(val_loader))
    print(f"  Image batch shape: {val_images.shape}")
    print(f"  Pixel range:       [{val_images.min():.2f}, {val_images.max():.2f}]")
    
    # ─── Show class distribution in one batch ───
    print("\n" + "=" * 60)
    print("Class distribution in one training batch:")
    print("=" * 60)
    for i in range(10):
        count = (labels == i).sum().item()
        print(f"  {CIFAR10_CLASSES[i]:>12s}: {count}")
    
    # ─── Verify de-normalization works ───
    denormed = denormalize_cifar10(images[:4])
    assert denormed.min() >= 0 and denormed.max() <= 1, \
        f"De-normalized range should be [0,1], got [{denormed.min():.2f}, {denormed.max():.2f}]"
    print(f"\nDe-normalization: pixel range [{denormed.min():.2f}, {denormed.max():.2f}] ")
    
    print("\n All dataset tests PASSED!")
    print(f"\nCIFAR-10 is ready for training!")