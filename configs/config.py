from dataclasses import dataclass


@dataclass
class ViTConfig:
    """
    Configuration for ViT trained on CIFAR-10.
    
    The @dataclass decorator auto-generates __init__, __repr__, etc.
    so you can do: config = ViTConfig() and immediately access config.d_model
    """
    
    # ──────────────────────────────────────────────
    # IMAGE SETTINGS
    # ──────────────────────────────────────────────
    image_size: int = 32        # CIFAR-10 images are 32x32 pixels
    patch_size: int = 4         # Each patch is 4x4 pixels
    in_channels: int = 3        # RGB = 3 channels
    
    # WHY patch_size=4?
    # 32/4 = 8, so we get an 8x8 grid = 64 patches
    # Each patch is 4×4×3 = 48 numbers, which gets projected to d_model
    # Smaller patches (2) = more patches (256) = better detail but MUCH slower (quadratic attention)
    # Larger patches (8) = fewer patches (16) = faster but too coarse
    # 4 is the sweet spot for 32x32 images
    
    # ──────────────────────────────────────────────
    # MODEL ARCHITECTURE
    # ──────────────────────────────────────────────
    d_model: int = 128          # Embedding dimension (size of each token vector)
    num_heads: int = 4          # Number of attention heads
    num_layers: int = 6         # Number of transformer blocks stacked
    ffn_hidden: int = 512       # Hidden dimension in feed-forward network (4 × d_model)
    dropout: float = 0.1        # Dropout rate for regularization
    
    # WHY these values?
    # d_model=128: Small enough to train on Colab, large enough to learn
    # num_heads=4: d_model/num_heads = 32 per head, which is reasonable
    # num_layers=6: Deep enough for complex features, not so deep it won't train
    # ffn_hidden=512: Standard practice is 4× d_model (the paper uses this ratio)
    # dropout=0.1: Standard regularization, prevents overfitting on small datasets
    
    # ──────────────────────────────────────────────
    # CLASSIFICATION
    # ──────────────────────────────────────────────
    num_classes: int = 10       # CIFAR-10 has 10 classes
    
    # ──────────────────────────────────────────────
    # TRAINING HYPERPARAMETERS
    # ──────────────────────────────────────────────
    batch_size: int = 256       # Number of images per training step
    learning_rate: float = 1e-3 # Peak learning rate (after warmup)
    min_lr: float = 1e-5        # Minimum learning rate (at end of cosine decay)
    weight_decay: float = 0.05  # L2 regularization strength
    warmup_epochs: int = 10     # Epochs of linear LR warmup
    total_epochs: int = 200     # Total training epochs
    
    # WHY warmup?
    # At initialization, attention scores are random → softmax gives ~uniform weights
    # → gradients are small and noisy. If we use a large learning rate immediately,
    # the model gets pushed to a bad region of parameter space.
    # Warmup lets the model first learn meaningful attention patterns with small steps,
    # then we increase the learning rate for faster convergence.
    
    # ──────────────────────────────────────────────
    # DERIVED VALUES (computed from the above)
    # ──────────────────────────────────────────────
    @property
    def num_patches(self) -> int:
        """Total number of patches the image is split into."""
        return (self.image_size // self.patch_size) ** 2
        # For CIFAR-10: (32 // 4)² = 8² = 64 patches
    
    @property
    def d_k(self) -> int:
        """Dimension per attention head."""
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        return self.d_model // self.num_heads
        # For default: 128 // 4 = 32 per head
    
    @property
    def patch_dim(self) -> int:
        """Flattened dimension of a single patch (before projection)."""
        return self.patch_size * self.patch_size * self.in_channels
        # For default: 4 × 4 × 3 = 48
    
    @property
    def seq_length(self) -> int:
        """Sequence length including [CLS] token."""
        return self.num_patches + 1
        # For default: 64 + 1 = 65


@dataclass
class EuroSATConfig(ViTConfig):
    """
    Configuration for ViT trained on EuroSAT satellite images.
    Inherits from ViTConfig and overrides what's different.
    
    EuroSAT images are 64x64 (larger than CIFAR-10), so we use:
    - Larger patch size (8 instead of 4) to keep num_patches = 64
    - Larger model to handle the richer data
    """
    image_size: int = 64
    patch_size: int = 8         # 64/8 = 8×8 grid = 64 patches (same count as CIFAR-10)
    d_model: int = 192          # Bigger model for richer satellite data
    num_heads: int = 6          # 192/6 = 32 per head
    num_layers: int = 8         # Deeper network
    ffn_hidden: int = 768       # 4 × 192
    learning_rate: float = 5e-4 # Slightly lower LR for the bigger model
    total_epochs: int = 150
    batch_size: int = 128       # Smaller batch (bigger model uses more memory)


# ──────────────────────────────────────────────────────
# Quick sanity check — run this file directly to verify
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    config = ViTConfig()
    print("=" * 50)
    print("ViT Configuration (CIFAR-10)")
    print("=" * 50)
    print(f"Image size:       {config.image_size}×{config.image_size}")
    print(f"Patch size:       {config.patch_size}×{config.patch_size}")
    print(f"Number of patches: {config.num_patches}")
    print(f"Patch dimension:   {config.patch_dim}")
    print(f"Sequence length:   {config.seq_length} (patches + [CLS])")
    print(f"Model dimension:   {config.d_model}")
    print(f"Attention heads:   {config.num_heads}")
    print(f"Dimension per head: {config.d_k}")
    print(f"Transformer layers: {config.num_layers}")
    print(f"FFN hidden dim:    {config.ffn_hidden}")
    print(f"Total epochs:      {config.total_epochs}")
    print()
    
    eurosat = EuroSATConfig()
    print("=" * 50)
    print("EuroSAT Configuration")
    print("=" * 50)
    print(f"Image size:       {eurosat.image_size}×{eurosat.image_size}")
    print(f"Patch size:       {eurosat.patch_size}×{eurosat.patch_size}")
    print(f"Number of patches: {eurosat.num_patches}")
    print(f"Sequence length:   {eurosat.seq_length}")
    print(f"Model dimension:   {eurosat.d_model}")
    print(f"Attention heads:   {eurosat.num_heads}")
    print(f"Dimension per head: {eurosat.d_k}")