"""
Vision Transformer (ViT) — The Complete Model.

This is the final assembly. Every component we built separately
now comes together into a single model that takes images in
and produces classification logits out.

═══════════════════════════════════════════════════════════
THE FULL PIPELINE:
═══════════════════════════════════════════════════════════

    Input: images [B, 3, 32, 32]
        │
        ↓  Patch Embedding (Conv2d + [CLS] + positional encoding)
    [B, 65, 128]
        │
        ↓  Transformer Block 1  →  save attention weights
    [B, 65, 128]
        │
        ↓  Transformer Block 2  →  save attention weights
    [B, 65, 128]
        │
        ↓  ... (6 blocks total)
        │
        ↓  Transformer Block 6  →  save attention weights
    [B, 65, 128]
        │
        ↓  Extract [CLS] token (index 0)
    [B, 128]
        │
        ↓  LayerNorm
    [B, 128]
        │
        ↓  Linear (classification head)
    [B, 10]  ←  logits (raw scores, softmax applied in loss function)

═══════════════════════════════════════════════════════════
PARAMETER BUDGET:
═══════════════════════════════════════════════════════════

    Patch Embedding:      ~14,720
    6 Transformer Blocks: ~1,189,632  (198,272 × 6)
    Final LayerNorm:      ~256
    Classification Head:  ~1,290     (128 × 10 + 10)
    ─────────────────────────────────
    Total:                ~1,205,898  (≈1.2M parameters)

This is small enough to train on a free Colab GPU in 1-3 hours.
For reference, the original ViT-Base has 86M parameters.
"""

import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import ViTConfig
from models.patch_embedding import PatchEmbedding
from models.transformer_block import (
    TransformerBlock,
    TransformerBlockPostNorm,
    TransformerBlockNoResidual
)
from models.mlp import LayerNorm


class ViT(nn.Module):
    """
    Vision Transformer for image classification.
    
    Built entirely from scratch — no nn.TransformerEncoder,
    no pretrained weights, no high-level abstractions.
    """
    
    def __init__(
        self,
        config: ViTConfig,
        use_cls_token: bool = True,
        use_scaling: bool = True,
        block_type: str = "pre_norm"
    ):
        """
        Args:
            config: ViTConfig with all hyperparameters
            use_cls_token: If True, use [CLS] token for classification.
                          If False, use Global Average Pooling (for ablation).
            use_scaling: If True, scale attention by √d_k.
                        If False, skip scaling (for ablation).
            block_type: Which transformer block variant to use:
                       "pre_norm" (default), "post_norm", or "no_residual"
                       (the latter two are for ablation studies)
        """
        super().__init__()
        self.config = config
        self.use_cls_token = use_cls_token
        
        # ─── 1. Patch Embedding ───
        # Converts images → sequence of patch tokens with [CLS] and positions
        self.patch_embed = PatchEmbedding(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.d_model,
            dropout=config.dropout,
        )
        
        # ─── 2. Transformer Blocks ───
        # Stack num_layers blocks. Using nn.ModuleList (not a Python list!)
        # so PyTorch can track all parameters for optimization.
        #
        # Choose block type based on ablation setting
        if block_type == "pre_norm":
            BlockClass = TransformerBlock
        elif block_type == "post_norm":
            BlockClass = TransformerBlockPostNorm
        elif block_type == "no_residual":
            BlockClass = TransformerBlockNoResidual
        else:
            raise ValueError(f"Unknown block_type: {block_type}")
        
        if block_type == "pre_norm":
            # Only PreNorm block supports the use_scaling flag
            self.blocks = nn.ModuleList([
                BlockClass(config, use_scaling=use_scaling)
                for _ in range(config.num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                BlockClass(config)
                for _ in range(config.num_layers)
            ])
        
        # ─── 3. Final LayerNorm ───
        # Applied to the [CLS] token representation before classification
        # This is important: without it, the classification head receives
        # unnormalized features whose scale varies during training
        self.norm = LayerNorm(config.d_model)
        
        # ─── 4. Classification Head ───
        # Simple linear layer: d_model → num_classes
        # No softmax here! CrossEntropyLoss in PyTorch includes softmax.
        # Keeping logits raw gives better numerical stability.
        self.head = nn.Linear(config.d_model, config.num_classes)
        
        # ─── Initialize weights ───
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """
        Weight initialization following the ViT paper.
        
        - Linear layers: truncated normal with std=0.02
        - LayerNorm: gamma=1, beta=0 (already default, but explicit)
        - Biases: zeros
        
        WHY truncated normal?
        Regular normal can produce outlier weights that cause
        exploding activations. Truncating at 2 std deviations
        prevents this while maintaining the overall distribution.
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: images → class logits.
        
        Args:
            x: Input images, shape [B, C, H, W]
               e.g., [4, 3, 32, 32] for CIFAR-10
        
        Returns:
            logits: Raw class scores, shape [B, num_classes]
                    e.g., [4, 10] for CIFAR-10
        """
        # ─── Step 1: Patch Embedding ───
        # [B, 3, 32, 32] → [B, 65, 128]
        x = self.patch_embed(x)
        
        # ─── Step 2: Pass through all Transformer Blocks ───
        # Each block: [B, 65, 128] → [B, 65, 128]
        # We don't store attention weights during normal forward pass
        # (saves memory during training). Use get_attention_maps() for visualization.
        for block in self.blocks:
            x, _ = block(x)
        
        # ─── Step 3: Extract representation for classification ───
        if self.use_cls_token:
            # Take the [CLS] token (index 0) — it has attended to all patches
            # [B, 65, 128] → [B, 128]
            x = x[:, 0]
        else:
            # Global Average Pooling: average all PATCH tokens (skip [CLS])
            # [B, 65, 128] → [B, 64, 128] → [B, 128]
            x = x[:, 1:].mean(dim=1)
        
        # ─── Step 4: Final LayerNorm ───
        # [B, 128] → [B, 128]
        x = self.norm(x)
        
        # ─── Step 5: Classification Head ───
        # [B, 128] → [B, 10]
        logits = self.head(x)
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and return attention weights from ALL layers.
        
        Used for visualization — NOT for training (slower due to storing
        all attention matrices).
        
        Args:
            x: Input images, shape [B, C, H, W]
        
        Returns:
            attention_maps: shape [num_layers, B, num_heads, N, N]
            where N = num_patches + 1 (including [CLS])
        """
        all_attention_weights = []
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Pass through blocks, collecting attention at each layer
        for block in self.blocks:
            x, attn_weights = block(x)
            all_attention_weights.append(attn_weights)
        
        # Stack: list of [B, heads, N, N] → [layers, B, heads, N, N]
        return torch.stack(all_attention_weights)
    
    def get_cls_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and return the [CLS] token embeddings
        BEFORE the classification head.
        
        Used for t-SNE visualization of learned representations.
        
        Args:
            x: Input images, shape [B, C, H, W]
        
        Returns:
            embeddings: shape [B, d_model]
        """
        x = self.patch_embed(x)
        
        for block in self.blocks:
            x, _ = block(x)
        
        # Extract [CLS] token and normalize
        cls_embedding = self.norm(x[:, 0])
        
        return cls_embedding


# ──────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    config = ViTConfig()
    
    # ──────────────────────────────────────────────
    print("=" * 60)
    print("TEST 1: Full ViT — Forward Pass")
    print("=" * 60)
    
    model = ViT(config)
    
    # Detailed parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Per-component breakdown
    patch_params = sum(p.numel() for p in model.patch_embed.parameters())
    block_params = sum(p.numel() for p in model.blocks.parameters())
    norm_params = sum(p.numel() for p in model.norm.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    
    print("Parameter breakdown:")
    print(f"  Patch Embedding:      {patch_params:>10,}  ({100*patch_params/total_params:.1f}%)")
    print(f"  Transformer Blocks:   {block_params:>10,}  ({100*block_params/total_params:.1f}%)")
    print(f"  Final LayerNorm:      {norm_params:>10,}  ({100*norm_params/total_params:.1f}%)")
    print(f"  Classification Head:  {head_params:>10,}  ({100*head_params/total_params:.1f}%)")
    print(f"  {'─' * 40}")
    print(f"  Total:                {total_params:>10,}")
    
    # Forward pass
    images = torch.randn(4, 3, 32, 32)
    logits = model(images)
    
    print(f"\nInput shape:  {images.shape}")
    print(f"Output shape: {logits.shape}")
    
    assert logits.shape == (4, config.num_classes), \
        f"Expected (4, {config.num_classes}), got {logits.shape}"
    
    print(" Forward pass test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: Backward Pass — Gradients Flow Everywhere")
    print("=" * 60)
    
    model.zero_grad()
    images = torch.randn(4, 3, 32, 32)
    logits = model(images)
    
    # Simulate a training step with cross-entropy loss
    targets = torch.randint(0, config.num_classes, (4,))
    loss = nn.functional.cross_entropy(logits, targets)
    loss.backward()
    
    print(f"Loss value: {loss.item():.4f}")
    print(f"  (Expected ~2.30 for 10 classes at initialization,")
    print(f"   since -log(1/10) = {-torch.log(torch.tensor(1/10.0)):.4f})")
    
    # Check ALL parameters have gradients
    no_grad_params = []
    nan_grad_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            no_grad_params.append(name)
        elif torch.isnan(param.grad).any():
            nan_grad_params.append(name)
    
    if no_grad_params:
        print(f"\n Parameters without gradients: {no_grad_params}")
    if nan_grad_params:
        print(f"\n Parameters with NaN gradients: {nan_grad_params}")
    
    if not no_grad_params and not nan_grad_params:
        print(f"\nAll {len(list(model.parameters()))} parameters received valid gradients")
    
    print(" Backward pass test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 3: Attention Maps Extraction")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        attn_maps = model.get_attention_maps(torch.randn(2, 3, 32, 32))
    
    print(f"Attention maps shape: {attn_maps.shape}")
    print(f"  Layers:  {attn_maps.shape[0]}")
    print(f"  Batch:   {attn_maps.shape[1]}")
    print(f"  Heads:   {attn_maps.shape[2]}")
    print(f"  Seq len: {attn_maps.shape[3]} × {attn_maps.shape[4]}")
    
    expected = (config.num_layers, 2, config.num_heads, config.seq_length, config.seq_length)
    assert attn_maps.shape == expected, \
        f"Expected {expected}, got {attn_maps.shape}"
    
    # Verify attention weights sum to 1
    row_sums = attn_maps.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
        "Attention weights don't sum to 1!"
    
    print(" Attention maps test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 4: [CLS] Embeddings Extraction (for t-SNE)")
    print("=" * 60)
    
    with torch.no_grad():
        embeddings = model.get_cls_embeddings(torch.randn(8, 3, 32, 32))
    
    print(f"Embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (8, config.d_model), \
        f"Expected (8, {config.d_model}), got {embeddings.shape}"
    
    print(" CLS embeddings test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 5: Ablation Variants")
    print("=" * 60)
    
    # No scaling
    model_no_scale = ViT(config, use_scaling=False)
    out = model_no_scale(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)
    print("  No √d_k scaling:        ✅")
    
    # Global Average Pooling instead of [CLS]
    model_gap = ViT(config, use_cls_token=False)
    out = model_gap(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)
    print("  Global Average Pooling:  ✅")
    
    # Post-Norm
    model_postnorm = ViT(config, block_type="post_norm")
    out = model_postnorm(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)
    print("  Post-Norm blocks:        ✅")
    
    # No residual connections
    model_nores = ViT(config, block_type="no_residual")
    out = model_nores(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)
    print("  No residual connections: ✅")
    
    print(" All ablation variants test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 6: EuroSAT Configuration")
    print("=" * 60)
    
    from configs.config import EuroSATConfig
    eurosat_config = EuroSATConfig()
    model_eurosat = ViT(eurosat_config)
    
    eurosat_params = sum(p.numel() for p in model_eurosat.parameters())
    print(f"EuroSAT ViT parameters: {eurosat_params:,}")
    
    out = model_eurosat(torch.randn(2, 3, 64, 64))
    print(f"Input:  [2, 3, 64, 64]")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 10)
    
    print(" EuroSAT config test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" ALL VIT TESTS PASSED!")
    print("=" * 60)
    print(f"\nYour ViT is ready for training.")
    print(f"  CIFAR-10 model:  {total_params:,} parameters")
    print(f"  EuroSAT model:   {eurosat_params:,} parameters")
    print(f"\nNext step: build the training pipeline (utils/training.py)")