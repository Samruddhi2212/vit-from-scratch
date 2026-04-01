"""
Model variants used only in ablation studies (see scripts/run_ablations.py).
"""

import torch
from configs.config import ViTConfig
from models.vit import ViT


class ViTNoPosition(ViT):
    """ViT with positional embeddings zeroed and frozen (ablation)."""

    def __init__(self, config: ViTConfig):
        super().__init__(config)
        with torch.no_grad():
            self.patch_embed.pos_embed.zero_()
        self.patch_embed.pos_embed.requires_grad = False
