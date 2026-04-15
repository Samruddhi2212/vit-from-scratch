"""
Siamese Swin Change Detection — shared hierarchical Swin backbone, multi-scale
difference module, and progressive decoder (same head contract as Siamese ViT).

Per-stage features (different H,W,C) are resized to 16×16 and projected to
``embed_dim`` tokens so MultiScaleDiffModule sees four (B, 256, embed_dim) lists.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_difference import MultiScaleDiffModule
from models.decoder import ProgressiveDecoder
from models.swin.backbone import SwinBackbone


class _ScaleAdapter(nn.Module):
    """Bilinear resize to grid_side×grid_side, then Linear(C → embed_dim)."""

    def __init__(self, in_channels: int, embed_dim: int, grid_side: int) -> None:
        super().__init__()
        self.grid_side = grid_side
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = F.interpolate(
            x,
            size=(self.grid_side, self.grid_side),
            mode="bilinear",
            align_corners=False,
        )
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return self.proj(x)


class SiameseSwinChangeDetection(nn.Module):
    """
    Args:
        img_size:       Input height/width (square).
        patch_size:     ViT-style semantic patch size for decoder grid only
                        (``num_patches_side = img_size // patch_size``); use 16 with Swin.
        fusion_embed_dim: Token dim for MultiScaleDiffModule (match ViT: 768).
        swin_* :        Backbone hyperparameters (Swin-Tiny defaults).
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        fusion_embed_dim: int = 768,
        swin_patch_size: int = 4,
        swin_embed_dim: int = 96,
        swin_depths: tuple[int, ...] = (2, 2, 6, 2),
        swin_num_heads: tuple[int, ...] = (3, 6, 12, 24),
        swin_window_size: tuple[int, int] = (7, 7),
        swin_mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        diff_out_dim: int = 256,
        decoder_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if decoder_dims is None:
            decoder_dims = [128, 64, 32, 16]

        self.img_size = img_size
        self.patch_size = patch_size
        self.fusion_embed_dim = fusion_embed_dim

        self.encoder = SwinBackbone(
            in_channels=in_channels,
            swin_patch_size=swin_patch_size,
            embed_dim=swin_embed_dim,
            depths=swin_depths,
            num_heads=swin_num_heads,
            window_size=swin_window_size,
            mlp_ratio=swin_mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path_rate=drop_path_rate,
        )

        stage_channels = [swin_embed_dim * (2**i) for i in range(4)]
        grid_side = img_size // patch_size
        self.scale_adapters = nn.ModuleList(
            _ScaleAdapter(c, fusion_embed_dim, grid_side) for c in stage_channels
        )

        self.diff_module = MultiScaleDiffModule(
            in_dim=fusion_embed_dim,
            out_dim=diff_out_dim,
            n_scales=4,
        )

        self.decoder = ProgressiveDecoder(
            in_dim=diff_out_dim,
            hidden_dims=decoder_dims,
            num_patches_side=grid_side,
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        feats1 = self.encoder(img1)
        feats2 = self.encoder(img2)
        tok1 = [self.scale_adapters[i](feats1[i]) for i in range(4)]
        tok2 = [self.scale_adapters[i](feats2[i]) for i in range(4)]
        diff_feat = self.diff_module(tok1, tok2)
        return self.decoder(diff_feat)

    @torch.no_grad()
    def predict(
        self, img1: torch.Tensor, img2: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        probs = torch.sigmoid(self.forward(img1, img2))
        return (probs > threshold).float()

    def get_param_count(self) -> dict[str, int]:
        enc_p = sum(p.numel() for p in self.encoder.parameters())
        ad_p = sum(p.numel() for p in self.scale_adapters.parameters())
        diff_p = sum(p.numel() for p in self.diff_module.parameters())
        dec_p = sum(p.numel() for p in self.decoder.parameters())
        return {
            "encoder": enc_p,
            "adapters": ad_p,
            "diff_module": diff_p,
            "decoder": dec_p,
            "total": enc_p + ad_p + diff_p + dec_p,
        }


def build_siamese_swin_cd(config: dict | None = None) -> SiameseSwinChangeDetection:
    """Build from training config dict (YAML + CLI)."""
    default = dict(
        img_size=256,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        swin_patch_size=4,
        swin_embed_dim=96,
        swin_depths=(2, 2, 6, 2),
        swin_num_heads=(3, 6, 12, 24),
        swin_window_size=(7, 7),
        swin_mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.0,
        drop_path_rate=0.0,
        diff_out_dim=256,
        decoder_dims=[128, 64, 32, 16],
    )
    if config:
        default.update(config)

    depths = default["swin_depths"]
    if isinstance(depths, list):
        depths = tuple(depths)
    heads = default["swin_num_heads"]
    if isinstance(heads, list):
        heads = tuple(heads)
    ws = default["swin_window_size"]
    if isinstance(ws, list):
        ws = tuple(ws)

    return SiameseSwinChangeDetection(
        img_size=default["img_size"],
        patch_size=default["patch_size"],
        in_channels=default["in_channels"],
        fusion_embed_dim=default["embed_dim"],
        swin_patch_size=default["swin_patch_size"],
        swin_embed_dim=default["swin_embed_dim"],
        swin_depths=depths,
        swin_num_heads=heads,
        swin_window_size=ws,
        swin_mlp_ratio=default["swin_mlp_ratio"],
        dropout=default["dropout"],
        attn_dropout=default["attn_dropout"],
        drop_path_rate=default["drop_path_rate"],
        diff_out_dim=default["diff_out_dim"],
        decoder_dims=list(default["decoder_dims"]),
    )


if __name__ == "__main__":
    from pathlib import Path
    import sys

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_siamese_swin_cd().to(device)
    counts = m.get_param_count()
    print("Params:", counts)
    x1 = torch.randn(2, 3, 256, 256, device=device)
    x2 = torch.randn(2, 3, 256, 256, device=device)
    logits = m(x1, x2)
    assert logits.shape == (2, 1, 256, 256)
    logits.sum().backward()
    print("SiameseSwinChangeDetection OK")
