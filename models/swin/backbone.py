"""
Hierarchical Swin backbone: patch embed + 4 stages, returns multi-scale maps (BCHW).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from models.swin.layers import PatchMerging, SwinTransformerBlock


class SwinBackbone(nn.Module):
    """
    Swin-Tiny–style hierarchy by default: depths (2,2,6,2), dims 96→192→384→768.

    Returns four feature maps (one per stage), each (B, C_k, H_k, W_k), shallow→deep.
    Input spatial size should be divisible by ``swin_patch_size`` (e.g. 256 / 4 = 64).
    """

    def __init__(
        self,
        in_channels: int = 3,
        swin_patch_size: int = 4,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        window_size: tuple[int, int] = (7, 7),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        self.swin_patch_size = swin_patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size

        # Patch embedding: B,3,H,W -> B,H',W',embed_dim
        self.patch_proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=swin_patch_size,
            stride=swin_patch_size,
        )
        self.norm0 = norm_layer(embed_dim)

        total_blocks = sum(depths)
        block_id = 0
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i_stage in range(len(depths)):
            dim = embed_dim * (2**i_stage)
            stage_blocks: list[nn.Module] = []
            depth = depths[i_stage]
            for i_layer in range(depth):
                if total_blocks > 1:
                    sd = drop_path_rate * float(block_id) / float(total_blocks - 1)
                else:
                    sd = 0.0
                shift = (0, 0) if i_layer % 2 == 0 else (window_size[0] // 2, window_size[1] // 2)
                stage_blocks.append(
                    SwinTransformerBlock(
                        dim=dim,
                        num_heads=num_heads[i_stage],
                        window_size=window_size,
                        shift_size=shift,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attn_dropout,
                        drop_path=sd,
                        norm_layer=norm_layer,
                    )
                )
                block_id += 1
            self.stages.append(nn.Sequential(*stage_blocks))

            if i_stage < len(depths) - 1:
                self.downsamples.append(PatchMerging(dim, norm_layer=norm_layer))

    def forward(self, x: Tensor) -> list[Tensor]:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            List of 4 tensors (B, C_k, H_k, W_k) for each stage.
        """
        x = self.patch_proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm0(x)

        outs: list[Tensor] = []
        for i_stage in range(len(self.depths)):
            x = self.stages[i_stage](x)
            # B,H,W,C -> B,C,H,W
            outs.append(x.permute(0, 3, 1, 2).contiguous())
            if i_stage < len(self.depths) - 1:
                x = self.downsamples[i_stage](x)

        return outs


def _test_backbone() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = SwinBackbone().to(device)
    x = torch.randn(2, 3, 256, 256, device=device)
    out = m(x)
    assert len(out) == 4
    # 256/4=64, then /2 each stage: 64,32,16,8
    expected_hw = [(64, 64), (32, 32), (16, 16), (8, 8)]
    expected_c = [96, 192, 384, 768]
    for i, (feat, (eh, ew), ec) in enumerate(zip(out, expected_hw, expected_c)):
        assert feat.shape == (2, ec, eh, ew), f"stage {i}: got {feat.shape}"
    y = sum(f.sum() for f in out)
    y.backward()
    print("SwinBackbone shape + backward OK")


if __name__ == "__main__":
    _test_backbone()
