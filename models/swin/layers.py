"""
Swin Transformer building blocks (from scratch).

Layout: tensors inside blocks are (B, H, W, C). Algorithm follows
Liu et al., "Swin Transformer" (ICCV 2021); shifted-window attention
matches the standard padding + roll + mask formulation used in common
PyTorch reference implementations.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _patch_merging_pad(x: Tensor) -> Tensor:
    """Pad H,W to even, then merge 2x2 spatial patches (NHWC)."""
    H, W, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    x0 = x[..., 0::2, 0::2, :]
    x1 = x[..., 1::2, 0::2, :]
    x2 = x[..., 0::2, 1::2, :]
    x3 = x[..., 1::2, 1::2, :]
    return torch.cat([x0, x1, x2, x3], -1)


class PatchMerging(nn.Module):
    """Downsample spatially by 2×, double channels (NHWC layout)."""

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor) -> Tensor:
        x = _patch_merging_pad(x)
        x = self.norm(x)
        return self.reduction(x)


def _get_relative_position_bias(
    relative_position_bias_table: Tensor,
    relative_position_index: Tensor,
    window_size: tuple[int, int],
) -> Tensor:
    Wh, Ww = window_size
    N = Wh * Ww
    relative_position_bias = relative_position_bias_table[relative_position_index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    return relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)


def shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: tuple[int, int],
    num_heads: int,
    shift_size: tuple[int, int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Tensor | None = None,
    proj_bias: Tensor | None = None,
    training: bool = True,
) -> Tensor:
    """Window MSA with optional shifted window; pads to multiples of window_size."""
    B, H, W, C = input.shape
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = list(shift_size)
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    Wh, Ww = window_size
    num_windows = (pad_H // Wh) * (pad_W // Ww)
    x = x.view(B, pad_H // Wh, Wh, pad_W // Ww, Ww, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, Wh * Ww, C)

    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (C // num_heads) ** -0.5
    attn = q.matmul(k.transpose(-2, -1))
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -Wh), (-Wh, -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -Ww), (-Ww, -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // Wh, Wh, pad_W // Ww, Ww)
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, Wh * Ww)
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    x = x.view(B, pad_H // Wh, pad_W // Ww, Wh, Ww, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    return x[:, :H, :W, :].contiguous()


class ShiftedWindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        shift_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        Wh, Ww = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = relative_coords.sum(-1).flatten()
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table,
            self.relative_position_index,
            self.window_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        rel_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            rel_bias,
            self.window_size,
            self.num_heads,
            self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
        )


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class _DropPath(nn.Module):
    """Stochastic depth (per-row); at prob=0 behaves like identity."""

    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int],
        shift_size: tuple[int, int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ShiftedWindowAttention(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.drop_path1 = _DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        self.mlp = _MLP(dim, mlp_ratio, dropout)
        self.drop_path2 = _DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
