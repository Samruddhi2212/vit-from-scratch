"""
Smoke tests: imports and tiny forward passes (no dataset files required).
Run from repository root: pytest
"""

from __future__ import annotations

import torch


def test_vit_cifar_forward():
    from configs.config import ViTConfig
    from models.vit import ViT

    cfg = ViTConfig()
    m = ViT(cfg)
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    assert y.shape == (2, 10)


def test_siamese_vit_cd_forward():
    from models.siamese_vit import build_siamese_vit_cd

    m = build_siamese_vit_cd()
    m.eval()
    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        logits = m(x1, x2)
    assert logits.shape == (1, 1, 256, 256)


def test_ablation_variants_import():
    from models.ablation_variants import ViTNoPosition
    from configs.config import ViTConfig

    m = ViTNoPosition(ViTConfig())
    x = torch.randn(1, 3, 32, 32)
    y = m(x)
    assert y.shape == (1, 10)


def test_cifar10_standalone_import():
    from utils.cifar10_standalone import ensure_cifar10_downloaded

    assert callable(ensure_cifar10_downloaded)


def test_metrics_import():
    from utils.metrics import ChangeDetectionMetrics

    assert ChangeDetectionMetrics is not None
