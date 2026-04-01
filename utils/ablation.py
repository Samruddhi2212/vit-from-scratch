"""
Shared ablation experiment runner (mirrors the training+ablation_studies notebook).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from configs.config import ViTConfig
from models.vit import ViT
from utils.training import train, evaluate, load_checkpoint


def run_ablation(
    experiment_name: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    ablation_dir: str,
    ablation_epochs: int,
    config: ViTConfig | None = None,
    use_cls_token: bool = True,
    use_scaling: bool = True,
    block_type: str = "pre_norm",
    description: str = "",
    model_factory: Callable[[ViTConfig], nn.Module] | None = None,
) -> dict[str, Any]:
    """
    Train one ablation, evaluate on test set, return a result dict.

    If ``model_factory`` is None, builds ``ViT(cfg, ...)`` for each fresh model.
    If provided, ``model_factory(cfg)`` must return a new module (e.g. ViTNoPosition(cfg)).
    ``cfg`` is a deep copy with ``total_epochs`` / ``warmup_epochs`` already set for this run.
    """
    if config is None:
        config = ViTConfig()
    cfg = copy.deepcopy(config)
    cfg.total_epochs = ablation_epochs
    cfg.warmup_epochs = min(5, max(1, ablation_epochs // 10))

    print("\n" + "=" * 70)
    print(f"ABLATION: {experiment_name}")
    print(f"  {description}")
    print("=" * 70)

    def _default_factory(c: ViTConfig) -> nn.Module:
        return ViT(
            c,
            use_cls_token=use_cls_token,
            use_scaling=use_scaling,
            block_type=block_type,
        )

    factory = model_factory or _default_factory
    model = factory(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device,
        save_dir=ablation_dir,
        experiment_name=experiment_name,
    )

    best_model = factory(cfg).to(device)
    ckpt = f"{ablation_dir}/{experiment_name}_best.pt"
    load_checkpoint(best_model, ckpt, device=device)
    test_loss, test_acc = evaluate(best_model, test_loader, device, desc="Test")

    result = {
        "name": experiment_name,
        "description": description,
        "history": history,
        "best_val_acc": max(history["val_acc"]),
        "test_acc": test_acc,
        "test_loss": test_loss,
        "params": total_params,
    }
    print(f"\n  Best val acc:  {result['best_val_acc']:.2f}%")
    print(f"  Test acc:      {result['test_acc']:.2f}%")
    return result
