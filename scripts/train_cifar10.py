#!/usr/bin/env python3
"""
Train ViT on CIFAR-10 from the command line (no Jupyter).

Example (from repository root):
  python scripts/train_cifar10.py --epochs 200 --num-workers 4

tmux / Slurm: run the same command inside an activated conda env on a GPU node.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from configs.config import ViTConfig
from models.vit import ViT
from scripts._paths import default_output_dir
from utils.dataset import get_cifar10_loaders, CIFAR10_CLASSES
from utils.evaluation import full_evaluation
from utils.training import train, load_checkpoint
from utils.visualization import plot_training_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ViT on CIFAR-10")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root for checkpoints/ and results/ (default: outputs/train or $VIT_OUTPUT_DIR/train)",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override config.total_epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override config.batch_size")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--experiment-name", type=str, default="vit_cifar10")
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib figures")
    p.add_argument("--seed", type=int, default=42, help="Torch RNG seed (DataLoader shuffle)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    root = _REPO
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = default_output_dir(root, "train")
    ckpt_dir = out / "checkpoints"
    res_dir = out / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    config = ViTConfig()
    if args.epochs is not None:
        config.total_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        config, num_workers=args.num_workers
    )

    model = ViT(config)
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir=str(ckpt_dir),
        experiment_name=args.experiment_name,
    )

    best_path = ckpt_dir / f"{args.experiment_name}_best.pt"
    best_model = ViT(config).to(device)
    load_checkpoint(best_model, str(best_path), device=device)

    results = full_evaluation(
        best_model,
        test_loader,
        device,
        class_names=list(CIFAR10_CLASSES),
        save_dir=str(res_dir),
        experiment_name=args.experiment_name,
    )

    hist_path = ckpt_dir / f"{args.experiment_name}_history.pt"
    torch.save(history, hist_path)
    print(f"Saved history to {hist_path}")

    if not args.no_plots:
        plot_training_curves(
            history,
            save_path=str(res_dir / "training_curves.png"),
        )
        print(f"Saved plot to {res_dir / 'training_curves.png'}")

    print("Done. Top-1 test accuracy: {:.2f}%".format(results["top1_accuracy"]))


if __name__ == "__main__":
    main()
