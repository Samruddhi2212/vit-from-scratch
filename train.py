"""
Training script for Siamese ViT Change Detection on OSCD.

Usage
-----
    python train.py                                   # uses configs/train_config.yaml
    python train.py --data_dir ./processed_oscd \\
                    --epochs 200 --batch_size 8 --lr 1e-3

Any CLI flag overrides the corresponding YAML key.  Unknown keys are ignored
so that the YAML can hold extra documentation fields.

Outputs (written to --output_dir)
----------------------------------
    best_model.pth      — checkpoint with best val-F1
    last_model.pth      — checkpoint after every epoch
    events.out.tfevents — TensorBoard log
    train.log           — plain-text log (same lines as stdout)
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.siamese_vit import build_siamese_vit_cd
from utils.losses import BCEDiceLoss, FocalDiceLoss
from utils.metrics import ChangeDetectionMetrics
from utils.oscd_dataset import get_oscd_dataloaders


# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = "configs/train_config.yaml"


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Siamese ViT Change Detection on OSCD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",        default=_DEFAULT_CONFIG, help="YAML config path")

    # ── data ──────────────────────────────────────────────────────────────
    p.add_argument("--data_dir",      type=str)
    p.add_argument("--num_workers",   type=int)

    # ── model ─────────────────────────────────────────────────────────────
    p.add_argument("--img_size",      type=int)
    p.add_argument("--patch_size",    type=int)
    p.add_argument("--in_channels",   type=int)
    p.add_argument("--embed_dim",     type=int)
    p.add_argument("--depth",         type=int)
    p.add_argument("--num_heads",     type=int)
    p.add_argument("--mlp_ratio",     type=float)
    p.add_argument("--dropout",       type=float)
    p.add_argument("--attn_dropout",  type=float)
    p.add_argument("--diff_type",     type=str,
                   choices=["subtract", "concat_project", "attention"])
    p.add_argument("--diff_out_dim",  type=int)

    # ── loss ──────────────────────────────────────────────────────────────
    p.add_argument("--loss",          type=str, choices=["bce_dice", "focal_dice"])
    p.add_argument("--bce_weight",    type=float)
    p.add_argument("--dice_weight",   type=float)
    p.add_argument("--pos_weight",    type=float)
    p.add_argument("--focal_alpha",   type=float)
    p.add_argument("--focal_gamma",   type=float)

    # ── optimiser ─────────────────────────────────────────────────────────
    p.add_argument("--lr",            type=float)
    p.add_argument("--weight_decay",  type=float)
    p.add_argument("--grad_clip",     type=float)

    # ── schedule ──────────────────────────────────────────────────────────
    p.add_argument("--epochs",        type=int)
    p.add_argument("--warmup_epochs", type=int)
    p.add_argument("--min_lr",        type=float)

    # ── batch sizes ───────────────────────────────────────────────────────
    p.add_argument("--batch_size",      type=int)
    p.add_argument("--eval_batch_size", type=int)

    # ── misc ──────────────────────────────────────────────────────────────
    p.add_argument("--log_every",     type=int)
    p.add_argument("--output_dir",    type=str)
    p.add_argument("--resume",        type=str, default=None)
    p.add_argument("--patience",      type=int)

    return p.parse_args(argv)


def _build_cfg(args: argparse.Namespace) -> dict:
    """Merge YAML defaults with CLI overrides.  CLI wins on conflict."""
    cfg = _load_yaml(args.config) if Path(args.config).exists() else {}

    # overlay only the CLI args that were explicitly provided (not None)
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    cfg.update(overrides)

    # decoder_dims may come from YAML as a list; keep it
    cfg.setdefault("decoder_dims", [128, 64, 32, 16])

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def _setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(output_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warmup + cosine decay
# ──────────────────────────────────────────────────────────────────────────────

def _cosine_schedule_with_warmup(
    optimizer: AdamW,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float,           # min_lr / base_lr
) -> LambdaLR:
    """Return a LambdaLR that implements warmup + cosine decay."""

    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    best_f1: float,
    cfg: dict,
) -> None:
    torch.save(
        {
            "epoch":      epoch,
            "best_f1":    best_f1,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "cfg":        cfg,
        },
        path,
    )


def _load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[int, float]:
    """Load checkpoint; returns (start_epoch, best_f1)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    epoch   = ckpt["epoch"] + 1
    best_f1 = ckpt.get("best_f1", 0.0)
    logger.info(f"Resumed from {path}  (epoch {ckpt['epoch']}, best F1={best_f1:.4f})")
    return epoch, best_f1


# ──────────────────────────────────────────────────────────────────────────────
# One training epoch
# ──────────────────────────────────────────────────────────────────────────────

def _train_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    criterion:  nn.Module,
    optimizer:  AdamW,
    scaler:     torch.amp.GradScaler,
    device:     torch.device,
    grad_clip:  float,
    log_every:  int,
    epoch:      int,
    writer:     SummaryWriter,
    logger:     logging.Logger,
    global_step: list[int],      # mutable int wrapper
) -> float:
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(loader):
        img1   = batch["image1"].to(device, non_blocking=True)
        img2   = batch["image2"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type):
            logits = model(img1, img2)
            loss   = criterion(logits, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        total_loss += loss_val
        global_step[0] += 1

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:03d}  step {step+1:4d}/{len(loader)}  "
                f"loss={loss_val:.4f}  lr={lr_now:.2e}  "
                f"[{elapsed:.0f}s]"
            )
            writer.add_scalar("train/loss_step", loss_val,     global_step[0])
            writer.add_scalar("train/lr",         lr_now,       global_step[0])

    return total_loss / len(loader)


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _validate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    metrics   = ChangeDetectionMetrics(threshold=0.5)
    total_loss = 0.0

    for batch in loader:
        img1   = batch["image1"].to(device, non_blocking=True)
        img2   = batch["image2"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type):
            logits = model(img1, img2)
            loss   = criterion(logits, target)

        total_loss += loss.item()
        metrics.update(logits, target)

    return total_loss / len(loader), metrics.compute()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = _parse_args(argv)
    cfg  = _build_cfg(args)

    output_dir = Path(cfg["output_dir"])
    logger     = _setup_logging(output_dir)
    writer     = SummaryWriter(log_dir=str(output_dir))

    # ── device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # ── dataloaders ───────────────────────────────────────────────────────
    loaders = get_oscd_dataloaders(
        data_root        = cfg["data_dir"],
        patch_size       = cfg.get("img_size", 256),
        train_batch_size = cfg["batch_size"],
        eval_batch_size  = cfg["eval_batch_size"],
        num_workers      = cfg["num_workers"],
        pin_memory       = device.type == "cuda",
    )

    # ── model ─────────────────────────────────────────────────────────────
    model_cfg = {k: cfg[k] for k in (
        "img_size", "patch_size", "in_channels",
        "embed_dim", "depth", "num_heads", "mlp_ratio",
        "dropout", "attn_dropout",
        "diff_type", "diff_out_dim", "decoder_dims",
    )}
    model = build_siamese_vit_cd(model_cfg).to(device)

    counts = model.get_param_count()
    logger.info(
        f"Model params — encoder={counts['encoder']:,}  "
        f"diff={counts['diff_module']:,}  "
        f"decoder={counts['decoder']:,}  "
        f"total={counts['total']:,}"
    )

    # ── loss ──────────────────────────────────────────────────────────────
    pw_val = cfg.get("pos_weight")
    pos_weight = (
        torch.tensor([float(pw_val)], device=device) if pw_val is not None else None
    )

    if cfg["loss"] == "bce_dice":
        criterion = BCEDiceLoss(
            bce_weight  = cfg["bce_weight"],
            dice_weight = cfg["dice_weight"],
            pos_weight  = pos_weight,
        )
    else:
        criterion = FocalDiceLoss(
            alpha        = cfg["focal_alpha"],
            gamma        = cfg["focal_gamma"],
            dice_weight  = cfg["dice_weight"],
            focal_weight = cfg["bce_weight"],   # reuse bce_weight slot as focal_weight
        )
    criterion = criterion.to(device)

    # ── optimiser ─────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )

    scheduler = _cosine_schedule_with_warmup(
        optimizer      = optimizer,
        warmup_epochs  = cfg["warmup_epochs"],
        total_epochs   = cfg["epochs"],
        min_lr_ratio   = cfg["min_lr"] / cfg["lr"],
    )

    scaler = torch.amp.GradScaler(device=device.type)

    # ── resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_f1     = 0.0
    no_improve  = 0

    resume_path = cfg.get("resume")
    if resume_path and Path(resume_path).exists():
        start_epoch, best_f1 = _load_checkpoint(
            resume_path, model, optimizer, scheduler, device, logger
        )

    global_step = [start_epoch * len(loaders["train"])]

    # ── training loop ─────────────────────────────────────────────────────
    logger.info(
        f"Starting training — epochs={cfg['epochs']}  "
        f"bs={cfg['batch_size']}  lr={cfg['lr']:.2e}  "
        f"loss={cfg['loss']}"
    )

    for epoch in range(start_epoch, cfg["epochs"]):
        # ── train ──────────────────────────────────────────────────────
        train_loss = _train_epoch(
            model       = model,
            loader      = loaders["train"],
            criterion   = criterion,
            optimizer   = optimizer,
            scaler      = scaler,
            device      = device,
            grad_clip   = cfg["grad_clip"],
            log_every   = cfg["log_every"],
            epoch       = epoch,
            writer      = writer,
            logger      = logger,
            global_step = global_step,
        )

        scheduler.step()

        # ── validate ───────────────────────────────────────────────────
        lr_now = optimizer.param_groups[0]["lr"]
        if len(loaders["val"].dataset) == 0:
            logger.warning("Val split is empty — skipping validation this epoch.")
            val_loss = float("nan")
            val_m    = {k: float("nan") for k in ("f1","iou","precision","recall","kappa","accuracy")}
        else:
            val_loss, val_m = _validate(
                model     = model,
                loader    = loaders["val"],
                criterion = criterion,
                device    = device,
            )

        logger.info(
            f"Epoch {epoch:03d}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"F1={val_m['f1']:.4f}  IoU={val_m['iou']:.4f}  "
            f"P={val_m['precision']:.4f}  R={val_m['recall']:.4f}  "
            f"Kappa={val_m['kappa']:.4f}  lr={lr_now:.2e}"
        )

        # TensorBoard epoch-level scalars
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("val/loss",         val_loss,   epoch)
        for metric_name, metric_val in val_m.items():
            writer.add_scalar(f"val/{metric_name}", metric_val, epoch)
        writer.add_scalar("train/lr_epoch", lr_now, epoch)

        # ── save last checkpoint ───────────────────────────────────────
        _save_checkpoint(
            output_dir / "last_model.pth",
            model, optimizer, scheduler, epoch, best_f1, cfg,
        )

        # ── save best checkpoint ───────────────────────────────────────
        if val_m["f1"] > best_f1:
            best_f1   = val_m["f1"]
            no_improve = 0
            _save_checkpoint(
                output_dir / "best_model.pth",
                model, optimizer, scheduler, epoch, best_f1, cfg,
            )
            logger.info(f"  *** New best F1={best_f1:.4f}  saved best_model.pth ***")
        else:
            no_improve += 1
            logger.info(
                f"  No improvement for {no_improve}/{cfg['patience']} epochs"
            )

        # ── early stopping ────────────────────────────────────────────
        if no_improve >= cfg["patience"]:
            logger.info(
                f"Early stopping triggered after {no_improve} epochs "
                f"without improvement."
            )
            break

    writer.close()
    logger.info(f"Training complete.  Best val-F1 = {best_f1:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
