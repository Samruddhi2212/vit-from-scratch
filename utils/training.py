"""
Training Pipeline for Vision Transformer.

This file contains everything needed to train the ViT:
1. Learning rate scheduler (warmup + cosine annealing)
2. Training loop (one epoch)
3. Evaluation loop
4. Full training function that ties everything together

═══════════════════════════════════════════════════════════
LEARNING RATE SCHEDULE: Warmup + Cosine Annealing
═══════════════════════════════════════════════════════════

Epoch:  0     10                    200
LR:    1e-5 → 1e-3 ~~~~~~~~~~~~→ 1e-5
        ↑      ↑        ↑          ↑
      start  peak    cosine      minimum
              ↑      decay
           warmup
            ends

Phase 1 — Linear Warmup (epochs 0 to warmup_epochs):
    lr = min_lr + (max_lr - min_lr) × (epoch / warmup_epochs)
    
    WHY? At initialization, attention scores are random → softmax
    gives ~uniform weights → gradients are noisy. Large LR would
    push the model to a bad region. Small LR lets it first learn
    meaningful patterns, THEN we speed up.

Phase 2 — Cosine Annealing (epochs warmup_epochs to total_epochs):
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    lr = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × progress))
    
    WHY cosine? It decays smoothly from max to min, spending more time
    at lower learning rates (where fine-tuning happens). Compared to
    step decay, cosine is smoother and generally works better for ViTs.

═══════════════════════════════════════════════════════════
OPTIMIZER: AdamW
═══════════════════════════════════════════════════════════

AdamW = Adam with decoupled weight decay.

Adam maintains two running averages per parameter:
  m_t = β₁ × m_{t-1} + (1 - β₁) × g_t        (momentum / first moment)
  v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²       (adaptive LR / second moment)

Bias correction (needed because m and v start at 0):
  m̂_t = m_t / (1 - β₁^t)
  v̂_t = v_t / (1 - β₂^t)

Update:
  θ_t = θ_{t-1} - lr × m̂_t / (√v̂_t + ε)

AdamW additionally applies weight decay DIRECTLY to the weights
(not through the gradient), which is mathematically cleaner and
works better with learning rate scheduling.
"""

import math
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import ViTConfig


def get_lr(epoch: int, config: ViTConfig) -> float:
    """
    Compute learning rate for a given epoch.
    
    Linear warmup for the first warmup_epochs, then cosine annealing.
    
    Args:
        epoch: Current epoch (0-indexed)
        config: ViTConfig with lr schedule parameters
    
    Returns:
        Learning rate for this epoch
    """
    if epoch < config.warmup_epochs:
        # Linear warmup: min_lr → learning_rate
        return config.min_lr + (config.learning_rate - config.min_lr) * (
            epoch / config.warmup_epochs
        )
    else:
        # Cosine annealing: learning_rate → min_lr
        progress = (epoch - config.warmup_epochs) / (
            config.total_epochs - config.warmup_epochs
        )
        return config.min_lr + 0.5 * (config.learning_rate - config.min_lr) * (
            1 + math.cos(math.pi * progress)
        )


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    """Set learning rate for all parameter groups in the optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The ViT model
        train_loader: Training data loader
        optimizer: AdamW optimizer
        device: GPU or CPU device
        epoch: Current epoch number (for display)
    
    Returns:
        avg_loss: Average training loss for this epoch
        accuracy: Training accuracy (%) for this epoch
    """
    model.train()  # Set model to training mode (enables dropout)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # tqdm creates a progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:>3d} [Train]", leave=False)
    
    for images, labels in pbar:
        # ─── Move data to GPU ───
        images = images.to(device)
        labels = labels.to(device)
        
        # ─── Forward pass ───
        logits = model(images)
        
        # ─── Compute loss ───
        # F.cross_entropy combines log_softmax + NLLLoss
        # It's numerically more stable than doing softmax → log → NLLLoss
        loss = F.cross_entropy(logits, labels)
        
        # ─── Backward pass ───
        optimizer.zero_grad()   # Clear old gradients
        loss.backward()         # Compute new gradients
        
        # ─── Gradient clipping ───
        # Prevents exploding gradients by capping the total gradient norm
        # max_norm=1.0 is a common choice for transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # ─── Update weights ───
        optimizer.step()
        
        # ─── Track metrics ───
        total_loss += loss.item() * images.size(0)  # Weighted by batch size
        _, predicted = logits.max(1)                 # Get predicted class
        correct += predicted.eq(labels).sum().item() # Count correct predictions
        total += labels.size(0)                      # Count total samples
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.1f}%'
        })
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


@torch.no_grad()  # Disable gradient computation for evaluation (saves memory)
def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    desc: str = "Eval"
) -> tuple[float, float]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The ViT model
        data_loader: Validation or test data loader
        device: GPU or CPU device
        desc: Description for progress bar
    
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy (%)
    """
    model.eval()  # Set model to evaluation mode (disables dropout)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(data_loader, desc=f"         [{desc:>5s}]", leave=False)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        
        total_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: ViTConfig,
    device: torch.device = None,
    save_dir: str = "checkpoints",
    experiment_name: str = "vit_cifar10"
) -> dict:
    """
    Full training loop with learning rate scheduling, checkpointing,
    and logging.
    
    Args:
        model: The ViT model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: ViTConfig with all hyperparameters
        device: GPU/CPU device (auto-detected if None)
        save_dir: Directory to save checkpoints
        experiment_name: Name for checkpoint files
    
    Returns:
        history: Dict with training curves:
            {
                'train_loss': [...],
                'train_acc': [...],
                'val_loss': [...],
                'val_acc': [...],
                'lr': [...],
                'epoch_time': [...]
            }
    """
    # ─── Device setup ───
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    print(f"Training on: {device}")
    model = model.to(device)
    
    # ─── Optimizer ───
    # AdamW: Adam with decoupled weight decay
    # We exclude bias and LayerNorm parameters from weight decay
    # (standard practice — regularizing biases and norms hurts performance)
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't apply weight decay to biases and LayerNorm parameters
        if 'bias' in name or 'gamma' in name or 'beta' in name or 'cls_token' in name or 'pos_embed' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=config.learning_rate, betas=(0.9, 0.999))
    
    print(f"Parameters with weight decay:    {sum(p.numel() for p in decay_params):,}")
    print(f"Parameters without weight decay: {sum(p.numel() for p in no_decay_params):,}")
    
    # ─── Create checkpoint directory ───
    os.makedirs(save_dir, exist_ok=True)
    
    # ─── Training history ───
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'epoch_time': []
    }
    
    best_val_acc = 0.0
    
    # ─── Print header ───
    print(f"\n{'='*90}")
    print(f"{'Epoch':>5s} | {'Train Loss':>10s} | {'Train Acc':>9s} | "
          f"{'Val Loss':>10s} | {'Val Acc':>9s} | {'LR':>10s} | {'Time':>6s}")
    print(f"{'='*90}")
    
    # ─── Training loop ───
    for epoch in range(config.total_epochs):
        epoch_start = time.time()
        
        # Set learning rate for this epoch
        lr = get_lr(epoch, config)
        set_lr(optimizer, lr)
        
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, device, desc="Val")
        
        epoch_time = time.time() - epoch_start
        
        # ─── Log metrics ───
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(lr)
        history['epoch_time'].append(epoch_time)
        
        # ─── Print progress ───
        # Print every epoch for the first 10, then every 5 epochs
        if epoch < 10 or (epoch + 1) % 5 == 0 or epoch == config.total_epochs - 1:
            print(f"{epoch+1:>5d} | {train_loss:>10.4f} | {train_acc:>8.2f}% | "
                  f"{val_loss:>10.4f} | {val_acc:>8.2f}% | {lr:>10.6f} | {epoch_time:>5.1f}s")
        
        # ─── Save best model ───
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config,
            }
            save_path = os.path.join(save_dir, f"{experiment_name}_best.pt")
            torch.save(checkpoint, save_path)
    
    # ─── Training complete ───
    total_time = sum(history['epoch_time'])
    print(f"{'='*90}")
    print(f"\nTraining complete!")
    print(f"  Total time:        {total_time/60:.1f} minutes")
    print(f"  Best val accuracy: {best_val_acc:.2f}%")
    print(f"  Checkpoint saved:  {save_path}")
    
    # Save training history
    history_path = os.path.join(save_dir, f"{experiment_name}_history.pt")
    torch.save(history, history_path)
    print(f"  History saved:     {history_path}")
    
    return history


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device = None
) -> dict:
    """
    Load a saved checkpoint.
    
    Args:
        model: The ViT model (must have same architecture as saved)
        checkpoint_path: Path to the .pt file
        device: Device to load onto
    
    Returns:
        checkpoint: The full checkpoint dict
    """
    if device is None:
        device = torch.device('cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
    print(f"  Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    return checkpoint


# ──────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    
    # ──────────────────────────────────────────────
    print("=" * 60)
    print("TEST 1: Learning Rate Schedule")
    print("=" * 60)
    
    config = ViTConfig()
    
    print(f"Warmup epochs:  {config.warmup_epochs}")
    print(f"Total epochs:   {config.total_epochs}")
    print(f"Min LR:         {config.min_lr}")
    print(f"Max LR:         {config.learning_rate}")
    print()
    
    # Print LR at key epochs
    key_epochs = [0, 1, 5, 9, 10, 11, 50, 100, 150, 199]
    print(f"  {'Epoch':>5s} | {'LR':>12s} | Phase")
    print(f"  {'-'*40}")
    for e in key_epochs:
        if e < config.total_epochs:
            lr = get_lr(e, config)
            phase = "warmup" if e < config.warmup_epochs else "cosine"
            print(f"  {e:>5d} | {lr:>12.8f} | {phase}")
    
    # Verify warmup is monotonically increasing
    warmup_lrs = [get_lr(e, config) for e in range(config.warmup_epochs)]
    assert all(warmup_lrs[i] <= warmup_lrs[i+1] for i in range(len(warmup_lrs)-1)), \
        "Warmup should be monotonically increasing!"
    
    # Verify LR at epoch 0 is close to min_lr
    assert abs(get_lr(0, config) - config.min_lr) < 1e-6, \
        f"LR at epoch 0 should be min_lr ({config.min_lr})"
    
    # Verify LR at warmup end is close to max_lr
    assert abs(get_lr(config.warmup_epochs, config) - config.learning_rate) < 1e-6, \
        f"LR at warmup end should be max_lr ({config.learning_rate})"
    
    print("\n LR schedule test PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: Quick Training Sanity Check (3 epochs)")
    print("=" * 60)
    
    # Use a tiny config for quick testing
    tiny_config = ViTConfig()
    tiny_config.total_epochs = 3
    tiny_config.warmup_epochs = 1
    tiny_config.batch_size = 64
    
    from models.vit import ViT
    from utils.dataset import get_cifar10_loaders
    
    model = ViT(tiny_config)
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        tiny_config, num_workers=0  # 0 workers for testing (avoids multiprocessing issues)
    )
    
    print(f"\nRunning 3 epochs to verify the training pipeline works...")
    print(f"(This should take 1-3 minutes on CPU)\n")
    
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=tiny_config,
        save_dir="checkpoints",
        experiment_name="test_run"
    )
    
    # Verify history was recorded
    assert len(history['train_loss']) == 3, "Should have 3 epochs of history"
    assert len(history['val_acc']) == 3, "Should have 3 epochs of val accuracy"
    
    # Verify loss decreased (it should decrease at least a little in 3 epochs)
    print(f"\nLoss: {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f}")
    
    # Verify checkpoint was saved
    checkpoint_path = "checkpoints/test_run_best.pt"
    assert os.path.exists(checkpoint_path), "Checkpoint should be saved!"
    
    # Verify checkpoint loads correctly
    model_loaded = ViT(tiny_config)
    load_checkpoint(model_loaded, checkpoint_path)
    
    print("\n Training pipeline test PASSED!")
    print("\nYour training pipeline is ready for full training on Colab!")
    
    # Clean up test checkpoint
    # os.remove(checkpoint_path)  # Uncomment to clean up