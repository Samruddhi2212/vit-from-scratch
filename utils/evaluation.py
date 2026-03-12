"""
Evaluation Utilities for Vision Transformer.

This file contains:
1. Full test set evaluation with Top-1 and Top-5 accuracy
2. Per-class accuracy breakdown
3. Confusion matrix generation and plotting
4. Prediction extraction for further analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import ViTConfig
from utils.dataset import CIFAR10_CLASSES


@torch.no_grad()
def get_all_predictions(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the model on an entire dataset and collect all predictions.
    
    This is the foundation for all other evaluation functions.
    We run inference once and reuse the results.
    
    Args:
        model: Trained ViT model
        data_loader: Test or validation data loader
        device: GPU/CPU device
    
    Returns:
        all_logits: Raw model outputs, shape [N, num_classes]
        all_preds: Predicted class indices, shape [N]
        all_labels: True class indices, shape [N]
    """
    model.eval()
    
    all_logits = []
    all_labels = []
    
    for images, labels in tqdm(data_loader, desc="Running inference", leave=False):
        images = images.to(device)
        logits = model(images)
        
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    
    all_logits = torch.cat(all_logits, dim=0)   # [N, num_classes]
    all_labels = torch.cat(all_labels, dim=0)    # [N]
    all_preds = all_logits.argmax(dim=1)         # [N]
    
    return all_logits, all_preds, all_labels


def compute_accuracy(
    all_logits: torch.Tensor,
    all_labels: torch.Tensor
) -> tuple[float, float]:
    """
    Compute Top-1 and Top-5 accuracy.
    
    Top-1: Is the highest-scoring class the correct one?
    Top-5: Is the correct class among the 5 highest-scoring classes?
    
    Top-5 is useful because sometimes the model's second or third
    guess is correct — this shows the model "almost" got it right.
    
    Args:
        all_logits: Raw model outputs, shape [N, num_classes]
        all_labels: True class indices, shape [N]
    
    Returns:
        top1_acc: Top-1 accuracy (%)
        top5_acc: Top-5 accuracy (%)
    """
    N = all_labels.shape[0]
    
    # Top-1: predicted class == true class
    top1_preds = all_logits.argmax(dim=1)
    top1_correct = (top1_preds == all_labels).sum().item()
    top1_acc = 100.0 * top1_correct / N
    
    # Top-5: true class is in the top 5 predictions
    # For CIFAR-10 (10 classes), top-5 is almost always ~99%+
    # It's more meaningful for datasets with many classes (like ImageNet)
    num_classes = all_logits.shape[1]
    k = min(5, num_classes)
    top5_preds = all_logits.topk(k, dim=1).indices  # [N, 5]
    top5_correct = 0
    for i in range(N):
        if all_labels[i] in top5_preds[i]:
            top5_correct += 1
    top5_acc = 100.0 * top5_correct / N
    
    return top1_acc, top5_acc


def compute_per_class_accuracy(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    class_names: list[str] = None
) -> dict:
    """
    Compute accuracy for each individual class.
    
    This reveals which classes the model struggles with.
    For example, ViT might confuse 'cat' with 'dog' more
    than 'airplane' with 'ship'.
    
    Args:
        all_preds: Predicted class indices, shape [N]
        all_labels: True class indices, shape [N]
        class_names: List of class name strings
    
    Returns:
        results: Dict with per-class stats
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES
    
    num_classes = len(class_names)
    results = {}
    
    for i in range(num_classes):
        mask = (all_labels == i)
        total = mask.sum().item()
        if total > 0:
            correct = ((all_preds == i) & mask).sum().item()
            accuracy = 100.0 * correct / total
        else:
            correct = 0
            accuracy = 0.0
        
        results[class_names[i]] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        }
    
    return results


def compute_confusion_matrix(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    num_classes: int = 10
) -> np.ndarray:
    """
    Compute the confusion matrix.
    
    Entry (i, j) = number of samples with true class i
    that were predicted as class j.
    
    Diagonal entries = correct predictions.
    Off-diagonal entries = errors (and which specific errors).
    
    Args:
        all_preds: Predicted class indices, shape [N]
        all_labels: True class indices, shape [N]
        num_classes: Number of classes
    
    Returns:
        cm: Confusion matrix, shape [num_classes, num_classes]
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for pred, label in zip(all_preds.numpy(), all_labels.numpy()):
        cm[label, pred] += 1
    
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] = None,
    normalize: bool = True,
    save_path: str = None,
    figsize: tuple = (10, 8)
):
    """
    Plot a confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix, shape [num_classes, num_classes]
        class_names: List of class name strings
        normalize: If True, show percentages instead of counts
        save_path: If provided, save the figure to this path
        figsize: Figure size
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES
    
    if normalize:
        # Normalize each row (true class) to sum to 100%
        cm_display = cm.astype(np.float64)
        row_sums = cm_display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm_display = cm_display / row_sums * 100
        fmt = '.1f'
        title = 'Confusion Matrix (% per true class)'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix (counts)'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Percentage (%)' if normalize else 'Count'}
    )
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()


def full_evaluation(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str] = None,
    save_dir: str = "results",
    experiment_name: str = "vit_cifar10"
) -> dict:
    """
    Run complete evaluation: accuracy, per-class breakdown, confusion matrix.
    
    This is the one-stop function you call after training.
    
    Args:
        model: Trained ViT model
        test_loader: Test data loader
        device: GPU/CPU device
        class_names: List of class name strings
        save_dir: Directory to save plots
        experiment_name: Name prefix for saved files
    
    Returns:
        results: Dict with all evaluation metrics
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES
    
    print("=" * 60)
    print("FULL MODEL EVALUATION")
    print("=" * 60)
    
    # ─── Get all predictions ───
    all_logits, all_preds, all_labels = get_all_predictions(model, test_loader, device)
    print(f"Total test samples: {len(all_labels):,}")
    
    # ─── Overall accuracy ───
    top1_acc, top5_acc = compute_accuracy(all_logits, all_labels)
    print(f"\nOverall Accuracy:")
    print(f"  Top-1: {top1_acc:.2f}%")
    print(f"  Top-5: {top5_acc:.2f}%")
    
    # ─── Per-class accuracy ───
    per_class = compute_per_class_accuracy(all_preds, all_labels, class_names)
    
    print(f"\nPer-Class Accuracy:")
    print(f"  {'Class':>12s} | {'Correct':>7s} | {'Total':>5s} | {'Accuracy':>8s}")
    print(f"  {'-'*42}")
    
    for cls_name, stats in sorted(per_class.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"  {cls_name:>12s} | {stats['correct']:>7d} | {stats['total']:>5d} | {stats['accuracy']:>7.2f}%")
    
    # ─── Confusion matrix ───
    cm = compute_confusion_matrix(all_preds, all_labels, len(class_names))
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save normalized version
    plot_confusion_matrix(
        cm, class_names, normalize=True,
        save_path=os.path.join(save_dir, f"{experiment_name}_confusion_matrix.png")
    )
    
    # ─── Find most confused pairs ───
    print(f"\nMost Confused Class Pairs:")
    cm_no_diag = cm.copy().astype(float)
    np.fill_diagonal(cm_no_diag, 0)
    
    # Get top 5 confused pairs
    for _ in range(5):
        i, j = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
        count = int(cm_no_diag[i, j])
        if count == 0:
            break
        total_i = cm[i].sum()
        pct = 100.0 * count / total_i if total_i > 0 else 0
        print(f"  {class_names[i]:>12s} → {class_names[j]:<12s}: {count} times ({pct:.1f}% of {class_names[i]})")
        cm_no_diag[i, j] = 0  # zero out to find next pair
    
    # ─── Compile results ───
    results = {
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'per_class': per_class,
        'confusion_matrix': cm,
        'all_logits': all_logits,
        'all_preds': all_preds,
        'all_labels': all_labels,
    }
    
    print(f"\n{'='*60}")
    
    return results


# ──────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    
    print("=" * 60)
    print("TEST 1: Evaluation functions with dummy data")
    print("=" * 60)
    
    # Create fake predictions to test the evaluation functions
    num_samples = 1000
    num_classes = 10
    
    # Simulate a decent model: 80% correct predictions
    all_labels = torch.randint(0, num_classes, (num_samples,))
    all_logits = torch.randn(num_samples, num_classes)
    
    # Make the correct class have a high logit 80% of the time
    for i in range(num_samples):
        if torch.rand(1).item() < 0.8:
            all_logits[i, all_labels[i]] += 5.0  # boost correct class
    
    all_preds = all_logits.argmax(dim=1)
    
    # Test accuracy computation
    top1, top5 = compute_accuracy(all_logits, all_labels)
    print(f"Simulated Top-1 accuracy: {top1:.1f}% (expected ~80%)")
    print(f"Simulated Top-5 accuracy: {top5:.1f}% (expected ~99%)")
    
    assert 70 < top1 < 90, f"Top-1 should be around 80%, got {top1:.1f}%"
    assert top5 > 85, f"Top-5 should be very high, got {top5:.1f}%"
    print(" Accuracy computation PASSED!")
    
    # Test per-class accuracy
    per_class = compute_per_class_accuracy(all_preds, all_labels)
    print(f"\nPer-class results for {len(per_class)} classes:")
    for cls, stats in list(per_class.items())[:3]:
        print(f"  {cls}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
    print(" Per-class accuracy PASSED!")
    
    # Test confusion matrix
    cm = compute_confusion_matrix(all_preds, all_labels, num_classes)
    assert cm.shape == (num_classes, num_classes)
    assert cm.sum() == num_samples, f"CM sum should be {num_samples}, got {cm.sum()}"
    print(f"\nConfusion matrix shape: {cm.shape}, total: {cm.sum()}")
    print(" Confusion matrix PASSED!")
    
    # Test plotting (save to file)
    os.makedirs("results", exist_ok=True)
    plot_confusion_matrix(
        cm, CIFAR10_CLASSES, normalize=True,
        save_path="results/test_confusion_matrix.png"
    )
    assert os.path.exists("results/test_confusion_matrix.png")
    print(" Confusion matrix plot PASSED!")
    
    print("\n All evaluation tests PASSED!")