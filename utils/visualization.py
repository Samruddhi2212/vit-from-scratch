"""
Visualization Utilities for Vision Transformer.

This file contains the visual analysis tools that will be
the stars of your presentation:

1. Attention Map Visualization — what does each head look at?
2. Attention Rollout — cumulative attention across all layers
3. Positional Embedding Similarity — did the model learn 2D structure?
4. t-SNE of [CLS] Embeddings — how does the model cluster classes?
5. Training Curve Plots — loss and accuracy over epochs
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — prevents macOS segfault
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import ViTConfig
from utils.cifar_paths import CIFAR10_RESULTS_DIR
from utils.dataset import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, denormalize_cifar10


# ═══════════════════════════════════════════════════════════
# 1. ATTENTION MAP VISUALIZATION
# ═══════════════════════════════════════════════════════════

def plot_attention_maps(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    save_path: str = None,
    figsize_per_cell: float = 2.0
):
    """
    Visualize attention maps for every head in every layer.
    
    Creates a grid where:
      Rows = layers (1 to num_layers)
      Columns = heads (1 to num_heads)
      Each cell = [CLS] token's attention over patches, overlaid on the image
    
    This shows HOW the model processes the image at each level:
    - Early layers: typically local, texture-focused attention
    - Later layers: typically global, semantic attention
    
    Args:
        model: Trained ViT model
        image: Single image tensor, shape [1, 3, H, W] (normalized)
        device: GPU/CPU device
        save_path: If provided, save the figure
    """
    model.eval()
    config = model.config
    
    # Get attention maps: [num_layers, 1, num_heads, N+1, N+1]
    with torch.no_grad():
        attn_maps = model.get_attention_maps(image.to(device))
    
    # Move to CPU for plotting
    attn_maps = attn_maps.cpu()
    
    num_layers = attn_maps.shape[0]
    num_heads = attn_maps.shape[2]
    grid_size = int(math.sqrt(config.num_patches))  # 8 for CIFAR-10
    
    # De-normalize image for display
    img_display = denormalize_cifar10(image[0].cpu()).permute(1, 2, 0).numpy()
    
    # Create figure
    fig, axes = plt.subplots(
        num_layers, num_heads + 1,  # +1 for original image column
        figsize=((num_heads + 1) * figsize_per_cell, num_layers * figsize_per_cell)
    )
    
    for layer in range(num_layers):
        # Show original image in first column
        axes[layer, 0].imshow(img_display)
        axes[layer, 0].set_title(f'Layer {layer+1}', fontsize=9)
        axes[layer, 0].axis('off')
        
        for head in range(num_heads):
            # Extract [CLS] token's attention to all patches
            # attn_maps[layer, 0, head, 0, 1:] = CLS attending to patches
            cls_attn = attn_maps[layer, 0, head, 0, 1:]  # [num_patches]
            
            # Reshape to 2D grid
            cls_attn = cls_attn.reshape(grid_size, grid_size)  # [8, 8]
            
            # Resize to image dimensions for overlay
            cls_attn = F.interpolate(
                cls_attn.unsqueeze(0).unsqueeze(0),
                size=(config.image_size, config.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            # Plot: image with attention overlay
            axes[layer, head + 1].imshow(img_display)
            axes[layer, head + 1].imshow(cls_attn, alpha=0.6, cmap='viridis')
            if layer == 0:
                axes[layer, head + 1].set_title(f'Head {head+1}', fontsize=9)
            axes[layer, head + 1].axis('off')
    
    plt.suptitle('Attention Maps: [CLS] Token Attending to Patches', fontsize=13, y=1.01)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention maps saved to: {save_path}")
    
    plt.close()


# ═══════════════════════════════════════════════════════════
# 2. ATTENTION ROLLOUT
# ═══════════════════════════════════════════════════════════

def compute_attention_rollout(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device
) -> np.ndarray:
    """
    Compute attention rollout — cumulative attention across all layers.
    
    The idea: multiply attention matrices across layers to see what the
    model looks at OVERALL, not just per layer.
    
    Because of residual connections, at each layer approximately half
    the information comes from attention and half passes through directly.
    So we mix each attention matrix with the identity matrix (50/50).
    
    Result: a single heatmap showing what the model focuses on overall.
    
    Args:
        model: Trained ViT model
        image: Single image tensor, shape [1, 3, H, W]
        device: GPU/CPU device
    
    Returns:
        rollout: Attention rollout for [CLS] token, shape [grid_size, grid_size]
    """
    model.eval()
    config = model.config
    
    with torch.no_grad():
        attn_maps = model.get_attention_maps(image.to(device))
    
    attn_maps = attn_maps.cpu()  # [layers, 1, heads, N+1, N+1]
    
    num_layers = attn_maps.shape[0]
    seq_len = attn_maps.shape[3]  # N+1
    
    # Start with identity matrix
    rollout = torch.eye(seq_len)
    
    for layer in range(num_layers):
        # Average attention across heads for this layer
        attn = attn_maps[layer, 0].mean(dim=0)  # [N+1, N+1]
        
        # Mix with identity (accounting for residual connection)
        # 0.5 * attention + 0.5 * identity
        attn = 0.5 * attn + 0.5 * torch.eye(seq_len)
        
        # Re-normalize rows to sum to 1
        attn = attn / attn.sum(dim=-1, keepdim=True)
        
        # Multiply with running rollout
        rollout = attn @ rollout
    
    # Extract [CLS] token's attention to patches (skip CLS-to-CLS)
    cls_rollout = rollout[0, 1:]  # [num_patches]
    
    # Reshape to 2D grid
    grid_size = int(math.sqrt(config.num_patches))
    cls_rollout = cls_rollout.reshape(grid_size, grid_size).numpy()
    
    return cls_rollout


def plot_attention_rollout(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    save_path: str = None,
    title: str = None
):
    """
    Plot the attention rollout overlaid on the original image.
    
    Args:
        model: Trained ViT model
        image: Single image tensor, shape [1, 3, H, W]
        device: GPU/CPU device
        save_path: If provided, save the figure
        title: Optional title (e.g., "Predicted: airplane")
    """
    config = model.config
    rollout = compute_attention_rollout(model, image, device)
    
    # Resize rollout to image dimensions
    rollout_resized = F.interpolate(
        torch.tensor(rollout).unsqueeze(0).unsqueeze(0),
        size=(config.image_size, config.image_size),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()
    
    # De-normalize image
    img_display = denormalize_cifar10(image[0].cpu()).permute(1, 2, 0).numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    axes[0].imshow(img_display)
    axes[0].set_title('Original Image', fontsize=11)
    axes[0].axis('off')
    
    # Attention rollout heatmap
    axes[1].imshow(rollout_resized, cmap='viridis')
    axes[1].set_title('Attention Rollout', fontsize=11)
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img_display)
    axes[2].imshow(rollout_resized, alpha=0.6, cmap='viridis')
    axes[2].set_title('Overlay', fontsize=11)
    axes[2].axis('off')
    
    if title:
        plt.suptitle(title, fontsize=13, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention rollout saved to: {save_path}")
    
    plt.close()


# ═══════════════════════════════════════════════════════════
# 3. POSITIONAL EMBEDDING SIMILARITY
# ═══════════════════════════════════════════════════════════

def plot_positional_embedding_similarity(
    model: nn.Module,
    save_path: str = None
):
    """
    Visualize the learned positional embeddings as a similarity matrix.
    
    If the model learned meaningful positions, nearby patches should
    have similar embeddings. The similarity matrix should show a
    block-diagonal or distance-based pattern.
    
    Args:
        model: Trained ViT model (or untrained — compare both!)
        save_path: If provided, save the figure
    """
    config = model.config
    grid_size = int(math.sqrt(config.num_patches))
    
    # Extract positional embeddings (skip [CLS] at index 0)
    pos_embed = model.patch_embed.pos_embed[0, 1:].detach().cpu()  # [num_patches, d_model]
    
    # Compute cosine similarity between all pairs
    pos_embed_norm = F.normalize(pos_embed, dim=-1)
    similarity = (pos_embed_norm @ pos_embed_norm.T).numpy()  # [num_patches, num_patches]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Full similarity matrix
    im = axes[0].imshow(similarity, cmap='viridis', vmin=-1, vmax=1)
    axes[0].set_title(f'Positional Embedding Similarity\n({config.num_patches} patches)', fontsize=11)
    axes[0].set_xlabel('Patch Index')
    axes[0].set_ylabel('Patch Index')
    plt.colorbar(im, ax=axes[0], label='Cosine Similarity')
    
    # For each patch, show its similarity to all other patches as a 2D grid
    # Pick 4 reference patches: corners and center
    ref_patches = [0, grid_size - 1, config.num_patches // 2, config.num_patches - 1]
    ref_labels = ['Top-Left', 'Top-Right', 'Center', 'Bottom-Right']
    
    # Create a 2×2 grid of similarity maps
    axes[1].axis('off')
    gs_inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=axes[1].get_subplotspec(), hspace=0.3, wspace=0.3)
    
    for idx, (ref, label) in enumerate(zip(ref_patches, ref_labels)):
        ax_inner = fig.add_subplot(gs_inner[idx])
        sim_map = similarity[ref].reshape(grid_size, grid_size)
        im = ax_inner.imshow(sim_map, cmap='viridis', vmin=-1, vmax=1)
        
        # Mark the reference patch position
        ref_row, ref_col = ref // grid_size, ref % grid_size
        ax_inner.plot(ref_col, ref_row, 'r*', markersize=10)
        
        ax_inner.set_title(f'{label} (patch {ref})', fontsize=9)
        ax_inner.set_xticks([])
        ax_inner.set_yticks([])
    
    plt.suptitle('Learned Positional Embeddings: Does the Model Know 2D Structure?', fontsize=13, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Positional embedding plot saved to: {save_path}")
    
    plt.close()


# ═══════════════════════════════════════════════════════════
# 4. t-SNE OF [CLS] EMBEDDINGS
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def plot_tsne_embeddings(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str] = None,
    max_samples: int = 2000,
    save_path: str = None
):
    """
    Extract [CLS] token embeddings from the model, reduce to 2D with t-SNE,
    and plot colored by class.
    
    If the model learned good representations, you should see clear clusters
    — one per class, well separated.
    
    Args:
        model: Trained ViT model
        data_loader: Test data loader
        device: GPU/CPU device
        class_names: List of class name strings
        max_samples: Maximum number of samples to plot (t-SNE is slow for large N)
        save_path: If provided, save the figure
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES
    
    model.eval()
    
    all_embeddings = []
    all_labels = []
    total = 0
    
    for images, labels in tqdm(data_loader, desc="Extracting embeddings", leave=False):
        images = images.to(device)
        embeddings = model.get_cls_embeddings(images)
        
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)
        
        total += labels.size(0)
        if total >= max_samples:
            break
    
    all_embeddings = torch.cat(all_embeddings, dim=0)[:max_samples]
    all_labels = torch.cat(all_labels, dim=0)[:max_samples]
    
    print(f"Running t-SNE on {len(all_embeddings)} embeddings...")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(all_embeddings.numpy())
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    num_classes = len(class_names)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for i in range(num_classes):
        mask = (all_labels.numpy() == i)
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=class_names[i],
            alpha=0.6,
            s=15
        )
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.set_title('t-SNE of [CLS] Token Embeddings', fontsize=13)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"t-SNE plot saved to: {save_path}")
    
    plt.close()


# ═══════════════════════════════════════════════════════════
# 5. TRAINING CURVES
# ═══════════════════════════════════════════════════════════

def plot_training_curves(
    history: dict,
    save_path: str = None
):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dict from the train() function with keys:
                 train_loss, val_loss, train_acc, val_acc, lr
        save_path: If provided, save the figure
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ─── Loss ───
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', alpha=0.8)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ─── Accuracy ───
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', alpha=0.8)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation', alpha=0.8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # ─── Learning Rate ───
    axes[2].plot(epochs, history['lr'], 'g-', alpha=0.8)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)
    
    # Add warmup boundary line
    if 'lr' in history and len(history['lr']) > 1:
        # Find warmup end (where LR is maximum)
        max_lr_epoch = np.argmax(history['lr']) + 1
        for ax in axes:
            ax.axvline(x=max_lr_epoch, color='gray', linestyle='--', alpha=0.5, label='Warmup End')
    
    plt.suptitle('Training Progress', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.close()


# ═══════════════════════════════════════════════════════════
# 6. MULTI-IMAGE ATTENTION ROLLOUT (for presentation)
# ═══════════════════════════════════════════════════════════

def plot_attention_rollout_grid(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    class_names: list[str] = None,
    save_path: str = None
):
    """
    Show attention rollout for multiple images in a grid.
    Great for presentation slides.
    
    Args:
        model: Trained ViT model
        images: Batch of images, shape [N, 3, H, W]
        labels: True labels, shape [N]
        device: GPU/CPU device
        class_names: List of class name strings
        save_path: If provided, save the figure
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES
    
    model.eval()
    config = model.config
    N = images.shape[0]
    
    # Get predictions
    with torch.no_grad():
        logits = model(images.to(device))
        preds = logits.argmax(dim=1).cpu()
    
    fig, axes = plt.subplots(2, N, figsize=(3 * N, 6))
    
    if N == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(N):
        img_single = images[i:i+1]
        rollout = compute_attention_rollout(model, img_single, device)
        
        # Resize rollout
        rollout_resized = F.interpolate(
            torch.tensor(rollout).unsqueeze(0).unsqueeze(0),
            size=(config.image_size, config.image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        # De-normalize image
        img_display = denormalize_cifar10(img_single[0].cpu()).permute(1, 2, 0).numpy()
        
        # Top row: original image
        axes[0, i].imshow(img_display)
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        color = 'green' if labels[i] == preds[i] else 'red'
        axes[0, i].set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=9, color=color)
        axes[0, i].axis('off')
        
        # Bottom row: attention rollout overlay
        axes[1, i].imshow(img_display)
        axes[1, i].imshow(rollout_resized, alpha=0.6, cmap='viridis')
        axes[1, i].set_title('Attention Rollout', fontsize=9)
        axes[1, i].axis('off')
    
    plt.suptitle('ViT Attention: What Does the Model Look At?', fontsize=13, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention rollout grid saved to: {save_path}")
    
    plt.close()


# ──────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from models.vit import ViT
    from utils.dataset import get_cifar10_loaders
    
    config = ViTConfig()
    device = torch.device('cpu')  # Use CPU for testing
    
    # Create an UNTRAINED model (visualizations still work, just not meaningful)
    model = ViT(config).to(device)
    model.eval()
    
    # Get a few test images
    _, _, test_loader = get_cifar10_loaders(config, num_workers=0)
    images, labels = next(iter(test_loader))
    
    _cifar_test_viz = os.path.join(CIFAR10_RESULTS_DIR, "test_viz")
    os.makedirs(_cifar_test_viz, exist_ok=True)
    
    # ──────────────────────────────────────────────
    print("=" * 60)
    print("TEST 1: Attention Maps")
    print("=" * 60)
    
    single_image = images[0:1]  # Take one image
    plot_attention_maps(
        model, single_image, device,
        save_path=os.path.join(_cifar_test_viz, "attention_maps.png"),
    )
    print(" Attention maps PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: Attention Rollout")
    print("=" * 60)
    
    rollout = compute_attention_rollout(model, single_image, device)
    print(f"Rollout shape: {rollout.shape}")
    
    grid_size = int(math.sqrt(config.num_patches))
    assert rollout.shape == (grid_size, grid_size), \
        f"Expected ({grid_size}, {grid_size}), got {rollout.shape}"
    
    plot_attention_rollout(
        model, single_image, device,
        save_path=os.path.join(_cifar_test_viz, "attention_rollout.png"),
        title="Test: Attention Rollout (untrained model)",
    )
    print(" Attention rollout PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 3: Positional Embedding Similarity")
    print("=" * 60)
    
    plot_positional_embedding_similarity(
        model,
        save_path=os.path.join(_cifar_test_viz, "pos_embed_similarity.png"),
    )
    print(" Positional embedding similarity PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 4: Attention Rollout Grid")
    print("=" * 60)
    
    plot_attention_rollout_grid(
        model,
        images[:6], labels[:6], device,
        save_path=os.path.join(_cifar_test_viz, "rollout_grid.png"),
    )
    print(" Attention rollout grid PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 5: Training Curves (with dummy data)")
    print("=" * 60)
    
    dummy_history = {
        'train_loss': [2.3, 2.0, 1.7, 1.4, 1.1, 0.9, 0.7, 0.5, 0.4, 0.3],
        'val_loss': [2.2, 1.9, 1.6, 1.4, 1.2, 1.1, 1.0, 0.95, 0.92, 0.90],
        'train_acc': [10, 25, 40, 55, 65, 72, 78, 84, 88, 91],
        'val_acc': [12, 27, 42, 52, 60, 65, 69, 72, 74, 75],
        'lr': [0.0001, 0.0005, 0.001, 0.001, 0.0009, 0.0007, 0.0005, 0.0003, 0.0001, 0.00005],
    }
    
    plot_training_curves(
        dummy_history,
        save_path=os.path.join(_cifar_test_viz, "training_curves.png"),
    )
    print(" Training curves PASSED!")
    
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 6: t-SNE Embeddings (small sample)")
    print("=" * 60)
    
    # Use very few samples for speed in testing
    plot_tsne_embeddings(
        model, test_loader, device,
        max_samples=200,  # Small for testing
        save_path=os.path.join(_cifar_test_viz, "tsne.png"),
    )
    print(" t-SNE PASSED!")
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATION TESTS PASSED!")
    print("=" * 60)
    print(f"\nCheck {_cifar_test_viz}/ for all generated plots.")
    print(f"(Note: plots are from an UNTRAINED model, so they won't show")
    print(f" meaningful patterns yet. After training, they'll look great!)")