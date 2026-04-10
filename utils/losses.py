"""
Loss functions for binary change-detection.

Both losses operate on raw logits (no sigmoid applied beforehand) so they
can use PyTorch's numerically stable fused implementations.

BCEDiceLoss   — Balanced BCE + Dice.  Good default for moderately imbalanced data.
FocalDiceLoss — Focal + Dice.  Better when change pixels are very rare (<5 %).

Dice loss:
    Dice = (2 * TP) / (2*TP + FP + FN)
         = (2 * Σ p*t) / (Σ p + Σ t + ε)
    Loss = 1 − Dice
    Differentiable because we use soft probabilities (sigmoid output), not binary.

Focal loss (Lin et al., 2017):
    FL(p) = −α (1−p)^γ log(p)   for positive examples
    FL(p) = −(1−α) p^γ log(1−p) for negative examples
    γ > 0 reduces the relative loss for easy examples → focus on hard ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy + Dice loss.

    Args:
        bce_weight : Weight for BCE term.          Default 0.5.
        dice_weight: Weight for Dice term.         Default 0.5.
        pos_weight : Scalar tensor to up-weight positive class in BCE.
                     Useful when change pixels are rare.  Default None.
        smooth     : Laplace smoothing for Dice denominator.  Default 1.0.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: torch.Tensor | None = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.smooth      = smooth
        self.register_buffer("pos_weight", pos_weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) raw model output
            targets : (B, 1, H, W) binary ground truth in {0, 1}
        Returns:
            Scalar loss.
        """
        # BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weight
        )

        # Soft Dice
        probs = torch.sigmoid(logits)
        probs_f  = probs.reshape(-1)
        target_f = targets.float().reshape(-1)
        intersection = (probs_f * target_f).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            probs_f.sum() + target_f.sum() + self.smooth
        )

        return self.bce_weight * bce + self.dice_weight * dice_loss


class FocalDiceLoss(nn.Module):
    """Focal Loss + Dice loss for handling severe class imbalance.

    Args:
        alpha      : Focal balancing factor for positives.   Default 0.25.
        gamma      : Focal modulating exponent.              Default 2.0.
        dice_weight: Weight for Dice term.                   Default 0.5.
        focal_weight: Weight for Focal term.                 Default 0.5.
        smooth     : Laplace smoothing for Dice denominator. Default 1.0.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha        = alpha
        self.gamma        = gamma
        self.dice_weight  = dice_weight
        self.focal_weight = focal_weight
        self.smooth       = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W)
            targets : (B, 1, H, W) in {0, 1}
        Returns:
            Scalar loss.
        """
        targets_f = targets.float()

        # Focal loss (element-wise)
        bce_elem  = F.binary_cross_entropy_with_logits(
            logits, targets_f, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt    = targets_f * probs + (1 - targets_f) * (1 - probs)
        alpha_t = targets_f * self.alpha + (1 - targets_f) * (1 - self.alpha)
        focal = alpha_t * (1 - pt) ** self.gamma * bce_elem
        focal_loss = focal.mean()

        # Soft Dice
        probs_f  = probs.reshape(-1)
        target_f = targets_f.reshape(-1)
        intersection = (probs_f * target_f).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            probs_f.sum() + target_f.sum() + self.smooth
        )

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
