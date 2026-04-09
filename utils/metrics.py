"""
Evaluation metrics for binary change detection.

ChangeDetectionMetrics accumulates confusion-matrix counts across batches
and computes Precision, Recall, F1, IoU, Accuracy, and Cohen's Kappa
at the end of an epoch.

All metrics are pixel-level binary classification metrics where:
    Positive class = change  (label 1)
    Negative class = no-change (label 0)
"""

import torch
import numpy as np


class ChangeDetectionMetrics:
    """Accumulate per-batch predictions and compute epoch-level metrics.

    Usage:
        metrics = ChangeDetectionMetrics(threshold=0.5)
        for batch in loader:
            logits, targets = model(img1, img2), batch['mask']
            metrics.update(logits, targets)
        results = metrics.compute()
        metrics.reset()

    Args:
        threshold: Probability cutoff for binarising sigmoid output.  Default 0.5.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @torch.no_grad()
    def update(
        self, pred_logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """
        Args:
            pred_logits : (B, 1, H, W) raw logits
            targets     : (B, 1, H, W) binary ground truth in {0, 1}
        """
        preds = (torch.sigmoid(pred_logits) > self.threshold).long()
        tgts  = targets.long()

        self.tp += int((preds *  tgts      ).sum())
        self.fp += int((preds * (1 - tgts) ).sum())
        self.fn += int(((1 - preds) * tgts ).sum())
        self.tn += int(((1 - preds) * (1 - tgts)).sum())

    def compute(self) -> dict[str, float]:
        """Return dict of all metrics for the accumulated predictions."""
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-8

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        iou       = tp / (tp + fp + fn + eps)
        accuracy  = (tp + tn) / (tp + fp + fn + tn + eps)

        # Cohen's Kappa
        total    = tp + fp + fn + tn
        p_o      = (tp + tn) / (total + eps)                     # observed agreement
        p_e_pos  = ((tp + fp) / (total + eps)) * ((tp + fn) / (total + eps))
        p_e_neg  = ((fn + tn) / (total + eps)) * ((fp + tn) / (total + eps))
        p_e      = p_e_pos + p_e_neg                              # expected agreement
        kappa    = (p_o - p_e) / (1 - p_e + eps)

        return {
            "precision": round(float(precision), 4),
            "recall"   : round(float(recall),    4),
            "f1"       : round(float(f1),        4),
            "iou"      : round(float(iou),       4),
            "accuracy" : round(float(accuracy),  4),
            "kappa"    : round(float(kappa),     4),
        }

    def __repr__(self) -> str:
        m = self.compute()
        return (
            f"Metrics("
            f"F1={m['f1']:.4f}  IoU={m['iou']:.4f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
            f"Kappa={m['kappa']:.4f})"
        )
