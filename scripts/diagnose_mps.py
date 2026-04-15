"""
Diagnostic script: find the exact operation whose backward fails on MPS.

torch.autograd.set_detect_anomaly(True) records a Python stack trace for
every autograd node during the forward pass.  When the backward fails, it
prints the forward-pass stack trace that *created* the failing node —
pinpointing the exact line of model code responsible.

Usage:
    python scripts/diagnose_mps.py
"""

import torch
import torch.nn as nn

def main():
    if not torch.backends.mps.is_available():
        print("MPS is not available on this machine.")
        return

    device = torch.device("mps")
    print(f"PyTorch {torch.__version__}  device={device}\n")

    # ── build model ──────────────────────────────────────────────────────
    from models.siamese_vit import SiameseViTChangeDetection
    from utils.losses import FocalDiceLoss

    model = SiameseViTChangeDetection(
        img_size=256, patch_size=16, in_channels=3,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        diff_type="concat_project",
    ).to(device)
    criterion = FocalDiceLoss().to(device)

    img1   = torch.randn(1, 3, 256, 256, device=device)
    img2   = torch.randn(1, 3, 256, 256, device=device)
    target = torch.randint(0, 2, (1, 1, 256, 256), dtype=torch.float32,
                           device=device)

    # ── forward + backward with anomaly detection ────────────────────────
    print("Running forward + backward with anomaly detection enabled...")
    print("(This will print the forward-pass traceback of the failing op)\n")

    torch.autograd.set_detect_anomaly(True)
    try:
        logits = model(img1, img2)
        loss = criterion(logits, target)
        loss.backward()
        print("SUCCESS — backward completed without error on MPS.")
    except RuntimeError as e:
        print(f"\nERROR CAUGHT: {e}")
        print("\nThe traceback above (printed by anomaly detection) shows")
        print("exactly which forward operation's backward failed.")
        print("Look for 'Traceback of forward call that caused the error'.")

    torch.autograd.set_detect_anomaly(False)


if __name__ == "__main__":
    main()
