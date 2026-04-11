#!/bin/bash
#================================================================
# SLURM Job Script: Siamese U-Net (FC-Siam-diff) Change Detection
# Baseline comparison for Siamese ViT on LEVIR-CD
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=siamese_unet_cd
#SBATCH --output=logs/unet_%j.out
#SBATCH --error=logs/unet_%j.err

echo "========================================"
echo "Siamese U-Net Baseline — LEVIR-CD"
echo "Date:   $(date)"
echo "Node:   $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# ── Load modules ──────────────────────────────────────────────
module purge
module load cuda/12.3.0
module load cuDNN/9.10.2
module load anaconda3/2024.06

# ── Activate environment ──────────────────────────────────────
PROJ_DIR="/home/patodia.pa/vit-from-scratch"
source "$PROJ_DIR/.venv_hpc/bin/activate"
export PYTHONNOUSERSITE=1

# ── GPU info ──────────────────────────────────────────────────
echo ""
echo "GPU Info:"
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"
echo ""

# ── Navigate to project ───────────────────────────────────────
cd "$PROJ_DIR"

# ── Paths ─────────────────────────────────────────────────────
DATA_DIR="$PROJ_DIR/LEVIR"
OUTPUT_DIR="$PROJ_DIR/outputs/siamese_unet"

# ── Create output dirs ────────────────────────────────────────
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# ─────────────────────────────────────────────────────────────
# STEP 1: Verify LEVIR-CD data
# ─────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "STEP 1: Verifying LEVIR-CD dataset"
echo "========================================"

if [ ! -d "$DATA_DIR/train/A" ]; then
    echo "ERROR: LEVIR-CD not found at $DATA_DIR"
    echo "Expected: $DATA_DIR/train/A/, $DATA_DIR/train/B/, $DATA_DIR/train/label/"
    exit 1
fi

echo "Image counts:"
for split in train val test; do
    count=$(ls "$DATA_DIR/$split/A/"*.png 2>/dev/null | wc -l)
    echo "  $split: $count images in A/"
done

# ─────────────────────────────────────────────────────────────
# STEP 2: Train Siamese U-Net
# ─────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "STEP 2: Training Siamese U-Net (FC-Siam-diff)"
echo "========================================"

python train.py \
    --config        configs/train_unet_config.yaml \
    --model         unet \
    --data_dir      "$DATA_DIR" \
    --output_dir    "$OUTPUT_DIR" \
    --device        cuda \
    --num_workers   "$SLURM_CPUS_PER_TASK"

TRAIN_EXIT=$?

# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────
echo ""
echo "========================================"
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "U-Net training complete at $(date)"
    echo "Best model : $OUTPUT_DIR/best_model.pth"
    echo "Last model : $OUTPUT_DIR/last_model.pth"
    echo "Logs       : $OUTPUT_DIR/train.log"
    echo "TensorBoard: tensorboard --logdir $OUTPUT_DIR"
else
    echo "Training FAILED (exit code $TRAIN_EXIT) at $(date)"
    exit $TRAIN_EXIT
fi
echo "========================================"
