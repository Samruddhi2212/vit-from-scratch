#!/bin/bash
#================================================================
# SLURM Job Script: Siamese ViT Change Detection (LEVIR-CD)
# Pipeline: train directly on LEVIR-CD (pre-split, no preprocessing)
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=8:00:00
#SBATCH --job-name=svit_multiscale
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "========================================"
echo "Siamese ViT Change Detection — LEVIR-CD"
echo "Date:   $(date)"
echo "Node:   $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# ── Load modules ──────────────────────────────────────────────
module purge
module load anaconda3/2024.06

# ── Activate environment ──────────────────────────────────────
ENV_NAME="vit-cd"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# ── GPU info ──────────────────────────────────────────────────
echo ""
echo "GPU Info:"
nvidia-smi
echo ""

# ── Navigate to project ───────────────────────────────────────
cd /home/katoch.aa/ondemand/vit-from-scratch-main

# ── Paths ─────────────────────────────────────────────────────
PROJECT_DIR="/home/katoch.aa/ondemand/vit-from-scratch-main"
LEVIR_DIR="$PROJECT_DIR/LEVIR CD"     # pre-split: train/A, train/B, train/label, val/, test/
OUTPUT_DIR="$PROJECT_DIR/outputs/siamese_vit"

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

if [ ! -d "$LEVIR_DIR/train/A" ]; then
    echo "ERROR: LEVIR-CD not found at $LEVIR_DIR"
    exit 1
fi

echo "Image counts (complete A+B+label triplets):"
for split in train val test; do
    count=$(ls "$LEVIR_DIR/$split/A/"*.png 2>/dev/null | wc -l)
    echo "  $split: $count images in A/"
done
echo "Note: Dataset loader will automatically filter to complete A/B/label triplets."

# ─────────────────────────────────────────────────────────────
# STEP 2: Train
# ─────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "STEP 2: Training Siamese ViT"
echo "========================================"

# Check for existing checkpoint to resume from
RESUME_CKPT="$OUTPUT_DIR/last_model.pth"
RESUME_FLAG=""
if [ -f "$RESUME_CKPT" ]; then
    echo "Found checkpoint at $RESUME_CKPT — resuming."
    RESUME_FLAG="--resume $RESUME_CKPT"
fi

python train.py \
    --data_dir          "$LEVIR_DIR" \
    --output_dir        "$OUTPUT_DIR" \
    --epochs            200 \
    --batch_size        8 \
    --eval_batch_size   16 \
    --lr                3e-4 \
    --weight_decay      0.05 \
    --warmup_epochs     20 \
    --min_lr            1e-6 \
    --grad_clip         1.0 \
    --patience          50 \
    --encoder_lr_scale  0.5 \
    --n_crops           4 \
    --num_workers       $SLURM_CPUS_PER_TASK \
    --loss              focal_dice \
    --pos_weight        20.0 \
    --threshold         0.35 \
    $RESUME_FLAG

TRAIN_EXIT=$?

# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────
echo ""
echo "========================================"
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "Pipeline complete at $(date)"
    echo "Best model : $OUTPUT_DIR/best_model.pth"
    echo "Last model : $OUTPUT_DIR/last_model.pth"
    echo "Logs       : $OUTPUT_DIR/train.log"
    echo "TensorBoard: tensorboard --logdir $OUTPUT_DIR"
else
    echo "Training FAILED (exit code $TRAIN_EXIT) at $(date)"
    exit $TRAIN_EXIT
fi
echo "========================================"
