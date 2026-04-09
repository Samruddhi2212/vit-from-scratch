#!/bin/bash
#================================================================
# SLURM Job Script: Siamese ViT Change Detection (OSCD)
# Full pipeline: preprocess → extract patches → train
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=8:00:00
#SBATCH --job-name=siamese_vit_cd
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "========================================"
echo "Siamese ViT Change Detection — OSCD"
echo "Date:   $(date)"
echo "Node:   $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# ── Load modules ──────────────────────────────────────────────
module purge
module load anaconda3/2024.06

# ── Activate environment ──────────────────────────────────────
ENV_NAME="vit-cd"
source activate $ENV_NAME

# ── GPU info ──────────────────────────────────────────────────
echo ""
echo "GPU Info:"
nvidia-smi
echo ""

# ── Navigate to project ───────────────────────────────────────
cd /home/katoch.aa/ondemand/vit-from-scratch-main

# ── Paths ─────────────────────────────────────────────────────
PROJECT_DIR="/home/katoch.aa/ondemand/vit-from-scratch-main"
OSCD_IMAGES_DIR="$PROJECT_DIR/images"              # Sentinel-2 region folders
OSCD_LABELS_DIR="$PROJECT_DIR/train_labels"        # change masks (cm/cm.png per region)
PREPROCESSED_DIR="$PROJECT_DIR/oscd_preprocessed"  # intermediate .npy files
PATCHES_DIR="$PROJECT_DIR/processed_oscd"          # final 256×256 PNG patches
OUTPUT_DIR="$PROJECT_DIR/outputs/siamese_vit"      # checkpoints + logs

# ── Create output dirs ────────────────────────────────────────
mkdir -p logs
mkdir -p $OUTPUT_DIR

# ─────────────────────────────────────────────────────────────
# STEP 1: Preprocess raw .tif → .npy
#   Skip if already done (preprocessed dir exists and is non-empty)
# ─────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "STEP 1: Preprocessing raw OSCD .tif files"
echo "========================================"

if [ -d "$PREPROCESSED_DIR" ] && [ "$(ls -A $PREPROCESSED_DIR 2>/dev/null)" ]; then
    echo "Preprocessed data already exists at $PREPROCESSED_DIR — skipping."
else
    echo "Running preprocess_oscd.py ..."
    python scripts/preprocess_oscd.py \
        --images_dir  "$OSCD_IMAGES_DIR" \
        --labels_dir  "$OSCD_LABELS_DIR" \
        --output_dir  "$PREPROCESSED_DIR"

    if [ $? -ne 0 ]; then
        echo "ERROR: Preprocessing failed. Exiting."
        exit 1
    fi
    echo "Preprocessing done."
fi

# ─────────────────────────────────────────────────────────────
# STEP 2: Extract 256×256 patches → PNG pairs
#   Skip if already done
# ─────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "STEP 2: Extracting 256×256 patches"
echo "========================================"

if [ -d "$PATCHES_DIR/train/A" ] && [ "$(ls -A $PATCHES_DIR/train/A 2>/dev/null)" ]; then
    echo "Patches already exist at $PATCHES_DIR — skipping."
else
    echo "Running extract_patches.py ..."
    python scripts/extract_patches.py \
        --input_dir  "$PREPROCESSED_DIR" \
        --output_dir "$PATCHES_DIR" \
        --patch_size 256 \
        --stride     128

    if [ $? -ne 0 ]; then
        echo "ERROR: Patch extraction failed. Exiting."
        exit 1
    fi
    echo "Patch extraction done."
fi

# Print patch counts
echo ""
echo "Patch counts:"
for split in train val test; do
    count=$(ls $PATCHES_DIR/$split/A/*.png 2>/dev/null | wc -l)
    echo "  $split: $count patches"
done

# ─────────────────────────────────────────────────────────────
# STEP 3: Train
# ─────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "STEP 3: Training Siamese ViT"
echo "========================================"

# Check for existing checkpoint to resume from
RESUME_CKPT="$OUTPUT_DIR/last_model.pth"
RESUME_FLAG=""
if [ -f "$RESUME_CKPT" ]; then
    echo "Found checkpoint at $RESUME_CKPT — resuming."
    RESUME_FLAG="--resume $RESUME_CKPT"
fi

python train.py \
    --data_dir       "$PATCHES_DIR" \
    --output_dir     "$OUTPUT_DIR" \
    --epochs         200 \
    --batch_size     8 \
    --eval_batch_size 16 \
    --lr             1e-3 \
    --weight_decay   0.05 \
    --warmup_epochs  10 \
    --min_lr         1e-6 \
    --grad_clip      1.0 \
    --patience       30 \
    --num_workers    $SLURM_CPUS_PER_TASK \
    --loss           bce_dice \
    --pos_weight     5.0 \
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
