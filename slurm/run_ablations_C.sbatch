#!/bin/bash
#================================================================
# CIFAR-10 ViT ablations — Job C (patch_2 only; many patches → slower)
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=abl_C
#SBATCH --output=logs/ablations_C_%j.out
#SBATCH --error=logs/ablations_C_%j.err

set -euo pipefail

module purge
module load cuda/12.3.0
module load cuDNN/9.10.2
module load anaconda3/2024.06

PROJ_DIR="${PROJ_DIR:-$HOME/vit-from-scratch}"
source "${PROJ_DIR}/.venv_hpc/bin/activate"
export PYTHONNOUSERSITE=1

cd "$PROJ_DIR"
mkdir -p logs outputs/ablations

echo "=== Ablation C — $(date) — $(hostname) — Job $SLURM_JOB_ID ==="
nvidia-smi

python scripts/run_ablations.py \
  --ablation-epochs 50 \
  --num-workers 4 \
  --only patch_2

echo "=== Ablation C done ==="
