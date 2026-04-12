#!/bin/bash
#================================================================
# CIFAR-10 ViT ablations — Job B (loads & appends to all_ablation_results.pt)
# heads_1, heads_2, heads_8, patch_8, patch_16
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=abl_B
#SBATCH --output=logs/ablations_B_%j.out
#SBATCH --error=logs/ablations_B_%j.err

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

echo "=== Ablation B — $(date) — $(hostname) — Job $SLURM_JOB_ID ==="
nvidia-smi

python scripts/run_ablations.py \
  --ablation-epochs 50 \
  --num-workers 4 \
  --only heads_1,heads_2,heads_8,patch_8,patch_16

echo "=== Ablation B done ==="
