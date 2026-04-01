#!/bin/bash
#SBATCH --job-name=vit-cifar10
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
#
# Submit from the repository root (directory containing scripts/ and configs/):
#   mkdir -p logs
#   sbatch slurm/train_cifar10.sbatch
#
# Edit --partition, --gres, --time to match your Explorer allocation and queue.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1
source activate pytorch_env

python scripts/train_cifar10.py \
  --epochs 200 \
  --num-workers 4 \
  --experiment-name vit_cifar10
