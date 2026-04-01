#!/bin/bash
#SBATCH --job-name=vit-cifar10
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
#
# Submit from the repository root:
#   mkdir -p logs && sbatch slurm/train_cifar10.sbatch
#
# Explorer: batch jobs belong on partition "gpu", not "gpu-interactive"
# (interactive is capped at ~1–2h and is for srun/OOD).
# GPU types: https://rc.northeastern.edu/compute/ — edit --gres if v100-pcie
# is unavailable (e.g. gpu:p100:1 or gpu:v100-sxm2:1 per cluster listing).
# If time limit still fails:  sinfo -o "%P %l"  and stay under Max for "gpu".

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1
source activate pytorch_env

python scripts/train_cifar10.py \
  --epochs 200 \
  --num-workers 4 \
  --experiment-name vit_cifar10
