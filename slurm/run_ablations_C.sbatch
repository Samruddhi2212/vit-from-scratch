#!/bin/bash
#SBATCH --job-name=vit-abl-C
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablations-C-%j.out
#SBATCH --error=logs/ablations-C-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=patodia.pa@northeastern.edu
#
# Part 3 of 3 — patch_size=2 only (256 patches; attention is much slower).
# If this hits TIMEOUT, re-submit with:  --only patch_2 --ablation-epochs 30

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1
source activate pytorch_env

python scripts/run_ablations.py \
  --ablation-epochs 50 \
  --num-workers 4 \
  --only patch_2
