#!/bin/bash
#SBATCH --job-name=vit-abl-A
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablations-A-%j.out
#SBATCH --error=logs/ablations-A-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=patodia.pa@northeastern.edu
#
# Part 1 of 3 — submitted by scripts/submit_ablations_chain.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1
source activate pytorch_env

python scripts/run_ablations.py \
  --ablation-epochs 50 \
  --num-workers 4 \
  --only baseline,no_scaling,no_position,gap,no_residual,post_norm
