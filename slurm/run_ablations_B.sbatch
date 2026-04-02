#!/bin/bash
#SBATCH --job-name=vit-abl-B
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablations-B-%j.out
#SBATCH --error=logs/ablations-B-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=patodia.pa@northeastern.edu
#
# Part 2 of 3 — merges outputs/ablations/all_ablation_results.pt from part A

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1
source activate pytorch_env

python scripts/run_ablations.py \
  --ablation-epochs 50 \
  --num-workers 4 \
  --only heads_1,heads_2,heads_8,patch_8,patch_16
