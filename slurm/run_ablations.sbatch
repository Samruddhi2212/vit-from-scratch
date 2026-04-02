#!/bin/bash
#SBATCH --job-name=vit-ablations
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablations-%j.out
#SBATCH --error=logs/ablations-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=patodia.pa@northeastern.edu
#
# Replace --mail-user with your real address (same idea as train_cifar10.sbatch).
# Email is handled by Slurm only; it does not change training or GPU usage.
#
# Explorer gpu partition often caps walltime (e.g. 8h). If sbatch rejects --time,
# run:  sinfo -o "%P %l"  and set --time below that max for "gpu".
# Ablations may not finish in one job; chunk with e.g.:
#   python scripts/run_ablations.py --only baseline,gap --ablation-epochs 50
#
# Explorer: full sweep usually exceeds one 8h job. Prefer one command on the login node:
#   bash scripts/submit_ablations_chain.sh
# That submits A -> B -> C (Slurm afterok); each step merges all_ablation_results.pt.
#
# Monolithic (single long GPU job — may TIMEOUT):
#   mkdir -p logs && sbatch slurm/run_ablations.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1
source activate pytorch_env

python scripts/run_ablations.py \
  --ablation-epochs 50 \
  --num-workers 4
