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
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@northeastern.edu
#
# Replace --mail-user with your real Northeastern email. Slurm sends mail only;
# this does not affect the training process or GPU allocation.
#
# Submit from the repository root:
#   mkdir -p logs && sbatch slurm/train_cifar10.sbatch
#
# Explorer notes:
# - Batch jobs use partition "gpu", not "gpu-interactive" (short walltime cap).
# - Standard "gpu" jobs are typically 1 GPU; requesting 2+ often fails unless you
#   have multigpu / reservation access: https://rc-docs.northeastern.edu/en/explorer-main/gpus/multigpu-partition-access.html
# - If you ARE allocated 2 GPUs on one node, add --data-parallel to the python line below.
# - To request 2 GPUs (only if your account allows), try e.g.:
#     #SBATCH --gres=gpu:v100-pcie:2
#   and raise --cpus-per-task / --mem as needed.
# - GPU types: https://rc.northeastern.edu/compute/
# - Walltime:  sinfo -o "%P %l"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1
source activate pytorch_env

python scripts/train_cifar10.py \
  --epochs 200 \
  --num-workers 4 \
  --experiment-name vit_cifar10
