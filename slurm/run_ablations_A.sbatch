#!/bin/bash
#================================================================
# CIFAR-10 ViT ablations — Job A of 3 (merges into outputs/ablations/)
# baseline, no_scaling, no_position, no_residual, post_norm, gap
#
# Must match the same PROJ_DIR / venv pattern as slurm/run_swin.sh.
# If job fails in <1 min, check: logs/ablations_A_<JOBID>.err
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=abl_A
#SBATCH --output=logs/ablations_A_%j.out
#SBATCH --error=logs/ablations_A_%j.err

set -eo pipefail

module purge
module load cuda/12.3.0
module load cuDNN/9.10.2
module load anaconda3/2024.06

# Same default as slurm/run_swin.sh (override: export PROJ_DIR=... before sbatch)
PROJ_DIR="${PROJ_DIR:-/home/patodia.pa/vit-from-scratch}"

if [[ ! -d "$PROJ_DIR" ]]; then
  echo "ERROR: PROJ_DIR does not exist: $PROJ_DIR"
  echo "Fix: export PROJ_DIR=/path/to/vit-from-scratch"
  exit 1
fi
if [[ ! -f "$PROJ_DIR/.venv_hpc/bin/activate" ]]; then
  echo "ERROR: venv not found: $PROJ_DIR/.venv_hpc/bin/activate"
  exit 1
fi

cd "$PROJ_DIR"
mkdir -p logs outputs/ablations

source "$PROJ_DIR/.venv_hpc/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== Ablation A — $(date) — $(hostname) — Job ${SLURM_JOB_ID:-local} ==="
echo "PROJ_DIR=$PROJ_DIR  PWD=$(pwd)"
which python
python -V

nvidia-smi
python scripts/verify_pytorch_stack.py

# num-workers 0 avoids fork/CUDA DataLoader issues on many Slurm nodes; increase to 2–4 if stable.
python scripts/run_ablations.py \
  --ablation-epochs 50 \
  --num-workers 0 \
  --only baseline,no_scaling,no_position,no_residual,post_norm,gap

echo "=== Ablation A done ==="
