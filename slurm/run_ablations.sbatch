#!/bin/bash
#SBATCH --job-name=vit-ablations
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ablations-%j.out
#SBATCH --error=logs/ablations-%j.err
#
# Full ablation sweep needs many hours. If 24h exceeds your qos/partition
# limit, lower --time or run chunks:  python scripts/run_ablations.py --only baseline,gap
#
# Submit from repository root:
#   mkdir -p logs && sbatch slurm/run_ablations.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1
source activate pytorch_env

python scripts/run_ablations.py \
  --ablation-epochs 50 \
  --num-workers 4
