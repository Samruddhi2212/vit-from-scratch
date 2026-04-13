#!/usr/bin/env bash
# Pull CIFAR ablation outputs from Northeastern Explorer into this repo (run on your laptop).
# Requires SSH access to the cluster (same as `ssh patodia.pa@login.explorer.northeastern.edu`).
set -euo pipefail

REMOTE_USER="${HPC_USER:-patodia.pa}"
REMOTE_HOST="${HPC_HOST:-login.explorer.northeastern.edu}"
REMOTE_PROJ="${HPC_PROJ:-~/vit-from-scratch}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/outputs/ablations"
mkdir -p "${DEST}"

echo "Syncing ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJ}/outputs/ablations/ -> ${DEST}/"
rsync -avz --progress "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJ}/outputs/ablations/" "${DEST}/"

echo "Done. Next:"
echo "  cd ${ROOT}"
echo "  git add outputs/ablations/all_ablation_results.pt outputs/ablations/*.png"
echo "  git status"
echo "  git commit -m \"Add CIFAR ablation merged results and plots\""
echo "  git push origin main"
