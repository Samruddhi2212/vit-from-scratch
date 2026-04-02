#!/usr/bin/env bash
# Submit three dependent GPU jobs so the full ablation sweep fits Explorer's ~8h cap.
# Run once from the repository root (login node):  bash scripts/submit_ablations_chain.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

J1=$(sbatch --parsable slurm/run_ablations_A.sbatch)
J2=$(sbatch --parsable --dependency=afterok:"${J1}" slurm/run_ablations_B.sbatch)
J3=$(sbatch --parsable --dependency=afterok:"${J2}" slurm/run_ablations_C.sbatch)

echo "Submitted ablation chain (each job loads prior results and appends):"
echo "  Job A ${J1}  — core + block variants (6 × 50 ep)"
echo "  Job B ${J2}  — heads + patch 8/16 (5 × 50 ep), after A OK"
echo "  Job C ${J3}  — patch size 2 only (slow), after B OK"
echo "Final merged file: outputs/ablations/all_ablation_results.pt"
