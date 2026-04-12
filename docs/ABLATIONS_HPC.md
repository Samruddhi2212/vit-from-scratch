# Running CIFAR ablations on HPC (Explorer)

Outputs go to **`outputs/ablations/`** by default:

- **`all_ablation_results.pt`** — merged metrics dict (this path is allowed in `.gitignore` for commits)
- **`ablation_bar_chart.png`**, **`all_ablations_curves.png`**
- Per-run checkpoints: `*_best.pt` (still ignored by `*.pt`; keep on disk or use `git add -f` if you must track one)

## One-time setup (login node)

```bash
ssh YOUR_NETID@login.explorer.northeastern.edu
cd ~/vit-from-scratch   # or your path
git pull

# Use the same venv as other GPU jobs (create if missing)
python3 -m venv .venv_hpc
source .venv_hpc/bin/activate
pip install -r requirements.txt

mkdir -p logs outputs/ablations
```

Edit **`slurm/run_ablations_*.sbatch`** if your project path is not `~/vit-from-scratch` (set `PROJ_DIR` or change the default).

## Recommended: three chained jobs (8 h each, fits Explorer limit)

From the repo root:

```bash
chmod +x scripts/submit_ablations_chain.sh   # once
bash scripts/submit_ablations_chain.sh
```

This submits **A → B → C**; each job **merges** into `outputs/ablations/all_ablation_results.pt`.

Watch logs:

```bash
tail -f logs/ablations_A_*.out    # use the job id from squeue
```

## Fresh start (no merge from old partial file)

If a broken `all_ablation_results.pt` exists on the cluster, remove it before job A:

```bash
rm -f outputs/ablations/all_ablation_results.pt
```

Or run job A with:

```bash
python scripts/run_ablations.py --no-merge --only baseline --ablation-epochs 50
```

(then continue with the chain or full script as needed).

## Single interactive test (GPU node / `salloc`)

```bash
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00
# on compute node:
source .venv_hpc/bin/activate
cd ~/vit-from-scratch
python scripts/run_ablations.py --ablation-epochs 2 --only baseline --num-workers 4
```

## Copy results to your laptop

```bash
# From your Mac:
scp -r YOUR_NETID@login.explorer.northeastern.edu:~/vit-from-scratch/outputs/ablations/ \
  ./outputs/
```

## Commit the merged results

```bash
git add outputs/ablations/all_ablation_results.pt outputs/ablations/*.png
git commit -m "Add CIFAR ablation merged results and plots"
```

Per-experiment `*_best.pt` files remain ignored unless you force-add them (large).
