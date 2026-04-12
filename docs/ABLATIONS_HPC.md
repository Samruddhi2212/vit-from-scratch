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

# Same CUDA modules as Slurm — needed so PyTorch can find libnvJitLink.so.12 etc. on login nodes
module load cuda/12.3.0
module load cuDNN/9.10.2

# Required for CIFAR loaders (utils/dataset.py); confirms venv matches requirements.txt
python scripts/verify_pytorch_stack.py

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

## If job A fails in seconds (ExitCode `1:0`)

1. Read the Slurm stderr (this is where Python tracebacks go):

   ```bash
   cat logs/ablations_A_*.err
   ```

2. **`libnvJitLink.so.12: cannot open shared object file`** — PyTorch’s CUDA build needs the toolkit on `LD_LIBRARY_PATH`. On Explorer, load the same modules as the Slurm scripts **before** `python`:

   ```bash
   module load cuda/12.3.0
   module load cuDNN/9.10.2
   source .venv_hpc/bin/activate
   python scripts/verify_pytorch_stack.py
   ```

   Batch jobs already run `module load` inside the `.sbatch` file; this mainly affects **interactive** checks on the login node.

3. **`ModuleNotFoundError: No module named 'torchvision'`** — `requirements.txt` already pins `torchvision`, but the cluster venv may have been created with only `torch`. On the **login node**:

   ```bash
   source .venv_hpc/bin/activate
   cd ~/vit-from-scratch
   pip install -r requirements.txt
   module load cuda/12.3.0
   module load cuDNN/9.10.2
   python scripts/verify_pytorch_stack.py
   ```

   Then resubmit the chain.

4. Confirm the repo path matches **`slurm/run_swin.sh`** (default `PROJ_DIR=/home/patodia.pa/vit-from-scratch`). Override if your clone lives elsewhere:

   ```bash
   export PROJ_DIR=/home/patodia.pa/vit-from-scratch
   bash scripts/submit_ablations_chain.sh
   ```

5. Batch scripts use **`--num-workers 0`** to avoid fork/CUDA + DataLoader issues on Slurm. If training is stable, you may raise to `2` or `4` in the `.sbatch` files.

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
