#!/usr/bin/env bash
# After: source .../.venv_hpc/bin/activate
# Installs torchvision into the project venv if absent (partial pip installs often skip it).
# Uses the same Python as the job. If pip fails (no network on compute), install on login:
#   pip install -r requirements.txt
set -e
if python -c "import torchvision" 2>/dev/null; then
  exit 0
fi
echo "torchvision not found in venv ($(command -v python)); installing..."
python -m pip install -q 'torchvision>=0.15.0' || {
  echo "ERROR: pip install torchvision failed (no network on compute node?)."
  echo "  On login node: source .venv_hpc/bin/activate && pip install -r requirements.txt"
  exit 1
}
