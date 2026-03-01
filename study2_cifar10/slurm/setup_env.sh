#!/usr/bin/env bash
# Setup Python venv with ROCm PyTorch for AMD MI210
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Setting up venv at $PROJECT_DIR/.venv ==="
python3 -m venv "$PROJECT_DIR/.venv"
source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Installing PyTorch (ROCm 6.0) ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

echo "=== Installing dependencies ==="
pip install POT torchdiffeq torch-fidelity pyyaml seaborn scipy tqdm

echo "=== Done ==="
echo "Activate with: source $PROJECT_DIR/.venv/bin/activate"
