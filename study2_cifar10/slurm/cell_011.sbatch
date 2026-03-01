#!/usr/bin/env bash
#SBATCH --job-name=fm_cell_011
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hermes-2
#SBATCH --output=logs/cell_011/%x-%j.out
#SBATCH --error=logs/cell_011/%x-%j.err
#SBATCH --requeue
#SBATCH --time=06:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source .venv/bin/activate
mkdir -p logs/cell_011

echo "=== Training cell_011 (seed 0) ==="
python train.py --config configs/cell_011.yaml --seed 0 --fast

echo "=== Evaluating cell_011 ==="
python evaluate.py \
    --checkpoint outputs/cell_011/seed_0/checkpoints/final.pt \
    --output_dir results/cell_011/seed_0 \
    --dataset cifar10

echo "=== Done cell_011 ==="
