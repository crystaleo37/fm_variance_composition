#!/usr/bin/env bash
# Launch all 8 ablation cells in parallel on SLURM
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Create log dirs
for cell in 000 001 010 011 100 101 110 111; do
    mkdir -p "logs/cell_${cell}"
done

echo "=== Submitting 8 ablation cells ==="

JOB_IDS=""
for cell in 000 001 010 011 100 101 110 111; do
    JOB_ID=$(sbatch --parsable "slurm/cell_${cell}.sbatch")
    echo "  cell_${cell} -> job $JOB_ID"
    if [ -z "$JOB_IDS" ]; then
        JOB_IDS="$JOB_ID"
    else
        JOB_IDS="$JOB_IDS:$JOB_ID"
    fi
done

echo ""
echo "=== All jobs submitted ==="
echo "Job IDs: $JOB_IDS"
echo ""

# Submit collection job that runs after all training is done
COLLECT_ID=$(sbatch --parsable --dependency=afterok:${JOB_IDS} \
    --job-name=fm_collect \
    --partition=hermes-2 \
    --nodes=1 --cpus-per-task=2 --time=00:10:00 \
    --output=logs/collect-%j.out \
    --error=logs/collect-%j.err \
    --wrap="bash $SCRIPT_DIR/collect_results.sh")

echo "Collect job -> $COLLECT_ID (runs after all 8 complete)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f logs/cell_*/fm_cell_*-*.out"
