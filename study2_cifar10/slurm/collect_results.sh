#!/usr/bin/env bash
# Collect and archive results from all 8 cells
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== Collecting results ==="

COMPLETE=0
MISSING=""
for cell in 000 001 010 011 100 101 110 111; do
    if [ -d "results/cell_${cell}/seed_0" ]; then
        echo "  cell_${cell}: OK"
        COMPLETE=$((COMPLETE + 1))
    else
        echo "  cell_${cell}: MISSING"
        MISSING="$MISSING $cell"
    fi
done

echo ""
echo "$COMPLETE/8 cells completed"

if [ -n "$MISSING" ]; then
    echo "Missing cells:$MISSING"
fi

# Archive everything
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE="all_results_${TIMESTAMP}.tar.gz"
tar -czf "$ARCHIVE" results/ outputs/
echo "Archive: $PROJECT_DIR/$ARCHIVE"

echo "=== Done ==="
