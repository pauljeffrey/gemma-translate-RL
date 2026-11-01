#!/bin/sh
# Copy the live final-run metrics from the GPU box (read-only: scp only reads), then rebuild fig3/fig9 and the final2 table.
# Usage: sh paper/scripts/refresh_final2.sh          (needs $SSH_KEY; box address as in HOW_TO_RUN.md)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
scp -o BatchMode=yes -i "$SSH_KEY" -P $BOX_PORT root@$BOX_HOST:/workspace/jobs/r7/runs/final2/metrics.jsonl "$HERE/../data/final2_metrics.jsonl"
wc -l "$HERE/../data/final2_metrics.jsonl"
cd "$HERE" && ${PYTHON:-python3} make_figures.py fig3 fig9 && ${PYTHON:-python3} make_tables.py
