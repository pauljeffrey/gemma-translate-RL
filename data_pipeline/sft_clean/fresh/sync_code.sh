#!/bin/bash
# Run on the LAPTOP: copy the chain code and the gemma_mt package to a fresh instance. No data, no .env, no git, no benchmark text.
#   SSH_TARGET=root@HOST SSH_PORT=PORT SSH_KEY=$SSH_KEY ./sync_code.sh
set -euo pipefail
: "${SSH_TARGET:?set SSH_TARGET=root@host}"; SSH_PORT=${SSH_PORT:-22}; SSH_KEY=${SSH_KEY:-$SSH_KEY}; ROOT=${ROOT:-/workspace/work}
REPO=${REPO:-"/path/to/repo"}
SSH="ssh -i $SSH_KEY -p $SSH_PORT -o StrictHostKeyChecking=accept-new"
$SSH "$SSH_TARGET" "mkdir -p $ROOT/code $ROOT/state $ROOT/logs"
EXC=(--exclude __pycache__ --exclude '.DS_Store' --exclude '*.pyc' --exclude '.env' --exclude '.git' --exclude 'env.sh')
rsync -a "${EXC[@]}" -e "$SSH" "$REPO/data_pipeline/sft_clean/" "$SSH_TARGET:$ROOT/code/sft_clean/"
rsync -a "${EXC[@]}" -e "$SSH" "$REPO/gemma_mt/gemma_mt" "$SSH_TARGET:$ROOT/code/"          # package gemma_mt (prepare_sft, packed_dataset, format, languages)
rsync -a -e "$SSH" "$REPO/reports/data/sft_clean_2026-10-28/heldout_split_cal_sel_v2.json" "$SSH_TARGET:$ROOT/code/"
echo "synced to $SSH_TARGET:$ROOT/code  (sft_clean/, gemma_mt/, heldout_split_cal_sel_v2.json)"
