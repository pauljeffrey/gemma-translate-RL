#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Stage 1: SFT. Single GPU by default; NGPU=2 uses torchrun (DDP).
#   DATA=/workspace/packed OUT=/workspace/runs/sft bash scripts/vast_train_sft.sh [key=value ...]
# Examples:
#   bash scripts/vast_train_sft.sh micro_batch_size=4 grad_accum_steps=16        # 11 GB GPU
#   bash scripts/vast_train_sft.sh attn_implementation=flash_attention_2         # Ampere+, flash-attn installed
# Resumes automatically from $OUT/last if it exists. Best checkpoint: $OUT/best, last: $OUT/final.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
DATA="${DATA:-data/packed}"
OUT="${OUT:-runs/sft}"
NGPU="${NGPU:-1}"

if [ "$NGPU" -gt 1 ]; then
  RUN=(torchrun --nproc_per_node="$NGPU" -m gemma_mt.train_sft)
else
  RUN=(python -m gemma_mt.train_sft)
fi
"${RUN[@]}" --config configs/sft.yaml data_dir="\"$DATA\"" out_dir="\"$OUT\"" "$@"
