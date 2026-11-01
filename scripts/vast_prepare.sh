#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Build the packed SFT dataset from raw parquet/jsonl (columns english, target, iso).
#   RAW=/workspace/raw_sft OUT=/workspace/packed bash scripts/vast_prepare.sh
# Downloads the tokenizer (needs HF_TOKEN if the repo is gated). Runs on CPU; ~minutes per million pairs.
# Data of any size < 150 GB may be downloaded to the instance first (never to a laptop).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

RAW="${RAW:?set RAW to the raw data file or directory}"
OUT="${OUT:-data/packed}"
SEQ_LEN="${SEQ_LEN:-2048}"
VAL_FRACTION="${VAL_FRACTION:-0.01}"

python -m gemma_mt.data.prepare_sft \
  --input "$RAW" --out "$OUT" \
  --tokenizer "${TOKENIZER:-google/gemma-3-270m-it}" \
  --seq-len "$SEQ_LEN" --val-fraction "$VAL_FRACTION" \
  --min-val-per-language "${MIN_VAL_PER_LANGUAGE:-50}" --seed "${SEED:-0}"
echo "wrote $OUT (meta.json has counts, per-language numbers and packing efficiency)"
