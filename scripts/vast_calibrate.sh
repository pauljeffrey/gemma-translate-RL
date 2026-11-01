#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Per-language reward calibration (run ONCE before RL, on the DEV split of FLORES+).
#   OUT=calibration/shaping.json MODEL=runs/sft/best bash scripts/vast_calibrate.sh
# Scores up to MAX_PER_LANGUAGE (default 997 = all FLORES+ dev sentences) human references per language with AfriCOMET-QE and writes anchors, AUC, the
# length statistics, the diacritics block and the usable / reliability flags that gemma_mt.rl.reward_shaping reads. Never calibrate on the split you report on.
# D-32 (2026-10-17): the 42-language file calibration/shaping_flores_dev_v4.json was made with
#   python scripts/build_flores_dev_all.py --out calibration/flores_dev_all_v2.jsonl                 # FLORES+ DEV rows (+ MAFAND dev for pcm / diacritic inventories)
#   BENCH_FILE=calibration/flores_dev_all_v2.jsonl FAIL_FLOOR=0.0 GPU_MEM_GB=6 OUT=... bash scripts/vast_calibrate.sh
#   python -m gemma_mt.rl.omission_eval --data calibration/flores_dev_all_v2.jsonl --calibrate-omission OUT --languages <the FLORES+ languages>   # CPU
# (comet 2.2.7 needs transformers < 5: put a transformers 4.x on PYTHONPATH first, e.g. `uv pip install --target pylibs_tf4 transformers==4.57.1 tokenizers==0.22.1 huggingface_hub==0.36.2`.)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT="${OUT:-calibration/shaping.json}"
if [ -n "${BENCH_FILE:-}" ]; then
  ARGS=(--bench-file "$BENCH_FILE" --out "$OUT" --max-per-language "${MAX_PER_LANGUAGE:-997}" --fail-floor "${FAIL_FLOOR:-0.0}" --resume)
else
  ARGS=(--bench "flores_plus${FLORES:+:$FLORES}" --split dev --out "$OUT" --max-per-language "${MAX_PER_LANGUAGE:-997}" --fail-floor "${FAIL_FLOOR:-0.0}")
fi
[ -n "${GPU_MEM_GB:-}" ] && ARGS+=(--gpu-mem-gb "$GPU_MEM_GB")
[ -n "${MODEL:-}" ] && ARGS+=(--model "$MODEL")
python -m gemma_mt.rl.calibrate_shaping "${ARGS[@]}"
