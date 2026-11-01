#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Evaluate one system on FLORES+ / NTREX-128 / MAFAND-MT; call once per system, then build the report.
#   MODEL=runs/rl/best NAME=rl OUT=results/rl FLORES=/workspace/flores_plus NTREX=/workspace/NTREX-128 \
#     bash scripts/vast_eval.sh
#   MODEL=google/gemma-3-270m-it NAME=base OUT=results/base bash scripts/vast_eval.sh      # the untouched base model
# FLORES+ is gated: accept the terms on the Hub and export HF_TOKEN. Leave FLORES / NTREX unset to use the Hub
# (FLORES+) or skip the benchmark (NTREX needs a local clone: git clone https://github.com/MicrosoftTranslator/NTREX).
# Report:
#   python -m gemma_mt.eval.make_report --run base=results/base/results.json --run sft=results/sft/results.json \
#       --run rl=results/rl/results.json --baseline-csv published.csv --reference base --out report.md
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
MODEL="${MODEL:?set MODEL}"
NAME="${NAME:?set NAME}"
OUT="${OUT:-results/$NAME}"

ARGS=(--model "$MODEL" --name "$NAME" --out "$OUT" --batch-size "${BATCH_SIZE:-32}" --max-new-tokens "${MAX_NEW_TOKENS:-256}")
ARGS+=(--bench "flores_plus${FLORES:+:$FLORES}")
[ -n "${NTREX:-}" ] && ARGS+=(--bench "ntrex:$NTREX")
ARGS+=(--bench "mafand${MAFAND:+:$MAFAND}")
[ "${NO_COMET:-0}" = "1" ] && ARGS+=(--no-comet)
python -m gemma_mt.eval.eval_translation "${ARGS[@]}"
