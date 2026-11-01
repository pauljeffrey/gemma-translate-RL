#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# AB2 evaluation of ONE model on yor kin orm lin: FLORES+ devtest, NTREX, MAFAND, SMOL-sent (greedy, 768 new tokens, vLLM at 10% of the card), then per-sentence surface metrics (chrF++, finished,
# GlotLID) and the neural metrics AfriCOMET-ref / -QE and SSA-COMET-ref / -QE (the E1 pipeline: pairwise_surface.py / pairwise_neural.py, copied unchanged next to this script).
#
#   bash scripts/ab2_eval.sh MODEL_DIR NAME [GPU_UTIL=0.1]      ->  $AB/eval/NAME/{gen/, sys.jsonl.gz, neural/*.jsonl, EVAL_DONE}
#   STAGES="gen surface neural" (default) selects the stages; each stage skips itself when its output exists (resumable).
# Next to a running trainer this needs ~10 GB (vLLM 0.1 of the card) + ~5 GB (COMET); it waits (GPU_WAIT s) until that much is free. Nothing here is pushed anywhere.
set -uo pipefail
AB="${AB:-/workspace/jobs/ab2}"
TREE="${TREE:-$AB/gemma_mt}"
PY="${PY:-/workspace/venv_gemma/bin/python}"
LANGS="${LANGS:-yor kin orm lin}"
STAGES="${STAGES:-gen surface neural}"
MODEL="${1:?model dir}"; NAME="${2:?name}"; UTIL="${3:-0.1}"
OUT="$AB/eval/$NAME"; mkdir -p "$OUT" "$AB/logs"
LOG="$AB/logs/eval_$NAME.log"
say() { echo "$(date -u +%FT%TZ) [ab2-eval $NAME] $*" | tee -a "$LOG"; }
[ -f /workspace/sft_build/env.sh ] && source /workspace/sft_build/env.sh >/dev/null 2>&1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
rm -f "$OUT/EVAL_FAILED"
if [[ " $STAGES " == *" gen "* ]] && [ ! -f "$OUT/gen/EVAL_DONE" ]; then
  say "generation: $MODEL ($LANGS)"
  export CONTAM_DIR=/workspace/sft_build/bench/test_sets/_contamination BENCHMARKS=flores_plus,ntrex,mafand,smol_sent A1_JOB=1
  OUT="$OUT/gen" GPU_UTIL="$UTIL" GPU_MIN_FREE_GIB="${GPU_MIN_FREE_GIB:-14}" GPU_WAIT="${GPU_WAIT:-3600}" TOTAL_TIMEOUT=5400 ROOT=/workspace/sft_build nice -n 10 \
    bash "$TREE/scripts/eval_sft_benchmarks.sh" "$MODEL" --skip-base --no-comet --max-new-tokens 768 --languages $LANGS >> "$LOG" 2>&1
  [ -f "$OUT/gen/EVAL_DONE" ] || { say "generation FAILED"; touch "$OUT/EVAL_FAILED"; exit 3; }
fi
if [[ " $STAGES " == *" surface "* ]] && [ ! -f "$OUT/sys.jsonl.gz" ]; then
  say "surface metrics"
  PYTHONPATH="$TREE" nice -n 10 "$PY" "$TREE/scripts/pairwise_surface.py" "$OUT/gen/predictions.jsonl" "$OUT/sys.jsonl.gz.tmp" --benchmarks flores_plus,ntrex,mafand,smol_sent --procs 3 >> "$LOG" 2>&1 \
    && mv "$OUT/sys.jsonl.gz.tmp" "$OUT/sys.jsonl.gz" || { say "surface FAILED"; touch "$OUT/EVAL_FAILED"; exit 4; }
fi
if [[ " $STAGES " == *" neural "* ]] && [ ! -f "$OUT/neural/NEURAL_DONE" ]; then
  say "neural metrics (AfriCOMET ref/QE, SSA-COMET ref/QE)"
  # the E1 scorer wants pre-filtered language rows only for speed
  PYTHONPATH="$TREE:/workspace/sft_build/pylibs_tf4:/workspace/sft_build/pylibs" HF_HOME=/workspace/sft_build/hf nice -n 10 "$PY" "$TREE/scripts/ab2_neural.py" --sys "s=$OUT/sys.jsonl.gz" \
    --out "$OUT/neural" --cap 0 --gpu-gib "${NEURAL_GPU_GIB:-5}" --metrics afri_ref,afri_qe,ssa_ref,ssa_qe >> "$LOG" 2>&1 \
    && touch "$OUT/neural/NEURAL_DONE" || { say "neural FAILED"; touch "$OUT/EVAL_FAILED"; exit 5; }
fi
touch "$OUT/EVAL_DONE"; say "done"
