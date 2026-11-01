#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Driver of the short science-SFT stage (D-58) on the rental box. One phase per call; every phase is restartable.
#
#   bash scripts/run_sci_sft_stage.sh baseline      # evaluate the starting model (final_1433) on the three suites           ~10 min GPU (vLLM 15 GB)
#   bash scripts/run_sci_sft_stage.sh train         # the SFT run (configs/sft_science.yaml), OOM fallback to micro-batch 1   ~20-70 min GPU
#   bash scripts/run_sci_sft_stage.sh sweep         # alpha in {0.3,0.5,0.7,1.0}: interpolate, evaluate, delete            ~10 min per alpha
#   bash scripts/run_sci_sft_stage.sh report        # before/after tables -> $WORK/report_tables.md
#   bash scripts/run_sci_sft_stage.sh finalize A    # write the chosen alpha as an HF model dir (float32) and evaluate it on dev + TEST (once)
#
# Nothing is pushed anywhere.  The GPU is shared: the script refuses to start a phase when less than MIN_FREE_GIB are free (default 30 for train, 18 for eval)
# unless FORCE=1, and everything runs under `nice -n 10` with 6 CPU threads.
set -uo pipefail
WORK="${WORK:-/workspace/jobs/r4}"
TREE="${TREE:-$WORK/gemma_mt_sft}"
INIT="${INIT:-/workspace/sft_build/runs/final_1433}"
CFG="${CFG:-$TREE/configs/sft_science.yaml}"
RUN="${RUN:-$WORK/runs/sci_sft}"
EVAL="${EVAL:-$WORK/eval}"
ALPHAS="${ALPHAS:-0.3 0.5 0.7 1.0}"
source "$WORK/env.sh"
cd "$TREE"
mkdir -p "$WORK/logs" "$EVAL"

need_gpu() {   # $1 = GiB
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  free=$((free / 1024))
  if [ "${FORCE:-0}" != "1" ] && [ "$free" -lt "$1" ]; then echo "only $free GiB of GPU memory free, need $1 (FORCE=1 to override)"; exit 3; fi
}

case "${1:-}" in
  baseline)
    need_gpu "${MIN_FREE_GIB:-18}"
    nice -n 10 $PY scripts/sft_stage_eval.py run --model "$INIT" --name final_1433 --out "$EVAL/final_1433" 2>&1 | tee "$WORK/logs/eval_baseline.log"
    ;;
  train)
    need_gpu "${MIN_FREE_GIB:-30}"
    nice -n 10 $PY -m gemma_mt.train_sft --config "$CFG" "${@:2}" 2>&1 | tee -a "$WORK/logs/sci_sft.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "[run_sci_sft_stage] train exited with $rc: retrying once with micro_batch_size=1 grad_accum_steps=8 gradient_checkpointing=true (same 65,536 tokens per step; resumes from the last checkpoint)"
      nice -n 10 $PY -m gemma_mt.train_sft --config "$CFG" micro_batch_size=1 grad_accum_steps=8 gradient_checkpointing=true "${@:2}" 2>&1 | tee -a "$WORK/logs/sci_sft.log"
    fi
    ;;
  sweep)
    need_gpu "${MIN_FREE_GIB:-18}"
    nice -n 10 $PY scripts/sft_stage_eval.py sweep --init "$INIT" --sft "$RUN/last/trainer_state.pt" --alphas $ALPHAS --out "$EVAL" --work "$WORK/interp_tmp" 2>&1 | tee -a "$WORK/logs/eval_sweep.log"
    ;;
  report)
    runs=""; for a in $ALPHAS; do d=$(printf "%s/alpha_%.2f" "$EVAL" "$a"); [ -f "$d/stage_eval.json" ] && runs="$runs $d"; done
    $PY scripts/sft_stage_eval.py compare --baseline "$EVAL/final_1433" --runs $runs --out "$WORK/report_tables.md" > /dev/null && echo "wrote $WORK/report_tables.md"
    ;;
  finalize)
    A="${2:?usage: finalize ALPHA}"
    out="$WORK/models/sci_sft_alpha_$A"
    $PY scripts/interpolate_weights.py --init "$INIT" --sft "$RUN/last/trainer_state.pt" --alpha "$A" --out "$out"
    need_gpu "${MIN_FREE_GIB:-18}"
    nice -n 10 $PY scripts/sft_stage_eval.py run --model "$out" --name "sci_sft_alpha_$A" --out "$EVAL/final_alpha_$A" --splits dev,test --suites afs 2>&1 | tee "$WORK/logs/eval_final.log"
    echo "model dir: $out"
    ;;
  *) sed -n 2,14p "$0"; exit 2;;
esac
