#!/bin/bash
# Science SFT waiter, version 2 (owner 2026-11-03): chain = weak run -> weak evaluation (inference) -> science SFT -> science RL. Starts when the weak run is finished, no trainer runs and the
# weak-evaluation inference is done (marker WEAK_EVAL_INFER_DONE). Fallbacks (the GPU must not idle): 90 min after the weak run ended without the marker; weak run never started within 20 min after the mixed stage ended.
R=/workspace/clean_rerun; S=/workspace/science_sft; LOG=$S/waiter.log; W=$R/runs_385m_weak/mixed/summary.json; M=$R/runs_385m/mixed/summary.json
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
say "science SFT waiter v2 started"
trainer() { pgrep -f "[g]emma_mt.rl.train_rl" >/dev/null; }
mfin=0; wfin=0
while true; do
  if [ -f $W ] && ! trainer; then
    [ $wfin -eq 0 ] && wfin=$(date +%s)
    if [ -f $S/WEAK_EVAL_INFER_DONE ]; then say "weak run finished, evaluation inference done: go"; break; fi
    if [ $(( $(date +%s) - wfin )) -gt 5400 ]; then say "no inference marker 90 min after the weak run: go"; break; fi
  fi
  if [ -f $M ] && ! trainer && [ ! -f $W ]; then
    [ $mfin -eq 0 ] && mfin=$(date +%s)
    if [ $(( $(date +%s) - mfin )) -gt 1200 ] && ! pgrep -f "[w]eak_waiter.sh" >/dev/null; then say "weak run never started: go"; break; fi
  fi
  sleep 20
done
bash $S/science_sft_pipeline.sh >> $S/waiter.log 2>&1
