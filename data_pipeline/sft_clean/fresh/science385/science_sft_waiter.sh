#!/bin/bash
# Queue behind the weak West-African group run: starts the science SFT pipeline the moment the GPU is free. HF_TOKEN inherited from the environment only. Fallback: if the mixed stage is
# finished for 20 min and the weak run never started (no summary, no trainer), start anyway so the GPU does not idle.
R=/workspace/clean_rerun; S=/workspace/science_sft; LOG=$S/waiter.log; W=$R/runs_385m_weak/mixed/summary.json; M=$R/runs_385m/mixed/summary.json
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
say "science SFT waiter started"
trainer() { pgrep -f "[g]emma_mt.rl.train_rl" >/dev/null; }
mfin=0
while true; do
  if [ -f $W ] && ! trainer; then say "weak run finished and no trainer: go"; break; fi
  if [ -f $M ] && ! trainer && [ ! -f $W ]; then
    [ $mfin -eq 0 ] && mfin=$(date +%s)
    if [ $(( $(date +%s) - mfin )) -gt 1200 ] && ! pgrep -f "[w]eak_waiter.sh" >/dev/null; then say "weak run never started: go"; break; fi
  fi
  sleep 20
done
bash $S/science_sft_pipeline.sh >> $S/waiter.log 2>&1
