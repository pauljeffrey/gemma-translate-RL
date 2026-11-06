#!/bin/bash
# After the weak run (summary + no trainer): run the weak-group evaluation (inference first, scoring may overlap the science SFT). HF_TOKEN not needed.
R=/workspace/clean_rerun; S=/workspace/science_sft; LOG=$S/weak_eval.log; W=$R/runs_385m_weak/mixed/summary.json
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
say "weak-eval waiter started"
while [ ! -f $W ] || pgrep -f "[g]emma_mt.rl.train_rl" >/dev/null; do sleep 15; done
say "weak run finished and no trainer: weak evaluation starts"
bash $S/weak_eval_infer.sh >> $LOG 2>&1
