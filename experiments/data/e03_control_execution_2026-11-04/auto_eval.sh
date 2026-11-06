#!/bin/bash
# E-03: evaluate the exported checkpoints next to training (util 0.15, one engine at a time) with a speed guard: abort an eval when the median training s/step of the last intervals exceeds 25% above the baseline
# (5.10 s/step -> 6.4 s); aborted checkpoints are evaluated after TRAINING_DONE with the whole GPU. Never pushes anything.
W=/dev/shm/control; M=$W/runs/c1/metrics.jsonl; HERE=/dev/shm/control_prep; LIMIT=6.4
log(){ echo "$(date -u +%H:%M:%S) $*" | tee -a $W/logs/auto_eval.log; }
med() { python3 - <<PY
import json,statistics
rs=[json.loads(l) for l in open("$M") if '"train_loss"' in l]
iv=[(b["time"]-a["time"])/(b["step"]-a["step"]) for a,b in zip(rs[-8:-1],rs[-7:]) if b["step"]>a["step"]]
print(round(statistics.median(iv),2) if iv else 0)
PY
}
run_eval() {  # name util ; returns 0 done, 1 aborted by the speed guard, 2 failed
  local N=$1 U=$2
  for d in /workspace/eval_385m/out_$N /workspace/eval_385m/out_afsci_$N; do [ -d $d ] && mv $d $d.partial_$(date +%s); done
  setsid bash $HERE/eval_c1.sh $N $W/export/$N $U > $W/logs/eval_$N.log 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 45
    if [ ! -f $W/TRAINING_DONE ]; then
      local s=$(med); 
      if python3 -c "import sys;sys.exit(0 if float('$s')>$LIMIT else 1)"; then
        log "SPEED GUARD: training median $s s/step > $LIMIT while evaluating $N: aborting the eval"
        kill -TERM -- -$pid 2>/dev/null; sleep 5; pkill -f "eval_385m/out_$N" 2>/dev/null; pkill -f "name $N " 2>/dev/null; kill -KILL -- -$pid 2>/dev/null
        echo "$N" >> $W/eval_pending; return 1
      fi
    fi
  done
  wait $pid; local rc=$?
  grep -q "EVAL_C1_DONE $N" $W/logs/eval_$N.log && return 0 || return 2
}
for N in c1_s3100 c1_s4800; do
  while [ ! -f $W/export/$N/EXPORT_DONE ]; do sleep 60; [ -f $W/TRAINING_DONE ] && break; done
  [ -f $W/export/$N/EXPORT_DONE ] || continue
  base=$(med); log "export $N present; training median $base s/step; starting eval at util 0.15"
  run_eval $N 0.15; rc=$?; log "eval $N rc=$rc"
done
while [ ! -f $W/TRAINING_DONE ]; do sleep 60; done
log "training done; pending: $(cat $W/eval_pending 2>/dev/null | tr '\n' ' ')"
for N in $(cat $W/eval_pending 2>/dev/null) c1_s6000; do
  [ -f $W/export/$N/EXPORT_DONE ] || { log "no export $N"; continue; }
  grep -q "EVAL_C1_DONE $N" $W/logs/eval_$N.log 2>/dev/null && continue
  log "evaluating $N after training with util 0.4"; run_eval $N 0.4; log "eval $N rc=$?"
done
touch $W/ALL_EVALS_DONE; log ALL_EVALS_DONE
