#!/bin/bash
# E-03: concurrent evaluation was aborted by the speed guard (training +55%); evaluate AFTER training with the whole GPU, final checkpoint first. Never pushes anything.
W=/dev/shm/control; HERE=/dev/shm/control_prep
log(){ echo "$(date -u +%H:%M:%S) $*" | tee -a $W/logs/auto_eval.log; }
while [ ! -f $W/TRAINING_DONE ]; do sleep 60; done
log "training done"
for N in c1_s6000 c1_s4800 c1_s3100; do
  [ -f $W/export/$N/EXPORT_DONE ] || { log "no export $N"; continue; }
  grep -q "EVAL_C1_DONE $N" $W/logs/eval_$N.log 2>/dev/null && continue
  for d in /workspace/eval_385m/out_$N /workspace/eval_385m/out_afsci_$N; do [ -d $d ] && mv $d $d.partial_$(date +%s); done
  log "evaluating $N (util 0.5)"
  bash $HERE/eval_c1.sh $N $W/export/$N 0.5 > $W/logs/eval_$N.log 2>&1
  grep -q "EVAL_C1_DONE $N" $W/logs/eval_$N.log && log "eval $N done" || log "eval $N FAILED"
done
touch $W/ALL_EVALS_DONE; log ALL_EVALS_DONE
