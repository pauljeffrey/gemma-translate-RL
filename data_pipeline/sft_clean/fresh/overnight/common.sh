#!/bin/bash
# Shared helpers of the overnight chain (2026-11-04). Source this file. One human-readable log: $LOGF (mirrored into reports/overnight_log_2026-11-04.md by hand).
O=/workspace/science_sft/overnight; E=/workspace/eval_385m; R=/workspace/clean_rerun; S=/workspace/science_sft
mkdir -p $O
LOGF=$O/overnight.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOGF; }
# step NAME TIMEOUT_SECONDS CMD...  : runs CMD with a timeout, never aborts the chain; writes $O/STEP_NAME.{DONE,FAILED}
step() {
  local name=$1 to=$2; shift 2
  rm -f $O/STEP_$name.DONE $O/STEP_$name.FAILED
  say "STEP $name start"
  local t0=$(date +%s)
  timeout --signal=TERM --kill-after=60 $to "$@" > $O/step_$name.out 2>&1
  local rc=$? dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then touch $O/STEP_$name.DONE; say "STEP $name DONE in ${dt}s"; else touch $O/STEP_$name.FAILED; say "STEP $name FAILED rc=$rc after ${dt}s (see $O/step_$name.out); chain continues"; fi
  return 0
}
gpu_busy_other() { pgrep -f "[g]emma_mt.rl.train_rl|[g]emma_mt.train_sft" >/dev/null; }
