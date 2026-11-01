#!/bin/bash
# usage: waitrun.sh <script.py> <logname> ; PLAN_JSON etc. from env. Starts when >= 32 GiB are free (twice, 20 s apart); retries a failed start (<= 10 tries). SFT keeps priority.
cd /workspace/jobs/r15/probe; source /workspace/datagen/code/scripts/env_dg.sh; export HF_HUB_OFFLINE=1 KV_GIB=${KV_GIB:-6.5} UTIL=${UTIL:-0.27}
for try in 1 2 3 4 5 6 7 8 9 10; do
  ok=0
  while :; do
    free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | awk -F, '{print int(($1-$2)/1024)}')
    if [ "$free" -ge 32 ]; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 2 ] && break
    sleep 20
  done
  echo "$(date -u +%FT%TZ) [$2] try $try starting $1, free ${free} GiB" >> waitstart.log
  nice -n 10 $PYVLLM $1 > $2.log 2>&1; rc=$?
  echo "$(date -u +%FT%TZ) [$2] $1 exited rc=$rc" >> waitstart.log
  grep -aq "ALL DONE" $2.log && break
  [ -f /workspace/jobs/r15/STOP_R15 ] && break
  sleep 30
done
