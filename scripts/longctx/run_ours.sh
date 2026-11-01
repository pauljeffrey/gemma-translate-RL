#!/bin/bash
# sequential, nice 19, one process at a time; retry on failure (e.g. GPU OOM because of other jobs)
cd /workspace/jobs/lc1; source env_lc.sh
retry() { for t in 1 2 3 4 5 6; do "$@" && return 0; echo "retry $t after failure" ; sleep 120; done; return 1; }
declare -A DIR=( [sft]=/workspace/jobs/a2/models/sft [rl]=/workspace/sft_build/runs/final_1433 [a2w]=/workspace/jobs/r7/models/sci_sft_A2W_alpha0.7 [live]=/dev/shm/lc1_models/live )
for m in "$@"; do
  retry nice -n 19 $PYV lc_gen.py $m ${DIR[$m]} --gpu-util ${GPU_UTIL:-0.07} >> logs/gen_$m.log 2>&1
done
echo ours_done > logs/OURS_DONE_$(echo "$@" | tr ' ' '_')
