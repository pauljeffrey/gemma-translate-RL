#!/bin/bash
# usage: run_gen.sh <logname> name [name ...]
cd /workspace/jobs/r12; export GPU_UTIL=${GPU_UTIL:-0.12} OMP_NUM_THREADS=4
V=/workspace/sft_build/vllm_cache; export VLLM_CACHE_ROOT=$V/vllm TRITON_CACHE_DIR=$V/triton TORCHINDUCTOR_CACHE_DIR=$V/inductor XDG_CACHE_HOME=$V/xdg VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONDONTWRITEBYTECODE=1
source /workspace/sft_build/env.sh
log=logs/$1.log; shift
for t in 1 2 3 4 5 6; do
  nice -n 10 /workspace/venv_vllm/bin/python gen_sent.py "$@" >> $log 2>&1 && break
  echo "attempt $t failed" >> $log; sleep 30
done
echo done >> $log
