#!/bin/bash
# usage: run_job.sh NAME JOBS UTIL
cd /workspace/jobs/r11
source env.sh
export VLLM_CACHE_ROOT=/workspace/jobs/r11/work/vllm_cache TRITON_CACHE_DIR=/workspace/jobs/r11/work/vllm_cache/triton TORCHINDUCTOR_CACHE_DIR=/workspace/jobs/r11/work/vllm_cache/inductor XDG_CACHE_HOME=/workspace/jobs/r11/work/vllm_cache/xdg HF_HOME=/workspace/sft_build/hf
for i in 1 2 3 4; do
  nice -n 10 $PYV code/scripts/mix2_gen_vllm.py --req work/$1/req_sent.jsonl --out work/$1/gen --jobs $2 --gpu-util $3 --max-seqs 160 --chunk 10000 --max-model-len 4096 && break
  sleep 20
done
echo JOB_DONE $1 $2
