#!/bin/bash
# usage: run_plain_eval.sh NAME MODEL_DIR  — the 385M harness, greedy, 768 new tokens, FLORES+ devtest / NTREX / MAFAND + AfriCOMET, single-token tag mode (E-20: B1 final), OUT=out_NAME
# disk rules (coordinator 2026-11-04): root disk is nearly full -> caches, temp and any download on /dev/shm; skip this step if < 1 GB free
mkdir -p /dev/shm/tmp /dev/shm/xdg_cache /dev/shm/vllm_cache_ov
export TMPDIR=/dev/shm/tmp XDG_CACHE_HOME=/dev/shm/xdg_cache
[ "$(df --output=avail -k / | tail -1)" -lt 1048576 ] && { echo "LOW DISK (< 1 GB free on /): step skipped"; exit 3; }
N=$1; M=$2; E=/workspace/eval_385m
cd $E
eval "$(grep "^export HF_TOKEN=" /workspace/clean_sft/env.sh)"
export HF_HOME=/workspace/clean_sft/hf_home GEMMA_MT_TAG_MODE=single_token
export ROOT=$E PY=/workspace/clean_sft/venv_train/bin/python VLLM_PY=/venv/main/bin/python COMET_PY=$E/venv_comet/bin/python COMET_PYTHONPATH=$E/pylibs_none
export OUT=$E/out_$N BENCH_DIR=$E/bench GPU_UTIL=0.25 GPU_MIN_FREE_GIB=6 EXTRA_LIBS=$E/pylibs VLLM_CACHE=/workspace/eval_385m/vllm_cache_guard
export BENCHMARKS=flores_plus,ntrex,mafand TOTAL_TIMEOUT=10800
export SAMPLER_FALLBACK=0   # no silent HF fallback (it ignores the decode-time guard and is 5x slower): retry the vLLM start instead
for attempt in 1 2 3 4; do
  bash gemma_mt_tree/scripts/eval_sft_benchmarks.sh $M --allow-missing-contamination --skip-base --max-new-tokens 768
  [ -f $OUT/EVAL_DONE ] && break
  echo "attempt $attempt did not finish (no EVAL_DONE); retrying in 180 s"; sleep 180
done
[ -f $OUT/EVAL_DONE ]
