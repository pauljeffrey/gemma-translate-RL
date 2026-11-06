#!/bin/bash
# usage: run_guard_eval_ext.sh NAME MODEL_DIR FREQ_PENALTY REP_PENALTY — POST-HOC extension of E-01 (stronger penalty than the pre-registered grid's edge); OUT=out_guardx_NAME
N=$1; M=$2; FP=$3; RP=$4
mkdir -p /dev/shm/tmp /dev/shm/xdg_cache /dev/shm/vllm_cache_ov; export TMPDIR=/dev/shm/tmp XDG_CACHE_HOME=/dev/shm/xdg_cache
[ "$(df --output=avail -k / | tail -1)" -lt 1048576 ] && { echo "LOW DISK: skipped"; exit 3; }
E=/workspace/eval_385m; cd $E
eval "$(grep "^export HF_TOKEN=" /workspace/clean_sft/env.sh)"
export HF_HOME=/workspace/clean_sft/hf_home GEMMA_MT_TAG_MODE=single_token
export ROOT=$E PY=/workspace/clean_sft/venv_train/bin/python VLLM_PY=/venv/main/bin/python COMET_PY=$E/venv_comet/bin/python COMET_PYTHONPATH=$E/pylibs_none
export OUT=/dev/shm/out_guardx_$N BENCH_DIR=$E/bench GPU_UTIL=0.15 GPU_MIN_FREE_GIB=6 EXTRA_LIBS=$E/pylibs VLLM_CACHE=/workspace/eval_385m/vllm_cache_guard
export BENCHMARKS=flores_plus,ntrex,mafand TOTAL_TIMEOUT=10800
export GEMMA_MT_EVAL_FREQ_PENALTY=$FP GEMMA_MT_EVAL_REP_PENALTY=$RP SAMPLER_FALLBACK=0
mkdir -p $OUT; echo "{\"frequency_penalty\": $FP, \"repetition_penalty\": $RP, \"model\": \"$M\", \"note\": \"post-hoc extension\"}" > $OUT/GUARD.json
for attempt in 1 2 3 4; do
  bash gemma_mt_tree/scripts/eval_sft_benchmarks.sh $M --allow-missing-contamination --skip-base --max-new-tokens 768 --comet-sequential --comet-batch-size 16
  [ -f $OUT/EVAL_DONE ] && break
  echo "attempt $attempt did not finish; retrying in 180 s"; sleep 180
done
[ -f $OUT/EVAL_DONE ]
