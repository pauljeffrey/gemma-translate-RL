#!/bin/bash
# usage: run_guard_eval.sh NAME MODEL_DIR   — the harness (generation + AfriCOMET) with the chosen decode-time guard (guard_dev.json 'chosen'), OUT=/workspace/eval_385m/out_guard_NAME
N=$1; M=$2; E=/workspace/eval_385m; O=/workspace/science_sft/overnight
read FP RP < <(python3 -c "import json;c=json.load(open('$O/guard_dev.json'))['chosen'];print(c['frequency_penalty'],c['repetition_penalty'])")
cd $E
eval "$(grep "^export HF_TOKEN=" /workspace/clean_sft/env.sh)"
export HF_HOME=/workspace/clean_sft/hf_home GEMMA_MT_TAG_MODE=single_token
export ROOT=$E PY=/workspace/clean_sft/venv_train/bin/python VLLM_PY=/venv/main/bin/python COMET_PY=$E/venv_comet/bin/python COMET_PYTHONPATH=$E/pylibs_none
export OUT=$E/out_guard_$N BENCH_DIR=$E/bench GPU_UTIL=0.25 GPU_MIN_FREE_GIB=6 EXTRA_LIBS=$E/pylibs VLLM_CACHE=$E/vllm_cache_guard
export BENCHMARKS=flores_plus,ntrex,mafand TOTAL_TIMEOUT=10800
export GEMMA_MT_EVAL_FREQ_PENALTY=$FP GEMMA_MT_EVAL_REP_PENALTY=$RP
mkdir -p $OUT; echo "{\"frequency_penalty\": $FP, \"repetition_penalty\": $RP, \"model\": \"$M\"}" > $OUT/GUARD.json
export SAMPLER_FALLBACK=0   # no silent HF fallback (it ignores the decode-time guard and is 5x slower): retry the vLLM start instead
for attempt in 1 2 3 4; do
  bash gemma_mt_tree/scripts/eval_sft_benchmarks.sh $M --allow-missing-contamination --skip-base --max-new-tokens 768
  [ -f $OUT/EVAL_DONE ] && break
  echo "attempt $attempt did not finish (no EVAL_DONE); retrying in 180 s"; sleep 180
done
[ -f $OUT/EVAL_DONE ]
