#!/bin/bash
# E-03 control evaluation of one exported checkpoint with the SAME harness as the RL models. usage: eval_c1.sh NAME MODEL_DIR [GPU_UTIL]
# FLORES+/NTREX/MAFAND greedy 768 tokens vLLM bf16 + held-out AfriCOMET (eval_sft_benchmarks.sh), per-sentence stats (ev_stats.py), AfriScience-MT dev/test sentence level.
N=$1; M=$2; U=${3:-0.15}
E=/workspace/eval_385m; S=/workspace/science_sft
mkdir -p /dev/shm/tmp /dev/shm/xdg_cache /dev/shm/vllm_cache_ov /dev/shm/evid_c1
export TMPDIR=/dev/shm/tmp XDG_CACHE_HOME=/dev/shm/xdg_cache
[ "$(df --output=avail -k / | tail -1)" -lt 1048576 ] && { echo "LOW DISK (< 1 GB free on /): skipped"; exit 3; }
cd $E
eval "$(grep "^export HF_TOKEN=" /workspace/clean_sft/env.sh)"
export HF_HOME=/workspace/clean_sft/hf_home GEMMA_MT_TAG_MODE=single_token
export ROOT=$E PY=/workspace/clean_sft/venv_train/bin/python VLLM_PY=/venv/main/bin/python COMET_PY=$E/venv_comet/bin/python COMET_PYTHONPATH=$E/pylibs_none
export OUT=$E/out_$N BENCH_DIR=$E/bench GPU_UTIL=$U GPU_MIN_FREE_GIB=6 EXTRA_LIBS=$E/pylibs VLLM_CACHE=/workspace/eval_385m/vllm_cache_guard
export BENCHMARKS=flores_plus,ntrex,mafand TOTAL_TIMEOUT=14400 SAMPLER_FALLBACK=0
for attempt in 1 2 3 4; do
  bash gemma_mt_tree/scripts/eval_sft_benchmarks.sh $M --allow-missing-contamination --skip-base --max-new-tokens 768
  [ -f $OUT/EVAL_DONE ] && break
  echo "attempt $attempt did not finish (no EVAL_DONE); retrying in 180 s"; sleep 180
done
[ -f $OUT/EVAL_DONE ] || exit 1
nice -n 19 $E/venv_comet/bin/python $S/ev_stats.py $N $OUT/predictions.jsonl /dev/shm/evid_c1/$N.npz $OUT/work/comet_cache/comet_ref.jsonl > /dev/shm/evid_c1/$N.log 2>&1
# AfriScience-MT sentence level (dev, test), plain chrF
cd $E/gemma_mt_tree
export PYTHONPATH=$E/gemma_mt_tree VLLM_CACHE_ROOT=$E/vllm_cache_af
$E/venv_comet/bin/python scripts/eval_afriscience.py --model $M --tokenizer $M --name $N --splits dev,test --levels sentence --vllm-python /venv/main/bin/python --vllm-gpu-util 0.12 --extra-libs $E/pylibs --out $E/out_afsci_$N > $E/afsci_$N.log 2>&1
echo EVAL_C1_DONE $N
