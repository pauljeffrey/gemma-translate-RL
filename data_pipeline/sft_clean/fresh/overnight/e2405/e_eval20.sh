#!/bin/bash
# usage: e_eval20.sh NAME MODEL_DIR [GPU_UTIL] [full]
# The standard harness (eval_sft_benchmarks.sh: vLLM 0.32.0 bf16, greedy, 768 new tokens, FLORES+ devtest + NTREX + MAFAND, held-out AfriCOMET-STL-1.1, no decode-time guard) on the 20 languages of the
# E-02 decoding study (yor hau ibo swh amh zul xho kin som wol lug sna twi ewe lin tir orm nso tsn afr), or on all languages when the 4th argument is `full` (E-14); then per-sentence statistics (ev_stats.py).
# Same code state and flags as the E-03 / E-01 evaluations (only --languages differs). Outputs: /workspace/eval_385m/out_NAME, /dev/shm/evid_e2405/NAME.npz
N=$1; M=$2; U=${3:-0.5}; MODE=${4:-20}
E=/workspace/eval_385m; S=/workspace/science_sft
mkdir -p /dev/shm/tmp /dev/shm/xdg_cache /dev/shm/vllm_cache_ov /dev/shm/evid_e2405
export TMPDIR=/dev/shm/tmp XDG_CACHE_HOME=/dev/shm/xdg_cache
[ "$(df --output=avail -k / | tail -1)" -lt 1048576 ] && { echo "LOW DISK (< 1 GB free on /): skipped"; exit 3; }
cd $E
eval "$(grep "^export HF_TOKEN=" /workspace/clean_sft/env.sh)"
export HF_HOME=/workspace/clean_sft/hf_home GEMMA_MT_TAG_MODE=single_token
export ROOT=$E PY=/workspace/clean_sft/venv_train/bin/python VLLM_PY=/venv/main/bin/python COMET_PY=$E/venv_comet/bin/python COMET_PYTHONPATH=$E/pylibs_none
export OUT=$E/out_$N BENCH_DIR=$E/bench GPU_UTIL=$U GPU_MIN_FREE_GIB=6 EXTRA_LIBS=$E/pylibs VLLM_CACHE=/workspace/eval_385m/vllm_cache_guard
export BENCHMARKS=flores_plus,ntrex,mafand TOTAL_TIMEOUT=14400 SAMPLER_FALLBACK=0
LANGS=""; [ "$MODE" = full ] || LANGS="--languages yor hau ibo swh amh zul xho kin som wol lug sna twi ewe lin tir orm nso tsn afr"
[ -d $OUT ] && [ ! -f $OUT/EVAL_DONE ] && mv $OUT $OUT.partial_$(date +%s)
for attempt in 1 2 3 4; do
  bash gemma_mt_tree/scripts/eval_sft_benchmarks.sh $M --allow-missing-contamination --skip-base --max-new-tokens 768 $LANGS
  [ -f $OUT/EVAL_DONE ] && break
  echo "attempt $attempt did not finish (no EVAL_DONE); retrying in 180 s"; sleep 180      # vLLM KV-budget collision with another engine: wait and retry
done
[ -f $OUT/EVAL_DONE ] || exit 1
nice -n 19 $E/venv_comet/bin/python $S/ev_stats.py $N $OUT/predictions.jsonl /dev/shm/evid_e2405/$N.npz $OUT/work/comet_cache/comet_ref.jsonl > /dev/shm/evid_e2405/$N.log 2>&1
echo EVAL_E2405_DONE $N
