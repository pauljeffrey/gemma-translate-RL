#!/bin/bash
# RL phase 2c supervisor on the new box: $1 = config file under gemma_mt_2c/configs, $2 = log tag. Same OOM ladder as before; resumes from runs/rl2c/last.
cd /workspace/sft_build; source env.sh
export PYTHONPATH=/workspace/sft_build/gemma_mt_2c PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv_gemma/bin/python; RLDIR=runs/rl2c; CFG=$1; TAG=$2
export VLLM_CACHE=/workspace/sft_build/vllm_cache
export VLLM_CACHE_ROOT=$VLLM_CACHE/vllm TRITON_CACHE_DIR=$VLLM_CACHE/triton TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE/inductor XDG_CACHE_HOME=$VLLM_CACHE/xdg VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
LADDER=("24576 128" "12288 64" "6144 32" "3072 16")
say() { echo "$(date -u +%FT%TZ) [sup-$TAG] $*" | tee -a logs/rental_run.log; }
rm -f $RLDIR/summary.json
IDX=0; STARTS=()
while [ ! -f $RLDIR/summary.json ]; do
  now=$(date +%s); keep=(); for t in "${STARTS[@]}"; do [ $((now - t)) -lt 3600 ] && keep+=($t); done; STARTS=("${keep[@]}")
  if [ ${#STARTS[@]} -ge 8 ]; then say "8 starts in the last hour: giving up"; touch logs/RL_GAVE_UP_$TAG; exit 1; fi
  STARTS+=($now); read T S < <(echo "${LADDER[$IDX]}")
  say "RL start ($CFG): micro_batch_tokens=$T micro_batch_seqs=$S"
  cp logs/rl_$TAG.log logs/rl_${TAG}_prev_$(date +%s).log 2>/dev/null
  $PY -m gemma_mt.rl.train_rl --config gemma_mt_2c/configs/$CFG micro_batch_tokens=$T micro_batch_seqs=$S > logs/rl_$TAG.log 2>&1
  rc=$?
  [ -f $RLDIR/summary.json ] && { say "RL finished (rc=$rc)"; break; }
  if tail -n 80 logs/rl_$TAG.log | grep -a -q "OutOfMemoryError"; then [ $IDX -lt $((${#LADDER[@]}-1)) ] && IDX=$((IDX+1)); say "OOM -> ladder $IDX"; else say "exited rc=$rc: restarting"; fi
  sleep 20
done
touch $RLDIR/RL_DONE_$TAG; say "DONE $TAG"
