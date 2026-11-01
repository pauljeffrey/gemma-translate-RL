#!/bin/bash
# RL supervisor (owner away): the 49k-token micro-batches OOM'd at step 14 every time (deterministic schedule): walk down the micro-batch ladder after an OOM, restart after any other crash.
cd /workspace/sft_build
source env.sh
export PYTHONPATH=/workspace/sft_build/gemma_mt PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv_gemma/bin/python; RLDIR=runs/rl
export VLLM_CACHE=/workspace/sft_build/vllm_cache
export VLLM_CACHE_ROOT=$VLLM_CACHE/vllm TRITON_CACHE_DIR=$VLLM_CACHE/triton TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE/inductor XDG_CACHE_HOME=$VLLM_CACHE/xdg VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
LADDER=("24576 128" "12288 64" "6144 32" "3072 16")
IDX=0; STARTS=()
say() { echo "$(date -u +%FT%TZ) [rl_sup] $*" | tee -a logs/rental_run.log; }
rm -f logs/RENTAL_RL_GAVE_UP
while [ ! -f $RLDIR/summary.json ]; do
  now=$(date +%s); keep=(); for t in "${STARTS[@]}"; do [ $((now - t)) -lt 3600 ] && keep+=($t); done; STARTS=("${keep[@]}")
  if [ ${#STARTS[@]} -ge 8 ]; then say "8 starts in the last hour: giving up"; touch logs/RENTAL_RL_GAVE_UP; exit 1; fi
  STARTS+=($now)
  read T S < <(echo "${LADDER[$IDX]}")
  say "RL start: micro_batch_tokens=$T micro_batch_seqs=$S"
  cp logs/rl_rtxpro.log logs/rl_rtxpro_prev_$(date +%s).log 2>/dev/null
  $PY -m gemma_mt.rl.train_rl --config gemma_mt/configs/rl_rtxpro.yaml micro_batch_tokens=$T micro_batch_seqs=$S > logs/rl_rtxpro.log 2>&1
  rc=$?
  [ -f $RLDIR/summary.json ] && { say "RL finished (rc=$rc)"; break; }
  if tail -n 80 logs/rl_rtxpro.log | grep -a -q "OutOfMemoryError"; then [ $IDX -lt $((${#LADDER[@]}-1)) ] && IDX=$((IDX+1)); say "OOM -> ladder $IDX"; else say "exited rc=$rc: restarting"; fi
  sleep 30
done
touch $RLDIR/RL_DONE; say "ALL DONE"
