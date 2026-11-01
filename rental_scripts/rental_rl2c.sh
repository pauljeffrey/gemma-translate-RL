#!/bin/bash
# Phase 2c of the RL (owner 2026-10-20): from the phase-2b stage-b64 model (branch stage-b64 of the rl2b repo), b256 onwards, KL guard on.
cd /workspace/sft_build
source env.sh
export PYTHONPATH=/workspace/sft_build/gemma_mt_2c PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv_gemma/bin/python; RLDIR=runs/rl2c
export VLLM_CACHE=/workspace/sft_build/vllm_cache
export VLLM_CACHE_ROOT=$VLLM_CACHE/vllm TRITON_CACHE_DIR=$VLLM_CACHE/triton TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE/inductor XDG_CACHE_HOME=$VLLM_CACHE/xdg VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
LADDER=("24576 128" "12288 64" "6144 32" "3072 16")
say() { echo "$(date -u +%FT%TZ) [rl2c_sup] $*" | tee -a logs/rental_run.log; }
say "fetching the phase-2b stage-b64 model"
OK=0
for try in $(seq 1 20); do
  rm -rf runs/rl_2b_stage_b64
  $PY - <<'PYEOF' 2>&1 | tail -1 | tee -a logs/rental_run.log
import os
from huggingface_hub import snapshot_download
p = snapshot_download("${HF_ORG}/gemma-3-270m-translate-rl2b", revision="stage-b64", local_dir="/workspace/sft_build/runs/rl_2b_stage_b64", token=os.environ["HF_TOKEN"])
print("2b stage-b64 model downloaded:", p)
PYEOF
  [ -f runs/rl_2b_stage_b64/model.safetensors ] && { OK=1; break; }
  say "download failed (try $try/20)"; sleep 30
done
[ $OK = 1 ] || { say "download failed: phase 2c not started"; touch logs/RENTAL_RL2C_GAVE_UP; exit 1; }
IDX=0; STARTS=()
while [ ! -f $RLDIR/summary.json ]; do
  now=$(date +%s); keep=(); for t in "${STARTS[@]}"; do [ $((now - t)) -lt 3600 ] && keep+=($t); done; STARTS=("${keep[@]}")
  if [ ${#STARTS[@]} -ge 8 ]; then say "8 starts in the last hour: giving up"; touch logs/RENTAL_RL2C_GAVE_UP; exit 1; fi
  STARTS+=($now); read T S < <(echo "${LADDER[$IDX]}")
  say "RL phase 2c start: micro_batch_tokens=$T micro_batch_seqs=$S"
  cp logs/rl2c_rtxpro.log logs/rl2c_rtxpro_prev_$(date +%s).log 2>/dev/null
  $PY -m gemma_mt.rl.train_rl --config gemma_mt_2c/configs/rl_rtxpro_phase2c.yaml micro_batch_tokens=$T micro_batch_seqs=$S > logs/rl2c_rtxpro.log 2>&1
  rc=$?
  [ -f $RLDIR/summary.json ] && { say "RL phase 2c finished (rc=$rc)"; break; }
  if tail -n 80 logs/rl2c_rtxpro.log | grep -a -q "OutOfMemoryError"; then [ $IDX -lt $((${#LADDER[@]}-1)) ] && IDX=$((IDX+1)); say "OOM -> ladder $IDX"; else say "exited rc=$rc: restarting"; fi
  sleep 30
done
touch $RLDIR/RL_DONE; say "PHASE 2c DONE"
