#!/bin/bash
# Phase 2 of the RL (owner 2026-10-19): starts when phase 1 is finished (runs/rl/RL_DONE) from the final mixed-stage model, ~380k completions, uniform language draws.
cd /workspace/sft_build
source env.sh
export PYTHONPATH=/workspace/sft_build/gemma_mt PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv_gemma/bin/python; RLDIR=runs/rl2
export VLLM_CACHE=/workspace/sft_build/vllm_cache
export VLLM_CACHE_ROOT=$VLLM_CACHE/vllm TRITON_CACHE_DIR=$VLLM_CACHE/triton TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE/inductor XDG_CACHE_HOME=$VLLM_CACHE/xdg VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
LADDER=("24576 128" "12288 64" "6144 32" "3072 16")
say() { echo "$(date -u +%FT%TZ) [rl2_sup] $*" | tee -a logs/rental_run.log; }
say "armed: waiting for runs/rl/RL_DONE"
while [ ! -f runs/rl/RL_DONE ]; do sleep 30; if [ -f logs/RENTAL_RL_GAVE_UP ]; then say "phase 1 gave up: phase 2 not started"; exit 1; fi; done
say "phase 1 finished: fetching its final (mixed-stage) model"
rm -rf runs/rl_phase1_final
$PY - <<'PYEOF' 2>&1 | tail -2 | tee -a logs/rental_run.log
import os
from huggingface_hub import snapshot_download
p = snapshot_download("${HF_ORG}/gemma-3-270m-translate-rl", revision="stage-mixed", local_dir="/workspace/sft_build/runs/rl_phase1_final", token=os.environ["HF_TOKEN"])
print("phase-1 final model downloaded:", p)
PYEOF
[ -f runs/rl_phase1_final/model.safetensors ] || { say "download failed: phase 2 not started"; exit 1; }
IDX=0; STARTS=()
while [ ! -f $RLDIR/summary.json ]; do
  now=$(date +%s); keep=(); for t in "${STARTS[@]}"; do [ $((now - t)) -lt 3600 ] && keep+=($t); done; STARTS=("${keep[@]}")
  if [ ${#STARTS[@]} -ge 8 ]; then say "8 starts in the last hour: giving up"; touch logs/RENTAL_RL2_GAVE_UP; exit 1; fi
  STARTS+=($now); read T S < <(echo "${LADDER[$IDX]}")
  say "RL phase 2 start: micro_batch_tokens=$T micro_batch_seqs=$S"
  cp logs/rl2_rtxpro.log logs/rl2_rtxpro_prev_$(date +%s).log 2>/dev/null
  $PY -m gemma_mt.rl.train_rl --config gemma_mt/configs/rl_rtxpro_phase2.yaml micro_batch_tokens=$T micro_batch_seqs=$S > logs/rl2_rtxpro.log 2>&1
  rc=$?
  [ -f $RLDIR/summary.json ] && { say "RL phase 2 finished (rc=$rc)"; break; }
  if tail -n 80 logs/rl2_rtxpro.log | grep -a -q "OutOfMemoryError"; then [ $IDX -lt $((${#LADDER[@]}-1)) ] && IDX=$((IDX+1)); say "OOM -> ladder $IDX"; else say "exited rc=$rc: restarting"; fi
  sleep 30
done
touch $RLDIR/RL_DONE; say "PHASE 2 DONE"
