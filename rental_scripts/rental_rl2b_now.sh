#!/bin/bash
# Phase 2b of the RL (owner 2026-10-19): starts when phase 2 has finished (runs/rl2/summary.json) from phase 2's final (mixed-stage) model.
# 1,300,000 completions, group 16 = 15 sampled + 1 greedy, advantage = reward - group mean, lr 3e-6. The GPU is never idle: this supervisor waits for the phase-2 trainer to exit, it never touches it.
cd /workspace/sft_build
source env.sh
export PYTHONPATH=/workspace/sft_build/gemma_mt_2b PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv_gemma/bin/python; RLDIR=runs/rl2b
export VLLM_CACHE=/workspace/sft_build/vllm_cache
export VLLM_CACHE_ROOT=$VLLM_CACHE/vllm TRITON_CACHE_DIR=$VLLM_CACHE/triton TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE/inductor XDG_CACHE_HOME=$VLLM_CACHE/xdg VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
LADDER=("24576 128" "12288 64" "6144 32" "3072 16")
say() { echo "$(date -u +%FT%TZ) [rl2b_sup] $*" | tee -a logs/rental_run.log; }
say "owner stopped phase 2 at step 375 (2026-10-19): phase 2b starts from the latest branch of the rl2 repo"
say "fetching the step-375 model (branch latest)"
OK=0
for try in $(seq 1 40); do
  rm -rf runs/rl_phase2_final
  $PY - <<'PYEOF' 2>&1 | tail -2 | tee -a logs/rental_run.log
import os
from huggingface_hub import snapshot_download
p = snapshot_download("${HF_ORG}/gemma-3-270m-translate-rl2", revision="latest", local_dir="/workspace/sft_build/runs/rl_phase2_final", token=os.environ["HF_TOKEN"])
print("phase-2 final model downloaded:", p)
PYEOF
  [ -f runs/rl_phase2_final/model.safetensors ] && { OK=1; break; }
  say "latest branch not available yet (try $try/40)"; sleep 60
done
[ $OK = 1 ] || { say "download failed: phase 2b not started"; touch logs/RENTAL_RL2B_GAVE_UP; exit 1; }
IDX=0; STARTS=()
while [ ! -f $RLDIR/summary.json ]; do
  now=$(date +%s); keep=(); for t in "${STARTS[@]}"; do [ $((now - t)) -lt 3600 ] && keep+=($t); done; STARTS=("${keep[@]}")
  if [ ${#STARTS[@]} -ge 8 ]; then say "8 starts in the last hour: giving up"; touch logs/RENTAL_RL2B_GAVE_UP; exit 1; fi
  STARTS+=($now); read T S < <(echo "${LADDER[$IDX]}")
  say "RL phase 2b start: micro_batch_tokens=$T micro_batch_seqs=$S"
  cp logs/rl2b_rtxpro.log logs/rl2b_rtxpro_prev_$(date +%s).log 2>/dev/null
  $PY -m gemma_mt.rl.train_rl --config gemma_mt_2b/configs/rl_rtxpro_phase2b.yaml micro_batch_tokens=$T micro_batch_seqs=$S > logs/rl2b_rtxpro.log 2>&1
  rc=$?
  [ -f $RLDIR/summary.json ] && { say "RL phase 2b finished (rc=$rc)"; break; }
  if tail -n 80 logs/rl2b_rtxpro.log | grep -a -q "OutOfMemoryError"; then [ $IDX -lt $((${#LADDER[@]}-1)) ] && IDX=$((IDX+1)); say "OOM -> ladder $IDX"; else say "exited rc=$rc: restarting"; fi
  sleep 30
done
touch $RLDIR/RL_DONE; say "PHASE 2b DONE"
