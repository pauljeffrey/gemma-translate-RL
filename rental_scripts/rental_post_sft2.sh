#!/bin/bash
# Unattended pipeline after the SFT (owner away, 2026-10-19): SFT_DONE -> benchmark evaluation (base vs SFT best) -> results to the Hub -> merge of the staged RL code -> RL.
cd /workspace/sft_build
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv_gemma/bin/python
SFT=runs/sft_unimax; RLDIR=runs/rl; EVAL=eval/sft_best
say() { echo "$(date -u +%FT%TZ) [post_sft2] $*" | tee -a logs/rental_run.log; }
gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
wait_gpu_empty() { local i=0; while [ "$(gpu_used_mib)" -gt 8000 ] && [ $i -lt 90 ]; do sleep 10; i=$((i+1)); done; say "GPU memory used: $(gpu_used_mib) MiB"; }
say "armed: waiting for $SFT/SFT_DONE"
while [ ! -f $SFT/SFT_DONE ]; do
  if [ -f logs/RENTAL_GAVE_UP ]; then say "the SFT supervisor gave up: nothing to do here"; exit 1; fi
  sleep 30
done
say "SFT finished"
# ------------------------------------------------------------------------------------------ benchmark evaluation
if [ ! -f $EVAL/EVAL_DONE ]; then
  wait_gpu_empty; mkdir -p eval
  say "EVAL start"
  nohup bash gemma_mt_eval_candidate/scripts/eval_sft_benchmarks.sh > eval/sft_best.launch.log 2>&1 &
  EP=$!
  for i in $(seq 1 330); do
    { [ -f $EVAL/EVAL_DONE ] || [ -f $EVAL/EVAL_FAILED ]; } && break
    kill -0 $EP 2>/dev/null || break
    sleep 60
  done
  if [ -f $EVAL/EVAL_DONE ]; then say "EVAL done"; elif [ -f $EVAL/EVAL_FAILED ]; then say "EVAL failed ($(head -c 300 $EVAL/EVAL_FAILED | tr '\n' ' ')): continuing"; else say "EVAL ended without a marker / timed out: continuing"; kill $EP 2>/dev/null; fi
  sleep 20
fi
# ------------------------------------------------------------------------------------------ results to the Hub (small files only)
$PY - <<'PYEOF' 2>&1 | tail -3 | tee -a logs/rental_run.log
import os, glob
from huggingface_hub import HfApi
d = "/workspace/sft_build/eval/sft_best"
files = [f for f in ("report.md", "results.json", "results.csv", "EVAL_DONE", "EVAL_FAILED") if os.path.exists(os.path.join(d, f))]
if files:
    api = HfApi(token=open("/workspace/sft_build/.hf_write_token").read().strip())
    for repo in ("${HF_ORG}/gemma3-270m-translate",):
        try:
            for f in files:
                api.upload_file(path_or_fileobj=os.path.join(d, f), path_in_repo="eval/sft_benchmarks/" + f, repo_id=repo, commit_message="SFT benchmark evaluation (post-SFT)")
            print("eval results uploaded to", repo, files)
        except Exception as e:
            print("eval results upload FAILED:", type(e).__name__, str(e)[-200:])
else:
    print("no eval result files to upload")
PYEOF
# ------------------------------------------------------------------------------------------ merge of the staged RL code (reward upgrades, GlotLID, consensus, new languages)
if [ ! -f gemma_mt_rl_new/STAGED_OK ]; then say "waiting up to 30 min for gemma_mt_rl_new/STAGED_OK"; for i in $(seq 1 30); do [ -f gemma_mt_rl_new/STAGED_OK ] && break; sleep 60; done; fi
if [ -f gemma_mt_rl_new/STAGED_OK ] && [ ! -f gemma_mt/.rl_overlay_applied ]; then
  rm -rf gemma_mt_before_rl_merge; cp -r gemma_mt gemma_mt_before_rl_merge
  cp -r gemma_mt_rl_new/. gemma_mt/ && touch gemma_mt/.rl_overlay_applied
  if PYTHONPATH=/workspace/sft_build/gemma_mt $PY -c "import gemma_mt.rl.train_rl, gemma_mt.languages" 2>logs/rl_overlay_import.log; then say "RL overlay merged and importable"
  else say "RL overlay import FAILED: restoring the previous code ($(tail -n 1 logs/rl_overlay_import.log | cut -c1-150))"; rm -rf gemma_mt; mv gemma_mt_before_rl_merge gemma_mt; fi
else say "RL overlay: ${STAGED:-not staged / already applied}: RL runs with the code in gemma_mt as it is"; fi
export PYTHONPATH=/workspace/sft_build/gemma_mt
wait_gpu_empty
# ------------------------------------------------------------------------------------------ RL
if [ -f $RLDIR/summary.json ] && grep -q placeholder $RLDIR/summary.json 2>/dev/null; then rm -f $RLDIR/summary.json; fi
export VLLM_CACHE=/workspace/sft_build/vllm_cache
export VLLM_CACHE_ROOT=$VLLM_CACHE/vllm TRITON_CACHE_DIR=$VLLM_CACHE/triton TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE/inductor XDG_CACHE_HOME=$VLLM_CACHE/xdg VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
mkdir -p $VLLM_CACHE
STARTS=()
while [ ! -f $RLDIR/summary.json ]; do
  now=$(date +%s); keep=(); for t in "${STARTS[@]}"; do [ $((now - t)) -lt 3600 ] && keep+=($t); done; STARTS=("${keep[@]}")
  if [ ${#STARTS[@]} -ge 4 ]; then say "RL: 4 starts in the last hour: giving up"; touch logs/RENTAL_RL_GAVE_UP; exit 1; fi
  STARTS+=($now)
  say "RL start (configs/rl_rtxpro.yaml)"
  cp logs/rl_rtxpro.log logs/rl_rtxpro_prev_$(date +%s).log 2>/dev/null
  $PY -m gemma_mt.rl.train_rl --config gemma_mt/configs/rl_rtxpro.yaml > logs/rl_rtxpro.log 2>&1
  rc=$?
  [ -f $RLDIR/summary.json ] && { say "RL finished (rc=$rc)"; break; }
  say "RL exited rc=$rc without summary.json: restarting (resume)"; sleep 30
done
touch $RLDIR/RL_DONE
say "ALL DONE"
