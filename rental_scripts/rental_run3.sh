#!/bin/bash
# SFT supervisor v3 (owner 2026-10-19): continues the SAME run (runs/sft_unimax, exact resume incl. optimizer) with the varlen flash attention + fused loss
# (about 3x faster per step) and falls back to the old sdpa configuration if the new path misbehaves (OOM / NaN / loss far from the pre-switch level).
# Same data order in every variant: micro_batch_size x grad_accum_steps = 32 packs per optimizer step. Touches runs/sft_unimax/SFT_DONE when summary.json exists.
cd /workspace/sft_build
source env.sh
export PYTHONPATH=/workspace/sft_build/gemma_mt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv_gemma/bin/python
RUN=runs/sft_unimax
LADDER=("varlen fused 32 1" "varlen fused 16 2" "varlen fused 8 4" "sdpa chunked 16 2" "sdpa chunked 8 4" "sdpa chunked 4 8")
IDX_FILE=logs/.sft_ladder3_idx; [ -f $IDX_FILE ] || echo $([ "$USE_SDPA_ONLY" = 1 ] && echo 3 || echo 0) > $IDX_FILE
say() { echo "$(date -u +%FT%TZ) [v3] $*" | tee -a logs/rental_run.log; }
STARTS=()
guard() { local now=$(date +%s) keep=(); for t in "${STARTS[@]}"; do [ $((now - t)) -lt 3600 ] && keep+=($t); done; STARTS=("${keep[@]}")
  if [ ${#STARTS[@]} -ge 6 ]; then say "6 starts in the last hour: giving up"; touch logs/RENTAL_GAVE_UP; exit 1; fi; STARTS+=($now); }
bump() { n=$(( $(cat $IDX_FILE) + 1 )); [ $n -ge ${#LADDER[@]} ] && n=$((${#LADDER[@]}-1)); echo $n > $IDX_FILE; say "ladder -> $n (${LADDER[$n]})"; }
while [ ! -f $RUN/summary.json ]; do
  guard
  read AT LI M A < <(echo "${LADDER[$(cat $IDX_FILE)]}")
  say "SFT resume: attn=$AT loss=$LI micro_batch_size=$M grad_accum_steps=$A (ladder $(cat $IDX_FILE))"
  cp logs/sft_unimax_rtxpro.log logs/sft_unimax_rtxpro_prev_$(date +%s).log 2>/dev/null
  $PY -m gemma_mt.train_sft --config gemma_mt/configs/sft_unimax.yaml attn_implementation=$AT loss_impl=$LI micro_batch_size=$M grad_accum_steps=$A \
      gradient_checkpointing=false loss_chunk_tokens=2048 > logs/sft_unimax_rtxpro.log 2>&1 &
  TP=$!
  # sanity check of the first logged steps: finite and near the pre-switch level (~1.0, never above 2.0)
  bad=0
  for i in $(seq 1 120); do
    sleep 5
    kill -0 $TP 2>/dev/null || break
    n=$(grep -a -c "\] step [0-9]*/[0-9]* loss" logs/sft_unimax_rtxpro.log)
    if [ "$n" -ge 3 ]; then
      m=$(grep -a "\] step [0-9]*/[0-9]* loss" logs/sft_unimax_rtxpro.log | head -3 | sed -E 's/.* loss ([^ ]+) .*/\1/' | awk '{s+=$1; if ($1!=$1 || $1>3) b=1} END {print (b?9:s/NR)}')
      say "first steps mean loss $m (n=$n)"
      awk -v m="$m" 'BEGIN{exit !(m>2.0)}' && bad=1
      break
    fi
  done
  if [ $bad = 1 ]; then say "loss out of range: killing this attempt"; kill $TP; wait $TP 2>/dev/null; bump; continue; fi
  wait $TP; rc=$?
  if [ -f $RUN/summary.json ]; then say "SFT finished (rc=$rc)"; break; fi
  if tail -n 60 logs/sft_unimax_rtxpro.log | grep -a -qi "out of memory"; then say "OOM"; bump
  elif tail -n 60 logs/sft_unimax_rtxpro.log | grep -a -qi "varlen\|Triton\|kernel"; then say "kernel error: ladder step down"; bump
  else say "exited rc=$rc without summary.json: restarting"; fi
  sleep 15
done
touch $RUN/SFT_DONE
say "SFT_DONE"
