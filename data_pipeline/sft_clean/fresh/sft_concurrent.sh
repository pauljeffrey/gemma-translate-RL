#!/bin/bash
# Run several SFT jobs at the same time on the ONE GPU (RTX PRO 6000, 96 GB: a 270M-1B model at micro batch 2 x 4,096 tokens uses a small part of it and one job leaves the SMs idle between kernels).
# Each job = NAME=CONFIG[,key=value,key=value]; every job gets its own out_dir (default runs/NAME) and log (logs/sft_NAME.log). Start is staggered (STAGGER_S, default 120 s) so that the
# VRAM of the first job is measured before the next one starts; a job is only started when free VRAM >= VRAM_MARGIN (default 1.25) x the VRAM the running jobs use per job.
#   cd $ROOT/run && bash $FR/sft_concurrent.sh a0=configs/clean_rerun_sft/a0.yaml s1=configs/clean_rerun_sft/s1.yaml
#   USE_MPS=1 bash sft_concurrent.sh ...    # CUDA MPS (nvidia-cuda-mps-control -d) so that the kernels of both jobs can overlap; off by default (try both, see sft_throughput_probe.sh)
# Different seeds/arms are independent runs: the only coupling is the speed each gets. Combined throughput is what sft_throughput_probe.sh measures. Needs: PYTHONPATH with gemma_mt, run from the
# folder that holds packed/ and sched/ (configs use relative paths).
set -uo pipefail
PY=${PY:-python}; STAGGER_S=${STAGGER_S:-120}; VRAM_MARGIN=${VRAM_MARGIN:-1.25}; LG=${LG:-logs}; mkdir -p "$LG"
[ "$#" -ge 1 ] || { echo "usage: $0 NAME=CONFIG[,key=value...] ..."; exit 2; }
if [ "${USE_MPS:-0}" = "1" ]; then export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps} CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}; mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; nvidia-cuda-mps-control -d 2>/dev/null || echo "MPS daemon not started (already running or not available)"; fi
used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc 0-9; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc 0-9; }
base=$(used_mib); pids=(); names=(); first_used=0
for spec in "$@"; do
  name=${spec%%=*}; rest=${spec#*=}; cfg=${rest%%,*}; extra=""; [ "$rest" != "$cfg" ] && extra=${rest#*,}
  ov=(); IFS=',' read -ra kv <<< "$extra"; for x in "${kv[@]}"; do [ -n "$x" ] && ov+=("$x"); done
  case " ${ov[*]:-} " in *" out_dir="*) ;; *) ov+=("out_dir=runs/$name");; esac
  if [ "${#pids[@]}" -gt 0 ]; then
    sleep "$STAGGER_S"; per_job=$(( ($(used_mib) - base) / ${#pids[@]} )); need=$(awk -v p="$per_job" -v m="$VRAM_MARGIN" 'BEGIN{printf "%d", p*m}')
    if [ "$(free_mib)" -lt "$need" ]; then echo "[SKIP] $name: free VRAM $(free_mib) MiB < $need MiB (1.25 x ${per_job} MiB per running job); not started"; continue; fi
  fi
  echo "[start] $name  config=$cfg  overrides=${ov[*]}  ($(date -u +%T))"
  nohup "$PY" -m gemma_mt.train_sft --config "$cfg" "${ov[@]}" > "$LG/sft_$name.log" 2>&1 &
  pids+=($!); names+=("$name")
done
echo "started ${#pids[@]} job(s): ${names[*]}; VRAM now $(used_mib) MiB of base $base MiB; logs $LG/sft_<name>.log; metrics <out_dir>/metrics.jsonl (tokens_per_s, step, time)"
rc=0; for i in "${!pids[@]}"; do wait "${pids[$i]}"; r=$?; echo "[exit] ${names[$i]} rc=$r"; [ "$r" = 0 ] || rc=1; done
exit $rc
