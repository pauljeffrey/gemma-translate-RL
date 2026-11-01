#!/bin/bash
# Runs the GPU side jobs next to the bulk QE scoring so that the 96 GB card never idles: drop one file per job into $ROOT/gpu_jobs.d/NAME.job
#     # vram_gb: 12            (required: the VRAM the job needs; it starts only when that much is free, plus GPU_RESERVE_GB for the QE workers)
#     # after: 31_hcal_refs    (optional: state/<step>.done files that must exist first, space separated)
#     # cpu_cores: 2           (optional, informational)
#     <shell commands; run with bash, cwd=$ROOT, env of env.sh; the job must be idempotent>
# Jobs the lead/calibration agents want next to QE: H_cal reference QE (built into run_chain.sh step 31), SSA-COMET scoring, the bigram table, the NLLB pseudo-reference precompute.
# Each job runs ONCE (state/gpujob_NAME.done), log logs/gpujob_NAME.log, failure -> state/gpujob_NAME.failed (delete it to retry). The runner exits when all jobs are done or when
# $ROOT/state/gpu_jobs.stop exists. Start:  nohup bash gpu_jobs_runner.sh >> logs/gpu_jobs_runner.log 2>&1 &     (run_chain.sh starts it at step 05)
set -uo pipefail
ROOT=${ROOT:-/workspace/work}; JD=$ROOT/gpu_jobs.d; ST=$ROOT/state; LG=$ROOT/logs; mkdir -p "$JD" "$ST" "$LG"; RES=${GPU_RESERVE_GB:-12}; POLL=${POLL_S:-30}
free_gb() { echo $(( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc 0-9) / 1024 )); }
declare -A running
while true; do
  [ -f "$ST/gpu_jobs.stop" ] && { echo "stop file"; exit 0; }
  pending=0
  for j in "$JD"/*.job; do
    [ -f "$j" ] || continue; n=$(basename "$j" .job)
    [ -f "$ST/gpujob_$n.done" ] || [ -f "$ST/gpujob_$n.failed" ] && continue
    pending=$((pending+1)); [ -n "${running[$n]:-}" ] && continue
    after=$(grep -m1 '^# after:' "$j" | cut -d: -f2); ok=1; for s in $after; do [ -f "$ST/$s.done" ] || ok=0; done; [ "$ok" = 1 ] || continue
    need=$(grep -m1 '^# vram_gb:' "$j" | tr -dc 0-9.); need=${need:-8}; need=${need%.*}
    if [ "$(free_gb)" -lt $((need + RES)) ]; then echo "$(date -u +%T) waiting: $n needs ${need} GB (+${RES} reserve), free $(free_gb) GB"; continue; fi
    echo "$(date -u +%T) start $n (vram ${need} GB, free $(free_gb) GB)"
    ( cd "$ROOT" && if bash "$j" >> "$LG/gpujob_$n.log" 2>&1; then date -u +%FT%TZ > "$ST/gpujob_$n.done"; else date -u +%FT%TZ > "$ST/gpujob_$n.failed"; fi ) &
    running[$n]=$!; sleep 45          # let the job claim its VRAM before the next free-memory reading
  done
  for n in "${!running[@]}"; do kill -0 "${running[$n]}" 2>/dev/null || unset "running[$n]"; done
  [ "$pending" = 0 ] && [ "${#running[@]}" = 0 ] && [ "${KEEP_ALIVE:-1}" = 0 ] && { echo "all jobs done"; exit 0; }
  sleep "$POLL"
done
