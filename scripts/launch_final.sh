#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Launcher / supervisor of the FINAL science + general GRPO run (D-59). Same OOM ladder and restart logic as /workspace/sft_build/sup.sh, plus a pidfile and STOP / HOLD / GO files.
#
#   bash scripts/launch_final.sh start [key=value ...]   # preflight, then run the supervisor in the background (setsid nohup); extra key=value go to train_rl (e.g. steps=1100)
#   bash scripts/launch_final.sh check                   # preflight only: state file, GPU memory, disk, assets, yaml loads strictly (no GPU use); exit 0 = ready
#   bash scripts/launch_final.sh status                  # pidfile, last log lines, step, the control files
#   bash scripts/launch_final.sh winddown [N]            # echo N (default 50) > $RUN/WINDDOWN : lr decays linearly to 0 over N steps, then a NORMAL finish (final eval, save, push) - the graceful way to end
#   bash scripts/launch_final.sh lr_scale X              # echo X > $RUN/LR_SCALE : multiplier on the learning rate, read every step
#   bash scripts/launch_final.sh hold | go | stop        # touch $RUN/HOLD | $RUN/GO | $RUN/STOP
#
# Control-file semantics (the supervisor only looks at them BETWEEN trainer processes; it NEVER kills a running trainer; ending a run is the owner's decision):
#   HOLD  present before a (re)start of the trainer  -> the supervisor waits (polls every 30 s) and starts nothing until GO appears or HOLD is removed. GO is consumed (HOLD and GO are deleted).
#   STOP  present when the trainer process exits     -> the supervisor does not restart it and exits (no RL_DONE marker). Also checked before the first start. STOP is not deleted.
#   (a trainer that is up keeps running; to end it gracefully use `winddown`; `kill` of the trainer is the owner's explicit decision and then STOP prevents the restart)
#   WINDDOWN / LR_SCALE are read by the trainer itself (rl/run_controls.py).
# Environment overrides (tests / shakedown): CFG, RUN (out_dir), TREE, TRAINER_CMD (replaces `python -m gemma_mt.rl.train_rl --config ...`), MIN_FREE_GIB (GPU, default 70: measured peak of the whole run 65-68 GB), MIN_DISK_GIB (default 8), POLL (default 30).
set -uo pipefail
R7="${R7:-/workspace/jobs/r7}"
TREE="${TREE:-$R7/gemma_mt_run}"
CFG="${CFG:-$TREE/configs/rl_science_final.yaml}"
RUN="${RUN:-$R7/runs/final}"
TAG="${TAG:-FINAL}"
LOGDIR="${LOGDIR:-$R7/logs}"
LOG="$LOGDIR/rl_$TAG.log"
PIDFILE="$RUN/supervisor.pid"
POLL="${POLL:-30}"
PY="${PY:-/workspace/venv_gemma/bin/python}"
MIN_FREE_GIB="${MIN_FREE_GIB:-70}"
MIN_DISK_GIB="${MIN_DISK_GIB:-8}"
LADDER=("24576 128" "12288 64" "6144 32" "3072 16")
mkdir -p "$RUN" "$LOGDIR"
say() { echo "$(date -u +%FT%TZ) [launch-$TAG] $*" | tee -a "$LOGDIR/launch_$TAG.log"; }

env_setup() {
  [ -f /workspace/sft_build/env.sh ] && source /workspace/sft_build/env.sh
  export PYTHONPATH="$TREE:/workspace/sft_build/pylibs" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export VLLM_CACHE=/workspace/sft_build/vllm_cache
  export VLLM_CACHE_ROOT=$VLLM_CACHE/vllm TRITON_CACHE_DIR=$VLLM_CACHE/triton TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE/inductor XDG_CACHE_HOME=$VLLM_CACHE/xdg VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
  export GEMMA_MT_LEXICON_DIR="${GEMMA_MT_LEXICON_DIR:-$R7/assets/lexicon}"
}

alive() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; }

preflight() {   # prints problems, returns their number
  local bad=0 free disk
  [ -f "$RUN/last/trainer_state.pt" ] || { say "PREFLIGHT: $RUN/last/trainer_state.pt is missing (run scripts/prepare_initial_state.py first)"; bad=$((bad+1)); }
  [ -f "$CFG" ] || { say "PREFLIGHT: config $CFG not found"; bad=$((bad+1)); }
  for p in "$R7/assets/rl_prompts_sci_v1" "$R7/assets/calibration/shaping_flores_dev_v8.json" "$R7/assets/calibration/anchors_v1_lex.json" "$R7/assets/calibration/science_anchor_v1.json" \
           "$R7/assets/lexicon" "$R7/assets/dev/bench_afrisci_dev_sent.jsonl" "$R7/assets/dev/bench_afrisci_dev_doc.jsonl" "$R7/assets/dev/bench_dev_clean.jsonl" "$R7/assets/dev/bench_weak_flores_dev.jsonl" \
           /workspace/sft_build/runs/final_1433/model.safetensors; do
    [ -e "$p" ] || { say "PREFLIGHT: missing $p"; bad=$((bad+1)); }
  done
  disk=$(df -P --block-size=1G "$RUN" | awk 'NR==2{print $4}')
  [ "${disk:-0}" -ge "$MIN_DISK_GIB" ] || { say "PREFLIGHT: only ${disk} GiB of disk free under $RUN (need $MIN_DISK_GIB: the 3.2 GB trainer state is rewritten via a .tmp copy at every save)"; bad=$((bad+1)); }
  if command -v nvidia-smi >/dev/null 2>&1; then
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); free=$((free / 1024))
    [ "$free" -ge "$MIN_FREE_GIB" ] || { say "PREFLIGHT: only $free GiB of GPU memory free (need $MIN_FREE_GIB: vLLM 0.20 of the card = 19 GB + trainer ~42 GB + the 2-model QE worker 3-7 GB)"; bad=$((bad+1)); }
  fi
  return $bad
}

supervise() {
  shift_args=("$@")
  env_setup
  cd "$TREE" || exit 2
  echo $$ > "$PIDFILE"
  trap 'rm -f "$PIDFILE"' EXIT
  rm -f "$RUN/summary.json"
  local idx=0 starts=() now keep t T S rc
  while [ ! -f "$RUN/summary.json" ]; do
    if [ -f "$RUN/STOP" ]; then say "STOP file present: not (re)starting the trainer, supervisor exits"; exit 0; fi
    if [ -f "$RUN/HOLD" ]; then
      say "HOLD file present: waiting for GO (touch $RUN/GO) or removal of HOLD"
      while [ -f "$RUN/HOLD" ] && [ ! -f "$RUN/GO" ] && [ ! -f "$RUN/STOP" ]; do sleep "$POLL"; done
      if [ -f "$RUN/GO" ]; then rm -f "$RUN/HOLD" "$RUN/GO"; say "GO received"; fi
      continue
    fi
    now=$(date +%s); keep=(); for t in "${starts[@]:-}"; do [ -n "$t" ] && [ $((now - t)) -lt 3600 ] && keep+=("$t"); done; starts=("${keep[@]:-}")
    if [ "${#keep[@]}" -ge 8 ]; then say "8 starts in the last hour: giving up"; touch "$LOGDIR/RL_GAVE_UP_$TAG"; exit 1; fi
    starts+=("$now"); read -r T S < <(echo "${LADDER[$idx]}")
    say "RL start ($CFG): micro_batch_tokens=$T micro_batch_seqs=$S ${shift_args[*]+${shift_args[*]}}"
    cp "$LOG" "$LOGDIR/rl_${TAG}_prev_$(date +%s).log" 2>/dev/null
    if [ -n "${TRAINER_CMD:-}" ]; then
      bash -c "$TRAINER_CMD" > "$LOG" 2>&1
    else
      "$PY" -m gemma_mt.rl.train_rl --config "$CFG" out_dir="$RUN" micro_batch_tokens=$T micro_batch_seqs=$S ${shift_args[@]+"${shift_args[@]}"} > "$LOG" 2>&1
    fi
    rc=$?
    if [ -f "$RUN/summary.json" ]; then say "RL finished (rc=$rc)"; break; fi
    if tail -n 80 "$LOG" | grep -a -q "OutOfMemoryError"; then [ $idx -lt $((${#LADDER[@]}-1)) ] && idx=$((idx+1)); say "OOM -> ladder $idx"; else say "exited rc=$rc: restarting"; fi
    if [ -f "$RUN/STOP" ]; then say "STOP file present after the trainer exited: supervisor exits"; exit 0; fi
    sleep "${RESTART_SLEEP:-20}"
  done
  touch "$RUN/RL_DONE_$TAG"; say "DONE $TAG"
}

cmd="${1:-}"; [ $# -gt 0 ] && shift
case "$cmd" in
  start)
    alive && { say "supervisor already running (pid $(cat "$PIDFILE")): refusing to start a second one"; exit 4; }
    [ -f "$RUN/STOP" ] && { say "STOP file present in $RUN: remove it first"; exit 5; }
    env_setup; preflight || { say "preflight failed (see above); nothing started"; exit 3; }
    setsid nohup bash "$0" _supervise "$@" > /dev/null 2>&1 < /dev/null &
    sleep 2; say "supervisor started (pidfile $PIDFILE); log $LOG; metrics $RUN/metrics.jsonl"
    ;;
  _supervise) supervise "$@";;
  check)
    env_setup
    if preflight; then
      (cd "$TREE" && "$PY" - <<PYEOF
import sys
sys.path.insert(0, "$TREE")
from gemma_mt.rl.train_rl import RLConfig, apply_overrides, validate_config
from gemma_mt.utils import dataclass_from_dict, load_yaml, parse_overrides
cfg = dataclass_from_dict(RLConfig, apply_overrides(load_yaml("$CFG"), parse_overrides(["out_dir=$RUN"])))
validate_config(cfg)
print("yaml loads strictly and validates: steps", cfg.steps, "lr", cfg.learning_rate, "prompts", cfg.prompts_path)
PYEOF
      ) 2>&1 | tail -3
      say "preflight OK"; exit 0
    fi
    exit 3;;
  status)
    if alive; then echo "supervisor: running, pid $(cat "$PIDFILE")"; else echo "supervisor: not running"; fi
    for f in HOLD GO STOP WINDDOWN WINDDOWN.state WINDDOWN.done LR_SCALE summary.json; do [ -e "$RUN/$f" ] && echo "$f: $(head -c 80 "$RUN/$f" 2>/dev/null | tr '\n' ' ')"; done
    tail -n 3 "$LOGDIR/launch_$TAG.log" 2>/dev/null; tail -n 2 "$RUN/metrics.jsonl" 2>/dev/null | cut -c1-300
    df -h "$RUN" | tail -1; command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader;;
  winddown) echo "${1:-50}" > "$RUN/WINDDOWN"; say "WINDDOWN ${1:-50} written";;
  lr_scale) [ -n "${1:-}" ] || { echo "usage: lr_scale X"; exit 2; }; echo "$1" > "$RUN/LR_SCALE"; say "LR_SCALE $1 written";;
  hold) touch "$RUN/HOLD"; say "HOLD set";;
  go) touch "$RUN/GO"; say "GO set";;
  stop) touch "$RUN/STOP"; say "STOP set (no restart after the current trainer exits; the running trainer is NOT killed)";;
  *) sed -n 2,22p "$0"; exit 2;;
esac
