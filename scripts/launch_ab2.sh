#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Launcher / supervisor of the AB2 ablation (cheap RL-vs-SFT-start ablation + continued-SFT controls). Same control-file semantics as launch_ab1.sh / launch_final2.sh.
#
#   bash scripts/launch_ab2.sh check  sft|full|weak|s1|s1p        # preflight only (no GPU use): start model, data, disk, /dev/shm, GPU memory, live-run guard, yaml loads strictly; exit 0 = ready
#                                                                  (SKIP_GPU=1 skips the GPU-memory and live-run checks: dry runs while the final run still trains)
#   bash scripts/launch_ab2.sh start  sft|full|weak|s1|s1p [k=v ..]   # preflight, then the supervisor in the background (setsid nohup)
#   bash scripts/launch_ab2.sh status [arm|all]
#   bash scripts/launch_ab2.sh winddown full|weak [N] | lr_scale full|weak X | hold|go|stop ARM
#   bash scripts/launch_ab2.sh convert sft|s1|s1p                  # SFT trainer state -> HF directory: sft -> models/sft_weak (float32, like SFT-final), s1 / s1p -> models/w_s1, models/w_s1p (bfloat16)
#   bash scripts/launch_ab2.sh eval MODEL_DIR NAME [UTIL]          # scripts/ab2_eval.sh: yor kin orm lin x FLORES+ / NTREX / MAFAND / SMOL, surface + neural metrics -> $AB/eval/NAME
#
# Arms: sft = weak SFT (3,500 steps), full / weak = RL from SFT-final / from the weak SFT, s1 = continued SFT (GPU-time matched), s1p = continued SFT (pair-matched).
# HOLD present before a (re)start of the trainer -> wait for GO; STOP present when the trainer exits -> no restart. WINDDOWN / LR_SCALE are read by the RL trainer itself. The supervisor NEVER kills a trainer.
# `$RUN/last` is a SYMLINK into /dev/shm/ab2/<name>/last (3.2 GB, rewritten through a .tmp copy): the disk has no room for it.
# Environment overrides: AB TREE CFG RUN TAG PY MIN_FREE_GIB MIN_DISK_GIB MIN_SHM_GIB POLL TRAINER_CMD FORCE=1 (skip the live-run guard) SKIP_GPU=1
set -uo pipefail
AB="${AB:-/workspace/jobs/ab2}"
TREE="${TREE:-$AB/gemma_mt}"
PY="${PY:-/workspace/venv_gemma/bin/python}"
POLL="${POLL:-30}"
MIN_DISK_GIB="${MIN_DISK_GIB:-2}"
MIN_SHM_GIB="${MIN_SHM_GIB:-10}"
LIVE_DIR="${LIVE_DIR:-/workspace/jobs/r7}"
LOGDIR="$AB/logs"
mkdir -p "$LOGDIR"

which="${2:-}"
case "$which" in
  sft)  CFG="${CFG:-$TREE/configs/ab2_sft_weak.yaml}"; RUN="${RUN:-$AB/runs/sft_weak}"; TAG="${TAG:-AB2_SFT}"; MIN_FREE_GIB="${MIN_FREE_GIB:-40}"
        LADDER=("16 2 false" "8 4 false" "4 8 false" "2 16 true"); MODULE="gemma_mt.train_sft"; KEYS=(micro_batch_size grad_accum_steps gradient_checkpointing);;
  s1)   CFG="${CFG:-$TREE/configs/ab2_sft_s1.yaml}"; RUN="${RUN:-$AB/runs/sft_s1}"; TAG="${TAG:-AB2_S1}"; MIN_FREE_GIB="${MIN_FREE_GIB:-40}"
        LADDER=("16 2 false" "8 4 false" "4 8 false" "2 16 true"); MODULE="gemma_mt.train_sft"; KEYS=(micro_batch_size grad_accum_steps gradient_checkpointing);;
  s1p)  CFG="${CFG:-$TREE/configs/ab2_sft_s1pair.yaml}"; RUN="${RUN:-$AB/runs/sft_s1pair}"; TAG="${TAG:-AB2_S1P}"; MIN_FREE_GIB="${MIN_FREE_GIB:-30}"
        LADDER=("8 1 false" "4 2 false" "2 4 true"); MODULE="gemma_mt.train_sft"; KEYS=(micro_batch_size grad_accum_steps gradient_checkpointing);;
  weak) CFG="${CFG:-$TREE/configs/rl_ab2_weak.yaml}"; RUN="${RUN:-$AB/runs/ab2_weak}"; TAG="${TAG:-AB2_W}"; MIN_FREE_GIB="${MIN_FREE_GIB:-62}"
        LADDER=("24576 128" "12288 64" "6144 32" "3072 16"); MODULE="gemma_mt.rl.train_rl"; KEYS=(micro_batch_tokens micro_batch_seqs);;
  full) CFG="${CFG:-$TREE/configs/rl_ab2_full.yaml}"; RUN="${RUN:-$AB/runs/ab2_full}"; TAG="${TAG:-AB2_F}"; MIN_FREE_GIB="${MIN_FREE_GIB:-62}"
        LADDER=("24576 128" "12288 64" "6144 32" "3072 16"); MODULE="gemma_mt.rl.train_rl"; KEYS=(micro_batch_tokens micro_batch_seqs);;
  *)    : ;;
esac
LOG="$LOGDIR/${TAG:-ab2}.log"
PIDFILE="${RUN:-$AB}/supervisor.pid"
say() { echo "$(date -u +%FT%TZ) [launch-${TAG:-ab2}] $*" | tee -a "$LOGDIR/launch_ab2.log"; }

env_setup() {
  [ -f /workspace/sft_build/env.sh ] && source /workspace/sft_build/env.sh
  export PYTHONPATH="$TREE:/workspace/sft_build/pylibs" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
  export VLLM_CACHE=/workspace/sft_build/vllm_cache
  export VLLM_CACHE_ROOT=$VLLM_CACHE/vllm TRITON_CACHE_DIR=$VLLM_CACHE/triton TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE/inductor XDG_CACHE_HOME=$VLLM_CACHE/xdg VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
}
alive() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; }
live_run_active() { pgrep -f "gemma_mt.rl.train_rl --config $LIVE_DIR" >/dev/null 2>&1 || pgrep -f "train_rl.*$LIVE_DIR/runs" >/dev/null 2>&1; }

prepare_dirs() {   # $RUN/last -> /dev/shm (idempotent); never replaces a real directory
  mkdir -p "$RUN" "/dev/shm/$(basename "$AB")/$(basename "$RUN")/last"
  if [ ! -e "$RUN/last" ] && [ ! -L "$RUN/last" ]; then ln -s "/dev/shm/$(basename "$AB")/$(basename "$RUN")/last" "$RUN/last"; fi
}

preflight() {      # prints problems, returns their number
  [ "${MOCK:-0}" = 1 ] && return 0
  local bad=0 free disk shm need
  [ -f "$CFG" ] || { say "PREFLIGHT: config $CFG not found"; bad=$((bad+1)); }
  case "$which" in
    sft)
      [ -f "$AB/data/packed_fwd/train/tokens.u32.bin" ] && [ -f "$AB/data/packed_fwd/val/tokens.u32.bin" ] || { say "PREFLIGHT: $AB/data/packed_fwd (symlink to the AB1 sparse set) is missing"; bad=$((bad+1)); }
      ls /workspace/sft_build/hf/hub/models--google--gemma-3-270m-it/snapshots/*/model.safetensors >/dev/null 2>&1 || { say "PREFLIGHT: google/gemma-3-270m-it is not in the HF cache"; bad=$((bad+1)); };;
    s1|s1p)
      local d="$AB/data/packed_s1"; [ "$which" = s1p ] && d="$AB/data/packed_s1_pair"
      [ -f "$d/train/tokens.u32.bin" ] && [ -f "$d/val/tokens.u32.bin" ] && [ -f "$d/meta.json" ] || { say "PREFLIGHT: $d is missing (scripts/ab2_build_s1_data.py needs the presigned URL in $AB/S1_URL)"; bad=$((bad+1)); }
      [ -f "$AB/models/sft_weak/model.safetensors" ] || { say "PREFLIGHT: start model $AB/models/sft_weak is missing (launch_ab2.sh convert sft)"; bad=$((bad+1)); };;
    weak|full)
      local m; [ "$which" = weak ] && m="$AB/models/sft_weak" || m="$AB/models/sft_final"
      [ -f "$m/model.safetensors" ] || { say "PREFLIGHT: start model $m/model.safetensors is missing"; bad=$((bad+1)); }
      for p in "$AB/data/rl_prompts_ab2/yor.parquet" "$TREE/calibration/shaping_flores_dev_v7.json" /workspace/sft_build/bench/bench_dev_clean.jsonl; do
        [ -e "$p" ] || { say "PREFLIGHT: missing $p"; bad=$((bad+1)); }
      done
      if [ -f "$RUN/last/trainer_state.pt" ]; then say "NOTE: $RUN/last/trainer_state.pt exists: the run RESUMES (resume: true); remove $RUN/last/* for a fresh start"; fi;;
  esac
  disk=$(df -P --block-size=1G "$AB" | awk 'NR==2{print $4}')
  [ "${disk:-0}" -ge "$MIN_DISK_GIB" ] || { say "PREFLIGHT: only ${disk} GiB of disk free (need $MIN_DISK_GIB)"; bad=$((bad+1)); }
  shm=$(df -P --block-size=1G /dev/shm | awk 'NR==2{print $4}')
  [ "${shm:-0}" -ge "$MIN_SHM_GIB" ] || { say "PREFLIGHT: only ${shm} GiB free in /dev/shm (need $MIN_SHM_GIB)"; bad=$((bad+1)); }
  if [ "${SKIP_GPU:-0}" != 1 ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); free=$((free / 1024))
      [ "$free" -ge "$MIN_FREE_GIB" ] || { say "PREFLIGHT: only $free GiB of GPU memory free (need $MIN_FREE_GIB)"; bad=$((bad+1)); }
    fi
    if [ "${FORCE:-0}" != 1 ] && live_run_active; then
      say "PREFLIGHT: the live run ($LIVE_DIR, final2) is still training: wait for it (or FORCE=1)"; bad=$((bad+1))
    fi
    if [ "${FORCE:-0}" != 1 ] && [ ! -f "$LIVE_DIR/runs/final2/summary.json" ]; then
      say "PREFLIGHT: $LIVE_DIR/runs/final2/summary.json does not exist yet (the final run has not finished)"; bad=$((bad+1))
    fi
  fi
  return $bad
}

supervise() {
  local shift_args=("$@") idx=0 starts=() now keep t a b c rc
  env_setup; prepare_dirs
  cd "$TREE" || exit 2
  export RUN TAG; echo $$ > "$PIDFILE"; trap 'rm -f "$PIDFILE"' EXIT
  rm -f "$RUN/summary.json"
  while [ ! -f "$RUN/summary.json" ]; do
    if [ -f "$RUN/STOP" ]; then say "STOP file present: not (re)starting the trainer, supervisor exits"; exit 0; fi
    if [ -f "$RUN/HOLD" ]; then
      say "HOLD file present: waiting for GO (touch $RUN/GO) or removal of HOLD"
      while [ -f "$RUN/HOLD" ] && [ ! -f "$RUN/GO" ] && [ ! -f "$RUN/STOP" ]; do sleep "$POLL"; done
      if [ -f "$RUN/GO" ]; then rm -f "$RUN/HOLD" "$RUN/GO"; say "GO received"; fi
      continue
    fi
    now=$(date +%s); keep=(); for t in "${starts[@]:-}"; do [ -n "$t" ] && [ $((now - t)) -lt 3600 ] && keep+=("$t"); done; starts=("${keep[@]:-}")
    if [ "${#keep[@]}" -ge 8 ]; then say "8 starts in the last hour: giving up"; touch "$LOGDIR/GAVE_UP_$TAG"; exit 1; fi
    starts+=("$now"); read -r a b c < <(echo "${LADDER[$idx]}")
    local ov=("${KEYS[0]}=$a" "${KEYS[1]}=$b"); [ -n "${c:-}" ] && ov+=("${KEYS[2]}=$c")
    say "start ($CFG): ${ov[*]} ${shift_args[*]+${shift_args[*]}}"
    cp "$LOG" "$LOGDIR/${TAG}_prev_$(date +%s).log" 2>/dev/null
    if [ -n "${TRAINER_CMD:-}" ]; then
      bash -c "$TRAINER_CMD" > "$LOG" 2>&1
    else
      "$PY" -m "$MODULE" --config "$CFG" out_dir="$RUN" "${ov[@]}" ${shift_args[@]+"${shift_args[@]}"} > "$LOG" 2>&1
    fi
    rc=$?
    if [ -f "$RUN/summary.json" ]; then say "finished (rc=$rc)"; break; fi
    if tail -n 80 "$LOG" | grep -a -q "OutOfMemoryError"; then [ $idx -lt $((${#LADDER[@]}-1)) ] && idx=$((idx+1)); say "OOM -> ladder $idx"; else say "exited rc=$rc: restarting"; fi
    if [ -f "$RUN/STOP" ]; then say "STOP file present after the trainer exited: supervisor exits"; exit 0; fi
    sleep "${RESTART_SLEEP:-20}"
  done
  touch "$RUN/DONE_$TAG"; say "DONE $TAG"
}

do_convert() {   # sft -> models/sft_weak (fp32); s1 / s1p -> bf16 evaluation models
  local w="${1:?sft|s1|s1p}" run out dt
  env_setup
  case "$w" in sft) run="$AB/runs/sft_weak"; out="$AB/models/sft_weak"; dt=float32;; s1) run="$AB/runs/sft_s1"; out="$AB/models/w_s1"; dt=bfloat16;; s1p) run="$AB/runs/sft_s1pair"; out="$AB/models/w_s1p"; dt=bfloat16;; *) echo "usage: convert sft|s1|s1p"; exit 2;; esac
  local st="$run/last/trainer_state.pt"
  [ -f "$st" ] || { say "convert: $st missing"; exit 3; }
  [ -f "$run/summary.json" ] || say "WARNING: $run/summary.json is missing: the state is an intermediate checkpoint"
  CUDA_VISIBLE_DEVICES= "$PY" "$TREE/scripts/interpolate_weights.py" --init "$AB/models/sft_final" --sft "$st" --alpha 1.0 --out "$out" --dtype "$dt" 2>&1 | tail -3
  [ -f "$out/model.safetensors" ] && say "convert $w: wrote $out ($(du -h "$out/model.safetensors" | cut -f1))"
}

status_one() {
  local w="$1" r tag
  case "$w" in sft) r="$AB/runs/sft_weak"; tag=AB2_SFT;; weak) r="$AB/runs/ab2_weak"; tag=AB2_W;; full) r="$AB/runs/ab2_full"; tag=AB2_F;; s1) r="$AB/runs/sft_s1"; tag=AB2_S1;; s1p) r="$AB/runs/sft_s1pair"; tag=AB2_S1P;; esac
  echo "== $w ($r)"
  if [ -f "$r/supervisor.pid" ] && kill -0 "$(cat "$r/supervisor.pid" 2>/dev/null)" 2>/dev/null; then echo "supervisor: running, pid $(cat "$r/supervisor.pid")"; else echo "supervisor: not running"; fi
  for f in HOLD GO STOP WINDDOWN WINDDOWN.state WINDDOWN.done LR_SCALE summary.json "DONE_$tag"; do [ -e "$r/$f" ] && echo "$f: $(head -c 80 "$r/$f" 2>/dev/null | tr '\n' ' ')"; done
  [ -f "$r/metrics.jsonl" ] && "$PY" - "$r/metrics.jsonl" "$w" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
tr = [r for r in rows if "t_step" in r or "reward" in r or "train_loss" in r]
ev = [r for r in rows if "bench/chrf_macro" in r]
if tr:
    r = tr[-1]
    keys = ("step", "reward", "kl", "finished_rate", "completion_len", "lr", "t_step", "train_loss", "grad_norm", "kl_guard_lr_scale")
    print("last train row:", {k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in keys if k in r})
    ts = [x["t_step"] for x in tr[-30:] if "t_step" in x]
    if ts: print("median t_step of the last %d steps: %.1f s" % (len(ts), sorted(ts)[len(ts)//2]))
if ev:
    print("dev chrF++ (yor kin orm, greedy 256 tokens): " + "  ".join("%s@%s" % (round(r["bench/chrf_macro"], 2), r["step"]) for r in ev[-8:]) + "   finished " + str(round(ev[-1].get("bench/finished_rate", float("nan")), 4)))
PYEOF
  tail -n 2 "$LOGDIR/$tag.log" 2>/dev/null | cut -c1-220
}

cmd="${1:-}"; [ $# -gt 0 ] && shift
case "$cmd" in
  start)
    [ -n "${CFG:-}" ] || { echo "usage: $0 start sft|full|weak|s1|s1p [k=v ...]"; exit 2; }
    shift   # the arm name
    alive && { say "supervisor already running (pid $(cat "$PIDFILE")): refusing to start a second one"; exit 4; }
    [ -f "$RUN/STOP" ] && { say "STOP file present in $RUN: remove it first"; exit 5; }
    env_setup; prepare_dirs; preflight || { say "preflight failed (see above); nothing started"; exit 3; }
    setsid nohup bash "$0" _supervise "$which" "$@" > /dev/null 2>&1 < /dev/null &
    sleep 2; say "supervisor started (pidfile $PIDFILE); log $LOG; metrics $RUN/metrics.jsonl";;
  _supervise) shift; supervise "$@";;
  check)
    [ -n "${CFG:-}" ] || { echo "usage: $0 check sft|full|weak|s1|s1p"; exit 2; }
    env_setup; prepare_dirs
    if preflight; then
      pycode=""
      case "$which" in
        weak|full) pycode="import sys
sys.path.insert(0, '$TREE')
from gemma_mt.rl.train_rl import RLConfig, apply_overrides, validate_config
from gemma_mt.utils import dataclass_from_dict, load_yaml, parse_overrides
cfg = dataclass_from_dict(RLConfig, apply_overrides(load_yaml('$CFG'), parse_overrides(['out_dir=$RUN'])))
validate_config(cfg)
print('yaml loads strictly and validates: steps', cfg.steps, 'lr', cfg.learning_rate, 'group', cfg.group_size, 'curriculum', cfg.curriculum_steps, 'export', cfg.export_steps, 'start', cfg.model_name)";;
        *) pycode="import sys
sys.path.insert(0, '$TREE')
from gemma_mt.train_sft import SFTConfig
from gemma_mt.utils import dataclass_from_dict, load_yaml
cfg = dataclass_from_dict(SFTConfig, load_yaml('$CFG'))
print('yaml loads strictly: max_steps', cfg.max_steps, 'lr', cfg.learning_rate, 'schedule_start_step', cfg.schedule_start_step, 'hold', cfg.lr_hold_fraction, 'data', cfg.data_dir)";;
      esac
      if ! out=$(cd "$TREE" && CUDA_VISIBLE_DEVICES= "$PY" -c "$pycode" 2>&1); then echo "$out" | tail -4; say "config check FAILED"; exit 3; fi
      echo "$out" | tail -1
      say "preflight OK"; exit 0
    fi
    exit 3;;
  status)
    env_setup
    if [ -z "$which" ] || [ "$which" = all ]; then for w in sft full weak s1 s1p; do status_one $w; done; else status_one "$which"; fi
    df -h "$AB" | tail -1; df -h /dev/shm | tail -1; command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    tail -n 3 "$LOGDIR/launch_ab2.log" 2>/dev/null;;
  winddown) echo "${3:-50}" > "$RUN/WINDDOWN"; say "WINDDOWN ${3:-50} written";;
  lr_scale) [ -n "${3:-}" ] || { echo "usage: lr_scale weak|full X"; exit 2; }; echo "$3" > "$RUN/LR_SCALE"; say "LR_SCALE $3 written";;
  hold) mkdir -p "$RUN"; touch "$RUN/HOLD"; say "HOLD set";;
  go)   mkdir -p "$RUN"; touch "$RUN/GO"; say "GO set";;
  stop) mkdir -p "$RUN"; touch "$RUN/STOP"; say "STOP set (no restart after the current trainer exits; the running trainer is NOT killed)";;
  convert) do_convert "${1:-}";;
  eval) env_setup; bash "$TREE/scripts/ab2_eval.sh" "${1:?model dir}" "${2:?name}" "${3:-0.1}";;
  *) sed -n 2,17p "$0"; exit 2;;
esac
