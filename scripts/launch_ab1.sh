#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Launcher / supervisor of the AB1 ablation (RL from a WEAK vs a FULL SFT checkpoint, identical RL settings; paper section 8.4). Style and control files of scripts/launch_final2.sh.
#
#   bash scripts/launch_ab1.sh check  sft|weak|full          # preflight only (no GPU use): start model, data, disk, /dev/shm, GPU memory, live-run guard, yaml loads strictly; exit 0 = ready
#   bash scripts/launch_ab1.sh start  sft|weak|full [k=v ..]  # preflight, then the supervisor in the background (setsid nohup); k=v go to the trainer (e.g. steps=300)
#   bash scripts/launch_ab1.sh status [sft|weak|full|all]     # pidfile, step, last metrics, control files, GPU / disk / shm, spend estimate
#   bash scripts/launch_ab1.sh winddown weak|full [N]         # echo N (default 50) > $RUN/WINDDOWN : lr decays linearly to 0 over N steps, then a NORMAL finish (final eval, snapshot, summary)
#   bash scripts/launch_ab1.sh lr_scale weak|full X           # echo X > $RUN/LR_SCALE (read every step)
#   bash scripts/launch_ab1.sh hold|go|stop sft|weak|full     # touch $RUN/HOLD | $RUN/GO | $RUN/STOP
#   bash scripts/launch_ab1.sh convert                        # SFT-weak trainer state -> models/sft_weak (float32 HF directory; same weights precision as SFT-final)
#   bash scripts/launch_ab1.sh eval MODEL_DIR NAME [N_GPU_UTIL]   # FLORES+ devtest, 45 languages, greedy 768 tokens, vLLM at 0.1 of the card (-> $AB/eval/NAME/{results.json,predictions.jsonl})
#   bash scripts/launch_ab1.sh quick MODEL_DIR NAME           # the <= 5 GB check: first 100 FLORES+ rows x 10 languages, HF sampler, nice 19
#
# Control-file semantics (as launch_final2.sh; the supervisor only looks at them BETWEEN trainer processes and NEVER kills a running trainer):
#   HOLD  present before a (re)start of the trainer -> wait (poll 30 s) until GO appears or HOLD is removed.   STOP present when the trainer exits -> no restart, supervisor exits.
#   WINDDOWN / LR_SCALE are read by the RL trainer itself (rl/run_controls.py). The SFT run has no wind-down: `stop` + `convert` from the last `save_every` checkpoint.
# Disk is the scarce resource (7 GB free): `$RUN/last` is a SYMLINK into /dev/shm/ab1/<name>/last (3.2 GB, rewritten through a .tmp copy at every save), made here.
# Environment overrides: AB TREE CFG RUN TAG PY MIN_FREE_GIB MIN_DISK_GIB MIN_SHM_GIB POLL TRAINER_CMD FORCE=1 (skip the live-run guard)
set -uo pipefail
AB="${AB:-/workspace/jobs/ab1}"
TREE="${TREE:-$AB/gemma_mt}"
PY="${PY:-/workspace/venv_gemma/bin/python}"
POLL="${POLL:-30}"
MIN_DISK_GIB="${MIN_DISK_GIB:-3}"
MIN_SHM_GIB="${MIN_SHM_GIB:-10}"
RATE="${RATE:-1.19}"
LOGDIR="$AB/logs"
mkdir -p "$LOGDIR"

which="${2:-}"
case "$which" in
  sft)  CFG="${CFG:-$TREE/configs/ab1_sft_weak.yaml}"; RUN="${RUN:-$AB/runs/sft_weak}"; TAG="${TAG:-AB1_SFT}"; MIN_FREE_GIB="${MIN_FREE_GIB:-40}"
        LADDER=("16 2 false" "8 4 false" "4 8 false" "2 16 true"); MODULE="gemma_mt.train_sft"; KEYS=(micro_batch_size grad_accum_steps gradient_checkpointing);;
  weak) CFG="${CFG:-$TREE/configs/rl_ablation_weak.yaml}"; RUN="${RUN:-$AB/runs/ab1_weak}"; TAG="${TAG:-AB1_W}"; MIN_FREE_GIB="${MIN_FREE_GIB:-62}"
        LADDER=("24576 128" "12288 64" "6144 32" "3072 16"); MODULE="gemma_mt.rl.train_rl"; KEYS=(micro_batch_tokens micro_batch_seqs);;
  full) CFG="${CFG:-$TREE/configs/rl_ablation_full.yaml}"; RUN="${RUN:-$AB/runs/ab1_full}"; TAG="${TAG:-AB1_F}"; MIN_FREE_GIB="${MIN_FREE_GIB:-62}"
        LADDER=("24576 128" "12288 64" "6144 32" "3072 16"); MODULE="gemma_mt.rl.train_rl"; KEYS=(micro_batch_tokens micro_batch_seqs);;
  *)    : ;;
esac
LOG="$LOGDIR/${TAG:-ab1}.log"
PIDFILE="${RUN:-$AB}/supervisor.pid"
say() { echo "$(date -u +%FT%TZ) [launch-${TAG:-ab1}] $*" | tee -a "$LOGDIR/launch_ab1.log"; }

env_setup() {
  [ -f /workspace/sft_build/env.sh ] && source /workspace/sft_build/env.sh
  export PYTHONPATH="$TREE:/workspace/sft_build/pylibs" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
  export VLLM_CACHE=/workspace/sft_build/vllm_cache
  export VLLM_CACHE_ROOT=$VLLM_CACHE/vllm TRITON_CACHE_DIR=$VLLM_CACHE/triton TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE/inductor XDG_CACHE_HOME=$VLLM_CACHE/xdg VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
}
alive() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; }

prepare_dirs() {   # $RUN/last -> /dev/shm (idempotent); never replaces a real directory
  mkdir -p "$RUN" "/dev/shm/ab1/$(basename "$RUN")/last"
  if [ ! -e "$RUN/last" ] && [ ! -L "$RUN/last" ]; then ln -s "/dev/shm/ab1/$(basename "$RUN")/last" "$RUN/last"; fi
}

preflight() {      # prints problems, returns their number
  local bad=0 free disk shm
  [ -f "$CFG" ] || { say "PREFLIGHT: config $CFG not found"; bad=$((bad+1)); }
  if [ "$which" = sft ]; then
    [ -f "$AB/data/packed_fwd/train/tokens.u32.bin" ] && [ -f "$AB/data/packed_fwd/val/tokens.u32.bin" ] || { say "PREFLIGHT: $AB/data/packed_fwd is missing (scripts/ab1_build_weak_sft_data.py)"; bad=$((bad+1)); }
    ls /workspace/sft_build/hf/hub/models--google--gemma-3-270m-it/snapshots/*/model.safetensors >/dev/null 2>&1 || { say "PREFLIGHT: google/gemma-3-270m-it is not in the HF cache"; bad=$((bad+1)); }
  else
    local m; [ "$which" = weak ] && m="$AB/models/sft_weak" || m="$AB/models/sft_final"
    [ -f "$m/model.safetensors" ] || { say "PREFLIGHT: start model $m/model.safetensors is missing"; bad=$((bad+1)); }
    for p in "$AB/data/rl_prompts_ab1/afr.parquet" "$TREE/calibration/shaping_flores_dev_v7.json" /workspace/sft_build/bench/bench_dev_clean.jsonl; do
      [ -e "$p" ] || { say "PREFLIGHT: missing $p"; bad=$((bad+1)); }
    done
    if [ -f "$RUN/last/trainer_state.pt" ]; then say "NOTE: $RUN/last/trainer_state.pt exists: the run RESUMES (resume: true); remove $RUN/last/* for a fresh start"; fi
  fi
  disk=$(df -P --block-size=1G "$AB" | awk 'NR==2{print $4}')
  [ "${disk:-0}" -ge "$MIN_DISK_GIB" ] || { say "PREFLIGHT: only ${disk} GiB of disk free (need $MIN_DISK_GIB)"; bad=$((bad+1)); }
  shm=$(df -P --block-size=1G /dev/shm | awk 'NR==2{print $4}')
  [ "${shm:-0}" -ge "$MIN_SHM_GIB" ] || { say "PREFLIGHT: only ${shm} GiB free in /dev/shm (need $MIN_SHM_GIB: last/ 3.2 GB + .tmp copy + the vLLM weight sync)"; bad=$((bad+1)); }
  if command -v nvidia-smi >/dev/null 2>&1; then
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); free=$((free / 1024))
    [ "$free" -ge "$MIN_FREE_GIB" ] || { say "PREFLIGHT: only $free GiB of GPU memory free (need $MIN_FREE_GIB)"; bad=$((bad+1)); }
  fi
  if [ "${FORCE:-0}" != 1 ] && pgrep -f "gemma_mt.rl.train_rl --config /workspace/jobs/r7" >/dev/null 2>&1; then
    say "PREFLIGHT: the live run (/workspace/jobs/r7, final2) is still training: wind it down first (or FORCE=1 if the owner has said so)"; bad=$((bad+1))
  fi
  return $bad
}

supervise() {
  local shift_args=("$@") idx=0 starts=() now keep t a b c rc
  env_setup; prepare_dirs
  cd "$TREE" || exit 2
  echo $$ > "$PIDFILE"; trap 'rm -f "$PIDFILE"' EXIT
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

do_convert() {
  env_setup
  local st="$AB/runs/sft_weak/last/trainer_state.pt"
  [ -f "$st" ] || { say "convert: $st missing"; exit 3; }
  [ -f "$AB/runs/sft_weak/summary.json" ] || say "WARNING: runs/sft_weak/summary.json is missing: the state is an intermediate checkpoint"
  CUDA_VISIBLE_DEVICES= "$PY" "$TREE/scripts/interpolate_weights.py" --init "$AB/models/sft_final" --sft "$st" --alpha 1.0 --out "$AB/models/sft_weak" 2>&1 | tail -3
  [ -f "$AB/models/sft_weak/model.safetensors" ] && say "convert: wrote $AB/models/sft_weak ($(du -h "$AB/models/sft_weak/model.safetensors" | cut -f1))"
}

do_eval() {   # FLORES+ devtest, 45 languages, greedy, 768 tokens, like reports/final_model_per_language (a1 pipeline); no AfriCOMET
  local model="$1" name="$2" util="${3:-0.1}"
  env_setup
  export CONTAM_DIR=/workspace/sft_build/bench/test_sets/_contamination OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BENCHMARKS=flores_plus A1_JOB=1
  OUT="$AB/eval/$name" GPU_UTIL="$util" GPU_MIN_FREE_GIB=14 GPU_WAIT=3600 TOTAL_TIMEOUT=5400 ROOT=/workspace/sft_build nice -n 10 \
    bash "$TREE/scripts/eval_sft_benchmarks.sh" "$model" --skip-base --no-comet --max-new-tokens 768 ${EVAL_EXTRA:-} > "$LOGDIR/eval_$name.log" 2>&1
  say "eval $name: $( [ -f "$AB/eval/$name/EVAL_DONE" ] && echo done || echo FAILED/partial ) -> $AB/eval/$name"
}

do_quick() {
  local model="$1" name="$2"
  env_setup
  nice -n 19 "$PY" -m gemma_mt.eval.sft_benchmarks --sft-model "$model" --skip-base --out "$AB/eval/quick_$name" --bench-dir /workspace/sft_build/bench --benchmarks flores_plus \
    --languages ${LANGS:-afr swh hau yor kin wol fon kam amh tir} --max-per-language "${N:-100}" --max-new-tokens 768 --sampler hf --hf-batch-size "${BS:-50}" --no-comet \
    --extra-libs /workspace/sft_build/pylibs --gpu-min-free-gib 4 --gpu-wait 120 --dtype bfloat16 --total-timeout 3300 > "$LOGDIR/quick_$name.log" 2>&1
}

status_one() {
  local w="$1" r tag
  case "$w" in sft) r="$AB/runs/sft_weak"; tag=AB1_SFT;; weak) r="$AB/runs/ab1_weak"; tag=AB1_W;; full) r="$AB/runs/ab1_full"; tag=AB1_F;; esac
  echo "== $w ($r)"
  if [ -f "$r/supervisor.pid" ] && kill -0 "$(cat "$r/supervisor.pid" 2>/dev/null)" 2>/dev/null; then echo "supervisor: running, pid $(cat "$r/supervisor.pid")"; else echo "supervisor: not running"; fi
  for f in HOLD GO STOP WINDDOWN WINDDOWN.state WINDDOWN.done LR_SCALE summary.json "DONE_$tag"; do [ -e "$r/$f" ] && echo "$f: $(head -c 80 "$r/$f" 2>/dev/null | tr '\n' ' ')"; done
  [ -f "$r/metrics.jsonl" ] && "$PY" - "$r/metrics.jsonl" "$w" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
tr = [r for r in rows if "t_step" in r or "reward" in r or "loss" in r]
ev = [r for r in rows if "bench/chrf_macro" in r]
if tr:
    r = tr[-1]
    keys = ("step", "reward", "kl", "finished_rate", "completion_len", "lr", "t_step", "loss", "grad_norm", "kl_guard_lr_scale", "lr_file_scale")
    print("last train row:", {k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in keys if k in r})
    ts = [x["t_step"] for x in tr[-30:] if "t_step" in x]
    if ts: print("median t_step of the last %d steps: %.1f s" % (len(ts), sorted(ts)[len(ts)//2]))
if ev:
    print("dev chrF++ (19 languages x 40, 256 tokens): " + "  ".join("%s@%s" % (round(r["bench/chrf_macro"], 2), r["step"]) for r in ev[-6:]) + "   finished " + str(round(ev[-1].get("bench/finished_rate", float("nan")), 4)))
PYEOF
  tail -n 2 "$LOGDIR/$tag.log" 2>/dev/null | cut -c1-220
}

cmd="${1:-}"; [ $# -gt 0 ] && shift
case "$cmd" in
  start)
    [ -n "${CFG:-}" ] || { echo "usage: $0 start sft|weak|full [k=v ...]"; exit 2; }
    shift   # the arm name
    alive && { say "supervisor already running (pid $(cat "$PIDFILE")): refusing to start a second one"; exit 4; }
    [ -f "$RUN/STOP" ] && { say "STOP file present in $RUN: remove it first"; exit 5; }
    env_setup; prepare_dirs; preflight || { say "preflight failed (see above); nothing started"; exit 3; }
    setsid nohup bash "$0" _supervise "$which" "$@" > /dev/null 2>&1 < /dev/null &
    sleep 2; say "supervisor started (pidfile $PIDFILE); log $LOG; metrics $RUN/metrics.jsonl";;
  _supervise) shift; supervise "$@";;
  check)
    [ -n "${CFG:-}" ] || { echo "usage: $0 check sft|weak|full"; exit 2; }
    env_setup; prepare_dirs
    if preflight; then
      pycode=""
      if [ "$which" != sft ]; then
        pycode="import sys
sys.path.insert(0, '$TREE')
from gemma_mt.rl.train_rl import RLConfig, apply_overrides, validate_config
from gemma_mt.utils import dataclass_from_dict, load_yaml, parse_overrides
cfg = dataclass_from_dict(RLConfig, apply_overrides(load_yaml('$CFG'), parse_overrides(['out_dir=$RUN'])))
validate_config(cfg)
print('yaml loads strictly and validates: steps', cfg.steps, 'lr', cfg.learning_rate, 'group', cfg.group_size, 'curriculum', cfg.curriculum_steps, 'export', cfg.export_steps, 'start', cfg.model_name)"
      else
        pycode="import sys
sys.path.insert(0, '$TREE')
from gemma_mt.train_sft import SFTConfig
from gemma_mt.utils import dataclass_from_dict, load_yaml
cfg = dataclass_from_dict(SFTConfig, load_yaml('$CFG'))
print('yaml loads strictly: max_steps', cfg.max_steps, 'lr', cfg.learning_rate, 'optimizer', cfg.optimizer, 'data', cfg.data_dir)"
      fi
      if ! out=$(cd "$TREE" && CUDA_VISIBLE_DEVICES= "$PY" -c "$pycode" 2>&1); then echo "$out" | tail -4; say "config check FAILED"; exit 3; fi
      echo "$out" | tail -1
      say "preflight OK"; exit 0
    fi
    exit 3;;
  status)
    env_setup
    if [ -z "$which" ] || [ "$which" = all ]; then for w in sft weak full; do status_one $w; done; else status_one "$which"; fi
    df -h "$AB" | tail -1; df -h /dev/shm | tail -1; command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    tail -n 3 "$LOGDIR/launch_ab1.log" 2>/dev/null;;
  winddown) echo "${3:-50}" > "$RUN/WINDDOWN"; say "WINDDOWN ${3:-50} written";;
  lr_scale) [ -n "${3:-}" ] || { echo "usage: lr_scale weak|full X"; exit 2; }; echo "$3" > "$RUN/LR_SCALE"; say "LR_SCALE $3 written";;
  hold) mkdir -p "$RUN"; touch "$RUN/HOLD"; say "HOLD set";;
  go)   mkdir -p "$RUN"; touch "$RUN/GO"; say "GO set";;
  stop) mkdir -p "$RUN"; touch "$RUN/STOP"; say "STOP set (no restart after the current trainer exits; the running trainer is NOT killed)";;
  convert) do_convert;;
  eval)  do_eval "${1:?model dir}" "${2:?name}" "${3:-0.1}";;
  quick) TAG=AB1_Q; do_quick "${1:?model dir}" "${2:?name}";;
  *) sed -n 2,19p "$0"; exit 2;;
esac
