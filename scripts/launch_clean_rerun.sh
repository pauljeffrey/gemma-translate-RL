#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Launcher / supervisor of the RL stages of the CLEAN RERUN (agent C1, 2026-10-28). Derived from launch_final2.sh (same OOM ladder, restart logic, pidfile, HOLD / GO / STOP files)
# and adds: the stage sequence with a FRESH reference at every curriculum switch (new out_dir, model_name = the previous stage's final/ model, fresh optimizer - exactly as phase 2c restarted
# from the 2b stage-b64 model), the 49-step preflight shakedown, stage gates, the planned WINDDOWN of the final run, and the launch manifest that pins the pre-registered items.
#
#   bash scripts/launch_clean_rerun.sh check [stage|all]   # yaml validation (real config dataclasses), files, GPU, disk; no GPU use
#   bash scripts/launch_clean_rerun.sh manifest            # pin: rule / amendment sha256, protected index .3 manifest sha, config + code hashes, git commit, GPU type, start model sha
#   bash scripts/launch_clean_rerun.sh preflight           # 49 b64 steps, greedy member on, then the preflight gate (KL, grad norm, greedy loop rate, finished rate, dev chrF++, step time)
#   bash scripts/launch_clean_rerun.sh run A|B1|B2|B3|mixed|final   # one supervised stage, then its gate
#   bash scripts/launch_clean_rerun.sh general             # preflight (if not passed yet) -> A -> B1 -> B2 -> B3 -> mixed, stopping at the first failed gate; then STOPS: the science stage is built outside
#   bash scripts/launch_clean_rerun.sh status | winddown [stage] [N] | lr_scale [stage] X | hold|go|stop [stage]
#   bash scripts/launch_clean_rerun.sh disclose [stage]    # LibreTexts / source_dataset disclosure of a finished stage (needs the trainer interpreter)
#
# Order of the whole pipeline (pre-registered): SFT arms -> arm chosen by rule R5 -> [preflight -> A b64 -> B1 b256 -> B2 b512 -> B3 b1024 -> C mixed] -> science inputs -> science SFT + alpha sweep (outside this
# script) -> F final run (SCI_MODEL = the chosen science-stage model).
#
# Environment (defaults in brackets):
#   ROOT [/workspace/clean_rerun]   TREE [the gemma_mt tree this script lives in]   CFGDIR [$TREE/configs/clean_rerun, or clean_rerun_specialist when ARM is S1 or S2]
#   ARM [A0] A0 | S1 | S2 (the arm that won rule R5; S1/S2 use the 20-language specialist configs and pool directories of symlinks)
#   SFT_MODEL [$ROOT/models/sft_final] (HF dir of the winning arm's SFT-final)     SCI_MODEL [$ROOT/models/sci_sft_start] (final run only)
#   HUB_PREFIX [${HF_ORG}/gemma-mt-clean-rl, ...-spec for S1/S2]  -> hub_repo_id=$HUB_PREFIX-<stage>, hub_private=true (PRIVATE; main = best, latest = end-of-stage model). Set it EMPTY (HUB_PREFIX=) to push nothing (the models then live only on this box)
#   HANDOFF [final] final | best : which model of a finished stage starts the next one (2c used the stage-end model; `best` = the stage's best dev chrF++, the AB2 decline pattern)
#   STEPS_<stage> e.g. STEPS_A=560 : override a hard cap (also rewrites curriculum_steps)       A_FROM_PREFLIGHT [0] 1 = stage A resumes from the passed preflight (identical schedule, saves 49 steps)
#   STAGE_PLAN [full] full | b256 : b256 = the 385M plan (owner 2026-11-02): B2 (b512) and B3 (b1024) are NOT run, mixed starts from B1's final/ (stages: preflight A B1 mixed final)
#   STAGE_PLAN=b256+long, or EXTRA_STAGES=F2 : also allow the OPTIONAL stage F2 (final_long, science, b1024 + b2048; 385M plan only). It is never part of `general` / `all` and never starts by itself:
#       `run F2` is refused without one of the two, after a failed `final` gate and for ARM != A0; decide with `clean_rerun_tools.py f2-precheck` first, keep or discard with `f2-decide` (reports/rl_stage_plan_385m_b256_2026-11-02.md section 6)
#   LADDER_START [0] index of the first micro-batch step of the OOM ladder (the 385M plan starts at 1)   POOL_V3 [$ROOT/prompts/clean_v3] pool directory pinned in the manifest
#   FORCE [0] 1 = continue after a failed gate        OVERRIDE_PIN [0] 1 = run although a yaml changed after the manifest      ALLOW_GPU [0] 1 = not an RTX PRO 6000
#   PY [/workspace/venv_gemma/bin/python]  MIN_FREE_GIB [70]  MIN_DISK_GIB [14]  POLL [30]  PRUNE [0] (1 = delete last/trainer_state.pt of a stage after its gate passed)
#   PREREG_DIR  INDEX_MANIFEST  HELDOUT_SPLIT  WEAK_DIR  (manifest inputs; default: reports/data/... next to the tree)   PROMPTS_HUB_REVISION (commit of the prompts repo, for the manifest)
# Control files per stage directory (the supervisor only looks at them BETWEEN trainer processes; it NEVER kills a running trainer):
#   HOLD / GO / STOP as in launch_final2.sh;  WINDDOWN / LR_SCALE / LANG_WEIGHTS.json are read by the trainer itself.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="${TREE:-$(cd "$HERE/.." && pwd)}"
ROOT="${ROOT:-/workspace/clean_rerun}"
ARM="${ARM:-A0}"
case "$ARM" in A0) CFGDIR="${CFGDIR:-$TREE/configs/clean_rerun}";; S1|S2) CFGDIR="${CFGDIR:-$TREE/configs/clean_rerun_specialist}";; *) echo "ARM must be A0, S1 or S2"; exit 2;; esac
case "$ARM" in A0) HUB_DEFAULT="${HF_ORG}/gemma-mt-clean-rl";; *) HUB_DEFAULT="${HF_ORG}/gemma-mt-clean-rl-spec";; esac
HUB_PREFIX="${HUB_PREFIX-$HUB_DEFAULT}"      # 2026-10-29 (Amendment 4): default = the ${HF_ORG} organisation, private repos; an explicitly EMPTY HUB_PREFIX disables the push
RUNS="${RUNS:-$ROOT/runs}"
LOGDIR="${LOGDIR:-$ROOT/logs}"
MANIFEST_DIR="${MANIFEST_DIR:-$ROOT/manifest}"
MANIFEST="$MANIFEST_DIR/clean_rerun_manifest_${ARM}.json"
SFT_MODEL="${SFT_MODEL:-$ROOT/models/sft_final}"
SCI_MODEL="${SCI_MODEL:-$ROOT/models/sci_sft_start}"
PY="${PY:-/workspace/venv_gemma/bin/python}"
tools() { python3 "$HERE/clean_rerun_tools.py" "$@"; }            # stdlib only
tools_trainer() { "$PY" "$HERE/clean_rerun_tools.py" "$@"; }       # needs the trainer environment (validate, manifest, disclose)
HANDOFF="${HANDOFF:-final}"
POLL="${POLL:-30}"
MIN_FREE_GIB="${MIN_FREE_GIB:-70}"
MIN_DISK_GIB="${MIN_DISK_GIB:-14}"
LADDER=("49152 320" "24576 128" "12288 64" "6144 32" "3072 16")
PREREG_DIR="${PREREG_DIR:-$TREE/../reports/data/prereg_clean_2026-10-28}"
WEAK_DIR="${WEAK_DIR:-$TREE/../reports/data/weak_selection_2026-10-26}"
INDEX_MANIFEST="${INDEX_MANIFEST:-$ROOT/protected_index/protected_manifest.json}"
HELDOUT_SPLIT="${HELDOUT_SPLIT:-$ROOT/protected_index/heldout_split_cal_sel_v2.json}"
SPEC_LANGS="amh tir fon dyu kbp nus mos kam bam umb kea kmb vmw kab seh wol ewe luo tum yor"
mkdir -p "$RUNS" "$LOGDIR" "$MANIFEST_DIR"
say() { echo "$(date -u +%FT%TZ) [clean-rerun $ARM] $*" | tee -a "$LOGDIR/launch_clean_rerun.log"; }

file_of() { case "$1" in preflight) echo rl_00_preflight.yaml;; A) echo rl_A_b64.yaml;; B1) echo rl_B1_b256.yaml;; B2) echo rl_B2_b512.yaml;; B3) echo rl_B3_b1024.yaml;; mixed) echo rl_C_mixed.yaml;; final) echo rl_F_final.yaml;; F2) echo rl_F2_long.yaml;; *) echo "";; esac; }
prev_of() { case "$1" in A) echo preflight;; B1) echo A;; B2) echo B1;; B3) echo B2;; mixed) echo B3;; *) echo "";; esac; }
bin_of()  { case "$1" in preflight|A) echo b64;; B1) echo b256;; B2) echo b512;; B3) echo b1024;; *) echo "";; esac; }   # bash 3.2 (macOS) has no associative arrays
STAGES=(A B1 B2 B3 mixed)
case "${STAGE_PLAN:-full}" in b256+long) STAGE_PLAN=b256; EXTRA_STAGES="${EXTRA_STAGES:-} F2";; esac     # F2 is OFF unless asked for (STAGE_PLAN=b256+long or EXTRA_STAGES=F2)
EXTRA_STAGES="$(echo "${EXTRA_STAGES:-}" | tr ',' ' ' | xargs)"
for x in $EXTRA_STAGES; do [ "$x" = F2 ] || { echo "EXTRA_STAGES: only F2 exists (got '$x')"; exit 2; }; done
[ -z "$EXTRA_STAGES" ] || [ "${STAGE_PLAN:-full}" = b256 ] || { echo "EXTRA_STAGES=F2 needs STAGE_PLAN=b256 (the 385M plan)"; exit 2; }
if [ "${STAGE_PLAN:-full}" = b256 ]; then     # 385M plan, owner 2026-11-02: the RL context is capped at b256 (production translates sentence by sentence)
  prev_of() { case "$1" in A) echo preflight;; B1) echo A;; mixed) echo B1;; F2) echo final;; *) echo "";; esac; }
  STAGES=(A B1 mixed)
elif [ "${STAGE_PLAN:-full}" != full ]; then echo "STAGE_PLAN must be full, b256 or b256+long"; exit 2; fi

env_setup() {
  [ -f /workspace/sft_build/env.sh ] && source /workspace/sft_build/env.sh
  export PYTHONPATH="$TREE:/workspace/sft_build/pylibs" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export VLLM_CACHE=/workspace/sft_build/vllm_cache
  export VLLM_CACHE_ROOT=$VLLM_CACHE/vllm TRITON_CACHE_DIR=$VLLM_CACHE/triton TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE/inductor XDG_CACHE_HOME=$VLLM_CACHE/xdg VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
  export GEMMA_MT_LEXICON_DIR="${GEMMA_MT_LEXICON_DIR:-$ROOT/assets/lexicon}"
}
runname() { echo "$1"; }
rundir()  { echo "$RUNS/$(runname "$1")"; }
alive()   { local pf="$(rundir "$1")/supervisor.pid"; [ -f "$pf" ] && kill -0 "$(cat "$pf" 2>/dev/null)" 2>/dev/null; }
cfg_of()  { echo "$CFGDIR/$(file_of "$1")"; }
yaml_steps() { grep -m1 -E '^steps:' "$(cfg_of "$1")" | sed -E 's/^steps:[[:space:]]*([0-9]+).*/\1/'; }
steps_of() { local v="STEPS_$1"; echo "${!v:-$(yaml_steps "$1")}"; }
model_of() {   # the model a stage starts from (= its KL reference)
  case "$1" in
    preflight|A) echo "$SFT_MODEL";;
    final) echo "$SCI_MODEL";;
    *) local p; p="$(prev_of "$1")"; tools handoff --run "$(rundir "$p")" --mode "$HANDOFF" 2>/dev/null;;
  esac
}
gate_verdict() { local g="$(rundir "$1")/gate.json"; [ -f "$g" ] && python3 -c "import json,sys;print(json.load(open('$g'))['verdict'])" || echo NONE; }

build_specialist_pools() {   # ARM S1/S2: pool directories holding only the 20 specialist languages (symlinks; the loader walks the directory)
  [ "$ARM" = A0 ] && return 0
  local src dst l
  for pair in "clean_v3:clean_v3_specialist" "clean_sci_v2:clean_sci_v2_specialist"; do
    src="$ROOT/prompts/${pair%%:*}"; dst="$ROOT/prompts/${pair##*:}"
    [ -d "$src" ] || { say "specialist pool: $src missing"; return 1; }
    mkdir -p "$dst"
    for l in $SPEC_LANGS; do [ -e "$src/$l.parquet" ] && ln -sf "$src/$l.parquet" "$dst/$l.parquet" || say "WARNING: $src/$l.parquet missing"; done
  done
}

preflight_resources() {   # prints problems, returns their number
  local bad=0 free disk st="$1" run
  run="$(rundir "$st")"; mkdir -p "$run"
  [ -f "$(cfg_of "$st")" ] || { say "PREFLIGHT: config $(cfg_of "$st") not found"; bad=$((bad+1)); }
  disk=$(df -Pk "$run" 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
  [ "${disk:-0}" -ge "$MIN_DISK_GIB" ] || { say "PREFLIGHT: only ${disk:-?} GiB free under $run (need $MIN_DISK_GIB: trainer state 3.2 GB x2 during a save, final/ + best/ 1.1 GB each)"; bad=$((bad+1)); }
  if command -v nvidia-smi >/dev/null 2>&1; then
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); free=$((free / 1024))
    [ "$free" -ge "$MIN_FREE_GIB" ] || { say "PREFLIGHT: only $free GiB of GPU memory free (need $MIN_FREE_GIB)"; bad=$((bad+1)); }
  elif [ -z "${TRAINER_CMD:-}" ]; then say "PREFLIGHT: no nvidia-smi"; bad=$((bad+1)); fi
  return $bad
}

pin_ok() {   # the yaml must be the one that the manifest pinned
  [ "${OVERRIDE_PIN:-0}" = 1 ] && return 0
  [ -f "$MANIFEST" ] || { say "no manifest $MANIFEST: run 'manifest' first (it pins rule file, index, configs, git commit, GPU)"; return 1; }
  local f want got; f="$(file_of "$1")"
  want=$(python3 -c "import json;print(json.load(open('$MANIFEST'))['config_sha256'].get('$f',''))")
  got=$( (sha256sum "$(cfg_of "$1")" 2>/dev/null || shasum -a 256 "$(cfg_of "$1")") | cut -d' ' -f1)
  [ "$want" = "$got" ] || { say "PIN MISMATCH: $f changed after the manifest was written (manifest $want, file $got). Re-run 'manifest' deliberately or set OVERRIDE_PIN=1"; return 1; }
}

supervise() {   # args: stage [extra key=value ...]
  local st="$1"; shift
  local cfg run steps model idx="${LADDER_START:-0}" starts=() now keep t T S rc extra=() wd_pid="" hub=""
  cfg="$(cfg_of "$st")"; run="$(rundir "$st")"; steps="$(steps_of "$st")"; model="$(model_of "$st")"
  env_setup; cd "$TREE" || exit 2
  mkdir -p "$run"; echo $$ > "$run/supervisor.pid"
  trap "rm -f '$run/supervisor.pid'" EXIT
  rm -f "$run/summary.json"
  extra=("model_name=$model" "out_dir=$run" "steps=$steps")
  case "$st" in preflight|A|B1|B2|B3) extra+=("curriculum_steps={\"$(bin_of "$st")\": $steps}");; esac
  [ "$ARM" != A0 ] && case "$st" in final) extra+=("prompts_path=$ROOT/prompts/clean_sci_v2_specialist");; *) extra+=("prompts_path=$ROOT/prompts/clean_v3_specialist");; esac
  if [ "${HUB_FROM_YAML:-}" = 1 ]; then :     # 385M plan (owner 2026-11-03): repo names and branches come from the yamls (gemma-0.4B-translate-rl / -science-rl), no per-stage override
  elif [ -n "${HUB_PREFIX:-}" ] && [ "$st" != preflight ]; then hub="${HUB_PREFIX}-$(echo "$st" | tr 'A-Z' 'a-z')"; extra+=("hub_repo_id=$hub" "hub_private=true"); fi
  [ -n "${HUB_PREFIX:-}" ] || say "WARNING: HUB_PREFIX is not set: stage $st pushes nothing (the models exist only under $run; a box wipe loses them)"
  # planned WINDDOWN of the last N steps: since 2026-10-29 the TRAINER does it (planned_winddown_last in the yaml) so that an extension of the cap moves it; the external watcher is only started for a yaml without that field
  if [ "$st" = final ] && ! grep -Eq '^planned_winddown_last: *[1-9]' "$cfg"; then   # (a plateau stop may end the run earlier; the watcher never overrides an existing WINDDOWN)
    tools winddown --run "$run" --steps "$steps" --n "${WINDDOWN_LAST:-30}" > "$LOGDIR/winddown_watch_$st.log" 2>&1 &
    wd_pid=$!
    trap "rm -f '$run/supervisor.pid'; kill $wd_pid 2>/dev/null" EXIT
  fi
  if [ -f "$run/last/trainer_state.pt" ]; then say "stage $st: RESUME of an interrupted run (last/ exists; the KL-guard state is restored)"; else say "stage $st: FRESH START: reference = model_name = $model; fresh optimizer; new out_dir"; fi
  while [ ! -f "$run/summary.json" ]; do
    if [ -f "$run/STOP" ]; then say "STOP present: supervisor of $st exits"; exit 0; fi
    if [ -f "$run/HOLD" ]; then
      say "HOLD present for $st: waiting for GO"
      while [ -f "$run/HOLD" ] && [ ! -f "$run/GO" ] && [ ! -f "$run/STOP" ]; do sleep "$POLL"; done
      [ -f "$run/GO" ] && { rm -f "$run/HOLD" "$run/GO"; say "GO received"; }
      continue
    fi
    now=$(date +%s); keep=(); for t in "${starts[@]:-}"; do [ -n "$t" ] && [ $((now - t)) -lt 3600 ] && keep+=("$t"); done; starts=("${keep[@]:-}")
    if [ "${#keep[@]}" -ge 8 ]; then say "8 starts in the last hour: giving up on $st"; touch "$LOGDIR/RL_GAVE_UP_$st"; exit 1; fi
    starts+=("$now"); read -r T S < <(echo "${LADDER[$idx]}")
    say "trainer start [$st] ($cfg): micro_batch_tokens=$T micro_batch_seqs=$S steps=$steps ${extra[*]}"
    cp "$LOGDIR/rl_$st.log" "$LOGDIR/rl_${st}_prev_$(date +%s).log" 2>/dev/null
    if [ -n "${TRAINER_CMD:-}" ]; then   # tests / dry runs: replaces the trainer (gets RUN, STEPS, MODEL in its environment)
      RUN="$run" STEPS="$steps" MODEL="$model" bash -c "$TRAINER_CMD" > "$LOGDIR/rl_$st.log" 2>&1
    else
      "$PY" -m gemma_mt.rl.train_rl --config "$cfg" "${extra[@]}" micro_batch_tokens=$T micro_batch_seqs=$S "$@" > "$LOGDIR/rl_$st.log" 2>&1
    fi
    rc=$?
    if [ -f "$run/summary.json" ]; then say "stage $st finished (rc=$rc)"; break; fi
    if tail -n 80 "$LOGDIR/rl_$st.log" | grep -a -q "OutOfMemoryError"; then [ $idx -lt $((${#LADDER[@]}-1)) ] && idx=$((idx+1)); say "OOM -> ladder $idx"; else say "trainer exited rc=$rc: restarting"; fi
    [ -f "$run/STOP" ] && { say "STOP present after the trainer exited: supervisor of $st exits"; exit 0; }
    sleep "${RESTART_SLEEP:-20}"
  done
  return 0
}

run_stage() {   # one stage in the foreground, then its gate
  local st="$1"; shift
  [ -n "$(file_of "$st")" ] || { say "unknown stage $st"; return 2; }
  alive "$st" && { say "supervisor of $st already running: refusing a second one"; return 4; }
  [ -f "$(rundir "$st")/STOP" ] && { say "STOP file in $(rundir "$st"): remove it first"; return 5; }
  env_setup
  pin_ok "$st" || return 6
  if [ -n "$(prev_of "$st")" ] && ! { [ "$st" = A ] && [ "${SKIP_PREFLIGHT:-0}" = 1 ]; }; then
    local pv; pv="$(gate_verdict "$(prev_of "$st")")"
    if [ "$pv" = NONE ] || { [ "$pv" = FAIL ] && [ "${FORCE:-0}" != 1 ]; }; then say "previous stage $(prev_of "$st") has gate verdict $pv: not starting $st (FORCE=1 overrides a FAIL)"; return 7; fi
  fi
  if [ "$st" = A ] && [ "${A_FROM_PREFLIGHT:-0}" = 1 ] && [ ! -f "$(rundir A)/last/trainer_state.pt" ]; then
    local pv; pv="$(gate_verdict preflight)"
    [ "$pv" = PASS ] || [ "$pv" = PASS_WITH_WARNINGS ] || { say "A_FROM_PREFLIGHT=1 needs a passed preflight (verdict $pv)"; return 8; }
    mkdir -p "$(rundir A)" && cp -a "$(rundir preflight)/last" "$(rundir A)/last" && cp "$(rundir preflight)/metrics.jsonl" "$(rundir A)/metrics.jsonl" && say "stage A continues from the preflight state (step 49)"
  fi
  local m; m="$(model_of "$st")"
  [ -e "$m/model.safetensors" ] || ls "$m"/*.safetensors >/dev/null 2>&1 || { say "start model of $st not found at '$m'"; return 3; }
  [ -n "${TRAINER_CMD:-}" ] && say "TRAINER_CMD is set: DRY RUN, the real trainer is not started"
  preflight_resources "$st" || { say "resource preflight failed for $st; nothing started"; return 3; }
  build_specialist_pools || return 3
  supervise "$st" "$@"
  local rc=$?
  [ -f "$(rundir "$st")/summary.json" ] || return $rc
  tools gate --stage "$st" --run "$(rundir "$st")" --out "$(rundir "$st")/gate.json" | tee "$LOGDIR/gate_$st.log" | tail -n 40
  local v; v="$(gate_verdict "$st")"
  say "stage $st gate verdict: $v (report $(rundir "$st")/gate.json)"
  if [ "$v" = FAIL ] && [ "${FORCE:-0}" != 1 ]; then touch "$(rundir "$st")/GATE_FAILED"; return 20; fi
  [ "${PRUNE:-0}" = 1 ] && [ "$v" != FAIL ] && rm -f "$(rundir "$st")/last/trainer_state.pt" && say "pruned last/trainer_state.pt of $st"
  [ "$st" = mixed ] && { echo "$(handoff_dir mixed)" > "$RUNS/RL_GENERAL_DONE"; say "GENERAL RL DONE. Model: $(cat "$RUNS/RL_GENERAL_DONE"). Next (outside this script): science-stage inputs from THIS model, science SFT, alpha sweep on the S sets, then SCI_MODEL=<chosen> $0 run final"; }
  return 0
}
handoff_dir() { tools handoff --run "$(rundir "$1")" --mode "$HANDOFF"; }

cmd="${1:-}"; [ $# -gt 0 ] && shift
case "$cmd" in
  check)
    st="${1:-all}"; env_setup; bad=0
    for s in preflight "${STAGES[@]}" final $EXTRA_STAGES; do [ "$st" = all ] || [ "$st" = "$s" ] || continue; preflight_resources "$s" || bad=$((bad+$?)); done
    tools_trainer validate --root "$ROOT" --arm "$([ "$ARM" = A0 ] && echo a0 || echo specialist)" --configs "$CFGDIR" --check-files $([ "${STAGE_PLAN:-full}" = b256 ] && echo --plan b256_385m) $([ -n "$EXTRA_STAGES" ] && echo --extra-stages F2) || bad=$((bad+1))
    for p in "$SFT_MODEL" "$ROOT/prompts/clean_v3" "$ROOT/prompts/clean_sci_v2" "$PREREG_DIR/DECISION_RULE.txt" "$INDEX_MANIFEST"; do [ -e "$p" ] || { say "CHECK: missing $p"; bad=$((bad+1)); }; done
    [ "$bad" -eq 0 ] && { say "check OK"; exit 0; } || { say "check found $bad problem(s)"; exit 3; };;
  manifest)
    env_setup
    tools_trainer manifest --tree "$TREE" --configs "$CFGDIR" --out "$MANIFEST" --prereg-dir "$PREREG_DIR" --weak-dir "$WEAK_DIR" --index-manifest "$INDEX_MANIFEST" --heldout-split "$HELDOUT_SPLIT" \
      --start-model "$SFT_MODEL" --pool-v3 "${POOL_V3:-$ROOT/prompts/clean_v3}" --pool-sci "$ROOT/prompts/clean_sci_v2" --hash-pools --arm "$ARM" $([ "${ALLOW_GPU:-0}" = 1 ] && echo --allow-gpu);;
  preflight) run_stage preflight "$@";;
  run) st="${1:-}"; shift || true
    if [ "${STAGE_PLAN:-full}" = b256 ] && { [ "$st" = B2 ] || [ "$st" = B3 ]; }; then echo "stage $st is not part of STAGE_PLAN=b256 (b512 / b1024 are not trained in the 385M plan)"; exit 2; fi
    if [ "$st" = F2 ]; then
      case " $EXTRA_STAGES " in *" F2 "*) ;; *) echo "stage F2 is OPTIONAL and OFF: set STAGE_PLAN=b256+long (or EXTRA_STAGES=F2) after clean_rerun_tools.py f2-precheck said RUN"; exit 2;; esac
      [ "$ARM" = A0 ] || { echo "stage F2 exists for ARM=A0 only"; exit 2; }
      F2_POOL="${F2_POOL:-$ROOT/prompts/clean_sci_v2_long}"
      ls "$F2_POOL"/zz_b2048/*.parquet >/dev/null 2>&1 || { echo "F2 pool $F2_POOL has no zz_b2048/*.parquet: run scripts/clean_rerun_385m_prepare_f2_pool.sh (needs the decon.py-scanned b2048 science rows)"; exit 3; }
    fi
    run_stage "$st" "$@";;
  general)
    for s in preflight "${STAGES[@]}"; do
      v="$(gate_verdict "$s")"
      if [ "$v" = PASS ] || [ "$v" = PASS_WITH_WARNINGS ]; then say "$s already passed ($v): skipping"; continue; fi
      [ "$s" = preflight ] && [ "${SKIP_PREFLIGHT:-0}" = 1 ] && continue
      run_stage "$s" "$@" || { say "stopped at stage $s (rc=$?)"; exit 1; }
    done;;
  status)
    for s in preflight "${STAGES[@]}" final F2; do
      r="$(rundir "$s")"; [ -d "$r" ] || continue
      printf '%-9s gate=%-18s' "$s" "$(gate_verdict "$s")"; alive "$s" && printf ' RUNNING(pid %s)' "$(cat "$r/supervisor.pid")"
      for f in HOLD GO STOP WINDDOWN LR_SCALE summary.json; do [ -e "$r/$f" ] && printf ' %s' "$f"; done; echo
      tail -n 1 "$r/metrics.jsonl" 2>/dev/null | cut -c1-200
    done; tail -n 3 "$LOGDIR/launch_clean_rerun.log" 2>/dev/null;;
  winddown) s="${1:-final}"; echo "${2:-30}" > "$(rundir "$s")/WINDDOWN"; say "WINDDOWN ${2:-30} written for $s";;
  lr_scale) s="${1:-final}"; [ -n "${2:-}" ] || { echo "usage: lr_scale <stage> X"; exit 2; }; echo "$2" > "$(rundir "$s")/LR_SCALE"; say "LR_SCALE $2 written for $s";;
  hold) touch "$(rundir "${1:-final}")/HOLD"; say "HOLD set for ${1:-final}";;
  go) touch "$(rundir "${1:-final}")/GO"; say "GO set for ${1:-final}";;
  stop) touch "$(rundir "${1:-final}")/STOP"; say "STOP set for ${1:-final} (no restart after the current trainer exits; the running trainer is NOT killed)";;
  disclose)
    s="${1:-final}"; env_setup
    tools_trainer disclose --tree "$TREE" --config "$(cfg_of "$s")" --steps "$(python3 -c "import json;print(json.load(open('$(rundir "$s")/summary.json')).get('steps',0))")" --archive "$(rundir "$s")/archive/archive.sqlite" --out "$(rundir "$s")/disclosure.json";;
  *) sed -n 2,34p "$0"; exit 2;;
esac
