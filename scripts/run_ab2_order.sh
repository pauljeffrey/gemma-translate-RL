#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# AB2 run order (cheap RL-vs-SFT-start ablation + continued-SFT controls). The owner approved running it as soon as the final run ends: ONE command starts everything.
#
#   bash scripts/run_ab2_order.sh go          # = start the chain in the background AND create $AB/ORDER_GO; the chain then waits for /workspace/jobs/r7/runs/final2/summary.json (and for
#                                              # the live trainer to be gone) before the first GPU job and before the budget clock starts. Log: $AB/logs/order.log
#   bash scripts/run_ab2_order.sh start       # the same without ORDER_GO (nothing GPU-related happens until `touch $AB/ORDER_GO`)
#   bash scripts/run_ab2_order.sh status      # what the chain is doing, the budget clock, the control files, all arms
#   bash scripts/run_ab2_order.sh dry-run     # prints the plan and checks every precondition that can be checked now (SKIP_GPU), starts and writes nothing
#   bash scripts/run_ab2_order.sh s1only      # after a chain that skipped S1 / S1P (no presigned URL at the time): build the data (needs $AB/S1_URL), set N, run S1 + S1P, evaluate, analyse
#   touch $AB/ORDER_STOP                       # no further job is started (running trainers are NOT killed; use launch_ab2.sh winddown / stop)
#
# S1 / S1P need the packed SFT set as a stream: put a presigned URL of  gemma_mt/data/gemma_packed_fwd_sft_windows.tar.zst  into the file $AB/S1_URL BEFORE `go` (the chain builds the data on the CPU
# while the weak SFT trains). Without it the chain runs everything else and writes ORDER_NEEDS_S1_URL (then: put the URL in and run `s1only`).
#
# Order (cheap, decisive things first; GPU hours in scripts/ab2_cost.py):
#   0. CPU, immediately: S1 data build (if S1_URL), config checks.        1. weak SFT (3,500 steps, ~2.7 h)  || eval F0
#   2. convert -> models/sft_weak ; arm F (RL from SFT-final, 49 steps) || eval W0 ; GATE on W0 (P1 gap >= 3 chrF++, P3 finished >= 60%)       3. arm W || eval F_RL
#   4. ab2_set_n.py (N of S1 from the measured F job) ; S1 || eval W_RL ; convert ; S1P || eval W_S1 ; convert ; eval W_S1P       5. ab2_analyze.py -> results/ (ablation_results.json, decision.json, ab2_report.md)
# Budget clock: starts when the final run is gone. At CAP_USD (7.8 = 9 / 1.15) no OPTIONAL job (S1, S1P and their evaluations) is started any more; at HARD_USD (9.0) ORDER_STOP and WINDDOWN 5 on a running
# RL arm. The chain never kills a trainer. MOCK=1 (tests): trainers are stubs, evaluations / conversions create empty markers, waits are short.
set -uo pipefail
AB="${AB:-/workspace/jobs/ab2}"
TREE="${TREE:-$AB/gemma_mt}"
L="bash $TREE/scripts/launch_ab2.sh"
PYV="${PYV:-/workspace/venv_gemma/bin/python}"
LIVE_DIR="${LIVE_DIR:-/workspace/jobs/r7}"
STEPS="${STEPS:-49}"
RATE="${RATE:-1.19}"; CAP_USD="${CAP_USD:-7.8}"; HARD_USD="${HARD_USD:-9.0}"
SLEEP="${SLEEP:-30}"; POLLRUN="${POLLRUN:-60}"
LOG="$AB/logs/order.log"; mkdir -p "$AB/logs" "$AB/results"
say() { echo "$(date -u +%FT%TZ) [order] $*" | tee -a "$LOG"; }
stopped() { [ -f "$AB/ORDER_STOP" ]; }
nomore() { [ -f "$AB/ORDER_NOMORE" ] || stopped; }
wait_file() { while [ ! -e "$1" ]; do stopped && { say "ORDER_STOP: leaving"; exit 0; }; sleep "$SLEEP"; done; }
live_gone() { [ -f "$LIVE_DIR/runs/final2/summary.json" ] && ! pgrep -f "gemma_mt.rl.train_rl --config $LIVE_DIR" > /dev/null 2>&1; }
spent() { python3 - "$AB/ORDER_T0" "$RATE" <<'PY'
import sys, time
t0 = float(open(sys.argv[1]).read()); print("%.2f" % ((time.time() - t0) / 3600 * float(sys.argv[2])))
PY
}
budget_guard() {
  local did=0 hard=0 s
  while [ ! -f "$AB/ORDER_DONE" ]; do
    s=$(spent)
    if [ "$did" = 0 ] && python3 -c "import sys; sys.exit(0 if float('$s') >= $CAP_USD else 1)"; then touch "$AB/ORDER_NOMORE"; say "BUDGET: \$$s >= \$$CAP_USD: no optional job (S1 / S1P) will be started"; did=1; fi
    if [ "$hard" = 0 ] && python3 -c "import sys; sys.exit(0 if float('$s') >= $HARD_USD else 1)"; then
      for a in weak full; do [ -d "$AB/runs/ab2_$a" ] && [ ! -f "$AB/runs/ab2_$a/summary.json" ] && echo 5 > "$AB/runs/ab2_$a/WINDDOWN"; done
      touch "$AB/ORDER_STOP"; say "BUDGET: \$$s >= \$$HARD_USD: WINDDOWN 5 on a running arm, ORDER_STOP"; hard=1
    fi
    sleep "$((SLEEP * 4))"
  done
}
reclaim() {   # disk is scarce (a few GB): remove EVALUATED snapshots / bf16 control models of THIS ablation only when less than 3 GiB is free; KEEP_SNAPS=1 never deletes
  [ "${KEEP_SNAPS:-0}" = 1 ] && return 0
  local free d n pair; free=$(df -P --block-size=1G "$AB" | awk 'NR==2{print $4}')
  [ "${free:-9}" -ge 3 ] && return 0
  for pair in "runs/ab2_full/snap_step$STEPS:F_RL" "runs/ab2_weak/snap_step$STEPS:W_RL" "models/w_s1:W_S1" "models/w_s1p:W_S1P"; do
    d="${pair%%:*}"; n="${pair##*:}"
    if [ -d "${AB:?}/${d:?}" ] && [ -f "${AB:?}/eval/${n:?}/EVAL_DONE" ]; then rm -r -- "${AB:?}/${d:?}"; say "disk: $free GiB free -> removed the evaluated $d"; fi
  done
}
run_and_wait() {   # $1 = sft|full|weak|s1|s1p
  reclaim
  $L start "$1" >> "$LOG" 2>&1 || { say "start $1 FAILED (preflight)"; touch "$AB/ORDER_FAILED"; exit 3; }
  local run tag
  case "$1" in sft) run=sft_weak; tag=AB2_SFT;; weak) run=ab2_weak; tag=AB2_W;; full) run=ab2_full; tag=AB2_F;; s1) run=sft_s1; tag=AB2_S1;; s1p) run=sft_s1pair; tag=AB2_S1P;; esac
  while [ ! -f "$AB/runs/$run/DONE_$tag" ]; do
    [ -f "$AB/logs/GAVE_UP_$tag" ] && { say "$1: supervisor gave up"; touch "$AB/ORDER_FAILED"; exit 4; }
    [ -f "$AB/runs/$run/STOP" ] && { say "$1: STOP file"; exit 0; }
    sleep "$POLLRUN"
  done
  say "$1 finished"
}
ev() { if [ "${MOCK:-0}" = 1 ]; then mkdir -p "$AB/eval/$2"; touch "$AB/eval/$2/EVAL_DONE"; say "MOCK eval $2 ($1)"; else $L eval "$1" "$2" 0.1 >> "$LOG" 2>&1; fi; }
conv() { if [ "${MOCK:-0}" = 1 ]; then case "$1" in sft) o=sft_weak;; s1) o=w_s1;; s1p) o=w_s1p;; esac; mkdir -p "$AB/models/$o"; touch "$AB/models/$o/model.safetensors"; say "MOCK convert $1"; else $L convert "$1" >> "$LOG" 2>&1; fi; }
EVPIDS=()
ev_bg() { ev "$@" & EVPIDS+=("$!"); }          # background evaluation; waitev waits for THESE only (never for the budget guard)
waitev() { local p; for p in "${EVPIDS[@]:-}"; do [ -n "$p" ] && wait "$p"; done; EVPIDS=(); }
s1_data_ready() { [ -f "$AB/data/packed_s1/BUILD_DONE" ]; }
build_s1_bg() {   # CPU/network only; idempotent
  s1_data_ready && return 0
  [ -f "$AB/S1_URL" ] || return 1
  ( set -o pipefail; curl -sSL --retry 3 "$(cat "$AB/S1_URL")" | zstd -dc | PYTHONPATH="$TREE" nice -n 10 "$PYV" "$TREE/scripts/ab2_build_s1_data.py" --out "$AB/data/packed_s1" --pair-out "$AB/data/packed_s1_pair" \
      --from-step 3500 --steps 4100 > "$AB/logs/build_s1.log" 2>&1 && touch "$AB/data/packed_s1/BUILD_DONE" && say "S1 data built" || say "S1 data build FAILED (see logs/build_s1.log)" ) &
  say "S1 data build started in the background (CPU)"
}
w_gate() {   # P1 / P3 of DECISION_RULE.txt from the FLORES+ no_pair rows of F0 and W0
  "$PYV" - "$AB/eval/F0/sys.jsonl.gz" "$AB/eval/W0/sys.jsonl.gz" <<'PY'
import gzip, json, sys
def rows(p):
    d = {}
    for l in gzip.open(p, "rt", encoding="utf-8"):
        r = json.loads(l)
        if r["b"] == "flores_plus" and r.get("pair_seen") is False and r["iso"] in ("yor", "kin", "orm", "lin"):
            d[(r["iso"], r["id"])] = r
    return d
try:
    f, w = rows(sys.argv[1]), rows(sys.argv[2])
    langs = ("yor", "kin", "orm", "lin")
    m = lambda d, k: sum(sum(float(r[k]) for kk, r in d.items() if kk[0] == l) / max(1, sum(1 for kk in d if kk[0] == l)) for l in langs) / 4
    gap = m(f, "chrf") - m(w, "chrf"); fin = 100 * m(w, "fin")
    print("F0 - W0 = %.2f chrF++ (sentence mean; P1 needs >= 3), W0 finished %.1f%% (P3 needs >= 60)" % (gap, fin)); sys.exit(0 if (gap >= 3 and fin >= 60) else 1)
except Exception as e:
    print("gate unreadable:", e); sys.exit(2)
PY
}
case "${1:-}" in
  go) touch "$AB/ORDER_GO"; setsid nohup bash "$0" _chain > /dev/null 2>&1 < /dev/null & echo "chain started AND released (ORDER_GO); it waits for $LIVE_DIR/runs/final2/summary.json; log $LOG"; [ -f "$AB/S1_URL" ] || echo "NOTE: $AB/S1_URL is missing: S1 / S1P will be skipped unless the file appears before the weak SFT ends";;
  start) setsid nohup bash "$0" _chain > /dev/null 2>&1 < /dev/null & echo "chain started; log $LOG; waiting for $AB/ORDER_GO";;
  status)
    ls "$AB"/ORDER_* 2>/dev/null; [ -f "$AB/ORDER_T0" ] && echo "spent (clock since the final run ended): \$$(spent)"; tail -n 10 "$LOG" 2>/dev/null; $L status all;;
  dry-run)
    echo "PLAN (nothing is started):"; sed -n '/^# Order/,/^# Budget/p' "$0" | sed 's/^# //'
    echo; echo "preconditions checkable now:"
    for a in sft full weak s1 s1p; do printf '  %-5s ' $a; SKIP_GPU=1 $L check $a 2>&1 | tail -n 2 | tr '\n' ' '; echo; done
    echo; [ -f "$AB/S1_URL" ] && echo "S1_URL: present" || echo "S1_URL: MISSING (S1 / S1P need it)"
    [ -f "$LIVE_DIR/runs/final2/summary.json" ] && echo "final run: finished" || echo "final run: not finished (the chain waits)"
    $PYV "$TREE/scripts/ab2_cost.py" | head -n 12;;
  s1only)
    say "s1only: building / waiting for the S1 data"
    build_s1_bg; while ! s1_data_ready; do [ -f "$AB/S1_URL" ] || { say "no $AB/S1_URL"; exit 2; }; stopped && exit 0; sleep "$SLEEP"; done
    [ -f "$AB/models/sft_weak/model.safetensors" ] || { say "models/sft_weak is missing: the main chain has not converted the weak SFT"; exit 2; }
    "$PYV" "$TREE/scripts/ab2_set_n.py" --write >> "$LOG" 2>&1
    run_and_wait s1; conv s1; ev_bg "$AB/models/w_s1" W_S1; run_and_wait s1p; conv s1p; ev "$AB/models/w_s1p" W_S1P; waitev
    $PYV "$TREE/scripts/ab2_analyze.py" --eval-dir "$AB/eval" --runs-dir "$AB/runs" --out "$AB/results" >> "$LOG" 2>&1; say "s1only done";;
  _chain)
    say "chain started (MOCK=${MOCK:-0}); waiting for $AB/ORDER_GO"
    if [ -f "$AB/S1_URL" ]; then build_s1_bg; else say "no S1_URL yet"; fi
    wait_file "$AB/ORDER_GO"
    sha256sum "$AB/DECISION_RULE.txt" > "$AB/results/DECISION_RULE.sha256" 2>/dev/null; say "DECISION_RULE.txt $(cut -c1-16 "$AB/results/DECISION_RULE.sha256")"
    while ! live_gone; do say "final run still training / no summary.json: waiting (the budget clock has not started)"; stopped && exit 0; sleep "$((SLEEP * 4))"; done
    date +%s > "$AB/ORDER_T0"; say "the final run is gone: budget clock started"; budget_guard &
    # 1. weak SFT (|| eval F0)
    ev_bg "$AB/models/sft_final" F0
    run_and_wait sft; stopped && exit 0
    conv sft
    [ -f "$AB/models/sft_weak/model.safetensors" ] || { say "convert FAILED"; touch "$AB/ORDER_FAILED"; exit 5; }
    # 2. arm F (|| eval W0), gate
    ev_bg "$AB/models/sft_weak" W0
    run_and_wait full; stopped && exit 0
    waitev
    if [ "${MOCK:-0}" != 1 ]; then w_gate > "$AB/W_GATE.txt" 2>&1; g=$?; else echo "MOCK gate" > "$AB/W_GATE.txt"; g=0; fi
    say "W0 gate: $(cat "$AB/W_GATE.txt") (rc=$g)"
    if [ "$g" != 0 ]; then say "GATE FAILED: holding before arm W (decide; touch $AB/ORDER_W_OK to continue)"; touch "$AB/ORDER_HOLD_W_GATE"; wait_file "$AB/ORDER_W_OK"; fi
    # 3. arm W (|| eval F_RL)
    ev_bg "$AB/runs/ab2_full/snap_step$STEPS" F_RL
    run_and_wait weak; stopped && exit 0
    ev_bg "$AB/runs/ab2_weak/snap_step$STEPS" W_RL
    # 4. controls (optional: skipped when the data is missing or the budget cap is reached)
    if ! s1_data_ready; then build_s1_bg; for i in 1 2 3; do s1_data_ready && break; sleep "$SLEEP"; done; fi
    if nomore; then say "budget cap / stop: the controls are skipped"; touch "$AB/ORDER_SKIPPED_CONTROLS"
    elif ! s1_data_ready; then say "S1 data not available (no / failed S1_URL): S1 and S1P skipped; later: put the URL into $AB/S1_URL and run: bash scripts/run_ab2_order.sh s1only"; touch "$AB/ORDER_NEEDS_S1_URL"
    else
      [ "${MOCK:-0}" = 1 ] || "$PYV" "$TREE/scripts/ab2_set_n.py" --write >> "$LOG" 2>&1
      run_and_wait s1; stopped && exit 0; conv s1
      ev_bg "$AB/models/w_s1" W_S1
      if nomore; then say "budget cap reached after S1: S1P skipped"; else run_and_wait s1p; conv s1p; ev "$AB/models/w_s1p" W_S1P; fi
    fi
    waitev
    # 5. analysis
    if [ "${MOCK:-0}" != 1 ]; then $PYV "$TREE/scripts/ab2_analyze.py" --eval-dir "$AB/eval" --runs-dir "$AB/runs" --out "$AB/results" >> "$LOG" 2>&1; fi
    say "ALL DONE: $AB/results/ablation_results.json , decision.json , ab2_report.md ; spent \$$(spent) (wall-clock since the final run ended)"; touch "$AB/ORDER_DONE";;
  *) sed -n 2,21p "$0"; exit 2;;
esac
