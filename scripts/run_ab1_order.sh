#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# AB1 run order (RL-vs-SFT-data ablation). NOTHING starts until the file $AB/ORDER_GO exists (created by the owner / lead once the live run is wound down).
#
#   bash scripts/run_ab1_order.sh start      # background chain (setsid nohup); log $AB/logs/order.log
#   bash scripts/run_ab1_order.sh status     # what the chain is doing, the budget clock, the control files
#   touch $AB/ORDER_GO                       # release;  touch $AB/ORDER_STOP  # no further step is started (running trainers are NOT killed; use launch_ab1.sh winddown)
#
# Order (GPU wall-clock ~10.7 h, ~$12.7 central / ~$17.2 high at 1.19/h; the cost table is in scripts/ab1_cost.py):
#   1. SFT-weak (3,500 steps, ~2.5 h)             in parallel: eval F0 (SFT-final, FLORES+ 45 languages, ~12 min, 10 GB)
#   2. convert -> models/sft_weak ; delete data/packed_fwd (2.3 GB of scarce disk) ; eval W0 in parallel with arm F ; GATE on W0 (finished >= 0.6 and macro chrF++ >= 15 = above the base model's 13.6, else the chain holds before arm W)
#   3. arm F  (SFT-final -> RL 400 steps = 100 b64 + 300 b256, ~3.5 h)   in parallel: eval W0
#   4. arm W  (SFT-weak  -> RL 400 steps, ~3.8 h)   in parallel: eval F_mid (step 200), F_end (step 400)
#   5. eval W_mid and W_end ; analysis (CPU) -> $AB/report/ab1_report.md
# Budget clock: starts when ORDER_GO exists AND no trainer of /workspace/jobs/r7 is running; at CAP_USD (default 17.0) both arms get a WINDDOWN of 30 steps; at HARD_USD (19.0) a WINDDOWN of 10; the chain never kills a trainer.
set -uo pipefail
AB="${AB:-/workspace/jobs/ab1}"
TREE="${TREE:-$AB/gemma_mt}"
L="bash $TREE/scripts/launch_ab1.sh"
MID="${MID:-200}"; END="${END:-400}"
RATE="${RATE:-1.19}"; CAP_USD="${CAP_USD:-17.0}"; HARD_USD="${HARD_USD:-19.0}"
LOG="$AB/logs/order.log"
say() { echo "$(date -u +%FT%TZ) [order] $*" | tee -a "$LOG"; }
stopped() { [ -f "$AB/ORDER_STOP" ]; }
wait_file() { while [ ! -e "$1" ]; do stopped && { say "ORDER_STOP: leaving"; exit 0; }; sleep 30; done; }
spent() { python3 - "$AB/ORDER_T0" "$RATE" <<'EOF'
import sys, time
t0 = float(open(sys.argv[1]).read()); print("%.2f" % ((time.time() - t0) / 3600 * float(sys.argv[2])))
EOF
}
budget_guard() {   # background: winddown both arms when the money is gone
  local did=0 hard=0 s
  while [ ! -f "$AB/ORDER_DONE" ]; do
    s=$(spent)
    if [ "$did" = 0 ] && python3 -c "import sys; sys.exit(0 if float('$s') >= $CAP_USD else 1)"; then
      for a in weak full; do [ -d "$AB/runs/ab1_$a" ] && [ ! -f "$AB/runs/ab1_$a/summary.json" ] && echo 30 > "$AB/runs/ab1_$a/WINDDOWN"; done; say "BUDGET: \$$s >= \$$CAP_USD: WINDDOWN 30 written to the running arm(s)"; did=1
    fi
    if [ "$hard" = 0 ] && python3 -c "import sys; sys.exit(0 if float('$s') >= $HARD_USD else 1)"; then
      for a in weak full; do [ -d "$AB/runs/ab1_$a" ] && [ ! -f "$AB/runs/ab1_$a/summary.json" ] && echo 10 > "$AB/runs/ab1_$a/WINDDOWN"; done; touch "$AB/ORDER_STOP"; say "BUDGET: \$$s >= \$$HARD_USD: WINDDOWN 10, ORDER_STOP"; hard=1
    fi
    sleep 120
  done
}
run_and_wait() {   # $1 = sft|weak|full, tag marker file
  $L start "$1" >> "$LOG" 2>&1 || { say "start $1 FAILED (preflight)"; touch "$AB/ORDER_FAILED"; exit 3; }
  local run tag
  case "$1" in sft) run=sft_weak; tag=AB1_SFT;; weak) run=ab1_weak; tag=AB1_W;; full) run=ab1_full; tag=AB1_F;; esac
  while [ ! -f "$AB/runs/$run/DONE_$tag" ]; do
    [ -f "$AB/logs/GAVE_UP_$tag" ] && { say "$1: supervisor gave up"; touch "$AB/ORDER_FAILED"; exit 4; }
    [ -f "$AB/runs/$run/STOP" ] && { say "$1: STOP file"; exit 0; }
    sleep 60
  done
  say "$1 finished"
}
case "${1:-}" in
  start) setsid nohup bash "$0" _chain > /dev/null 2>&1 < /dev/null & echo "chain started; log $LOG; waiting for $AB/ORDER_GO";;
  status)
    ls "$AB"/ORDER_* 2>/dev/null; [ -f "$AB/ORDER_T0" ] && echo "spent (clock since GO): \$$(spent)"; tail -n 8 "$LOG" 2>/dev/null; $L status all;;
  _chain)
    say "waiting for $AB/ORDER_GO"; wait_file "$AB/ORDER_GO"
    while pgrep -f "gemma_mt.rl.train_rl --config /workspace/jobs/r7" > /dev/null 2>&1; do say "live run still training: waiting (the budget clock has not started)"; stopped && exit 0; sleep 120; done
    date +%s > "$AB/ORDER_T0"; say "GO and the live run is gone: budget clock started"; budget_guard &
    # 1. SFT-weak + eval F0
    ( $L eval "$AB/models/sft_final" F0 0.1 ) &
    run_and_wait sft; stopped && exit 0
    $L convert >> "$LOG" 2>&1
    [ -f "$AB/models/sft_weak/model.safetensors" ] || { say "convert FAILED"; touch "$AB/ORDER_FAILED"; exit 5; }
    rm -rf "$AB/data/packed_fwd"; say "deleted data/packed_fwd (weak SFT data)"
    # 2./3. arm F, W0 eval in parallel
    ( $L eval "$AB/models/sft_weak" W0 0.1 ) &
    $L start full >> "$LOG" 2>&1 || { say "start full FAILED (preflight)"; touch "$AB/ORDER_FAILED"; exit 3; }
    sleep 1500                                     # W0 eval needs ~15 min next to the trainer
    python3 - "$AB/eval/W0/results.json" > "$AB/W_GATE.txt" <<'EOF'
import json, sys
try:
    r = json.load(open(sys.argv[1])); s = r["systems"]["sft"]; m = s["macro_all"]
    fl = s["benchmarks"]["flores_plus"]["languages"]
    fin = sum(v.get("finished_rate", 0) for v in fl.values()) / max(len(fl), 1)
    print("W0 macro chrF++ %.2f finished %.3f" % (m["chrf"], fin)); sys.exit(0 if (m["chrf"] >= 15 and fin >= 0.6) else 1)
except Exception as e:
    print("W0 gate unreadable:", e); sys.exit(2)
EOF
    g=$?; say "W0 gate: $(cat "$AB/W_GATE.txt") (rc=$g)"
    [ "$g" = 0 ] || { say "W0 GATE FAILED: holding after arm F (decide: weaker/longer SFT, or the interpolation fallback)"; touch "$AB/ORDER_HOLD_W_GATE"; }
    wait_file "$AB/runs/ab1_full/DONE_AB1_F"; say "arm F finished"; stopped && exit 0
    if [ -f "$AB/ORDER_HOLD_W_GATE" ]; then wait_file "$AB/ORDER_W_OK"; fi
    # 4. arm W, F evals in parallel
    ( $L eval "$AB/runs/ab1_full/snap_step$MID" F_mid 0.1; $L eval "$AB/runs/ab1_full/snap_step$END" F_end 0.1 ) &
    run_and_wait weak; stopped && exit 0
    # 5. W evals + analysis
    $L eval "$AB/runs/ab1_weak/snap_step$MID" W_mid 0.1; $L eval "$AB/runs/ab1_weak/snap_step$END" W_end 0.1
    wait
    /workspace/venv_gemma/bin/python "$TREE/scripts/ab1_analyze.py" --eval-dir "$AB/eval" --runs-dir "$AB/runs" --out "$AB/report" >> "$LOG" 2>&1
    /workspace/venv_gemma/bin/python "$TREE/scripts/ab1_analyze.py" --eval-dir "$AB/eval" --runs-dir "$AB/runs" --out "$AB/report" --common-finished >> "$LOG" 2>&1
    say "ALL DONE: $AB/report/ab1_report.md ; spent \$$(spent) (wall-clock since GO)"; touch "$AB/ORDER_DONE";;
  *) sed -n 2,17p "$0"; exit 2;;
esac
