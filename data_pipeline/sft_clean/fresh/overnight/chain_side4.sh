#!/bin/bash
# SIDE QUEUE, PHASE 4 (coordinator 2026-11-04 ~18:00 UTC; E-03 drives the box time): cut-off 19:45 UTC (side jobs still running are stopped and logged; unfinished E-09b / E-05 are "not completed", partial outputs kept);
# GPU_HANDOFF_E03 is written as soon as the long-RL evaluation is finished (POST_LONG_RL_DONE), even if side jobs are not, or at 19:45 if the evaluation is still running (the marker says so). The long-RL evaluation is
# never stopped. No side job is started after the marker. Order: [fp32 SFT run already in progress since 17:46] -> E-05 (light) -> fp32 RL run.
source /workspace/science_sft/overnight/common.sh
export GEMMA_MT_TAG_MODE=single_token HF_HOME=/workspace/clean_sft/hf_home TMPDIR=/dev/shm/tmp; mkdir -p /dev/shm/tmp
SFT=$R/models/g15_385m_sft_final; RL=$R/runs_385m/mixed/best
DEADLINE=$(date -u -d "today 19:45" +%s); MARK=$O/GPU_HANDOFF_E03
say "side queue phase 4 started: cut-off 19:45 UTC, marker as soon as the long-RL evaluation is done; order fp32-SFT (running) -> E-05 -> fp32-RL"
stop_side() { for pat in "run_fp32_eval.sh" "sent_neural_all.py" "gemma_mt.eval.sft_benchmarks"; do for p in $(pgrep -f "$pat"); do
    # only side-job processes: the long-RL evaluation uses eval_afriscience*.py / run_*_x.sh / documents scripts, never these patterns, except sft_benchmarks of its NTREX/MAFAND pass: check the out dir
    cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null); case "$cmd" in *out_fp32_*|*sent_neural_all*|*run_fp32_eval*) kill_tree $p;; esac; done; done; }
kill_tree() { for c in $(pgrep -P $1); do kill_tree $c; done; kill $1 2>/dev/null; }
write_marker() {
  local why=$1
  local sft="not completed"; [ -f /dev/shm/out_fp32_sft/EVAL_DONE ] && sft="completed"
  local rl="not completed"; [ -f /dev/shm/out_fp32_rl/EVAL_DONE ] && rl="completed"
  local e5="not completed"; [ -f $O/STEP_SIDE_E05.DONE ] && e5="completed"
  local ev="STILL RUNNING (not stopped: main job)"; [ -f $S/POST_LONG_RL_DONE ] && ev="done"
  echo "$(date -u +%FT%TZ) GPU handoff to the E-03 executor ($why). long-RL evaluation: $ev. E-09b fp32 SFT: $sft; E-09b fp32 RL: $rl; E-05: $e5. Side jobs still running are stopped at 19:45 UTC at the latest; no new side job is started. E-02, E01x (stronger penalty), E-04 not run." > $MARK
  say "GPU_HANDOFF_E03 written ($why): long-RL evaluation $ev; E-09b SFT $sft, E-09b RL $rl, E-05 $e5"
}
# watcher: marker + cut-off
( while true; do
    if [ ! -f $MARK ]; then
      if [ -f $S/POST_LONG_RL_DONE ]; then write_marker "long-RL evaluation finished"; elif [ $(date +%s) -ge $DEADLINE ]; then write_marker "19:45 UTC cut-off reached"; fi
    fi
    if [ $(date +%s) -ge $DEADLINE ]; then stop_side; say "19:45 UTC cut-off: side jobs still running were stopped"; break; fi
    sleep 30
  done ) &
WATCH=$!
# 1. the fp32 SFT run that is already running (started by chain_side3, orphaned when that chain was replaced)
while pgrep -f "run_fp32_eval.sh sft" >/dev/null && [ $(date +%s) -lt $DEADLINE ]; do sleep 30; done
[ -f /dev/shm/out_fp32_sft/EVAL_DONE ] && say "SIDE E09b_sft_fp32 DONE (EVAL_DONE)" || say "SIDE E09b_sft_fp32 not completed"
side4() {
  local name=$1 to=$2; shift 2
  [ -f $MARK ] && { say "SIDE $name NOT STARTED: the GPU handoff marker exists"; return 0; }
  local left=$(( DEADLINE - $(date +%s) )); [ $left -lt 600 ] && { say "SIDE $name NOT STARTED: less than 10 min to the 19:45 cut-off"; return 0; }
  [ $to -gt $left ] && to=$left
  step SIDE_$name $to "$@"
}
side4 E05 7200 bash -c "export PYTHONPATH=$E/pylibs_none:$E/gemma_mt_tree; $E/venv_comet/bin/python $O/sent_neural_all.py $O/e05_all.json 8"
side4 E09b_rl_fp32 10800 bash $O/run_fp32_eval.sh rl $RL
say "side queue phase 4: all side jobs done, skipped or stopped; waiting for the marker/cut-off"
wait $WATCH
