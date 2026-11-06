#!/bin/bash
# SIDE QUEUE, PHASE 3 (owner order 2026-11-04 ~16:35 UTC; E-03 is mandatory and takes over the GPU): after the long-document RL training ends, E-09b (fp32 SFT, fp32 RL) and E-05 run one at a time next to
# the long-RL evaluation (at most 2 vLLM engines in all). E-02 and the post-hoc stronger-penalty runs are DROPPED, E-04 and the earlier E-03 plan too. When E-09b, E-05 and the long-RL evaluation are done,
# or at 20:30 UTC at the latest (then the side job still running is stopped and logged), write GPU_HANDOFF_E03 and leave the GPU alone: the E-03 executor takes over from that marker.
source /workspace/science_sft/overnight/common.sh
export GEMMA_MT_TAG_MODE=single_token HF_HOME=/workspace/clean_sft/hf_home TMPDIR=/dev/shm/tmp; mkdir -p /dev/shm/tmp
SFT=$R/models/g15_385m_sft_final; RL=$R/runs_385m/mixed/best; RUN=$R/runs_385m_long/final
DEADLINE=$(date -u -d "today 20:30" +%s)
say "side queue phase 3 started; E-02 and the stronger-penalty runs DROPPED (owner order, E-03 mandatory); deadline for side jobs $(date -u -d @$DEADLINE +%H:%M) UTC"
until [ -f $RUN/summary.json ] && ! pgrep -f "[g]emma_mt.rl.train_rl" >/dev/null; do sleep 30; done
say "long-document RL training ended; side jobs start next to its evaluation"
sleep 120
side3() {   # side3 NAME TIMEOUT CMD...
  local name=$1 to=$2; shift 2
  local left=$(( DEADLINE - $(date +%s) ))
  if [ $left -lt 600 ]; then say "SIDE $name NOT RUN: less than 10 min left before the 20:30 UTC deadline"; return 0; fi
  [ $to -gt $left ] && to=$left
  local f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "${f:-0}" -lt 15360 ] && say "SIDE $name: only ${f} MiB GPU memory free (< 15 GB), starting anyway after the evaluation's current phase" && sleep 120
  step SIDE_$name $to "$@"
  [ -f $O/STEP_SIDE_$name.FAILED ] && say "SIDE $name did not finish (timeout at the deadline or failure; partial output stays where it is)"
}
side3 E09b_sft_fp32 10800 bash $O/run_fp32_eval.sh sft $SFT
side3 E09b_rl_fp32 10800 bash $O/run_fp32_eval.sh rl $RL
side3 E05 7200 bash -c "export PYTHONPATH=$E/pylibs_none:$E/gemma_mt_tree; $E/venv_comet/bin/python $O/sent_neural_all.py $O/e05_all.json 8"
say "E-09b and E-05 finished or stopped; waiting for the long-RL evaluation (POST_LONG_RL_DONE) until the deadline"
until [ -f $S/POST_LONG_RL_DONE ] || [ $(date +%s) -ge $DEADLINE ]; do sleep 30; done
note="E-09b: $([ -f $O/STEP_SIDE_E09b_rl_fp32.DONE ] && echo done || echo 'not finished'); E-05: $([ -f $O/STEP_SIDE_E05.DONE ] && echo done || echo 'not finished'); long-RL evaluation: $([ -f $S/POST_LONG_RL_DONE ] && echo done || echo 'STILL RUNNING at the deadline (not stopped: it is the main job)')"
echo "$(date -u +%FT%TZ) GPU handoff to the E-03 executor. $note. E-02, E01x (stronger penalty), E-04 not run." > $O/GPU_HANDOFF_E03
say "GPU_HANDOFF_E03 written: $note"
