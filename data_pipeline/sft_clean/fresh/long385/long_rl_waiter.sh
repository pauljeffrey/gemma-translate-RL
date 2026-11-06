#!/bin/bash
# Queue behind the science RL, its post-run evaluation AND the overnight controls E-01/E-09/E-13 (coordinator 2026-11-04): when POST_SCI_RL_DONE and overnight/CONTROLS_DONE exist (or the summary is 8 h old),
# pick the start checkpoint from the science RL's dev series (pick_long_start.py), then start the long-document RL (launcher stage `final` of the long config dir). HF_TOKEN from the environment only.
R=/workspace/clean_rerun; S=/workspace/science_sft; RS=$R/runs_385m_long; SCI=$R/runs_385m_sci/final; LOG=$S/long_rl_waiter.log; CFG=$R/tree/gemma_mt/configs/clean_rerun_385m_long
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
say "long RL waiter started"
busy() { pgrep -f "[g]emma_mt.rl.train_rl|[g]emma_mt.train_sft|[e]val_sft_benchmarks|[e]val_afriscience|[r]un_doc_eval|[p]ost_sci_rl_eval" >/dev/null; }
while true; do
  if [ -f "$SCI/summary.json" ] && ! busy; then
    if [ -f $S/POST_SCI_RL_DONE ] && [ -f $S/overnight/CONTROLS_DONE ]; then break; fi
    age=$(( $(date +%s) - $(stat -c %Y "$SCI/summary.json") ))
    [ "$age" -gt 28800 ] && { say "science RL summary is ${age}s old and no process runs: starting without the markers"; break; }
  fi
  sleep 20
done
python3 $S/pick_long_start.py $SCI $S/long_start_choice.json >> $LOG 2>&1
MODEL=$(python3 -c "import json;print(json.load(open('$S/long_start_choice.json'))['model_dir'])")
[ -f "$MODEL/config.json" ] || { say "no usable start model ($MODEL): NOT launching, see long_start_choice.json"; touch $S/LONG_RL_SKIPPED; exit 3; }
say "science RL done; long RL start model $MODEL ($(python3 -c "import json;print(json.load(open('$S/long_start_choice.json'))['reason'])"))"
cd $R/tree/gemma_mt; source $R/rl_env.sh
mkdir -p $RS/final /dev/shm/rl_long_ckpt $R/logs_long
for d in best final last; do mkdir -p /dev/shm/rl_long_ckpt/$d; [ -e $RS/final/$d ] || ln -s /dev/shm/rl_long_ckpt/$d $RS/final/$d; done
export RUNS=$RS LOGDIR=$R/logs_long CFGDIR=$CFG STAGE_PLAN=b256 SCI_MODEL=$MODEL FORCE=1 OVERRIDE_PIN=1 LADDER_START=1 MIN_DISK_GIB=3 MIN_FREE_GIB=60 HUB_NO_MODEL_CARD=1 A_FROM_PREFLIGHT=0
say "launching the long-document RL (stage final of the long config dir) from $MODEL"
nohup bash scripts/launch_clean_rerun_385m.sh run final >> $R/logs_long/launch.log 2>&1 &
sleep 3; cd $R; sed -e 's#/workspace/clean_rerun/runs_385m_sci/#/workspace/clean_rerun/runs_385m_long/#g' -e 's#/workspace/clean_rerun/logs_sci/#/workspace/clean_rerun/logs_long/#g' watchdog_sci.sh > watchdog_long.sh
nohup bash watchdog_long.sh final >/dev/null 2>&1 &
say "long RL launched"
# the post-run evaluation of best and final starts when the stage has ended (summary.json, no trainer)
nohup bash $S/post_long_rl_waiter.sh >/dev/null 2>&1 &
