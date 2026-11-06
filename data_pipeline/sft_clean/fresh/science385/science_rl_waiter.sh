#!/bin/bash
# Queue behind the science SFT pipeline: when science_sft/PIPELINE_DONE exists and the GPU is free, run the GPU preflight of the reference-based SSA worker (falls back to ref_ssa: false =
# the other science terms still apply), then start the science RL stage (launcher stage `final` of the science config dir) from the chosen science-SFT model. HF_TOKEN from the environment only.
R=/workspace/clean_rerun; S=/workspace/science_sft; RS=$R/runs_385m_sci; LOG=$S/rl_waiter.log; CFG=$R/tree/gemma_mt/configs/clean_rerun_385m_sci
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
say "science RL waiter started"
while [ ! -f $S/PIPELINE_DONE ]; do sleep 20; done
while pgrep -f "[g]emma_mt.rl.train_rl|[g]emma_mt.train_sft|[e]val_sft_benchmarks|[e]val_afriscience" >/dev/null; do sleep 10; done
MODEL=$(cat $S/final_model_dir); say "science SFT done; model $MODEL (alpha $(cat $S/chosen_alpha))"
cd $R/tree/gemma_mt; source $R/rl_env.sh
if timeout 900 /workspace/venv_gemma/bin/python $S/check_ref_ssa.py >> $LOG 2>&1; then say "reference-based SSA worker preflight OK"; else
  say "reference-based SSA worker preflight FAILED: running with ref_ssa false (recall / precision / length / preservation terms only)"
  sed -i 's/^  ref_ssa: true.*/  ref_ssa: false/' $CFG/rl_F_final.yaml; fi
mkdir -p $RS/final /dev/shm/rl_sci_ckpt $R/logs_sci
for d in best final last; do mkdir -p /dev/shm/rl_sci_ckpt/$d; [ -e $RS/final/$d ] || ln -s /dev/shm/rl_sci_ckpt/$d $RS/final/$d; done
export RUNS=$RS LOGDIR=$R/logs_sci CFGDIR=$CFG STAGE_PLAN=b256 SCI_MODEL=$MODEL FORCE=1 OVERRIDE_PIN=1 LADDER_START=2 MIN_DISK_GIB=3 HUB_NO_MODEL_CARD=1 A_FROM_PREFLIGHT=0
say "launching science RL (stage final of the science config dir) from $MODEL"
nohup bash scripts/launch_clean_rerun_385m.sh run final >> $R/logs_sci/launch.log 2>&1 &
sleep 3; cd $R; sed -e 's#/workspace/clean_rerun/runs_385m/#/workspace/clean_rerun/runs_385m_sci/#g' -e 's#/workspace/clean_rerun/logs/#/workspace/clean_rerun/logs_sci/#g' watchdog_rl.sh > watchdog_sci.sh
nohup bash watchdog_sci.sh final >/dev/null 2>&1 &
say "science RL launched"
