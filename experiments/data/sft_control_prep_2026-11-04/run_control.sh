#!/bin/bash
# E-03 control: run order on the A100 AFTER the science RL and the long-document RL (and their evaluation) have ended and the GPU is free. Training only; evaluation is in the report.
# Never pushes anything (hub_repo_id null). Needs: $WORK/combined, $WORK/sched_ctl (build_control_data.sh), train_sft_eot.py in the current folder (arm C2).
set -euo pipefail
ROOT=/workspace/clean_sft; WORK=${WORK:-/dev/shm/control}; cd $ROOT/run; source $ROOT/env.sh
PY=$ROOT/venv_train/bin/python; HERE=${HERE:-/workspace/control_prep}          # folder holding the yaml files and train_sft_eot.py
export HF_HOME=$ROOT/hf_home PYTHONPATH=$ROOT/code:$ROOT/code/sft_clean/chain GEMMA_MT_PKG=$ROOT/code PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GEMMA_MT_TAG_MODE=single_token
export HUB_NO_MODEL_CARD=1; unset HF_WRITE_TOKEN                                  # the run must not be able to push
ARM=${ARM:-c1}                                                                    # c1 | c2_eot
CFG=$HERE/control_${ARM}.yaml
EXPORT="$PY $ROOT/code/gemma_mt_root/scripts/export_trainer_state.py"
LAUNCH=("$PY" -m gemma_mt.train_sft)
[ "$ARM" = c2_eot ] && { export EOT_LOSS_WEIGHT=5 EOT_TOKEN_IDS=106; LAUNCH=("$PY" $HERE/train_sft_eot.py); }

# 0. smoke test (GPU, 20 steps, about 2 min): prints loss / tokens per s / lr; abort if tokens/s is far below 25 k (then re-plan the step budget, see the yaml header)
"${LAUNCH[@]}" --config $CFG micro_batch_size=8 grad_accum_steps=4 stop_after_steps=20 out_dir=$WORK/runs/smoke_$ARM

# 1. segment 1: steps 1..3,100 (stage-A matched), then export the weights for evaluation
"${LAUNCH[@]}" --config $CFG micro_batch_size=8 grad_accum_steps=4 stop_after_steps=3100
$EXPORT --state $WORK/runs/$ARM/last/trainer_state.pt --like $WORK/sft_model --out $WORK/export/${ARM}_s3100 --dtype fp32
# 2. segment 2: resume (exact: optimiser state kept) to 4,800 (mixed-best matched), export
"${LAUNCH[@]}" --config $CFG micro_batch_size=8 grad_accum_steps=4 stop_after_steps=4800
$EXPORT --state $WORK/runs/$ARM/last/trainer_state.pt --like $WORK/sft_model --out $WORK/export/${ARM}_s4800 --dtype fp32
# 3. segment 3: to the end of the 6,000-step schedule (chain matched, 8.57 h of training); the trainer writes <out_dir>/final only without keep_single_checkpoint, so export last/ again
"${LAUNCH[@]}" --config $CFG micro_batch_size=8 grad_accum_steps=4
$EXPORT --state $WORK/runs/$ARM/last/trainer_state.pt --like $WORK/sft_model --out $WORK/export/${ARM}_s6000 --dtype fp32
echo CONTROL_${ARM}_DONE
