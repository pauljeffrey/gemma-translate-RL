#!/bin/bash
# E-03 GPU phase: wait for GPU_HANDOFF_E03, LR probe (pre-registered), then C1 to step 6,000 with exports at 3,100 / 4,800 / 6,000. Never pushes anything (tokens unset, offline).
source /dev/shm/control_prep/common_env.sh
H=/workspace/science_sft/overnight/GPU_HANDOFF_E03
cd $WORK; mkdir -p runs export probe_logs
until [ -f $H ]; do sleep 60; done
log "GPU_HANDOFF_E03 present: $(head -c 300 $H | tr '\n' ' ')"
train() { env -u HF_TOKEN -u HFT -u HF_WRITE_TOKEN HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -m gemma_mt.train_sft --config $HERE/control_c1.yaml \
   micro_batch_size=4 grad_accum_steps=8 model_name=$WORK/sft_model tokenizer_name=$WORK/sft_model "$@"; }
run_seg() {  # retry once; if the second attempt ends with the same code, stop for diagnosis
  local rc1=0 rc2=0; train "$@" || rc1=$?
  if [ $rc1 -ne 0 ]; then log "trainer rc=$rc1, restarting once"; train "$@" || rc2=$?
    if [ $rc2 -ne 0 ]; then log "trainer failed again rc=$rc2"; echo "rc1=$rc1 rc2=$rc2" > $WORK/STOPPED_FOR_DIAGNOSIS; exit 1; fi; fi; }
if [ ! -f $WORK/lr_choice.json ]; then
  for LR in 3e-6 1e-5 3e-5; do
    [ -f $WORK/runs/probe_$LR/PROBE_DONE ] && continue
    log "probe lr=$LR start"; rm -rf $WORK/runs/probe_$LR
    run_seg out_dir=$WORK/runs/probe_$LR learning_rate=$LR stop_after_steps=300 eval_every=100 eval_at_start=true save_every=300 keep_prev_checkpoint=false
    touch $WORK/runs/probe_$LR/PROBE_DONE; cp $WORK/runs/probe_$LR/metrics.jsonl $WORK/probe_logs/metrics_$LR.jsonl; log "probe lr=$LR done"
  done
  $PY $HERE/select_lr.py $WORK > $WORK/lr_choice.log 2>&1 || { log "select_lr failed"; cat $WORK/lr_choice.log; exit 1; }
  cat $WORK/lr_choice.log
fi
WIN=$($PY -c "import json;print(json.load(open('$WORK/lr_choice.json'))['winner'])"); log "winning LR $WIN"
if [ ! -d $WORK/runs/c1 ]; then mv $WORK/runs/probe_$WIN $WORK/runs/c1; rm -f $WORK/runs/c1/PROBE_DONE; for d in $WORK/runs/probe_*; do rm -rf $d; done; fi
seg() {  # target step, export name
  local T=$1 N=$2 DT=${3:-bf16}
  if [ ! -f $WORK/export/$N/model.safetensors ]; then
    log "C1 segment to step $T"
    run_seg out_dir=$WORK/runs/c1 learning_rate=$WIN stop_after_steps=$T eval_every=250 save_every=100 keep_prev_checkpoint=true
    env -u HF_TOKEN -u HFT HF_HUB_OFFLINE=1 $PY $ROOT/code/gemma_mt_root/scripts/export_trainer_state.py --state $WORK/runs/c1/last/trainer_state.pt --like $WORK/sft_model --out $WORK/export/$N --dtype $DT
    touch $WORK/export/$N/EXPORT_DONE; log "exported $N"
  fi
}
seg 3100 c1_s3100; seg 4800 c1_s4800
if [ ! -f $WORK/export/c1_s6000/EXPORT_DONE ]; then
  log "C1 segment to the end"; run_seg out_dir=$WORK/runs/c1 learning_rate=$WIN eval_every=250 save_every=100 keep_prev_checkpoint=true
  env -u HF_TOKEN -u HFT HF_HUB_OFFLINE=1 $PY $ROOT/code/gemma_mt_root/scripts/export_trainer_state.py --state $WORK/runs/c1/last/trainer_state.pt --like $WORK/sft_model --out $WORK/export/c1_s6000 --dtype fp32
  touch $WORK/export/c1_s6000/EXPORT_DONE; log "exported c1_s6000"
fi
touch $WORK/TRAINING_DONE; log TRAINING_DONE
