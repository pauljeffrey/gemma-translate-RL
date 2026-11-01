#!/bin/bash
# Runs on the A100 after the bundle (run/packed_all, run/sched_all, their .sha256 files) has arrived: verify, config a0f_up6.yaml, wait for a free GPU, resume A0 from step 5,625
# on the full data (same run directory runs/a0u_up6, micro-batch 8 x accumulation 4, checkpoint every 20 steps, Hub push every 500).
set -uo pipefail
ROOT=/workspace/work; cd $ROOT/run; source $ROOT/env.sh
ALERT=$ROOT/logs/ALERT_a100.txt
fail() { echo "$(date -u +%FT%TZ) [A0F_LAUNCH_FAILED] $1" >> $ALERT; echo "FAIL $1"; exit 1; }
( cd packed_all && sha256sum -c ../packed_all.sha256 --quiet ) || fail "packed_all checksum"
( cd sched_all && sha256sum -c ../sched_all.sha256 --quiet ) || fail "sched_all checksum"
echo "$(date +%T) bundle verified"
python3 - <<PY || fail config
import re
t = open("configs/clean_rerun_sft/a0u_up6.yaml").read()
def setk(t, k, v): return re.sub(rf"^{k}:.*$", f"{k}: {v}", t, flags=re.M)
t = setk(t, "data_dir", "packed_all"); t = setk(t, "schedule_file", "sched_all/A0F/schedule.npy")
open("configs/clean_rerun_sft/a0f_up6.yaml", "w").write("# A0-up6 continued on ALL scored data (uncapped QE, NLLB capped at 150k/language) + the instruction stream, UniMax floor 1 / cap 4 epochs, stitched after step 5,625\n" + t)
PY
# the GPU may be in use by another job: wait for it (never kill anything)
for i in $(seq 1 720); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
  [ "${used:-99999}" -lt 4000 ] && break
  sleep 60
done
[ "${used:-99999}" -lt 4000 ] || fail "GPU still busy after 12 h"
rm -f runs/a0u_up6/summary.json
export HF_HOME=$ROOT/hf_home PYTHONPATH=$ROOT/code PY=$ROOT/venv_train/bin/python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HUB_NO_MODEL_CARD=1
echo "$(date +%T) launching A0 on the full data" >> $ROOT/logs/arms_queue.log
nohup setsid bash $ROOT/code/sft_clean/fresh/sft_concurrent.sh "a0u_up6=configs/clean_rerun_sft/a0f_up6.yaml,micro_batch_size=8,grad_accum_steps=4" > $ROOT/logs/a0f_launch.log 2>&1 < /dev/null &
for i in $(seq 1 60); do
  sleep 15
  tr '\r' '\n' < $ROOT/run/logs/sft_a0u_up6.log | grep -a "resumed from step\|rror" | tail -n 2 | grep -aq "resumed from step 5625" && { echo "$(date -u +%FT%TZ) [A0F_RESUMED] A0 continues on the full data (packed_all, sched_all/A0F)" >> $ALERT; echo resumed; exit 0; }
done
fail "no 'resumed from step 5625' in sft_a0u_up6.log (see logs/a0f_launch.log)"
