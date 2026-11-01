#!/bin/bash
# Runs on the A100 (deploy_growth.sh puts the code there first). Growth 1, two phases, data streamed in two bundles (owner 2026-11-01):
#   PHASE 1  marker run/.arrived_bundle_g0: run/packed_g0/ (+ packed_g0.sha256, paths relative to packed_g0/), run/sched_g/A/ (+ sched_g_A.sha256, relative to sched_g/A/)
#            -> stage A: configs/growth1/g1a.yaml, data packed_g0, sched_g/A (1,500 steps), only the 15 new layers + the 125 tag rows train
#   PHASE 2  marker run/.arrived_bundle_delta: run/packed_delta_tag/ (+ packed_delta_tag.sha256), run/sched_g/B/ (+ sched_g_B.sha256), run/packed_g_arrays.sha256 (of the FINAL packed_g0
#            train/*.npy + train/tokens.u32.bin, relative to packed_g0/). After stage A: verify the delta, extend packed_g0 IN PLACE (chain/extend_packed.py --inplace), verify the arrays,
#            delete packed_delta_tag -> stage B: configs/growth1/g1b.yaml, data packed_g0 (the extended directory keeps its name), sched_g/B, everything trains, weights from runs/g1a/last.
# The grown checkpoint (A0 step 5,625 + 15 identity layers + 125 tag rows, 385,144,448 parameters) is built here when missing. Polls every 60 s, waits for a free GPU (never kills anything).
# Alerts in /workspace/work/logs/ALERT_a100.txt: [G1A_STARTED] [G1A_DONE] [G1_WAITING_PHASE2] [G1B_STARTED] [G1B_DONE] and [G1_LAUNCH_FAILED] <reason>.
# Restartable: the trainer resumes from runs/g1a|g1b/last; a finished stage (summary.json) is skipped; the in-place extension is skipped once packed_g_arrays.sha256 verifies.
set -uo pipefail
ROOT=/workspace/work; cd $ROOT/run; source $ROOT/env.sh
SC=$ROOT/code/gemma_mt_root/scripts; CH=$ROOT/code/sft_clean/chain; CK=$ROOT/ckpt; GROWN=gemma3-up6-g15-tags
ALERT=$ROOT/logs/ALERT_a100.txt
export HF_HOME=$ROOT/hf_home PYTHONPATH=$ROOT/code:$ROOT/code/sft_clean/chain GEMMA_MT_PKG=$ROOT/code PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HUB_NO_MODEL_CARD=1
PY=$ROOT/venv_train/bin/python
fail() { echo "$(date -u +%FT%TZ) [G1_LAUNCH_FAILED] $1" >> $ALERT; echo "FAIL $1"; exit 1; }
free_gb() { df -BG --output=avail /workspace | tail -n 1 | tr -dc 0-9; }
wait_gpu() {
  local used
  for i in $(seq 1 720); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -dc 0-9)
    [ "${used:-99999}" -lt 4000 ] && return 0
    sleep 60
  done
  fail "GPU still busy after 12 h"
}
wait_disk() {   # wait_disk GB WHAT: poll until GB are free (alert once); nothing is deleted here
  local told=0
  while [ "$(free_gb)" -lt "$1" ]; do
    [ $told = 0 ] && echo "$(date -u +%FT%TZ) [G1_WAITING_DISK] $2 needs $1 GB free on /workspace, $(free_gb) GB free; polling every 60 s (free space: the owner / lead decides what to delete)" >> $ALERT
    told=1; sleep 60
  done
}
verify() {      # verify DIR SUMFILE: the checksums of SUMFILE (paths relative to DIR); a few retries, the files may still be flushing
  local n
  for n in 1 2 3; do ( cd "$1" && sha256sum -c "$2" --quiet ) 2>/dev/null && return 0; sleep 60; done
  return 1
}
mkdir -p logs
for f in configs/growth1/g1a.yaml configs/growth1/g1b.yaml $SC/upcycle_layers.py $SC/init_tag_rows.py $SC/export_trainer_state.py $ROOT/code/gemma_mt/growth.py $CH/extend_packed.py; do
  [ -f "$f" ] || fail "missing $f (run deploy_growth.sh from the laptop first)"
done
grep -q "train_embedding_tags" $ROOT/code/gemma_mt/train_sft.py || fail "code/gemma_mt/train_sft.py is the old trainer (no freeze_regex / train_embedding_tags): run deploy_growth.sh"
grep -q "inplace" $CH/extend_packed.py || fail "chain/extend_packed.py has no --inplace: run deploy_growth.sh"

# ---- 1. the grown checkpoint (built once; the two intermediates are deleted afterwards to save 2.7 GB of the tight disk)
if [ ! -f $CK/$GROWN/model.safetensors ]; then
  ST=$ROOT/run/runs/a0u_up6/last/trainer_state.pt; [ -f $ST ] || fail "A0 trainer_state.pt missing"
  rm -rf $CK/gemma3-270m-up6-a0 $CK/gemma3-up6-g15 $CK/$GROWN
  $PY $SC/export_trainer_state.py --state $ST --like $CK/gemma3-270m-up6 --out $CK/gemma3-270m-up6-a0 > $ROOT/logs/g1_export.log 2>&1 || fail "export A0"
  grep -q '"step": 5625' $CK/gemma3-270m-up6-a0/export.json || fail "A0 export is not step 5625"
  $PY $SC/upcycle_layers.py --model $CK/gemma3-270m-up6-a0 --out $CK/gemma3-up6-g15 --n-new 15 --post-norm-gain 0.01 > $ROOT/logs/g1_upcycle.log 2>&1 || fail "upcycle"
  $PY $SC/init_tag_rows.py --model $CK/gemma3-up6-g15 --out $CK/$GROWN > $ROOT/logs/g1_tags.log 2>&1 || fail "tag rows"
  rm -rf $CK/gemma3-270m-up6-a0 $CK/gemma3-up6-g15
fi
python3 -c "import json; u=json.load(open('$CK/$GROWN/upcycle.json')); assert u['n_new_layers']==15 and u['parameters_upcycled']==385144448 and u['sliding_window_pattern']==13" || fail "grown checkpoint is not the expected 39-layer model"
echo "$(date +%T) grown checkpoint ok"

# ---- 2. stage A: wait for phase 1, verify, train
if [ ! -f runs/g1a/summary.json ]; then
  until [ -f .arrived_bundle_g0 ]; do sleep 60; done
  verify packed_g0 ../packed_g0.sha256 || fail "packed_g0 checksum"
  verify sched_g/A ../../sched_g_A.sha256 || fail "sched_g/A checksum"
  [ -f sched_g/A/schedule.npy ] && [ -f sched_g/A/schedule.json ] || fail "sched_g/A/schedule.npy or its sidecar missing"
  echo "$(date +%T) stage A inputs verified"
  wait_disk 9 "stage A (last 3.2 GB + last_prev 3.2 GB + Hub staging copy 1.5 GB)"
  wait_gpu
  echo "$(date -u +%FT%TZ) [G1A_STARTED] g1a: 15 new layers + 125 tag rows, data packed_g0, $(python3 -c "import numpy as np;print(np.load('sched_g/A/schedule.npy').shape[0])") steps" >> $ALERT
  "$PY" -m gemma_mt.train_sft --config configs/growth1/g1a.yaml micro_batch_size=8 grad_accum_steps=4 >> logs/sft_g1a.log 2>&1
  rc=$?
  [ $rc = 0 ] && [ -f runs/g1a/summary.json ] || fail "stage A exited rc=$rc (see $ROOT/run/logs/sft_g1a.log)"
  rm -rf runs/g1a/last_prev                                  # stage B only needs runs/g1a/last (init_from, read again on every resume of B)
  echo "$(date -u +%FT%TZ) [G1A_DONE] g1a finished: $(python3 -c "import json;print(json.load(open('runs/g1a/summary.json')).get('steps'))") steps; runs/g1a/last/trainer_state.pt kept ($(du -sh runs/g1a/last | cut -f1))" >> $ALERT
fi

# ---- 3. stage B: wait for phase 2, extend packed_g0 in place, train
if [ ! -f runs/g1b/summary.json ]; then
  if ! ( cd packed_g0 && sha256sum -c ../packed_g_arrays.sha256 --quiet ) 2>/dev/null; then      # not extended yet
    [ -f .arrived_bundle_delta ] || echo "$(date -u +%FT%TZ) [G1_WAITING_PHASE2] stage A is done; waiting for run/.arrived_bundle_delta (packed_delta_tag, sched_g/B, packed_g_arrays.sha256); polling every 60 s" >> $ALERT
    until [ -f .arrived_bundle_delta ]; do sleep 60; done
    [ -d packed_delta_tag ] || fail "phase 2 marker present but packed_delta_tag/ is missing"
    verify packed_delta_tag ../packed_delta_tag.sha256 || fail "packed_delta_tag checksum"
    echo "$(date +%T) extending packed_g0 in place"
    $PY $CH/extend_packed.py --base packed_g0 --add packed_delta_tag --out packed_g0 --inplace > $ROOT/logs/g1_extend_inplace.log 2>&1 || fail "extend_packed --inplace (packed_g0 may be half-extended: see $ROOT/logs/g1_extend_inplace.log; restore from the bundle before retrying)"
    tail -n 1 $ROOT/logs/g1_extend_inplace.log
    ( cd packed_g0 && sha256sum -c ../packed_g_arrays.sha256 --quiet ) || fail "extended packed_g0 does not match packed_g_arrays.sha256"
  fi
  [ -f .arrived_bundle_delta ] || until [ -f .arrived_bundle_delta ]; do sleep 60; done
  verify sched_g/B ../../sched_g_B.sha256 || fail "sched_g/B checksum"
  [ -f sched_g/B/schedule.npy ] && [ -f sched_g/B/schedule.json ] || fail "sched_g/B/schedule.npy or its sidecar missing"
  rm -rf packed_delta_tag                                    # 4 GB back; the extension is verified
  echo "$(date +%T) packed_g0 extended and verified, packed_delta_tag deleted"
  wait_disk 10 "stage B (last 3.75 GB + last_prev 3.75 GB + Hub staging copy 1.5 GB; runs/g1a/last 3.2 GB stays)"
  wait_gpu
  echo "$(date -u +%FT%TZ) [G1B_STARTED] g1b: all parameters, data packed_g0 (extended), weights from runs/g1a/last" >> $ALERT
  "$PY" -m gemma_mt.train_sft --config configs/growth1/g1b.yaml micro_batch_size=8 grad_accum_steps=4 >> logs/sft_g1b.log 2>&1
  rc=$?
  [ $rc = 0 ] && [ -f runs/g1b/summary.json ] || fail "stage B exited rc=$rc (see $ROOT/run/logs/sft_g1b.log)"
fi
echo "$(date -u +%FT%TZ) [G1B_DONE] growth 1 finished: $(python3 -c "import json;print(json.load(open('runs/g1b/summary.json')).get('steps'))") steps" >> $ALERT
