#!/bin/bash
# After the uncapped scoring + gate (run_chain.sh steps 32-43 with QE_CAP=0): windows and packs of the NEW kept pairs only, appended to packed_ext (so every pack index
# A0 was trained on stays valid), then a UniMax schedule (floor 1, cap 4 epochs per language) stitched onto the 5,625 steps already trained.
#   LAMBDA = UniMax water level in packs: a language with <= LAMBDA/4 packs gets 4 epochs, one with >= LAMBDA packs gets 1, in between LAMBDA/n.
set -uo pipefail
ROOT=${ROOT:-/workspace/work}; cd $ROOT; source env.sh
export HF_HOME=$ROOT/hf_home PYTHONPATH=$ROOT/code/sft_clean/chain:$ROOT/code GEMMA_MT_PKG=$ROOT/code
CH=$ROOT/code/sft_clean/chain; PD=$ROOT/venv_data/bin/python; W=$ROOT/work; HUB=$ROOT/hub; LG=$ROOT/logs
LAMBDA=${LAMBDA:-12000}
until [ -f state/43_sanity.done ]; do sleep 30; done
echo "$(date +%T) gate done"
SRC="$HUB/data $HUB/data_devhalf $HUB/specialist $HUB/extra"; NCA=(--nc-allow howard-nlp/ibom-mt)
$PD - <<PY || exit 1
import pyarrow.parquet as pq, pyarrow.compute as pc
new = pq.read_table("$W/gate/keep_final.parquet"); old = pq.read_table("$W/gate_old/keep_final.parquet")
print("columns", new.column_names, "new", new.num_rows, "old", old.num_rows)
k = "key"
d = new.filter(pc.invert(pc.is_in(new[k], value_set=old[k])))
print("delta", d.num_rows, "in old but not in new:", pc.sum(pc.invert(pc.is_in(old[k], value_set=new[k]))).as_py())
pq.write_table(d, "$W/gate/keep_delta.parquet")
PY
rm -rf $W/windows_delta $W/packed_delta $W/packed_full
$PD $CH/windows.py --src $SRC --keep-file $W/gate/keep_delta.parquet --out $W/windows_delta --afro-tokenizer $W/tok_afro/tokenizer.json ${NCA[@]} > $LG/windows_delta.log 2>&1 || { echo "FAIL windows"; exit 1; }
$PD $CH/nc_filter.py guard-windows --windows $W/windows_delta >> $LG/windows_delta.log 2>&1 || { echo "FAIL guard"; exit 1; }
$PD $CH/pack.py prepare --windows $W/windows_delta --out $W/packed_delta --workers 8 --val-fraction 0 > $LG/pack_delta.log 2>&1 || { echo "FAIL pack"; exit 1; }
$PD $CH/extend_packed.py --base $W/packed_ext --add $W/packed_delta --out $W/packed_full > $LG/extend_full.log 2>&1 || { echo "FAIL extend"; exit 1; }
echo "$(date +%T) packs built"; tail -n 3 $LG/extend_full.log
# total draws from the real pack counts: sum_i clip(LAMBDA, n_i, 4 n_i), rounded down to whole steps of 32 packs
B=$($PD - <<PY
import numpy as np, sys, os
import sampler as S
PD_, ALL = S._pd()
ds = PD_.PackedDataset("$W/packed_full", None, pad=False) if os.path.exists("$W/packed_full/tokens.u32.bin") else PD_.PackedDataset("$W", "packed_full", pad=False)
pl = np.asarray(ds.pack_languages(), dtype=np.int64); c = np.bincount(pl); c = c[c > 0]
B = int(sum(min(max($LAMBDA, n), 4 * n) for n in c)); print(B - B % 32)
PY
)
echo "budget draws $B"
$PD $CH/sampler.py stitch --old-schedule $W/sched_ext/A0U --prefix-steps 5625 --packed-dir $W/packed_full --out $W/sched_full/A0F --budget-draws $B --min-epochs 1 --max-epochs 4 --seed 1234 > $LG/stitch_full.log 2>&1 || { echo "FAIL stitch"; tail -n 5 $LG/stitch_full.log; exit 1; }
tail -n 2 $LG/stitch_full.log
echo "$(date +%T) schedule ready"

echo "$(date -u +%FT%TZ) [PACKS_READY] packed_full + sched_full/A0F built (lambda $LAMBDA); training NOT launched (owner: pause until the extra intake is appended)" >> $ROOT/logs/ALERT_a100.txt
[ -z "${LAUNCH:-}" ] && exit 0
# ---- A0 continues on the full data: same run directory (runs/a0u_up6, last/trainer_state.pt at step 5625), longer stitched schedule, packs = packed_full
cd $ROOT/run
ln -sfn ../work/packed_full packed_full; ln -sfn ../work/sched_full sched_full
python3 - <<PY
import re
t = open("configs/clean_rerun_sft/a0u_up6.yaml").read()
def setk(t, k, v): return re.sub(rf"^{k}:.*$", f"{k}: {v}", t, flags=re.M)
t = setk(t, "data_dir", "packed_full"); t = setk(t, "schedule_file", "sched_full/A0F/schedule.npy")
open("configs/clean_rerun_sft/a0f_up6.yaml", "w").write("# A0-up6 continued on ALL scored data (uncapped QE), UniMax floor 1 / cap 4 epochs, stitched after step 5,625\n" + t)
PY
rm -f runs/a0u_up6/summary.json
export HF_HOME=$ROOT/hf_home PYTHONPATH=$ROOT/code PY=$ROOT/venv_train/bin/python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HUB_NO_MODEL_CARD=1
echo "$(date +%T) launching A0 on the full data (lambda $LAMBDA)" >> $ROOT/logs/arms_queue.log
nohup setsid bash $ROOT/code/sft_clean/fresh/sft_concurrent.sh "a0u_up6=configs/clean_rerun_sft/a0f_up6.yaml,micro_batch_size=8,grad_accum_steps=4" > $ROOT/logs/a0f_launch.log 2>&1 < /dev/null &
for i in $(seq 1 40); do sleep 15; tr '\r' '\n' < $ROOT/run/logs/sft_a0u_up6.log | grep -a "resumed from step\|rror" | tail -n 2 | grep -aq "resumed from step 5625" && { echo "$(date -u +%FT%TZ) [A0F_RESUMED] A0 continues on the full data" >> $ROOT/logs/ALERT_a100.txt; echo resumed; exit 0; }; done
echo "$(date -u +%FT%TZ) [A0F_START_FAILED] see logs/a0f_launch.log and run/logs/sft_a0u_up6.log" >> $ROOT/logs/ALERT_a100.txt; exit 1
