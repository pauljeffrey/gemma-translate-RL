#!/bin/bash
set -euo pipefail
source /dev/shm/control_prep/common_env.sh; cd $WORK
mkdir -p reports_keep
for d in tp_clean_afr tp_clean_mlg tp_clean_tsn; do cp $d/prepare_report.json reports_keep/prepare_report_$d.json; cp $d/per_language.csv reports_keep/per_language_$d.csv; done
cp tp_select_table.csv tp_equal_samples_table.csv reports_keep/ 2>/dev/null || true
log extend; nice -n 19 $PY $CODE/sft_clean/chain/extend_packed.py --base $WORK/replay_sel --add $WORK/tp_packed --out $WORK/replay_sel --inplace
mv replay_sel combined; cp -r tp_valpack/train combined/val_tp; ls combined combined/val_tp
rm -rf tp_packed tp_train tp_val tp_clean_afr tp_clean_mlg tp_clean_tsn pidx tp_valpack
df -h /dev/shm | tail -1
log schedule; nice -n 19 $PY $HERE/make_control_schedule_equal.py --packed-dir $WORK/combined --n-replay 96600 --steps 6000 --out $WORK/sched_ctl
$PY - <<PYEOF
import numpy as np
from gemma_mt.data.packed_dataset import PackedDataset
from gemma_mt.train_sft import load_schedule
ds = PackedDataset("$WORK/combined", "train", pad=False)
arr, info = load_schedule("$WORK/sched_ctl/schedule.npy", ds, 32)
print(arr.shape, info["checks_passed"], "unique packs", len(np.unique(arr)), "max draws per pack", int(np.bincount(arr.reshape(-1)).max()))
PYEOF
log BUILD_DONE
