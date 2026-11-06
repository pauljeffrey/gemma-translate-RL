#!/bin/bash
# stage C1: calibration pack -> N (equal samples per language) 
set -euo pipefail
source /dev/shm/control_prep/common_env.sh; cd $WORK
TARGET_P=${TARGET_P:-96300}
mkdir -p tp_calib
for d in tp_clean_*; do for f in $d/en2x_*.jsonl; do head -n 20000 $f > tp_calib/$(basename $f); done; done
ls tp_calib | wc -l
nice -n 19 $PY -m gemma_mt.data.prepare_sft --input tp_calib --out tp_calib_packed --tokenizer $WORK/sft_model --seq-len 4096 --max-sample-tokens 4096 --directions en2x --pack-by-language --val-fraction 0 --workers 3 2>&1 | tail -15
$PY - <<PYEOF
import numpy as np, json, collections
from gemma_mt.languages import ALL_ISOS
d="$WORK/tp_calib_packed/train"
import os; print(sorted(os.listdir(d)))
iso=np.load(d+"/iso_id.npy"); print("iso_id shape", iso.shape, "n_packs?")
meta=json.load(open("$WORK/tp_calib_packed/meta.json")); print({k:meta[k] for k in list(meta)[:12]})
PYEOF
