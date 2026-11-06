#!/bin/bash
# stage C2: equal-samples TranslatePsy half: select N per language, pack train (+val), verify
set -euo pipefail
source /dev/shm/control_prep/common_env.sh; cd $WORK
N=${N:-281500}; V=${V:-600}
[ -f .c2a ] || { log select; nice -n 19 $PY $HERE/tp_select_equal.py --in tp_clean_afr tp_clean_mlg tp_clean_tsn --out-train tp_train --out-val tp_val --n $N --val $V --table $WORK/tp_select_table.csv; touch .c2a; }
[ -f .c2b ] || { log pack_train; rm -rf tp_packed
  nice -n 19 $PY -m gemma_mt.data.prepare_sft --input tp_train --out tp_packed --tokenizer $WORK/sft_model --seq-len 4096 --max-sample-tokens 4096 --directions en2x --pack-by-language --val-fraction 0 --workers 4 2>&1 | grep -v "incorrect regex" | tail -6; touch .c2b; }
[ -f .c2c ] || { log pack_val; rm -rf tp_valpack
  nice -n 19 $PY -m gemma_mt.data.prepare_sft --input tp_val --out tp_valpack --tokenizer $WORK/sft_model --seq-len 4096 --max-sample-tokens 4096 --directions en2x --pack-by-language --val-fraction 0 --workers 2 2>&1 | grep -v "incorrect regex" | tail -4; touch .c2c; }
log verify
$PY -m gemma_mt.data.audit_packed --data $WORK/tp_packed --split train --check-samples 4000 --expect-single-language --decode 2 2>&1 | grep -v "incorrect regex" | tail -12
log stage_C2_done
