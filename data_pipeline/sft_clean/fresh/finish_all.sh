#!/bin/bash
# After ALL scoring (main chain step 34 + extra2k): gate over every source, keys NEW compared with A0's keep set, windows + packs of the new pairs only (append-only later: extend_packed on the A100),
# per-language token table. Nothing here touches the GPU.
set -uo pipefail
ROOT=/workspace/work; cd $ROOT; source env.sh
export HF_HOME=$ROOT/hf_home PYTHONPATH=$ROOT/code/sft_clean/chain:$ROOT/code GEMMA_MT_PKG=$ROOT/code
CH=$ROOT/code/sft_clean/chain; PD=$ROOT/venv_data/bin/python; W=$ROOT/work; HUB=$ROOT/hub; LG=$ROOT/logs
until [ -f state/34_qe_extra.done ] && [ -f logs/qe_extra2_done.txt ] && [ -f logs/qe_hold_done.txt ]; do sleep 60; done
echo "$(date +%T) all scoring done"
for p in $(pgrep -f "[r]un_chain.sh"); do kill $p; done          # the chain would now run its own gate with the old source list
SRC="$HUB/data $HUB/data_devhalf $HUB/specialist $HUB/extra $(find $HUB/extra2k $HUB/extra2_hold -name '*.parquet' -printf '%h\n' | sort -u | tr '\n' ' ')"
NCA=(--nc-allow howard-nlp/ibom-mt)
rm -rf $W/gate_all; mkdir -p $W/gate_all
$PD $CH/gate.py fit --src $SRC --scores $W/scores_cpu $W/scores_gpu --refs-scores $W/scores_refs --refs $W/refs_keys --out $W/gate_all --exclude-nc ${NCA[@]} > $LG/gate_all_fit.log 2>&1 || { echo FAIL fit; exit 1; }
$PD $CH/gate.py apply --rule cal --src $SRC --scores $W/scores_cpu $W/scores_gpu --out $W/gate_all --exclude-nc ${NCA[@]} > $LG/gate_all_apply.log 2>&1 || { echo FAIL apply; exit 1; }
$PD $CH/nc_hidden.py merge-keep --keep $W/gate_all/keep_cal.parquet --drop $W/nc_hidden_keys.parquet --out $W/gate_all/keep_final.parquet > $LG/gate_all_merge.log 2>&1 || { echo FAIL merge; exit 1; }
$PD - <<PY || exit 1
import pyarrow.parquet as pq, pyarrow.compute as pc
new = pq.read_table("$W/gate_all/keep_final.parquet"); old = pq.read_table("$W/gate_old/keep_final.parquet")
d = new.filter(pc.invert(pc.is_in(new["key"], value_set=old["key"])))
print("keep now", new.num_rows, "keep in A0", old.num_rows, "delta", d.num_rows, "dropped vs A0", pc.sum(pc.invert(pc.is_in(old["key"], value_set=new["key"]))).as_py())
pq.write_table(d, "$W/gate_all/keep_delta.parquet")
PY
rm -rf $W/windows_delta $W/packed_delta
$PD $CH/windows.py --src $SRC --keep-file $W/gate_all/keep_delta.parquet --out $W/windows_delta --afro-tokenizer $W/tok_afro/tokenizer.json ${NCA[@]} > $LG/windows_delta.log 2>&1 || { echo FAIL windows; exit 1; }
$PD $CH/nc_filter.py guard-windows --windows $W/windows_delta >> $LG/windows_delta.log 2>&1 || { echo FAIL guard; exit 1; }
$PD $CH/pack.py prepare --windows $W/windows_delta --out $W/packed_delta --workers 16 --val-fraction 0 > $LG/pack_delta.log 2>&1 || { echo FAIL pack; exit 1; }
$PD - <<PY
import json, numpy as np, os, collections
d = "$W/packed_delta"; m = json.load(open(d + "/meta.json")); iso = m["iso_table"]
ids = np.load(d + "/train/iso_id.npy"); ps = np.load(d + "/train/pack_start.npy"); sl = np.load(d + "/train/sample_len.npy")
pack_of = np.searchsorted(ps, np.arange(len(ids)), side="right") - 1 if len(ids) == len(sl) else None
c = collections.Counter(); 
for i, n in zip(ids, sl): c[iso[int(i)]] += int(n)
tot = sum(c.values()); print("delta tokens (sample tokens)", tot)
json.dump({k: v for k, v in c.most_common()}, open("$LG/delta_tokens_by_language.json", "w"), indent=1)
print([(k, round(v / 1e6, 1)) for k, v in c.most_common(25)])
PY
echo "$(date -u +%FT%TZ) [DELTA_PACKS_READY] work/packed_delta (new pairs only) + gate_all" >> $ROOT/ALERT.txt
echo "$(date +%T) finished"
