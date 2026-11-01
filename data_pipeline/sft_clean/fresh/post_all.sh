#!/bin/bash
# Runs on the DATA box after finish_all.sh ([DELTA_PACKS_READY]). Growth 1, phase 2 (phase 1 = build_g0.sh: packed_g0 + stage-A schedule, shipped to the A100 earlier):
#   retag_packs.py: the new translation packs get the single-token language tags               -> packed_delta_tag
#   reference copy of the final set  packed_g = packed_g0 + packed_delta_tag (APPENDED at the end, indices of packed_g0 unchanged)
#   make_stage_schedules.py B: ONE epoch of every pack that stage A did not draw               -> sched_g/B  (sidecar pins packed_g)
#   bundle_delta/: packed_delta_tag + sched_g/B + checksums; the A100 appends packed_delta_tag to its packed_g0 IN PLACE (extend_packed.py --inplace) when stage A is done and checks
#   the arrays against packed_g_arrays.sha256. Writes [DELTA_BUNDLE_READY] to ALERT.txt; nothing is trained here.
set -uo pipefail
ROOT=${ROOT:-/workspace/work}; cd $ROOT; source env.sh
export HF_HOME=$ROOT/hf_home PYTHONPATH=$ROOT/code/sft_clean/chain:$ROOT/code GEMMA_MT_PKG=$ROOT/code
CH=$ROOT/code/sft_clean/chain; FR=$ROOT/code/sft_clean/fresh; PD=$ROOT/venv_data/bin/python; W=$ROOT/work; LG=$ROOT/logs
fail() { echo "$(date -u +%FT%TZ) [POST_ALL_FAILED] $1" >> $ROOT/ALERT.txt; echo "FAIL $1"; exit 1; }
until grep -q "DELTA_PACKS_READY" $ROOT/ALERT.txt 2>/dev/null && grep -q "G0_READY" $ROOT/ALERT.txt 2>/dev/null; do sleep 60; done
echo "$(date +%T) inputs ready"
rm -rf $W/packed_delta_tag $W/packed_g $W/sched_g/B $W/bundle_delta
$PD $CH/retag_packs.py --src $W/packed_delta --out $W/packed_delta_tag --verify 200 > $LG/retag_delta.log 2>&1 || fail retag_delta
$PD $CH/extend_packed.py --base $W/packed_g0 --add $W/packed_delta_tag --out $W/packed_g > $LG/extend_g.log 2>&1 || fail extend_g
NOLD=$(awk '{print $2}' $LG/g0_nold.txt)
$PD $FR/make_stage_schedules.py B --packed $W/packed_g --stage-a $W/sched_g/A --out $W/sched_g/B > $LG/stageB_schedule.log 2>&1 || fail schedB
tail -n 3 $LG/stageB_schedule.log
# per-language pack counts of the final set (for the report)
cat > $W/packs_per_language.py <<'PYEOF'
import csv, sys
import numpy as np
import sampler as S
PD_, ALL = S._pd()
ds = PD_.PackedDataset(sys.argv[1], "train", pad=False)
pl = np.asarray(ds.pack_languages(), dtype=np.int64); c = np.bincount(pl, minlength=len(ALL))
with open(sys.argv[2], "w") as f:
    w = csv.writer(f); w.writerow(["iso", "packs"])
    for i in np.flatnonzero(c): w.writerow([ALL[i], int(c[i])])
print("packs", len(pl), "languages", int((c > 0).sum()))
PYEOF
$PD $W/packs_per_language.py $W/packed_g $W/packs_per_language_g.csv | tail -n 1
# what the A100 needs
mkdir -p $W/bundle_delta/sched_g; ln -s $W/packed_delta_tag $W/bundle_delta/packed_delta_tag; ln -s $W/sched_g/B $W/bundle_delta/sched_g/B
( cd $W/packed_delta_tag && find . -type f | sort | xargs sha256sum > $W/bundle_delta/packed_delta_tag.sha256 )
( cd $W/sched_g/B && find . -type f | sort | xargs sha256sum > $W/bundle_delta/sched_g_B.sha256 )
( cd $W/packed_g && sha256sum train/*.npy train/tokens.u32.bin > $W/bundle_delta/packed_g_arrays.sha256 )
du -shL $W/bundle_delta/packed_delta_tag | tail -n 1
echo "$(date -u +%FT%TZ) [DELTA_BUNDLE_READY] work/bundle_delta: packed_delta_tag + sched_g/B (n_old $NOLD)" >> $ROOT/ALERT.txt
echo "$(date +%T) done"
