#!/bin/bash
# E-13 analysis (CPU, outputs on /dev/shm): old 270M SFT / RL (rl2c) re-run in the 385M harness vs the 385M SFT / RL, on rows that are clean for BOTH generations (385M flags OR old flags removed)
S=/workspace/science_sft; E=/workspace/eval_385m; P=$E/venv_comet/bin/python; O=/dev/shm/evid_e13
mkdir -p $O/contam /dev/shm/tmp; export TMPDIR=/dev/shm/tmp
until [ -f $E/out_old_sft/EVAL_DONE ] && [ -f $E/out_old_rl2c/EVAL_DONE ]; do sleep 30; done
cp $S/evid/sft.npz $S/evid/mixed.npz $O/
$P - <<PY
import json
for b in ("flores_plus", "ntrex", "mafand"):
    a = json.load(open("$E/contamination_385m_2026-11-03/contamination_%s.json" % b)); o = json.load(open("$E/old_flags/contamination_%s.json" % b))
    m = {}
    for d in (a, o):
        for part in ("exact", "sentence", "near"):
            for k, v in d.get(part, {}).items():
                if v: m[k] = True
    json.dump({"meta": {"benchmark": b, "note": "union of the 385M audit flags and the old (first research) audit flags"}, "exact": m, "sentence": {}, "near": {}, "pair": {}}, open("$O/contam/contamination_%s.json" % b, "w"))
    print(b, "flagged keys", len(m))
PY
nice -n 19 $P $S/ev_stats.py old_sft $E/out_old_sft/predictions.jsonl $O/old_sft.npz $E/out_old_sft/work/comet_cache/comet_ref.jsonl > $O/old_sft.log 2>&1 &
nice -n 19 $P $S/ev_stats.py old_rl2c $E/out_old_rl2c/predictions.jsonl $O/old_rl2c.npz $E/out_old_rl2c/work/comet_cache/comet_ref.jsonl > $O/old_rl2c.log 2>&1 &
wait; cat $O/old_*.log
LANGS=afr,amh,bam,bem,dyu,ewe,fon,hau,ibo,kab,kam,kbp,kea,kik,kin,kmb,knc,lin,lua,lug,luo,mlg,mos,nde,nso,nus,nya,orm,pcm,run,sag,shi,sna,som,sot,ssw,swh,tir,tsn,tso,tum,twi,umb,ven,vmw,wol,xho,yor,zul
cd $S; nice -n 19 $P ev_analyse2.py $O $O/contam $E $O/out --systems old_sft,old_rl2c,sft,mixed --pairs old_sft:old_rl2c,sft:mixed,old_sft:sft,old_rl2c:mixed --languages $LANGS --tag _e13 2>&1 | tail -3
echo E13_ANALYSIS_DONE
