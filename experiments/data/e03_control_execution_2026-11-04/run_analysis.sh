#!/bin/bash
# E-03 paired comparison control vs RL on identical sentences (CPU only). Primary pair "CONTROL:RL" -> diff = RL - CONTROL (positive = RL better).
S=/workspace/science_sft; E=/workspace/eval_385m; P=$E/venv_comet/bin/python; O=/dev/shm/evid_c1; OUT=$O/out
mkdir -p $OUT /dev/shm/tmp; export TMPDIR=/dev/shm/tmp
for s in sft stageA b1 mixed; do cp $S/evid/$s.npz $O/; done
SYS=sft,stageA,b1,mixed,c1_s6000
TP19=afr,amh,hau,ibo,kin,lin,lug,mlg,nya,orm,sna,som,sot,swh,tsn,wol,xho,yor,zul
OTH=$($P - <<PY
import numpy as np
z=np.load("$O/sft.npz",allow_pickle=True); tp=set("$TP19".split(","))
print(",".join(sorted(set(z["iso"].tolist())-tp)))
PY
)
cd $S
run(){ tag=$1; shift; nice -n 19 $P ev_analyse2.py $O $E/contamination_385m_2026-11-03 $E $OUT --systems $SYS --tag $tag "$@" 2>&1 | tail -2; }
# owner instruction 2026-11-05: only the 6,000-step checkpoint is evaluated (3,100 / 4,800 not evaluated)
run _mixed_vs_c6000 --pairs c1_s6000:mixed,c1_s6000:stageA,c1_s6000:b1,sft:c1_s6000,sft:mixed,sft:stageA,sft:b1
run _stageA_vs_c6000 --pairs c1_s6000:stageA,sft:c1_s6000,sft:stageA
run _b1_vs_c6000 --pairs c1_s6000:b1,sft:c1_s6000,sft:b1
run _mixed_vs_c6000_TP19 --pairs c1_s6000:mixed,sft:c1_s6000,sft:mixed --languages $TP19
run _mixed_vs_c6000_OTHERS --pairs c1_s6000:mixed,sft:c1_s6000,sft:mixed --languages $OTH
run _stageA_vs_c6000_TP19 --pairs c1_s6000:stageA,sft:c1_s6000,sft:stageA --languages $TP19
run _stageA_vs_c6000_OTHERS --pairs c1_s6000:stageA,sft:c1_s6000,sft:stageA --languages $OTH
echo ANALYSIS_DONE
