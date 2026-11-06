#!/bin/bash
# usage: e11_analyse.sh ARM [ARM ...]   (ARM tags: a_base c_noref d_noguide e_kl025 that have a finished evaluation) ; CPU only, niced
cd /workspace/e2405; E=/workspace/eval_385m; V=$E/venv_comet/bin/python; mkdir -p e11
L=yor,hau,ibo,swh,amh,zul,xho,kin,som,wol,lug,sna,twi,ewe,lin,tir,orm,nso,tsn,afr
SYS="sft=base/sft.npz stageA=base/stageA.npz mixed=base/mixed.npz s2=e04/e04_s2.npz s3=e04/e04_s3.npz"; PAIRS=""; DEV=""
for a in "$@"; do
  [ -f e11/e04_$a.npz ] || nice -n 15 $V e_iloop.py /dev/shm/evid_e2405/e04_$a.npz e11/e04_$a.npz $E/out_e04_$a/predictions.jsonl --langs $L | tail -1
  SYS="$SYS $a=e11/e04_$a.npz"; PAIRS="$PAIRS $a:s2 $a:s3 $a:sft"
  DEV="$DEV $a=/workspace/clean_rerun/runs_e04_$a/A"
done
bash e04_analyse.sh /workspace/e2405/e11/final $SYS -- $PAIRS s2:s3 > /dev/null 2>&1
nice $V dev_compare.py 100 s2=/workspace/clean_rerun/runs_e04_s2/A $DEV --pair $(for a in "$@"; do echo -n "$a:s2 "; done) > e11/dev_compare_step100.txt 2>&1
nice $V dev_compare.py 50 s2=/workspace/clean_rerun/runs_e04_s2/A $DEV --pair $(for a in "$@"; do echo -n "$a:s2 "; done) > e11/dev_compare_step50.txt 2>&1
for a in "$@"; do python3 e04_summ.py /workspace/clean_rerun/runs_e04_$a/A e11/summ_$a.json > /dev/null; done
echo E11_ANALYSE_DONE
