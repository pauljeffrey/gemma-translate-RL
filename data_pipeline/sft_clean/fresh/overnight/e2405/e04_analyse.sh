#!/bin/bash
# usage: e04_analyse.sh OUTDIR NAME=npz ... -- PAIR ...    (CPU; 20-language scope; contamination-flagged rows removed; 1000 resamples seed 20261104)
O=$1; shift; mkdir -p $O; SYS=(); while [ "$1" != "--" ]; do SYS+=("$1"); shift; done; shift
L=yor,hau,ibo,swh,amh,zul,xho,kin,som,wol,lug,sna,twi,ewe,lin,tir,orm,nso,tsn,afr
nice /workspace/eval_385m/venv_comet/bin/python /workspace/e2405/ecompare.py --sys "${SYS[@]}" --langs $L --comet-langs-from /dev/shm/evid_e20/mixed.npz --per-language --out $O/cmp --pair "$@" > $O/cmp.log 2>&1
tail -3 $O/cmp.log
