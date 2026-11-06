#!/bin/bash
# E-02 CPU post-processing (npz per system, paired bootstrap), niced; run after the AfriCOMET pass so that the GPU queue can move on.
O=/workspace/e2405/e02; E=/workspace/eval_385m
say() { echo "$(date -u +%FT%TZ) [e02-post] $*" | tee -a $O/log.txt; }
for m in sft rl; do
  cf=$O/$m.comet.json; [ -f $cf ] || cf=none
  PYTHONPATH=$E/pylibs_none:$E/gemma_mt_tree nice -n 15 $E/venv_comet/bin/python $O/../e02_mknpz.py $O/$m.json $cf $O/npz $m >> $O/mknpz.log 2>&1
done
say "npz done"
cd $O; S=""; for f in npz/*.npz; do n=$(basename $f .npz); S="$S $n=$f"; done
nice -n 15 $E/venv_comet/bin/python $O/../ecompare.py --sys $S --out $O/cmp_all --comet-langs-from /dev/shm/evid_e20/mixed.npz --per-language \
  --pair sft_beam4:sft_greedy sft_mbr16:sft_greedy sft_beam4:sft_mbr16 rl_beam4:rl_greedy rl_mbr16:rl_greedy rl_beam4:rl_mbr16 sft_beam4hf:sft_greedy rl_beam4hf:rl_greedy sft_beam4hf:sft_beam4 rl_beam4hf:rl_beam4 \
         rl_greedy:sft_greedy rl_beam4:sft_beam4 rl_mbr16:sft_mbr16 rl_greedy:sft_beam4 rl_greedy:sft_mbr16 rl_beam4:sft_greedy rl_mbr16:sft_greedy > $O/ecompare.log 2>&1
touch $O/E02_POST_DONE; say "E02_POST_DONE"
