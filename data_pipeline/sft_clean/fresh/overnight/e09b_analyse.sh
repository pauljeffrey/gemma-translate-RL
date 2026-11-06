#!/bin/bash
# E-09b analysis on CPU (outputs on /dev/shm): fp32 vs bf16 for SFT and RL, paired bootstrap over the headline cells
S=/workspace/science_sft; E=/workspace/eval_385m; P=$E/venv_comet/bin/python; O=/dev/shm/evid_e09b
mkdir -p $O /dev/shm/tmp; export TMPDIR=/dev/shm/tmp
until [ -f /dev/shm/out_fp32_sft/EVAL_DONE ] && [ -f /dev/shm/out_fp32_rl/EVAL_DONE ]; do sleep 30; done
cp $S/evid/sft.npz $S/evid/mixed.npz $O/
nice -n 19 $P $S/ev_stats.py sft_fp32 /dev/shm/out_fp32_sft/predictions.jsonl $O/sft_fp32.npz /dev/shm/out_fp32_sft/work/comet_cache/comet_ref.jsonl > $O/sft_fp32.log 2>&1 &
nice -n 19 $P $S/ev_stats.py rl_fp32 /dev/shm/out_fp32_rl/predictions.jsonl $O/rl_fp32.npz /dev/shm/out_fp32_rl/work/comet_cache/comet_ref.jsonl > $O/rl_fp32.log 2>&1 &
wait; cat $O/*.log
cd $S; nice -n 19 $P ev_analyse2.py $O $E/contamination_385m_2026-11-03 $E $O/out --systems sft,sft_fp32,mixed,rl_fp32 --pairs sft:sft_fp32,mixed:rl_fp32,sft_fp32:rl_fp32,sft:mixed --tag _e09b 2>&1 | tail -3
echo E09B_ANALYSIS_DONE
