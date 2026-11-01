#!/bin/bash
# a1: final RL model and SFT-final on the repo benchmark pipeline (greedy, vLLM worker), no AfriCOMET, no base.
cd /workspace/jobs/a1; export A1_JOB=1
export CONTAM_DIR=/workspace/sft_build/bench/test_sets/_contamination OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BENCHMARKS=flores_plus,ntrex,mafand,smol_sent,ibom_mt,flores_moz,glot_story,salt
CODE=/workspace/sft_build/gemma_mt_2c
for sys in rl sft; do
  if [ $sys = rl ]; then M=${HF_ORG_LEGACY}/gemma-3-270m-translate-rl2c; R=latest; else M=${HF_ORG}/gemma3-270m-translate; R=step-27638; fi
  [ -f out_$sys/EVAL_DONE ] && continue
  OUT=/workspace/jobs/a1/out_$sys SFT_REVISION=$R GPU_UTIL=0.1 nice -n 10 bash $CODE/scripts/eval_sft_benchmarks.sh $M --skip-base --no-comet --max-new-tokens 768 > launch_$sys.log 2>&1
done
echo ALLDONE > ALLDONE
