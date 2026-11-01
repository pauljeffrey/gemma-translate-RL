#!/bin/bash
# a1: same pipeline with the 320-token budget of the older reports (SFT: all benchmarks; RL: FLORES+ only) to separate the loop-truncation effect
cd /workspace/jobs/a1; export A1_JOB=1 CONTAM_DIR=/workspace/sft_build/bench/test_sets/_contamination OMP_NUM_THREADS=4
CODE=/workspace/sft_build/gemma_mt_2c
OUT=/workspace/jobs/a1/out_sft320 SFT_REVISION=step-27638 GPU_UTIL=0.1 BENCHMARKS=flores_plus,ntrex,mafand,smol_sent,ibom_mt,flores_moz,glot_story,salt nice -n 10 bash $CODE/scripts/eval_sft_benchmarks.sh ${HF_ORG}/gemma3-270m-translate --skip-base --no-comet --max-new-tokens 320 > launch_sft320.log 2>&1
OUT=/workspace/jobs/a1/out_rl320 SFT_REVISION=latest GPU_UTIL=0.1 BENCHMARKS=flores_plus nice -n 10 bash $CODE/scripts/eval_sft_benchmarks.sh ${HF_ORG_LEGACY}/gemma-3-270m-translate-rl2c --skip-base --no-comet --max-new-tokens 320 > launch_rl320.log 2>&1
echo done > ALLDONE320
