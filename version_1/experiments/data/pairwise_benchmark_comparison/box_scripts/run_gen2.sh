#!/bin/bash
# e1 gen v2 (after the coordinator's GPU-etiquette message): ONE vLLM engine at a time, util 0.04, short context, 128 seqs, nice 19, no HF fallback, wait-and-retry.
cd /workspace/jobs/e1; export A1_JOB=1
export CONTAM_DIR=/workspace/sft_build/bench/test_sets/_contamination OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
ALL=flores_plus,ntrex,mafand,smol_sent,ibom_mt,flores_moz,glot_story,salt
CODE=/workspace/jobs/e1/code2c; export E1_MNBT=1024
run() {  # name model revision maxnew benchmarks
  local n=$1 M=$2 R=$3 T=$4 B=$5
  for try in $(seq 1 30); do
    [ -f out_$n/EVAL_DONE ] && return
    OUT=/workspace/jobs/e1/out_$n SFT_REVISION=$R GPU_UTIL=0.05 GPU_MIN_FREE_GIB=7 SAMPLER_FALLBACK=0 BENCHMARKS=$B nice -n 19 bash $CODE/scripts/eval_sft_benchmarks.sh $M --skip-base --no-comet --max-new-tokens $T --vllm-max-model-len 2048 --vllm-max-num-seqs 128 >> launch_$n.log 2>&1
    [ -f out_$n/EVAL_DONE ] && return
    sleep 180
  done
}
run sft ${HF_ORG}/gemma3-270m-translate step-27638 768 $ALL
run rl /workspace/sft_build/runs/final_1433 "" 768 $ALL
run sft320 ${HF_ORG}/gemma3-270m-translate step-27638 320 flores_plus
run rl320 /workspace/sft_build/runs/final_1433 "" 320 flores_plus
echo ALLDONE > ALLDONE_GEN
