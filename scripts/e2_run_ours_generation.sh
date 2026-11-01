#!/bin/bash
# E2: FLORES+ devtest generations of our four models with the same pipeline/decoding as the final table (greedy, vLLM, 768 new tokens)
cd /workspace/jobs/e2; export A1_JOB=1
export CONTAM_DIR=/workspace/sft_build/bench/test_sets/_contamination OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BENCHMARKS=flores_plus
CODE=/workspace/sft_build/gemma_mt_2c
run() {  # name model revision
  local n=$1 M=$2 R=$3
  [ -f out_$n/EVAL_DONE ] && return
  for try in 1 2 3 4 5 6; do
    SAMPLER_FALLBACK=0 OUT=/workspace/jobs/e2/out_$n SFT_REVISION=$R GPU_UTIL=0.05 GPU_MIN_FREE_GIB=5 BENCHMARKS=flores_plus nice -n 10 bash $CODE/scripts/eval_sft_benchmarks.sh $M --skip-base --no-comet --max-new-tokens 768 >> launch_$n.log 2>&1
    [ -f out_$n/EVAL_DONE ] && break
    sleep 120
  done
}
run a2w /workspace/jobs/r7/models/sci_sft_A2W_alpha0.7 ""
run live250 /dev/shm/e2/live_latest ""
run sft ${HF_ORG}/gemma3-270m-translate step-27638
run rl /workspace/sft_build/runs/final_1433 ""
echo done > OURS_GEN_DONE
