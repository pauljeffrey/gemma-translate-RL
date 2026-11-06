#!/bin/bash
# E-01 SFT: the first attempt started while the post-science evaluation held the GPU, the vLLM worker failed to start and the harness fell back to HF generate (which ignores the guard): discarded.
source /workspace/science_sft/overnight/common.sh
for p in $(pgrep -f "eval_sft_benchmarks.sh /workspace/clean_rerun/models/g15_385m_sft_final"); do kill $p; done
for p in $(pgrep -f "gemma_mt.eval.sft_benchmarks --sft-model /workspace/clean_rerun/models/g15_385m_sft_final"); do kill $p; done
sleep 5
[ -d $E/out_guard_sft ] && mv $E/out_guard_sft $E/out_guard_sft_DISCARDED_hf_fallback
say "E-01 SFT first attempt (HF fallback, guard ignored) killed and moved to out_guard_sft_DISCARDED_hf_fallback; rerun with vLLM concurrently with the chain"
SFT=$R/models/g15_385m_sft_final
step E01_sft_rerun 10800 bash $O/run_guard_eval.sh sft $SFT
