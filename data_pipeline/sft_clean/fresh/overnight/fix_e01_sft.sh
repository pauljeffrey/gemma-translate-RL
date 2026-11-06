#!/bin/bash
# Two instances of the SFT guard evaluation (the chain's, whose script I replaced while it ran, and my rerun) wrote to the same OUT and one fell back to HF: kill both, discard OUT, start ONE clean vLLM-only rerun.
source /workspace/science_sft/overnight/common.sh
M=/workspace/clean_rerun/models/g15_385m_sft_final
kill_tree() { for c in $(pgrep -P $1); do kill_tree $c; done; kill $1 2>/dev/null; }
for p in $(pgrep -f "run_guard_eval.sh sft $M") $(pgrep -f "rerun_e01_sft.sh") $(pgrep -f "eval_sft_benchmarks.sh $M"); do kill_tree $p; done
for p in $(pgrep -f "sft_benchmarks --sft-model $M"); do kill_tree $p; done
sleep 8
pgrep -af "sft_benchmarks|run_guard_eval" | cut -c1-120
mv $E/out_guard_sft $E/out_guard_sft_DISCARDED_hf_fallback2 2>/dev/null
say "E-01 SFT: duplicate/HF-fallback instances killed, OUT discarded; ONE clean vLLM-only rerun starts (chain continues with E01_rl)"
step E01_sft_rerun 10800 bash $O/run_guard_eval.sh sft $M
