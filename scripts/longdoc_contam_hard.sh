#!/bin/bash
# Contamination scan of the HARD-split candidates (CPU, nice 10, 10 processes) in a PRIVATE pool directory: symlinks to every pool file plus the staged hard_*.jsonl files, and a copy of
# the coordinator's scan cache, so that only the new files are scanned (one pass over the SFT stream / RL prompts v3 / benchmarks) and the duplicate check runs over ALL pool documents.
# The coordinator's contam/ directory is not touched.  Extra arguments are passed to longdoc_contam_scan.py.  Output: longdoc_bench/contam_hard/<stem>.flags.jsonl, SUMMARY.json.
B=/workspace/sft_build/longdoc_bench
mkdir -p $B/contam_hard/cache $B/pool_hard_scan $B/logs
cp -n $B/contam/cache/* $B/contam_hard/cache/ 2>/dev/null
for f in $B/pool/*.jsonl $B/hard_stage/hard_*.jsonl; do ln -sf "$f" $B/pool_hard_scan/; done
cd $B/code || exit 1
export PYTHONPATH=$B/code:/workspace/sft_build/pylibs CUDA_VISIBLE_DEVICES=""
LOG=$B/logs/contam_hard_$(date +%Y%m%d_%H%M%S).log
nice -n 10 /workspace/venv_gemma/bin/python scripts/longdoc_contam_scan.py \
  --pool-dir $B/pool_hard_scan --out-dir $B/contam_hard \
  --sft-dir /workspace/sft_build/gemma/packed_fwd --rl-dir /workspace/sft_build/gemma/rl_prompts_v3 \
  --bench-root /workspace/sft_build/bench/test_sets --extra-bench bench_dev=/workspace/sft_build/bench/bench_dev.jsonl \
  --tokenizer "/workspace/sft_build/hf/hub/models--google--gemma-3-270m-it/snapshots/*/tokenizer.json" --processes 10 "$@" > "$LOG" 2>&1
echo "log: $LOG"
