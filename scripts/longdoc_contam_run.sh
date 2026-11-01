#!/bin/bash
# Incremental contamination scan of the benchmark pool on the rental box (CPU only, nice 10, 14 processes). Extra arguments are passed through
# (e.g. --force, --only <stem> ..., --clean-below 0.02). Log: longdoc_bench/logs/contam_scan_<timestamp>.log
# Writes longdoc_bench/contam/<pool stem>.flags.jsonl and longdoc_bench/contam/SUMMARY.json.
cd /workspace/sft_build/longdoc_bench/code || exit 1
export PYTHONPATH=/workspace/sft_build/longdoc_bench/code:/workspace/sft_build/pylibs CUDA_VISIBLE_DEVICES=""
LOG=/workspace/sft_build/longdoc_bench/logs/contam_scan_$(date +%Y%m%d_%H%M%S).log
nice -n 10 /workspace/venv_gemma/bin/python scripts/longdoc_contam_scan.py \
  --pool-dir /workspace/sft_build/longdoc_bench/pool --out-dir /workspace/sft_build/longdoc_bench/contam \
  --sft-dir /workspace/sft_build/gemma/packed_fwd --rl-dir /workspace/sft_build/gemma/rl_prompts_v3 \
  --bench-root /workspace/sft_build/bench/test_sets --extra-bench bench_dev=/workspace/sft_build/bench/bench_dev.jsonl \
  --tokenizer "/workspace/sft_build/hf/hub/models--google--gemma-3-270m-it/snapshots/*/tokenizer.json" --processes 14 "$@" > "$LOG" 2>&1
echo "log: $LOG"
