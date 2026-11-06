#!/bin/bash
source /workspace/science_sft/overnight/common.sh
mkdir -p /dev/shm/tmp /dev/shm/vllm_cache_ov; export TMPDIR=/dev/shm/tmp
export GEMMA_MT_TAG_MODE=single_token HF_HOME=/workspace/clean_sft/hf_home
step E01_ext_sft 1800 bash -c "cd $E/gemma_mt_tree && PYTHONPATH=$E/gemma_mt_tree VLLM_CACHE_ROOT=/dev/shm/vllm_cache_ov /venv/main/bin/python $O/guard_ext.py --model sft=$R/models/g15_385m_sft_final --dev $R/assets/dev/bench_dev_hsel.jsonl --out $O/guard_grid_ext_sft.json --gpu-util 0.15"
