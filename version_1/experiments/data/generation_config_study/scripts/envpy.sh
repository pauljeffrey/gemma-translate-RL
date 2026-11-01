#!/bin/bash
# run python (venv_gemma) in r12 dir
cd /workspace/jobs/r12; export OMP_NUM_THREADS=4 PYTHONPATH=/workspace/sft_build/gemma_mt_2c:/workspace/sft_build/pylibs PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
source /workspace/sft_build/env.sh
exec nice -n 19 /workspace/venv_gemma/bin/python "$@"
