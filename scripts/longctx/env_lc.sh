export HF_HOME=/workspace/sft_build/hf HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export PYTHONPATH=/workspace/jobs/lc1:/workspace/sft_build/gemma_mt_2c:/workspace/sft_build/pylibs
source /workspace/sft_build/env.sh >/dev/null 2>&1
export PYG=/workspace/venv_gemma/bin/python PYV=/workspace/venv_vllm/bin/python
