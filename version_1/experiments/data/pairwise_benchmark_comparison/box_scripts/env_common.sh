export PYTHONPATH=/workspace/jobs/e1/code2c:/workspace/sft_build/pylibs_tf4:/workspace/sft_build/pylibs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
set +u; source /workspace/sft_build/env.sh; set -u
export HF_HOME=/workspace/sft_build/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv_gemma/bin/python
