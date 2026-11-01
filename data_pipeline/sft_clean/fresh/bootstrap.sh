#!/bin/bash
# On the INSTANCE (once; idempotent). Two environments, as before: comet pins numpy<2 / transformers<5, the data tools use numpy 2.
#   venv_comet : AfriCOMET-QE (unbabel-comet 2.2.7, CUDA torch, pyarrow<20)           -> qe_score, qe_crosscheck, hcal refs scoring
#   venv_data  : pyarrow, xxhash, tokenizers, transformers (chat template for prepare_sft) -> gate, windows, pack, sampler, nc_filter
#   venv_vllm  : only with WITH_VLLM=1 (vllm 0.31.0, cu130; the evaluation/RL stages, not this chain)
set -euo pipefail
source "${ROOT:-/workspace/work}/env.sh" 2>/dev/null || true
ROOT=${ROOT:-/workspace/work}; mkdir -p "$ROOT"/{code,state,logs,hub,work}
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "WARNING: no nvidia-smi (CPU only)"
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH)
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux rsync)
cd "$ROOT"
[ -d venv_data ] || { uv venv --python 3.12 venv_data -q; uv pip install -q --python venv_data/bin/python "numpy>=2" pyarrow xxhash tokenizers "transformers>=5,<6" jinja2 sentencepiece protobuf huggingface_hub pytest; }
if [ ! -d venv_comet ]; then
  uv venv --python 3.12 venv_comet -q
  uv pip install -q --python venv_comet/bin/python torch --torch-backend=auto        # picks the CUDA build that matches the driver (RTX PRO 6000 = sm_120 needs CUDA >= 12.8)
  uv pip install -q --python venv_comet/bin/python "unbabel-comet==2.2.7" "numpy<2" "pyarrow<20" xxhash "huggingface_hub<1" tokenizers pytest "setuptools<81"   # torchmetrics (comet) imports pkg_resources, removed in setuptools>=81
fi
if [ "${WITH_VLLM:-0}" = "1" ] && [ ! -d venv_vllm ]; then
  uv venv --python 3.12 venv_vllm -q; uv pip install -q --python venv_vllm/bin/python vllm==0.31.0 --torch-backend cu130
fi
venv_comet/bin/python -c "import torch,comet;print('comet env: torch',torch.__version__,'cuda',torch.cuda.is_available())"
venv_data/bin/python -c "import numpy,pyarrow,transformers,tokenizers;print('data env: numpy',numpy.__version__,'pyarrow',pyarrow.__version__,'transformers',transformers.__version__)"
echo "bootstrap OK"
