#!/bin/bash
# usage: codestate.sh OUTFILE  - code/config/env state of the run: sha256 of the trainer tree, vLLM/torch versions, GPU, git commit pin of the launcher
O=$1; T=/workspace/clean_rerun/tree/gemma_mt
{ echo "date $(date -u +%FT%TZ)"; echo "GIT_COMMIT_PIN $(grep -o 'GIT_COMMIT=[0-9a-f]*' /workspace/clean_rerun/rl_env.sh)"
  echo "venv_gemma $(/workspace/venv_gemma/bin/python -c 'import torch,transformers;print("torch",torch.__version__,"transformers",transformers.__version__)' 2>&1 | tail -1)"
  echo "venv_vllm  $(/workspace/venv_vllm/bin/python -c 'import vllm;print("vllm",vllm.__version__)' 2>&1 | tail -1)"
  echo "venv_main  $(/venv/main/bin/python -c 'import vllm;print("vllm",vllm.__version__)' 2>&1 | tail -1)"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  echo "--- sha256 of gemma_mt/rl/*.py, gemma_mt/*.py and launcher"; (cd $T && sha256sum gemma_mt/rl/*.py gemma_mt/*.py scripts/launch_clean_rerun.sh scripts/launch_clean_rerun_385m.sh scripts/clean_rerun_tools.py 2>/dev/null)
  echo "--- start model"; sha256sum /workspace/clean_rerun/models/g15_385m_sft_final/model.safetensors /dev/shm/rl_mixed_ckpt/best/model.safetensors 2>/dev/null
  echo "--- reward assets"; sha256sum /workspace/clean_rerun/assets/calibration/*.json 2>/dev/null | head -20
} > $O 2>&1
