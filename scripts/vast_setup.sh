#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# One-time environment setup on a fresh Vast.ai instance (Linux, CUDA GPU, 11-24 GB).
#   bash scripts/vast_setup.sh
# Creates .venv in the module root, installs the pinned requirements, optionally flash-attn
# (INSTALL_FLASH_ATTN=1), and runs the offline unit tests as a sanity check. Needs HF_TOKEN in the environment
# for gated assets (FLORES+, Gemma). Never writes the token to disk.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
# torch: pick the wheel that matches the instance's CUDA driver; the default PyPI wheel works on most images.
python -m pip install -r requirements.txt
if [ "${INSTALL_FLASH_ATTN:-0}" = "1" ]; then
  python -m pip install flash-attn --no-build-isolation
fi
if [ "${INSTALL_FASTTEXT:-0}" = "1" ]; then
  python -m pip install fasttext-wheel
fi
python -m pip install pytest

python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(p.name, round(p.total_memory / 2**30, 1), "GiB", "bf16 supported:", torch.cuda.is_bf16_supported())
PY
python -m pytest -q tests
echo "setup done. Next: scripts/vast_prepare.sh"
