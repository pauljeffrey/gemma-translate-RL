#!/bin/bash
# usage: URL=<presigned> bash build_weak_data.sh STEPS
cd /workspace/jobs/ab1
export PYTHONPATH=/workspace/jobs/ab1/gemma_mt
curl -sSL --retry 3 "$URL" | tee >(sha256sum > data/packed_fwd.sha256.txt) | zstd -dc | nice -n 10 /workspace/venv_gemma/bin/python gemma_mt/scripts/ab1_build_weak_sft_data.py --out /workspace/jobs/ab1/data/packed_fwd --steps $1
echo exit $? 
