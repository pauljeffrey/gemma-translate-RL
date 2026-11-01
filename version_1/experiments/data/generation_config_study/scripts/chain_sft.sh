#!/bin/bash
cd /workspace/jobs/r12
while [ ! -f logs/chain_doc2.done ]; do sleep 30; done
export MAX_SEQS=96 MAXBT=2048 EAGER=1 GPU_UTIL=0.028 MODEL_PATH=/workspace/jobs/a2/models/sft NAME_PREFIX=sft_
V=/workspace/sft_build/vllm_cache; export VLLM_CACHE_ROOT=$V/vllm TRITON_CACHE_DIR=$V/triton TORCHINDUCTOR_CACHE_DIR=$V/inductor XDG_CACHE_HOME=$V/xdg VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONDONTWRITEBYTECODE=1
source /workspace/sft_build/env.sh
for t in 1 2 3 4; do nice -n 19 /workspace/venv_vllm/bin/python gen_sent.py greedy T0.3 T0.5 T0.7 T1.0 T1.0_p0.9 T1.0_mp0.05 T1.0_mp0.1 T0.7_mp0.05 T0.7_n16 >> logs/sft_gen.log 2>&1 && break; sleep 30; done
./envpy.sh sft_eval.py >> logs/sft_eval.log 2>&1
echo done > logs/chain_sft.done
