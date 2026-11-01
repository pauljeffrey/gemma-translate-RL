#!/bin/bash
# wait for the sentence generation queue, then RL rollouts -> rewards -> bias (strictly sequential, <= 7 GB GPU at any time)
cd /workspace/jobs/r12
while [ ! -f logs/chain2.done ]; do sleep 30; done
V=/workspace/sft_build/vllm_cache; export VLLM_CACHE_ROOT=$V/vllm TRITON_CACHE_DIR=$V/triton TORCHINDUCTOR_CACHE_DIR=$V/inductor XDG_CACHE_HOME=$V/xdg VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONDONTWRITEBYTECODE=1
source /workspace/sft_build/env.sh
NAMES="T1.0 T1.0_p0.95 T1.0_mp0.05 T1.0_p0.9 T0.7 T0.5 T1.0_mp0.1 T1.0_k64_p0.95 T1.0_p0.8 T0.8 T0.3"
for t in 1 2 3 4 5 6; do GPU_UTIL=0.028 MAXBT=2048 nice -n 19 /workspace/venv_vllm/bin/python rl/rl_gen.py $NAMES >> logs/rl_gen.log 2>&1 && break; sleep 30; done
./envpy.sh rl/rl_reward.py greedy T1.0 T1.0+g T1.0_p0.95 T1.0_mp0.05 T1.0_p0.9 T0.7 T0.5 T1.0_mp0.1 T1.0_k64_p0.95 T1.0_p0.8 T0.8 T0.3 >> logs/rl_reward.log 2>&1
./envpy.sh rl/rl_bias.py >> logs/rl_bias.log 2>&1
echo done >> logs/chain_rl.done
