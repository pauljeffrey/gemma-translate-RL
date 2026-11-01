#!/bin/bash
cd /workspace/jobs/r12
true
./envpy.sh rl/rl_bias.py 100 >> logs/rl_bias.log 2>&1
./envpy.sh rl/rl_analyze.py >> logs/rl_analyze.log 2>&1
echo done > logs/chain_rl2.done
