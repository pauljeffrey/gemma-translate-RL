#!/bin/bash
S=/workspace/science_sft; R=/workspace/clean_rerun
while [ ! -f $R/runs_385m_long/final/summary.json ] || pgrep -f "[g]emma_mt.rl.train_rl" >/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) long RL finished: post-RL evaluation starts" >> $S/post_long_rl.log
bash $S/post_long_rl_eval.sh
