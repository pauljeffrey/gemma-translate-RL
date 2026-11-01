#!/bin/bash
# tmux window qe_cpu:push : every 2 h (and on request) push the score shards + progress to the Hub folder qe_scores_cpu/ (keys and scores only, no pair text)
cd /workspace/work
source /workspace/tr/env.sh
while true; do
  venv/bin/python chain/qe_cpu_push.py >> qe_cpu/push.log 2>&1
  sleep ${PUSH_EVERY:-7200}
done
