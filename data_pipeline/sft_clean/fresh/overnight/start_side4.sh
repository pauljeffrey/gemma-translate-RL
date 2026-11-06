#!/bin/bash
cd /workspace/science_sft
# stop ONLY the chain script (its running side job child keeps running; chain_side4 waits for it)
for p in $(pgrep -f "bash overnight/chain_side3.sh"); do kill $p; done
sleep 1
pgrep -af "run_fp32_eval|sent_neural_all" | cut -c1-110
setsid nohup bash overnight/chain_side4.sh >/dev/null 2>&1 < /dev/null &
sleep 3; pgrep -af "chain_side" | cut -c1-90; tail -3 overnight/overnight.log | cut -c1-220; date -u
