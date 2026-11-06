#!/bin/bash
cd /workspace/science_sft
for p in $(pgrep -f "bash overnight/chain_p3.sh") $(pgrep -f "bash e09b_analyse.sh"); do kill $p; done
sleep 1
setsid nohup bash overnight/chain_side.sh >/dev/null 2>&1 < /dev/null &
setsid nohup bash e09b_analyse.sh > e09b_analyse.out 2>&1 < /dev/null &
sleep 4; pgrep -af "chain_side|e09b_analyse|chain_p" | cut -c1-90; tail -3 overnight/overnight.log | cut -c1-200
