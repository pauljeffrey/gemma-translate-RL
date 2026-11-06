#!/bin/bash
cd /workspace/science_sft
for p in $(pgrep -f "bash overnight/chain_p1.sh"); do kill $p; done
sleep 1
setsid nohup bash overnight/chain_p1.sh >/dev/null 2>&1 < /dev/null &
sleep 2; pgrep -af "bash overnight/chain_p" | cut -c1-90; tail -2 overnight/overnight.log
