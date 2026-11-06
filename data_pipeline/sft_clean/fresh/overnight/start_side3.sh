#!/bin/bash
cd /workspace/science_sft
for p in $(pgrep -f "bash overnight/chain_side2.sh"); do kill $p; done
sleep 1
setsid nohup bash overnight/chain_side3.sh >/dev/null 2>&1 < /dev/null &
sleep 3; pgrep -af "chain_side" | cut -c1-90; tail -2 overnight/overnight.log | cut -c1-200
