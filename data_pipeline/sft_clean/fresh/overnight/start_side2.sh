#!/bin/bash
cd /workspace/science_sft
setsid nohup bash overnight/chain_side2.sh >/dev/null 2>&1 < /dev/null &
sleep 3; pgrep -af "chain_side2" | cut -c1-90
