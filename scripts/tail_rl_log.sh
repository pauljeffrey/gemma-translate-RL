#!/bin/bash
[ -f .env ] && { set -a; . ./.env; set +a; }   # SSH_KEY, BOX_HOST, BOX_PORT, GPU_BOX_HOST, GPU_BOX_PORT (see .env.example)
# Live view of an RL stage log on the A100 (Terminal panel tab "Growth training"):  bash scripts/tail_rl_log.sh [preflight|A|B1|mixed|final]
STAGE="${1:-preflight}"
O="-i ${SSH_KEY} -o IdentitiesOnly=yes -o LogLevel=ERROR -o ServerAliveInterval=30 -o ConnectTimeout=25"
exec ssh $O -p ${GPU_BOX_PORT} root@${GPU_BOX_HOST} "tail -n 80 -F /workspace/clean_rerun/logs/rl_${STAGE}.log 2>&1 | grep --line-buffered -E 'rl step|sample|eval|bench|OOM|Error|Traceback|guard|ladder|pushed|vLLM'"
