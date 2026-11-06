#!/bin/bash
[ -f .env ] && { set -a; . ./.env; set +a; }   # SSH_KEY, BOX_HOST, BOX_PORT, GPU_BOX_HOST, GPU_BOX_PORT (see .env.example)
# Live view of the post-editor box (3090 Ti): training and evaluation logs of every editor run.
# Every INTERVAL seconds prints the tail of each train_*/eval_*/comet_* log touched in the last 2 h,
# with progress-bar carriage returns split and vLLM engine noise removed.
INTERVAL="${INTERVAL:-30}"
OPTS="-i ${SSH_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o ServerAliveInterval=30 -o LogLevel=ERROR -p ${BOX_PORT}"
REMOTE='cd /workspace/logs; echo "== $(date -u +%H:%M:%S) UTC  GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)";
for f in $(find . -maxdepth 1 \( -name "train_*.log" -o -name "eval_*.log" -o -name "comet_*.log" \) -mmin -120 | sort); do
  echo "--- $f";
  tr "\r" "\n" < $f | grep -v -E "EngineCore|INFO|WARNING|Rendering prompts|^$" | tail -n 6 | cut -c1-200;
done'
while true; do
  ssh $OPTS root@${BOX_HOST} "$REMOTE" 2>&1 | grep -v -E "^(Welcome to vast|Have fun|AI agents:)"
  echo
  sleep "$INTERVAL"
done
