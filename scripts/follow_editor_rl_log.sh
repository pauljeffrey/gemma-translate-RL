#!/bin/bash
[ -f .env ] && { set -a; . ./.env; set +a; }   # SSH_KEY, BOX_HOST, BOX_PORT, GPU_BOX_HOST, GPU_BOX_PORT (see .env.example)
# Waits on the post-editor box (3090 Ti) until the editor-RL training log exists, then streams it live (tail -F) with the per-step lines and the guard / checkpoint messages.
OPTS="-i ${SSH_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o ServerAliveInterval=30 -o LogLevel=ERROR -p ${BOX_PORT}"
LOG="${LOG:-/workspace/logs/train_pe_rl.log}"
while true; do
  echo "[$(date +%H:%M:%S)] waiting for $LOG ..."
  ssh $OPTS root@${BOX_HOST} "while [ ! -s $LOG ]; do sleep 10; done; echo STARTED \$(date -u +%H:%M:%S) UTC; tail -n 40 -F $LOG" 2>&1 | grep -v -E "^(Welcome to vast|Have fun|AI agents:)"
  echo "[$(date +%H:%M:%S)] connection ended, retrying in 15 s"
  sleep 15
done
