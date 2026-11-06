#!/bin/bash
[ -f .env ] && { set -a; . ./.env; set +a; }   # SSH_KEY, BOX_HOST, BOX_PORT, GPU_BOX_HOST, GPU_BOX_PORT (see .env.example)
# usage: bx.sh 'remote command'
OPTS="-p ${GPU_BOX_PORT} -i ${SSH_KEY} -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30"
ssh $OPTS root@${GPU_BOX_HOST} "$1" 2>&1 | grep -v "^Welcome to vast\|^Have fun\|^AI agents:"
