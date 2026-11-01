#!/bin/bash
# Live board on the laptop: refreshes the instance status every INTERVAL seconds. Ctrl-C to leave (nothing on the instance is touched).
#   bash dash.sh            # defaults to the current instance
HOST=${HOST:-root@$BOX_HOST}; PORT=${PORT:-$BOX_PORT}; KEY=${KEY:-$SSH_KEY}; INTERVAL=${INTERVAL:-20}
DIR=$(cd "$(dirname "$0")" && pwd)
B64=$(base64 < "$DIR/dash_remote.sh" | tr -d '\n')
while true; do
  OUT=$(ssh -i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=15 -o LogLevel=ERROR -p "$PORT" "$HOST" "echo '$B64' | base64 -d | bash" 2>&1 | grep -v "Welcome to vast\|Have fun\|AI agents")
  clear; printf '%s\n' "$OUT"; echo; echo "(refresh every ${INTERVAL}s, Ctrl-C to leave)"
  sleep "$INTERVAL"
done
