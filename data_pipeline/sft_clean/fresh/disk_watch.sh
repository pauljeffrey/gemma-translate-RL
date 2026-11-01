#!/bin/bash
# background disk watcher: df every 60 s into logs/disk.log; marks state/DISK_LOW when free space < MIN_FREE_GB
ROOT=${ROOT:-/workspace/work}; MIN=${MIN_FREE_GB:-12}
while true; do
  free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
  echo "$(date -u +%FT%TZ) free=${free}G root_use=$(du -sh "$ROOT" 2>/dev/null | cut -f1)" >> "$ROOT/logs/disk.log"
  if [ "$free" -lt "$MIN" ]; then touch "$ROOT/state/DISK_LOW"; else rm -f "$ROOT/state/DISK_LOW"; fi
  sleep 60
done
