#!/bin/bash
# Runs on the A100: pulls shard k from the bucket when it is published (DONE marker) and there is room: shard 1 at once, shard 2 after stage A (packed_g0 is deleted then), shard k >= 3
# once shard k-2 has finished training (at most the running shard and the next one are on disk). Writes run/shards/s<k>/READY_LOCAL after the download. Restartable.
set -uo pipefail
K=${K:-6}
ROOT=/workspace/work; cd $ROOT; P=${S3_PREFIX}; BS="python3 code/sft_clean/fresh/bucket_sync.py"
mkdir -p run/shards
for k in $(seq 1 $K); do
  [ -f run/runs/g1b_s$k/summary.json ] && continue
  [ -f run/shards/s$k/READY_LOCAL ] && continue
  if [ $k -ge 2 ]; then until [ -f run/runs/g1a/summary.json ]; do sleep 60; done; fi
  if [ $k -ge 3 ]; then until [ -f run/runs/g1b_s$((k - 2))/summary.json ]; do sleep 60; done; fi
  until $BS has $P/s$k.DONE 2>/dev/null; do sleep 30; done
  rm -rf run/shards/s$k
  $BS get $P run/shards s$k s$k.sha256 > logs/bucket_get_s$k.log 2>&1 || { echo "$(date -u +%FT%TZ) [SHARD_PULL_FAILED] $k" >> logs/ALERT_a100.txt; exit 1; }
  if [ -d run/shards/s$k.sha256 ]; then mv run/shards/s$k.sha256/s$k.sha256 run/shards/s$k.sha256.tmp && rmdir run/shards/s$k.sha256 && mv run/shards/s$k.sha256.tmp run/shards/s$k.sha256; fi   # bucket_sync put of a FILE stores it as <name>/<name>
  ( cd run/shards/s$k && sha256sum -c ../s$k.sha256 --quiet ) || { echo "$(date -u +%FT%TZ) [SHARD_PULL_FAILED] checksum $k" >> logs/ALERT_a100.txt; exit 1; }
  touch run/shards/s$k/READY_LOCAL
  echo "$(date -u +%FT%TZ) [SHARD_ARRIVED] $k" >> logs/ALERT_a100.txt
done
