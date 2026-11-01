#!/bin/bash
# Runs on the DATA box after post_all.sh: cut stage B into K shards (make_shards.py), checksum each, push each to the bucket and publish a DONE marker.
set -uo pipefail
K=${K:-6}
ROOT=/workspace/work; cd $ROOT; source env.sh
export HF_HOME=$ROOT/hf_home PYTHONPATH=$ROOT/code/sft_clean/chain:$ROOT/code GEMMA_MT_PKG=$ROOT/code
PD=$ROOT/venv_data/bin/python; W=$ROOT/work; P=${S3_PREFIX}; BS="$PD code/sft_clean/fresh/bucket_sync.py"
fail() { echo "$(date -u +%FT%TZ) [SHARDS_FAILED] $1" >> $ROOT/ALERT.txt; echo FAIL $1; exit 1; }
until grep -q DELTA_BUNDLE_READY ALERT.txt 2>/dev/null; do sleep 30; done
rm -rf $W/shards
$PD code/sft_clean/fresh/make_shards.py --packed $W/packed_g --stage-a $W/sched_g/A --k $K --out $W/shards > logs/make_shards.log 2>&1 || fail make_shards
tail -n 8 logs/make_shards.log
for k in $(seq 1 $K); do
  ( cd $W/shards/s$k && find . -type f ! -name READY | sort | xargs sha256sum > ../s$k.sha256 )
  $BS put $P $W/shards/s$k $W/shards/s$k.sha256 >> logs/bucket_put_shards.log 2>&1 || fail "push shard $k"
  $BS mark $P/s$k.DONE && echo "$(date -u +%FT%TZ) [SHARD_PUSHED] $k" >> ALERT.txt
done
echo "$(date -u +%FT%TZ) [SHARDS_ALL_PUSHED] $K shards in bucket $P" >> ALERT.txt
