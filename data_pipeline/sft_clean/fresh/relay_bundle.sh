#!/bin/bash
# Runs on the LAPTOP (the vast boxes cannot reach each other and share no bucket credentials): waits for MARKER in ALERT.txt on the data box, then streams
# the listed items of work/<BUNDLE_DIR> (symlinks followed, tar through this machine) into the A100's run directory. Restartable (starts over). Keep the laptop awake and online.
#   relay_bundle.sh G0_READY bundle_g0 packed_g0 sched_g packed_g0.sha256 sched_g_A.sha256
#   relay_bundle.sh DELTA_BUNDLE_READY bundle_delta packed_delta_tag sched_g packed_delta_tag.sha256 sched_g_B.sha256 packed_g_arrays.sha256
MARKER=$1; BDIR=$2; shift 2; ITEMS="$*"
K=$SSH_KEY; O="-i $K -o IdentitiesOnly=yes -o LogLevel=ERROR -o ConnectTimeout=30 -o ServerAliveInterval=30"
DATA="-p $BOX_PORT root@$BOX_HOST"; GPU="-p $BOX_PORT root@$BOX_HOST"
until ssh $O $DATA "grep -q $MARKER /workspace/work/ALERT.txt" 2>/dev/null; do
  ssh $O $DATA "grep -q 'POST_ALL_FAILED\|G0_FAILED' /workspace/work/ALERT.txt && ! grep -q $MARKER /workspace/work/ALERT.txt && tail -n 1 /workspace/work/ALERT.txt | grep -q FAILED" 2>/dev/null && { echo "$(date +%T) data box reported a failure, stopping"; exit 1; }
  sleep 120
done
echo "$(date +%T) $MARKER seen, streaming $ITEMS"
for try in 1 2 3; do
  ssh $O $DATA "cd /workspace/work/work/$BDIR && tar chf - $ITEMS" | ssh $O $GPU "mkdir -p /workspace/work/run && cd /workspace/work/run && tar xf -" && { echo "$(date +%T) transfer done"; ssh $O $GPU "echo $BDIR > /workspace/work/run/.arrived_$BDIR"; exit 0; }
  echo "$(date +%T) transfer try $try failed"; sleep 30
done
echo "$(date +%T) transfer FAILED"; exit 1
