#!/bin/bash
# watcher: as soon as the E-02 AfriCOMET pass is done, stop the queue's sequential CPU tail and run it niced in parallel so that the GPU moves on to E-04 at once
O=/workspace/e2405/e02
while [ ! -f $O/COMET_DONE ]; do sleep 10; done
for p in $(pgrep -f "^bash /workspace/e2405/e02_run.sh"); do kill $p; done; pkill -f "e02_mknpz.py" ; sleep 3
nohup bash /workspace/e2405/e02_post.sh > $O/post.out 2>&1 &
echo "$(date -u +%FT%TZ) [e02] handover: CPU tail runs niced in parallel (e02_post.sh)" >> $O/log.txt
