#!/bin/bash
# LC1 queue part 5 (time-boxed): TranslatePsy-2B on a reduced bin/language set, then Gemma-3-1B-it sentence mode only
cd /workspace/jobs/lc1; source env_lc.sh
retry() { for t in 1 2 3 4 5 6; do "$@" && return 0; echo "retry $t: $*"; sleep 120; done; return 1; }
log() { echo "$(date +%H:%M:%S) $*" >> logs/queue.log; }
rm -rf /dev/shm/lc1_models/qwen08
nice -n 19 $PYG dl.py qvac/TranslatePsy-AfriSLM-2B tpsy2b >> logs/dl_tpsy2b.log 2>&1
retry nice -n 19 $PYV lc_chat.py tpsy2b /dev/shm/lc1_models/tpsy2b translatepsy hau,yor,swh,amh,zul --mm --skip-bins b256,b512,b1024,b4096,sm,st512,sci_test --max-seqs 128 --gpu-util 0.075 >> logs/chat_tpsy2b.log 2>&1; log "tpsy2b done"
rm -rf /dev/shm/lc1_models/tpsy2b
nice -n 19 $PYG dl.py google/gemma-3-1b-it gemma1b >> logs/dl_gemma1b.log 2>&1
if [ -f /dev/shm/lc1_models/gemma1b/model.safetensors ]; then
  retry nice -n 19 $PYV lc_chat.py gemma1b /dev/shm/lc1_models/gemma1b gemma hau,yor,swh,amh,zul --modes sent --skip-bins b256,b512,b1024,b4096,sm,st512,sci_test --max-seqs 256 --gpu-util 0.06 >> logs/chat_gemma1b.log 2>&1; log "gemma1b done"
fi
rm -rf /dev/shm/lc1_models/gemma1b
log "QUEUE DONE"; touch logs/QUEUE_DONE
