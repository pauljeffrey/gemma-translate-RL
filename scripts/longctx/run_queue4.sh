#!/bin/bash
# LC1 queue part 4: after the qwen08 job (started by queue 3): TranslatePsy-2B first (more informative), then Gemma-3-1B-it if time allows
cd /workspace/jobs/lc1; source env_lc.sh
retry() { for t in 1 2 3 4 5 6; do "$@" && return 0; echo "retry $t: $*"; sleep 120; done; return 1; }
log() { echo "$(date +%H:%M:%S) $*" >> logs/queue.log; }
until [ -f hyp/qwen08__whole.jsonl ]; do sleep 30; done
log "qwen08 done"
rm -rf /dev/shm/lc1_models/qwen08
nice -n 19 $PYG dl.py qvac/TranslatePsy-AfriSLM-2B tpsy2b >> logs/dl_tpsy2b.log 2>&1
retry nice -n 19 $PYV lc_chat.py tpsy2b /dev/shm/lc1_models/tpsy2b translatepsy hau,yor,swh,amh,zul,ibo,lug --mm --skip-bins b4096,b256,sm,st512 --max-seqs 128 --gpu-util 0.075 >> logs/chat_tpsy2b.log 2>&1; log "tpsy2b done"
rm -rf /dev/shm/lc1_models/tpsy2b
nice -n 19 $PYG dl.py google/gemma-3-1b-it gemma1b >> logs/dl_gemma1b.log 2>&1
if [ -f /dev/shm/lc1_models/gemma1b/model.safetensors ]; then
  retry nice -n 19 $PYV lc_chat.py gemma1b /dev/shm/lc1_models/gemma1b gemma hau,yor,swh,amh,zul --skip-bins b4096,b256,sm,st512 --max-seqs 256 --gpu-util 0.06 >> logs/chat_gemma1b.log 2>&1; log "gemma1b done"
fi
rm -rf /dev/shm/lc1_models/gemma1b
log "QUEUE DONE"; touch logs/QUEUE_DONE
