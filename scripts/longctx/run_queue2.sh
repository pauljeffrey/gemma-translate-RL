#!/bin/bash
# LC1 queue part 2 (compute-reduced): chat models on 6 core/human languages + the AfriScience languages they support
cd /workspace/jobs/lc1; source env_lc.sh
retry() { for t in 1 2 3 4 5 6; do "$@" && return 0; echo "retry $t: $*"; sleep 120; done; return 1; }
log() { echo "$(date +%H:%M:%S) $*" >> logs/queue.log; }
TP="hau,yor,swh,amh,zul,ibo,lug"
ZS="hau,yor,swh,amh,zul,ibo,lug,nso"
[ -f hyp/tpsy08__whole.jsonl ] || { retry nice -n 19 $PYV lc_chat.py tpsy08 /dev/shm/lc1_models/tpsy08 translatepsy $TP --mm --max-seqs 256 --gpu-util 0.07 >> logs/chat_tpsy08.log 2>&1; log "tpsy08 done"; }
rm -rf /dev/shm/lc1_models/tpsy08
nice -n 19 $PYG dl.py Qwen/Qwen3.5-0.8B qwen08 >> logs/dl_qwen08.log 2>&1
retry nice -n 19 $PYV lc_chat.py qwen08 /dev/shm/lc1_models/qwen08 translatepsy $ZS --mm --skip-bins b4096 --max-seqs 256 --gpu-util 0.07 >> logs/chat_qwen08.log 2>&1; log "qwen08 done"
rm -rf /dev/shm/lc1_models/qwen08
nice -n 19 $PYG dl.py google/gemma-3-1b-it gemma1b >> logs/dl_gemma1b.log 2>&1
if [ -f /dev/shm/lc1_models/gemma1b/model.safetensors ]; then
  retry nice -n 19 $PYV lc_chat.py gemma1b /dev/shm/lc1_models/gemma1b gemma $ZS --skip-bins b4096 --max-seqs 256 --gpu-util 0.07 >> logs/chat_gemma1b.log 2>&1; log "gemma1b done"
fi
rm -rf /dev/shm/lc1_models/gemma1b
nice -n 19 $PYG dl.py qvac/TranslatePsy-AfriSLM-2B tpsy2b >> logs/dl_tpsy2b.log 2>&1
retry nice -n 19 $PYV lc_chat.py tpsy2b /dev/shm/lc1_models/tpsy2b translatepsy $TP --mm --skip-bins b4096 --max-seqs 128 --gpu-util 0.085 >> logs/chat_tpsy2b.log 2>&1; log "tpsy2b done"
rm -rf /dev/shm/lc1_models/tpsy2b
nice -n 19 $PYG dl.py Qwen/Qwen3.5-2B qwen2b >> logs/dl_qwen2b.log 2>&1
retry nice -n 19 $PYV lc_chat.py qwen2b /dev/shm/lc1_models/qwen2b translatepsy $ZS --mm --skip-bins b4096 --max-seqs 128 --gpu-util 0.085 >> logs/chat_qwen2b.log 2>&1; log "qwen2b done"
rm -rf /dev/shm/lc1_models/qwen2b
log "QUEUE DONE"; touch logs/QUEUE_DONE
