#!/bin/bash
# LC1 sequential queue: ONE GPU process at a time, nice 19, GPU footprint <= 8 GB; every step is skipped when its outputs exist and retried on failure (OOM from other jobs).
cd /workspace/jobs/lc1; source env_lc.sh
export LD_LIBRARY_PATH=/workspace/jobs/r14/pylibs/nvidia/cublas/lib:/workspace/jobs/r14/pylibs/nvidia/cudnn/lib:/workspace/jobs/r14/pylibs/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
retry() { for t in 1 2 3 4 5 6; do "$@" && return 0; echo "retry $t: $*"; sleep 120; done; return 1; }
log() { echo "$(date +%H:%M:%S) $*" >> logs/queue.log; }
until [ -f logs/OURS_DONE_sft_rl_a2w ]; do sleep 30; done
log "ours 1 done"
retry ./run_ours.sh live; log "live done"
for sz in 3b3 600m; do retry nice -n 19 $PYG lc_nllb.py $sz sent whole >> logs/nllb_$sz.log 2>&1; log "nllb $sz done"; done
TP="hau,yor,swh,amh,ibo,mlg,zul,xho,kin,som,wol,lug"
nice -n 19 $PYG dl.py qvac/TranslatePsy-AfriSLM-0.8B tpsy08 >> logs/dl_tpsy08.log 2>&1
retry nice -n 19 $PYV lc_chat.py tpsy08 /dev/shm/lc1_models/tpsy08 translatepsy $TP --mm --gpu-util 0.07 >> logs/chat_tpsy08.log 2>&1; log "tpsy08 done"
rm -rf /dev/shm/lc1_models/tpsy08
nice -n 19 $PYG dl.py Qwen/Qwen3.5-0.8B qwen08 >> logs/dl_qwen08.log 2>&1
retry nice -n 19 $PYV lc_chat.py qwen08 /dev/shm/lc1_models/qwen08 translatepsy all --mm --skip-bins b4096 --gpu-util 0.07 >> logs/chat_qwen08.log 2>&1; log "qwen08 done"
rm -rf /dev/shm/lc1_models/qwen08
nice -n 19 $PYG dl.py google/gemma-3-1b-it gemma1b >> logs/dl_gemma1b.log 2>&1
if [ -f /dev/shm/lc1_models/gemma1b/model.safetensors ]; then
  retry nice -n 19 $PYV lc_chat.py gemma1b /dev/shm/lc1_models/gemma1b gemma all --skip-bins b4096 --gpu-util 0.07 >> logs/chat_gemma1b.log 2>&1; log "gemma1b done"
fi
rm -rf /dev/shm/lc1_models/gemma1b
nice -n 19 $PYG dl.py qvac/TranslatePsy-AfriSLM-2B tpsy2b >> logs/dl_tpsy2b.log 2>&1
retry nice -n 19 $PYV lc_chat.py tpsy2b /dev/shm/lc1_models/tpsy2b translatepsy $TP --mm --skip-bins b4096 --gpu-util 0.085 --max-seqs 32 >> logs/chat_tpsy2b.log 2>&1; log "tpsy2b done"
rm -rf /dev/shm/lc1_models/tpsy2b
nice -n 19 $PYG dl.py Qwen/Qwen3.5-2B qwen2b >> logs/dl_qwen2b.log 2>&1
retry nice -n 19 $PYV lc_chat.py qwen2b /dev/shm/lc1_models/qwen2b translatepsy all --mm --skip-bins b4096 --gpu-util 0.085 --max-seqs 32 >> logs/chat_qwen2b.log 2>&1; log "qwen2b done"
rm -rf /dev/shm/lc1_models/qwen2b
log "QUEUE DONE"; touch logs/QUEUE_DONE
