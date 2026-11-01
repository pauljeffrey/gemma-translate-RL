#!/bin/bash
# tmux session qe_cpu: CPU AfriCOMET-QE over the priority queue, two processes (part 0/2 on cores 4-5, part 1/2 on cores 6-7; measured faster than one 4-thread process).
# Resumable: re-running skips every chunk file and every already scored key (CPU or GPU shards). NEVER touches the GPU (CUDA_VISIBLE_DEVICES="").
# usage: qe_cpu_run.sh <part 0|1>
P=${1:-0}
cd /workspace/work
source /workspace/tr/env.sh                                   # HF_TOKEN only (never printed)
export CUDA_VISIBLE_DEVICES="" HF_HOME=/workspace/work/hf_comet OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/work/chain
CORES=$([ "$P" = "0" ] && echo 4-5 || echo 6-7)
exec taskset -c $CORES nice -n 19 ionice -c3 venv_comet/bin/python chain/qe_score.py --src qe_cpu/queue --out qe_cpu/scores_cpu --device cpu --threads 2 --batch 16 --workers 1 --chunk 500 --gpus 0 \
  --part $P/2 --manifest qe_cpu/queue/queue_manifest.json --progress qe_cpu/progress_$P.json --skip-scored qe_cpu/scores_cpu qe_cpu/scores_gpu --max-load 20 >> qe_cpu/qe_cpu_$P.log 2>&1
