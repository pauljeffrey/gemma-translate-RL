#!/bin/bash
# E2 sequential queue, GPU footprint <= 6 GB, nice 19 (coordinator rule 2026-10-23). SFT/RL predictions: read-only copies of e1 outputs (same pipeline, greedy vLLM, 768 tokens).
cd /workspace/jobs/e2
source /workspace/sft_build/env.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv_gemma/bin/python
retry() { for t in 1 2 3 4 5 6 7 8; do "$@" && return 0; sleep 90; done; return 1; }
[ -f scores/live250.pkl ] || nice -n 19 $PY e2_score.py live250 pipe out_live250/predictions.jsonl > score_live250.log 2>&1
[ -f scores/sft.pkl ] || nice -n 19 $PY e2_score.py sft pipe /workspace/jobs/e1/out_sft/predictions.jsonl > score_sft.log 2>&1
retry nice -n 19 $PY e2_comet.py live250 sft > comet_ours.log 2>&1
retry nice -n 19 $PY e2_ssa.py nllb600m nllb3b3 tpsy08 a2w live250 sft > ssa_all.log 2>&1
retry nice -n 19 $PY e2_comet.py tpsy2b > comet_tpsy2b.log 2>&1
retry nice -n 19 $PY e2_ssa.py tpsy2b > ssa_tpsy2b.log 2>&1
echo phase1 > CHAIN_PHASE1_DONE
for i in $(seq 1 1500); do [ -f /workspace/jobs/e1/out_rl/EVAL_DONE ] && break; sleep 30; done
[ -f scores/rl.pkl ] || nice -n 19 $PY e2_score.py rl pipe /workspace/jobs/e1/out_rl/predictions.jsonl > score_rl.log 2>&1
retry nice -n 19 $PY e2_comet.py rl > comet_rl.log 2>&1
retry nice -n 19 $PY e2_ssa.py rl > ssa_rl.log 2>&1
echo done > CHAIN_DONE
