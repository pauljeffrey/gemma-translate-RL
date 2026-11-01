#!/bin/bash
# SUPERSEDED 2026-10-29 by r4_run_all.py (box 2 is lost; IbomMT removed). Kept for the record only, do not run.
# r4: the rest of the QE scoring on box 2 CPU after phase A of AfriCOMET (run inside tmux session calib_cpu_q; 2 cores, nice 19). Resumable (every (language,set) is a file).
# Order: AfriCOMET phase B (half, append: reliability_same_source) -> SSA-COMET-MTL phase A (qe2 anchors) -> SSA phase B. Re-run r4_assemble.py after each stage.
cd /workspace/calib_r4/work
export CUDA_VISIBLE_DEVICES= HF_HOME=/workspace/calib_r4/hf_comet OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
PY=/workspace/work/venv_comet/bin/python
while pgrep -f "r4_qe_score.py --refs /workspace/calib_r4/refs/cal.jsonl --split heldout_split_cal_sel_v2.json --work /workspace/calib_r4/qe_work --model africomet --phase A" >/dev/null; do sleep 60; done
for spec in "africomet B" "ssa_mtl A" "ssa_mtl B"; do
  set -- $spec
  taskset -c 8-9 nice -n 19 ionice -c3 $PY r4_qe_score.py --refs /workspace/calib_r4/refs/cal.jsonl --split heldout_split_cal_sel_v2.json --work /workspace/calib_r4/qe_work --model $1 --phase $2 --batch 16 --code /workspace/calib_r4/code --threads 2 2>&1 | tee -a /workspace/calib_r4/logs/qe_$1_$2.log
done
