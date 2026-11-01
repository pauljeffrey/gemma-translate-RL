#!/bin/bash
# SUPERSEDED 2026-10-29 by r4_run_all.py (box 2 is lost; IbomMT removed). Kept for the record only, do not run.
# r4: GPU part of the clean reward calibration. NO trained model is needed for steps 1-3 (they score HUMAN references); step 4+ need the new policy.
# Any CUDA GPU >= 8 GB, venv with torch + unbabel-comet (e.g. /workspace/work/venv_comet on a GPU box). Resumable: every (language, set) is a file in qe_work/.
# Expected GPU time (ESTIMATED from 290-350 pairs/s fp16 on an RTX 5070): AfriCOMET B 22k pairs ~1.3 min, SSA-COMET-MTL A+B 44k pairs ~2.5 min, model loads ~2 min  => ~7 GPU-min in total.
# The CPU job (tmux calib_cpu / calib_cpu_q on box 2) does the same work at ~1 pair/s; whatever it has finished is skipped here.
set -e
W=/workspace/calib_r4                 # same layout as on box 2: refs/cal.jsonl (benchmark text, box only), qe_work/, clean/, code/ (repo gemma_mt/), work/ (r4_*.py)
cd $W/work
export HF_HOME=$W/hf_comet TOKENIZERS_PARALLELISM=false
PY=${PY:-/workspace/work/venv_comet/bin/python}
SPLIT=heldout_split_cal_sel_v2.json
# 1. AfriCOMET-QE-STL-1.1: human + mismatch (phase A) and half + append (phase B)
$PY r4_qe_score.py --refs $W/refs/cal.jsonl --split $SPLIT --work $W/qe_work --model africomet --phase ALL --batch 128 --device cuda --code $W/code
# 2. SSA-COMET-MTL (reference-free): same sets
$PY r4_qe_score.py --refs $W/refs/cal.jsonl --split $SPLIT --work $W/qe_work --model ssa_mtl --phase ALL --batch 64 --device cuda --code $W/code
# 3. assemble (numbers only), anchors v1/v2 with the final content modes, push
PYTHONPATH=$W/code $PY r4_assemble.py --code $W/code --old shaping_flores_dev_v8.json --work $W --split $SPLIT --out $W/clean/shaping_clean_v1.json --status $W/clean/status.json
# (r4_assemble.py also writes the qe2 / ens blocks (delta 1.0 of D-55) wherever SSA scores exist)
# 4. NEEDS THE NEW POLICY (not here): validate_consensus.py on efi/ibb hold-out (D-43 weight), NLLB-signal rule on H_sel (scripts/r14/*), tail_gap (scripts/longdoc_v2/reward_run.py + calibrate_tailgap.py)
