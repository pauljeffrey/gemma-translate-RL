#!/bin/bash
# SUPERSEDED 2026-10-29 by r4_run_all.py (box 2 is lost; IbomMT removed). Kept for the record only, do not run.
# r4: runs in tmux session calib_cpu_fin on box 2 (2 cores, nice 19). Every 30 min: if new QE score files exist -> r4_assemble.py + push the numbers to the Hub (calibration/).
# When the 84 AfriCOMET phase-A files are complete (content modes final) -> anchors v1 + v2 are refit once more with the final modes, then pushed. Ends when all 169+169 score files exist.
cd /workspace/calib_r4/work
export HF_TOKEN=$(cat ~/.cache/huggingface/token) PYTHONPATH=/workspace/calib_r4/code
PY=/workspace/calib_r4/venv_lid/bin/python
LID=/workspace/calib_r4/hf_lid/hub/models--cis-lmu--glotlid/snapshots/49990ca512e170d705827d69c1064ce262434be6/model_v3.bin
last=-1
while true; do
  na=$(ls /workspace/calib_r4/qe_work/africomet 2>/dev/null | wc -l); ns=$(ls /workspace/calib_r4/qe_work/ssa_mtl 2>/dev/null | wc -l); tot=$((na+ns))
  if [ "$tot" != "$last" ]; then
    taskset -c 8-9 nice -n 19 $PY r4_assemble.py --code /workspace/calib_r4/code --old shaping_flores_dev_v8.json --work /workspace/calib_r4 --split heldout_split_cal_sel_v2.json --out ../clean/shaping_clean_v1.json --status ../clean/status.json >> ../logs/finisher.log 2>&1
    if [ "$na" -ge 85 ] && [ ! -f ../clean/.anchors_final ]; then
      taskset -c 8-9 nice -n 19 $PY r4_anchors_v1.py --code /workspace/calib_r4/code --cal /workspace/calib_r4/refs/cal.jsonl --split heldout_split_cal_sel_v2.json --kept /workspace/work/out/kept --ibom-halves /workspace/calib_r4/clean/ibom_halves --afs s3/afriscience_clean_train/afriscience_clean_train.parquet --shaping ../clean/shaping_clean_v1.json --lid-model $LID --out ../clean/anchors_clean_v1.json --dump-human ../clean/human_units_clean.jsonl >> ../logs/finisher.log 2>&1
      taskset -c 8-9 nice -n 19 $PY /workspace/calib_r4/code/scripts/build_anchors_v2_calibration.py --v1 ../clean/anchors_clean_v1.json --human ../clean/human_units_clean.jsonl --funcwords ../clean/funcwords_clean_v1.json --bigrams ../clean/bigrams_clean_v1.npz --out ../clean/anchors_clean_v2.json >> ../logs/finisher.log 2>&1 && touch ../clean/.anchors_final
    fi
    taskset -c 8-9 nice -n 19 $PY r4_push_calibration.py --work /workspace/calib_r4 --split heldout_split_cal_sel_v2.json >> ../logs/finisher.log 2>&1
    last=$tot
  fi
  [ "$na" -ge 169 ] && [ "$ns" -ge 169 ] && break
  sleep 1800
done
