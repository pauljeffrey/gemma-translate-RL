#!/bin/bash
# How much faster do two SFT jobs on one GPU run than one? Measures COMBINED steps/s (and tokens/s) of 1 job vs 2 concurrent jobs on the first STEPS steps of A0's schedule. ~15-25 minutes.
#   cd $ROOT/run && PYTHONPATH=$CODE:$CH PY=$ROOT/venv_train/bin/python bash $FR/sft_throughput_probe.sh [CONFIG] [STEPS]
# Needs packed/ and sched/A0 (run_chain.sh steps 60 and 70). Uses configs/clean_rerun_sft/a0.yaml (270M; pass a 1B config to probe that). No evaluation, no checkpoints (eval_every/save_every huge).
# Decision rule printed at the end: run two jobs at once if the combined steps/s is >= 1.15 x one job and the measurement windows overlapped (>= 0.8).
set -uo pipefail
CFG=${1:-configs/clean_rerun_sft/a0.yaml}; STEPS=${2:-80}; PY=${PY:-python}; CH=${CH:-$(dirname "$0")/../chain}; FR=$(cd "$(dirname "$0")" && pwd); P=${PROBE_DIR:-runs/_probe}
mkdir -p "$P" logs; rm -rf "$P"/1job "$P"/2job_a "$P"/2job_b
"$PY" "$CH/sampler.py" head --schedule sched/A0 --out sched/_probe_head --steps "$STEPS" --name probe_head || { echo "sampler.py head failed"; exit 1; }
COMMON="schedule_file=sched/_probe_head/schedule.npy,max_steps=0,eval_every=1000000,save_every=1000000,quick_eval_every=0,log_every=5,early_stopping_patience=0,run_name=probe"
echo "== phase 1: ONE job ($STEPS steps) =="
t0=$(date +%s); bash "$FR/sft_concurrent.sh" "p1=$CFG,out_dir=$P/1job,$COMMON"; t1=$(date +%s); echo "one job: wall $((t1-t0)) s"
echo "== phase 2: TWO concurrent jobs (same schedule, separate out_dirs; ${STAGGER_S:-120}s stagger is shortened to 20 s: the second job's warm-up must overlap the first job's run) =="
t0=$(date +%s); STAGGER_S=${STAGGER_S_PROBE:-20} bash "$FR/sft_concurrent.sh" "pa=$CFG,out_dir=$P/2job_a,$COMMON" "pb=$CFG,out_dir=$P/2job_b,$COMMON"; t1=$(date +%s); echo "two jobs: wall $((t1-t0)) s"
echo "== result (steady state: the first 10 logged steps of every run are skipped; peak VRAM: see nvidia-smi log in logs/) =="
"$PY" "$FR/sft_probe_summary.py" --group "1job=$P/1job" --group "2jobs=$P/2job_a,$P/2job_b"
echo "Also try USE_MPS=1 for the two-job phase (kernel overlap across processes) and 3 jobs by hand with sft_concurrent.sh; keep the best by the same table."
