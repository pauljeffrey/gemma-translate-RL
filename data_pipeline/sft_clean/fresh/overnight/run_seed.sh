#!/bin/bash
# usage: run_seed.sh SEED  — E-04: one stage-A-recipe seed run (100 steps) with the launcher, then the harness evaluation of its final/ on the 20 languages
# disk rules (coordinator 2026-11-04): root disk is nearly full -> caches, temp and any download on /dev/shm; skip this step if < 1 GB free
mkdir -p /dev/shm/tmp /dev/shm/xdg_cache /dev/shm/vllm_cache_ov
export TMPDIR=/dev/shm/tmp XDG_CACHE_HOME=/dev/shm/xdg_cache
[ "$(df --output=avail -k / | tail -1)" -lt 1048576 ] && { echo "LOW DISK (< 1 GB free on /): step skipped"; exit 3; }
SEED=$1; source /workspace/science_sft/overnight/common.sh
R=/workspace/clean_rerun; CFG=$R/tree/gemma_mt/configs/clean_rerun_385m_seed$SEED; RS=$R/runs_385m_seed$SEED
mkdir -p $CFG $RS/A $R/logs_seed$SEED /dev/shm/rl_seed${SEED}_ckpt
python3 $O/make_seed_cfg.py $SEED > $CFG/rl_A_b64.yaml || exit 2
for d in best final last hub_stage hub_stage_latest; do mkdir -p /dev/shm/rl_seed${SEED}_ckpt/$d; [ -e $RS/A/$d ] || ln -s /dev/shm/rl_seed${SEED}_ckpt/$d $RS/A/$d; done   # hub_stage* too: a real 1.5 GB dir on the root disk caused ENOSPC in the long RL (2026-11-04)
cd $R/tree/gemma_mt; source $R/rl_env.sh
export RUNS=$RS LOGDIR=$R/logs_seed$SEED CFGDIR=$CFG STAGE_PLAN=b256 FORCE=1 OVERRIDE_PIN=1 LADDER_START=1 MIN_DISK_GIB=2 MIN_FREE_GIB=60 HUB_NO_MODEL_CARD=1 A_FROM_PREFLIGHT=0 STEPS_A=100 HUB_PREFIX=
bash scripts/launch_clean_rerun_385m.sh run A
