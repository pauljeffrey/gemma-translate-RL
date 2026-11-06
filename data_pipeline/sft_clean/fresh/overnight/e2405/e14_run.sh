#!/bin/bash
# E-14: the executed science RL config (runs_385m_sci/final/config.json, same seed 1234, same pool, same reward incl. the science terms and the NLLB pseudo-reference reward signal of 5 languages),
# started from MIXED BEST (Hub main; = science SFT alpha 0) instead of the science-SFT interpolation alpha 0.5; 160 steps, lr schedule as executed (flat to step 150, WINDDOWN 10 steps to step 160), early stops off.
R=/workspace/clean_rerun; E2=/workspace/e2405; RUNS=$R/runs_e14; RS=$RUNS/final; LOGDIR=$R/logs_e14; CFG=$R/tree/gemma_mt/configs/e14; CK=/dev/shm/e14; L=$E2/log_e14.txt
say() { echo "$(date -u +%FT%TZ) [e14] $*" | tee -a $L; }
mkdir -p /dev/shm/tmp /dev/shm/xdg_cache; export TMPDIR=/dev/shm/tmp XDG_CACHE_HOME=/dev/shm/xdg_cache
[ "$(df --output=avail -k / | tail -1)" -lt 1048576 ] && { say "LOW DISK (< 1 GB free on /): not started"; exit 3; }
[ "$(df --output=avail -k /dev/shm | tail -1)" -lt 12582912 ] && { say "LOW /dev/shm (< 12 GB free): not started"; exit 3; }
mkdir -p $RS $LOGDIR $CFG $CK; PY=/workspace/venv_gemma/bin/python
for d in best final last hub_stage hub_stage_latest; do mkdir -p $CK/$d; [ -e $RS/$d ] || ln -s $CK/$d $RS/$d; done
$PY $E2/e04_cfg.py sci 1234 $RS > $CFG/rl_F_final.yaml; cp $CFG/rl_F_final.yaml $E2/cfg_e14.yaml
bash $E2/codestate.sh $RS/code_state.txt
cd $R/tree/gemma_mt; source $R/rl_env.sh
export RUNS LOGDIR CFGDIR=$CFG STAGE_PLAN=b256 SCI_MODEL=/dev/shm/rl_mixed_ckpt/best FORCE=1 OVERRIDE_PIN=1 LADDER_START=2 MIN_DISK_GIB=1 MIN_FREE_GIB=60 HUB_NO_MODEL_CARD=1 A_FROM_PREFLIGHT=0 HUB_PREFIX= GEMMA_MT_TAG_MODE=single_token
say "launching the supervisor (science RL from mixed best, 160 steps, seed 1234); config sha256 $(sha256sum $CFG/rl_F_final.yaml | cut -c1-16)"
nohup bash scripts/launch_clean_rerun_385m.sh run final > $LOGDIR/launch.out 2>&1 &
for i in $(seq 1 1800); do [ -f $RS/summary.json ] && break; sleep 20; done
[ -f $RS/summary.json ] || { say "TIMEOUT: no summary.json after 10 h"; exit 5; }
sleep 20; say "stage finished: $(python3 -c "import json;d=json.load(open('$RS/summary.json'));print('steps',d['steps'],'seconds',d['seconds'],'stop_reason',d['stop_reason'])")"
rm -rf $CK/last $CK/hub_stage $CK/hub_stage_latest; mkdir -p $CK/last; echo "E14_RUN_DONE" > $RS/E14_RUN_DONE; say "E14_RUN_DONE"
