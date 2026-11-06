#!/bin/bash
# usage: e04_run.sh TAG SEED PHASE2 [FORK_FROM_DIR]
#   TAG     run name (run dir /workspace/clean_rerun/runs_e04_TAG/A, weights /dev/shm/e04_TAG/{best,final,last})
#   SEED    trainer seed (prompt order + sampling)
#   PHASE2  config of steps 41-100: full | noref | noguide | kl025 | base      (steps 1-40 = base, features off, exactly as stage A was run: 40 steps without the features, resumed with them)
#   FORK_FROM_DIR  (E-11 arms) a saved step-40 state (made by SAVE_FORK40=dir of the parent run): the run starts from it at step 41 with the PHASE2 config; no steps 1-40 are repeated
# env: SAVE_FORK40=/dev/shm/e11_fork40_xx  copy last/ + sidecars of THIS run at its step-40 checkpoint (parent of the E-11 arms)
# Mirrors the stage-A procedure of 2026-11-03: trainer started with the base config, stopped (SIGTERM) right after the step-40 checkpoint, the config swapped, the supervisor restarts the trainer from last/.
TAG=$1; SEED=$2; PH2=$3; FORK=${4:-}
R=/workspace/clean_rerun; E2=/workspace/e2405
mkdir -p /dev/shm/tmp /dev/shm/xdg_cache; export TMPDIR=/dev/shm/tmp XDG_CACHE_HOME=/dev/shm/xdg_cache
RUNS=$R/runs_e04_$TAG; RS=$RUNS/A; LOGDIR=$R/logs_e04_$TAG; CFG=$R/tree/gemma_mt/configs/e04_$TAG; CK=/dev/shm/e04_$TAG; L=$E2/log_$TAG.txt
say() { echo "$(date -u +%FT%TZ) [e04 $TAG] $*" | tee -a $L; }
[ "$(df --output=avail -k / | tail -1)" -lt 1048576 ] && { say "LOW DISK (< 1 GB free on /): not started"; exit 3; }
[ "$(df --output=avail -k /dev/shm | tail -1)" -lt 12582912 ] && { say "LOW /dev/shm (< 12 GB free): not started"; exit 3; }
mkdir -p $RS $LOGDIR $CFG $CK; PY=/workspace/venv_gemma/bin/python
for d in best final last hub_stage hub_stage_latest; do mkdir -p $CK/$d; [ -e $RS/$d ] || ln -s $CK/$d $RS/$d; done      # every weight dir on /dev/shm (a real hub_stage dir on the root disk crash-looped the long RL)
if [ -n "$FORK" ]; then
  say "FORK from $FORK at step 40: phase-2 config '$PH2', seed $SEED"
  rm -rf $CK/last; mkdir -p $CK/last; cp -a $FORK/last/. $CK/last/ || { say "fork copy failed"; exit 4; }
  cp -a $FORK/sidecars/. $RS/ 2>/dev/null; cp $FORK/metrics_upto40.jsonl $RS/metrics.jsonl
  $PY $E2/e04_cfg.py $PH2 $SEED $RS > $CFG/rl_A_b64.yaml
else
  $PY $E2/e04_cfg.py base $SEED $RS > $CFG/rl_A_b64.yaml
  $PY $E2/e04_cfg.py $PH2 $SEED $RS > $CFG/phase2.yaml
fi
cp $CFG/rl_A_b64.yaml $E2/cfg_${TAG}_phase1.yaml; [ -f $CFG/phase2.yaml ] && cp $CFG/phase2.yaml $E2/cfg_${TAG}_phase2.yaml
bash $E2/codestate.sh $RS/code_state.txt
cd $R/tree/gemma_mt; source $R/rl_env.sh
export RUNS LOGDIR CFGDIR=$CFG STAGE_PLAN=b256 FORCE=1 OVERRIDE_PIN=1 LADDER_START=2 MIN_DISK_GIB=1 MIN_FREE_GIB=60 HUB_NO_MODEL_CARD=1 A_FROM_PREFLIGHT=0 SKIP_PREFLIGHT=1 STEPS_A=100 HUB_PREFIX= GEMMA_MT_TAG_MODE=single_token
say "launching the supervisor (steps 100, seed $SEED); config sha256 $(sha256sum $CFG/rl_A_b64.yaml | cut -c1-16)"
nohup bash scripts/launch_clean_rerun_385m.sh run A > $LOGDIR/launch.out 2>&1 &
if [ -z "$FORK" ] && [ "$PH2" != base ]; then
  ( while true; do
      [ -f $RS/summary.json ] && exit 0
      if [ "$($PY $E2/e04_wait40.py $RS 40)" = READY ]; then
        if [ -n "${SAVE_FORK40:-}" ]; then   # parent of the E-11 arms: keep the step-40 state
          rm -rf $SAVE_FORK40; mkdir -p $SAVE_FORK40/last $SAVE_FORK40/sidecars
          cp -a $CK/last/. $SAVE_FORK40/last/
          for f in LANG_WEIGHTS.state.json; do [ -f $RS/$f ] && cp $RS/$f $SAVE_FORK40/sidecars/; done
          $PY - <<PYEOF
import json
out = open("$SAVE_FORK40/metrics_upto40.jsonl", "w")
for l in open("$RS/metrics.jsonl"):
    try: r = json.loads(l)
    except Exception: continue
    if r.get("step", 0) <= 40: out.write(l if l.endswith("\n") else l + "\n")
PYEOF
          say "step-40 state saved to $SAVE_FORK40 ($(du -sh $SAVE_FORK40 | cut -f1))"
        fi
        cp $CFG/phase2.yaml $CFG/rl_A_b64.yaml.new && mv $CFG/rl_A_b64.yaml.new $CFG/rl_A_b64.yaml
        say "step-40 checkpoint is on disk: config swapped to '$PH2' (sha256 $(sha256sum $CFG/rl_A_b64.yaml | cut -c1-16)); SIGTERM to the trainer (the supervisor restarts it from last/)"
        pkill -TERM -f "gemma_mt.rl.train_rl --config $CFG/rl_A_b64.yaml"
        exit 0
      fi
      sleep 8
    done ) &
fi
# wait for the end of the stage
for i in $(seq 1 1800); do [ -f $RS/summary.json ] && break; sleep 20; done
[ -f $RS/summary.json ] || { say "TIMEOUT: no summary.json after 10 h"; exit 5; }
sleep 20; say "stage finished: $(python3 -c "import json;d=json.load(open('$RS/summary.json'));print('steps',d['steps'],'seconds',d['seconds'],'stop_reason',d['stop_reason'])")"
rm -rf $CK/last $CK/hub_stage $CK/hub_stage_latest      # trainer state of a finished 100-step run is not needed (best/final stay for the evaluation)
mkdir -p $CK/last; echo "E04_RUN_DONE $TAG" > $RS/E04_RUN_DONE; say "E04_RUN_DONE; /dev/shm free $(df -h /dev/shm | tail -1 | awk '{print $4}')"
