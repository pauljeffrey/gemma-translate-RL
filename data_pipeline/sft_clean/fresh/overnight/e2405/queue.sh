#!/bin/bash
# Sequential GPU queue of 2026-11-05 (E-02, E-04, E-11, E-14). One job at a time, nothing next to a training run. Idempotent: a finished stage has a marker in /workspace/e2405/q/ and is skipped on restart.
# Starts only when /workspace/e2405/GO exists, the E-03 evaluation has finished (ALL_EVALS_DONE) and the GPU has been idle for 5 min. Halt by `touch /workspace/e2405/STOP_QUEUE` (checked between stages).
E2=/workspace/e2405; Q=$E2/q; R=/workspace/clean_rerun; E=/workspace/eval_385m; mkdir -p $Q
say() { echo "$(date -u +%FT%TZ) [queue] $*" | tee -a $E2/queue.log; }
gpu_idle() { [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)" -lt 5 ]; }
wait_gpu() { local need=${1:-5} ok=0; while [ $ok -lt $need ]; do if gpu_idle; then ok=$((ok+1)); else ok=0; fi; sleep 60; done; }
stop_check() { [ -f $E2/STOP_QUEUE ] && { say "STOP_QUEUE present: queue halts"; exit 0; }; }
active() { echo "$1 $2" > $E2/ACTIVE; }
stage() {  # name command...  : run once, marker on success
  local n=$1; shift; [ -f $Q/$n.DONE ] && { say "stage $n already done"; return 0; }
  stop_check; say "stage $n start"; "$@"; local rc=$?
  if [ $rc -eq 0 ]; then touch $Q/$n.DONE; say "stage $n DONE"; else touch $Q/$n.FAILED; say "stage $n FAILED rc=$rc (queue continues with the next stage)"; fi
  return 0
}
# ---------------------------------------------------------------- waits
while [ ! -f $E2/GO ]; do sleep 30; done
# (the E-03 executor released the GPU by message at 05:30 UTC and stopped its own auto_eval; GO is written by the E-02/E-04 agent after checking that nvidia-smi shows no process)
say "GO received (E-03 executor released the GPU); waiting for 5 idle GPU minutes"; wait_gpu; say "GPU idle"
# ---------------------------------------------------------------- E-02
e02() { active /workspace/e2405/e02 /workspace/e2405/e02/log.txt; bash $E2/e02_run.sh smoke && python3 $E2/e02_smoke_check.py | tee -a $E2/e02/log.txt && bash $E2/e02_run.sh; }
stage e02 e02
# ---------------------------------------------------------------- E-04 seeds, each with its harness evaluation
seedrun() {  # tag seed phase2 [fork] ; then evaluation and weight cleanup
  local tag=$1 seed=$2 ph2=$3 fork=${4:-}
  wait_gpu 1; active $R/runs_e04_$tag/A $E2/log_$tag.txt
  bash $E2/e04_run.sh $tag $seed $ph2 $fork || return 1
  [ -f $R/runs_e04_$tag/A/summary.json ] || return 1
  wait_gpu 1; bash $E2/e_eval20.sh e04_$tag /dev/shm/e04_$tag/final 0.5 || return 1
  local nm=$tag; case $tag in s2) nm=seed2;; s3) nm=seed3;; esac
  nohup bash $E2/push_run.sh /dev/shm/e04_$tag e04-$nm-step100 "E-04/E-11 run $tag (seed $seed, stage-A recipe, 100 steps, phase 2 = $ph2, fork=${fork:-none})" "stage-A b64 recipe, 100 steps, 64 prompts x 16 samples, steps 1-40 features off, 41-100 config '$ph2'; exact yaml in the reports" >> $E2/push.log 2>&1 &
  say "push of $tag started in the background (weights are deleted by the pusher only after verification)"
}
S2=20261105; S3=20261106
stage e04_s2 env SAVE_FORK40=/dev/shm/e11_fork40_s2 bash -c "$(declare -f seedrun wait_gpu gpu_idle active say); E2=$E2; R=$R; export E2 R; seedrun s2 $S2 full"
stage e04_s3 bash -c "$(declare -f seedrun wait_gpu gpu_idle active say); E2=$E2; R=$R; export E2 R; seedrun s3 $S3 full"
touch $Q/E04_ALL
# ---------------------------------------------------------------- E-11 arms (forks of the seed-2 step-40 state), a / c / d, then e if the rule says so
for arm in "a_base base" "c_noref noref" "d_noguide noguide"; do
  set -- $arm
  stage e11_$1 bash -c "$(declare -f seedrun wait_gpu gpu_idle active say); E2=$E2; R=$R; export E2 R; seedrun $1 $S2 $2 /dev/shm/e11_fork40_s2"
done
dec=$(python3 $E2/decide_e.py $R/runs_e04_s2/A $R/runs_e04_s3/A); say "arm (e) rule: $(echo "$dec" | head -1) | $(echo "$dec" | tail -n +2 | tr '\n' ';')"; echo "$dec" > $E2/e11_e_rule.txt
if [ "$(echo "$dec" | head -1)" = RUN ]; then stage e11_e_kl025 bash -c "$(declare -f seedrun wait_gpu gpu_idle active say); E2=$E2; R=$R; export E2 R; seedrun e_kl025 $S2 kl025 /dev/shm/e11_fork40_s2"; fi
rm -rf /dev/shm/e11_fork40_s2; touch $Q/E11_ALL
# ---------------------------------------------------------------- E-14
while [ ! -f $E2/e14_ready ]; do stop_check; sleep 60; done
e14() { wait_gpu 1; active $R/runs_e14/final $E2/log_e14.txt; bash $E2/e14_run.sh && wait_gpu 1 && bash $E2/e14_eval.sh; local rc=$?; FORCE_BEST=1 bash $E2/push_run.sh /dev/shm/e14 e14-science-rl-only-final "E-14 science RL from mixed best without the science SFT" "executed science-RL recipe (seed 1234, 160 steps, lr 1e-6) started from mixed best (Hub main)" e14-science-rl-only-best >> $E2/push.log 2>&1; return $rc; }
stage e14 e14
touch $Q/QUEUE_DONE; say "QUEUE_DONE"
