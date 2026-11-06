#!/bin/bash
# SIDE QUEUE next to the long-document RL (owner order 2026-11-04 15:45 UTC; instance will be terminated, work must end today): one side job at a time in spare GPU memory
# (vLLM gpu_memory_utilization 0.15, only when >= 20 GB of the GPU are free), each under side_guard.py (abort at once if the RL step time rises > 25 % or the RL restarts / alerts).
# Order: E-09b (fp32 SFT, fp32 RL), E-05, then E-02 and the post-hoc stronger-penalty runs only if they can finish before the long-RL evaluation ends. E-04 and E-03 are DROPPED (instance termination).
# Nothing is deleted; outputs of harness runs go to /dev/shm.
source /workspace/science_sft/overnight/common.sh
export GEMMA_MT_TAG_MODE=single_token HF_HOME=/workspace/clean_sft/hf_home TMPDIR=/dev/shm/tmp; mkdir -p /dev/shm/tmp /dev/shm/side_out
SFT=$R/models/g15_385m_sft_final; RL=$R/runs_385m/mixed/best
RUN=$R/runs_385m_long/final
free_gpu_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
rl_running() { [ ! -f $RUN/summary.json ] && pgrep -f "[g]emma_mt.rl.train_rl" >/dev/null; }
projected_eval_end() {   # epoch seconds: RL end (remaining steps x recent median step time + 4 min per remaining 25-step evaluation) + 3.0 h of post-run evaluation
  python3 - <<PY
import json, time, statistics
L = [json.loads(l) for l in open("$RUN/metrics.jsonl")]
tr = {r["step"]: r["time"] for r in L if "reward" in r}
ks = sorted(tr); iv = [tr[b] - tr[a] for a, b in zip(ks, ks[1:]) if b == a + 1 and b % 25 not in (0, 1)][-8:]
step = ks[-1]; per = statistics.median(iv) if iv else 145.0
rem = 80 - step; end = time.time() + rem * per + (rem // 25 + 1) * 240
print(int(end + 3.0 * 3600))
PY
}
# side job: side NAME TIMEOUT EST_SECONDS CMD...   (EST_SECONDS > 0: start only if now + EST < projected end of the long-RL evaluation)
side() {
  local name=$1 to=$2 est=$3; shift 3
  [ -f $O/SIDE_ABORTED ] && { say "SIDE $name skipped: a side job was aborted by the guard ($(cat $O/SIDE_ABORTED)); no further side jobs while the RL runs"; return 0; }
  rl_running || { say "SIDE $name skipped: the long-document RL is not running any more (its evaluation owns the GPU now)"; return 0; }
  if [ "$est" -gt 0 ]; then
    local end=$(projected_eval_end); [ $(( $(date +%s) + est )) -gt "$end" ] && { say "SIDE $name skipped: would end after the projected end of the long-RL evaluation ($(date -u -d @$end +%H:%M))"; return 0; }
  fi
  local f=$(free_gpu_mib); [ "${f:-0}" -lt 20480 ] && { say "SIDE $name skipped: only ${f} MiB of GPU memory free (< 20 GB)"; return 0; }
  local pf=$O/side_$name.pid; rm -f $pf
  ( timeout --signal=TERM --kill-after=60 $to "$@" > $O/step_$name.out 2>&1 & echo $! > $pf; wait $!; echo $? > $O/side_$name.rc ) &
  local sub=$!; sleep 3
  say "SIDE $name start (pid $(cat $pf 2>/dev/null))"
  python3 $O/side_guard.py $name $pf; local g=$?
  wait $sub 2>/dev/null; local rc=$(cat $O/side_$name.rc 2>/dev/null)
  if [ "$rc" = 0 ] && [ $g -eq 0 ]; then touch $O/SIDE_$name.DONE; say "SIDE $name DONE"; else touch $O/SIDE_$name.FAILED; say "SIDE $name FAILED/ABORTED (rc=$rc guard=$g)"; fi
}
say "side queue started; waiting for the long-document RL to be past its first steps"
until rl_running && [ -s $RUN/metrics.jsonl ]; do sleep 30; done
side E09b_sft_fp32 10800 0 bash $O/run_fp32_eval.sh sft $SFT
side E09b_rl_fp32 10800 0 bash $O/run_fp32_eval.sh rl $RL
side E05 7200 0 bash -c "export PYTHONPATH=$E/pylibs_none:$E/gemma_mt_tree; $E/venv_comet/bin/python $O/sent_neural_all.py $O/e05_all.json 7"
mkdir -p $O/e02
side E02_sft 5400 3600 env PYTHONPATH=$E/gemma_mt_tree VLLM_CACHE_ROOT=$E/vllm_cache_guard /venv/main/bin/python $O/decode_study2.py --model $SFT --flores $E/bench/test_sets/flores_plus/devtest --out $O/e02/sft.json --gpu-util 0.15
side E02_rl 5400 3600 env PYTHONPATH=$E/gemma_mt_tree VLLM_CACHE_ROOT=$E/vllm_cache_guard /venv/main/bin/python $O/decode_study2.py --model $RL --flores $E/bench/test_sets/flores_plus/devtest --out $O/e02/rl.json --gpu-util 0.15
side E02_comet 3600 2400 bash -c "export PYTHONPATH=$E/pylibs_none:$E/gemma_mt_tree; $E/venv_comet/bin/python $O/decode_comet.py $O/e02/comet.json $O/e02/sft.json $O/e02/rl.json"
side E01x_sft_f050 7200 3600 bash $O/run_guard_eval_ext.sh sft_f050 $SFT 0.5 1.0
side E01x_rl_f050 7200 3600 bash $O/run_guard_eval_ext.sh rl_f050 $RL 0.5 1.0
say "E-04 (two extra RL seeds) and E-03 (continued SFT) NOT RUN: dropped by the owner's order of 2026-11-04 15:45 UTC (the instance will be terminated; the work has to end today)"
touch $O/SIDE_QUEUE_DONE; say "side queue finished"
