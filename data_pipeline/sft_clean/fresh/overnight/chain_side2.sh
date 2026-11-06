#!/bin/bash
# SIDE QUEUE, PHASE 2 (after the guard abort of 15:57: a side job next to the RL slowed it by 65 %): the side jobs run after the long-document RL has ended, one at a time next to its evaluation
# (at most 2 vLLM engines in all). E-09b and E-05 always run; E-02 and the post-hoc stronger-penalty runs only if they can finish before the projected end of the long-RL evaluation (RL end + 3.0 h).
source /workspace/science_sft/overnight/common.sh
export GEMMA_MT_TAG_MODE=single_token HF_HOME=/workspace/clean_sft/hf_home TMPDIR=/dev/shm/tmp; mkdir -p /dev/shm/tmp /dev/shm/side_out
SFT=$R/models/g15_385m_sft_final; RL=$R/runs_385m/mixed/best; RUN=$R/runs_385m_long/final
say "side queue phase 2 started; waiting for the long-document RL to end"
until [ -f $RUN/summary.json ] && ! pgrep -f "[g]emma_mt.rl.train_rl" >/dev/null; do sleep 30; done
END=$(( $(stat -c %Y $RUN/summary.json) + 3*3600 ))
say "long-document RL ended; side jobs start next to its evaluation (projected evaluation end $(date -u -d @$END +%H:%M))"
sleep 120
side2() {   # side2 NAME TIMEOUT EST_SECONDS(0 = always) CMD...
  local name=$1 to=$2 est=$3; shift 3
  if [ "$est" -gt 0 ] && [ $(( $(date +%s) + est )) -gt $END ]; then say "SIDE $name skipped: would end after the projected end of the long-RL evaluation ($(date -u -d @$END +%H:%M))"; return 0; fi
  local f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${f:-0}" -lt 20480 ]; then say "SIDE $name waits: only ${f} MiB GPU memory free"; for i in $(seq 1 30); do sleep 60; f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "${f:-0}" -ge 20480 ] && break; done; fi
  step SIDE_$name $to "$@"
}
side2 E09b_sft_fp32 10800 0 bash $O/run_fp32_eval.sh sft $SFT
side2 E09b_rl_fp32 10800 0 bash $O/run_fp32_eval.sh rl $RL
side2 E05 7200 0 bash -c "export PYTHONPATH=$E/pylibs_none:$E/gemma_mt_tree; $E/venv_comet/bin/python $O/sent_neural_all.py $O/e05_all.json 8"
mkdir -p $O/e02
side2 E02_sft 5400 3000 env PYTHONPATH=$E/gemma_mt_tree VLLM_CACHE_ROOT=$E/vllm_cache_guard /venv/main/bin/python $O/decode_study2.py --model $SFT --flores $E/bench/test_sets/flores_plus/devtest --out $O/e02/sft.json --gpu-util 0.15
side2 E02_rl 5400 3000 env PYTHONPATH=$E/gemma_mt_tree VLLM_CACHE_ROOT=$E/vllm_cache_guard /venv/main/bin/python $O/decode_study2.py --model $RL --flores $E/bench/test_sets/flores_plus/devtest --out $O/e02/rl.json --gpu-util 0.15
side2 E02_comet 3600 1500 bash -c "export PYTHONPATH=$E/pylibs_none:$E/gemma_mt_tree; $E/venv_comet/bin/python $O/decode_comet.py $O/e02/comet.json $O/e02/sft.json $O/e02/rl.json"
side2 E01x_sft_f050 7200 3600 bash $O/run_guard_eval_ext.sh sft_f050 $SFT 0.5 1.0
side2 E01x_rl_f050 7200 3600 bash $O/run_guard_eval_ext.sh rl_f050 $RL 0.5 1.0
touch $O/SIDE2_DONE; say "side queue phase 2 finished"
