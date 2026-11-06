#!/bin/bash
# PHASE 3: after the long-document RL and its post-run evaluation: E-05 (SSA-COMET-QE for all stages), E-02 (beam-4 / MBR-16 at 768 tokens, SFT and RL), E-04 (two extra RL seeds, 100 steps,
# 20 languages), then E-03 only if its config and data exist. Starts on POST_LONG_RL_DONE, or LONG_RL_SKIPPED, or when the GPU has been idle for 15 minutes after CONTROLS_DONE.
source /workspace/science_sft/overnight/common.sh
export GEMMA_MT_TAG_MODE=single_token HF_HOME=/workspace/clean_sft/hf_home
SFT=$R/models/g15_385m_sft_final; RL=$R/runs_385m/mixed/best
say "P3 chain started; waiting for the long-document RL stage and its evaluation"
idle=0
while true; do
  [ -f $S/POST_LONG_RL_DONE ] && { say "POST_LONG_RL_DONE seen"; break; }
  [ -f $S/LONG_RL_SKIPPED ] && { say "LONG_RL_SKIPPED seen"; break; }
  if [ -f $O/CONTROLS_DONE ]; then
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$apps" -eq 0 ]; then idle=$((idle + 20)); else idle=0; fi
    [ $idle -ge 900 ] && { say "GPU idle for 15 min after CONTROLS_DONE: P3 starts without waiting for the long-document RL"; break; }
  fi
  sleep 20
done
step E05 7200 bash -c "export PYTHONPATH=$E/pylibs_none:$E/gemma_mt_tree; $E/venv_comet/bin/python $O/sent_neural_all.py $O/e05_all.json 14"
mkdir -p $O/e02
for mn in sft:$SFT rl:$RL; do
  n=${mn%%:*}; m=${mn#*:}
  step E02_$n 5400 env PYTHONPATH=$E/gemma_mt_tree VLLM_CACHE_ROOT=$E/vllm_cache_guard /venv/main/bin/python $O/decode_study2.py --model $m --flores $E/bench/test_sets/flores_plus/devtest --out $O/e02/$n.json --gpu-util 0.3
done
step E02_comet 3600 bash -c "export PYTHONPATH=$E/pylibs_none:$E/gemma_mt_tree; $E/venv_comet/bin/python $O/decode_comet.py $O/e02/comet.json $O/e02/sft.json $O/e02/rl.json"
# E-09b (coordinator 2026-11-04): full vLLM fp32 run of the SFT and the RL model on the headline cells (harness: --dtype float32 now reaches the vLLM engine)
step E09b_sft_fp32 10800 bash $O/run_fp32_eval.sh sft $SFT
step E09b_rl_fp32 10800 bash $O/run_fp32_eval.sh rl $RL
# E-01 post-hoc extension (stronger frequency penalty 0.50 on SFT and RL; coordinator order 2026-11-04: after the long-document RL, at most 2 vLLM engines at once)
step E01x_sft_f050 7200 bash $O/run_guard_eval_ext.sh sft_f050 $SFT 0.5 1.0 &
sleep 90
step E01x_rl_f050 7200 bash $O/run_guard_eval_ext.sh rl_f050 $RL 0.5 1.0 &
wait
step E04_pool 300 bash $O/prep_seed_pool.sh
for seed in 2 3; do
  step E04_seed${seed}_train 14400 bash $O/run_seed.sh $seed
  step E04_seed${seed}_eval 5400 bash $O/run_seed_eval.sh $seed
done
if [ -f $O/E03_READY ] && [ -f $O/E03_run.sh ]; then step E03 36000 bash $O/E03_run.sh; else say "E-03 not started: no $O/E03_READY (config and data not present)"; fi
touch $O/P3_DONE; say "P3 DONE"
