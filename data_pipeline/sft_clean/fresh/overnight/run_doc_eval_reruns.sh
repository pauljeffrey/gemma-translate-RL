#!/bin/bash
# (named so that the long-document waiter's busy check sees it) Reruns of the four harness evaluations that failed 09:00-09:16 because I had put the vLLM/Triton compile cache on /dev/shm,
# which is mounted noexec ("Triton cache error: compiled module cuda_utils.so could not be loaded"). Cache is back on /workspace/eval_385m/vllm_cache_guard (47 MB).
source /workspace/science_sft/overnight/common.sh
SFT=$R/models/g15_385m_sft_final; RL=$R/runs_385m/mixed/best
for d in out_old_sft out_old_rl2c out_guardx_sft_f050 out_guardx_rl_f050; do [ -d $E/$d ] && [ ! -f $E/$d/EVAL_DONE ] && mv $E/$d $E/${d}_FAILED_noexec_cache; done
( step E01x_sft_f050_rerun 7200 bash $O/run_guard_eval_ext.sh sft_f050 $SFT 0.5 1.0 ) &
sleep 45
( step E01x_rl_f050_rerun 7200 bash $O/run_guard_eval_ext.sh rl_f050 $RL 0.5 1.0 ) &
( step E13_sft_rerun 5400 bash $O/run_old_eval.sh sft ${HF_ORG}/gemma3-270m-translate; step E13_rl2c_rerun 5400 bash $O/run_old_eval.sh rl2c /dev/shm/model_rl2c ) &
wait
touch $O/RERUNS_DONE; say "reruns of E-13 and the E-01 stronger-penalty evaluations finished"
