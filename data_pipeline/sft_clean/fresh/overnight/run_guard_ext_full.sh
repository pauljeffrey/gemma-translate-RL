#!/bin/bash
source /workspace/science_sft/overnight/common.sh
SFT=$R/models/g15_385m_sft_final; RL=$R/runs_385m/mixed/best
step E01x_sft_f050 7200 bash $O/run_guard_eval_ext.sh sft_f050 $SFT 0.5 1.0 &
sleep 60
step E01x_rl_f050 7200 bash $O/run_guard_eval_ext.sh rl_f050 $RL 0.5 1.0 &
wait
touch $O/E01X_DONE
