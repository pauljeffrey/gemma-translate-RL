#!/bin/bash
cd /workspace/science_sft/overnight
kill_tree() { for c in $(pgrep -P $1); do kill_tree $c; done; kill $1 2>/dev/null; }
for p in $(pgrep -f "run_guard_eval_ext.sh rl_f050") $(pgrep -f "run_old_eval.sh rl2c") $(pgrep -f "run_guard_ext_full.sh"); do kill_tree $p; done
sleep 8
setsid nohup bash run_doc_eval_reruns.sh > run_reruns.out 2>&1 < /dev/null &
sleep 3; pgrep -af "run_doc_eval_reruns|chain_p" | cut -c1-90
