#!/bin/bash
# coordinator reorder 2026-11-04: stop the post-hoc E-01 stronger-penalty runs (they move to P3, after the long-document RL); E-13 and E-20 keep running (2 engines)
cd /workspace/science_sft/overnight
kill_tree() { for c in $(pgrep -P $1); do kill_tree $c; done; kill $1 2>/dev/null; }
for p in $(pgrep -f "run_guard_eval_ext.sh"); do kill_tree $p; done
sleep 5
for d in out_guardx_sft_f050 out_guardx_rl_f050; do [ -d /workspace/eval_385m/$d ] && [ ! -f /workspace/eval_385m/$d/EVAL_DONE ] && mv /workspace/eval_385m/$d /workspace/eval_385m/${d}_STOPPED_reorder; done
bash restart_p3.sh
pgrep -af "run_old_eval|run_plain_eval|run_guard_eval_ext|run_doc_eval_reruns" | cut -c1-110
