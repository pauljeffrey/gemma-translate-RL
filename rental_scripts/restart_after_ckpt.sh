#!/bin/bash
# one-off: after the next 2c checkpoint (runs/rl2c/last/trainer_state.pt rewritten) restart the trainer once so it uses the new hub_repo_id (supervisor resumes it)
cd /workspace/sft_build
CK=runs/rl2c/last/trainer_state.pt; base=$(stat -c %Y $CK)
say() { echo "$(date -u +%FT%TZ) [restart_hub] $*" | tee -a logs/rental_run.log; }
say "armed: waiting for the next checkpoint (mtime $base)"
while true; do m=$(stat -c %Y $CK 2>/dev/null || echo 0); if [ "$m" != "$base" ] && [ ! -e $CK.tmp ]; then sleep 20; m2=$(stat -c %Y $CK); [ "$m2" = "$m" ] && break; fi; sleep 5; done
P=$(ps -eo pid,args | grep -a "[t]rain_rl --config gemma_mt_2c" | awk '{print $1}' | head -1)
say "checkpoint complete: restarting trainer pid $P (new hub_repo_id ${HF_ORG_LEGACY}/gemma-3-270m-translate-rl2c)"
kill $P
