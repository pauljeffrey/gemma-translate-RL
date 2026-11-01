#!/bin/bash
# Owner 2026-10-20: push the RL best-translation archives to a PRIVATE dataset under ${HF_ORG}, only if they pass the quality gate, at the END of all training.
# Waits for the phase-2c supervisor to finish (RL_DONE) or give up; then runs scripts/archive_publish.py. A missing/invalid Hub token is retried every 30 min for 48 h.
cd /workspace/sft_build
export PYTHONPATH=/workspace/sft_build/gemma_mt_2c
PY=/workspace/venv_gemma/bin/python; OUT=/workspace/sft_build/publish_final
say() { echo "$(date -u +%FT%TZ) [archive_publish] $*" | tee -a logs/archive_publish.log; }
say "armed: waiting for runs/rl2c/RL_DONE (or logs/RENTAL_RL2C_GAVE_UP)"
while [ ! -f runs/rl2c/RL_DONE ] && [ ! -f logs/RENTAL_RL2C_GAVE_UP ]; do sleep 120; done
say "training finished: running the gate"
for try in $(seq 1 97); do
  $PY gemma_mt_2c/scripts/archive_publish.py --archive rl2=runs/rl2/archive/archive.sqlite --archive rl2b=runs/rl2b/archive/archive.sqlite --archive rl2c=runs/rl2c/archive/archive.sqlite \
      --drop-after rl2b=488 --out $OUT >> logs/archive_publish.log 2>&1
  st=$(cat $OUT/PUBLISH_STATUS 2>/dev/null)
  say "attempt $try: status=$st"
  [ "$st" != "no_token" ] && break
  sleep 1800
done
say "done: $st (report: $OUT/PUBLISH_REPORT.md)"
