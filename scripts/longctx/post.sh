#!/bin/bash
# CPU watcher: assemble our models' outputs and score every hyp file once; waits for the NLLB-3.3B sentence outputs (the pseudo-reference of the non-core items) first.
cd /workspace/jobs/lc1; source env_lc.sh
until [ -f work/refs_nllb_sent.jsonl ]; do sleep 60; done
while true; do
  for m in sft rl a2w live; do
    [ -f out/$m/fix.jsonl.gz ] && [ ! -f hyp/${m}__sentfix.jsonl ] && nice -n 19 $PYG lc_assemble_ours.py $m >> logs/assemble.log 2>&1
  done
  for f in hyp/*__whole.jsonl hyp/*__sent.jsonl hyp/*__sentfix.jsonl; do
    [ -f "$f" ] || continue
    n=$(basename $f .jsonl)
    if [ ! -f sc/$n.pkl ] || [ $f -nt sc/$n.pkl ] || [ work/refs_nllb_sent.jsonl -nt sc/$n.pkl ]; then
      [ -f logs/QUEUE_DONE ] || sleep 20   # let writers finish
      nice -n 19 $PYG lc_score.py $n >> logs/score.log 2>&1
    fi
  done
  [ -f logs/QUEUE_DONE ] && [ -z "$(ls hyp/*.jsonl 2>/dev/null | while read f; do n=$(basename $f .jsonl); case $n in *__whole|*__sent|*__sentfix) [ -f sc/$n.pkl ] || echo $n;; esac; done)" ] && break
  sleep 90
done
touch logs/POST_DONE
