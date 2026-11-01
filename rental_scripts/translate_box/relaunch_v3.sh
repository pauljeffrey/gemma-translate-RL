#!/bin/bash
cd /workspace/translate
mkdir -p out_old_set/nllb out_old_set/madlad
for pair in "nllb nllb-200-3.3b" "madlad madlad400-3b-mt"; do
  set -- $pair; d=out/$1/$2
  for f in $d/*.jsonl $d/*.summary.json; do [ -e "$f" ] || continue; b=$(basename $f); [ "$b" = sentence_cache.jsonl ] || mv "$f" out_old_set/$1/; done
  mv out/$1/progress.log out_old_set/$1/progress_oldset.log 2>/dev/null; mv out/$1/progress.json out_old_set/$1/progress_oldset.json 2>/dev/null
done
sed -i 's#../source_set/core/docs.jsonl#../source_set_v3/core/docs.jsonl#' run_nllb.sh run_madlad.sh
grep -o "source_set[a-z_0-9]*/core/docs.jsonl" run_nllb.sh run_madlad.sh
nohup setsid /workspace/translate/run_nllb.sh > logs/nllb.out 2>&1 < /dev/null &
sleep 2
nohup setsid /workspace/translate/run_madlad.sh > logs/madlad.out 2>&1 < /dev/null &
echo relaunched at $(date -u +%T)
