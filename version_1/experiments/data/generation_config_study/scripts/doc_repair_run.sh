#!/bin/bash
# CPU: assemble + score + aggregate the sentence-by-sentence documents with each repair variant (sel = all 3,418 selected items)
cd /workspace/jobs/r12
names="greedy_sent_r12 sentfix_r12"
./envpy.sh doc/asm.py fix greedy t07x4 greedy_sent_r12 none > logs/asm_gs.log 2>&1
for f in doc/out/rl2c/*__sentfail8.jsonl.gz; do
  cfg=$(basename $f | sed 's/__sentfail8.jsonl.gz//'); n=fix_${cfg}
  ./envpy.sh doc/asm.py fix greedy $cfg $n cons >> logs/asm_gs.log 2>&1 && names="$names $n"
done
./envpy.sh doc/score_items.py rl2c $names >> logs/score_gs.log 2>&1
./envpy.sh doc/docagg.py doc/res/repair.json sentfix sel $names greedy greedy_sent > logs/docagg_repair.log 2>&1
echo $names > doc/res/repair_names.txt
echo done > logs/doc_repair.done
