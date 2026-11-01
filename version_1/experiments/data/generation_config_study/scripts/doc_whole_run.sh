#!/bin/bash
# CPU: assemble + score the finished whole-document configs; aggregate on the sel2 subset against R6's whole-document configs and sentfix
cd /workspace/jobs/r12
names=""
for f in doc/out/rl2c/*__docfull2.jsonl.gz; do
  [ -e "$f" ] || continue
  cfg=$(basename $f | sed 's/__docfull2.jsonl.gz//')
  if [ ! -f doc/sc/rl2c/$cfg.pkl ]; then
    ./envpy.sh doc/asm.py full2 $cfg >> logs/asm_whole.log 2>&1 && ./envpy.sh doc/score_items.py rl2c $cfg >> logs/score_whole.log 2>&1
  fi
  names="$names $cfg"
done
echo $names > doc/res/whole_names.txt
./envpy.sh doc/docagg.py doc/res/whole.json greedy sel2 $names t03 t05 t07 t10 bon4 sentfix > logs/docagg_whole.log 2>&1
echo done > logs/doc_whole.done
