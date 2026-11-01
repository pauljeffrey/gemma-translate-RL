#!/bin/bash
# after the 5th whole-document config: stop chain_doc1 (the remaining configs are low value) and run the min_tokens variants
cd /workspace/jobs/r12
while [ ! -f doc/out/rl2c/T0.5_mp0.05__docfull2.jsonl.gz ]; do sleep 30; done
./killmine.sh "chain_doc1.sh|run_doc_gen.sh"; sleep 1; ./killmine.sh "gen_doc.py|EngineCore"; sleep 15
export GPU_UTIL=0.028 MAXBT=2048 MAX_SEQS=96
./run_doc_gen.sh doc2 T0.7_mp0.05_mt1.4:docfull2 T1.0_mp0.05_mt1.4:docfull2
echo done > logs/chain_doc2.done
