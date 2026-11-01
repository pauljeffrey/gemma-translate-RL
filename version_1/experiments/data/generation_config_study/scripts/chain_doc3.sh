#!/bin/bash
cd /workspace/jobs/r12
while [ ! -f logs/chain_sft.done ]; do sleep 30; done
export GPU_UTIL=0.028 MAXBT=2048 MAX_SEQS=96
./run_doc_gen.sh doc3 T0.7_mp0.05_mt1.4:docfull2 T1.0_mp0.05_mt1.4:docfull2
echo done > logs/chain_doc3.done
