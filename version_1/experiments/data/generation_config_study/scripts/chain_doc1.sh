#!/bin/bash
# document level, GPU strictly sequential and <= 7 GB: repairs of unfinished sentence units, then whole-document sampling configs
cd /workspace/jobs/r12
while [ ! -f logs/chain_rl2.done ]; do sleep 30; done
export GPU_UTIL=0.030 MAX_SEQS=96
JOBS="T0.5_n4:sentfail8 T0.7_p0.9_n4:sentfail8 T0.7_mp0.05_n4:sentfail8 T1.0_p0.9_n4:sentfail8 T1.0_mp0.05_n4:sentfail8 T1.0_n4:sentfail8 T0.3_mp0.05_n4:sentfail8 T0.5_mp0.1_n4:sentfail8 T1.0_mp0.1_n4:sentfail8 T0_rp1.05:sentfail8 T0_rp1.1:sentfail8 T0.7_n8:sentfail8 T1.0_mp0.05_n8:sentfail8 T0.7_mp0.05_n8:sentfail8"
JOBS="$JOBS T1.0_mp0.05:docfull2 T0.7_mp0.05:docfull2 T1.0_k64_p0.95:docfull2 T1.0_mp0.1:docfull2 T0.5_mp0.05:docfull2 T1.0_p0.9:docfull2 T0.7_mp0.05_n4:docfull2 T0.7_p0.9:docfull2"
./run_doc_gen.sh doc1 $JOBS
echo done > logs/chain_doc1.done
