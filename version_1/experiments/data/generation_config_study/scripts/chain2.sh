#!/bin/bash
# sentence-level tail: after generation -> pool evals -> grid eval -> QE (GPU, ~3 GB, sequential) -> RL part
cd /workspace/jobs/r12
while ! tail -1 logs/grid2b.log | grep -q "^done"; do sleep 30; done
./envpy.sh eval_grid.py "*" >> logs/eval4.log 2>&1
./envpy.sh pool_eval.py T1.0_n16 T1.0_p0.9_n16 T1.0_p0.95_n16 T1.0_mp0.05_n16 T1.0_mp0.1_n16 T0.7_n16 T0.7_mp0.05_n16 T0.5_n16 T0.3_n16 T0.7_p0.9_n16 T0.5_mp0.05_n16 >> logs/pool5.log 2>&1
./envpy.sh qe_pools.py T0.7_n16:16 T1.0_mp0.05_n16:16 T0.5_n16:8 T0.7_n16:8 T1.0_mp0.05_n16:8 >> logs/qep.log 2>&1
./envpy.sh qe_eval.py greedy T0.3 T0.5 T0.7 T1.0 T1.0_p0.9 T1.0_p0.95 T1.0_mp0.05 T1.0_mp0.1 T1.0_k64_p0.95 T0.2_mp0.05 T0.4_p0.8 T0.6_p0.9 hf_T1.0_typ0.9 hf_T1.0_eps0.02 \
  mbr4__T0.7_n16 mbr8__T0.7_n16 mbr16__T0.7_n16 mbr8__T0.5_n16 mbr16__T0.5_n16 mbr16__T0.3_n16 mbr8__T0.3_n16 mbr8__T1.0_p0.9_n16 mbr16__T1.0_p0.9_n16 mbr8__T1.0_mp0.05_n16 mbr16__T1.0_mp0.05_n16 mbr16__T0.7_mp0.05_n16 mbr16__T1.0_p0.95_n16 mbr16__T1.0_n16 \
  qe8__T0.7_n16 qe16__T0.7_n16 mbrqe8__T0.7_n16 mbrqe16__T0.7_n16 qe16__T1.0_mp0.05_n16 mbrqe16__T1.0_mp0.05_n16 qe8__T0.5_n16 mbrqe8__T0.5_n16 oracle8__T0.7_n16 >> logs/qe_eval.log 2>&1
echo done > logs/chain2.done
