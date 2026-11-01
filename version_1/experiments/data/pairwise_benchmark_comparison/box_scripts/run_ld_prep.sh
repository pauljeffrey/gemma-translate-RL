#!/bin/bash
cd /workspace/jobs/e1
export PYTHONPATH=/workspace/sft_build/pylibs_tf4:/workspace/sft_build/pylibs:/workspace/sft_build/gemma_mt_2c PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
PY=/workspace/venv_gemma/bin/python
nice -n 10 $PY pairwise_longdoc_prep.py --a sft --b rl2c --hyp greedy_sent --out longdoc > ld_prep.log 2>&1
for k in seg doc; do for s in sft rl2c; do
  nice -n 10 $PY pairwise_surface.py longdoc/${k}_$s.jsonl longdoc/sys_${k}_$s.jsonl.gz --procs 2 > longdoc/surface_${k}_$s.log 2>&1
done; done
echo done > longdoc/PREP_DONE
