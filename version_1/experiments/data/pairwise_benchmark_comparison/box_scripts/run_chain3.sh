#!/bin/bash
cd /workspace/jobs/e1
source ./env_common.sh
# CPU: stats of the 320-token FLORES+ comparison (no neural metrics)
mkdir -p in320; ln -sf ../sys_sft320.jsonl.gz in320/sys_sft.jsonl.gz; ln -sf ../sys_rl320.jsonl.gz in320/sys_rl.jsonl.gz
(nice -n 19 $PY pairwise_stats.py --dir /workspace/jobs/e1/in320 --a sft --b rl --out summary --neural none --lang-groups /workspace/jobs/e1/lang_groups.json --benchmarks flores_plus > stats_320.log 2>&1; echo done > STATS_320_DONE) &
# GPU (one job at a time, <= 4.5 GB cap, batch 32)
nice -n 19 $PY pairwise_neural.py --sys sft=sys_sft.jsonl.gz --sys rl=sys_rl.jsonl.gz --out neural --cap 400 --gpu-gib 4.5 --batch 32 --metrics ssa_ref,afri_ref,afri_qe,ssa_qe > neural.log 2>&1
echo done > CHAIN_NEURAL_DONE
(nice -n 19 $PY pairwise_stats.py --dir /workspace/jobs/e1 --a sft --b rl --out summary --neural neural --lang-groups /workspace/jobs/e1/lang_groups.json > stats_main.log 2>&1; echo done > STATS_MAIN_DONE) &
nice -n 19 $PY pairwise_neural.py --sys sft=longdoc/sys_seg_sft.jsonl.gz --sys rl=longdoc/sys_seg_rl2c.jsonl.gz --out longdoc/neural --cap 500 --gpu-gib 4.5 --batch 32 --metrics ssa_ref,afri_ref,afri_qe,ssa_qe > longdoc/neural.log 2>&1
echo done > CHAIN_LD_NEURAL_DONE
nice -n 19 $PY pairwise_stats.py --dir /workspace/jobs/e1/longdoc/seg_in --a sft --b rl --out /workspace/jobs/e1/longdoc/seg_summary --neural /workspace/jobs/e1/longdoc/neural --lang-groups /workspace/jobs/e1/lang_groups.json --min-n 5 > longdoc/stats_seg.log 2>&1
echo done > CHAIN3_DONE
wait
