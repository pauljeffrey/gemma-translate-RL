#!/bin/bash
# Evaluation of the weak West-African group run (owner 2026-11-03): weak best and weak final against the mixed best (the start point), B1 best, stage A and SFT, same harness as before.
# 1 inference (3 models in parallel, FLORES+ devtest + NTREX + MAFAND greedy, then AfriScience dev/test sentences)  -> marker WEAK_EVAL_INFER_DONE (the science SFT may start)
# 2 AfriCOMET passes (small GPU share, may overlap the science SFT)  3 scoring with paired bootstrap (CPU)  -> marker WEAK_EVAL_SCORED
set -uo pipefail
R=/workspace/clean_rerun; S=/workspace/science_sft; E=/workspace/eval_385m; LOG=$S/weak_eval.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
declare -A MODEL=( [mixedbest]=$R/runs_385m/mixed/best [weakbest]=$R/runs_385m_weak/mixed/best [weakfinal]=$R/runs_385m_weak/mixed/final )
mk() {  # name model util comet?
  local n=$1 m=$2 u=$3 c=$4 tpl=run_m385.sh; [ "$c" = comet ] && tpl=run_m385_comet.sh
  sed -e "s#/workspace/eval_385m/model_m385#$m#g" -e "s#out_m385#out_wk_$n#g" -e "s#^export BENCHMARKS=.*#export BENCHMARKS=flores_plus,ntrex,mafand TOTAL_TIMEOUT=10800#" -e "s#GPU_UTIL=0.4#GPU_UTIL=$u#" -e "s#GPU_MIN_FREE_GIB=20#GPU_MIN_FREE_GIB=6#" $E/$tpl > $E/run_wk_${n}_$c.sh; chmod +x $E/run_wk_${n}_$c.sh
}
infer() {  # one model: generation, then AfriScience
  local n=$1 m=${MODEL[$1]}
  mk $n $m 0.25 gen
  (cd $E && ./run_wk_${n}_gen.sh > run_wk_${n}_gen.out 2>&1)
  (cd $E/gemma_mt_tree && HF_HOME=/workspace/clean_sft/hf_home PYTHONPATH=$E/gemma_mt_tree GEMMA_MT_TAG_MODE=single_token VLLM_CACHE_ROOT=$E/vllm_cache_af $E/venv_comet/bin/python scripts/eval_afriscience.py --model $m --tokenizer $m --name wk_$n --splits dev,test --levels sentence --vllm-python /venv/main/bin/python --vllm-gpu-util 0.15 --extra-libs $E/pylibs --out $E/out_afsci_wk_$n > $E/afsci_wk_$n.log 2>&1)
  say "inference done: $n"
}
for n in mixedbest weakbest weakfinal; do infer $n & sleep 90; done
wait
touch $S/WEAK_EVAL_INFER_DONE; say "WEAK_EVAL_INFER_DONE (inference of the three models finished)"
for n in mixedbest weakbest weakfinal; do mk $n ${MODEL[$n]} 0.08 comet; (cd $E && ./run_wk_${n}_comet.sh > run_wk_${n}_comet.out 2>&1); say "AfriCOMET pass done: $n"; done
# scoring: six systems; stage A and B1 best keep NTREX / MAFAND in their *_x dirs
cd $E; cat out_stageA/predictions.jsonl out_stageA_x/predictions.jsonl > pred_wk_stageA.jsonl; cat out_b1best/predictions.jsonl out_b1best_x/predictions.jsonl > pred_wk_b1best.jsonl
python3 - <<PY
import json
t8 = "mos fon kbp bam wol ewe twi dyu".split()
json.dump({l: ("target8" if l in t8 else "other") for l in json.load(open("$E/groups_family.json"))} | {l: "target8" for l in t8}, open("$E/groups_target8.json", "w"))
PY
export HF_HOME=/workspace/clean_sft/hf_home GEMMA_MT_TAG_MODE=single_token PYTHONPATH=$E/gemma_mt_tree
nice -n 15 /workspace/venv_gemma/bin/python gemma_mt_tree/scripts/eval_385m_score.py --system sft=out_m385/predictions.jsonl --system stageA=pred_wk_stageA.jsonl --system b1best=pred_wk_b1best.jsonl \
  --system mixedbest=out_wk_mixedbest/predictions.jsonl --system weakbest=out_wk_weakbest/predictions.jsonl --system weakfinal=out_wk_weakfinal/predictions.jsonl \
  --pair weakbest:mixedbest --pair weakfinal:mixedbest --pair weakbest:b1best --pair mixedbest:b1best --pair weakfinal:weakbest --groups groups_target8.json --subsets all --boot 1000 --processes 6 --out scores_weakgroup > scores_weakgroup.log 2>&1
touch $S/WEAK_EVAL_SCORED; say "WEAK_EVAL_SCORED"
