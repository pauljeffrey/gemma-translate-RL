#!/bin/bash
# Science SFT of the 385M model, end to end on the A100 box (arm 2). Called by science_sft_waiter.sh when the GPU is free; needs HF_TOKEN in the environment only for the final Hub push.
#   1 pick the start checkpoint (pick_start.py)  2 SFT (AdamW 3e-6, ~282 steps, early stop on the AfriScience dev loss)  3 interpolation alpha in {0.3 0.5 0.7 1.0} against the start (alpha 0 = start)
#   4 evaluation of every alpha: AfriScience dev+test (sentence), NTREX, MAFAND   5 choose alpha (choose_alpha.py)   6 AfriCOMET pass for the chosen model, push to the Hub branch science-sft
set -uo pipefail
R=/workspace/clean_rerun; S=/workspace/science_sft; E=/workspace/eval_385m; LOG=$S/pipeline.log; PYV=/workspace/clean_sft/venv_train/bin/python; TREE=$R/tree/gemma_mt
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
export GEMMA_MT_TAG_MODE=single_token HF_HOME=/workspace/clean_sft/hf_home PYTHONPATH=$TREE
cd $S
# 1 start checkpoint
CANDS="--candidate mixed=$R/runs_385m/mixed"
[ -f $R/runs_385m_weak/mixed/summary.json ] && CANDS="$CANDS --candidate weak=$R/runs_385m_weak/mixed"
python3 $S/pick_start.py $CANDS --out $S/start_choice.json | tail -n 40 >> $LOG
START=$(python3 -c "import json;print(json.load(open('$S/start_choice.json'))['model_dir'])")
say "start checkpoint: $START ($(python3 -c "import json;d=json.load(open('$S/start_choice.json'));print(d['chosen'],'-',d['reason'])"))"
# 2 SFT
mkdir -p /dev/shm/science_sft_models
say "SFT start"
nice -n 5 $PYV -m gemma_mt.train_sft --config $S/sft_science_385m.yaml model_name=$START >> $S/sft.log 2>&1
rc=$?; say "SFT finished rc=$rc"
[ -f /dev/shm/science_sft_run/last/trainer_state.pt ] || { say "no trainer_state.pt: stop"; exit 2; }
# 3 interpolation (alpha = weight of the SFT model; 0 = start)
ln -sfn $START /dev/shm/science_sft_models/a0
for al in 0.3 0.5 0.7 1.0; do
  $PYV $TREE/scripts/interpolate_weights.py --init $START --sft /dev/shm/science_sft_run/last/trainer_state.pt --alpha $al --out /dev/shm/science_sft_models/a$al >> $LOG 2>&1
done
# 4 evaluation (GPU alone, one model at a time)
for al in 0 0.3 0.5 0.7 1.0; do
  M=/dev/shm/science_sft_models/a$al; tag=a$al
  sed -e "s#/workspace/eval_385m/model_m385#$M#g" -e "s#out_m385#out_sci_${tag}_x#g" -e "s#^export BENCHMARKS=.*#export BENCHMARKS=ntrex,mafand TOTAL_TIMEOUT=7200#" $E/run_m385.sh > $E/run_sci_${tag}_x.sh
  chmod +x $E/run_sci_${tag}_x.sh
  say "eval NTREX/MAFAND generation $tag"; (cd $E && ./run_sci_${tag}_x.sh > run_sci_${tag}_x.out 2>&1)
  say "eval AfriScience dev+test $tag"
  (cd $E/gemma_mt_tree && HF_HOME=$HF_HOME PYTHONPATH=$E/gemma_mt_tree VLLM_CACHE_ROOT=$E/vllm_cache_af $E/venv_comet/bin/python scripts/eval_afriscience.py --model $M --tokenizer $M --name sci_$tag --splits dev,test --levels sentence --vllm-python /venv/main/bin/python --vllm-gpu-util 0.3 --extra-libs $E/pylibs --out $E/out_afsci_sci_$tag > $E/afsci_sci_$tag.log 2>&1)
done
# 5 choose alpha
python3 $S/choose_alpha.py --root $E --alphas 0 0.3 0.5 0.7 1.0 --out $S/choice_alpha.json >> $LOG 2>&1
AL=$(python3 -c "import json;print(json.load(open('$S/choice_alpha.json'))['chosen_alpha'])")
say "chosen alpha $AL"; echo $AL > $S/chosen_alpha; echo /dev/shm/science_sft_models/a$AL > $S/final_model_dir
# 6 AfriCOMET pass of the chosen model + Hub branch
sed -e "s#/workspace/eval_385m/model_m385#/dev/shm/science_sft_models/a$AL#g" -e "s#out_m385#out_sci_a${AL}_x#g" -e "s#^export BENCHMARKS=.*#export BENCHMARKS=ntrex,mafand TOTAL_TIMEOUT=7200#" $E/run_m385_comet.sh > $E/run_sci_chosen_comet.sh; chmod +x $E/run_sci_chosen_comet.sh
(cd $E && ./run_sci_chosen_comet.sh > run_sci_chosen_comet.out 2>&1)
if [ -n "${HF_TOKEN:-}" ]; then
  /workspace/venv_gemma/bin/python - <<PY >> $LOG 2>&1
from huggingface_hub import HfApi
a = HfApi(); R = "${HF_ORG}/gemma-0.4B-translate-rl"; a.create_branch(R, branch="science-sft", exist_ok=True)
print(a.upload_folder(repo_id=R, folder_path="/dev/shm/science_sft_models/a$AL", revision="science-sft", commit_message="science SFT (arm 2), alpha $AL"))
PY
fi
say "PIPELINE_DONE"; touch $S/PIPELINE_DONE
