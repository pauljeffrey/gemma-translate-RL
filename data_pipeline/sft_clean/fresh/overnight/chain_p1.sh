#!/bin/bash
# PHASE 1 (coordinator order 2026-11-04): the moment the science RL has ended (summary.json, no trainer), run the cheap paper controls next to the science post-RL evaluation:
# E-01 loop guard (guard chosen on the FLORES+ dev held-out half, applied to SFT and RL), E-09 engine parity (vLLM fp32 vs HF fp32), E-13 old 270M models in the same harness.
# Every step has a timeout; a failed step is noted and the chain goes on. CONTROLS_DONE releases the long-document RL waiter.
source /workspace/science_sft/overnight/common.sh
export GEMMA_MT_TAG_MODE=single_token HF_HOME=/workspace/clean_sft/hf_home
SFT=$R/models/g15_385m_sft_final; RL=$R/runs_385m/mixed/best  # = Hub ${HF_ORG}/gemma-0.4B-translate-rl branch main: model.safetensors sha256 8cb0dd5a... identical (checked 2026-11-04)
say "P1 chain started; waiting for the science RL to end"
while [ ! -f $R/runs_385m_sci/final/summary.json ] || pgrep -f "[g]emma_mt.rl.train_rl" >/dev/null; do sleep 20; done
say "science RL ended; P1 controls start (post-RL evaluation runs next to them)"
sleep 60
for mn in sft:$SFT rl:$RL; do
  n=${mn%%:*}; m=${mn#*:}
  step E01_grid_$n 3600 bash -c "cd $E/gemma_mt_tree && PYTHONPATH=$E/gemma_mt_tree VLLM_CACHE_ROOT=$E/vllm_cache_guard /venv/main/bin/python $O/guard_select.py --model $n=$m --dev $R/assets/dev/bench_dev_hsel.jsonl --out $O/guard_grid_$n.json --gpu-util 0.2"
done
step E01_choose 120 python3 $O/guard_choose.py $O/guard_grid_sft.json $O/guard_grid_rl.json $O/guard_dev.json
if [ ! -f $O/guard_dev.json ]; then
  say "guard selection failed: falling back to the report's starting point, frequency penalty 0.05 (recorded as a fallback, NOT selected on dev)"
  python3 -c "import json;json.dump({'chosen':{'name':'freq005_FALLBACK','frequency_penalty':0.05,'repetition_penalty':1.0},'note':'selection step failed; fallback to reports/longdoc_eval_final_model_2026-10-21.md starting point'},open('$O/guard_dev.json','w'))"
fi
say "guard chosen: $(python3 -c "import json;print(json.dumps(json.load(open('$O/guard_dev.json'))['chosen']))")"
step E01_sft 7200 bash $O/run_guard_eval.sh sft $SFT
step E01_rl 7200 bash $O/run_guard_eval.sh rl $RL
# E-09: four engines x two models on the 240-sentence parity set
cd $E; mkdir -p $O/parity
for mn in sft:$SFT rl:$RL; do
  n=${mn%%:*}; m=${mn#*:}
  for mode in vllm32 vllm; do step E09_${n}_$mode 900 env PYTHONPATH=$E/gemma_mt_tree VLLM_CACHE_ROOT=$E/vllm_cache_guard /venv/main/bin/python $O/parity3.py $mode $m $O/parity/${n}_$mode.json; done
  for mode in hf32 hf; do step E09_${n}_$mode 1200 env PYTHONPATH=$E/gemma_mt_tree /workspace/clean_sft/venv_train/bin/python $O/parity3.py $mode $m $O/parity/${n}_$mode.json; done
done
step E09_compare 300 /workspace/clean_sft/venv_train/bin/python $O/parity_compare.py $O/parity_summary.json sft=$O/parity/sft rl=$O/parity/rl
# E-13: old 270M models in the 385M harness
step E13_sft 5400 bash $O/run_old_eval.sh sft ${HF_ORG}/gemma3-270m-translate
step E13_rl2c 5400 bash $O/run_old_eval.sh rl2c /dev/shm/model_rl2c
# E-20 (coordinator 2026-11-04): the B1 FINAL weights (Hub branch stage-b1-final of the public repo), never evaluated
step E20_download 1500 env HF_HOME=/workspace/clean_sft/hf_home /venv/main/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('${HF_ORG}/gemma-0.4B-translate-rl', revision='stage-b1-final', local_dir='/dev/shm/model_b1final')"
[ -f /dev/shm/model_b1final/config.json ] && step E20_eval 5400 bash $O/run_plain_eval.sh b1final /dev/shm/model_b1final || say "E-20 skipped: no /dev/shm/model_b1final/config.json"
touch $O/CONTROLS_DONE; say "P1 DONE (CONTROLS_DONE): long-document RL waiter released"
