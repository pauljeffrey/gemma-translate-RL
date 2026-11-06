#!/bin/bash
# E-14 evaluation = the E-16 evaluation (AfriScience sentence dev/test, NTREX/MAFAND, documents whole + chunked, neural passes) on E-14 final (and best when it differs), plus the same sentence-level
# AfriScience run for the B1 final (b256 RL; never run on AfriScience before) and the full benchmark harness (FLORES+ 54 / NTREX / MAFAND with held-out AfriCOMET) so that E-14 can be compared with B1 final
# (E-20) and mixed best on identical sentences.
E=/workspace/eval_385m; S=/workspace/science_sft; E2=/workspace/e2405; L=$E2/log_e14.txt
say() { echo "$(date -u +%FT%TZ) [e14-eval] $*" | tee -a $L; }
export HF_HOME=/workspace/clean_sft/hf_home GEMMA_MT_TAG_MODE=single_token
CORPUS=$(ls -d /workspace/clean_sft/hf_home/hub/datasets--dsfsi--afriscience_mt/snapshots/*/corpus)
M14=/dev/shm/e14/final; B1=/dev/shm/rl_B1_ckpt/final
afsci() {  # name model
  say "AfriScience sentence level $1"
  (cd $E/gemma_mt_tree && PYTHONPATH=$E/gemma_mt_tree VLLM_CACHE_ROOT=$E/vllm_cache_af $E/venv_comet/bin/python scripts/eval_afriscience.py --model $2 --tokenizer $2 --name $1 --splits dev,test --levels sentence --vllm-python /venv/main/bin/python --vllm-gpu-util 0.3 --extra-libs $E/pylibs --out $E/out_afsci_$1 --corpus-dir $CORPUS > $E/afsci_$1.log 2>&1)
}
afsci e14_final $M14
afsci b1final_af $B1
BEST_DIFF=0; [ -f /dev/shm/e14/best/model.safetensors ] && ! cmp -s /dev/shm/e14/best/model.safetensors /dev/shm/e14/final/model.safetensors && BEST_DIFF=1
[ $BEST_DIFF = 1 ] && afsci e14_best /dev/shm/e14/best
say "full benchmark harness e14_final (FLORES+ 54, NTREX, MAFAND, AfriCOMET)"; bash $E2/e_eval20.sh e14_final $M14 0.5 full
say "documents e14_final (whole + chunked, 168 test documents)"
mkdir -p $E/out_doc_e14_final
(cd $E/gemma_mt_tree && PYTHONPATH=$E/gemma_mt_tree VLLM_CACHE_ROOT=$E/vllm_cache_doc $E/venv_comet/bin/python scripts/eval_afriscience_documents.py --model $M14 --tokenizer $M14 --name e14_final --out $E/out_doc_e14_final --vllm-python /venv/main/bin/python --vllm-gpu-util 0.25 --vllm-max-num-seqs 64 --max-new-doc 4096 --extra-libs $E/pylibs --corpus-dir $CORPUS > $E/doc_e14_final.log 2>&1)
say "neural passes"
export PYTHONPATH=$E/pylibs_none:$E/gemma_mt_tree
$E/venv_comet/bin/python $E2/e14_neural.py $E2/e14_neural.json 8 > $E2/e14_neural.log 2>&1
$E/venv_comet/bin/python $E/gemma_mt_tree/scripts/eval_afriscience_documents_neural.py --pred $E/out_doc_e14_final/documents_predictions.json $E/out_doc_sci_rl_final/documents_predictions.json $E/out_doc_a0/documents_predictions.json $E/out_doc_a05/documents_predictions.json --names e14_final sci_rl_final sci_a0 sci_a0.5 --out $E2/e14_doc_neural.json > $E2/e14_doc_neural.log 2>&1
$E/venv_comet/bin/python $E2/e14_pairs.py $E2/e14_pairs.json 8 > $E2/e14_pairs.log 2>&1
touch $E2/E14_EVAL_DONE; say "E14_EVAL_DONE"
