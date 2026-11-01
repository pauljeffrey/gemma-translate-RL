#!/bin/bash
# tokswap_run_all.sh -- every command used to build the artefacts (on the box: /workspace/tokswap; CPU only).  Stages: A (tokenizers), B (checkpoints + sanity), C (llama.cpp converter test).
# Usage: [SKIP_SANITY=1] [VARIANTS="geez_N3000 ..."] bash tokswap_run_all.sh A|A2|B|C   (SKIP_SANITY=1: surgery + control dirs only; the sanity runs were then done on the laptop CPU because the box CPU quota is shared with the owner's job)   (A2 = tokenizer tests; it was split from A after the simulation check turned out slow)
set -euo pipefail
R=${R:-/workspace/tokswap}; S=$R/scripts; PY=$R/venv/bin/python; IN=$R/in; O=$R/out; L=$R/logs
T=$IN/tok_orig/tokenizer.json; TD=$IN/tok_orig
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-16} OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} TOKENIZERS_PARALLELISM=true
mkdir -p $O $L
SW="$PY $S/tokswap_swap_vocab.py"
case "$1" in
A)
  $SW classify --tok $T --out $O/free_list
  $SW train --tok $T --kind geez --text $IN/text/amh.train.txt $IN/text/tir.train.txt --n 3000 --max-chars 25200000 --min-count 5 --out $O/merges_geez_N3000.json
  $SW train --tok $T --kind fon  --text $IN/text/fon.wiki.txt $IN/text/fon.mt560.txt $IN/text/fon.fineweb2.txt --n 3000 --min-count 5 --out $O/merges_fon_N3000.json
  for n in 1000 3000; do
    $SW build --tok $T --geez $O/merges_geez_N3000.json --n-geez $n --out $O/tok_geez_N$n
    $SW build --tok $T --fon  $O/merges_fon_N3000.json  --n-fon  $n --out $O/tok_fon_N$n
    $SW spm --spm $IN/spm/tokenizer.model --swap $O/tok_geez_N$n
    $SW spm --spm $IN/spm/tokenizer.model --swap $O/tok_fon_N$n
  done
  ;;
A2)
  $SW simcheck --orig $T --swapped $O/tok_geez_N3000 --merges $O/merges_geez_N3000.json --kind geez --n 3000 --n-is-total --text $IN/text/tir.train.txt --max-chars 400000 | tee $L/simcheck_geez.txt
  $SW simcheck --orig $T --swapped $O/tok_fon_N3000  --merges $O/merges_fon_N3000.json  --kind fon  --n 3000 --n-is-total --text $IN/text/fon.fineweb2.txt --max-chars 400000 | tee $L/simcheck_fon.txt
  for v in geez_N1000 geez_N3000 fon_N1000 fon_N3000; do
    mode=${v%%_*}
    $PY $S/tokswap_eval.py identity --orig $T --swapped $O/tok_$v --flores $IN/flores --mode $mode --code-dir "$($PY -c 'import os;print(os.path.dirname(os.__file__))')" --out $O/identity_$v.json | tee $L/identity_$v.txt
  done
  $PY $S/tokswap_eval.py fertility --orig $T --flores $IN/flores --swapped geez_N1000=$O/tok_geez_N1000 geez_N3000=$O/tok_geez_N3000 fon_N1000=$O/tok_fon_N1000 fon_N3000=$O/tok_fon_N3000 \
      --langs eng fra amh tir fon hau yor swh kin ibo ewe bam lin wol twi --out $O/fertility.csv
  $PY $S/tokswap_spm_check.py --orig-json $T --orig-spm $IN/spm/tokenizer.model --flores $IN/flores --out $O/spm_check.json \
      --variants geez_N1000=$O/tok_geez_N1000 geez_N3000=$O/tok_geez_N3000 fon_N1000=$O/tok_fon_N1000 fon_N3000=$O/tok_fon_N3000
  ;;
B)
  # variant checkpoints from the SFT-final (fp32 on the Hub, cast to bf16), the sanity run for each, then the control (same weights, ORIGINAL tokenizer)
  M=$IN/sft_final
  # N=3000 variants: 300 sentences for the loss, 50 sources for the greedy runs; N=1000 variants: 100 / 25 (the CPU quota of the container is 19.2 cores
  # shared with the owner's generation job, so everything runs niced and with few threads)
  for v in ${VARIANTS:-geez_N3000 fon_N3000 geez_N1000 fon_N1000}; do
    mode=${v%%_*}
    if [ "${v#*_N}" = 3000 ]; then NL=300; NG=50; else NL=100; NG=25; fi
    [ -f $O/model_$v/model.safetensors ] || $SW surgery --model $M --swap $O/tok_$v --orig-tok $T --dtype bf16 --out $O/model_$v | tee $L/surgery_$v.txt
    if [ "$mode" = geez ]; then G="hau_Latn yor_Latn swh_Latn amh_Ethi tir_Ethi"; else G="hau_Latn yor_Latn swh_Latn amh_Ethi fon_Latn"; fi
    [ -n "${SKIP_SANITY:-}" ] || $PY $S/tokswap_sanity.py --orig-model $M --orig-tok $T --model $O/model_$v --swap-map $O/tok_$v/swap_map.json --flores $IN/flores --names $IN/text/language_names.json \
        --gen-langs $G --n-loss $NL --n-gen $NG --threads ${THREADS:-6} --out $O/sanity_$v.json 2>&1 | grep -v Loading | tee $L/sanity_$v.txt
    # control: identical weights, original tokenizer files (hard link of the weights: no extra disk)
    mkdir -p $O/control_$v; ln -f $O/model_$v/model.safetensors $O/control_$v/model.safetensors
    for f in config.json generation_config.json surgery.json; do cp $O/model_$v/$f $O/control_$v/; done
    for f in tokenizer.json tokenizer_config.json chat_template.jinja special_tokens_map.json; do [ -f $TD/$f ] && cp $TD/$f $O/control_$v/ || true; done
    if [ -z "${SKIP_SANITY:-}" ] && { [ "$v" = geez_N3000 ] || [ "$v" = fon_N3000 ]; }; then
      $PY $S/tokswap_sanity.py --orig-model $M --orig-tok $T --model $O/control_$v --swap-map $O/tok_$v/swap_map.json --flores $IN/flores --names $IN/text/language_names.json \
          --loss-langs amh_Ethi tir_Ethi hau_Latn yor_Latn swh_Latn fon_Latn --gen-langs hau_Latn amh_Ethi --n-loss 150 --n-gen 25 --threads ${THREADS:-6} --out $O/sanity_control_$v.json 2>&1 | grep -v Loading | tee $L/sanity_control_$v.txt
    fi
    touch $O/DONE_$v
  done
  ;;
C)
  # llama.cpp converter test (already installed at /workspace/llama.cpp; NOTHING is built or started, the running llama-server is not touched). Python converter only,
  # q8_0 output (about 0.3 GB), deleted after the check.
  LC=/workspace/llama.cpp; mkdir -p $R/gguf
  for v in geez_N3000 fon_N3000; do
    ln -f $O/tok_$v/tokenizer.model $O/model_$v/tokenizer.model 2>/dev/null || cp $O/tok_$v/tokenizer.model $O/model_$v/tokenizer.model
    PYTHONPATH=$LC/gguf-py $PY $LC/convert_hf_to_gguf.py $O/model_$v --outfile $R/gguf/$v.q8_0.gguf --outtype q8_0 2>&1 | tail -15 | tee $L/gguf_convert_$v.txt
    PYTHONPATH=$LC/gguf-py $PY $S/tokswap_gguf_check.py --gguf $R/gguf/$v.q8_0.gguf --swap $O/tok_$v --orig-spm $IN/spm/tokenizer.model --out $O/gguf_check_$v.json | tee $L/gguf_check_$v.txt
  done
  ;;
esac

# ------------------------------------------------------------------------------------------------------------------------------------------------
# LAPTOP steps (repo root; read token inside tokswap_fetch_laptop.py; W = gemma_mt/artifacts/tokswap_2026-10-25/_work)
#  inputs : .venv-clf/bin/python gemma_mt/scripts/tokswap_fetch_laptop.py --out $W/in           (+ --only fon for the Fon text);  tokenizer.model: google/gemma-3-270m-it/tokenizer.model
#  SFT-final fp32: huggingface_hub.snapshot_download("${HF_ORG}/gemma3-270m-translate", revision="step-27638")  (model.safetensors md5 4838a75997e09842e98fb8c50fe66975)
#  tokenizer tests (identical to the box run; the box run of stage A2 was stopped because the container CPU quota is 19.2 cores shared with the owner's job):
#     tokswap_eval.py identity|fertility, tokswap_spm_check.py, tokswap_hf_check.py   (arguments as in stage A/A2, flores = $W/in/flores)
#  checkpoints: tokswap_swap_vocab.py surgery --model $W/in/sft_final --swap <tok dir> --orig-tok $W/in/tok_orig/tokenizer.json --dtype bf16 --out gemma_mt/artifacts/tokswap_2026-10-25/model_<variant>
#     -> sha256 identical to the box run (reports/data/tokswap_2026-10-25/tests/box_model_sha256.txt)
#  control: same model.safetensors (APFS clone) + ORIGINAL tokenizer files;  sanity: _work/lane.sh <geez|fon>  (tokswap_sanity.py, n-loss 300 / n-gen 50 for N=3000, control 150/25, N=1000 100/25)
#  tables : .venv-clf/bin/python gemma_mt/scripts/tokswap_report_tables.py --dir <results dir> --names cpu_translate/language_names.json
