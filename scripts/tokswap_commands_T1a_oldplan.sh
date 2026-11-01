#!/bin/bash
# NOTE (2026-10-26): this file documents the OTHER T1 session (T1a, old plan: re-used ids = leaf CJK tokens that are REMOVED from the vocabulary, N up to 8,000, Fon up to 5,000).
# It is NOT the main result of reports/token_swap_build_2026-10-25.md (main = tokswap_run_all.sh: re-use of the 6,143 <unused> ids, nothing removed).
# tokswap_commands_T1.sh -- the exact command sequence that produced the artefacts of reports/token_swap_build_2026-10-25.md (agent T1, 2026-10-25/26).
# Box: vast.ai, CPU only, /workspace/tokswap.  Laptop: repo root.  Not meant to be run blindly: it documents the order and the arguments.
# Every artefact below is deterministic given the inputs (inputs: sha256 in reports/data/tokswap_2026-10-25/SHA256SUMS.txt).

# ------------------------------------------------------------------ 0. laptop: inputs (READ token from .env is loaded INSIDE the script, never printed; no Hub write)
#   .venv-clf/bin/python gemma_mt/scripts/tokswap_fetch_laptop.py --out $W/in            # models, tokenizer, FLORES+ (all langs, dev+devtest), training text
#   .venv-clf/bin/python gemma_mt/scripts/tokswap_fetch_laptop.py --out $W/in --only fon # Fon text rebuilt WITHOUT NFC + JW spacing repair (see report)
#   SFT-final fp32 -> bf16 file (the reference 'original' checkpoint; the Hub file is fp32):  safetensors load_file -> .to(torch.bfloat16)
#   trimmed text (<= 20 M characters per file) gzipped and uploaded with `rsync -a --partial --inplace` in a retry loop (the link resets on long transfers)

# ------------------------------------------------------------------ 1. box: python = /workspace/tokswap/venv/bin/python (venv --system-site-packages over /venv/main, + sacrebleu)
PY=/workspace/tokswap/venv/bin/python; cd /workspace/tokswap; T=in/tok_orig/tokenizer.json
# 1a free list
$PY swap_vocab.py classify --tok $T --out out/free
# 1b merges (continuation BPE on the original segmentation; amh+tir balanced to about 20 M characters each; Fon: Wikipedia fon + mt560 target side)
$PY swap_vocab.py train --tok $T --kind geez --n 8000 --min-count 3 \
    --text in/text/amh.wiki.txt in/text/amh.ft.txt@12000000 in/text/tir.wiki.txt in/text/tir.ft.txt@19700000 --out out/merges_geez_amh_tir.json
$PY swap_vocab.py train --tok $T --kind fon --n 8000 --min-count 3 --text in/text/fon.wiki.txt in/text/fon.mt560.txt --out out/merges_fon.json   # stops at 5,278 merges (count 3)
# 1c tokenizers (prefix of the merge list = N; ids = free_order[0:8000] for Ge'ez, free_order[8000:16000] for Fon)
for n in 1000 3000 8000; do $PY swap_vocab.py build --tok $T --free out/free/free_pool.json --geez out/merges_geez_amh_tir.json --n-geez $n --out out/tok/geez_N$n; done
for n in 1000 3000 5000; do $PY swap_vocab.py build --tok $T --free out/free/free_pool.json --fon out/merges_fon.json --n-fon $n --out out/tok/fon_N$n; done
$PY swap_vocab.py build --tok $T --free out/free/free_pool.json --geez out/merges_geez_amh_tir.json --n-geez 3000 --fon out/merges_fon.json --n-fon 1000 --out out/tok/both_g3000_f1000
# 1d tests of the tokenizers
$PY swap_vocab.py simcheck --orig $T --swapped out/tok/geez_N3000 --merges out/merges_geez_amh_tir.json --kind geez --n 3000 --text in/text/amh.ft.txt --max-chars 1500000
$PY tokswap_eval.py identity --orig $T --swapped out/tok/<variant> --flores in/flores --mode geez|fon|both --code-dir /usr/lib/python3.12 --out out/tests/identity_<variant>.json
$PY tokswap_eval.py fertility --orig $T --flores in/flores --langs eng fra amh tir fon hau yor swh kin ibo ewe bam lin wol twi --swapped geez_N1000=out/tok/geez_N1000 ... --out out/tests/fertility.csv
$PY tokswap_hf_check.py --orig in/tok_orig --swapped out/tok/geez_N3000 out/tok/fon_N3000 out/tok/both_g3000_f1000 --flores in/flores --out out/tests/hf_check.json
# 1e checkpoints (bf16) and the control (identical weights, ORIGINAL tokenizer files)
$PY swap_vocab.py surgery --model in/sft_final_bf16 --swap out/tok/<variant> --orig-tok $T --dtype bf16 --out out/model/<variant>
$PY swap_vocab.py surgery --model in/sft_final_bf16 --swap out/tok/<variant> --orig-tok $T --dtype bf16 --control --orig-tok-dir in/tok_orig --out out/model/control_<variant>
# 1f model sanity (CPU, fp32 compute from the bf16 weights)
$PY tokswap_sanity_fast.py --orig-model in/sft_final_bf16 --orig-tok $T --model out/model/<variant> --swap-map out/tok/<variant>/swap_map.json --flores in/flores \
    --names in/text/language_names.json --n-loss 300 --loss-langs amh_Ethi tir_Ethi hau_Latn yor_Latn swh_Latn kin_Latn ibo_Latn fon_Latn --gen-langs hau_Latn yor_Latn swh_Latn amh_Ethi tir_Ethi --out out/tests/sanity_<variant>.json
# ------------------------------------------------------------------ 2. laptop: tables
#   .venv-clf/bin/python gemma_mt/scripts/tokswap_report_tables.py --dir reports/data/tokswap_2026-10-25/tests --names cpu_translate/language_names.json
