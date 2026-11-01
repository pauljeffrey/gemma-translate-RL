#!/bin/bash
# usage: lane.sh <geez|fon>   (laptop CPU sanity lane)
cd "/path/to/repo"
W="$PWD/gemma_mt/artifacts/tokswap_2026-10-25/_work"; A="$PWD/gemma_mt/artifacts/tokswap_2026-10-25"; PY=.venv-clf/bin/python
k=$1
if [ $k = geez ]; then G="hau_Latn yor_Latn swh_Latn amh_Ethi tir_Ethi"; else G="hau_Latn yor_Latn swh_Latn amh_Ethi fon_Latn"; fi
S() { # name model_dir variant loss_langs gen_langs nl ng
  nice -n 5 $PY gemma_mt/scripts/tokswap_sanity.py --orig-model "$W/in/sft_final" --orig-tok "$W/in/tok_orig/tokenizer.json" --model "$2" --swap-map "$W/out/tok_$3/swap_map.json" \
    --flores "$W/in/flores" --names "$W/in/text/language_names.json" --loss-langs $4 --gen-langs $5 --n-loss $6 --n-gen $7 --threads 5 --out "$W/res/sanity_$1.json" > "$W/res/sanity_$1.txt" 2>&1
}
LL="amh_Ethi tir_Ethi hau_Latn yor_Latn swh_Latn kin_Latn ibo_Latn fon_Latn"
S ${k}_N3000 "$A/model_${k}_N3000" ${k}_N3000 "$LL" "$G" 300 50
S control_${k}_N3000 "$A/control_${k}_N3000" ${k}_N3000 "amh_Ethi tir_Ethi hau_Latn yor_Latn swh_Latn fon_Latn" "hau_Latn amh_Ethi" 150 25
S ${k}_N1000 "$A/model_${k}_N1000" ${k}_N1000 "$LL" "$G" 100 25
echo LANE_DONE > "$W/res/lane_$k.done"
