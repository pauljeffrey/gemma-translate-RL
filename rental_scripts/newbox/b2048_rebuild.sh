#!/bin/bash
# rebuild the b2048 prompts (rl_prompts_v3) on the new box: same shards/seeds as the lost build; CPU only, nice 19
cd /workspace/sft_build; source env.sh
export PYTHONPATH=/workspace/sft_build/gemma_mt_2c:/workspace/sft_build/pylibs CUDA_VISIBLE_DEVICES=""
PY=/workspace/venv_gemma/bin/python; B=b2048_build; mkdir -p $B
say() { echo "$(date -u +%T) [b2048] $*" | tee -a logs/b2048_rebuild.log; }
S=gemma_mt_2c/scripts/build_rl_prompts_b2048.py
say start
nice -n 19 $PY $S windows --source fineweb --shard 3 --out $B/win_fineweb.parquet --workers 6 > logs/b2048_fineweb.log 2>&1 &
nice -n 19 $PY $S windows --source wiki --shard 12 --out $B/win_wiki.parquet --workers 3 > logs/b2048_wiki.log 2>&1 &
nice -n 19 $PY $S hashes --v2 gemma/rl_prompts_v2 --out $B/v2_hashes.npy --workers 3 > logs/b2048_hashes.log 2>&1 &
wait; say "windows+hashes done"
nice -n 19 $PY $S assemble --windows $B/win_fineweb.parquet $B/win_wiki.parquet --v2 gemma/rl_prompts_v2 --hashes $B/v2_hashes.npy --out gemma/rl_prompts_v3 --per-language 600 --doc-ids gemma/rl_prompts_b2048_doc_ids.txt > logs/b2048_assemble.log 2>&1
say "assemble rc=$?"; ls gemma/rl_prompts_v3 | wc -l; ls gemma/rl_prompts_v3/zz_b2048 2>/dev/null | wc -l
touch $B/DONE
