#!/bin/bash
# Assets of the 385M RL plan on the RL box (run once, CPU only): the general prompt pool with the 8 new consensus languages, shaping_clean_v2.json, the renamed tokenizer.
#   ROOT=/workspace/clean_rerun scripts/clean_rerun_385m_prepare_assets.sh
# Needs next to the tree: artifacts/rl_prompts_new/*.parquet (8 files), calibration/clean_2026-10-28/shaping_clean_v2.json, artifacts/tokenizer_gemma3_tags/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TREE="$(cd "$HERE/.." && pwd)"; ROOT="${ROOT:-/workspace/clean_rerun}"
POOL_OLD="$ROOT/prompts/clean_v3"; POOL_NEW="$ROOT/prompts/clean_v3_385m"
[ -d "$POOL_OLD" ] || { echo "missing $POOL_OLD"; exit 1; }
mkdir -p "$POOL_NEW"
for f in "$POOL_OLD"/*.parquet; do ln -sfn "$f" "$POOL_NEW/$(basename "$f")"; done        # symlinks: the old pool is not copied (disk)
for f in "$TREE"/artifacts/rl_prompts_new/*.parquet; do cp -f "$f" "$POOL_NEW/"; done
n_old=$(ls "$POOL_OLD"/*.parquet | wc -l); n_new=$(ls "$POOL_NEW"/*.parquet | wc -l)
echo "prompt pool $POOL_NEW: $n_new files ($n_old from clean_v3 + $((n_new - n_old)) new)"
mkdir -p "$ROOT/assets/calibration" "$ROOT/assets"
cp -f "$TREE/calibration/clean_2026-10-28/shaping_clean_v2.json" "$ROOT/assets/calibration/shaping_clean_v2.json"
rm -rf "$ROOT/assets/tokenizer_gemma3_tags"; cp -r "$TREE/artifacts/tokenizer_gemma3_tags" "$ROOT/assets/tokenizer_gemma3_tags"
echo "done. Remember: the exported SFT-final model dir ($ROOT/models/g15_385m_sft_final) needs the renamed tokenizer files from $ROOT/assets/tokenizer_gemma3_tags"
