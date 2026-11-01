#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Builds the S2 START checkpoint of the clean rerun: google/gemma-3-270m-it + 3,000 Ge'ez tokens (ids 256001-259000 of the unused tail 256001-262143), each new embedding row =
# mean (float32) of the ORIGINAL embedding rows of the sub-tokens the original tokenizer produces for the token's string. The tokenizer (61 Ethiopic letters + 2,939 merges appended)
# is the one built and tested on 2026-10-25 (reports/token_swap_build_2026-10-25.md); NOTHING is retrained here, only the surgery is re-run on the BASE model
# (the checkpoints in gemma_mt/artifacts/tokswap_2026-10-25 were made from the old SFT-final model and must NOT be used for S2).
#
#   bash scripts/upcycle_s2_start.sh                          # run from anywhere; needs HF_TOKEN (read) in the environment for the gated Gemma repo, never echoed
#   BASE=/path/to/gemma-3-270m-it OUT=ckpt/gemma3-270m-it-geez3000 bash scripts/upcycle_s2_start.sh
# Output: $OUT (model.safetensors fp32, config.json, tokenizer files, surgery.json) and the check report $OUT/s2_start_check.json. Exit code != 0 if any check fails.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${PY:-python}
W=${W:-$PWD/s2_start_work}
BASE=${BASE:-$W/gemma-3-270m-it}
OUT=${OUT:-$PWD/ckpt/gemma3-270m-it-geez3000}
SRC=${SWAP_SRC:-$REPO/reports/data/tokswap_2026-10-25/tokenizers/geez_N3000}
EXPECT_TOK_SHA=6772d90b0914f94091e9f33a38616c7d129d09c5bd7a509437bbabcdb54dce8b      # reports/data/tokswap_2026-10-25/SHA256SUMS_artefacts.txt (uncompressed geez_N3000/tokenizer.json)
mkdir -p "$W"
if [ ! -f "$BASE/model.safetensors" ]; then
  $PY - <<PYEOF
from huggingface_hub import snapshot_download      # reads HF_TOKEN from the environment
snapshot_download("google/gemma-3-270m-it", local_dir="$BASE", allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja"])
PYEOF
fi
SWAP="$W/tok_geez_N3000"
mkdir -p "$SWAP"
cp "$SRC/swap_map.json" "$SRC/tokenizer_config.json" "$SWAP/"
[ -f "$SRC/tokenizer.model" ] && cp "$SRC/tokenizer.model" "$SWAP/" || true
if [ -f "$SRC/tokenizer.json.gz" ]; then gunzip -c "$SRC/tokenizer.json.gz" > "$SWAP/tokenizer.json"; else cp "$SRC/tokenizer.json" "$SWAP/tokenizer.json"; fi
for f in chat_template.jinja special_tokens_map.json; do [ -f "$BASE/$f" ] && cp "$BASE/$f" "$SWAP/" || true; done      # as `build` did: copied from the original tokenizer folder
GOT=$(shasum -a 256 "$SWAP/tokenizer.json" | cut -d' ' -f1)
[ "$GOT" = "$EXPECT_TOK_SHA" ] || { echo "swapped tokenizer.json sha256 $GOT != expected $EXPECT_TOK_SHA" >&2; exit 2; }
$PY "$HERE/tokswap_swap_vocab.py" surgery --model "$BASE" --swap "$SWAP" --orig-tok "$BASE/tokenizer.json" --dtype fp32 --out "$OUT"
$PY "$HERE/upcycle_s2_start_check.py" --base "$BASE" --start "$OUT" --swap-map "$SWAP/swap_map.json" --report "$OUT/s2_start_check.json"
