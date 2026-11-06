#!/bin/bash
# RL launcher for the 385M grown model (owner 2026-11-01): the clean-rerun launcher (scripts/launch_clean_rerun.sh) with the 385M configs, the SINGLE-TOKEN tag mode and the 385M model / hub names.
#   ROOT=/workspace/clean_rerun scripts/launch_clean_rerun_385m.sh [args of launch_clean_rerun.sh]
# STAGE_PLAN=b256 (default here): the RL stages are preflight -> A (b64) -> B1 (b256) -> mixed (b64 + b256) -> final (science, b64 + b256); B2 / B3 are not run (owner 2026-11-02).
# OPTIONAL stage F2 (final_long, science, b1024 + b2048): OFF. STAGE_PLAN=b256+long (or EXTRA_STAGES=F2) allows `run F2`; decide with clean_rerun_tools.py f2-precheck / f2-decide (reports/rl_stage_plan_385m_b256_2026-11-02.md section 6).
# Before the first run: scripts/clean_rerun_385m_prepare_assets.sh (prompt pool with the 8 new languages, shaping_clean_v2.json, renamed tokenizer).
# NOTE (launcher defaults of the 270M plan that matter on other cards): the launcher refuses a GPU that is not an RTX PRO 6000 unless ALLOW_GPU=1 and wants MIN_FREE_GIB (70) free GPU memory;
# on an 80 GB A100 set ALLOW_GPU=1 and check MIN_FREE_GIB against reports/rl_gpu_ram_385m_2026-11-01.md.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="${TREE:-$(cd "$HERE/.." && pwd)}"
ROOT="${ROOT:-/workspace/clean_rerun}"
export GEMMA_MT_TAG_MODE="${GEMMA_MT_TAG_MODE:-single_token}"
[ "$GEMMA_MT_TAG_MODE" = single_token ] || { echo "refusing to run the 385M plan with GEMMA_MT_TAG_MODE=$GEMMA_MT_TAG_MODE (the SFT data and the model use single-token tags)"; exit 2; }
export CFGDIR="${CFGDIR:-$TREE/configs/clean_rerun_385m}"
export RUNS="${RUNS:-$ROOT/runs_385m}"
export SFT_MODEL="${SFT_MODEL:-$ROOT/models/g15_385m_sft_final}"
export SCI_MODEL="${SCI_MODEL:-$ROOT/models/sci_sft_start_385m}"
export STAGE_PLAN="${STAGE_PLAN:-b256}"   # owner 2026-11-02: RL context capped at b256: stages preflight A B1 mixed final (no B2 / B3)
export LADDER_START="${LADDER_START:-1}"   # 385M: 39 layers double the activation memory of the 270M model; start at 24,576 tokens per micro-batch (reports/rl_gpu_ram_385m_2026-11-01.md)
export POOL_V3="${POOL_V3:-$ROOT/prompts/clean_v3_385m}"
export HUB_PREFIX="${HUB_PREFIX-${HF_ORG}/gemma-mt-385m-rl}"
export HUB_FROM_YAML=1      # repo names / branches from the yamls: ${HF_ORG}/gemma-0.4B-translate-rl (stages A, B1, mixed) and gemma-0.4B-science-rl (final, F2)
PY="${PY:-/workspace/venv_gemma/bin/python}"
"$PY" - <<PYEOF || { echo "tag-mode check failed"; exit 2; }
import os, sys
sys.path.insert(0, "$TREE")
from gemma_mt import format as fmt, tags
assert fmt.get_tag_mode() == "single_token", fmt.get_tag_mode()
assert tags.tag_token("yor") == "<unused0>" and tags.tag_token("adh") == "<unused6216>"
print("tag mode single_token OK")
PYEOF
for m in "$SFT_MODEL"; do [ -f "$m/tokenizer.json" ] && grep -q '"<yor>"' "$m/tokenizer.json" || echo "WARNING: $m/tokenizer.json does not contain the renamed tags (<yor>): copy $ROOT/assets/tokenizer_gemma3_tags/* into the exported model dir"; done
exec bash "$HERE/launch_clean_rerun.sh" "$@"
