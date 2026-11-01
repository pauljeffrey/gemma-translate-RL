#!/usr/bin/env bash
# The "$3 stage": AfriLongDoc check of the OLD intermediate RL checkpoints (agent C2, 2026-10-28). GPU session command list; nothing has been run on a GPU.
#   HF_TOKEN=... PY=python PY_VLLM=/path/venv_vllm/bin/python bash s4_stagecheck_driver.sh [fetch prep gen score report]
# Needs: read access of HF_TOKEN to ${HF_ORG}/AfriLongDoc (private), ${HF_ORG}/gemma3-270m-translate, ${HF_ORG}/gemma-3-270m-translate-rl2b, ${HF_ORG_LEGACY}/gemma-3-270m-translate-rl2c
#        (all private; the 7 checkpoint branches were verified to exist on 2026-10-28), zstd + tar, ~10 GB disk (7 x 1.07 GB fp32 checkpoints; --delete-after trims it to ~2 GB).
# Time/cost (ESTIMATED from the measured item counts, RTX PRO 6000 at USD 1.19/h): ~0.9 h [0.7-2.0] = USD 1.1 [0.8-2.4]; the E.4 guess was 2-2.5 h / USD 3. Only `gen` uses the GPU (~0.5 h of it).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="${W:-/workspace/stagecheck}"
PY="${PY:-python}"; PY_VLLM="${PY_VLLM:-$PY}"
export GEMMA_MT_PKG="${GEMMA_MT_PKG:-$(cd "$HERE/../.." && pwd)/gemma_mt}"
steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(fetch prep gen score report)
S="$PY $HERE/s4_afrilongdoc_stagecheck.py"
has() { local s; for s in "${steps[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
has fetch  && $S fetch  --out "$W"                                                    # AfriLongDoc core docs, NLLB refs of hau yor swh amh zul wol kea sag, 7 checkpoints
has prep   && $S prep   --out "$W"                                                    # 342 documents x 8 languages (b256 90, b512 100, b1024 89, b2048 63)
has gen    && $S gen    --out "$W" --engine vllm --python "$PY_VLLM" --gpu-util 0.85 --delete-after   # whole-document greedy, one vLLM engine per checkpoint
has score  && $S score  --out "$W"
has report && $S report --out "$W" --boot 400 --min-chrf 1.0 --min-fin 0.05            # W/stagecheck_report.md + stagecheck_results.json (verdict rule printed in the report)
echo "done: ${steps[*]}"
