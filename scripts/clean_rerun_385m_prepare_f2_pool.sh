#!/bin/bash
# Prompt pool of the OPTIONAL stage F2 (final_long, owner 2026-11-02): $ROOT/prompts/clean_sci_v2_long = the clean science pool (b64..b1024, decontaminated) + zz_b2048/<iso>.parquet.
#   ROOT=/workspace/clean_rerun B2048_CLEAN_DIR=<dir with <iso>.parquet> scripts/clean_rerun_385m_prepare_f2_pool.sh
# WHY: reports/clean_rl_science_data_2026-10-28.md section 5 dropped the 26,094 b2048 rows of the ORIGINAL sci_v2 (jobs/r7/assets2/rl_prompts_sci_v2/<iso>.parquet) WITHOUT scanning them, so clean_sci_v2 has
#   b64..b1024 only. F2 needs b2048 science prompts, and F2 is judged on the AfriScience-MT DEV papers, so those rows MUST first go through the same decontamination as every other clean pool
#   (data_pipeline/sft_clean/decon.py: R1 exact sentence, R2 contained sentence, R3 shared word 8-gram against every protected set incl. afriscience dev / test, afrilongdoc, smol, afridocmt ...; a prompt with
#   any hit is removed, never repaired). The scan is NOT done by this script. B2048_CLEAN_DIR = its output: one parquet per language, the columns of the original sci_v2 files (`domain`, `src_afr_tokens`, ...).
# CPU only. The clean pool is symlinked, not copied.
set -euo pipefail
ROOT="${ROOT:-/workspace/clean_rerun}"; SRC="$ROOT/prompts/clean_sci_v2"; DST="${F2_POOL:-$ROOT/prompts/clean_sci_v2_long}"; B="${B2048_CLEAN_DIR:?set B2048_CLEAN_DIR to the decon-scanned b2048 science rows (one <iso>.parquet per language)}"
PY="${PY:-/workspace/venv_gemma/bin/python}"
[ -d "$SRC" ] || { echo "missing $SRC"; exit 2; }
ls "$B"/*.parquet >/dev/null 2>&1 || { echo "no parquet files in $B"; exit 2; }
mkdir -p "$DST/zz_b2048"
for f in "$SRC"/*.parquet; do ln -sfn "$f" "$DST/$(basename "$f")"; done
for f in "$B"/*.parquet; do ln -sfn "$f" "$DST/zz_b2048/$(basename "$f")"; done
"$PY" - "$DST" <<'PYEOF'
import glob, os, sys
import pandas as pd
d = sys.argv[1]
bad, tot, per = [], 0, {}
for f in sorted(glob.glob(os.path.join(d, "zz_b2048", "*.parquet"))):
    df = pd.read_parquet(f, columns=["domain", "src_afr_tokens"])
    iso = os.path.basename(f)[:-8]
    n = int(((df.domain == "science") & df.src_afr_tokens.between(1025, 2048)).sum()); per[iso] = n; tot += n
    if len(df) != n: bad.append(f"{iso}: {len(df) - n} rows are not science or not in 1,025..2,048 AfriCOMET tokens")
print(f"zz_b2048: {len(per)} languages, {tot} science b2048 rows, min per language {min(per.values())}, max {max(per.values())}")
if bad: print("PROBLEMS:\n  " + "\n  ".join(bad)); sys.exit(1)
if min(per.values()) < 50: print("WARNING: a language has < 50 b2048 science rows: F2 draws ~15 b2048 prompts per weight-1 language"); 
PYEOF
echo "F2 pool ready: $DST"
