#!/usr/bin/env bash
# SMOKE test of the whole science-stage rebuild chain on CPU (agent C2, 2026-10-28). Proves the PLUMBING only; every number it produces is SMOKE, none is a result.
#   * model: google/gemma-3-270m-it (tokenizer + one real CPU generation check); the long-document / anchor generations used downstream are the FAKE engine (target-language
#     sentences of the replay file by length; not translations) because the base model cannot translate;
#   * QE: the 'fake' scorer of gemma_mt.rl.comet (no AfriCOMET / SSA-COMET on CPU here); calibration = the OLD 2026-10-19 files, used ONLY to exercise the code path;
#   * counts: 8 AfriScience papers, a 38k-pair sample of the clean SFT pairs -> windows -> packs (the dry run on UNGATED windows), 4 languages for long documents, 2 for anchors.
#   bash s4_smoke.sh      (env: HF_TOKEN, SMOKE_WORK, BASE, PREFIX="taskset -c 16-17 nice -n 19 ionice -c3", PY, DATA, PROTECTED_DIR, SRC_PAIRS, GLOTLID, TOK_AFRO, CAL_DIR)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="${SMOKE_WORK:-/workspace/s4/work/smoke2}"
PY="${PY:-/workspace/work/venv/bin/python}"
PREFIX="${PREFIX:-taskset -c 16-17 nice -n 19 ionice -c3}"
BASE="${BASE:?path of the google/gemma-3-270m-it snapshot (weights + tokenizer)}"
DATA="${DATA:-/workspace/s4/work/data}"
PROT="${PROTECTED_DIR:-/workspace/work/protected}"
SRC_PAIRS="${SRC_PAIRS:-/workspace/work/out/final /workspace/work/out/specialist}"
GLOTLID="${GLOTLID:?GlotLID model_v3.bin}"
TOK_AFRO="${TOK_AFRO:-/workspace/work/tok_afro/tokenizer.json}"
TOK_GEMMA="${TOK_GEMMA:-/workspace/work/tok_google/tokenizer.json}"
CAL_DIR="${CAL_DIR:-$W/cal}"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
IDX="--index $PROT/protected_index.npz --manifest $PROT/protected_manifest.json"
CALARGS="--calibration $CAL_DIR/shaping_SMOKE_old_v8.json --anchors $CAL_DIR/anchors_SMOKE_old_v1.json --glotlid $GLOTLID --scorer $CAL_DIR/scorer_fake.json"
export GEMMA_MT_PKG="${GEMMA_MT_PKG:-/workspace/work/gemma_mt_pkg}" CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false
r() { echo "[$(date +%T)] SMOKE: $*"; $PREFIX "$PY" "$@"; }
mkdir -p "$W"; cd "$HERE"
t0=$(date +%s)
# ---- SMOKE prerequisites (idempotent): the pools of 6 languages, the OLD calibration files (code-path exercise only), the fake scorer spec, the checker controls
if [ ! -f "$DATA/clean_v3/kea.parquet" ] || [ ! -f "$DATA/clean_sci_v2/kea.parquet" ] || [ ! -f "$DATA/afriscience_clean_train/afriscience_clean_train.parquet" ]; then
  r s4_fetch.py --out "$DATA" prompts --langs yor,hau,amh,zul,swh,kea --pools clean_v3,clean_sci_v2 --afs
fi
mkdir -p "$CAL_DIR"
[ -f "$CAL_DIR/shaping_SMOKE_old_v8.json" ] || cp "${CAL_SRC:-$REPO_ROOT/gemma_mt/calibration}/shaping_flores_dev_v8.json" "$CAL_DIR/shaping_SMOKE_old_v8.json"
[ -f "$CAL_DIR/anchors_SMOKE_old_v1.json" ] || cp "${CAL_SRC:-$REPO_ROOT/gemma_mt/calibration}/anchors_v1.json" "$CAL_DIR/anchors_SMOKE_old_v1.json"
echo '{"type": "fake"}' > "$CAL_DIR/scorer_fake.json"
CONTROLS="${CONTROLS:-$W/controls_devhalf_english.jsonl}"
if [ ! -f "$CONTROLS" ]; then      # 300 English sentences of the FLORES+ dev train half (rows of data_devhalf): the strict check MUST flag all of them
  $PREFIX "$PY" - "$CONTROLS" "${DEVHALF_PAIRS:-/workspace/work/out/data_devhalf/pairs-0000.parquet}" <<'PYEOF'
import json, sys, pyarrow.parquet as pq
seen, out = set(), []
for e in pq.read_table(sys.argv[2]).to_pydict()["english"]:
    if e not in seen:
        seen.add(e); out.append({"english": e})
    if len(out) >= 300: break
open(sys.argv[1], "w").write("\n".join(json.dumps(r) for r in out) + "\n")
PYEOF
fi
# (a) AfriScience windows, both multi-parallel options exercised
r s4_afs_windows.py --afs-train "$DATA/afriscience_clean_train/afriscience_clean_train.parquet" --out "$W/afs" --limit-papers 8 --dev-from-hub --tokenizer "$BASE" $IDX
r s4_afs_windows.py --afs-train "$DATA/afriscience_clean_train/afriscience_clean_train.parquet" --out "$W/afs_any_split" --limit-papers 8 --multi-parallel any --gap-policy split --tokenizer "$BASE"
# (b) replay: UNGATED windows from a sample of the clean pairs -> packs (chain/pack.py, the old command) -> s4_replay.py --pack-dir
r s4_make_sample.py --src $SRC_PAIRS --out "$W/sample_pairs" --n 40000 --per-group 150
r chain/windows.py --src "$W/sample_pairs" --out "$W/windows_sft" --afro-tokenizer "$TOK_AFRO" --gemma-tokenizer "$TOK_GEMMA"
r chain/pack.py prepare --windows "$W/windows_sft" --out "$W/packed" --tokenizer "$BASE" --workers 2
r s4_replay.py --pack-dir "$W/packed" --windows-dir "$W/windows_sft" --out "$W/replay" --replay-tokens 600000 --min-examples 20 $IDX --tokenizer "$BASE"
# (c) archive: synthetic archive.sqlite -> s4_archive
r s4_smoke_archive.py --pool-v3 "$DATA/clean_v3" --pool-sci "$DATA/clean_sci_v2" --bank "$W/replay/replay_train.jsonl" --out "$W/arch/archive.sqlite" --langs yor,hau,amh,kea --per-lang 40
r s4_archive.py --archive newrl_SMOKE="$W/arch/archive.sqlite" --out "$W/arch" $IDX $CALARGS --pools "$DATA/clean_v3,$DATA/clean_sci_v2" --model "$BASE" --model-label SMOKE-synthetic
# (d) long documents (LibreTexts included) and (e) anchors
for kind in longdoc anchor; do
  if [ $kind = longdoc ]; then D="$W/ld"; SEL="--per-lang b512=3,b1024=2 --languages yor,hau,kea,amh"; else D="$W/an"; SEL="--per-lang b64=6,b256=4,b512=2 --exclude $W/ld/docs.jsonl"; fi
  r s4_longdoc.py select --kind $kind --pool-v3 "$DATA/clean_v3" --pool-sci "$DATA/clean_sci_v2" --out "$D" --model "$BASE" $IDX $SEL
  r s4_longdoc.py gen --work "$D" --model "$BASE" --engine fake --fake-bank "$W/replay/replay_train.jsonl"
  r s4_longdoc.py score --work "$D" --model "$BASE" $CALARGS --allow-fake
  r s4_longdoc.py pick --work "$D" --model "$BASE" --kind $kind --model-label SMOKE-FAKE-GEN
done
# real CPU generation with the base model on 4 requests (engine hf), separate work dir: proves the model path, output not used downstream
mkdir -p "$W/hf_check"; head -n 4 "$W/ld/req_sent.jsonl" > "$W/hf_check/req_sent.jsonl"
r s4_longdoc.py gen --work "$W/hf_check" --model "$BASE" --engine hf --hf-max-new 24
# (f) mix + pack + strict protected-index check of the decoded training tokens
r s4_mix.py --out "$W/mix" --model "$BASE" --science "$W/afs/science_windows.jsonl" --val-sci "$W/afs/val_sci.jsonl" --replay "$W/replay/replay_train.jsonl" --replay-val "$W/replay/replay_val.jsonl" \
    --archive "$W/arch/archive_selected.jsonl" --longdoc "$W/ld/longdoc_selected.jsonl" --anchor "$W/an/anchor_selected.jsonl" $IDX --smoke
r s4_decon_final.py --packed "$W/mix" $IDX --out "$W/mix/decon_final.json" --controls-jsonl "$CONTROLS" \
    --jsonl "$W/afs/science_windows.jsonl" "$W/replay/replay_train.jsonl" "$W/arch/archive_selected.jsonl" "$W/ld/longdoc_selected.jsonl" "$W/an/anchor_selected.jsonl"
echo "[$(date +%T)] SMOKE chain finished in $(( $(date +%s) - t0 )) s"
