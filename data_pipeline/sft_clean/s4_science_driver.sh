#!/usr/bin/env bash
# Clean science-stage rebuild: ONE driver with the exact commands in order (agent C2, 2026-10-28).
#
#   NEW_RL_MODEL=/workspace/clean_rerun/runs/mixed/final  PACK_DIR=/workspace/work/packed  [ARCHIVES=A=...,B1=...,B2=...,B3=...,mixed=...] \
#   CAL=... ANCHORS=... SCORER_JSON=... HF_TOKEN=... bash s4_science_driver.sh [step ...]
#
# Steps (default: all, in this order; every step is resumable and skips finished work):
#   fetch      CPU   clean pools + clean AfriScience-MT train + the protected index from the Hub
#   afs        CPU   (a) AfriScience-MT windows, 2 passes                                  A2W reference 6.62 M tokens
#   replay     CPU   (b) replay windows from the CLEAN packs (PACK_DIR = gated packs)       A2W reference 8.99 M tokens
#   archive    GPU   (c) RL best-translation archive of the NEW run, QE-ensemble re-judged  ~0.1 GPU-h  (A2W reference 1.04 M tokens)
#   longdoc    GPU   (d) long-document distillation, best-of-4, clean_v3 + clean_sci_v2 (LibreTexts INCLUDED, tagged)   (A2W 2.83 M tokens)
#   anchors    GPU   (e) self-distilled anchors on short clean windows of the 47 weak / Ge'ez languages                  (A2W 1.41 M tokens)
#   mix        CPU   (f) mixer + packer: A2W ratios, per-row provenance/licence, config; then the strict protected-index check (must report 0 hits)
#   baseline   GPU   guardrail evaluation of the start model (NEW_RL_MODEL)                       ~0.15 h
#   train      GPU   319 steps x 65,536 tokens, lr 3e-6, start = NEW_RL_MODEL
#   sweep      GPU   theta = (1-a) theta_RL + a theta_SFT, a in 0.3 0.5 0.7 1.0, guardrail evaluation (A2W chose a = 0.7: 0.3 RL + 0.7 SFT)
#   report     CPU   before/after tables of the sweep
#
# GPU time (ESTIMATED, RTX PRO 6000 at USD 1.19/h; nothing here has run on a GPU). Bottom-up from per-document counts MEASURED on the real clean pools in the CPU smoke
#   (sentence units / source tokens per document: b64 1.7 / 34, b256 4.4 / 128, b512 10.8 / 293, b1024 22.8 / 647): 14,428 candidate documents (long documents 77 languages x (70 b512 +
#   60 b1024) = 10,010; anchors 47 languages x (30 b64 + 40 b256 + 24 b512) = 4,418) = ~186 k unique sentence units, ~5.2 M source tokens; x4 candidates x 2.0 generated tokens per
#   source token (R6 measurement) = ~41.6 M generated tokens: 0.46 h at 25 k tok/s [0.29 h at 40 k, 0.96 h at 12 k]; QE of the valid candidates <= 744 k pairs x 2 models: 0.38 h at 1,100
#   pairs/s per model [0.28 h at 1,500, 0.86 h at 480]; archive re-judgement 0.02 h; vLLM/QE start-up and restarts 0.1 h.
#   (c)-(e) bottom-up 1.0 h [0.7-2.0]; the lean-schedule plan M1 keeps 1.5 h [0.8-3.0] (the old job: ~2.5 h on a shared card in 3 restarts): BUDGET 1.5 h.
#   baseline 0.15 h [0.10-0.20]; train 319 steps 0.35 h [0.30-0.80] ; sweep 0.55 h [0.40-0.80] (A2W logs: 20.7 min on a quiet card; 8-10 min per alpha). Rebuild + stage: 2.05 h
#   bottom-up = USD 2.4; with M1's 1.5 h for (c)-(e): 2.55 h = USD 3.0.
set -euo pipefail

: "${NEW_RL_MODEL:?path of the new RL model (HF dir: weights + tokenizer)}"
: "${PACK_DIR:?packed dataset of the gated clean SFT data (chain/pack.py prepare output; ungated packs for a dry run)}"
ARCHIVES="${ARCHIVES:-A=/workspace/clean_rerun/runs/A/archive/archive.sqlite,B1=/workspace/clean_rerun/runs/B1/archive/archive.sqlite,B2=/workspace/clean_rerun/runs/B2/archive/archive.sqlite,B3=/workspace/clean_rerun/runs/B3/archive/archive.sqlite,mixed=/workspace/clean_rerun/runs/mixed/archive/archive.sqlite}"   # one archive.sqlite per stage of the clean RL run (configs/clean_rerun/rl_*.yaml)
: "${CAL:?refit shaping calibration json of the clean rerun}"
: "${ANCHORS:?refit anchors json of the clean rerun}"
: "${SCORER_JSON:?json with the QE scorer spec, e.g. {\"type\":\"worker\",\"python\":\".../venv_comet/bin/python\",\"pythonpath\":\"...\",\"model_id\":\"masakhane/africomet-qe-stl-1.1\",\"ensemble_model_id\":\"McGill-NLP/ssa-comet-mtl\",\"batch_size\":96,\"max_batch_chars\":48000}}"
WORK="${WORK:-/workspace/s4}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
export GEMMA_MT_PKG="${GEMMA_MT_PKG:-$REPO_ROOT/gemma_mt}"
export S4_OLD_SCRIPTS="${S4_OLD_SCRIPTS:-$REPO_ROOT/gemma_mt/scripts}"
PY="${PY:-python}"                     # data / CPU environment (torch, transformers, pyarrow, xxhash, fasttext, sacrebleu)
PY_VLLM="${PY_VLLM:-$PY}"              # environment with vLLM
GLOTLID="${GLOTLID:?path of GlotLID model_v3.bin}"
DATA="$WORK/data" ; IDX_DIR="$WORK/protected"
IDX="--index $IDX_DIR/protected_index/protected_index.npz --manifest $IDX_DIR/protected_index/protected_manifest.json"   # protected-2026-10-28.3, STRICT: no exemption is ever attached
CALARGS="--calibration $CAL --anchors $ANCHORS --glotlid $GLOTLID --scorer $SCORER_JSON"
MIX="$WORK/mix_clean_science"
MODEL_LABEL="${MODEL_LABEL:-$NEW_RL_MODEL}"
MULTI_PARALLEL="${MULTI_PARALLEL:-pair}"           # pair (default) | any: drop an English sentence in all six languages if it was removed in any
EXCLUDE_SOURCE="${EXCLUDE_SOURCE:-}"                # e.g. libretexts to switch the unresolved-licence rows off; default: INCLUDED and tagged
WEAK_LANGS="${WEAK_LANGS:-}"                        # default: the 45 weak-signal languages of A2W (s4_common.WEAK45)
WINDOWS_DIR="${WINDOWS_DIR:-$(dirname "$PACK_DIR")/windows_sft}"   # side table with source_dataset / licence of the pack samples (chain/windows.py output); [ -d ] checked below
[ -d "$WINDOWS_DIR" ] || WINDOWS_DIR=""
cd "$HERE"; mkdir -p "$WORK"/{data,protected,logs}
steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(fetch afs replay archive longdoc anchors mix baseline train sweep report)
run() { echo "[$(date +%T)] $*"; "$@"; }
has() { local s; for s in "${steps[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

if has fetch; then
  run $PY s4_fetch.py --out "$DATA" prompts --pools clean_v3,clean_sci_v2 --afs
  run $PY s4_fetch.py --out "$IDX_DIR" sft-index
fi
if has afs; then
  run $PY s4_afs_windows.py --afs-train "$DATA/afriscience_clean_train/afriscience_clean_train.parquet" --out "$WORK/afs" --passes 2 --multi-parallel "$MULTI_PARALLEL" --dev-from-hub \
      --tokenizer "$NEW_RL_MODEL" $IDX
fi
if has replay; then
  run $PY s4_replay.py --pack-dir "$PACK_DIR" ${WINDOWS_DIR:+--windows-dir "$WINDOWS_DIR"} --out "$WORK/replay" --science-tokens "$($PY -c "import json;print(json.load(open('$WORK/afs/science_windows_stats.json'))['tokens_total'])")" \
      ${WEAK_LANGS:+--weak-langs "$WEAK_LANGS"} $IDX
fi
if has archive; then
  ARCH_ARGS=(); for kv in ${ARCHIVES//,/ }; do ARCH_ARGS+=(--archive "$kv"); done
  run $PY s4_archive.py "${ARCH_ARGS[@]}" --out "$WORK/arch" $IDX $CALARGS --pools "$DATA/clean_v3,$DATA/clean_sci_v2" --model "$NEW_RL_MODEL" --model-label "$MODEL_LABEL" \
      ${EXCLUDE_SOURCE:+--exclude-source "$EXCLUDE_SOURCE"}
fi
longdoc_like() {   # $1 = kind (longdoc | anchor), $2 = work dir, $3.. = extra select args
  local kind=$1 w=$2; shift 2
  run $PY s4_longdoc.py select --kind "$kind" --pool-v3 "$DATA/clean_v3" --pool-sci "$DATA/clean_sci_v2" --out "$w" --model "$NEW_RL_MODEL" $IDX \
      ${EXCLUDE_SOURCE:+--exclude-source "$EXCLUDE_SOURCE"} "$@"
  run $PY s4_longdoc.py gen   --work "$w" --model "$NEW_RL_MODEL" --engine vllm --python "$PY_VLLM" --gpu-util 0.85 --max-seqs 512
  run $PY s4_longdoc.py score --work "$w" --model "$NEW_RL_MODEL" $CALARGS
  run $PY s4_longdoc.py pick  --work "$w" --model "$NEW_RL_MODEL" --kind "$kind" --model-label "$MODEL_LABEL"
}
if has longdoc; then longdoc_like longdoc "$WORK/ld"; fi
if has anchors; then longdoc_like anchor "$WORK/an" --exclude "$WORK/ld/docs.jsonl"; fi
if has mix; then
  run $PY s4_mix.py --out "$MIX" --model "$NEW_RL_MODEL" --science "$WORK/afs/science_windows.jsonl" --val-sci "$WORK/afs/val_sci.jsonl" --replay "$WORK/replay/replay_train.jsonl" \
      --replay-val "$WORK/replay/replay_val.jsonl" --archive "$WORK/arch/archive_selected.jsonl" --longdoc "$WORK/ld/longdoc_selected.jsonl" --anchor "$WORK/an/anchor_selected.jsonl" $IDX \
      ${EXCLUDE_SOURCE:+--exclude-source "$EXCLUDE_SOURCE"} --ratio-mode "${RATIO_MODE:-report}"
  # the number to report: HITS_samples must be 0 (strict index, decoded training tokens, English AND target, no exemption)
  run $PY s4_decon_final.py --packed "$MIX" $IDX --out "$MIX/decon_final.json" \
      --jsonl "$WORK/afs/science_windows.jsonl" "$WORK/replay/replay_train.jsonl" "$WORK/arch/archive_selected.jsonl" "$WORK/ld/longdoc_selected.jsonl" "$WORK/an/anchor_selected.jsonl"
fi
if has train; then
  ( cd "$GEMMA_MT_PKG" && run $PY -m gemma_mt.train_sft --config "$MIX/sft_science_clean.yaml" 2>&1 | tee -a "$WORK/logs/sci_sft_clean.log" )
fi
evalargs() {   # guardrail sets of the clean rerun (NOT the old bench_dev_clean / rl-vs-sft pool: they were partly contaminated): AFS_CORPUS = dir with dev-00000.parquet (AfriScience-MT dev
               # = an S set), GENERAL_DEV = general dev jsonl rebuilt on H_sel, EVAL_POOL = weak-language dev pool jsonl; supplied by the calibration / evaluation owners of the clean rerun
  : "${AFS_CORPUS:?}" "${GENERAL_DEV:?}" "${EVAL_POOL:?}"
  echo --afs-corpus "$AFS_CORPUS" --general-dev "$GENERAL_DEV" --eval-pool "$EVAL_POOL" --vllm-python "$PY_VLLM" --vllm-gpu-util 0.5 --kv-cache-gib 0
}
if has baseline; then
  ( cd "$GEMMA_MT_PKG" && run $PY scripts/sft_stage_eval.py run --model "$NEW_RL_MODEL" --name rl_start --out "$WORK/eval/rl_start" $(evalargs) )
fi
if has sweep; then
  ( cd "$GEMMA_MT_PKG" && run $PY scripts/sft_stage_eval.py sweep --init "$NEW_RL_MODEL" --sft "${MIX}_run/last/trainer_state.pt" --alphas 0.3 0.5 0.7 1.0 --out "$WORK/eval" \
      --work "$WORK/interp_tmp" $(evalargs) )
fi
if has report; then
  ( cd "$GEMMA_MT_PKG" && run $PY scripts/sft_stage_eval.py compare --baseline "$WORK/eval/rl_start" --runs "$WORK"/eval/alpha_0.30 "$WORK"/eval/alpha_0.50 "$WORK"/eval/alpha_0.70 "$WORK"/eval/alpha_1.00 --out "$WORK/eval/report.md" )
fi
echo "[$(date +%T)] done: ${steps[*]}"
