#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Post-SFT benchmark evaluation (owner 2026-10-18): the SFT model and the base gemma-3-270m-it, English -> African, on FLORES+ devtest, NTREX-128 and
# MAFAND-MT test, contamination-filtered, AfriCOMET included. Runs on the rental between "SFT finished" and "RL starts".
#
#   bash scripts/eval_sft_benchmarks.sh [SFT_MODEL_ID_OR_DIR] [extra args of gemma_mt.eval.sft_benchmarks ...]
#
# Output (OUT, default /workspace/sft_build/eval/sft_best): results.json results.csv predictions.jsonl report.md eval.log, work/ (resume caches), and ONE of
#   EVAL_DONE     everything ran, both systems, all benchmarks            (exit 0)
#   EVAL_FAILED   json with the reasons; partial results are still written (exit 1)
# It never waits forever: GPU wait (GPU_WAIT), generation / scoring limits per benchmark, a total limit (TOTAL_TIMEOUT) that kills the process tree.
# A second run resumes from the caches in OUT/work (generation units, AfriCOMET scores); remove OUT/work to start from scratch.
#
# Environment (defaults are the rental's):
#   ROOT=/workspace/sft_build   PY=/workspace/venv_gemma/bin/python   VLLM_PY=/workspace/venv_vllm/bin/python
#   COMET_PY=$PY   COMET_PYTHONPATH=$ROOT/pylibs_tf4:$ROOT/pylibs   EXTRA_LIBS=$ROOT/pylibs (appended to sys.path: sacrebleu etc.)
#   BENCH_DIR=$ROOT/bench (contamination_<benchmark>.json; test sets are downloaded to $BENCH_DIR/test_sets)   OUT=$ROOT/eval/sft_best
#   SAMPLER_FALLBACK=1 (default: HF generate when the vLLM worker cannot start; 0 = fail)   SFT_REVISION=  (Hub branch/tag/commit; default main)   SAMPLER=vllm|hf   GPU_UTIL=0.2   GPU_MIN_FREE_GIB=24   GPU_WAIT=1800
#   TOTAL_TIMEOUT=14400   GEN_TIMEOUT=5400   SCORE_TIMEOUT=5400   BUDGET=0 (seconds per invocation, then exit 75 and loop; for shared GPUs)
#   GPU_LOCK_CMD=""  (e.g. "flock /workspace/sft_build/.gpu.lock": prefix of every python invocation, used on the shared old box)
#   ITER_TIMEOUT=   (seconds for ONE python invocation; default TOTAL_TIMEOUT, or BUDGET+GPU_WAIT+600 when BUDGET is set)
#   CONTAM_DIR=     (D-42, default unset = $BENCH_DIR) directory of the contamination_<benchmark>.json files
#   BENCHMARKS=     (D-42, default unset = flores_plus ntrex mafand, unchanged) benchmarks to run, comma or space separated, e.g. BENCHMARKS=smol_sent,smol_doc,gatitos
#                   (Google SMOL / GATITOS, CC-BY-4.0, downloaded to $BENCH_DIR/test_sets/<name> on first use; their contamination_<name>.json must exist in $BENCH_DIR).
#                   Use OUT=... of its own for such a run. Documents (smol_doc) generate up to --doc-max-new-tokens 1024, words (gatitos) up to --word-max-new-tokens 48;
#                   extra args after the model id are passed on, e.g. --bench-limit smol_doc=50 --max-per-language 300
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # the gemma_mt tree this script belongs to (gemma_mt_eval_candidate/ until the supervisor copies it)
ROOT="${ROOT:-/workspace/sft_build}"
PY="${PY:-/workspace/venv_gemma/bin/python}"
VLLM_PY="${VLLM_PY:-/workspace/venv_vllm/bin/python}"
COMET_PY="${COMET_PY:-$PY}"
COMET_PYTHONPATH="${COMET_PYTHONPATH:-$ROOT/pylibs_tf4:$ROOT/pylibs}"
EXTRA_LIBS="${EXTRA_LIBS:-$ROOT/pylibs}"
OUT="${OUT:-$ROOT/eval/sft_best}"
BENCH_DIR="${BENCH_DIR:-$ROOT/bench}"
SAMPLER="${SAMPLER:-vllm}"
GPU_UTIL="${GPU_UTIL:-0.2}"
GPU_MIN_FREE_GIB="${GPU_MIN_FREE_GIB:-24}"
GPU_WAIT="${GPU_WAIT:-1800}"
TOTAL_TIMEOUT="${TOTAL_TIMEOUT:-14400}"
GEN_TIMEOUT="${GEN_TIMEOUT:-5400}"
SCORE_TIMEOUT="${SCORE_TIMEOUT:-5400}"
BUDGET="${BUDGET:-0}"
GPU_LOCK_CMD="${GPU_LOCK_CMD:-}"
if [ "$BUDGET" != "0" ]; then ITER_TIMEOUT="${ITER_TIMEOUT:-$((BUDGET + GPU_WAIT + 600))}"; else ITER_TIMEOUT="${ITER_TIMEOUT:-$TOTAL_TIMEOUT}"; fi
SFT_MODEL="${1:-${HF_ORG}/gemma3-270m-translate}"
[ $# -gt 0 ] && shift

mkdir -p "$OUT"
[ -f "$ROOT/env.sh" ] && { set +u; source "$ROOT/env.sh"; set -u; }           # HF_TOKEN (READ), HF_HOME=$ROOT/hf
export HF_HOME="${HF_HOME:-$ROOT/hf}"
export PYTHONPATH="$HERE"                                                       # nothing else: env.sh's pylibs would shadow the interpreter's own packages
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
# vLLM worker: compile caches on a real disk (/dev/shm is mounted noexec on the rentals), same variables as scripts/vast_train_rl.sh
VLLM_CACHE="${VLLM_CACHE:-$ROOT/vllm_cache}"
mkdir -p "$VLLM_CACHE"
export VLLM_CACHE_ROOT="$VLLM_CACHE/vllm" TRITON_CACHE_DIR="$VLLM_CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$VLLM_CACHE/inductor" \
       XDG_CACHE_HOME="$VLLM_CACHE/xdg" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

exec > >(tee -a "$OUT/eval.log") 2>&1
export GEMMA_EVAL_RUN_ID="eval-$$-$(date +%s)"                                 # inherited by every child (but not by tee above): used to find leftovers
echo "[eval_sft_benchmarks] $(date -u +%FT%TZ) start: model=$SFT_MODEL out=$OUT code=$HERE sampler=$SAMPLER run=$GEMMA_EVAL_RUN_ID"

kill_leftovers() {   # children of THIS run (vLLM worker, COMET workers) are found by their environment, never by name: the trainer's processes are not touched
  local me=$$ pid
  for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    [ "$pid" = "$me" ] && continue
    if { tr '\0' '\n' < "/proc/$pid/environ" | grep -qx "GEMMA_EVAL_RUN_ID=$GEMMA_EVAL_RUN_ID"; } 2>/dev/null; then kill -9 "$pid" 2>/dev/null || true; fi
  done
}
trap kill_leftovers EXIT

rm -f "$OUT/EVAL_DONE" "$OUT/EVAL_FAILED" "$OUT/EVAL_RUNNING"
ARGS=(--sft-model "$SFT_MODEL" --out "$OUT" --bench-dir "$BENCH_DIR" --sampler "$SAMPLER" --vllm-python "$VLLM_PY" --vllm-gpu-util "$GPU_UTIL"
      --comet-python "$COMET_PY" --comet-pythonpath "$COMET_PYTHONPATH" --extra-libs "$EXTRA_LIBS"
      --gpu-min-free-gib "$GPU_MIN_FREE_GIB" --gpu-wait "$GPU_WAIT" --total-timeout "$((TOTAL_TIMEOUT > 300 ? TOTAL_TIMEOUT - 120 : TOTAL_TIMEOUT))"
      --gen-timeout "$GEN_TIMEOUT" --score-timeout "$SCORE_TIMEOUT" --budget-seconds "$BUDGET")
[ -n "${SFT_REVISION:-}" ] && ARGS+=(--sft-revision "$SFT_REVISION")
# if the vLLM worker cannot start, generate with HF instead (slower, ~1 h on the 96 GB card) rather than block the RL run: recorded as a WARNING in results.json / EVAL_DONE
[ "${SAMPLER_FALLBACK:-1}" = "1" ] && ARGS+=(--sampler-fallback-hf)
[ -n "${BENCHMARKS:-}" ] && ARGS+=(--benchmarks "$BENCHMARKS")
[ -n "${CONTAM_DIR:-}" ] && ARGS+=(--contamination-dir "$CONTAM_DIR")
ARGS+=("$@")

rc=0
while :; do
  # shellcheck disable=SC2086   # GPU_LOCK_CMD is a command prefix (several words)
  $GPU_LOCK_CMD timeout --kill-after=60 "$ITER_TIMEOUT" "$PY" -m gemma_mt.eval.sft_benchmarks "${ARGS[@]}"
  rc=$?
  kill_leftovers
  [ "$rc" -eq 75 ] || break
  echo "[eval_sft_benchmarks] budget of this invocation used up, continuing from the caches"
  sleep 5
done

if [ ! -e "$OUT/EVAL_DONE" ] && [ ! -e "$OUT/EVAL_FAILED" ]; then   # the python process died before it could write a marker (import error, kill, timeout)
  reason="gemma_mt.eval.sft_benchmarks exited with code $rc without writing a marker"
  [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ] && reason="$reason (timeout of ${ITER_TIMEOUT}s)"
  printf '{"reason": "%s; see %s/eval.log", "exit_code": %s}\n' "$reason" "$OUT" "$rc" > "$OUT/EVAL_FAILED"
  rm -f "$OUT/EVAL_RUNNING"
  [ "$rc" -eq 0 ] && rc=1
fi
echo "[eval_sft_benchmarks] $(date -u +%FT%TZ) exit $rc ($(ls "$OUT" | grep -E '^EVAL_' | tr '\n' ' '))"
exit "$rc"
