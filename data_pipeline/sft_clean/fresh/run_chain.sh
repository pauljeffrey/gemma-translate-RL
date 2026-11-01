#!/bin/bash
# Idempotent runner of the whole post-build chain on a FRESH instance. Every step writes state/<name>.done and logs/<name>.log; a finished step is skipped
# (FORCE=<step name> re-runs one; STEPS="a b c" runs only those). Timing of every step: logs/timing.tsv. HF_TOKEN comes from the environment (env.sh, placed by the lead; never written anywhere).
#   source $ROOT/env.sh; bash $ROOT/code/sft_clean/fresh/run_chain.sh            # whole chain
#   STOP_AFTER_FIT=1 bash .../run_chain.sh                                        # stops after the OLD+CAL keep-rate report and the hidden-NC report (read both, then run again to apply)
# GPU use (RTX PRO 6000, $0.58/h): 25 probes 1/2/4 scoring processes and picks the best; the bulk QE (32/33) runs N processes while H_cal reference QE, the cross-check, the CPU-only NC steps
# and any $ROOT/gpu_jobs.d/*.job (SSA-COMET, bigram table, NLLB pseudo-references, ...) run beside it (gpu_jobs_runner.sh). vram.log is written every 30 s.
set -uo pipefail
ROOT=${ROOT:-/workspace/work}; CODE=$ROOT/code; CH=$CODE/sft_clean/chain; FR=$CODE/sft_clean/fresh
HUB=$ROOT/hub; W=$ROOT/work; ST=$ROOT/state; LG=$ROOT/logs; mkdir -p "$HUB" "$W" "$ST" "$LG" "$ROOT/gpu_jobs.d"
HUB_REPO=${HUB_REPO:-${HF_ORG_LEGACY}/sabiyarn-clean-sft-2026-10-28}; PACK_REPO=${PACK_REPO:-${HF_ORG}/sabiyarn-clean-sft-packs-2026-10-29}
GATE_RULE=${GATE_RULE:-cal}; QE_PROCS=${QE_PROCS:-auto}; QE_BATCH=${QE_BATCH:-128}; QE_BATCHES=${QE_BATCHES:-"128 256"}; PACK_WORKERS=${PACK_WORKERS:-8}; MIN_FREE_GB=${MIN_FREE_GB:-12}
NC_ALLOW=${NC_ALLOW-howard-nlp/ibom-mt}                      # owner exceptions (2026-10-29 evening): NC sources that STAY, tagged nc_exception=<last path component>; "" = none
HIDDEN_NC=${HIDDEN_NC:-1}                                    # 1 = drop the rows whose NC origin is identifiable per row (rules R1/R2 of nc_hidden.py); whole sources are only LISTED
HIDDEN_DROP_SOURCES=${HIDDEN_DROP_SOURCES:-}                 # owner decision: whole unsplittable sources to drop (repo ids, space separated), e.g. "UBC-NLP/AfroLingu-MT"
HIDDEN_ALLOW_BELOW=${HIDDEN_ALLOW_BELOW:-0}                  # 1 = allow languages to end below 2,000 pairs because of the hidden-NC filter (default: the step stops and says which)
HIDDEN_CORPORA=${HIDDEN_CORPORA-'fiifinketia/pristine-twi-english|train.*parquet|english_translation|twi|twi|pristine-twi (CC-BY-NC upstream)'}
FUL_MIN_SIM=${FUL_MIN_SIM:-1.06}; EXTRA_LANGS=${EXTRA_LANGS:-"nde ful knc tzm"}; EXTRA_ALLOW_MISSING=${EXTRA_ALLOW_MISSING:-}      # Amendment 6: new SFT languages, built by extra_sources.py (NLLB-mined Fulah: LASER margin >= FUL_MIN_SIM, the set runs from 1.05 up)
QE_FAST=${QE_FAST:-1}; QE_RATE=${QE_RATE:-129}       # QE_FAST=1: qe_fast.py batching (same model and scores, see its docstring); the first chunk of every run is cross-checked against comet predict and falls back by itself
QE_CAP=${QE_CAP:-150000}      # pairs per language that are QE-scored (deterministic, monotone hash sample; languages with fewer pairs in full); 0 = no cap. A pair that is not scored is not kept.
A2_FRAC=${A2_FRAC:-0.25}; A2_SEED=${A2_SEED:-1234}; A2_FLOOR=${A2_FLOOR:-}        # A2_FLOOR: optional per-language floor of draws (sampler.py subset --floor-draws), agent T1's decision
export ROOT MIN_FREE_GB GEMMA_MT_PKG=$CODE PYTHONPATH=$CH:$CODE HF_HOME=${HF_HOME:-$ROOT/hf} TOKENIZERS_PARALLELISM=false
PD=$ROOT/venv_data/bin/python; PC=$ROOT/venv_comet/bin/python
NCA=(); [ -n "$NC_ALLOW" ] && NCA=(--nc-allow $NC_ALLOW)
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is not set (source env.sh; it is never stored)"; exit 2; }
pgrep -f disk_watch.sh >/dev/null || (nohup bash "$FR/disk_watch.sh" >/dev/null 2>&1 &)

step() {      # step <name> <need_free_GB> <command...>
  local name=$1 need=$2; shift 2
  if [ -n "${STEPS:-}" ] && [[ " $STEPS " != *" $name "* ]]; then return 0; fi
  if [ -f "$ST/$name.done" ] && [ "${FORCE:-}" != "$name" ]; then echo "[skip] $name (done $(cat "$ST/$name.done"))"; return 0; fi
  local free; free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
  if [ "$free" -lt "$need" ]; then echo "[STOP] $name needs about ${need} GB free, only ${free} GB: free space (see README_RUNBOOK.md, 'what to delete') and re-run"; exit 3; fi
  echo "[run ] $name ($(date -u +%T))"; local t0; t0=$(date +%s)
  if "$@" 2>&1 | tee -a "$LG/$name.log"; [ "${PIPESTATUS[0]}" = "0" ]; then
    local dt=$(( $(date +%s) - t0 )); echo "$(date -u +%FT%TZ) ${dt}s" > "$ST/$name.done"; printf "%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$name" "$dt" >> "$LG/timing.tsv"; echo "[done] $name ${dt}s"
  else echo "[FAIL] $name (see $LG/$name.log); fix and re-run the same command (finished steps are skipped)"; exit 1; fi
}
get() { "$PD" "$FR/hub_get.py" "$@"; }
BGP=()
bg() { ( "$@" ) & BGP+=($!); }                                       # run a function (group of steps) in the background
wait_bg() { local rc=0 p; for p in "${BGP[@]:-}"; do [ -n "$p" ] && { wait "$p" || rc=1; }; done; BGP=(); [ "$rc" = 0 ] || { echo "[FAIL] a background group failed (see logs/); fix and re-run"; exit 1; }; }
qe_cfg() {     # sets QE_N, QE_B, QE_W : the probe result (step 25) when QE_PROCS=auto
  QE_N=$QE_PROCS; QE_B=$QE_BATCH
  if [ "$QE_PROCS" = auto ]; then
    if [ -f "$W/qe_probe/qe_procs.json" ]; then read -r QE_N QE_B < <("$PD" -c "import json;d=json.load(open('$W/qe_probe/qe_procs.json'));print(d['best_procs'],d['best_batch'])"); else QE_N=3; fi
  fi
  QE_W=$(( $(nproc) / QE_N )); [ "$QE_W" -gt 8 ] && QE_W=8; [ "$QE_W" -lt 2 ] && QE_W=2
}
CAPARG=(); [ "${QE_CAP:-0}" -gt 0 ] && CAPARG=(--cap-per-lang "$QE_CAP" --cap-counts "$W/cap_counts.json")
qe_parts() {   # qe_parts <label> <extra args...> : N processes on the GPU, resumable, NC rows not scored (except the owner exceptions)
  local label=$1; shift; local pids=(); qe_cfg; echo "QE: $QE_N processes, batch $QE_B, $QE_W loader workers each"
  for i in $(seq 0 $((QE_N-1))); do
    "$PC" "$CH/qe_score.py" "$@" $([ "$QE_FAST" = 1 ] && [ ! -f "$ST/qe_fast_failed" ] || echo --no-fast) --out "$W/scores_gpu" --device cuda --half --batch "$QE_B" --workers "$QE_W" --chunk 100000 --part "$i/$QE_N" --exclude-nc ${NCA[@]+"${NCA[@]}"} ${CAPARG[@]+"${CAPARG[@]}"} \
      --skip-scored "$W/scores_cpu" "$W/scores_gpu" --budget-pairs-per-s 1100 >> "$LG/qe_${label}_$i.log" 2>&1 & pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; return $rc
}

# ---------------------------------------------------------------- preflight, data (sizes in README_RUNBOOK.md), background helpers
step 00_preflight 5 bash -c "nvidia-smi -L && '$PC' -c 'import torch;assert torch.cuda.is_available()' && '$PD' -c 'import pyarrow,tokenizers,transformers' && echo preflight ok"
(pgrep -f "query-gpu=timestamp" >/dev/null || nohup nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 30 >> "$LG/vram.log" 2>&1 &)
(pgrep -f gpu_jobs_runner.sh >/dev/null || KEEP_ALIVE=1 nohup bash "$FR/gpu_jobs_runner.sh" >> "$LG/gpu_jobs_runner.log" 2>&1 &)      # side jobs from $ROOT/gpu_jobs.d/*.job run beside the QE
step 10_fetch_tokenizers 1 bash -c "'$PD' '$FR/hub_get.py' --repo Davlan/afro-xlmr-large-76L --type model --include tokenizer.json --out '$W/tok_afro'"      # the Gemma tokenizer (chat template) is fetched by prepare_sft through HF_HOME
step 11_fetch_cpu_scores 2 bash -c "'$PD' '$FR/hub_get.py' --repo '$HUB_REPO' --include 'qe_scores_cpu/scores/*' 'qe_scores_cpu/*.json' --out '$W/_hub_scores' && mkdir -p '$W/scores_cpu' && cp '$W'/_hub_scores/qe_scores_cpu/scores/*.parquet '$W/scores_cpu/'"
step 12_fetch_pairs 8 get --repo "$HUB_REPO" --include 'data/*' 'data_devhalf/*' 'specialist/*' --out "$HUB"
step 13_fetch_split 1 bash -c "cp '$CODE/heldout_split_cal_sel_v2.json' '$W/heldout_split_cal_sel_v2.json' && echo copied the v2 H_cal H_sel split"
# ---------------------------------------------------------------- 15: the four languages of Amendment 6 (nde ful knc tzm): download -> standard pair schema -> dedupe -> STRICT decontamination (protected index, both sides) -> $HUB/extra
# runs BEFORE the QE, the NC steps and the windows so that these pairs go through exactly the same chain; raw downloads are deleted as they are read, the 2.3 GB protected index is deleted afterwards
step 15_extra_sources 8 bash -c "'$PD' '$FR/hub_get.py' --repo '$HUB_REPO' --include 'protected_index/protected_index.npz' 'protected_index/protected_manifest.json' 'protected_index/protected_exemption_devhalf_v3.npz' --out '$W/_hub_protected' && '$PD' '$CH/extra_sources.py' --out '$HUB/extra' --index '$W/_hub_protected/protected_index/protected_index.npz' --manifest '$W/_hub_protected/protected_index/protected_manifest.json' --tmp '$W/_extra_dl' --languages $EXTRA_LANGS --ful-min-sim '$FUL_MIN_SIM' ${EXTRA_ALLOW_MISSING:+--allow-missing $EXTRA_ALLOW_MISSING} && rm -rf '$W/_hub_protected' '$W/_extra_dl'"
SRC="$HUB/data $HUB/data_devhalf $HUB/specialist $HUB/extra"

# ---------------------------------------------------------------- 16: unique scoreable pairs per language, counted ONCE (the QE cap thresholds come from this file; it is never overwritten, so QE_CAP can be raised later and only the new keys are scored)
step 16_cap_counts 2 "$PD" "$CH/qe_cap.py" count --src $SRC --out "$W/cap_counts.json" ${NCA[@]+"${NCA[@]}"}
[ "${QE_CAP:-0}" -gt 0 ] && [ -f "$W/cap_counts.json" ] && "$PD" "$CH/qe_cap.py" table --counts "$W/cap_counts.json" --caps 100000 150000 200000 300000 --rate "${QE_RATE:-129}" | tee "$LG/qe_cap_table.txt"

# ---------------------------------------------------------------- 25: throughput probe (a few minutes): pairs/s for 1, 2, 4 scoring processes (and the batch sizes in QE_BATCHES) -> work/qe_probe/qe_procs.json
if [ "$QE_PROCS" = auto ]; then
  step 25_qe_probe 6 bash -c "'$PD' '$CH/qe_probe.py' --pairs '$HUB/data' --out '$W/qe_probe' --python '$PC' --procs 1 2 4 --batches $QE_BATCHES --n 3000 && cp '$W/qe_probe/qe_procs.json' '$LG/qe_procs.json'"
fi

# ---------------------------------------------------------------- 24: the fast scorer against comet predict on the real GPU (about 2 minutes): Pearson must be >= 0.9999, speeds of both paths are printed; a failure switches the whole chain to --no-fast
if [ "$QE_FAST" = 1 ]; then
  step 24_qe_fast_check 4 bash -c "'$PC' '$CH/qe_fast.py' check --pairs '$HUB/data' --n 800 --device cuda --half --batch 64 --cache-gb 2 || { touch '$ST/qe_fast_failed'; echo 'fast scorer rejected: the chain continues with --no-fast'; }"
fi

# ---------------------------------------------------------------- CPU-only side steps (run beside the QE): licence report, hidden-NC filter (reports BEFORE anything is dropped)
cpu_side() {
  step 20_nc_report 1 "$PD" "$CH/nc_filter.py" report --src $SRC --out "$W/nc_report" ${NCA[@]+"${NCA[@]}"}
  [ "$HIDDEN_NC" = 1 ] || return 0
  local hc=(); [ -n "$HIDDEN_CORPORA" ] && hc=(--hub-corpus "$HIDDEN_CORPORA")
  step 21_nc_hidden_index 2 "$PD" "$CH/nc_hidden.py" build-index --out "$W/nc_origin_index.npz" ${hc[@]+"${hc[@]}"} --tmp "$W/_nc_origin_dl"
  step 22_nc_hidden_report 1 "$PD" "$CH/nc_hidden.py" report --src $SRC --index "$W/nc_origin_index.npz" --out "$W/nc_hidden_report" ${NCA[@]+"${NCA[@]}"} ${HIDDEN_DROP_SOURCES:+--drop-source $HIDDEN_DROP_SOURCES}
  local ab=(); [ "$HIDDEN_ALLOW_BELOW" = 1 ] && ab=(--allow-below-min)
  step 23_nc_hidden_apply 1 "$PD" "$CH/nc_hidden.py" apply --src $SRC --index "$W/nc_origin_index.npz" --out "$W/nc_hidden_keys.parquet" ${NCA[@]+"${NCA[@]}"} ${HIDDEN_DROP_SOURCES:+--drop-source $HIDDEN_DROP_SOURCES} ${ab[@]+"${ab[@]}"}
}
bg cpu_side

# ---------------------------------------------------------------- GPU side steps beside the bulk scoring: cross-check, H_cal references (benchmark text exists only here, only for these minutes)
gpu_side() {
  step 30_crosscheck 6 bash -c "'$PD' '$CH/qe_crosscheck.py' prepare --from-scores '$W/scores_cpu' --pairs $HUB/data $HUB/data_devhalf $HUB/specialist --out '$W/crosscheck_pairs' && '$PC' '$CH/qe_score.py' --src '$W/crosscheck_pairs' --out '$W/scores_gpu_check' --device cuda --half --batch 64 --chunk 500 --workers 4 && '$PD' '$CH/qe_crosscheck.py' compare --cpu '$W/scores_cpu' --gpu '$W/scores_gpu_check'"
  step 31_hcal_refs 4 bash -c "'$PD' '$CH/hcal.py' refs --split '$W/heldout_split_cal_sel_v2.json' --out '$W/refs_hcal' --tmp '$W/_hcal_dl' && '$PC' '$CH/qe_score.py' --src '$W/refs_hcal' --out '$W/scores_refs' --device cuda --half --batch 64 --chunk 20000 --workers 4 && '$PD' -c \"import pyarrow.parquet as pq,os;os.makedirs('$W/refs_keys',exist_ok=True);pq.write_table(pq.read_table('$W/refs_hcal/pairs-0000.parquet',columns=['key','iso']),'$W/refs_keys/pairs-0000.parquet')\" && rm -rf '$W/refs_hcal' '$W/_hcal_dl'"
}
bg gpu_side

# ---------------------------------------------------------------- bulk QE: N processes (probe), NC rows not scored (owner exceptions are)
step 32_qe_main 12 qe_parts main --src $HUB/data $HUB/data_devhalf
step 33_qe_specialist 6 qe_parts specialist --src $HUB/specialist --skip-keys-of $HUB/data $HUB/data_devhalf
step 34_qe_extra 3 qe_parts extra --src $HUB/extra --skip-keys-of $HUB/data $HUB/data_devhalf $HUB/specialist
wait_bg          # the side groups (and with them nc_hidden/H_cal) must be complete before the gate; gpu_jobs.d jobs keep running independently

# ---------------------------------------------------------------- gate (b): both rules side by side, then apply the chosen one
step 40_gate_fit 6 "$PD" "$CH/gate.py" fit --src $SRC --scores "$W/scores_cpu" "$W/scores_gpu" --refs-scores "$W/scores_refs" --refs "$W/refs_keys" --out "$W/gate" --exclude-nc ${NCA[@]+"${NCA[@]}"}
if [ "${STOP_AFTER_FIT:-0}" = "1" ]; then echo "STOP_AFTER_FIT=1: read $W/gate/keep_rate_per_language.csv (OLD and CAL side by side), $W/nc_report/ and $W/nc_hidden_report/nc_hidden_by_language_source.csv (nothing dropped yet), then re-run"; exit 0; fi
step 41_gate_apply 6 "$PD" "$CH/gate.py" apply --rule "$GATE_RULE" --src $SRC --scores "$W/scores_cpu" "$W/scores_gpu" --out "$W/gate" --exclude-nc ${NCA[@]+"${NCA[@]}"}
KEEP="$W/gate/keep_$GATE_RULE.parquet"
if [ "$HIDDEN_NC" = 1 ]; then
  step 42_keep_minus_hidden 2 "$PD" "$CH/nc_hidden.py" merge-keep --keep "$KEEP" --drop "$W/nc_hidden_keys.parquet" --out "$W/gate/keep_final.parquet"; KEEP="$W/gate/keep_final.parquet"
fi
step 43_sanity 3 bash -c "'$PD' '$FR/sanity_after_gate.py' --src $SRC --keep '$KEEP' --scores '$W/scores_cpu' '$W/scores_gpu' --out '$W/gate/sanity_after_gate'"

# ---------------------------------------------------------------- windows (c), pack (d)
step 50_windows 12 "$PD" "$CH/windows.py" --src $SRC --keep-file "$KEEP" --out "$W/windows_sft" --afro-tokenizer "$W/tok_afro/tokenizer.json" ${NCA[@]+"${NCA[@]}"}
step 51_nc_guard 1 "$PD" "$CH/nc_filter.py" guard-windows --windows "$W/windows_sft"
step 60_pack 30 "$PD" "$CH/pack.py" prepare --windows "$W/windows_sft" --out "$W/packed" --workers "$PACK_WORKERS"

# ---------------------------------------------------------------- schedules (e): A0 stage A, stage B, S1/S2, G0 (T1 sampler.py select), A2 (T1 sampler.py subset: ONE file shared by A2-base, A2-up6, A2-up12 and Gemma-3-1B-it; no fallback)
step 70_sched_A0 1 "$PD" "$CH/sampler.py" plan --stage A --packed-dir "$W/packed/train" --out "$W/sched/A0" --seed 1234
step 71_sched_B0 1 "$PD" "$CH/sampler.py" plan --stage B --packed-dir "$W/packed/train" --out "$W/sched/B0" --seed 1234 --budget-tokens 568197120
step 72_sched_S1S2 1 "$PD" "$CH/sampler.py" arms --schedule "$W/sched/A0" --out "$W/sched"
step 73_sched_G0 1 "$PD" "$CH/sampler.py" select --schedule "$W/sched/A0" --out "$W/sched/G0" --languages amh,tir --name G0
step 74_sched_A2 1 "$PD" "$CH/sampler.py" subset --schedule "$W/sched/A0" --out "$W/sched/A2" --frac "$A2_FRAC" --stratify language --seed "$A2_SEED" ${A2_FLOOR:+--floor-draws $A2_FLOOR} --name A2
# S2 (re-tokenised packs) needs the extended tokenizer and is run later:  python pack.py replay --like work/packed --out work/packed_S2 --tokenizer <extended tokenizer>
# two concurrent SFT jobs on the one GPU: fresh/sft_throughput_probe.sh measures combined steps/s with 1 vs 2 jobs; fresh/sft_concurrent.sh launches them (README_RUNBOOK.md)

# ---------------------------------------------------------------- push (private) and the disk clean-up that the runbook allows
step 90_push 1 "$PD" "$FR/push_packs.py" --repo "$PACK_REPO" --work "$W"
echo "chain finished; timing: $LG/timing.tsv; disk: $LG/disk.log; GPU memory: $LG/vram.log. Optional clean-up after the push is verified: rm -rf $W/windows_sft $W/_hub_scores $HUB (pairs are on the Hub)"
