# Post-GPU chain for the clean rerun (D2, 2026-10-28)

Everything here was written and tested on CPU (a 20k-pair sample of the clean output on box 2). Nothing needs to be edited when the GPU is available; the only real unknown is the QE
throughput (`qe_score.py` prints pairs/s and an ETA). The final per-language, per-bin numbers need the real QE scores: the tables in `reports/data/sft_clean_2026-10-28/bins/` are the
UNGATED counts and a labelled ESTIMATE of the gated ones.

```
data/ (main, 23,406,354 pairs)   data_devhalf/ (8,747)   specialist/ (5,096,868 rows, 20 languages; 4,731,094 of them repeat main rows, 365,774 are external)
   -> unique union to score: about 23.77 M pairs
qe_score.py  ->  scores/ (key, qe)          scores_refs/ (QE of the FLORES+ dev H_cal human references)
gate.py fit  ->  gate/keep_rate_per_language.csv  (rules OLD and CAL side by side, nothing dropped)
gate.py apply --rule old|cal -> gate/keep_<rule>.parquet
windows.py   ->  windows_sft/ (old definitions: bins b64 b256 b512 b1024, pcm x2)
pack.py prepare -> packed/ (4,096 tokens, one language per pack, old chat format and loss mask)  [A0]
sampler.py plan --stage A|B, arms -> schedules; S1 = the draws of A0 that fall on specialist-language packs, same order
pack.py replay -> packed_S2/ (S1's packs re-tokenised with the extended tokenizer)               [S2]
```

## Environment (GPU box)
```
pip install torch                      # CUDA build
pip install unbabel-comet "pyarrow<20" xxhash tokenizers transformers numpy pandas   # comet pins numpy<2 -> pyarrow<20 (a separate venv from the data tools is simplest)
export HF_TOKEN=...                    # read access to <HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28 (private) and the gated FLORES+ dev
export GEMMA_MT_PKG=<repo>/gemma_mt    # folder that contains the python package gemma_mt (prepare_sft, packed_dataset, format)
hf download <HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28 --repo-type dataset --include "data/*" "data_devhalf/*" "specialist/*" "protected_index/*" "chain/*" --local-dir .
tokenizers: Davlan/afro-xlmr-large-76L tokenizer.json (window lengths, = "AfriCOMET tokens"), google/gemma-3-270m-it (packing)
```

## Command sequence (one resumable command per step; every step skips work already on disk)
```
cd chain
# 0. H_cal human references of FLORES+ dev (benchmark text, written locally ONLY, never push it)
python hcal.py split --devhalf-split ../stats/devhalf_split.json --out ../protected_index/heldout_split_cal_sel.json     # already in the repo; re-run is deterministic
python hcal.py refs  --split ../protected_index/heldout_split_cal_sel.json --out refs_hcal
# 1. QE (a): ~23.8 M pairs; run N processes on the GPU with --part i/N (one process is CPU/dataloader bound, as in the old run)
python qe_score.py --src ../data ../data_devhalf --out scores --half --batch 128 --workers 8 --part 0/3 --budget-pairs-per-s 1100 &
python qe_score.py --src ../data ../data_devhalf --out scores --half --batch 128 --workers 8 --part 1/3 --budget-pairs-per-s 1100 &
python qe_score.py --src ../data ../data_devhalf --out scores --half --batch 128 --workers 8 --part 2/3 --budget-pairs-per-s 1100 &
python qe_score.py --src ../specialist --out scores --half --batch 128 --workers 8 --skip-keys-of ../data ../data_devhalf     # only the 365,774 external specialist rows
python qe_score.py --src refs_hcal --out scores_refs --half --batch 64
# 2. gate (b): fit prints keep rates of BOTH rules, drops nothing
python gate.py fit --src ../data ../data_devhalf ../specialist --scores scores --refs-scores scores_refs --refs refs_hcal --out gate
#    -> read gate/keep_rate_per_language.csv, decide the rule, then
python gate.py apply --rule old --src ../data ../data_devhalf ../specialist --scores scores --out gate      # or --rule cal
# 3. windows (c)
python windows.py --src ../data ../data_devhalf ../specialist --scores scores --keep-file gate/keep_old.parquet --out windows_sft
# 4. pack (d) = A0
python pack.py prepare --windows windows_sft --out packed --workers 5
# 5. stage samplers (e)
python sampler.py plan --stage A --packed-dir packed/train --out sched/A0 --seed 1234
python sampler.py plan --stage B --packed-dir packed/train --out sched/B0 --seed 1234
python sampler.py arms --schedule sched/A0 --out sched            # sched/S1 (and S2) = A0's draws on the specialist-language packs, same order
# 6. S2 after the Ge'ez tokenizer exists: same packs, new token ids
python pack.py replay --like packed --out packed_S2 --tokenizer <extended tokenizer>        # the S1 pack list is applied through sched/S2
```
Tests: `python -m pytest chain/test_chain.py` (windows order, dedupe, stub refusal, gate rules, S1 = subsequence of A0).

## Expected runtime (estimates; the measured values are printed by each script)
| step | work | estimate |
|---|---|---|
| qe_score | 23.77 M pairs on one RTX PRO 6000 | **assumption 1,100 pairs/s** (old measurements on an RTX 5070, fp16: 290-350 pairs/s per process, 480 pairs/s with three processes; an RTX PRO 6000 Blackwell is assumed at 2.3x that): **6.0 GPU-hours**; at the old 480/s 13.8 h; at 1,500/s 4.4 h. The CPU smoke test gave 1.5 pairs/s on 2 cores (only a functional test). |
| gate fit/apply | 24 M scores in memory | minutes (numpy) |
| windows | 23.79 M unique pairs, 2 sides (Rust tokenizer, 2 threads) + python window logic | **measured: 4,395 s (73 min) on 2 CPU cores** for the stats-only run with a 3% Gemma-token sample (exact Gemma counting of every pair would take about 2 h); use more cores when available (`--src` order is the only state: not parallel inside one run) |
| pack prepare | about 12 M windows, 5 workers (old note: about 1,000 windows/s per process) | about 40 min |
| audit_packed | 4,000 samples per split | minutes |
| sampler plan / arms | pure numpy | seconds to a minute |
| pack replay (S2) | decode + re-encode of the specialist packs | about 10 min |

## What had to be assumed (differences to the old scripts, all deliberate)
1. **Source order.** The old windows used the file order of `pairs_final`. The D2 output keeps the row order of every source file (dedupe keeps the first occurrence; `data/` = one shard group per input file in the original order; `specialist/` = main rows first, then external sources, each in its file order). Consecutive pairs of one run (language, source, direction) are therefore consecutive source rows, with gaps where rows were removed by the filters. The old audit already found that most sources are not in document order (14 of 170 weakly), so a window is a pseudo-paragraph with exact sentence-by-sentence alignment. No document order is assumed or added.
2. **Run key** = (iso, full `source_dataset`, flipped), not `source_dataset.split(":")[0]`: the specialist set has sources such as `weak-repo:<id>` whose base name would merge unrelated sources.
3. **Direction.** The D2 output is English -> X only (`flipped` records that a pair was read X -> English in its source); the old `x2en` rows are not built.
4. **Cross-directory duplicates** (specialist repeats main rows) are removed by `key` before windowing; the first directory in `--src` wins.
5. **Tokeniser for window budgets:** `tokenizers` JSON of Davlan/afro-xlmr-large-76L (identical ids to the old transformers tokenizer; +2 specials as before).
6. **S1/S2 packs.** `sampler.py arms` makes S1 an exact subsequence of A0's seeded draws, so every specialist language keeps A0's passes. S2 keeps the same pack list through `pack.py replay` (decode with the source tokenizer, re-encode with the new one through `gemma_mt.format.build_example`; selftest: same tokenizer -> byte-identical tokens). **The trainer reads a schedule file** (done 2026-10-28, agent T1): `schedule_file: sched/<arm>/schedule.npy` in `train_sft.py` (int64 `[steps, packs_per_step]`, sidecar `schedule.json` pinning sha256 and the pack structure; `embed_only_steps` / `embed_only_min_row` for the S2 warm-up). `pack.py replay` now records the new tokenizer's fingerprint. The S2 start checkpoint must be rebuilt from google/gemma-3-270m-it with `tokswap_swap_vocab.py surgery` (the 2026-10-25 artifact was built from the old SFT-final). Derived schedules: `sampler.py select` (G0), `subset --frac 0.25 --floor-draws 100` (the one A2 subset shared by A2-base/up6/up12/1b), `head` (300-step sanity pair); configs in `gemma_mt/configs/clean_rerun_sft/`. Upcycled A2 arms: `scripts/upcycle_layers.py ... --post-norm-gain 0.01` (decision 2026-10-30).
7. QE gate rule CAL, `H_cal` and the pooled median offset are defined in `gate.py` / `hcal.py` (the pre-registration says only that H is split again by sha256 parity; the exact rule is fixed in `hcal.py` and the list is in `protected_index/heldout_split_cal_sel.json`).

## CPU QE job (started 2026-10-29 on box 2) and how the GPU run continues from it
* **Stable pair id** = `key` = xxh64(english.casefold, iso, target.casefold) (the `key` column of `out/kept`; `qe_score.py`, `gate.py`, `windows.py` use the same id).
* **Queue** (`qe_cpu_queue.py`, on the box in `qe_cpu/queue/`, 500-pair shards `q-NNNNNN.parquet`, `queue_manifest.json`, `queue_meta.json`): inside every language the order is a seeded hash order, so any prefix is an unbiased random sample. Stages in queue order: P1a (500 pairs per language, round-robin), P2a (all pairs, cap 50k, of nus kbp dyu bam kea), P1b (ranks 500-1999 of every other language), P2b (fon ewe wol seh kmb vmw kam, then ascending by pair count through the 20 specialist languages). The REF stage (H_cal references) was removed from the queue: those QE scores come from the calibration agent (`/workspace/calib_r4/qe_work/africomet/<iso>.human.json`, split `heldout_split_cal_sel_v2.json`, article level, H_cal 261 / H_sel 296); `hcal.py` and rule CAL use v2, the v1 split (282/275) is superseded.
* **Running job:** tmux session `qe_cpu` windows `s0` and `s1` (`chain/qe_cpu_run.sh 0|1`: two processes, cores 4-5 and 6-7, fp32, CUDA hidden, `taskset -c ... nice -n 19 ionice -c3`, backs off above load 20), window `push` (`qe_cpu_push.sh`: every 2 h to the Hub folder `qe_scores_cpu/`, keys and scores only). Score shards: `qe_cpu/scores_cpu/queue__q-NNNNNN-0000.parquet` with columns key, qe, scorer, model_sha, device=cpu, dtype=fp32, ts. Live status: `python chain/qe_cpu_status.py` (writes `qe_cpu/status.md`), raw progress `qe_cpu/progress_0.json`, `progress_1.json`, logs `qe_cpu/qe_cpu_0.log`.
* **GPU resume:** fetch `qe_scores_cpu/scores/` into `qe_cpu/scores_cpu/` (or any directory) and run `qe_score.py --src data data_devhalf specialist --out qe_cpu/scores_gpu --half --device cuda --skip-scored qe_cpu/scores_cpu ...`: every key that already has a CPU score is skipped; shards from both devices are interchangeable (`gate.py --scores qe_cpu/scores_cpu qe_cpu/scores_gpu`, the first score per key wins). Before the bulk run do the 500-pair cross-check (`qe_crosscheck.py`: the first queue shard q-000000 scored on CPU fp32 is re-scored on the GPU; pass = Pearson >= 0.999 and mean |diff| <= 0.003); if it fails, re-score the CPU part on the GPU and do not mix devices.
* **Rule CAL with the calibration agent's references:** `gate.py fit ... --refs-json-dir /workspace/calib_r4/qe_work/africomet` reads `<iso>.human.json` (scores of the 261 H_cal references); languages without a file use the pooled median offset until the GPU has scored them.

## Commercial-use rule (Amendment 4, 2026-10-29) and the fresh-instance runbook
Rows of licence class `non_commercial` never enter the chain: `windows.py` drops them BEFORE the cross-directory dedupe (`--include-nc` exists for tests only; `stats.json` reports `dropped_nc_by_iso`), `qe_score.py --exclude-nc` does not score them, `gate.py fit/apply --exclude-nc` leaves them out of the thresholds and the keep list, `nc_filter.py report | guard-windows | write-clean` reports and guards (the guard fails if a window carries the class), `common.assert_clean_src` refuses quarantine/kept/meta directories. `data_devhalf/` is FLORES+ dev (class share_alike). The whole sequence for a new instance (bootstrap, storage, idempotent `run_chain.sh`, cross-check, gate with OLD and CAL side by side, schedules A0/S1/S2/G0/A2, private push) is in `../fresh/README_RUNBOOK.md`. H_cal/H_sel: the v2 split (`heldout_split_cal_sel_v2.json`).

## Update 2026-10-30 (owner evening decisions; details: fresh/README_RUNBOOK.md and Addendum 5 of reports/sft_clean_rebuild_2026-10-28.md)
* `--nc-allow REPO[=tag]` (qe_score, gate, windows, nc_filter report/guard): NC sources that stay (IbomMT, tag `nc_exception=ibom-mt` in every window).
* `nc_hidden.py build-index | report | apply | merge-keep`: hidden-NC filter (rules R1 licence text, R2 NC-origin pairs; whole sources only listed); never leaves a language below 2,000 pairs silently. `hidden_impact.py`: the laptop-side impact table from the build tables.
* `qe_probe.py`: throughput probe for 1/2/4 scoring processes (+ batch sizes) -> `qe_procs.json`; `qe_score.py` DONE line has `elapsed_s`.
* G0 = `sampler.py select --languages amh,tir`, A2 = `sampler.py subset --frac 0.25 --stratify language` (one file for the four A2 arms); `arms_extra.py` is gone.

## Update 2026-10-30 (Amendment 6)
* `extra_sources.py`: nde / ful / knc / tzm pairs (download, standard schema, dedupe, STRICT protected-set filter) -> an extra pair directory read by windows / QE / gate like data/ (run_chain.sh steps 15 and 34). `--estimate` = laptop counts with an R1-only estimate.
* `common.SPECIALIST` = 20 + `NEW_LANGS` (nde ful knc tzm); `common.owner_tag` / `OWNER_TAGS` -> windows column `owner_exception` (AfroLingu-MT is kept and tagged, not dropped).
* `project_nc.py --extra-bins DIR` adds the four languages (window statistics from `windows.py --stats-only` on the estimate output + `project.py bins`).

## Update 2026-10-30 (QE cap and fast scorer; RTX 4500 Ada, 129 pairs/s measured)
* `qe_score.py --cap-per-lang N --cap-counts cap_counts.json`: deterministic MONOTONE hash sample (`(key & 0xFFFFFFFF) < ceil(N * 2^32 / pairs_of_language)`); `qe_cap.py count|table|check`; `common.cap_thresholds / in_cap / cap_counts`. A pair that is not scored is not kept (gate and windows already work on the keep list); `gate.py` and `fresh/sanity_after_gate.py` report pairs before cap / scored after cap / kept after gate.
* `qe_fast.py` (`FastQE`, `check`): same model, separate length-sorted source / translation passes, token-budget batches, English-source cache; `qe_score.py` uses it by default (`--no-fast` for comet predict) and checks its first chunk against predict (Pearson >= 0.9999, else fallback).
* BUG FIXED in `qe_score.py`: after a restart (`--skip-scored`) chunk numbers began at 0 again and a chunk whose file name already existed was silently NOT written (those pairs stayed unscored); chunk files now carry a per-run tag.
* `project_nc.py --cap N` projects the capped data.
