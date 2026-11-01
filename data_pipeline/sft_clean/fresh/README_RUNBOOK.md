# Runbook: the clean-SFT chain on a FRESH instance (written 2026-10-29, updated 2026-10-30 for the owner's evening decisions and Amendment 6; laptop only, nothing here has been run on the new instance)

Goal: from the clean pair set on the Hub (`<HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28`: `data/`, `data_devhalf/`, `specialist/`, `qe_scores_cpu/`) to **gated, commercial-only packed SFT data and the schedules of A0, S1, S2, G0 and A2**, pushed to a private repo.
Owner decisions applied (Amendment 4, 2026-10-29, plus the evening update): rows of licence class `non_commercial` are removed everywhere (31,209 amh pairs: AfriDocMT, opus, AfricanFineTranslations, polynews) **except IbomMT (`howard-nlp/ibom-mt`, efi 994 + ibb 998 pairs), an owner exception kept with `--nc-allow howard-nlp/ibom-mt` and tagged `nc_exception=ibom-mt` in every window, so it can be removed again with one flag**; **Amendment 6 (2026-10-30): AfroLingu-MT is KEPT everywhere (windows tagged `owner_exception=afrolingu-mt`; the hidden-NC filter keeps only its per-row rules R1/R2) and four languages are added to A0 and the specialist set: nde, ful, knc, tzm (section 10; S1/S2 now train on 24 languages, fat and gaa are skipped)**; a hidden-NC filter (section 6) drops rows whose NC origin is identifiable per row, whole unsplittable sources are only listed; unverified classes stay (unspecified, other/multi-open, mt560 incl. the specialist set); the QE gate rule is CAL (FLORES-calibrated on H_cal, split v2) and is applied after the OLD and CAL keep rates were shown; GPU scoring, the CPU shards pushed from box 2 are reused.

## 0. What you need (from the lead)
SSH access to the instance, and `HF_TOKEN` (read of the clean repo and the gated FLORES+ dev; write to the private pack repo): the lead places it in the instance's `env.sh` (never committed, nothing in the scripts is hardcoded; the scripts only read `$HF_TOKEN`). Disk: **at least 50 GB free (60 GB comfortable)**, RAM >= 16 GB (key sets of 24 M pairs), one GPU (RTX PRO 6000 96 GB assumed).

## 1. Commands
```
# laptop
cd "/path/to/repo/data_pipeline/sft_clean/fresh"
SSH_TARGET=root@HOST SSH_PORT=PORT SSH_KEY=~/.ssh/<key> ./sync_code.sh            # chain code + gemma_mt package + the v2 H_cal split (no data, no .env, no git)
# instance
cp $ROOT/code/sft_clean/fresh/env.example $ROOT/env.sh; nano $ROOT/env.sh; source $ROOT/env.sh          # paste HF_TOKEN here only
bash $ROOT/code/sft_clean/fresh/bootstrap.sh                                      # venv_data (numpy 2, transformers 5), venv_comet (torch CUDA, unbabel-comet 2.2.7, numpy<2, pyarrow<20); WITH_VLLM=1 adds venv_vllm (vllm 0.31.0, cu130)
tmux new -s chain
STOP_AFTER_FIT=1 bash $ROOT/code/sft_clean/fresh/run_chain.sh                      # steps 00-40: download, GPU probe, QE (+ side jobs), NC + hidden-NC reports, gate FIT (nothing dropped)
less $ROOT/work/gate/keep_rate_per_language.csv                                    # read both keep rates; owner rule: apply CAL
less $ROOT/work/nc_hidden_report/nc_hidden_by_language_source.csv                  # the hidden-NC impact table (per language x source, pairs + tokens BEFORE dropping)
bash $ROOT/code/sft_clean/fresh/run_chain.sh                                       # continues: apply CAL, sanity, windows, NC guard, pack, schedules, push
```
`run_chain.sh` is idempotent: every step writes `state/<step>.done` and `logs/<step>.log`; finished steps are skipped; `FORCE=<step>` re-runs one, `STEPS="a b"` runs only those; `logs/timing.tsv` has the wall time of every step; `logs/disk.log` is written every 60 s by `disk_watch.sh`; a step refuses to start when less than its listed free space is available (`[STOP] ... needs about N GB`). A failure prints `[FAIL]` and exits; fix and run the same command again.

| step | what | notes |
|---|---|---|
| 00_preflight | GPU + both environments | |
| 10_fetch_tokenizers | Davlan/afro-xlmr-large-76L tokenizer.json (window budgets) | the Gemma tokenizer is fetched by prepare_sft through HF_HOME |
| 11_fetch_cpu_scores | `qe_scores_cpu/scores/*` -> `work/scores_cpu` | CPU fp32 shards from box 2 (about 12-25k pairs); reused by `--skip-scored` |
| 12_fetch_pairs | `data/`, `data_devhalf/`, `specialist/` | 3.2 GB (`extra/` is built locally by step 15: about 0.15 GB) |
| 05 (inside run_chain) | starts `vram.log` (nvidia-smi every 30 s) and `gpu_jobs_runner.sh` (side GPU jobs, section 7) | |
| 15_extra_sources | `extra_sources.py` (section 10): downloads the sources of nde ful knc tzm, standardises, dedupes, applies the STRICT protected-set filter (index fetched from the Hub folder `protected_index/`, 2.3 GB, deleted afterwards) -> `hub/extra/` (a normal pair directory, last in `$SRC`) | runs before everything that reads pairs; needs ~8 GB free while it runs; `EXTRA_LANGS`, `FUL_MIN_SIM`, `EXTRA_ALLOW_MISSING` in env.sh |
| 16_cap_counts | `qe_cap.py count`: unique scoreable pairs per language, written ONCE to `work/cap_counts.json` (never overwritten; `--force` to recount) and the cap table (pairs to score and hours at `QE_RATE` for caps 100k/150k/200k/300k) | the thresholds of the QE cap come from this file, so a later `QE_CAP` increase scores a superset |
| 20_nc_report | `nc_filter.py report --nc-allow ...`: NC rows per language/source, `nc_kept_by_owner_exception.csv`, plus a list of rows whose licence text looks NC but whose class is not (CHECK it) | CPU only, runs beside the QE |
| 21-23 nc_hidden_* | `nc_hidden.py build-index` (hashes of MAFAND-MT all splits from GitHub + `fiifinketia/pristine-twi-english`), `report` (the impact table), `apply` (keys of rows with identifiable NC origin; **refuses** if a language would end below 2,000 pairs: say so to the owner, or `HIDDEN_ALLOW_BELOW=1`) | CPU only, beside the QE; `HIDDEN_NC=0` disables; `HIDDEN_DROP_SOURCES="repo ..."` = owner decision to drop a whole source |
| 24_qe_fast_check | `qe_fast.py check` on the GPU: 800 pairs scored by comet predict and by the fast scorer; **Pearson must be >= 0.9999**, speeds of both printed; a failure creates `state/qe_fast_failed` and the chain continues with `--no-fast` | ~2 min; `QE_FAST=0` skips it |
| 25_qe_probe | `qe_probe.py`: scores 3,000 different pairs per process with 1, 2, 4 processes (batch 128 and 256), polls VRAM, writes `work/qe_probe/qe_procs.json`; a bigger configuration must be >= 5% faster to win; stops trying bigger ones above 80% VRAM | 3-6 min; `QE_PROCS=3` fixes N and skips it |
| 30_crosscheck | 500 pairs (queue shard q-000000, recovered **by key** from the pair shards because the queue text was on the lost box) CPU fp32 vs GPU fp16; **fails the chain if Pearson < 0.999 or mean abs diff > 0.003** | if it fails: do not mix devices (delete `work/scores_cpu`, re-score everything on the GPU) |
| 31_hcal_refs | FLORES+ dev H_cal references (split v2, 261 ids) of every language with a dev file, scored on the GPU; the benchmark text is deleted right after scoring, only key + iso (hashes) stay in `work/refs_keys` | local only, never pushed |
| 32_qe_main / 33_qe_specialist / 34_qe_extra | `qe_score.py` x N processes (N, batch from the probe), **`--cap-per-lang $QE_CAP --cap-counts work/cap_counts.json`** (default 150,000 pairs per language, 0 = no cap), the fast scorer (first chunk of every run cross-checked against comet predict, automatic fallback), fp16, `--exclude-nc --nc-allow ...`, `--skip-scored` (CPU + own shards), specialist and extra skip the keys already scored in data/ | resumable per chunk of 100k pairs; prints pairs/s and the ETA at the assumed 1,100/s; **30_crosscheck and 31_hcal_refs run beside it** (background groups), the chain waits for them before the gate |
| 40_gate_fit | `gate.py fit --exclude-nc --nc-allow ...`: OLD and CAL side by side; `--scores` takes the CPU and GPU directories | nothing is dropped |
| 41_gate_apply | `gate.py apply --rule $GATE_RULE` (default cal) -> `keep_cal.parquet` | |
| 42_keep_minus_hidden | `nc_hidden.py merge-keep`: gate keep list minus the hidden-NC keys -> `keep_final.parquet` | |
| 43_sanity | `sanity_after_gate.py` (on keep_final): pairs per language before/after the gate, the table of languages under 2,000 pairs (`work/gate/sanity_after_gate*.csv`) | read it before packing |
| 50_windows / 51_nc_guard | `windows.py --keep-file keep_final --nc-allow ...` (drops NC rows before dedupe; IbomMT windows carry `nc_exception=ibom-mt`), then `nc_filter.py guard-windows` (fails on an UNTAGGED non_commercial window) | |
| 60_pack | `pack.py prepare` = the old prepare_sft command (4,096 tokens, one language per pack, same chat format and loss mask, val fraction 0.002, 60 val windows per language, pcm decontaminated) + `audit_packed` | |
| 70-74 | `sampler.py plan` stage A (A0) and stage B (B0), `sampler.py arms` (S1, S2), **`sampler.py select --languages amh,tir --name G0`** (G0 = A0's draws of the amh/tir packs), **`sampler.py subset --frac 0.25 --stratify language --name A2`** (T1's commands; no fallback and no `arms_extra.py`: if a command is missing the step fails) | ONE A2 file (`work/sched/A2/schedule.npy`, one sha256 in its sidecar) is shared by A2-base, A2-up6, A2-up12 and Gemma-3-1B-it; S2's re-tokenised packs: later, `pack.py replay --like work/packed --out work/packed_S2 --tokenizer <extended tokenizer>` |
| 90_push | `push_packs.py`: packs, schedules, gate outputs, GPU scores, logs, code -> the PRIVATE repo `$PACK_REPO` | no pair text, no windows, no benchmark text |

## 2. Storage plan (sizes from the build tables; ESTIMATES marked ~)
| after step | adds | on disk (cumulative) |
|---|---|---:|
| bootstrap | venv_data ~1.4 GB, venv_comet ~8 GB (CUDA torch) [venv_vllm ~10 GB only with WITH_VLLM=1] | ~9.4 GB |
| fetch (11, 12) | pair shards 3.2 GB (data 2.6, specialist 0.6), CPU scores ~0.01 GB | ~12.6 GB |
| 30-33 QE | AfriCOMET checkpoint + XLM-R encoder in the HF cache ~4.5 GB; GPU scores ~0.4 GB (23.8 M keys + qe + short strings, zstd) | ~17.5 GB |
| 40-42 gate | keep list ~0.16 GB, csv < 0.01 GB | ~17.7 GB |
| 50 windows | window text ~3.5 GB (same text as the pairs) | ~21.2 GB |
| 60 pack | prepare_sft writes a raw token store (~8.1 GB: 2.03 B sample tokens x 4 B, commercial + CAL estimate; 8.7 GB without the gate) and then the packed file (~8.1 GB): **peak +16.2 GB**, then the raw store is deleted | **peak ~37 GB**, after pack ~29 GB |
| 70-74 schedules | A0, B0, S1, S2, G0, A2: each `schedule.npy` (steps x 32 x 8 B = 4 MB) + `pack_lang.npy` (4 MB): < 0.1 GB | ~29 GB |

What may be deleted (only after the push in step 90 is verified on the Hub; the order is also the safe order): `work/windows_sft` (3.5 GB, regenerable from the pairs), `hub/` pair shards (3.2 GB, on the Hub), `$HF_HOME` AfriCOMET cache (4.5 GB, unless more scoring follows), `work/_hub_scores`. **Keep**: `work/packed` (~8.1 GB, the training input), `work/sched`, `work/gate`, `work/scores_gpu` and `work/scores_cpu` (0.4 GB, needed to reproduce the gate), `logs/`. Final footprint without venvs: about 9 GB.

## 3. Expected time and cost (estimates; every step logs its real time in `logs/timing.tsv`)
**The instance is an RTX 4500 Ada (24 GB), not the Pro 6000.** Measured: 129 pairs/s with comet predict (1 process, batch 128, 18 GB VRAM peak; 2 processes no gain). Scoring all 24.4 M pairs would take about 52 h. Therefore the QE is capped per language (`QE_CAP`, default 150,000; deterministic hash sample, monotone; thin languages in full; **a pair that is not scored is not kept**):

| cap per language | pairs to score | languages capped | hours at 129 pairs/s | hours at 210 / 280 pairs/s (expected with the fast scorer) |
|---|---:|---:|---:|---:|
| 100,000 | 5.68 M | 45 | 12.2 | 7.5 / 5.6 |
| **150,000** | 7.80 M | 40 | 16.8 | 10.3 / 7.7 |
| 200,000 | 9.68 M | 34 | 20.8 | 12.8 / 9.6 |
| 300,000 | 12.49 M | 25 | 26.9 | 16.5 / 12.4 |
| none | 24.41 M | 0 | 52.6 | 32.3 / 24.2 |
(laptop estimate from `qe_cpu/queue_meta.json` + the four new languages; the instance prints the exact table in step 16 and `logs/qe_cap_table.txt`.) The cap shrinks the training set: with cap 150,000 the commercial CAL projection is 3.14 M windows, 393 M Gemma target tokens, 163,618 packs, A0 5,113 steps, S1/S2 1,391, A2 (25%) 1,278 (uncapped: 9.89 M windows, 1.20 B tokens, 503,024 packs, A0 15,719, S1/S2 3,914, A2 3,929); `reports/data/sft_clean_2026-10-28/bins_cap_<N>/` has the four caps with the per-language table (pairs before cap / after cap / after gate). Raising the cap later: set a larger `QE_CAP` and re-run 32-34 (only the new keys are scored: `--skip-scored`), then 40 onwards with `FORCE`.
**Fast scorer** (`qe_fast.py`, same model and scores): separate length-sorted source and translation passes, token-budget batches, English-source cache. On real pair samples (ful 100k, nde, tzm) comet predict pads to 1.4-1.9 times the real tokens (the translation side 1.6-2.2 times, because it sorts by the source only), the fast scorer to 1.01-1.11: **1.3-1.85 times fewer padded tokens, so expect roughly 1.5-1.8 times more pairs/s if the GPU is compute-bound (210-230/s), and more if English sources repeat across languages** (cache hit rate is printed at the end of every run; it was 0% within one language of the extra sources). Not done on purpose: truncation to 256 tokens (saves 0.1% of the tokens, changes scores), bf16 (same throughput as fp16 on this card, lower precision), torch.compile (variable shapes), SDPA (the model already runs `sdpa`). Everything else of the old estimate (windows 73 min on 2 cores, pack 20-30 min, push 10-20 min) still holds; the QE of the four new languages is inside the cap table.

## 4. After the run (the lead's sanity)
`work/gate/sanity_after_gate.csv` (every language) and `sanity_after_gate_under_2000.csv`; compare with the projection in `reports/data/sft_clean_2026-10-28/bins_commercial/languages_under_2000_pairs_ESTIMATE.csv`; `work/windows_sft/stats.json` (`dropped_nc_by_iso` must equal `nc_report/nc_rows_by_language.csv`); `work/packed/meta.json` (packs, tokens, efficiency); `work/sched/*/schedule.json` sidecars (hashes the trainer checks).

## 5. Risks
1. **NC-origin text in the "unverified" classes** (kept by the owner). The hidden-NC filter (section 6) catches what is identifiable per row; the big whole sources (mt560 16.2 M pairs, multi-open 3.7 M, AfroLingu-MT 0.21 M, eng_twi 0.55 M) cannot be split and are only listed in `nc_hidden_report/`.
2. **Rule CAL thresholds are not final.** The partial CPU fit (about 155 pairs per language) gave keep rates 83-95% on the specialist languages; the final numbers need the GPU scores and the H_cal references (step 31 recomputes them on the GPU; a CPU-fp32 vs GPU-fp16 difference of < 0.003 is accepted by the cross-check).
3. The cross-check pairs are recovered by key from the Hub pair shards; if the CPU shard `queue__q-000000-0000.parquet` is missing from the Hub the step fails (then skip it and re-score everything on the GPU: delete `work/scores_cpu`).
4. Several QE processes on one GPU: if the probe picks 4 and a worker is killed (OOM), `qe_score.py` is resumable per chunk, re-run the same command. The loader workers use `nproc / N` CPU cores each; on a small instance the CPU, not the GPU, limits the rate (the probe shows it).
5. Disk: the pack step needs 16 GB on top of everything else; the guard stops the chain instead of filling the disk.
6. S2 (re-tokenised packs) needs the extended tokenizer, not available yet; `pack.py replay` is tested only on the same-tokenizer identity.
7. `train_sft.py` and `rl/` were not touched; the trainer's `schedule_file` option and sidecar check are agent T1's work and must match `sampler.py` sidecars. The A2 floor (`A2_FLOOR`, `--floor-draws`) is T1's decision; the default is none.
8. Windows are built from the clean pairs: ibb stays with its 998 IbomMT pairs (below 2,000 anyway, about 950 after the gate) and efi keeps its 994 (about 325 k pairs in total with the other sources); nbl (188 pairs) and men (about 1,600) stay below 2,000; bbj, dag, fat, gaa have no data (fat and gaa are skipped by the owner; nde, ful, knc, tzm now have data from step 15, section 10).
10. **Amendment 6 sources**: the NLLB-mined Fulah set is 92% of ful and NLLB-mined text has unknown rights (owner-approved, tag `nllb_mined`); after the QE gate it may keep far less than the estimate (sensitivity in Addendum 6). KanuriMT adds almost nothing (it merges Gamayun and TICO-19). The nde alignment depends on the verse ids of two eBible files (both 31,103 lines); the Adamawa Bible is a different Fulfulde dialect under the `ful` tag. If a source URL has moved, step 15 fails with the source name; `EXTRA_ALLOW_MISSING` continues without it.
9. The IbomMT exception is a licence exception of the owner for a non-commercial dataset: the windows carry `nc_exception=ibom-mt`; remove them by re-running 50_windows without `NC_ALLOW`.

## 6. Hidden-NC filter (owner request 2026-10-29 evening)
Rules (`chain/nc_hidden.py`), applied PER ROW and only where the origin is identifiable:
- R1 the row's own `licence_raw` / source text contains an NC marker (non-commercial, CC-BY-NC, NC-SA, Tanzil, TED, ...);
- R2 the pair is in a corpus whose licence is NC: MAFAND-MT (CC-BY-NC-4.0; all splits fetched from GitHub `masakhane-io/lafand-mt`) and `fiifinketia/pristine-twi-english` (NC upstream); exact normalised pair match (NFKC, casefold, punctuation-insensitive);
- R3 whole sources that carry no per-row origin (AfroLingu-MT "gated, non-commercial use only, 43 datasets", multi-open "other", eng_twi, the mt560 OPUS mixtures) are **listed, not dropped**; `HIDDEN_DROP_SOURCES="UBC-NLP/AfroLingu-MT"` makes the owner decision.
`nc_hidden_report/nc_hidden_by_language_source.csv` gives per language and source: pairs, estimated Gemma target tokens, R1 and R2 hits, what the default removes, whether it is a whole-source candidate and what the language keeps if that source were dropped (and whether that is under 2,000 pairs). **No language is left below 2,000 pairs silently**: `23_nc_hidden_apply` refuses and names the languages. Laptop-side estimate before the run: `reports/data/sft_clean_2026-10-28/bins_commercial/hidden_nc/`.

## 7. Using the whole GPU
- **Bulk QE**: N processes (probe) x batch (probe). Check `logs/vram.log` (30 s samples) and `logs/qe_procs.json`.
- **Side GPU jobs** (calibration agents, NLLB pseudo-reference precompute, SSA-COMET, bigram table, the H_cal reference QE is already built in): put one file per job into `$ROOT/gpu_jobs.d/NAME.job`:
  ```
  # vram_gb: 12
  # after: 13_fetch_split        (optional: state/<step>.done that must exist)
  cd $ROOT/work && $ROOT/venv_comet/bin/python ... 
  ```
  `gpu_jobs_runner.sh` (started by run_chain.sh, KEEP_ALIVE=1) starts a job when `vram_gb` + 12 GB (GPU_RESERVE_GB) are free, runs it once (`state/gpujob_NAME.done|failed`, `logs/gpujob_NAME.log`) and keeps the QE untouched. Jobs must be idempotent. To restart the runner after dropping files: `nohup bash $FR/gpu_jobs_runner.sh >> $ROOT/logs/gpu_jobs_runner.log 2>&1 &`.
- **Packing-independent steps** (CPU only, run during the QE): 20_nc_report, 21-23 hidden-NC, the cross-check and H_cal references (GPU, small).

## 8. Two SFT jobs on the same GPU (and how to measure it)
`fresh/sft_concurrent.sh NAME=CONFIG[,key=value,...] ...` starts several `python -m gemma_mt.train_sft` jobs on the one GPU (own `out_dir` and log each, 120 s stagger, VRAM check: the next job starts only when free VRAM >= 1.25 x the VRAM per running job; `USE_MPS=1` starts the CUDA MPS daemon). To MEASURE whether it pays:
```
cd $ROOT/work     # holds packed/ and sched/ (A0 from step 70); configs use relative paths
PYTHONPATH=$ROOT/code:$ROOT/code/sft_clean/chain PY=<training venv python> bash $ROOT/code/sft_clean/fresh/sft_throughput_probe.sh configs/clean_rerun_sft/a0.yaml 80
```
It builds the first 80 steps of A0 (`sampler.py head`), runs ONE job, then TWO concurrent jobs (same schedule, separate out_dirs, no evaluation/checkpoints), and prints with `sft_probe_summary.py` the **combined steps/s and tokens/s** (steady state, first 10 logged steps skipped, from the `step` and `time` fields of each `metrics.jsonl`, which `train_sft.py` writes every `log_every` steps; the run's own `tokens_per_s` is in the same file). Rule: run two jobs at once if combined throughput is >= 1.15 x one job and the measurement windows overlapped. Repeat with `USE_MPS=1`, and by hand with 3 jobs (`sft_concurrent.sh a=.. b=.. c=..`); for the 1B config pass its config as the first argument. Two jobs also need two sets of data loader processes: watch `top` for CPU saturation.

## 9. A2 arms
`work/sched/A2` is built ONCE: `sampler.py subset --schedule work/sched/A0 --frac 0.25 --stratify language --seed 1234 --name A2` (25% of A0's draws, stratified per language, A0's order); A2-base, A2-up6, A2-up12 and Gemma-3-1B-it all use that file (`schedule_file: sched/A2/schedule.npy`), so the arms see exactly the same packs. The sidecar's sha256 is the proof.

## 10. Amendment 6: the four new languages (nde, ful, knc, tzm) and `chain/extra_sources.py`
- **Registry:** nothing to change. `gemma_mt/languages.py` already has the four as SFT-only rows (North Ndebele, Fulah, Kanuri, Tamazight; ids nde 35, knc 55, tzm 83, ful 84 in `ALL_ISOS` of 85 languages; `ALL_ISOS[-2:] == (tzm, ful)`), so no id is shifted and packs, schedules, sampler and calibration keep their indexes (`test_new_languages_registered_and_ids_stable`). They are not RL-eligible (no RL pools, no reward calibration). The language code stays plain text in the target (no tokenizer change). `common.SPECIALIST` is now 24 languages (the 20 + `NEW_LANGS`, appended), so `sampler.py arms` (S1/S2) trains on 24; the evaluation groups G, W, C are unchanged.
- **Sources** (all in `extra_sources.py`, priority = order): nde: eBible `nde` (Biblica, CC BY-SA 4.0) aligned by verse id to eBible `engwebp` (public domain); cross-check against `SalomonMetre13/nde_en` is written to `hub/extra/nde_crosscheck.json` (not trained on). ful: OPUS wikimedia v20260327 en-ff, OPUS XLEnt v1.2 en-ff, `stephanedonna/english_fulfulde` (Adamawa Fulfulde Bible, disclosure), `michsethowusu/english-fulah_sentence-pairs` (NLLB-mined, owner-approved, tag `nllb_mined`; streamed in row groups, rows with LASER margin `similarity` < `FUL_MIN_SIM` (default 1.06) dropped, the margin is kept in column `aux`). knc: CLEAR-Global/Gamayun-kits `kau-eng` kit5k (TWB terms, CC-BY-4.0), `IsahMBukar/kanurimt` (only what survives the dedupe against Gamayun and the strict filter). tzm: Tamazight-NLP NLLB-Seed `eng_Latn-tzm_Tfng.csv` (CC-BY-SA-4.0), OPUS translatewiki en-tzm, Tamazight-NLP Pontoon `en-tzm.tsv` (MPL-2.0). Not used: fat, gaa, the NLLB-mined tzm set, GATITOS.
- **What it does:** NFC + whitespace normalisation, the old builder's length / ratio / letter rules, placeholder strings dropped for translatewiki and Pontoon, exact dedupe by `key`, then the STRICT filter (protected-2026-10-28.3, both sides, R1/R2/R3). Output schema = the pair schema (english, target, iso, source_dataset `extra-<tag>:<detail>`, licence class, licence_raw, kind, source_family, flipped, key, aux) + `stats/per_language_source.csv`, `stats/removed_by_protected_set.csv`, `stats/summary.json`. Raw downloads are deleted as they are read (no raw copies; peak extra disk about 0.4 GB plus the 2.3 GB protected index during the step).
- **Disclosure:** windows carry `owner_exception=<tag>`: `afrolingu-mt`, `nllb_mined`, `nllb_seed`, `bible_derived` (nde, Adamawa), `twb_gamayun_terms`, `kanurimt_compilation`; `nc_filter.py report` writes `owner_exception_rows.csv`, `windows/stats.json` has `owner_exception_windows`.
- **Laptop estimate** (no protected index, exact-sentence R1 estimate only): `python extra_sources.py --estimate --sent-hashes protected_sentence_hashes.npz --out /tmp/extra_est` (numbers in Addendum 6).
