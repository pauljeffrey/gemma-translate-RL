# Pipeline: from sources to a trained model

All paths are relative to the repository root. Environment variables come from `.env` (see `.env.example`).
`PYTHONPATH=.:data_pipeline` makes both `gemma_mt` and `sft_clean` importable.

## 1. Sourcing and cataloguing the data

* `data_pipeline/catalog_crawl/crawl_hf.py`, `crawl_detail.py`, `analyze.py`: list Hugging Face datasets per African language, read their cards, count rows,
  classify the licence and the provenance (human, machine-translated, LLM-generated).
* `data_pipeline/sft_new/`: first-generation builders. One converter per source (`converters.py`, `ext_*.py`) maps a source to a common table;
  `dedup.py`, `langdetect.py`, `license_scan.py`, `compose.py`, `combine.py` deduplicate, verify the language and licence, and combine.
* `data_pipeline/sft_clean/catalog_plan.py`, `crawl_plan.py`, `ext_catalog.py`, `ext_big.py`: plans for what a catalogue audit says is still missing
  (report: `experiments/reports/catalogue_audit_2026-10-31.md`).
* `data_pipeline/sft_clean/chain/nllb_probe.py`, `nllb_intake.py`: quality probe and capped intake of the NLLB mined bitext;
  `intake2_pairs.py`, `intake2_instr*.py`, `intake2_aya*.py`: parallel pairs, instruction/chat/QA data, and the aligned part of the Aya collection.

## 2. Decontamination

`data_pipeline/sft_clean/fetch_protected.py` builds the **protected index**: every sentence of the dev, devtest and test splits of every benchmark we evaluate on.
`decon.py` removes any training pair whose English or African side matches (exact sentence, contained sentence, normalised match).
Reports: `experiments/reports/benchmark_contamination_audit_2026-10-26.md`, `sft_clean_rebuild_2026-10-28.md`.

## 3. Scoring and gating (the "chain")

`data_pipeline/sft_clean/chain/run_chain.sh` runs numbered steps; each writes `state/NN_name.done`:

| Step | Script | What |
|---|---|---|
| 12-15 | `extra_sources.py` | fetch pairs and extra sources, rule-based cleaning |
| 20-23 | `nc_filter.py`, `nc_hidden.py` | licence reports; find sources that hide a non-commercial origin |
| 32-34 | `qe_score.py` | AfriCOMET-QE-STL score of every pair (fp16, several processes per GPU, resumable by shard) |
| 40-43 | `gate.py` | fit a per-language threshold: the 10th percentile of the QE of **human reference** sentences (pooled fallback for languages without references), apply, remove hidden-NC sources |
| 50-51 | `windows.py` | group consecutive same-source pairs into token-balanced windows (so long inputs and several sentences are learned) |
| 60 | `pack.py` | pack windows into sequences of 4,096 tokens with a per-sample loss mask |
| 70-74 | `sampler.py` | UniMax schedules (each language seen at least once, small languages up to four times), stitched continuations |

`data_pipeline/sft_clean/fresh/` holds the scripts of the later operations: `finish_all.sh` (gate, windows and packs after scoring),
`post_all.sh` (retag, extend, schedules), `make_stage_schedules.py`, `make_shards.py`, `add_replay_to_shard.py`, `bucket_sync.py` (large transfers through object storage),
`launch_growth.sh`, `launch_shards.sh` (training launchers with watchdog behaviour).

## 4. SFT

```bash
python -m gemma_mt.train_sft --config configs/clean_rerun_sft/a0u_up6.yaml
```

Packed data with block-diagonal attention (flash varlen), a fused chunked loss for the 262k vocabulary, Muon on the 2-D decoder matrices and AdamW on embeddings
and norms, a schedule file that fixes the packs of every step (sha256 pinned in a sidecar), checkpoints every 20 steps with optimizer state, optional Hub push.
Design notes: `DECISIONS.md` (D-01 to D-12), `docs/MODULE_README.md`.

## 5. Growth

See `docs/GROWTH.md`.

## 6. RL

See `docs/RL.md`. Entry point: `python -m gemma_mt.rl.train_rl --config configs/rl.yaml`.

## 7. Evaluation

`gemma_mt/eval/`: benchmark adapters (FLORES+, NTREX-128, MAFAND-MT, SMOL, SALT, IbomMT, FLORES-MOZ), chrF++ and AfriCOMET scoring, a long-document benchmark.
`scripts/` has the report builders (`build_final_per_language.py`, `model_comparison_*`).

## 8. Language ID and fluency (reference-free signals used by the RL guards)

```bash
python data_pipeline/sft_clean/fresh/lid_measure.py --shard-prefix <prefix> --per-lang 800 --out lid_out      # GlotLID precision and recall per language
python tools/language_id/train_fasttext_lid.py --train a.tsv --test b.tsv --out lid.ftz                       # 1 MB classifier for languages without a GlotLID label
python scripts/fluency_lm.py validate --train a.tsv --test b.tsv --out fluency_validation                      # character n-gram fluency model + validity test
```
