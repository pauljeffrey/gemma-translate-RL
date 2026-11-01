# LongCtx-MT: long-context benchmark and the hypothesis test (LC1, 2026-10-23)

(Report text returned by agent LC1, saved by the lead because the agent's sandbox refused report-style .md files. Tables: `paper/tables/t_longctx_*.md`; figures: `paper/figures/fig14..16_longctx_*`; data: `reports/data/longctx/`; Gemini plan: `reports/long_context_gemini_plan_2026-10-23.md`; scripts: `paper/scripts/make_longctx.py`, `paper/scripts/gemini_*`, `gemma_mt/scripts/longctx/`; box: `/workspace/jobs/lc1`.)

## 0. Verdict on the hypothesis H
H = "our 270M model translates documents up to ~2,048 tokens far better than other models, which do not support such inputs". **H is only partly true, and the part that holds is narrower than the paper's sentence.**

1. **Other models fail on whole-document input of 1,000-2,000 tokens, but "do not support" is wrong for some.**
   - NLLB-200 (600M, 3.3B): 1,024 positions, trained on inputs <= 512 tokens; its model card says "not intended for document translation". Input was cut at 1,024 tokens for 100% of ~2,000-token documents, 90% of ~1,000-token ones and 36% (dev) / 57% (test) of the AfriScience papers. Output is short and repetitive (median 260-360 tokens whatever the input length), source-sentence coverage 0.33-0.41. Hausa example, ~2,000-token document (reference 9,188 chars): NLLB-3.3B returns 800 invented, repetitive characters.
   - TranslatePsy-AfriSLM 0.8B/2B: 262,144 context, so they "support" the input on paper, but loop or never stop: finished 0.77/0.61 (0.8B, ~1,024/~2,048 tokens), 0.75/0.76 (2B), 0.21 (0.8B, AfriScience papers).
   - Qwen3.5-0.8B and Gemma-3-1B-it zero-shot do not really write these languages.
2. **Our native whole-document generation beats theirs clearly at >= 1,000 tokens, but only for the science-stage models** (A2W alpha 0.7 and the live run at step 250); the SFT-final / RL-final (2c) checkpoints of the paper's headline do not. Whole-document chrF++ vs human references at ~1,024 / ~2,048 English tokens (6 languages): live 39.6 / 28.2, A2W 39.4 / 28.8, RL-final 25.6 / 17.9, SFT-final 14.7 / 12.3. Others (whole-document): NLLB-600M 12.4 / 8.5, NLLB-3.3B 9.7 / 5.9, TranslatePsy-0.8B 11.7 / 8.1, TranslatePsy-2B 14.9 / 7.5. AfriScience papers (~900 tokens) dev / test: live 43.8 / 41.5 vs 8-23 for every other whole-document system.
3. **Even our best native result is far below what the other models get with a sentence-by-sentence wrapper.** Live model sentence-by-sentence (with best-of-4 repair): 52.6 / 55.5 (+13 / +27 over its own whole-document mode). Wrapped baselines: 53.9-56.0 / 56.6-58.7 (NLLB-600M, NLLB-3.3B, TranslatePsy-0.8B/2B): 1-4 chrF++ above ours at every length; on AfriScience papers 57.6-58.8 vs 55.4-55.6. "Far better than other models" holds for native long input only; as a statement about document translation quality it is false: a user of NLLB or TranslatePsy gets better output by splitting the text.
4. **What the native result still shows.** The science-stage models finish 95-100% of documents up to ~2,048 tokens and write the target language (GlotLID 0.94-0.99), but they omit text: source-sentence coverage 0.87 at ~1,024 and 0.70 at ~2,048 tokens, length ratio 0.78 / 0.56. RL-final finishes only 51% / 39% at 1,024 / 2,048 with loops in 67-90% of documents; SFT-final finishes 3-7%. Our own sentence wrapper (sentfix) has 100% coverage and works up to b4096.
5. **The pseudo-reference needs care.** TranslatePsy-0.8B scores 73.4 chrF++ against NLLB-3.3B but 56.0 against humans (+17.5); NLLB-600M +15.6, TranslatePsy-2B +19.1, ours +7.8 to +8.6. Against human references NLLB-3.3B beats our best model by >= 2 chrF++ only for amh, kin, mlg, zul; it is below ours for lug (-2.6), nso (-5.2), wol (-3.2), yor (-0.8).

**Suggested paper wording.** "Whole-document decoding beyond ~512 tokens fails for NLLB (input truncated at 1,024 tokens), for TranslatePsy-AfriSLM (repetition loops) and for our SFT/RL checkpoints. After the science stage our 270M model is the only system we tested that finishes and writes the target language on 1,000-2,000-token documents (chrF++ 40 / 28 on human references), although it omits 13-30% of the sentences. A sentence-by-sentence wrapper removes the failure for every system and gives higher scores (53-59 chrF++) than native long input, with the other systems 1-4 chrF++ ahead of ours." Do NOT write that the model translates long documents "far better than other models".

| claim in H | measured | verdict |
|---|---|---|
| other models do not support such inputs | NLLB truncated; TranslatePsy accepts the input but loops; Qwen/Gemma do not write the languages | true in practice, not by context length |
| our model translates documents up to ~2,048 tokens | only A2W / live finish >= 95%; coverage 0.70-0.87 | partly: stable but omits 13-30% |
| far better than other models | whole-document 39.6 / 28.2 vs <= 14.9 / 8.5; but 52.6 / 55.5 vs 53.9-56.0 / 56.6-58.7 for their wrappers | true for native long input only |

## 1. Benchmark definition (LongCtx-MT; seed 20261023; `lc_build.py`)
Items are document x language. Files `/workspace/jobs/lc1/work/{docs,items,refs_human,refs_nllb_core}.jsonl`; summary `reports/data/longctx/bench_summary.json`, `paper/tables/t_longctx_benchmark.md`.
- **AfriLongDoc core, real documents**: bins b256 (control; real documents only 60-224 Gemma tokens), b512, b1024, b2048 (20 documents each), b4096 (10); median tokens 120 / 518 / 1,035 / 2,057 / 4,109; clean (not in SFT/RL data), NLLB-valid in 12 languages; reference = NLLB-3.3B sentence-by-sentence pseudo-reference (beam 4, CT2 int8); no human reference.
- **SmolDoc human-reference set (12 languages)**: G1 = hau yor swh amh ibo mlg (181 clean documents in common), G2 = zul xho kin som wol twi (50); 60 short documents (121-527 tokens) plus stitched documents of ~512 / ~1,024 / ~2,048 tokens (38 / 33 / 16). Stitched = unrelated texts concatenated, so they test length robustness, not discourse; `sm` and stitched bins share English sentences.
- **AfriScience-MT** (6 languages amh hau lug nso yor zul): dev 25 papers (531-1,165 tokens), test 28 papers (577-1,370 tokens); the dataset's own sentence split for the sentence mode.
- **Hold-out / contamination**: core documents and AfriScience dev/test are outside our training data (exact + 8-gram). A2W alpha 0.7 and the live run were trained on the AfriScience-MT TRAIN split (in-domain advantage on the AfriScience sets); the live run also selects checkpoints on AfriScience dev, so its dev column is not independent. AfriScience TEST was scored for SFT-final, RL-final, A2W and the live step-250 checkpoint; treat the live row's test column as dev-like (no selection used it). Unknown: whether NLLB/TranslatePsy/Qwen/Gemma pretraining contains SmolDoc or AfriScience (TranslatePsy lists Smol among its evaluation sets). Our SFT/RL data used NLLB-style translations, so part of our agreement with NLLB is lineage.

## 2. Systems and documented context (`t_longctx_models.md`)
| system | context / limits | fits |
|---|---|---|
| SabiYarn-270M: SFT-final (step 27,638), RL-final 2c (step 1,433), A2W a0.7, live run (`latest` = step 250) | 32,768 positions; trained on windows <= 2,048 tokens; tested to 10,240 | all bins (b4096 outputs hit the 3.5x budget) |
| NLLB-200-distilled-600M, NLLB-3.3B | 1,024 positions; trained on <= 512 tokens; "not intended for document translation"; CT2 truncates input at 1,024 | b512 yes; b1024 no (90% cut); b2048 no (100%); AfriScience dev 64% fit, test 43% |
| TranslatePsy-AfriSLM 0.8B and 2B | 262,144; quality "not guaranteed for very long documents"; 19 languages, not twi/nso | all fit by context |
| Qwen3.5-0.8B zero-shot | 262,144 | all |
| Gemma-3-1B-it zero-shot | 32,768 | all |

Not run: Gemma-3-4B-it, TranslatePsy-4B (weights above the 8 GB cap), Qwen3.5-2B, AfriNLLB. Two decoding modes per system: whole document, and sentence-by-sentence (our models: unfinished sentences re-decoded by best-of-4 at T=0.7 = sentfix; no repair for the others). Settings: ours vLLM greedy, 3.5x token budget (R6 protocol); NLLB CT2 beam 4, input truncated at 1,024, output <= 1,024; TranslatePsy/Qwen/Gemma vLLM greedy, TranslatePsy prompt, thinking off, output budget 2.5x expected length. Compute-driven reductions: NLLB whole-document on hau yor swh amh zul ibo + the 6 AfriScience languages (no b4096); TranslatePsy-0.8B on hau yor swh amh zul ibo lug; TranslatePsy-2B on hau yor swh amh zul (stitched ~1,024/~2,048, b2048, AfriScience dev only); Qwen3.5-0.8B and Gemma-3-1B sentence-by-sentence only. Main tables use the common languages: hau yor swh amh zul ibo (core, SmolDoc); amh hau lug yor zul (AfriScience). Full data: `reports/data/longctx/cells.csv`, `macro.csv`.

## 3. Results
Metrics: document-pooled chrF++ (word_order 2) and spBLEU (flores200), macro over languages; finished rate, loop rate (word 3-gram), source-sentence coverage (alignment plan of the long-document reward v2), length ratio, GlotLID share; 95% bootstrap CIs over documents (300 draws, shared across systems, paired).

### 3.1 chrF++ against human references (`t_longctx_human_chrf.md`, `fig14_longctx_chrf`)
| model | mode | SmolDoc short | stitched ~512 | ~1,024 | ~2,048 | AfriSci dev | AfriSci test |
|---|---|---|---|---|---|---|---|
| SFT-final | whole | 43.1 | 40.4 | 14.7 | 12.3 | 19.4 | 16.0 |
| SFT-final | sent+repair | 46.1 | 46.9 | 49.6 | 52.3 | 53.0 | 53.0 |
| RL-final 2c | whole | 48.2 | 47.0 | 25.6 | 17.9 | 33.8 | 31.5 |
| RL-final 2c | sent+repair | 49.0 | 49.7 | 52.1 | 55.1 | 53.3 | 53.0 |
| A2W a0.7 | whole | 48.8 | 47.8 | 39.4 | 28.8 | 43.7 | 41.6 |
| A2W a0.7 | sent+repair | 49.1 | 49.8 | 52.5 | 55.4 | 55.7 | 55.4 |
| live run (step 250) | whole | 49.0 | 47.9 | 39.6 | 28.2 | 43.8 | 41.5 |
| live run (step 250) | sent+repair | 49.3 | 50.0 | 52.6 | 55.5 | 55.6 | 55.4 |
| NLLB-600M | whole | 34.8 | 23.0 | 12.4 | 8.5 | 11.5 | 10.8 |
| NLLB-600M | sent | 50.5 | 51.6 | 53.9 | 56.6 | 55.1 | 54.6 |
| NLLB-3.3B | whole | 39.1 | 23.5 | 9.7 | 5.9 | 9.8 | 8.4 |
| NLLB-3.3B | sent | 51.7 | 53.1 | 55.0 | 57.6 | 57.1 | 56.9 |
| TranslatePsy-0.8B | whole | 35.5 | 22.9 | 11.7 | 8.1 | 18.3 | 17.9 |
| TranslatePsy-0.8B | sent | 53.2 | 54.2 | 56.0 | 58.7 | 57.9 | 57.6 |
| TranslatePsy-2B (5 lang., reduced bins) | whole | n/a | n/a | 14.9 | 7.5 | 23.3 | n/a |
| TranslatePsy-2B | sent | n/a | n/a | 55.7 | 58.3 | 58.8 | n/a |
| Qwen3.5-0.8B zero-shot | sent | 10.4 | 11.2 | 12.6 | 13.7 | 10.4 | 10.0 |
| Gemma-3-1B-it zero-shot | sent | n/a | n/a | 21.3 | 23.3 | 14.3 | n/a |

Reading: up to ~512 tokens our science-stage and RL models match their own wrapper; NLLB and TranslatePsy already lose about 30 chrF++ at ~512 tokens (23.0-23.5 stitched ~512). From ~1,024 tokens only the science-stage models keep a usable whole-document score. With the wrapper the order is TranslatePsy-0.8B ~ 2B > NLLB-3.3B > NLLB-600M > live/A2W ~ RL-final > SFT-final at every length (at ~1,024: 56.0 / 55.0 / 53.9 vs 52.6; at ~2,048: 58.7 / 57.6 / 56.6 vs 55.5).

### 3.2 Failure anatomy (`t_longctx_failure_*.md`, `fig15_longctx_failure`)
Whole-document, stitched ~1,024 / ~2,048 tokens: finished / loop / coverage / length ratio / GlotLID.
| model | ~1,024 | ~2,048 |
|---|---|---|
| live | 1.00 / 0.04 / 0.87 / 0.78 / 0.95 | 0.95 / 0.10 / 0.70 / 0.56 / 0.94 |
| A2W | 0.98 / 0.15 / 0.87 / 0.80 / 0.95 | 0.80 / 0.30 / 0.72 / 0.75 / 0.95 |
| RL-final | 0.51 / 0.67 / 0.95 / 1.29 / 0.93 | 0.39 / 0.90 / 0.94 / 1.27 / 0.89 |
| SFT-final | 0.07 / 0.90 / 0.78 / 1.80 / 0.90 | 0.03 / 1.00 / 0.74 / 1.80 / 0.96 |
| NLLB-3.3B | 0.87 / 0.53 / 0.39 / 0.25 / 0.90 | 0.85 / 0.49 / 0.33 / 0.16 / 0.93 |
| NLLB-600M | 0.55 / 0.64 / 0.41 / 0.41 / 0.86 | 0.45 / 0.69 / 0.31 / 0.23 / 0.90 |
| TranslatePsy-0.8B | 0.77 / 0.23 / 0.46 / 0.66 / 0.93 | 0.61 / 0.38 / 0.53 / 0.94 / 0.91 |
| TranslatePsy-2B | 0.75 / 0.15 / 0.55 / 0.79 / 0.94 | 0.76 / 0.18 / 0.54 / 0.64 / 0.94 |

All sentence-by-sentence rows: coverage ~1.00, loop <= 0.05, finished 0.95-1.00 (exceptions: NLLB sentence-mode loop 0.09-0.33 from repeated sentences/list items; NLLB-600M finished 0.84-0.96). Hausa ~2,000 tokens: NLLB-3.3B returns 800 characters; TranslatePsy-0.8B writes 25,145 characters and loops to the budget; the live model finishes with 4,490 of the ~9,188 reference characters (about half untranslated).

### 3.3 AfriScience-MT
Whole paper, dev / test: live finished 0.96 / 0.95, A2W 0.90 / 0.86, RL-final 0.67 / 0.71, TranslatePsy-0.8B 0.21, NLLB-3.3B 0.83 / 0.74 (but 36-57% of papers truncated, coverage 0.60-0.67, length ratio 0.30-0.33). The live finished rate 0.96 is below the "1.000" of the live-run integration notes (different decoding budget: here 3.5x source tokens, greedy). **Scoring caveat**: document-pooled chrF++ forgives clause shifts of the dataset's sentence alignment and is about 10 chrF++ above sentence-level scores; do not compare with published sentence-level scores. Sentence-level chrF++ (5 common languages), dev / test: RL-final 43.5 / 43.1, live 45.7 / 45.4, NLLB-3.3B 48.3 / 48.0, TranslatePsy-0.8B 49.4 / 49.0, Qwen 4.6 / 4.2 (`reports/data/longctx/sentlevel.csv`; for 6 languages RL-final and SFT-final reproduce the earlier reports' 45.6 / 44.8 vs 45.65 / 44.90, validating the harness).

### 3.4 Core documents vs the NLLB-3.3B pseudo-reference (`t_longctx_core_nllbref.md`; biased, use for ordering only)
Whole-document at ~512 / 1,024 / 2,048 tokens: NLLB-600M 24.6 / 14.5 / 8.9, live 56.5 / 46.1 / 32.1, RL-final 55.1 / 34.3 / 18.9. Sentence-wrapped NLLB-600M and TranslatePsy score 65-76 here against 52-65 for ours: lineage bias, not a 10-point quality gap. Control b256: no length effect for any system.

### 3.5 Sentence minus whole, paired (`t_longctx_sentence_minus_whole.md`)
Live vs NLLB pseudo-reference: +0.4 (b256), +4 (b512), +14.0 (b1024), +30.9 (b2048), +49 (b4096). Live vs humans: +13 (stitched ~1,024), +27 (~2,048), +12 / +14 (AfriScience dev / test). RL-final: +26.5 / +37.2 / +19.5 / +21.5. SFT-final: +34.8 / +40.0 / +33.6 / +37.0.

### 3.6 Zero-shot models
Qwen3.5-0.8B: chrF++ 10-14, GlotLID 0.14-0.23, finished ~0 (Hausa in Arabic script, chrF++ 1.7; Amharic 1.7; Swahili 23.1, Zulu 22.6, Yoruba 13.9 with loops). Gemma-3-1B-it: chrF++ 21-23 (swh 40.0, zul 29.1, hau 25.1, yor 11.6, amh 0.7; GlotLID 0.48; rarely stops). Neither can write most of these languages, so context length is irrelevant for them.

## 4. Pseudo-reference validation (`t_longctx_pseudoref_validation.md`, `t_longctx_agreement_nllb.md`, `fig16_longctx_pseudoref`)
NLLB-3.3B (sentence-by-sentence) vs human references, mean over SmolDoc bins: amh 37.7, hau 57.9, ibo 59.2, kin 61.5, mlg 68.4, som 58.2, swh 66.9, twi 45.0, wol 40.7, xho 57.8, yor 45.5, zul 59.0, lug (AfriScience) 55.2, nso (AfriScience) 63.1. NLLB-3.3B minus best SabiYarn: amh +9.8, hau +0.5, ibo +1.8, kin +8.7, mlg +6.9, som +0.9, swh +0.7, twi +0.8, wol -3.2, xho +0.4, yor -0.8, zul +2.7, lug -2.6, nso -5.2. Usable as a better-than-ours yardstick (>= +2): amh, kin, mlg, zul; marginal: hau, ibo, som, swh, twi, xho; not above ours: lug, nso, wol, yor.

Agreement gap (chrF++ vs NLLB-3.3B minus vs humans; stitched ~1,024 / AfriScience dev): SFT-final +6.6 / +3.8; RL-final +8.6 / +4.5; A2W +7.8 / +2.2; live +8.0 / +2.3; NLLB-600M +15.6 / +12.8; TranslatePsy-0.8B +17.5 / +15.3; TranslatePsy-2B +19.1 / +15.6; Qwen3.5-0.8B and Gemma-3-1B ~0 (zero-shot, near-zero scores). R6's "NLLB-referenced scores 1-16 chrF++ above human-referenced" is confirmed and attributed: NLLB-derived systems sit at the top of that range, ours in the middle. Failure metrics (coverage, loop, length ratio) depend on the pseudo-reference only through the length model and loop veto, so are far less affected.

## 5. Gemini Batch API cost math (no call made; full detail in `reports/long_context_gemini_plan_2026-10-23.md`)
Prices read from the official pricing page on 2026-10-23 (USD per 1M tokens; Batch = 50% of standard): 2.5 Flash-Lite 0.05 in / 0.20 out; 2.5 Flash 0.15 / 1.25; 2.5 Pro 0.625 / 5.00 (output includes thinking tokens; 2.5 models may be limited to users who already used them, a new key may only get 3.x). Benchmark: 2,280 items from 290 documents, 2.27M Gemma source tokens. Whole-document, Batch, thinking off, output = 2.75x English tokens: 2.5 Flash $9.64 (full set) / $1.18 (reduced set of 270 items); Flash-Lite $1.62 / $0.20; 2.5 Pro (default thinking) $61 / $7.4. Chunked requests: chunk-8 about +1% (thinking off), chunk-1 about +16%; with default thinking 3x and 18x. 50-sentence pilot (14 languages): $0.05-0.06 (Flash, thinking off) up to $1.2-7.3 (Pro, default thinking). CAVEAT: Gemini is uneven in low-resource African languages (the Urhobo silver set was doubtful): evidence in the repo is Gemini-2.5-Flash FLORES chrF++ on only 10 sentences per language vs NLLB-3.3B re-scored on 1,012 sentences. Trust (Gemini ahead of NLLB-3.3B): hau, swh, ibo, mlg, zul, xho, kin, som, twi, lug, nso; marginal: yor, wol; no number: amh. Gemini adds most where NLLB is not above our model (lug, nso); for wol and yor neither pseudo-reference is useful. Recommendation: run the ~$0.06 pilot (50 sentences x 14 languages vs human references), then the ~$1.2 reduced set; the full set (~$10) only if the pilot passes; use Gemini outputs as evaluation references only (API terms forbid using the Services to develop competing models). Scripts: `paper/scripts/gemini_batch_requests.py`, `gemini_batch_ingest.py`, `gemini_batch_cost.py`, `gemini_count_tokens.py`, `gemini_language_trust.py`, `test_gemini_batch.py` (11 pytest cases pass); the key via `GEMINI_API_KEY` env var, nothing secret in the repo; scoring a Gemini reference: write `/workspace/jobs/lc1/work/refs_gemini.jsonl` rows `{iid, ref_sents}` and re-run `lc_score.py` / `lc_agg.py`.

## 6. Commands, deviations, what remains
Box folder `/workspace/jobs/lc1` (code in `gemma_mt/scripts/longctx/`): `source env_lc.sh; $PYG lc_build.py; $PYG lc_prep.py; ./run_ours.sh sft rl a2w live; $PYG lc_assemble_ours.py <model>; $PYG lc_nllb.py 3b3 sent whole; $PYG lc_nllb.py 600m sent whole; $PYV lc_chat.py tpsy08 ...; $PYG lc_score.py <name>__<mode>; $PYG lc_agg.py --boot 300; $PYG lc_sentlevel.py ...`. Laptop (numpy + matplotlib): `python paper/scripts/make_longctx.py [tables|figs]`. To re-score a later live checkpoint: `python dl.py <HF_ORG_LEGACY>/gemma-science-translate live latest`, then `./run_ours.sh live`, assemble, score, aggregate (~25 min).
- GPU footprint: cap 8 GB; ours peaked 7.2-7.7 GB, TranslatePsy-0.8B whole-document 9.1 GB, NLLB-3.3B 7.5 GB; no OOM hit LC1 processes; vLLM start-ups sometimes failed the memory-profile assertion when other jobs freed memory during profiling (retry loop handled it).
- Not done: Qwen3.5-2B, Gemma-3-4B-it, TranslatePsy-4B, AfriNLLB; whole-document mode for Qwen3.5-0.8B and Gemma-3-1B-it; AfriCOMET/SSA-COMET scoring (chrF++, spBLEU and failure metrics only); any human or LLM judge of discourse quality; repair/sampling variants for non-SabiYarn systems; multi-sentence windows for NLLB and TranslatePsy; the Gemini pseudo-reference itself (needs the owner's key).
- Metric caveats: pooled document-level chrF++ (3.3); NLLB sentence-mode loop flags inflated; the TranslatePsy-2B agreement macro covers 5 languages.
- Disk: `/workspace/jobs/lc1` 523 MB (`out/` = token-id files, deletable). `/dev/shm/lc1_models` removed. No LC1 process is running.
- For the paper: decide the wording of H (section 0); if the native long-context claim stays for the science-stage model, quote coverage and length ratio next to chrF++.

## Addendum 2026-10-24 (G1): the final checkpoint `final2-step401` on LongCtx-MT

Same scripts, documents, languages and settings as the other systems (`/workspace/jobs/lc1`: `lc_gen.py final2 /dev/shm/lc1_models/final2 --gpu-util 0.25`, then `lc_assemble_ours.py final2`, `lc_score.py final2__{whole,sent,sentfix}`, `lc_agg.py --only final2,live --boot 300` with and without `--langs amh,hau,ibo,swh,yor,zul,lug`; rows added to `reports/data/longctx/{macro,macro_common,cells,cells_common,pair_mode,pair_mode_common}.csv`, step-250 rows re-computed and identical to the existing ones; `sentlevel.csv` has the `final2__sentfix` rows; `final2_paired_vs_other.csv` = paired document bootstrap final2 minus live / A2W / RL-final, 1,000 draws, `paper/scripts/box_g1_lc_paired.py`). GPU: one vLLM process, 25 GB reserved (gpu_memory_utilization 0.25), whole-document 249 s, sentence units 6 min, repair 6 s; nothing else was run on the GPU.

All parts were run: whole-document mode, sentence-by-sentence and sentence-by-sentence + repair (best-of-four of 262 unfinished units: 222 fixed, 40 still unfinished), all bins (b256-b4096, SmolDoc short, stitched ~512 / ~1,024 / ~2,048, AfriScience dev and test), all languages. Not run: nothing of the existing protocol; the NLLB-referenced and human-referenced tables are as for the other systems.

Macro over the common languages (chrF++ against human references, 95% document-bootstrap interval; whole document | sentence + repair):

| bin | final2 whole | live (step 250) whole | A2W whole | RL-final whole | final2 sent+repair |
|---|---|---|---|---|---|
| stitched ~1,024 | 39.6 [38.3, 40.7] | 39.6 [38.3, 40.8] | 39.4 | 25.6 | 52.6 [52.1, 53.2] |
| stitched ~2,048 | 27.7 [25.8, 29.7] | 28.2 [26.6, 29.9] | 28.8 | 17.9 | 55.5 [55.0, 56.0] |
| AfriScience dev | 43.2 [40.7, 45.3] | 43.8 [41.7, 45.8] | 43.7 | 33.8 | 55.6 [54.3, 56.8] |
| AfriScience test | 41.3 [39.2, 43.3] | 41.5 [39.3, 43.6] | 41.6 | 31.5 | 55.4 [54.1, 56.6] |

Failure metrics (stitched ~1,024 / ~2,048, whole document): finished 0.98 / 0.92, loop 0.08 / 0.25, coverage 0.88 / 0.73, length ratio 0.81 / 0.59, GlotLID 0.95 / 0.95 (live: 1.00 / 0.95, 0.04 / 0.10, 0.87 / 0.70, 0.78 / 0.56). Paired final2 minus live (whole): chrF++ -0.03 [-1.01, +0.95] at ~1,024, -0.46 [-2.31, +1.40] at ~2,048, AfriScience dev -0.61 [-1.80, +0.48], test -0.20 [-1.15, +0.71]; loop rate +0.04 [+0.01, +0.08] and +0.15 [+0.10, +0.20]. Sentence-level chrF++ on the AfriScience papers, 6 languages, sentence + repair: 48.06 / 47.26 (dev / test), the separate evaluation script of the paper's Section 6.7 gives 48.12 / 47.30.

Reading: the final checkpoint is indistinguishable from the step-250 checkpoint in chrF++ in every column and omits as much text; its whole-document outputs loop more often at ~1,024-2,048 tokens. It does not change section 0.
