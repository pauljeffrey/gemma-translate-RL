# Gemma-4-31B-it as a data teacher for the 23 consensus-only languages (R15, D-67, 2026-10-22)

(Report text as returned by agent R15; saved by the lead because the sandbox refused the agent's write. CSVs and scripts: `reports/data/gemma31b_weak_language_probe_2026-10-22/`, `reports/data/gemma31b_probe_*_2026-10-22.csv`. Box: `/workspace/jobs/r15`.)

## 0. Verdict
- Does Gemma-4-31B-it know these languages? Mostly no. Zero-shot it scores chrF++ 1-14 on 16 languages. It does know Kabuverdianu (40.8 against our 29.4), handles Cameroon Pidgin (36.6), South Ndebele (29.4, but GlotLID accepts only 33% of its output) and Ndau (26.0). It writes Krio only with examples (22.2 zero-shot, 38.8 few-shot).
- Decision rule (fixed before running): Gemma few-shot-8 (fs8: the 8 most similar pairs, BM25 on the English side) must beat our `final_1433` by >= +3 chrF++ with the paired-bootstrap 95% CI excluding 0, and pass the checks (language ID, copy of English, repetition, finish, copying of retrieved examples).
- Result: 2 of 23 pass. kea (+16.1) passes clearly; kri (+3.5) is a conditional pass.
- All six named languages fail: ish -1.1, bin -5.9, bbj -2.4, oke -9.7, tsc -3.3, dag -10.3. The other 15 fail too: her +0.2 and nbl -0.8 are level; the rest are clearly worse (-2.7 to -12.2), ndc -4.6, wes -5.9.
- The few-shot results look decent because retrieval returns the JW/Bible phrases our model already absorbed. k=3 is 6-12 points below k=8 on the six named languages. A language note adds nothing (+1.8 to -1.9, CIs contain 0).

## 1. Design and caveats
Caveat first: the held-out references are mostly JW/Bible-style text from our own SFT corpus, which our model was trained on, so the comparison FAVOURS OUR MODEL. The out-of-domain benchmark rows are the fairer side.

Retrieval pool: R10's decoded replay pool (927-1,311 windows per language); the original SFT windows are no longer stored per row.

References per language: `val` (R4's held-out validation windows, 54-127 per language; items with 6-gram containment >= 0.5 against the pool dropped); benchmark rows (at most 100 per set): FLORES+ devtest (kea, vmw), FLORES-MOZ (seh), SmolSent (nbl, ach, nyn, ndc, kri), GlotStoryBook (dag, tsc, nbl, her), IbomMT (ibb), SALT (ach, nyn), Wikimedia (dag), 5 gold rows (bin); `seen`: 40 training windows held out of retrieval (robustness check, favours our model). Ten languages have no benchmark at all: ish, bin, bbj, oke, urh, yao, sid, dua, bas, wes. Primary population: val + benchmark rows, capped at 150 items per language.

Systems: `final_1433` (greedy, max_new_tokens 512), `gemma_zs` (zero-shot, prompt naming language and region), `gemma_fs8` (8 BM25 pairs, prompt capped 2,600 tokens), `gemma_fs3`, `gemma_fs8note`. Decoding: thinking off, T=0, 600 new tokens, bf16 KV, max_model_len 3,328. Ablations on a fixed 80-item subset per language.

Metrics: chrF++ (NFC) with paired bootstrap (1,000 resamples); GlotLID top-1 (GlotLID accepts only 0.49 of ndc references, 0.35 of shi, 0.73 of dag); copy of English; 3-gram repetition; finish rate; length ratio; example copying (share of output 4-grams found in the retrieved example targets).

PASS rule: delta >= +3 with CI lower bound > 0, copy <= 5%, GlotLID >= max(0.5, our rate - 0.05), finished >= 90%, repetition <= 0.25, example-copy excess <= 0.15.

Other caveats: possible Gemma contamination on FLORES+, SmolSent, GlotStoryBook (kea's pass does not depend on it; kri's does). shi NTREX reference is in another script (unusable). Populations are 47-150 items, so differences below ~2 chrF++ are noise.

## 2. Results: fs8 against final_1433 (primary population)

| lang | name | n | final_1433 | Gemma fs8 | delta | 95% CI | verdict |
|---|---|---|---|---|---|---|---|
| ish | Esan | 61 | 31.9 | 30.8 | -1.1 | [-4.9,+2.7] | FAIL (level) |
| bin | Edo | 132 | 32.2 | 26.2 | -5.9 | [-7.6,-4.4] | FAIL |
| bbj | Ghomala | 54 | 25.6 | 23.2 | -2.4 | [-5.7,+1.2] | FAIL (level) |
| oke | Okpe | 109 | 41.2 | 31.5 | -9.7 | [-12.9,-6.3] | FAIL |
| tsc | Tswa | 150 | 34.0 | 30.7 | -3.3 | [-6.7,-0.2] | FAIL |
| dag | Dagbani | 150 | 38.7 | 28.5 | -10.3 | [-13.0,-7.5] | FAIL |
| ibb | Ibibio | 150 | 37.5 | 25.3 | -12.2 | [-15.1,-9.3] | FAIL |
| urh | Urhobo | 113 | 38.5 | 28.7 | -9.9 | [-12.7,-6.6] | FAIL |
| nbl | S. Ndebele | 150 | 34.7 | 33.9 | -0.8 | [-3.5,+1.9] | FAIL (level; GlotLID 0.33) |
| shi | Tachelhit | 150 | 20.1 | 17.4 | -2.7 | [-4.7,-0.6] | FAIL |
| **kea** | Kabuverdianu | 150 | 28.8 | **44.9** | **+16.1** | [+13.9,+18.1] | **PASS** |
| ach | Acholi | 150 | 35.3 | 28.8 | -6.5 | [-9.1,-3.9] | FAIL |
| vmw | Makhuwa | 150 | 32.5 | 20.4 | -12.1 | [-14.0,-9.9] | FAIL |
| seh | Sena | 150 | 39.2 | 28.3 | -10.9 | [-14.2,-7.2] | FAIL |
| nyn | Nyankore | 150 | 33.4 | 25.6 | -7.8 | [-10.9,-4.4] | FAIL |
| her | Herero | 150 | 27.6 | 27.8 | +0.2 | [-2.3,+2.7] | FAIL (tie) |
| yao | Yao | 94 | 39.1 | 31.8 | -7.3 | [-10.0,-4.5] | FAIL |
| sid | Sidamo | 96 | 35.8 | 24.5 | -11.3 | [-14.5,-7.7] | FAIL |
| ndc | Ndau | 150 | 34.9 | 30.3 | -4.6 | [-6.9,-2.2] | FAIL |
| **kri** | Krio | 150 | 35.3 | **38.7** | **+3.5** | [+1.4,+5.8] | **PASS (conditional)** |
| dua | Duala | 88 | 41.2 | 29.9 | -11.3 | [-15.6,-5.9] | FAIL |
| bas | Basaa | 79 | 39.7 | 31.1 | -8.7 | [-12.2,-5.6] | FAIL |
| wes | Cameroon Pidgin | 47 | 54.0 | 48.1 | -5.9 | [-8.0,-3.6] | FAIL |

Sets that matter (fs8 minus final_1433): kea FLORES+ +20.6 [+18.5,+22.7] (n=96); kea val +8.9; kea seen +11.5; kri SmolSent +10.8 [+8.4,+13.2] (n=89); kri val (JW) -3.6 [-5.9,-1.3]; kri seen -3.5; tsc GlotStoryBook +2.6 [-0.8,+5.5]; tsc val -14.3; nbl GlotStoryBook +3.2 [-1.6,+7.4]; dag GlotStoryBook -10.3; ish seen +1.7 [-0.2,+3.7] (ish has only 1,697 SFT samples).

Checks: the fs8 outputs of the failing languages are not broken (GlotLID 0.74-0.99 except nbl 0.33; copy-of-English about 0; example-copy excess small except bbj 0.218 vs 0.033 for the reference = example pasting). Repetition and finish rates are worse than ours for dag, vmw, shi, dua. The failures are quality failures, not format failures.

## 3. Ablations (80-item subsets, chrF++)

| lang | n | final_1433 | zero-shot | k=3 | k=8 | k=8 + note | note vs fs8 |
|---|---|---|---|---|---|---|---|
| ish | 61 | 31.9 | 3.2 | 21.3 | 30.8 | 31.7 | +0.9 |
| bin | 80 | 31.3 | 4.3 | 19.0 | 25.5 | 25.5 | 0.0 |
| bbj | 54 | 25.6 | 1.2 | 16.5 | 23.2 | 21.3 | -1.9 |
| oke | 80 | 41.9 | 2.7 | 20.0 | 32.0 | 31.3 | -0.7 |
| tsc | 80 | 30.8 | 11.5 | 21.6 | 28.3 | 30.0 | +1.8 |
| dag | 80 | 38.2 | 9.7 | 21.1 | 27.6 | 28.6 | +1.0 |
| kea | 80 | 29.4 | 40.8 | 43.1 | 44.7 | 44.9 | +0.2 |
| kri | 80 | 32.1 | 22.2 | 35.4 | 38.8 | 38.6 | -0.2 |

Zero-shot elsewhere: nbl 29.4, ndc 26.0, wes 36.6, ibb 6.4, ach 5.4, urh 3.9, dua 3.1, shi 2.1, nyn 16.5, seh 13.4, yao 12.7, her 7.1, vmw 4.3, sid 3.4, bas 2.1. Note effects: nbl +0.1, shi -0.5, her -0.5, ndc -0.8 (all CIs contain 0).

## 4. The passing languages
- kea: Gemma 44.9 vs 28.8 on 150 items. The pass does not depend on possible FLORES contamination (val +8.9 and seen +11.5 agree). Clean: GlotLID 0.97 (references 0.96), copy 0, repetition 0.003, finish 1.00.
- kri: CONDITIONAL. Driven by general-domain SmolSent (+10.8), also the domain of the generated data. On JW val Gemma is worse (-3.6); SmolSent is public, so contamination is possible. Recommend a lower mixture weight than kea until a clean general-domain Krio reference exists.

## 5. Generation for kea and kri
- Sources: rl_prompts_v3, rl_prompts_extra_v1, rl_prompts_sci_v2 (FineWeb-Edu, Wikipedia, PubMed, LibreTexts, arXiv papers, arXiv abstracts, AfriScience train side), split into 5-80-word chunks, R10 hold-out filter; about 124k chunks per language, 12k kept.
- Generation: k=8 BM25 shots; greedy candidate + one T 0.7 / top-p 0.95 / top-k 64 partner for a consistency filter. Consistency correlates weakly with quality (Spearman 0.12 kea, 0.31 kri); tau=60 drops 0% of kea and 14% of kri (+0.4 / +1.1 chrF++ on kept). No QE ensemble (not validated for these languages).
- Filters: empty/multi-line/truncated, meta text, too short, copy of English, word-ratio bounds from real pairs (kea [0.46,3.30], kri [0.58,3.59]), 3-gram repetition > 0.25, digits lost, copy of a retrieved example, GlotLID, consistency chrF++ >= 60, exact duplicate targets.
- Packs: 8,192 tokens, 3% validation, tokenizer fingerprint equal to R4's mix, `audit_packed` OK. Columns: source_dataset=gemma_synthetic, teacher, licence_class=gemma_synthetic, synthetic_target=True, src_source_dataset, src_licence, domain, consistency_chrf, sid.

| lang | generated | kept | yield | target tokens | train windows | val windows |
|---|---|---|---|---|---|---|
| kea | 2,640 | 2,515 | 95.3% | 177,507 | 387 | 8 |
| kri | 2,160 | 1,887 | 87.4% | 175,200 | 266 | 11 |

Final pack: `/workspace/jobs/r15/gemma_synth_v1_full/` (`READY_GEMMA`, `manifest_full.json`); partial pack unchanged in `gemma_synth_v1/`. Diversity: distinct 1-grams kea 0.24 vs 0.08 for the replay pool, kri 0.19 vs 0.06; ~91% of synthetic word types are absent from the pool. Quality limits: Portuguese leakage in kea ("Igreja Anglikana"), English words inside kri science text; a code-mixing filter would help before heavy use. Far below the 2M-token target because of throughput (30 GiB engine, ~4 concurrent sequences at 8-shot prompt length, 70-130 generated tok/s; 2M tokens per language would need ~30 GPU-hours each).

## 6. Licence
- Checkpoint `google/gemma-4-31B-it-qat-w4a16-ct`: model card `license: apache-2.0`, linking the Gemma 4 licence page, which carries the Apache-2.0 text and says nothing about outputs or training other models. The page is marked deprecated and points to gemma.ai: re-check before any public release.
- Prohibited Use Policy has no item on synthetic data or training other models. Gemma 1-3 Terms of Use did define "Model Derivatives" to include models trained on Gemma outputs; that clause is absent from the Apache text of Gemma 4. Our base `gemma-3-270m-it` remains under the old Terms.
- Apache-2.0 permits commercial and non-commercial use; restrictions come from the English source texts (`src_licence` per row; Wikipedia rows are CC-BY-SA). Not legal advice.

## 7. Process
Datagen STOP/HOLD untouched; own engine only when >= 32 GiB free, ~30 GiB footprint, under nice, never two engines at once; scratch deleted (`/workspace/jobs/r15` is 47 MB). Scripts: `reports/data/gemma31b_weak_language_probe_2026-10-22/scripts/` (prep1b/prep2/prep3, flags, gen_base2, gen_gemma/gen_gemma2, score, agg, sub_compare, make_tables, calib, prep_src, gen_synth, filter_pack, curve).
