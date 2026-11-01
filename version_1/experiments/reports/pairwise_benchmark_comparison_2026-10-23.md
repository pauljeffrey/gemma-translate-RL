# Pairwise benchmark comparison of SFT-final and RL-final, a stand-in for a human preference evaluation (E1, 2026-10-23)

Question: the owner has no native-speaker raters, so can the paper support "RL improved the model over SFT" with benchmarks alone? This report builds, for the same sentences, a **sentence-level pairwise comparison** (win / tie / loss of RL-final over SFT-final) under several metrics, with paired-bootstrap intervals and sign tests, and says explicitly which of the metrics are independent of what the RL stage optimised. Nothing was pushed; the per-sentence hypotheses are kept on the box (`/workspace/jobs/e1/`), only the small summary CSVs are in the repository (`reports/data/pairwise_benchmark_comparison/`).

Systems: **SFT-final** = `<HF_ORG>/gemma3-270m-translate` tag `step-27638` (the copy already in the box's HF cache, nothing downloaded); **RL-final** = `/workspace/sft_build/runs/final_1433` (GRPO step 1433, the same snapshot R5 used). Greedy decoding, prompt `Translate the text to {language}:  {english}`, language tag stripped, NFC, `max_new_tokens` 768 (headline) and 320 (FLORES+). Benchmarks and subsets exactly as R5 (FLORES+ devtest, NTREX-128, MAFAND-MT, SmolSent, IbomMT, FLORES-MOZ, SALT, GlotStoryBook; test splits only; 161,272 English-to-X sentence pairs per system, 66 languages with at least 20 `no_pair` sentences). The new run reproduces R5 (section 8): macro corpus chrF++ per benchmark within 0.01-0.2 of the R5 tables (`no_pair`, e.g. FLORES+ 28.68 / 34.08 against R5 28.67 / 34.09).

## 0. Summary (read this first)

1. **RL-final is preferred over SFT-final on every metric, in every language group, on every benchmark and at every document length.** Headline (`no_pair`: rows whose reference pair was not in the SFT text; macro over languages, all benchmarks pooled per language): chrF++ 27.7 -> 31.1 (delta +3.32 [+3.26, +3.38]); RL is better in 54.8% of the sentences, tied (|delta| < 1 chrF++) in 20.0%, worse in 25.1%, i.e. **preferred in 63.7% of the non-tied sentences** [63.5, 64.0]. With the neural metrics against the human references: AfriCOMET (reference-based) +0.059, RL preferred in **70.0%** of the non-tied sentences [69.7, 70.4]; SSA-COMET-MTL (reference-based) +0.059, **70.4%** [70.1, 70.7]; majority vote of the three reference-based metrics **70.2%** [69.8, 70.5]. On the contamination-robust `clean` subset (English sentence not in the SFT text) the picture is the same: 61.6% (chrF++), 69.0% (AfriCOMET-ref), 69.6% (SSA-COMET-ref), 68.9% (majority). Sign tests: p < 1e-300 for every metric (optimistic: the sentences are not independent, section 9), but the language-level result does not need them: **RL is significantly better (CI of the mean delta above 0) in 65 of 66 languages on chrF++ and in all 51 AfriCOMET-covered languages on all four neural metrics; in no language is it significantly worse.**
2. **The reference-based evidence does not depend on the RL reward model.** RL-final was optimised on AfriCOMET-QE (+ the consensus signal and anchors). chrF++ against human references is independent of every learned model; SSA-COMET-MTL is a different model trained on different data (per D-55 the SSA-COMET ensemble is off by default, so it should not be in the RL-final reward; confirm against the `rl2c` config before the paper says so), and its reference-based variant also prefers RL in 70.4% of the sentences. AfriCOMET reference-based comes from the same model family (same encoder, same training data) as the reward model and is therefore *not* independent in the strict sense; the independent evidence is chrF++ and SSA-COMET.
3. **No sign of metric gaming.** The RL reward model (AfriCOMET-QE) prefers RL in 72.9% of the non-tied sentences, only 2.9 points above AfriCOMET with a reference (70.0%) and 2.5 above the independent SSA-COMET-ref (70.4%); a model that had learned to please the QE would show a much larger QE gap than reference gap. Of the 38,604 sentences on which RL wins on **both** QE metrics, the reference-based majority agrees in **89.7%**, is neutral in 5.0% and prefers SFT in 5.3% (section 6). Languages: no language in which RL wins significantly on a QE metric while losing on a reference-based one when all benchmarks are pooled; one language (`sot`) wins on both QE metrics and both reference-based COMET metrics but is flat on chrF++ (+0.29 [-0.01, +0.62]). At the (benchmark, language) level 20 of 127 cells show "QE up, a reference-based metric not up"; in 19 of them chrF++ is the metric that is flat (-0.1 to +0.9) while AfriCOMET-ref is up (and SSA-ref in 18); these are the high-resource Bantu, Sotho-Tswana, Malagasy, Afrikaans, Hausa languages, in one (SALT `lug`, 81 sentences) chrF++ is -2.98 [-5.84, -0.47] while the COMET references are flat (section 6.3).
4. **Part of the gain is the removal of the runaway / looping outputs of SFT, and that is a real quality difference (a translation that does not stop is worse), not a metric artefact.** SFT does not finish 11.4% of its sentences (loop 13.0%, mean length 1.51 x the reference) against 0.3% (1.0%, 1.15 x) for RL. Removing every sentence on which either model loops or does not finish (8.5% of them), the chrF++ delta halves (+1.67 macro) but RL is still preferred in 60.0% of the non-tied sentences by chrF++ and in 67.7% / 68.6% / 70.8% / 68.7% by AfriCOMET-ref / SSA-ref / AfriCOMET-QE / SSA-QE (section 4).
5. **Long inputs** (R6 documents, both models decoded sentence by sentence, clean documents): documents: RL is preferred in 63 / 71 / 75 / 84% of the non-tied documents by chrF++ for b64 / b256 / b512 / b1024 (+3.4 / +3.2 / +2.3 / +2.3 chrF++), 79-80% at b2048 / b4096, and on the human-reference SmolDoc documents 80% (short), 79% (~512 tokens), 100% (~1,024 tokens, 80 documents) with +2.0 ... +3.7 chrF++; **it finishes 94-98% of the documents at b64-b512 against 48-93% for SFT and loops in 1-21% against 7-55%**. Inside the documents, per aligned sentence, RL is preferred in 60-65% (chrF++), 67-71% (AfriCOMET-ref), 66-70% (SSA-ref) of the non-tied sentences, with human references and NLLB pseudo-references alike (section 7).
6. **Honest limits** (section 9): no human judgement; FLORES+ / NTREX are public and the models have seen 95% of the FLORES+ English sentences in other pairs; the neural metrics were validated by their authors on a subset of the languages and see only one reference; chrF++ and the COMET-family metrics agree on the sentence winner in only 64-73% of the sentences where both call one; the benchmark domains (news, Wikipedia-like, religious / story text) are not the pivot / JW-domain in which the training data lives.

## 1. What was run

* **Generation.** The repo's `gemma_mt.eval.sft_benchmarks` pipeline through `scripts/eval_sft_benchmarks.sh` (vLLM worker, greedy, `--no-comet --skip-base`), exactly as `reports/data/final_model_runs/box_scripts/run_bench.sh` (R5), but with the **hypotheses kept**: `sys_sft.jsonl.gz`, `sys_rl.jsonl.gz` (27 MB each: english, reference, hypothesis, per-sentence chrF++, finished / loop / repetition / LID flags), `sys_{sft,rl}320.jsonl.gz` (FLORES+ at 320 tokens). 161,272 rows per system; two ids occur twice in SmolSent `afr` (counted twice, negligible).
* **GPU footprint.** The first attempt (vLLM utilisation 0.05, default 4,096 positions / 8,192 batched tokens) used 6.9 GB per engine (the coordinator measured up to 10.8 GB) because the 262k-token vocabulary makes the profiling logits large, and the coordinator reported that the card was then full for the live run; I stopped, copied the code tree to a private overlay (`/workspace/jobs/e1/code2c`, one patched line: `max_num_batched_tokens` = 1,024 from the environment) and reran with ONE engine at a time, 0.05 utilisation, `max_model_len` 2,048, 128 sequences, `nice -n 19`: peak footprint 6.2 GB for the process pair, COMET scoring 4.1 GB with batch 32. The generations are deterministic (greedy), so nothing depends on that restart; SFT-final was generated with the first settings, and the first RL-final unit of the interrupted run was reused from its cache (same model, same rows digest).
* **Metrics per sentence.** chrF++ (sacrebleu `word_order=2`, NFC, sentence level, 0-100); **AfriCOMET reference-based** (`masakhane/africomet-stl-1.1`); **SSA-COMET-MTL with reference** (`McGill-NLP/ssa-comet-mtl`, mean of the three views of the model card; one joint 512-position sequence, so a long source + hypothesis + reference is truncated by the tokenizer: sentences are short, the 768-token runaway hypotheses of SFT are cut and scored as such); **AfriCOMET-QE** (`masakhane/africomet-qe-stl-1.1`, the RL reward model); **SSA-COMET-MTL QE** (reference-free; pairs over the joint budget are split and averaged, `rl/joint_budget.py`). Also finished, loop (`not finished` or `rl.guards.is_repetitive / is_self_repetitive` or empty, the decoding study's definition), word-3-gram repetition (>= 15 words and < 50% distinct 3-grams), GlotLID v3 right-language (and sibling-tolerant), length ratio (hypothesis / reference characters, clipped at 5).
* **Neural-metric sample.** The COMET-family models were run on the AfriCOMET-covered languages (54; 51 reach 20 `no_pair` sentences) on **every `clean` row plus a hash-ordered sample of at most 400 rows per (benchmark, language)**: 79,062 sentence pairs per system and metric (49,552 `clean`). chrF++ and the surface metrics use all 161,272 rows; "chrF++ (rows scored by COMET)" is chrF++ on exactly the neural sample so that the metrics can be compared on the same sentences.
* **Wins, ties, losses.** For a metric, delta = RL - SFT per sentence; RL **wins** if delta > threshold, **loses** if delta < -threshold, otherwise **ties**; thresholds chrF++ 1.0 (0-100 scale) and 0.01 for the COMET-family (scale about 0-1; tables of other thresholds in section 5.3). "RL preferred in X%" = wins / (wins + losses). Identical outputs (1.8% of the sentences) are ties.
* **Statistics.** Units are (benchmark, language, subset) and (all benchmarks pooled, language, subset). Paired bootstrap: 1,000 resamples of the sentences of a unit, the same resampled sentences for both systems. *Macro* rows: mean over languages (every language weighs the same; every language is resampled independently); *micro* rows: sentences pooled, stratified by language. Sign test: exact two-sided binomial test of wins against losses (ties dropped); per-language p-values are Benjamini-Hochberg adjusted across the languages of the table (`q`). Subsets: `all`, `no_pair` (headline), `clean` (contamination-robust), as in R5.
* **Not done.** The optional science-stage start model (`sci_sft_A2W_alpha0.7`) and the live run's `latest` checkpoint were not evaluated (GPU and disk budget; the live run was never touched). The scripts take any number of systems (`--sys name=file`), so the same pairwise tables can be produced for them.

## 2. Headline: RL-final against SFT-final

### 2.1 `no_pair` (rows whose reference pair is unseen), all benchmarks, macro over languages

| metric | langs | sentences | SFT | RL | delta | win / tie / loss % (macro) | RL preferred % of non-tied [95% CI] (micro) | sign test p |
|---|---|---|---|---|---|---|---|---|
| chrF++ (all rows) | 66 | 153220 | 27.74 | 31.06 | +3.32 [+3.26, +3.38] | 54.8 / 20.0 / 25.1 | 63.7 [63.5, 64.0] | <1e-300 |
| chrF++ (rows scored by COMET) | 51 | 76804 | 29.54 | 32.75 | +3.21 [+3.13, +3.28] | 54.2 / 19.1 / 26.7 | 62.1 [61.8, 62.5] | <1e-300 |
| AfriCOMET reference-based | 51 | 76804 | 0.530 | 0.589 | +0.059 [+0.058, +0.060] | 64.1 / 12.4 / 23.5 | 70.0 [69.7, 70.4] | <1e-300 |
| SSA-COMET-MTL reference-based | 51 | 76804 | 0.415 | 0.474 | +0.059 [+0.058, +0.060] | 64.2 / 12.0 / 23.8 | 70.4 [70.1, 70.7] | <1e-300 |
| majority of the 3 reference-based | 51 | 76804 |  |  |  | 66.4 / 10.4 / 23.2 | 70.2 [69.8, 70.5] | <1e-300 |
| AfriCOMET-QE (RL reward model) | 51 | 76804 | 0.478 | 0.547 | +0.069 [+0.068, +0.070] | 67.2 / 11.3 / 21.5 | 72.9 [72.6, 73.2] | <1e-300 |
| SSA-COMET-MTL QE | 51 | 76804 | 0.425 | 0.494 | +0.069 [+0.068, +0.070] | 65.3 / 10.8 / 23.9 | 70.8 [70.4, 71.1] | <1e-300 |

Reading: "win / tie / loss" are macro percentages of the sentences; "RL preferred" is the micro share of the non-tied sentences with a bootstrap interval; the neural metrics cover the AfriCOMET-covered languages, so their rows are a subset (51 languages) of the chrF++ rows (66). The two chrF++ rows differ because the second restricts to the sentences the neural metrics scored.

### 2.2 `clean` (English sentence not in the SFT text)

| metric | langs | sentences | SFT | RL | delta | win / tie / loss % (macro) | RL preferred % of non-tied [95% CI] (micro) | sign test p |
|---|---|---|---|---|---|---|---|---|
| chrF++ (all rows) | 66 | 53751 | 25.38 | 29.25 | +3.86 [+3.72, +4.00] | 57.2 / 19.8 / 23.0 | 61.6 [61.1, 62.1] | <1e-300 |
| chrF++ (rows scored by COMET) | 51 | 49552 | 27.88 | 31.50 | +3.62 [+3.46, +3.78] | 55.7 / 19.2 / 25.2 | 60.7 [60.2, 61.2] | <1e-300 |
| AfriCOMET reference-based | 51 | 49552 | 0.494 | 0.557 | +0.063 [+0.061, +0.065] | 66.1 / 11.7 / 22.2 | 69.0 [68.6, 69.4] | <1e-300 |
| SSA-COMET-MTL reference-based | 51 | 49552 | 0.384 | 0.443 | +0.059 [+0.057, +0.061] | 65.4 / 11.8 / 22.7 | 69.6 [69.2, 70.0] | <1e-300 |
| majority of the 3 reference-based | 51 | 49552 |  |  |  | 68.4 / 9.8 / 21.8 | 68.9 [68.5, 69.3] | <1e-300 |
| AfriCOMET-QE (RL reward model) | 51 | 49552 | 0.419 | 0.490 | +0.072 [+0.069, +0.074] | 69.1 / 10.6 / 20.3 | 71.7 [71.3, 72.2] | <1e-300 |
| SSA-COMET-MTL QE | 51 | 49552 | 0.391 | 0.462 | +0.071 [+0.069, +0.074] | 67.5 / 10.0 / 22.5 | 70.0 [69.6, 70.4] | <1e-300 |

`clean` is smaller for FLORES+ / SmolSent (48 / 76 sentences per language) and large for NTREX-128 (about 1,200 per language) and MAFAND-MT (about 1,000), so the `clean` macro is dominated by the latter two.

### 2.3 By benchmark (`no_pair`)

| benchmark | chrF++ (all rows) | AfriCOMET reference-based | SSA-COMET-MTL reference-based | AfriCOMET-QE (RL reward model) | SSA-COMET-MTL QE |
|---|---|---|---|---|---|
| FLORES+ | +3.19 (68%) n=45 | +0.062 (74%) n=42 | +0.061 (74%) n=42 | +0.073 (78%) n=42 | +0.070 (74%) n=42 |
| NTREX-128 | +2.19 (62%) n=24 | +0.053 (70%) n=22 | +0.050 (71%) n=22 | +0.056 (72%) n=22 | +0.057 (71%) n=22 |
| MAFAND-MT | +1.88 (57%) n=15 | +0.039 (66%) n=15 | +0.042 (66%) n=15 | +0.041 (70%) n=15 | +0.047 (68%) n=15 |
| SmolSent | +2.47 (63%) n=45 | +0.054 (71%) n=40 | +0.054 (71%) n=40 | +0.064 (74%) n=40 | +0.063 (71%) n=40 |
| IbomMT | +7.28 (87%) n=2 | +0.084 (78%) n=1 | +0.179 (87%) n=1 | +0.098 (77%) n=1 | +0.185 (86%) n=1 |
| FLORES-MOZ | +2.91 (71%) n=1 |  |  |  |  |
| SALT | +1.53 (59%) n=5 | +0.019 (63%) n=3 | +0.024 (62%) n=3 | +0.020 (62%) n=3 | +0.030 (61%) n=3 |
| GlotStoryBook | +3.92 (69%) n=8 | +0.043 (68%) n=4 | +0.048 (68%) n=4 | +0.049 (69%) n=4 | +0.054 (66%) n=4 |

Cells: macro delta (RL - SFT) (RL-preferred % of non-tied sentences, micro) and the number of languages. The neural metrics cover the AfriCOMET-covered languages only.

The ordering is the same as in R5: the largest gains on the weakest benchmarks (IbomMT +7.3 chrF++, GlotStoryBook +3.9, FLORES+ +3.2) and the smallest on the benchmark closest to the training domain (MAFAND-MT +1.9, SALT +1.5). RL is preferred by chrF++ in 57-87% of the non-tied sentences on every benchmark and by every neural metric in 61-87% (the upper end is IbomMT, one AfriCOMET-covered language).

### 2.4 By training group, language family and resource tier (`no_pair`, all benchmarks)

| scope | langs | chrF++ (all rows) | AfriCOMET reference-based | SSA-COMET-MTL reference-based | AfriCOMET-QE (RL reward model) | SSA-COMET-MTL QE |
|---|---|---|---|---|---|---|
| all_langs | 66 | +3.32 [+3.26, +3.38]; 64% | +0.059 [+0.058, +0.060]; 70% | +0.059 [+0.058, +0.060]; 70% | +0.069 [+0.068, +0.070]; 73% | +0.069 [+0.068, +0.070]; 71% |
| rl77 | 64 | +3.27 [+3.21, +3.33]; 63% | +0.059 [+0.058, +0.060]; 70% | +0.059 [+0.058, +0.060]; 70% | +0.069 [+0.068, +0.070]; 73% | +0.069 [+0.068, +0.070]; 71% |
| group:qe_validated | 42 | +2.96 [+2.91, +3.01]; 62% | +0.059 [+0.058, +0.060]; 70% | +0.060 [+0.059, +0.061]; 70% | +0.069 [+0.068, +0.070]; 73% | +0.068 [+0.067, +0.069]; 70% |
| group:blend | 9 | +3.91 [+3.71, +4.11]; 72% | +0.058 [+0.055, +0.061]; 74% | +0.056 [+0.053, +0.059]; 73% | +0.069 [+0.066, +0.072]; 76% | +0.071 [+0.067, +0.074]; 74% |
| family:Afroasiatic | 7 | +1.77 [+1.70, +1.86]; 62% | +0.068 [+0.066, +0.071]; 73% | +0.056 [+0.054, +0.058]; 72% | +0.079 [+0.077, +0.081]; 75% | +0.067 [+0.065, +0.069]; 72% |
| family:Atlantic-Congo (non-Bantu) | 13 | +4.30 [+4.15, +4.46]; 68% | +0.070 [+0.068, +0.072]; 74% | +0.071 [+0.069, +0.074]; 76% | +0.076 [+0.074, +0.079]; 79% | +0.079 [+0.076, +0.081]; 76% |
| family:Bantu | 35 | +2.91 [+2.82, +2.99]; 61% | +0.051 [+0.049, +0.052]; 67% | +0.054 [+0.053, +0.056]; 67% | +0.061 [+0.059, +0.062]; 69% | +0.064 [+0.063, +0.066]; 68% |
| family:Creole / Pidgin | 3 | +4.13 [+3.86, +4.40]; 70% | +0.017 [+0.015, +0.020]; 73% | +0.007 [+0.005, +0.009]; 65% | +0.024 [+0.021, +0.027]; 75% | +0.005 [+0.003, +0.007]; 56% |
| family:Mande | 2 | +4.66 [+4.43, +4.90]; 85% | +0.083 [+0.077, +0.087]; 86% | +0.079 [+0.075, +0.084]; 89% | +0.128 [+0.122, +0.134]; 90% | +0.097 [+0.092, +0.102]; 89% |
| family:Nilo-Saharan | 4 | +6.53 [+6.32, +6.71]; 81% | +0.121 [+0.114, +0.127]; 83% | +0.116 [+0.111, +0.121]; 84% | +0.119 [+0.114, +0.124]; 86% | +0.132 [+0.124, +0.137]; 84% |
| family:Other (Indo-European, Austronesian) | 2 | +0.57 [+0.42, +0.72]; 52% | +0.021 [+0.018, +0.024]; 61% | +0.015 [+0.012, +0.017]; 59% | +0.027 [+0.024, +0.030]; 63% | +0.018 [+0.015, +0.021]; 61% |
| script:Ge'ez (amh, tir) | 2 | +1.74 [+1.61, +1.88]; 63% | +0.100 [+0.096, +0.104]; 77% | +0.070 [+0.066, +0.074]; 74% | +0.119 [+0.115, +0.124]; 79% | +0.086 [+0.081, +0.090]; 74% |
| tier:strong | 5 | +1.84 [+1.73, +1.96]; 58% | +0.033 [+0.031, +0.035]; 65% | +0.034 [+0.033, +0.036]; 66% | +0.034 [+0.032, +0.035]; 66% | +0.037 [+0.036, +0.039]; 66% |
| tier:medium | 5 | +2.08 [+1.93, +2.21]; 59% | +0.045 [+0.043, +0.047]; 70% | +0.053 [+0.051, +0.055]; 71% | +0.048 [+0.046, +0.050]; 71% | +0.060 [+0.058, +0.063]; 71% |
| tier:weak | 10 | +4.72 [+4.60, +4.82]; 71% | +0.097 [+0.094, +0.100]; 77% | +0.081 [+0.079, +0.083]; 76% | +0.097 [+0.094, +0.099]; 80% | +0.090 [+0.087, +0.093]; 76% |
| group:consensus_only | 13 | +3.84 [+3.64, +4.05]; 75% |  |  |  |  |
| group:sft_only | 2 | +4.82 [+4.59, +5.04]; 77% |  |  |  |  |

Cell = macro delta [95% CI]; micro RL-preferred share. The groups the RL stage trained with a QE signal (`qe_validated`, `blend`) and the ones it trained with the consensus signal only (`consensus_only`: no AfriCOMET model exists for them, so only chrF++ can be computed, and RL is preferred in 75%) are all above 61% (chrF++: 62% for `qe_validated`, 72% `blend`, 75% `consensus_only`, 77% `sft_only`). `sft_only` (`nde`, `knc`; not RL-trained, zero-shot targets with no SFT data) gains +4.8 chrF++ (77%) almost entirely by no longer running away (`knc` finished 37% -> 100%). The gain is larger the weaker the language (weak tier +4.7 chrF++ / 71%, medium +2.1 / 59%, strong +1.8 / 58%), and smallest for the languages SFT already handled best (Malagasy +0.29, Sotho +0.29, Luganda +0.41 chrF++).

### 2.5 Per language

![Pairwise preference per language](../paper/figures/figpw2_languages.png)

Figure PW2 (`paper/figures/figpw2_languages.png`): RL-preferred share per language and metric (`no_pair`, all benchmarks pooled; `*` = BH-adjusted sign test q < 0.05). The full table is in the appendix (A1). Counting languages with at least 20 sentences: chrF++ 65 of 66 languages significantly better for RL, 1 neutral (`sot`; the lower end is `mlg` +0.29, `sot` +0.29, `lug` +0.41, `twi` +0.75), none worse; AfriCOMET-ref, SSA-ref, AfriCOMET-QE and SSA-QE: 51 of 51 significantly better, none worse. On `clean`: chrF++ 63 better / 3 neutral / 0 worse. At the (benchmark, language) level (`no_pair`, n >= 20): chrF++ 128 of 145 cells significantly better, 16 neutral, 1 worse (SALT `lug`, 81 sentences); the four neural metrics 123-126 of 127 better, 1-4 neutral, none worse.

![Win, tie and loss per metric](../paper/figures/figpw1_wtl.png)

Figure PW1 (`paper/figures/figpw1_wtl.png`): A, macro win / tie / loss per metric; B, RL-preferred share per benchmark; C, per training group.

## 3. Failure modes, language identification and length (`no_pair`, macro over languages)

| scope | finished | loop (guards) | word-3-gram repetition | GlotLID right language | GlotLID (sibling-tolerant) | length ratio hyp/ref |
|---|---|---|---|---|---|---|
| all_langs | 88.6 -> 99.7 (+11.1 [+10.8, +11.3]) | 13.0 -> 1.0 (-12.0 [-12.3, -11.8]) | 11.5 -> 0.5 (-11.0 [-11.3, -10.7]) | 91.3 -> 94.3 (+3.0 [+2.8, +3.2]) | 94.2 -> 96.8 (+2.6 [+2.4, +2.8]) | 1.51 -> 1.15 |
| group:qe_validated | 91.9 -> 100.0 (+8.1 [+7.9, +8.3]) | 9.2 -> 0.1 (-9.1 [-9.3, -9.0]) | 8.7 -> 0.0 (-8.7 [-8.9, -8.5]) | 96.2 -> 98.1 (+1.9 [+1.8, +2.0]) | 97.6 -> 99.1 (+1.5 [+1.4, +1.6]) | 1.37 -> 1.07 |
| group:blend | 86.1 -> 99.8 (+13.6 [+12.9, +14.4]) | 16.5 -> 1.1 (-15.5 [-16.2, -14.7]) | 15.8 -> 0.5 (-15.3 [-16.0, -14.5]) | 95.1 -> 97.0 (+1.9 [+1.4, +2.5]) | 96.6 -> 97.9 (+1.4 [+0.8, +1.9]) | 1.68 -> 1.18 |
| group:consensus_only | 83.0 -> 98.9 (+15.9 [+14.8, +16.8]) | 19.1 -> 3.1 (-16.0 [-17.0, -14.9]) | 13.7 -> 2.0 (-11.7 [-12.7, -10.6]) | 86.8 -> 94.4 (+7.7 [+6.8, +8.5]) | 89.0 -> 96.2 (+7.2 [+6.5, +8.0]) | 1.74 -> 1.38 |
| group:sft_only | 68.2 -> 100.0 (+31.7 [+30.3, +33.3]) | 38.0 -> 6.2 (-31.8 [-33.5, -30.2]) | 36.8 -> 1.3 (-35.4 [-36.9, -33.9]) | 1.3 -> 1.3 (-0.0 [-0.4, +0.4]) | 49.5 -> 49.8 (+0.3 [+0.1, +0.5]) | 2.32 -> 1.13 |

RL finishes 99.7% of the sentences (SFT 88.6%), loops in 1.0% (13.0%), and the hypothesis length moves from 1.51 to 1.15 x the reference. GlotLID: right language 91.3% -> 94.3%. The two groups where SFT is worst, `consensus_only` (finished 83% -> 99%, LID 87% -> 94%) and `sft_only`, gain most. `sft_only` LID is about 1% for both models, as in R5 (`nde` / `knc` are zero-shot targets without SFT data: the model does not write their tag and the output is not recognised as the language).

## 4. How much of the preference is the removal of loops? (robustness)

| sentences | metric | n | mean delta (micro) | macro delta over languages | win / tie / loss % | RL preferred % |
|---|---|---|---|---|---|---|
| all rows | chrF++ (all rows) | 153220 | +2.60 | +3.32 | 50.7 / 20.4 / 28.9 | 63.7 |
| all rows | chrF++ (rows scored by COMET) | 76804 | +2.45 | +3.21 | 49.6 / 20.1 / 30.3 | 62.1 |
| all rows | AfriCOMET reference-based | 76804 | +0.050 | +0.059 | 60.4 / 13.8 / 25.8 | 70.0 |
| all rows | SSA-COMET-MTL reference-based | 76804 | +0.051 | +0.059 | 60.6 / 14.0 / 25.5 | 70.4 |
| all rows | AfriCOMET-QE (RL reward model) | 76804 | +0.057 | +0.069 | 63.5 / 13.0 / 23.6 | 72.9 |
| all rows | SSA-COMET-MTL QE | 76804 | +0.058 | +0.069 | 61.7 / 12.8 / 25.5 | 70.8 |
| neither model loops / unfinished | chrF++ (all rows) | 140204 | +1.41 | +1.67 | 47.1 / 21.4 / 31.5 | 60.0 |
| neither model loops / unfinished | chrF++ (rows scored by COMET) | 71794 | +1.33 | +1.57 | 46.2 / 21.5 / 32.3 | 58.8 |
| neither model loops / unfinished | AfriCOMET reference-based | 71794 | +0.035 | +0.039 | 57.7 / 14.7 / 27.6 | 67.7 |
| neither model loops / unfinished | SSA-COMET-MTL reference-based | 71794 | +0.040 | +0.048 | 58.4 / 14.8 / 26.8 | 68.6 |
| neither model loops / unfinished | AfriCOMET-QE (RL reward model) | 71794 | +0.040 | +0.048 | 61.0 / 13.8 / 25.1 | 70.8 |
| neither model loops / unfinished | SSA-COMET-MTL QE | 71794 | +0.045 | +0.053 | 59.3 / 13.6 / 27.1 | 68.7 |
| neither loops, both length ratio in [0.5, 2] | chrF++ (all rows) | 136993 | +1.35 | +1.54 | 46.9 / 21.4 / 31.7 | 59.6 |
| neither loops, both length ratio in [0.5, 2] | chrF++ (rows scored by COMET) | 70207 | +1.28 | +1.48 | 45.9 / 21.5 / 32.6 | 58.5 |
| neither loops, both length ratio in [0.5, 2] | AfriCOMET reference-based | 70207 | +0.034 | +0.038 | 57.5 / 14.8 / 27.7 | 67.5 |
| neither loops, both length ratio in [0.5, 2] | SSA-COMET-MTL reference-based | 70207 | +0.039 | +0.047 | 58.1 / 14.9 / 27.0 | 68.3 |
| neither loops, both length ratio in [0.5, 2] | AfriCOMET-QE (RL reward model) | 70207 | +0.040 | +0.047 | 60.9 / 13.9 / 25.2 | 70.7 |
| neither loops, both length ratio in [0.5, 2] | SSA-COMET-MTL QE | 70207 | +0.044 | +0.051 | 59.1 / 13.7 / 27.2 | 68.4 |

The macro delta is halved (chrF++ +3.32 -> +1.67; AfriCOMET-ref +0.059 -> +0.039; SSA-ref +0.059 -> +0.048) but the *preference* barely moves (chrF++ 63.7 -> 60.0%; AfriCOMET-ref 70.0 -> 67.7%; SSA-ref 70.4 -> 68.6%; AfriCOMET-QE 72.9 -> 70.8%; SSA-QE 70.8 -> 68.7%): the looping sentences are 8.5% of the rows but carry about half of the average gain; on the other 91.5% of the sentences RL still wins about 3 non-tied sentences in 5 (chrF++) to 2 in 3 (COMET) in every language group.

## 5. Sensitivity

### 5.1 Tie threshold (`no_pair`, all sentences of the metric, pooled)

| metric | threshold | n | win | tie | loss | pref |
|---|---|---|---|---|---|---|
| chrF++ (all rows) | 0.0 | 153220 | 60.3 | 2.4 | 37.3 | 61.8 |
| chrF++ (all rows) | 0.5 | 153220 | 55.3 | 12.0 | 32.7 | 62.8 |
| chrF++ (all rows) | 1.0 | 153220 | 50.7 | 20.4 | 28.9 | 63.7 |
| chrF++ (all rows) | 2.0 | 153220 | 43.0 | 34.5 | 22.5 | 65.7 |
| chrF++ (all rows) | 5.0 | 153220 | 26.8 | 62.5 | 10.7 | 71.5 |
| AfriCOMET reference-based | 0.0 | 76804 | 66.0 | 2.9 | 31.1 | 68.0 |
| AfriCOMET reference-based | 0.005 | 76804 | 63.2 | 8.4 | 28.4 | 69.0 |
| AfriCOMET reference-based | 0.01 | 76804 | 60.4 | 13.8 | 25.8 | 70.0 |
| AfriCOMET reference-based | 0.02 | 76804 | 54.8 | 24.1 | 21.1 | 72.2 |
| AfriCOMET reference-based | 0.05 | 76804 | 39.8 | 48.9 | 11.3 | 77.9 |
| AfriCOMET-QE (RL reward model) | 0.0 | 76804 | 68.7 | 2.8 | 28.5 | 70.7 |
| AfriCOMET-QE (RL reward model) | 0.005 | 76804 | 66.1 | 8.0 | 25.9 | 71.8 |
| AfriCOMET-QE (RL reward model) | 0.01 | 76804 | 63.5 | 13.0 | 23.6 | 72.9 |
| AfriCOMET-QE (RL reward model) | 0.02 | 76804 | 58.0 | 22.7 | 19.4 | 75.0 |
| AfriCOMET-QE (RL reward model) | 0.05 | 76804 | 42.9 | 46.6 | 10.5 | 80.4 |
| SSA-COMET-MTL reference-based | 0.0 | 76804 | 66.3 | 2.8 | 30.9 | 68.2 |
| SSA-COMET-MTL reference-based | 0.005 | 76804 | 63.4 | 8.6 | 28.0 | 69.4 |
| SSA-COMET-MTL reference-based | 0.01 | 76804 | 60.6 | 14.0 | 25.5 | 70.4 |
| SSA-COMET-MTL reference-based | 0.02 | 76804 | 55.2 | 23.8 | 21.0 | 72.4 |
| SSA-COMET-MTL reference-based | 0.05 | 76804 | 41.3 | 46.9 | 11.8 | 77.7 |
| SSA-COMET-MTL QE | 0.0 | 76804 | 66.8 | 2.8 | 30.3 | 68.8 |
| SSA-COMET-MTL QE | 0.005 | 76804 | 64.3 | 7.8 | 27.9 | 69.7 |
| SSA-COMET-MTL QE | 0.01 | 76804 | 61.7 | 12.8 | 25.5 | 70.8 |
| SSA-COMET-MTL QE | 0.02 | 76804 | 56.7 | 21.8 | 21.5 | 72.5 |
| SSA-COMET-MTL QE | 0.05 | 76804 | 43.9 | 43.2 | 12.8 | 77.4 |

The preference is stable: 62-72% (chrF++) for thresholds 0-5 and 68-80% (neural) for 0-0.05. With threshold 0 (no ties) RL wins 60.3% / 66.0% / 66.3% (chrF++ / AfriCOMET-ref / SSA-ref) of all sentences.

### 5.2 Decoding budget: FLORES+ at 320 and at 768 tokens (`no_pair`; macro over 45 languages)

| subset | metric | SFT | RL | delta [95% CI] | win/tie/loss % (macro) | RL preferred % (micro) |
|---|---|---|---|---|---|---|
| no_pair | chrF++ | 30.93 | 33.84 | +2.91 [+2.83, +2.98] | 55.1 / 17.8 / 27.1 | 67.4 |
| no_pair | finished | 89.91 | 99.99 | +10.08 [+9.85, +10.31] |  |  |
| no_pair | loop | 11.38 | 0.40 | -10.97 [-11.23, -10.71] |  |  |
| no_pair | GlotLID | 95.26 | 96.75 | +1.48 [+1.33, +1.63] |  |  |
| no_pair | length ratio | 1.34 | 1.07 | -0.27 [-0.27, -0.26] |  |  |
| clean | chrF++ | 27.46 | 30.99 | +3.53 [+3.24, +3.84] | 58.1 / 16.9 / 25.0 | 69.9 |
| clean | finished | 85.69 | 100.00 | +14.31 [+13.01, +15.46] |  |  |
| clean | loop | 15.74 | 0.28 | -15.46 [-16.67, -14.17] |  |  |
| clean | GlotLID | 94.17 | 96.94 | +2.78 [+1.90, +3.57] |  |  |
| clean | length ratio | 1.47 | 1.08 | -0.39 [-0.43, -0.35] |  |  |
| all | chrF++ | 31.13 | 34.07 | +2.94 [+2.87, +3.02] | 55.3 / 17.7 / 27.0 | 67.1 |
| all | finished | 89.90 | 99.99 | +10.09 [+9.84, +10.32] |  |  |
| all | loop | 11.38 | 0.41 | -10.97 [-11.23, -10.72] |  |  |
| all | GlotLID | 95.26 | 96.75 | +1.49 [+1.34, +1.64] |  |  |
| all | length ratio | 1.34 | 1.07 | -0.27 [-0.28, -0.26] |  |  |

| subset | metric | SFT | RL | delta [95% CI] | win/tie/loss % (macro) | RL preferred % (micro) |
|---|---|---|---|---|---|---|
| no_pair | chrF++ | 30.63 | 33.83 | +3.19 [+3.11, +3.27] | 55.2 / 17.8 / 27.0 | 67.6 |
| no_pair | finished | 90.44 | 100.00 | +9.55 [+9.32, +9.81] |  |  |
| no_pair | loop | 11.17 | 0.39 | -10.78 [-11.04, -10.53] |  |  |
| no_pair | GlotLID | 95.21 | 96.78 | +1.56 [+1.42, +1.73] |  |  |
| no_pair | length ratio | 1.42 | 1.07 | -0.35 [-0.36, -0.34] |  |  |
| clean | chrF++ | 26.98 | 31.00 | +4.02 [+3.69, +4.37] | 58.1 / 17.1 / 24.8 | 70.1 |
| clean | finished | 86.34 | 100.00 | +13.66 [+12.41, +15.00] |  |  |
| clean | loop | 15.46 | 0.14 | -15.32 [-16.62, -13.98] |  |  |
| clean | GlotLID | 94.35 | 96.94 | +2.59 [+1.76, +3.38] |  |  |
| clean | length ratio | 1.58 | 1.08 | -0.50 [-0.55, -0.45] |  |  |
| all | chrF++ | 30.83 | 34.06 | +3.23 [+3.16, +3.30] | 55.4 / 17.7 / 26.9 | 67.4 |
| all | finished | 90.47 | 100.00 | +9.53 [+9.29, +9.76] |  |  |
| all | loop | 11.14 | 0.39 | -10.75 [-10.98, -10.51] |  |  |
| all | GlotLID | 95.20 | 96.78 | +1.57 [+1.42, +1.73] |  |  |
| all | length ratio | 1.41 | 1.07 | -0.35 [-0.36, -0.34] |  |  |

At 320 tokens the SFT model is truncated earlier (R5: its runaways lengthen with the budget): chrF++ delta +2.91 [+2.83, +2.98] against +3.19 [+3.11, +3.27] at 768; RL preferred in 67.4% vs 67.6% of the non-tied sentences. The preference does not depend on the budget; the size of the delta does.

### 5.3 Reproduction of R5

| benchmark | langs | SFT here | SFT R5 | RL here | RL R5 | max abs diff per language |
|---|---|---|---|---|---|---|
| FLORES+ | 45 | 28.68 | 28.67 | 34.08 | 34.09 | 0.483 |
| NTREX-128 | 24 | 33.21 | 33.22 | 37.09 | 37.08 | 0.303 |
| MAFAND-MT | 15 | 38.36 | 38.36 | 42.04 | 42.02 | 0.148 |
| SmolSent | 45 | 26.08 | 26.08 | 31.44 | 31.44 | 0.206 |
| IbomMT | 2 | 15.72 | 15.72 | 26.76 | 26.75 | 0.010 |
| FLORES-MOZ | 1 | 23.66 | 23.66 | 29.70 | 29.70 | 0.003 |
| SALT | 5 | 35.77 | 35.77 | 38.57 | 38.46 | 0.543 |
| GlotStoryBook | 8 | 19.86 | 19.86 | 27.84 | 28.02 | 0.857 |


## 6. Agreement between metrics, and the metric-gaming question

### 6.1 Do the metrics pick the same winner?

| pair | sentences both call a winner | same winner % | same 3-class label % | Cohen kappa (3 class) | Spearman of the deltas |
|---|---|---|---|---|---|
| chrF++ vs AfriC-ref | 54853 | 72.7 | 57.3 | 0.28 | 0.50 |
| chrF++ vs AfriC-QE | 55152 | 64.1 | 51.0 | 0.17 | 0.32 |
| chrF++ vs SSA-ref | 54668 | 70.6 | 55.6 | 0.25 | 0.44 |
| chrF++ vs SSA-QE | 55248 | 65.5 | 52.0 | 0.19 | 0.35 |
| AfriC-ref vs AfriC-QE | 60395 | 87.0 | 73.8 | 0.51 | 0.79 |
| AfriC-ref vs SSA-ref | 59554 | 82.1 | 69.0 | 0.43 | 0.67 |
| AfriC-ref vs SSA-QE | 60215 | 79.0 | 66.9 | 0.39 | 0.61 |
| AfriC-QE vs SSA-ref | 59957 | 81.2 | 68.4 | 0.41 | 0.64 |
| AfriC-QE vs SSA-QE | 60747 | 80.9 | 68.9 | 0.41 | 0.65 |
| SSA-ref vs SSA-QE | 61660 | 94.9 | 83.2 | 0.69 | 0.92 |

Sentences where RL wins on BOTH QE metrics (38604): the majority of the three reference-based metrics prefers RL in 89.7%, is neutral in 5.0%, prefers SFT in 5.3%.

Agreement of chrF++ with each neural metric by training group (same winner %, sentences where both call a winner):

| group | metric_2 | n_both_nontie | same winner % |
|---|---|---|---|
| qe_validated | afri_ref | 50776 | 72.4 |
| qe_validated | ssa_ref | 50514 | 70.4 |
| qe_validated | afri_qe | 50981 | 63.7 |
| qe_validated | ssa_qe | 51020 | 65.1 |
| blend | afri_ref | 4077 | 76.7 |
| blend | ssa_ref | 4154 | 73.5 |
| blend | afri_qe | 4171 | 69.4 |
| blend | ssa_qe | 4228 | 70.4 |

* chrF++ and the neural metrics agree on the sentence winner in 64-73% of the sentences where both call one (Cohen's kappa 0.17-0.28 over three classes): they are weakly related at sentence level, as is well known; they agree most with the reference-based COMET metrics (73 / 71%) and least with the QE metrics (64 / 66%).
* The two AfriCOMET variants agree in 87%, the two SSA-COMET variants in 95%, the different model families in 79-82%.
* Because the metrics are only weakly correlated sentence by sentence, the pooled preference of each is the meaningful quantity, and the fact that **all five metrics give 62-73%** with different error sources (surface overlap, two different learned models, with and without reference) is the central evidence.

### 6.2 Is RL optimising the QE at the expense of the references?

* The RL reward model (AfriCOMET-QE) prefers RL in 72.9% of non-tied sentences; the independent reference-based metrics 70.0-70.4% and the independent SSA QE 70.8%. A gap of 2-3 points is small: had the policy learned to exploit AfriCOMET-QE, the QE should prefer RL much more than the references do and the two should disagree on the "wins" (the QE-win sentences would be neutral or SFT-better under the reference).
* It does not: of the **38,604** sentences on which RL wins on both QE metrics, the majority of the three reference-based metrics prefers RL in **89.7%**, is neutral in 5.0% and prefers SFT in 5.3%.
* Per training group the agreement of chrF++ with each neural metric is stable (QE-validated 64-72%, blend 69-77%).

### 6.3 Languages where RL wins on QE but not on the reference-based metrics

Rule (`flags.csv`): RL wins on AfriCOMET-QE or SSA-QE with the 95% CI of the mean delta above 0, but not (CI above 0) on all of chrF++, AfriCOMET-ref, SSA-ref.

* **All benchmarks pooled, `no_pair` (51 languages): one language.** `sot` (Southern Sotho): QE +0.032 / +0.030 (significant), AfriCOMET-ref +0.027, SSA-ref +0.027 (significant, RL-preferred 65% / 64%), chrF++ +0.29 [-0.01, +0.62] (RL-preferred 51%). Not a contradiction: the neural metrics, with and without reference, say better; chrF++ says no change.
* **Per benchmark (127 cells): 20 flagged.** 19: QE up, AfriCOMET-ref up (and SSA-ref up in 18; FLORES+ `afr` is the exception), chrF++ flat (not significantly above 0; all deltas between -0.10 and +0.93): FLORES+ `afr` (also SSA-ref not significant), `ibo`, `mlg`, `nso`, `sot`, `xho`, `zul`; MAFAND-MT `hau`, `lug`, `tsn`, `twi`, `zul`; NTREX `mlg`; SmolSent `lug`, `pcm` (SSA-QE flat), `som`, `sot`, `tso`, `zul`. 1: **SALT `lug`**, the only cell in which a reference-based metric is significantly *down*: chrF++ -2.98 [-5.84, -0.47] on 81 sentences, AfriCOMET-ref / SSA-ref +0.001 (flat) with QE +0.025 / +0.023 (the QE-up that is not supported by the reference). With 81 sentences and one benchmark this is one weak signal in 127 cells, and exactly the kind of case a human check would settle.
* Nothing flagged is a language where RL is clearly worse; the flagged cells are the high-resource languages where SFT is already good and chrF++ resolves less than 1 point.

## 7. Long-input check (R6 documents, both models decoded sentence by sentence)

Data: the `greedy_sent` hypotheses of agent A2 for both models (`/workspace/jobs/a2/hyp/{sft,rl2c}/greedy_sent.jsonl`; the SFT model was run on the 8-language subset hau yor swh amh zul wol kea sag, so the comparison is on those 4,282 clean documents, 8 languages; `core` documents have an NLLB-200 pseudo-reference sentence by sentence, `human_*` documents (SmolDoc, and SmolDoc stitched to about 512 / 1,024 / 2,048 tokens) have human references; `human_st2048` has only 5 clean documents of one language). Whole documents: chrF++ of the joined text, finished = every sentence unit ended with the stop token, loop = guards on the joined text. Per sentence: the aligned sentences (1:1 with the source sentences) of bins b64 ... b1024 and the human bins up to ~1,024 tokens, neural metrics on at most 500 sentences per (bin, language) (7 of the 8 languages for AfriCOMET: `kea` is not AfriCOMET-covered). Decoding is sentence by sentence for both models, so the long-document failure of whole-document decoding (R6) is not in play; what is tested is whether the RL-vs-SFT difference survives long inputs when both are used in the setting that works.

![Agreement and long input](../paper/figures/figpw3_agreement.png)

Figure PW3: A, winner agreement between metrics; B, what the reference-based majority says about the sentences on which RL wins both QE metrics; C, per-sentence chrF++ win / tie / loss per bin.

| bin | metric | langs | units | SFT | RL | delta [95% CI] | win/tie/loss % (micro) | RL preferred % [CI] |
|---|---|---|---|---|---|---|---|---|
| core_b64 | chrF++ (all rows) | 8 | 584 | 44.69 | 48.10 | +3.41 [+2.50, +4.37] | 54 / 15 / 31 | 63 [59, 68] |
| core_b64 | finished % | 8 | 584 | 92.6 | 98.8 | +6.2 [+4.1, +8.4] | 7 / 92 / 1 | 88 [78, 96] |
| core_b64 | loop % | 8 | 584 | 7.5 | 1.2 | -6.3 [-8.6, -4.3] | 7 / 92 / 1 | 88 [79, 96] |
| core_b256 | chrF++ (all rows) | 8 | 720 | 45.59 | 48.83 | +3.23 [+2.74, +3.76] | 60 / 16 / 25 | 71 [67, 74] |
| core_b256 | finished % | 8 | 720 | 71.5 | 98.1 | +26.5 [+23.8, +29.4] | 28 / 72 / 1 | 97 [94, 99] |
| core_b256 | loop % | 8 | 720 | 29.0 | 2.9 | -26.1 [-29.0, -23.3] | 27 / 72 / 1 | 97 [94, 99] |
| core_b512 | chrF++ (all rows) | 8 | 800 | 51.30 | 53.56 | +2.26 [+1.97, +2.54] | 54 / 28 / 18 | 75 [72, 78] |
| core_b512 | finished % | 8 | 800 | 48.4 | 94.1 | +45.8 [+43.0, +48.6] | 47 / 52 / 1 | 97 [96, 99] |
| core_b512 | loop % | 8 | 800 | 54.9 | 21.1 | -33.8 [-36.9, -30.8] | 35 / 63 / 2 | 96 [94, 98] |
| core_b1024 | chrF++ (all rows) | 8 | 712 | 53.38 | 55.69 | +2.31 [+2.03, +2.58] | 56 / 34 / 11 | 84 [81, 87] |
| core_b1024 | finished % | 8 | 712 | 39.2 | 92.3 | +53.1 [+50.4, +55.9] | 54 / 46 / 1 | 99 [98, 100] |
| core_b1024 | loop % | 8 | 712 | 62.2 | 32.3 | -29.9 [-32.7, -27.2] | 31 / 68 / 1 | 97 [95, 99] |
| core_b2048 | chrF++ (all rows) | 8 | 504 | 56.32 | 58.39 | +2.07 [+1.76, +2.41] | 50 / 37 / 12 | 80 [77, 85] |
| core_b2048 | finished % | 8 | 504 | 24.0 | 85.9 | +61.9 [+58.3, +65.5] | 62 / 38 / 0 | 100 [99, 100] |
| core_b2048 | loop % | 8 | 504 | 77.8 | 43.1 | -34.7 [-37.9, -31.7] | 35 / 65 / 0 | 99 [98, 100] |
| core_b4096 | chrF++ (all rows) | 8 | 280 | 57.87 | 60.20 | +2.33 [+1.87, +2.78] | 48 / 39 / 12 | 79 [74, 85] |
| core_b4096 | finished % | 8 | 280 | 13.2 | 81.8 | +68.6 [+64.3, +72.9] | 69 / 31 / 0 | 100 [100, 100] |
| core_b4096 | loop % | 8 | 280 | 87.9 | 59.6 | -28.2 [-32.5, -23.9] | 28 / 72 / 0 | 100 [100, 100] |
| human_sm | chrF++ (all rows) | 6 | 360 | 44.19 | 46.84 | +2.64 [+2.18, +3.13] | 56 / 29 / 14 | 80 [75, 85] |
| human_sm | finished % | 6 | 360 | 83.9 | 98.3 | +14.4 [+11.1, +18.1] | 15 / 84 / 1 | 95 [89, 100] |
| human_sm | loop % | 6 | 360 | 16.4 | 1.9 | -14.4 [-18.3, -10.8] | 16 / 83 / 1 | 93 [87, 98] |
| human_st512 | chrF++ (all rows) | 6 | 224 | 46.98 | 49.03 | +2.05 [+1.57, +2.51] | 47 / 41 / 12 | 79 [72, 86] |
| human_st512 | finished % | 6 | 224 | 75.9 | 96.7 | +20.8 [+15.6, +26.0] | 24 / 74 / 3 | 90 [82, 97] |
| human_st512 | loop % | 6 | 224 | 24.1 | 4.2 | -19.9 [-25.1, -14.7] | 23 / 75 / 3 | 89 [82, 96] |
| human_st1024 | chrF++ (all rows) | 4 | 80 | 47.03 | 50.74 | +3.70 [+3.04, +4.42] | 68 / 32 / 0 | 100 [100, 100] |
| human_st1024 | finished % | 4 | 80 | 53.8 | 96.2 | +42.5 [+32.5, +52.5] | 44 / 55 / 1 | 97 [91, 100] |
| human_st1024 | loop % | 4 | 80 | 46.2 | 3.8 | -42.5 [-52.5, -32.5] | 44 / 55 / 1 | 97 [91, 100] |
| human_st2048 | chrF++ (all rows) | 1 | 5 | 60.68 | 61.92 | +1.24 [+0.64, +2.02] | 60 / 40 / 0 | 100 [100, 100] |
| human_st2048 | finished % | 1 | 5 | 40.0 | 100.0 | +60.0 [+20.0, +100.0] | 60 / 40 / 0 | 100 [100, 100] |
| human_st2048 | loop % | 1 | 5 | 60.0 | 0.0 | -60.0 [-100.0, -20.0] | 60 / 40 / 0 | 100 [100, 100] |

| bin | metric | langs | units | SFT | RL | delta [95% CI] | win/tie/loss % (micro) | RL preferred % [CI] |
|---|---|---|---|---|---|---|---|---|
| core_b64 | chrF++ (all rows) | 8 | 960 | 43.20 | 46.71 | +3.51 [+2.64, +4.33] | 50 / 19 / 31 | 62 [59, 65] |
| core_b64 | AfriCOMET reference-based | 7 | 840 | 0.645 | 0.697 | +0.053 [+0.045, +0.061] | 57 / 17 / 26 | 68 [65, 72] |
| core_b64 | SSA-COMET-MTL reference-based | 7 | 840 | 0.486 | 0.536 | +0.050 [+0.042, +0.058] | 58 / 16 / 25 | 70 [66, 73] |
| core_b64 | AfriCOMET-QE (RL reward model) | 7 | 840 | 0.591 | 0.641 | +0.050 [+0.043, +0.056] | 62 / 16 / 23 | 73 [70, 76] |
| core_b64 | SSA-COMET-MTL QE | 7 | 840 | 0.480 | 0.531 | +0.051 [+0.043, +0.059] | 57 / 16 / 27 | 68 [65, 71] |
| core_b64 | loop % | 8 | 960 | 4.2 | 0.1 | -4.1 [-5.3, -2.9] | 4 / 96 / 0 | 100 [100, 100] |
| core_b256 | chrF++ (all rows) | 8 | 4304 | 40.18 | 43.82 | +3.65 [+3.23, +4.05] | 52 / 17 / 31 | 62 [61, 64] |
| core_b256 | AfriCOMET reference-based | 7 | 3766 | 0.619 | 0.671 | +0.052 [+0.048, +0.055] | 57 / 15 / 28 | 67 [65, 69] |
| core_b256 | SSA-COMET-MTL reference-based | 7 | 3766 | 0.462 | 0.511 | +0.049 [+0.045, +0.052] | 58 / 15 / 26 | 69 [67, 70] |
| core_b256 | AfriCOMET-QE (RL reward model) | 7 | 3766 | 0.561 | 0.610 | +0.050 [+0.046, +0.053] | 60 / 15 / 25 | 71 [69, 72] |
| core_b256 | SSA-COMET-MTL QE | 7 | 3766 | 0.455 | 0.506 | +0.051 [+0.047, +0.055] | 59 / 14 / 27 | 68 [67, 70] |
| core_b256 | loop % | 8 | 4304 | 6.6 | 0.2 | -6.3 [-7.0, -5.6] | 6 / 94 / 0 | 99 [98, 100] |
| core_b512 | chrF++ (all rows) | 8 | 16264 | 41.88 | 44.81 | +2.93 [+2.73, +3.13] | 51 / 16 / 33 | 61 [60, 61] |
| core_b512 | AfriCOMET reference-based | 7 | 14231 | 0.628 | 0.677 | +0.049 [+0.047, +0.051] | 57 / 15 / 28 | 67 [66, 68] |
| core_b512 | SSA-COMET-MTL reference-based | 7 | 14231 | 0.467 | 0.517 | +0.050 [+0.048, +0.052] | 59 / 15 / 26 | 69 [68, 70] |
| core_b512 | AfriCOMET-QE (RL reward model) | 7 | 14231 | 0.566 | 0.616 | +0.050 [+0.048, +0.052] | 61 / 15 / 25 | 71 [70, 72] |
| core_b512 | SSA-COMET-MTL QE | 7 | 14231 | 0.459 | 0.511 | +0.052 [+0.050, +0.053] | 59 / 14 / 26 | 69 [69, 70] |
| core_b512 | loop % | 8 | 16264 | 6.2 | 0.2 | -6.0 [-6.4, -5.7] | 6 / 94 / 0 | 99 [98, 99] |
| core_b1024 | chrF++ (all rows) | 8 | 29408 | 41.17 | 44.28 | +3.12 [+2.97, +3.26] | 51 / 16 / 33 | 60 [60, 61] |
| core_b1024 | AfriCOMET reference-based | 7 | 25732 | 0.622 | 0.671 | +0.049 [+0.048, +0.050] | 57 / 15 / 28 | 67 [66, 67] |
| core_b1024 | SSA-COMET-MTL reference-based | 7 | 25732 | 0.465 | 0.514 | +0.049 [+0.048, +0.050] | 58 / 15 / 27 | 68 [68, 69] |
| core_b1024 | AfriCOMET-QE (RL reward model) | 7 | 25732 | 0.562 | 0.611 | +0.048 [+0.047, +0.050] | 60 / 15 / 25 | 71 [70, 71] |
| core_b1024 | SSA-COMET-MTL QE | 7 | 25732 | 0.458 | 0.509 | +0.050 [+0.049, +0.052] | 59 / 14 / 27 | 69 [68, 69] |
| core_b1024 | loop % | 8 | 29408 | 5.7 | 0.2 | -5.5 [-5.8, -5.3] | 6 / 94 / 0 | 99 [98, 99] |
| human_sm | chrF++ (all rows) | 6 | 4273 | 37.08 | 40.10 | +3.01 [+2.72, +3.31] | 48 / 21 / 31 | 61 [59, 63] |
| human_sm | AfriCOMET reference-based | 6 | 4273 | 0.659 | 0.706 | +0.046 [+0.043, +0.049] | 56 / 18 / 26 | 68 [67, 70] |
| human_sm | SSA-COMET-MTL reference-based | 6 | 4273 | 0.511 | 0.553 | +0.042 [+0.039, +0.046] | 55 / 17 / 28 | 66 [65, 68] |
| human_sm | AfriCOMET-QE (RL reward model) | 6 | 4273 | 0.637 | 0.676 | +0.039 [+0.036, +0.042] | 55 / 19 / 26 | 68 [66, 69] |
| human_sm | SSA-COMET-MTL QE | 6 | 4273 | 0.515 | 0.557 | +0.042 [+0.039, +0.046] | 55 / 16 / 29 | 65 [64, 67] |
| human_sm | loop % | 6 | 4273 | 1.7 | 0.0 | -1.6 [-2.1, -1.2] | 2 / 98 / 0 | 99 [95, 100] |
| human_st512 | chrF++ (all rows) | 6 | 5000 | 37.77 | 40.19 | +2.42 [+2.13, +2.70] | 48 / 21 / 31 | 60 [59, 62] |
| human_st512 | AfriCOMET reference-based | 6 | 5000 | 0.664 | 0.707 | +0.043 [+0.040, +0.046] | 56 / 19 / 26 | 69 [67, 70] |
| human_st512 | SSA-COMET-MTL reference-based | 6 | 5000 | 0.515 | 0.555 | +0.040 [+0.037, +0.043] | 55 / 17 / 27 | 67 [66, 68] |
| human_st512 | AfriCOMET-QE (RL reward model) | 6 | 5000 | 0.638 | 0.676 | +0.038 [+0.035, +0.041] | 55 / 18 / 26 | 68 [66, 69] |
| human_st512 | SSA-COMET-MTL QE | 6 | 5000 | 0.517 | 0.560 | +0.042 [+0.039, +0.046] | 56 / 16 / 28 | 66 [65, 68] |
| human_st512 | loop % | 6 | 5000 | 1.2 | 0.0 | -1.1 [-1.4, -0.8] | 1 / 99 / 0 | 100 [100, 100] |
| human_st1024 | chrF++ (all rows) | 6 | 4259 | 35.26 | 38.34 | +3.08 [+2.48, +3.65] | 52 / 20 / 28 | 65 [64, 67] |
| human_st1024 | AfriCOMET reference-based | 6 | 4259 | 0.643 | 0.690 | +0.047 [+0.040, +0.054] | 59 / 17 / 24 | 71 [70, 73] |
| human_st1024 | SSA-COMET-MTL reference-based | 6 | 4259 | 0.501 | 0.543 | +0.043 [+0.036, +0.049] | 57 / 17 / 26 | 69 [67, 70] |
| human_st1024 | AfriCOMET-QE (RL reward model) | 6 | 4259 | 0.623 | 0.666 | +0.044 [+0.038, +0.050] | 56 / 18 / 26 | 68 [67, 70] |
| human_st1024 | SSA-COMET-MTL QE | 6 | 4259 | 0.504 | 0.548 | +0.043 [+0.035, +0.051] | 57 / 16 / 27 | 67 [66, 69] |
| human_st1024 | loop % | 6 | 4259 | 2.1 | 0.0 | -2.1 [-3.3, -1.1] | 1 / 99 / 0 | 100 [100, 100] |

Reading: (i) whole documents: RL is preferred in 63% (b64) to 84% (b1024) of the non-tied documents by chrF++, and on the human-reference documents in 80% (short), 79% (~512) and 100% (~1,024, 80 documents); **the gap in finished documents grows with length** (b64 +6 points, b256 +27, b512 +46, b1024 +53, b2048 +62, b4096 +69): the SFT model still does not stop reliably after a sentence, whatever the chunking, which makes a document-level quality comparison an unfair one for it, so the **per-sentence** rows are the cleaner evidence; (ii) per aligned sentence the preference is flat across length, 60-62% (chrF++), 67-71% (AfriCOMET-ref), 66-70% (SSA-ref), with human and pseudo references alike; the sentence-level delta is +2.4 ... +3.7 chrF++, +0.04 ... +0.05 COMET; (iii) caveat: against NLLB pseudo-references the neural and surface metrics reward NLLB-likeness (R6, section 8: the NLLB-reference scores are 1-16 chrF++ above the human-reference scores); the human-reference rows agree with the pseudo-reference rows.

## 8. Reproduction

Section 5.3 above (R5 reproduced). The macro "chrF++" of the headline differs from the R5 numbers on purpose: R5 reports the corpus-level chrF++ per benchmark on FLORES+ (45 languages), here the sentence-level mean pooled over all benchmarks of a language (66 languages), and the two systems' difference is smaller because MAFAND-MT / NTREX-128 / SALT weigh in (+1.5 ... +2.2) next to FLORES+ (+3.2).

## 9. Limitations (for the paper) and what this does not show

* **No human judgement.** The "preferred in X% of sentences" numbers are *metric* preferences. Sentence-level agreement between chrF++, AfriCOMET and SSA-COMET is only 64-73% (kappa 0.17-0.28), and the agreement of each of them with human judgement on low-resource African languages is moderate (the authors' own validation covers a subset of our languages; in this repository AfriCOMET-QE is validated on 42 of them). A pairwise metric preference is a probabilistic proxy, not a rater study.
* **Metric overlap with the RL reward.** RL-final was optimised on AfriCOMET-QE (plus consensus and anchors). The reward-family metrics (AfriCOMET-QE, and AfriCOMET reference-based, which shares encoder and training data) therefore cannot count as independent evidence; chrF++ against human references and SSA-COMET-MTL (not in this reward) are. The result that they agree with the reward family (sections 6.1, 6.2) is reassuring, not a proof of the absence of reward hacking.
* **Contamination.** FLORES+ devtest and NTREX-128 are public; 95% of the FLORES+ English sentences are in our SFT text (other pairs). `no_pair` removes the rows whose English-reference *pair* was seen, `clean` keeps only rows whose English sentence was not seen (4.7% of the FLORES+ rows, about 60% of NTREX-128, about 67% of MAFAND-MT). The contamination flags refer to the SFT text; I did not re-check the RL prompts against the benchmark sentences (they are drawn from the SFT / mined pool; check `rl_prompts_v3` before claiming "no leakage" for RL). The preference is slightly lower on `clean` in chrF++ (61.6 against 63.7%) and equal in the COMET metrics (69-70%).
* **Pivot / JW domain.** The training data is religious and pivot-translated (English-to-X from a Bible / JW-style domain); the benchmarks are news (NTREX-128, MAFAND-MT), Wikipedia-like (FLORES+), children's stories (SmolSent, GlotStoryBook), and the gains are smallest where the domain is closest to the training text (MAFAND-MT +1.9, SALT +1.5). Domain shift towards user text, science or dialogue is not tested here.
* **Other.** Single reference per sentence; sentence level only (documents only through the R6 hypotheses); the neural metrics were run on a sample (all `clean` rows + at most 400 per benchmark-language), not on every sentence; SSA-COMET is truncated at 512 joint positions; one decoding (greedy); sign tests and bootstrap treat sentences as independent although FLORES+ / NTREX / SmolSent translate the *same* English sentences into every language (the language-level CIs do not need that assumption, the pooled p-values do); equal weight per language (not per sentence or per speaker).

## 10. Text for the paper

**Pairwise benchmark comparison (no human raters).** Because native-speaker ratings were not available, we compare the final RL model with the SFT model it started from sentence by sentence on eight human-reference benchmarks (161k English-to-X sentences, 66 languages; FLORES+ devtest, NTREX-128, MAFAND-MT, SmolSent, IbomMT, FLORES-MOZ, SALT, GlotStoryBook; greedy decoding, 768 tokens). For every sentence and metric the RL translation wins, ties or loses against the SFT translation (tie: |delta| < 1 chrF++ or < 0.01 COMET), and we report the share of the non-tied sentences in which RL is preferred with paired-bootstrap intervals. On the subset whose reference pair was not in the SFT text, RL is preferred in 63.7% of the non-tied sentences by chrF++ (95% CI 63.5-64.0; mean delta +3.3 chrF++, macro over languages), in 70.0% by AfriCOMET with reference, in 70.4% by SSA-COMET with reference, and in 70.2% by the majority of the three reference-based metrics; on the contamination-robust subset (English sentence not in the SFT text) the shares are 61.6, 69.0, 69.6 and 68.9%. RL is better in 65 of 66 languages on chrF++ and in all 51 AfriCOMET-covered languages on every neural metric (language-level bootstrap intervals above zero) and significantly worse in none; the share is above 50% on every metric for every training group, language family, resource tier and benchmark (lowest: 52% by chrF++ for Afrikaans and Malagasy), and 60-100% for every document length from 64 to 1,024 source tokens (sentence-by-sentence decoding; also on human-reference SmolDoc documents). The RL reward model (AfriCOMET-QE) prefers RL in 72.9% of the sentences, close to the 70-71% of the reference-based metrics, and on the 38,604 sentences where RL wins on both QE metrics the reference-based majority agrees in 89.7% and prefers SFT in 5.3%, which gives no sign of the policy exploiting the QE reward. Part of the gain is the removal of the SFT model's runaway outputs (11.4% of its sentences do not finish, against 0.3% for RL): without the 8.5% of the sentences on which either model loops, the mean chrF++ delta halves to +1.7 but RL is still preferred in 60.0% (chrF++) and 67.7-70.8% (neural) of the non-tied sentences. The comparison is a metric-based stand-in for a preference study, not a replacement for one.

**Limitations paragraph.** These preferences are metric preferences, not human judgements: chrF++ and the two learned metrics agree on the sentence winner in only 64-73% of the sentences where both call one, and their validity for several of our languages rests on small meta-evaluations. RL was optimised on AfriCOMET-QE, so AfriCOMET (with or without reference) is not independent evidence; chrF++ against human references and SSA-COMET-MTL, which was not in the reward, are, and they point the same way. FLORES+ and NTREX-128 are public benchmarks and the models have seen most FLORES+ English sentences paired with other languages; we therefore report a `no_pair` subset (reference pair unseen) and a `clean` subset (English sentence unseen), and the conclusion does not change, but we did not audit the RL prompt pool against the test sentences. The training data are pivot-translated and religious in domain, the benchmarks are news, encyclopedic and children's-story text, and the smallest gains occur where the domain is closest to the training data, so the results may not transfer to other domains. The significance tests treat sentences as independent although several benchmarks translate the same English sentences into every language.

## 11. Rerun

Files: `gemma_mt/scripts/pairwise_surface.py`, `pairwise_neural.py`, `pairwise_stats.py`, `pairwise_longdoc_prep.py`, `pairwise_robust_noloop.py`, `pairwise_make_tables.py`, `pairwise_fig_{wtl,langs,agreement}.py`; box drivers in `reports/data/pairwise_benchmark_comparison/box_scripts/` (`run_gen2.sh`, `run_chain3.sh`, `env_common.sh`, `run_ld_prep.sh`, `patch.py`). Box paths below are `/workspace/jobs/e1`; the interpreter is `/workspace/venv_gemma/bin/python`; `env_common.sh` sets `PYTHONPATH=/workspace/jobs/e1/code2c:/workspace/sft_build/pylibs_tf4:/workspace/sft_build/pylibs` (the box's `gemma_mt_2c` predates SSA-COMET support: `code2c` is a copy of it with the repo's `gemma_mt/rl/comet.py`, `joint_budget.py`, `segments.py` and the `max_num_batched_tokens` patch of `patch.py`), `HF_HOME=/workspace/sft_build/hf`, `HF_HUB_OFFLINE=1`.

```bash
# 0. on the box: private code overlay (copy the repo's comet.py, joint_budget.py, segments.py into code2c/gemma_mt/rl/, run patch.py), then the scripts
mkdir -p /workspace/jobs/e1/code2c && rsync -a --exclude=__pycache__ /workspace/sft_build/gemma_mt_2c/ /workspace/jobs/e1/code2c/ && python3 patch.py
# 1. generation (one vLLM engine at a time, 0.05 utilisation; keeps predictions.jsonl per system)
./run_gen2.sh                       # out_sft, out_rl (768 tokens, 8 benchmarks), out_sft320, out_rl320 (FLORES+)
# 2. per-sentence surface metrics -> sys_<name>.jsonl.gz
python pairwise_surface.py out_sft/predictions.jsonl sys_sft.jsonl.gz ; python pairwise_surface.py out_rl/predictions.jsonl sys_rl.jsonl.gz
python pairwise_surface.py out_sft320/predictions.jsonl sys_sft320.jsonl.gz ; python pairwise_surface.py out_rl320/predictions.jsonl sys_rl320.jsonl.gz
# 3. neural metrics (GPU, <= 4.5 GB, batch 32): all `clean` rows + 400 rows per (benchmark, language)
python pairwise_neural.py --sys sft=sys_sft.jsonl.gz --sys rl=sys_rl.jsonl.gz --out neural --cap 400 --gpu-gib 4.5 --batch 32 --metrics ssa_ref,afri_ref,afri_qe,ssa_qe
# 4. statistics (CPU, about 15 min): summary/{unit_stats,macro,agreement,agreement_by_group,flags,threshold_sensitivity,corpus_chrf_units}.csv
python pairwise_stats.py --dir /workspace/jobs/e1 --a sft --b rl --out summary --neural neural --lang-groups lang_groups.json
python pairwise_stats.py --dir in320 --a sft --b rl --out summary --neural none --lang-groups lang_groups.json --benchmarks flores_plus   # in320/sys_{sft,rl}.jsonl.gz = the 320-token files
python pairwise_robust_noloop.py                      # summary/robustness_no_loop.csv
# 5. long input (reads /workspace/jobs/a2 read-only): documents and aligned sentences
python pairwise_longdoc_prep.py --a sft --b rl2c --hyp greedy_sent --out longdoc        # then pairwise_surface.py on longdoc/{doc,seg}_{sft,rl2c}.jsonl
python pairwise_neural.py --sys sft=longdoc/sys_seg_sft.jsonl.gz --sys rl=longdoc/sys_seg_rl2c.jsonl.gz --out longdoc/neural --cap 500 --gpu-gib 4.5 --batch 32
python pairwise_stats.py --dir longdoc/doc_in --a sft --b rl --out ../doc_summary --neural none --lang-groups lang_groups.json --min-n 5
python pairwise_stats.py --dir longdoc/seg_in --a sft --b rl --out ../seg_summary --neural /workspace/jobs/e1/longdoc/neural --lang-groups lang_groups.json --min-n 5
# 6. on the laptop: copy the summary directories (CSV only) to reports/data/pairwise_benchmark_comparison/{main,max320,longdoc_docs,longdoc_segments}/, then
python gemma_mt/scripts/pairwise_make_tables.py                                        # tables.md
python gemma_mt/scripts/pairwise_fig_wtl.py; python gemma_mt/scripts/pairwise_fig_langs.py; python gemma_mt/scripts/pairwise_fig_agreement.py   # paper/figures/figpw{1,2,3}_*.{png,pdf}
```

`lang_groups.json` (language -> group, family, tier) is generated from `build_final_per_language.py`'s `group_of` / `FAMILY` and `box_scripts/groups.json`; a copy is in `reports/data/pairwise_benchmark_comparison/lang_groups.json`.

## Appendix A1. Per language (`no_pair`, all benchmarks pooled)

| iso | group | n | chrF++ | AfriCOMET ref | SSA ref | AfriC-QE | SSA QE | fin SFT/RL % | flag |
|---|---|---|---|---|---|---|---|---|---|
| dyu | blend | 1875 | +4.22 / 88* | +0.071 / 85* | +0.073 / 88* | +0.115 / 88* | +0.088 / 88* | 89/100 |  |
| kon | blend | 863 | +9.72 / 93* | +0.112 / 93* | +0.060 / 75* | +0.114 / 85* | +0.129 / 90* | 39/100 |  |
| kqn | blend | 420 | +4.33 / 72* | +0.062 / 72* | +0.058 / 70* | +0.069 / 70* | +0.068 / 68* | 89/100 |  |
| kua | blend | 520 | +2.98 / 71* | +0.043 / 73* | +0.040 / 70* | +0.042 / 71* | +0.040 / 65* | 93/98 |  |
| loz | blend | 960 | +2.33 / 59* | +0.031 / 63* | +0.043 / 64* | +0.039 / 67* | +0.047 / 64* | 96/100 |  |
| lun | blend | 420 | +2.75 / 68* | +0.036 / 69* | +0.050 / 72* | +0.046 / 70* | +0.060 / 71* | 95/100 |  |
| sag | blend | 1012 | +2.37 / 62* | +0.052 / 75* | +0.069 / 79* | +0.072 / 78* | +0.078 / 77* | 94/100 |  |
| tiv | blend | 863 | +3.53 / 71* | +0.063 / 75* | +0.057 / 74* | +0.067 / 78* | +0.061 / 72* | 89/100 |  |
| ven | blend | 2860 | +2.96 / 61* | +0.052 / 69* | +0.057 / 70* | +0.056 / 74* | +0.066 / 72* | 92/100 |  |
| ach | cons | 1363 | +2.89 / 66* | - | - | - | - | 94/100 |  |
| dag | cons | 135 | +5.29 / 78* | - | - | - | - | 90/100 |  |
| her | cons | 549 | +4.40 / 77* | - | - | - | - | 80/96 |  |
| ibb | cons | 1012 | +6.06 / 88* | - | - | - | - | 84/100 |  |
| kea | cons | 1012 | +7.41 / 88* | - | - | - | - | 78/100 |  |
| kri | cons | 863 | +4.12 / 73* | - | - | - | - | 84/100 |  |
| nbl | cons | 1184 | +1.86 / 60* | - | - | - | - | 98/100 |  |
| ndc | cons | 863 | +2.94 / 72* | - | - | - | - | 89/100 |  |
| nyn | cons | 1363 | +2.45 / 68* | - | - | - | - | 96/100 |  |
| seh | cons | 1012 | +2.91 / 71* | - | - | - | - | 94/100 |  |
| shi | cons | 1997 | +0.76 / 97* | - | - | - | - | 24/98 |  |
| tsc | cons | 101 | +4.37 / 72* | - | - | - | - | 78/91 |  |
| vmw | cons | 1012 | +4.46 / 79* | - | - | - | - | 89/100 |  |
| afr | qe | 3872 | +0.86 / 55* | +0.014 / 60* | +0.011 / 58* | +0.016 / 62* | +0.012 / 59* | 100/100 |  |
| amh | qe | 4756 | +1.57 / 61* | +0.078 / 74* | +0.055 / 71* | +0.104 / 76* | +0.065 / 71* | 89/100 |  |
| bam | qe | 1311 | +5.10 / 81* | +0.094 / 88* | +0.085 / 91* | +0.142 / 92* | +0.106 / 90* | 84/100 |  |
| bem | qe | 3872 | +4.02 / 69* | +0.072 / 76* | +0.095 / 80* | +0.076 / 74* | +0.112 / 80* | 93/100 |  |
| efi | qe | 1875 | +7.40 / 82* | +0.066 / 73* | +0.153 / 85* | +0.077 / 72* | +0.158 / 83* | 89/100 |  |
| ewe | qe | 3348 | +1.42 / 56* | +0.042 / 71* | +0.047 / 73* | +0.065 / 80* | +0.057 / 73* | 97/100 |  |
| fon | qe | 1874 | +1.87 / 69* | +0.044 / 76* | +0.065 / 73* | +0.066 / 85* | +0.073 / 73* | 98/100 |  |
| hau | qe | 3806 | +0.96 / 54* | +0.031 / 66* | +0.036 / 67* | +0.033 / 69* | +0.039 / 67* | 99/100 |  |
| ibo | qe | 3927 | +2.74 / 59* | +0.050 / 76* | +0.060 / 76* | +0.048 / 76* | +0.060 / 74* | 98/100 |  |
| kab | qe | 1874 | +2.63 / 71* | +0.050 / 75* | +0.049 / 73* | +0.047 / 74* | +0.051 / 69* | 95/100 |  |
| kam | qe | 1012 | +7.90 / 88* | +0.106 / 85* | +0.121 / 81* | +0.170 / 88* | +0.146 / 84* | 70/100 |  |
| kbp | qe | 1012 | +11.72 / 96* | +0.186 / 94* | +0.071 / 81* | +0.175 / 95* | +0.083 / 82* | 35/100 |  |
| kik | qe | 1875 | +4.79 / 70* | +0.103 / 83* | +0.105 / 81* | +0.119 / 85* | +0.122 / 81* | 85/100 |  |
| kin | qe | 4878 | +4.53 / 69* | +0.056 / 70* | +0.070 / 75* | +0.054 / 71* | +0.075 / 75* | 95/100 |  |
| kmb | qe | 1012 | +4.52 / 81* | +0.072 / 84* | +0.094 / 82* | +0.112 / 85* | +0.107 / 82* | 86/100 |  |
| lin | qe | 1875 | +1.25 / 53* | +0.064 / 73* | +0.060 / 71* | +0.074 / 76* | +0.066 / 70* | 95/100 |  |
| lua | qe | 1012 | +2.53 / 62* | +0.056 / 75* | +0.073 / 76* | +0.073 / 75* | +0.084 / 75* | 94/100 |  |
| lug | qe | 3454 | +0.41 / 51 | +0.021 / 61* | +0.023 / 61* | +0.030 / 66* | +0.030 / 63* | 98/100 |  |
| luo | qe | 3271 | +6.39 / 79* | +0.097 / 82* | +0.139 / 84* | +0.105 / 85* | +0.154 / 84* | 91/100 |  |
| mlg | qe | 3872 | +0.29 / 49 | +0.028 / 62* | +0.019 / 60* | +0.038 / 64* | +0.024 / 62* | 98/100 |  |
| mos | qe | 1875 | +4.04 / 84* | +0.062 / 78* | +0.063 / 77* | +0.093 / 85* | +0.075 / 77* | 94/100 |  |
| nso | qe | 3872 | +1.02 / 56* | +0.039 / 67* | +0.039 / 66* | +0.050 / 71* | +0.045 / 66* | 98/100 |  |
| nus | qe | 1012 | +8.40 / 89* | +0.144 / 87* | +0.093 / 87* | +0.133 / 92* | +0.109 / 87* | 50/100 |  |
| nya | qe | 4876 | +2.09 / 61* | +0.039 / 70* | +0.041 / 71* | +0.041 / 71* | +0.044 / 71* | 98/100 |  |
| orm | qe | 3790 | +3.47 / 62* | +0.099 / 77* | +0.085 / 81* | +0.120 / 81* | +0.110 / 82* | 87/100 |  |
| pcm | qe | 2237 | +0.86 / 59* | +0.017 / 73* | +0.007 / 65* | +0.024 / 75* | +0.005 / 56* | 100/100 |  |
| run | qe | 1875 | +1.86 / 59* | +0.044 / 67* | +0.052 / 70* | +0.050 / 70* | +0.059 / 70* | 96/100 |  |
| sna | qe | 4877 | +1.37 / 58* | +0.036 / 68* | +0.033 / 67* | +0.039 / 70* | +0.039 / 68* | 99/100 |  |
| som | qe | 3871 | +1.11 / 55* | +0.031 / 67* | +0.027 / 67* | +0.035 / 67* | +0.030 / 66* | 99/100 |  |
| sot | qe | 1875 | +0.29 / 51 | +0.027 / 65* | +0.027 / 64* | +0.032 / 71* | +0.030 / 66* | 99/100 | QE up, ref not all up |
| ssw | qe | 3871 | +1.47 / 57* | +0.037 / 63* | +0.038 / 65* | +0.040 / 64* | +0.042 / 65* | 97/100 |  |
| swh | qe | 6086 | +1.79 / 56* | +0.025 / 62* | +0.019 / 61* | +0.021 / 62* | +0.019 / 61* | 100/100 |  |
| tir | qe | 3872 | +1.91 / 66* | +0.121 / 80* | +0.085 / 77* | +0.135 / 83* | +0.106 / 78* | 79/100 |  |
| tsn | qe | 5340 | +1.22 / 53* | +0.025 / 61* | +0.029 / 63* | +0.030 / 67* | +0.035 / 65* | 98/100 |  |
| tso | qe | 1875 | +0.83 / 51 | +0.035 / 66* | +0.029 / 64* | +0.044 / 71* | +0.036 / 65* | 98/100 |  |
| tum | qe | 1875 | +4.40 / 74* | +0.062 / 70* | +0.076 / 73* | +0.070 / 72* | +0.089 / 74* | 95/100 |  |
| twi | qe | 3375 | +0.75 / 52 | +0.018 / 61* | +0.042 / 69* | +0.043 / 79* | +0.053 / 72* | 98/100 |  |
| umb | qe | 1012 | +3.96 / 72* | +0.068 / 82* | +0.089 / 78* | +0.098 / 81* | +0.102 / 75* | 91/100 |  |
| wol | qe | 3242 | +1.36 / 59* | +0.031 / 65* | +0.039 / 67* | +0.043 / 72* | +0.049 / 69* | 97/100 |  |
| xho | qe | 4872 | +1.01 / 54* | +0.023 / 61* | +0.022 / 60* | +0.028 / 64* | +0.025 / 61* | 99/100 |  |
| yor | qe | 4018 | +7.40 / 77* | +0.153 / 85* | +0.120 / 86* | +0.091 / 84* | +0.117 / 84* | 96/100 |  |
| zul | qe | 4566 | +0.81 / 53* | +0.022 / 60* | +0.021 / 59* | +0.024 / 61* | +0.024 / 60* | 99/100 |  |
| knc | sft-only | 1012 | +8.42 / 97* | - | - | - | - | 37/100 |  |
| nde | sft-only | 1997 | +1.21 / 64* | - | - | - | - | 99/100 |  |

Cell = delta (RL - SFT) / RL-preferred % of the non-tied sentences; `*` = sign test significant after Benjamini-Hochberg across languages (q < 0.05). chrF++ on all rows; AfriCOMET / SSA on the sampled rows (see text).

