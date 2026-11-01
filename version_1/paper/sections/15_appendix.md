<div style="page-break-before: always"></div>

## Appendix A Per-language results

{{table:t_master|Every language on its primary benchmark (`no\_pair` rows; first available in the order FLORES+, NTREX, MAFAND, SmolSent, IbomMT, FLORES-MOZ, SALT, GlotStoryBook), SFT-final to RL-final, greedy, 768 tokens. Different benchmarks have different difficulty, so values are not comparable across rows. The $\Delta$ interval is a 95% paired bootstrap over sentences (500 resamples), asterisk if it excludes 0. Tachelhit (`shi`, NTREX) is a script mismatch (Arabic-script reference, Latin output), not a quality measure; Kanuri and North Ndebele have no SFT data. Source: `reports/final_model_per_language_2026-10-21.md` section 3.}}

{{table:t_nobench|Languages without any human-reference benchmark (19 of 85; 13 RL-trained). Their quality was judged only by the reward models.}}

{{table:t_published_lang|Per-language chrF++ of RL-final (all FLORES+ rows) next to the published English-to-X values (FLORES-200 devtest; NA = not published). 44 of the 77 RL languages have a published NLLB number (43 of them have a FLORES+ row of ours and appear here); 33 have no published number from any system we compared.}}

## Appendix B Hyper-parameters

SFT: Section 3.5. RL: {{tref:t_hparams}}; common to all phases: temperature 1.0, top-$p$ 1.0, top-$k$ off, one update per batch, token-level loss normalisation, gradient clip 1.0, fp32 master weights with bf16 autocast, vLLM rollouts with `max_new_tokens` 384 / 1,280 / 2,304 / 3,584 for b64 / b256 / b512 / b1024, and `max_model_len` 8,192 with 3,584 new tokens for the mixture, in phases 1, 2, 2b and 2c up to the b2048 restart; from the b2048 restart and in the final run `max_model_len` 10,240 and 7,168 new tokens for b2048 and the mixture. Phase 1: learning rate 1e-6, 8 completions per prompt, 42 prompts per step in the mixture, 300 steps. Phase 2: 2e-6, group 8, 497 planned steps (the log we have ends at step 378). Phase 2b: 3e-6, group 16 with a greedy member, $\beta=0.05$ with $k_3$, advantages without division by the group standard deviation, 1,623 planned steps (stopped at 839; the model of step 488 was kept). Draw weights: 77 language files with weights 1.0 for most languages, 1.5-4.0 for the weakest (Esan 4.0; Edo, Ghomala, Okpe 3.5; Kongo, Tswa, Dagbani, Nuer, Tigrinya 3.0; 16 other weak languages 2.5; Hausa, Luganda, Sepedi, Zulu, Ndau, Ibibio 2.0; ten no-benchmark or script-artefact languages 1.5).

{{table:t_hparams|Reinforcement-learning hyper-parameters of phase 2c and the final run (`paper/data/rl2c_config.json`, `final2_config.json`).}}

## Appendix C Reward: formulas and calibration facts

**Shaping.** With per-language anchors $(t_\text{fail}, t_\text{mid}, t_\text{great})$ the shaped reward is the piecewise-linear interpolation through $(0,-2), (t_\text{fail}, 0), (t_\text{mid}, +1), (t_\text{great}, +2), (\max(1, t_\text{great}+0.001), +2.25)$, constant outside. Anchors: $t_\text{fail}=\max(\text{floor}, P_{90}$ of the score of mismatched human references$)$ with floor 0 for the ensemble ($0.46$ in the first version, which is wrong for languages whose human references score below it); $t_\text{mid}=\max(P_{50}$ of human references, $t_\text{fail}+0.05)$; $t_\text{great}=\max(P_{90}$ of human references, $t_\text{mid}+0.05)$. Calibration data: FLORES+ dev (997 sentences per language; Pidgin: MAFAND dev; Efik: the IbomMT test set, 997 of its 1,012 sentences, its only human-reference set, which Section 6.1 also reports as a result, Section 6.9a). In the first, QE-only calibration the $+0.05$ spacing floor was active for $t_\text{mid}$ in 21 of 41 calibrated languages, so in those languages the band between failure line and human median is only 0.05 QE units wide.

**Advantage weight.** $w_\ell=\mathrm{clip}((h-0.5)/0.4,0,1)$ with $h$ the mean of the accuracies of ranking the human reference above a mismatched and above a half-omitted version of the same source ("same-source" reliability); weights 1.0 for 31 languages and 0.79-0.96 for umb, kam, pcm, kmb, kbp, nus, mos, fon, kab, lua in the first calibration; hand-set 0.3-0.5 for blend and consensus languages without validation; Efik 0.5 after consensus validation (within-prompt Spearman of consensus with chrF++ 0.380).

**Ensemble.** $r=\tfrac12(r_1+r_2)$ if $|r_1-r_2|\le 1.0$ else $\min(r_1,r_2)$; the min branch takes 14.0% of policy chunks (12.3% in QE languages, 19.3% in blend languages; Pidgin 69% because SSA rates Pidgin far above AfriCOMET). Joint-encoder budget: the translation and source together must fit in 508 tokens, so longer pairs are split at sentence boundaries and the piece scores averaged by source characters.

**Consensus.** For completion $i$ of a group with at least three valid members, $c_i=\frac{1}{|V|-1}\sum_{j\in V, j\ne i}\mathrm{chrF{+}{+}}(y_i,y_j)$ (sentence-level, word order 2, NFC), $r_i=-2+4.25\,c_i/100$; archive threshold for consensus-only languages 58.8 chrF++.

**Hard failure classes** (reward $-2$, never scored; first match wins): no end token; text after the end token; turn marker before it; special tokens (optional); missing or wrong language tag; empty; copy of source (Jaccard $\ge 0.9$); degenerate (word repeated $>4$ times, $<30\%$ distinct words of $\ge 8$, zlib ratio $<0.18$); character length ratio outside $[0.2, 5]$; GlotLID mismatch (75 of 77 languages; minimum text length 15-120 characters per language); stuffed output (an optional gate of the anchors: 0.9 in the first launch of the final run, which produced 26-63% failures in six languages, and off, 0.0, in the current run).

**Omission and extra-sentence detection** ({{tref:t_omission}}). The language-aware Gale-Church cost with constant omission and extra beads raises the share of dropped sentences found from 0% to 56%.

{{table:t_omission|Omission detection on FLORES+ dev documents of 3-12 sentences (macro over 8 languages, 3 tilings; `reports/rl_reward_explained_2026-10-19_v3.md` section 4.5).}}

**Long-document reward v2** (documents of at least 150 words). Severity $s\in[0,1]$ is the largest of five ramps (word 4-gram repeat share, words in repeated 8-grams, duplicated-sentence characters, repeated blocks of at least two sentences, periodic word cycles), each measured as excess over the English source and ramped from $t_0$ = 1.15 times the 99.5th percentile of human documents of the length bucket (floor 0.02-0.10) to $t_1=0.5$ (0.6 for the 4-gram share); the reward is $r-s(r+2)$. Aggregation: $R=\bar r-\alpha\max(0,(\bar r-\mathrm{CVaR}_{25})-g_0)$ with $g_0$ the human median of $\bar r-\mathrm{CVaR}_{25}$ per bucket (0.76 / 0.76 / 0.89 / 0.95 / 0.96 / 0.99), $\alpha=1$; plans of fewer than 8 chunks use the mean. Coverage: $u$ = failed share of the source weight; the reward moves towards $-2$ by $\kappa u$ with $\kappa=0.5$.

## Appendix D Detector calibrations

Every threshold is a quantile of human text. *Repetition*: language-specific quantiles with cross-validated false-positive rate at most 4.5% (63 of 63 languages with references pass). *Boilerplate*: five-grams occurring in at least three other prompts' outputs in a buffer of 600 outputs (4 per prompt). *Entities (v2)*: deficit $=1-$ recall, channel weight $(1-d)$ with $d$ the share of judged human chunks that lost every entity (names $\times 0.5$), channel off if $d>0.6$ or if it does not separate a human from another source's translation; numbers enabled in 67 of 74 calibrated languages, names in 54 of 76. *Substring copy*: fraction of unprotected source words covered by word runs of length at least 3 and character runs of at least 20 found verbatim in the output; threshold $\max(0.08, P_{99.5}$ of human fraction$)$ (v3: $P_{99.8}$, floor 0.30); chunk reward $\min(r,-2\cdot\text{fraction})$; enabled in 73 of 77 languages. *Function-word profile*: 80 words of high document frequency in both training text and human references; enabled in 55 of 77 languages (0.3 weight); off for agglutinative languages and Ge'ez. *Padding*: v2 bigram-order gain plus type/token ratio with length-bucket thresholds; v3 segmental type/token ratio only. *Lexicon*: English-to-X lexicons mined from our pivot data, enabled for 10 languages (sag kam fon kab kmb kea umb vmw mos kbp) with weight at most 0.25 and a threshold at the human median (a graded, small term). *Diacritics*: penalty for 13 languages with active density threshold (e.g. yor 0.182 from the marked sentences, fon 0.146, kbp 0.189, nus 0.166), guards only for hau, kin, mlg, tum, umb.

## Appendix E Prompts and formats

SFT / RL prompt (Gemma chat template):

```
<bos><start_of_turn>user
Translate the text to Yoruba:  Good morning<end_of_turn>
<start_of_turn>model
yor E kaaro<end_of_turn>
```

The text after `Translate the text to {language}:` is separated by two spaces and contains no newline; the model answer starts with the three-letter tag. For documents the English windows are whole-sentence windows of up to 2,048 AfroXLM-R tokens and the answer is the whole translation in one pass; the chunked decoder calls the same prompt once per sentence.

## Appendix F Hardware, cost and what was run

| item | value | source |
|:--|:--|:--|
| GPU for RL and late SFT | RTX PRO 6000 Blackwell, 96 GB, rented on vast.ai at about 1.19 USD/h | project lead's statement; `reports/final_run_integration_2026-10-21.md` |
| GPU for early SFT | RTX 5070 12 GB (22 s per step at micro-batch 2 x 16 with gradient checkpointing); rental 6.4 s per step | `reports/gemma_sft_training_plan_2026-10-14.md`, memory notes |
| SFT | 27,638 steps x 131,072 tokens = 3.62 B training tokens | `sft_unimax.yaml`, training log |
| RL phase 2c | 40.5 h wall-clock including restarts, median 107 s per step; 900,384 sampled completions | `paper/data/rl2c_metrics.jsonl` |
| Final run | 401 steps: 26.2 h of step time, median 218 s per step; 26.6 h of wall-clock for the final lineage and 28.4 h between the first and the last logged step (the difference is the 112 minutes lost around the resume, with the discarded steps 251-267); last process (steps 251-401) 45,456 s = 12.6 h; one planned restart after step 150 (2026-10-22 21:22 UTC), two OOM kills (2026-10-23 04:22 and 04:49 UTC) with resume from the step-250 checkpoint; wind-down steps 351-401; about 34 USD for the 28.4 h at 1.19 USD/h (rate TODO(owner: confirm the invoice)) | `paper/data/final2_metrics.jsonl`, `/workspace/jobs/r7/runs/final2/summary.json`, `launch_FINAL2.log` |
| Final checkpoint | private Hub repository `<HF_ORG_LEGACY>/gemma-science-translate`: branches `latest` and `final2/step401` and tag `final2-step401` (commit 8bb1a029, step 401); `main` of the repository holds the SFT-stage models; no mid-run checkpoint was kept; evaluated on FLORES+, NTREX-128, MAFAND-MT, SmolSent and AfriScience-MT dev and test on 2026-10-23 (Section 6.7) | Hub listing of 2026-10-23 (read-only) |
| Ablation AB2 (finished; chain released 2026-10-23 17:35 UTC, result files of 2026-10-24 03:17 UTC) | planned: 4.60 h = 5.48 USD (central) to 6.51 h = 7.75 USD (high) at 1.19 USD/h, cap 9.00 USD including a 15% margin. Realised (derived, not an invoice): 9.7 h wall-clock from 17:35 to 03:17 UTC = about 11.5 USD at 1.19 USD/h if all of it was billed, above the plan and the cap; the excess includes about 2 h of GPU idle after a configuration error and one re-run of the evaluations of F0, W0 and W_S1 (deterministic greedy decoding, so the re-runs do not change the results). S1 ran 367 steps (about 48 M tokens), fewer than the 432 (central) or 487 (high) of the cost model because the rule sets N from the measured arm-F job. TODO(owner: confirm the invoice and what was billed) | `gemma_mt/scripts/ab2_cost.py` (planned; `paper/data/ab2/ab2_cost_output.txt`); wall-clock from the project lead's message (no file); steps of S1 from `paper/data/ablation_results.json` |
| All-models comparison | GPU jobs of about 6 GB on the shared box (NLLB re-scoring, TranslatePsy-0.8B generation, AfriCOMET and SSA-COMET scoring); GPU-hours and cost not recorded | `reports/model_comparison_2026-10-23.md` |
| RL phases 1, 2, 2b | wall-clock not recorded in the copied logs TODO(owner: add GPU-hours from the vast.ai invoice) | -- |
| Science SFT stage | 293-514 steps per variant (about 17 minutes for the 293-step first run) | `reports/final_run_integration_2026-10-21.md` section 3 |

Estimated RL budget of the path to the final general model: 1.76 M sampled completions, 0.86 B generated tokens ({{tref:t_rl_budget}}).

<div style="page-break-before: always"></div>

## Appendix G Comparison with other systems: audit, per-language tables and coverage survey

*Metric audit.* {{tref:t_model_comparison_comet_audit}} shows which COMET-family metrics are published for which model family (sources, versions and URLs per model are in `reports/data/published_comet_audit.csv`, 122 rows). The published TranslatePsy aggregates (19 languages, direction not stated) and AfroBench pooled chrF cannot be put next to per-language English-to-X numbers and are in {{tref:t_model_comparison_published_aggregates}}.

{{table:t_model_comparison_comet_audit|Which COMET-family numbers are published, per model family (E2 audit, 2026-10-22/23).}}

{{table:t_model_comparison_published_aggregates|Published aggregates that cannot be paired with our per-language numbers: TranslatePsy-AfriSLM paper (19 languages, both directions averaged by inference) and AfroBench pooled chrF.}}

*The final run.* The computed rows and tables of this appendix include the final checkpoint of the final run ("SabiYarn-270M final run (final2 step 401)") since 2026-10-23: the E2 aggregation was re-run with that checkpoint added, which reproduced every earlier row exactly (886 per-language, 718 macro and 628 paired rows compared; the new files are `*_with_final2.csv`, the old ones are kept) and added paired differences to SFT-final, A2W and the step-250 checkpoint beside those to RL-final (`paper/scripts/box_e2_model_comparison_aggregate_with_final2.py`).

*Main tables with intervals and paired differences.* {{tref:t_cmp_main_full}} is the unabridged version of {{tref:t_cmp_main}} (with intervals, all 43 baseline models, published and computed rows); {{tref:t_model_comparison_paired}} gives the paired differences to RL-final used in Section 6.8; {{tref:t_model_comparison_validation}} the check of our metric stack against the published NLLB-200-3.3B chrF++.

{{table:t_cmp_main_full|Table M1 of the comparison: chrF++ and AfriCOMET, English to X, FLORES+ devtest (computed rows: `no_pair` rows, 95% sentence-bootstrap intervals; published rows: the papers' whole-devtest numbers, not paired).}}

{{table:t_model_comparison_paired|Table M3: paired difference to RL-final (system minus RL-final, macro over the set, same sentences, 95% intervals). Positive values favour the system in the first column.}}

{{table:t_model_comparison_validation|Table M4: our chrF++ for NLLB-200-3.3B on the whole devtest minus the published value, per language.}}

*Per-language results.* {{tref:t_cmp_perlang_chrf}} and {{tref:t_cmp_perlang_africomet}} give the per-language values behind the macros (45 FLORES+ languages; the full versions with all systems are `t_model_comparison_perlang_*.md`).

{{table:t_cmp_perlang_chrf|Per-language chrF++ on FLORES+ devtest `no_pair` rows (n sentences); NLLB-3.3B published = the NLLB release value on the whole FLORES-200 devtest (n/a: no published value). n/a: the system does not cover the language.}}

{{table:t_cmp_perlang_africomet|Per-language AfriCOMET (reference-based africomet-stl-1.1, 0-1) on the same sentences. n/a: the language is not on the AfriCOMET card (Kabuverdianu, Kanuri, Makhuwa) or the system does not translate it.}}

*Language-coverage survey.* {{tref:t_coverage_claim}} lists the African-language coverage of 35 systems (the full text of the evidence is in `reports/data/language_coverage_claim_check.md`). "Verified" marks whether the number was read from the system's own paper or model card; the MADLAD-400 and TranslateGemma counts are manual tallies of language tables and are unverified.

{{table:t_coverage_claim|African-language coverage of other systems (survey of 2026-10-23).}}


<div style="page-break-before: always"></div>

## Appendix H Pairwise comparison of SFT-final and RL-final: per language

{{figure:figpw2_languages|Share of non-tied sentences in which RL-final is preferred over SFT-final, per language and metric (`no_pair`, all benchmarks pooled; asterisk: Benjamini-Hochberg adjusted sign test q < 0.05, which treats sentences as independent; the language-level intervals in the table below do not). Languages of the QE-validated group in blue, blend in orange; the two columns on the right are the reward family (AfriCOMET-QE, the reward model, and SSA-COMET without reference). The dot marks Southern Sotho, the one language without a significant chrF++ difference. Source: `gemma_mt/scripts/pairwise_fig_langs.py`.|55}}

{{table:t_pw_perlang|Per language (`no_pair` rows, all benchmarks pooled): macro chrF++ difference (RL minus SFT; asterisk: the 95% bootstrap interval is above zero), share of non-tied sentences in which RL is preferred, the same for the reference-based learned metrics (51 languages), and the finished rate of the two models. Source: `reports/data/pairwise_benchmark_comparison/main/unit_stats.csv`.}}

*The final checkpoint against RL-final, per language.* {{tref:t_final2_perlang}} gives the per-language difference in corpus chrF++ (final2, step 401, minus RL-final; `no_pair` rows) on the four benchmarks for which the final checkpoint was scored on the same sentences (Section 6.7). The pairwise comparison with the learned reference-based metrics was added afterwards for NTREX-128, MAFAND-MT and SmolSent pooled ({{tref:t_final2_pw_perlang}}; Section 6.7): the final checkpoint is clearly lower than RL-final in AfriCOMET with reference in 17 of 40 languages and clearly higher in one (Luganda), the largest falls being Amharic (-0.0154), Yoruba (-0.0099), Oromo (-0.0083), Igbo (-0.0068) and Northern Sotho (-0.0064), in each of which sentence chrF++ is flat or higher (Amharic +0.55, Yoruba +0.55, Oromo +0.23, Northern Sotho +0.14; Igbo -0.06 with an interval that includes zero).

{{table:t_final2_perlang|Per language: chrF++ difference of the final checkpoint (final2, step 401) minus RL-final (corpus level, `no_pair` rows; an asterisk marks a paired 95% bootstrap interval that excludes zero; "--": the language is not in the benchmark) and the FLORES+ values of both. Source: `paper/data/final2_eval/per_language_bench.csv`.}}

{{table:t_final2_pw_perlang|Per language (NTREX-128, MAFAND-MT and SmolSent pooled, `no_pair` rows): sentence-level mean difference of the final checkpoint (final2, step 401) minus RL-final in chrF++, AfriCOMET with reference and SSA-COMET with reference (an asterisk marks a paired 95% sentence-bootstrap interval that excludes zero; "--": no SSA-COMET with reference for that language, or fewer than 20 sentences) and the share of sentences in which final2 wins / loses under AfriCOMET with reference (ties: difference below 0.01). Source: `paper/data/final2_eval/g1_pairwise_final2_per_language.csv`.}}

<div style="page-break-before: always"></div>

## Appendix I Long-context benchmark: systems, and an optional Gemini pseudo-reference (future work)

{{tref:t_longctx_models}} lists the systems of Section 6.10 with their documented context limits and the languages each was run on; the per-language tables behind {{tref:t_longctx_human_chrf}} are `paper/tables/t_longctx_perlang_*.md`, the paired wrapper-minus-whole differences `t_longctx_sentence_minus_whole.md` and all cells `reports/data/longctx/cells.csv`.

{{table:t_longctx_models|Systems compared in the long-context benchmark: size, documented maximum context and limits for documents, whether the benchmark documents fit, and the languages each system was run on (compute-driven reductions).}}

*Gemini Batch API as a second pseudo-reference: plan only, nothing run.* (The LLM-judge runs of Section 6.12 are a different use of the Gemini API, scoring existing outputs with a judge prompt; they do not produce a pseudo-reference and are not part of this plan, whose own cost estimates and absence of any call stand.) The NLLB-3.3B pseudo-reference of the core documents is biased towards NLLB-like systems and is not above our model in several languages (Section 6.10). LC1 therefore prepared, but did **not** run, a plan to translate the benchmark with the Gemini Batch API as an additional *evaluation reference* (`reports/long_context_gemini_plan_2026-10-23.md`, scripts `paper/scripts/gemini_*.py` with 11 passing tests; the API key would come from an environment variable; no key was available and no API call, paid or free, was made for this plan; the judge validation pilot of Section 7.1a later used the Gemini API, with a key from an environment variable, to score 300 segments of public human-annotated MT data; it translated nothing of this benchmark). The benchmark is small: 2,280 items (290 documents times languages), about 2.3-2.7 M English source tokens. With the prices that LC1 read from the provider's pricing page on 2026-10-23 (we did not verify them independently; Batch = 50% of the standard price; output prices include thinking tokens) and an *assumed* output of 2.75 times the English tokens (not measured; Amharic may exceed 3.5), whole-document requests with thinking switched off cost about 9.6 USD for the full set and 1.2 USD for a reduced set of 270 items with Gemini 2.5 Flash (Flash-Lite: 1.6 and 0.2 USD; Gemini 2.5 Pro with default thinking about 61 and 7.4 USD); sentence-by-sentence requests cost about 1% more with eight sentences per request and 16% more with one (three and eighteen times more with default thinking); a 50-sentence pilot over 14 languages costs about 0.06 USD (up to 1.2-7.3 USD for Pro with thinking). Money is not the constraint; quality is: Gemini's quality in low-resource African languages is uneven (the only evidence in the repository is 10 FLORES sentences per language for Gemini 2.5 Flash against 1,012 for NLLB-3.3B), and an earlier Gemini-generated Urhobo set was of doubtful quality. By that thin evidence Gemini is ahead of NLLB-3.3B in Hausa, Swahili, Igbo, Malagasy, Zulu, Xhosa, Kinyarwanda, Somali, Twi, Luganda and Northern Sotho, marginal in Yoruba and Wolof, and unmeasured in Amharic; it would add most where NLLB is not above our model (Luganda, Northern Sotho). The recommended order is the pilot against human references first, then the reduced set, then the full set only if the pilot passes. Gemini outputs would be used as evaluation references only (the provider's terms forbid using the service to develop competing models), never as training data. The cost tables are generated by `paper/scripts/gemini_batch_cost.py` into `paper/tables/t_longctx_gemini_cost.md`.


<div style="page-break-before: always"></div>

## Appendix J LLM-judge prompts, run details and cost

**Judge model.** The SSA-COMET paper [@li2025ssacomet] judges with Gemini-2.5-Pro. For our API key that model returned "no longer available to new users" (Batch API and standard API), so we did not run it. We compared Gemini-3.1-Pro-preview (Google's named successor) with Gemini-3.8-Flash on the same 300 SSA-MTE and AfriMTE segments (`reports/judge_pilot_2026-10-25.md`, `reports/data/judge_pilot_2026-10-25/`): pooled Spearman with the human z-score 0.511 [0.41, 0.60] for Flash and 0.454 [0.35, 0.55] for 3.1 Pro, paired difference Pro minus Flash -0.057 [-0.09, -0.02], per-language intervals of the difference including zero except for Hausa and Somali, at about one fifth of the cost (USD 0.29 and 1.53 for the 300 segments). We therefore used Gemini-3.8-Flash; we do not claim that it is better than 2.5 Pro. All other settings are listed below.

The independent judge of Section 6.12 uses the two prompts of Li et al. [@li2025ssacomet] without change: Figure 8 (reference-based) and Figure 9 (reference-free, "QE"), five-shot, score only. The text below is the exact request text built by `gemma_mt/scripts/judge_j2_common.py` (`reports/data/judge_j2_2026-10-25/prompt_templates_and_examples.txt`); it is verbatim from the publication except that "[67]Most" is written "[67] Most". Figure 9 is Figure 8 with the `Reference: ...` field removed from the five demonstrations and from the test line. **Header (both prompts):**

> Assess the translation adequacy on a continuous scale [0 ~ 100] using the quality levels described below:  
> [0] Nonsense/No meaning preserved: Nearly all information is lost between the translation and source.  
> [34] Some meaning preserved: The translation preserves some of the meaning of the source but misses significant parts.  
> [67] Most meaning preserved: The translation retains most of the meaning of the source.  
> [100] Perfect meaning: The meaning of the translation is completely consistent with the source.  
> Instruction: Please assess the given machine translation based on the source sentence. Note that you should only output the final score  
> Given examples:

**Demonstrations (five, numbered "Example 1:" to "Example 5:", one per adequacy interval; reference-free prompt, Swahili, example 1 of the Run 1 request):**

> Example 1:  
> Source: After the accident occurred, Gibson was transported to a hospital but died shortly afterwards. Translation: Baada ya ajali ilitokea, Gibson alichaguliwa hospitalini lakini alikufa baada ya muda mfupi. Score: 91

**The reference-based prompt has the same demonstrations with the human reference inserted between the translation and the score** (verbatim format, Hausa, example 3 of the Run 2 request; stored in `prompt_example_run2_flores_hau_rl.txt`): "Example 3:  Source: *source sentence* Translation: *system translation* Reference: *human reference* Score: 60".

**Closing line and test line (both prompts; the request ends with "Score: " and a trailing space, and the judge answers with a bare number):**

> Based on the examples given, generate the output in exactly the same format, give the score and do not give any commentary response.  
> Source: *source sentence or aligned source block* Translation: *system output* [Reference: *human reference*, reference-based prompt only] Score:

The demonstrations are those of the pilot (`reports/data/judge_pilot_2026-10-25/demo_ids.json`): the SSA-MTE train split for Amharic, Hausa, Kinyarwanda and Yoruba, other rows of the same pair of AfriMTE-WMT2024 (scores min-max scaled to 0-100) for Swahili, Igbo, Somali and Xhosa; the demonstrations of languages that were not validated were not used. One complete request per prompt type and language is stored in `reports/data/judge_j2_2026-10-25/prompt_example_*.txt` (Run 1: Yoruba and Swahili, QE; Run 2: Hausa, reference-based, RL-final output; Run 3: final checkpoint whole-document, final checkpoint wrapper and NLLB-200 3.3B, QE), and every request actually sent is in `prompts_run{1,2,3}_*.jsonl.gz`.

**Settings and failures.** gemini-3.8-flash as served on 2026-10-25; temperature 0; thinking at the model default; Batch API only; one request per segment or unit and one run; output cap 1,500 tokens (Run 1) and 1,000 tokens (Runs 2 and 3), thinking included. Of 6,190 responses none failed to parse and none ended for a reason other than STOP; the largest thinking budget was 912 tokens. {{tref:t_judge_j2_cost}} gives the requests and the cost of each batch job (input USD 0.375 and output, thinking included, USD 1.875 per million tokens at the Batch price on that date). Runs 2 and 3 were each split in two jobs by language so that the worst-case sum stayed below the budget cap of USD 8; the split does not change any request.

{{table:t_judge_j2_cost|Requests, token usage and cost of the judge runs (actual usage metadata of the Batch API; the worst case is the guard's bound at submission). Source: `reports/data/judge_j2_2026-10-25/cost_summary.csv`, `reports/data/judge_pilot_2026-10-25/cost_and_failures.csv`.}}
