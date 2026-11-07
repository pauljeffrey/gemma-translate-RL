# PAPER PLAN: the 385M English-to-African translation paper (updated 2026-11-04, second draft)

Companion files in this folder: `CLAIMS_LEDGER.md` (every claim, number, source, independence, status), `EXPERIMENTS_LEFT.md`, `DISCLOSURES.md`, `paper.md` (second draft), `refs_new.bib` (new, verified references). Claim IDs below (R1, S3, DL2, ...) refer to the ledger. Report keys (HYG, NLLBRL, FAMILY, SPLIT, DEGEN, CFU, CTLP) are those of ledger section 0.

What changed since the first plan (2026-11-04 morning): E-23 (hygiene), E-06 (RL models against NLLB), E-12 (family/script/resource), E-08 (independent degeneration audit), E-11a (stage-A feature split), E-07 (contamination follow-ups) are done on the CPU and folded into `paper.md`; E-03 is prepared, not run. Still open: E-01, E-02, E-03, E-04, E-05, E-09, E-10, E-11, E-13, E-15, E-16, E-20.

## 1. The thesis the evidence supports

Write this, and nothing stronger:

> We grow Gemma-3-270m to 385M parameters (39 layers, hidden size 640, 385,144,448 parameters), give it 125 single-token language tags (a 125-language registry) and train it with supervised fine-tuning (Muon) on decontaminated data, and then apply GRPO (Muon; learning rate 2e-6 and KL coefficient 0.05 in stages A to mixed, 1.0e-6 and 0.1 in the science and long-document stages) with reference-free quality-estimation rewards (mean of the two shaped QE rewards if they differ by at most 1.0, else their minimum). RL raises FLORES+ chrF++ from 22.85 to 31.66 (+8.81, 95% CI [+8.59, +9.03]) and held-out AfriCOMET by +0.063, with every FLORES+ language improving. **About half of the chrF++ gain is the model learning to finish its sentences and stop looping** (49% is recovered by cutting the SFT's loops); what remains on well-formed outputs is +2.4 to +4.1 chrF++ and +0.037 AfriCOMET. The reward's own loop flag is a lower bound (an independent detector finds 1.77 times as many degenerate FLORES+ outputs), but the extra rows cost at most 0.19 chrF++, 3 to 4% of the gain. Nearly all of the gain arrives in the first RL stage (94% to B1); further general RL adds +0.54 and +0.31 on FLORES+, and RL targeted at the eight weakest West-African languages adds +0.03. On 43 languages shared with the published NLLB-200 table the final model reaches 35.54 chrF++ against 38.80 for NLLB-3.3B (92%; the SFT 68%), with caveats. Science SFT improves surface metrics on AfriScience-MT but not SSA-COMET-QE; whole-document decoding fails and sentence-by-sentence decoding is required; the first research's 270M model is as good as or better at document level. The four features added to stage A cannot be separated by the log and are described as design, not as contributions. We have no matched-compute SFT control, no decode-time loop-guard control, no second seed and no human evaluation, so the paper claims a measured, decomposed gain, not that RL is a better use of compute than more SFT.

What the paper must NOT say (ledger status `weak` or `pending`): that RL is more data/compute-efficient than SFT (R14); that the finishing gain needs RL rather than a decode-time guard (R13); that the RL gain is "real translation quality" beyond the +2.4 to +4.1 floor (R5, R6); that any reward component or any of the four stage-A features (reference-anchored reward, dynamic sampling, beam/MBR guide, std floor) helped (M10, R17, R18, no ablation); that the KL slope "tripled" when the features began (it is 2.0x once the warm-up is excluded, and was already rising); that the 385M beats the 270M first-research model (S7, R16, DL3: it does not on document level, and not on several clean sets); that NLLB data caused the regressions, or did not (K1); that the science stage improves quality (SC3); that the model is competitive with NLLB or beats it on low-resource languages in general (N2: 3.26 chrF++ below NLLB-3.3B on 43 languages, 7 wins all in the lowest tertile, one of them the suspect Yoruba value, five languages with NLLB in the reward); that family, script or resource explain the gain (R12b-R12f: exploratory, fragile); that the independent degeneration detector proves the RL is free of loops (it is blind to non-repetitive failure); any "first" claim (see the related-work hedges of the first paper).

Honest framing options for the abstract: (a) "a decomposition study of RL for small African translators" (the strongest framing: the measurement is the contribution); (b) "a 385M open translator for N African languages". Prefer (a) with (b) as the artifact.

## 2. Title options

1. *Half of the Gain Is Finishing the Sentence: Reinforcement Learning for a 385M English-to-African Translator* (current working title; **conditional on E-01**: it states a fact about RL, and E-01 may show it is a fact about unguarded greedy decoding).
2. *Small but Sabi, Revisited: What Reinforcement Learning Adds to a 385M African Translator, and What It Only Repairs* (conditional on E-03 in the same way: "only repairs" must survive the control).
3. *Decomposing the Gain of Quality-Estimation RL for Small African Machine-Translation Models* (survives both gates).
4. *Growing a Small Translator for a 125-Language African Registry: Supervised Fine-Tuning, GRPO with Quality-Estimation Rewards, and Honest Controls* (survives, but long; the "honest controls" wording requires E-01 and E-03 to exist).
5. (neutral) *A 385M Model for English-to-African Translation: Data, RL, and Where the Gains Come From*
6. (new) *Where the Gain of Quality-Estimation RL Comes From: Finishing, Looping and What Remains, in a 385M African Translator* (descriptive, survives both gates, matches the independent degeneration audit and the stage-by-stage decomposition).
7. (new, only if E-03 shows the control equals or beats SFT plus RL) *Is Reinforcement Learning Needed? A Matched-Compute Study of Quality-Estimation RL for a 385M African Translator*.

Recommendation now: 6 or 3. Use 1 only if E-01 leaves at least roughly half of the 49% standing after the guard (the threshold must be fixed before E-01 runs; proposal: "the guarded SFT recovers at most 25 percentage points of the 49%" keeps title 1, anything larger retitles). If E-03 shows the control equal or better, 3 or 7; title 1 and 2 must go.

## 3. Section outline (matches `paper.md`, second draft)

Status column: W = written, results final; P = pending experiment; S = skeleton.

| # | section | contents | ledger IDs | tables / figures | status |
|---|---|---|---|---|---|
| 0 | Abstract | draft written with gates and [PENDING] slots | R1, R4-R6, R8, R19-R20, N2, W4, SC3, DL3 | | P (gates, section 7) |
| 1 | Introduction | problem; what we find; what we do not claim; contributions (decomposition + independent degeneration audit, decontaminated pipeline and its follow-ups, negative results, artifacts) | all | Fig 1 teaser (decomposition bar) | S |
| 2 | Related work | MT for African languages; RL for MT; metrics and hacking; growth; documents | | | W |
| 3 | Languages, tokenizer, growth | 125 tags, 74 scored languages, verified architecture (39 layers, hidden 640, 4 heads, 1 KV head, 385,144,448 parameters) | M1, M2, M3, M6 | T1 language/group table, Fig 2 growth diagram | W |
| 4 | SFT data and recipe | NLLB-derived shares with both denominators, Muon recipe, per-shard learning rates, validation loss | M4, M5, L1-L4, CT1-CT3 | T2 data composition, Fig 3 validation curve (data exists, plot missing) | W |
| 5 | RL method | GRPO as used (Muon, 2e-6), reward rule (mean if abs(r1 - r2) <= 1.0, else min), the four features from step 41, stage-A learning-rate decay, prompts, monitors | M7-M10, R9, R10, W6 | Fig 4 reward diagram, T3 stage table | W |
| 6 | Evaluation protocol | benchmarks and counts, metric independence table (now with the independent detector and NLLB-218 LID), contamination audit and follow-ups | M11, CT1-CT13 | T4 metric independence | W |
| 7.1 | SFT vs A0, vs old 270M | S1-S10 | S1-S10 | T5 benchmark deltas, Fig 5 | W |
| 7.2 | RL vs SFT | headline, finishing/looping, decomposition, where RL does not help; **7.2.1 independent degeneration audit** (lower bound 1.77x, cost <= 0.19 chrF++, English-output rise) | R1-R7, R19-R24 | T6, Fig 6 waterfall, T-degen (reward vs independent flag), T-english, Fig 7 per-language gain | W for R1-R7, R19-R24; P for E-01, E-03, E-04 |
| 7.3 | RL chain | stage deltas, dev curve | R8, R9 | Fig 8 | W |
| 7.4 | Weak languages, weak-group run | W1-W6 | W1-W6 | T9, T10 | W |
| 7.5 | Decoding | beam/MBR on SFT; RL models, guard [P] | D1-D6 | T8, Fig 9 | W (E-01, E-02, E-09, E-09b) |
| 7.6 | Comparison with other systems | **sentence-paired RL vs NLLB-600M / NLLB-3.3B / TranslatePsy-0.8B on FLORES+ (FRC1-FRC3), NTREX/MAFAND language-level rows (FRC4), 270M rows (FRC5)**; published 43-language block (92% vs 68%), tertiles, 7 wins, caveats; AfriqueLLM indicative | N1-N6, N2, N2b, N3b | T11 NLLB table, Fig 10 gap by tertile with RL | W (E-06b optional) |
| 7.7 | Family, script, resource (exploratory) | Glottolog 5.3, gain by family/script, token-budget null that flips when Afro-Asiatic is dropped, regression R2 | R12b-R12f | T12 (appendix candidate), Fig 11 | W (labelled exploratory) |
| 7.8 | Science SFT and RL | alpha sweep and SSA-COMET-QE null; science RL [P]; SSA-COMET-QE for all stages [P] | SC1-SC5 | T13 alpha sweep, Fig 12 | partial (E-16, E-05) |
| 7.9 | Documents | whole vs chunked, 270M vs 385M; long-document RL [P] | DL1-DL5 | T14, Fig 13 | partial (E-15) |
| 8 | Analysis: stage-A log | KL slope 2.0x not 3x; no speed-up of reward/finished/loop; features inseparable; first stage explanation (marked as our reading) | R9, R10, R17, R18, M10 | Fig 14 (`fig_feature_split.png`) | W |
| 9 | Stage chain summary | steps, prompts, bins, cost, delta per stage | R15, W4, SC1 | T3 | W (rows for science RL and long-document RL pending) |
| 10 | NLLB data and the regression | K1-K4, plus the family regression's NLLB-share ambiguity | K1-K4 | Fig 15 scatter (exists) | W (inconclusive) |
| 11 | Negative results | weak-group RL, science SFT null, whole-document, post-editor | W4, SC3, DL1, PE1 | | W |
| 12 | Discussion | thesis in 8 numbered claims, "our reading, not tested", what is not claimed, what would change it | | | W (gated) |
| 13 | Limitations | 14 items | | | W |
| 14 | Ethics, licences, disclosure | NLLB-derived shares, IbomMT, MT560, RL prompts, TranslatePsy (control only), benchmark exposure, process, release status, intended use | DISCLOSURES | | W (VERIFY items) |
| 15 | Reproducibility | code, seeds, hardware, Hub branches, data, what cannot be reproduced | | | W (VERIFY items) |
| 16 | Conclusion | | | | S (gated) |
| A | Appendices | per-language tables, hyper-parameters, reward formulas, contamination, family regressions, detector definitions, feature-split slopes, control design | | | S |

## 4. Figure and table list

Figures (all from stored data; none needs a GPU):
1. Pipeline overview: A0 -> growth -> SFT -> RL A -> B1 -> mixed -> [weak] -> science SFT (alpha) -> science RL -> long-document RL, with the measured gain at each arrow (R8, W4, SC3).
2. Decomposition of the RL gain (waterfall): total +7.96 chrF++ (93 cells); recovered by cutting SFT loops 3.90 (49%); remaining 4.06; on loop-free rows 2.41/2.61; AfriCOMET 0.0631 vs 0.0366/0.0370 (R5). Source EVID Table 1. Add a marker for the independent-detector bound (<= 0.19 chrF++ more).
3. Per-language RL minus SFT (FLORES+, 54 languages, sorted), all rows vs paired loop-free rows, with the 8 added languages marked and family colours (R3, R7, R12d). EVID Table 6, FAMILY.
4. Stage chain: SFT -> A -> B1 -> mixed on FLORES+, NTREX, MAFAND (R8) and the stage-A dev curve with KL-guard cuts (R9, STA).
5. Decoding study: greedy / beam / MBR per language (D1, D3); caveat markers for the 256-cap (D2).
6. SFT 385M vs A0 and vs old 270M on unseen sentences (S1, S7), with the language count printed.
7. **NLLB gap by tertile, all five of our systems and NLLB-3.3B (N2)**: high 38.43/46.82 vs 51.05 etc.; mark the five languages with an NLLB pseudo-reference reward and the suspect yor value.
8. Science alpha sweep: SSA-COMET-QE (flat) against chrF (rising) (SC3). SCI alpha table.
9. Whole vs chunked document decoding, three models (DL1-DL3).
10. NLLB share vs regression scatter (K1), existing figure `reports/data/nllb_regression_link_2026-11-03/fig_nllb_share_vs_change.png`.
11. Reward-independence map (diagram of which metric is in/out of the reward) (metric table).
12. **Reward loop flag vs independent detector** by system and benchmark (R19), with the human-reference floor; **English-output share in aeb/ary/arz/zgh by system** (R23).
13. **Stage-A KL and reward slopes before/after step 41** with the 15-step window slopes (`reports/data/stage_A_feature_split_2026-11-04/fig_feature_split.png`) (R17, R18).
14. **Gain by Glottolog family and script** (appendix; `fig_gain_sft_tokens_family.png`, `fig_gain_by_glottolog_branch.png`), labelled exploratory (R12d-R12f).
Tables: T1 languages/groups; T2 data composition and licences; T3 stage chain; T4 metric independence; T5 SFT benchmark deltas; T6 headline RL vs SFT; T7 failure classes; T8 decoding; T9 weak and added languages; T10 weak-group run; T11 external comparison (NLLB 43 languages, tertiles, same sentences, Afrique); T12 family/script; T13 alpha sweep; T14 documents; T15 degeneration reward vs independent; appendix per-language tables.

## 5. Related work (what to cite)

Keys already in `paper/refs.bib` (use as they are; entries were checked in the first paper's audit; the arXiv ids of `abdulmumin2026afriscience` (2605.29741) and `li2025ssacomet` (2506.04557) match the brief, not re-fetched):

| topic | keys |
|---|---|
| AfriScience-MT (arXiv:2605.29741) | `abdulmumin2026afriscience` |
| SSA-COMET (arXiv:2506.04557, EMNLP 2025) | `li2025ssacomet` |
| AfriCOMET / AfriMTE | `wang2024africomet` |
| NLLB-200, FLORES-101/200, FLORES+ | `nllb2022nllb`, `nllb2024nature`, `nllb200_card`, `goyal2022flores101`, `oldi2025floresplus`, `dale2025oldi` |
| African MT data and benchmarks | `adelani2022mafand` (MAFAND-MT), `nekoto2020participatory`, `reid2021afromt`, `akera2022salt`, `caswell2025smol` (SMOL, GATITOS), `alabi2025afridocmt`, `ojo2025afrobench`, `kudugunta2023madlad`, `agic2019jw300`, `akerman2023ebible` |
| African LLM and small-model baselines | `yu2026afriquellm`, `akera2025sunflower`, `gritta2026translatepsy`, `tonja2024inkubalm`, `moslem2026afrinllb`, `finkelstein2026translategemma` |
| GRPO and analyses | `shao2024deepseekmath`, `guo2025deepseekr1`, `liu2025understanding` (Dr. GRPO), `yu2025dapo` (dynamic sampling), `schulman2020approximating` (k3 KL) |
| RL for MT with QE / reference rewards | `feng2025mtr1zero`, `he2025r1t1`, `wang2025deeptrans`, `wang2025extrans`, `cheng2025seedx`, `zheng2025hunyuanmt`, `zheng2025hymt15`, `attia2026roundtrip`, `garciaestrada2026reffree`, `hu2026rlunseen`, `liu2026walar`, `han2026milmmtrl` |
| MBR and quality-aware decoding | `freitag2022mbr`, `eikema2020map`, `fernandes2022qualityaware`, `shen2016mrt`, `xu2024cpo` |
| consensus / label-free RL | `zuo2025ttrl`, `wang2023selfconsistency` |
| reward hacking, overoptimisation, ensembles | `gao2023scaling`, `karwowski2023goodhart`, `krakovna2020specification`, `coste2024ensembles`, `singhal2024long` |
| metrics and statistics | `rei2020comet`, `rei2022cometkiwi`, `guerreiro2024xcomet`, `juraska2024metricx24`, `kocmi2024metricsmaze`, `popovic2017chrfpp` (chrF++), `post2018call` (sacreBLEU), `koehn2004significance`, `kargaran2023glotlid` |
| documents and long output | `wang2023document`, `wu2024adapting`, `varis2021sequence`, `liu2024lost`, `gale1993program`, `thompson2019vecalign`, `holtzman2020curious`, `welleck2020unlikelihood`, `song2025formatrl`, `wu2025longwriterzero`, `song2025fastcurl`, `hammoud2025trainlong` |
| small models, multilinguality, forgetting | `gemmateam2025gemma3`, `google2025gemma270m`, `gemma_terms`, `conneau2020unsupervised`, `chang2024multilinguality`, `petrov2023tokenizer`, `chung2023unimax`, `wortsman2022robust` (WiSE-FT interpolation, used for alpha), `wortsman2022soups`, `ilharco2023editing`, `rolnick2019experience`, `biderman2024lora`, `shenfeld2025razor`, `mukherjee2025subnetworks`, `chu2025sft`, `yue2025doesrl`, `shumailov2023curse` |

New references the 385M paper needs and that are NOT in `paper/refs.bib`. Each was fetched and checked on 2026-11-04 (title, authors, year); BibTeX is in `refs_new.bib`:

| need | reference | verified |
|---|---|---|
| NTREX-128 (benchmark) | Federmann, Kocmi, Xin. NTREX-128 - News Test References for MT Evaluation of 128 Languages. SUMEval 2022, pp. 21-24 (aclanthology 2022.sumeval-1.4) | yes |
| chrF (original metric, plain chrF is the AfriScience metric) | Popovic. chrF: character n-gram F-score for automatic MT evaluation. WMT 2015 (aclanthology W15-3049) | yes |
| depth growth with identity-initialised blocks | Wu et al. LLaMA Pro: Progressive LLaMA with Block Expansion. arXiv:2401.02415 (2024); Kim et al. SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling. arXiv:2312.15166 (NAACL 2024 Industry) | yes |
| Muon optimiser (SFT and RL optimiser, verified in the run configs) | Liu et al. Muon is Scalable for LLM Training. arXiv:2502.16982 (2025) | yes |
| Aya aligned instruction data | Singh et al. Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning. arXiv:2402.06619 (2024) | yes (the "Aya collection" used here is the aligned part; confirm the collection's own citation) |

Still to find and verify before use (not added): **Glottolog 5.3** (Hammarstrom, Forkel, Haspelmath, Bank; cite the v5.3 release, commit 072ca0d, CC BY 4.0; the CLDF metadata labels its citation 5.2.1, so cite the tag; add the bib entry in `refs_new.bib` once the DOI/Zenodo record is fetched); fastText language identification as used by the NLLB-218 LID model (`facebook/fasttext-language-identification`, cc-by-nc-4.0; evaluation only); IbomMT, GlotStoryBook and FLORES-MOZ dataset citations (the first paper cites them as dataset cards without bib keys); a proper citation for the Muon reference implementation if the blog post is preferred; Hugging Face model cards of `masakhane/africomet-stl-1.1`, `masakhane/africomet-qe-stl-1.1`, `McGill-NLP/ssa-comet-mtl`, `McGill-NLP/ssa-comet-qe` (cite the papers; record the licence of each checkpoint, unrecorded for `ssa-comet-*`); a citation for beam-guided/off-policy guide members in GRPO (candidates: LUFFY and related off-policy-guidance work; not verified, do not cite until fetched); a reference for reference-anchored (reference-plus-QE) rewards beyond `feng2025mtr1zero`; any 2026 paper that compares matched-compute SFT and RL for MT (search before submission: the first paper's novelty search dates from 2026-10-22/23); a citation for interrupted-time-series analysis only if the Section 8 analysis is kept in the main text.

## 6. Build setup

The first paper builds from `paper/sections/*.md` -> `paper/paper.md` (assembled by hand/`build_paper.py`) -> HTML/PDF, and `build_latex.py` -> Overleaf project, with `[@key]` citations resolved against a `refs.bib` in the same tree and `TODO(...)` markers highlighted. To reuse without touching `paper/`: copy `paper/scripts/build_paper.py`, `build_latex.py`, `paper.css` into `paper_385m/scripts/` (both scripts take `PAPER` = the parent of `scripts/`), copy `paper/refs.bib` to `paper_385m/refs.bib` and append `refs_new.bib`. Directives used in `paper.md`: `[@key]`, `{{table:NAME|caption}}`, `{{fref:}}`/`{{tref:}}`, `TODO(owner: ...)`. The draft uses plain markdown tables and `<!-- src: ... -->` comments so that it renders without the scripts; `<!-- GATE E-xx -->` comments mark the blocks listed in section 8.

## 7. Gates before writing the abstract

**No gate is open.** The abstract of `paper.md` states only supported numbers. Closed: E-01 (decode-time guard: 49% of the chrF++ gain, 29% of the AfriCOMET gain; RL + guard minus SFT + guard +4.37 [+4.28, +4.46] chrF++), **E-03 (matched-compute continued-SFT control: 29.97 chrF++ against 34.55 for the RL mixed best over 93 cells, +4.58 [+4.45, +4.70]; registered verdict "inconclusive" because d on the 19 TranslatePsy languages is -0.03 [-0.13, +0.09]; one seed, one learning rate, only the 6,000-step point evaluated)**, E-15 and E-16, E-05, E-09 and E-09b, E-13, E-20. E-02 (beam/MBR at 768 tokens), E-04 (two extra seeds) and E-11 (RL ablations) are in (ledger D7, R25-R28; M10 corrected: the features do not cause the KL rise; the mixed step is not separable from seed noise). E-14 (RL-only science arm: the registered rule says the science SFT is needed; chrF -1.44 [-1.61, -1.25] against the science RL final) is in (ledger SC7, SC8). Nothing is running or open except the owner answers below. Not run: E-10 human evaluation (needs native-speaker annotators), P8 data-box scans, E-17. Open owner answers: X14 (release status and licence), X17 (34 dev-train-half languages), unspecified-licence sources, `SetFit/bbc-news` licence. Nothing from `paper/` is reused: the first paper's FLORES+ numbers were contaminated.

Not a gate but a risk: the independent degeneration detector and the family analysis were run after the results were known; the text says so (Sections 7.2.1, 7.7, 13 item 6).

## 8. Text blocks that must change if E-01 or E-03 changes the story

Block identifiers are the `<!-- GATE ... -->` comments in `paper.md` plus the section names below.

**Status 2026-11-04 (late): E-01 is in** (guard: 49% of the chrF++ gain, 29% of the AfriCOMET gain; RL + guard minus SFT + guard +4.37 chrF++ and +0.0479 AfriCOMET), so the blocks of 8.1 have been rewritten in `paper.md` (abstract, title, Sections 7.2, 12, 13); the table is kept as the record of what changed. 8.2 has also been applied: the E-03 verdict is "inconclusive" (a third case in addition to "RL better", "comparable" and "control better" of the registered rule: the registered conditions on the overall endpoint are met and the one on the 19 TranslatePsy languages is not), and `paper.md` Sections 7.2.2, 12 and 13 and the abstract state it exactly as registered.

### 8.1 If E-01 closes most of the finishing gap (the guarded SFT recovers most of the 49% or more)

The claim "about half of the gain is finishing" would become "about half of the gain is a decoding artefact that a guard also removes", and RL's own contribution is the remainder.

| block (paper.md) | change |
|---|---|
| Title (`GATE E-01 E-03`) | drop "Half of the Gain Is Finishing the Sentence" as a statement about RL; use option 3 or 6; or reword as "... Is a Decoding Problem". |
| Abstract (`GATE ... whole abstract`), sentence on "49% recovered by cutting loops" | add the guarded-SFT row: "a repetition guard at decode time recovers X% of the gain without any RL"; state RL's gain over the guarded SFT with its interval; the thesis sentence becomes "RL's contribution beyond decoding is Y". |
| Section 1 bullets "What we find" and "What we do not claim" and Contributions (1) (`GATE E-01 E-03`) | the first contribution (decomposition) gains the decoding control; delete "RL had to teach"; keep the independent degeneration audit. |
| Section 7.2 "One seed ... no decode-time guard control" sentence (`GATE E-01, GATE E-03`) | replace the [PENDING: E-01] clause by the guarded-SFT result; add the guarded RL row for symmetry (E-01 also runs the guard on the RL model). |
| Section 7.2 Finishing/looping table | add rows "SFT + guard" and "RL + guard" (finished, loop by both detectors, hypothesis-to-reference length). |
| Section 7.2 Decomposition table (`GATE E-01`) and the paragraph below it | add a row "RL minus guarded SFT"; rewrite the sentence "49% of the gain is recovered by cutting loops" to give both the post-hoc cut and the genuine decode-time guard; state which is the headline. The sentence on the independent detector ("not redone with the independent detector") must be completed with the guard parameters chosen on the dev half. |
| Section 7.2.1 | add the independent flag for the guarded outputs; the bound "at most 0.19 chrF++" is recomputed if the guard changes which rows are flagged. |
| Section 7.5 Decoding (`GATE E-01`) | replace the "[PENDING: ... loop guard ...]" sentence; the estimate "+5.0 over a 768-token greedy" is replaced by the measured beam-at-768 value (E-02). |
| Section 7.6 caveat (2) (beam vs greedy) | add the guarded-greedy and beam rows against NLLB if E-02 runs on the 43 languages. |
| Section 8 "Why the first RL stage does nearly everything" (`GATE E-01 E-03`) | the explanation (QE reward teaches the model to stop) moves from "our reading" to "tested" or is withdrawn. |
| Section 12.1 intro and claim 2 (`GATE E-01`), "Our reading, not tested" paragraph, 12.2 bullet 2, 12.3 | claim 2 becomes "about half of the gain is finishing and looping, and a decode-time guard recovers P of it"; the thesis sentence changes accordingly; delete the alternative-account paragraph or turn it into a result. |
| Section 13 limitation 2 (`GATE E-01 E-03`) | remove "no decode-time loop-guard control"; if the guard hurts translation, state that. |
| Section 16 Conclusion (`GATE E-01 E-03 E-04`) | restate with the new split. |
| PAPER_PLAN section 1 (thesis, the sentence "About half ... finish") and section 2 (title options 1, 2) | same changes as above. |
| Figure 2 (waterfall) | add the guard bar between "cut at loop" and "RL". |

If the guard closes only a small share, the same blocks lose the [PENDING: E-01] clauses and gain the sentence "a decode-time guard recovers only P% of the finishing gain, so most of it needed training"; the title can stay.

### 8.2 If E-03 beats SFT-plus-RL (the matched-compute control equals or beats the RL)

By the rule of CTLP section 7: "control better" if the upper 95% bound of (RL - control) chrF++ over the 93 cells < -1.0; "comparable" if both the chrF++ and AfriCOMET intervals are inside +-1.0 / contain 0. In either case "RL is a better use of compute" is not supportable.

| block (paper.md) | change |
|---|---|
| Title | drop options 1 and 2; use 3, 6 or 7. |
| Abstract | replace "single-seed, without a matched-compute SFT control" by the control result; the thesis becomes "quality-estimation RL repairs what more SFT also repairs, at the cost of X" (comparable) or "continued SFT on Y data beats RL" (control better). Remove any sentence implying RL's necessity. |
| Section 1 bullets and Contributions (`GATE E-01 E-03`) | the central question changes from "what does RL add" to "RL versus the same GPU time of SFT"; add the control as a contribution; the sentence "RL is a better use of compute ... (no control, E-03)" in "What we do not claim" is replaced by the result. |
| Section 4 and Section 14.3 | the control's data (TranslatePsy, 50% of the mix, 19 languages) is now evidence; add its decontamination counts (0.67% of the sample rows removed), the release status ("not released") and the two caveats (neural metrics favour the control through the TranslatePsy filters; TranslatePsy is NLLB-3.3B output, so the control partly distils NLLB). If the control wins, say that it may win by distilling NLLB, not by being "more of our SFT". |
| Section 6.3 metric table | the control's AfriCOMET/SSA-COMET gains are in the family of its data filter: add a row; chrF/chrF++ remain the evidence. |
| Section 7.2 sentence "no matched-compute SFT control" (`GATE E-01, GATE E-03`) | replace with the control table: SFT, stage A and C@3,100, mixed best and C@4,800, C@6,000, C2 (end-of-turn x5), with the primary endpoint, the 19 TranslatePsy languages and the other languages separately, and the seven near-failed languages; share of the SFT-to-RL gap closed (C - SFT) / (RL - SFT) for chrF++, AfriCOMET, finished and loop rates. |
| Section 7.3 | the stage-by-stage statement gains the control at the stage-A budget. |
| Section 9 (`GATE E-03`) | add control rows; the cost column is the quantity matched. |
| Section 12.1 intro and claims 1, 2 (`GATE E-01 E-03`), "Our reading" paragraph | claim 1 says "SFT plus RL beats SFT" only in the sense of "the same SFT without further training"; add the control sentence; the alternative account is now the finding. |
| Section 12.2 bullet 1 (`GATE E-03`) | delete or replace with what the control shows. |
| Section 12.3 | delete the E-03 sentence. |
| Section 13 limitation 2 | remove "no matched-compute SFT control"; add: one control run, one LR (3e-6 unless the probe decides), 19 languages of new data, NLLB-derived data; and for "RL better" outcome the caveat that the margin is 1.0 chrF++. |
| Section 14.3 (`GATE E-03`) | status "prepared, not run" becomes "run"; the licence paragraph stays. |
| Section 16 and PAPER_PLAN sections 1, 2, 7 | restate. |

If E-03 shows RL better (lower bound > +1.0), the same blocks lose "[PENDING: E-03]" and the Discussion may state "RL is a better use of 4.4 to 8.7 A100-hours than continued SFT on the data we tried (one seed, one LR, one control mix)", never more.

### 8.3 Blocks that change under other pending results (for completeness)

**Applied 2026-11-05:** E-04, E-02, E-11 are in; `paper.md` Sections 5, 7.3, 7.5, 7.6 (caveat 2), 8, 12 and 13, the abstract and the ledger (D7, R25-R28, M10) carry them. The list below is kept as the record.

* E-04 (seeds): Section 7.3 sentence "of the size of evaluation noise ... no seed estimate", Section 12.1 claim 4, Section 13 item 1, abstract "single-seed" clause.
* E-05 and E-16: Section 6.3 row for SSA-COMET-QE, Section 7.8 "Science RL" paragraph, Section 9 science row, Section 12.1 claim 5.
* E-15 and P9: Section 7.9 "Long-document RL" paragraph, Section 9, Section 14.2 (LibreTexts [VERIFY]), Section 12.1 claim 6.
* E-02, E-09: Section 7.5, Section 7.6 caveat (2), Section 13 item 9.
* E-10 (human evaluation): Section 12.2 bullet 3, Section 13 item 3, abstract.
* E-11 (ablations): Section 8, Section 12.2 bullet 4, Section 13 item 5; the features may be described as contributions only if an arm shows a benefit.
