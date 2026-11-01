# Review checklist for the short paper (2026-10-24)

Maps every claim of the abstract of `main.tex` (`sections/00_abstract.tex`) to the ledger id(s) (`paper/CLAIMS_LEDGER.md`) and to the section of the long paper (`paper/sections/*.md`, 106-page version). Section numbers are those of the long paper. No number was changed; where the long paper's parts differ, the ledger's later parts (G-K) were used. Long-paper section references appear only in `%` comments of the `.tex` files, never in the text.

## 1. Abstract claims

| # | claim in the abstract | ledger id | long paper |
|:--|:--|:--|:--|
| A1 | SFT leaves failure modes (unfinished/looping output, copying, wrong language, weak low-resource quality) | B14, I13 (unfinished/loop repair), B117-B124, B135 (copy, loops) | 6.1(i), 7.2, 7.3, 8.4 |
| A2 | Gemma 3 270M, English to 77 African languages | A1, A5 | 3.1, 3.2 |
| A3 | reward = QE ensemble taking the lower judge when they disagree (mean-or-min), consensus for 23 languages without a metric, language-aware alignment charging omissions, constraint detectors calibrated on human text | B121, B124, C2, A57, A67, A55, B135 | 4.3, 7.2, 7.3, App. C, D |
| A4 | RL needs no extra parallel data | (design fact: RL prompts are monolingual English, no target references) | 4.1, 3.6, 6.9 |
| A5 | FLORES+ macro chrF++ 28.7 to 34.1, +5.42 [5.22, 5.61], 45 of 45 languages improve | B2, B5, C4, C5 | 6.1 |
| A6 | gain largest where SFT was weakest | B20, C85, B21; per-language fig2s from `reports/data/final_model_per_language_wide.csv` | 6.1, 6.3 |
| A7 | part of the gain is repair of unfinished or looping output | B14 (320-token budget: 30.4 to 34.1), B2 (finished 90.4 to 100%), I13 | 6.1 (i), 8.4 |
| A8 | ablation: four languages, 49 RL steps, one run per arm, pre-registered | G27, I2, I9, I20, I22 | 8.4 |
| A9 | gain survives weak start: +7.36 against +5.72 | I2, I3 | 8.4 |
| A10 | RL beats continued SFT of equal GPU time: +7.04 chrF++ | I5 (RL - S1 +7.04 [6.49, 7.54]; also RL - S1P +8.37) | 8.4 |
| A11 | does not show that RL replaces SFT | I4 (W_RL - F0 = -6.25; W_RL - F_RL = -11.97), decision.json flags, I13 | 8.2, 8.4 |
| A12 | improvement measured against human references and human-judgement-correlated metrics; no human evaluation | E35, E36, K6; Limitations 6 | 6.9, 11 |
| A13 | 3.32 chrF++ behind NLLB-200 3.3B on identical sentences over the 42 languages without Amharic and Tigrinya (3.85 on all 44) (was 3.7 over the 43 NLLB languages; section 3e) | T1, T2 (earlier E4, E5 for the 43-language 3.7) | 6.8 |
| A14 | Amharic and Tigrinya (left out of the headline as an outlier with an untested cause) are 19.0 and 11.0 behind | T3, T7 (earlier B30, C17, C54, C77: Amharic -19.0, Tigrinya -11.0 vs published NLLB-3.3B; RL gains +4.4/+4.3) | 6.2, 6.8, 9.1 |
| A15 | GGUF Q8_0 runs on four CPU cores without measurable quality loss | J13 (-0.02 [-0.22, +0.16]), J5 | 6.11 |

Body numbers not in the abstract keep their ledger ids in the `%` comments' sections: pairwise 63.7/70.0/70.4% (E35, E36, K6), TranslatePsy 45.3 vs 41.2 (E8), wins 17/44 and 8/44 (E7), long-input table (F2, J27, K8, K10), final run (G1, G20, H2, H3, H10, K6), CPU table and tokenisation cost (J5, J7, J13, J22, J30), AB2 rows (I2-I15, I20).

## 2. Claims softened or changed relative to the long paper

1. **"Hardened reward" is split by model.** The long paper's reward description mixes phases. In the short paper RL-final (headline, phase 2c) is stated to have used AfriCOMET-QE alone (plus consensus, shaping, penalties, hard failures, GlotLID, chunked alignment); the QE ensemble with mean-or-min, the anchors and the long-document reward v2 are stated as final-run components only (long paper 4.3, 4.7, 6.9). Consequently the "mean or min" evidence is presented as a measured design rule, with the explicit statement that no RL A/B of ensemble against AfriCOMET alone was run (7.2).
2. **Equal-cost claim scoped.** "Beats continued SFT at equal cost" is stated as: equal GPU time (S1, 367 steps) and equal number of sentences (S1P), one untuned recipe, our own SFT windows, 49 RL steps against ~400x the training tokens; the chain's realised wall-clock exceeded its cost model (9.7 h, about 11.5 USD against a 9 USD cap, ledger I22 / G29; invoice open).
3. **"Gain survives a weak start"** is accompanied everywhere by the finished-only result (+1.13 against +2.23; DiD -1.10), the metric-dependence of "who gains more", and the weak start being a re-created annealed SFT.
4. **Contamination wording (rewritten 2026-10-26, section 3g).** The 2026-10-25 wording ("larger on clean rows for FLORES+ only") is replaced by the per-benchmark `clean` results of the audit; the ledger C139 note is corrected (Part V, V14: NTREX clean +4.09 is larger than all and `no_pair`).
5. **Dropped from the abstract/body (not softened, omitted for length):** the novelty sentence W2 ("first open 270M model ... RL against an African-language QE", long paper 2/6.8/14) and coverage claims W1/W3 beyond the plain facts 77 vs 55 languages and 33 languages absent from FLORES-200; "level with or slightly below published 8-14B LLMs" (removed 2026-10-26: the `clean` comparison with published numbers was not like for like; 5.2 now claims neither superiority nor inferiority); case studies (Urhobo silver set, consensus-only languages), the diverged-phase and teacher-failure stories (only one sentence on the divergence), the Gemini pseudo-reference and widened NLLB signal (7.8, not used in any run), AfriLongDoc details, 6.4/6.5/6.6 decoding and science-stage guardrail results.
6. **Long documents:** the long paper's "omits 13-30% of sentences" (A2W, step 250, final) is replaced by the final-run numbers only (about 12% and 27%, coverage 0.88 and 0.73, ledger K8) because Table 4 shows only RL-final and the final run; science-stage native results are described as the supervised science stage plus RL final run, not as an RL achievement. The AfriScience columns are marked in-domain.
7. **Metric coverage ("about 17-22 of 77")** was replaced by sourced numbers: 42 validated (A5/C2: 42/12/23 groups), 22 languages with both AfriCOMET and SSA-COMET with reference (E-part, K), 23 without any learned metric, human-annotated data for only a handful of pairs. The figure 17 does not occur in the ledger.
8. **Licence of the ablation controls.** The long paper says the controls use "our own commercially clean SFT data" and no non-commercial data anywhere in the ablation; since the corpus-wide licence verification is not attached (Limitations 11, TODO(owner)), the short paper says only that the controls use the SFT windows of the main corpus and no NLLB-derived text.
9. **NLLB disclosure** added to Setup and Method: NLLB-200 3.3B pseudo-reference signal for five languages (kea, kbp, nus, bam, kab; weight 0.23-0.35; ledger A60, B150) in the final run only; NLLB/TranslatePsy otherwise reference baselines only; no TranslatePsy data anywhere.
10. **"AfriCOMET independent":** nowhere; independence of metrics is stated per model (RL-final vs final run), and AfriCOMET-QE is called the reward model throughout. The wording "human preferences" is not used.
11. **CPU:** the measured model is the final-run checkpoint, not the headline RL-final (said in the text and Limitations 10); "300 MB file" and the RAM figures come from the report only (ledger J18, J30; no CSV); laptop speed is labelled an estimate.
12. **Release:** the long paper's private-repo list and the draft addendum are summarised without repo names; "open" is used only in a TODO(owner) (W1/W2 depend on release).

## 3. Content not in the ledger (new since the long paper's last ledger part), please verify

- Release state and "empty correction columns" (Section 8): from `reports/hf_release_2026-10-24_plan.md` and the long paper's Section 13 (repo list).
- Metric-coverage breakdown 22 (both) / 20 (AfriCOMET validated only) / 12 (listed) / 23 (none): same release plan file, `stats/language_metric_coverage.csv`.

## 3b. LLM-judge validation pilot (added 2026-10-25, P11)

Analysis paragraph "An independent judge, validated on human ratings (pilot)", Limitations item 1 and Appendix "LLM-judge validation pilot" (Table 7): ledger part N (N1-N23), long paper Section 7.1a (numbered 7.1a so that the references to Sections 7.2-7.8 stay valid). Numbers in the text: 0.511 [0.41, 0.60], 0.454 [0.35, 0.55], 0.536 (N8); -0.057 [-0.09, -0.02] (N13); USD 0.29 and 1.53 (N19). Open: the AfriMTE licence and the "11 languages" count are coordinator statements (N20, N21); the 2.5 Pro 404 is a coordinator statement (N4).

## 3c. Independent LLM judge, Runs 1-3 (added 2026-10-25, P13; ledger Part R, long paper Section 6.12 and Appendix J)

Analysis paragraph "An independent LLM judge" (replaces the pilot paragraph), the sentence in Section 5 "Long documents", the clause in the abstract, the intro contribution (4), Limitations items 1 and 9, and the Appendix "LLM judge: protocol, validation and results" with Tables 7 (validation, `tab_judge_pilot.tex`), 8 (Run 2, `tab_judge_run2.tex`) and 9 (Run 3, `tab_judge_run3.tex`; all three generated by `paper/scripts/make_judge.py`). Numbers in the text: pooled Spearman 0.51 (reference) / 0.49 [0.394, 0.586] (QE) / SSA-COMET-MTL 0.54 (R1-R3); FLORES+ 62.2 vs 58.2, +4.04 [3.21, 4.86], AfriCOMET +5.32, SSA-COMET +4.40 (R9-R12); whole document 25.8, 16.4% omitted, wrapper 67.0, NLLB-3.3B 85.6 (five languages; R15-R18). Space was made for it by shortening the Analysis paragraphs on reward hacking, hardening and overlap, and by dropping three sentences that only pointed to the long report or the plateau hypotheses. Numbers removed from the short paper (all still in the long paper): the v1 repetition guard flagging 47% of good sentence-by-sentence translations at 4,096 tokens (long paper 7.4), the +11.4 rise of AfriCOMET-QE in the ablation's weak arm (long paper 8.4), and the 100% padding-flag figure (kept in Appendix C as 5/14/31/81/100%). The main body ends on page 9.

## 4. Figures and tables chosen

Figures (3 + 1): Fig. 1 pipeline (new compact `fig1s_pipeline`, replaces long Fig. 1), Fig. 2 per-language gain (new compact `fig2s_gain`, replaces long Fig. 2), Fig. 3 ablation (`fig11_ablation`, unchanged), Fig. 4 CPU trade-off (`fig18_cpu_tradeoff`, unchanged). The long-context figure (fig14) is not used (too dense at column width; the table carries it). Tables (5): SFT vs RL; comparison with other systems; ablation; long-context; CPU and tokenisation cost; plus one hyper-parameter table in the appendix.

## 5. Open owner items (red TODO(owner: ...) in the PDF)

Licence verification of the SFT corpus; licence of the weights and derived data and legal review; release plan and the use of "open"; hourly GPU rate and the AB2 invoice; native-speaker review plan and funding. Also decide before submission: the venue template (swap `main.tex` only), the headline model (RL-final is used), and whether the two 2026-10-24 reports of Section 3 above should be added to the long paper and ledger.

## 4. Reward hardening: claims softened on 2026-10-25 (P12; owner decision "the body can describe the hardened reward according to our experiments and its results")

Ledger part Q (`paper/CLAIMS_LEDGER.md`), long paper Section 7.9. Source: `reports/hardening_evidence_2026-10-25.md`, data `reports/data/hardening_evidence_2026-10-25/`.

| where in the short paper | before | after | ledger |
|:--|:--|:--|:--|
| Intro, setting | "a 270M policy finds the cheap exploits of a learned reward quickly" | "the cheap exploits of a learned reward are well known"; "for the extra terms of the final run we find no demonstrated protective effect" | Q1, Q2, Q18 |
| Intro, Claim | "Reward-hardened RL is a cheap complement to SFT" | "RL with a reward surrounded by hard failure classes and calibration on human text is a cheap complement to SFT" | Q1 |
| Intro, contribution 1 | "A hardened reward and its evidence" | "A reward design and its evidence" | Q1 |
| Method | paragraph "The hardened reward"; mean-or-min "earns the lower score" | paragraph "The reward" with "Hardened ... describes the design of the final run; its measured effect is in Section 6"; "(tested on corrupted references; effect on the policy not measured)" | Q1, Q13 |
| Analysis, "Reward hacking" | "a policy under RL searches for the residual" | "could search for the residual (our runs show no copying, below)" | Q2, Q3 |
| Analysis, new paragraph "What the extra hardening did and did not do" (no table: page limit) | (none) | source copies at most 0.03% of RL-final's training completions and 0.08% of benchmark rows for both models; level on copying, entity loss, stop-word profile; names lost +1.6 pp; padding and repetition 2-4 pp lower only in weak and consensus-only languages; 2,048-token documents unfinished -40.6 pp of which -29.3 pp with the science stage; severe repetition +6.9 pp [2.8, 11.3] from step 250 to the end; min branch 56% on real copies, base rate on padded rows; no control run | Q2, Q3, Q5, Q6, Q13, Q14, Q15, Q19 |
| Limitations | (none) | new item 5 "No control for the extra reward hardening" | Q19 |
| Conclusion | "Reward-hardened RL is a cheap, additional lever"; "The reward carries most of the engineering: a pessimistic judge ensemble, ..., detectors" | "RL with a reward surrounded by hard failure classes ... "; shaping, alignment and consensus signal in the headline run; ensemble and detectors in the final run, "we could not show that these extra terms protected against anything (no control run)" | Q1, Q19 |
| Results, long documents | "they finish whole documents natively"; "With a sentence-by-sentence wrapper every system scores higher" | "they generate whole documents ... natively, in one pass"; "(a way to compare with systems that cannot take long input, not how our models are trained to work)" | long paper 6.10 |

Page budget: the main body (to the end of the Conclusion) ends at the bottom of page 9; references start on page 10. Nothing can be added to the body without a cut.

## 3d. Judge choice, magnitude, and results without the Ge'ez-script languages (added 2026-10-25, P14; ledger Part S, long paper Sections 6.8, 6.10, 6.12, 10, 11, Appendix J)

| where in the short paper | what changed | ledger |
|:--|:--|:--|
| Analysis, "An independent LLM judge" | one sentence: Gemini-2.5-Pro (the judge of the SSA-COMET paper) was unavailable to our key; Flash not worse than Gemini-3.1-Pro-preview on the same 300 segments (0.511 against 0.454) | S1-S4 |
| Results 5.2, "Honest losses: Ge'ez script" | tokenisation numbers inline (1.66 / 1.35 against 3.11 characters per token); "we have not tested what causes the gap"; without Amharic and Tigrinya: gap to NLLB 3.3B 3.85 to 3.32 chrF++, to the 600M model 1.69 to 1.13, AfriCOMET 0.624 against 0.667 and 0.651 (40 languages) | S5-S10, S16, S17 |
| Results 5.4, "Long documents" | one sentence: chrF++ against pseudo-references is much less harsh than the judge (wrapper 19.9 points behind NLLB 3.3B); the whole-document failure is not a Ge'ez effect (Yoruba 9.0, Kinyarwanda 14.6, Amharic 7.7) | S11-S15 |
| Appendix C, Protocol | the judge choice in full (2.5 Pro unavailable, 3.1 Pro against Flash: 0.454 [0.35, 0.55] against 0.511 [0.41, 0.60], difference -0.057 [-0.09, -0.02], Hausa and Somali the only per-language exceptions, one fifth of the cost; no claim that Flash beats 2.5 Pro) | S1-S4 |
| New Appendix D "Without the Ge'ez-script languages" (Tables 9 and 10) | FLORES+ macros with and without amh, tir; Run 3 judge macros over hau yor swh ibo with document-cluster intervals; cause untested, future work | S5-S16 |
| New Appendix E "CPU deployment tables" | the CPU table (Table 11) moved here from Section 5.5 to keep the main body within 9 pages; numbers unchanged and still inline | S17 |

Page budget (2026-10-25c): main body ends on page 9 with about 14 lines to spare; total 15 pages. The pilot report `reports/judge_pilot_2026-10-25.md` is now saved (Part N said it was not).


## 3e. Headline comparison without the Ge'ez-script languages, four untested candidate causes (added 2026-10-25, P15; ledger Part T, long paper Sections 6.8, 10, 11 item 16)

| where in the short paper | before | after | ledger |
|:--|:--|:--|:--|
| Abstract | "The model stays 3.7 chrF++ behind NLLB-200 3.3B on identical sentences and far behind on Ge'ez-script languages" (3.7 = paired deficit over the 43 NLLB-covered languages, A13) | "On identical sentences the model stays 3.32 chrF++ behind NLLB-200 3.3B on the 42 languages without the two Ge'ez-script languages (3.85 on all 44); Amharic and Tigrinya, left out as an outlier with an untested cause, are 19.0 and 11.0 behind" (A13 now maps to T1, T2, T3) | T1-T3, T6 |
| Intro, contribution list | "the model does not beat them, and is far behind on Ge'ez-script languages" | adds "(3.32 chrF++ behind NLLB-200 3.3B over 42 languages, 3.85 over all 44, the two Ge'ez-script languages reported alongside)" | T1, T2 |
| Results 5.2, first paragraph | starts with the 43-language numbers (34.7 against 36.4 and 38.4) | new "Headline." opening: 42 languages 3.32 / 0.043 and 1.13 / 0.028, all 44 3.85 / 0.050 and 1.69 / 0.035, Table 9 named; then the 43-language numbers with the paired deficits 1.7 and 3.7 (still there, now labelled "on these 43") | T1-T6 |
| Results 5.2, caption of Table 3 | (none) | "These sets include Amharic and Tigrinya; Table 9 gives the headline without them" | T6 |
| Results 5.2, "Honest losses: Ge'ez script" | gaps without the two languages (3.32 / 3.85, 1.13 / 1.69, AfriCOMET) in the last sentence; CPU-test sentence for NLLB-600M (37.4 / 22.6 against 19.6 / 13.2) | last sentence shortened to the AfriCOMET numbers and the 34 of 42 count (the chrF++ gaps are now in the headline); the CPU-test sentence moved to Appendix D | T4, T5 |
| Limitations item 8 | "3.7 chrF++ and 0.050 AfriCOMET behind NLLB-200 3.3B ... 19 and 11 chrF++ behind ... on Amharic and Tigrinya (published values)" | 3.32 / 0.043 over 42 languages (3.85 / 0.050 over 44); 19.0 and 11.0 on Amharic and Tigrinya; one new sentence: the cause is untested (capacity over 77 languages, tokenisation with 862 Ethiopic entries in 262,144, amount and quality of amh / tir data, Gemma's pretraining coverage), NLLB's size, data and language pairs mean the comparison cannot separate them, a matched specialist or larger model is future work | T8, T9 |
| Appendix D, text and Table 9 | table: rows = macro sets, columns = systems and two gap columns | table: columns all 44 / without Amharic, Tigrinya (42): headline / Amharic / Tigrinya; rows RL-final, final run, NLLB 3.3B, NLLB 600M and the two gaps for chrF++ and AfriCOMET; text states the exclusion rule, the single-language gaps (19.0 and 11.0; 18.3 and 8.6 to the 600M model; AfriCOMET 0.229 and 0.139), the four causes and the NLLB confound, the CPU-test sentence | T3, T4, T7, T8 |

Page budget (2026-10-25d): main body ends at the bottom of page 9 with no spare line; total 15 pages. Abstract claim A13 of section 1 above now reads "3.32 chrF++ behind NLLB-200 3.3B over 42 languages (3.85 over 44)", ledger T1, T2, long paper 6.8. Not added, by instruction: triage scores, any TranslatePsy-derived training samples.

## 3f. Skew of the gap to NLLB, Ge'ez-script languages stated separately, largest gains highlighted (added 2026-10-25, P16; ledger Part U, long paper Sections 6.1, 6.8, 8.1, 11 item 16, 14)

Owner instruction of 2026-10-25: keep the large Kabuverdianu gain, add all languages, state the Ge'ez-script languages' share of the skew separately, highlight the very large gains.

| where in the short paper | before | after | ledger |
|:--|:--|:--|:--|
| Abstract | "most where the fine-tuned model was weakest; part of the gain is repaired unfinished or looping output" / "3.32 ... (3.85 on all 44); Amharic and Tigrinya, left out as an outlier with an untested cause, are 19.0 and 11.0 behind; we do not beat NLLB" | "(Kabuverdianu +14.2, Kabiye +14.1); part of that gain ..." / "(median 2.78; 3.85 and 3.17 on all 44, because a few languages carry the gap: Kabuverdianu, Amharic and Tigrinya 30%); ... are 19.0 and 11.0 behind, and Kabuverdianu, our largest gain over SFT, is also the largest gap (21.0)" | U1, U3, U7, U11 |
| Intro (2) and (4) | "largest for the weakest languages" / "3.32 ... over 42 languages, 3.85 over all 44, the two Ge'ez-script languages reported alongside" | adds "(Kabuverdianu +14.2, Kabiye +14.1)" / adds "medians 2.78 and 3.17, because a few languages, Kabuverdianu first, carry the gap" | U1, U7 |
| Results 5.1, "Per-language gains and weak languages" | "+11.22 for the two consensus-only languages (Kabuverdianu +14.6, Makhuwa), +14.2 for Kabiye, +11.5 for Nuer" (generation-run table) | "+11.22 for the two consensus-only languages"; new sentences: largest gains over the 44 languages with an NLLB row (kea +14.2, kbp +14.1, kam +12.1, luo +11.9, nus +11.4, knc +11.0; mean +5.35, median +4.38); caveat (SFT finished 34%, 51%, 76% of Kabiye, Nuer, Kabuverdianu outputs, Kanuri has no SFT data, ablation gain on finished sentences much smaller); Kabuverdianu both the largest gain and the largest remaining gap (21.0) | U7-U11 |
| Results 5.2, Headline | "... higher than the 3.3B model in 8 of 44 (6 clear wins, 30 clear losses); the wins include Yoruba, Fon, Wolof and Igbo" | new skew sentences (median 2.78 and 3.17; kea, amh, tir 30%; Ge'ez script alone 17.7%, Kabuverdianu 12.4%; without the three mean 2.89, median 2.48); the eight languages at or above NLLB-3.3B named, labelled "chrF++" with "on AfriCOMET 4 of 42: Fon, Igbo, Wolof, Yoruba"; "point estimates" | U1-U6 |
| Limitations 8 | "3.32 ... (3.85 and 0.050 over all 44), clearly worse in 30 of 44" | adds "the median chrF++ gaps are 2.78 and 3.17, because a few languages, Kabuverdianu first, carry the mean" | U1 |
| Appendix D | Table 9 = FLORES+ without the Ge'ez-script languages; text | Table 8 (was 9) gains median rows and a Kabuverdianu column; paragraph "Skew of the gap"; new Tables 9 (`tab:skewsummary`) and 10 (`tab:skewlangs`); the NLLB comparison table (Table 3 before) is now Table 11 at the end of Appendix D | U1-U13 |

Page budget (2026-10-25e): main body (to the end of the Conclusion) ends on page 9, a few lines above the bottom of the second column, References heading follows on page 9; total 16 pages (Table 3 moved to Appendix D; its numbers stay inline in 5.2). Per-language intervals for the "at or above NLLB" count are the stored unpaired 95% intervals (non-overlap), not paired intervals: per-sentence NLLB scores are not stored. Not added, by instruction: triage scores, any TranslatePsy-derived training samples.

## 3g. Contamination audit applied (added 2026-10-26, P17; ledger Part V, long paper Section 6.9a)

Abstract: one added sentence (exposure: SFT text held 95% of the FLORES+ devtest English, five English texts, test-fitted reward calibration, inflated levels 3.8 / 2.6 chrF++, gain survives on never-seen rows, fresh evaluation planned not done). Intro: contribution (2) one clause, scope paragraph one sentence. Setup: "Evaluation sets, and how much of them our pipeline had seen" (replaces "Evaluation sets and contamination protocol"; the sentence "Published baselines did not see these sentences" is removed). Results 5.1: Table 1 restructured (`clean` block first, `no_pair` block second, IbomMT daggered; group block removed, numbers inline), paragraph rewritten. Results 5.2: the 0.7 / 3.0 `clean` gaps kept (identical sentences) with 0.70 / 3.16 over 44 languages added; the Afrique `clean` comparison removed. Results 5.4: one clause on NTREX-128 and SmolSent. Limitations item 7 rewritten, new item 12 (fresh evaluation). Conclusion: one sentence. Appendix A: Efik calibration data; Appendix H (new): Tables `tab:exposure` and `tab:exposuresize` (generated by `paper/scripts/make_exposure.py`); Appendix I (new): ablation table and figure and the long-input table, moved from the main text.

| # | claim | ledger id | long paper |
|:--|:--|:--|:--|
| A16 | the SFT text held the English side of 95.3% of FLORES+ devtest; for 7 languages the reference translation; the eight test sets are five English texts | V1, V15 | 3.6, 6.9a |
| A17 | levels inflated by 3.8 (SFT) and 2.6 (RL) chrF++ on FLORES+ (all minus clean) | V6 | 6.1, 6.9a |
| A18 | on never-seen rows RL beats SFT on all 8 benchmarks: FLORES+ +6.62 [5.83, 7.24], NTREX-128 +4.09, MAFAND-MT +3.66, SmolSent +7.06, SALT +2.31 [0.85, 3.57] | V5 | 6.1, 6.9a |
| A19 | reward calibrated on test splits: Efik thresholds on 997 of 1,012 IbomMT test sentences (RL-final and final run); final run's anchors on six test splits | V9, V12, V13, V18 | 6.9, 6.9a |
| A20 | NLLB-200 3.3B stays 3.16 chrF++ ahead on never-seen rows of identical sentences (3.84 all, 3.85 `no_pair`; 44 languages) | V7, V8 | 6.2, 6.8, 6.9a |
| A21 | a fresh evaluation is planned, not done; cost estimate 7-18 kUSD unsourced | V22 | 11 (17) |

Removed or not claimed: "6.6 behind on `clean` rows"; "turns negative against the Afrique LLMs"; "published systems did not see these sentences"; "level with or slightly below 8-14B LLMs once contamination is controlled" (V7).

