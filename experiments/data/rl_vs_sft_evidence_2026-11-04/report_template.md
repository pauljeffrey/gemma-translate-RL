# Does RL beat SFT? Evidence from the stored evaluations (2026-11-04)

Question (owner): how much of the RL gain over the SFT is real translation improvement, and how much is the RL model finishing its sentences and not looping? Everything here was computed on the CPUs of the A100 box (`nice 19`, no GPU) from predictions and AfriCOMET scores that were already stored; no model was run, the science RL was not touched. Scripts and raw output: `reports/data/rl_vs_sft_evidence_2026-11-04/` (`ev_stats.py` per-sentence statistics, `ev_analyse.py` paired bootstrap, `ev_extra.py` retention and detector floor, `build_evidence_tables.py`, `evidence.json`, `extra.json`, `tables.md`). Section 5 (science) is filled in when the science RL has ended and its on-box evaluation is done; it is marked below.

## 0. Short answer

* **The RL model is much better than the SFT on every benchmark, and the difference is far outside sampling noise.** FLORES+ devtest chrF++ (54 languages, language macro) 22.85 -> 31.66, **+8.81 [+8.59, +9.03]**; NTREX (24) +7.91 [+7.69, +8.12]; MAFAND (15) +5.01 [+4.76, +5.23]; all 93 benchmark-language cells **+7.96 [+7.82, +8.12]**. Plain chrF: +8.59 [+8.43, +8.76]. The independent neural check (held-out `masakhane/africomet-stl-1.1`, reference-based, 76 cells): 0.5653 -> 0.6284, **+0.0631 [+0.0620, +0.0643]**. All 54 FLORES+ languages and all 24 NTREX languages improve on all three metrics with confidence intervals above zero; MAFAND: 14 of 15 (pcm: -0.25 [-0.69, +0.30], not different from zero).
* **A large part of this is finishing and looping, not translation quality.** On FLORES+ the SFT finishes 82.9 % of the sentences and the loop detector flags 20.9 % of its outputs (RL: 99.5 % and 4.8 %; the same detector flags 0.01 % of the human references, so the RL's 4.8 % are real repetition; 95 % of the RL's flagged FLORES+ rows come from 7 near-failed languages: taq, knc, zgh, dik, aeb, arz, ary). SFT outputs are 3.0 times as long as the reference (RL 1.14).
* **Real translation improvement exists and is smaller: +2.4 to +4.1 chrF++ and +0.037 AfriCOMET.** Cutting the SFT's looping outputs at the loop onset (no translation change at all) already gives the SFT +3.9 chrF++ (26.59 -> 30.49), i.e. **49 % of the chrF++ gain is recovered by cutting loops**; the RL then still wins by **+4.06 [+3.97, +4.15]**. On the rows where the SFT output is itself finished and loop-free the RL still wins by **+2.41 [+2.33, +2.49] chrF++** and **+0.0366 [+0.0357, +0.0374] AfriCOMET** (59 % of the neural gain), and by +2.61 [+2.54, +2.70] / +0.0370 when the RL's own failures are removed too. In 50 of 54 FLORES+ languages the RL is significantly better on those clean rows.
* **Where it does not help beyond finishing:** the Arabic dialects aeb, ary (and arz when the SFT is cut at its loop) and Tamazight zgh (FLORES+, chrF++ 2 to 9 for both models, the RL still loops in 11 to 46 % of them), and on MAFAND hau, ibo, pcm (-0.3 to -0.5 chrF++ on clean rows). The 8 languages added after the A0 set show **no demonstrated translation gain beyond finishing** (-0.13 [-0.54, +0.27]); their whole gain is finishing.
* **Nearly all of the gain comes from stage A** (SFT -> stage A +7.96 of the +8.50 to the B1 best on FLORES+; B1 adds +0.54, mixed +0.31, both ~0 on MAFAND).
* **What this cannot establish** (section 6): whether an equally long continued SFT, or a decoding fix, would give most of the same gain (no control was run), seed-to-seed variation (one run), and independence from the reward for everything that is a QE or chrF++-agreement metric. Estimates for the controls are in section 5.

## 1. What was compared and how

Systems (same prompts, greedy, 768 new tokens, vLLM, the 385M evaluation harness of `reports/eval_post_B1_385m_2026-11-03.md`): `sft` = the 385M SFT final (`g15_385m_sft_final`); `stage A` = final of stage A (400 steps); `B1 best` (stage B1, 135 steps run, best checkpoint by dev); `mixed best` = the final RL model (stage mixed, 60 steps, best checkpoint by dev). "RL" below is the mixed best unless stated; table 3 shows the three stages. Benchmarks: FLORES+ devtest (54 languages x 1,012 rows), NTREX (24 x ~2,000), MAFAND (15 x ~1,500), 122,585 rows in all, all English -> African.

Rows removed everywhere: 214 rows whose English side is flagged by the 385M SFT contamination audit (`reports/contamination_audit_385m_2026-11-03.md`; exact / sentence / near hit): FLORES+ 0, NTREX 120, MAFAND 94 (one- and two-word items). Nothing else was filtered. (The earlier statement in this session that the 385M audit was missing was wrong; it exists and its flags are used here.)

Scoring: sacrebleu chrF++ (word order 2) and plain chrF (word order 0) recomputed from per-sentence n-gram statistics (NFC); the recomputed SFT / stage A / B1 FLORES+ macros (22.85, 30.81, 31.35) equal the numbers already reported, which checks the pipeline. AfriCOMET = `masakhane/africomet-stl-1.1` per-sentence scores from the stored caches (joined by the cache key; coverage 83.6 % of rows, i.e. the 76 benchmark-language cells of AfriCOMET-supported languages). Macro = mean over benchmark-language cells. Intervals: 95 % percentile intervals of a paired bootstrap, 1,000 resamples, Poisson(1) weights per source sentence shared by the two systems and (for FLORES+ and NTREX, which are parallel) by all languages, so the macro intervals include the cross-language correlation of the same English sentence. They quantify test-sentence sampling only, not training-run variation (section 6).

### Which metrics are in the reward family

| metric | in the RL reward? | use here |
|---|---|---|
| `masakhane/africomet-qe-stl-1.1` (reference-free) | yes (QE ensemble member) | **excluded** (not computed) |
| `McGill-NLP/ssa-comet-mtl` (and SSA-COMET-QE, same model family and training data) | yes (QE ensemble member; in the science RL also reference-based) | **excluded** from the RL-vs-SFT evidence; SSA-COMET-QE appears only as a hacking monitor in the science reports |
| consensus utility (chrF++ agreement among the 16 samples of a prompt, no reference) | yes, all stages | not a reference score; but it optimises chrF++-style agreement |
| NLLB-3.3B pseudo-reference chrF++ (reward only, CC-BY-NC) | yes, 5 languages (kea kbp nus bam kab), calibrated weight | chrF++ of those 5 languages is not independent of the reward |
| human-reference chrF++ term (weight 0.35) | yes, B1 and mixed only, 15 % of prompts: 7 languages (amh efi hau lug nso yor zul), source AfriScience-MT **train** pairs | the benchmarks here are different data, so chrF / chrF++ on FLORES+, NTREX, MAFAND is not a reward input, but the language-level chrF++ gain on those 7 languages can be partly trained |
| guards and penalties (language ID, copy, repetition, length, unfinished = hard failure -2) | yes | the finished / loop rates are optimised by construction; the loop flag here is the same detector as the reward guard (`is_repetitive`, `is_self_repetitive`), see section 6 |
| `masakhane/africomet-stl-1.1` (reference-based) | **no**: held out since patch 3 and used only as a monitor and in the hacking alarm; it never entered the reward | **the independent neural check**. Caveat: it shares architecture and training data (AfriMTE) with the QE reward member, so it is independent in weights and in use of the reference, not in family |
| chrF / chrF++ against the human references of FLORES+, NTREX, MAFAND | no | the main lexical evidence; checkpoint choice used a separate dev set (720 sentences in 16 languages in stage A; the dev macro of B1 and mixed), never these test sets |

## 2. Gain of RL over SFT per language and macro (task 1)

<<T1>>

Per language (FLORES+ devtest, sorted by gain; "paired rows" = both models finished and neither loops; the last column cuts the SFT at its loop and leaves the RL as is):

<<T6>>

NTREX and MAFAND per language:

<<T7a>>

<<T7b>>

## 3. Finishing and looping against translation improvement (task 2)

<<T2>>

Four ways to remove the finishing effect, all with the same bootstrap (table 1):

| row set | chrF++ macro, 93 cells | share of the +7.96 | AfriCOMET macro, 76 cells | share of +0.0631 |
|---|---|---:|---|---:|
| all clean rows | +7.96 [+7.82, +8.12] | 100 % | +0.0631 [+0.0620, +0.0643] | 100 % |
| SFT cut at its first loop onset, RL as is | +4.06 [+3.97, +4.15] | 51 % | not scorable (cut text has no cached score) | |
| both cut at their first loop | +4.18 [+4.10, +4.28] | 53 % | | |
| rows where the SFT is finished and loop-free (RL failures kept) | +2.41 [+2.33, +2.49] | 30 % | +0.0366 [+0.0357, +0.0374] | 58 % |
| rows where both are finished and loop-free | +2.61 [+2.54, +2.70] | 33 % | +0.0370 [+0.0362, +0.0379] | 59 % |
| rows where the SFT failed (unfinished or loop; 21 % of FLORES+ rows) | +24.57 [+24.26, +24.90] | | +0.3342 [+0.3295, +0.3391] | |

Reading: on the SFT's failed rows its chrF++ is 5.95 and its AfriCOMET 0.24 (the SFT keeps generating after the translation), the RL gets 30.5 and 0.58, so the failed rows carry a large part of the macro gain. The finishing effect is therefore **about half of the chrF++ gain (49 % recovered by loop-cutting alone; 67 to 70 % attributable to rows where either model failed) and about 41 % of the AfriCOMET gain**. What is left, **+2.4 to +4.1 chrF++ and +0.037 AfriCOMET**, is translation quality on outputs that are well-formed in both systems; it is significant in 50 of 54 languages on clean rows (table 5 and the per-language table).

How far each estimate can be trusted: the "both finished" subset removes the RL's own failures (favours RL) and the SFT's (favours SFT) and keeps only the sentences the SFT can handle, which are easier (rows kept: FLORES+ 78 %, NTREX 88 %, MAFAND 96 %; for knc 4 %, taq 7 %, zgh 7 %, dik 13 %, so the subset of those languages is a handful of easy sentences); the "SFT finished" subset keeps RL failures and is the fairer lower bound; cutting the SFT at the loop is the most generous treatment of the SFT but leaves unfinished-without-loop rows and any other trailing text untouched. The three estimates agree on the sign and rough size, and none of them makes the gain vanish.

## 4. Where the gain is: language groups and the number of languages with RL worse (task 3)

<<T4>>

* **Weak list vs other** (the x2.5 RL list, 20 FLORES+ languages): all rows +9.29 vs +8.53 chrF++, AfriCOMET +0.087 vs +0.063, clean rows +3.58 vs +2.30. The weak list gains more on FLORES+, but on NTREX the six weak-list languages present gain less (+6.17 vs +8.50), so the advantage is not uniform across benchmarks.
* **Resource level** (no external labels exist for the 54 languages; terciles of the SFT's own FLORES+ chrF++ are the proxy): weakest third +9.13, middle third +10.90, strongest third +6.40 chrF++ (all rows). The strongest languages gain least (afr +1.0, the largest gain in a language near the SFT ceiling is small) and the weakest third's clean-row gain is only +1.95 because its clean subset is tiny (survivor bias) while its AfriCOMET gain is the largest (+0.12 all rows).
* **Family:** Nilo-Saharan +11.6, Bantu +9.8, Afroasiatic +9.6 (dragged down by the Arabic dialects, see below), Mande +8.4, Atlantic-Congo non-Bantu +7.1, Indo-European / Austronesian (afr, mlg) +3.3; pcm +20.3 on FLORES+ (but -0.25 on MAFAND).
* **Languages added after A0** (8 on FLORES+: aeb ary arz cjk dik mfe taq zgh): +6.28 all rows, clean rows -0.13 [-0.54, +0.27]. Inside the group, cjk (+2.13), mfe (+3.91), arz (+0.67), dik (+1.8), taq (+3.3) are positive on clean rows, aeb (-2.46), ary (-3.70), zgh (-6.69) negative; the Arabic dialects and Tamazight sit at chrF++ 2 to 9 in both models.
* **Number of languages where the RL is worse than the SFT** (table 5): on all rows, FLORES+ 0 / 54, NTREX 0 / 24, MAFAND 1 / 15 (pcm, not significant) for chrF++ and chrF, and 0 on AfriCOMET everywhere. On clean rows FLORES+ chrF++ 3 / 54 (aeb, ary, zgh, all significant), MAFAND 4 / 15 (hau, ibo, pcm, sna; significant for hau, ibo, pcm, about -0.3 to -0.5 chrF++; AfriCOMET is not significantly lower for any language).

<<T5>>

### Along the RL chain (which stage produced the gain)

<<T3>>

SFT -> stage A is 94 % of the SFT -> B1 gain on FLORES+ (7.96 of 8.50) and 96 % on NTREX; B1 and the mixed stage add +0.54 and +0.31 on FLORES+, +0.30 and +0.18 on NTREX, and nothing measurable on MAFAND (AfriCOMET on MAFAND -0.001, a significant but negligible decrease at each of the two later stages). The dev series of stage A (`reports/rl_stage_A_385m_2026-11-03.md`) agrees: +3.3 of +3.7 dev chrF++ in the first 100 of 400 steps.

## 5. Science: SFT alpha 0.5 against the science RL (task 4) -- PENDING

The science RL (`runs_385m_sci/final`, started from the alpha-0.5 interpolation of the science SFT) is at step ~50 of 150 at the time of writing; the on-box job `post_sci_rl_waiter.sh` starts the sentence-level AfriScience dev/test, NTREX/MAFAND and document evaluation of its best and final checkpoints when `summary.json` appears. This section will be filled from that output with the same tooling (paired bootstrap per language on AfriScience-MT dev and test, sentence level), under these rules fixed in advance:

* independent evidence = plain chrF against the human references of AfriScience-MT test (the science RL's chrF++ term uses AfriScience-MT **train** pairs, not test) and the reference-based `africomet-stl-1.1` (held out of the science reward); **excluded as reward family**: `africomet-qe-stl-1.1`, `ssa-comet-mtl` (reference-free and the reference-based use of it in the science reward), and therefore also SSA-COMET-QE, which stays in the science reports only as the headline of the AfriScience-MT paper with the Goodhart caution already written there;
* the SFT alpha-0.5 numbers already exist (`reports/eval_science_sft_385m_2026-11-04.md`): AfriScience test chrF 45.40 (alpha 0) -> 46.63 (alpha 0.5), NTREX -0.26, MAFAND +0.27 chrF++, SSA-COMET-QE test 0.6130 -> 0.6118 (no gain);
* finishing and looping are reported separately, with the same cut-at-loop and clean-row variants as above.

## 6. What the stored data cannot prove, and what the controls would cost (task 5)

**Cannot be established from what exists**

1. **No matched-compute control.** The SFT was not continued for the 8.7 GPU-hours (553,600 rollouts, 34,600 prompts) the RL used. The comparison is "SFT" against "SFT + RL", so it does not show that RL is better use of the compute than more SFT. The finishing part in particular (SFT 83 % finished on FLORES+, 17 % unfinished or looping) looks like something more SFT with end-of-turn emphasis, or any decoding guard, could fix.
2. **No decoding control.** The SFT was decoded greedily with 768 new tokens and no repetition guard (no repetition penalty, no n-gram blocking, no stop on a repeated block). A decode-time loop guard on the SFT would remove part of the +3.9 chrF++ that loop-cutting recovers; it was not run.
3. **One seed.** The intervals are over test sentences. Variation between training runs (seed, prompt draw, sampling) is not measured; the dev macro's own noise was estimated at sd 0.09 to 0.10 chrF++ per evaluation, but that is evaluation noise, not training-run noise. The gain of +8 is large compared with plausible seed noise, the +2.4 beyond finishing less obviously so, and the per-language differences of under 1 chrF++ (pcm, hau, ibo) cannot be separated from it.
4. **Reward-family overlap.** AfriCOMET-STL is held out of the reward but is the same family as one reward member; the chrF++ gain is partly in the optimised family through consensus and, for 5 + 7 languages, the NLLB and human-reference terms. Both agree here (chrF++ +7.96, chrF +8.59, AfriCOMET +0.063), which is evidence against a pure metric artefact but not independent evidence of quality. No human evaluation is part of this analysis.
5. **The loop flag is the reward's own guard.** `is_repetitive` / `is_self_repetitive` feed the penalties during RL, so the RL is optimised against exactly this detector. The detector is quiet on the human references (0.01 % of FLORES+ rows) and the chrF++ and AfriCOMET results do not depend on it except through the subsets, but a loop type the detector misses would be undercounted for the RL.
6. **Survivor and selection effects** of the clean-row subsets (section 3); and the cut-at-loop rule is a heuristic (earliest block of up to 20 words repeated three times, else the second occurrence of the first 4-gram that occurs three times).
7. **Low-score languages.** For aeb, ary, zgh and several others both models score chrF++ 2 to 9; differences there say little about translation quality.
8. **Benchmarks were seen several times** (stage A, B1, mixed and weak-run evaluations; the "keep RL" decision used NTREX and MAFAND). Checkpoint choice itself used only the dev set, so test-set selection bias is small, but the reported test numbers are not from a single pre-registered look.
9. **Contamination:** the audit covers the 385M SFT text only; the base model's pretraining text is unknown, and it is the same for SFT and RL, so it does not change the paired comparison.
10. **Same-data caveat on the RL prompts:** the RL used English prompts from the clean pools with the protected-index filter; the 7 new-language prompt files that needed filtering are disclosed in `reports/rl_stage_A_385m_2026-11-03.md`. No benchmark source sentence is a prompt by that filter, which is a property of the filter, not independently re-checked here.

**Options for closing the gaps** (estimates only, nothing was run; the GPU is on the science RL, then the long-document RL, so each option displaces that queue)

Measured rates used: stage A 15,920 s for 400 steps (39.8 s per step, 64 prompts x 16 samples), B1 10,479 s for 135 steps (77.6 s per step, 48 prompts), mixed 4,786 s for 60 steps (79.8 s per step, 42 prompts); the chain A + B1 + mixed is 8.66 GPU-h on one A100 (B1 started from stage A's final, mixed from B1's best checkpoint). SFT throughput measured in the science SFT: ~16.5 k tokens/s at micro-batch 4 (65,536 tokens per step), other jobs sharing the GPU part of the time.

| option | what it tests | estimated GPU-h | notes |
|---|---|---:|---|
| A. continued SFT, matched wall-clock / GPU time | is RL a better use of 8.7 GPU-h than more SFT | **~8.7 h training (~510 M tokens at 16.5 k tok/s) + ~0.8 h evaluation** | needs a data and learning-rate choice (continue on the shard mix at the SFT's final LR, with and without end-of-turn emphasis); the faster the SFT code runs, the more tokens fit |
| A'. continued SFT, matched in rollouts / tokens | same data volume as the RL saw | ~1 h training (~55 M target tokens) + 0.8 h evaluation | there are no references for RL prompts, so the targets must be distilled (best-of-16 or consensus outputs): add ~3 to 4 GPU-h of generation and scoring (rough, 70 to 80 % of the 4.4 h of stage A), i.e. rejection-sampling fine-tuning ~5 h in all |
| B. decode-time loop guard on the SFT (repetition penalty / n-gram block / stop on repeated block) | how much of the finishing gain is a decoding problem | **~0.5 h** (inference of 122,585 rows + AfriCOMET), no training | cheapest and most informative first control |
| C. second RL seed on a subset | seed variance of the gain | stage A recipe, 100 steps (64 x 16): 1.1 h; with half the prompts per step on a 20-language subset ~0.6 to 0.8 h; + 0.5 h evaluation: **~1.2 to 1.7 h per seed** | most of the gain arrives by step 100 (dev +3.3 of +3.7), so 100 steps is enough to measure the gain; 2 extra seeds ~3 h |
| D. full replication of the chain | seed variance of the whole pipeline | 8.7 h (+ 0.8 h evaluation) | only if C shows large variation |

Recommended order if GPU time is made available: B (0.5 h, decides how much of the finishing gain needs RL at all), then C with two seeds (about 3 h), then A if the paper wants the "RL against continued SFT" claim.
