# Generation-config study for the final model: temperature, top-p, top-k, min-p, typical / epsilon sampling, best-of-n and MBR, at sentence level, document level and in RL rollouts (2026-10-22)

Task R12 / D-64. Question from the owner: which generation configuration should be used for short text, long documents, RL rollouts and as the fast config, given that greedy decoding is "very repetitive". Model: `<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c@latest` (RL final, step 1433). Builds on `reports/final_model_decoding_study_2026-10-21.md` (R5: sentence level, greedy / beam / repetition penalty / n-gram blocking / MBR) and `reports/longdoc_eval_final_model_2026-10-21.md` (R6: documents). Data, box scripts and JSON: `reports/data/generation_config_study/` (`export/*.csv`, `json/*.json`, `scripts/`); renderer `gemma_mt/scripts/build_generation_config_report.py`. Nothing was pushed; scratch on the box was deleted at the end.

## 0. Answer

**No sampler beats greedy on a single draw, at any level; the gains come from search (beam) or from several samples plus a selection rule; for long documents the sampler is irrelevant and the unit of decoding must be the sentence; for RL rollouts keep T = 1.0 with no truncation.**

1. **Short text (quality): beam search with 4 beams and a greedy fallback** (R5: +0.68 chrF++, +0.7 spBLEU, +0.011 AfriCOMET over greedy); where HF `generate` is not available or the language is weak / medium, **MBR over 16 vLLM samples at T 0.7 + min-p 0.05** gives +0.83 chrF++ (+0.22 spBLEU; strong +0.69, medium +0.79, weak +0.92, against beam 4's +0.57 / +0.95 / +0.56) at 16x the tokens. Single-sample sampling is *worse*: 80 temperature x cut-off combinations lose 0 ... 2.0 chrF++ (T <= 0.4: tied; T 1.0: -2.0; the Gemma default `T 1, top-k 64, top-p 0.95` that ships in `generation_config.json`: -1.40 [-1.55, -1.24]); seed noise is 0.02-0.09, typical / epsilon / eta sampling are no different from min-p / top-p. Selecting by AfriCOMET-QE *looks* best on AfriCOMET (+0.045) and is below greedy on chrF++ (-0.10 ... -0.41): metric gaming, do not use.
2. **Long documents: translate sentence by sentence with greedy and re-decode the 0.6% unfinished sentences with best-of-8 at T 0.7** (R6's `sentfix`, repair upgraded from 4 to 8 candidates: 92.1% of the repaired units finish instead of 85.2%). On 1,278 documents no whole-document sampler comes within 20 chrF++ of it beyond 1,024 tokens (b1024: greedy 29.1, best sampler 35.6, sentence-by-sentence 55.6; b2048: 17.9 / 27.3 / 58.7) because the sampled output is 22-47% too short; and which sampler repairs the unfinished sentences changes the document chrF++ by < 0.05. For a single pass that cannot be split, T 1.0 + min-p 0.05 + `min_tokens = 1.4 x source tokens` is the best I found (37.4 / 30.0 at b1024 / b2048).
3. **RL rollouts: keep `temperature 1.0, top_p 1.0, top_k 0, min_p 0` (unbiased, as now) with the greedy group member (D-59).** Lower T or a cut-off raises the mean reward of the rollouts (QE-mode languages: T 1.0 0.445, top-p 0.95 0.519, top-p 0.9 0.543, min-p 0.05 0.557, T 0.7 0.575, T 0.5 0.628, T 0.3 0.654, greedy 0.648) but shrinks the within-group std (0.289 -> 0.272 / 0.266 / 0.267 / 0.259 / 0.238 / 0.214), brings back the looping in the weakest languages (kbp, nus degenerate: 0.42% at T 0.3 against 0), and samples from a distribution that is 0.02 (top-p 0.95) ... 0.13 (T 0.3) nats per token away from the policy; with top-p 0.95, **74% of the T 1.0 sequences contain a token the truncated sampler can never produce**, and those sequences score 0.35 lower. If a milder rollout is wanted, top-p 0.95 is the least harmful and needs `allow_truncated_sampling: true`.
4. **Fast config: greedy** (vLLM ~200 sentences/s on the shared card, R5), no penalty, no n-gram blocking; its 0.1% loop rate is fine at sentence level. Greedy *is* repetitive in the weak languages (repeated word 3-gram in 52% of Kabiye and Nuer sentences, 36% Acholi, 33% Fon, 17% Yoruba and Efik, against 2.7% in the references), but removing the repetition by sampling costs more chrF++ than the repetition does, and beam search / consensus do not remove it either.
5. **Does the best sampler differ by language group? Not in the ranking, only in the size of the effect.** Every sampler is no better than greedy in every group; the strong group (swh hau zul kin som) is the most sensitive (T 0.5 already -0.70, T 1.0 -2.75), the weak group (including Yoruba, Amharic, Wolof, Tigrinya, Kabiye, Nuer, Kabyle, Fon, Acholi, Efik) the least (T <= 0.3 tied, T 1.0 -1.59; T 1.0 + min-p 0.1 only -0.42). Consensus-type selection gains **more** in the weak and medium groups (MBR n = 16: weak +0.9, medium +0.8 ... +1.0, strong +0.3 ... +0.7) and the best pool sampler is a higher temperature for the weak group (T 1.0 + min-p 0.1: +0.98) than for the strong one (T 0.7 + min-p 0.05: +0.69), but the intervals overlap: **T 0.7 + min-p 0.05 is within noise of the best in every group**. Yoruba (weak, 26.9 chrF++): single samples T <= 0.3 tie, T 1.0 -1.4, beam 4 +0.8, MBR n = 16 +0.5 ... +0.7; Igbo and the other medium languages behave like the weak ones for MBR (+0.4 ... +1.6) and like the strong ones for single samples. **For the SFT-final checkpoint the answer is different** (section 5): its greedy loops (10.9% of the sentences) and a low temperature helps the weak group (T 0.5: weak +1.40, strong -1.31), and MBR over 16 samples at T 0.7 gives +2.45 chrF++.

### Decision table

| use | configuration | expected effect (final model, vs greedy) | cost | exact parameters |
|---|---|---|---|---|
| **(1) short text, quality** (a sentence to a short paragraph, up to ~300 output tokens) | **beam search 4 + greedy fallback** (R5) | chrF++ +0.68, spBLEU +0.69, AfriCOMET +0.011; strong +0.57, medium +0.95, weak +0.60; the fallback removes the beam loops of Nuer and Kabiye | ~4.6x HF greedy time | HF: `model.generate(input_ids, attention_mask, num_beams=4, do_sample=False, max_new_tokens=320, early_stopping=False, length_penalty=1.0, eos_token_id=[106, 1], pad_token_id=0, temperature=None, top_p=None, top_k=None)`; if the beam output has no end token, is empty or `rl.guards.is_repetitive` / `is_self_repetitive`, use the greedy output |
| **(1b) short text, quality, vLLM only / weak and medium languages** | **MBR over 16 samples, T 0.7 + min-p 0.05**, pick the finished candidate with the highest mean sentence chrF++ (word order 2) against the other 15 (add the greedy output to the pool if spBLEU matters) | chrF++ +0.83 [+0.72, +0.94], spBLEU +0.22; strong +0.69, medium +0.79, weak +0.92; AfriCOMET +0.007; Yoruba +0.5, Bemba +1.2, Kabiye +1.7, Nuer +0.6 (where beam 4 gives -1.0) | 16x the tokens of greedy + ~0.1 s CPU per sentence (`gemma_mt.rl.consensus`) | `SamplingParams(n=16, temperature=0.7, top_p=1.0, top_k=0, min_p=0.05, max_tokens=320, stop_token_ids=[106, 1], seed=<per sentence>, detokenize=False, skip_special_tokens=False)`; n = 8 gives +0.57; n = 4 only +0.2 |
| **(2) long documents** (> ~300 source tokens) | **sentence-by-sentence greedy + repair of the unfinished sentences by best-of-8** (R6 `sentfix`, repair upgraded to n = 8) | b256-b2048 chrF++ 49.0 / 53.5 / 55.6 / 58.4 against NLLB (+1.4 / +5.8 / +25.9 / +40.2 over whole-document greedy), 100% sentence coverage, loop 0-3%; the repair sampler changes the document chrF++ by < 0.05 | 2.0 generated tokens per source token (as whole-document greedy); the repair touches 0.6% of the units | per unit: `SamplingParams(temperature=0.0, max_tokens=min(int(3.5 * src_tokens) + 16, 1024), stop_token_ids=[106, 1])`; repair of the units that did not emit the end token: `SamplingParams(n=8, temperature=0.7, top_p=1.0, top_k=0, min_p=0.0, max_tokens=<same>, stop_token_ids=[106, 1], seed=<per unit>)`, keep the finished candidate with the highest chrF++ consensus (greedy output if none finished) |
| (2b) long documents when a single pass is forced | **T 1.0 + min-p 0.05 + `min_tokens` = 1.4 x source tokens** (finishes 91-100%, loops 3-12%); do not go beyond ~512 source tokens if avoidable | b1024 37.4, b2048 30.0 chrF++ (greedy 29.1 / 17.9; plain T 1.0 + min-p 0.05 35.6 / 27.3) - still 18-29 below (2) | | `SamplingParams(temperature=1.0, top_p=1.0, top_k=0, min_p=0.05, min_tokens=min(int(1.4 * src_tokens), max_tokens - 1), max_tokens=min(int(3.5 * src_tokens), ctx - prompt), stop_token_ids=[106, 1], seed=<per document>)` |
| **(3) RL rollouts** | **keep T 1.0, top-p 1.0, top-k 0, min-p 0** (unbiased) for the sampled members + the greedy member (D-59) | mean reward of a T 1.0 rollout is 0.445 (QE-mode languages) against 0.648 for greedy; any cut-off or T < 1 raises it (+0.07 ... +0.21) at the price of -6% ... -26% group std, KL(q \|\| p) 0.02 ... 0.13 nats/token, 74-96% of the T 1.0 sequences containing a token the truncated sampler never emits, and loops back in kbp / nus | none | trainer: `temperature: 1.0, top_p: 1.0, top_k: 0, allow_truncated_sampling: false, group_size: 16, greedy_in_group: true` = `SamplingParams(n=15, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, max_tokens=<bin budget>, stop_token_ids=[106, 1], seed=<per prompt>)` + one greedy request |
| **(4) fast** | **greedy** | reference (32.52 chrF++); 0.1% loops at sentence level | ~200 sentences/s (R5) | `SamplingParams(temperature=0.0, max_tokens=320, stop_token_ids=[106, 1])` (no repetition_penalty, no frequency_penalty, no n-gram blocking: R5/R6) |
| not recommended | HF defaults of the released model (`do_sample=true, top_k=64, top_p=0.95`, T = 1), any single-sample T >= 0.4, repetition_penalty, no_repeat_ngram, QE-based selection, top-k on its own | -1.40 chrF++ (defaults) ... -2.0 (T 1.0) | | the shipped `generation_config.json` should say `do_sample: false` (and callers should pass `num_beams=4` explicitly) |

`top_k=0` disables top-k in vLLM 0.31 (`-1` is also accepted; HF treats 0 as off too); `min_p=0.0` disables min-p; all configurations use `max_tokens` / `max_new_tokens` from the table, stop ids 106 (`<end_of_turn>`) and 1 (`<eos>`), the prompt of `gemma_mt.format.build_prompt`, and strip the leading language tag from the completion before use.


## 1. What was run

* **Model:** `<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c@latest` = RL final, step 1433, `runs/final_1433` (weights byte-identical to the Hub snapshot `80baefff...`, sha256 `09cfba3c...`). The SFT-final model (`/workspace/jobs/a2/models/sft`, step 27,638) was run at sentence level only (section 5).
* **Sentence level** = R5's set unchanged: 20 languages x 200 `no_pair` sentences (n = 3,970), same prompt format, `max_new_tokens` 320, stop ids 106 / 1, groups strong / medium / weak as in R5. Metrics as R5: macro chrF++ (word order 2) and spBLEU over the 20 languages, equal language weight, paired bootstrap of the delta to greedy (1,000 resamples of the sentences inside each language, the *same* resamples for every configuration), finished / loop (`rl.guards`) / GlotLID right-language rate / number and name recall, AfriCOMET (with reference) and AfriCOMET-QE on the 18 validated languages. New: **repeated 3-gram rate** (share of outputs with a repeated word 3-gram that the human reference does not have) and `dist2`. My greedy reproduces R5 (32.516 chrF++, 12.149 spBLEU); my T 0.3 / 0.5 / 0.7 reproduce R5's to 0.01.
* **Engines:** vLLM 0.31 for everything that has `temperature`, `top_p`, `top_k`, `min_p`, `n`, `seed` (SamplingParams); HF `generate` for typical-p, epsilon and eta sampling (not in vLLM). One seed per sentence (`1234 + i`), the seed offset changes for the repeated-seed runs. Selection (MBR, oracle, QE) is done offline on pools of 16 samples per sentence.
* **Document level** = R6's documents and scoring pipeline unchanged (`assemble.py`, `score_items.py`, `agg.py` copied to `/workspace/jobs/r12/doc`, R6's request files, items and pseudo-references read in place): core documents of bins b256 / b512 / b1024 / b2048, the 8-language subset B8 (hau yor swh amh zul wol kea sag), chrF++ against the NLLB-3.3B pseudo-reference (macro over 8 languages, bootstrap over documents) and against the **clean** SmolDoc human references (6 languages; bins `sm`, `st512`, `st1024`, `st2048`), finished / loop / sentence coverage / length ratio. My re-assembly of R6's `sentfix` from its unit outputs reproduces R6's table to the last digit (49.0 / 53.5 / 55.6 / 58.4). **Because of the GPU budget (7 GB, then 4 GB, nice 19) the whole-document sampling configurations and the sentence-unit experiments were run on a document subset**: 1,278 documents = 240 core documents per bin for b256 / b512 / b1024 (30 per bin x 8 languages), 160 for b2048 (20 x 8), and the clean human documents (157 `sm`, 143 `st512`, 83 `st1024`, 15 `st2048`); R6's configurations (greedy, T 0.3 / 0.5 / 0.7 / 1.0, best-of-4, sentfix) are re-aggregated on exactly the same documents, so every comparison is paired. The unfinished-unit repair experiments use all 3,418 selected documents (b256-b2048 core + clean human).
* **RL measurement** (section 4): 280 prompts, 16 samples each, real reward code.
* **Budget / priority:** the GPU was shared with the running RL and SFT jobs; after the first half of the sentence level I was limited to 7 GB, later 4 GB, under `nice -n 19`. The vLLM engine ran with 2.7-2.9 GB; I did not touch other jobs except for one slip (section 7).

## 2. Sentence level (R5's 20 languages x 200 `no_pair` sentences, n = 3,970)

### 2.1 One sample from a sampler: nothing beats greedy

The grid has 8 temperatures x 10 cut-off rules (none, top-p 0.8 / 0.9 / 0.95, top-k 10 / 20 / 50, min-p 0.02 / 0.05 / 0.1) = 80 configurations, one sample per sentence, seed per sentence (the same seeds as R5; my T 0.3 / 0.5 / 0.7 reproduce R5's numbers to 0.01). Delta of macro chrF++ against greedy (32.52), `*` = the 95% paired-bootstrap interval excludes 0:

| temperature | none | top-p 0.8 | top-p 0.9 | top-p 0.95 | top-k 10 | top-k 20 | top-k 50 | min-p 0.02 | min-p 0.05 | min-p 0.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| T 0.2 | -0.05 | -0.03 | -0.06 | -0.01 | -0.06 | -0.05 | -0.05 | -0.05 | +0.02 | -0.00 |
| T 0.3 | -0.13* | -0.08 | -0.04 | -0.08 | -0.08 | -0.10 | -0.13* | -0.07 | -0.04 | -0.03 |
| T 0.4 | -0.25* | -0.01 | -0.13* | -0.17* | -0.28* | -0.24* | -0.27* | -0.25* | -0.16* | -0.13* |
| T 0.5 | -0.35* | -0.11 | -0.22* | -0.31* | -0.40* | -0.40* | -0.39* | -0.27* | -0.24* | -0.20* |
| T 0.6 | -0.65* | -0.25* | -0.35* | -0.37* | -0.57* | -0.56* | -0.62* | -0.40* | -0.40* | -0.30* |
| T 0.7 | -0.79* | -0.29* | -0.44* | -0.59* | -0.79* | -0.79* | -0.84* | -0.62* | -0.42* | -0.41* |
| T 0.8 | -1.14* | -0.33* | -0.58* | -0.83* | -1.03* | -1.07* | -1.12* | -0.89* | -0.66* | -0.46* |
| T 1.0 | -2.02* | -0.71* | -1.25* | -1.50* | -1.69* | -1.77* | -1.93* | -1.45* | -1.15* | -0.81* |

* **No sampler configuration is better than greedy.** The best point estimate is T 0.2 + min-p 0.05 (+0.02, interval [-0.07, +0.12]); 20 of the 30 configurations at T <= 0.4 are statistically tied with greedy (all ten at T 0.2, nine of ten at T 0.3, T 0.4 + top-p 0.8). Without a cut-off every temperature from 0.3 upward is significantly worse (T 0.5 + top-p 0.8, -0.11, is the only tie above T 0.4), and the loss grows monotonically with T: plain T 1.0 loses 2.02 chrF++ [-2.19, -1.84] and 1.89 spBLEU.
* **A cut-off rule recovers part of the loss, never all of it.** At T 1.0 the order is top-p 0.8 (-0.71) > min-p 0.1 (-0.81) > min-p 0.05 (-1.15) > top-p 0.9 (-1.25) > min-p 0.02 (-1.45) > top-p 0.95 (-1.50) > top-k 10 (-1.69) > top-k 20 (-1.77) > top-k 50 (-1.93) > nothing (-2.02): the more probability mass the rule removes, the closer to greedy; a fixed top-k is the worst because it does not adapt to the sharpness of the distribution (and does nothing at all at T <= 0.2, where `T0.2_k20` = `T0.2_k50` = `T0.2` exactly). The **Gemma default** that ships in the model's `generation_config.json` (`do_sample: true`, `top_k: 64`, `top_p: 0.95`, temperature unset = 1.0) is `T1.0_k64_p0.95`: **-1.40 chrF++ [-1.55, -1.24], -1.38 spBLEU, numbers kept 92.7% (greedy 95.7%)**. Anyone who calls `model.generate(...)` without arguments gets this, not greedy.
* **Typical, epsilon and eta sampling** (HF `generate`, vLLM 0.31 has none of them) behave like the other cut-offs: at T 1.0 typical-p 0.8 -0.87, 0.9 -1.20, 0.95 -1.49; epsilon 0.02 -1.21, 0.01 -1.38, 0.003 -1.67, 0.0003 -1.91; eta 0.01 -1.37 (epsilon and eta give the same numbers at 0.001 and 0.0003 to 0.01). Plain HF T 1.0 (-2.00) equals vLLM T 1.0 (-2.02), so the two engines agree. None is better than min-p or top-p at the same removed mass.

| HF sampling config (T, then cut-off) | chrF++ | delta chrF++ | 95% CI | strong | medium | weak |
|---|---|---|---|---|---|---|
| `T0.5_typ0.9` | 32.26 | -0.25* | [-0.37, -0.14] | -0.36* | -0.45* | -0.11 |
| `T0.7_typ0.9` | 32.07 | -0.45* | [-0.59, -0.32] | -0.81* | -0.54* | -0.22* |
| `T0.7_eps0.001` | 31.77 | -0.75* | [-0.90, -0.61] | -1.30* | -0.86* | -0.41* |
| `T1.0_typ0.8` | 31.65 | -0.87* | [-1.00, -0.72] | -1.34* | -0.91* | -0.61* |
| `T1.0_typ0.9` | 31.31 | -1.20* | [-1.36, -1.05] | -1.94* | -1.17* | -0.85* |
| `T1.0_eps0.02` | 31.30 | -1.21* | [-1.36, -1.06] | -2.22* | -1.25* | -0.69* |
| `T1.0_eta0.01` | 31.15 | -1.37* | [-1.54, -1.20] | -2.28* | -1.51* | -0.85* |
| `T1.0_eps0.01` | 31.14 | -1.38* | [-1.54, -1.21] | -2.26* | -1.52* | -0.86* |
| `T1.0_typ0.95` | 31.03 | -1.49* | [-1.65, -1.33] | -2.29* | -1.56* | -1.05* |
| `T1.0_eps0.003` | 30.85 | -1.67* | [-1.83, -1.49] | -2.59* | -1.83* | -1.13* |
| `T1.0_eta0.003` | 30.82 | -1.69* | [-1.86, -1.52] | -2.61* | -1.82* | -1.17* |
| `T1.0_eps0.001` | 30.71 | -1.81* | [-1.96, -1.63] | -2.75* | -1.94* | -1.27* |
| `T1.0_eta0.001` | 30.71 | -1.81* | [-1.96, -1.63] | -2.75* | -1.94* | -1.27* |
| `T1.0_eta0.0003` | 30.62 | -1.90* | [-2.06, -1.73] | -2.85* | -2.05* | -1.36* |
| `T1.0_eps0.0003` | 30.61 | -1.91* | [-2.06, -1.73] | -2.85* | -2.05* | -1.37* |
| `T1.0` | 30.52 | -2.00* | [-2.16, -1.82] | -2.86* | -2.16* | -1.48* |

* **Seed variance is small** (5 independent seeds per configuration, sd of the macro chrF++ 0.02-0.09), so the differences above are not seed noise; a single draw is as reliable as the average of five to +-0.1:

| config | chrF++ of the 5 seeds | mean | sd | mean delta vs greedy | spBLEU mean +- sd |
|---|---|---|---|---|---|
| `T0.3` | 32.39, 32.44, 32.39, 32.42, 32.41 | 32.41 | 0.02 | -0.11 | 12.03 +- 0.03 |
| `T0.5` | 32.17, 32.23, 32.18, 32.17, 32.15 | 32.18 | 0.03 | -0.34 | 11.80 +- 0.04 |
| `T0.7` | 31.73, 31.71, 31.71, 31.62, 31.63 | 31.68 | 0.05 | -0.84 | 11.31 +- 0.03 |
| `T1.0` | 30.50, 30.48, 30.60, 30.55, 30.52 | 30.53 | 0.05 | -1.99 | 10.28 +- 0.05 |
| `T1.0_p0.9` | 31.26, 31.33, 31.37, 31.33, 31.30 | 31.32 | 0.04 | -1.20 | 10.97 +- 0.04 |
| `T1.0_mp0.05` | 31.37, 31.47, 31.56, 31.48, 31.37 | 31.45 | 0.08 | -1.07 | 11.04 +- 0.09 |
| `T1.0_mp0.1` | 31.71, 31.70, 31.72, 31.59, 31.58 | 31.66 | 0.07 | -0.86 | 11.26 +- 0.05 |
| `T0.7_mp0.05` | 32.09, 32.04, 32.07, 31.86, 31.99 | 32.01 | 0.09 | -0.51 | 11.59 +- 0.10 |

By quality group (delta chrF++, T x cut-off, strong = swh hau zul kin som; medium = ibo sna lug bem nbl; weak = the other 10):

Strong group:

| temperature | none | top-p 0.8 | top-p 0.9 | top-p 0.95 | top-k 10 | top-k 20 | top-k 50 | min-p 0.02 | min-p 0.05 | min-p 0.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| T 0.2 | -0.27* | -0.12 | -0.22* | -0.12 | -0.18 | -0.27* | -0.27* | -0.15 | -0.07 | -0.19 |
| T 0.3 | -0.28* | -0.16 | -0.13 | -0.27* | -0.14 | -0.22 | -0.28* | -0.14 | -0.04 | -0.04 |
| T 0.4 | -0.36* | -0.03 | -0.23* | -0.22 | -0.34* | -0.37* | -0.37* | -0.34* | -0.22 | -0.21 |
| T 0.5 | -0.70* | -0.22 | -0.40* | -0.52* | -0.73* | -0.72* | -0.80* | -0.49* | -0.46* | -0.30* |
| T 0.6 | -1.10* | -0.32* | -0.66* | -0.76* | -1.05* | -1.11* | -1.15* | -0.79* | -0.73* | -0.62* |
| T 0.7 | -1.38* | -0.50* | -0.76* | -1.12* | -1.33* | -1.47* | -1.37* | -1.18* | -0.82* | -0.81* |
| T 0.8 | -1.96* | -0.77* | -0.97* | -1.39* | -1.80* | -1.82* | -1.84* | -1.59* | -1.33* | -0.91* |
| T 1.0 | -2.75* | -1.23* | -1.66* | -2.02* | -2.70* | -2.80* | -2.85* | -2.11* | -1.83* | -1.60* |

Weak group:

| temperature | none | top-p 0.8 | top-p 0.9 | top-p 0.95 | top-k 10 | top-k 20 | top-k 50 | min-p 0.02 | min-p 0.05 | min-p 0.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| T 0.2 | +0.03 | -0.02 | +0.00 | +0.01 | -0.04 | +0.03 | +0.03 | +0.02 | +0.05 | +0.01 |
| T 0.3 | -0.07 | -0.08 | +0.00 | +0.02 | -0.04 | -0.02 | -0.07 | +0.04 | -0.00 | -0.03 |
| T 0.4 | -0.19* | +0.06 | -0.09 | -0.13 | -0.22* | -0.12 | -0.20* | -0.22* | -0.09 | -0.06 |
| T 0.5 | -0.16 | -0.03 | -0.10 | -0.22* | -0.23* | -0.19* | -0.13 | -0.10 | -0.12 | -0.12 |
| T 0.6 | -0.39* | -0.19* | -0.16 | -0.13 | -0.31* | -0.25* | -0.33* | -0.14 | -0.18* | -0.16 |
| T 0.7 | -0.44* | -0.16 | -0.32* | -0.32* | -0.45* | -0.37* | -0.51* | -0.37* | -0.21* | -0.21* |
| T 0.8 | -0.66* | -0.13 | -0.39* | -0.49* | -0.62* | -0.64* | -0.69* | -0.51* | -0.30* | -0.24* |
| T 1.0 | -1.59* | -0.46* | -1.01* | -1.17* | -1.11* | -1.19* | -1.38* | -1.00* | -0.70* | -0.42* |

The strong group is the most sensitive to randomness (T 0.5 already -0.70*, T 1.0 -2.75*), the weak group the least (T <= 0.3 within noise of greedy, T 1.0 -1.59*, T 1.0 + min-p 0.1 -0.42*). So **mid- and low-resource languages tolerate sampling better than the strong ones, but even there nothing wins on a single draw**. Medium group and the best-of-group listing:

**all** (greedy 32.52 chrF++, 12.15 spBLEU)

| config | chrF++ | delta chrF++ | 95% CI | delta spBLEU |
|---|---|---|---|---|
| `T0.2_mp0.05` | 32.54 | +0.02 | [-0.07, +0.12] | +0.10 |
| `T0.2_mp0.1` | 32.51 | -0.00 | [-0.10, +0.08] | +0.02 |
| `T0.4_p0.8` | 32.51 | -0.01 | [-0.12, +0.10] | -0.01 |
| `T0.2_p0.95` | 32.50 | -0.01 | [-0.12, +0.08] | +0.06 |
| `T0.3_mp0.1` | 32.49 | -0.03 | [-0.13, +0.08] | +0.02 |

**strong** (greedy 47.02 chrF++, 22.91 spBLEU)

| config | chrF++ | delta chrF++ | 95% CI | delta spBLEU |
|---|---|---|---|---|
| `T0.4_p0.8` | 47.00 | -0.03 | [-0.25, +0.20] | +0.16 |
| `T0.3_mp0.05` | 46.98 | -0.04 | [-0.26, +0.20] | -0.02 |
| `T0.3_mp0.1` | 46.98 | -0.04 | [-0.26, +0.17] | +0.07 |
| `T0.2_mp0.05` | 46.95 | -0.07 | [-0.27, +0.12] | +0.09 |
| `T0.2_p0.95` | 46.90 | -0.12 | [-0.33, +0.09] | +0.01 |

**medium** (greedy 36.71 chrF++, 12.61 spBLEU)

| config | chrF++ | delta chrF++ | 95% CI | delta spBLEU |
|---|---|---|---|---|
| `T0.2_mp0.1` | 36.86 | +0.15 | [-0.03, +0.34] | +0.05 |
| `T0.2_mp0.05` | 36.78 | +0.07 | [-0.14, +0.29] | +0.13 |
| `T0.2_p0.95` | 36.76 | +0.05 | [-0.16, +0.25] | +0.16 |
| `T0.2_p0.8` | 36.75 | +0.04 | [-0.16, +0.24] | +0.05 |
| `T0.2` | 36.75 | +0.04 | [-0.17, +0.24] | -0.03 |

**weak** (greedy 23.16 chrF++, 6.54 spBLEU)

| config | chrF++ | delta chrF++ | 95% CI | delta spBLEU |
|---|---|---|---|---|
| `T0.4_p0.8` | 23.22 | +0.06 | [-0.09, +0.20] | -0.04 |
| `T0.2_mp0.05` | 23.21 | +0.05 | [-0.09, +0.18] | +0.10 |
| `T0.3_mp0.02` | 23.21 | +0.04 | [-0.11, +0.19] | -0.02 |
| `T0.2` | 23.19 | +0.03 | [-0.11, +0.16] | +0.06 |
| `T0.2_k20` | 23.19 | +0.03 | [-0.11, +0.16] | +0.06 |

Canonical rows with the quality-adjacent metrics (loop = unfinished / repetitive / empty by `rl.guards`; repeated 3-gram = share of outputs with a repeated word 3-gram that the human reference does not have):

| config | chrF++ | delta chrF++ | delta spBLEU | delta strong | delta medium | delta weak | loop % | repeated 3-gram % | numbers kept % | names kept % | LID % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `greedy` | 32.52 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | 0.10 | 13.2 | 95.7 | 92.6 | 99.9 |
| `T0.2_mp0.05` | 32.54 | +0.02 | +0.10 | -0.07 | +0.07 | +0.05 | 0.23 | 12.3 | 96.0 | 93.2 | 99.9 |
| `T0.3` | 32.39 | -0.13* | -0.11 | -0.28* | -0.09 | -0.07 | 0.15 | 11.6 | 96.1 | 92.8 | 99.9 |
| `T0.5` | 32.17 | -0.35* | -0.37* | -0.70* | -0.37* | -0.16 | 0.00 | 10.0 | 96.2 | 92.2 | 99.9 |
| `T0.7` | 31.73 | -0.79* | -0.78* | -1.38* | -0.91* | -0.44* | 0.00 | 7.5 | 96.3 | 92.0 | 99.9 |
| `T0.7_mp0.05` | 32.09 | -0.42* | -0.47* | -0.82* | -0.46* | -0.21* | 0.00 | 9.2 | 95.9 | 92.4 | 99.9 |
| `T0.8_p0.9` | 31.93 | -0.58* | -0.65* | -0.97* | -0.58* | -0.39* | 0.00 | 8.7 | 95.5 | 91.9 | 99.9 |
| `T1.0` | 30.50 | -2.02* | -1.89* | -2.75* | -2.16* | -1.59* | 0.00 | 4.5 | 93.1 | 91.3 | 99.9 |
| `T1.0_p0.95` | 31.02 | -1.50* | -1.41* | -2.02* | -1.64* | -1.17* | 0.00 | 4.9 | 93.6 | 91.8 | 99.9 |
| `T1.0_k64_p0.95` | 31.12 | -1.40* | -1.38* | -1.99* | -1.48* | -1.06* | 0.00 | 5.8 | 92.7 | 91.9 | 99.9 |
| `T1.0_p0.9` | 31.26 | -1.25* | -1.23* | -1.66* | -1.33* | -1.01* | 0.00 | 5.6 | 94.3 | 92.0 | 99.9 |
| `T1.0_mp0.05` | 31.37 | -1.15* | -1.20* | -1.83* | -1.37* | -0.70* | 0.00 | 6.7 | 94.5 | 92.2 | 99.9 |
| `T1.0_mp0.1` | 31.71 | -0.81* | -0.85* | -1.60* | -0.81* | -0.42* | 0.00 | 7.5 | 94.8 | 92.0 | 99.9 |
| `hf_T1.0_typ0.9` | 31.31 | -1.20* | -1.23* | -1.94* | -1.17* | -0.85* | 0.00 | 5.9 | 93.8 | 92.4 | 100.0 |
| `hf_T1.0_eps0.02` | 31.30 | -1.21* | -1.28* | -2.22* | -1.25* | -0.69* | 0.00 | 6.7 | 94.2 | 91.5 | 99.9 |

### 2.2 Is greedy "very repetitive" at sentence level?

Yes, in the weak languages; and the repetition is not what costs chrF++. The human references have a repeated word 3-gram in 2.7% of the sentences; the greedy output in **14.7%** (13.2% not present in the reference). By language (greedy, repeated word 3-gram not in the reference): kbp 52%, nus 52%, ach 36%, fon 33%, yor 17.5%, efi 17%, wol 9%, kab 8.5%, bem 7.5%, amh 5.5%, tir 5%, som 5%; the strong languages 0-3% (zul 0%, swh 2.5%, hau 2.4%). Sampling removes it (T 0.7: weak group 23.5% -> 12.6%, T 1.0: 6.9%) but costs 0.44 / 1.59 chrF++ in that group; cut-offs keep part of the repetition (T 1.0 + min-p 0.05: 11.0%). The hard failure mode (unfinished / looping, the `loop` column) is 0.10% under greedy, so the repetition is the mild kind (repeated function words and phrases), not runaway loops. Consensus selection does not remove it either (beam search was not re-measured: R5's beam hypotheses are no longer on the box; R5 found beam 4 loops more than greedy, 0.6% against 0.1%, mostly in Nuer and Kabiye):

| config | delta chrF++ | repeated 3-gram, all % | strong % | medium % | weak % |
|---|---|---|---|---|---|
| `greedy` | +0.00 | 13.1 | 2.6 | 3.0 | 23.5 |
| `T0.2` | -0.05 | 12.6 | 2.9 | 3.3 | 22.0 |
| `T0.3` | -0.13* | 11.5 | 2.6 | 3.1 | 20.1 |
| `T0.2_mp0.05` | +0.02 | 12.2 | 3.0 | 3.2 | 21.3 |
| `T0.5` | -0.35* | 10.0 | 2.2 | 2.6 | 17.6 |
| `T0.7` | -0.79* | 7.5 | 2.3 | 2.5 | 12.6 |
| `T1.0` | -2.02* | 4.5 | 1.6 | 2.6 | 6.9 |
| `T1.0_p0.9` | -1.25* | 5.5 | 2.3 | 2.7 | 8.6 |
| `T1.0_mp0.05` | -1.15* | 6.7 | 2.6 | 2.2 | 11.0 |
| `T1.0_mp0.1` | -0.81* | 7.4 | 1.9 | 2.2 | 12.8 |
| `T0.7_mp0.05` | -0.42* | 9.2 | 2.0 | 2.8 | 16.0 |
| `mbr8:T0.5_n16` | +0.60* | 12.0 | 3.1 | 2.9 | 21.1 |
| `mbr16:T0.5_n16` | +0.73* | 12.2 | 2.6 | 3.3 | 21.3 |
| `mbr16:T0.7_n16` | +0.81* | 10.0 | 2.8 | 2.6 | 17.2 |
| `mbr16:T0.7_mp0.05_n16` | +0.83* | 10.8 | 2.5 | 2.2 | 19.3 |
| `mbr16:T1.0_mp0.05_n16` | +0.69* | 8.5 | 1.8 | 2.4 | 14.9 |
| `mbrg16:T0.7_n16` | +0.73* | 11.3 | 2.6 | 3.1 | 19.7 |
| `mbrqe16:T0.7_n16` | +0.67* | 8.8 | 2.2 | 3.0 | 15.0 |
| `qe16:T0.7_n16` | -0.10 | 6.8 | 1.7 | 2.3 | 11.6 |
| `oracle16:T0.7_n16` | +4.93* | 7.1 | 1.8 | 2.7 | 12.1 |

(`no_repeat_ngram` and `repetition_penalty` remove the repetition by construction, but R5 showed that they hurt chrF++ and delete copied numbers.) I cannot say from these data whether the remaining repetitions are an error a reader would object to: the reference-based metrics do not penalise them (chrF++ is unchanged between greedy and T 0.2 / T 0.3 + cut-off while the repetition falls a little), and no human judgement was collected.

### 2.3 Several samples and a selection rule

Pools of 16 samples per sentence (n = 4, 8 are the first 4 / 8 of the 16), selection by **MBR with chrF++ utility** (the candidate with the highest mean sentence chrF++ against the other candidates; R5's rule), delta chrF++ against greedy:

| sampler of the pool | mbr n=4 | mbr n=8 | mbr n=16 |
|---|---|---|---|
| `T0.3` | +0.28* | +0.44* | +0.57* |
| `T0.5` | +0.29* | +0.60* | +0.73* |
| `T0.5_mp0.05` | +0.29* | +0.57* | +0.69* |
| `T0.7` | +0.19* | +0.49* | +0.81* |
| `T0.7_mp0.05` | +0.24* | +0.57* | +0.83* |
| `T0.7_p0.9` | +0.27* | +0.57* | +0.80* |
| `T1.0` | -0.70* | -0.20* | +0.21* |
| `T1.0_p0.95` | -0.26* | +0.21* | +0.59* |
| `T1.0_p0.9` | -0.09 | +0.29* | +0.74* |
| `T1.0_mp0.1` | +0.01 | +0.43* | +0.79* |
| `T1.0_mp0.05` | -0.06 | +0.34* | +0.69* |

* **MBR needs enough samples and a sampler that does not add noise.** With 16 samples the gain is **+0.7 ... +0.8 chrF++** for every sampler except plain T 1.0 (+0.21) and T 1.0 + top-p 0.95 (+0.59): T 0.7 + min-p 0.05 +0.83 [+0.72, +0.94], T 0.7 +0.81, T 0.7 + top-p 0.9 +0.80, T 1.0 + min-p 0.1 +0.79, T 0.5 +0.73. That equals R5's beam search 4 / 8 (+0.66 / +0.74 chrF++) but at 16x the tokens (beam 4 is about 4.6x the HF greedy time), and it **does not improve spBLEU** the way beam does (-0.2 ... +0.2 against beam 4's +0.69), because the utility is chrF++. With 8 samples +0.5 ... +0.6 (as in R5); with 4 samples about +0.2 ... +0.3 or nothing. The oracle (best candidate against the human reference) shows the headroom: +1.7 / +3.1 / +4.4 chrF++ for n = 4 / 8 / 16 at T 1.0 (+2.4 / +3.8 / +5.0 with min-p 0.05), so a better selector than chrF++ consensus could gain several points; none of the ones I tried does.
* **Which sampler for a pool?** For n = 16 the ranking by chrF++ is T 0.7 + min-p 0.05 (+0.83) ~ T 0.7 (+0.81) ~ T 0.7 + top-p 0.9 (+0.80) ~ T 1.0 + min-p 0.1 (+0.79) > T 0.5 (+0.73) ~ T 1.0 + top-p 0.9 (+0.74) > T 0.5 + min-p 0.05 (+0.69) ~ T 1.0 + min-p 0.05 (+0.69) > T 0.3 (+0.57); the intervals overlap. For n = 8: T 0.5 (+0.60) ~ T 0.7 + min-p 0.05 (+0.57) ~ T 0.7 + top-p 0.9 (+0.57) > T 0.7 (+0.49) > T 1.0 + min-p 0.1 (+0.43). **Diversity from a higher temperature pays only when n is large; for n <= 8 use T 0.5 - 0.7 (with min-p 0.05 as a guard).**
* **By group** (MBR n = 16 / 8):

| MBR (chrF++ consensus) | chrF++ | delta chrF++ | delta spBLEU | strong | medium | weak | tokens vs greedy |
|---|---|---|---|---|---|---|---|
| `T0.5` n=8 | 33.11 | +0.60* | +0.08 | +0.45* | +0.57* | +0.69* | 8x |
| `T0.5` n=16 | 33.25 | +0.73* | +0.16* | +0.53* | +0.70* | +0.85* | 16x |
| `T0.7` n=8 | 33.01 | +0.49* | -0.11 | +0.15 | +0.63* | +0.59* | 8x |
| `T0.7` n=16 | 33.33 | +0.81* | +0.10 | +0.47* | +1.01* | +0.88* | 16x |
| `T0.7_mp0.05` n=8 | 33.09 | +0.57* | +0.01 | +0.38* | +0.57* | +0.67* | 8x |
| `T0.7_mp0.05` n=16 | 33.35 | +0.83* | +0.22* | +0.69* | +0.79* | +0.92* | 16x |
| `T1.0_mp0.05` n=8 | 32.86 | +0.34* | -0.42* | -0.01 | +0.24 | +0.57* | 8x |
| `T1.0_mp0.05` n=16 | 33.20 | +0.69* | -0.19* | +0.29 | +0.57* | +0.94* | 16x |
| `T1.0_mp0.1` n=8 | 32.95 | +0.43* | -0.21* | +0.04 | +0.39* | +0.65* | 8x |
| `T1.0_mp0.1` n=16 | 33.31 | +0.79* | +0.01 | +0.39* | +0.82* | +0.98* | 16x |

MBR helps every group, and **helps the weak and medium languages more than the strong ones at the same n** (n = 16 T 0.7: strong +0.47, medium +1.01, weak +0.88; Yoruba +0.7, Bemba +1.6, Kabiye (`kbp`) +1.6, Swahili +0.7, Hausa +0.3). Consensus is a worse selector for the languages where the model is already near the reference (hau, som, zul +0.3 ... +0.4).
* **Adding the greedy output to the pool** (R5's "MBR + greedy"): +0.55 (n = 8) / +0.73 (n = 16) at T 0.7 against +0.49 / +0.81 without; the greedy output is picked for 44-49% of the sentences. It helps when n is small and not when n is large; its spBLEU is a little better (+0.06 ... +0.25 against -0.11 ... +0.22), which is the only reason to include it.
* **Selection by AfriCOMET-QE is metric gaming, not a gain.** Picking the best of 16 by reference-free AfriCOMET-QE raises AfriCOMET (with reference) from 0.623 to 0.668 (+0.045), more than any other config in this study, but **chrF++ is below greedy (-0.10 [T 0.7 pool] and -0.41 [T 1.0 + min-p 0.05 pool])** and spBLEU -0.37 / -0.59: the QE metric and the reference-based metric share an encoder and a training set, so the QE selector and the AfriCOMET judge agree with each other and not with the n-gram metrics. A 50/50 z-score mixture of MBR and QE (`mbrqe`) is +0.67 chrF++ / +0.036 AfriCOMET at n = 16 (T 0.7): a little less chrF++ than plain MBR (+0.81) for a large AfriCOMET gain; I would not use it without a human check because I cannot tell which of the two metrics is right. The oracle selection (best chrF++ against the human reference among 8) gains +3.8 chrF++ but only +0.012 AfriCOMET, which shows how far the two metrics are from each other in what they reward.

| selection (pool, n) | chrF++ | delta chrF++ | delta spBLEU | numbers kept % | AfriCOMET ref (delta) | AfriCOMET-QE (delta) |
|---|---|---|---|---|---|---|
| `mbr16:T0.7_n16` | 33.33 | +0.81* | +0.10 | 96.9 | 0.6288 (+0.0058) | 0.5736 (+0.0041) |
| `mbrg16:T0.7_n16` | 33.24 | +0.73* | +0.16* | 96.9 | - | - |
| `mbrqe16__T0.7_n16` | 33.19 | +0.67* | +0.04 | 96.6 | 0.6586 (+0.0356) | 0.6192 (+0.0497) |
| `qe16__T0.7_n16` | 32.42 | -0.10 | -0.37* | 96.3 | 0.6681 (+0.0451) | 0.6384 (+0.0689) |
| `mbr16:T1.0_mp0.05_n16` | 33.20 | +0.69* | -0.19* | 96.4 | 0.6271 (+0.0041) | 0.5723 (+0.0028) |
| `mbrg16:T1.0_mp0.05_n16` | 33.28 | +0.76* | +0.10 | 96.6 | - | - |
| `mbrqe16__T1.0_mp0.05_n16` | 33.07 | +0.55* | -0.03 | 97.0 | 0.6581 (+0.0351) | 0.6189 (+0.0494) |
| `qe16__T1.0_mp0.05_n16` | 32.10 | -0.41* | -0.59* | 96.9 | 0.6670 (+0.0440) | 0.6385 (+0.0691) |
| `mbr8:T0.5_n16` | 33.11 | +0.60* | +0.08 | 97.0 | 0.6282 (+0.0052) | 0.5735 (+0.0041) |
| `mbrg8:T0.5_n16` | 33.05 | +0.54* | +0.11* | 96.6 | - | - |
| `mbrqe8__T0.5_n16` | 33.07 | +0.55* | +0.12 | 96.8 | 0.6529 (+0.0298) | 0.6096 (+0.0401) |
| `qe8__T0.5_n16` | 32.65 | +0.14* | -0.01 | 96.3 | 0.6624 (+0.0394) | 0.6260 (+0.0566) |

| pool | oracle n=4 | oracle n=8 | oracle n=16 |
|---|---|---|---|
| `T0.7` | +2.53* | +3.80* | +4.93* |
| `T1.0_mp0.05` | +2.37* | +3.79* | +4.96* |
| `T1.0` | +1.72* | +3.13* | +4.38* |

### 2.4 AfriCOMET (18 validated languages)

| config | AfriCOMET ref | delta [95% CI] | AfriCOMET-QE | delta QE | delta chrF++ |
|---|---|---|---|---|---|
| `qe16:T0.7_n16` | 0.6681 | +0.0451 [+0.0432, +0.0470] | 0.6384 | +0.0689 | -0.10 |
| `qe16:T1.0_mp0.05_n16` | 0.6670 | +0.0440 [+0.0420, +0.0459] | 0.6385 | +0.0691 | -0.41* |
| `qe8:T0.5_n16` | 0.6624 | +0.0394 [+0.0377, +0.0412] | 0.6260 | +0.0566 | +0.14* |
| `qe8:T0.7_n16` | 0.6601 | +0.0371 [+0.0352, +0.0390] | 0.6250 | +0.0555 | -0.07 |
| `mbrqe16:T0.7_n16` | 0.6586 | +0.0356 [+0.0338, +0.0374] | 0.6192 | +0.0497 | +0.67* |
| `mbrqe16:T1.0_mp0.05_n16` | 0.6581 | +0.0351 [+0.0331, +0.0369] | 0.6189 | +0.0494 | +0.55* |
| `mbrqe8:T0.5_n16` | 0.6529 | +0.0298 [+0.0280, +0.0316] | 0.6096 | +0.0401 | +0.55* |
| `mbrqe8:T0.7_n16` | 0.6515 | +0.0285 [+0.0267, +0.0304] | 0.6083 | +0.0389 | +0.49* |
| `oracle8:T0.7_n16` | 0.6354 | +0.0124 [+0.0103, +0.0144] | 0.5683 | -0.0012 | +3.80* |
| `mbr16:T0.5_n16` | 0.6312 | +0.0082 [+0.0064, +0.0100] | 0.5760 | +0.0066 | +0.73* |
| `mbr16:T0.7_mp0.05_n16` | 0.6301 | +0.0071 [+0.0052, +0.0088] | 0.5760 | +0.0066 | +0.83* |
| `mbr16:T0.3_n16` | 0.6298 | +0.0067 [+0.0051, +0.0084] | 0.5757 | +0.0062 | +0.57* |
| `mbr16:T0.7_n16` | 0.6288 | +0.0058 [+0.0038, +0.0077] | 0.5736 | +0.0041 | +0.81* |
| `mbr8:T0.5_n16` | 0.6282 | +0.0052 [+0.0033, +0.0071] | 0.5735 | +0.0041 | +0.60* |
| `mbr8:T0.3_n16` | 0.6277 | +0.0047 [+0.0031, +0.0063] | 0.5739 | +0.0045 | +0.44* |
| `mbr16:T1.0_mp0.05_n16` | 0.6271 | +0.0041 [+0.0019, +0.0060] | 0.5723 | +0.0028 | +0.69* |
| `mbr16:T1.0_p0.9_n16` | 0.6261 | +0.0031 [+0.0010, +0.0051] | 0.5702 | +0.0007 | +0.74* |
| `mbr8:T0.7_n16` | 0.6257 | +0.0027 [+0.0006, +0.0047] | 0.5706 | +0.0011 | +0.49* |
| `T0.3` | 0.6238 | +0.0008 [-0.0009, +0.0025] | 0.5694 | -0.0000 | -0.13* |
| `T0.2_mp0.05` | 0.6235 | +0.0005 [-0.0012, +0.0022] | 0.5701 | +0.0007 | +0.02 |
| `T0.4_p0.8` | 0.6230 | +0.0000 [-0.0019, +0.0019] | 0.5690 | -0.0005 | -0.01 |
| `greedy` | 0.6230 | +0.0000 [+0.0000, +0.0000] | 0.5694 | +0.0000 | +0.00 |
| `mbr8:T1.0_mp0.05_n16` | 0.6218 | -0.0013 [-0.0036, +0.0009] | 0.5668 | -0.0027 | +0.34* |
| `T0.5` | 0.6215 | -0.0015 [-0.0035, +0.0005] | 0.5682 | -0.0013 | -0.35* |
| `mbr16:T1.0_p0.95_n16` | 0.6215 | -0.0016 [-0.0034, +0.0005] | 0.5663 | -0.0032 | +0.59* |
| `mbr4:T0.7_n16` | 0.6211 | -0.0019 [-0.0040, +0.0002] | 0.5664 | -0.0030 | +0.19* |
| `T0.6_p0.9` | 0.6211 | -0.0019 [-0.0041, +0.0002] | 0.5664 | -0.0031 | -0.35* |
| `mbr8:T1.0_p0.9_n16` | 0.6200 | -0.0030 [-0.0052, -0.0009] | 0.5648 | -0.0048 | +0.29* |
| `T1.0_mp0.1` | 0.6156 | -0.0074 [-0.0097, -0.0050] | 0.5607 | -0.0087 | -0.81* |
| `T0.7` | 0.6144 | -0.0086 [-0.0108, -0.0063] | 0.5602 | -0.0093 | -0.79* |
| `mbr16:T1.0_n16` | 0.6140 | -0.0091 [-0.0112, -0.0070] | 0.5599 | -0.0096 | +0.21* |
| `T1.0_mp0.05` | 0.6121 | -0.0109 [-0.0133, -0.0088] | 0.5571 | -0.0124 | -1.15* |
| `T1.0_p0.9` | 0.6091 | -0.0139 [-0.0164, -0.0115] | 0.5546 | -0.0148 | -1.25* |
| `hf_T1.0_typ0.9` | 0.6087 | -0.0143 [-0.0166, -0.0119] | 0.5544 | -0.0151 | -1.20* |
| `hf_T1.0_eps0.02` | 0.6086 | -0.0144 [-0.0166, -0.0119] | 0.5546 | -0.0148 | -1.21* |
| `T1.0_k64_p0.95` | 0.6041 | -0.0190 [-0.0215, -0.0165] | 0.5497 | -0.0197 | -1.40* |
| `T1.0_p0.95` | 0.6036 | -0.0195 [-0.0220, -0.0171] | 0.5499 | -0.0196 | -1.50* |
| `T1.0` | 0.5923 | -0.0308 [-0.0332, -0.0282] | 0.5386 | -0.0309 | -2.02* |

### 2.5 Per language (delta chrF++ against greedy; S / M / W = strong / medium / weak group)

| config | swh (S) | hau (S) | yor (W) | ibo (M) | zul (S) | amh (W) | kin (S) | som (S) | wol (W) | sna (M) | lug (M) | bem (M) | tir (W) | kbp (W) | nus (W) | kab (W) | fon (W) | ach (W) | nbl (M) | efi (W) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy (chrF++) | 57.8 | 45.6 | 26.9 | 37.7 | 49.0 | 19.1 | 42.1 | 40.5 | 29.3 | 38.8 | 35.3 | 34.3 | 13.4 | 22.3 | 21.1 | 26.9 | 21.1 | 25.9 | 37.4 | 25.8 |
| `T0.3` | -0.1 | -0.6 | +0.1 | -0.1 | -0.3 | -0.7 | -0.4 | -0.0 | +0.4 | -0.1 | -0.1 | +0.4 | -0.1 | +0.2 | -0.2 | -0.2 | +0.2 | -0.2 | -0.5 | -0.2 |
| `T0.2_mp0.05` | -0.3 | -0.4 | +0.1 | +0.0 | +0.1 | -0.1 | +0.1 | +0.1 | +0.3 | -0.2 | +0.0 | +0.3 | -0.0 | +0.0 | -0.1 | -0.1 | +0.2 | +0.2 | +0.2 | +0.2 |
| `T0.5` | -0.9 | -0.5 | -0.4 | -0.2 | -0.5 | -0.6 | -0.7 | -0.8 | +0.3 | -0.6 | -0.2 | +0.0 | -0.5 | +0.3 | +0.1 | -0.5 | -0.1 | +0.1 | -0.9 | -0.3 |
| `T0.7` | -1.1 | -1.2 | -0.3 | -1.1 | -1.8 | -1.5 | -1.6 | -1.2 | -0.3 | -1.2 | -0.6 | -0.5 | -0.8 | +0.4 | +0.3 | -0.8 | +0.1 | -0.7 | -1.1 | -0.8 |
| `T1.0` | -2.5 | -2.8 | -1.4 | -1.3 | -3.0 | -3.3 | -3.2 | -2.2 | -1.2 | -2.2 | -2.1 | -1.9 | -1.9 | -1.6 | -0.2 | -2.2 | -0.6 | -2.0 | -3.3 | -1.4 |
| `T1.0_p0.9` | -1.4 | -1.8 | -0.6 | -0.9 | -2.1 | -2.9 | -1.6 | -1.4 | -0.8 | -1.4 | -1.0 | -1.2 | -1.3 | -0.5 | +0.0 | -1.7 | -0.1 | -1.1 | -2.1 | -1.0 |
| `T1.0_mp0.05` | -1.6 | -2.1 | -0.8 | -0.7 | -2.1 | -2.0 | -1.8 | -1.5 | -0.6 | -1.4 | -1.3 | -1.2 | -0.9 | -0.2 | +0.1 | -1.2 | +0.1 | -0.7 | -2.2 | -0.9 |
| `mbr8:T0.7_n16` | +0.4 | -0.0 | +0.5 | +0.4 | +0.2 | +0.3 | +0.3 | -0.1 | +0.8 | +0.3 | +0.5 | +1.1 | +0.6 | +1.5 | +0.7 | -0.0 | +0.5 | +0.6 | +0.9 | +0.4 |
| `mbr16:T0.7_n16` | +0.7 | +0.3 | +0.7 | +0.7 | +0.3 | +0.8 | +0.6 | +0.4 | +1.1 | +0.9 | +0.8 | +1.6 | +1.1 | +1.6 | +0.7 | +0.3 | +0.7 | +0.9 | +1.1 | +1.0 |
| `mbr16:T0.5_n16` | +0.7 | +0.6 | +0.6 | +0.4 | +0.5 | +0.6 | +0.6 | +0.3 | +0.8 | +0.3 | +0.7 | +1.2 | +1.0 | +1.7 | +0.7 | +0.6 | +0.7 | +1.0 | +0.8 | +1.0 |
| `mbr16:T1.0_mp0.05_n16` | +0.4 | -0.1 | +0.6 | +0.2 | +0.9 | +0.7 | +0.5 | -0.3 | +1.1 | +0.5 | +0.4 | +1.4 | +0.9 | +1.6 | +0.8 | +0.8 | +0.8 | +0.9 | +0.3 | +1.2 |
| `mbrg16:T0.7_n16` | +0.6 | +0.6 | +0.5 | +0.5 | +0.4 | +1.1 | +0.4 | +0.4 | +0.9 | +0.7 | +0.5 | +1.1 | +0.7 | +1.6 | +0.7 | +0.3 | +0.6 | +1.0 | +1.1 | +0.9 |
| R5 `beam4` (HF) | +0.6 | +0.4 | +0.8 | +0.6 | +0.9 | +1.4 | +0.5 | +0.4 | +0.9 | +1.0 | +0.7 | +1.0 | +0.8 | +0.6 | -1.0 | +0.6 | +0.2 | +0.3 | +1.4 | +1.0 |
| R5 `beam8` (HF) | +0.7 | +0.4 | +0.8 | +0.8 | +1.3 | +1.4 | +0.9 | +0.3 | +1.0 | +1.2 | +0.8 | +1.6 | +1.0 | +0.5 | -1.2 | +0.5 | +0.3 | +0.1 | +1.7 | +0.9 |

For Yoruba (26.9 chrF++ greedy; weak group): T 0.3 +0.1, T 0.2 + min-p 0.05 +0.1 (ties), T 0.5 -0.4, T 0.7 -0.3, T 1.0 -1.4 (+top-p 0.9 -0.6, +min-p 0.05 -0.8), MBR n = 16 +0.5 ... +0.7 (every sampler), MBR + greedy +0.5, R5's beam 4 / 8 +0.8 / +0.8. Igbo (medium): T 1.0 -1.3 but T 0.5 only -0.2; MBR n = 16 +0.2 ... +0.7, beam 4 / 8 +0.6 / +0.8.

## 3. Document level (R6's documents, 8-language subset B8)

### 3.1 Whole-document sampling with the new samplers: more of the same

Core documents, chrF++ against the NLLB-3.3B reference (macro over 8 languages; in brackets the difference to whole-document greedy with the 95% paired bootstrap over documents, `*` = interval excludes 0). Same 1,278 documents for every row; R6's rows are re-aggregated on them. The last row is R6's best configuration, `sentfix` (sentence-by-sentence, greedy, unfinished sentences re-decoded by best-of-4 at T 0.7).

| config | b256 | b512 | b1024 | b2048 |
|---|---|---|---|---|
| greedy, whole document | 47.3 | 47.7 | 29.1 | 17.9 |
| `T0.5_mp0.05` | 47.2 (-0.1) | 47.5 (-0.2) | 32.1 (+3.0*) | 22.0 (+4.1*) |
| `T0.7_mp0.05` | 46.3 (-0.9*) | 47.7 (+0.0) | 34.2 (+5.1*) | 24.5 (+6.6*) |
| `T0.7_mp0.05_mt1.4` | 46.5 (-0.8*) | 47.7 (+0.0) | 34.8 (+5.7*) | 25.8 (+7.9*) |
| `T1.0_k64_p0.95` | 45.2 (-2.0*) | 46.7 (-1.0*) | 35.6 (+6.5*) | 27.3 (+9.4*) |
| `T1.0_mp0.05` | 45.7 (-1.6*) | 47.3 (-0.4*) | 35.6 (+6.5*) | 27.3 (+9.5*) |
| `T1.0_mp0.05_mt1.4` | 45.8 (-1.4*) | 47.1 (-0.6*) | 37.4 (+8.3*) | 30.0 (+12.2*) |
| `T1.0_mp0.1` | 46.1 (-1.1*) | 47.5 (-0.2) | 35.0 (+5.9*) | 26.1 (+8.2*) |
| T 0.3 (R6) | 47.1 (-0.2) | 47.4 (-0.3) | 31.2 (+2.1*) | 20.3 (+2.4*) |
| T 0.5 (R6) | 46.9 (-0.4*) | 47.9 (+0.2) | 33.5 (+4.4*) | 23.5 (+5.7*) |
| T 0.7 (R6) | 46.4 (-0.8*) | 47.6 (-0.1) | 35.3 (+6.2*) | 27.1 (+9.2*) |
| T 1.0 (R6) | 44.3 (-2.9*) | 45.1 (-2.6*) | 33.9 (+4.8*) | 25.8 (+7.9*) |
| best-of-4 T 0.7, chrF++ consensus (R6) | 46.9 (-0.4*) | 48.3 (+0.6*) | 36.5 (+7.4*) | 27.1 (+9.2*) |
| sentence-by-sentence, greedy + R6 repair (best-of-4 T 0.7) | 48.8 (+1.5*) | 53.2 (+5.5*) | 55.6 (+26.5*) | 58.7 (+40.8*) |

Failure metrics (finished = end token generated, loop = repeated word cycle the reference does not have, cov = sentence coverage, len = median hypothesis/reference length):

| config | b512 fin % | b512 loop % | b512 cov % | b512 len | b1024 fin % | b1024 loop % | b1024 cov % | b1024 len | b2048 fin % | b2048 loop % | b2048 cov % | b2048 len |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy, whole document | 93 | 12 | 99 | 0.99 | 64 | 52 | 95 | 1.04 | 36 | 79 | 96 | 1.22 |
| `T0.5_mp0.05` | 97 | 5 | 98 | 1.00 | 88 | 27 | 94 | 0.83 | 69 | 64 | 94 | 0.75 |
| `T0.7_mp0.05` | 98 | 0 | 98 | 0.99 | 99 | 3 | 93 | 0.79 | 89 | 34 | 93 | 0.60 |
| `T0.7_mp0.05_mt1.4` | 99 | 2 | 98 | 1.00 | 95 | 14 | 98 | 0.90 | 79 | 43 | 98 | 0.84 |
| `T1.0_k64_p0.95` | 100 | 0 | 98 | 1.00 | 100 | 0 | 95 | 0.78 | 99 | 1 | 90 | 0.53 |
| `T1.0_mp0.05` | 100 | 0 | 98 | 0.99 | 100 | 0 | 94 | 0.78 | 98 | 4 | 92 | 0.56 |
| `T1.0_mp0.05_mt1.4` | 100 | 1 | 98 | 1.01 | 100 | 3 | 98 | 0.88 | 91 | 12 | 99 | 0.75 |
| `T1.0_mp0.1` | 100 | 0 | 98 | 0.99 | 98 | 0 | 93 | 0.77 | 91 | 13 | 91 | 0.56 |
| T 0.3 (R6) | 95 | 9 | 98 | 0.99 | 82 | 39 | 95 | 0.86 | 56 | 69 | 94 | 0.86 |
| T 0.5 (R6) | 98 | 3 | 98 | 0.99 | 96 | 14 | 95 | 0.81 | 85 | 41 | 92 | 0.61 |
| T 0.7 (R6) | 100 | 0 | 98 | 0.99 | 100 | 1 | 94 | 0.77 | 95 | 6 | 91 | 0.57 |
| T 1.0 (R6) | 100 | 0 | 98 | 1.01 | 100 | 0 | 94 | 0.78 | 99 | 0 | 91 | 0.57 |
| best-of-4 T 0.7, chrF++ consensus (R6) | 100 | 0 | 98 | 1.00 | 100 | 0 | 94 | 0.79 | 99 | 6 | 91 | 0.56 |
| sentence-by-sentence, greedy + R6 repair (best-of-4 T 0.7) | 99 | 0 | 100 | 1.01 | 98 | 2 | 100 | 1.01 | 98 | 2 | 100 | 1.01 |

* **No sampler fixes whole-document translation beyond ~512 tokens.** Greedy loops (b1024 52% of the documents, b2048 79%) and does not finish (64% / 36%). Every sampler removes the loops and the unfinished documents (T 1.0 + min-p 0.05 finishes 100% / 98%, Gemma default 100% / 99%) but the output stays 22% (b1024) and 44-47% (b2048) shorter than the reference: **the model stops early or skips text** (median length ratio 0.77-0.79 / 0.53-0.60 in every sampled row without `min_tokens`), so chrF++ is 34-36 (b1024) and 24-28 (b2048) against 55.6 / 58.7 for sentence-by-sentence. The best whole-document rows are T 1.0 + min-p 0.05 (35.6 / 27.3), Gemma default (35.6 / 27.3), R6's best-of-4 (36.5 / 27.1): all +6 ... +9 over greedy, all -20 ... -31 below `sentfix`. **Truncation to a nucleus (top-p, min-p, top-k) does not repair the early stop**: T 0.7 + min-p 0.05 is *worse* than R6's plain T 0.7 at b2048 (24.5 against 27.1, loop 34% against 6%) because cutting the tail makes the sampler greedier and brings the loops back; T 0.5 + min-p 0.05 is between greedy and T 0.7 (22.0 at b2048, loop 64%).
* **Short documents (b256 / b512): greedy is as good as any sampler** (b256 -0.9 ... -2.0 for T 0.7 + min-p 0.05 / T 1.0 + cut-offs, b512 -0.4 ... -1.0 or tied), as at sentence level.
* **Human references** (SmolDoc clean documents, 6 languages; `st1024` has 20 documents for 4 of the languages and 1-2 for the other two, `st2048` 15 documents in total, so these two columns are indicative only):

| config | sm (<=680 tok) | st512 | st1024 | st2048 |
|---|---|---|---|---|
| greedy, whole document | 45.7 | 47.0 | 29.0 | 18.2 |
| `T0.5_mp0.05` | 45.6 (-0.0) | 46.7 (-0.3) | 32.0 (+3.0*) | 20.1 (+1.9) |
| `T0.7_mp0.05` | 45.3 (-0.4*) | 46.4 (-0.6*) | 33.3 (+4.3*) | 24.0 (+5.8*) |
| `T0.7_mp0.05_mt1.4` | 45.5 (-0.2) | 46.3 (-0.7*) | 34.6 (+5.6*) | 25.8 (+7.6*) |
| `T1.0_k64_p0.95` | 44.2 (-1.5*) | 45.4 (-1.6*) | 34.4 (+5.4*) | 27.6 (+9.5*) |
| `T1.0_mp0.05` | 45.2 (-0.5*) | 45.8 (-1.2*) | 33.3 (+4.3*) | 24.2 (+6.0*) |
| `T1.0_mp0.05_mt1.4` | 45.0 (-0.7*) | 46.0 (-1.1*) | 36.6 (+7.6*) | 30.9 (+12.7*) |
| `T1.0_mp0.1` | 45.2 (-0.5*) | 45.9 (-1.1*) | 33.7 (+4.7*) | 26.6 (+8.4*) |
| T 0.3 (R6) | 45.4 (-0.3) | 46.7 (-0.3*) | 28.9 (-0.1) | 19.8 (+1.6) |
| T 0.5 (R6) | 45.5 (-0.2) | 46.6 (-0.4*) | 31.6 (+2.6*) | 22.5 (+4.3*) |
| T 0.7 (R6) | 45.3 (-0.4*) | 46.1 (-0.9*) | 33.7 (+4.7*) | 25.4 (+7.2*) |
| T 1.0 (R6) | 43.7 (-2.0*) | 44.9 (-2.1*) | 33.9 (+4.9*) | 26.3 (+8.1*) |
| best-of-4 T 0.7, chrF++ consensus (R6) | 45.4 (-0.3) | 46.5 (-0.5*) | 33.3 (+4.3*) | 25.0 (+6.8*) |
| sentence-by-sentence, greedy + R6 repair (best-of-4 T 0.7) | 47.0 (+1.3*) | 49.3 (+2.3*) | 50.6 (+21.7*) | 52.4 (+34.2*) |

  The ranking is the same as with the pseudo-reference: samplers do +4 ... +9 over greedy on the stitched long documents and nothing on short ones; sentence-by-sentence is +22 (st1024) and +34 (st2048) over greedy.
* **min_tokens** (suppress the end token until a fraction of the expected length) is the one decoding-side cure for the early stop that I tried; see 3.3 (`T1.0_mp0.05_mt1.4` row: the best whole-document row, 37.4 / 30.0 at b1024 / b2048, still 18 / 29 points below sentence-by-sentence).

### 3.2 Repairing the unfinished sentence units (sentfix) with other samplers

In sentence-by-sentence translation 0.6% of the units (636 of the units of the B8 documents; 390 of them in the documents selected here) hit the token limit under greedy (looped or truncated); R6 re-decodes them with best-of-4 at T 0.7 and keeps the candidate with the highest chrF++ consensus among the finished ones. Unit level (the 636 units; chrF++ against the NLLB sentence for the 571 units that are whole sentences of a core document; "finished" = share of units whose chosen output ends with the end token):

| repair of the unfinished units | units | re-decoded | finished % | loop or unfinished % | chrF++ vs NLLB (units with a reference) | len ratio | units with >= 1 finished candidate % | candidates finished % |
|---|---|---|---|---|---|---|---|---|
| greedy (unrepaired) | 636 |  | 0.0 | 100.0 | 35.0 | 1.23 | - | - |
| T0.3_mp0.05_n4 rule=cons | 636 | 636 | 79.6 | 22.0 | 35.2 | 1.05 | 80 | 50 |
| T0.3_mp0.05_n4 rule=first | 636 | 636 | 79.6 | 22.2 | 35.0 | 1.04 | 80 | 50 |
| T0.5_mp0.1_n4 rule=cons | 636 | 636 | 81.4 | 19.3 | 35.2 | 1.04 | 81 | 54 |
| T0.5_mp0.1_n4 rule=first | 636 | 636 | 81.4 | 19.8 | 34.9 | 1.04 | 81 | 54 |
| T0.5_n4 rule=cons | 636 | 636 | 82.9 | 17.8 | 34.9 | 1.02 | 83 | 57 |
| T0.5_n4 rule=first | 636 | 636 | 82.9 | 17.6 | 34.4 | 1.03 | 83 | 57 |
| T0.7_mp0.05_n4 rule=cons | 636 | 636 | 84.1 | 16.0 | 34.6 | 1.02 | 84 | 57 |
| T0.7_mp0.05_n4 rule=first | 636 | 636 | 84.1 | 16.2 | 34.1 | 1.02 | 84 | 57 |
| T0.7_mp0.05_n8 rule=cons | 636 | 636 | 91.4 | 8.8 | 35.1 | 1.01 | 91 | 59 |
| T0.7_mp0.05_n8 rule=first | 636 | 636 | 91.4 | 9.6 | 33.9 | 1.01 | 91 | 59 |
| T0.7_n8 rule=cons | 636 | 636 | 92.1 | 7.9 | 35.0 | 1.01 | 92 | 60 |
| T0.7_n8 rule=first | 636 | 636 | 92.1 | 7.9 | 33.9 | 1.00 | 92 | 60 |
| T0.7_p0.9_n4 rule=cons | 636 | 636 | 83.5 | 16.8 | 34.7 | 1.02 | 83 | 58 |
| T0.7_p0.9_n4 rule=first | 636 | 636 | 83.5 | 17.0 | 34.2 | 1.02 | 83 | 58 |
| T0_rp1.05 rule=first | 636 | 636 | 55.2 | 45.6 | 32.1 | 1.11 | 55 | 55 |
| T0_rp1.1 rule=first | 636 | 636 | 48.4 | 52.4 | 25.7 | 1.17 | 48 | 48 |
| T1.0_mp0.05_n4 rule=cons | 636 | 636 | 86.5 | 13.5 | 33.8 | 1.01 | 86 | 61 |
| T1.0_mp0.05_n4 rule=first | 636 | 636 | 86.5 | 13.5 | 33.1 | 1.01 | 86 | 61 |
| T1.0_mp0.05_n8 rule=cons | 636 | 636 | 92.9 | 7.1 | 34.2 | 1.00 | 93 | 61 |
| T1.0_mp0.05_n8 rule=first | 636 | 636 | 92.9 | 7.1 | 33.1 | 1.01 | 93 | 61 |
| T1.0_mp0.1_n4 rule=cons | 636 | 636 | 86.2 | 14.0 | 33.9 | 1.02 | 86 | 60 |
| T1.0_mp0.1_n4 rule=first | 636 | 636 | 86.2 | 14.2 | 33.5 | 1.01 | 86 | 60 |
| T1.0_n4 rule=cons | 636 | 636 | 86.3 | 13.7 | 32.2 | 1.01 | 86 | 60 |
| T1.0_n4 rule=first | 636 | 636 | 86.3 | 13.7 | 31.4 | 1.01 | 86 | 60 |
| T1.0_p0.9_n4 rule=cons | 636 | 636 | 86.0 | 14.0 | 33.7 | 1.01 | 86 | 60 |
| T1.0_p0.9_n4 rule=first | 636 | 636 | 86.0 | 14.0 | 32.9 | 1.00 | 86 | 60 |
| t07x4 rule=cons | 636 | 636 | 85.2 | 14.8 | 34.3 | 1.01 | 85 | 58 |
| t07x4 rule=first | 636 | 636 | 85.2 | 14.8 | 33.8 | 1.01 | 85 | 58 |

* **Finished rate grows with n and with temperature, chrF++ does not move.** 4 candidates: 80% (T 0.3 + min-p 0.05) ... 82.9% (T 0.5) ... 85.2% (R6, T 0.7) ... 86.5% (T 1.0 + min-p 0.05); 8 candidates: **91.4 - 92.9%** (T 0.7: 92.1%, T 1.0 + min-p 0.05: 92.9%). chrF++ of the repaired units is 34-35 for every T <= 0.7 setting and 32-34 for T 1.0; the consensus rule is better than "first finished candidate" by 0.2 - 1.2 chrF++ (`rule=cons` against `rule=first`) at no cost. The unrepaired greedy output (truncated or looped) already has chrF++ 35.0 on these units, so chrF++ cannot show a gain from the repair; only the finished rate and the loop rate do.
* **A repetition penalty on the repair is a bad idea:** greedy + repetition_penalty 1.05 finishes 55% of the units and 1.1 only 48% (chrF++ 32.1 and 25.7); the retry is itself unfinished in 45-52% of the units.
* **Document level: nothing changes.** The sentence-by-sentence documents with each of the 14 repair variants (and with no repair at all) have identical chrF++ to one decimal in every bin (49.0 / 53.5 / 55.6 / 58.4 against NLLB, same against the human references); the only differences are the finished rate (finished, b512 / b1024 / b2048: no repair 94 / 92 / 86%; repaired 97-100 / 95-99 / 93-98%, the repetition-penalty repairs at the low end) and a 0-1 point loop rate. Because only 0.6% of the units are repaired, **the repair sampler is irrelevant for translation quality**; choose it for coverage. n = 8 at T 0.7 is the best at the unit level (92.1% finished, 7.9% unfinished) and costs nothing at 636 units per 100k.

| config | b256 | b512 | b1024 | b2048 |
|---|---|---|---|---|
| sentence-by-sentence, greedy + R6 repair (best-of-4 T 0.7) | 49.0 | 53.5 | 55.6 | 58.4 |
| sentence-by-sentence, greedy, no repair | 49.0 (-0.0) | 53.5 (-0.0) | 55.6 (-0.0) | 58.4 (-0.0) |
| R6 repair re-assembled (best-of-4 T 0.7) | 49.0 (+0.0) | 53.5 (+0.0) | 55.6 (+0.0) | 58.4 (+0.0) |
| `fix_T0.5_n4` | 49.0 (+0.0) | 53.5 (-0.0) | 55.6 (-0.0) | 58.4 (-0.0) |
| `fix_T0.7_mp0.05_n8` | 49.0 (-0.0) | 53.5 (-0.0) | 55.6 (+0.0) | 58.4 (-0.0) |
| `fix_T0.7_n8` | 49.0 (-0.0) | 53.5 (+0.0) | 55.6 (+0.0) | 58.4 (-0.0) |
| `fix_T0_rp1.1` | 49.0 (-0.0) | 53.5 (-0.0*) | 55.6 (-0.0*) | 58.4 (-0.0*) |
| `fix_T1.0_mp0.05_n4` | 49.0 (-0.0) | 53.5 (-0.0) | 55.6 (+0.0) | 58.4 (+0.0) |
| `fix_T1.0_mp0.05_n8` | 49.0 (-0.0) | 53.5 (-0.0) | 55.6 (+0.0) | 58.4 (-0.0) |
| `fix_T1.0_n4` | 49.0 (-0.0*) | 53.5 (-0.0*) | 55.6 (-0.0) | 58.4 (+0.0) |
| greedy, whole document | 47.7 (-1.4*) | 47.7 (-5.8*) | 29.7 (-25.9*) | 18.2 (-40.2*) |

Failure metrics:

| config | b512 fin % | b512 loop % | b512 cov % | b512 len | b1024 fin % | b1024 loop % | b1024 cov % | b1024 len | b2048 fin % | b2048 loop % | b2048 cov % | b2048 len |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sentence-by-sentence, greedy + R6 repair (best-of-4 T 0.7) | 99 | 1 | 100 | 1.01 | 98 | 2 | 100 | 1.01 | 97 | 3 | 100 | 1.01 |
| sentence-by-sentence, greedy, no repair | 94 | 1 | 100 | 1.01 | 92 | 2 | 100 | 1.01 | 86 | 4 | 100 | 1.01 |
| R6 repair re-assembled (best-of-4 T 0.7) | 99 | 1 | 100 | 1.01 | 98 | 2 | 100 | 1.01 | 97 | 3 | 100 | 1.01 |
| `fix_T0.5_n4` | 98 | 1 | 100 | 1.01 | 98 | 2 | 100 | 1.01 | 97 | 3 | 100 | 1.01 |
| `fix_T0.7_mp0.05_n8` | 99 | 1 | 100 | 1.01 | 99 | 2 | 100 | 1.01 | 98 | 3 | 100 | 1.01 |
| `fix_T0.7_n8` | 99 | 1 | 100 | 1.01 | 99 | 2 | 100 | 1.01 | 98 | 3 | 100 | 1.01 |
| `fix_T0_rp1.1` | 97 | 1 | 100 | 1.01 | 95 | 2 | 100 | 1.01 | 93 | 3 | 100 | 1.01 |
| `fix_T1.0_mp0.05_n4` | 99 | 1 | 100 | 1.01 | 99 | 2 | 100 | 1.01 | 97 | 3 | 100 | 1.01 |
| `fix_T1.0_mp0.05_n8` | 100 | 1 | 100 | 1.01 | 99 | 2 | 100 | 1.01 | 98 | 3 | 100 | 1.01 |
| `fix_T1.0_n4` | 99 | 1 | 100 | 1.01 | 99 | 2 | 100 | 1.01 | 97 | 3 | 100 | 1.01 |
| greedy, whole document | 92 | 13 | 98 | 1.00 | 65 | 53 | 95 | 1.03 | 42 | 73 | 95 | 1.19 |

Human references:

| config | sm (<=680 tok) | st512 | st1024 | st2048 |
|---|---|---|---|---|
| sentence-by-sentence, greedy + R6 repair (best-of-4 T 0.7) | 47.2 | 49.1 | 50.6 | 52.4 |
| sentence-by-sentence, greedy, no repair | 47.2 (-0.0) | 49.1 (+0.0) | 50.6 (-0.0) | 52.4 (-0.0) |
| R6 repair re-assembled (best-of-4 T 0.7) | 47.2 (+0.0) | 49.1 (+0.0) | 50.6 (+0.0) | 52.4 (+0.0) |
| `fix_T0.5_n4` | 47.2 (-0.0) | 49.1 (+0.0) | 50.6 (-0.0) | 52.4 (-0.0*) |
| `fix_T0.7_mp0.05_n8` | 47.2 (-0.0) | 49.1 (-0.0) | 50.6 (-0.0) | 52.4 (-0.0) |
| `fix_T0.7_n8` | 47.2 (-0.0) | 49.1 (-0.0) | 50.6 (-0.0) | 52.4 (-0.0) |
| `fix_T0_rp1.1` | 47.2 (-0.0*) | 49.1 (-0.0) | 50.6 (+0.0) | 52.4 (-0.0*) |
| `fix_T1.0_mp0.05_n4` | 47.2 (-0.0*) | 49.1 (-0.0) | 50.6 (-0.0) | 52.4 (-0.0*) |
| `fix_T1.0_mp0.05_n8` | 47.2 (-0.0*) | 49.1 (-0.0) | 50.6 (-0.0) | 52.4 (-0.0) |
| `fix_T1.0_n4` | 47.2 (-0.0) | 49.1 (-0.0*) | 50.6 (-0.0) | 52.4 (-0.0*) |
| greedy, whole document | 46.0 (-1.2*) | 46.8 (-2.3*) | 29.0 (-21.7*) | 18.2 (-34.2*) |

### 3.3 min_tokens, and what was not run

`min_tokens` = 1.4 x the source token count (vLLM `SamplingParams(min_tokens=...)` suppresses the stop ids until that many tokens exist; a complete translation of these documents has about 2.0 generated tokens per source token, so 1.4 only forbids ending before 70% of the expected length). Two configurations, same 1,278 documents:

| config | b1024 chrF++ (d vs greedy) | b1024 fin / loop / cov / len | b2048 chrF++ (d vs greedy) | b2048 fin / loop / cov / len |
|---|---|---|---|---|
| T 0.7 + min-p 0.05 | 34.2 (+5.1*) | 99 / 3 / 93 / 0.79 | 24.5 (+6.6*) | 89 / 34 / 93 / 0.60 |
| T 0.7 + min-p 0.05 + min_tokens 1.4 | 34.8 (+5.7*) | 95 / 14 / 98 / 0.90 | 25.8 (+7.9*) | 79 / 43 / 98 / 0.84 |
| T 1.0 + min-p 0.05 | 35.6 (+6.5*) | 100 / 0 / 94 / 0.78 | 27.3 (+9.5*) | 98 / 4 / 92 / 0.56 |
| **T 1.0 + min-p 0.05 + min_tokens 1.4** | **37.4 (+8.3*)** | 100 / 3 / 98 / 0.88 | **30.0 (+12.2*)** | 91 / 12 / 99 / 0.75 |
| sentence-by-sentence + repair | 55.6 (+26.5*) | 98 / 2 / 100 / 1.01 | 58.7 (+40.8*) | 98 / 2 / 100 / 1.01 |

It does what it should (sentence coverage 93-94% -> 98-99%, length ratio 0.78 -> 0.88 at b1024 and 0.56 -> 0.75 at b2048, chrF++ +1.8 and +2.7 at T 1.0) but forcing the model to continue also brings loops back at the lower temperature (T 0.7: loop 14% / 43%) and leaves 12% loops and 9% unfinished at b2048 even at T 1.0. It is the best single-pass setting I found for documents that cannot be split (**T 1.0, min-p 0.05, `min_tokens = 1.4 x source tokens`**), and still 18-29 chrF++ below chunking. Not run: higher `min_tokens` ratios, greedy with `min_tokens` (loops), and combinations with a repetition-aware stop.

**Not run at document level:** sentence-by-sentence translation with a sampler / MBR / beam for *every* unit (the unit is a sentence, so the sentence-level result of section 2 applies unit by unit: beam 4 +0.66 and MBR-16 +0.8 chrF++ would be expected, not measured on documents; 8 samples x 37k units or HF beam on 100k units did not fit the 4 GB / nice 19 budget), the SFT model on documents, and a human check of the sentence-boundary effects of any of this.

## 4. RL rollouts: what T < 1, top-p or min-p would change (question 3)

**Measurement.** 280 prompts of the RL prompt pool (`rl_prompts_v3`, 20 languages = R5's set, 10 `b64` + 4 `b256` prompts per language; `b1024` was cut for time), 16 completions per prompt under each sampler (the trainer's group size), generated by vLLM from the final model with the trainer's budgets (384 / 1,280 new tokens), then scored with **the real reward code** (`rl_rtxpro_phase2c.yaml` reward + scorer blocks = what the policy was trained with, calibration v7, AfriCOMET-QE worker, GlotLID guard, group consensus with `group_size = 16`). Not the reward of the running science run (D-59: QE ensemble, anchors, lexicon): I did not re-measure with it; I expect the QE-mode ordering of the samplers to hold but have not tested it. 13 languages are in `qe` mode (reward = shaped AfriCOMET-QE), 5 in `blend` (kbp nus kab fon efi), 2 in `consensus` mode (ach nbl; reward = chrF++ agreement with the other 15 completions). Group-consensus rewards rise mechanically when the samples become more alike, so the **QE-mode column is the one to read as quality**.

| rollout setting | reward, 13 QE-mode languages | reward, 5 blend languages | reward, 2 consensus languages | reward, all (macro) | b64 | b256 | failed % (wrong language, degenerate, ...) | within-group std | mean best of group | flat groups % (std < 0.05) |
|---|---|---|---|---|---|---|---|---|---|---|
| `greedy` | 0.648 | 0.240 | -0.075 | 0.474 | 0.508 | 0.388 | 1.07 | - | - | - |
| `T1.0` | 0.445 | -0.182 | -0.244 | 0.220 | 0.241 | 0.167 | 0.22 | 0.289 | 0.693 | 0.7 |
| `T1.0+g` | 0.459 | -0.166 | -0.210 | 0.236 | 0.258 | 0.181 | 0.29 | 0.296 | 0.720 | 0.7 |
| `T1.0_p0.95` | 0.519 | -0.095 | -0.128 | 0.301 | 0.320 | 0.252 | 0.22 | 0.272 | 0.737 | 0.7 |
| `T1.0_p0.9` | 0.543 | -0.039 | -0.055 | 0.338 | 0.359 | 0.283 | 0.20 | 0.266 | 0.766 | 1.1 |
| `T1.0_p0.8` | 0.595 | 0.043 | 0.101 | 0.408 | 0.428 | 0.358 | 0.22 | 0.248 | 0.824 | 1.1 |
| `T1.0_mp0.05` | 0.557 | 0.026 | 0.028 | 0.371 | 0.389 | 0.326 | 0.22 | 0.267 | 0.807 | 1.4 |
| `T1.0_mp0.1` | 0.585 | 0.080 | 0.110 | 0.411 | 0.429 | 0.367 | 0.25 | 0.259 | 0.829 | 0.7 |
| `T1.0_k64_p0.95` | 0.516 | -0.079 | -0.105 | 0.305 | 0.324 | 0.258 | 0.27 | 0.273 | 0.753 | 0.7 |
| `T0.8` | 0.545 | 0.010 | 0.021 | 0.359 | 0.372 | 0.326 | 0.25 | 0.268 | 0.802 | 0.7 |
| `T0.7` | 0.575 | 0.088 | 0.158 | 0.411 | 0.427 | 0.372 | 0.29 | 0.259 | 0.823 | 1.4 |
| `T0.5` | 0.628 | 0.210 | 0.428 | 0.503 | 0.521 | 0.459 | 0.36 | 0.238 | 0.887 | 1.4 |
| `T0.3` | 0.654 | 0.338 | 0.749 | 0.585 | 0.610 | 0.522 | 0.78 | 0.214 | 0.935 | 2.9 |

(`T1.0+g` = 15 samples at T 1.0 + the greedy completion as 16th member, the D-59 recipe `greedy_in_group`; `greedy` alone is a group of one, so its blend / consensus numbers are rubric-only values and not comparable. All rollouts finish; the failures are `wrong_language` and `degenerate`.)

Reading, for the trainer's current setting (T 1.0, top-p 1, top-k 0) against the alternatives:

* **What T < 1 or a cut-off gains: a higher mean reward and a better best-of-group, not a better policy.** On the 13 QE languages the mean reward of a T 1.0 rollout is 0.445 and greedy's is 0.648; top-p 0.95 / 0.9 / 0.8 give 0.519 / 0.543 / 0.595, min-p 0.05 / 0.1 give 0.557 / 0.585, T 0.7 / 0.5 / 0.3 give 0.575 / 0.628 / 0.654. The mean of a group is subtracted before the update (advantages are `reward - group mean`, std-normalised), so the level does not matter; what the group provides is the ranking and the spread.
* **What it loses, 1: advantage signal.** The within-group std of the reward falls from 0.289 (T 1.0) to 0.272 (top-p 0.95), 0.266 (top-p 0.9), 0.267 (min-p 0.05), 0.259 (T 0.7 or min-p 0.1), 0.248 (top-p 0.8), 0.238 (T 0.5), 0.214 (T 0.3); with the std normalisation the update magnitude is the same but it is computed from smaller, noisier differences; the share of flat groups (std < 0.05) rises from 0.7% to 2.9% at T 0.3. The best sample of a group is better under sharper sampling (0.693 -> 0.766 top-p 0.9 -> 0.935 T 0.3), i.e. the policy would be shown "good" outputs more often and nearer to what it already does.
* **What it loses, 2: the failure modes come back.** At T 0.3 the `degenerate` rate (loops, near-loops) is 0.42% against 0.00% at T 1.0 (T 0.5: 0.11%), all in the weakest languages (kbp, nus: reward -2.0 for every looping sample); with T 1.0 the model basically never loops, which is why the sampled rollouts looked "healthy" in the long-document study while greedy decoding of the same model loops. A trainer that rolls out at low temperature stops punishing the looping that greedy users will see; the D-59 greedy group member is what keeps that visible.
* **What it loses, 3: unbiasedness.** The trainer's loss assumes the rollouts come from the policy it differentiates (ratio = 1, `allow_truncated_sampling: false`). The distance of the truncated or tempered sampler q from the policy p, exact per position, summed along the sampled prefixes (400 sequences per sampler):

| sampler q | sequences | KL(q, p) per token (nats) | KL(q, p) per sequence | % of T=1 sequences containing a token q can never emit | % of T=1 tokens outside q | reward of T=1 sequences inside q | reward of T=1 sequences with such a token |
|---|---|---|---|---|---|---|---|
| `T1.0` | 400 | 0.0000 | 0.00 | - | - | - | - |
| `T1.0_p0.95` | 400 | 0.0198 | 2.19 | 74.4 | 2.04 | 0.718 | 0.373 |
| `T1.0_p0.9` | 400 | 0.0355 | 3.92 | 88.2 | 3.55 | 0.854 | 0.409 |
| `T1.0_p0.8` | 400 | 0.0636 | 7.01 | 95.8 | 6.32 | 1.015 | 0.437 |
| `T1.0_mp0.05` | 400 | 0.0426 | 4.71 | 88.3 | 4.33 | 0.936 | 0.398 |
| `T1.0_mp0.1` | 400 | 0.0582 | 6.41 | 93.5 | 5.79 | 1.009 | 0.423 |
| `T1.0_k64_p0.95` | 400 | 0.0219 | 2.42 | 75.5 | 2.29 | 0.734 | 0.373 |
| `T0.8` | 400 | 0.0158 | 1.75 | - | - | - | - |
| `T0.7` | 400 | 0.0323 | 3.58 | - | - | - | - |
| `T0.5` | 400 | 0.0770 | 8.43 | - | - | - | - |
| `T0.3` | 400 | 0.1331 | 14.68 | - | - | - | - |

  top-p 0.95 is the mildest (0.020 nats per token, 2.2 nats per ~110-token sequence), top-p 0.8 / min-p 0.1 cost 0.06, T 0.5 0.08, T 0.3 0.13 nats per token. The middle columns show *where* the mismatch sits: with plain T 1.0 samples, **74% of the sequences contain at least one token that top-p 0.95 can never emit (88% for top-p 0.9 or min-p 0.05, 96% for top-p 0.8)**, and those sequences have a clearly lower reward (0.37 against 0.72 for top-p 0.95; 0.40 against 0.94 for min-p 0.05; the sequences differ in length and bin, so this is association, not a causal estimate). The tail tokens are where the T 1.0 errors are; a trainer that never samples them never gives them a negative advantage, and it also no longer sees the sequences that an unconstrained user of `model.generate` (or the Gemma default `top_k 64, top_p 0.95`) will produce. For a pure truncation the per-token importance ratio p/q = Z_t (the kept mass) is known exactly and bounded by 1, so a token-level correction would be cheap; I did not test one and the per-sequence product (e^-2 ... e^-7 here) is useless as a weight.
* **What it would change for the policy, in one sentence:** it shows the policy higher-reward groups that are nearer to what it already prefers and starves the tail; the cost is that the KL estimator and the gradient are biased and that low-probability errors are never penalised.

**Recommendation for RL: keep T 1.0, top-p 1.0, top-k 0 (disabled), min-p 0 for the sampled members, and keep the greedy member (D-59).** The greedy completion beats the mean of its 15 siblings by 0.19 raw reward (greedy 0.412 against 0.225 for the sample mean on the same 280 groups, 77% of the groups have the greedy member above the group mean), so it contributes a strong, correctly-signed advantage to the exploit side while the 15 T 1.0 samples keep the update unbiased and exploratory. If a sharper rollout is wanted anyway, **top-p 0.95 is the least harmful** (KL 0.02, +0.074 reward, -6% group std) and needs `allow_truncated_sampling: true` and a note that the estimator is biased; T < 1 and top-p <= 0.9 are not recommended.

## 5. The SFT-final model (sentence level)

Cheap to add (10 sampling configurations and a 16-sample pool, same sentences, `/workspace/jobs/a2/models/sft` = the SFT final, step 27,638; deltas against the SFT model's own greedy, 29.14 chrF++):

| SFT-final, sentence level | chrF++ | delta chrF++ | strong | medium | weak | spBLEU | finished % | loop % | repeated 3-gram % | numbers kept % |
|---|---|---|---|---|---|---|---|---|---|---|
| `greedy` | 29.14 | - | - | - | - | 10.23 | 90.2 | 10.9 | 23.1 | 87.3 |
| `T0.3` | 29.53 | +0.39* | -0.07 | -0.09 | +0.86* | 10.32 | 94.6 | 6.3 | 18.3 | 90.5 |
| `T0.5` | 29.27 | +0.13 | -1.31* | -0.97* | +1.40* | 9.94 | 98.3 | 2.2 | 13.5 | 91.7 |
| `T0.7` | 27.81 | -1.34* | -3.91* | -3.05* | +0.81* | 8.59 | 99.7 | 0.4 | 7.6 | 87.3 |
| `T0.7_mp0.05` | 28.99 | -0.15 | -1.88* | -1.49* | +1.38* | 9.61 | 98.9 | 1.4 | 11.7 | 89.6 |
| `T1.0_mp0.1` | 28.05 | -1.09* | -3.35* | -2.79* | +0.88* | 8.68 | 99.7 | 0.6 | 7.5 | 87.5 |
| `T1.0_mp0.05` | 27.25 | -1.89* | -4.96* | -3.61* | +0.50* | 8.09 | 99.9 | 0.2 | 5.4 | 85.4 |
| `T1.0_p0.9` | 25.46 | -3.68* | -7.83* | -5.58* | -0.65* | 6.73 | 100.0 | 0.0 | 3.1 | 80.3 |
| `T1.0` | 22.91 | -6.23* | -12.58* | -8.25* | -2.04* | 4.65 | 100.0 | 0.0 | 1.2 | 67.6 |
| `mbr8:T0.7_n16` | 30.91 | +1.77* | +0.41 | +0.62* | +3.03* | 10.29 | 100.0 | 0.0 | 9.5 | 95.1 |
| `mbr16:T0.7_n16` | 31.60 | +2.45* | +1.26* | +1.20* | +3.68* | 10.72 | 100.0 | 0.0 | 8.6 | 95.5 |

* **The SFT model is a different case: greedy is the problem.** 9.8% of its greedy outputs do not end (loop 10.9%, 23% have a repeated 3-gram, numbers kept 87%), against 0.1% for the RL model. A little sampling therefore *helps*: T 0.3 +0.39 [+0.14, +0.65] and T 0.5 +0.13 (tied), T 0.7 + min-p 0.05 -0.15 (tied) while T >= 0.7 without a cut-off, top-p 0.9 and T 1.0 are -1.3 ... -6.2; **and here the best sampler differs by group**: the weak group gains from T 0.3 - 0.7 (T 0.5 +1.40*, T 0.7 + min-p 0.05 +1.38*, T 0.3 +0.86*) while the strong and medium groups lose from T 0.5 on (-1.3 / -1.0 at T 0.5) and tie at T 0.3. This is the clearest evidence in the study that the owner's observation holds for the SFT model: greedy loops and repeats, and a low temperature (with min-p) repairs it for the languages where it loops.
* **MBR is much larger for the SFT model**: 16 samples at T 0.7 give **+2.45 chrF++ [+2.20, +2.71]** (weak +3.68, medium +1.20, strong +1.26), +0.5 spBLEU, 100% finished, numbers kept 95.5%; 8 samples +1.77. For the SFT checkpoint (and for any checkpoint whose greedy output loops) the right config is T 0.7 x 16 + MBR, or T 0.3 - 0.5 if one sample is all that is affordable. The RL training removed most of this gap: the RL model's greedy is +3.38 chrF++ above SFT greedy and +0.9 above SFT's best MBR.

## 6. Cost summary

| configuration | generated tokens vs greedy | wall clock vs greedy (same engine) | where |
|---|---|---|---|
| greedy | 1x | 1x (vLLM ~200 sentences/s, R5) | |
| one sample, any T / cut-off | 1x | 1.0 - 1.3x | |
| beam 4 (+ fallback) | ~4x (HF) | 4.6x HF greedy (R5) | HF |
| MBR n = 8 / 16 | 8x / 16x | ~10x / ~20x (R5: MBR8 10x including the CPU chrF++ matrix; n = 16: 120 pairs per sentence, 58 s on 10 cores for 3,970 sentences) | vLLM + CPU |
| sentence-by-sentence documents | 2.0 per source token (as whole-document greedy) | R6: faster than whole-document greedy in wall time | |
| repair best-of-8 of the 0.6% unfinished units | +8 x 0.6% of the units, up to 1,024 tokens each | negligible | |

## 7. Caveats

* **Metric.** chrF++ / spBLEU / AfriCOMET against one reference per sentence; the owner's concern ("very repetitive") is about readability, and the only evidence on it here is the repeated-3-gram statistic; no human judgement was collected. Selection by consensus (MBR) optimises chrF++ and gains little on spBLEU; selection by QE optimises AfriCOMET and loses chrF++; the oracle selection shows several chrF++ points of headroom that none of the selectors reach.
* **Sample sizes.** 200 sentences per language, equal language weight: differences below ~0.15 chrF++ (macro) are noise; per-language differences below ~0.5 are not resolved. `st1024` human documents: 20 per language for 4 languages (hau yor swh amh), 1-2 for the other two; `st2048`: 3-5 per language for 4 languages (15 in total): indicative only. Document-level sampling rows use a subset of 1,278 documents (see section 1), R6 rows are re-aggregated on it.
* **Not run (and why).** Beam search on the sentence units of long documents (HF-only, 100k units); MBR of sentence units at document level (the sentence-level result applies unit by unit; 8 samples x 37k units did not fit the 4 GB / nice 19 budget); the SFT model on documents; repetition-aware selection (MBR with a repetition penalty on the candidates) which would be the natural next experiment if the repetition matters to readers; RL rollouts for `b1024` and longer bins (generation of 16 x 3,584 tokens per prompt at 2k tokens/s) and with the live reward (QE ensemble, anchors, lexicon of D-55 - D-57); the rollout statistics are for the final model's *own* samples, not for the policy while it trains.
* **Engines.** vLLM and HF sampling agree (T 1.0: 30.50 vs 30.52 chrF++). vLLM applies min-p, top-k and top-p after the temperature; the bias numbers (section 4) rebuild this order in PyTorch with bf16 logits (the vLLM sample sometimes sits on the exact nucleus boundary, which is why I report the exact per-position KL instead of the log-ratio of the sampled tokens).
* **Housekeeping slip (R8's job).** One of my cleanup commands matched `comet_worker` by name and terminated the AfriCOMET worker of R8's `reward_run.py` (nice 19, `rw_abl_norep.jsonl`) at about 05:3x UTC on 2026-10-22. R8's driver relaunched the script within a minute (new process ids), so that run restarted from its beginning; I did not touch it again. Every kill after that was restricted to processes whose working directory is under `/workspace/jobs/r12`. R8 should know that one run of theirs was restarted.

## 8. Reproduction

Box scripts are in `reports/data/generation_config_study/scripts/` (sentence level: `gen_sent.py` -> `gen_hf.py` -> `eval_grid.py` -> `pool_eval.py` -> `qe_pools.py` / `qe_eval.py` -> `export_csv.py`; document level `doc/`: `mkreq.py`, `mkreq2.py`, `gen_doc.py`, `asm.py`, `score_items.py` (R6's, unchanged), `docagg.py`, `unit_repair.py`; RL `rl/`: `rl_prompts.py` -> `rl_gen.py` -> `rl_reward.py` -> `rl_bias.py` -> `rl_analyze.py`; the queue scripts `chain*.sh` document the order and the GPU limits). Results: `json/*.json`, `export/*.csv`. The tables of this report are rendered by `gemma_mt/scripts/build_generation_config_report.py` from the JSON.
