# Sentence-level decoding study for the final RL model (2026-10-21)

Task D-49 (3). Which decoding rule should the released `<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c` use for English -> African sentence translation? 20 languages x 200 sentences x 19 configurations, bootstrap CIs, plus AfriCOMET on the validated languages. Data and scripts: `reports/data/final_model_decoding_study/` (`results.json`, `qe_results.json`, `extra.json`, `study_*.py` = the box scripts in the order prep -> vLLM -> HF -> score -> QE -> extra); renderer `gemma_mt/scripts/build_decoding_study_report.py`. Companion: `reports/final_model_per_language_2026-10-21.md`.

## 0. Result and recommendation

* **Beam search wins on every aggregate**: macro chrF++ over the 20 languages greedy **32.52** -> beam 4 **33.18** (+0.66 [+0.55, +0.78]*) -> beam 8 **33.26** (+0.74 [+0.62, +0.86]*); spBLEU 12.15 -> 12.84 -> 12.88; AfriCOMET (reference) on the 18 validated languages 0.6230 -> 0.6338 (+0.0108), QE 0.5694 -> 0.5808. Brackets: 95% paired bootstrap of the delta to greedy (1,000 resamples of the sentences of every language, equal language weight).
* Beam 8 is only 0.08 chrF++ better than beam 4 and costs 1.2x more time per sentence (HF generate): **beam 4 is the knee**. Beam 2 gives +0.49 [+0.38, +0.60]* at 2.3x the HF greedy time.
* **Beam search loops where greedy does not**: loop rate (unfinished, repetitive or empty after decoding) greedy 0.1% -> beam 4 0.6% -> beam 8 1.0%, concentrated in Nuer (`nus`: greedy 0.5%, beam 4 8%, beam 8 11.5%) and Kabiye (`kbp` 0% -> 2.5% -> 4%); this is why beam 8 loses to greedy on `nus` (chrF++ 21.1 -> 19.9). A **greedy fallback** (keep the beam-4 output unless it is unfinished / repetitive / empty by `rl.guards`, then use the greedy one; 25 of 3,970 sentences fall back) gives chrF++ 33.20 (+0.68 [+0.57, +0.80]), better than plain beam 4 in the weak group (+0.04 [+0.02, +0.06] against plain beam 4).
* **MBR / consensus** (8 samples at T 0.7 or 0.5, pick the candidate with the highest mean chrF++ against the others) gives +0.5 chrF++ (33.03 at T 0.7, 33.06 at T 0.5), but **no BLEU gain** (+0.08 [-0.05, +0.21]): MBR on chrF++ selects what chrF++ likes. It costs about 10-15x the vLLM greedy time as measured (8 samples + 26 s of CPU chrF++ on 4 processes for 3,970 sentences) and is the best config in the weak group by a hair (chrF++ weak 23.81 vs beam 4 23.73, intervals overlap).
* **Things that hurt**: repetition_penalty 1.1 (-1.48 [-1.64, -1.33]* chrF++, spBLEU -2.56 [-2.76, -2.38]*) and `no_repeat_ngram_size` 3 (-1.06 [-1.18, -0.95]*) and 4 (-0.56 [-0.67, -0.46]*) are clearly worse than greedy; temperature sampling is monotonically worse (T 0.3 -0.13 [-0.23, -0.02]*, 0.5 -0.35 [-0.48, -0.22]*, 0.7 -0.79 [-0.93, -0.65]*). Repetition penalty 1.05 is neutral on chrF++ (-0.01 [-0.13, +0.12]) but lowers spBLEU (-0.52 [-0.67, -0.38]*) and number recall (below), and the model does not loop under greedy (0.1%), so there is nothing for it to fix.
* **Copied numbers and names** (section 3): `no_repeat_ngram_size 3` keeps only 20% of the source numbers that the human reference keeps (greedy 96%) because a number copied twice in a sentence is a repeated n-gram; repetition_penalty 1.1 over prompt + output keeps 77% of the numbers and 86% of the names (greedy 93%), because the source tokens are in the prompt and are penalised; restricting the penalty to generated tokens repairs most of it (89% numbers) but still costs 1.40 chrF++.

**Recommended default (quality): beam search with 4 beams, with a greedy fallback when the beam output is unfinished, repetitive or empty.** Expected gain over greedy on sentences: about +0.7 chrF++ and +0.7 spBLEU macro, +0.011 AfriCOMET; the gain is positive in every group (strong +0.57, medium +0.95, weak +0.56) and largest in the middle of the quality range (South Ndebele, Shona, Bemba, Igbo, Amharic).
**Recommended fast config: plain greedy, no penalty, no n-gram blocking** (vLLM, ~200 sentences/s on the shared RTX PRO 6000 at 0.08 GPU share, 12k tokens/s). If a consensus output is wanted (for example for pseudo-label generation), MBR over 8 samples at T 0.5 (+0.5 chrF++, no spBLEU gain) is the alternative; for plain translation serving it is dominated by beam 4.

## 1. Setup

* Model: RL-final `<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c` `latest`, sha `80baefff3b16b120b3e37e70afeb51f1fc248d96` (GRPO step 1433). Prompt `Translate the text to {language}:  {english}` in the chat-less format of `gemma_mt.format.build_prompt`; the language tag at the start of the completion is stripped before scoring.
* Languages (20, spread over the quality range): `swh`, `hau`, `yor`, `ibo`, `zul`, `amh`, `kin`, `som`, `wol`, `sna`, `lug`, `bem`, `tir`, `kbp`, `nus`, `kab`, `fon`, `ach`, `nbl`, `efi`. Test sentences: FLORES+ devtest for 17 languages and SmolSent for `ach`, `nbl`, `efi` (no FLORES+ code). **200 sentences per language, drawn at random (fixed seed) from the rows whose reference pair is not in the SFT text (`no_pair`)**; Hausa has only 170 such rows, so n = 3,970. The same sentences for every configuration (paired).
* Quality groups (fixed assignment in `groups.json`, made before the final-model runs): **strong** = swh hau zul kin som (greedy chrF++ here 40.5-57.8), **medium** = ibo sna lug bem nbl (34.3-38.8), **weak** = yor amh wol tir kbp nus kab fon ach efi (13.4-29.3). Group numbers are macro means over the languages of the group.
* Budget `max_new_tokens` 320 for every configuration. Engines: vLLM (greedy, sampling, repetition_penalty, 8-sample pools; one seed per sentence) and HF `generate` (beam, no_repeat_ngram, generated-token-only penalty, HF greedy cross-check), both bf16 on the same GPU slice. vLLM greedy and HF greedy agree: chrF++ 32.52 vs 32.49 (paired delta -0.03 [-0.11, +0.05]), so the two engines can be mixed in one table. HF beam search: default `length_penalty` 1.0, `early_stopping` False, batches of 80 / 40 / 20 sentences for 2 / 4 / 8 beams, left-padded.
* Metrics per (configuration, language): chrF++ (word order 2) and spBLEU (sacrebleu `flores200` tokenizer), NFC; macro over languages; bootstrap = 1,000 paired resamples of the sentences within each language, languages weighted equally; `p_better` in the JSON = share of resamples where the config beats greedy. `finished` = stop token inside 320 tokens. `loop` = not finished, or `rl.guards.is_repetitive` / `is_self_repetitive`, or empty. `LID` = GlotLID right-language (sibling-tolerant) share. **Source-anchor recall** = of the numbers and capitalised names in the English sentence that the human reference also contains, the share the hypothesis contains (numbers compared with and without thousands/decimal separators, names case-folded). The ceilings: of 1157 source numbers 1048 survive into the references, of 5705 capitalised tokens 2345 do (the rest are sentence-initial words, titles or translated names); recall is computed over the surviving ones only.
* Tokens/s: generated tokens (up to and including the stop token) over wall-clock of the whole run; **vLLM and HF numbers are not comparable with each other** (continuous batching vs static padded batches); use the `x HF greedy` / `x vLLM greedy` column for the cost of a rule on the same engine.

## 2. All configurations

Macro over 20 languages. Delta = config minus greedy with the 95% paired-bootstrap interval, `*` = interval excludes 0.

| config | chrF++ | delta chrF++ [95% CI] | spBLEU | delta spBLEU [95% CI] | AfriCOMET (18 langs) | finished % | loop % | LID % | numbers kept % | names kept % | tokens/s | cost vs same-engine greedy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy (vLLM) = reference | 32.52 | 0 | 12.15 | 0 | 0.6230 | 100.0 | 0.1 | 99.9 | 95.7 | 92.6 | 12201 (vllm) | 1.0x |
| greedy (HF) cross-check | 32.49 | -0.03 [-0.11, +0.05] | 12.13 | -0.02 [-0.11, +0.07] | 0.6229 | 100.0 | 0.2 | 99.9 | 95.3 | 92.9 | 1681 (hf) | 1.0x |
| greedy + repetition_penalty 1.05 (vLLM, prompt + output) | 32.51 | -0.01 [-0.13, +0.12] | 11.62 | -0.52 [-0.67, -0.38]* | 0.6316 | 100.0 | 0.0 | 99.9 | 89.8 | 91.1 | 11226 (vllm) | 1.1x |
| greedy + repetition_penalty 1.1 (vLLM, prompt + output) | 31.03 | -1.48 [-1.64, -1.33]* | 9.58 | -2.56 [-2.76, -2.38]* | 0.6116 | 100.0 | 0.0 | 99.9 | 77.5 | 86.1 | 11970 (vllm) | 1.1x |
| greedy + repetition penalty 1.05, generated tokens only (HF) | 32.33 | -0.19 [-0.30, -0.08]* | 11.77 | -0.37 [-0.51, -0.24]* | 0.6293 | 100.0 | 0.0 | 99.9 | 94.4 | 92.8 | 1983 (hf) | 0.8x |
| greedy + repetition penalty 1.1, generated tokens only (HF) | 31.11 | -1.40 [-1.54, -1.27]* | 10.55 | -1.59 [-1.76, -1.43]* | 0.6155 | 100.0 | 0.0 | 99.8 | 88.5 | 93.4 | 1803 (hf) | 0.9x |
| greedy + repetition_penalty 1.1 (HF, prompt + output) | 30.98 | -1.53 [-1.69, -1.38]* | 9.58 | -2.56 [-2.76, -2.37]* | 0.6098 | 100.0 | 0.0 | 99.9 | 77.9 | 86.0 | 1680 (hf) | 1.0x |
| greedy + no_repeat_ngram_size 3 (HF) | 31.45 | -1.06 [-1.18, -0.95]* | 10.34 | -1.81 [-1.98, -1.64]* | 0.6182 | 100.0 | 0.0 | 100.0 | 19.7 | 79.8 | 1533 (hf) | 1.1x |
| greedy + no_repeat_ngram_size 4 (HF) | 31.95 | -0.56 [-0.67, -0.46]* | 11.06 | -1.08 [-1.23, -0.94]* | 0.6215 | 100.0 | 0.0 | 100.0 | 48.5 | 86.5 | 1256 (hf) | 1.3x |
| sampling T 0.3 (vLLM, 1 sample) | 32.39 | -0.13 [-0.23, -0.02]* | 12.03 | -0.11 [-0.24, +0.02] | 0.6238 | 100.0 | 0.1 | 99.9 | 96.1 | 92.8 | 9230 (vllm) | 1.3x |
| sampling T 0.5 (vLLM, 1 sample) | 32.17 | -0.35 [-0.48, -0.22]* | 11.78 | -0.37 [-0.51, -0.21]* | 0.6215 | 100.0 | 0.0 | 99.9 | 96.2 | 92.2 | 10073 (vllm) | 1.2x |
| sampling T 0.7 (vLLM, 1 sample) | 31.73 | -0.79 [-0.93, -0.65]* | 11.36 | -0.78 [-0.95, -0.61]* | 0.6144 | 100.0 | 0.0 | 99.9 | 96.3 | 92.0 | 11954 (vllm) | 1.0x |
| beam search 2 (HF) | 33.00 | +0.49 [+0.38, +0.60]* | 12.62 | +0.48 [+0.36, +0.60]* | 0.6302 | 100.0 | 0.4 | 99.9 | 96.2 | 93.7 | 723 (hf) | 2.3x |
| beam search 4 (HF) | 33.18 | +0.66 [+0.55, +0.78]* | 12.84 | +0.69 [+0.56, +0.82]* | 0.6338 | 100.0 | 0.6 | 99.9 | 97.2 | 93.5 | 368 (hf) | 4.6x |
| beam search 8 (HF) | 33.26 | +0.74 [+0.62, +0.86]* | 12.88 | +0.74 [+0.61, +0.88]* | 0.6344 | 100.0 | 1.0 | 99.9 | 97.1 | 93.9 | 305 (hf) | 5.6x |
| MBR: 8 samples T 0.7, pick the candidate with the highest mean chrF++ to the other 7 | 33.03 | +0.51 [+0.39, +0.63]* | 12.05 | -0.09 [-0.23, +0.05] | 0.6249 | 100.0 | 0.0 | 99.9 | 96.6 | 93.7 | 9769 (vllm+cpu) | 10.0x |
| MBR: same pool + the greedy output (9 candidates) | 33.05 | +0.53 [+0.44, +0.61]* | 12.17 | +0.03 [-0.07, +0.14] | 0.6267 | 100.0 | 0.1 | 99.9 | 96.4 | 93.3 | 8677 (vllm+cpu) | 12.6x |
| MBR: 8 samples T 0.5 | 33.06 | +0.54 [+0.44, +0.65]* | 12.22 | +0.08 [-0.05, +0.21] | 0.6285 | 100.0 | 0.0 | 99.9 | 96.8 | 93.1 | 6236 (vllm+cpu) | 15.6x |
| MBR: 8 samples T 0.5 + greedy | 32.99 | +0.47 [+0.39, +0.55]* | 12.20 | +0.07 [-0.03, +0.16] | 0.6287 | 100.0 | 0.0 | 99.9 | 96.8 | 93.2 | 5993 (vllm+cpu) | 18.3x |

AfriCOMET = `masakhane/africomet-stl-1.1` with reference, 18 of the 20 languages (the AfriCOMET-validated ones; not `ach`, `nbl`); the QE (no reference) macro and the paired CIs are in `qe_results.json`. The MBR rows' cost includes 8 samples (vLLM) and the CPU chrF++ matrix.

## 3. Repetition penalty and n-gram blocking against copied names and numbers

The prompt contains the English sentence; names and numbers are copied into the translation. HF's and vLLM's `repetition_penalty` divide the logit of every token already present in the *prompt or output*, so it penalises copying the source. `no_repeat_ngram_size` forbids any repeated n-gram, including a repeated number or name.

| config | numbers kept % | names kept % | chrF++ | spBLEU | length ratio hyp/ref | tokens per sentence |
|---|---|---|---|---|---|---|
| greedy (vLLM) = reference | 95.7 | 92.6 | 32.52 | 12.15 | 1.077 | 59.9 |
| greedy + repetition_penalty 1.05 (vLLM, prompt + output) | 89.8 | 91.1 | 32.51 | 11.62 | 1.095 | 59.4 |
| greedy + repetition penalty 1.05, generated tokens only (HF) | 94.4 | 92.8 | 32.33 | 11.77 | 1.051 | 57.1 |
| greedy + repetition_penalty 1.1 (vLLM, prompt + output) | 77.5 | 86.1 | 31.03 | 9.58 | 1.183 | 62.0 |
| greedy + repetition penalty 1.1, generated tokens only (HF) | 88.5 | 93.4 | 31.11 | 10.55 | 1.056 | 56.3 |
| greedy + no_repeat_ngram_size 4 (HF) | 48.5 | 86.5 | 31.95 | 11.06 | 1.069 | 59.3 |
| greedy + no_repeat_ngram_size 3 (HF) | 19.7 | 79.8 | 31.45 | 10.34 | 1.068 | 59.3 |

Reading: (1) the effect on copied tokens is large and in the expected direction, the numbers are hit hardest (`nr3` 20%, `nr4` 49%, `rep1.1` 78%); (2) the generated-token-only penalty (`...out`) keeps the numbers and names but is still worse than greedy on chrF++, so the loss is not only from copied tokens (with the prompt+output penalty at 1.1 the hypotheses are about 10% longer relative to the reference: ratio 1.18 against 1.08 for greedy); (3) the greedy output of this model does not loop (0.1%), so the fixes have no failure to repair on this set. They remain relevant for hard cases (long documents, unseen languages), which this sentence study does not test.

## 4. Winner per quality group

### chrF++ (macro; delta to greedy with 95% CI)

| config | all (20) | strong (5) | medium (5) | weak (10) |
|---|---|---|---|---|
| greedy (vLLM) = reference | 32.52 (-) | 47.02 (-) | 36.71 (-) | 23.16 (-) |
| beam search 2 (HF) | 33.00 (+0.49 [+0.38, +0.60]*) | 47.51 (+0.49 [+0.28, +0.70]*) | 37.38 (+0.67 [+0.44, +0.92]*) | 23.56 (+0.39 [+0.26, +0.54]*) |
| beam search 4 (HF) | 33.18 (+0.66 [+0.55, +0.78]*) | 47.59 (+0.57 [+0.34, +0.80]*) | 37.66 (+0.95 [+0.72, +1.19]*) | 23.73 (+0.56 [+0.41, +0.72]*) |
| beam search 8 (HF) | 33.26 (+0.74 [+0.62, +0.86]*) | 47.75 (+0.73 [+0.50, +0.96]*) | 37.92 (+1.21 [+0.96, +1.45]*) | 23.68 (+0.51 [+0.36, +0.67]*) |
| MBR: 8 samples T 0.7, pick the candidate with the highest mean chrF++ to the other 7 | 33.03 (+0.51 [+0.39, +0.63]*) | 47.38 (+0.35 [+0.10, +0.62]*) | 37.13 (+0.42 [+0.16, +0.66]*) | 23.81 (+0.63 [+0.48, +0.79]*) |
| MBR: same pool + the greedy output (9 candidates) | 33.05 (+0.53 [+0.44, +0.61]*) | 47.41 (+0.39 [+0.20, +0.58]*) | 37.26 (+0.56 [+0.37, +0.74]*) | 23.75 (+0.59 [+0.47, +0.71]*) |
| MBR: 8 samples T 0.5 | 33.06 (+0.54 [+0.44, +0.65]*) | 47.40 (+0.38 [+0.15, +0.61]*) | 37.27 (+0.56 [+0.33, +0.78]*) | 23.78 (+0.61 [+0.47, +0.77]*) |
| greedy + repetition_penalty 1.05 (vLLM, prompt + output) | 32.51 (-0.01 [-0.13, +0.12]) | 46.63 (-0.40 [-0.67, -0.14]*) | 36.57 (-0.14 [-0.40, +0.11]) | 23.42 (+0.25 [+0.07, +0.42]*) |
| sampling T 0.3 (vLLM, 1 sample) | 32.39 (-0.13 [-0.23, -0.02]*) | 46.74 (-0.28 [-0.53, -0.03]*) | 36.62 (-0.09 [-0.33, +0.17]) | 23.10 (-0.07 [-0.21, +0.09]) |
| greedy + no_repeat_ngram_size 4 (HF) | 31.95 (-0.56 [-0.67, -0.46]*) | 46.09 (-0.94 [-1.15, -0.73]*) | 35.86 (-0.84 [-1.04, -0.65]*) | 22.93 (-0.24 [-0.38, -0.11]*) |

Best point estimate per group: all: `beam8` (33.26), strong: `beam8` (47.75), medium: `beam8` (37.92), weak: `mbr8_T0.7` (23.81).

### spBLEU (macro; delta to greedy with 95% CI)

| config | all (20) | strong (5) | medium (5) | weak (10) |
|---|---|---|---|---|
| greedy (vLLM) = reference | 12.15 (-) | 22.91 (-) | 12.61 (-) | 6.54 (-) |
| beam search 2 (HF) | 12.62 (+0.48 [+0.36, +0.60]*) | 23.54 (+0.65 [+0.37, +0.91]*) | 13.20 (+0.59 [+0.34, +0.85]*) | 6.87 (+0.33 [+0.18, +0.48]*) |
| beam search 4 (HF) | 12.84 (+0.69 [+0.56, +0.82]*) | 23.69 (+0.79 [+0.48, +1.10]*) | 13.62 (+1.00 [+0.73, +1.29]*) | 7.02 (+0.49 [+0.32, +0.65]*) |
| beam search 8 (HF) | 12.88 (+0.74 [+0.61, +0.88]*) | 23.83 (+0.94 [+0.62, +1.26]*) | 13.77 (+1.16 [+0.89, +1.44]*) | 6.97 (+0.43 [+0.26, +0.59]*) |
| MBR: 8 samples T 0.7, pick the candidate with the highest mean chrF++ to the other 7 | 12.05 (-0.09 [-0.23, +0.05]) | 22.93 (+0.05 [-0.28, +0.40]) | 12.59 (-0.02 [-0.30, +0.25]) | 6.34 (-0.19 [-0.36, -0.02]*) |
| MBR: same pool + the greedy output (9 candidates) | 12.17 (+0.03 [-0.07, +0.14]) | 23.05 (+0.16 [-0.06, +0.38]) | 12.77 (+0.16 [-0.04, +0.35]) | 6.44 (-0.09 [-0.21, +0.03]) |
| MBR: 8 samples T 0.5 | 12.22 (+0.08 [-0.05, +0.21]) | 23.05 (+0.17 [-0.12, +0.46]) | 12.78 (+0.18 [-0.09, +0.42]) | 6.51 (-0.02 [-0.17, +0.13]) |
| greedy + repetition_penalty 1.05 (vLLM, prompt + output) | 11.62 (-0.52 [-0.67, -0.38]*) | 22.18 (-0.72 [-1.10, -0.34]*) | 12.07 (-0.54 [-0.82, -0.24]*) | 6.11 (-0.42 [-0.61, -0.23]*) |
| sampling T 0.3 (vLLM, 1 sample) | 12.03 (-0.11 [-0.24, +0.02]) | 22.71 (-0.18 [-0.51, +0.14]) | 12.60 (-0.02 [-0.28, +0.26]) | 6.41 (-0.13 [-0.28, +0.03]) |
| greedy + no_repeat_ngram_size 4 (HF) | 11.06 (-1.08 [-1.23, -0.94]*) | 21.12 (-1.77 [-2.11, -1.45]*) | 11.24 (-1.37 [-1.70, -1.07]*) | 5.95 (-0.58 [-0.76, -0.41]*) |

Best point estimate per group: all: `beam8` (12.88), strong: `beam8` (23.83), medium: `beam8` (13.77), weak: `beam4` (7.02).

Beam 8 and beam 4 win the strong and medium groups; in the weak group the four best (MBR at T 0.7 23.81, MBR at T 0.5 23.78, MBR + greedy 23.75, beam 4 23.73, beam 4 with fallback 23.77) are inside each other's intervals, beam 8 loses ground (loops) and plain greedy is 0.6 behind. On spBLEU the beams gain +0.3 to +1.2 in every group and **MBR gains nothing** in any group.

### Beam search with greedy fallback

| config | sentences falling back | chrF++ all | delta vs greedy [CI] | delta vs the plain beam [CI] | chrF++ strong / medium / weak (delta vs greedy) |
|---|---|---|---|---|---|
| beam2 + greedy fallback | 18 | 33.01 | +0.50 [+0.39, +0.61] | +0.01 [+0.00, +0.02] | 47.5 (+0.49) / 37.4 (+0.67) / 23.6 (+0.41) |
| beam4 + greedy fallback | 25 | 33.20 | +0.68 [+0.57, +0.80] | +0.02 [+0.01, +0.03] | 47.6 (+0.57) / 37.7 (+0.95) / 23.8 (+0.60) |
| beam8 + greedy fallback | 38 | 33.29 | +0.77 [+0.65, +0.88] | +0.03 [+0.01, +0.05] | 47.7 (+0.74) / 37.9 (+1.21) / 23.7 (+0.57) |

The fallback costs one extra greedy decode for 0.6-1.0% of the sentences. It adds +0.02 chrF++ over plain beam 4 (CI [+0.01, +0.03]): small, but it removes the visible failures (a looping Nuer or Kabiye sentence).

## 5. Per language

chrF++ per language (200 sentences each; per-language intervals are not computed, see the group tables for intervals). Loop rates in the last column.

| iso | group | greedy | beam 2 | beam 4 | beam 8 | MBR8 T0.5 | MBR9+g T0.7 | rep 1.05 | T 0.3 | nr4 | loop % greedy / beam 4 / beam 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `swh` | strong | 57.8 | 58.3 | 58.4 | 58.5 | 58.1 | 58.2 | 57.6 | 57.8 | 57.0 | 0.0 / 0.0 / 0.0 |
| `hau` | strong | 45.6 | 45.8 | 46.0 | 46.0 | 45.9 | 46.2 | 44.7 | 45.0 | 44.6 | 0.0 / 0.0 / 0.6 |
| `zul` | strong | 49.0 | 49.9 | 50.0 | 50.4 | 49.6 | 49.6 | 48.8 | 48.7 | 48.6 | 0.0 / 0.0 / 0.0 |
| `kin` | strong | 42.1 | 42.7 | 42.7 | 43.1 | 42.7 | 42.4 | 41.9 | 41.7 | 40.7 | 0.0 / 0.5 / 0.5 |
| `som` | strong | 40.5 | 40.9 | 40.9 | 40.8 | 40.7 | 40.7 | 40.2 | 40.5 | 39.6 | 0.0 / 0.0 / 0.0 |
| `ibo` | medium | 37.7 | 38.3 | 38.3 | 38.5 | 38.2 | 38.2 | 36.6 | 37.6 | 36.6 | 0.0 / 0.0 / 0.0 |
| `sna` | medium | 38.8 | 39.4 | 39.8 | 40.0 | 39.1 | 39.2 | 38.9 | 38.7 | 37.7 | 0.0 / 0.0 / 0.0 |
| `lug` | medium | 35.3 | 35.8 | 36.1 | 36.1 | 35.8 | 35.7 | 35.4 | 35.2 | 34.5 | 0.0 / 0.0 / 0.0 |
| `bem` | medium | 34.3 | 34.8 | 35.3 | 35.9 | 35.4 | 35.1 | 34.7 | 34.7 | 33.2 | 0.0 / 0.0 / 0.0 |
| `nbl` | medium | 37.4 | 38.5 | 38.8 | 39.1 | 37.9 | 38.1 | 37.2 | 36.9 | 37.3 | 0.0 / 0.0 / 0.0 |
| `yor` | weak | 26.9 | 27.4 | 27.7 | 27.8 | 27.3 | 27.4 | 26.3 | 27.0 | 26.6 | 0.0 / 0.0 / 0.0 |
| `amh` | weak | 19.1 | 20.2 | 20.4 | 20.4 | 19.7 | 19.6 | 19.2 | 18.4 | 18.7 | 0.0 / 0.0 / 0.0 |
| `wol` | weak | 29.3 | 29.9 | 30.2 | 30.2 | 30.0 | 30.0 | 29.6 | 29.7 | 28.8 | 0.0 / 0.0 / 0.0 |
| `tir` | weak | 13.4 | 13.9 | 14.2 | 14.4 | 13.9 | 13.9 | 13.2 | 13.3 | 13.0 | 0.0 / 0.0 / 0.0 |
| `kbp` | weak | 22.3 | 22.8 | 22.9 | 22.8 | 23.5 | 23.5 | 23.2 | 22.5 | 22.2 | 0.0 / 2.5 / 4.0 |
| `nus` | weak | 21.1 | 20.4 | 20.1 | 19.9 | 21.4 | 21.5 | 21.8 | 20.8 | 21.0 | 0.5 / 8.0 / 11.5 |
| `kab` | weak | 26.9 | 27.3 | 27.6 | 27.4 | 27.3 | 27.0 | 27.3 | 26.7 | 26.5 | 0.0 / 0.0 / 0.0 |
| `fon` | weak | 21.1 | 21.2 | 21.3 | 21.4 | 21.4 | 21.5 | 21.4 | 21.3 | 20.9 | 0.0 / 1.0 / 1.0 |
| `ach` | weak | 25.9 | 26.1 | 26.2 | 25.9 | 26.8 | 26.7 | 26.2 | 25.7 | 25.9 | 0.5 / 0.5 / 1.5 |
| `efi` | weak | 25.8 | 26.5 | 26.8 | 26.7 | 26.4 | 26.4 | 26.0 | 25.6 | 25.7 | 1.0 / 0.0 / 0.0 |

The beams help 19 of 20 languages at 4 beams (largest: South Ndebele +1.4, Amharic +1.3, Bemba +1.0, Zulu +1.0, Shona +1.0); the exception is Nuer (`nus`: -1.0 at 4 beams, -1.2 at 8) where beam search loops and MBR is better (+0.3). Acholi and Efik (SmolSent) behave like the FLORES+ weak languages.

## 6. AfriCOMET

AfriCOMET with reference and AfriCOMET-QE on the 18 validated languages (3570 sentences), macro, paired 95% CI of the delta to greedy:

| config | AfriCOMET | delta [CI] | AfriCOMET-QE | delta [CI] |
|---|---|---|---|---|
| greedy (vLLM) = reference | 0.6230 | - | 0.5694 | - |
| greedy (HF) cross-check | 0.6229 | -0.0001 [-0.0013, +0.0013] | 0.5693 | -0.0001 [-0.0014, +0.0013] |
| greedy + repetition_penalty 1.05 (vLLM, prompt + output) | 0.6316 | +0.0086 [+0.0067, +0.0105] | 0.5751 | +0.0056 [+0.0036, +0.0077] |
| greedy + repetition_penalty 1.1 (vLLM, prompt + output) | 0.6116 | -0.0115 [-0.0137, -0.0092] | 0.5555 | -0.0139 [-0.0162, -0.0115] |
| greedy + repetition penalty 1.05, generated tokens only (HF) | 0.6293 | +0.0062 [+0.0044, +0.0080] | 0.5743 | +0.0048 [+0.0029, +0.0067] |
| greedy + repetition penalty 1.1, generated tokens only (HF) | 0.6155 | -0.0075 [-0.0097, -0.0054] | 0.5620 | -0.0076 [-0.0099, -0.0054] |
| greedy + repetition_penalty 1.1 (HF, prompt + output) | 0.6098 | -0.0132 [-0.0155, -0.0109] | 0.5551 | -0.0144 [-0.0167, -0.0120] |
| greedy + no_repeat_ngram_size 3 (HF) | 0.6182 | -0.0048 [-0.0065, -0.0032] | 0.5617 | -0.0077 [-0.0095, -0.0059] |
| greedy + no_repeat_ngram_size 4 (HF) | 0.6215 | -0.0015 [-0.0029, +0.0000] | 0.5659 | -0.0036 [-0.0052, -0.0020] |
| sampling T 0.3 (vLLM, 1 sample) | 0.6238 | +0.0008 [-0.0009, +0.0025] | 0.5694 | -0.0000 [-0.0020, +0.0018] |
| sampling T 0.5 (vLLM, 1 sample) | 0.6215 | -0.0015 [-0.0035, +0.0005] | 0.5682 | -0.0013 [-0.0034, +0.0007] |
| sampling T 0.7 (vLLM, 1 sample) | 0.6144 | -0.0086 [-0.0108, -0.0063] | 0.5602 | -0.0093 [-0.0116, -0.0070] |
| beam search 2 (HF) | 0.6302 | +0.0072 [+0.0055, +0.0089] | 0.5768 | +0.0073 [+0.0055, +0.0091] |
| beam search 4 (HF) | 0.6338 | +0.0108 [+0.0089, +0.0126] | 0.5808 | +0.0113 [+0.0094, +0.0133] |
| beam search 8 (HF) | 0.6344 | +0.0113 [+0.0094, +0.0131] | 0.5813 | +0.0118 [+0.0098, +0.0138] |
| MBR: 8 samples T 0.7, pick the candidate with the highest mean chrF++ to the other 7 | 0.6249 | +0.0018 [-0.0001, +0.0039] | 0.5697 | +0.0002 [-0.0017, +0.0023] |
| MBR: same pool + the greedy output (9 candidates) | 0.6267 | +0.0037 [+0.0022, +0.0050] | 0.5725 | +0.0030 [+0.0015, +0.0045] |
| MBR: 8 samples T 0.5 | 0.6285 | +0.0055 [+0.0035, +0.0074] | 0.5744 | +0.0049 [+0.0030, +0.0069] |
| MBR: 8 samples T 0.5 + greedy | 0.6287 | +0.0056 [+0.0042, +0.0071] | 0.5743 | +0.0049 [+0.0034, +0.0065] |

The neural metrics agree with chrF++ that beam 8 ~ beam 4 are best (+0.011 AfriCOMET), that greedy ~ T 0.3 and that T 0.7, `no_repeat_ngram_size` and repetition_penalty 1.1 are worse. They disagree on two configurations: repetition_penalty 1.05 gains +0.009 AfriCOMET (+0.006 QE), more than MBR (+0.006), although it is neutral on chrF++, lowers spBLEU by 0.5 and drops the number recall from 95.7% to 89.8%; AfriCOMET does not seem to punish a changed or dropped number. We do not adopt it: the metric that disagrees seems insensitive to exactly the error the penalty introduces. MBR (T 0.5) gains +0.0055 AfriCOMET, half of beam 4; on chrF++ it gains 80% of beam 4's gain, as expected from a selection by chrF++.

## 7. Cost

| config | engine | seconds for 3,970 sentences | sentences/s | tokens/s | cost vs same-engine greedy |
|---|---|---|---|---|---|
| greedy (vLLM) = reference | vllm | 20 | 203.3 | 12201 | 1.0x |
| greedy (HF) cross-check | hf | 141 | 28.1 | 1681 | 1.0x |
| greedy + repetition_penalty 1.05 (vLLM, prompt + output) | vllm | 21 | 188.7 | 11226 | 1.1x |
| greedy + no_repeat_ngram_size 3 (HF) | hf | 154 | 25.8 | 1533 | 1.1x |
| greedy + no_repeat_ngram_size 4 (HF) | hf | 188 | 21.1 | 1256 | 1.3x |
| sampling T 0.5 (vLLM, 1 sample) | vllm | 24 | 168.4 | 10073 | 1.2x |
| beam search 2 (HF) | hf | 331 | 12.0 | 723 | 2.3x |
| beam search 4 (HF) | hf | 655 | 6.1 | 368 | 4.6x |
| beam search 8 (HF) | hf | 791 | 5.0 | 305 | 5.6x |
| MBR: 8 samples T 0.7, pick the candidate with the highest mean chrF++ to the other 7 | vllm+cpu | 194 | 20.4 | 9769 | 10.0x |
| MBR: same pool + the greedy output (9 candidates) | vllm+cpu | 246 | 16.1 | 8677 | 12.6x |
| MBR: 8 samples T 0.5 | vllm+cpu | 305 | 13.0 | 6236 | 15.6x |

The box GPU is shared with other jobs (this run had <= 0.1 of its memory and a fraction of its compute), absolute speeds are indicative; the ratios between rows of the same engine are the useful part. Beam search was timed in HF `generate`, which is 7x slower than vLLM greedy at equal work; a serving stack with an efficient beam implementation would narrow the gap (not measured here).

## 8. Caveats

* 200 sentences per language, equal language weight: the intervals measure sentence-sampling noise only; per-language differences below ~0.5 chrF++ are not resolved. The three SmolSent languages use a different domain from the 17 FLORES+ ones.
* Sampling configurations use one sample per sentence (seed per sentence), which is what a user would get; the average over several seeds would be a little less noisy but not different in expectation.
* MBR utility = chrF++ (the metric also used for scoring). Optimising chrF++ to select among candidates is expected to favour chrF++ over BLEU or neural metrics; the AfriCOMET column is the check.
* Sentence level only. Long documents (SmolDoc, LongDoc) have a different failure profile (truncation, repetition across many sentences); the conclusions about repetition penalty and n-gram blocking do not transfer to them without a separate test.
* The conclusion is for the RL-final model. Beam search was not tried on the SFT model, which may benefit differently.
