# The Gemma-3-270m-it en->African RL reward, step by step (v3: state of 2026-10-19, after D-37 .. D-40)

For the owner to vet. This file supersedes `rl_reward_explained_2026-10-17_v2.md` (kept next to it). **Sections 0-21 are the v2 text; the passages that changed are patched in place and marked [v3]; sections 22-30 are new.**
What v3 adds (all in `gemma_mt/`, suite green, staged on the rental, run starts automatically after SFT and the benchmark evaluation):

* **GlotLID guard ON** (D-37): `wrong_language` is a -2 failure for the 75 of 77 languages whose measured human recall is >= 0.90 (off: pcm, kon), labels / minimum length per language from the calibration (section 22).
* **35 more RL languages = 77** (D-38): 12 named on the AfriCOMET card but unvalidated (QE blended with consensus, advantage weight 0.3 / 0.4) and 23 outside AfriCOMET (consensus only, weight 0.5); calibration `shaping_flores_dev_v5.json` (section 24).
* **QE-free consensus signal** (D-38): mean chrF++ of a completion against the other valid completions of its GRPO group, in modes `qe` (33 languages) / `blend` (21) / `consensus` (23) (section 23).
* **Sampling / normalisation audit** (D-39, section 27) and the launch blockers it found: `load_rl_prompts` crashed on the prompt directory, `sacrebleu` was missing in the trainer's environment (section 28).
* **Hub storage** (D-40): org repos, squashed history, pruned tags (section 29).

Abbreviations as in v2, plus (paths under `gemma_mt/`): **CO** `rl/consensus.py`, **LD** `rl/lid.py`, **CAL5** `calibration/shaping_flores_dev_v5.json`, **ML** `scripts/measure_lid.py`, **B5** `scripts/build_calibration_v5.py`, **DR** `scripts/dry_check_rl.py`.
Line numbers of NEW code are those of the working copy at the time of writing (R = `rl/reward.py`); the v2 citations of R, TR and the other files have drifted (new code was inserted above them) and are cited by name in the new sections.

---------------------------------------------------------------------------------------------------------------------------------------------

## 0. The loop in one picture **[v2]**

```
prompt (English source, target iso)  --generate G=8 samples, T=1--> token ids (cut at first stop id)         SM:32-75, TR:232-234 (vLLM: D-28)
  --> TranslationReward(samples)                                                                             R:154-304
        0a source -> NFC; decoded completion -> NFC                                                          R:161,166, TX:59-77             (D-29, new)
        0b end-of-turn checks + failure classes -> penalty, never scored                                     R:169-200
        1 split source/output into sentences; equal counts: i-th to i-th; counts differ: language-aware
          Gale-Church (omission / extra beads) + shortfall check -> chunks + FAILED chunks                   R:218-239, CP:43-125, SEG:119-172   (D-30, new)
        2 AfriCOMET-QE on every aligned chunk (one scorer call for the whole batch)                          R:241-250, CM:72-88
        3 per-chunk shaping curve (per-language anchors from CAL4; t_fail floor = reward.fail_floor)         RS:77-94,238-242, R:267
        4 per-chunk penalties, subtracted, never below -2: length, lexical, DIACRITICS (new)                  R:268-280                       (D-31, new)
        5 weighted mean over chunks (weight = source characters; failed chunk = -2)                         CP:128-142, R:282
        6 reliability weight of the language (separate output, used on the advantage; efi = 0)              R:287
  --> rewards r_i in [-2, +2.25]  (asserted)                                                                 TR:509-512
  --> GRPO advantage: standardise inside each group of 8, multiply by the language weight                    GR:32-48, TR:513-514
  --> token-level policy-gradient loss + 0.05 * k3-KL to the frozen SFT model, one on-policy update          GR:63-98, TR:565-582
```

## 1. Inputs

| input | where it comes from | cite |
|---|---|---|
| English source `english`, target tag `iso` | RL prompt file (`english, iso, language, src_afr_tokens, source_dataset, length_bin`); no reference is read | `gemma_mt/rl/data.py:1-12`, D-18 |
| prompt ids | `fmt.build_prompt(english, iso, tokenizer)` (Gemma chat template, ends with the generation prompt, first generated token = the `iso` tag) | TR:226, FMT:159-163 |
| sampled completion ids | HF `generate`, `num_return_sequences=G` (G=8), T=1.0, top_p=1.0, top_k=0, `eos_token_id=[<end_of_turn>, <eos>]`, `pad_token_id=pad` | Y:18,21-23; SM:55-57 |
| budget | `max_new_tokens` per length bin: b64 384, b256 1280, b512 2304, b1024 3584, mixed 3584 (Gemma tokens); global fallback 1024 | Y:44,59-64; TR:493-495 |
| what the reward receives | `Sample(token_ids, iso, source, post_stop_tokens)`. `token_ids` = ids up to and including the first stop id (or all `max_new_tokens` ids if none) | R:108-113, SM:68-70, TR:235-236 |
| `post_stop_tokens` (new) | number of non-padding, non-stop tokens the sampler produced after the first stop id and that the cut removed | SM:65-71, R:173 |

Generation facts relevant to the reward: finished rows are right-padded by HF with `pad_token_id`, the rollout cuts each row at the first stop id
(`cut_at_stop`, TX:50-56, SM:68). So with the stock sampler nothing after the end token ever reaches the reward (D-26 explains what that implies).

## 2. Preprocessing of the completion (`decode_completion`, TX:64-79) **[v2]**

1. Cut at the FIRST stop id (`<end_of_turn>` or `<eos>`), the stop id itself is excluded from the text: TX:68-69, `cut_at_stop` TX:50-56.
2. Decode twice: with and without special tokens (TX:70-71). If they differ, `special_ids_stripped` is set (TX:72). The clean text is then scrubbed of special-token
   STRINGS typed out as characters (`<start_of_turn>`, `<end_of_turn>`, `<bos>`, `<eos>`, `<pad>`, `<unk>`, `<mask>`, `<unusedNN>` ...: TX:30-32,73) and `.strip()`ped (TX:78).
3. Language tag: `strip_iso_prefix` (FMT:292-313): the text must start with a 3-letter tag (`^([A-Za-z]{3})(?:\s+|$)`, FMT:288) that is a known language tag (FMT:289);
   `ok` = the expected tag, `wrong` = a different known tag, `missing` = no tag. With the default policy `require` anything but `ok` is a `bad_prefix` failure
   (R:182-185, FMT:316-324). The tag is removed before scoring (R:186). Only the ends are whitespace-stripped, the interior of the translation is never touched (FMT:300-301).
4. **[v2, D-29] NFC.** `decode_completion` normalises the decoded text to Unicode NFC after the special-string scrub and before `.strip()` and the tag split (`nfc()` TX:59-61, call TX:77; switch
   `normalise`, config `reward.nfc`, default True: R:86, R:166). The reward ALSO normalises the English source (R:161) and uses that normalised source for the role-tail test (R:176), the copy test
   (R:190), the hard length-ratio test (R:196), the chunk planner (R:226) and therefore every lexical / diacritic feature, so source and output are always compared in the same form. The only other
   normalisation in this path remains NFKC + lower-casing + punctuation stripping inside the copy test (`G:14-16`), which is not applied to the scored text. Why it matters (measured on the FLORES+ dev references,
   `scripts/build_flores_dev_all.py` data): 29.8% of the Yoruba and 31.8% of the Fon references are NOT in NFC; the NFD form of Yoruba is 22.6% longer in characters (12.3% in UTF-8 bytes), Fon +8.3% (+6.7%),
   Kikuyu +11.3% (+10.2%), Wolof +3.5%, Ewe +1.0%. Every length rule counts characters: the hard `length_ratio` failure (G:53-59), the length-window penalty (RS:176-183), the appended-text rule (CP:106-111)
   and the word forms of the lexical features; the degenerate test (G:33-50) also looks at UTF-8 bytes (zlib ratio). Without NFC a policy could move all of them by sampling decomposed diacritics (finding F1, now resolved).
   QE itself (SentencePiece NFKC) was never affected. `tests/test_reward_nfc.py` pins: NFC and NFD completions of Yoruba and Fon get the identical reward, text and features; the source is normalised too;
   with `nfc: false` a decomposed completion crosses the `max_len_ratio` limit that the composed one respects.

## 3. Failure classes (hard failures: no QE call, reward = the class penalty)

Penalty value of every class: `-neg_scale` = **-2.0** by default, i.e. the floor of the shaped range (`R:138-139`, `RS:65-68`); `penalty` may be set lower
but never above the floor, otherwise a failure could outscore a scored translation (R:140-142). The three end-of-turn classes can each be set lower with
`eos_*_penalty` (R:143-151). Checks run in this order, first hit wins (R:169-210):

| # | class (`status`) | detection | cite | default penalty |
|---|---|---|---|---|
| 1 | `no_end_of_turn` | the ids contain no stop id (generation hit `max_new_tokens`) = TRUNCATED. Always on | TX:68, R:169-171 | -2.0 (`eos_truncated_penalty`) |
| 2 | `post_eos_rubbish` (new, D-26) | after the first stop id there is a token that is neither a pad id nor another stop id (counted from the ids given AND from `Sample.post_stop_tokens`). Needs `eos_discipline` | TX:75-76, R:172-175 | -2.0 (`eos_post_end_penalty`) |
| 3 | `dangling_marker` (new, D-26) | before the stop id the raw decode contains `<start_of_turn>`, `<end_of_turn>`, `<\|assistant\|>`, `<\|user\|>`, `<\|system\|>`, `<\|im_start\|>`, `<\|im_end\|>` (TX:36,79), or the text ends with a bare role line `\nmodel` / `\nuser:` / `\nassistant` (not if the source does too) (G:62-70). Needs `eos_discipline` | R:176-178 | -2.0 (`eos_dangling_penalty`) |
| 4 | `special_tokens` | any special id or special string was stripped. **Off by default** (`penalize_special_tokens: false`); otherwise they are silently removed and only counted in `special_stripped_rate` | R:68,179-181,167 | -2.0 |
| 5 | `bad_prefix` | tag missing / wrong (policy `require`) | R:182-185 | -2.0 |
| 6 | `empty` | nothing left after the tag | R:186-189 | -2.0 |
| 7 | `copy_source` | normalised (NFKC, lower, punctuation -> space) output equals the source, or word-set Jaccard >= `copy_jaccard` 0.9 | G:14-30, R:190-192 | -2.0 |
| 8 | `degenerate` | one word repeated more than 4 times in a row; or >= 8 words with < 30% distinct (lower-cased); or >= 80 UTF-8 bytes that zlib-compress to < 18% | G:33-50, R:193-195 | -2.0 |
| 9 | `length_ratio` | `len(text)/len(source)` in characters (both stripped) outside [0.2, 5.0] | G:53-59, R:196-198 | -2.0 |
| 10 | `wrong_language` **[v3, D-37]** | GlotLID top-1 label not in the language's accepted label set (calibration `lid`); only when `lid_enabled` (**true in `Y`** now), the language's guard is on (recall >= 0.90) and the text has >= max(15, the language's `min_chars`) characters; section 22 | R:280 | -2.0 |

A failed row gets `rewards = class_penalty[status]` (R:258-259), `qe = None`, and its weight is still the language weight (R:287). Per-class rates are logged as `fail/<class>` (R:315-316).

## 4. Sentence chunks and the Gale-Church alignment **[v2: 4.5 is new]**

Entry: `plan_chunks(source, text, ...)`, called once per surviving row (R:218-239, CP:43-125). With `chunking: sentences` (Y:98; the dataclass default is `off`, R:75):

**4.1 Sentence splitting** (`split_sentences`, SEG:56-73). Whitespace (including newlines) is collapsed (SEG:58). A terminator run `.`, `!`, `?` or Ethiopic `።` (SEG:23) plus closing quotes/brackets (SEG:24)
ends a sentence only if it is followed by whitespace/end of text (SEG:28), the next word does not start with a lowercase letter (not for `።`, SEG:35-37), the word before it is not an
abbreviation from the list at SEG:25-27, not a single capital initial, not a dotted acronym (`U.S.`), and not a list number of <= 3 digits at the start of a segment (SEG:42-52).

**4.2 Pairing** (`align_chunks`, SEG:175-198):

| status | condition | result |
|---|---|---|
| `single` | the source has < 2 sentences | one chunk = whole source vs whole output (SEG:185-186) |
| `aligned` | equal sentence counts | i-th source sentence vs i-th output sentence (SEG:187-188) |
| `whole` | counts differ and the output has <= 1 sentence (or `chunk_align: whole`) | one chunk: all source sentences joined by a space vs the output (SEG:189-190) |
| `gale_church` | counts differ, output has >= 2 sentences | beads from the DP below (SEG:191-198) |

**4.3 Gale-Church DP** (`gale_church`, SEG:119-172), a banded dynamic programme over beads (source sentences consumed, output sentences consumed):

* bead priors (SEG:81-82): (1,1) 0.89; (1,0) omission 0.01; (0,1) extra output 0.01; (2,1) 0.045; (1,2) 0.045; (2,2) 0.011; (3,1) 0.006; (1,3) 0.006; (3,2) 0.002; (2,3) 0.002; (4,1) 0.001; (1,4) 0.001.
* length model (SEG:110-116): `c = (total output chars) / (total source chars)`, floored at 1e-3 (SEG:129, "the observed ratio of the two texts, so no per-language constant");
  for a bead with source length `l_s` and output length `l_o` (characters): `delta = (l_o - l_s*c) / sqrt(max(l_s,1) * 6.8)` (variance constant `_S2 = 6.8`, SEG:83,114);
  `p = max(2*(1 - Phi(|delta|)), 1e-12)` (SEG:105-107,115); `cost = -ln(prior) - ln(p)` (SEG:116); a bead of two empty sides costs 1e9 (SEG:112-113).
* search: minimise the summed cost; only cells with `|i*m/n - j| <= max(8, 0.25*max(n,m))` are visited (SEG:137-143); if the corner is unreachable the whole thing is one big bead (SEG:165-166); traceback SEG:167-172.

**4.4 What happens to leftovers** (`align_chunks` SEG:191-198 then `plan_chunks` CP:78-117):

* a bead with sources and outputs -> one chunk (joined with a space) (SEG:194-195);
* an omission bead (sources, no output) -> a chunk with an EMPTY output and `n_failed += len(sources)` (SEG:196-198); in `plan_chunks` an empty output side (or a copy of its source, Jaccard >= 0.9) with more than one pair becomes a FAILED chunk with weight = the source characters, not sent to QE (CP:112-115);
* an extra-output bead (outputs, no source) is dropped by `align_chunks` (SEG:194-198 only emits `si and oi` and `si` beads). Its characters are recovered in `plan_chunks`: `extra = len(" ".join(all output sentences)) - len(" ".join(matched outputs))`; if `extra > 0` and `extra >= 5%` of the output length, one failed chunk of weight `extra` output characters (CP:89-95);
* inside every pair with both sides non-empty, text appended beyond the language window: `over = len(b) - lim*len(a)` with `lim = p95_ratio*(1+0.15)` (calibrated) or `1.8` (uncalibrated); if `over > 10%` of `len(b)`, one more failed chunk of weight `over` characters (CP:106-111). `p95_ratio` is `len_window[lang][1]` (R:220, RS:148-151);
* weights: `chunk_agg: source_length` -> the chunk's SOURCE characters (CP:97), `mean` -> 1; the two "extra text" failed chunks weigh output characters / 1 (CP:94,110). Whole-text mode (`chunking: off`) is a single chunk weighing the source length (CP:74-75).

**4.5 [v2, D-30] The omission / extra-sentence detector** (`omission_detector: true`, `Y:102`; only when the sentence counts of source and output DIFFER and the language has a `len_ratio.corpus` in CAL4).
Why: in v1 (finding F2) a translation that DROPPED a sentence was practically never penalised as a failed chunk: Gale-Church found an omission bead in 2 of 20,000 synthetic trials, the dropped sentence was
absorbed into a 2:1 bead (a long source scored against a short output, which QE rates mediocre but not as failed), and an extra output sentence was caught in only 40-42% of the cases. The cause is the length
model of `gale_church`: the ratio `c` is the OBSERVED one of the very pair (SEG:129), so a dropped sentence just makes the output "shorter overall", and an empty output is 4+ sigma away from any sentence
(`cost = -ln(0.01) - ln p(delta)`, SEG:110-116), so the cheaper explanation is always a merge. Raising the priors alone does not help (measured below).

What the detector does (all optional arguments, default off; with them off the planner is bit-identical to v1):
1. `LengthModel` (SEG:87-102): the expected output/source character ratio and the Gale-Church variance per source character come from the LANGUAGE (CAL4 `len_ratio.corpus` and `.s2`, computed on the human dev
   references; RS:148-160), not from the pair. The omission bead (1,0) and the extra-output bead (0,1) cost a CONSTANT `omit_cost` / `extra_cost` nats (SEG:152-160) instead of the Gaussian length term, and are
   *directional*: an omission bead only exists when the output has fewer sentences than the source, an extra bead only when it has more (so a dropped sentence can never be explained as an extra one and
   vice versa). The (k,l) merge/split beads keep the priors of SEG:81-82 but use the language's ratio and variance (SEG:110-116).
2. `align_beads` (SEG:201-215) returns the beads with their sentence indices; `plan_chunks` (CP:79-88) turns an omission bead into the existing failed chunk with weight = the SOURCE characters of the dropped
   sentence(s) (CP:112-115) and records `n_failed` and `flagged_src`. Extra sentences are the existing `unmatched_output` failed chunk (CP:89-95), now actually produced because the (0,1) bead is available.
3. Shortfall check (CP:98-105, 117-122) for what the alignment still merged: a bead of >= 2 source sentences, or the "whole" case (one output sentence for several source sentences), whose output is shorter
   than `short_frac` (0.7) x the expected length `ratio * len(source side)` and whose missing part `len(a) - len(b)/ratio` is >= `short_min` (15%) of the bead is split into a scored chunk of the covered
   source characters and a FAILED chunk of the missing source characters (total weight unchanged).
4. Per-language operating point (`choose_omission_cost`, OE:197-225; CAL4 `omission.cost`): the omission cost is the smallest of 4, 5, 6, 8, 12 for which the language's UNTOUCHED human dev documents
   are flagged in at most 2% of the cases (`reward.omission_cost: null` = use it; a number overrides all). 34 of the 40 languages with a FLORES+ document order keep 4; fon and nus need 12, kbp 8, kik and lug 6, nya 5.
   Ewe cannot reach the target at any cost (14.3% at 4, 10.6% at 12): its FLORES+ references themselves drop and merge sentences (F16). When even the largest cost leaves more than 5% flagged the smallest
   cost is kept (a higher one would only blunt the detector), so ewe stays at 4 (`omission.note` in CAL4); fon (2.1% at 12) keeps 12.

**Choice by measurement** (`python -m gemma_mt.rl.omission_eval`; OE:1-30). Documents = windows of 3-12 consecutive FLORES+ DEV sentence pairs (random tilings, 8 per language = ~1,050 documents), per
language: *untouched* (false alarms), *drop* (k = 1 .. n/2 random sentences removed from the target side), *keep one* (only one sentence of >= 4 is translated), *extra* (1-2 sentences of another document appended /
inserted). Sentence recall = share of the dropped sentences that sit in an omission / shortfall bead; the false-alarm column is measured on ALL untouched documents, including the 13-55% of documents
whose human translation has a different sentence count. Macro over yor hau ibo swh zul amh kin ewe (calibration 3 tilings, then the final 8-tiling runs):

| variant (macro of 8 languages, 3 tilings) | false alarm | sentence recall | precision | docs with >= 1 found | k=1 recall | dropped-char coverage | extra appended / inserted found | false omission flags when sentences are appended / inserted |
|---|---|---|---|---|---|---|---|---|
| v1 planner (legacy Gale-Church) | 0.3% | 0% | - | 0% | 0% | 0.1% | 42% / 40% | 0% / 0% |
| legacy cost, omission/extra priors raised to 0.1 | 0.4% | 0% | - | 0% | 0% | 0.1% | 42% / 40% | 0% / 0% |
| legacy cost, priors 0.3 | 0.6% | 0% | - | 0% | 0% | 1.3% | 42% / 40% | 0% / 0% |
| shortfall check only (legacy alignment, 0.7) | 1.3% | 53% | 36% | 64% | 32% | 61% | 42% / 40% | 0% / 0% |
| language-aware DP, cost 3 | 2.6% | 61% | 66% | 80% | 60% | 92% | 94% / 92% | 0.0% / 0.0% |
| language-aware DP, cost 4 | 2.1% | 56% | 64% | 75% | 56% | 89% | 92% / 91% | 0.0% / 0.0% |
| language-aware DP, cost 5 | 1.6% | 51% | 63% | 72% | 54% | 86% | 91% / 89% | 0.0% / 0.0% |
| same, cost 4, NOT directional | 2.3% | 56% | 64% | 75% | 57% | 89% | 92% / 91% | 0.8% / 0.5% |
| DP cost 4 + shortfall 0.7 (**chosen**, per-language cost) | 2.2% (0.6% without ewe) | 56% | 64% | 77% | 59% | 89% | 93% / 92% | 0.0% / 0.0% |
| DP cost 4 + shortfall 0.8 | 2.7% | 57% | 63% | 76% | 58% | 90% | 92% / 91% | 0.2% / 0.3% |

* Raised omission/extra priors (0.1, 0.3) with the legacy Gaussian cost: 0.0% / 0.0% recall. The length term dominates the prior; the idea is dead.
* Shortfall check alone (legacy alignment): recall 53%, precision 36%, extra sentences still 42% - cheap but blunt (it cannot say which sentence is missing, and does nothing for extras).
* Language-aware DP with constant costs: the large gain (recall 0 -> 56%, extras 42% -> 93%); the cost 4 is the elbow of recall vs false alarms; the shortfall check on top adds nothing after the DP except the
  "keep one" case (100%).
* Sentence-count mismatch + per-chunk QE was NOT used: QE sees a half-translated chunk only as "mediocre" (its half-omitted failure rate is 0-8% in the SSA study), it needs the GPU, and it cannot say which part is missing.

Results of the final configuration (language-aware DP, per-language cost, shortfall check), 8 tilings, before (v1 planner) -> after:

| lang | docs | human count differs | det. docs before -> after | sentence recall | precision | char cov. | keep-1 | extras appended before -> after | inserted | FA before -> after | FA any before -> after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yor | 1048 | 20.7% | 0.0% -> 73.3% | 51.6% | 60.5% | 87.2% | 100.0% | 31.9% -> 91.7% | 31.8% -> 91.4% | 0.0% -> 0.8% | 3.1% -> 4.7% |
| hau | 1048 | 13.4% | 0.0% -> 82.9% | 62.6% | 68.3% | 92.9% | 100.0% | 44.0% -> 95.0% | 38.0% -> 96.0% | 0.0% -> 0.0% | 3.7% -> 3.7% |
| ibo | 1048 | 22.8% | 0.0% -> 75.1% | 54.1% | 61.0% | 90.9% | 100.0% | 42.3% -> 94.4% | 37.5% -> 92.3% | 0.0% -> 0.0% | 3.9% -> 3.5% |
| swh | 1048 | 15.3% | 0.0% -> 84.4% | 64.6% | 70.4% | 94.7% | 100.0% | 51.1% -> 94.2% | 49.5% -> 94.1% | 0.0% -> 0.7% | 1.9% -> 2.6% |
| zul | 1048 | 16.2% | 0.0% -> 80.8% | 59.7% | 65.4% | 93.7% | 100.0% | 41.0% -> 94.1% | 36.6% -> 93.2% | 0.0% -> 0.0% | 3.0% -> 2.8% |
| amh | 1048 | 35.6% | 0.0% -> 77.3% | 56.1% | 64.1% | 90.0% | 100.0% | 62.8% -> 94.7% | 57.6% -> 94.2% | 0.0% -> 2.6% | 8.8% -> 10.4% |
| kin | 1048 | 13.1% | 0.0% -> 79.3% | 58.6% | 65.7% | 91.8% | 100.0% | 35.2% -> 94.9% | 32.7% -> 94.2% | 0.0% -> 0.0% | 6.4% -> 6.8% |
| ewe | 1048 | 54.8% | 0.0% -> 59.8% | 39.2% | 53.3% | 73.2% | 100.0% | 31.1% -> 86.9% | 38.1% -> 83.5% | 2.2% -> 13.6% | 12.5% -> 26.2% |
| **macro** | | | 0.0% -> 76.6% | 55.8% | 63.6% | 89.3% | 100.0% | 42.4% -> 93.2% | 40.2% -> 92.4% | 0.3% -> 2.2% | 5.4% -> 7.6% |

Reading the table: `det.doc` = share of dropped-sentence documents with at least one dropped sentence found; `char cov.` = flagged failed weight capped at the dropped characters / dropped characters
(the weight the reward actually moves); `FA` = untouched documents with an omission-type flag (target < 2%); `FA any` = untouched documents with ANY failed chunk (includes the old appended-text rule, see F15).
All 40 languages with a FLORES+ document order (3 tilings, per-language cost): FA <= 2% in 33 of 40 languages (macro 1.1%, recall 54.5%, extra sentences found 93.1%); the languages above 2% are those whose human translations often change the sentence count:

| lang | FA | docs found | sentence recall | char cov. | extras found | cost |
|---|---|---|---|---|---|---|
| afr | 0.0% | 85.6% | 70.3% | 97.1% | 95.5% | 4 |
| amh | 2.6% | 77.8% | 57.6% | 90.0% | 94.4% | 4 |
| bam | 3.4% | 73.9% | 52.5% | 89.5% | 92.9% | 4 |
| bem | 0.0% | 75.7% | 57.3% | 88.3% | 95.4% | 4 |
| ewe | 13.3% | 59.2% | 38.8% | 72.9% | 86.8% | 4 |
| fon | 2.6% | 63.7% | 51.7% | 66.2% | 89.1% | 12 |
| hau | 0.0% | 84.1% | 63.0% | 92.7% | 94.3% | 4 |
| ibo | 0.0% | 73.8% | 54.9% | 90.4% | 94.2% | 4 |
| kab | 1.8% | 78.0% | 58.7% | 91.5% | 93.1% | 4 |
| kam | 0.0% | 73.3% | 53.6% | 87.1% | 94.0% | 4 |
| kbp | 2.1% | 56.7% | 38.4% | 67.4% | 85.2% | 8 |
| kik | 1.0% | 66.1% | 44.8% | 79.5% | 94.0% | 6 |
| kin | 0.0% | 80.8% | 59.8% | 92.8% | 93.4% | 4 |
| kmb | 1.6% | 78.0% | 56.6% | 88.2% | 94.5% | 4 |
| lin | 0.0% | 75.5% | 58.5% | 90.9% | 95.4% | 4 |
| lua | 0.3% | 75.9% | 55.1% | 89.4% | 95.8% | 4 |
| lug | 1.3% | 66.7% | 46.8% | 83.8% | 96.2% | 6 |
| luo | 2.1% | 75.3% | 52.3% | 89.6% | 91.6% | 4 |
| mlg | 0.3% | 79.0% | 54.2% | 92.7% | 93.9% | 4 |
| mos | 0.8% | 68.3% | 49.2% | 85.4% | 87.6% | 4 |
| nso | 0.0% | 84.9% | 66.1% | 94.8% | 95.3% | 4 |
| nus | 0.8% | 63.7% | 49.4% | 66.8% | 91.2% | 12 |
| nya | 1.6% | 74.5% | 50.3% | 86.0% | 95.3% | 5 |
| orm | 0.0% | 79.1% | 56.7% | 91.6% | 95.4% | 4 |
| run | 0.5% | 76.0% | 57.4% | 88.9% | 91.6% | 4 |
| sna | 0.0% | 76.3% | 54.1% | 89.8% | 92.8% | 4 |
| som | 0.5% | 79.2% | 57.8% | 93.8% | 95.3% | 4 |
| sot | 0.5% | 81.7% | 60.5% | 93.7% | 97.0% | 4 |
| ssw | 0.0% | 77.0% | 54.2% | 91.8% | 91.6% | 4 |
| swh | 0.5% | 85.5% | 64.4% | 95.1% | 94.1% | 4 |
| tir | 0.5% | 73.2% | 53.7% | 87.3% | 91.8% | 4 |
| tsn | 0.8% | 77.2% | 53.1% | 90.0% | 94.2% | 4 |
| tso | 0.0% | 81.4% | 57.5% | 90.3% | 92.5% | 4 |
| tum | 0.8% | 66.7% | 46.9% | 81.5% | 89.6% | 4 |
| twi | 2.1% | 66.1% | 51.0% | 85.2% | 92.7% | 4 |
| umb | 1.6% | 71.1% | 50.7% | 86.0% | 89.6% | 4 |
| wol | 0.8% | 75.8% | 55.8% | 90.3% | 93.5% | 4 |
| xho | 0.0% | 78.0% | 56.5% | 93.3% | 96.1% | 4 |
| yor | 0.8% | 72.7% | 50.2% | 86.3% | 90.1% | 4 |
| zul | 0.0% | 79.8% | 60.1% | 93.2% | 95.0% | 4 |
| **macro (40)** | 1.1% | 74.7% | 54.5% | 87.5% | 93.1% | |
Caveats: documents come from one corpus (FLORES+ news/wiki), the English side is the same for all languages, a "dropped" sentence is a whole sentence (the model's real omissions are often partial),
and the length statistics are measured on the same dev sentences (2 numbers per language, but not out of sample). Equal sentence counts (a drop plus an extra) are not examined: the planner keeps the
i-th-to-i-th pairing there, bit-identically to v1 (fuzz test: 400 random samples, features off = identical rewards, status, qe, weights; detector on = identical on all 132 single-sentence and 150 equal-count samples,
74 of 118 count-differing samples change; `tests/test_chunk_plan.py`).

## 5. AfriCOMET-QE scoring of each chunk

* Model: `masakhane/africomet-qe-stl-1.1` (`Y:122`), a `referenceless_regression_metric` over `Davlan/afro-xlmr-large-76L` with average pooling and a sparsemax layer mix (HF cache `hparams.yaml:2,9-12`). Source and translation are embedded SEPARATELY, then `[mt, src, |mt-src|, mt*src]` go through a feed-forward net with NO final activation (COMET `regression/referenceless.py:191-197`, `final_activation: None` line 93), so a score can fall outside [0, 1].
* Truncation: each side is tokenised with `truncation=True, max_length = max_positions - 2` = 510 (COMET `encoders/base.py:170-174`, `max_positions` 512). Nothing in the reward truncates; chunking exists to keep each side under it (D-23). `CM:26,67-70` only measures lengths.
* One scorer call for ALL chunks of ALL rows of the batch: `self.scorer(seg_src, seg_mt)` (R:241-243); length mismatch raises (R:244-245).
* Batching: `batched_in_length_order` sorts by `len(src)+len(mt)` characters, runs `batch_size` (32, `Y:123`) at a time and un-sorts (CM:32-44, 88). Inference under `torch.no_grad()` (CM:83); **fp16 on CUDA only** (`model.half()` when the device is cuda, CM:57-58), fp32 on CPU.
* Segments are the strings from the plan; the raw QE value `c` is NOT clipped before shaping (R:243,267). `observe()` is a no-op unless `mode: adaptive` (R:249-250, RS:225-236).

## 6. The shaping function

Piecewise linear, monotone non-decreasing, per language (`shape`, RS:88-94; knots `make_curve`, RS:77-85). With anchors `(t_fail, t_mid, t_great)`:

```
xs = [0,  t_fail,  t_mid,  t_great,  max(1.0, t_great + 0.001)]
ys = [-2, 0,       +1,     +2,       +2.25 ]          (preset "repo"; ys[0] = -neg_scale)     RS:58-62,83-84
shape(c) = -2 for c <= 0 ; +2.25 for c >= xs[-1] ; np.interp(c, xs, ys) otherwise               RS:90-94
```

So: failure zone `c < t_fail` runs from -2 (at c=0) to 0 (at t_fail); acceptable -> good `t_fail..t_mid` is 0..+1; good -> great `t_mid..t_great` is +1..+2; above `t_great` there is a tiny slope to +2.25 at c=1 ("a score above what human translations get is where a learned metric is gamed", RS:20-26). Reward range: **[-2, +2.25]** (`preset_range`, RS:65-68, R:138). The alternative preset `owner_note` (ys = 0,1,1,1; range [-2,+1]) exists but is not used (Q-01 resolved to `repo`, RS:60-61, `Y:88`).

Anchor sanitation (`ContentShaper.anchors`, RS:218-223; `make_curve`): `t_fail >= fail_floor` (reward parameter, 0.0 in `rl.yaml`; the constant `FAIL_FLOOR = 0.46` RS:55 is only its default, RS:107-111); `t_mid >= t_fail + 0.001`, `t_great >= t_mid + 0.001` (RS:80-82). A language missing from the file uses `default_anchors = (0.46, 0.60, 0.75)` (RS:56,219) with a warning if the whole file is missing (RS:130-133).

**Where the thresholds come from [v2, D-32].** `CAL:54-141` scores (english, human reference) pairs of the language with the QE model on the DEV split, and also the reference of ANOTHER sentence ("mismatched"),
the source copied, the reference with half its words removed, and the reference with an unrelated sentence appended (CAL:86-93). Both texts are NFC-normalised first (CAL:66-67). Then (`pair_anchors`, RS:288-296, called at CAL:101):

```
t_fail  = max(fail_floor, P90 of QE on MISMATCHED references)          fail_floor = 0.0 in CAL4 (0.46 in v1)
t_mid   = max(P50 of QE on human references, t_fail + 0.05)
t_great = max(P90 of QE on human references, t_mid  + 0.05)
```

CAL4 was produced on the new instance (RTX PRO 6000, shared, fp16, batch 32, <= 6 GB, 7 minutes) from `/workspace/sft_build/calibration/flores_dev_all_v2.jsonl` (FLORES+ DEV, 997 sentences per language, 41 languages;
pcm: MAFAND dev, 997 of 1,484; efi: no reference set). `fail_floor` is now a reward parameter (`reward.fail_floor`, R:97,137; RS:107-111,218-223,238-242; yaml `Y:91`): v1's constant 0.46 (RS:55) is wrong for
languages whose human references score below it (the SSA study: kam, lua, kmb, bam, umb ... have human medians of 0.40-0.46). It moves t_fail of only three languages (pcm 0.459, kam 0.452, lua 0.438 instead of 0.46; the file stores both: `anchors`
and `anchors_at_floor_0.46`); the +0.05 spacing floor of `t_mid` is active in 21 languages, that of `t_great` only in kbp (see F10).

**Anchors before / after for the 12 languages that had the MAFAND-based v3 file** (v3: MAFAND dev news, ~500 sentences; v4: FLORES+ dev, 997; all 12 stay usable):

| lang | v3 (MAFAND, n) t_fail / t_mid / t_great | v4 (FLORES+, 997) t_fail / t_mid / t_great | change | AUC v3 -> v4 | weight v3 -> v4 |
|---|---|---|---|---|---|
| hau | 0.663 / 0.719 / 0.838 (500) | 0.549 / 0.683 / 0.792 | -0.113 / -0.036 / -0.045 | 0.890 -> 0.941 | 1.00 -> 1.00 |
| ibo | 0.629 / 0.679 / 0.789 (500) | 0.533 / 0.583 / 0.708 | -0.095 / -0.095 / -0.081 | 0.843 -> 0.844 | 1.00 -> 1.00 |
| kin | 0.628 / 0.678 / 0.764 (460) | 0.592 / 0.682 / 0.788 | -0.036 / +0.004 / +0.024 | 0.884 -> 0.935 | 1.00 -> 1.00 |
| nya | 0.584 / 0.634 / 0.721 (483) | 0.565 / 0.640 / 0.763 | -0.019 / +0.005 / +0.042 | 0.815 -> 0.904 | 0.98 -> 1.00 |
| pcm | 0.463 / 0.513 / 0.564 (500) | 0.459 / 0.509 / 0.563 | -0.003 / -0.003 / -0.001 | 0.757 -> 0.766 | 0.80 -> 0.83 |
| sna | 0.633 / 0.683 / 0.770 (500) | 0.571 / 0.646 / 0.752 | -0.062 / -0.037 / -0.018 | 0.879 -> 0.908 | 1.00 -> 1.00 |
| swh | 0.626 / 0.728 / 0.846 (500) | 0.553 / 0.721 / 0.827 | -0.073 / -0.008 / -0.019 | 0.931 -> 0.964 | 1.00 -> 1.00 |
| tsn | 0.598 / 0.648 / 0.776 (500) | 0.553 / 0.625 / 0.754 | -0.045 / -0.022 / -0.022 | 0.869 -> 0.900 | 1.00 -> 1.00 |
| twi | 0.535 / 0.585 / 0.641 (500) | 0.518 / 0.568 / 0.666 | -0.017 / -0.017 / +0.025 | 0.776 -> 0.831 | 0.93 -> 1.00 |
| xho | 0.593 / 0.659 / 0.786 (486) | 0.515 / 0.614 / 0.759 | -0.078 / -0.044 / -0.027 | 0.902 -> 0.909 | 1.00 -> 1.00 |
| yor | 0.613 / 0.663 / 0.724 (500) | 0.525 / 0.575 / 0.666 | -0.087 / -0.087 / -0.058 | 0.794 -> 0.803 | 1.00 -> 1.00 |
| zul | 0.569 / 0.619 / 0.767 (500) | 0.554 / 0.670 / 0.792 | -0.015 / +0.051 / +0.025 | 0.856 -> 0.939 | 1.00 -> 1.00 |

t_fail fell in all 12 languages (by 0.003-0.113), t_mid in 9 (by up to 0.095: yor, ibo; it rose in kin +0.004, nya +0.005, zul +0.051) and t_great in 8 (it rose in kin +0.024, nya +0.042, twi +0.025, zul +0.025): FLORES+ (Wikinews / WikiJunior / WikiVoyage) is a different domain from MAFAND news, the references are written by other translators, and the
mismatched-reference P90 sets t_fail. For most languages the same QE score therefore earns MORE reward than under v3 (yor: a QE of 0.60 was -0.04 under v3 and is +1.27 now). The first RL run will look better than the v3
numbers suggest without the policy being better; compare with chrF++ (Y:131-139), not with the reward level. The 8 languages that were uncalibrated before (afr efi ewe fon orm som sot wol): seven now have anchors, length window, lexical
thresholds, diacritic and omission blocks (afr 0.535 / 0.743 / 0.844, ewe 0.550 / 0.600 / 0.698, fon 0.471 / 0.521 / 0.573, orm 0.591 / 0.641 / 0.761, som 0.515 / 0.620 / 0.767, sot 0.538 / 0.636 / 0.758,
wol 0.539 / 0.589 / 0.673); **efi** has no FLORES+ and no MAFAND: it is written as `usable: false` (reason in the file), default anchors, **advantage weight 0** - the 360,000 efi prompts (6.9% of all draws) teach nothing until a
reference set exists. **No language is unusable on the AUC criterion** (AUC vs mismatched >= 0.70 everywhere; lowest nus 0.714, umb 0.720, kbp 0.720, kab 0.733, fon 0.736, kmb 0.739, kam 0.739, mos 0.748, pcm 0.766);
the fragile ones carry an advantage weight below 1 (section 9).

## 7. Per-chunk length penalty, lexical penalty, diacritics penalty (subtracted from the shaped chunk reward) **[v2: 7.3 is new]**

Applied to every SCORED chunk, in this order (length, lexical, then the diacritics penalty of 7.3), each floored at the failure value (R:267-281):

```
r = shape(c)
if length_penalty > 0:   ex = length_excess(lang, len(src_chunk), len(mt_chunk), slack=0.15);  if ex > 0: r = max(-2, r - length_penalty * ex)
if lexical_penalty > 0:  d  = lexical_deficit(lang, src_chunk, mt_chunk);                     if d  > 0: r = max(-2, r - lexical_penalty * d)
```
`rl.yaml` sets both penalties to 1.0 (Y:109-110); the dataclass default is 0 = off (R:82,84).

* `length_excess` (RS:176-183): window `[p05*(1-slack), p95*(1+slack)]` of the language's HUMAN `len(reference)/len(source)` ratio in characters; `ex = max(0, ln(lo/r), ln(r/hi))` with `r = len(mt)/len(src)`. Zero if the language (or `"*"`) has no window or either length is 0. The window comes from the calibration file `len_ratio {p05, p95}` (CAL:93,98), clamped to `[max(p05, 0.3), min(p95, 2.5)]` when loaded (RS:148-151; e.g. zul's measured p95 of 8.13 becomes 2.5). yor: (0.77, 1.63); pcm (0.76, 1.17); hau (0.77, 2.04) before slack.
* `lexical_deficit` (RS:185-192, `LX:63-86`). Four features between English chunk and output chunk: `anchor_recall` and `anchor_precision` over "anchors" = tokens containing a digit and capitalised words that are not sentence-initial, longer than 1 letter and not all caps (LX:47-56); `cognate_recall` and `cognate_precision` over content words = letters/digits/apostrophe tokens of >= 4 characters that are not in the 120-word stop list and contain no digit (LX:26-30, 59-60). A feature is `None` (not penalised) when there is nothing to measure: no source anchors (recall); content words need >= 3 on BOTH sides (LX:37, 72); `anchor_precision` is 1.0 when the output has no anchors (LX:69). Only `anchor_recall`, `anchor_precision`, `cognate_precision` can be penalised (LX:34; `cognate_recall` is reported but never penalised because a paraphrase legitimately drops words, LX:33).
  Thresholds are PER LANGUAGE human 10th percentiles written to the calibration file under `lexical`, kept only for features whose human median is >= 0.5 (anchors) / 0.25 (cognates), with >= 30 samples (`summarise`, LX:89-105, MIN_MEDIAN LX:36). `deficit = mean over the enabled features of max(0, p10 - f)/p10` (LX:78-86). Penalty only; it is never a bonus (LX:15-18).

### 7.3 [v2, D-31] Diacritics penalty (`diacritic_penalty: 1.0`, `Y:106`; new module DI)

Applied after the length and lexical penalties, to every SCORED chunk of a language whose human references use diacritics (R:276-281; the penalty is subtracted and floored like the other two; it never adds reward):

```
d   = diacritic density of the chunk = (combining marks + special letters) / number of characters, on the NFD form              DI:59-66 (copy of data_pipeline/sft_new/text_restore.py: SPECIAL, is_mark, diacritic_units, diacritic_density)
t   = the language's threshold: human P10 of the per-sentence density (CAL4 diacritics.threshold)                              DI:111-143
deficit = max(0, t - d) / t                         (chunks of >= 25 NFD characters; a long output cannot hide stripped chunks)   DI:168-178
deficit = 1.0  if one guard fires: a combining mark / special letter outside the language's human inventory, a mark attached to
               a non-letter, a (letter, marks) cluster that no human reference has, or d above the human P99 (over-decoration)   DI:145-165
r -= diacritic_penalty * deficit               only if the English chunk has >= 3 content words (rl/lexical.py)                R:276-281, RS:194-204, DI:180-185
```

* **Which languages qualify** (human median density >= 0.008 = `text_restore.MIN_DENSITY`, DI:44): 18 of the 42 - **bam ewe fon hau ibo kab kbp kik kin mlg mos nso nus tum twi umb wol yor**. Not diacritic languages (median below 0.008):
  afr amh bem kam kmb lin lua lug luo nya orm pcm run sna som sot ssw swh tir tsn tso xho zul (efi unknown). Nothing is computed for them; their reward is unchanged.
* **Density penalty active** (threshold > 0) for 13: bam 0.046, ewe 0.045, fon 0.146, ibo 0.017, kab 0.016, kbp 0.189, kik 0.067, mos 0.017, nso 0.006, nus 0.166, twi 0.027, wol 0.013, yor 0.182. **Guards only** for 5 (hau kin mlg tum umb:
  1% of the characters, 14-31% of the human sentences have no special letter at all, so a single sentence with density 0 is normal; the P10 is 0 and the density deficit would punish human text).
* **Yoruba needs a deviation from the literal rule.** FLORES+ Yoruba is a MIX of fully marked sentences (median density 0.24) and fully unmarked ones (15.4% of the references contain not a single tone mark or under-dot), so the
  human P10 is 0 and "P10" would never penalise stripped Yoruba. For languages where the P10 is 0 but the marked sentences are dense (median >= 0.05, so zeros cannot be chance: yor) the threshold is the P10 of the
  MARKED sentences (0.182); `threshold_source` in CAL4 says which rule applied (`p10`, `p10_of_marked_sentences`, `disabled`). Consequence: ~15% of the human Yoruba references would receive a full deficit.
* **Inventory** (allowed marks / letters / clusters) is the union of FLORES+ dev and, where it exists, MAFAND dev: MAFAND Yoruba writes the under-dot as U+0329 (combining vertical line below), FLORES+ as U+0323; with FLORES+ alone,
  10% of the human MAFAND sentences were flagged as "foreign mark" (17% as unseen cluster). Apostrophes are in SPECIAL (the restoration tasks fold them) but are never counted as foreign letters (DI:43).
* **Measured** on the FLORES+ dev references, calibrating on one half and testing on the other (diacritic languages with an active density threshold): stripping every mark / special letter is penalised in 99.8-100% of
  the sentences (mean deficit 1.00); sprinkling the language's OWN marks at random on a stripped sentence: 100% flagged (unseen clusters / over-decoration); sprinkling foreign marks: 100%; the guards flag 0.4-3.8% of the HUMAN
  held-out sentences (over-decoration 1.0-1.7% by construction of the P99, unseen clusters 0.2-2.3%, orphan 0-0.8%, foreign 0-0.9%); the density deficit is > 0 for the 10% below the P10 by construction (28% for yor, whose references are 15% unmarked; the 28% for twi is an artefact of the even / odd
  split, which separates the two Akuapem / Asante varieties that alternate by sentence in the pooled set). Domain shift (full FLORES+ calibration tested on MAFAND dev): over-decoration flags 1.6% of Yoruba, 3.9% of Igbo, 1.7% of Twi sentences.
* Diagnostics: `RewardOutput.diacritic_deficit[i]` (per sample, None when not measured), stats `diacritic/measured_rate`, `diacritic/deficit_mean`, `diacritic/penalised_rate` and one `diacritic/deficit/<iso>` per language in the
  batch (R:290-296; they reach the logger as `reward/diacritic/...`, TR:605), the sample line of the log prints `diacritic_deficit=` (TR:613-616). `diacritic_report(text, lang, stats=None, src=None)` (DI:206-223)
  returns `{lang, applies, chars, units, density, p10, threshold, p99, flags, deficit}` for any text - the plain function for the future best-translation archive.
* Limits: the P10 rule fines ~10% of correctly written human chunks by up to 1.0 (= 1/4.25 of the reward range) in every density language; the very low thresholds (nso 0.006, wol 0.013, kab 0.016, ibo 0.017, mos 0.017) mean a chunk
  with no mark at all is penalised but a chunk with a few too few is not; the density of a 25-60 character chunk is noisy.

## 8. Aggregation across chunks

```
reward_i = ( sum_k w_k * r_k  +  sum_failed w_f * (-2) ) / ( sum_k w_k + sum_failed w_f )         CP:128-142, R:282
```
`r_k` = the shaped, penalised reward of scored chunk k (section 7); `w_k` = source characters of the chunk; every failed chunk contributes the penalty (-2.0) with its weight (omitted/copied sentence: its source characters; appended text: the excess output characters; unmatched output sentences: their characters). A one-chunk plan is the same formula with a single term. A zero total weight (blank source) now falls back to the plain mean instead of raising `ZeroDivisionError` (CP:139-141). The diagnostic `qe[i]` is the UNWEIGHTED mean of the raw chunk QE values (R:283); `None` if nothing was scored.

## 9. Reliability weights (same-source)

`weights[i] = ContentShaper.reliability(lang)` (R:287, RS:172-174), or 1.0 when `train_unusable` (R:69). With `reliability_from: same_source` (Y:111, RS:126-146) the weight is the calibration field `reliability_same_source`, falling back to `reliability` (the AUC version) when absent (RS:145-147):

* `hard = (accuracy of ranking the human reference above the mismatched one + above the half-omitted one) / 2`; `reliability_same_source = clamp((hard - 0.5)/0.4, 0, 1)`, 0 if the language is unusable (CAL:136-137) - D-24.
* AUC version (`reliability`): linear in `auc_vs_mismatched` between 0.65 (-> 0) and 0.90 (-> 1) (RS:283-285, CAL:133).
* A language with `auc_vs_mismatched < 0.7` (`--min-auc`) or without any reference set (efi) is `usable: false` (CAL:55,132): the loader puts it in `unusable` and skips its anchors (RS:140-143), so it scores with the default anchors, but its weight is **0** (RS:174) and its advantage is zeroed (TR:514). A language absent from the file has weight **1.0** (RS:174).
* **[v2]** CAL4 `reliability_same_source` / AUC-based `reliability`: **1.00 (weight)** for 31 languages; below 1: umb 0.79, kam 0.82, pcm 0.83, kmb 0.86, kbp 0.86, nus 0.86, mos 0.90, fon 0.91, kab 0.96, lua 0.96; **efi 0 (unusable)**.
  Full per-language list in section 18. The 12 languages of v3 changed: hau 1.0 -> 1.0, ibo 1.0 -> 1.0, kin 1.0 -> 1.0, nya 0.98 -> 1.0, pcm 0.80 -> 0.83, sna, swh, tsn, xho, yor, zul 1.0 -> 1.0, twi 0.935 -> 1.0.
  The AUC version of the same field (`reliability`, used with `reliability_from: auc`) is lower for the hard languages (yor 0.61, twi 0.72, wol 0.57, fon 0.34, kbp 0.28, nus 0.26, umb 0.28).
* The weight does NOT change the reward number; it multiplies the standardised advantage (section 12), because group standardisation would otherwise cancel any per-language scale (TR:514; same remark in the SabiYarn reward).

## 10. End-of-turn (EOS) discipline penalty (D-26)

Implemented as failure classes 1-3 of section 3 (so the penalty is the same -2.0 floor, un-scored): truncation (R:169-171), anything but padding after the first stop id (TX:75-76, R:172-175, `Sample.post_stop_tokens` R:113, SM:65-71), turn/role marker before the stop id (TX:36,79; G:62-70; R:176-178). The text sent to QE is always the part before the first stop id (TX:68-69) so rubbish after it can never be scored. Config: `eos_discipline` (switch for classes 2-3, R:102), `eos_truncated_penalty`, `eos_post_end_penalty`, `eos_dangling_penalty` (R:103-105; `Y:115-118`); a value above the floor raises (R:148-149). With the stock sampler classes 2 cannot occur (finished rows are padded, `tests/test_rl_loop.py::test_hf_generate_pads_after_the_stop...`), classes 1 and 3 can; see D-26 for the exact coverage and limits.

## 11. Final clipping / normalisation

There is no final clip or rescale of the reward. The range [-2, +2.25] holds by construction: every shaped value is inside it (RS:90-94), the length and lexical penalties are floored at the failure penalty (R:271,275), failed chunks equal it (CP:137), the mean of numbers in a range stays in it. `reward_range = (min(lo, penalty, class penalties), hi)` (R:151) and the trainer ASSERTS every reward inside it every step (TR:511-512). Nothing is normalised across the batch (no running mean); the only normalisation is the group standardisation of the advantage.

## 12. GRPO advantage and update (grpo.py, train_rl.py)

* Group = the G = 8 completions of one prompt, consecutive rows (`group_size: 8`, Y:18; GR:34-37).
* `A_i = (r_i - mean_group) / (std_group + std_eps)`, population std (`unbiased=False`), `std_eps = 1e-4` (Y:79); `advantage_norm: mean` would skip the division (GR:40-46, Y:78). A group whose std < `flat_threshold = 1e-6` gets A = 0 for every member (GR:33, 43, 47), e.g. a group where all 8 failed.
* Then `A_i *= weights[i]` (the language weight of section 9) (TR:514). No clipping of A, no value function.
* Loss: per token `-A_i * ratio + kl_coef * k3`, `k3 = exp(d) - d - 1`, `d = log p_ref - log p`, `kl_coef = 0.05`, `kl_estimator: k3` (GR:51-53, 76-88; Y:80-81). `num_updates_per_batch: 1` (Y:20) -> exactly on-policy, `ratio = 1`, `clip_eps: null` (Y:82) so nothing clips; PPO-style clipping only exists for `num_updates_per_batch > 1` (GR:76-83, D-16). Aggregation `token`: sum over all completion tokens of the optimizer step / total completion tokens (Y:83, GR:89-90, TR:520). Gradient-norm clip 1.0 (Y:77, TR:581), lr 1e-6 with 10 warmup steps (Y:74-75), reference = the frozen SFT checkpoint (Y:5, D-16).
* The in-loop asserts (finite rewards, range, ratio == 1, KL ~ 0 at step 0) are TR:507-516,500-505.

## 13. Worked numeric examples **[v2: recomputed with CAL4; E and F are new]** (computed with the real functions; QE values invented)

Setup **[v2]**: language `yor`, CAL4 anchors (0.525, 0.575, 0.666), `len_window` (0.71, 1.26), lexical thresholds (all 0.0: the FLORES+ Yoruba references share too few names / numbers with the English for a P10 above 0, so the lexical penalty is inert for yor in CAL4), diacritic threshold 0.182, `reliability_same_source` 1.00, config as in `rl.yaml` (chunking `sentences`, `source_length`, length / lexical / diacritic penalty 1.0, omission detector on, `reliability_from same_source`, eos discipline on). Reward range printed by the code: (-2.0, 2.25).

**A. One short sentence** (source 75 characters, fake QE 0.70, shaped = `1 + (0.70 - t_mid)/(t_great - t_mid)` = 2.0253).
* v1 text, unaccented: `yor Awon agbe lati Ibadan mu agbon ewebe tutu 20 wa si oja.` -> status `ok`, 1 chunk (`single`), length ratio 0.733 inside the window, lexical deficit 0, **diacritic deficit 1.00** (density 0, below the threshold 0.182) -> reward **1.0253** = 2.0253 - 1.0 * 1.0.
* the same sentence with its tone marks: `yor Àwọn àgbẹ̀ láti Ibadan mú ẹ̀fọ́ tútù 20 wá sí ọjà.` -> diacritic deficit 0.00, reward **2.0253** (= the shaped value; stripping the marks costs exactly `diacritic_penalty`).

**B. Three chunks** (`aligned`; the Yoruba of v1 with its tone marks added).
source = `The market opens at eight o'clock in the morning. The farmers from Ibadan bring 20 baskets of fresh vegetables. Everyone is happy to see the new bridge.`
output = `Ọjà ń ṣí ní agogo mẹ́jọ òwúrọ̀. Àwọn àgbẹ̀ láti Ibadan mú ẹ̀fọ́ tútù 20 wá. Inú gbogbo ènìyàn dùn láti rí afárá tuntun tí wọ́n kọ́, tí wọ́n sì ti parí ní oṣù tó kọjá ní ìlú náà.`
fake QE: 0.72, 0.60, 0.68. Chunk 2 is below `t_mid`; chunk 3 is much longer than its source.

| chunk | weight | src chars | mt chars | ratio | QE | shaped | length excess | lexical | diacritic | chunk reward |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 49 | 49 | 31 | 0.633 | 0.72 | 2.0403 | 0.0000 | 0.0000 | 0.0000 | 2.0403 |
| 2 | 61 | 61 | 43 | 0.705 | 0.60 | 1.2710 | 0.0000 | 0.0000 | 0.0000 | 1.2710 |
| failed | 42.89 | - | - | - | - | - | - | - | - | -2.0 |
| 3 | 40 | 40 | 101 | 2.525 | 0.68 | 2.0103 | 0.5529 | 0.0000 | 0.0000 | 1.4575 |

The failed row exists because chunk 3's output is longer than the window allows: `over = 101 - 1.263*1.15*40 = 42.89 > 10% of 101` (CP:106-111), so the excess characters count as a failed chunk of weight 42.89. Aggregate = 150.01/192.89 = **0.7777** (code: 0.7777); `qe` diagnostic = 0.667; stats `chunk/aligned_rate 1.0`, `chunk/segments_per_sample 3.0`, `diacritic/deficit_mean 0.00`.

**C. End-of-turn cases on the accented output of A**: no end token -> `no_end_of_turn` **-2.0**; end token followed by a second copy of the answer -> `post_eos_rubbish` **-2.0**; `<start_of_turn>model\n` before the end token -> `dangling_marker` **-2.0**; end token followed by pad ids -> `ok`, **2.0253** (padding is benign).

**D. GRPO on a group** (shortened to 4 members for the arithmetic; real G = 8): rewards [2.0253 (A), 0.7777 (B), -2.0 (truncated), -2.0 (rubbish)]; mean -0.2993; population std 1.7570; advantages `(r-mean)/(std+1e-4)` = [1.3229, 0.6129, -0.9679, -0.9679]. With weight 1.0 (yor) unchanged; with a weight of 0.83 (pcm, CAL4) [1.0980, 0.5087, -0.8034, -0.8034]. A group whose members all failed (all -2.0) has std 0 < 1e-6 -> all advantages 0.

**E. [v2] A dropped sentence** (4 source sentences of 49/70/40/76 characters; the Yoruba translation of the third is left out; QE 0.70 everywhere):

| translation | detector | plan | reward |
|---|---|---|---|
| complete (4 sentences) | on | aligned; chunk 49, chunk 70, chunk 40, chunk 76 | **2.0253** |
| third sentence dropped | off (v1) | gale_church; chunk 49, chunk 70, chunk 117 | **2.0253** |
| third sentence dropped | on | gale_church; chunk 49, chunk 70, FAILED 40, chunk 76 | **1.3401** |

With the detector off the dropped sentence is merged with its neighbour into one 117-character chunk scored against 72 characters of output and the reward equals the complete translation's (the fake QE does not notice, and the real one only rates such a chunk as mediocre); on, the omission bead is a failed chunk of exactly the 40 dropped source characters, which carry 40/235 of the weight at -2.0.

**F. [v2] NFC** (the accented sentence of A): 50 characters / 72 UTF-8 bytes in NFC, 64 / 81 in NFD (+28% characters). Reward NFC 2.0253, NFD input 2.0253 (identical; the scorer receives the NFC text), NFD with `nfc: false` 2.0253 here (the window is wide enough; the length-ratio failure of `tests/test_reward_nfc.py` shows the case where it flips the status).

## 14. All weights and thresholds (config key, dataclass default, `rl.yaml` value)

| quantity | key | default | rl.yaml | cite |
|---|---|---|---|---|
| shaping preset / range | `preset` | `repo` [-2, +2.25] | `repo` | R:52, RS:58-62, Y:88 |
| failure/negative scale | `neg_scale` | 2.0 | 2.0 | R:55, Y:92 |
| failure penalty | `penalty` | `None` = -neg_scale | null | R:60, Y:93 |
| fail floor of `t_fail` | constant `FAIL_FLOOR` (default of `fail_floor`, see below) | 0.46 | - | RS:55 |
| default anchors | `default_anchors` | (0.46, 0.60, 0.75) | - | R:56, RS:56 |
| mode | `mode` | `calibrated` | `calibrated` | R:53, Y:89 |
| prefix policy | `prefix_policy` | `require` | `require` | R:61, Y:94 |
| copy test | `copy_jaccard` | 0.9 | (default) | R:62, G:19-30 |
| length-ratio hard failure | `min_len_ratio`/`max_len_ratio` | 0.2 / 5.0 | (default) | R:63-64, G:53-59 |
| repetition guard | `repetition_guard` (+ fixed thresholds 4 / 0.3 / 8 words / 0.18 / 80 B) | True | (default) | R:65, G:33-50 |
| GlotLID **[v3]** | `lid_enabled`, `lid_min_chars`, `lid_model_path`, `lid_disable` | False, 15, null, [] | **true**, 15, rental path, [] | R:83 |
| penalise special tokens | `penalize_special_tokens` | False | (default) | R:68 |
| QE truncation (stat only) | `max_comet_tokens` | 510 | 510 | R:70, Y:96 |
| chunking | `chunking` | `off` | `sentences` | R:75, Y:98 |
| chunk weights | `chunk_agg` | `source_length` | `source_length` | R:76, Y:107 |
| count-mismatch handling | `chunk_align` | `gale_church` | (default) | R:77 |
| appended-text failed chunk | `extra_output_penalty`, `extra_ratio_limit`, `length_slack` | True, 1.8, 0.15 | true, (defaults) | R:80-83, Y:108 |
| unmatched-output / appended thresholds (fixed) | 5% of output / 10% of chunk output | - | - | CP:93,109 |
| length penalty per log-unit | `length_penalty` | 0.0 | 1.0 | R:82, Y:109 |
| lexical penalty per unit deficit | `lexical_penalty` | 0.0 | 1.0 | R:84, Y:110 |
| lexical thresholds | calibration `lexical.<feature>.p10` (human P10) | - | per language | LX:89-105, RS:152-153 |
| reliability source | `reliability_from` | `auc` | `same_source` | R:98, Y:111 |
| reliability scales | AUC 0.65 -> 0, 0.90 -> 1; same-source `(hard-0.5)/0.4`; unusable if AUC < 0.7 | - | - | RS:283-285, CAL:55,132-137 |
| Gale-Church | priors, `_S2 = 6.8`, band `max(8, 0.25 max(n,m))`, floor 1e-12 | fixed | - | SEG:81-83,115,137 |
| len-window clamp | p05 >= 0.3, p95 <= 2.5 | fixed | - | RS:151 |
| EOS discipline | `eos_discipline` | True | true | R:102, Y:115 |
| EOS class values | `eos_truncated_penalty`, `eos_post_end_penalty`, `eos_dangling_penalty` | None (= penalty) | null | R:103-105, Y:116-118 |
| QE model / batch | `scorer.model_id`, `scorer.batch_size` | africomet-qe-stl-1.1, 32 | same | `Y:121-123`, TR:113 |
| GRPO | `group_size` 8, `prompts_per_step` 16, `temperature` 1.0, `top_p` 1.0, `top_k` 0, `num_updates_per_batch` 1 | as listed | same | TR:56-62, Y:17-23 |
| advantage | `advantage_norm` `mean_std`, `std_eps` 1e-4, flat threshold 1e-6 | as listed | same | TR:102-103, GR:33, Y:78-79 |
| KL / loss | `kl_coef` 0.05, `kl_estimator` k3, `loss_agg` token, `clip_eps` null | as listed | same | TR:104-108, Y:80-83 |
| optimiser | `learning_rate` 1e-6, `warmup_steps` 10, `grad_clip` 1.0 | TR:96-101 (lr default 1e-6) | same | Y:74-77 |
| generation budget | `max_new_tokens_by_bin` b64 384 / b256 1280 / b512 2304 / b1024 3584 / mixed 3584 | {} | as listed | TR:91, Y:59-64 |
| NFC of completion and source **[v2]** | `nfc` | True | true | R:86, Y:97, TX:59-77 |
| diacritics penalty per unit deficit **[v2]** | `diacritic_penalty` | 0.0 | 1.0 | R:89, Y:106, DI:168-178 |
| diacritic guards (fixed) | min 25 NFD characters for the density tests, >= 3 English content words, `DENSE_MIN` 0.05, `MIN_DENSITY` 0.008 | - | - | DI:44-48,108, RS:194-204 |
| omission detector **[v2]** | `omission_detector` | False | true | R:92, Y:102 |
| omission / extra bead cost **[v2]** | `omission_cost` | None = the language's CAL4 `omission.cost` (4 / 5 / 6 / 8 / 12) | null | R:93, RS:206-216, Y:103 |
| shortfall check **[v2]** | `omission_short_frac`, `omission_short_min` | 0.7, 0.15 | 0.7, 0.15 | R:94-95, CP:98-105, Y:104-105 |
| lowest `t_fail` **[v2]** | `fail_floor` | 0.46 (= the old constant) | 0.0 | R:97,137, RS:107-111,218-223, Y:91 |
| calibration file **[v2]** | `calibration_path` | None | `calibration/shaping_flores_dev_v4.json` (replaces `calibration/shaping.json`) | R:54, Y:90 |

## 15. Things that could be reward-hacked, and what guards them

| exploit | guard | gap |
|---|---|---|
| Copy the English source (QE sees a fluent source) | `copy_source` hard failure (G:19-30, R:190); per-chunk copy = failed chunk when several chunks (CP:112); the lexical penalty is built so copying cannot earn reward (LX:15-18) | a light paraphrase of the English that stays below Jaccard 0.9 is only caught by QE, GlotLID (off by default) and the human-calibrated lexical precision |
| Answer in the wrong language / wrong tag | `prefix_policy: require` (R:182-185); QE; **[v3]** GlotLID ON (section 22) | wrong language with the right tag: 75 of 77 languages are checked (not pcm kon); siblings of a language are accepted |
| Markup, turn headers, special tokens instead of language | cut at first stop id + special strings removed (TX:68-78); `dangling_marker` (R:176-178); optional `special_tokens` | `special_tokens` is off: a stripped `<unused5>` etc. is just counted |
| Never stopping / running into the budget | `no_end_of_turn` -2.0 (R:169-171); per-bin budgets (Y:59-64); selection requires `finished_rate >= 0.95` (Y:135, TR:136) | budget too tight punishes the policy for the sampler (see sampling limits in D-26) |
| Junk after the end token | `post_eos_rubbish` (R:172-175); text cut at the stop id (TX:68) | cannot fire with the stock sampler (it pads) - armed for samplers that keep generating |
| Loops / stuttering | `degenerate` (G:33-50) | |
| Very short output / dropped sentences | `length_ratio` < 0.2 (G:53-59); window penalty (RS:176-183); **[v2]** omission bead = failed chunk with the dropped source characters, shortfall check for merged beads and one-sentence outputs (section 4.5); QE | recall 56% of the dropped sentences / 77% of the documents / 89% of the dropped characters on FLORES+ (section 4.5); a drop plus an extra sentence (equal counts) is not detected; dropped partial sentences only by QE and the length window |
| Padding with unrelated/hallucinated text (QE accuracy 0.47 = chance, D-24) | `extra_output_penalty`: unmatched output sentences and over-window chunk text = failed chunks (CP:89-111); length-window penalty; lexical precision | text that stays inside the human window passes (only QE and lexical precision see it) |
| Gaming the sentence splitter (no full stops -> one output sentence -> `whole`) | `whole` still gets the in-chunk appended-text rule (CP:106-111), the hard length ratio, the window penalty | the omitted-sentence/copy rules need > 1 pair (CP:112); a very long source scored as one chunk is truncated at 510 tokens by the encoder (F9) **[v2]** the shortfall check (4.5) flags a one-sentence output that is far too short for several source sentences |
| Decomposed (NFD) diacritics to move character counts | **[v2]** NFC of the completion and the source before every rule (R:161,166, TX:59-77); pinned by `tests/test_reward_nfc.py` | none left; the sampler can still emit NFD, it just does not matter |
| Over-optimising QE above human level | slope above `t_great` is 0.25 per (1 - t_great) (RS:20-26); KL 0.05 to the SFT model (GR:51-53, Y:80); benchmark chrF++ model selection against a human-reference dev sample (`Y:127-135`; `select_by: benchmark_chrf` Y:133) | |
| Languages where QE is blind | `usable:false` -> weight 0 (RS:174); weight = same-source accuracy (RS:145-147) | **[v2]** every language has an entry now; efi is explicit `usable: false`, weight 0 (F11 resolved) |
| Noise amplification by narrow bands (pcm: 0.01 QE = 0.2 reward) | reliability weight 0.83 for pcm (CAL4), 0.79-0.91 for umb kam kmb kbp nus mos fon; group standardisation | F10 |
| Stripped or invented diacritics **[v2]** | `diacritic_penalty` (section 7.3): density deficit against the human P10, guards for foreign / orphan / unseen / over-decorated marks | the 5 languages with sparse special letters (hau kin mlg tum umb) have guards only; ~10% of correct human chunks get a deficit by construction; efi and the 23 non-diacritic languages are not judged |

## 16. Findings: things that look wrong or inconsistent

* **F1 - No NFC on the hypothesis. RESOLVED [v2] (D-29).** Was: the reward scored and measured the text as sampled, FLORES+ Yoruba / Fon references are 30% / 32% non-NFC, NFD Yoruba is 22.6% longer in characters. Now both completion and source are NFC before any rule (section 2, item 4).
* **F2 - Omitted sentences were practically never a "failed chunk". RESOLVED [v2] (D-30)** for count-differing pairs: sentence recall 0% -> 56%, document detection 0% -> 77%, dropped-character coverage 0% -> 89%, extra sentences 42% -> 93% at 0.6% false alarms on 7 of 8 languages (section 4.5). Still open: pairs with EQUAL sentence counts are paired i-th to i-th (a drop plus an extra sentence is invisible), and the shortfall check cannot see a dropped half-sentence.
* **F3 - Stale comment in `rl.yaml`** ("when the sentence counts differ the whole pair is scored as one segment"). Fixed [v2]: the comment now describes the Gale-Church alignment, the omission detector and the `whole` case (Y:98-101).
* **F4 - DECISIONS.md numbering/staleness.** Two entries are called D-22 (offline testing; reverse direction). D-17 still says prompts are filtered to `src_afr_tokens <= 256` and `max_new_tokens 256` (superseded by D-23 and `Y:46,59-64`); `data.py:MAX_SRC_AFR_TOKENS = 256` and its docstring are stale against `max_src_tokens: 1024`. D-17's "Nothing after the first `<end_of_turn>` exists for the reward" was amended by D-26.
* **F5 - `reward_shaping.py` documents the wrong `t_fail`.** RS:9,12-13 and D-17 say `t_fail` = P10 of human QE, "further raised" to the P90 of mismatched QE. The code that produces the file (`pair_anchors`, RS:288-296, CAL:101) uses `max(0.46, P90 mismatched)` and never uses the human P10. Behaviour is fine; the text is not.
* **F6 - Weight units are mixed.** Normal chunks weigh SOURCE characters, the two "extra text" failed chunks weigh OUTPUT characters (CP:94,97,110). Harmless but a long hallucination can outweigh a short sentence; `matched` also counts the separators of empty chunks (CP:91).
* **F7 - Division by zero removed.** `tot / wsum` raised on a blank source (old R:261); `aggregate` falls back to the plain mean (CP:139-141). Numerics for non-blank sources are bit-identical (checked on 120 samples x 6 configurations).
* **F8 - Misleading stat in chunk mode.** `comet_mt_truncated_rate` (R:319-321) measures the UNSPLIT text, so it is high for long windows even though the chunks are fine; use `chunk/mt_truncated_rate` (R:305-307). Neither looks at the SOURCE side (a long unsplittable sentence or the `whole` fallback is truncated at 510 tokens unseen).
* **F9 - `whole` fallback can exceed the encoder limit** for b1024/b2048 when the output has one sentence (SEG:189-190).
* **F10 - Narrow, sensitive bands.** The `+0.05` spacing floors of `pair_anchors` (RS:288-296) are active for `t_mid` in 21 of the 41 calibrated languages (t_mid = t_fail + 0.05 because the human median is not above the mismatched P90): bam bem ewe fon ibo kab kam kbp kik kmb lua lug luo mos nus orm pcm twi umb wol yor (e.g. pcm 0.459 / 0.509 / 0.563, fon 0.471 / 0.521 / 0.573, yor 0.525 / 0.575 / 0.666), and for `t_great` in kbp only. In those languages the 0 .. +1 band is 0.05 of QE wide (0.01 QE = 0.2 reward) and QE separates human from mismatched only at AUC 0.71-0.85; the advantage weights (0.79-0.91 for umb kam pcm kmb kbp nus mos fon) only partly compensate.
* **F11 - Weight 1.0 for uncalibrated languages. RESOLVED [v2] (D-32)** except efi: all 41 languages with a reference set are calibrated (section 18); efi is written as `usable: false`, advantage weight **0**, default anchors, until someone provides a reference set. Side effect: efi is still 6.9% of the prompt draws (section 19).
* **F12 - Defaults vs yaml.** The dataclass defaults (`chunking off`, `length_penalty 0`, `lexical_penalty 0`, `reliability_from auc`) differ from `rl.yaml`; a run that does not use the yaml silently gets a different reward (R:75,82,84,98 vs Y:98,109-111).
* **F13 - Unbounded raw QE.** There is no final activation (COMET `referenceless.py:93`), and the raw `c` is not clipped (the SabiYarn reward clips to [0, 1]); the shaping handles it (RS:90-94), but `qe_mean_scored` can leave [0,1].
* **F14 - One consequence of D-26 to be aware of:** `has_role_tail` flags a translation that legitimately ends on a line consisting of `model`, `user` or `assistant` unless the source does too (G:62-70); this is deliberate and rare.
* **F15 - [v2] The old `appended_text` rule flags 2-10% of FAITHFUL human documents.** In the false-alarm runs of section 4.5 the v1 planner (no detector) marks a failed chunk in 3.1% (yor), 3.7% (hau), 3.9% (ibo), 1.9% (swh), 3.0% (zul), 8.8% (amh), 6.4% (kin), 12.5% (ewe) of the untouched human documents, almost entirely through the in-chunk rule `over > 10% of the chunk output` with `lim = p95 x 1.15` (CP:106-111): by construction ~5% of human sentences exceed their language's p95 ratio, and the 15% slack only absorbs part of that. The rule costs a faithful long chunk a failed chunk of the excess characters. Not changed (it is D-24, outside this task); the new CAL4 windows are narrower than the v3 ones (yor 0.71-1.26 instead of 0.77-1.63), so it fires more often than before for the same text.
* **F16 - [v2] FLORES+ is not a clean reference everywhere.** Yoruba: 15.4% of the references carry no tone mark at all and 29.8% are not NFC; Twi alternates two varieties (Akuapem / Asante) by sentence in the pooled set; Ewe: only 45-55% of the 3-12 sentence windows have the same number of sentences as the English, and a third of the flagged "false alarms" are source sentences the Ewe reference really leaves out (section 4.5); Amharic uses `፡፡` as a full stop in places, which the sentence splitter (`SEG:22`, `።` only) does not split on (amh FA 2.6%). The calibration treats all of this as human behaviour; it makes the reward more lenient for those languages, not stricter.
* **F17 - [v2] Anchors moved mostly down versus v3** (t_fail -0.003 .. -0.113 in all 12, t_mid and t_great down in 8-9, up in kin / nya / zul / twi; domain: FLORES+ vs MAFAND news, more sentences, other translators). For most languages the same QE gives more reward than under v3 (section 6); track chrF++ on the dev sample, not the reward level, when comparing runs with v3-calibrated ones.
* **F18 - [v2] The 22 added languages get 0.38% of the prompt draws each** (20,000 prompts of 5.24M, `language_alpha: 1.0`, no weight): ~14 prompts of the 3,600 a 300-step run draws, against ~82 for a standard old language. They are RL-eligible and calibrated but will not be learned in this schedule unless `language_weights` (e.g. 6.0 = 2.3%) is raised (section 19). Not changed (training schedule).
* **F20 - [v2] The launcher wiped the reward block (found while confirming the owner's point 5; FIXED).** `scripts/vast_train_rl.sh` passed `reward='{"preset":..,"calibration_path":..,"mode":..}'` and `train_rl.main` merged overrides with a shallow `dict.update` (old TR:738), so launching through the script REPLACED the whole `reward:` block of `rl.yaml`: chunking, `extra_output_penalty`, length / lexical penalties, `reliability_from`, the end-of-turn discipline and now NFC / omission / diacritics all fell back to their dataclass defaults (off). `train_rl.apply_overrides` now merges `reward` and `scorer` dict overrides key by key, and the script only overrides `calibration_path` / `preset` when `CAL` / `PRESET` are set (`tests/test_configs.py::test_a_reward_override_is_merged_into_the_yaml_block_not_substituted`). Direct `python -m gemma_mt.rl.train_rl --config configs/rl.yaml` was never affected.
* **F19 - [v2] Low-density diacritic languages.** hau kin mlg tum umb have `applies: true` (median density >= 0.008) but no usable density threshold (P10 = 0); only the guards run. nso (0.006), wol (0.013), kab (0.016), ibo (0.017), mos (0.017) have thresholds so low that only chunks with (almost) no marks at all are penalised. Yoruba's threshold (0.182) comes from the marked sentences because 15% of the references are unmarked (section 7.3).

## 17. Appendix - what was ported to the SabiYarn RL reward (for completeness)

`rl/chunking.py` (file-for-file port of SEG + CP + the copy guard, equality asserted by `tests/test_chunking.py`), `StructuredTranslationReward` in `rl/rewards.py` (chunking is applied to the content term only; scaffold term, gate, copy rule and length guard unchanged; one-sentence prompts are scored exactly like before, bit-identical on 150 random items for the four shaping modes), `rl/eos_discipline.py` (text-level `</s>` rules), `rl/gen_limits.py` (per-bin `max_new_tokens`: 2.5 x the bin's p99 prompt tokens + 48, multiple of 64, cap 8192, clipped by `max_seq_len` - prompt; b64 320, b256 704, b512 1344, b1024 2496, b2048 4672), config keys `reward_chunking ... reward_eos_*`, `max_new_tokens_*` in `rl/config.py`, example `rlhf/configs/rl_bin_v2.yaml`.

**[v2] Not ported to SabiYarn:** NFC at the start of the reward path, the diacritics score and the per-language fail floor exist only in the Gemma reward. The omission detector is ADDITIVE in `segments.py` / `chunk_plan.py` (new optional arguments `length_model`, `short_frac`, `short_min`; `align_beads`, `LengthModel`): with the defaults the functions return exactly what they returned in v1, and `tests/test_chunking.py` of the research repo (the equality test against the SabiYarn port) still passes (4 passed on 2026-10-17). D-27's rule "change both or neither" therefore holds for the old behaviour; porting the detector to `rl/chunking.py` is a separate decision.

## 18. [v2] The 42 RL languages: calibration (CAL4)

Columns: data set the anchors come from, n sentences, anchors, AUC human-vs-mismatched (QE validity), advantage weight (`reliability_same_source`, the weight of the language's advantages), human length-ratio window p05-p95 (clamped to 0.3-2.5) and the corpus ratio used by the omission model, diacritics (threshold of the density penalty), omission cost. `old` = in the first 20 languages, `new` = added 2026-10-17. All 41 languages with data are usable (AUC >= 0.70); efi has no data.

| tag | name | set | data | n | t_fail / t_mid / t_great | AUC | advantage weight | human len ratio p05-p95 (corpus) | diacritics | omission cost |
|---|---|---|---|---|---|---|---|---|---|---|
| yor | Yoruba | old | FLORES+ dev yor_Latn | 997 | 0.525 / 0.575 / 0.666 | 0.803 | 1.00 | 0.71-1.26 (0.96) | yes, threshold 0.182 | 4 |
| hau | Hausa | old | FLORES+ dev hau_Latn | 997 | 0.549 / 0.683 / 0.792 | 0.941 | 1.00 | 0.87-1.36 (1.08) | yes, guards only (density too sparse) | 4 |
| ibo | Igbo | old | FLORES+ dev ibo_Latn | 997 | 0.533 / 0.583 / 0.708 | 0.844 | 1.00 | 0.76-1.30 (1.01) | yes, threshold 0.017 | 4 |
| pcm | Pidgin | old | MAFAND dev (no FLORES+) | 997 | 0.459 / 0.509 / 0.563 | 0.766 | 0.83 | 0.76-1.18 (0.95) | no | 4 (default) |
| efi | Efik | old | none (no FLORES+, no MAFAND) | 0 | default 0.46/0.60/0.75 (unused) | - | **0** (unusable) | - | - | - |
| twi | Twi | old | FLORES+ dev twi_Latn | 997 | 0.518 / 0.568 / 0.666 | 0.831 | 1.00 | 0.75-1.31 (0.99) | yes, threshold 0.027 | 4 |
| ewe | Ewe | old | FLORES+ dev ewe_Latn | 997 | 0.550 / 0.600 / 0.698 | 0.835 | 1.00 | 0.62-1.40 (0.98) | yes, threshold 0.045 | 4 (FA target not met) |
| fon | Fon | old | FLORES+ dev fon_Latn | 997 | 0.471 / 0.521 / 0.573 | 0.736 | 0.91 | 0.67-1.40 (1.00) | yes, threshold 0.146 | 12 (FA target not met) |
| wol | Wolof | old | FLORES+ dev wol_Latn | 997 | 0.539 / 0.589 / 0.673 | 0.792 | 1.00 | 0.73-1.22 (0.96) | yes, threshold 0.013 | 4 |
| swh | Swahili | old | FLORES+ dev swh_Latn | 997 | 0.553 / 0.721 / 0.827 | 0.964 | 1.00 | 0.85-1.28 (1.05) | no | 4 |
| kin | Kinyarwanda | old | FLORES+ dev kin_Latn | 997 | 0.592 / 0.682 / 0.788 | 0.935 | 1.00 | 0.88-1.43 (1.12) | yes, guards only (density too sparse) | 4 |
| som | Somali | old | FLORES+ dev som_Latn | 997 | 0.515 / 0.620 / 0.767 | 0.919 | 1.00 | 0.89-1.44 (1.14) | no | 4 |
| orm | Oromo | old | FLORES+ dev gaz_Latn | 997 | 0.591 / 0.641 / 0.761 | 0.848 | 1.00 | 0.93-1.50 (1.19) | no | 4 |
| amh | Amharic | new | FLORES+ dev amh_Ethi | 997 | 0.575 / 0.727 / 0.839 | 0.964 | 1.00 | 0.53-0.87 (0.68) | no | 4 |
| tir | Tigrinya | new | FLORES+ dev tir_Ethi | 997 | 0.596 / 0.660 / 0.779 | 0.893 | 1.00 | 0.52-0.92 (0.69) | no | 4 |
| lug | Luganda | new | FLORES+ dev lug_Latn | 997 | 0.529 / 0.579 / 0.694 | 0.818 | 1.00 | 0.78-1.30 (1.02) | no | 6 |
| lin | Lingala | new | FLORES+ dev lin_Latn | 997 | 0.500 / 0.569 / 0.723 | 0.881 | 1.00 | 0.86-1.36 (1.08) | no | 4 |
| zul | Zulu | old | FLORES+ dev zul_Latn | 997 | 0.554 / 0.670 / 0.792 | 0.939 | 1.00 | 0.88-1.41 (1.12) | no | 4 |
| xho | Xhosa | old | FLORES+ dev xho_Latn | 997 | 0.515 / 0.614 / 0.759 | 0.909 | 1.00 | 0.81-1.37 (1.07) | no | 4 |
| afr | Afrikaans | old | FLORES+ dev afr_Latn | 997 | 0.535 / 0.743 / 0.844 | 0.982 | 1.00 | 0.91-1.24 (1.06) | no | 4 |
| sna | Shona | old | FLORES+ dev sna_Latn | 997 | 0.571 / 0.646 / 0.752 | 0.908 | 1.00 | 0.84-1.49 (1.12) | no | 4 |
| nya | Chichewa | old | FLORES+ dev nya_Latn | 997 | 0.565 / 0.640 / 0.763 | 0.904 | 1.00 | 0.84-1.49 (1.13) | no | 5 |
| sot | Sesotho | old | FLORES+ dev sot_Latn | 997 | 0.538 / 0.636 / 0.758 | 0.921 | 1.00 | 0.95-1.54 (1.21) | no | 4 |
| tsn | Setswana | old | FLORES+ dev tsn_Latn | 997 | 0.553 / 0.625 / 0.754 | 0.900 | 1.00 | 0.93-1.67 (1.26) | no | 4 |
| mlg | Malagasy | new | FLORES+ dev plt_Latn | 997 | 0.568 / 0.690 / 0.804 | 0.947 | 1.00 | 0.95-1.56 (1.23) | yes, guards only (density too sparse) | 4 |
| bem | Bemba | new | FLORES+ dev bem_Latn | 997 | 0.523 / 0.573 / 0.696 | 0.860 | 1.00 | 0.94-1.62 (1.24) | no | 4 |
| tum | Tumbuka | new | FLORES+ dev tum_Latn | 997 | 0.595 / 0.663 / 0.768 | 0.897 | 1.00 | 0.92-1.84 (1.31) | yes, guards only (density too sparse) | 4 |
| run | Kirundi | new | FLORES+ dev run_Latn | 997 | 0.575 / 0.650 / 0.763 | 0.912 | 1.00 | 0.81-1.49 (1.12) | no | 4 |
| tso | Tsonga | new | FLORES+ dev tso_Latn | 997 | 0.569 / 0.639 / 0.762 | 0.895 | 1.00 | 0.91-1.53 (1.20) | no | 4 |
| nso | Sepedi | new | FLORES+ dev nso_Latn | 997 | 0.521 / 0.608 / 0.754 | 0.897 | 1.00 | 0.93-1.42 (1.16) | yes, threshold 0.006 | 4 |
| ssw | Swati | new | FLORES+ dev ssw_Latn | 997 | 0.569 / 0.643 / 0.772 | 0.893 | 1.00 | 0.85-1.46 (1.12) | no | 4 |
| umb | Umbundu | new | FLORES+ dev umb_Latn | 997 | 0.471 / 0.521 / 0.595 | 0.720 | 0.79 | 0.68-1.36 (1.01) | yes, guards only (density too sparse) | 4 |
| kmb | Kimbundu | new | FLORES+ dev kmb_Latn | 997 | 0.469 / 0.519 / 0.596 | 0.739 | 0.86 | 0.82-1.45 (1.11) | no | 4 |
| mos | Mossi | new | FLORES+ dev mos_Latn | 997 | 0.536 / 0.586 / 0.654 | 0.748 | 0.90 | 0.68-1.30 (0.96) | yes, threshold 0.017 | 4 |
| kik | Kikuyu | new | FLORES+ dev kik_Latn | 997 | 0.523 / 0.573 / 0.693 | 0.854 | 1.00 | 0.87-1.62 (1.17) | yes, threshold 0.067 | 6 |
| luo | Luo | new | FLORES+ dev luo_Latn | 997 | 0.580 / 0.630 / 0.746 | 0.852 | 1.00 | 0.77-1.38 (1.05) | no | 4 |
| kam | Kamba | new | FLORES+ dev kam_Latn | 997 | 0.452 / 0.502 / 0.610 | 0.739 | 0.82 | 0.72-1.30 (0.98) | no | 4 |
| bam | Bambara | new | FLORES+ dev bam_Latn | 997 | 0.469 / 0.519 / 0.629 | 0.809 | 1.00 | 0.72-1.25 (0.96) | yes, threshold 0.046 | 4 |
| kbp | Kabiye | new | FLORES+ dev kbp_Latn | 997 | 0.543 / 0.593 / 0.643 | 0.720 | 0.86 | 0.73-1.56 (1.10) | yes, threshold 0.189 | 8 |
| kab | Kabyle | new | FLORES+ dev kab_Latn | 997 | 0.493 / 0.543 / 0.597 | 0.733 | 0.96 | 0.77-1.26 (0.99) | yes, threshold 0.016 | 4 |
| nus | Nuer | new | FLORES+ dev nus_Latn | 997 | 0.461 / 0.511 / 0.565 | 0.714 | 0.86 | 0.75-1.42 (1.08) | yes, threshold 0.166 | 12 |
| lua | Luba-Kasai | new | FLORES+ dev lua_Latn | 997 | 0.438 / 0.488 / 0.634 | 0.812 | 0.96 | 0.81-1.39 (1.08) | no | 4 |


## 19. [v2] Cost-relevant facts after the language change (nothing in the training schedule was changed)

* Per-step workload does not depend on the number of languages: `prompts_per_step_by_bin` b64 32 / b256 16 / b512 8 / b1024 4 / mixed 8 prompts x G = 8 = 256 / 128 / 64 / 32 / 64 completions per step (Y:51-56); the 300-step schedule (40 / 60 / 60 / 60 steps of b64 / b256 / b512 / b1024, then 80 mixed) draws
  1,280 + 960 + 480 + 240 + 640 = **3,600 prompts = 28,800 completions**. The `max_new_tokens` per bin (Y:59-64) are unchanged; the Gemma-token length of the new languages' prompts is the same distribution (the prompts are English; windows are shared between languages).
* Prompt files: 22 new `<iso>.parquet` of 20,000 rows (5,000 per bin) each = **440,000 prompts**, next to the 20 old files (4.8M rows) in `rl_prompts_v2/` and with Gemma-token columns in `rl_prompts_v2_tok/` (section 20). Total pool 5.24M prompts. With `language_alpha: 1.0` the draw shares are:

| group | prompts each | share each | prompts drawn in a 300-step run (each) |
|---|---|---|---|
| pcm | 600,000 | 11.45% | ~412 |
| fon efi ewe twi wol sot orm nya | 360,000 | 6.87% | ~247 |
| the other 11 old languages | 120,000 | 2.29% | ~82 |
| the 22 added languages | 20,000 | 0.38% | ~14 |

  efi (advantage weight 0) takes 6.9% of the draws = ~247 prompts = 6.9% of the sampling and QE cost for no gradient (its groups have advantage 0). `language_weights` multiplies the draw weight: 6.0 on the added languages makes each as frequent as a standard old language (1.6% each; standard old 1.6%, boosted 4.8%, pcm 8.1%).
* Reward CPU time: the language-aware alignment costs 0.4 / 2.3 / 10 ms per document of 5 / 12 / 30 sentences (v1: 0.26 / 1.3 / 5.8 ms) and only runs when the sentence counts differ; the diacritic score is microseconds per chunk. QE time is unchanged (the shortfall and omission beads send FEWER chunks to QE, never more).
* Calibration: 41 languages x 997 pairs x 5 scorings (human, mismatched, copy, half, appended) = 204k QE pair-scores in 7 minutes on the shared card (fp16, batch 32, <= 6 GB cap).

## 20. [v2] How the files were produced (reproducible)

* **Languages** (`LG`): the five main-table rows amh tir lug lin mlg and 17 `_SFT_ONLY` rows (run bem tso nso tum mos kik luo bam kmb ssw lua umb kab kam kbp nus) are `africomet_status = VERIFIED` with `africomet_code` = the tag, FLORES+ codes amh_Ethi tir_Ethi lin_Latn lug_Latn plt_Latn (mlg) run_Latn ... nus_Latn (existing `flores` field kept or set), GlotLID labels checked against the `cis-lmu/glotlid` `model_v3.bin` (2,102 labels): amh_Ethi, tir_Ethi, plt_Latn for Malagasy (there is no mlg_Latn label), the others `<iso>_Latn`. Justification: the AfriCOMET-QE-1.1 card lists the same 76 languages as the encoder, COMET QE takes text only,
  and the validity study (`reports/ssa_comet_vs_africomet_2026-10-15.md`, section 4) measured human-vs-mismatched AUC >= 0.80 for the first 17 and 0.70-0.80 for umb kab kam kbp nus; on the full 997 dev sentences the AUCs are amh .964 tir .893 lin .881 lug .818 mlg .947 run .912 bem .860 tso .895 nso .897 tum .897 mos .748 kik .854 luo .852 bam .809 kmb .739 ssw .893 lua .812 | umb .720 kab .733 kam .739 kbp .720 nus .714. Not added: dyu sag nqo (AUC < 0.70), fuv knc taq cjk dik (no SFT coverage).
  `ALL_ISOS` order (= language ids stored in packed data) is unchanged (verified identical to the previous table). `SUPPORTED_RL` has 42 tags.
* **Prompts** (old box, CPU only, `nice -n 10 taskset -c 8,9,10`): `gemma_rl_assemble.py --only <22 isos> --per-language 20000` (no boost; it loaded the cached filtered window pool of 1,424,598 windows, 12 s) -> `/workspace/sft_build/gemma/rl_prompts_v2/<iso>.parquet`;
  `gemma_rl_tokenize_22new.py` (a copy of `gemma_rl_tokenize.py` that takes the file list and writes `report_22new.json`) -> `/workspace/sft_build/gemma/rl_prompts_v2_tok/<iso>.parquet` + `report_22new.json` (440,000 rows, 0 empty / iso-mismatch / pivot rows, 49 rows above `max_src_tokens` 1024);
  both streamed `ssh old tar c | ssh new tar x` to `/workspace/sft_build/gemma/rl_prompts_v2{,_tok}/` on the new instance (md5 of all 46 files verified). The old `stats.json` was restored (the new rows' stats are `stats_22new.json`); `languages.py` on the old box has a `.bak_20261017`.
* **Dev data**: `scripts/build_flores_dev_all.py` -> `/workspace/sft_build/calibration/flores_dev_all_v2.jsonl` (FLORES+ DEV only; the 22 new languages from the SSA study's dev downloads, 11 old ones from `bench_dev.jsonl`'s `flores_plus_dev` rows, Twi from the Hub (two varieties, alternating by id parity), pcm from MAFAND dev; MAFAND dev rows of the other languages are inventory-only).
* **Calibration**: `python -m gemma_mt.rl.calibrate_shaping --bench-file flores_dev_all_v2.jsonl --max-per-language 997 --fail-floor 0.0 --gpu-mem-gb 6 --batch-size 32 --resume --out /workspace/sft_build/calibration/shaping.json` (new instance; `transformers 4.57` in a private `pylibs_tf4` because comet 2.2.7 does not run on transformers 5), then
  `python -m gemma_mt.rl.omission_eval --calibrate-omission shaping.json --languages ...` (CPU) for the per-language omission cost. Result copied to `calibration/shaping_flores_dev_v4.json` (112 KB) in the repo and kept at `/workspace/sft_build/calibration/shaping.json` (the GPU-only version before the omission pass: `shaping_gpu_raw.json`).
* **Tests**: 248 pass / 3 skipped (was 199 / 3; 247 / 3 with the real pytest on the new instance). New: `tests/test_diacritics.py` (15), `tests/test_reward_nfc.py` (4), 8 detector tests in `tests/test_chunk_plan.py`, calibration / loader / yaml tests in `test_reward_shaping.py` and `test_configs.py`, language tests.

## 21. [v2] Open problems

1. **Ewe (and, less, fon / kbp / lug / nus / kik)**: FLORES+ references drop and merge sentences; the omission detector's false-alarm target (2%) cannot be met for ewe (14.3% at cost 4, 10.6% at 12; cost 4 kept) and is met for fon only at 2.1% (cost 12). For these languages "false alarms" are partly real omissions of the reference, and the penalty for them is partly deserved by the reference, not the policy.
2. **efi** has no reference set: weight 0, 6.9% of the draws wasted. Options: lower efi's prompt count, or build a small human-checked dev set.
3. **Prompt shares of the 22 new languages** (0.38% each) are too small to learn from in 300 steps; decide `language_weights`.
4. **Equal sentence counts** after a drop plus an extra sentence are not detected (paired i-th to i-th by design: bit-identity with v1).
5. **Diacritic threshold for hau kin mlg tum umb** is disabled (density too sparse); a lexicon-based check (words that change meaning without the special letter) would be needed.
6. **Amharic / Tigrinya full stop** `፡፡` is not a sentence boundary in `split_sentences` (shared with the SabiYarn port, D-27).
7. The `appended_text` rule (F15) fires on 2-10% of faithful human documents; with the narrower FLORES+ windows more often than under v3. Needs its own decision (slack, or a per-document instead of per-chunk rule).
8. A second GPU pass could measure the effect of the new rules on the SFT model's greedy outputs (share of chunks penalised by diacritics, omission flags per language) - the calibration `model` block (`--model`) was not run to keep the shared card free.

## 22. [v3, D-37] The GlotLID guard

**What it does.** After the hard checks of section 3 (so only completions that would otherwise be scored), one batched call classifies every gated completion whose language has the guard on: the top-1 GlotLID label must be in the language's accepted set, else the row is `wrong_language`, reward -2, no QE call, no consensus (R:280; `lid_spec` R:207). The text classified is the translation (NFC, tag stripped), cut at 1,000 characters (LD `clean_for_fasttext`).

**Classifier** (LD:36). `cis-lmu/glotlid` `model_v3.bin` (fastText, 2,102 labels, pre-downloaded on the rental; `reward.lid_model_path`). numpy 2 breaks `fasttext.FastText.predict` (`np.array(probs, copy=False)`: "Unable to avoid copy"), so the wrapper calls the C++ binding `model.f.predict(line + "\n", k, 0.0, "strict")`, which returns `(prob, label)` pairs without numpy (LD:69); `multilinePredict` is NOT used (in fasttext-wheel 0.9.2 it returns the top probability for every rank). Newlines become spaces. A cache keyed by text serves the repeats of a group. `make_lid` (LD:176) loads the model and classifies three canned sentences (English, Yoruba, Hausa) in `TranslationReward.__init__`: a missing fastText, a wrong model file or a numpy problem STOPS the run before step 1 (tested on the rental with numpy 2.5.3). Speed on the rental CPU: 336 completions of ~900 characters in 0.38 s per step. Checked end to end on the rental with the real model and the real `rl_rtxpro.yaml`: Yoruba text for `yor` passes, Swahili or English for `yor` and Yoruba for `swh` are `wrong_language` (-2), Swahili for `zul` is `wrong_language`.

**Measurement** (ML; 3 minutes on CPU). Human text per language: FLORES+ DEV references (997 sentences) for the 42 validated languages and dyu sag; for the 35 languages without a dev set (and pcm, efi) a random sample of 1,000 SFT training targets (30-500 characters, real corpora; a lower-bound proxy: other domain, some noisy rows). Own-label recall: FLORES+ languages mean 0.993 (min 0.926 bam), SFT-target languages mean 0.981 (min 0.77 kon). Per-language numbers are in the table of section 24.

| quantity | result |
|---|---|
| languages with strict own-label recall < 0.95 | bam 0.926, dyu 0.667, kin 0.936, kon 0.000 (no `kon_Latn` label: GlotLID says kng / kwy), nbl 0.868, pcm 0.809, shi 0.944 |
| sibling labels added (>= 3% of the language's human sentences, same close-language group, LD:126) | dyu += bam (0.667 -> 0.986), bam += dyu (0.926 -> 0.957), kin += run (0.936 -> 1.000), swh += swc (0.956 -> 1.000), nbl += xho zul nde (0.868 -> 0.998), kon += kng kwy ktu (0.000 -> 0.770) |
| never added | English or any label outside the language's group: Pidgin text that GlotLID calls English (15%) keeps pcm at 0.809 |
| guard on (final recall >= 0.90) | **75 of 77**; **off: pcm, kon** (reported, not applied) |
| price of the sibling labels (other languages' human sentences accepted by a set) | 1.3% on average, 2.6% at most (nbl) |
| minimum text length (smallest of 15 / 30 / 60 / 120 characters with prefix recall >= 0.90) | 15: amh ewe ibo kab kbp kin nbl nus sna; 30: 41 languages; 60: 19 languages; 120: dag kam kon pcm run shi sot wes |

Recall on short prefixes is far below whole-sentence recall (e.g. swh before swc 0.43 / 0.68 / 0.86 / 0.95 at 15 / 30 / 60 / 120 characters), which is why the minimum length is per language. RL completions are windows of at least a sentence, so the minimum rarely binds.

**Diagnostics** (to `metrics.jsonl` as `reward/...`): `fail/wrong_language`, `fail/wrong_language/<iso>` (and the same for every other failure class that occurs), `lid/checked_rate`, `lid/fail_rate_of_checked`, `lid/fail_rate/<iso>`, `lid/seconds`; per sample `RewardOutput.lid_label`, and the sample line of the log prints `lid=`.
**Limits.** A bag-of-n-grams classifier: a completion that is mostly right and ends in English passes; the model's own outputs may be confused more often than clean human text (watch `lid/fail_rate/<iso>` over the first 20 steps: a language whose rate is > 30% has a label set that is too strict for the model, switch it off with `reward.lid_disable`); the SFT-target recalls are a proxy.

## 23. [v3, D-38] Content modes and the consensus signal

**Modes** (calibration `content`, override `reward.content_modes`; `content_spec` R:192): `qe` (33 languages: the reward is exactly the v2 reward), `blend` (21: per chunk `r = (1 - beta) * shape(QE) + beta * consensus_value`, beta 0.5; R:396), `consensus` (23: `r = consensus_value`, the chunk is never sent to the scorer, R:362). Penalties that have a calibration (length, lexical, diacritics), the omission / extra-text failed chunks and every failure class then apply as for any language. A blend language whose group has no consensus is scored by QE alone; a consensus language gets `consensus_skip_value` (0.0 = the shaping curve's `t_fail` point). With `reward.consensus: false` everything is v2 (blend -> QE, consensus -> the constant).

**The signal** (CO; block R:304). For the completions of one GRPO group (consecutive rows, `__call__(samples, group_size=G)`, passed by `rollout_and_reward`: TR:248) that are VALID (they passed every failure check, including wrong_language), the consensus of completion i is the mean over the other valid completions j of the sentence chrF++ with i as hypothesis and j as reference (`group_consensus` CO:123). chrF++ is sacrebleu's `CHRF(word_order=2)` (char 6-grams, word 2-grams, beta 2, whitespace ignored) on the texts the reward already derived; the implementation caches the 8 n-gram counters per text and computes the match counts of a pair once (CO:76); it equals sacrebleu to 1e-14 on random texts (`tests/test_consensus.py`). Map to the reward scale (CO:144): `value = -2 + 4.25 * chrF / 100` (fixed, linear, the preset's range). A group needs >= 3 valid samples; ragged batches and a block that changes prompt are split (`split_groups` CO:149); greedy evaluation (G = 1) has no consensus.

**Worked example A, language `ach` (consensus mode, no calibration): sample 0 is truncated, 1 is a word loop, 2 copies the English; the five valid samples agree only with each other** (real code, fake tokenizer):

| # | status | chrF++ vs the other valid | reward |
|---|---|---|---|
| 0 | no_end_of_turn | - | -2.000 |
| 1 | degenerate | - | -2.000 |
| 2 | copy_source | - | -2.000 |
| 3 | ok | 81.9 | 1.481 |
| 4 | ok | 84.1 | 1.576 |
| 5 | ok | 82.7 | 1.514 |
| 6 | ok | 88.7 | 1.770 |
| 7 | ok | 79.8 | 1.390 |

**Worked example B, language `kam` in blend mode, QE from the fake scorer; reward = 0.5 * QE reward + 0.5 * (-2 + 4.25 * chrF/100)**:

| # | status | chrF++ vs the other valid | reward |
|---|---|---|---|
| 0 | ok | 91.9 | 1.978 |
| 1 | ok | 87.2 | 1.872 |
| 2 | ok | 88.6 | 1.914 |
| 3 | ok | 85.2 | 1.843 |
| 4 | ok | 86.1 | 1.859 |
| 5 | ok | 84.6 | 1.836 |
| 6 | ok | 91.9 | 1.978 |
| 7 | ok | 81.3 | 1.752 |

The same eight completions with the consensus switched off (QE only): 2.050, 2.036, 2.063, 2.067, 2.058, 2.079, 2.050, 2.050.

**Failure modes and the guards (tests `tests/test_consensus.py`).**

| mode | guard | test |
|---|---|---|
| (a) collapse: all 8 identical | chrF 100 for all, the group is flat, advantage 0 (`flat_threshold`); mode-seeking is what MBR does, so the trainer log must be watched: `consensus/chrf_mean` > ~80 means the language is collapsing | `test_a_collapsed_group_is_flat...` |
| (b) a majority of garbage agreeing with itself | identical loops are `degenerate` already; new `guards.is_self_repetitive` (G:85: > 35% of the word 4-grams repeat an earlier one, >= 24 words) removes near-loops (a paragraph repeated inside the answer, which the plain guard does not see) in blend / consensus languages (R:268); threshold from measurement: human FLORES+ documents of 3-40 sentences reach at most 0.21 over 6,000 documents of 40 languages, a doubled answer > 0.5; copies of the English and wrong-language outputs are failures and so never in the pool | `..._majority_of_loops_is_excluded` |
| (c) very short outputs agreeing | hard length bound 0.4 x source for blend / consensus languages WITHOUT a human length window (R:271; below every calibrated language's p01 = 0.44) instead of 0.2; the calibrated ones (dyu sag and the validated blends) keep the window penalty. **Not caught:** agreeing outputs between 0.4 x source and the true length of the uncalibrated languages | `..._pooled_length_bound...` |
| (d) cost | 9 / 34 / 105 ms per group of 8 for ~370 / ~1,600 / ~5,200 character completions (rental CPU, one thread, FLORES+ text) = **0.4 s / 1.4 s / 4.4 s per step of 42 groups** in the b64 / b256 / b1024 stages, only blend / consensus groups pay it | `test_cost_of_a_step_is_small` |

Discrimination on perturbed FLORES+ documents (8 samples: 5 with 5% words dropped / swapped, one 30%, one another document, one 60% dropped), mean chrF++: light noise 66 / 69 / 71 (3 / 12 / 40 sentences), heavy noise 46 / 49 / 51, other document 20 / 27 / 35, 60% dropped 30 / 31 / 33.
Parameters: `consensus`, `consensus_min_valid` 3, `consensus_skip_value` 0.0, `content_modes`, `consensus_beta`, `consensus_beta_default` 0.5, `selfrep_guard` consensus, `min_len_ratio_uncalibrated` 0.4 (R:126). Diagnostics: `consensus/{applied_rate,chrf_mean,groups,groups_skipped,seconds}`, `consensus/chrf/<iso>`, `mode/<mode>_rate`, per sample `consensus_chrf`, `content_mode`.

## 24. [v3, D-38] The 77 RL languages: mode, beta, advantage weight, anchors, GlotLID

77 = the SFT targets with data (85 table rows minus nde fat knc gaa men luy tzm ful). Groups: **validated** (42, section 18: QE validity measured); **listed** (12: named on the AfriCOMET-QE-1.1 model card, which prints 75 languages although text and metadata say 76 / 77; QE unvalidated or weak: loz ven iso tiv kqn wal lun kua kon dyu sag, and swc); **outside AfriCOMET** (23: not on the card; `dua` was missing from the owner's list).
Decisions: **swc** (Congo Swahili) is listed through Swahili (the card names swa, validated AUC 0.964): swh's anchors, blend, weight 0.3, no calibration of its own. **dyu, sag**: own FLORES+ DEV calibration (997 sentences, CPU on the old box): dyu AUC 0.683 (anchors 0.461 / 0.511 / 0.561, omission cost 5; below the 0.70 bar, forced usable), **sag AUC 0.783** (0.428 / 0.478 / 0.581, omission cost 4; the 0.689 of the 60-sentence study was a small-sample estimate: sag passes the bar and is as good as the weak validated languages); the owner's weight 0.4 is kept for both (capped by the measured same-source value, 0.92 for sag): raising sag is one number in the file. **The nine untested** (ven has only NTREX = a test set, never used): pooled-prior anchors = the median of the 41 calibrated languages' anchors (0.536 / 0.608 / 0.746; NOT the default 0.46 / 0.60 / 0.75), weight 0.3, no length / lexical / diacritic penalty (no human reference). **Weights** (owner): 0.3 untested, 0.4 dyu / sag, 0.5 consensus-only; justification: an untested language's QE half may point the wrong way (dyu shows it can be below the usability bar), a consensus-only language has no learned metric that can mislead, only the model's agreement with itself, a bounded-risk signal that still carries information. None of the three numbers is measured. **Weak-QE validated languages** in blend mode (beta 0.5): the owner's kbp nus kam umb kmb mos bam, plus kab fon by the same rule (AUC < 0.75: nus .714 kbp .720 umb .720 kab .733 fon .736 kmb .739 kam .739 mos .748); bam (AUC 0.809) is on the owner's list because its human QE scores are low (median 0.46). Override per language with `reward.content_modes: {kab: qe}`.
Columns: tag, name, group, mode, beta, advantage weight (`reliability_same_source`), anchors t_fail/t_mid/t_great (source), AUC human-vs-mismatched, GlotLID guard + accepted labels, recall with that set (source: flo(res dev) / sft(targets)), minimum characters.

| tag | name | group | mode | beta | weight | anchors | AUC | GlotLID | recall | min chars |
|---|---|---|---|---|---|---|---|---|---|---|
| yor | Yoruba | validated | qe | - | 1.00 | 0.525/0.575/0.666 (FLORES+ dev) | 0.803 | on yor | 1.000 (flo) | 30 |
| hau | Hausa | validated | qe | - | 1.00 | 0.549/0.683/0.792 (FLORES+ dev) | 0.941 | on hau | 1.000 (flo) | 30 |
| ibo | Igbo | validated | qe | - | 1.00 | 0.533/0.583/0.708 (FLORES+ dev) | 0.844 | on ibo | 1.000 (flo) | 15 |
| pcm | Pidgin | validated | qe | - | 0.83 | 0.459/0.509/0.563 (MAFAND dev) | 0.766 | **OFF** pcm,wes | 0.809 (sft) | 120 |
| efi | Efik | validated | qe | - | 0.00 | 0.000/0.600/0.750 (none) | - | on efi | 0.995 (sft) | 30 |
| twi | Twi | validated | qe | - | 1.00 | 0.518/0.568/0.666 (FLORES+ dev) | 0.831 | on twi | 0.998 (flo) | 30 |
| ewe | Ewe | validated | qe | - | 1.00 | 0.550/0.600/0.698 (FLORES+ dev) | 0.835 | on ewe | 1.000 (flo) | 15 |
| fon | Fon | validated | blend | 0.5 | 0.91 | 0.471/0.521/0.573 (FLORES+ dev) | 0.736 | on fon | 1.000 (flo) | 30 |
| wol | Wolof | validated | qe | - | 1.00 | 0.539/0.589/0.673 (FLORES+ dev) | 0.792 | on wol | 1.000 (flo) | 30 |
| swh | Swahili | validated | qe | - | 1.00 | 0.553/0.721/0.827 (FLORES+ dev) | 0.964 | on swc,swh | 1.000 (flo) | 30 |
| kin | Kinyarwanda | validated | qe | - | 1.00 | 0.592/0.682/0.788 (FLORES+ dev) | 0.935 | on kin,run | 1.000 (flo) | 15 |
| som | Somali | validated | qe | - | 1.00 | 0.515/0.620/0.767 (FLORES+ dev) | 0.919 | on som | 1.000 (flo) | 30 |
| orm | Oromo | validated | qe | - | 1.00 | 0.591/0.641/0.761 (FLORES+ dev) | 0.848 | on gaz | 1.000 (flo) | 30 |
| amh | Amharic | validated | qe | - | 1.00 | 0.575/0.727/0.839 (FLORES+ dev) | 0.964 | on amh | 1.000 (flo) | 15 |
| tir | Tigrinya | validated | qe | - | 1.00 | 0.596/0.660/0.779 (FLORES+ dev) | 0.893 | on tir | 0.978 (flo) | 30 |
| lug | Luganda | validated | qe | - | 1.00 | 0.529/0.579/0.694 (FLORES+ dev) | 0.818 | on lug | 0.997 (flo) | 30 |
| lin | Lingala | validated | qe | - | 1.00 | 0.500/0.569/0.723 (FLORES+ dev) | 0.881 | on lin | 1.000 (flo) | 30 |
| zul | Zulu | validated | qe | - | 1.00 | 0.554/0.670/0.792 (FLORES+ dev) | 0.939 | on zul | 0.961 (flo) | 60 |
| xho | Xhosa | validated | qe | - | 1.00 | 0.515/0.614/0.759 (FLORES+ dev) | 0.909 | on xho | 0.987 (flo) | 30 |
| afr | Afrikaans | validated | qe | - | 1.00 | 0.535/0.743/0.844 (FLORES+ dev) | 0.982 | on afr | 1.000 (flo) | 30 |
| sna | Shona | validated | qe | - | 1.00 | 0.571/0.646/0.752 (FLORES+ dev) | 0.908 | on sna | 1.000 (flo) | 15 |
| nya | Chichewa | validated | qe | - | 1.00 | 0.565/0.640/0.763 (FLORES+ dev) | 0.904 | on nya | 0.997 (flo) | 30 |
| sot | Sesotho | validated | qe | - | 1.00 | 0.538/0.636/0.758 (FLORES+ dev) | 0.921 | on sot | 0.986 (flo) | 120 |
| tsn | Setswana | validated | qe | - | 1.00 | 0.553/0.625/0.754 (FLORES+ dev) | 0.900 | on tsn | 0.997 (flo) | 30 |
| mlg | Malagasy | validated | qe | - | 1.00 | 0.568/0.690/0.804 (FLORES+ dev) | 0.947 | on plt | 0.998 (flo) | 60 |
| bem | Bemba | validated | qe | - | 1.00 | 0.523/0.573/0.696 (FLORES+ dev) | 0.860 | on bem | 0.999 (flo) | 30 |
| tum | Tumbuka | validated | qe | - | 1.00 | 0.595/0.663/0.768 (FLORES+ dev) | 0.897 | on tum | 1.000 (flo) | 30 |
| run | Kirundi | validated | qe | - | 1.00 | 0.575/0.650/0.763 (FLORES+ dev) | 0.912 | on run | 0.977 (flo) | 120 |
| tso | Tsonga | validated | qe | - | 1.00 | 0.569/0.639/0.762 (FLORES+ dev) | 0.895 | on tso | 0.999 (flo) | 30 |
| nso | Sepedi | validated | qe | - | 1.00 | 0.521/0.608/0.754 (FLORES+ dev) | 0.897 | on nso | 0.996 (flo) | 30 |
| ssw | Swati | validated | qe | - | 1.00 | 0.569/0.643/0.772 (FLORES+ dev) | 0.893 | on ssw | 0.994 (flo) | 60 |
| umb | Umbundu | validated | blend | 0.5 | 0.79 | 0.471/0.521/0.595 (FLORES+ dev) | 0.720 | on umb | 0.989 (flo) | 60 |
| kmb | Kimbundu | validated | blend | 0.5 | 0.86 | 0.469/0.519/0.596 (FLORES+ dev) | 0.739 | on kmb | 0.991 (flo) | 60 |
| mos | Mossi | validated | blend | 0.5 | 0.90 | 0.536/0.586/0.654 (FLORES+ dev) | 0.748 | on mos | 0.987 (flo) | 30 |
| kik | Kikuyu | validated | qe | - | 1.00 | 0.523/0.573/0.693 (FLORES+ dev) | 0.854 | on kik | 0.974 (flo) | 30 |
| luo | Luo | validated | qe | - | 1.00 | 0.580/0.630/0.746 (FLORES+ dev) | 0.852 | on luo | 1.000 (flo) | 30 |
| kam | Kamba | validated | blend | 0.5 | 0.82 | 0.452/0.502/0.610 (FLORES+ dev) | 0.739 | on kam | 0.959 (flo) | 120 |
| bam | Bambara | validated | blend | 0.5 | 1.00 | 0.469/0.519/0.629 (FLORES+ dev) | 0.809 | on bam,dyu | 0.957 (flo) | 30 |
| kbp | Kabiye | validated | blend | 0.5 | 0.86 | 0.543/0.593/0.643 (FLORES+ dev) | 0.720 | on kbp | 1.000 (flo) | 15 |
| kab | Kabyle | validated | blend | 0.5 | 0.96 | 0.493/0.543/0.597 (FLORES+ dev) | 0.733 | on kab | 1.000 (flo) | 15 |
| nus | Nuer | validated | blend | 0.5 | 0.86 | 0.461/0.511/0.565 (FLORES+ dev) | 0.714 | on nus | 0.998 (flo) | 15 |
| lua | Luba-Kasai | validated | qe | - | 0.96 | 0.438/0.488/0.634 (FLORES+ dev) | 0.812 | on lua | 0.996 (flo) | 30 |
| ibb | Ibibio | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on efi | 1.000 (sft) | 30 |
| urh | Urhobo | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on urh | 0.955 (sft) | 60 |
| ven | Venda | listed | blend | 0.5 | 0.30 | 0.536/0.608/0.746 (pooled prior) | - | on ven | 0.998 (sft) | 30 |
| nbl | South Ndebele | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on nbl,nde,xho,zul | 0.998 (sft) | 15 |
| sag | Sango | listed | blend | 0.5 | 0.40 | 0.428/0.478/0.581 (own FLORES+ dev) | 0.783 | on sag | 1.000 (flo) | 30 |
| tiv | Tiv | listed | blend | 0.5 | 0.30 | 0.536/0.608/0.746 (pooled prior) | - | on tiv | 0.996 (sft) | 60 |
| bin | Edo | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on bin | 0.999 (sft) | 30 |
| dag | Dagbani | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on dag | 0.953 (sft) | 120 |
| oke | Okpe | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on oke | 0.990 (sft) | 60 |
| shi | Tachelhit | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on shi | 0.944 (sft) | 120 |
| dyu | Dyula | listed | blend | 0.5 | 0.40 | 0.461/0.511/0.561 (own FLORES+ dev) | 0.683 | on bam,dyu | 0.986 (flo) | 60 |
| bbj | Ghomala | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on bbj | 1.000 (sft) | 30 |
| kon | Kongo | listed | blend | 0.5 | 0.30 | 0.536/0.608/0.746 (pooled prior) | - | **OFF** kng,ktu,kwy | 0.770 (sft) | 120 |
| kea | Kabuverdianu | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on kea | 0.981 (sft) | 60 |
| swc | Congo Swahili | listed | blend | 0.5 | 0.30 | 0.553/0.721/0.827 (borrowed from swh) | - | on swc,swh | 1.000 (sft) | 30 |
| kqn | Kaonde | listed | blend | 0.5 | 0.30 | 0.536/0.608/0.746 (pooled prior) | - | on kqn | 0.999 (sft) | 30 |
| lun | Lunda | listed | blend | 0.5 | 0.30 | 0.536/0.608/0.746 (pooled prior) | - | on lun | 0.999 (sft) | 30 |
| iso | Isoko | listed | blend | 0.5 | 0.30 | 0.536/0.608/0.746 (pooled prior) | - | on iso | 1.000 (sft) | 30 |
| loz | Lozi | listed | blend | 0.5 | 0.30 | 0.536/0.608/0.746 (pooled prior) | - | on loz | 1.000 (sft) | 30 |
| wal | Wolaytta | listed | blend | 0.5 | 0.30 | 0.536/0.608/0.746 (pooled prior) | - | on wal | 1.000 (sft) | 30 |
| ach | Acholi | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on ach | 0.994 (sft) | 60 |
| tsc | Tswa | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on tsc | 0.992 (sft) | 60 |
| vmw | Makhuwa | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on vmw | 0.999 (sft) | 30 |
| kua | Kuanyama | listed | blend | 0.5 | 0.30 | 0.536/0.608/0.746 (pooled prior) | - | on kua | 1.000 (sft) | 30 |
| seh | Sena | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on seh | 0.998 (sft) | 60 |
| nyn | Nyankore | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on nyn | 0.995 (sft) | 60 |
| her | Herero | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on her | 0.996 (sft) | 30 |
| yao | Yao | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on yao | 0.998 (sft) | 30 |
| sid | Sidamo | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on sid | 1.000 (sft) | 60 |
| ndc | Ndau | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on ndc | 0.997 (sft) | 30 |
| kri | Krio | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on kri | 0.999 (sft) | 60 |
| ish | Esan | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on ish | 0.999 (sft) | 60 |
| dua | Duala | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on dua | 0.997 (sft) | 60 |
| bas | Basaa | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on bas | 1.000 (sft) | 60 |
| wes | Cameroon Pidgin | outside AfriCOMET | consensus | 1.0 | 0.50 | - (-) | - | on wes | 0.974 (sft) | 120 |

**Prompts.** 35 new files x 20,000 rows (5,000 per length bin), built on the old box with the recipe of D-32 (`gemma_rl_assemble_add35.py` = `gemma_rl_assemble.py` with the full language list of the table instead of `SUPPORTED_RL`, cached filtered pool of 1,424,598 windows, no boost, CPU only), streamed to `/workspace/sft_build/gemma/rl_prompts_v2/` on the rental (md5 of all 35 files + `stats_35new.json` verified); the directory now holds 77 language files. Draw shares with `language_alpha 0.5` (unchanged): 0.84% per new language (~106 of the 12,600 prompts of a 300-step x 42 run), 2.1% per standard old language (was 2.9%), the 57 added languages together 48% of the draws, pcm 4.6%. `language_weights` is untouched (efi 0.0001); lowering the weights of the 35 new languages (e.g. 0.5) is the knob if the validated languages should keep their old share. The Gemma-token columns (`rl_prompts_v2_tok`) were not built for the new files (nothing in the training path reads them).

## 25. [v3] Config keys added (`reward.` unless stated; defaults are the v2 behaviour)

| key | default | `rl.yaml` / `rl_rtxpro.yaml` | meaning |
|---|---|---|---|
| `lid_enabled` | false | **true** | GlotLID guard (section 22) |
| `lid_min_chars` | 15 | 15 | global minimum text length (a language may need more) |
| `lid_model_path` | null | null / rental snapshot path | local `model_v3.bin`; null = Hub cache / download |
| `lid_disable` | [] | [] | languages whose guard is off whatever the calibration says |
| `consensus` | false | **true** | consensus signal (section 23) |
| `consensus_min_valid` | 3 | 3 | valid samples a group needs |
| `consensus_skip_value` | 0.0 | 0.0 | rubric-only value of a consensus-mode sample without consensus |
| `consensus_scale` | null | (null) | (lo, hi) of the chrF -> reward map; null = [-2, +2.25] |
| `content_mode_default` | qe | qe | languages without a calibration `content` block |
| `content_modes` / `consensus_beta` | {} | {} | per-language overrides |
| `consensus_beta_default` | 0.5 | 0.5 | |
| `selfrep_guard` | consensus | consensus | near-loop guard: off / consensus (blend + consensus languages) / all |
| `min_len_ratio_uncalibrated` | 0.4 | 0.4 | hard lower length bound without a human length window |
| `calibration_path` | null | **v5** (rental: `/workspace/sft_build/gemma_mt/calibration/shaping_flores_dev_v5.json`) | |
| top level `extra_libs` | [] | [] / `[/workspace/sft_build/pylibs]` | appended to `sys.path` (sacrebleu) |
| top level `hub_squash`, `hub_prune_tags` | true, true | true, true | section 29 |
| top level `hub_repo_id`, `model_name` | | `<HF_ORG>/gemma-3-270m-translate-rl`, `<HF_ORG>/gemma3-270m-translate` (rtxpro) | section 29 |

Updated loop (section 0): ... `TranslationReward(samples, group_size=G)`: 0 decode / NFC / failure checks (+ self-repetition and the uncalibrated length bound in blend / consensus languages) -> **0c GlotLID** -> **0d consensus value per sample** -> 1 chunks -> 2 QE only for `qe` / `blend` chunks -> 3 shaping -> **3b blend / consensus value** -> 4 penalties -> 5 aggregate -> 6 weight (0.3 .. 1.0). The advantage, loss and update are unchanged.

## 26. [v3] The advantage weights of the added languages (reminder)

`A_i = standardise_group(r_i) * weight(language)` (TR, unchanged): the weight multiplies AFTER the group standardisation, so a consensus-only language with weight 0.5 pushes the policy half as hard as a validated one at equal reward spread. A group that is flat (all failed, or all identical) has A = 0.

## 27. [v3, D-39] Audit: sampling and normalisation against standard GRPO / RLVR practice

Evidence: code cites below and `tests/test_sampling_audit.py` (8 tests). vLLM 0.31.0 defaults were read on the rental by constructing `SamplingParams` there.

**(a) Sampling parameters.**

| item | HF path (`sampling.py`) | vLLM path (`vllm_engine.py`) | standard? |
|---|---|---|---|
| temperature | 1.0 (`temperature=T`) | 1.0 | yes (RLVR default; the loss divides logits by the same T) |
| top_p | 1.0 | 1.0 | yes; < 1 would bias the estimator (guarded by `allow_truncated_sampling`) |
| top_k | 0 = off in HF | 0 = off: vLLM 0.31 `SamplingParams` validation: "top_k must be 0 (disable), or at least 1"; -1 is the legacy spelling | yes, disabled |
| min_p / repetition / presence / frequency penalty | HF defaults (none; Gemma's generation_config has top_k 64, top_p 0.95, which the explicit arguments override) | min_p 0, 1.0, 0, 0 (vLLM defaults; `generation_config="vllm"` ignores the checkpoint's defaults) | yes, none |
| min_tokens / logit bias / bad words | none | 0 / none / none | yes |
| max length | `max_new_tokens` = per-bin budget | `max_tokens` = same | yes; truncation = failure (-2) |
| stop | `eos_token_id=[<end_of_turn>, <eos>]` | `stop_token_ids=[106, 1]`, `ignore_eos=False` | yes |
| n samples | `num_return_sequences=8` (independent multinomial draws) | `n=8`, one seed per prompt (`derive_seeds`), child k gets seed + k (`parallel_sampling.py`): independent streams | yes |
| token ids | as generated | as generated: prompts in as ids (`skip_tokenizer_init`), `detokenize=False`, no re-tokenisation | yes |
| log-probs for the loss | recomputed by teacher forcing (`completion_logprobs`) | same; vLLM log-probs only every 25 steps as a consistency check (stops at mean gap > 0.25) | yes (no sampler log-probs in the loss) |
| sampling precision | policy dtype under autocast | bf16 copy of the policy, logits to fp32 for sampling | **non-standard, known**: trainer fp32-under-autocast vs sampler bf16; no importance correction (TIS) |

**(b) Normalisation before any reward component.** One chain, the same function for the reward and the evaluation: cut at the first stop id -> decode (with and without specials) -> scrub special-token strings -> NFC -> strip -> language tag (`text.decode_completion` + `format.strip_iso_prefix`); the source is NFC-normalised in the reward only (the evaluation metrics NFC both sides). `generate_translations` (evaluation, SFT benchmarks, dev chrF++ of the RL loop) and `TranslationReward` call `decode_completion` with the same arguments; the reward adds pad ids (no effect on the text). **Test**: eight scripted id sequences (plain, NFD diacritics, spaces around the tag, typed-out special strings, special ids inside, text after the first stop, a second stop id, interior whitespace) go through the reward and through the REAL `generate_translations` with a scripted sampler: the hypothesis strings are identical wherever the reward scores the row; the benchmark metric adds only NFC (idempotent). The only intended difference is the tag policy: a missing / wrong tag is a reward failure, the evaluation scores the rest and reports `prefix_status`.

**(c) Group advantage and loss conventions.**

| convention | here | GRPO (DeepSeekMath) | Dr.GRPO / DAPO | verdict |
|---|---|---|---|---|
| advantage | `(r - mean_group) / (std_group + 1e-4)`, population std, group of 8, consecutive rows | same (sample std, small eps) | Dr.GRPO: no std; DAPO: same as GRPO | standard (GRPO); `advantage_norm: mean` = Dr.GRPO |
| flat group | A = 0 (std < 1e-6) | A = 0 by the formula | DAPO drops such groups from the batch | deviation: kept, their tokens stay in the loss denominator (dilutes the step by the flat fraction) |
| loss aggregation | token level: sum over all completion tokens of the step / total completion tokens of the step | per sequence 1/|o_i|, then mean | DAPO: token level; Dr.GRPO: constant normaliser | DAPO-like (no per-sequence length bias) |
| KL | k3 estimator to the frozen SFT reference, coefficient 0.05 | k3, 0.04 | DAPO / Dr.GRPO often drop KL | standard (GRPO) |
| updates per batch | 1, exactly on-policy: ratio == 1 (asserted every step), no clipping | mu updates with clipping | DAPO: clip-higher | standard for fully on-policy GRPO; clipping exists for `num_updates_per_batch > 1` |
| baseline / critic | group mean, no value function | same | same | standard |
| per-language weight | `A *= weight` after the standardisation | none | none | **deliberate deviation** (QE reliability of the language) |
| rewards | bounded [-2, +2.25], failures -2 (truncation too, no soft overlong ramp) | rule-based 0/1 | DAPO: soft overlong punishment | deliberate; truncation is a hard failure |
| reference | frozen SFT model | frozen SFT | | standard |

**(d) Non-standard or risky.**

1. *Sampler/trainer precision mismatch* (bf16 vLLM vs fp32-under-autocast): measured mean |dlogp| 0.01-0.02, no correction; only the 25-step gap check guards it.
2. *Std normalisation on near-flat groups* amplifies noise: a group whose members differ by 1e-3 gets unit-variance advantages (`std_eps` 1e-4). With the consensus signal rewards are continuous, so a near-collapsed group (7 identical + 1 almost identical) is not exactly flat. A std floor (e.g. 0.05 reward units) or `advantage_norm: mean` would remove it; not changed (the repo's GRPO choice).
3. *Flat groups stay in the token denominator* (all-failed groups, collapsed groups): the effective step size shrinks with the flat fraction (`flat_group_frac` is logged).
4. *Consensus is mode-seeking* and its reward couples the samples of a group; diversity can fall over the run (KL 0.05 and the weights 0.3-0.5 limit it; monitor `consensus/chrf_mean`).
5. *Greedy evaluation vs T = 1 training*: `eval/reward` of blend / consensus languages is QE-only / the constant 0.0 (no groups), so the held-out reward carries no signal for the 23 consensus languages; model selection is by dev chrF++ (`select_by: benchmark_chrf`), which exists only for the languages in `bench_dev_clean.jsonl`.
6. *Truncation = -2* with per-bin budgets (b64 384 tokens ...): a budget that is too tight punishes the policy for the sampler (v2 section 15).

## 28. [v3, D-39] Launch blockers found while checking the path (fixed, staged)

* **F21 - `load_rl_prompts` crashed on the real prompt directory.** `data.io.iter_rows` (since D-34) parses `.json` files and `yield from data` on a dict yields its KEYS; `rl_prompts_v2/` contains `stats.json`, `stats_22new.json` (and now `stats_35new.json`): `AttributeError: 'str' object has no attribute 'get'` at the first prompt load. Fixed (`io.py:78`: only dict rows are yielded; checked by `dry_check_rl.py` on the real directory), and the three stats files were moved to `gemma/rl_prompts_v2_meta/` on the rental.
* **F22 - `sacrebleu` is not importable in the trainer's environment** (`venv_gemma` + `PYTHONPATH=gemma_mt`): the step-0 dev evaluation (`evaluate_benchmark` -> `metrics.chrf_pp`) would have raised `ModuleNotFoundError`. New `RLConfig.extra_libs` (TR:341) appends `/workspace/sft_build/pylibs` to `sys.path` (after the environment's own packages).
* **F23 - `reward.lid_enabled: true` did nothing from the CLI** (`train_rl.main` passes `lid=None`; the reward ignored a missing LID silently). The reward now builds and self-tests GlotLID itself.
* **F24 - the post-SFT evaluation defaults to the old repo.** `rental_post_sft.sh` starts `gemma_mt_eval_candidate/scripts/eval_sft_benchmarks.sh`, whose default model is `<HF_ORG_LEGACY>/gemma3-270m-translate` (quota full, no longer written). The staged `scripts/eval_sft_benchmarks.sh` and `eval/sft_benchmarks.py` default to the org repo; the candidate copy was not touched (not mine): pass the model as the first argument or merge the staged script.

## 29. [v3, D-40] Hub storage

SFT repo `<HF_ORG>/gemma3-270m-translate`, RL repo `<HF_ORG>/gemma-3-270m-translate-rl`, RL policy start `<HF_ORG>/gemma3-270m-translate` (the old <HF_ORG_LEGACY> ids remain in `sft_long.yaml`, the manual push tools and old reports). `CheckpointPusher(squash=True, prune_tags=True)` (`hub/push_checkpoint.py:116`): after an upload, older `rl-step-N` tags are deleted (the newest stays), the history of the pushed branch is squashed (`HfApi.super_squash_history`) and the tag is created afterwards on the squashed head; best effort. RL config: `hub_squash`, `hub_prune_tags` true: the best (main), `latest` and `stage-*` pushers each keep one copy: at most ~7 model copies. Tests with a fake API: `tests/test_hub_latest.py`.

## 30. [v3] Open problems

1. **No GlotLID guard for pcm (recall 0.809: GlotLID calls 15% of Pidgin English) and kon (GlotLID has no `kon` label; kng / kwy / ktu cover 0.77).** Wrong-language Pidgin is caught only by the copy guard and QE.
2. **The LID recalls of the 35 new languages are measured on SFT training targets**, not on a dev set; a language whose model outputs are confused more often than that will lose reward unfairly: watch `lid/fail_rate/<iso>` early and use `lid_disable`.
3. **The untested listed languages have an unvalidated QE half** (pooled anchors, no penalties): weight 0.3 limits the damage; a small human-checked dev set for loz iso tiv kqn wal lun kua kon (and a non-test set for ven) would allow the same calibration as the others.
4. **Consensus cannot see agreeing-but-short or agreeing-but-hallucinated outputs** in uncalibrated languages beyond the 0.4 x source bound and the self-repetition guard; no anchor-recall (numbers / names) check exists without a calibration.
5. **Consensus is sample-level** (one value per completion, the same for every chunk of it); the per-chunk structure comes only from QE (blend) and the penalties.
6. **Eval/held-out reward is blind for the 23 consensus languages** (greedy, no group): the dev chrF++ covers only languages with FLORES+ dev rows in the benchmark file; no selection signal exists for the others.
7. **Draw shares**: the 57 added languages take 48% of the draws; the validated languages' share fell (2.9% -> 2.1% for a standard old one). `language_weights` is the knob.
8. **Consensus cost** is single-threaded Python: 4.4 s per b1024 step (measured on FLORES+ text; real completions may be longer); acceptable against the generation time, trivially parallelisable if needed.
9. **Test environment**: on the rental the suite needs a shell without the Hub token (`test_stratified::test_pusher_is_disabled_without_repo_or_token` sees the write token of `env.sh` / `.hf_write_token`); and the repo's own `tests/run_tests.py` only understands `parametrize` without a real pytest (use `python -m pytest` there).
10. v2's open problems 1-8 (section 21) stand.

## 31. [v3, D-41] Best-translation archive (reads the reward, changes nothing in it)

Owner 2026-10-19: save good translations for later review. `rl/archive.py`, hooked into `train_rl.py` right after the reward of a step (before the advantages): per source (key = `sha1(iso + "\n" + NFC(source))[:16]`) one SQLite row with the best ELIGIBLE translation seen so far (status ok, no failed chunk, GlotLID passed, diacritic deficit 0, reward >= qe 1.0 / blend 1.0 / consensus 0.5, blend languages also consensus chrF++ >= 50), replaced when a strictly better eligible one appears; written by one background thread through a bounded queue (a step is never delayed; hook 0.3-1 ms per 1,000 samples). `RewardOutput` gained the per-sample lists `n_chunks` and `failed_chunks` (values `TranslationReward` already had). Config keys `archive`, `archive_path`, `archive_min_reward_by_mode`, `archive_min_consensus_chrf`, `archive_flush_seconds`, `archive_queue_size` (off in `rl.yaml` / `rl_rtxpro.yaml`, on in `rl_rtxpro_phase2.yaml`). Review with `python -m gemma_mt.rl.archive stats|export|review PATH`. Full description, thresholds, limits and tests: `reports/rl_archive_2026-10-19.md`, DECISIONS D-41.

## 32. [v4, D-47 wired, 2026-11-01] Reference-free confidence rules (gate / trust / cap) — OFF by default

`rl/confidence.py` (the pure D-47 module) is now called by `TranslationReward` and `train_rl.py`. **With `reward.confidence: false` (the default) nothing changes**: rewards, weights and advantages are bit-identical (test `test_off_by_default_is_bit_identical`).

**When it runs.** Only in TRAINING calls (`group_size > 1`) and only for groups whose content mode is in `confidence_modes` (default consensus, blend; `qe` languages have the real signal). Greedy evaluation (`group_size` None / 1) neither reads nor writes the tracker.

**Where the per-sample mean log-prob comes from.** The sampler's per-token log-probs (`Rollout.sampler_logp`, the log-prob under the sampling distribution; T=1, top_p=1, top_k=0 = the policy's). `train_rl` asks for them (`return_logprobs`) whenever `reward.confidence` is on; vLLM sampler: sampled-token log-probs only (cheap); HF `generate`: `output_scores` over the whole vocabulary (use it only for short CPU runs). `rollout_and_reward` turns them into `Sample.logp` with `confidence.completion_mean_logp`: the mean over the CONTENT tokens, i.e. WITHOUT the leading language tag (`_tag_len`: 1 token for the dedicated tag token `<unusedK>` / the renamed `<yor>`, 1-3 text pieces before growth 1) and without the trailing end token. A sampler without log-probs gives `None`, which counts as "no information" (neutral), never as a penalty.

**The three rules** (`TranslationReward._confidence_rules`, called after GlotLID and consensus):

| rule | flag | effect |
|---|---|---|
| 1 gate | `confidence_gate` (True) | `confidence.group_gate`: a group with < `min_valid` (3) valid samples, a GlotLID pass rate < `lid_pass_min` (0.5) or a collapsed valid set (distinct < 0.5) gets advantage weight 0 for ALL its samples (`out.weights[i] = 0`; `RewardOutput.conf_gate[i]` holds the reason) |
| 2 trust | `confidence_trust` (True) | the language's advantage weight is multiplied by its running trust in [0, 1] = weakest of the GlotLID pass rate, the mean log-prob of the valid samples and the valid fraction (`ConfidenceTracker`, EMA over groups; plain mean for the first 20). Trust and cap stay NEUTRAL (1.0 / no cap) until the language's state holds `confidence_min_groups` (10) groups |
| 3 cap | `confidence_cap` (True) | the reward of a consensus-MODE sample is `min(reward, consensus_cap(trust))`: skip value (0.0) at trust 0, the top of the range at trust 1 |

Other keys: `confidence_modes`, `confidence_alpha` (0.05), `confidence_relative` (True: log-prob minus the median log-prob of the languages that currently work, `reference_logp`: >= 5 languages with GlotLID pass >= 0.9 and >= 20 groups seen), `confidence_absolute_fallback` (False: until that reference exists the log-prob ramp is skipped; True = use the absolute bounds, which were measured on another policy and are NOT valid for the grown 385M model), `confidence_params` (overrides of `ConfidenceParams`: `lid_lo lid_hi logp_rel_lo logp_rel_hi valid_lo valid_hi min_valid lid_pass_min distinct_min cap_floor min_trust_to_pay ...`; a typo raises). A missing observation (GlotLID guard off for the language, no text long enough, no log-probs) counts as 1 in the trust (`language_trust(..., missing_ok=True)`), so pcm / kon / languages without a label are not starved.

**Example yaml (`reward:` block):** `confidence: true`, `confidence_min_groups: 10`, `confidence_modes: [consensus, blend]`.

**Logged (metrics.jsonl, keys exactly as written, one record per step; only languages with a group in the step):** `confidence/groups`, `confidence/gate_rate`, `confidence/weight_zeroed_rate`, `confidence/gate_reason/{few_valid,wrong_language,collapsed}`, `confidence/logp_ref` (when the reference exists), and per language `confidence/gate_rate/<iso>`, `confidence/trust/<iso>`, `confidence/logp_mean/<iso>` (this step), `confidence/logp_running/<iso>`, `confidence/lid_pass_running/<iso>`, `confidence/cap/<iso>` (consensus-mode languages). The startup line prints the active rules.

**State.** `TranslationReward.conf_state()` / `load_conf_state()`; `train_rl.save_last` stores it in `last/trainer_state.pt` under `state["confidence"]` and the resume block restores it (a checkpoint without the key: the tracker starts empty, nothing breaks).

**Tests** (CPU, no torch): `tests/test_confidence_wiring.py` (off-by-default identity, gate on few valid and on the GlotLID pass rate, trust 0 + cap on a low log-prob, neutral until enough groups and without log-probs, relative form waiting for the reference, greedy / qe languages untouched, checkpoint round trip, tag-length and mean-log-prob helpers) and the old `tests/test_confidence.py`. **Not run:** the trainer loop with the rules on (needs torch / a GPU), a real vLLM or HF rollout with `return_logprobs`, the real Gemma tokenizer in `_tag_len` (the fake tokenizer was used).

## 33. [v4, 2026-11-01] Eight more consensus-only RL languages (guw lue tll zne lub crs mfe arz)

**Decision (owner 2026-11-01).** The 40 languages appended to the table on 2026-11-01 are SFT targets; those with at least 2,000 packs in the final stage-B set AND a usable GlotLID plan join the RL as QE-free consensus-only languages (D-38 mode `consensus`, beta 1.0, advantage weight 0.5, no AfriCOMET code). Included (8, packs in brackets): guw 4,962, lue 4,495, arz 4,909, tll 3,517, zne 2,845, lub 2,791, mfe 2,171, crs 2,022. Excluded (32): 28 have fewer than 2,000 packs (rnd 1,810, nyk 1,530, ndo 1,445, kwn 1,366, dje 1,242, ada 991, bci 705 ... teo 77, adh 73), `nup` (142 packs, no GlotLID label, guard custom/off), `toi` (51 packs), and `ber`, `zgh`, `aar` (0 packs; ber and aar also without a label or a tested classifier).

**Files.** `languages._RL_CONSENSUS_ONLY` += the 8 (ids and table order unchanged; `SUPPORTED_RL` 77 -> 85, `CONSENSUS_ONLY` 23 -> 31). `calibration/shaping_flores_dev_v6.json` = v5 (every v5 entry byte-identical) + one entry per added language: `content` {consensus, 1.0}, `reliability` 0.5, no anchors (the global `min_len_ratio_uncalibrated` bound 0.4 applies), `lid` block from `calibration/lid_blocks_new8_2026-11-01.json` (GlotLID v3 on 800 training targets per language of stage-B shard 6, a proxy for human text; recall 0.976-1.000, minimum length 30 for guw lue tll lub and 60 for zne crs mfe arz, `cross_accept` <= 0.0012). `arz` carries `weak: true`: its label set includes Standard Arabic (`arb_Arab`), so the guard cannot tell Egyptian from Standard Arabic text. Builder: `scripts/build_calibration_v6.py` (reads v5, never edits it). Prompts: `scripts/build_rl_prompts_new.py` draws 20,000 English sources per language (5,000 in each of b64 b256 b512 b1024, seeded by crc32(iso)) from the union of four existing prompt files (afr, hau, ibo, kin of `rl_prompts_v2`, 421,339 unique sources); output `artifacts/rl_prompts_new/<iso>.parquet`, not in git.

**To use them in a run** the config must point `calibration_path` at v6 (the live `rl.yaml` still names v5) and the eight parquet files must be copied into the prompt directory; both are one-line changes made at RL launch. Tests: `tests/test_calibration_v6.py`, `tests/test_rl_prompts_new.py`; `test_languages`, `test_calibration_v5` (now sliced to the original 77) and `test_longdoc_systems` were adapted to the new counts.
