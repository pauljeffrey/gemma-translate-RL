# The Gemma-3-270m-it en->African RL reward, step by step (state of 2026-10-17, after D-26 / D-27)

For the owner to vet. Everything below describes the code as it is NOW (after the end-of-turn discipline and the chunk-planner refactor), in the
order the numbers are computed. Every statement carries `file:line`. Abbreviations (all paths under `gemma_mt/` unless stated):

| tag | file | tag | file |
|---|---|---|---|
| R | `gemma_mt/rl/reward.py` | CP | `gemma_mt/rl/chunk_plan.py` (new, pure) |
| RS | `gemma_mt/rl/reward_shaping.py` | SEG | `gemma_mt/rl/segments.py` |
| G | `gemma_mt/rl/guards.py` | TX | `gemma_mt/rl/text.py` |
| LX | `gemma_mt/rl/lexical.py` | SM | `gemma_mt/rl/sampling.py` |
| CM | `gemma_mt/rl/comet.py` | GR | `gemma_mt/rl/grpo.py` |
| CAL | `gemma_mt/rl/calibrate_shaping.py` | TR | `gemma_mt/rl/train_rl.py` |
| FMT | `gemma_mt/format.py` | Y | `configs/rl.yaml` |
| D-nn | `DECISIONS.md` entry nn | COMET | `.venv-comet/lib/python3.11/site-packages/comet/...` (unbabel-comet 2.2.7) |

**Two things to know before reading.** (1) The per-language thresholds are NOT in the repo: `Y:69` points at `calibration/shaping.json`, written on the GPU
box by `CAL`. The only calibration files I could find are on the old vast box, `/workspace/sft_build/gemma_calibration/shaping_mafand_v3.json`
(MAFAND dev, ~500 sentences per language, 12 languages: hau ibo kin nya pcm sna swh tsn twi xho yor zul). The numbers quoted below as "yor" etc. come
from that file; the worked examples use its `yor` entry. If the file you train with differs, the numbers differ, the formulas do not. (2) All QE scores in
the worked examples are MADE UP (a fake scorer returning chosen numbers); everything else is computed by the real functions.

---------------------------------------------------------------------------------------------------------------------------------------------

## 0. The loop in one picture

```
prompt (English source, target iso)  --generate G=8 samples, T=1--> token ids (cut at first stop id)         SM:32-75, TR:433
  --> TranslationReward(samples)                                                                             R:140-266
        0 decode + end-of-turn checks     -> failure class?  (reward = class penalty, never scored)         R:147-185
        1 split source/output into sentences, align (Gale-Church), plan chunks + failed chunks              CP:41-85, SEG:143-166
        2 AfriCOMET-QE on every aligned chunk (one scorer call for the whole batch)                        R:219-228, CM:72-88
        3 per-chunk shaping curve (per language) -> [-2, +2.25]                                            RS:74-91, R:242
        4 per-chunk length penalty and lexical penalty (subtracted, never below -2)                        R:243-250
        5 weighted mean over chunks (weight = source characters; failed chunk = -2)                        CP:88-104, R:252
        6 reliability weight of the language (separate output, used on the advantage)                      R:254
  --> rewards r_i in [-2, +2.25]  (asserted)                                                                 TR:438-441
  --> GRPO advantage: standardise inside each group of 8, multiply by the language weight                    GR:32-48, TR:442-443
  --> token-level policy-gradient loss + 0.05 * k3-KL to the frozen SFT model, one on-policy update          GR:63-98, TR:490-506
```

## 1. Inputs

| input | where it comes from | cite |
|---|---|---|
| English source `english`, target tag `iso` | RL prompt file (`english, iso, language, src_afr_tokens, source_dataset, length_bin`); no reference is read | `gemma_mt/rl/data.py:1-12`, D-18 |
| prompt ids | `fmt.build_prompt(english, iso, tokenizer)` (Gemma chat template, ends with the generation prompt, first generated token = the `iso` tag) | TR:197, FMT:159-163 |
| sampled completion ids | HF `generate`, `num_return_sequences=G` (G=8), T=1.0, top_p=1.0, top_k=0, `eos_token_id=[<end_of_turn>, <eos>]`, `pad_token_id=pad` | Y:18,21-23; SM:55-57 |
| budget | `max_new_tokens` per length bin: b64 384, b256 1280, b512 2304, b1024 3584, mixed 3584 (Gemma tokens); global fallback 1024 | Y:25,40-45; TR:428-430 |
| what the reward receives | `Sample(token_ids, iso, source, post_stop_tokens)`. `token_ids` = ids up to and including the first stop id (or all `max_new_tokens` ids if none) | R:95-100, SM:68-70, TR:204-206 |
| `post_stop_tokens` (new) | number of non-padding, non-stop tokens the sampler produced after the first stop id and that the cut removed | SM:65-71, R:158 |

Generation facts relevant to the reward: finished rows are right-padded by HF with `pad_token_id`, the rollout cuts each row at the first stop id
(`cut_at_stop`, TX:47-53, SM:68). So with the stock sampler nothing after the end token ever reaches the reward (D-26 explains what that implies).

## 2. Preprocessing of the completion (`decode_completion`, TX:56-69)

1. Cut at the FIRST stop id (`<end_of_turn>` or `<eos>`), the stop id itself is excluded from the text: TX:59-60, `cut_at_stop` TX:47-53.
2. Decode twice: with and without special tokens (TX:61-62). If they differ, `special_ids_stripped` is set (TX:63). The clean text is then scrubbed of special-token
   STRINGS typed out as characters (`<start_of_turn>`, `<end_of_turn>`, `<bos>`, `<eos>`, `<pad>`, `<unk>`, `<mask>`, `<unusedNN>` ...: TX:27-29, 64) and `.strip()`ped (TX:68).
3. Language tag: `strip_iso_prefix` (FMT:292-313): the text must start with a 3-letter tag (`^([A-Za-z]{3})(?:\s+|$)`, FMT:288) that is a known language tag (FMT:289);
   `ok` = the expected tag, `wrong` = a different known tag, `missing` = no tag. With the default policy `require` anything but `ok` is a `bad_prefix` failure
   (R:167-170, FMT:316-324). The tag is removed before scoring (R:171). Only the ends are whitespace-stripped, the interior of the translation is never touched (FMT:300-301).
4. **No Unicode normalisation of the translation happens in the reward.** The only normalisation in this path is NFKC + lower-casing + punctuation stripping inside the
   copy test (`G:14-16`), which is not applied to the text that is scored. NFC is applied elsewhere (English windows upstream, `reports/sabiyarn_rl_bin_v2_2026-10-15.md:49`;
   hypotheses/references for chrF++ in `gemma_mt/eval/metrics.py:15-18`) but not here. See finding F1.

## 3. Failure classes (hard failures: no QE call, reward = the class penalty)

Penalty value of every class: `-neg_scale` = **-2.0** by default, i.e. the floor of the shaped range (`R:124-125`, `RS:62-65`); `penalty` may be set lower
but never above the floor, otherwise a failure could outscore a scored translation (R:126-128). The three end-of-turn classes can each be set lower with
`eos_*_penalty` (R:129-137). Checks run in this order, first hit wins (R:154-195):

| # | class (`status`) | detection | cite | default penalty |
|---|---|---|---|---|
| 1 | `no_end_of_turn` | the ids contain no stop id (generation hit `max_new_tokens`) = TRUNCATED. Always on | TX:59, R:154-156 | -2.0 (`eos_truncated_penalty`) |
| 2 | `post_eos_rubbish` (new, D-26) | after the first stop id there is a token that is neither a pad id nor another stop id (counted from the ids given AND from `Sample.post_stop_tokens`). Needs `eos_discipline` | TX:66-67, R:157-160 | -2.0 (`eos_post_end_penalty`) |
| 3 | `dangling_marker` (new, D-26) | before the stop id the raw decode contains `<start_of_turn>`, `<end_of_turn>`, `<\|assistant\|>`, `<\|user\|>`, `<\|system\|>`, `<\|im_start\|>`, `<\|im_end\|>` (TX:33, 69), or the text ends with a bare role line `\nmodel` / `\nuser:` / `\nassistant` (not if the source does too) (G:62-70). Needs `eos_discipline` | R:161-163 | -2.0 (`eos_dangling_penalty`) |
| 4 | `special_tokens` | any special id or special string was stripped. **Off by default** (`penalize_special_tokens: false`); otherwise they are silently removed and only counted in `special_stripped_rate` | R:68, 164-166, 152 | -2.0 |
| 5 | `bad_prefix` | tag missing / wrong (policy `require`) | R:167-170 | -2.0 |
| 6 | `empty` | nothing left after the tag | R:171-174 | -2.0 |
| 7 | `copy_source` | normalised (NFKC, lower, punctuation -> space) output equals the source, or word-set Jaccard >= `copy_jaccard` 0.9 | G:14-30, R:175-177 | -2.0 |
| 8 | `degenerate` | one word repeated more than 4 times in a row; or >= 8 words with < 30% distinct (lower-cased); or >= 80 UTF-8 bytes that zlib-compress to < 18% | G:33-50, R:178-180 | -2.0 |
| 9 | `length_ratio` | `len(text)/len(source)` in characters (both stripped) outside [0.2, 5.0] | G:53-59, R:181-183 | -2.0 |
| 10 | `wrong_language` | GlotLID top-1 label not in the language's label set; only when `lid_enabled` (**false in `Y:73`**) and the text has >= 15 characters and the language has labels | R:187-195 | -2.0 |

A failed row gets `rewards = class_penalty[status]` (R:234-235), `qe = None`, and its weight is still the language weight (R:254). Per-class rates are logged as `fail/<class>` (R:273-274).

## 4. Sentence chunks and the Gale-Church alignment

Entry: `plan_chunks(source, text, ...)`, called once per surviving row (R:203-217, CP:41-85). With `chunking: sentences` (Y:75; the dataclass default is `off`, R:75):

**4.1 Sentence splitting** (`split_sentences`, SEG:55-72). Whitespace (including newlines) is collapsed (SEG:57). A terminator run `.`, `!`, `?` or Ethiopic `።` (SEG:22) plus closing quotes/brackets (SEG:23)
ends a sentence only if it is followed by whitespace/end of text (SEG:27), the next word does not start with a lowercase letter (not for `።`, SEG:34-36), the word before it is not an
abbreviation from the list at SEG:24-26, not a single capital initial, not a dotted acronym (`U.S.`), and not a list number of <= 3 digits at the start of a segment (SEG:41-51).

**4.2 Pairing** (`align_chunks`, SEG:143-166):

| status | condition | result |
|---|---|---|
| `single` | the source has < 2 sentences | one chunk = whole source vs whole output (SEG:153-154) |
| `aligned` | equal sentence counts | i-th source sentence vs i-th output sentence (SEG:155-156) |
| `whole` | counts differ and the output has <= 1 sentence (or `chunk_align: whole`) | one chunk: all source sentences joined by a space vs the output (SEG:157-158) |
| `gale_church` | counts differ, output has >= 2 sentences | beads from the DP below (SEG:159-166) |

**4.3 Gale-Church DP** (`gale_church`, SEG:99-140), a banded dynamic programme over beads (source sentences consumed, output sentences consumed):

* bead priors (SEG:80-81): (1,1) 0.89; (1,0) omission 0.01; (0,1) extra output 0.01; (2,1) 0.045; (1,2) 0.045; (2,2) 0.011; (3,1) 0.006; (1,3) 0.006; (3,2) 0.002; (2,3) 0.002; (4,1) 0.001; (1,4) 0.001.
* length model (SEG:90-96): `c = (total output chars) / (total source chars)`, floored at 1e-3 (SEG:107, "the observed ratio of the two texts, so no per-language constant");
  for a bead with source length `l_s` and output length `l_o` (characters): `delta = (l_o - l_s*c) / sqrt(max(l_s,1) * 6.8)` (variance constant `_S2 = 6.8`, SEG:82, 94);
  `p = max(2*(1 - Phi(|delta|)), 1e-12)` (SEG:85-87, 95); `cost = -ln(prior) - ln(p)` (SEG:96); a bead of two empty sides costs 1e9 (SEG:92-93).
* search: minimise the summed cost; only cells with `|i*m/n - j| <= max(8, 0.25*max(n,m))` are visited (SEG:113-119); if the corner is unreachable the whole thing is one big bead (SEG:133-134); traceback SEG:135-140.

**4.4 What happens to leftovers** (`align_chunks` SEG:159-166 then `plan_chunks` CP:62-84):

* a bead with sources and outputs -> one chunk (joined with a space) (SEG:162-163);
* an omission bead (sources, no output) -> a chunk with an EMPTY output and `n_failed += len(sources)` (SEG:164-166); in `plan_chunks` an empty output side (or a copy of its source, Jaccard >= 0.9) with more than one pair becomes a FAILED chunk with weight = the source characters, not sent to QE (CP:79-82);
* an extra-output bead (outputs, no source) is dropped by `align_chunks` (SEG:162-166 only emits `si and oi` and `si` beads). Its characters are recovered in `plan_chunks`: `extra = len(" ".join(all output sentences)) - len(" ".join(matched outputs))`; if `extra > 0` and `extra >= 5%` of the output length, one failed chunk of weight `extra` output characters (CP:64-70);
* inside every pair with both sides non-empty, text appended beyond the language window: `over = len(b) - lim*len(a)` with `lim = p95_ratio*(1+0.15)` (calibrated) or `1.8` (uncalibrated); if `over > 10%` of `len(b)`, one more failed chunk of weight `over` characters (CP:73-78). `p95_ratio` is `len_window[lang][1]` (R:205, RS:139-142);
* weights: `chunk_agg: source_length` -> the chunk's SOURCE characters (CP:72), `mean` -> 1; the two "extra text" failed chunks weigh output characters / 1 (CP:69, 77). Whole-text mode (`chunking: off`) is a single chunk weighing the source length (CP:60-61).

## 5. AfriCOMET-QE scoring of each chunk

* Model: `masakhane/africomet-qe-stl-1.1` (`Y:93`), a `referenceless_regression_metric` over `Davlan/afro-xlmr-large-76L` with average pooling and a sparsemax layer mix (HF cache `hparams.yaml:2,9-12`). Source and translation are embedded SEPARATELY, then `[mt, src, |mt-src|, mt*src]` go through a feed-forward net with NO final activation (COMET `regression/referenceless.py:191-197`, `final_activation: None` line 93), so a score can fall outside [0, 1].
* Truncation: each side is tokenised with `truncation=True, max_length = max_positions - 2` = 510 (COMET `encoders/base.py:170-174`, `max_positions` 512). Nothing in the reward truncates; chunking exists to keep each side under it (D-23). `CM:26,67-70` only measures lengths.
* One scorer call for ALL chunks of ALL rows of the batch: `self.scorer(seg_src, seg_mt)` (R:219-221); length mismatch raises (R:222-223).
* Batching: `batched_in_length_order` sorts by `len(src)+len(mt)` characters, runs `batch_size` (32, `Y:94`) at a time and un-sorts (CM:32-44, 88). Inference under `torch.no_grad()` (CM:83); **fp16 on CUDA only** (`model.half()` when the device is cuda, CM:57-58), fp32 on CPU.
* Segments are the strings from the plan; the raw QE value `c` is NOT clipped before shaping (R:221, 242). `observe()` is a no-op unless `mode: adaptive` (R:227-228, RS:184-195).

## 6. The shaping function

Piecewise linear, monotone non-decreasing, per language (`shape`, RS:85-91; knots `make_curve`, RS:74-82). With anchors `(t_fail, t_mid, t_great)`:

```
xs = [0,  t_fail,  t_mid,  t_great,  max(1.0, t_great + 0.001)]
ys = [-2, 0,       +1,     +2,       +2.25 ]          (preset "repo"; ys[0] = -neg_scale)     RS:55-59, 80-81
shape(c) = -2 for c <= 0 ; +2.25 for c >= xs[-1] ; np.interp(c, xs, ys) otherwise               RS:87-91
```

So: failure zone `c < t_fail` runs from -2 (at c=0) to 0 (at t_fail); acceptable -> good `t_fail..t_mid` is 0..+1; good -> great `t_mid..t_great` is +1..+2; above `t_great` there is a tiny slope to +2.25 at c=1 ("a score above what human translations get is where a learned metric is gamed", RS:20-26). Reward range: **[-2, +2.25]** (`preset_range`, RS:62-65, R:124). The alternative preset `owner_note` (ys = 0,1,1,1; range [-2,+1]) exists but is not used (Q-01 resolved to `repo`, RS:57-58, `Y:67`).

Anchor sanitation (`ContentShaper.anchors`, RS:177-182; `make_curve`): `t_fail >= FAIL_FLOOR = 0.46` (RS:52, 76, 180); `t_mid >= t_fail + 0.001`, `t_great >= t_mid + 0.001` (RS:77-79). A language missing from the file uses `default_anchors = (0.46, 0.60, 0.75)` (RS:53, 178) with a warning if the whole file is missing (RS:123-126).

**Where the thresholds come from.** `CAL:39-102` scores up to 500 (english, human reference) pairs per language with the QE model on the DEV split (FLORES+ dev; or MAFAND dev for the v3 file), and also the reference of ANOTHER sentence ("mismatched"), the source copied, the reference with half its words removed, and the reference with an unrelated sentence appended (CAL:53-62). Then (`pair_anchors`, RS:242-250, called at CAL:66):

```
t_fail  = max(0.46, P90 of QE on MISMATCHED references)
t_mid   = max(P50 of QE on human references, t_fail + 0.05)
t_great = max(P90 of QE on human references, t_mid  + 0.05)
```

Values in `shaping_mafand_v3.json` (box): hau (0.663, 0.719, 0.838); ibo (0.629, 0.679, 0.789); kin (0.628, 0.678, 0.764); nya (0.584, 0.634, 0.721); pcm (0.463, 0.513, 0.564); sna (0.633, 0.683, 0.770); swh (0.626, 0.728, 0.846); tsn (0.598, 0.648, 0.776); twi (0.535, 0.585, 0.641); xho (0.593, 0.659, 0.786); yor (0.613, 0.663, 0.724); zul (0.569, 0.619, 0.767).
Note the `+0.05` floors are active for several languages (pcm: both gaps are exactly 0.05), see F5/F10.

## 7. Per-chunk length penalty, lexical penalty (subtracted from the shaped chunk reward)

Applied to every SCORED chunk, in this order, each floored at the failure value (R:242-250):

```
r = shape(c)
if length_penalty > 0:   ex = length_excess(lang, len(src_chunk), len(mt_chunk), slack=0.15);  if ex > 0: r = max(-2, r - length_penalty * ex)
if lexical_penalty > 0:  d  = lexical_deficit(lang, src_chunk, mt_chunk);                     if d  > 0: r = max(-2, r - lexical_penalty * d)
```
`rl.yaml` sets both penalties to 1.0 (Y:80-81); the dataclass default is 0 = off (R:82, 84).

* `length_excess` (RS:159-166): window `[p05*(1-slack), p95*(1+slack)]` of the language's HUMAN `len(reference)/len(source)` ratio in characters; `ex = max(0, ln(lo/r), ln(r/hi))` with `r = len(mt)/len(src)`. Zero if the language (or `"*"`) has no window or either length is 0. The window comes from the calibration file `len_ratio {p05, p95}` (CAL:60,64), clamped to `[max(p05, 0.3), min(p95, 2.5)]` when loaded (RS:139-142; e.g. zul's measured p95 of 8.13 becomes 2.5). yor: (0.77, 1.63); pcm (0.76, 1.17); hau (0.77, 2.04) before slack.
* `lexical_deficit` (RS:168-175, `LX:63-86`). Four features between English chunk and output chunk: `anchor_recall` and `anchor_precision` over "anchors" = tokens containing a digit and capitalised words that are not sentence-initial, longer than 1 letter and not all caps (LX:47-56); `cognate_recall` and `cognate_precision` over content words = letters/digits/apostrophe tokens of >= 4 characters that are not in the 120-word stop list and contain no digit (LX:26-30, 59-60). A feature is `None` (not penalised) when there is nothing to measure: no source anchors (recall); content words need >= 3 on BOTH sides (LX:37, 72); `anchor_precision` is 1.0 when the output has no anchors (LX:69). Only `anchor_recall`, `anchor_precision`, `cognate_precision` can be penalised (LX:34; `cognate_recall` is reported but never penalised because a paraphrase legitimately drops words, LX:33).
  Thresholds are PER LANGUAGE human 10th percentiles written to the calibration file under `lexical`, kept only for features whose human median is >= 0.5 (anchors) / 0.25 (cognates), with >= 30 samples (`summarise`, LX:89-105, MIN_MEDIAN LX:36). `deficit = mean over the enabled features of max(0, p10 - f)/p10` (LX:78-86). Penalty only; it is never a bonus (LX:15-18).

## 8. Aggregation across chunks

```
reward_i = ( sum_k w_k * r_k  +  sum_failed w_f * (-2) ) / ( sum_k w_k + sum_failed w_f )         CP:88-104, R:252
```
`r_k` = the shaped, penalised reward of scored chunk k (section 7); `w_k` = source characters of the chunk; every failed chunk contributes the penalty (-2.0) with its weight (omitted/copied sentence: its source characters; appended text: the excess output characters; unmatched output sentences: their characters). A one-chunk plan is the same formula with a single term. A zero total weight (blank source) now falls back to the plain mean instead of raising `ZeroDivisionError` (CP:99-101). The diagnostic `qe[i]` is the UNWEIGHTED mean of the raw chunk QE values (R:253); `None` if nothing was scored.

## 9. Reliability weights (same-source)

`weights[i] = ContentShaper.reliability(lang)` (R:254, RS:155-157), or 1.0 when `train_unusable` (R:69). With `reliability_from: same_source` (Y:82, RS:119-137) the weight is the calibration field `reliability_same_source`, falling back to `reliability` (the AUC version) when absent (RS:136-138):

* `hard = (accuracy of ranking the human reference above the mismatched one + above the half-omitted one) / 2`; `reliability_same_source = clamp((hard - 0.5)/0.4, 0, 1)`, 0 if the language is unusable (CAL:99-100) - D-24.
* AUC version (`reliability`): linear in `auc_vs_mismatched` between 0.65 (-> 0) and 0.90 (-> 1) (RS:237-239, CAL:96).
* A language with `auc_vs_mismatched < 0.7` (`--min-auc`) is `usable: false` (CAL:40, 95): the loader puts it in `unusable` and skips its anchors (RS:131-134), so it scores with the default anchors, but its weight is **0** (RS:157) and its advantage is zeroed (TR:443). A language absent from the file has weight **1.0** (RS:157).
* v3 file, `reliability_same_source` / AUC-based `reliability`: hau 1.0/0.959, ibo 1.0/0.773, kin 1.0/0.934, nya 0.978/0.659, pcm 0.80/0.426, sna 1.0/0.917, swh 1.0/1.0, tsn 1.0/0.876, twi 0.935/0.504, xho 1.0/1.0, yor 1.0/0.577, zul 1.0/0.824.
* The weight does NOT change the reward number; it multiplies the standardised advantage (section 12), because group standardisation would otherwise cancel any per-language scale (TR:443; same remark in the SabiYarn reward).

## 10. End-of-turn (EOS) discipline penalty (D-26)

Implemented as failure classes 1-3 of section 3 (so the penalty is the same -2.0 floor, un-scored): truncation (R:154-156), anything but padding after the first stop id (TX:66-67, R:157-160, `Sample.post_stop_tokens` R:100, SM:65-71), turn/role marker before the stop id (TX:33,69; G:62-70; R:161-163). The text sent to QE is always the part before the first stop id (TX:59-60) so rubbish after it can never be scored. Config: `eos_discipline` (switch for classes 2-3, R:89), `eos_truncated_penalty`, `eos_post_end_penalty`, `eos_dangling_penalty` (R:90-92; `Y:86-89`); a value above the floor raises (R:134-135). With the stock sampler classes 2 cannot occur (finished rows are padded, `tests/test_rl_loop.py::test_hf_generate_pads_after_the_stop...`), classes 1 and 3 can; see D-26 for the exact coverage and limits.

## 11. Final clipping / normalisation

There is no final clip or rescale of the reward. The range [-2, +2.25] holds by construction: every shaped value is inside it (RS:87-91), the length and lexical penalties are floored at the failure penalty (R:246, 250), failed chunks equal it (CP:97), the mean of numbers in a range stays in it. `reward_range = (min(lo, penalty, class penalties), hi)` (R:137) and the trainer ASSERTS every reward inside it every step (TR:440-441). Nothing is normalised across the batch (no running mean); the only normalisation is the group standardisation of the advantage.

## 12. GRPO advantage and update (grpo.py, train_rl.py)

* Group = the G = 8 completions of one prompt, consecutive rows (`group_size: 8`, Y:18; GR:34-37).
* `A_i = (r_i - mean_group) / (std_group + std_eps)`, population std (`unbiased=False`), `std_eps = 1e-4` (Y:58); `advantage_norm: mean` would skip the division (GR:40-46, Y:57). A group whose std < `flat_threshold = 1e-6` gets A = 0 for every member (GR:33, 43, 47), e.g. a group where all 8 failed.
* Then `A_i *= weights[i]` (the language weight of section 9) (TR:443). No clipping of A, no value function.
* Loss: per token `-A_i * ratio + kl_coef * k3`, `k3 = exp(d) - d - 1`, `d = log p_ref - log p`, `kl_coef = 0.05`, `kl_estimator: k3` (GR:51-53, 76-88; Y:59-60). `num_updates_per_batch: 1` (Y:20) -> exactly on-policy, `ratio = 1`, `clip_eps: null` (Y:61) so nothing clips; PPO-style clipping only exists for `num_updates_per_batch > 1` (GR:76-83, D-16). Aggregation `token`: sum over all completion tokens of the optimizer step / total completion tokens (Y:62, GR:89-90, TR:449). Gradient-norm clip 1.0 (Y:56, TR:506), lr 1e-6 with 10 warmup steps (Y:53-54), reference = the frozen SFT checkpoint (Y:5, D-16).
* The in-loop asserts (finite rewards, range, ratio == 1, KL ~ 0 at step 0) are TR:436-443, 500-505.

## 13. Worked numeric examples (computed with the real functions; QE values invented)

Setup: language `yor`, anchors (0.613, 0.663, 0.724), `len_window` (0.77, 1.63), lexical p10 thresholds `anchor_precision 0.333`, `cognate_precision 0.2`, `reliability_same_source 1.0` (the v3-file entry), config as in `rl.yaml` (chunking `sentences`, `source_length`, `length_penalty 1.0`, `lexical_penalty 1.0`, `reliability_from same_source`, eos discipline on). Reward range printed by the code: (-2.0, 2.25).

**A. One short sentence.**
source `The farmers from Ibadan bring 20 baskets of fresh vegetables to the market.` (75 chars)
output `yor Awon agbe lati Ibadan mu agbon ewebe tutu 20 wa si oja.` -> text after tag stripped `Awon agbe lati Ibadan mu agbon ewebe tutu 20 wa si oja.` (55 chars).

| step | value |
|---|---|
| status | `ok`; plan: 1 chunk (`single`), weight 75 |
| fake QE `c` | 0.70 |
| shaped | between `t_mid` 0.663 and `t_great` 0.724: `1 + (0.70-0.663)/(0.724-0.663)` = **1.6066** |
| length | ratio 55/75 = 0.733; window with slack `[0.77*0.85, 1.63*1.15]` = [0.655, 1.875]; inside -> excess 0 |
| lexical | anchors {ibadan, 20}: recall 1.0, precision 1.0 -> no deficit; content words: precision 1/7 = 0.1429 < p10 0.2 -> `(0.2-0.1429)/0.2` = 0.2857; mean over the 2 enabled features (0 and 0.2857) = **0.1429** |
| chunk reward | 1.6066 - 1.0*0.1429 = **1.4637** (the code returned 1.4637) |
| aggregate | one chunk -> 1.4637; `qe` diagnostic 0.70; weight 1.0 |

**B. Three chunks** (`aligned`).
source = `The market opens at eight o'clock in the morning. The farmers from Ibadan bring 20 baskets of fresh vegetables. Everyone is happy to see the new bridge.`
output = `Oja n si ni agogo mejo owuro. Awon agbe lati Ibadan mu agbon ewebe tutu 20 wa. Inu gbogbo eniyan dun lati ri afara tuntun ti won ko, ti won si ti pari ni osu to koja ni ilu naa.`
fake QE: 0.72, 0.60, 0.68.

| chunk | src chars (weight) | mt chars | ratio | QE | shaped | length excess | lexical deficit | chunk reward |
|---|---|---|---|---|---|---|---|---|
| 1 | 49 | 29 | 0.592 | 0.72 | 1.9344 | 0.1006 (ratio < 0.655) | 0.5000 | 1.3338 |
| 2 | 61 | 48 | 0.787 | 0.60 | -0.0424 (below `t_fail`: `-2 + 2*0.60/0.613`) | 0 | 0.1429 | -0.1853 |
| 3 | 40 | 98 | 2.450 | 0.68 | 1.2787 | 0.2677 (ratio > 1.875) | 0.5000 | 0.5109 |
| failed | 23.02 (output chars) | - | - | - | - | - | - | -2.0 |

The failed row exists because chunk 3's output is longer than the window allows: `over = 98 - 1.63*1.15*40 = 23.02 > 10% of 98` (CP:73-78), so the excess characters count as a failed chunk of weight 23.02.
Aggregate = (49*1.3338 + 61*(-0.1853) + 40*0.5109 + 23.02*(-2.0)) / (49+61+40+23.02) = 28.45/173.02 = **0.1644** (code: 0.16444). Without the failed chunk it would have been 0.4966. `qe` diagnostic = (0.72+0.60+0.68)/3 = 0.667; stats `chunk/aligned_rate 1.0`, `chunk/segments_per_sample 3.0`.

**C. End-of-turn cases on the same output as A** (token ids built with the fake byte tokenizer): no end token -> `no_end_of_turn` **-2.0**; end token followed by a second copy of the answer -> `post_eos_rubbish` **-2.0**; `<start_of_turn>model\n` before the end token -> `dangling_marker` **-2.0**; end token followed by pad ids -> `ok`, **1.4637** (padding is benign).

**D. GRPO on a group** (shortened to 4 members for the arithmetic; real G = 8): rewards [1.4637 (A), 0.1644 (B), -2.0 (truncated), -2.0 (rubbish)]; mean -0.5930; population std 1.4801; advantages `(r-mean)/(std+1e-4)` = [1.3894, 0.5117, -0.9506, -0.9506]. With weight 1.0 (yor) unchanged; with a weight of 0.80 (pcm, same-source) [1.1115, 0.4093, -0.7604, -0.7604]. A group whose members all failed (all -2.0) has std 0 < 1e-6 -> all advantages 0.

## 14. All weights and thresholds (config key, dataclass default, `rl.yaml` value)

| quantity | key | default | rl.yaml | cite |
|---|---|---|---|---|
| shaping preset / range | `preset` | `repo` [-2, +2.25] | `repo` | R:52, RS:55-59, Y:67 |
| failure/negative scale | `neg_scale` | 2.0 | 2.0 | R:55, Y:70 |
| failure penalty | `penalty` | `None` = -neg_scale | null | R:60, Y:71 |
| fail floor of `t_fail` | `FAIL_FLOOR` (constant) | 0.46 | - | RS:52 |
| default anchors | `default_anchors` | (0.46, 0.60, 0.75) | - | R:56, RS:53 |
| mode | `mode` | `calibrated` | `calibrated` | R:53, Y:68 |
| calibration file | `calibration_path` | None | `calibration/shaping.json` | R:54, Y:69 |
| prefix policy | `prefix_policy` | `require` | `require` | R:61, Y:72 |
| copy test | `copy_jaccard` | 0.9 | (default) | R:62, G:19-30 |
| length-ratio hard failure | `min_len_ratio`/`max_len_ratio` | 0.2 / 5.0 | (default) | R:63-64, G:53-59 |
| repetition guard | `repetition_guard` (+ fixed thresholds 4 / 0.3 / 8 words / 0.18 / 80 B) | True | (default) | R:65, G:33-50 |
| GlotLID | `lid_enabled`, `lid_min_chars` | False, 15 | false | R:66-67, Y:73 |
| penalise special tokens | `penalize_special_tokens` | False | (default) | R:68 |
| QE truncation (stat only) | `max_comet_tokens` | 510 | 510 | R:70, Y:74 |
| chunking | `chunking` | `off` | `sentences` | R:75, Y:75 |
| chunk weights | `chunk_agg` | `source_length` | `source_length` | R:76, Y:78 |
| count-mismatch handling | `chunk_align` | `gale_church` | (default) | R:77 |
| appended-text failed chunk | `extra_output_penalty`, `extra_ratio_limit`, `length_slack` | True, 1.8, 0.15 | true, (defaults) | R:80-83, Y:79 |
| unmatched-output / appended thresholds (fixed) | 5% of output / 10% of chunk output | - | - | CP:68, 76 |
| length penalty per log-unit | `length_penalty` | 0.0 | 1.0 | R:82, Y:80 |
| lexical penalty per unit deficit | `lexical_penalty` | 0.0 | 1.0 | R:84, Y:81 |
| lexical thresholds | calibration `lexical.<feature>.p10` (human P10) | - | per language | LX:89-105, RS:143-144 |
| reliability source | `reliability_from` | `auc` | `same_source` | R:85, Y:82 |
| reliability scales | AUC 0.65 -> 0, 0.90 -> 1; same-source `(hard-0.5)/0.4`; unusable if AUC < 0.7 | - | - | RS:237-239, CAL:40, 95-100 |
| Gale-Church | priors, `_S2 = 6.8`, band `max(8, 0.25 max(n,m))`, floor 1e-12 | fixed | - | SEG:80-82, 95, 113 |
| len-window clamp | p05 >= 0.3, p95 <= 2.5 | fixed | - | RS:142 |
| EOS discipline | `eos_discipline` | True | true | R:89, Y:86 |
| EOS class values | `eos_truncated_penalty`, `eos_post_end_penalty`, `eos_dangling_penalty` | None (= penalty) | null | R:90-92, Y:87-89 |
| QE model / batch | `scorer.model_id`, `scorer.batch_size` | africomet-qe-stl-1.1, 32 | same | `Y:92-94`, TR:94-95 |
| GRPO | `group_size` 8, `prompts_per_step` 16, `temperature` 1.0, `top_p` 1.0, `top_k` 0, `num_updates_per_batch` 1 | as listed | same | TR:56-62, Y:17-23 |
| advantage | `advantage_norm` `mean_std`, `std_eps` 1e-4, flat threshold 1e-6 | as listed | same | TR:83-84, GR:33, Y:57-58 |
| KL / loss | `kl_coef` 0.05, `kl_estimator` k3, `loss_agg` token, `clip_eps` null | as listed | same | TR:85-89, Y:59-62 |
| optimiser | `learning_rate` 1e-6, `warmup_steps` 10, `grad_clip` 1.0 | TR:77-82 (lr default 1e-6) | same | Y:53-56 |
| generation budget | `max_new_tokens_by_bin` b64 384 / b256 1280 / b512 2304 / b1024 3584 / mixed 3584 | {} | as listed | TR:72, Y:40-45 |

## 15. Things that could be reward-hacked, and what guards them

| exploit | guard | gap |
|---|---|---|
| Copy the English source (QE sees a fluent source) | `copy_source` hard failure (G:19-30, R:175); per-chunk copy = failed chunk when several chunks (CP:79); the lexical penalty is built so copying cannot earn reward (LX:15-18) | a light paraphrase of the English that stays below Jaccard 0.9 is only caught by QE, GlotLID (off by default) and the human-calibrated lexical precision |
| Answer in the wrong language / wrong tag | `prefix_policy: require` (R:167-170); QE; optional GlotLID (off, Q-10) | wrong language with the right tag is not checked unless GlotLID is enabled |
| Markup, turn headers, special tokens instead of language | cut at first stop id + special strings removed (TX:59-68); `dangling_marker` (R:161-163); optional `special_tokens` | `special_tokens` is off: a stripped `<unused5>` etc. is just counted |
| Never stopping / running into the budget | `no_end_of_turn` -2.0 (R:154-156); per-bin budgets (Y:40-45); selection requires `finished_rate >= 0.95` (Y:106, TR:117) | budget too tight punishes the policy for the sampler (see sampling limits in D-26) |
| Junk after the end token | `post_eos_rubbish` (R:157-160); text cut at the stop id (TX:59) | cannot fire with the stock sampler (it pads) - armed for samplers that keep generating |
| Loops / stuttering | `degenerate` (G:33-50) | |
| Very short output (omission) | `length_ratio` < 0.2 (G:53-59); window penalty (RS:159-166); Gale-Church merges + QE | omission of a whole sentence is NOT flagged as a failed chunk in practice (F2); QE's half-omitted accuracy enters the reliability (CAL:99-100) |
| Padding with unrelated/hallucinated text (QE accuracy 0.47 = chance, D-24) | `extra_output_penalty`: unmatched output sentences and over-window chunk text = failed chunks (CP:64-78); length-window penalty; lexical precision | text that stays inside the human window passes (only QE and lexical precision see it) |
| Gaming the sentence splitter (no full stops -> one output sentence -> `whole`) | `whole` still gets the in-chunk appended-text rule (CP:73-78), the hard length ratio, the window penalty | the omitted-sentence/copy rules need > 1 pair (CP:79); a very long source scored as one chunk is truncated at 510 tokens by the encoder (F9) |
| Decomposed (NFD) diacritics to move character counts | none | F1 |
| Over-optimising QE above human level | slope above `t_great` is 0.25 per (1 - t_great) (RS:20-26); KL 0.05 to the SFT model (GR:51-53, Y:59); benchmark chrF++ model selection against a human-reference dev sample (`Y:98-106`; `select_by: benchmark_chrf` Y:104) | |
| Languages where QE is blind | `usable:false` -> weight 0 (RS:157); weight = same-source accuracy (RS:136-138) | languages with NO calibration entry get weight 1.0 (F11) |
| Noise amplification by narrow bands (pcm: 0.01 QE = 0.2 reward) | reliability weight 0.8 for pcm; group standardisation | F10 |

## 16. Findings: things that look wrong or inconsistent

* **F1 - No NFC on the hypothesis.** The reward scores and measures the decoded text as sampled (TX:56-69, R:171). Yoruba/Fon references are mixed NFC/NFD (`eval/metrics.py:15-18`). Character-based rules (`length_ratio`, window penalty, appended-text rule, lexical word forms) therefore change with the Unicode form, and a policy could shift lengths by emitting decomposed diacritics. QE's sentencepiece normalises (NFKC), so QE itself is mostly unaffected. Suggest `unicodedata.normalize("NFC", ...)` in `decode_completion` (not done: it changes numbers).
* **F2 - Omitted sentences are practically never a "failed chunk", and Gale-Church can shift pairings.** The priors (SEG:80-81) make a 2:1 merge (0.045) cheaper than an omission bead (0.01) in almost every case. Measured on 20,000 synthetic documents with one sentence dropped and character lengths from 1 to 300: an omission bead appeared in 2 (0.01%); by hand, a dropped middle sentence became a (2,1) chunk (a long source scored against a short output). So `chunk/omitted_sentences_per_sample` will read ~0 and D-23's "an omitted source sentence a failed chunk with the penalty" is dead code in practice; the omission is left to QE (whose half-omitted accuracy is part of the reliability weight). Also the comment at R:71-74 ("never a shifted pairing") is not true for the extra-sentence case: 4 source sentences vs 5 output sentences produced chunks ('Children sell oranges near the gate.' scored against 'Inu gbogbo eniyan dun lati ri afara tuntun.'). The branch is tested with a stub (`tests/test_chunk_plan.py`).
* **F3 - Stale comment in `rl.yaml:75-77`**: "when the sentence counts differ the whole pair is scored as one segment". The default `chunk_align` is `gale_church` (R:77); `whole` is only used when the output is one sentence.
* **F4 - DECISIONS.md numbering/staleness.** Two entries are called D-22 (offline testing; reverse direction). D-17 still says prompts are filtered to `src_afr_tokens <= 256` and `max_new_tokens 256` (superseded by D-23 and `Y:27,40-45`); `data.py:MAX_SRC_AFR_TOKENS = 256` and its docstring are stale against `max_src_tokens: 1024`. D-17's "Nothing after the first `<end_of_turn>` exists for the reward" was amended by D-26.
* **F5 - `reward_shaping.py` documents the wrong `t_fail`.** RS:9,12-13 and D-17 say `t_fail` = P10 of human QE, "further raised" to the P90 of mismatched QE. The code that produces the file (`pair_anchors`, RS:242-250, CAL:66) uses `max(0.46, P90 mismatched)` and never uses the human P10. Behaviour is fine; the text is not.
* **F6 - Weight units are mixed.** Normal chunks weigh SOURCE characters, the two "extra text" failed chunks weigh OUTPUT characters (CP:69, 72, 77). Harmless but a long hallucination can outweigh a short sentence; `matched` also counts the separators of empty chunks (CP:66).
* **F7 - Division by zero removed.** `tot / wsum` raised on a blank source (old R:237); `aggregate` falls back to the plain mean (CP:99-101). Numerics for non-blank sources are bit-identical (checked on 120 samples x 6 configurations).
* **F8 - Misleading stat in chunk mode.** `comet_mt_truncated_rate` (R:277-279) measures the UNSPLIT text, so it is high for long windows even though the chunks are fine; use `chunk/mt_truncated_rate` (R:263-265). Neither looks at the SOURCE side (a long unsplittable sentence or the `whole` fallback is truncated at 510 tokens unseen).
* **F9 - `whole` fallback can exceed the encoder limit** for b1024/b2048 when the output has one sentence (SEG:157-158).
* **F10 - Narrow, sensitive bands.** pcm anchors (0.463, 0.513, 0.564): the whole 0..+2 range is spread over 0.10 of QE (0.01 QE = 0.2 reward) and both +0.05 floors are active (RS:77-79, 242-250). Together with Pidgin's low reliability this is the most noise-prone language.
* **F11 - Weight 1.0 for uncalibrated languages.** The v3 file covers 12 of the 20 RL languages; afr efi ewe fon orm som sot wol have no entry (D-24 says FLORES+ for them is still missing), so they use the default anchors (0.46/0.60/0.75), no length window (1.8 fallback), no lexical thresholds, and weight 1.0 (RS:157) - the highest advantage weight for the least-verified languages, above calibrated ones such as twi (0.935) and pcm (0.80).
* **F12 - Defaults vs yaml.** The dataclass defaults (`chunking off`, `length_penalty 0`, `lexical_penalty 0`, `reliability_from auc`) differ from `rl.yaml`; a run that does not use the yaml silently gets a different reward (R:75,82,84,85 vs Y:75,80-82).
* **F13 - Unbounded raw QE.** There is no final activation (COMET `referenceless.py:93`), and the raw `c` is not clipped (the SabiYarn reward clips to [0, 1]); the shaping handles it (RS:87-91), but `qe_mean_scored` can leave [0,1].
* **F14 - One consequence of D-26 to be aware of:** `has_role_tail` flags a translation that legitimately ends on a line consisting of `model`, `user` or `assistant` unless the source does too (G:62-70); this is deliberate and rare.

## 17. Appendix - what was ported to the SabiYarn RL reward (for completeness)

`rl/chunking.py` (file-for-file port of SEG + CP + the copy guard, equality asserted by `tests/test_chunking.py`), `StructuredTranslationReward` in `rl/rewards.py` (chunking is applied to the content term only; scaffold term, gate, copy rule and length guard unchanged; one-sentence prompts are scored exactly like before, bit-identical on 150 random items for the four shaping modes), `rl/eos_discipline.py` (text-level `</s>` rules), `rl/gen_limits.py` (per-bin `max_new_tokens`: 2.5 x the bin's p99 prompt tokens + 48, multiple of 64, cap 8192, clipped by `max_seq_len` - prompt; b64 320, b256 704, b512 1344, b1024 2496, b2048 4672), config keys `reward_chunking ... reward_eos_*`, `max_new_tokens_*` in `rl/config.py`, example `rlhf/configs/rl_bin_v2.yaml`.
