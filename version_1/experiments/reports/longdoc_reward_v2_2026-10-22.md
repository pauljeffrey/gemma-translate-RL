# Why the RL reward fails at long lengths, and reward v2 for long documents - 2026-10-22 (D-60, agent R8)

Scope: diagnosis with numbers (b256 ... b4096), a flag-gated reward v2 (`gemma_mt/rl/longdoc.py`, OFF by default), offline validation on the saved outputs of the final model (R6, `reports/longdoc_eval_final_model_2026-10-21.md`) and on freshly sampled GRPO groups, the effect of the run's greedy member, cost, risks. No training was run, nothing was pushed, the live tree (`/workspace/sft_build/gemma_mt_2c`) and the final run (`/workspace/jobs/r7`) were not touched (read-only look at its config and logs). Work tree on the box: `/workspace/jobs/r8/gemma_mt_r8`; laptop mirror: `gemma_mt/` of this repo (patch + new files, list in section 9). Raw tables: `reports/longdoc_reward_v2_2026-10-22/` (`tables.txt` has every table below, plus the JSON files).

Evidence base, so that the strength of each claim is clear: 6 QE languages (hau yor swh amh zul wol), 12 core documents per bin (b256 ... b4096; 360 document x language items per system), R6's saved outputs (`sentfix`, NLLB, 9 whole-document decoding settings, windows) and 11 corruptions of `sentfix`; 72 GRPO-style groups per bin (b1024, b2048, b4096) sampled from the final model (8 samples at T = 1 plus the greedy member; 4 samples at b4096); human documents = SmolDoc references stitched to the median source length of each bin (45 languages for the repetition limits, 8 languages for the reward). Every AUC has a 95 % CI from a bootstrap over (document, language) clusters unless stated.

## 0. Summary

1. **Where the reward loses signal depends on the bin.** b1024 / b2048: the reward is mostly right and the *policy* is what fails (greedy: 49 / 76 % of the completions contain a loop, only 81 / 47 % stop; T = 1: 100 % stop but the text is truncated, reward -0.88 / -1.41 with a within-group sd of only 0.18 / 0.11). At b4096 and partly at b2048 the **reward is wrong**: the v1 repetition guards (`is_repetitive` distinct-word ratio < 0.3) fire on length alone - on 47 % of the *good* sentence-by-sentence translations and 49 % of NLLB's at b4096 (11 % / 15 % at b2048), and on 10 % of human text of ~3,100 words - so the reward cannot rank there (P(sentfix > loop-free whole-document output) = 0.61 at b4096). The same guards miss real loops at b1024 (a repeated-sentence tail adding 30 % of the text is flagged in 18 % of the cases at b1024).
2. **Alignment is a real problem**, not only a theoretical one: only 0 - 3 % of the policy's outputs have exactly the source's number of sentences, so the i-th-to-i-th shortcut never applies and every sample goes through the length DP. On documents with a known alignment the production aligner (language-aware Gale-Church, D-30) reaches pair F1 0.78 (macro over 12 corruptions; legacy 0.66): 0.62 for a repeated-sentence tail, 0.33 for a text cut at 50 %, 0.89 for 10 % dropped sentences. The new anchored aligner (numbers / names as anchors, redundant copies removed first, length model only between anchors) reaches 0.89, is exact for loops / duplicates (0.97) and is 3 - 6x faster.
3. **Aggregation**: `min` of the sentence scores is useless (AUC 0.72 - 0.79 even for a text cut in half; its sd on good output, 0.7 - 1.0, exceeds the mean), the 25th percentile is worse than the mean for local failures, the mean already sees every failed (omitted / extra / copied) chunk as -2. A tail term (CVaR 25) adds only +0.02 - 0.08 AUC against other good systems for 1 - 10 % dropped / replaced sentences, in both QE models; SSA-COMET-MTL is more tail-sensitive than AfriCOMET-QE (robust AUC for one dropped sentence 0.65 vs 0.57). The honest conclusion: *which aggregator* matters much less than *which chunks are failed* and *which texts are flagged degenerate*.
4. **Reward v2** (`reward.longdoc_v2`): continuous multi-level repetition severity with limits from human text per length bucket and measured as the excess over the source's own repetition; a stuttering sentence is a failed chunk, not a failed document; anchored alignment; `cvar_excess` aggregation (mean minus the tail deficit beyond what human documents show); a coverage term. Reward = r - s (r - (-2)), s in [0, 1]: never below the failure penalty.
5. **Validation** (section 4): good sentence-by-sentence output no longer fails (flagged 0 % at every bin, v1: 47 % at b4096); P(sentfix > loop-free whole-document) 0.98 / 1.00 / 1.00 at b1024 / b2048 / b4096 (v1: 0.96 / 0.91 / 0.61); P(sentfix > unfinished) 1.00 at b1024 - b4096 (v1 0.99 / 0.94 / 0.76); P(loop-free whole-document > looping) 0.98 / 0.89 / 0.68; looping vs unfinished are both at the floor (P = 0.53 / 0.68 / 0.83, only partly separated). Human documents: 0 % flagged, reward 0.99 - 1.08 (v1 1.08 - 1.21; the shift is the tail term, by design). The current policy under v2: T = 1 samples -1.21 / -1.65 / -1.86 (b1024 / b2048 / b4096), greedy -1.58 / -1.98 / -2.00.
6. **Greedy member** (live run: 15 sampled + 1 greedy, `greedy_adv_scale` 0.1): at b1024 / b2048 it loops in 49 / 76 % of the groups and is the only member that ever does; it inflates the group std by 1.8 / 2.0x under v1 (1.4 / 1.3x under v2), which shrinks the 15 sampled members' |advantage| by a quarter to a third; at scale 0.1 it still carries 6 / 11 % (v1) or 5 / 7 % (v2) of the advantage mass; about 40 - 50 % of a looping greedy completion's words are good text *before* the loop and get the same negative advantage. Scale 0.1 is not obviously too weak to matter (R7: grad norm 0.9 - 1.7 against 0.5 without the member) but whether it *teaches* loop avoidance cannot be shown offline; section 6 gives the monitor and the switch.
7. **Cost**: v2 scores 6 - 15 % fewer chunks than v1 (QE time unchanged or lower), its CPU planning is 1.7 - 3x faster than v1's DP (b2048 T = 1 sample: v1 128 ms, v2 58 + 21 ms), the new checks add 1 - 4 ms. Flags-off equivalence: 900 reward evaluations identical to the pristine tree; the full suite of the patched pristine tree passes (893 passed, 11 skipped).

## 1. What was measured and how

* **Items** (`scripts/longdoc_v2/build_items.py`): per bin 12 core documents (hash-selected), 6 languages; systems = `sentfix` (A), NLLB-200 3.3B (N), 256-token windows `winfix` (W), nine whole-document settings of R6 (greedy, rep105, rep110, freq005, t03, t05, t07, t10 = T 1.0, bon4) classified with R6's own flags into B = finished and loop-free, C = finished with a loop, D = did not finish; 11 corruptions of sentfix with a known alignment (drop 1 / 3 / 10 %, duplicate 1 / 3 sentences, loop tail of sentences / of words adding 30 % of the text, cut to 80 / 50 %, 10 / 30 % of the sentences replaced by another document's). `finished` of sentfix / NLLB / corruptions is set True (the question is content, not stop token).
* **Reward**: the real `TranslationReward` with the reward block of `configs/rl_rtxpro_phase2c.yaml` + `qe_ensemble` (D-55: AfriCOMET-QE-STL-1.1 and SSA-COMET-MTL, delta 1.0, calibration v8), GlotLID off, QE scores from a disk cache (`qe_cache.py`; 300 k distinct pairs x 2 models), joint-budget handling for SSA is the production code (`joint_budget.py`; chunks are sentence pairs and fit). `reward_run.py` runs items or groups.
* **Groups**: `gen_groups.py` (vLLM, 5 GB, nice) from `/workspace/jobs/a2/models/rl2c` (= final model) with R6's prompts and token budgets: T = 1, top_p 1, top_k 0, 8 samples + greedy (b4096: 4 + greedy), layout of the trainer (greedy LAST).

## 2. Diagnosis

### 2.1 (a) Alignment accuracy (`align_study.py`, `align_table.py`, `reports/.../align_v3.json`)

Documents with a *known* alignment: sentfix outputs and NLLB references of the core set (b256 ... b4096, 6 languages, 4 documents per bin) and SmolDoc human documents stitched to the median length of the bins (equal-count documents, 2 per cell); the joined text is re-split with the reward's splitter and each output sentence is labelled with the source sentence(s) it came from. Metric: pairs (source sentence, output sentence) inside one bead; F1 of recall / precision. `legacy` = Gale-Church with the pair's own length ratio (D-23); `lm` = language-aware Gale-Church + omission / extra beads (D-30, what the 2c / final runs use); `anchored` = new.

| corruption (known truth) | legacy | lm (production) | anchored | anchored, b4096 only |
|---|---|---|---|---|
| none (re-split of 1:1 translations) | 0.967 | 0.958 | **0.974** | 0.973 |
| 1 sentence dropped | 0.935 | 0.931 | **0.949** | 0.959 |
| 3 dropped | 0.898 | 0.930 | **0.933** | 0.953 |
| 10 % dropped | 0.769 | 0.885 | **0.897** | 0.897 |
| 1 sentence duplicated | 0.890 | 0.888 | **0.969** | 0.969 |
| 3 sentences duplicated | 0.841 | 0.890 | **0.967** | 0.963 |
| tail loops over 2 sentences (+30 % text) | 0.080 | 0.624 | **0.971** | 0.971 |
| tail loops over a 3-word cycle (+30 %) | 0.569 | 0.629 | **0.819** | 0.839 |
| text cut to 80 % | 0.347 | 0.762 | **0.875** | 0.892 |
| text cut to 50 % | 0.028 | 0.340 | **0.774** | 0.819 |
| 10 % of the sentences from another document | 0.894 | 0.859 | 0.891 | 0.884 |
| 30 % from another document | 0.699 | 0.664 | 0.685 | 0.673 |
| **mean of the 12** | 0.660 | 0.780 | **0.892** | 0.899 |
| ms per clean document (all bins / b4096) | 42 / 178 | 39 / 160 | **16 / 47** | |

* Per set (mean over corruptions): human 0.68 / 0.81 / **0.89**, NLLB 0.65 / 0.77 / **0.90**, sentfix 0.66 / 0.78 / **0.89** (legacy / lm / anchored).
* The two cases that matter for RL are the weakest of the production aligner: a truncated output (the T = 1 failure) and a looped tail (the greedy failure). With the 50 % cut the DP pairs a quarter of the true pairs; the shortfall and omission rules of D-30 then act on a shifted alignment. The anchored aligner removes exact duplicate sentences / repeated blocks first (budget: only the surplus over the source's own duplicates, otherwise a repeated table line of the English is "omitted"), pins sentences with numbers, kept names and (optionally) lexicon words, and aligns only the gaps with the same language-aware length model. Where it loses is where nothing in the text identifies the sentence (30 % foreign sentences of the same length): equal to `lm` there; that is the QE model's job.
* Lexicon anchors (`longdoc_lexicon`, 4 weak languages with a lexicon: kea kam sag nus; no-corruption / drop3 / drop10 / cut 50 % / replace 10 %): mean F1 `lm` 0.661, anchored 0.822, anchored + lexicon 0.829 (b4096: 0.646 / 0.835 / 0.843). Small gain, +10 - 25 ms; left OFF.
* On the **policy's own outputs** (72 groups per bin) the sentence count equals the source's in 1.7 / 1.4 / 1.0 % of the sampled members and 2.8 / 1.4 / 0 % of the greedy ones (b1024 / b2048 / b4096), and the median output / source sentence ratio is 1.00 / 1.15 / 0.75 for T = 1 (36 - 64 % of the samples have fewer sentences than the source). So the i-th-to-i-th path of D-23 is dead at these lengths and the alignment quality is exactly the DP quality above.
* Untouched human documents: legacy 0.988 > lm 0.983 > anchored 0.984 (equal within noise). Where the anchored result differs from the identity on equal-count sentfix documents (4 of 116 at b >= 1024) the anchors show a consistent shift by one sentence from some point on (a split and a merge that cancel in the count): the v1 shortcut was wrong there.

### 2.2 (c) Rule-based penalties (`calibrate_repetition.py`, `rep_study.py`)

Human limits: SmolDoc references stitched to the median source length of each bin (45 languages, 756 documents, 120 - 220 per bucket) and the NLLB references of the core set (19,014 documents of 16 languages; they are a test, not a calibration source: NLLB sentence translations do contain real word loops, e.g. "shaidar" x 8 in hau).

**False positives of the v1 guards** (share of documents flagged by `is_repetitive`; the distinct-word-ratio rule < 0.3 is the culprit, the zlib rule never fires, the run rule only on true stutter):

| source words (bucket) | human | NLLB refs |
|---|---|---|
| 230 - 560 (b512) | 0 % | 3.0 % |
| 560 - 1000 (b1024) | 0 % | 12.3 % |
| 1000 - 2000 (b2048) | 0.8 % | 32.1 % |
| > 2000 (b4096) | **10.3 %** | **56.2 %** |

On the study items (v1 flag rate | v2 severity > 0 | v2 severity >= 0.5; b1024 / b2048 / b4096):

| output | v1 flagged | v2 sev > 0 | v2 sev >= 0.5 |
|---|---|---|---|
| sentfix (good) | 1 / 11 / 47 % | 0 / 0 / 0 % | 0 / 0 / 0 % |
| NLLB | 11 / 15 / 49 % | 14 / 11 / 21 % (real NLLB repeats) | 1 / 0 / 0 % |
| windows (winfix) | 1 / 6 / 44 % | 0 / 0 / 1 % | 0 / 0 / 0 % |
| T = 1 samples | 0 / 1 / 0 % | 6 / 0 / 0 % | 0 / 0 / 0 % |
| greedy | 56 / 94 / 100 % | 68 / 100 / 100 % | 57 / 96 / 99 % (R6 loop flag: 40 / 71 / 67 %) |
| sentence tail loop (+30 % text) | 18 / 47 / 67 % | 100 / 100 / 100 % (mean severity 0.47 / 0.45 / 0.44) | 10 / 0 / 1 % |
| word-cycle tail loop (+30 %) | 21 / 44 / 67 % | 100 / 100 / 100 % (0.43 / 0.43 / 0.41) | 19 / 15 / 14 % |
| 3 duplicated sentences | 6 / 15 / 47 % | 94 / 79 / 17 % (0.09 / 0.03 / 0.00) | 0 |
| 1 dropped sentence (clean text) | 1 / 11 / 46 % | 0 | 0 |

Reading: (i) at b1024 v1 *misses* sentence-level loops (82 % of a 30 % repeated tail); at b2048 / b4096 it "detects" them with the same rule that flags 11 - 47 % of clean text, i.e. its recall at b4096 (67 %) is hardly above its false-positive rate (47 %); (ii) a single dropped sentence turns a document into "degenerate" at b4096 in 46 % of the cases (pure false positive); (iii) loop types: word cycles ("muna muna ina") and repeated sentences / blocks are found by the v2 levels at any size, a single duplicated sentence in a 177-sentence document is below the human limit (0.5 % of the text) and is intentionally free; (iv) the v1 `selfrep` rule (4-gram share > 0.35) almost never fired on these texts (0.1 % of NLLB documents); (v) NLLB loses at b4096 because of the unique-ratio rule, not because of its translations.

### 2.3 (b) Aggregation of sentence scores, both QE models (`agg_study.py`, `agg_from_rw.py`; chunk rewards of the plan after the v2 aligner; paired / robust AUC for b1024 - b4096)

*Paired* AUC = P(score(sentfix) > score(corrupted sentfix)); *robust* AUC = P(score(NLLB or window translation of the same document) > score(corrupted sentfix)), i.e. the corruption must fall below another *good* system, which is the case that matters inside a GRPO group. Ensemble channel (the production reward of D-55); AfriCOMET alone / SSA alone have the same ordering.

| corruption | mean | min | p25 | CVaR25 | 0.5 mean + 0.5 CVaR25 | mean - 1 sd |
|---|---|---|---|---|---|---|
| 1 dropped | 0.96 / 0.60 | 0.72 / 0.70 | 0.78 / 0.54 | 0.93 / 0.65 | 0.96 / 0.64 | 0.96 / 0.65 |
| 3 dropped | 0.99 / 0.74 | 0.74 / 0.72 | 0.92 / 0.61 | 0.98 / 0.79 | 0.98 / 0.78 | 0.98 / 0.79 |
| 10 % dropped | 1.00 / 0.92 | 0.74 / 0.72 | 0.98 / 0.81 | 1.00 / 0.94 | 1.00 / 0.95 | 1.00 / 0.95 |
| 10 % replaced | 0.99 / 0.84 | 0.74 / 0.70 | 0.97 / 0.81 | 0.98 / 0.86 | 0.98 / 0.86 | 0.98 / 0.85 |
| 3 duplicated | 0.74 / 0.59 | 0.62 / 0.60 | 0.72 / 0.54 | 0.73 / 0.58 | 0.74 / 0.58 | 0.73 / 0.58 |
| cut 80 % / 50 %, loops, 30 % replaced | 1.00 / 0.97 - 1.00 | 0.74 / 0.72 | 1.00 / 0.95 - 0.99 | 1.00 / 0.97 - 0.99 | 1.00 / 0.97 - 1.00 | 1.00 / 0.97 - 1.00 |
| Spearman with the known damage (8 graded versions) | 0.95 | n/a | 0.94 | 0.92 | 0.94 | 0.94 |
| sd of the score of good output (b2048) | 0.47 | 0.72 | 0.44 | 0.49 | 0.47 | 0.48 |

Per model (robust AUC of 1 dropped / 3 dropped / 10 % replaced, mean -> CVaR25): AfriCOMET-QE 0.57 -> 0.65 / 0.71 -> 0.79 / 0.80 -> 0.83; SSA-COMET-MTL 0.65 -> 0.70 / 0.79 -> 0.82 / 0.87 -> 0.90. SSA is the more tail-sensitive judge; the ensemble sits between.

Findings: (1) **min** is dominated by the extreme-value noise of 100+ sentence scores (sd 0.72 - 1.05 on good text): AUC 0.72 - 0.79 even for a text cut in half. (2) **p25** is blind to a handful of failures. (3) Against the *mean*, **CVaR25 or mean - 1 sd gains +0.02 - 0.08 robust AUC** (ensemble +0.02 - 0.05, AfriCOMET alone up to +0.08) for local failures (1 - 10 % dropped / replaced), loses 0.01 - 0.02 paired AUC for duplicates / replaced sentences and is noisier alone; a 50/50 blend with the mean keeps the gain without the noise. (4) A plain blend lowers *every* document by ~0.43 (it measures the natural spread of sentence scores), which is why v2 charges only the tail deficit **beyond the human value of the length bucket** (`cvar_excess`, shift of good systems -0.0 - -0.1). (5) Everything else is decided by failed chunks: the mean over chunks already counts omitted / extra / copied chunks as -2 (full separation of cut / looped / replaced-30 % texts, AUC >= 0.97 against other good systems). The aggregator is therefore a second-order choice; the repetition guards and the alignment are first-order.

### 2.4 (d) How much of the b1024 / b2048 drop is the reward and how much the model / sampling

v1 reward on the same documents (mean reward / share status != ok), from `tables.txt`:

| system | b1024 | b2048 | b4096 |
|---|---|---|---|
| sentfix | 0.59 / 1 % | 0.35 / 11 % | -0.64 / 47 % |
| NLLB (reward ceiling) | 0.52 / 11 % | 0.48 / 15 % | -0.54 / 49 % |
| T = 1 (whole document) | -0.83 / 0 % | -1.42 / 4 % | -1.79 / 24 % |
| T = 0.7 | -0.90 / 15 % | -1.65 / 42 % | -1.88 / 60 % |
| greedy | -1.29 / 54 % | -1.95 / 94 % | -2.00 / 100 % |

and in the sampled groups (final model; v1): greedy member stops 81 / 47 / 35 %, loops 49 / 76 / 97 % (R6's loop detector, median loop covers 25 - 30 % of the words and starts after 38 - 49 % of them); the 8 T = 1 members stop 100 / 100 / 97 % but are cut short (reward -0.88 / -1.41 / -1.76, sd inside a group 0.18 / 0.11 / 0.08; completeness and reward correlate with Spearman 0.60 / 0.82 / 0.78 inside a group).

* **b1024 / b2048: the drop is real model / sampling behaviour**. The reward punishes it correctly: at b1024 the guards flag 1 % of the good output; the loss of reward between sentence-by-sentence translation (0.59 / 0.35) and whole-document T = 1 (-0.83 / -1.42) or greedy (-1.29 / -1.95) is 1.4 - 2.3 reward points of true failure (loops, truncation) against at most 0.0 - 0.3 points of detector error (sentfix 0.35 at b2048 under v1 against 0.64 under v2 = 0.29).
* **b4096: the reward is the problem** (sentfix -0.64 under v1, +0.51 under v2: 1.15 points of false failure, ceiling NLLB -0.54 / +0.47); this is the "reward cannot rank" effect of R6.
* What the reward does *not* do at b1024 / b2048 is give the T = 1 samples a usable ranking by completeness: their rewards sit in a narrow band (sd 0.1 - 0.2), and the greedy member's -2 inflates the std. v2 fixes the second part (section 6) and makes the first more reliable (Spearman with completeness 0.89 / 0.91 / 0.92).

## 3. Reward v2

Behind `reward.longdoc_v2` (default false) and for samples whose English source has >= `longdoc_min_src_words` words (150; b256 median 110, b512 405, b1024 786, b2048 1,579, b4096 3,141); shorter samples are scored exactly as before. Code: `gemma_mt/rl/longdoc.py` (pure python, 450 lines), hooks in `reward.py` / `chunk_plan.py`.

1. **Multi-level repetition severity.** Five shares of the output: word 4-gram repeat share (`rep4`), share of words in a repeated word 8-gram (`cover8`), characters in exactly duplicated sentences of >= 4 words (`dup_sent`), characters in repeated blocks of >= 2 sentences (`dup_block`), words in the extra copies of a periodic word cycle of 1 - 30 words repeated >= 3 times (>= 6 times for one word; `cycle`, also reports whether it reaches the end of the text). Each is measured as the **excess over the same share of the English source** (an exercise list that repeats "Which of the following" is translated into a text that repeats it) and ramps linearly from the human limit of the length bucket `t0` (p99.5 of human documents x 1.15, floor 0.02 - 0.10, `calibration/longdoc_v2.json`, built by `scripts/longdoc_v2/calibrate_repetition.py`) to `t1` = 0.5 (0.6 for `rep4`) where the level is fully degenerate. Severity = the largest ramp. Reward = r - s (r - penalty): s = 0 leaves it, s = 1 is the failure penalty exactly, in between it is linear in the repeated fraction. Severity 1 gives status `degenerate`; severity >= `longdoc_pool_severity` (0.5) keeps the sample out of the consensus pool (near-loops would agree with each other).
2. **Chunk-level stutter.** A sentence with one word repeated > 4 times in a row, or with >= 30 % of its words in a cycle, is a *failed chunk* (its source weight counts as -2), not a failed document. (The v1 same-word-run rule failed the whole document: 6 % of good sentence-by-sentence documents at b4096 and 28 % of NLLB's.)
3. **Anchored alignment** (section 2.1) through the new `beads_fn` argument of `plan_chunks`; the D-30 omission / shortfall / appended-text / copy rules then act on better beads.
4. **Aggregation** `cvar_excess`: R = mean - alpha max(0, (mean - CVaR25) - gap0), gap0 = the median human value of mean - CVaR25 per bucket (0.76 / 0.76 / 0.89 / 0.95 / 0.96 / 0.99 for the six buckets; `calibrate_tailgap.py`), alpha 1; plans of < 8 chunks use the plain mean. Other modes: `mean`, `cvar`, `p25`, `min`.
5. **Coverage.** u = failed share of the source weight (omitted, copied, shortfall chunks, extra output counted at its weight); extra drop kappa x u towards the penalty (kappa 0.5), combined with the repetition severity as 1 - (1 - s_rep)(1 - kappa u). Floored by construction at the failure penalty.
6. Unfinished outputs, the eos classes, copy, wrong language, length ratio stay failure classes with -2 in every mode.

## 4. Validation (`validate_v2.py`, `reward_run.py`; items of section 1; v1 and v2 differ only in the flags)

Mean reward per class and bin:

| bin | A sentfix v1 / v2 | N NLLB v1 / v2 | W windows v1 / v2 | B loop-free whole-doc v1 / v2 | C looping v1 / v2 | D unfinished |
|---|---|---|---|---|---|---|
| b1024 | 0.59 / 0.54 | 0.52 / 0.56 | 0.49 / 0.29 | -0.83 / -1.17 | -1.94 / -1.98 | -2.00 |
| b2048 | 0.35 / 0.64 | 0.48 / 0.70 | 0.40 / 0.40 | -1.58 / -1.69 | -1.87 / -1.95 | -2.00 |
| b4096 | -0.64 / 0.51 | -0.54 / 0.47 | -0.61 / 0.52 | -1.81 / -1.85 | -1.97 / -1.92 | -2.00 |

Share of the class with status != ok (v1 -> v2): sentfix 1 / 11 / 47 % -> 0 / 0 / 0 %; NLLB 11 / 15 / 49 % -> 0 / 0 / 0 %; windows 1 / 6 / 44 % -> 0 / 0 / 0 %.

P(reward(X) > reward(Y)), 95 % CI over documents; v1 -> v2:

| pair | b1024 | b2048 | b4096 |
|---|---|---|---|
| A > B (sentfix > loop-free whole-doc) | 0.955 -> **0.979** [0.954, 0.993] | 0.907 -> **1.000** [1.00, 1.00] | 0.611 [0.52, 0.69] -> **1.000** [1.00, 1.00] |
| B > C (loop-free > looping) | 0.931 -> **0.975** [0.946, 0.992] | 0.738 -> **0.885** [0.823, 0.942] | 0.761 -> 0.680 [0.555, 0.799] |
| C > D (looping > unfinished) | 0.531 -> 0.531 [0.500, 0.605] | 0.600 -> 0.680 [0.591, 0.759] | 0.558 -> 0.827 [0.725, 0.914] |
| A > C | 0.992 -> 1.000 | 0.933 -> 1.000 | 0.737 -> 1.000 |
| A > D (sentfix > unfinished) | 0.993 -> 1.000 | 0.944 -> 1.000 | 0.764 -> 1.000 |
| B > D | 0.947 -> 0.984 | 0.841 -> 0.969 | 0.823 -> 0.913 |
| A > N (not a requirement: QE prefers NLLB slightly) | 0.46 [0.39, 0.54] | 0.43 [0.36, 0.51] | 0.50 [0.44, 0.59] |

* The requested order **sentfix > loop-free whole-document > looping greedy > truncated holds from b1024 on for the first three links (b4096: B > C only 0.68, both classes sit at -1.9)**; "looping" vs "unfinished" is only partly separated (0.53 / 0.68 / 0.83): both are failures at or near the floor, a finished loop that covers most of the output has severity ~1 just like a truncated one. That is deliberate (a loop is not worth more than a truncation) but it means the strict four-way order is *not* achieved at b1024 (C = D within CI) and b4096 (B ~ C).
* b256 / b512 (only samples >= 150 words use v2): A > B 0.61 / 0.77; at b512 class C (R6's mild loops, 13 % of greedy) scores 0.52 (about as good as sentfix 0.57) because those loops cover a few percent of the text and the penalty is proportional - B > C = 0.21 there. I regard this as correct behaviour and a sign that the R6 loop flag is stricter than a proportional penalty.
* **Human documents** (240 stitched SmolDoc documents, 8 languages, 48 per bin): status != ok 0 % at every bin under v1 and v2 (v1's distinct-word rule would flag ~10 % at b4096 in the 45-language calibration set; in this 8-language set none), severity 0, mean reward v1 1.08 / 1.12 / 1.21 / 1.19 / 1.16 -> v2 0.99 / 1.05 / 1.05 / 1.08 / 1.08 (b256 ... b4096): the -0.07 - -0.16 is the tail term (human documents above the median tail gap pay the excess). FP rate of the repetition severity on the 756-document calibration set: 0 % at every bucket (by construction p99.5, so >= 0.5 % could have been flagged; none was).
* **The current policy under v2** (reward of the final model's outputs, mean): whole-document T = 1: -0.83 -> -1.14 / -1.42 -> -1.65 / -1.79 -> -1.88 (b1024 / b2048 / b4096; v1 -> v2, items); greedy -1.29 -> -1.58 / -1.95 -> -1.98 / -2.00; in the sampled groups T = 1 members -0.88 -> -1.21, -1.41 -> -1.65, -1.76 -> -1.86 and the greedy member -1.31 -> -1.58, -1.96 -> -1.98, -2.00. The reward of a truncated sample is lower under v2 because omission is now found reliably (alignment) and charged twice (mean over chunks + coverage).
* **Ablations** (mean reward of sentfix / P(A > B) / P(B > C) / P(A > D) at b1024, b2048, b4096; `tables.txt`): full v2 0.54 / 0.98 / 0.98 / 1.00, 0.64 / 1.00 / 0.89 / 1.00, 0.51 / 1.00 / 0.68 / 1.00; **repetition v1 guards but everything else v2: -0.68 and P(A > B) 0.61 at b4096** (the ablation that matters: the whole b2048 / b4096 improvement is the repetition fix); v1 aligner inside v2: sentfix 0.60 / 0.68 / 0.58 (v2's aligner flags more failed chunks on sentfix: 1.6 vs 1.0 per document at b1024, partly real count shifts, partly omissions of garbled sentfix output) with B > C lower at b4096 (0.61 vs 0.68); no coverage term / kappa 1: no visible change in these class tests (their effect is on the size of the drop of truncated samples, not on the order); `mean` instead of `cvar_excess`: +0.03 - 0.06 higher reward for the good systems, same ordering; the plain 50/50 blend `cvar`: sentfix 0.17 / 0.26 / 0.13 (-0.4 for every good system), same ordering.

## 5. Cost (per step; box shared with the final run, nice 19)

* **QE calls**: the chunks to score are the same pairs as before - v2 scores 30.5 vs 32.3 (b1024) and 41.1 vs 48.1 (b2048) pairs per completion (-6 / -15 %). Measured cold (12 groups = 108 completions per bin, batch 32, AfriCOMET worker 2.9 GB, GPU 87 of 98 GB used by other jobs, so a pessimistic number): **b1024: 3,295 pairs, AfriCOMET 18.3 s + SSA 15.5 s = 33.8 s (0.31 s per completion, 10 ms per pair for both models)**; **b2048: 4,493 pairs, 22.3 + 21.5 = 43.8 s (0.41 s per completion)**. 130 - 210 pairs/s per model at batch 32 under contention; the live run (205 s per step with 1,344 completions, ~49 % b1024 / b2048 in the science schedule) must run the same calls several times faster than this measurement (batch 256, dedicated card); v2 does not add a QE call.
* **CPU in the reward process** (ms per sample, one core, box load ~10; plan = chunk plan, profile = repetition profile of the output + the cached source profile; `time_cpu.py`):

| sample | b1024 v1 plan | b1024 v2 plan + profile | b2048 v1 plan | b2048 v2 plan + profile | b4096 v1 plan | b4096 v2 plan + profile |
|---|---|---|---|---|---|---|
| complete (sentfix) | 28 | 16 + 11 | 61 | 27 + 23 | 305 | 56 + 47 |
| T = 1 sample | 46 | 24 + 9 | 128 | 58 + 21 | 509 | 170 + 32 |
| greedy | 74 | 25 + 13 | 305 | 80 + 43 | 705 | 218 + 55 |

  The stutter check adds 1 - 4 ms per sample. Net: v2 planning + checks cost 0 - 40 % less than v1's plan for long samples (T = 1 samples 28 - 38 % less, complete ones equal at b1024): ~38 s instead of ~57 s per step for 1,344 completions with the science bin mix (49 % b1024 / b2048 T = 1-like samples), serial in the trainer process. The repetition profile is linear (the cycle search marks positions incrementally; the first version was quadratic on a loop of thousands of words: 407 ms -> 30 ms).
* The calibrations are two json files read at construction.

## 6. The greedy member of the current run (greedy_in_group, `greedy_adv_scale` 0.1)

Read-only look at `/workspace/jobs/r7` (config `rl_science_final.yaml`, `logs/rl_FINAL.log`, `logs/rl_probe_*.log`, `logs/rl_SHAKE.log`, `runs/final/metrics.jsonl`): the run has 84 prompts x 16 per step; science (70 % of the prompts) is b1024 35 %, b2048 35 % -> about 41 of 84 prompts per step are long and each carries a greedy member. The live `metrics.jsonl` holds only the step-0 dev evaluation (when I looked the log announced a WINDDOWN at step 5), so **there is no training evidence on loop avoidance yet**; what exists is R7's measurement of gradient norm (greedy at full weight 10 - 17, no greedy member 0.5, scale 0.1: 0.9 - 1.7) and my group data (final model, T = 1 x 8 + greedy, 72 groups per bin; `group_study.py`):

| | b1024 v1 / v2 | b2048 v1 / v2 | b4096 v1 / v2 |
|---|---|---|---|
| greedy stops / loops | 81 % / 49 % | 47 % / 76 % | 35 % / 97 % |
| greedy status != ok | 53 / 49 % | 93 / 86 % | 100 / 94 % |
| sampled reward mean (sd inside group) | -0.88 (0.18) / -1.21 (0.21) | -1.41 (0.11) / -1.65 (0.14) | -1.76 (0.08) / -1.86 (0.07) |
| greedy reward | -1.31 / -1.58 | -1.96 / -1.98 | -2.00 / -2.00 |
| std inflation by the greedy member (std all / std sampled) | 1.82 / 1.37 | 2.01 / 1.30 | 2.26 / 1.44 |
| mean abs advantage of the sampled members, with / without the greedy member in the statistics | 0.60 / 0.81 (v1), 0.67 / 0.81 (v2) | 0.53 / 0.81, 0.67 / 0.82 | 0.66 / 0.86, 0.72 / 0.88 |
| greedy advantage before scaling | -0.80 / -1.07 | -2.26 / -1.75 | -1.43 / -1.30 |
| share of the token-weighted advantage mass at scale 1 / 0.3 / 0.1 | 35 / 16 / 6 % (v1), 28 / 12 / 5 % (v2) | 52 / 27 / 11 %, 41 / 18 / 7 % | 47 / 24 / 10 %, 43 / 21 / 9 % |
| Spearman(reward, completeness) of the sampled members | 0.60 / 0.89 | 0.82 / 0.91 | 0.78 / 0.92 |

Findings:
1. The greedy member is the **only** source of loop information: the T = 1 members never loop (R6: 0 - 1 % loops), they stop early instead. So loop avoidance is learnt from ~41 completions per step at most, with one sign and no contrast.
2. Its side effect is the group statistics: with `mean_std` normalisation a -2 outlier inflates the std by 1.8 - 2.3x (v1) and cuts the sampled members' advantage by a quarter to a third (0.81 -> 0.53 - 0.60) *whatever `greedy_adv_scale` is*, because the scale is applied after the statistics. v2 reduces the inflation (1.3 - 1.4x) because the T = 1 members are lower and spread more, and raises the Spearman with completeness to ~0.9.
3. About 38 - 49 % of the words of a looping greedy completion precede the loop and receive the same negative advantage (one advantage per sequence): the update pushes good text down together with the loop. At scale 1 the greedy member carries 28 - 52 % of the token-weighted advantage mass (hence R7's grad norm 10 - 17); at 0.1 it carries 5 - 11 %.
4. **Is 0.1 enough to teach loop avoidance?** The mass is small (5 - 11 %) but it is one coherent direction (the repeated tokens), and in R7's measurements the total gradient norm with the member at 0.1 is 1.8 - 3.4x the norm without it; so it is not negligible, but nothing in the logs tests that it works, and the pre-loop good tokens dilute it. I would not raise the scale before there is evidence (at full weight the gradient clip crushed the rest of the update and the KL rose fastest in R7's probes). The direct monitor already exists in the run: `bench_aux/afrisci_dev_doc/finished_rate` (whole papers, greedy: 0.773 at step 0, `yor` / `lug` 0.56) and the per-bin greedy evaluation; if that rate has not clearly risen by step ~150 - 200, the cheapest next steps in order are: (a) compute the group baseline / std over the sampled members only and apply the greedy advantage separately (removes the 25 - 35 % compression of the sampled signal; needs a small `grpo.py` change, not done here to keep my hunks inside the reward), (b) then raise the scale to 0.2 - 0.3, (c) longer term: apply the greedy member's negative advantage only from the loop onset (`rep_prof` has the cycle position; token-level advantages are a trainer change).
5. With v2 the greedy member gets a graded reward (a loop at the end of an otherwise complete document is not -2), which makes its advantage informative: b2048 raw advantage -1.75 under v2 against -2.26 under v1.

## 7. Optional sentence-marker format (design note only; needs SFT; not implemented)

Source gets numbered markers (`[1] ... [2] ...` or a pair of existing special tokens), the model must repeat them in front of each translated sentence; the reward strips them (after checking that they are present, in order, once). *Benefit*: alignment becomes exact (the anchored F1 0.89 -> ~1.0, omission / duplication / reordering are counted, not guessed; the 30 % foreign-sentence case that no aligner solves becomes visible); loops show up as repeated or missing markers; the last marker present is a completion test; the model gets an explicit position signal for long inputs, which is the most plausible way to attack the T = 1 early-stop behaviour. *Cost*: an SFT stage on marked data (the data builders for long windows exist, `data-gen` pipeline; ~2 - 3 tokens per sentence = +5 - 8 % tokens), a new prompt convention that all downstream users and evaluations must apply (R6's sentence-by-sentence decoding gets the same effect at inference for free), a copy / format hack surface (markers without content: needs a content-per-marker length check), reward changes (marker validity as a failure class), and a mismatch between marked training and unmarked deployment prompts unless every prompt is marked. *Expected effect size*: the reward gains it would add over v2 are bounded by the remaining alignment error (F1 0.89 -> 1.0) and the localised-failure sensitivity (+0.05 AUC); the larger potential gain is behavioural (completion), which cannot be estimated offline. Recommendation: do not do it before v2 has been tried in the run; if long-document completion stays poor after v2, test it on a 2 - 3 k-example SFT mix first.

## 8. Risks and honest limits

* **Evidence**: 6 languages, 12 documents per bin, one model; human long documents are stitched SmolDoc (no discourse, no real b4096 human text); the repetition limits rest on 120 - 220 human documents per bucket; weak languages (blend / consensus modes) were not run through v2 (no QE there) - their v2 repetition and anchor behaviour is covered by tests only; the alignment study uses 2 - 4 documents per cell; the lexicon result is on 4 languages.
* **Reward scale shift**: long samples lose 0.1 - 0.15 on average (tail term) and truncated samples sit near the floor (b2048 T = 1: -1.65 +- 0.14, b4096: -1.86 +- 0.07); GRPO normalises by the group std, so small QE noise differences can be amplified there (Spearman with completeness 0.9 says the signal dominates). The archive thresholds (qe 1.0) will admit fewer long samples. `reward/*` logs of long bins are not comparable with earlier runs.
* **Discontinuity at `longdoc_min_src_words`** (150 words): a 149-word and a 151-word source are scored by different code; the two give similar numbers for good output but not for failures.
* **Interactions**: v1 anchors (`anchors_repeat_weight`) still subtract their own repetition penalty for blend / consensus languages (both are floored at -2, so stacking cannot go below the penalty); `hard_fail_mask` (D-61) is untouched; the consensus pool excludes samples with severity >= 0.5.
* **Looping vs unfinished are not separated** (both ~ -2): there is no gradient between "stopped in a loop" and "never stopped".
* **Omissions are charged twice** (mean over chunks and coverage kappa 0.5): the ablation shows no effect on ordering, only on size; set `longdoc_coverage_kappa: 0` for a milder drop.
* The aligner flags more failed chunks on sentfix than v1 does (1.6 vs 1.0 per document at b1024); part of it is true (shifted counts), part is garbled sentfix output; the effect on the reward is -0.05 - -0.10.
* Timings were taken on a contended GPU/CPU and are upper bounds.

## 9. Files, flags, how to enable, tests

**Hunks of `reward.py`** (all listed in `patches/longdoc_v2_D60_reward.patch`, applies cleanly to the pristine file and, `--dry-run`, to `/workspace/jobs/r7/gemma_mt_run`; nothing of R9's anchors v2 / D-61 is touched): (1) `from gemma_mt.rl import longdoc`; (2) 12 new `RewardConfig` keys after `anchors_related_min`; (3) two new `RewardOutput` fields (`chunk_rewards`, `longdoc`); (4) `__init__`: build the calibration + validate the keys (before the `qe_ens` block); (5) start of `__call__`: `long_i`, `rep_sev`, `rep_lvl`, `rep_prof`; (6) the two repetition checks wrapped in `if long_i[i] and cfg.longdoc_repetition: ... else: <unchanged>`; (7) `okset` of the consensus block; (8) `beads_fn` added to the `plan_chunks` call; (9) one line in the chunk loop (stutter chunk = penalty) before the length penalty; (10) `rewards.append(aggregate(...))` -> long / not long branch, `ld_info` / `chunk_rw` lists; (11) after `out.science_deficit`: `chunk_rewards`, `longdoc`, `longdoc/*` stats; (12) three new methods `_longdoc_reward`, `_src_profile`, `_longdoc_stats`. `chunk_plan.py` (patch `longdoc_v2_D60_chunk_plan.patch`): optional argument `beads_fn` of `plan_chunks`, one `if/else` before the existing alignment call (default None = bit-identical). New files: `gemma_mt/rl/longdoc.py`, `calibration/longdoc_v2.json`, `configs/longdoc_v2_overlay.yaml`, `tests/test_longdoc_reward_v2.py`, `scripts/longdoc_v2/*` (study scripts, `make_patch.py`).

**Exact yaml additions** (in the `reward:` block; same as `configs/longdoc_v2_overlay.yaml`; or `reward.longdoc_v2=true reward.longdoc_calibration_path=...` as launch overrides):

```yaml
  longdoc_v2: true
  longdoc_calibration_path: /workspace/jobs/r7/assets/calibration/longdoc_v2.json   # copy of gemma_mt/calibration/longdoc_v2.json
  longdoc_min_src_words: 150
  longdoc_repetition: true
  longdoc_align: anchored          # gale_church = v1 aligner
  longdoc_lexicon: false
  longdoc_agg: cvar_excess         # mean | cvar | cvar_excess | p25 | min
  longdoc_agg_alpha: 1.0
  longdoc_agg_q: 0.25
  longdoc_coverage_kappa: 0.5
  longdoc_pool_severity: 0.5
```

New log keys (via `reward/*`): `longdoc/long_rate`, `longdoc/agg_mean`, `longdoc/agg_tail`, `longdoc/rep_severity`, `longdoc/rep_penalised_rate`, `longdoc/rep_full_rate`, `longdoc/cover_failed`, `longdoc/severity`. Resume-safe and deterministic: no state besides a pure cache of source profiles, no new checkpoint fields; switching the flag on at a restart only changes the reward of long samples from that step on.

**Tests**: `tests/test_longdoc_reward_v2.py` (27 tests: detectors, severity ramps and the exact endpoints, excess over the source, calibration file, aggregators, coverage, anchors / chain / aligner on drops, duplicates, loop tails, truncation and source duplicates, `beads_fn` = v1 on equal counts, reward ordering complete > cut > loop > unfinished, floor, natural long text v1 flags / v2 does not, stutter = failed chunk, flags off = unchanged, config validation). Full suite on the pristine tree + patch + new files: **893 passed, 11 skipped**. Flags-off equivalence: 900 study items rewarded by the pristine tree and the patched tree - reward, status and QE identical (`flags_off_check.py`).

**Scripts** (`scripts/longdoc_v2/`): `common.py corrupt.py build_items.py build_human_items.py align_study.py align_table.py calibrate_repetition.py calibrate_tailgap.py rep_study.py chunk_study.py agg_study.py agg_from_rw.py reward_run.py validate_v2.py gen_groups.py group_study.py greedy_onset.py time_cpu.py flags_off_check.py qe_cache.py make_patch.py`. Box data (scratch, removed at the end except the tree): `/workspace/jobs/r8/`.

## 10. Recommendation

Merge the patch flag-gated and enable the yaml block above at the restart; look first at `reward/longdoc/rep_penalised_rate` (expect ~0 for sampled members, 50 - 100 % for greedy members of b2048) and `reward/fail/degenerate/*` (v1 flagged ~11 % at b2048); keep `greedy_adv_scale` 0.1 and watch the whole-paper greedy finished rate; if it stays flat, change the group statistics first (section 6). Do not use `min` or `p25`. Revisit `longdoc_coverage_kappa` and the `cvar_excess` shift after the first 100 steps if long-bin rewards look compressed against the floor.
