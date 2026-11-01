## 3. Document level (R6's documents, 8-language subset B8)

### 3.1 Whole-document sampling with the new samplers: more of the same

Core documents, chrF++ against the NLLB-3.3B reference (macro over 8 languages; in brackets the difference to whole-document greedy with the 95% paired bootstrap over documents, `*` = interval excludes 0). Same 1,278 documents for every row; R6's rows are re-aggregated on them. The last row is R6's best configuration, `sentfix` (sentence-by-sentence, greedy, unfinished sentences re-decoded by best-of-4 at T 0.7).

{{doc_whole}}

* **No sampler fixes whole-document translation beyond ~512 tokens.** Greedy loops (b1024 52% of the documents, b2048 79%) and does not finish (64% / 36%). Every sampler removes the loops and the unfinished documents (T 1.0 + min-p 0.05 finishes 100% / 98%, Gemma default 100% / 99%) but the output stays 22% (b1024) and 44-47% (b2048) shorter than the reference: **the model stops early or skips text** (median length ratio 0.77-0.79 / 0.53-0.60 in every sampled row without `min_tokens`), so chrF++ is 34-36 (b1024) and 24-28 (b2048) against 55.6 / 58.7 for sentence-by-sentence. The best whole-document rows are T 1.0 + min-p 0.05 (35.6 / 27.3), Gemma default (35.6 / 27.3), R6's best-of-4 (36.5 / 27.1): all +6 ... +9 over greedy, all -20 ... -31 below `sentfix`. **Truncation to a nucleus (top-p, min-p, top-k) does not repair the early stop**: T 0.7 + min-p 0.05 is *worse* than R6's plain T 0.7 at b2048 (24.5 against 27.1, loop 34% against 6%) because cutting the tail makes the sampler greedier and brings the loops back; T 0.5 + min-p 0.05 is between greedy and T 0.7 (22.0 at b2048, loop 64%).
* **Short documents (b256 / b512): greedy is as good as any sampler** (b256 -0.9 ... -2.0 for T 0.7 + min-p 0.05 / T 1.0 + cut-offs, b512 -0.4 ... -1.0 or tied), as at sentence level.
* **Human references** (SmolDoc clean documents, 6 languages; `st1024` has 20 documents for 4 of the languages and 1-2 for the other two, `st2048` 15 documents in total, so these two columns are indicative only):

{{doc_human}}

  The ranking is the same as with the pseudo-reference: samplers do +4 ... +9 over greedy on the stitched long documents and nothing on short ones; sentence-by-sentence is +22 (st1024) and +34 (st2048) over greedy.
* **min_tokens** (suppress the end token until a fraction of the expected length) is the one decoding-side cure for the early stop that I tried; see 3.3 (`T1.0_mp0.05_mt1.4` row: the best whole-document row, 37.4 / 30.0 at b1024 / b2048, still 18 / 29 points below sentence-by-sentence).

### 3.2 Repairing the unfinished sentence units (sentfix) with other samplers

In sentence-by-sentence translation 0.6% of the units (636 of the units of the B8 documents; 390 of them in the documents selected here) hit the token limit under greedy (looped or truncated); R6 re-decodes them with best-of-4 at T 0.7 and keeps the candidate with the highest chrF++ consensus among the finished ones. Unit level (the 636 units; chrF++ against the NLLB sentence for the 571 units that are whole sentences of a core document; "finished" = share of units whose chosen output ends with the end token):

{{doc_units}}

* **Finished rate grows with n and with temperature, chrF++ does not move.** 4 candidates: 80% (T 0.3 + min-p 0.05) ... 82.9% (T 0.5) ... 85.2% (R6, T 0.7) ... 86.5% (T 1.0 + min-p 0.05); 8 candidates: **91.4 - 92.9%** (T 0.7: 92.1%, T 1.0 + min-p 0.05: 92.9%). chrF++ of the repaired units is 34-35 for every T <= 0.7 setting and 32-34 for T 1.0; the consensus rule is better than "first finished candidate" by 0.2 - 1.2 chrF++ (`rule=cons` against `rule=first`) at no cost. The unrepaired greedy output (truncated or looped) already has chrF++ 35.0 on these units, so chrF++ cannot show a gain from the repair; only the finished rate and the loop rate do.
* **A repetition penalty on the repair is a bad idea:** greedy + repetition_penalty 1.05 finishes 55% of the units and 1.1 only 48% (chrF++ 32.1 and 25.7); the retry is itself unfinished in 45-52% of the units.
* **Document level: nothing changes.** The sentence-by-sentence documents with each of the 14 repair variants (and with no repair at all) have identical chrF++ to one decimal in every bin (49.0 / 53.5 / 55.6 / 58.4 against NLLB, same against the human references); the only differences are the finished rate (finished, b512 / b1024 / b2048: no repair 94 / 92 / 86%; repaired 97-100 / 95-99 / 93-98%, the repetition-penalty repairs at the low end) and a 0-1 point loop rate. Because only 0.6% of the units are repaired, **the repair sampler is irrelevant for translation quality**; choose it for coverage. n = 8 at T 0.7 is the best at the unit level (92.1% finished, 7.9% unfinished) and costs nothing at 636 units per 100k.

{{doc_repair}}

### 3.3 min_tokens, and what was not run

{{doc_minTokens}}
