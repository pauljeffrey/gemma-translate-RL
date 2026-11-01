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