# Contamination and de-duplication scan of the candidate documents (2026-10-20)

Purpose: the English source documents of the long-context benchmark must not (a) have been in the SFT training data, (b) have been RL prompts, (c) be existing benchmark text, (d) duplicate each other. This part describes the scanner that checks all four, the evidence for its parameters, the validation, the result on the pool as it stood at 07:14 UTC (16,741 documents in 13 files; the pool is still being filled) and how to re-run it.

Code (laptop repo `gemma_mt/`, synced to the rental `longdoc_bench/code/`):

| file | role |
|---|---|
| `gemma_mt/eval/longdoc_contam.py` | library: normalisation, rolling hashes, membership, packed-stream reader, scans, duplicate detection, flag rules |
| `scripts/longdoc_contam_scan.py` | CLI (`--pool-dir --out-dir --sft-dir --rl-dir --bench-root --processes`, all thresholds) |
| `scripts/longdoc_contam_run.sh` | the rental invocation with the right paths (incremental) |
| `scripts/longdoc_contam_validate.py` | builds the controls (`build`) and scores them (`evaluate`) |
| `scripts/longdoc_contam_ngram_study*.py`, `longdoc_contam_fetch_negatives.py` | the n-gram length study and the negative-control download |
| `tests/test_longdoc_contam.py` | 14 offline tests (tiny synthetic corpora, a synthetic packed dataset, the CLI end to end); pass on the laptop (python 3.14, numpy 2.5) and on the rental with venv_gemma |

Outputs on the rental (`/workspace/sft_build/longdoc_bench/contam/`): `<pool stem>.flags.jsonl` (one line per pool document, `id` unchanged), `SUMMARY.json` (per pool file: documents, flag counts, share quantiles, per-benchmark hits, scan provenance; plus the validation block and run time/RAM), `validation.json` and `validation/` (control pool files, their flags, `controls.json`), `exp_n.json`, `exp_n_excerpts.json` (n-gram study), `cache/` (per-file scan cache).

## 1. Method

One primitive serves all four checks. A document is a sequence of units (normalised words, or Gemma token ids). Every window of `n` consecutive units is hashed to a uint64: each unit id is first mixed (splitmix64 for token ids, blake2b-8 for words, so the ids are the same in every process and every run), then a polynomial rolling hash `h = (...(x0*B + x1)*B + x2...)` mod 2^64 is evaluated for all windows at once with numpy (Horner form, `n` vectorised passes; a test checks it against a pure-python hash). The **share** of a document for a source is the fraction of its windows found in that source; the **longest run** is the longest stretch of consecutive matching windows (r matches = r + n - 1 units).

The candidate side (the pool) is hashed once and kept as a sorted unique uint64 array (`np.unique`); the huge training side is streamed in chunks, every chunk's window hashes are looked up with `np.searchsorted` in that array, and a hit sets a byte in a shared (fork-inherited) array. No python set holds a training hash. Per-document statistics are read back through each document's own positions in the sorted array. Windows never cross a training sample / RL window / benchmark row; in a candidate they do cross paragraph breaks (a paragraph break is a plain space in the single-line training text).

| source | space | n | what is scanned |
|---|---|---|---|
| `sft` (`sft_share`, `sft_longest_run`) | Gemma tokens | **12** | the English side of the packed SFT stream, train 11,329,709 + val 26,938 samples, all en2x: 687,225,030 English tokens (of 2.20 B in the stream) |
| `sft_word` (`sft_word_share`, `sft_word_longest_run`; extra) | words | 8 | the same English regions DECODED back to text, then normalised like the RL text; robust to formatting that changes the tokenisation |
| `rl` | words | 8 | `gemma/rl_prompts_v2/*.parquet`, column `english`: 77 files (not 78), 5,940,000 rows, 1,406,812 distinct windows (76% are shared between languages), about 314 M words (FineWeb-Edu + Wikipedia 20231101). Duplicates inside a file are skipped; across files they are scanned again (cost: 4x, 100 s) |
| `bench` | words | 8 | English of flores_plus devtest, flores_moz, ntrex, mafand, smol_doc, smol_sent, glot_story, salt, ibom_mt, and `bench_dev.jsonl` (13,600 distinct dev sentences = the FLORES+ dev file `flores_dev_all_v2.jsonl` has the same 13,600; counted once). Consecutive rows are also joined (the n-grams across the join) so a copy of two neighbouring FLORES sentences is one run. GATITOS is word level (< 8 words): not scannable |
| `dup` | words | 8 | the other pool documents, pairwise |

Normalisation is `contamination.normalise` (NFKC, casefold, every run of non-alphanumerics to one space); the word n-gram is `contamination.NGRAM` = 8. For the token path the pool document's whitespace is collapsed to single spaces first (the English side of the stream never contains a newline token: 0 of 200,000 sampled samples) and it is tokenised alone with the Gemma tokenizer without special tokens.

**Packed layout** (read from `prepare_sft.py`, `packed_dataset.py`, `format.py` and checked on the data): sample `i` is `tokens[sample_offset[i] : +sample_len[i]]` = `<bos><start_of_turn>user\nTranslate the text to {Language}:  {english}<end_of_turn>\n<start_of_turn>model\n` (that is `prompt_len` tokens) followed by `{iso} {target}<end_of_turn>\n`. The English is what lies between the double-space token (id 138, the first occurrence after the header; header length 10-14 tokens) and the fixed 5-token tail (106, 107, 105, 4368, 107). All 11,356,647 samples of train and val matched this layout (0 invalid; a mismatch would be counted and reported), `direction` is 0 (en2x) everywhere. A re-tokenisation of the decoded English of 300 random samples gave exactly the in-stream token ids (300/300), which is the premise of the token path. `--sft-scope all` scans every token of the stream instead (it would also see English-lexified targets such as Pidgin).

**Duplicates inside the pool.** Documents are ranked by number of words (longer first), then pool file name, then line; the highest-ranked copy is kept. For every later document the earlier document sharing the largest number of its unique windows is found (pairs generated per hash group with a vectorised triu index; windows contained in more than 25 documents - licence footers, stock phrases - are ignored). `dup_share` = shared windows / windows of the later document (a short document wholly inside a longer one has share 1.0); `dup_of` is set when it is >= 0.5. This stage is always recomputed over all pool files, because a new file may hold the longer copy of an old document.

**Other columns:** `paragraph_exact` = paragraphs of >= 12 words all of whose windows occur in RL or benchmark English (verbatim up to the hash); `bench_hit_benchmarks` = benchmarks with >= 8 matching windows, `bench_hit_ngrams` the counts; `n_ngrams_word`, `n_ngrams_tok`, `flag_reasons`.

## 2. Parameters and the evidence for them

### Token n-gram length (12)

Setting: 300 random SFT English windows (whole windows, re-tokenised alone) as positives, 80 documents of 1,000 words from 20 Project Gutenberg novels as negatives, the full train+val stream scanned once for n = 6..32 (54 s with 12 processes, cold). `exp_n.json`:

| n (tokens) | positives: share of n-grams found | Gutenberg: n-grams found (mean) | Gutenberg: largest document share | Gutenberg documents with any hit (of 80) |
|---|---|---|---|---|
| 6 | 1.000 | 2.37 % | 7.7 % | 80 |
| 8 | 1.000 | 0.234 % | 2.5 % | 40 |
| 10 | 1.000 | 0.107 % | 2.2 % | 13 |
| **12** | 1.000 | **0.075 %** | **2.0 %** | 9 |
| 16 | 1.000 | 0.036 % | 1.7 % | 5 |
| 20 | 1.000 | 0.017 % | 1.4 % | 1 |
| 24 | 1.000 | 0.013 % | 1.1 % | 1 |
| 32 | 1.000 | 0.005 % | 0.4 % | 1 |

Whole windows are trivially recovered at any n. The question is a copied excerpt: 92 excerpts of 60-300 words cut at word boundaries out of SFT windows and re-tokenised alone (`exp_n_excerpts.json`): the share of their n-grams found is 0.9922 (n = 8), 0.9920 (12), 0.9918 (16), 0.9915 (20), with a mean of 1.10 missed windows per excerpt for every n: the interior tokens are identical, only the first token of a standalone text differs (no leading-space marker), so the loss is one window at the start and does not depend on n. n < 10 lets common phrases through (6: every Gutenberg document has hits); beyond 12 the false positives do not fall much further, and the remaining Gutenberg hits are real overlaps (the matching spans are passages of Alice in Wonderland, The Yellow Wallpaper and A Study in Scarlet that are in the training stream), so they do not shrink with n. n = 12 tokens (about 9 words, the same strength as the word 8-gram) keeps the most sensitivity to lightly edited copies at 0.075 % noise. 64-bit collisions: about 1.2e7 candidate windows x 6.9e8 stream windows / 1.8e19 = 5e-4 expected spurious matches per run.

### Thresholds

Defaults (all CLI options): `clean` if every share < **0.02**; `weak` if the largest share < **0.10**; `contaminated` otherwise, and also when a verbatim run >= **50 words** (rl, bench, decoded sft) or >= **64 tokens** (sft) exists, or `dup_share` >= **0.5**. The shares in the rule are `sft_share`, `sft_word_share`, `rl_share`, `bench_share` and `dup_share`.

Evidence, negative control = 112 Wikinews documents written 2025-08 .. 2026-05 (90 articles, 25 concatenations of 3; after the data was collected, and not in the 2023 Wikipedia / FineWeb-Edu windows), scored without the duplicate rule (the triples contain the articles):

| share | median | q90 | q95 | q99 | max |
|---|---|---|---|---|---|
| `sft_share` (tokens) | 0 | 0.15 % | 0.38 % | 1.15 % | 2.0 % |
| `sft_word_share` | 0 | 0.30 % | 0.72 % | 2.2 % | 3.2 % |
| `rl_share` | 0 | 0.31 % | 0.67 % | 2.6 % | 3.9 % |
| `bench_share` | 0 | 0 | 0 | 0 | 0 |
| longest run (words / tokens) | 0 | - | 10-13 | - | 12-15 |

`clean_below` = 0.02 sits at about the q99 of the noise (stock phrases of 8-13 words, e.g. "inducted into the Rock and Roll Hall of Fame in 2019", in short documents of ~150-260 words; the three `weak` documents are exactly these). `weak_below` = 0.10 is 2.6x the largest negative share and a third of the smallest tested copy (30 %). The run limits (50 words / 64 tokens) are 3.5-4x the longest accidental run (15). The real pool has documents far longer than this control (median 0.5-3 k tokens), where one stock phrase weighs less.

## 3. Validation (`validation.json`, controls in `validation/pool`, seeded)

Run on 2026-10-20 with the same code and parameters as the pool scan. Controls are judged without the duplicate rule except the duplicate control itself (the controls contain each other's text on purpose).

| control | n | result |
|---|---|---|
| positive: random RL English windows (whole window = document; 70/76/73/81 per length bin b64/b512/b256/b1024) | 300 | **300/300 contaminated** (rl_share = 1.0 for all) |
| positive: random SFT English windows decoded from the packed stream (270 train + 30 val) | 300 | **271/300 (90.3 %)**; 248/248 of the windows of >= 12 tokens (179 of 12-39 tokens, 69 of >= 40 tokens), 23/52 of the windows shorter than 12 tokens. All 29 misses have no 8-word / 12-token window at all (fragments such as "A vine supports the branches ."), so no n-gram method can see them. Token-path share >= 0.5 alone: 248/300; the decoded-word path adds the 23 windows of 8-11 words |
| positive: benchmark passages (50 FLORES+ devtest x 8 consecutive rows, 25 NTREX x 8 rows, 25 SMOL documents) | 100 | **100/100 contaminated, the right benchmark named in all 100**; bench_share 0.96-1.0 |
| negative: Wikinews 2025-08 .. 2026-05 | 112 | **109 clean (97.3 %)**, 3 weak (shares 2.0-3.9 %, stock phrases), 0 contaminated |
| negative(?): Project Gutenberg slices (80 x 1000 words) | 80 | 67 clean, 6 weak, 7 contaminated: NOT a valid negative control, the hits are real: the RL windows (FineWeb-Edu / Wikipedia) quote long classic passages (Frankenstein 18 % of one slice, a Moby Dick chapter 57 words in a row, Dorian Gray 39 words) and the SFT stream contains passages of Alice in Wonderland, The Yellow Wallpaper and A Study in Scarlet |
| partial: 40 docs, 100-200 words copied from an RL window + fresh Wikinews words, 30 % copied | 40 | expected share (exact, with the n-1 window loss) 0.291, measured mean 0.292 (0.279-0.312), mean abs error 0.001, max 0.011; all 40 contaminated |
| partial: same, 60 % | 40 | expected 0.590, measured 0.590 (0.571-0.624), max error 0.006; 40 contaminated |
| partial: SFT window slices, 30 % (tokens) | 40 | expected 0.255, measured 0.256 (0.207-0.308), max error 0.008 (the decoded-word share is 0.285 of words); 40 contaminated |
| partial: SFT window slices, 60 % (tokens) | 40 | expected 0.554, measured 0.554 (0.459-0.646), max error 0.004; 40 contaminated |
| duplicates: 40 Wikinews originals + full copy, 60 % prefix, 30 % prefix (own pool, own scan) | 160 | **120/120 copies flagged contaminated with `dup_of` = the original, share 1.0** (a prefix is wholly contained); 38/40 originals clean, 1 weak (2.4 % shared with an older article) and 1 contaminated (11.6 % shared with another article) - real overlaps between Wikinews articles |

The shares track the construction to within one or two windows, so "share" can be read as "fraction of the document that is training text" (the nominal fraction is slightly above it because a copy of c units yields c - n + 1 windows).

## 4. Result on the pool (snapshot 2026-10-20 07:14 UTC, 16,741 documents)

Flag counts (`clean / weak / contaminated`; "docs >= 2 %" = documents whose share for that source is at least 2 %; "dominant" = which share is largest among the contaminated documents):

| pool file | docs | clean | weak | contaminated | docs >= 2 %: sft tok / sft decoded | rl | bench | dup | contaminated, dominant sft / rl / bench / dup | benchmarks hit (docs) |
|---|---|---|---|---|---|---|---|---|---|---|
| federalregister__pd | 600 | 297 (49.5 %) | 156 (26.0 %) | 147 (24.5 %) | 0 / 1 | 2 | 0 | 301 | 0 / 0 / 0 / 147 | |
| fedhealth__pd | 150 | 82 (54.7 %) | 41 (27.3 %) | 27 (18.0 %) | 18 / 21 | 43 | 0 | 47 | 0 / 1 / 0 / 26 | |
| govuk__permissive | 750 | 552 (73.6 %) | 110 (14.7 %) | 88 (11.7 %) | 1 / 9 | 8 | 0 | 188 | 0 / 3 / 0 / 85 | |
| gutenberg__pd | 702 | 680 (96.9 %) | 11 (1.6 %) | 11 (1.6 %) | 2 / 5 | 11 | 0 | 6 | 1 / 7 / 0 / 3 | |
| legislation__permissive | 0 | | | | | | | | | (empty file) |
| nasa__pd | 1,600 | 894 (55.9 %) | 321 (20.1 %) | 385 (24.1 %) | 8 / 24 | 501 | 0 | 532 | 1 / 122 / 0 / 262 | |
| openstax_ccby_current__permissive | 130 | 109 (83.8 %) | 18 (13.8 %) | 3 (2.3 %) | 0 / 2 | 4 | 0 | 15 | 0 / 1 / 0 / 2 | |
| openstax_ccby_snapshot__permissive | 831 | 716 (86.2 %) | 60 (7.2 %) | 55 (6.6 %) | 9 / 22 | 32 | 0 | 72 | 2 / 21 / 0 / 32 | |
| owid__permissive | 485 | 338 (69.7 %) | 96 (19.8 %) | 51 (10.5 %) | 0 / 4 | 5 | 0 | 138 | 0 / 3 / 0 / 48 | smol_doc 1 |
| wikinews__pd | 560 | 515 (92.0 %) | 24 (4.3 %) | 21 (3.8 %) | 4 / 9 | 14 | 3 | 23 | 0 / 6 / 2 / 13 | flores_plus 2, flores_moz 2, ibom_mt 2, bench_dev 1 |
| wikinews__permissive | 5,856 | 5,182 (88.5 %) | 413 (7.1 %) | 261 (4.5 %) | 122 / 169 | 131 | 72 | 404 | 7 / 24 / 58 / 172 | flores_plus 38, flores_moz 38, ibom_mt 38, bench_dev 34, mafand 1 |
| wikipedia__share_alike | 2,577 | 1,877 (72.8 %) | 246 (9.5 %) | 454 (17.6 %) | 42 / 45 | 651 | 0 | 47 | 10 / 433 / 0 / 11 | |
| wikivoyage__share_alike | 2,500 | 2,198 (87.9 %) | 200 (8.0 %) | 102 (4.1 %) | 21 / 32 | 64 | 9 | 205 | 4 / 35 / 8 / 55 | flores_plus 9, flores_moz 9, ibom_mt 9, bench_dev 9 |
| **total** | 16,741 | 13,440 (80.3 %) | 1,696 (10.1 %) | 1,605 (9.6 %) | | | | | | |

(`flores_moz` and `ibom_mt` contain the same FLORES English, hence identical hit counts; the dev sets in `bench_dev` include FLORES+ dev.)

What the training data clearly contained:

* **Wikipedia (`wikipedia__share_alike`)**: 651 of 2,577 documents (25 %) have >= 2 % of their windows in the RL windows (the RL set contains Wikipedia 20231101); 433 are contaminated mainly by RL. Wikipedia as a source is the most contaminated one by design. Another 42 documents overlap the SFT stream (>= 2 % tokens).
* **NASA (`nasa__pd`)**: 501 of 1,600 (31 %) documents have >= 2 % in RL windows, 122 are contaminated by RL; one page ("A GOLDen Look at the Ionosphere", `nasa:science-1050515`) is 100 % identical to RL windows, so FineWeb-Edu contains nasa.gov pages. Plus heavy internal duplication (532 documents >= 2 % shared with another NASA page, 105 of them copies >= 50 % of a longer page; 262 contaminated by the dup rule: news / feature families that repeat each other).
* **Wikinews**: the FLORES+ source articles are in the pool: 38 `wikinews__permissive` and 2 `wikinews__pd` documents hit FLORES+ / FLORES-MOZ / IBOM (the 2013 IPC Alpine World Championships reports, the Iraq Study Group Report, Brzezinski obituary ...: 27-42 % of their windows are FLORES sentences), 34 hit the dev sets. 122 documents have >= 2 % of their tokens in the SFT stream, 7 of them more than 20 % (9 by decoded words) (e.g. `wikinews:2890848` about toilet-paper limits in Australia: 41 % of tokens, 60 % of decoded words), so the SFT data contains Wikinews text beyond FLORES.
* **OpenStax**: little (current 4 documents, snapshot 32 documents in RL; 9 snapshot documents >= 2 % SFT tokens) but 72 snapshot / 15 current documents share >= 2 % with each other (the same chapters in several textbooks, e.g. the Federalist Papers in two books).
* **Gutenberg**: 11 of 702 contaminated (7 by RL: widely quoted passages).
* **Fedhealth (`fedhealth__pd`)**: 18 of 150 documents (12 %) have >= 2 % of their tokens in the SFT stream (21 by decoded words) and 43 (29 %) in the RL windows: US federal health text is in the training data, the only government source of the pool for which the SFT stream itself matters.
* **GOV.UK, OWID, Federal Register**: almost no overlap with the training data (RL >= 2 %: 8, 5, 2 documents; SFT tokens: 1, 0, 0) but a lot of template text shared inside the pool (GOV.UK 188 of 750 documents, OWID 138 of 485, Federal Register 301 of 600 have >= 2 % dup_share; 85 / 48 / 147 documents are contaminated by it). These are the dup rule's work, not a training-data problem: the shorter / later copies are what the selection step would drop.
* **Wikivoyage**: 9 documents hit the benchmarks (FLORES+ includes Wikivoyage text), 21 SFT-token overlaps.

Selection guidance: `flag == clean` is the strict set (80 % of the pool today); `weak` (largest share 2-10 %, usually a few stock phrases) can be admitted if the pool is short on a domain. `flag_reasons` says what triggered a document. A document dropped for `dup` has a kept sibling named in `dup_of`.

## 5. Runtime, memory, re-running

Measured on the rental (128 cores shared with the trainer and other agents, `nice -n 10`, 14 worker processes, CPU only); the figures in the table are from the pass that scanned 7,727 new documents / 12.1 M Gemma tokens (07:02-07:06), the following pass (750 new documents, box busier) took 314 s with RL 136 s, SFT tokens 24 s, SFT decoded words 111 s, i.e. the cost is dominated by the fixed pass over the training data and varies by about 1.3x with the load (`SUMMARY.json` -> `run`, logs in `longdoc_bench/logs/contam_scan_*.log`):

| stage | seconds | note |
|---|---|---|
| loading + word-hashing the pool, tokenising the new files | ~20 | single process / tokenizers |
| benchmark English (10 sets, ~96 k texts) | 8 | single process |
| RL scan (77 parquet files, 5.94 M rows, 314 M distinct words) | 118 | one task per file; the dominant cost is python normalisation, not the lookup |
| SFT token scan (687 M English tokens, 115 chunks) | 20 (9 with a small pool, 54 cold for 8 n values at once) | numpy rolling hash + searchsorted |
| SFT decoded-word scan (decode + normalise 11.4 M windows) | 68 | tokenizers `decode_batch`, single-threaded per worker (rayon would deadlock after the fork) |
| duplicate detection (16,291 documents; 15 s for 16,741) | 5 | always over all files |
| **total wall time** | **263 s** | peak RSS 2.0 GB (main) and 2.9 GB (largest worker; the candidate arrays are shared by fork), so at most ~45 GB with 14 workers, far less in practice |

Estimate (not measured): the lookup cost grows with the number of candidate windows times log of their count, so a pool 10x larger (about 120 M windows, 1 GB sorted array) should take about 10-15 min for the scans and a few GB more in the main process.

Re-run: `bash /workspace/sft_build/longdoc_bench/code/scripts/longdoc_contam_run.sh [--force] [--only <stem> ...] [thresholds]` (needs the code dir with `gemma_mt/eval/longdoc_contam.py`; `setsid nohup ... &` for the background). It is incremental: a pool file whose size / mtime and the parameters match its cache (`contam/cache/<stem>.scan.{jsonl,meta.json}`) is not rescanned; all new or changed files go through ONE pass over the training data (the cost above, independent of how many files); the duplicate stage and the flags/SUMMARY of all files are rewritten every time, so flags can change for old files when a longer copy arrives. Pool files are read as `*.jsonl` (the `.part` / `.rejects` files are ignored). Validation: `longdoc_contam_validate.py build --work <dir>`, scan `<dir>/pool` into `<dir>/out` and `<dir>/pool_dup` into `<dir>/out_dup`, then `evaluate`. Tests: `python tests/run_tests.py tests/test_longdoc_contam.py` (rental: `cd longdoc_bench/code; PYTHONPATH=/workspace/sft_build/pylibs /workspace/venv_gemma/bin/python tests/run_tests.py tests/test_longdoc_contam.py`; 14 passed on both).

## 6. Limitations

* **Token space for the SFT stream.** The text windows were deleted, so the `sft_share` token path needs the same tokenisation: text with different quotes, hyphens or spacing around punctuation (some SFT sources are pre-tokenised: "branches ."), or any re-formatting, is missed. The decoded-word path (`sft_word_*`) normalises like the other sources and closes most of that gap (it found 23 more of the 300 positive windows than the token path, and the Wikinews documents with the highest word shares are higher than their token shares), but it can only see what the tokens still say.
* **Windows shorter than the n-gram** (8 words / 12 tokens) cannot be matched: 17 % of the SFT English windows are shorter than 12 tokens. A benchmark document made of such fragments cannot be checked; single short sentences copied into a long document are only visible through the windows they contribute (a 25-word sentence is 18 word windows, ~2 % of a 1,000-word document, i.e. it alone makes a document `weak` at best).
* **Target side / translation-memory leakage is out of scope.** The scan is of English text only; a document that was translated into an African language in the training data, or that appears in the target of an x2en sample, is not seen (this SFT set has no x2en samples; `--sft-scope all` scans every token including English-lexified targets).
* **Paraphrase, near-duplicates with edits, and the same event written up by another outlet** (a Wikinews article about a story that Wikipedia or FineWeb-Edu also tell) are not detected; neither are translations of English training text into English-like variants. The windows catch copies, quotations and lightly edited copies (edits every <12 tokens defeat them).
* **Short documents are noisier** (a stock phrase of 12 words is 3-4 % of a 150-word document; the thresholds were calibrated on documents of 140-1,500 words, the benchmark documents are 512-4096 tokens).
* **Duplicate stage.** Priority is global over all pool files, longer first, so the kept copy may come from a different source / licence than the coordinator prefers; `dup_share` is containment (shared / own windows), so a short document inside a longer one counts as a full duplicate; hash groups larger than 25 documents are ignored.
* **Existing benchmarks** are the English sides found under `bench/test_sets` plus `bench_dev.jsonl`; a benchmark that is not on the box (e.g. AfroBench, MAFAND train/dev) is not checked. `paragraph_exact` is exact up to the 64-bit hash and only for paragraphs of >= 12 words.
* **The RL scan covers the prompts' English windows, not the model's later rollouts**, and the pool snapshot above will go stale: the pool files are still changing (an empty `legislation__permissive.jsonl` was present at the time of the last run).
* Gutenberg negatives turned out to be partly in the training data (see the validation table), so only Wikinews 2025-26 is a clean negative control; it is small (112 documents, 43 k words) and from one domain, so the false-positive rate on other domains (legal, technical) is not measured beyond the pool's own distribution.
