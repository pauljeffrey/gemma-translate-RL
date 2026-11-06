# News-style English RL prompt pool for the 385M English->African model (2026-11-03)

VERIFIED = measured in this work. REPORTED = read from a file or card. Nothing was trained, no GPU, no paid API, nothing pushed to the Hub.

## 0. Hand-over note
A first attempt (an earlier session of this task) had already built release 1 (BBC, window-level decontamination, 84 languages x 22,000 rows, full b64..b1024) and uploaded it to the bucket, but left no report and nothing on the A100. This session re-checked it (loader, independent index re-check, source licences), cut it to the bins the 385M plan uses (b64+b256), put it on the A100, and then built release 2.

**Release 1 is on the A100 now**: `/workspace/clean_rerun/prompts/clean_v3_385m/news/` (84 files, 201 MB, sha256 checked against the laptop copy). Release 2 is only STAGED on the A100 in `/workspace/clean_rerun/prompts/news_r2_staged/` (outside the pool directory, so nothing reads it). To switch before a launch: `cd /workspace/clean_rerun/prompts/clean_v3_385m && mv news /workspace/clean_rerun/prompts/news_r1_kept && cp -r /workspace/clean_rerun/prompts/news_r2_staged news` (about 200 MB, 7.9 GB free).

## 1. Sources and licences (Hub API + dataset cards, VERIFIED 2026-11-03)
| source (Hub id) | used | articles | licence on the Hub | what that means |
|---|---|---:|---|---|
| `RealTimeData/bbc_news_alltime` (BBC News 2017-2025) | R1, R2 | 158,416 after cleaning | **none declared** (no licence field, no licence text in the card) | BBC-copyrighted news text, scraped. Research / reward-only use, disclose. |
| `permutans/c4-bbc-news` (C4 realnewslike, BBC domains, ~2019) | R1, R2 | 77,420 | `odc-by` ("inherited from C4") | The ODC-By label is on the C4 dump, not on the BBC text; BBC copyright still applies to the articles. |
| `SetFit/bbc-news` (Greene & Cunningham 2004-05, 2,225 articles) | R1 only (10 windows, because nearly all of its articles are also in the sources above or fail the filters) | 2,077 | **none declared**; card says "All rights, including copyright, in the content of the original articles are owned by the BBC" | The original dataset page allows non-commercial research only (REPORTED from memory of mlg.ucd.ie, not re-fetched). Not in release 2. |
| `gopalkalpande/bbc-news-summary` | **not used** | | `cc0-1.0` on the card, but it is a 417-article subset of the same 2004-05 BBC set (same copyright statement), so the CC0 label cannot be right for the text. | skipped (adds nothing) |
| `RealTimeData/bbc_alltime` (BBC 2017-2022) | **not used** (downloaded in the first attempt) | | `cc-by-2.0` on the card, but the content is BBC articles; the label looks wrong, and `bbc_news_alltime` is the newer version of the same crawl. | skipped |
| `izumi-lab/wikinews-en-20230728` (English Wikinews, 43,246 pages) | R2 | 36,961 after dropping namespace pages (Category, User, Talk, ...) | `cc-by-2.5` | Wikinews text is CC BY 2.5; attribution needed if ever redistributed. |
| `alvations/globalvoices-en-es` (English side, document order, 355,136 sentence rows) | R2 | 20,849 documents (split at the `... · Global Voices` title rows) | **none declared on this card**. Global Voices content is CC BY 3.0 (the sister card `Nicolas-BZRD/Parallel_Global_Voices_English_French` states `cc-by-3.0` for the same crawl, 2015) | CC BY 3.0 attribution. |
| `darkpredator/stealth-wikivoyage-en` (Wikivoyage 2026-09, 32,840 guides) | R2 | 31,987 | `cc-by-sa-4.0` | **Share-alike**; prose paragraphs only (listings, bullets, phone numbers dropped). Travel guide, not news: only 5 % of the draw. |

Not used because no clean article-order English text is on the Hub: `wecover/OPUS_GlobalVoices` and `sentence-transformers/parallel-sentences-global-voices` (shuffled, subsampled sentence pairs, no consecutive text), `bigscience-data/roots_en_wikinews` / `roots_en_wikivoyage` (gated, accepting the terms was not authorised). The `licence` column of every row carries the string from this table (`bbc-copyright-no-hub-licence`, `odc-by`, `bbc-copyright-noncommercial-research`, `cc-by-2.5`, `cc-by-3.0-original-no-hub-licence`, `cc-by-sa-4.0`).

The data is used as RL INPUT text only (the model's output is rewarded, the source is never trained on as a target). Owner accepts non-commercial / unclear-licence research data when disclosed: BBC rows are the ones to disclose.

## 2. Pipeline (scripts `gemma_mt/scripts/build_rl_prompts_news.py` = release 1, `build_rl_prompts_news_r2.py` = release 2; both resumable stage by stage)
1. Clean paragraphs (>= 6 words, ends in sentence punctuation, no caption / "follow us" / copyright lines, paragraphs repeated in >= 8 articles dropped as boilerplate), split into sentences (abbreviation-aware), keep sentences of 3-70 words without URLs / odd characters.
2. One window per article and bin, cut on sentence boundaries, from consecutive good sentences (60 % start at a paragraph start). Bins by **AfriCOMET-encoder tokens** (`Davlan/afro-xlmr-large-76L`, with `<s></s>`; this is the `src_afr_tokens` column and the unit of `max_src_tokens` / `length_bin` in the loader and in the 385M yaml: b64 24-64, b256 65-256, b512 257-512, b1024 513-1024). The brief said "Gemma tokens"; the existing pools, the loader and the yaml all use the encoder tokens, so I kept them (a Gemma-token column was not added).
3. Decontamination, the same rule as the earlier RL pools (`data_pipeline/sft_clean/decon.py`, via `s3_common.Checker`, index **protected-2026-10-28.3**, file sha256 `f094fd2b24d6a0e4...`, 35 protected sets, FLORES+ dev train-half exemption NOT attached): R1 exact sentence, R2 contained sentence, R3 shared word 8-gram, both normal forms, English side only. A window with any hit is removed, never repaired.
4. Release 2 adds an **article-level guard**: the cleaned full text of every article (all sources, BBC included) is checked, and an article with any hit loses all its windows (FLORES was cut from Wikinews / Wikijunior / Wikivoyage; MAFAND and NTREX are news; neighbours of a benchmark sentence come from the same document).
5. Per language (84 = SUPPORTED_RL minus `arz`), seeded draw without replacement per bin (seed `crc32(iso)`), so every language gets a different subset. Release 2 draws by source shares (below).

## 3. Removals (VERIFIED)
**Release 1 (BBC, window level):** 606,814 raw windows -> 8,532 removed (1.41 %) -> 598,282. By bin removed: b64 478 (0.27 %), b256 1,583 (0.85 %), b512 3,084 (2.0 %), b1024 3,387 (3.6 %). By source: bbc_news_alltime 5,866, c4-bbc-news 2,666. By rule (a window can hit several): R1 968, R2 186, R3 7,627. By protected set (any rule): afrilongdoc 3,167; smol 2,122; mafand:test 2,034; mafand:dev 1,690; ntrex128 1,148; ntrex_african 1,107; glot_story_book 351; afridocmt:test 308; tico19:test 278; naijas2st_news_mt:test 141; wmt24pp 114; xsim++:dev 106; flores_plus:dev 90; ibom_mt:dev 90; flores_moz:dev 90; afridocmt:dev 63; naijas2st validation 55; flores_plus:devtest 35; ssa_mte:test 23; afriscience test 15 / dev 2; salt test 33 / dev 2; the rest under 40 each. (Full dict: `work/decon_stats.json` of the first attempt, copied into the bucket directory `rl_prompts_news/stats/`.)

**Release 2 article-level check** (articles with >= 1 hit, all dropped completely):
| source | articles checked | flagged | % |
|---|---:|---:|---:|
| RealTimeData/bbc_news_alltime | 158,416 | 7,316 | 4.6 |
| permutans/c4-bbc-news | 77,420 | 2,498 | 3.2 |
| SetFit/bbc-news | 2,077 | 52 | 2.5 |
| izumi-lab/wikinews-en | 36,961 | 1,939 | 5.2 |
| alvations/globalvoices-en-es | 20,849 | 406 | 1.9 |
| darkpredator/stealth-wikivoyage-en | 31,987 | 1,125 | 3.5 |
The article guard removes **18,751 of the 598,282** release-1 BBC windows (3.1 %) on top of the window rule. Flags by set (articles): afrilongdoc 5,958; smol 3,645; mafand:test 2,509; mafand:dev 2,185; ntrex128 1,355; ntrex_african 1,305; afridocmt:test 1,138; glot_story_book 542; xsim++ dev/devtest 439/357; tico19:test 409; flores_plus:dev 401 / devtest 350 (these include Wikinews, Wikivoyage articles as expected); naijas2st test 202; afrimte 195; wmt24pp 172; ssa_mte:test 98; afriscience dev/test 226/23. After the article guard the new windows of Wikinews / Global Voices / Wikivoyage had **0** further window-level hits (186,231 windows in, 0 removed).

**Independent re-check of the shipped files (VERIFIED here, same index):** release 1 slim: 1,008,000 rows, 339,157 distinct windows, 0 hits. Release 2 slim: 1,008,000 rows, 382,101 distinct windows, 0 hits. Release 2 full: 1,848,000 rows, 626,906 distinct windows, 0 hits. The first attempt additionally ran `s3_verify.py` (a Python reference built from FLORES+ dev/devtest, NTREX-128 English, AfriScience dev/test, AfriLongDoc core/hard, 29,192 sentences / 539,745 8-grams; 300 planted controls all caught): 71,543 sampled rows, 0 hits (REPORTED from `verify_result.json` of the first attempt).

## 4. Counts
Per language and bin (every one of the 84 files has exactly these; a different random subset per language):
| release | b64 | b256 | b512 | b1024 | rows/language | where |
|---|---:|---:|---:|---:|---:|---|
| 1 slim | 6,000 | 6,000 | - | - | 12,000 | A100 `news/` (201 MB), laptop `gemma_mt/artifacts/rl_prompts_news/release1/` |
| 1 full (first attempt) | 6,000 | 6,000 | 5,000 | 5,000 | 22,000 | bucket `rl_prompts_news/*.parquet` (flat), ~1.0 GB |
| 2 slim | 6,000 | 6,000 | - | - | 12,000 | A100 `news_r2_staged/`, laptop `.../release2/` (197 MB) |
| 2 full | 6,000 | 6,000 | 5,000 | 5,000 | 22,000 | bucket `rl_prompts_news/release2_full/`, laptop scratch only (965 MB) |
All counts are >= the 5,000-per-language-per-bin goal for b64 and b256. Slim = b64 + b256 only because the 385M plan sets `max_src_tokens: 256` (b512 and longer are dropped by the loader anyway) and the loader holds all rows as Python strings.

Distinct English windows behind the 84 draws (release 2): b64 189,905; b256 192,196; b512 147,408; b1024 97,397 (a window serves about 2-3 languages on average for b64/b256, more for the scarce Wikinews / Global Voices b1024 windows).

Pool available to the draw (release 2, after both guards) per bin: b64 BBC 168,520 / Wikinews 19,411 / Global Voices 18,023 / Wikivoyage 28,310; b256 178,835 / 20,940 / 13,568 / 28,815; b512 144,161 / 13,054 / 6,885 / 18,783; b1024 88,005 / 3,637 / 3,790 / 11,015.

Release 2 composition of the 12,000 slim rows per language (identical in every language): bbc_news_alltime 30 %, c4-bbc-news 25 %, Wikinews 25 %, Global Voices 15 %, Wikivoyage 5 %. BBC share overall 55 %. In release 1 it is bbc_news_alltime 62 % / c4-bbc 38 % (about 10 rows of SetFit overall).

## 5. Schema and how the loader mixes it
Columns (same as `rl_prompts_new` / `clean_v3_385m`, plus `domain`, `licence`, `n_sentences`): `english` str, `iso` str, `language` str (canonical name, 0 mismatches), `src_afr_tokens` int64, `length_bin` str (b64 / b256 / b512 / b1024), `n_sentences` int64, `source_dataset` str, `licence` str, `domain` str = `news`, `prov_pool` (`news_v1_2026-11-03` / `news_v2_2026-11-03`), `prov_file`, `prov_row` (file and row / article id in the source), `prov_en_xxh64` (hash of the English window), `prov_index` = `protected-2026-10-28.3`, `flag_ssa_mte_train` (always False: the SSA-MTE train flag index was not applied, no flag-only check was run on news).

`gemma_mt.rl.data.load_rl_prompts(path)` walks the directory recursively, reads the optional `domain` column and, when any row has one, labels rows without it `general` (code comment: "385M plan: the FineWeb/Wikipedia files have none, the news files say news"). So `<prompts_path>/news/*.parquet` with `domain = news` plus the unchanged `clean_v3_385m/*.parquet` gives exactly `domain_weights: {general: 0.7, news: 0.3}` of `rl_00_preflight.yaml`. **No code change is needed.** Tested here (VERIFIED): `load_rl_prompts(release1 or release2, max_src_tokens=256)` -> 1,008,000 rows, 0 dropped, 0 name mismatches, 84 languages, no `arz`, domain set {news}; `PromptSchedule(..., domain_weights={'news': 1.0})` draws. Not tested here: the combined general+news load with the real `clean_v3_385m` files (that pool is only on the A100; the loader code path for it is the one quoted above).

Things the launch agent should know:
* The domain draw happens first (30 % news), then bin, then language (by prompt count of the domain+bin pool, i.e. uniform over languages for news because every language has 6,000), then the prompt. So the news share of language l is the same 30 % in every language; the general pool is natural-frequency weighted as before.
* Mixed stage: news has only b64 and b256, general has b64..b1024 but b512+ rows are cut by `max_src_tokens: 256`, so all bins present are b64 / b256 and the default uniform bin draw applies to both domains.
* The eval hold-out is "hash-selected from the same pool", so news prompts will be in it too (at 30 %), unless the per-bin minimum logic picks otherwise.
* Reward modes with an NLLB pseudo-reference (`nllb_pseudoref_dir`) have no pseudo-reference for these windows: prompts are new, so any pseudo-reference lookup falls back as for other unseen prompts (not checked in the trainer code; the pseudo-reference directory is built for the clean_v3 pool).
* Wikivoyage / Global Voices / Wikinews are only in release 2; release 1 is BBC only (the owner's first choice).

## 6. Caveats and unresolved
* Licences: BBC text has no Hub licence (bbc_news_alltime) or an inherited ODC-By label that does not cover BBC copyright (c4-bbc-news). Wikinews CC BY 2.5, Global Voices CC BY 3.0 (not declared on the card used), Wikivoyage CC BY-SA 4.0 share-alike. Used as reward-time input only; disclose in the paper's data table.
* Time range: bbc_news_alltime covers 2017 to mid-2025, c4-bbc-news is ~2019, Wikinews mostly 2004-2019, Global Voices 2005-2015. Newer events (2024-2025) are only in bbc_news_alltime. A cut-off rule for the model's training data (the memory note "May-2025 rule" belongs to the data-expansion project; I did not apply a date filter here) is not applied.
* The SSA-MTE train flag index (`aux/ssa_mte_train`) was not applied (flag-only in the earlier pools); not needed for correctness, but the news windows were not counted against it.
* Near-duplicates: exact duplicate windows (case-folded) were removed within a source and across sources (0 across); near-duplicate articles between `bbc_news_alltime` and `c4-bbc-news` (the same story in both) are only caught by the > 50 % repeated-sentence rule within one run, not across the two BBC sources in release 1. Release 2 did not re-check that either, so the same BBC story can appear twice in different windows.
* Windows in a language file are shuffled; the 6,000 b64 windows of one language are mostly from different articles (one window per article and bin), but a language file can contain a b64 and a b256 window from the same article.
* Global Voices rows come from an automatically sentence-aligned corpus: sentences the aligner dropped leave gaps inside a "window", and the document split uses the title rows (20,849 documents); a few windows may skip a sentence.
* Wikinews articles are 2004-2008 heavy and contain older style and some US/Canada local news; the 15,197 duplicate/empty Wikinews pages (Wikinews Shorts, stubs) were dropped.
* Wikivoyage is a travel guide (instructions / "you can take bus 313"), not news; kept at 5 % as the brief said "if easy". Dropping it is `SHARE[...] = 0` in the draw script.
* The A100 has 7.9 GB free (89 % full); the news directory is 201 MB. Release 2 staging adds 197 MB.
* A100 files are the released ones; nothing under `clean_v3_385m/*.parquet` was touched. The earlier test directory `news_r1_incoming` was renamed to `news` (no leftovers).

## 7. Where everything is
* Laptop: `gemma_mt/artifacts/rl_prompts_news/release1/` and `release2/` (b64+b256, git-ignored), scripts `gemma_mt/scripts/build_rl_prompts_news.py`, `build_rl_prompts_news_r2.py`, this report.
* Bucket `<S3_BUCKET>`, prefix `clean_rerun_2026-11-01/rl_prompts_news/`: `release1/` (slim), `release2/` (slim), `release2_full/` (b64..b1024), flat `*.parquet` = release 1 full from the first attempt (b64..b1024, window-level rule only).
* A100: `/workspace/clean_rerun/prompts/clean_v3_385m/news/` = release 1 slim (ACTIVE); `/workspace/clean_rerun/prompts/news_r2_staged/` = release 2 slim (staged).
* Working data (sources ~3 GB, work parquet files) in temporary working directories, not kept.
