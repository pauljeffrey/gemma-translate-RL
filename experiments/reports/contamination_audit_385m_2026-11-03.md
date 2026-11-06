# Benchmark contamination audit of the 385M SFT text (2026-11-03)

Question: do the benchmark rows used to evaluate the 385M Gemma translation model (and the A0 270M it was grown from) occur in its SFT training text? Scope: every training sample of the A0 packs (stage A draws from them) and of the six stage-B shards (including the tail-language replay packs appended to shards 2..6). CPU only, nothing downloaded to the laptop, no GPU, nothing pushed to the Hub, no training data modified.

## 1. Result in one table

Rows of each benchmark whose English side is found in the training text (either side of any sample; rule = sentence / contiguous run, `n_contaminated` = the scorer's `contaminated` flag, which counts strings of any length). `pair` = English and reference seen together. `ref (same lang)` = the reference sentence is in a sample of the same language (target-side exposure). `>= 3 tok` = rows whose exposed English string has 3 or more normalised tokens (the length below which the intake filter `contam_audit_12` never matches).

| benchmark (languages) | rows | n_contaminated | % | n_sentence_rule | n_near | n_pair | ref (same lang) | exposed rows with >= 3 tok | distinct English exposed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLORES+ devtest (54) | 54,648 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 of 1,012 |
| NTREX (24) | 47,928 | 120 | 0.25 | 120 | 0 | 13 | 44 | 0 | 5 of 1,993 |
| MAFAND (15) | 20,009 | 94 | 0.47 | 94 | 0 | 40 | 54 | 0 | 85 of 17,738 |
| SmolSent (45) | 38,833 | 45 | 0.12 | 45 | 0 | 0 | 1 | 0 | 1 of 863 |
| SALT (5) | 2,500 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 of 500 |
| GlotStory (8) | 3,426 | 15 | 0.44 | 15 | 0 | 0 | 2 | 0 | 9 of 937 |
| IbomMT (2) | 2,024 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| FLORES-MOZ (1) | 1,012 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| FLORES+ dev, held-out half (54) | 30,078 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 of 557 |
| AfriScience-MT test (6) | 5,820 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| AfriScience-MT dev (6) | 5,058 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| *control:* FLORES+ dev, train half (exempt by the D2 rule: its pairs are in the training data on purpose) | 23,760 | 23,706 | 99.8 | 20,574 | 0 | 7,655 | 7,681 | 23,706 | 439 of 440 |

Verdict: **no benchmark sentence of 3 or more words is in the new text, on either side, for any of the eight evaluation benchmarks, FLORES+ dev held-out or AfriScience.** FLORES+ devtest is fully clean (0 of 54,648 rows), against 95% of devtest English in the old SFT text of the first paper. The only hits are strings of one or two words (see section 4); they are 0.12 to 0.47% of the rows of four benchmarks. The 99.8% on the exempt FLORES+ dev train half is the positive control: the scanner finds exposure where it exists, and it was confirmed independently (section 3). The 8-gram near-duplicate rule found nothing at all (`n_near` 0 everywhere; no row has half of its 8-grams in the training English).

`n_pair` counts only the one or two word strings seen together with their reference; `n_pair` with 3 or more tokens is 0.

Sensitivity sets (same flags, other conventions) are in section 5. Per benchmark x language counts: `reports/data/contamination_385m_2026-11-03/results_by_language.csv` (columns `n, n_contaminated, n_sentence_rule, n_near, n_pair`, as in `results.csv` of `sft_benchmarks`, plus target-side, >= 3 token and strict columns, shares in percent); benchmark totals: `results_by_benchmark.csv`.

### Per language (non-zero cells only; full table in the CSV)

* NTREX: the same 5 distinct English strings (two words or one) are exposed in all 24 languages, so every language has 5 of about 1,997 rows (0.25%); `pair` in 7 languages (afr, hau, ibo, ssw, tsn, xho 2 each, wol 1; 13 rows), reference-side exposure in 16 languages (zul 7, wol 4, swh 4, sna 4, tsn 4, xho 4, afr 3, ibo 3, the others 1 to 2).
* MAFAND: ibo 42 of 1,500 (2.8%), amh 23 of 1,037 (2.2%), swh 13 of 1,835 (0.7%), yor 8 of 1,558 (0.5%), hau 2, pcm 2, nya 1, sna 1, kin 1, xho 1; `pair`: ibo 24, amh 11, swh 3, hau 2. These are rows whose whole English side is one or two words or a single numeral (MAFAND has such rows in ibo and amh).
* SmolSent: one English string (two words) in each of the 45 languages, 1 of 863 rows per language (0.12%); `pair` 0.
* GlotStory: dag 9 of 135 (6.7%), loz 2 of 960, her 1, kqn 1, kua 1, lun 1; these are one to four word items of the word-level part of GlotStory (single verbs, "a + noun" phrases).
* SALT: no English hit; 1 nyn reference of 2 words appears in a nyn sample. AfriScience dev: 2 amh references of one word.

## 2. What was scanned and how

Training text (decoded from the packed datasets in the object-storage bucket by ranged reads, never stored; 25,911,095 samples):

| dataset | bucket prefix | samples | translation-wrapper samples | tokens |
|---|---|---:|---:|---:|
| g0 (A0 old translation packs retagged + instruction packs; stage A drew from this set) | `clean_rerun_2026-11-01/g0/packed_g0` | 7,204,705 | 2,940,410 | 1.814 B |
| s1 to s6 (stage B, one epoch each, replay packs of shards 2..6 included) | `clean_rerun_2026-11-01/shards/s1..s6` | 3.11 to 3.12 M each, 18.7 M in all | 2.46 to 2.48 M each | 0.628 B each |

* The replay packs are cut from `packed_g0` and appended in place to the shards, so they are inside both counts; the stage-A schedule draws only from g0, so g0 covers stage A (as an upper bound: stage A saw a subset). All "new" text of the brief (stage-B new translation data, the 40 added languages, NLLB-mined bitext, Aya aligned part, Tumbuka, Efik NT, hub/extra2k) is inside the shards or the instruction part of g0, because the packs are the only form in which the trainer saw it.
* Both sides of every sample are scanned: the packed samples are not reliably English to target (flipped pairs in mined and JW/Bible bitext, instruction chat turns), so the benchmark English can sit on either side. Chat markers, the translation wrapper and the language tag are removed before matching.
* Rules (`gemma_mt.eval.contamination`, the scorer's own, plus the strict rule of `contam_audit_12`): normalisation NFKC, case-fold, every run of non-alphanumerics to one space (this also folds quotes); `exact` = normalised benchmark sentence equals a sentence of the text or occurs inside it as a contiguous token run; `sentence` = only as a whole sentence; `near` = at least 50% of its 8-grams occur; `pair` = exact and the reference sentence in the other side of the same sample, same language. Texts under 8 tokens have no n-grams. Target side: the same machinery on an index of all reference sentences, scanned against both sides of every sample (`ref_any`, `ref_same_iso`, `ref_near`). R4 of `contam_audit_12` (diacritics dropped after NFD: Yoruba, Fon, Ewe, Igbo written with and without under-dots or tone marks) was run as a second full pass; the final flags are the union of the plain and the mark-stripped scan (the variant adds 1 MAFAND exact row, 1 NTREX pair row and 3 reference-side rows, nothing else).
* Benchmarks (from `/workspace/eval_385m/bench/test_sets` on the A100, benchmark text never left the box; the loader is the evaluation's, so the row keys `<iso>:<id>` are the evaluation's): FLORES+ devtest, NTREX, MAFAND, SmolSent, IbomMT, FLORES-MOZ, GlotStory, SALT; FLORES+ dev split into the D2 held-out half (557 ids, protected) and train half (440 ids, exempt); AfriScience-MT test and dev (en to X, 6 languages).
* Check of the flag keys on the real predictions: all 170,380 rows of the 385M run and all 162,284 of the A0 run receive a flag (0 rows without), by `contam_audit_14_attach.py`.

## 3. Checks that the null result is real

1. Positive control inside the scan: FLORES+ dev train half, which the D2 rule puts in the training pairs on purpose, is found for 99.8% of its rows (439 of 440 English sentences; 7,655 pairs).
2. Independent re-implementation: `contam_audit_12_decontaminate.py` (R1 exact, R2 contiguous, R3 any shared 8-gram and near-50%, R4 variants, strings of fewer than 3 tokens ignored) run on two slices of 100,000 real training samples (s1 samples 0 to 100k, g0 samples 2.0M to 2.1M, the g0 region with the most target-side hits). With the dev train half in the protected set it flags exactly the same samples as the scanner (72 of 72 and 139 of 139, nothing extra on either side). Without it, both flag 0 samples of 3 or more tokens, and the scanner's own hits in those slices (459 and 102 samples) are all strings below 3 tokens. Logs: `logs/crosscheck_*.log`.
3. The plain scan was run twice by the previous attempt (second run bit-identical in the per-unit counts) and the mark-stripped scan is a third full pass; the three agree except for the 3 extra rows of R4.
4. Strict 8-gram rule: 54 FLORES+ devtest rows share one 8-gram with the text, all with one English sentence (one of its 16 8-grams, a common phrase about a world heritage designation); IbomMT and FLORES-MOZ contain the same FLORES English sentence (1 distinct sentence each); the other benchmarks have 1 (NTREX), 15 (MAFAND), 1 (SmolSent), 4 (GlotStory), 1 (AfriScience dev) distinct sentences sharing one 8-gram, always a common phrase (religious formula, news boilerplate, children's-story phrase, recipe ingredient list); only one GlotStory sentence shares more than 2 of its 61 8-grams (7). None reaches 50%. Under the intake filter with `--strict` those few training samples would be dropped; this is phrase-level overlap, not sentence-level.

## 4. Where the hits come from (attribution)

The packs carry no source field, so attribution is by dataset, sample kind, side and language, not by corpus.

* NTREX, SmolSent, GlotStory (1,600, 17, 219 string hits): 88 to 96% are in stage-B and g0 translation-wrapper samples (NTREX 1,531 of 1,600, of which stage B 1,252), 62% of them as the English side: short English words and phrases paired with a translation, which points to word-level and phrase-level bitext (word lists, glossaries, phrasebook-type pairs; an inference from the form of the strings, not read from metadata). Sample languages with the most NTREX English hits: wol 96, afr 81, luo 70, kab 59; on the reference side ibo has 443 of 654 hits.
* MAFAND (3.3 M string-hits, capped): 93% are in instruction ("other") samples, g0 1.64 M and stage B 1.45 M, mostly on the target side. Single-digit numerals (`1`, `2`, `3`: 1.05 M hits together) are the bulk, presumably numbered-list items and short numeric answers in the instruction chat data (sample languages with most hits: yor, swh, zul). The remaining 7% are translation prompts with one to two English words (some personal names). A numeral cannot contaminate a translation score; it is flagged because the scorer's `exact` rule has no length floor.
* AfriScience dev: 22 string hits, all one-word amh references in instruction data and mined pairs.
* No FLORES+ devtest, IbomMT, FLORES-MOZ or SALT English sentence is present; the one SALT hit is a 2-word nyn reference.

The earlier clean rebuild (intake filter against the protected index .3 with R1 to R4 on both sides, strings under 3 tokens exempt) is therefore effective for the A0 data, and the new data added since (NLLB-mined, Aya aligned part, 40 added languages, Tumbuka, Efik NT, hub/extra2k) brought in no benchmark sentence.

## 5. Files and how to use them

`reports/data/contamination_385m_2026-11-03/` (also on the A100 at `/workspace/eval_385m/contamination_385m_2026-11-03/`; the eight MAIN files are also copied to `/workspace/eval_385m/contamination/`, the `CONTAM_DIR` of the plan):

* `contamination_<benchmark>.json` x 8 (flores_plus ntrex mafand smol_sent ibom_mt flores_moz glot_story salt): the file `sft_benchmarks --contamination-dir` / `CONTAM_DIR` / `gemma_mt.eval.contamination.apply_flags` read: `{"meta", "exact", "sentence", "near", "pair"}` keyed `<iso>:<row id>`; `exact` becomes `contaminated`, `near` `near_dup`, `pair` `pair_seen`; these feed `n_contaminated, n_sentence_rule, n_near, n_pair` of `results.csv`. MAIN = English side, any length.
* `strict.tar.gz`: `exact` is also true when the reference is in a sample of the same language, `near` also for reference near-duplicates (sensitivity, target-side exposure).
* `min3tok.tar.gz`, `strict_min3tok.tar.gz`: the two sets above with exact / sentence / pair cleared for strings under 3 normalised tokens (the convention of the intake filter). With these, the `clean` subset equals `all` for all eight benchmarks.
* `extra/`: FLORES+ dev held-out and train half, AfriScience test and dev; `analysis/`: exposure tables, `exposed_rows.csv` (ids and lengths only, no text), `hits_by_dataset.csv`, summaries; `logs/`: scan and cross-check logs.
* Re-scoring without regenerating (done for the flags, not for the scores): `python gemma_mt/scripts/contam_audit_14_attach.py --predictions OUT/predictions.jsonl --contam-dir DIR --out OUT/predictions_flagged.jsonl`, then `eval_385m_score.py --system m385=predictions_flagged.jsonl ... --subsets all,clean,clean_strict,no_pair`. Flagged copies already exist on the A100: `/workspace/eval_385m/contam_audit_v2/predictions_flagged_m385.jsonl` (170,380 rows) and `predictions_flagged_a0.jsonl` (162,284 rows), 212 MB each.

Effect on the clean subsets of the evaluation (rows removed from `clean`): FLORES+ 0, NTREX 120 of 47,928, MAFAND 94 of 20,009, SmolSent 45 of 38,833, GlotStory 15 of 3,426, SALT / IbomMT / FLORES-MOZ 0. `no_pair` removes 13 NTREX and 40 MAFAND rows. The ordinary `all` tables and the `clean` tables will differ by at most 0.5% of the rows of a benchmark (language-level up to 2.8% for MAFAND ibo, 6.7% for GlotStory dag); the headline can be labelled `clean` (standard rule) after the scorer has been re-run on the flagged predictions. A0 270M: its text is a subset of g0, so the same files apply to both models.

Scripts (all in `gemma_mt/scripts/`): `contam_audit_14_385m_scan.py` (scan / finalize, resumable, ranged reads), `contam_audit_14b_strip.py` (R4 variant, imports the scanner unchanged), `contam_audit_14c_assemble.py` (union, MAIN / strict / min3tok sets, tables), `contam_audit_14_report.py`, `contam_audit_14d_crosscheck.py` (comparison with `contam_audit_12`), `contam_audit_14_attach.py` (flags onto predictions).

## 6. What the audit could not cover

* Source attribution: packs have no provenance field. Hits are attributed to dataset, sample kind, side and language only; the corpus behind the word-level pairs (word lists, glossaries, JW headings) is inferred from the form of the strings, not read from metadata.
* Sentences split across two training samples (a window cut inside a sentence) cannot match the sentence rule; the 8-gram sharing check partly covers this (section 3.4).
* Paraphrase and machine-translation contamination: a training target that is an MT or NLLB output of a benchmark English sentence is only caught if it equals the human reference; semantic overlap is not measured.
* Benchmarks not in the index: GATITOS and SmolDoc (word and document level, skipped by the scorer), SSA-MTE (excluded earlier), AfriDocMT, AfriLongDoc, AfriScience train, any benchmark version not on the A100 (`bench/test_sets`). FLORES+ languages that exist only as evaluation-only mappings (arz ary aeb dik taq zgh cjk mfe seh) are inside the 54-language FLORES+ set.
* Not part of this audit: the RL prompt pools and NLLB pseudo-reference reward data (the RL stages), the pretraining of google/gemma-3-270m-it, and the 270M A0 weights' own pretraining history. Stage A is bounded by the whole of g0 (7.2 M samples), not by the exact 48,000 packs it drew.
* Strings under 3 tokens: the scorer's rule flags them, the intake filter ignores them by design; both views are provided (MAIN and `min3tok`).
* Hit lists of very frequent strings (MAFAND numerals) are capped per unit, so the string-hit counts of section 4 for MAFAND are lower bounds; the row flags are not affected (a string is flagged on its first hit).

## 7. Removal recommendation

Nothing needs removing for the 385M or A0. No benchmark sentence of 3 or more words occurs in the new text; the one and two word hits are generic words, interjections, numerals and names in word-level pairs and in instruction answers, so removing them from the shards would change training in no way that matters for the benchmark and would alter shard checksums and schedules for no gain. If a zero-tolerance statement at every length is wanted for a later run, filter at intake: drop translation pairs whose English side is a single benchmark string of 1 to 2 tokens from the word-level sources in shards s1 to s6 (the number of such pairs was not counted here; count them first), and drop the numeral-only instruction answers from the MAFAND matches; the training shards were not modified. Report the headline as `clean` with the MAIN flags and state that "clean" removes at most 0.5% of rows; use `strict` as the sensitivity row.

## 8. Cost and compliance

Runs on the A100 (box 2, <BOX_HOST>:<BOX_PORT>, refused the connection): `nice -n 19`, 4 processes for my passes (the previous attempt's scan used 6), shards streamed by ranged GETs, nothing downloaded to the laptop, bucket keys read from the box's `/root/.bucket_env`, never printed; no process of anyone else touched; disk use of my passes 641 MB on the A100 (flag sets 47 MB, the two flagged prediction files 425 MB, the rest scan units and hit lists), the previous attempt's directory 883 MB. Runtime: plain scan 26M samples 1,699 s with 6 processes (previous attempt; its second identical run 1,916 s), mark-stripped pass 2,678 s with 4 processes (partly alongside that second run), flag assembly and tables about 2 minutes, each cross-check slice pair about 4 minutes. The earlier attempt did not "die without leaving files": its scanner, results and a still-running duplicate scan were on the A100 (`/workspace/eval_385m/contam_audit`, untouched); I reviewed the code, reproduced and extended it (R4 pass, cross-check, assembly) rather than repeating the plain scan.
