# Weak-language data: what exists, what is legitimately usable, what was built (D-62, agent R10, 2026-10-22)

Owner request: raise the sample weight of the weak languages (ish, bin, bbj, tsc, kon, nus, dag, oke) and add the training parts of benchmark datasets / any unused English<->language data. Later additions of the same day: Ge'ez script (amh, tir), commercial licence first, the owner's own `African-Code-Switch`, and the TranslatePsy question. Everything ran on the rental box (`/workspace/jobs/r10`, 1.4 GB left there after cleaning); nothing was pushed to the Hub or the bucket, nothing was downloaded to the laptop (only the small CSV/JSON files under `reports/data/weak_language_data_2026-10-22/`), the GPU was used only for QE scoring (AfriCOMET-QE + SSA-COMET, < 7 GB; the 270M model was not run), `/workspace/datagen`, `/workspace/jobs/r7` (the running final RL) and R4's files were only read.

## 0. Headline

1. **For the eight weak languages almost nothing new and legitimately usable exists.** ish, bin, bbj, oke and tsc: no English-parallel resource that is not already in the SFT corpus (all of it is JW / Bible text through the `michsethowusu/*_mt560` sets). kon: only an NLLB-mined set (2.2 M pairs, no licence on the repo, about half of the pairs misaligned in a 14-row read). dag: 717 Wikimedia pairs. nus: 6,181 OLDI-seed pairs (professional translations, CC-BY-SA) + a few hundred web chunks, on top of an already huge (9.4 M tokens) but synthetic-story corpus. **Where nothing exists the only lever is the draw weight (RL) and not repeating the same ~1.7 k JW windows more than once (SFT)**, and for a later round teacher data (R14) or new human translation.
2. **Benchmarks give no training part for them.** FLORES+ and FLORES-MOZ: dev = our calibration / evaluation, devtest = evaluation (FLORES-MOZ's card says "should not be used as training data"); SmolSent / SmolDoc / GATITOS: no split, they are our benchmarks; GlotStoryBook (dag, tsc): no split, our benchmark; NTREX, SALT test, IbomMT test, MAFAND test: evaluation. BOUQuet (`facebook/bouquet`, would have nus, bas, dua, tir, bam, kam, kea dev splits) is **gated: our token is not authorised**; JW300 Edo (`kinged0043/eng_edo-bin_jw300`) is gated too. ish, bin, bbj, oke have **no benchmark at all** (confirmed against `languages.py`).
3. **Built and audited (R4 packed format, `audit_packed` OK on every split), by licence class** (`/workspace/jobs/r10/sft_extra_v1/`):

| dir | licence_class | languages | windows | tokens | what |
|---|---|---|---|---|---|
| `commercial_ok` | commercial_ok | 13 | 35,177 | 37.7 M | tir 8.3 M, amh 11.2 M, kua 2.9 M, kam 2.6 M, her 2.6 M, kea 2.5 M, bam 2.3 M, nyn 2.1 M, sid 2.0 M, bas 0.65 M, shi 0.32 M, nus 0.26 M (FineTranslations real-target web text with Gemma-3-27B English, Tatoeba, GoURMET CC0 news, GlobalVoices) |
| `share_alike` | share_alike | 8 | 5,787 | 3.2 M | CC-BY-SA: OLDI seed (nus, bam), SALT train+dev (ach, nyn), Wikimedia (dag, kon, shi, amh) |
| `owner_code_switch` | owner | 44 | 1,969 | 2.7 M | `<HF_ORG>/African-Code-Switch` eng -> m_text, QE-filtered, capped 150 k tokens / language |
| `NC_noncommercial` | nc | 3 | 8,730 | 6.7 M | amh: AfriDocMT (sentence + document), AfricanFineTranslations, MAFAND-MT dev, PolyNews, + IbomMT train (ibb, efi). Separate file, only if NC is accepted |
| `EXCLUDED_unknown_licence` | unknown | 4 | 19,534 | 11.3 M | amh (habtew, OPUS-100, CCAligned, Bible), shi (Bible), wal (Bible), tir (CCAligned): licence unstated -> not to be merged |

   Plus `rl_prompts_sci_v2` (drop-in prompts dir, +1,210 prompts for each of 25 weak languages) and nested data-scaling probe mixes (`sft_probe_v1/mix_{0.25M,1M,4M}`). Markers: `READY_COMMERCIAL`, `READY` (same content), `manifest.json`.
4. **The owner's belief that the existing SFT corpus is commercial-clean is only partly true** (section 5): non-commercial repos were skipped at repo level, but 213 of 292 scanned repos have an unstated licence and were used (flagged), JW / Watchtower text was explicitly allowed by the owner (publisher copyright not cleared) and makes up about **58 % of the replay tokens on average and 75-93 % for most weak languages**, a few NC sets were used on owner instruction, and OpenSubtitles / GNOME strings are in the Amharic data.
5. **TranslatePsy (NC, NLLB-3.3B generated): analysed, excluded, not built** (section 3.3). It covers 19 strong / mid languages and none of the eight weak ones, so it could not have helped them in any case.
6. **Draw weights (RL)**: ish 4.0, bin / bbj / oke 3.5, kon / tsc / dag / nus / tir 3.0, ndc / ibb 2.0, the ten no-benchmark / script-artefact languages 1.5-2.0 (table in section 8), as `LANG_WEIGHTS_recommended.json` and a yaml line. Honest limit: R13's regression gives +1.38 chrF++ per doubling of samples with a 95 % CI of [-0.28, +3.11], i.e. not significant; the weights are a bet, not a measured effect.

## 1. Method

* Candidate discovery: the crawl catalogue (`data-catalog/african_hf/{rl,sft,pretrain,_excluded}`, 135 language CSVs, parallel-looking rows kept: `reports/data/weak_language_data_2026-10-22/weak_language_catalogue_sweep.csv`, **2,232 dataset x language rows over 82 languages**; licence class by rule: `commercial_ok` 479 rows / `share_alike` 89 / `nc` 147 / `unknown` 1,517 - the unknown mass is the `michsethowusu/*_sentence-pairs` NLLB-mined sets with no licence on the repo), plus direct Hub listing / search on the box and the OPUS API (JW300 is gone from OPUS; per-language corpus lists in `opus_probe.json`).
* "Already used" is **measured**, not assumed: the SFT windows are not stored per row any more, so R4's replay pool (87,498 original windows, all 77 languages) was decoded to text (`pool/<iso>.jsonl`) and every candidate pair was matched against it (target AND English 5-gram containment >= 80 %). A source is `USED` when hit rate / pool coverage >= 0.5, `PARTLY USED` >= 0.15 (only unseen rows kept), else `NEW`. Sanity: GNOME, OpenSubtitles, XLEnt (amh), Wikimedia (fon, tir) come out USED; OLDI seed, SALT, GoURMET come out NEW with expected-if-used counts of 70-1,400 hits and observed 0-137.
* Contamination: R4's hold-out (AfriScience dev/test, AfriLongDoc hard/core, `bench_dev_clean`, FLORES+ dev English, the RL-vs-SFT pool) **plus the English side of every evaluation test set in `bench/test_sets`** (FLORES+ devtest, NTREX, SmolSent, SmolDoc, GATITOS, SALT test, GlotStoryBook, IbomMT, FLORES-MOZ, MAFAND): exact sentence or word 8-gram => dropped (79,578 sentences, 1.85 M 8-grams). It removed e.g. 977 of 997 IbomMT dev rows (= FLORES+ dev English), 827 of 1,672 GlobalVoices rows, 191 of 341 PolyNews rows.
* Row filters: length / ratio, URL, letters, copy, dupes, GlotLID (English with p >= 0.5 dropped; other major language with p >= 0.8 dropped), and, at build time, a code-mix / untranslated-tail filter (>= 45 % of the target's words also in the English).
* Windows: consecutive pairs of one source / document joined by single spaces (as the original SFT, D-23), four length bins with equal token share, English side measured in Gemma tokens (the original used AfriCOMET tokens, same edges); **tokenizer fingerprint of `final_1433` checked equal to R4's `mix_v1`**, `seq_len` 8,192, `iso_table` equal.

## 2. The eight weak languages and the rest of the bottom group, one by one

Today = original SFT (`gemma_packed_fwd_sft_windows`), R4 `mix_v1` (science + replay), RL pool (`rl_prompts_sci_v1`: 598 general + 791 science prompts per language), draws in the 800-step plan (`language_coverage_800.json`). "NEW" = tokens of the files built here (prompt + answer), by class. `JW %` = share of the language's replay-sample tokens that look like JW / Watchtower text (spaced-punctuation tokenisation or Jehovah vocabulary; a heuristic, section 5).

| iso | language | RL chrF++ (benchmark) | SFT target tok (original) | JW % | R4 mix tok | NEW commercial_ok | NEW share_alike | owner (ACS) | NC | excluded unknown | RL pool now -> v2 | draws now -> recommended |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ish | Esan | none | 0.16M | 78 | 0.15M | 0 | 0 | 0 | 0 | 0 | 1389 -> 2599 | 1x (311) -> 4x (1014) |
| bin | Edo | none | 1.71M | 22 | 0.15M | 0 | 0 | 1k | 0 | 0 | 1388 -> 2598 | 1x (311) -> 3.5x (888) |
| bbj | Ghomala | none | 0.38M | 2 | 0.15M | 0 | 0 | 0 | 0 | 0 | 1389 -> 2599 | 1x (311) -> 3.5x (888) |
| oke | Okpe | none | 0.74M | 78 | 0.15M | 0 | 0 | 0 | 0 | 0 | 1390 -> 2600 | 1x (311) -> 3.5x (888) |
| tsc | Tswa | 17.9 (GlotStoryBook) | 2.49M | 83 | 0.15M | 0 | 0 | 0 | 0 | 0 | 1388 -> 2598 | 1x (311) -> 3x (761) |
| kon | Kongo | 16.7 (SmolSent) | 0.14M | 93 | 0.15M | 0 | 13k | 916 | 0 | 0 | 1388 -> 2598 | 1x (311) -> 3x (761) |
| nus | Nuer | 20.8 (FLORES+) | 9.37M | 0 | 0.15M | 0.26M | 0.70M | 0 | 0 | 0 | 1391 -> 2601 | 2.5x (778) -> 3x (761) |
| dag | Dagbani | 27.9 (GlotStoryBook) | 1.93M | 6 | 0.15M | 0 | 54k | 0 | 0 | 0 | 1390 -> 2600 | 1x (311) -> 3x (761) |
| shi | Tachelhit | 1.0 (NTREX, script artefact) | 0.93M | 82 | 0.15M | 0.32M | 31k | 0 | 0 | 1.42M | 1388 -> 2598 | 1x (311) -> 1.5x (380) |
| tir | Tigrinya | 13.8 (FLORES+) | 5.60M | 83 | 0.15M | 8.31M | 0 | 4k | 0 | 0.54M | 1391 -> 2601 | 2.5x (778) -> 3x (761) |
| dyu | Dyula | 18.3 (FLORES+) | 0.48M | 80 | 0.15M | 0 | 0 | 0 | 0 | 0 | 1389 -> 2599 | 2.5x (778) -> 2.5x (634) |
| amh | Amharic | 18.9 (FLORES+) | 8.29M | 58 | 1.34M | 11.18M | 0.39M | 0.14M | 6.50M | 8.14M | 1388 -> 2598 | 2.5x (778) -> 2.5x (634) |
| fon | Fon | 21.6 (FLORES+) | 34.03M | 10 | 0.15M | 0 | 0 | 1k | 0 | 0 | 1390 -> 2600 | 2.5x (778) -> 2.5x (634) |
| kbp | Kabiye | 21.7 (FLORES+) | 0.43M | 71 | 0.15M | 0 | 0 | 0 | 0 | 0 | 1388 -> 2598 | 2.5x (778) -> 2.5x (634) |
| ndc | Ndau | 22.0 (SmolSent) | 0.83M | 86 | 0.15M | 0 | 0 | 0 | 0 | 0 | 1389 -> 2599 | 1x (311) -> 2x (507) |
| urh | Urhobo | none | 1.22M | 77 | 0.15M | 0 | 0 | 2k | 0 | 0 | 1389 -> 2599 | 1x (311) -> 1.5x (380) |
| yao | Yao | none | 1.34M | 82 | 0.15M | 0 | 0 | 0 | 0 | 0 | 1388 -> 2598 | 1x (311) -> 1.5x (380) |
| sid | Sidamo | none | 1.36M | 86 | 0.15M | 2.03M | 0 | 0 | 0 | 0 | 1388 -> 2598 | 1x (311) -> 1.5x (380) |
| dua | Duala | none | 0.53M | 77 | 0.15M | 0 | 0 | 0 | 0 | 0 | 1390 -> 2600 | 1x (311) -> 1.5x (380) |
| bas | Basaa | none | 0.82M | 82 | 0.15M | 0.65M | 0 | 0 | 0 | 0 | 1389 -> 2599 | 1x (311) -> 1.5x (380) |
| wes | Cameroon Pidgin | none | 0.43M | 76 | 0.15M | 0 | 0 | 6k | 0 | 0 | 1390 -> 2600 | 1x (311) -> 1.5x (380) |
| iso | Isoko | none | 7.35M | 82 | 0.15M | 0 | 0 | 0 | 0 | 0 | 1391 -> 2601 | 1x (311) -> 1.5x (380) |
| wal | Wolaytta | none | 3.77M | 86 | 0.15M | 0 | 0 | 0 | 0 | 1.24M | 1387 -> 2597 | 1x (311) -> 1.5x (380) |
| swc | Congo Swahili | none | 6.76M | 88 | 0.15M | 54 | 0 | 0 | 0 | 0 | 1387 -> 2597 | 1x (311) -> 1.5x (380) |

Notes per language (what was looked at and where):

* **ish (Esan).** Only `michsethowusu/english-esan_sentence-pairs_mt560` (5,221 rows, JW, already in the corpus: 1,697 windows, 0.16 M target tokens, 56 % of it replayed by R4). Translatewiki 128 strings (software UI, mixed licences, not used). Omnilingual-ASR corpus: 481 monolingual transcripts (target only, CC-BY-4.0), fineweb-2: 46 documents: monolingual, usable only after an X -> English teacher pass. No benchmark, no Bible in `biblenlp-corpus`.
* **bin (Edo).** mt560 Edo (11,635, JW, in the corpus) + Bible-style text (24 % of the replay tokens). The only extra set, `kinged0043/eng_edo-bin_jw300` (JW300), is **gated** (access not granted to our token) and would be the same JW domain. Translatewiki 21 strings. African-Code-Switch: 5 human-gold rows.
* **bbj (Ghomala).** SFT = Bible (17 %) + other; MAFAND-MT has **French <-> bbj** (not English; CC-BY-NC) and MasakhaNER2 has bbj monolingual NER text (afl-3.0): neither is an English-parallel training set. French -> English by `opus-mt-fr-en` would give a pivot English side for ~2.2 k train / 1.1 k dev pairs (NC); not built (no English-parallel claim could be made, and the owner asked for commercial first). **Monitoring for bbj** is possible with MAFAND-MT fr-bbj test via the French pivot (source = machine-translated English) as a proxy; today only R4's 60 validation windows exist.
* **oke (Okpe).** mt560 Okpe (22,471, JW, in the corpus). Nothing else on the Hub or OPUS.
* **tsc (Tswa).** mt560 Tswa (84,311, JW, in the corpus: 22,954 windows); GlotStoryBook tsc (101 rows) is the benchmark; FLORES-MOZ has Xichangana (`tso`), not Tswa, and is evaluation only.
* **kon (Kongo).** 4 commercial-looking sets are all in the corpus (mt560 206 k). The 2.2 M-pair `michsethowusu/english-kongo_sentence-pairs` (OPUS NLLB v1 mined, sorted by similarity, licence unstated) was streamed (top 200 k) and cleaned to 163 k pairs; **a 14-row read of the kept pairs showed about 6 of 14 misaligned** (e.g. "Chapter 6: The promise:" -> "Mbongiseni Solar"), so it was **not packed** (raw and cleaned files deleted); a QE gate (AfriCOMET covers kon) would be needed before any use. Wikimedia: 95 pairs (share_alike). GATITOS / SmolSent / SmolDoc kg are our benchmarks.
* **nus (Nuer).** 90,891 windows / 9.4 M target tokens already (synthetic story corpus, NaathNLP-type), RL chrF++ 20.8 and 50 % unfinished in SFT: volume is not the problem. New and clean: OLDI seed `nus_Latn` (6,181 pairs, professional translations of Wikipedia sentences, CC-BY-SA-4.0, 0.70 M tokens) and 431 FineTranslations chunks. TICO-19 nus (3,071) is CC0 but is a dev / test evaluation set (not used). BOUQuet nus dev is gated. NaathNLP / dayomtechnologies sets are NC.
* **dag (Dagbani).** In the corpus: Bible, navigation corpus, GlotStoryBook is the benchmark. New: Wikimedia 717 pairs (CC-BY-SA, noisy: 206 of 922 dropped as code-mixed / English). Ghana NLP community sets (Bible, navigation, youth conversations) are NC / already used / monolingual.
* **Languages with new commercial-OK volume** (tir, amh, sid, bas, her, kea, kam, kua, nyn, bam, shi, ach): section 3.

## 3. Licence classes and the commercial gap

### 3.1 Sets that passed (all measured NEW or PARTLY USED against the SFT pool; full per-set numbers in `reports/data/weak_language_data_2026-10-22/audit.csv`)

| source (id, split) | languages | rows after filters | licence (where looked) | class | note |
|---|---|---|---|---|---|
| `HuggingFaceFW/finetranslations` train | tir 19.3k, amh 21.1k, kua 14.0k, kea 7.5k, bam 4.7k, kam 4.9k, her 4.4k, sid 3.7k, nyn 3.5k, bas 1.1k, nus 0.4k | 85k | ODC-By 1.0 (card) | commercial_ok | **real FineWeb-2 text in the language, English side = Gemma-3-27B translation (synthetic SOURCE)**; underlying web text copyright unvetted; JW pages are a large part of tir / amh |
| `openlanguagedata/oldi_seed` | nus 6.2k, bam 6.2k | 12k | CC-BY-SA-4.0 (card) | share_alike | human translation of Wikipedia sentences |
| `Sunbird/salt` text-all train + dev | ach 24.4k, nyn 24.4k | 49k | CC-BY-SA-4.0 (card) | share_alike | human translations, agriculture / health / community; the dev split (496) is the extras' validation split |
| OPUS GoURMET (en-am) | amh 55.8k | 56k | **CC0** (OPUS README inside the zip) | commercial_ok | human-translated news |
| OPUS GlobalVoices | amh 0.8k | 0.8k | CC-BY-3.0 (Global Voices site terms; OPUS README silent) | commercial_ok | 827 of 1,672 rows dropped by the hold-out filter |
| Tatoeba (`abdelhaqueidali/...Tatoeba...`, OPUS Tatoeba) | shi 15.7k + 1.7k, amh 0.2k, tir 0.1k | 17.7k | CC-BY-2.0-FR (card / OPUS README) | commercial_ok | short sentences; the shi target column used is `shi_latn` (our SFT writes Latin) |
| OPUS Wikimedia | dag 0.7k, shi 0.4k, kon 0.1k, amh 1.6k | 2.9k | CC-BY-SA-4.0 (OPUS README) | share_alike | content-translation |
| `<HF_ORG>/African-Code-Switch` | 44 | see 6 | `other` / private | owner | |
| OPUS Bible-uedin | amh 59k, shi 15.8k, wal 15.2k | 90k | CC0 for the compilation (OPUS README); the Bible translations underneath have their own copyright | **unknown** | excluded from the default files |
| OPUS CCAligned (amh 298k, tir 6.3k), `habtew/english-amharic-translation` (133k), OPUS-100 en-am (62k), tir NLLB-mined (133k, 3 of 3 read pairs misaligned) | amh, tir | 630k | none / unknown | unknown | CCAligned is web-mined with no licence; OPUS-100 contains Quran / Tanzil / Bible / subtitles |
| AfriDocMT sentence (health + tech, 15.2k), documents (cut to 5/10/25 sentences, 3.0k windows), AfricanFineTranslations (4.5k), MAFAND-MT dev (0.85k), PolyNews (0.15k), Tanzil (90.8k), TED2020 (1.0k), IbomMT train (ibb 998, efi 996) | amh, ibb, efi | 120k | CC-BY-NC-SA-3.0 (catalogue; the card states none) / CC-BY-NC-4.0 / Tanzil terms (non-commercial) / TED policy | **nc** | separate dir |

Not usable and why: FLORES+ / FLORES-MOZ (evaluation, card forbids training), NTREX, SmolSent / SmolDoc / GATITOS / GlotStoryBook (our benchmarks), SALT test, TICO-19 (CC0 but a dev / test benchmark), BOUQuet and JW300-Edo (gated), `Walelign/EthioMT` (gender-bias test set), XL-Sum / MasakhaNEWS (monolingual summarisation / classification, NC), WMT (no amh / tir training track exists; the WMT22 African set is evaluation), the `michsethowusu/*_sentence-pairs` NLLB-mined families (licence unspecified, noisy), Weblate / Pontoon / translatewiki / GNOME / Ubuntu (UI strings, mixed licences; GNOME and OpenSubtitles are already in the Amharic data).

### 3.2 Gap: commercial-OK tokens that exist vs what is missing (tokens = prompt + answer, built files)

| class | weak8 | bottom15 (shi tir dyu amh fon kbp ndc) | no benchmark (9) | other (kua kam kea her bam nyn ach ...) |
|---|---|---|---|---|
| commercial_ok | 0.26M (nus only; 1 language) | 19.8M (3 languages) | 2.7M (3) | 15.0M (6) |
| share_alike | 0.76M (nus, dag, kon) | 0.42M | 0 | 2.1M |
| owner (ACS) | 2k | 0.15M | 8k | 2.6M |
| NC | 0 | 6.5M (amh) | 0 | 0.17M |
| unknown (excluded) | 0 | 10.1M | 1.2M | 0 |

* **Languages where commercial-OK new data covers < 5 % of the original SFT tokens: ish, bin, bbj, oke, tsc, kon, dag, dyu, kbp, ndc, urh, yao, dua, wes, iso, wal, swc, fon.** For these, a teacher (R14), new human translation, or NC / unknown data is the only way to add anything; sample weight is the only free lever.
* Languages where commercial data is large relative to what exists: tir (+8.3 M vs 5.6 M target tokens in the whole original SFT), kea (+2.5 M vs 0.11 M), her (+2.6 M vs 0.23 M), bam (+2.3 M), sid, kua, kam, nyn. These are **web text in the language with synthetic English sources** (FineTranslations): natural targets, imperfect sources, JW-heavy in tir / amh.

### 3.3 TranslatePsy-AfriSLM-Synthetic-Mix (`qvac/`, CC-BY-NC-4.0): analysed, NOT built (owner: SFT stays commercial-clean)

* Card: 215,653,192 bidirectional training rows (the catalogue's 5.67 M is a stale count), 182 GB in 364 arrow shards (streamed with `pyarrow.ipc` over the Hub, never stored), columns `source, target, messages, source_lang, target_lang, source_pair`. **Languages: English + af am ha ig rw ln mg ny om sn so st sw tn wo xh yo zu lg (19): all strong or mid; none of ish bin bbj tsc kon nus dag oke or tir.**
* Origin: MADLAD-400 monolingual sentences (CC-BY-4.0) translated by **`facebook/nllb-200-3.3B` (CC-BY-NC-4.0 weights), beam 3**, scored in both directions with AfriCOMET, SSA-COMET and MetricX, robust-z filtered, then expanded both ways. Precedent: our own SFT build already skipped this repo (182 GB, NC) and the Aya-translated sets (NLLB-3.3B output) for the same reason (`SFT_BUILD.md`).
* Directions: each shard holds ONE direction (`eng->xx` 4-23 shards per language, `xx->eng` 4-22); the corpus is **mirrored** (3 shards of `eng-sot` vs 3 of `sot-eng`: 7,531 identical pairs in 120 k rows), so the `xx->eng` rows add no new sentence pairs for en -> xx; for a future X -> English component they are the same data flipped. The row does not say which side was the original web text.
* Quality (40 k en->xx rows streamed per language, 19 languages; QE on 4,000 per language, statistics on the first 300): AfriCOMET-QE mean 0.71-0.87 vs 0.53-0.73 for the HUMAN FLORES+ references of the same languages (+0.09 to +0.25), SSA-COMET 0.65-0.77 vs 0.49-0.64: the corpus is above human references on both judges (it was filtered by them, and NLLB-like output is what they like; QE headroom is not chrF headroom, D-65). GlotLID ok 73-99 % (wol 73 %, mlg 79 %, swh 87 %), English-word overlap >= 45 %: 0-12 % (wol 11.6 %), 3-gram repetition >= 30 %: ~0, mean English length 73-120 characters (sentences). Overlap: 1,168 8-gram + 24 sentence hits against our hold-outs in 760 k rows (0.16 %), 104 rows (0.014 %) inside R4's SFT pool sample. `reports/data/weak_language_data_2026-10-22/translatepsy_analysis.csv`.
* **What NC means** (not legal advice): the dataset is CC-BY-NC-4.0 and its targets are NLLB-3.3B (CC-BY-NC-4.0) outputs, so a model trained on it should be treated as NC: private / research-only, or released under CC-BY-NC with the same restriction; the SFT dataset built from it could not be published under a permissive licence, and the existing commercial-clean SFT / RL line would be contaminated by it. Because the owner decided to keep SFT commercial-clean, nothing was packed.
* Value even if allowed: none for the weak languages (not covered), and for the strong ones low (section 9). `tp_build.py subset` (kept at `/workspace/jobs/r10/`) builds the QE-filtered subset in about 20 minutes should the decision change.

## 4. Ge'ez script (amh, tir)

Only **amh and tir** use Ge'ez among our 85 languages (checked in `languages.py` and by the Ethiopic-letter share of the SFT pool: ~100 % for amh / tir, < 1 % for every other language; sidamo, wolaytta and Oromo are written in Latin).

| | amh | tir |
|---|---|---|
| original SFT samples / target tokens | 59,035 / 8.29 M | 48,410 / 5.60 M |
| R4 mix_v1 tokens | 1.05 M science + 0.30 M replay | 0.15 M replay |
| RL prompts (v3 pool / sci_v1 / draws in 800 steps) | 20,000 / 1,388 / 660 | 20,000 / 1,391 / 639 |
| best-translation archive rows (2c run, 6,142 in total) | 5 | 24 |
| RL chrF++ (FLORES+ devtest) | 18.9 | 13.8 (worst QE-validated language) |
| new commercial_ok / share_alike (this work) | 11.18 M / 0.39 M tokens | 8.31 M / 0 |
| new NC / unknown | 6.50 M / 8.14 M | 0 / 0.54 M |

**What exists** (rows after filters; licence; verdict): OPUS **GoURMET** en-am 55.8 k (CC0; built, commercial_ok); FineTranslations amh 21 k / tir 19 k (ODC-By; built); Tatoeba 189 / 102; Wikimedia amh 1.6 k (share_alike; tir's is already in the SFT); GlobalVoices 0.8 k; Bible-uedin amh 59 k (CC0 compilation, underlying translation unknown; excluded); CCAligned amh 298 k / tir 6.3 k and `habtew` 133 k, OPUS-100 62 k (unknown; excluded); tir NLLB-mined 133 k (misaligned in 3 of 3 pairs; excluded); **AfriDocMT** (sentence-level health + tech 15.2 k, document-level 3.0 k windows of 5-25 sentences, CC-BY-NC-SA, NC dir; the only document-level amh set found, useful for the whole-paper behaviour), AfricanFineTranslations amh 4.5 k, MAFAND-MT en-amh dev 0.85 k (no train file was found at the published `lafand-mt` path (404), nothing at all for tir), PolyNews 150, Tanzil 90.8 k, TED2020 1.0 k (all NC); already in the SFT corpus (measured USED): GNOME, OpenSubtitles, XLEnt, Wikimedia tir. **Do not exist / cannot be used:** WMT has no amh / tir training track (WMT22 African is evaluation); NLLB-seed / OLDI seed has neither amh nor tir (only nus, bam, knc, fuv among our languages); HornMT and EthioMT were not found on the Hub (EthioMT is a gender-bias test set); XL-Sum amh / tir and MasakhaNEWS amh / tir are monolingual (NC); `facebook/bouquet` has tir (and amh) dev / test but is gated; TICO-19 tir (6.1 k, CC0) is a dev / test evaluation set; FLORES+ amh / tir dev and devtest are excluded by our own rules; JW300 is no longer on OPUS (the JW text we have comes through mt560 and is 58 % / 83 % of the amh / tir replay tokens by the heuristic).

**Tokenisation** (Gemma tokenizer of `final_1433`, FLORES+ devtest, 1,012 aligned sentences, no language tag; `tokenizer_flores_devtest.csv`):

| | tokens / word | tokens / character | characters / token | bytes / token | tokens vs English | p95 | words vs English |
|---|---|---|---|---|---|---|---|
| English | 1.24 | 0.205 | 4.87 | 4.87 | 1.00 | 1.00 | 1.00 |
| **amh** | 3.03 | **0.603** | 1.66 | 4.29 | 1.94 | 2.62 | 0.79 |
| **tir** | 3.46 | **0.740** | 1.35 | 3.45 | **2.48** | 3.33 | 0.89 |
| hau | 1.86 | 0.335 | 2.99 | 3.02 | 1.73 | 2.30 | 1.15 |
| swh | 2.09 | 0.322 | 3.11 | 3.11 | 1.64 | 2.11 | 0.97 |
| yor | 2.59 | 0.514 | 1.94 | 2.59 | 2.41 | 3.68 | 1.15 |
| zul | 3.37 | 0.360 | 2.78 | 2.78 | 1.97 | 2.62 | 0.72 |
| fon | 2.34 | 0.531 | 1.88 | 2.33 | 2.65 | 4.04 | 1.40 |
| nus | 2.80 | 0.558 | 1.79 | 2.20 | 2.93 | 4.20 | 1.30 |

Only 862 of the 262,145 vocabulary entries contain Ethiopic characters: most syllables are single tokens (ሰላም = 3 tokens for 3 characters), so a Ge'ez character costs about 3x a Latin one in tokens (0.60-0.74 vs 0.21-0.36 tokens per character) and **tir needs 2.5x the tokens of the English source (p95 3.3x), amh 1.9x (p95 2.6x)**; per word amh and tir are in the Latin agglutinative range (Zulu 3.4) because Ge'ez words are short. Implications: (a) context use: a b1024 tir window is ~1.4 k English tokens + ~3.5 k answer tokens, a b2048 tir window ~2.3 k + ~5.7 k = 8.0 k, i.e. at the edge of R4's 8,192 pack (a few b2048 tir samples will be dropped by `seq_len`); (b) RL caps `max_new_tokens_by_bin` (b64 384, b256 1,280, b512 2,304, b1024 3,584, b2048 7,168) are 3.5x or more of the source bin edge: amh and tir fit on average (tir mean 2.48x), but the per-sentence p95 of tir (3.3x), **nus (4.2x) and fon (4.0x) exceeds the 3.5x cap of b1024 / b2048**, so a few percent of their long completions can hit the cap and be scored as `no_end_of_turn`; (c) per-token loss in the SFT is dominated by Ge'ez syllables, so tokens-per-language budgets (section 9) should be read in "words": 1 M amh tokens ~ 330 k words, 1 M hau tokens ~ 540 k words. The tokenizer was not touched.

## 5. Is the existing SFT corpus commercial-clean? Partly - flagged, not confirmed

Sources: `docs/report/06_data_campaign.md` (rule: no non-commercial licence; Watchtower / JW text explicitly allowed; `sft_new/license_scan.py` classes `commercial_ok` / `non_commercial` (skipped) / `unknown` (used, flagged); owner overrides), `old_box_snapshot/sft_build/report/SFT_BUILD.md` and `license_scan.json`, R4's replay pool.

* **Repo level (the SabiYarn SFT/pretrain build that the Gemma windows come from):** 63 repos (18.95 TB) `commercial_ok`, **213 repos (1.27 TB) `unknown`, used and flagged**, 16 repos (0.21 TB) skipped as non-commercial (incl. TranslatePsy, `ghanaopenai/ghok-*`, `masakhane/african-translated-alpaca`). Owner overrides recorded: `ghananlpcommunity/pristine-twi-english` (declared CC-BY-NC-4.0, used), `AfriSpeech/african-corpus-jw`, `AfriSpeech/africa-corpus`, `African-Languages-Lab/multi-open`; the `naija` / `naija2` sets carry XL-Sum rows (CC-BY-NC-SA) that were kept and flagged. Sets of unverifiable MT origin were used (`<HF_ORG>/tds` 10.4 M rows, `naija_sft` 8.5 M, ...) and the Aya NLLB-3.3B-translated collections were held out.
* **Window level (what R4's `mix_v1` replays):** the per-row `licence` column of the Gemma pairs (`permissive_pct`, `unspecified_pct`, ...) existed in `gemma_stats.py` but its output is not on the box any more, and the packed windows carry no source column, so **the per-row licence mix cannot be tabulated: unknown**. Measured instead from the decoded pool (`pool_domain_share.csv`): by tokens, **57.9 % of the replay is JW-style text on average** (ish 78, kon 93, tsc 83, oke 78, tir 83, amh 58, kea 87, her 84, ...; low for hau 3, kin 4, nus 0, bam 0, fon 10), 3.7 % Bible-style, 38.4 % other. JW text is allowed by the owner but the publisher's copyright is not cleared, so I class it **unknown / owner-allowed**, not commercial_ok. The Amharic pool also shows OpenSubtitles (copyrighted subtitles) and GNOME strings (GPL-family) as USED.
* So: the corpus was cleaned of **declared** NC licences at repo level; it was not cleaned of unknown licences (77 % of scanned repos), JW text, and a few owner-approved NC sets. If a commercial release is planned, the replay (65 % of `mix_v1`) is the part to review; the new files here are class-labelled to make a clean alternative possible: for the weak languages the clean alternative does not exist (section 3.2).

## 6. `<HF_ORG>/African-Code-Switch` (private)

* 631 files; the Hub `final/<iso>.jsonl` holds 1,780 rows in total (a few dozen per language), `generated/out_main/<iso>.raw.jsonl` holds the raw generated rows: ~1,716-1,750 for 21 high-resource languages (afr amh bam ful hau ibo kin lug mlg nso nya orm sna som sot swh tsn wol xho yor zul), 4-6 k for pcm / gpe / wes, and **only 18-56 rows for the weak and very-low languages (bin 26, efi 32, ibb 38, kon 48, tir 56, urh 36, kmb, umb, mos, sag ...)**.
* Columns: `eng` = "formal, neutral English"; **`m_text` = "the official translation of `eng`" written by `google/gemma-4-31b-it`** (teacher field in the raw rows; 266 human-gold / native-corrected rows overall); `hm` = code-mixed variant, `ood` = informal paraphrase of the situation (**not** a translation of `eng`, not used). Seeds include NC / research sources as inspiration only (card), so the owner licence applies.
* Scored (AfriCOMET-QE + SSA-COMET, GlotLID, repetition, English-word overlap, length ratio, hold-out; `african_code_switch_scores.csv`): kept = gold rows + generated rows passing the cheap filters and QE >= p25 of the language's human FLORES+ references (languages without FLORES: 0.55 / 0.50). Kept per language, e.g. amh 1,389 of 1,744, mlg 1,462 of 1,745, yor 1,481, som 1,612, zul 1,047, hau 980, afr 711, ibo 749, **bam 8 of 1,729 (QE mean 0.35: the Bambara translations are poor)**, wes 188 (GlotLID agrees on 22 %), pcm 20 of 6,380, ful 6 (no usable LID label), and for the weak languages only the gold rows (bin 5, kon 5, ibb 18, efi 14, urh 12, tir 27, ...). Hold-out hits: 0-4 rows per language.
* Packed: `sft_extra_v1/owner_code_switch` (1,969 windows, 2.7 M tokens, 44 languages, `licence_class = owner`, `quality_flag` in `rows.parquet`: 257 `human_gold`, 13,347 `generated_pass`), capped at 150 k tokens per language, gold first. Treat like archive-quality data: cap its share per language at <= 15 % of the language's tokens; for the weak languages it adds essentially nothing (bin / kon ~1 k tokens).

## 7. Files, format, and the command for R11

All under `/workspace/jobs/r10/`:

| path | content |
|---|---|
| `sft_extra_v1/<class>/{train,val}/`, `meta.json` | R4 packed format (`gemma_mt.data.prepare_sft`): uint32 tokens + `sample_offset/len`, `prompt_len`, `target_len`, `iso_id`, `pack_start`; `seq_len` 8,192; template `Translate the text to {Language}:  {English}`, answer `{iso} {translation}<end_of_turn>`; tokenizer fingerprint `69f7f26e2deede417601468df553168c39714aeb3e171eea3482408fe6e3c218` = `final_1433` = `mix_v1`; `iso_table` = `ALL_ISOS` (85) |
| `.../rows.parquet` | one row per sentence pair: iso, english, target, source_dataset, source_file, orig_split, licence, **licence_class**, synthetic_en, doc, window_id, in_split, lid, lid_p (+ `quality_flag` in `owner_code_switch`) |
| `.../windows.parquet` | one row per window: window_id, iso, english, target, n_sent, en_tokens, total_tokens, target_tokens, source_dataset, source_file, licence, licence_class, synthetic_en, split, note |
| `.../manifest.json` and `/workspace/jobs/r10/manifest.json` | per language: tokens per class, sources, licences, original SFT tokens, "NO new commercial-OK data" note; `stats/manifest_flat.csv` |
| `merge_mix.py` | merges any of the extra dirs into an R4-format mixture with per-language token budgets and an optional per-language multiplier on the base samples |
| `rl_prompts_sci_v2/`, `rl_prompts_extra_v1/` | RL prompts, section 8 |
| `sft_probe_v1/mix_{0.25M,1M,4M}` | probe mixtures, section 9 |
| `LANG_WEIGHTS_recommended.json` | draw weights in the D-65 `LANG_WEIGHTS.json` format |

**R11 command** (default = commercial_ok + share_alike, budgets = total prompt + answer tokens per language; R4's `mix_v1` has 148 k replay tokens per language, 19.2 M in all; the budgets below add about 11.4 M tokens (+59 %; halve them if the mixture should stay near 19 M tokens - the order of priority is tir, nus, ach, nyn, amh, bam, sid, then the FineTranslations languages)):

```bash
ssh -o BatchMode=yes -i $SSH_KEY -p $BOX_PORT root@$BOX_HOST
cd /workspace/jobs/r10; source /workspace/jobs/r4/env.sh
export PYTHONPATH=/workspace/jobs/r4/gemma_mt_sft:$PYTHONPATH
$PY merge_mix.py --base /workspace/jobs/r4/mix_v1 --out /workspace/jobs/r11/mix_v2 --seed 1 --max-repeat 1 \
  --extra sft_extra_v1/commercial_ok --budget '{"tir":2000000,"amh":1500000,"sid":600000,"bas":400000,"her":500000,"kea":500000,"kam":500000,"kua":500000,"nyn":300000,"bam":500000,"shi":300000,"nus":250000}' \
  --extra sft_extra_v1/share_alike  --budget2 '{"nus":690000,"ach":700000,"nyn":800000,"bam":400000,"dag":54000,"kon":12500,"shi":30000,"amh":390000}' \
  --extra sft_extra_v1/owner_code_switch --budget3 '{"amh":140000,"mlg":120000,"orm":120000,"som":120000}'
$PY -m gemma_mt.data.audit_packed --data /workspace/jobs/r11/mix_v2 --split train --tokenizer /workspace/sft_build/runs/final_1433 --check-samples 3000
```
Output: `train/` (base + extra, repacked), `val/` and `val_sci/` copied from the base, **`val_extra/`** (the extras' own validation windows: add `val_extra` to `extra_val_splits` in the SFT yaml; its SALT-dev part is a human-translated validation set for ach / nyn), `merge_report.json`. `--scale-base '{"ish":1.0,...}'` scales base samples of a language; do not push weak languages above one pass (section 9). Never add `NC_noncommercial` or `EXCLUDED_unknown_licence` to a commercial mixture; the NC dir is only for a model the owner accepts as NC (build recipe identical: `--extra sft_extra_v1/NC_noncommercial --budgetN '{"amh":2000000}'`).

## 8. RL prompts and the draw weights

**Format**: `rl_prompts_sci_v2/<iso>.parquet` has the exact 12 columns / dtypes of `rl_prompts_sci_v1` (english, iso, language, src_afr_tokens (int32, `Davlan/afro-xlmr-large-76L` tokens with specials), length_bin, n_sentences (int32), source_dataset, licence, domain, field, doc_id, window_id); the 77 files are sci_v1 unchanged, plus **+1,210 rows for each of 25 languages** (ish bin bbj tsc kon nus dag oke shi tir dyu amh fon kbp ndc ibb urh yao sid dua bas wes iso wal swc), all `domain = general`: 450 NEW English windows (SALT agri / health, OLDI seed Wikipedia, Tatoeba, FineTranslations English of African-language web pages: local-topic, disjoint per language, hold-out filtered; field = `salt:...`, `oldi_seed:wikipedia`, `finetranslations:<iso>`, ...) in bins b64 150 / b256 150 / b512 100 / b1024 50, and 760 unused windows of `rl_prompts_v3` (fineweb-edu / Wikipedia, 190 per bin, `field = general_v3_unused`). `rl_prompts_extra_v1/` has only the added rows; `_licence_classes.csv` maps source_dataset to class (the schema has no class column; odc-by -> commercial_ok, cc-by-sa -> share_alike, cc-by-2.0-fr -> commercial_ok). No b2048 rows were added (the v3 pool stops at b1024 and the added English windows are sentence-level sources).

**What the sampler sees** (`language_alpha 0`, 800 steps x 42 prompts = 33,600 draws, `domain_weights` science 0.7 / general 0.3): a language now has **1,389 prompts (598 general + 791 science)**; with the 25 extra languages at v2 **2,599 prompts (1,808 general + 791 science)**. Total pool 106,922 -> 137,172 rows.

| iso | weight now -> recommended | draws now -> recommended (of 33,600) | passes over its pool (v1 / v2) |
|---|---|---|---|
| ish | 1 -> 4.0 | 311 -> 1,014 | 0.73 / 0.39 |
| bin, bbj, oke | 1 -> 3.5 | 311 -> 888 | 0.64 / 0.34 |
| kon, tsc, dag | 1 -> 3.0 | 311 -> 761 | 0.55 / 0.29 |
| nus | 2.5 -> 3.0 | 778 -> 761 | 0.55 / 0.29 |
| tir | 2.5 -> 3.0 | 778 -> 761 | 0.55 / 0.29 |
| dyu, amh, fon, kbp | 2.5 -> 2.5 | 778 -> 634 | 0.46 / 0.24 |
| ndc, ibb | 1 -> 2.0 | 311 -> 507 | 0.37 / 0.20 |
| shi, urh, yao, sid, dua, bas, wes, iso, wal, swc | 1 -> 1.5 | 311 -> 380 | 0.27 / 0.15 |
| every other language at weight 1 | 1 | 311 -> 254 | |

(the draw budget is fixed, so raising 24 weights lowers every untouched language by 18 % and the 2.5x languages by 18 %; sum of weights 108 -> 132.5.) `LANG_WEIGHTS_recommended.json` (D-65 format, mode `replace`) or, for the yaml of a restart:
`language_weights: {tir: 3.0, amh: 2.5, kbp: 2.5, nus: 3.0, kam: 2.5, bam: 2.5, mos: 2.5, umb: 2.5, kmb: 2.5, dyu: 2.5, kea: 2.5, luo: 2.5, kik: 2.5, orm: 2.5, fon: 2.5, yor: 2.5, vmw: 2.5, efi: 2.5, hau: 2.0, lug: 2.0, nso: 2.0, zul: 2.0, ish: 4.0, bin: 3.5, bbj: 3.5, oke: 3.5, kon: 3.0, tsc: 3.0, dag: 3.0, shi: 1.5, ndc: 2.0, ibb: 2.0, urh: 1.5, yao: 1.5, sid: 1.5, dua: 1.5, bas: 1.5, wes: 1.5, iso: 1.5, wal: 1.5, swc: 1.5}`.

Reasons: (1) the eight are the languages with the least reward-independent signal (no benchmark for four) and the loops measured by the anchors report (ish 86 % of outputs flagged, bin 74 %, bbj 66 %); RL is the only stage that can move them because SFT has nothing new for them; (2) consensus-only RL gained +5 chrF++ on the languages it trained on, so more prompts there are the cheapest lever; (3) the cap: weights above ~4 push a language over one pass of even the v2 pool within 800 steps and, per R13, the effect of more samples is +1.4 chrF++ per doubling with a CI that includes 0, so I stopped at 4.0 for ish and did not touch the ordinary languages; (4) shi gets only 1.5 because its low score is a script artefact (Arabic-script reference vs Latin output), amh keeps 2.5 because it is a science language already drawn more through the science domain; (5) no-benchmark languages get 1.5 only: their quality is judged by consensus alone, so extra weight there cannot be verified. Expected prompts per language over 800 steps are in the table; a language at 4.0 sees ~1,014 prompts (~0.4 passes of 2,599).

**SFT replay** (R4's mixture): ish already has 56 % of its whole corpus in the replay (916 of 1,697 windows), kon 60 %, bbj 25 %. Raising their weight above one pass only repeats ~1.7 k JW windows (JW % 78 / 93), and R4's full-weight result is that pulling the RL policy back toward noisy JW / Bible references **lowered the weak-blend languages by 2.3 chrF++ and the consensus-only languages by 1.2** (alpha 1.0, weak suite). So: **keep ish, kon, bbj, oke, tsc, bin at <= 1.0 pass of their own data (ish 164 k and kon 136 k target tokens would be the whole pass), do not up-weight them in the SFT replay**, spend the weight in RL, and use the new data only where it is cleaner than the replay (section 7 budgets).

## 9. Does more data help? (honest judgement, evidence, probe)

* **Cross-language association is positive but confounded.** Spearman between log10(original SFT target tokens) and RL chrF++ over the 64 languages with a benchmark: +0.66 (blend +0.78, n = 9; QE-validated +0.52, n = 42; consensus-only +0.38, n = 13, not significant). Counter-examples: kea (0.11 M tokens) 26.8, kon (0.14 M) 16.7, ibb (0.20 M) 22.3, her (0.23 M) 24.9 vs nus (9.4 M) 20.8, fon (34 M) 21.6, amh (8.3 M) 18.9, tir (5.6 M) 13.8: volume does not explain nus, fon, amh, tir.
* **RL samples (R13, D-65):** +1.38 chrF++ per doubling, 95 % CI [-0.28, +3.11] over 64 languages; 2c was a plateau at the dev resolution.
* **Added SFT references can hurt:** R4 `mix_v1`, alpha 1.0: weak-blend (19 languages) -2.34 [-2.93, -1.75], weak-consensus -1.19 [-1.85, -0.44], controls -0.59, all 63 languages -1.31, while the science languages gained; partial interpolation (alpha 0.3-0.5) is neutral (+0.07 / -0.02 overall). The replay content is 58 % JW. New data helps only if it is cleaner than what the RL policy already reproduces.
* **My expectation by group** (no new training run was made; this is a judgement from the above):
  * strong languages (afr swh zul xho hau yor ibo): saturated (chrF++ 38-63), at risk of translationese when synthetic targets are added, and no commercial-OK new data exists for them anyway -> **0 extra tokens**.
  * mid (kin sna som sot tsn nya mlg orm lug, 33-47 chrF++): possible small gains, but only TranslatePsy-type synthetic data exists -> not available commercially; ACS adds 0.1 M each at QE level of the RL model itself -> budget <= 0.15 M, low expected value.
  * weak with human-quality new data (nus: OLDI 0.7 M; ach / nyn: SALT 0.7 / 0.8 M; bam OLDI 0.5 M): **the best bet** for measurable gains, because the sources are professional translations in new domains and the languages are low (nus 20.8, ach 26.3, nyn 25.0, bam 24.9).
  * weak with web-text-in-language data and synthetic English (tir, amh, sid, bas, her, kea, kam, kua): natural targets; probably useful for tir (13.8, the worst QE-validated language, 2.1 M extra tokens recommended) and amh; for languages that already score 25-31 expect little.
  * the eight weak languages without any new data: none from data; RL weight only.
* **Recommended token budgets** (extra, prompt + answer): tir 2.0 M, amh 1.5-1.9 M (GoURMET + FT + OLDI-type), nus 0.95 M, ach 0.7 M, nyn 1.1 M, bam 0.9 M, sid 0.6 M, bas 0.4 M, her / kea / kam / kua 0.5 M each, shi 0.33 M, dag 54 k, kon 12 k; synthetic-source share per language <= 40 % of the language's tokens (FineTranslations rows: natural target / synthetic source), ACS <= 15 %.
* **Probe (GPU runs belong to R7).** Files: `sft_probe_v1/mix_0.25M`, `mix_1M`, `mix_4M` (R4 `mix_v1` + extra tokens for nus, ach, nyn, tir: 0.25 M each / 0.7-1.0 M each / the available maximum, nus 0.95 M, ach 0.7 M, nyn 2.9 M, tir 4.0 M; +1.0 M, +3.6 M, +8.5 M tokens in total; audit OK). A strong-language arm cannot be built from commercial data (none exists) - the saturation claim rests on the R4 / R13 evidence above.
  ```bash
  cd /workspace/jobs/r4/gemma_mt_sft; source /workspace/jobs/r4/env.sh
  for T in 0.25M 1M 4M; do
    python -m gemma_mt.train_sft --config configs/sft_science.yaml data_dir=/workspace/jobs/r10/sft_probe_v1/mix_$T \
      out_dir=/workspace/jobs/r10/probe_runs/$T "extra_val_splits=[val_sci,val_extra]"      # 293-300 steps, lr 3e-6, ~25 min each (R4 section 6)
    python scripts/sft_stage_eval.py sweep --init /workspace/sft_build/runs/final_1433 --sft /workspace/jobs/r10/probe_runs/$T/last/trainer_state.pt --alphas 0.5 1.0 --out /workspace/jobs/r10/probe_eval/$T
  done   # plus the control: R4's mix_v1 run (T = 0)
  ```
  **Decision rule**: per probe language, FLORES+ **dev** chrF++ (60 fixed sentences for nus and tir, the `weak` suite; SmolSent / SALT-dev-free references for ach / nyn: use the 63-language `weak` suite and `val_extra` loss) and `val_extra` loss. Adopt a tier if (a) the group mean over the four languages beats the T = 0 control by >= +0.5 chrF++ with a bootstrap CI over sentences excluding 0, (b) the gain at 1 M is >= half the gain at 4 M (otherwise the extra tokens are wasted), (c) the general-dev macro (19 languages) does not drop by more than 0.5 and the `finished` rate does not drop, and (d) the controls do not lose. If (a) fails at 1 M, stop adding data for that group and spend the budget on RL weight. Noise floor from R4: +-0.4 chrF++ at 5 k sentences, +-3 per language at 60 sentences, so decide on group means only.

## 10. Open points and caveats

* The overlap with the original SFT is inferred from a random sample of its windows (1-76 % per language, 2.9 % for amh, 1.2 % for nus); a source with low pool coverage could be partly used without showing up (nus, tir, amh). The methodology is sound for the large sets (expected hits 70-1,400, observed 0-137) and weak for tiny ones.
* `licence_class` of FineTranslations is commercial_ok on the dataset's declared licence; the English side is Gemma-3-27B output (Gemma terms apply, consistent with the Gemma-3-270m base) and the target web text carries the copyright of its sites (unvetted). GoURMET's CC0 comes from the OPUS README inside the zip, not from a Hub card. GlobalVoices' CC-BY-3.0 comes from the site's terms.
* The English-source windows of ACS / FineTranslations / SALT are unrelated sentences joined in order (like most of the original windows), so they teach sentence-level behaviour, not document coherence; only AfriDocMT (NC) and FineTranslations (same page chunks) keep document order.
* Not built: MAFAND fr-bbj with a machine-translated English pivot (NC), the monolingual target texts (Omnilingual-ASR ish 481, fineweb-2 ish 46 / bin 812 / tsc 397 / dag 1,037, Wikipedia kg 1,329 / dag 10,071, Waxal dag transcripts, MasakhaNER2 bbj) - they need an X -> English pass by a teacher (R14) to become en -> X data, and then they are the only route for ish / bin / oke / bbj.
* The first `commercial_clean` build was deleted and rebuilt as separate `commercial_ok` / `share_alike` dirs (with GoURMET / GlobalVoices added) while R11 may have been reading the first one; the dirs were absent for roughly 10 minutes during the rebuild. All markers point to the final dirs.
