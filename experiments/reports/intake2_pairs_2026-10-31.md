# extra2 parallel-pair intake for the clean SFT rerun (2026-10-31)

Output: `/workspace/work/hub/extra2/<source>/pairs-NNNN.parquet` (17 sources, 200k rows per shard, same schema as `hub/extra`: `english, target, iso, source_dataset, licence, licence_raw, kind, source_family, flipped, key, aux`), stats in `hub/extra2/_stats/` (`per_language_source.csv`, `removed_by_protected_set.csv`, `summary.json`, plus `tokens_by_source_language.csv`, `tokens_by_language.json`). Code: `data_pipeline/sft_clean/chain/intake2_pairs.py` (+ `intake2_report.py` for the tables below), also on the instance in `/workspace/work/code/sft_clean/chain/`. The directory `hub/extra2/nllb/` belongs to the lead and was not touched.

**Result: 8,860,021 pairs kept (15.93 M raw rows read), about 507 M Gemma tokens (english + target), 80 iso codes, 39 of them new.** Every kept key is unique inside extra2 and absent from the chain inputs (checked against `work/existing_keys.npy`, 24,441,133 keys from hub/data, data_devhalf, specialist, extra). No row has licence class `non_commercial`.

## Method

* Pattern of `extra_sources.py`: NFC + whitespace, `ok()` (length/ratio/letters/URL rules), dedupe key `common.key_of(english, iso, target)`, STRICT protected-set filter (`extra_sources.Filter`, index `protected-2026-10-28.3`, both sides, any language, rules R1 exact, R2 contiguous, R3 shared 8-gram). Order per row: clean -> source rules (>= 3 English words for UI/word-level sets, placeholders dropped, per-row licence) -> dedupe against the chain-input key set and inside the task -> protected filter -> staging -> finalize (removes pairs that an EARLIER source in the priority order below already kept, caps, writes shards). The protected filter ran on every row that survived the dedupes (removed: 96,516 rows = 38,329 R1 english, 37,486 R3 english, 9,802 R1 target, 7,773 R3 target, 1,056 + 730 R2).
* Raw files were downloaded one at a time to `work/_intake2/raw`, converted and deleted (the raw dir is empty). The staged converted shards (`work/_intake2/stage`, 0.87 GB) are kept so that `finalize` can be re-run; delete them when no longer needed.
* Test per source on 1,000 rows (3 tasks per source) before the full run; full run 24 workers at `nice -n 10`, 10 minutes for 235 tasks (plus 4 autshumato tasks later); no GPU use, no process touched.
* Licence: `licence` = class (cc-by, cc0, mit, apache, share_alike, mpl, unspecified, other), `licence_raw` = card string or the per-row licence field. Splits named `test*` were never read (Mehgoss, Inioluwa, Tommy0201, iamwille, Gazette, govza); validation/eval splits of non-benchmark corpora were read. Rows whose per-row licence says non-commercial would be dropped (1 Weblate row was).
* Disclosure: kept sets with `licence=unspecified` (no licence on the Hub): `igbo_bundles` (Tommy0201/* copies; 526,275 rows), `weblate` (per-string licences, class derived per row), `nupe` (card tag `cc`), `twi_kweku` (12 rows), `xhosa_silvano` (per-row licence column, the kept rows are labelled CC-BY there but the card says other), `swh_rogendo`, `kin_gazette` (card tag `cc`). `okwu` is CDLA-Permissive-2.0 (class `other`). `weblate` carries 45,829 `share_alike` rows (GPL/CC-BY-SA strings) and 1,827 `mpl` rows: the lead decides whether they stay.

## Source by source

1. `mt560_new` (priority 1). The 31 `michsethowusu/english-<language>_sentence-pairs_mt560` repos that the clean build did not use (99 on the Hub minus the 68 in the existing `source_dataset` values): alur alz, bulu bum, chokwe cjk, chopi cce, chuwabu chw, coptic cop, dangme ada, egyptian-arabic arz, gun guw, konzo koo, kwangali kwn, lamba lam, luba-katanga lub, lukpa dop, luvale lue, mambwe-lungu mgr, morisyen mfe, ndonga ndo, nyaneka nyk, nyemba nba, nyungwe nyu, ruund rnd, seychelles-french-creole crs, songe sop, southwestern-dinka dik, tamashek taq, tetela tll, tonga toi, tooro ttj, zande zne, zarma dje (iso3 as in the audit CSV). cc-by-4.0 by card, JW-derived religious text. `arz` (Egyptian Arabic) and `cop` (Coptic) are the 31st and a non-living-language repo; drop them if out of scope.
2. `mehgoss`. `Mehgoss/sa-languages-translation` train + validation (10 SA languages; test split not read). The afr/sot/ssw/ven parts are small (about 3k rows each, 78-82% already in the chain inputs via Vukuzenzele); the other six languages are almost all new (zul 223k, tsn 223k, tso 162k, nso 130k, nbl 121k, xho 121k kept). Component (`vukuzenzele`, `autshumato`, JW) is in `source_family`.
3. **`NaathNLP/Nuer_Dinka_Parallel_translation_pairs`: SKIPPED.** 899,372 English/Nuer/Dinka triples in one 268 MB JSON. Sampled 200 random rows plus corpus statistics: the English side has 243 distinct words over the first 200k rows, four sentence openings account for 90k rows ("at the cattle camp", "by the riverbank the", "during the dry season", "among the elders the"), 60% of the sample is the same greeting / cattle-camp register, 8 of 200 Dinka sides contain a repeated-bigram loop ("thää ke thää ke thää ke"), provenance is not stated (card is empty apart from `license: mit`), and both African sides read as machine-generated. Judged obviously degenerate: not ingested (sample file: `work/_intake2/naath_sample200.jsonl`). nus/din therefore still have no new pairs from this repo.
4. `igbo_bundles` (priority 4, siblings counted once by the key dedupe, in this order): `Tommy0201/JW300_Igbo_To_Eng`, `WebScraped_Igbo_To_Eng`, `IGBONLP_Igbo_To_Eng` (gated "auto" but readable with the token), `Igbo_To_Eng_Amazon_Translator` (csv), `igbo_to_english_split`, `igbo-to-english` (a copy of the split: 111,610 of its 117,964 rows dropped as duplicates), `iamwille/igbo_language_translation_dataset` (mit) and, as the last source, `ccibeekeoc42/english_to_igbo` (mit; train only; its leftover after iamwille is 33.8k rows). 1,762,028 raw rows -> 563,671 kept: 299,278 were already in the chain inputs (mt560 Igbo is JW) and 804,071 were duplicates among the siblings. **Quality flag:** in `igbo_to_english_split` / `igbo-to-english` the English side often does not translate the Igbo (first sample row: Igbo "Ndịagha Naịjrịa gbarambọ dajụọ ..." vs English "The people of Nigeria are urged to report any suspicious activity ..."); 1,121 + 110,881 rows are kept from them. The QE gate should catch these.
5. `weblate` (`Tamazight-NLP/Weblate-Translations`, only the `en-<code>` and `en_GB-<code>` configs of African languages: 84 configs mapped through `WEBLATE_MAP`, Arabic dialects excluded). `state >= 20` or approved; placeholders (`%s`, `{x}`, `<tags>`, `&amp;`) dropped with the `extra_sources.PLACEHOLDER` regex; English side >= 3 words. 206,744 raw -> 64,679 kept, 46 languages (swh 15.9k, afr 13.4k, kab 12.2k, amh 4.8k, kin 2.5k, lin 2.3k, wol 1.8k ...). Tiny sets for aar, saq, sus, son, ssy, abr, rcf, zgh, ber.
6. `kabyle_tm` (`taqbaylit/kabyle-english-TM`, cc-by-4.0, UI strings, >= 3 words: 121,725 -> 55,723), `nupe` (`Inioluwa/nigerianLanguageTranslator`, train only, English/Nupe `nup`, >= 3 words because most rows are single words: 204,900 -> 12,194; the longer rows are Quran-translation sentences with the Yusuf Ali English text, licence of the Nupe side unknown), `twi_kweku` (`kweku2088/twi_to_english`: 561,216 raw, 551,374 already in the chain inputs, i.e. it is the same data as KwekuAinoo/eng_twi, 12 new rows).
   The audit's second Nupe repo `Inioluwa/nupe-english` is empty (README only).
7. `dochplt` (`HPLT/DocHPLT`, cc0): configs af-en, en-sw, en-xh, all 70 files, streamed. **Deviation from the brief:** only 10% of documents have equal sentence-list lengths and pairing them by position is unreliable (the data carries its own alignment), so I used the dataset's alignment entries instead of the line-count rule: one source id and one target id (1:1), `bicleaner-score >= 0.5` and `aligner-score >= 0.4`, at most 40 pairs per document, a per-file quota (2x cap / files, spreads the cap over the domain-sorted files), English side >= 4 words; the bicleaner score is stored in `aux`. Documents are never split by position. 7.76 M pairs read -> 3.59 M kept (afr 2.27 M, swh 1.17 M, xho 0.15 M; below the 3 M cap per language, the cap was not reached). 3.4 M were duplicates inside the source (boilerplate, spam pages repeated across crawls). **Quality flag:** web-crawl text with obvious MT-style English on the non-English-original documents (casino/spam pages, "Qhov no yog ib tus Best Data Entry Jobs"); the QE gate is the filter here.
8. Extras, largest first (all tested at 1,000 rows, all English-pivot, none NLLB/benchmark): `xhosa_silvano` (silvanosolutions/xhosa-nlp-dataset `parallel.parquet`, 110,681 -> 45,056; 65k already in chain inputs), `swh_rogendo` (Rogendo/English-Swahili-Sentence-Pairs, >= 3 words, 210,471 -> 188,357), `kin_gazette` (DigitalUmuganda Rwandan gazette, train + val, 167,738 -> 165,025), `tatoeba_kab_shi` (abdelhaqueidali Kabyle and Tashelhit Tatoeba, latn column only, cc-by-2.0, kab 86k + shi 21k new beyond the specialist Tatoeba rows), `govza` (dsfsi/govza-sa-cabinet-statements: only the afr-eng pair is English-paired, the other 9 pairs are Afrikaans-pivot and not ingested; score >= 0.9; most lines are identical in both languages: 3,731 -> 2,768), `lug_pky` (apache-2.0, 78,238 -> 58,959), `paracrawl` (Helsinki-NLP/opus_paracrawl en-sw + en-so, cc0, 147,400 -> 131,992), `autshumato` (nwu-ctext/autshumato: the repo is a loading script, but the auto-converted parquet branch `refs/convert/parquet` of the same repo has the four configs en-tn, en-ts, en-ts-manual, en-zu; cc-by-2.5; 736,885 -> 271,371 because 427k duplicate Mehgoss/Vukuzenzele or each other), `okwu` (Okwu/african-language-parallel-corpus, `verified == true` rows only, pcm/swa/yor, 90,976 -> 89,987).

## Skipped (and why)

* `NaathNLP/Nuer_Dinka_Parallel_translation_pairs` (degenerate, see 3).
* `quranlab/quran`, `nassimjp/quran` (per-config Quran translation licences, Tanzil-type non-commercial terms), `mteb/biblenlp-corpus-mmteb` (MTEB bitext-mining benchmark data), `nordpolemil/biblenlp-corpus` (207 M whole-repo Bible translations, per-translation licences), `GiJoeHansFranz/Liseli` (88% Bible rows, licence "per-source" not recorded per row, AI-dictionary rows), `DigitalLearningGmbH/tatoeba_mt_parquet` (repo holds only test/validation splits: evaluation data), `MALIBA-AI/bambara-mt-dataset` (contains `google_smol_gatitos` rows), `mbazaNLP/Kinyarwanda_English_parallel_dataset` (stop-word-stripped lower-case text), `CapitainVigs/mece-fongbe-corpus` (chat format, no plain pair columns), `dsfsi/za-mafoko-*` (NOODL licence, not commercial-clear), `Helsinki-NLP/opus_infopankki` (47k, below the 50k threshold), French-pivot sets (FrancophonIA, galsenai, cidjeu/louisbertson moore), `Tadesse/COMET_score`, `risaleinur`, `sello-ralethe/SA-Knowledge` (mixed licences), the 47.9 M synthetic Twi set (lowest priority, old build took a QE-selected part), `Tamazight-NLP/Anki-Translations` and the other small Kabyle sets (below threshold or fr-pivot), `Tommy0201/pidgin-*` (not Igbo).
* Not read at all: any split called test; nothing from FLORES, NTREX, SMOL/GATITOS, MAFAND, TICO, AfriDocMT, NLLB, CCMatrix, CCAligned, TranslatePsy, the unsuffixed michsethowusu mirrors.

## Notes for the lead

* **New iso codes** (not in `work/gate/thresholds.json`, rows kept): lue 309,484; guw 280,465; tll 216,578; lub 192,492; zne 185,323; crs 184,129; mfe 177,242; rnd 129,274; nyk 112,716; kwn 103,680; ndo 96,974; ada 61,298; dje 60,270; arz 53,550; sop 49,452; mgr 42,510; koo 35,341; cjk 34,885; bum 27,063; nba 26,633; nyu 24,013; chw 17,601; cce 17,539; ttj 16,124; dop 15,652; cop 15,646; lam 14,367; dik 13,273; nup 12,194; alz 7,340; taq 5,347; toi 3,793; zgh 1,044; ber 926; aar 383; plus crumbs (sus 53, rcf 28, son 9, saq 1). Each needs a tokenizer tag or a gate entry before it can enter windows (token-swap policy). `ber` is the macro code used by Weblate, `taq` is Tamasheq (Tuareg), `nup` Nupe, `dop` Lukpa (Lokpa), `nyu` Nyungwe, `sop` Songe, `toi` Tonga (Zambia).
* Per-source overlap with existing rows is large for mehgoss (26k), igbo_bundles (299k), twi_kweku (551k), xhosa_silvano (65k), tatoeba (48k), autshumato (23k) and DocHPLT afr/swh (162k); all removed by key before scoring.
* `licence_raw` for mt560/Mehgoss rows says "JW-derived": the card claims cc-by-4.0 but the upstream text is Watch Tower material; same situation as the 68 mt560 repos already used.
* Rows with `source_dataset` = `HPLT/DocHPLT:<config>`; all other `source_dataset` values are the repo ids (the NC-exception lookup in `common.nc_exception_tag` splits on `:`).
* The Gemma token estimate samples up to 2,000 rows per (source, language) from the final shards and scales by rows kept (`google/gemma-3-270m-it`, english + target, no special tokens).


## Tables

### Per source

| # | source | repo(s) | licence class | raw rows read | clean/short/licence drops | dup in chain inputs | dup within / earlier source | removed by protected set | kept | est. tokens (Gemma, en+target) | languages (kept) |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `mt560_new` | michsethowusu/english-<language>_sentence-pairs_mt560 (31 repos) | cc-by | 2,593,880 | 31,367 | 0 | 2,034 | 30,476 | 2,530,003 | 147,545,604 | lue 309,484, guw 280,465, tll 216,578, lub 192,492, zne 185,323, crs 184,129, mfe 177,191, rnd 129,274, nyk 112,716, kwn 103,680, ndo 96,974, ada 61,298, dje 60,270, arz 53,550, sop 49,452, mgr 42,510, koo 35,341, cjk 34,885, bum 27,063, nba 26,633, nyu 24, ... (31 languages) |
| 2 | `mehgoss` | Mehgoss/sa-languages-translation | cc-by | 1,018,706 | 3,943 | 25,986 | 3,391 | 4,194 | 981,192 | 57,623,997 | zul 223,205, tsn 222,926, tso 162,392, nso 129,609, nbl 120,727, xho 120,595, afr 562, sot 562, ssw 559, ven 55 |
| 3 | `igbo_bundles` | Tommy0201/* , iamwille/igbo_language_translation_dataset, ccibeekeoc42/english_to_igbo | unspecified | 1,762,028 | 57,402 | 299,278 | 804,071 | 37,606 | 563,671 | 38,861,169 | ibo 563,671 |
| 4 | `weblate` | Tamazight-NLP/Weblate-Translations | unspecified | 206,744 | 139,812 | 141 | 2,023 | 89 | 64,679 | 1,763,149 | swh 15,859, afr 13,400, kab 12,229, amh 4,810, kin 2,527, lin 2,272, wol 1,783, zul 1,198, twi 1,121, zgh 1,044, tzm 945, ber 926, mlg 887, som 871, ibo 817, hau 505, ven 394, xho 392, aar 383, yor 361, sna 319, orm 291, ewe 266, tir 226, nya 151, ful 128, ... (46 languages) |
| 5 | `kabyle_tm` | taqbaylit/kabyle-english-TM | cc-by | 121,725 | 61,299 | 2,972 | 1,661 | 70 | 55,723 | 1,699,077 | kab 55,723 |
| 6 | `nupe` | Inioluwa/nigerianLanguageTranslator | unspecified | 204,900 | 130,591 | 0 | 62,050 | 65 | 12,194 | 592,043 | nup 12,194 |
| 7 | `twi_kweku` | kweku2088/twi_to_english | unspecified | 561,216 | 4,541 | 551,374 | 7 | 5,282 | 12 | 658 | twi 12 |
| 8 | `dochplt` | HPLT/DocHPLT | cc0 | 7,755,532 | 577,793 | 162,401 | 3,412,490 | 11,156 | 3,591,692 | 203,505,924 | afr 2,266,811, swh 1,174,988, xho 149,893 |
| 9 | `xhosa_silvano` | silvanosolutions/xhosa-nlp-dataset | unspecified | 110,681 | 385 | 65,081 | 1 | 158 | 45,056 | 2,725,527 | xho 45,056 |
| 10 | `swh_rogendo` | Rogendo/English-Swahili-Sentence-Pairs | unspecified | 210,471 | 9,467 | 1,005 | 7,940 | 3,702 | 188,357 | 5,115,493 | swh 188,357 |
| 11 | `kin_gazette` | DigitalUmuganda/NMT_Rwandan-Gazette_parallel_data_en_kin | unspecified | 167,738 | 2,550 | 1 | 23 | 139 | 165,025 | 14,195,863 | kin 165,025 |
| 12 | `tatoeba_kab_shi` | abdelhaqueidali/{Kabyle-Latin-to-Tifinagh-Parallel-Corpus, Tashelhit-Tifinagh-Latin-Parallel-Tatoeba-Dataset} | cc-by | 156,586 | 38 | 48,031 | 58 | 1,119 | 107,340 | 2,166,008 | kab 86,123, shi 21,217 |
| 13 | `govza` | dsfsi/govza-sa-cabinet-statements-sentence-aligned | cc-by | 3,731 | 880 | 8 | 39 | 36 | 2,768 | 272,385 | afr 2,768 |
| 14 | `lug_pky` | pkyoyetera/luganda_english_dataset | apache | 78,238 | 1,816 | 3,224 | 14,184 | 55 | 58,959 | 2,895,594 | lug 58,959 |
| 15 | `paracrawl` | Helsinki-NLP/opus_paracrawl | cc0 | 147,400 | 1,269 | 6,832 | 6,904 | 403 | 131,992 | 12,071,585 | swh 119,561, som 12,431 |
| 16 | `autshumato` | nwu-ctext/autshumato (refs/convert/parquet) | cc-by | 736,885 | 13,623 | 23,334 | 427,217 | 1,340 | 271,371 | 10,700,636 | tso 187,925, tsn 60,753, zul 22,693 |
| 17 | `okwu` | Okwu/african-language-parallel-corpus | other | 90,976 | 318 | 16 | 29 | 626 | 89,987 | 5,130,690 | pcm 30,176, swh 29,934, yor 29,877 |
| | **total** | | | 15,927,437 | 1,037,094 | 1,189,684 | 4,744,122 | 96,516 | 8,860,021 | 506,865,402 | |

### Per language (rows kept, estimated tokens of english+target, protected-set removals)

| iso | in thresholds.json | rows kept | est. tokens | raw rows read | dup in chain inputs | removed by protected set |
|---|---|---:|---:|---:|---:|---:|
| afr | yes | 2,283,541 | 125,492,768 | 5,131,836 | 91,842 | 5,861 |
| swh | yes | 1,528,699 | 86,829,366 | 2,830,944 | 70,690 | 8,598 |
| ibo | yes | 564,488 | 38,879,103 | 1,764,417 | 299,278 | 37,607 |
| tso | yes | 350,355 | 13,318,616 | 709,386 | 13,395 | 1,277 |
| xho | yes | 315,936 | 19,371,300 | 491,677 | 75,897 | 1,487 |
| lue | **new** | 309,484 | 17,544,647 | 317,092 | 0 | 3,762 |
| tsn | yes | 283,722 | 16,197,538 | 391,070 | 15,660 | 1,971 |
| guw | **new** | 280,465 | 18,317,028 | 286,908 | 0 | 3,514 |
| zul | yes | 247,096 | 15,748,902 | 268,832 | 6,642 | 615 |
| tll | **new** | 216,578 | 13,509,377 | 222,225 | 0 | 2,901 |
| lub | **new** | 192,492 | 10,861,264 | 197,423 | 0 | 2,703 |
| zne | **new** | 185,323 | 11,124,383 | 190,082 | 0 | 2,599 |
| crs | **new** | 184,129 | 9,202,491 | 188,361 | 0 | 2,429 |
| mfe | **new** | 177,242 | 9,098,856 | 181,666 | 0 | 2,410 |
| kin | yes | 167,552 | 14,244,025 | 178,246 | 12 | 143 |
| kab | yes | 154,075 | 3,853,888 | 295,582 | 51,020 | 997 |
| nso | yes | 129,609 | 8,788,138 | 134,935 | 2,565 | 1,129 |
| rnd | **new** | 129,274 | 6,962,956 | 133,631 | 0 | 1,696 |
| nbl | yes | 120,765 | 7,358,674 | 122,586 | 188 | 220 |
| nyk | **new** | 112,716 | 6,035,772 | 116,364 | 0 | 1,479 |
| kwn | **new** | 103,680 | 5,285,347 | 106,595 | 0 | 1,300 |
| ndo | **new** | 96,974 | 5,622,552 | 99,817 | 0 | 1,203 |
| ada | **new** | 61,298 | 3,720,727 | 63,021 | 0 | 725 |
| dje | **new** | 60,270 | 4,766,935 | 60,515 | 0 | 99 |
| lug | yes | 59,019 | 2,896,860 | 78,473 | 3,224 | 55 |
| arz | **new** | 53,550 | 3,003,324 | 54,590 | 0 | 105 |
| sop | **new** | 49,452 | 2,826,849 | 51,070 | 0 | 646 |
| mgr | **new** | 42,510 | 2,306,550 | 43,911 | 0 | 538 |
| koo | **new** | 35,341 | 2,001,820 | 36,378 | 0 | 377 |
| cjk | **new** | 34,885 | 1,828,444 | 35,767 | 0 | 387 |
| yor | yes | 30,238 | 2,200,779 | 31,233 | 0 | 279 |
| pcm | yes | 30,176 | 1,224,602 | 30,370 | 0 | 143 |
| bum | **new** | 27,063 | 1,438,696 | 27,996 | 0 | 313 |
| nba | **new** | 26,633 | 1,389,829 | 27,317 | 0 | 264 |
| nyu | **new** | 24,013 | 1,381,299 | 24,491 | 0 | 194 |
| shi | yes | 21,217 | 352,000 | 21,438 | 0 | 212 |
| chw | **new** | 17,601 | 857,925 | 17,988 | 0 | 121 |
| cce | **new** | 17,539 | 892,840 | 17,935 | 0 | 126 |
| ttj | **new** | 16,124 | 861,956 | 16,442 | 0 | 135 |
| dop | **new** | 15,652 | 1,704,596 | 15,711 | 0 | 44 |
| cop | **new** | 15,646 | 2,073,243 | 15,706 | 0 | 43 |
| lam | **new** | 14,367 | 819,608 | 14,846 | 0 | 208 |
| som | yes | 13,302 | 1,413,070 | 18,162 | 2,385 | 27 |
| dik | **new** | 13,273 | 1,028,418 | 13,319 | 0 | 35 |
| nup | **new** | 12,194 | 592,043 | 204,900 | 0 | 65 |
| alz | **new** | 7,340 | 385,394 | 7,567 | 0 | 68 |
| taq | **new** | 5,347 | 487,491 | 5,363 | 0 | 13 |
| amh | yes | 4,810 | 138,580 | 16,867 | 37 | 8 |
| toi | **new** | 3,793 | 206,149 | 3,898 | 0 | 39 |
| lin | yes | 2,272 | 43,609 | 4,109 | 0 | 0 |
| wol | yes | 1,783 | 73,468 | 2,915 | 0 | 1 |
| twi | yes | 1,133 | 25,341 | 563,830 | 551,374 | 5,283 |
| zgh | **new** | 1,044 | 33,652 | 5,550 | 0 | 0 |
| tzm | yes | 945 | 15,059 | 3,338 | 0 | 0 |
| ber | **new** | 926 | 34,044 | 3,151 | 0 | 0 |
| mlg | yes | 887 | 25,615 | 1,727 | 1 | 7 |
| sot | yes | 610 | 47,464 | 3,361 | 2,605 | 25 |
| ssw | yes | 597 | 45,189 | 3,270 | 2,614 | 18 |
| hau | yes | 505 | 14,091 | 1,242 | 0 | 1 |
| ven | yes | 449 | 11,588 | 7,426 | 254 | 3 |
| aar | **new** | 383 | 6,555 | 1,506 | 0 | 2 |
| sna | yes | 319 | 8,522 | 835 | 0 | 2 |
| orm | yes | 291 | 9,148 | 619 | 0 | 1 |
| ewe | yes | 266 | 5,173 | 650 | 0 | 0 |
| tir | yes | 226 | 6,943 | 3,956 | 1 | 0 |
| nya | yes | 151 | 3,211 | 574 | 0 | 3 |
| ful | yes | 128 | 3,162 | 2,789 | 0 | 0 |
| sus | **new** | 53 | 1,215 | 94 | 0 | 0 |
| nde | yes | 38 | 1,048 | 55 | 0 | 0 |
| run | yes | 38 | 1,037 | 55 | 0 | 0 |
| sag | yes | 38 | 968 | 55 | 0 | 0 |
| rcf | **new** | 28 | 946 | 175 | 0 | 0 |
| wal | yes | 21 | 329 | 134 | 0 | 0 |
| bem | yes | 13 | 151 | 107 | 0 | 0 |
| ach | yes | 9 | 150 | 581 | 0 | 0 |
| son | **new** | 9 | 104 | 214 | 0 | 0 |
| kik | yes | 6 | 51 | 118 | 0 | 0 |
| bam | yes | 2 | 159 | 17 | 0 | 0 |
| tum | yes | 2 | 294 | 3 | 0 | 0 |
| saq | **new** | 1 | 98 | 9 | 0 | 0 |
| **total** | | 8,860,021 | 506,865,402 | 15,927,437 | 1,189,684 | 96,516 |

New iso codes (not in work/gate/thresholds.json): aar (383), ada (61,298), alz (7,340), arz (53,550), ber (926), bum (27,063), cce (17,539), chw (17,601), cjk (34,885), cop (15,646), crs (184,129), dik (13,273), dje (60,270), dop (15,652), guw (280,465), koo (35,341), kwn (103,680), lam (14,367), lub (192,492), lue (309,484), mfe (177,242), mgr (42,510), nba (26,633), ndo (96,974), nup (12,194), nyk (112,716), nyu (24,013), rcf (28), rnd (129,274), saq (1), son (9), sop (49,452), sus (53), taq (5,347), tll (216,578), toi (3,793), ttj (16,124), zgh (1,044), zne (185,323)
