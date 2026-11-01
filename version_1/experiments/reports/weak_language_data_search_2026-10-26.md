# New commercially usable English-parallel data for the weak languages (agent D1, 2026-10-26)

Scope after the coordinator's priority update: the 20 lowest languages on W1's FLORES+ dev ranking (`reports/weak_language_selection_2026-10-26.md`: tir dyu amh kbp fon nus mos kam bam umb kea kmb vmw kab seh yor wol ewe luo tum), the owner's additions efi, ibb, urh, then the owner's earlier list (knc, run, ish, bin, bbj, tiv, oke, tsc ...). Laptop only: metadata, cards and samples of at most 200 rows (stored in `reports/data/weak_data_search_2026-10-26/samples/`); nothing bulk was downloaded, no Hub write, no paid API, vast box and `paper/` untouched.

Files: `reports/data/weak_data_search_2026-10-26/` = `sources.csv` (42 candidate sources, licence quotes, URLs), `lang_summary.csv` (per-language totals and fetch plan), `gap_analysis.csv`, `leakage_check.py` + `leakage_results.csv`, `samples/`.

## 0. Headline

1. **The 100-200k new commercially usable pairs per language the owner hopes for do not exist for any of the 23 priority languages, except through synthetic-English FineTranslations (tir, amh, luo, run) and, for kab, one Tatoeba block that may already be in the SFT.** Counting only human translations with a permissive or share-alike licence (plus FLORES+ dev, which the owner allows), realistic NEW pairs per language are: kab ~72k, yor ~47k (not needed), tir ~22k, amh ~7.7k, kea ~8.2k, mos ~6.8k, nus ~4.1k, tum ~3.9k, ewe ~3.1k, wol ~2.8k, vmw ~1.5k, kbp ~1.5k; **dyu, kam, bam, umb, kmb, seh ~0.9k each (only the FLORES+ dev 897)**; efi 0.5k, ibb 0.8k; urh, ish, bin, oke, tsc, tiv: nothing. Table in section 4.
2. **Biggest real levers found.** (a) **Kanuri (knc) is no longer empty**: Gamayun kau-eng 5,000 pairs (CC-BY-4.0), OLDI Seed knc_Latn 6,193 pairs (CC-BY-SA, plus another 6,193 in Ajami script), TICO-19 3,070 pairs (CC0) = ~14k human pairs / ~0.7M target tokens against 0 today (but knc is not one of the 77 model languages). (b) **Tigrinya**: Gamayun 5,000 (CC-BY-4.0), TICO-19 6,140 (CC0, two translators), Travis Foundation crowd-sourced block ~5,000 (CC-BY-SA), plus the unused remainder of FineTranslations tir (72k real VOA/assenna documents; ~50k not yet used). (c) **FineTranslations for Kirundi** (74,480 real news documents) and Dholuo (2,692): in the original build list but never measured; R10 only measured 11 other languages.
3. **Everything else is JW/Bible text already in the SFT (58-93% of the replay), UI strings, or tiny Tatoeba stubs.** Fully checked dead ends: ish, bin, oke, tsc, tiv, umb, kam, dyu, bam, seh, urh, ibb, efi, kmb (Portuguese only), bbj (Bible only).
4. **Leakage:** 0 exact and 0 8-gram hits for 10 of 11 sampled sources (up to 1,195 sentences each) against NTREX (all 1,997), SmolSent/SmolDoc English of 8 languages and W1's 200 dev-sample sentences; only TICO-19 (1 of 600 sentences, 0.2%) and Tatoeba rn (1 of 300 exact) hit. **Limit: FLORES+, SALT and MAFAND English are gated or unreachable from here, so this is a lower bound; `leakage_check.py` is ready to run on the box with the full pools.**
5. **Licence traps found:** Zenodo Dholuo corpus (CC-BY 4.0 on the record, Apache-2.0 in the zip, CC-BY-NC-4.0 on its English extension); Travis Foundation Tigrinya (CC-BY-SA declared, but ~90% scraped JW text "without permission"); `flagship-ai` Ghomala/Cameroon Bibles (CC-BY-4.0 label, "rights belong to publishers"); kabuverdianu-english (ODC-By but revised NLLB-mined text); FineTranslations (ODC-By, English side = Gemma-3-27B output, web copyright and CommonCrawl terms unvetted).

## 1. How I searched

Hub API (`filter=language:<2-letter and 3-letter codes>`, 37 languages, then diff against the 7,273 dataset ids of `data-catalog/african_hf`; plus name searches such as "kanuri", "kikamba", "ghomala", "kabiye" for repos without language tags; plus author listings of CLEAR-Global, openlanguagedata, Sunbird, masakhane, google, cis-lmu, dsfsi, Tamazight-NLP, burkimbia, flagship-ai ...), OPUS API (all moses pairs for en/fr/pt x each language, licence lines read from the README of the small corpora), GitHub repository search, Zenodo, Wikipedia API (article counts), eBible `translations.csv`, FineWeb-2 / FineTranslations config sizes and 300-document samples per config (domains, JW share), web search for Masakhane/Lanfrica/HornMT/EthioMT/Tigrinya/Kanuri/Kabiye/Esan/Edo/Tiv. The Hub search finds only ~3% of repos that were not already in the crawl catalogue; the finds that mattered were already in the catalogue but filed under other languages (Gamayun, kanuri-books) or never tested for these languages (OLDI knc, FineTranslations run/luo/knc, TICO).

Token estimates = target words x tokens/word measured on FLORES+ devtest (tir 3.46, amh 3.03, nus 2.8, Latin default 2.3-2.8); +-30%.

## 2. Gap analysis (what we have), 23 priority languages

Full CSV with all 53 languages: `gap_analysis.csv`. "SFT" = target tokens of the original Gemma SFT windows; "R10 built" = commercial-OK extra files built on the box 2026-10-22 (target tokens; **not verified whether merged into the continuation mix**); JW = share of replay tokens that look like JW/Watchtower text.

| iso | W1 dev chrF++ | SFT tok | JW % | R10 built (comm. / SA / unknown) | licence picture of what exists |
|---|---|---|---|---|---|
| tir | 14.4 | 5.60M | 83 | 5.69M / 0 / 0.54M | JW + FT synthetic-English; very little human clean data |
| dyu | 18.4 | 0.48M | 80 | 0 | JW only |
| amh | 18.9 | 8.29M | 58 | 7.21M / 0.39M / 8.14M (+6.5M NC) | large; not data-limited |
| kbp | 20.3 | 0.43M | 71 | 0 | JW only |
| fon | 20.4 | 34.0M | 10 | 0 | large |
| nus | 20.4 | 9.37M | 0 | 0.17M / 0.70M | synthetic stories + OLDI/FT |
| mos | 23.4 | 4.76M | 76 | 0 | JW |
| kam | 23.5 | 1.86M | 79 | 1.72M (FT) | JW + FT |
| bam | 24.4 | 0.38M | 0 | 1.48M (FT) / 0.53M (OLDI) | |
| umb | 24.9 | 3.68M | 84 | 0 | JW |
| kea | 25.3 | 0.11M | 87 | 1.44M (FT) | tiny base |
| kmb | 25.6 | 2.52M | 83 | 0 | JW |
| vmw | 27.2 | 2.13M | 86 | 0 | JW |
| kab | 27.5 | 3.89M | 14 | 0 | Tatoeba status unmeasured |
| seh | 29.1 | 1.43M | 84 | 0 | JW |
| yor | 29.2 | 145M | 27 | 0 | large; weakness is not volume |
| wol | 29.9 | 5.03M | 10 | 0 | |
| ewe | 30.2 | 3.10M | 9 | 0 | |
| luo | 32.2 | 17.1M | 22 | 0 | large |
| tum | 32.6 | 3.82M | 83 | 0 | JW |
| efi | unmeasured | 15.6M | 79 | 0 (0.08M NC) | |
| ibb | unmeasured | 0.20M | 76 | 0 (0.09M NC) | |
| urh | unmeasured | 1.22M | 77 | 0 | |
| knc | not in the 77 | 0 | - | 0 | |
| run | 33.3 (rank 24) | 4.11M | 82 | 0 | |

## 3. Ranked NEW sources per language

Class legend: **P** permissive (verified licence line), **P\*** permissive label but flagged (provenance, synthetic English, upstream mixed licences), **SA** share-alike (listed separately), **U** unstated or conflicting (not counted), **NC** excluded. Pair counts are NEW, after removing what R10 already built. Details, licence quotes and fetch commands per source: `sources.csv` (id in brackets).

**All tier-A languages (tir dyu amh kbp fon nus mos kam bam umb kea kmb vmw kab seh yor wol ewe luo tum, + run, knc):** FLORES+ dev, 897 pairs each after W1's 100 held-out ids (SA, CC-BY-SA-4.0, gated: needs the box token; 0.45-0.5 MB per language) [S40]. The English side is 98.7% inside the SFT text already (W1 section 6), so only the target side is new information.

1. **tir.** (1) Gamayun-kits `tig-eng`, 5,000 pairs, P (CC-BY-4.0), professional translation, short sentences [S01]. (2) TICO-19 en-ti, 3,070 English sentences with two Tigrinya translations each = 6,140 pairs, P (CC0), health/news [S03]. (3) FineTranslations tir_Ethi: R10 used 19.3k rows of 72,009 documents; ~50k documents unused (est.), P\* — real Ge'ez target text (59% VOA Tigrigna, US-government news; assenna.com; dimtsiwoyane.com), English = Gemma-3-27B; ~10-15M target tokens extrapolated [S07]. (4) Travis Foundation, first ~5,000 crowd-sourced Wikipedia-snippet pairs, SA (declared CC-BY-SA-4.0); the other ~58k pairs are JW and must be excluded; split point approximate [S21]. (5) translatewiki 4,776 UI strings, P\* [S09]. (6) Common Voice `ti` 4,519 sentences (CC0, monolingual) [S12]. U: HornMT 2,030 news pairs (benchmark, no licence), eyosidesta compiled set, NLLB-mined 1.4M (3 of 3 read pairs misaligned in R10). NC/unobtainable: EthioMT (140k tir, never released), Tanzil.
   **Clean human pairs ~22k (0.9M tokens) vs 11.3M held; reaching 100k needs FT.**
2. **amh.** (1) TICO-19 amh 3,071 pairs, P, only in the original release, not on OPUS [S03]. (2) translatewiki 3,166 UI, Tatoeba 216, P/P\* [S09, S11]. (3) FineTranslations amh_Ethi: 421,933 documents, R10 used 21k rows: **>100k further synthetic-English pairs available**, P\* [S07]. SA: wikimedia 300, FLORES dev. NC/restricted: Gezmu 145k (research only), QED, Tanzil, AfriDocMT. amh is not data-limited; its chrF++ problem is script/segmentation (W1 section 7).
3. **dyu, kam, bam, umb, seh, kmb (English side), kbp.** Only FLORES+ dev (897, SA). Beyond that: kbp translatewiki 625 UI; kam Tatoeba 10; umb Tatoeba 32; bam: OLDI and FT already used; QED is research-only; kmb: **KmbPtMT 18,100 Kimbundu-Portuguese pairs (Apache-2.0 label, provenance "books, religious websites, news") — Portuguese source, needs Pt->En by a licensed teacher** [S33]. NLLB-mined sets exist for all (dyu 286k, kbp 1.59M, kam 1.63M, umb 298k, kmb 196k) = class U, noisy, and FineWeb-2 text for these (tiv/kng/kmb/umb/bin) is 84-98% jw.org in the first 300 documents.
4. **fon.** FLORES dev 897 + wikimedia 112 (SA). **French-pivot only:** Zenodo 4266935 French->Fongbe / French->Ewe, CC-BY-4.0, "70k+" pairs over the two files (~35k each, unverified) [S41]; OPUS FFR fr-fon 82k (licence unstated, U). Fon already has 34M SFT tokens.
5. **nus.** TICO-19 en-nus 3,070 (P, CC0, previously noted as eval-only and skipped by R10), Tatoeba 110 (>1,000 via indirect links on tatoeba.org) [S03, S11]; FLORES dev. NaathNLP MIT-labelled Nuer-Dinka sets are templated, LLM-looking triples with no card: U [X08]. 9.4M tokens exist already.
6. **mos, vmw, kea, kab, wol, ewe, tum (UI + wiki translation + Tatoeba):**
   * **kab:** OPUS Tatoeba en-kab 58,674 pairs, P (CC BY 2.0 FR), 0.65M tokens — **the only permissive human block above 50k for a W1-weak language, but R10 never checked it against the pool and `tatoeba_mt_parquet` is in the SFT catalogue: measure before use** [S10]; translatewiki 11,992 UI [S09]; wikimedia 1,136 SA [S08].
   * **kea:** translatewiki 7,320 UI (P\*); frgr3618 2,000 manually revised pairs (ODC-By; upstream NLLB-mined, JW-like) [S32]; FT kea already used (1.44M). Kreyol-MT public kea-eng contains FLORES-200 and NLLB parts: do not use [X09].
   * **mos:** translatewiki 5,952 UI; burkimbia sets are gated and NLLB-derived [X10].
   * **tum:** wikimedia 1,757 (SA), translatewiki 1,227. **ewe:** wikimedia 899 (SA), translatewiki 1,343, French pivot above; ghananlp sets NC. **wol:** translatewiki 1,700, wikimedia 226 (SA); bible-uedin 15.8k is a CC0-labelled Bible compilation (U); XLEnt are entity pairs, not sentences. **vmw:** translatewiki 610.
   * **yor:** wikimedia 40,048 pairs (SA) — available but pointless at 145M SFT tokens.
7. **luo.** FineTranslations luo_Latn 2,692 documents (~9k chunk pairs, ~3M tokens, radio sites + 15% jw.org), P\* [S06]; eBible `luo` (Biblica 2020, full Bible, CC-BY-SA-4.0, ~31k verses, SA) probably already in the SFT via biblenlp [X13]; **U: Dholuo-Kiswahili-English 29,230 pairs with three conflicting licences** [S31]. Luo has 17M tokens.
8. **efi, ibb, urh.** efi: translatewiki 539; IbomMT NC; QED research-only; the English-Efik corpus (13,865 pairs, arXiv 2603.14873) has no release. ibb: translatewiki 847; IbomMT NC. urh: nothing found (the repo's Urhobo silver set is the only data).
9. **knc (not a model language).** Gamayun kau-eng 5,000 (P) [S02]; TICO en-kr 3,070 (P) [S03]; FT knc_Latn 220 documents (P\*) [S04]; Tatoeba 43 (~200 indirect); OLDI Seed knc_Latn 6,193 (SA) + knc_Arab 6,193 (SA, Ajami script) [S20]; FLORES+ dev knc. Monolingual only: CLEAR-Global kanuri-books-corpus (10,281 sentences, CC-BY with author copyright, gated terms), FW2 knc 221 documents. ~15.8k clean pairs (+6.2k Ajami), ~1M tokens.
10. **run (Kirundi).** FT run_Latn 74,480 news documents (VOA Kirundi, igihe.bi, BBC, indundi.com; jw.org only ~4%; est. 5-6M target tokens at R10's row cap), P\*, usage unmeasured [S05]; Tatoeba 1,706 (P) [S11]; FLORES dev; the eligapris Kirundi sets (English-Kirundi CSV gated, Bible 32,725 verses) have no licence (U).
11. **ish, bin, oke, tsc, tiv, bbj, kbp, umb (nothing new):** see section 5.

## 4. Per-language totals, fetch method and download size

NEW pairs by class (human, excluding what R10 built). "UI" = translatewiki strings. "FT" = synthetic-English FineTranslations, estimate of rows. "Pivot" = non-English source. Download MB excludes FT unless noted. **API-filterable** = Hub `load_dataset`/`hf_hub_download` that fetches only the language's config or file; **bulk** = OPUS/Zenodo/GitHub archives (all small, one pair per file).

| iso | human P | UI | SA (incl. FLORES dev) | FT (synthetic EN) | pivot | est. target tokens (human+UI+SA) | reaches 100k? | fetch | download |
|---|---|---|---|---|---|---|---|---|---|
| tir | 11,251 | 4,776 | 5,897 | ~50k | - | 0.9M | no (yes with FT) | API: Gamayun file, FLORES file, FT config (streaming, take); bulk: TICO zip, translatewiki zip, Travis GitHub raw range | ~4 MB; FT up to 670 MB (50k docs ~250 MB) |
| amh | 3,287 | 3,166 | 1,197 | >100k | - | 0.4M | only via FT | API: FLORES, FT config (4.1 GB total; stream); bulk: TICO original release, OPUS zips | ~8 MB + FT stream |
| dyu | 0 | 0 | 897 | - | - | 0.06M | no | API: FLORES file | 0.5 MB |
| kbp | 0 | 625 | 897 | - | - | 0.07M | no | API: FLORES; bulk: translatewiki | 0.7 MB |
| fon | 0 | 0 | 1,009 | - | ~35k | 0.07M | no | API: FLORES; bulk: wikimedia, Zenodo 4266935 | 4 MB |
| nus | 3,180 | 0 | 897 | - | - | 0.31M | no | bulk: TICO zip; API: FLORES | 1 MB |
| mos | 0 | 5,952 | 897 | - | - | 0.2M | no | bulk: translatewiki; API: FLORES | 1 MB |
| kam | 10 | 0 | 897 | - | - | 0.06M | no | API: FLORES | 0.5 MB |
| bam | 0 | 0 | 897 | - | - | 0.05M | no | API: FLORES | 0.5 MB |
| umb | 32 | 0 | 897 | - | - | 0.06M | no | API: FLORES | 0.5 MB |
| kea | 0 | 7,320 | 897 | - | - | 0.2M (+2,000 flagged ODC-By) | no | bulk: translatewiki, frgr3618 GitHub raw; API: FLORES | 1 MB |
| kmb | 0 | 0 | 897 | - | 18,100 (Pt) | 0.06M | no | API: FLORES, KmbPtMT csv | 3 MB |
| vmw | 0 | 610 | 897 | - | - | 0.05M | no | bulk: translatewiki; API: FLORES | 0.6 MB |
| kab | 58,674 | 11,992 | 2,033 | - | - | 0.9M | no (73k) | bulk: Tatoeba, translatewiki, wikimedia zips; API: FLORES | 4 MB |
| seh | 0 | 0 | 897 | - | - | 0.05M | no | API: FLORES | 0.5 MB |
| yor | 176 | 5,995 | 40,945 | - | - | 2.9M | no (47k, not needed) | bulk: wikimedia en-yo zip; API: FLORES | 6 MB |
| wol | 0 | 1,700 | 1,123 | - | - | 0.12M | no | bulk: translatewiki, wikimedia; API: FLORES | 1 MB |
| ewe | 0 | 1,343 | 1,796 | - | ~35k | 0.15M | no | bulk: wikimedia, translatewiki, Zenodo 4266935; API: FLORES | 4 MB |
| luo | 0 | 0 | 31,897 (eBible) | ~9k | - | 1.2M | no | API: FT config (22 MB), FLORES; bulk: eBible zip | ~30 MB |
| tum | 19 | 1,227 | 2,654 | - | - | 0.2M | no | bulk: wikimedia (28 KB), translatewiki; API: FLORES | 1 MB |
| efi | 0 | 539 | 0 | - | - | 0.01M | no | bulk: translatewiki | 0.1 MB |
| ibb | 0 | 847 | 0 | - | - | 0.02M | no | bulk: translatewiki | 0.1 MB |
| urh | 0 | 0 | 0 | - | - | 0 | no | - | - |
| knc | 8,113 | 597 | 7,090 (+6,193 Ajami) | 760 | - | 1.0M | no | API: Gamayun file, OLDI files, FLORES file, FT config (0.7 MB); bulk: TICO zip | 8 MB |
| run | 1,706 | 902 | 897 | ~20k | - | 0.1M | only via FT | API: FT config (349 MB; stream 20-30k docs ~120 MB), FLORES; bulk: Tatoeba zip | 2 MB + FT stream |

Exact commands (all URLs verified on 2026-10-26): OPUS `https://object.pouta.csc.fi/OPUS-<corpus>/<version>/moses/<src>-<tgt>.txt.zip` (get the current version and code from `https://opus.nlpl.eu/opusapi?source=en&target=<code>`; codes: ti, am, kea, kr (TICO) / knc (Tatoeba), nus, rn, mos, vmw, kab, wo, ee, tum, ibb, efi, kg); Gamayun `hf_hub_download('CLEAR-Global/Gamayun-kits','parallel/tig-eng/kit5k.tsv', repo_type='dataset')`; OLDI `seed/knc_Latn.jsonl` + `seed/eng_Latn.jsonl`, join on `id`; FineTranslations `load_dataset('HuggingFaceFW/finetranslations','run_Latn',split='train',streaming=True)`; FLORES+ `dev/<iso>_<Script>.jsonl` with the box token, then drop the ids in `reports/data/weak_selection_2026-10-26/sample_ids.json`. Filtering needed on the box: pool match against the SFT windows (R10 audit pipeline), GlotLID, hold-out filter against the English sides of FLORES+ devtest, NTREX, MAFAND, SmolSent/SmolDoc, SALT test, AfriScience dev/test, and W1's 100 validation sentences per language.

## 5. Languages where nothing usable exists

Nothing new, permissive and human beyond the FLORES+ dev (or nothing at all): **ish, bin, oke, tsc, tiv** (only JW/Bible in the SFT; FineWeb-2 for tiv 2,160 documents, bin 812, tsc 397, ish 46, all jw.org), **urh, ibb, efi** (UI strings < 1k), **dyu, kam, bam, umb, seh, vmw, kbp** (< 1k outside the FLORES dev), **kmb** (Portuguese only), **bbj** (Bible: english_ghomala 7,916 KJV pairs without a licence, flagship-ai New Testament with publisher rights, French dictionary; probably the Bible slice already in the SFT), **dua, bas, wes, ndc, yao, iso, her, kua** (FT already used where it existed). For these the options are the owner's: paid translation of an English seed set (a Gamayun-style 5,000-10,000 sentences per language is the proven format), the teacher route (R14), or the draw weight alone. Cheapest high-value targets by W1 rank: tir, dyu, kbp, nus, mos, kam, bam, umb, kmb, vmw, seh.

## 6. Science / long-document sources (AfriScience-MT has amh hau lug nso yor zul only)

* **FineTranslations documents** (`translated_chunks`/`og_chunks` per document): real long Ge'ez/Latin target documents with synthetic English for tir (72k), run (74k), amh, luo, kea, knc — the only long-document material for tir and run. News/web, not science; English side synthetic.
* **OLDI Seed** knc/nus/bam: sentences are ordered per Wikipedia article (~51 sentences per article, ~120 articles; film/biography), so 10-60-sentence pseudo-documents (b256-b1024) for CC-BY-SA. Same English for all three languages.
* **TICO-19** (CC0) knc, nus, tir, amh: COVID/health, news and Wikipedia text, original release keeps document structure (OPUS flattens it; unverified).
* **No permissive document-level science translation exists for tir or any of the 23 other languages.** HornMT is sentence-level news with no licence; AfriDocMT is NC. For RL the English science prompts are the owner's existing set; references for tir would need commissioned translation of CC-BY open-access abstracts (PLOS, eLife, PMC OA CC-BY subset) in the AfriScience-MT format.
* Wikipedia (CC-BY-SA, monolingual) for consensus/anchor targets: am 15.7k articles, dag 14.5k, fon 5.6k, kg 2.0k, wo 1.7k, kbp 1.7k, ee 1.4k, bm 0.9k, rn 0.7k, ti 0.4k.

## 7. Monolingual target-side corpora (later consensus/anchor use)

FineWeb-2 (ODC-By): tir 72.8k documents, run 75.4k, amh 432k, luo 2.7k, kea 2.9k, kam 1.5k, tiv 2.2k, kng 2.2k, dyu 2.8k, kbp 1.3k, kmb 1.3k, umb 1.3k, bin 812, knc 221, nus 131, ish 46; JW-dominated for tiv/kng/kmb/umb/bin. Wikipedia as above. Common Voice sentences ti 4,519 and am (CC0). CLEAR-Global kanuri-books-corpus. Details in `sources.csv` X14, S12, S22.

## 8. Leakage check (English sides)

`leakage_results.csv`: Gamayun tig 0/200, Gamayun kau 0/200, OLDI English 0/460, TICO first/last 300: 0 exact, 1 and 0 8-gram hits, Tatoeba rn 1 exact/300 (0.3%), HornMT 0/300, FineTranslations run 0/550, luo 0/1,195, knc 0/714, kea 1 exact/1,089. Pool: NTREX (1,997), SmolSent + SmolDoc English for am ti rn kg kr luo tiv nus, W1 dev sample (200 sentences, i.e. the held-out validation text). Not covered: FLORES+ devtest/dev in full, SALT test, MAFAND, AfriScience (gated or unreachable here); HornMT's English looks like Wikinews, a FLORES source, so treat it as risky until checked on the box.

## 9. Open questions for the owner

1. **FineTranslations as the only route to volume (tir, amh, run, luo).** English side is Gemma-3-27B output: do the Gemma Terms of Use matter for a commercial model, and is "web text of unknown copyright under ODC-By + CommonCrawl terms" acceptable (it was accepted for 11 languages in R10)?
2. **Travis Foundation Tigrinya:** use only the first ~5k crowd-sourced pairs (my split is approximate), or skip because the repo also hosts JW text?
3. **TICO-19 and FLORES+ dev:** train on them (CC0 / CC-BY-SA, FLORES+ dev allowed), or keep TICO as the extra external test for knc, nus, tir, amh?
4. **Kanuri:** is knc to be added as a language (tag, tokenizer for the Ajami script)? It is the only language where this search changes the picture from nothing to ~1M tokens.
5. **Pivot sources** (Fon/Ewe French 35k each, Kimbundu Portuguese 18k): allow Fr/Pt -> En by a licensed teacher to create English sources?
6. **Kab Tatoeba (58k):** run the pool-match first on the box; if unused it is the single largest clean block.
7. **Paid translation** for the languages in section 5: which ones, and which seed English set (Gamayun core v2 30k English sentences is CC-BY-4.0 and already curated; 5,000 per language is its standard kit).
8. Merge status of R10's `sft_extra_v1` into the continuation mix is unverified from the laptop; the "have" column assumes it may or may not be merged.
