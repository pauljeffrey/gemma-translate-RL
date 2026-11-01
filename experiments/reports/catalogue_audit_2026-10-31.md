# Catalogue audit: what the African HF catalogue could still add to the clean SFT build (2026-10-31)

Metadata only. No dataset was downloaded. Sources used: the catalogue CSVs (`data-catalog/african_hf/{sft,rl}/`, 7,884 dataset-language rows, 2,747 distinct repos), Hub API `dataset_info` with file listings for 1,018 candidate repos, the datasets-server `size` endpoint (row counts per config) for the same repos, and `first-rows` (3 rows each) for 7 repos whose schema was unclear (Vast-Urdu, Emulated-Inc pool, Inioluwa Nupe set, NaathNLP Nuer-Dinka, Svngoku central-africa, Tommy0201 JW300 Igbo, kweku2088 Twi). The Hub token was read from `.env` and never printed or stored.
Data table: `reports/data/catalogue_audit_2026-10-31.csv` (5,348 rows, 2,179 datasets, one row per dataset x language).

## 1. Method and the "already used" set

**Already used (set size).** Confirmed from the Hub repo `<HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28` (`rebuild/meta/` = the 181 finished input units, `specialist/stats/`, `stats/per_source.csv`), `data_pipeline/sft_clean/build_pairs.py`, `chain/extra_sources.py` and `reports/language_data_hunt_2026-10-29.md`:
* **104 Hub dataset repos** named as used by the clean pairs build (68 mt560 + 36 others): 68 `michsethowusu/english-*_sentence-pairs_mt560` repos (of the 99 that exist; see below), the other public pair sources (multi-open, finetranslations, OPUS-100 via `sentence-transformers`, KwekuAinoo/eng_twi, AfroLingu-MT, BrianMwangi luo, mamakobe luhya, two Wolof sets, Helsinki-en-fon, Pontoon, Vukuzenzele, SSA-MTE train, IbomMT, Davlan ibom-mt-en-efi, and the two quarantined sets pristine-twi / synthetic-external), the specialist set sources (AfriDocMT, AfricanFineTranslations, African-Code-Switch, polynews-parallel, oldi_seed, Gamayun-kits, the weak-language and R10 backup repos) and the Amendment-6 extras (eBible nde, OPUS wikimedia/XLEnt en-ff, `stephanedonna/english_fulfulde`, `michsethowusu/english-fulah_sentence-pairs`, kanurimt, NLLB-Seed tzm, Pontoon tzm).
* **Instruction / task side:** `<HF_ORG>/sft-new` and `<HF_ORG_LEGACY>/sft-all` (sft-new card: swim-ir (swh, yor), aya_dataset new-language rows, NCHLT, Universal Dependencies, Injongo, WolBanking77, lr-sum) plus **62 more sources** named in the old `final_v2` inventory / `data_pipeline/sft_new` code (xP3 family, muri-it split, alpaca-ewe, OpenHermes, glaive, big_math, EthioEmo, tswana-emotions, mich_* corpora ...). These 62 were in the OLD build; the clean rebuild is pairs-only, and `final_v2` itself is not on the Hub.
* **Catalogue overlap:** 89 of the 104 and 44 of the 62 appear in the catalogue; they are class `mirror_of_used` (105 datasets incl. re-uploads and copies), not "new".
* **Old-build conversion of catalogue datasets (inferred):** the old build converted catalogue datasets in rounds r1-r5 with `data_pipeline/sft_new/catalog_plan.py` (only the r4/r5 plans were saved). I re-ran its `decide()` logic on the catalogue: datasets with verdict `convert:<shape>` were almost certainly converted in the old build (capped per repo) but are NOT in the clean pairs rebuild. They stay `new_*` here and are labelled "old build (inferred)" in the CSV (`already_used_equivalent`). **196 candidate datasets / 11.0M rows were never converted by any build** (table in section 3).
* **Languages already covered:** 76 tags in the main pairs output + nde, ful, knc, tzm (Amendment 6) + the 20 weak languages of the specialist set (current counts are max(main, specialist)).

**Classification (heuristic, applied in this order):** protected (benchmark by id/name: FLORES, NTREX, SMOL/GATITOS, MAFAND, TICO, SIB-200, SALT, GlotStoryBook, AfriDocMT, IrokoBench-family, masakhanews/afriqa/masakhapos, FLEURS-derived, wmt24pp, any `eval/test/dev/blindspot` name) -> NLLB/TranslatePsy (ids with nllb/ccmatrix/ccaligned/translatepsy, the 759-repo unsuffixed `michsethowusu/<l>-<l>_sentence-pairs` family that the old audit showed to be NLLB/CCMatrix with a `score` column, Aya collections, any card/hints mentioning NLLB) -> mirror/copy of a used set -> non_commercial (catalogue `commercial_use`) -> new_parallel / new_instruction / other by catalogue task tags. Row counts: per-language catalogue counts, replaced by Hub config sizes where config names are language pairs (57 new_parallel/new_instruction datasets); repos whose `num_rows` is the whole multi-language repo with no per-language config are marked "whole-repo" and are NOT summed into the verified totals. Re-uploads with identical (rows, languages, task) are counted once (`duplicate/re-upload of` in notes; 103 datasets).
Licence rule: commercial_ok and share_alike (disclose) and unknown (no licence declared, disclose) are listed in the main tables; non_commercial only in section 5. `synthetic_or_mt` means a keyword (machine translation, synthetic, translated, LLM, generated ...) appears in the card: precision is low (e.g. DocHPLT and Tatoeba cards mention MT).

## 2. Headline numbers (gross rows, before dedupe against the clean pairs and between datasets)

Cells show rows (number of datasets). Only datasets with usable licence; duplicates counted once; Wikipedia mirrors and pure speech/pretraining repos excluded. "verified" = single-language repo, per-language config count, or Hub config sizes; "whole-repo" = repo-level count of a multi-language repo whose African share could not be isolated without reading rows (not comparable, shown separately).

| bucket | reliability of row count | commercial_ok | share_alike | unknown (disclose) | total |
|---|---|---:|---:|---:|---:|
| parallel en<->X | verified | 53.68M (57) | 164.2k (4) | 6.06M (40) | 59.90M (101) |
| parallel en<->X | whole-repo | 1.74M (7) | 0 (0) | 116.0k (6) | 1.86M (13) |
| parallel fr<->X (no English) | verified | 56.7k (6) | 169.7k (3) | 2.68M (29) | 2.90M (38) |
| parallel fr<->X (no English) | whole-repo | 10.0k (1) | 0 (0) | 0 (1) | 10.0k (2) |
| parallel X<->X / unknown pivot | verified | 506.0k (33) | 120.6k (5) | 1.44M (57) | 2.06M (95) |
| parallel X<->X / unknown pivot | whole-repo | 12.85M (16) | 28.4k (1) | 201.0k (7) | 13.08M (24) |
| instruction/chat/QA/summ | verified | 83.13M (147) | 2.27M (4) | 11.27M (96) | 96.68M (247) |
| instruction/chat/QA/summ | whole-repo | 20.60M (36) | 6.1k (1) | 34.76M (10) | 55.36M (47) |
| task/label (classification, NER, headline ...) | verified | 42.41M (102) | 135.3k (9) | 7.47M (60) | 50.01M (171) |
| task/label (classification, NER, headline ...) | whole-repo | 1.93M (21) | 0 (1) | 238.2k (4) | 2.16M (26) |

Synthetic / machine-translated share inside the verified rows:

| bucket (verified/single-language rows only) | not flagged | flagged synthetic_or_mt (keyword in card) |
|---|---:|---:|
| instruction/chat/QA/summ | 62.01M | 34.67M |
| parallel X<->X / unknown pivot | 1.71M | 0.35M |
| parallel en<->X | 4.16M | 55.74M |
| parallel fr<->X (no English) | 2.72M | 0.19M |
| task/label (classification, NER, headline ...) | 21.43M | 28.59M |

Reading guide: the verified en<->X parallel total is dominated by one synthetic set (`michsethowusu/twi-english-parallel-synthetic-50m`, 47.9M, old build already took a QE-selected part). **Without repos carrying a synthetic/MT keyword the verified en<->X pool is about 4.2M gross rows; the flagged 55.7M are 47.9M synthetic Twi, 2.3M DocHPLT document pairs (flagged only because its card mentions MT) and smaller sets. On top of that come 2.54M rows in 30 mt560 repos that are not in the catalogue (row 1 of section 3).** Expect a large part of the Bible/JW-style rows to duplicate text already in mt560/multi-open (no overlap measurement was possible without reading rows). Class counts over the 2,747 audited repos: new_parallel 282, new_instruction 331, mirror_of_used 105, non_commercial 269, NLLB/TranslatePsy-excluded 792, protected 47, other (label tasks, demoted non-pairs, speech/pretraining-only) 921. Streamability of the 613 new_parallel/new_instruction repos checked on the Hub: yes 458, no 10, unknown 145 (48 are gated and need access approval first).

## 3. Top 40 candidates ranked by new usable rows (en<->X parallel first, then instruction / chat / task)

Parallel entries are ranked by verified rows (synthetic last); instruction entries are listed after them, ranked by rows. "Rows" are gross rows for the languages listed (see section 2 caveats). `[old build ...]` = the old final_v2 used or probably used it (confirmed = named in the inventory/code, inferred = catalog_plan verdict), but it is not in the clean pairs rebuild. Streamable = parquet/jsonl/csv with a plain text column and a working dataset viewer (checked with the Hub file listing and datasets-server); the repo is read with `load_dataset(..., streaming=True)` and the listed configs.

| # | dataset | languages | rows | licence | why new | caveats | streamable |
|---:|---|---|---:|---|---|---|---|
| 1 | 30 x `michsethowusu/english-<language>_sentence-pairs_mt560` (NOT in the catalogue; found on the Hub; a 31st, egyptian-arabic 54.6k, is catalogued) | alz,bum,cjk,cce,chw+25 | 2.54M | cc-by-4.0 (commercial_ok per card) | OPUS MT560 / JW-derived en pairs for languages the clean build has no tag for (Gun 287k, Luvale 317k, Tetela 222k, Luba-Katanga 197k, Zande 190k, Seychelles creole 188k, Morisyen 182k, Ruund 134k, Nyaneka 116k, Ndonga 100k, Kwangali 107k ...) | JW religious domain, no new tags exist; each language needs a tokenizer tag first (token-swap policy); MT560 is not NLLB per its cards | yes (parquet) |
| 2 | `HPLT/DocHPLT` | afr,swh,xho | 2.30M | cc0-1.0 (commercial_ok) | only doc-level en pairs for afr/swh/xho at this scale; web-crawled | document pairs, not sentences; sentence-split needed; stream 3 configs of a 511 GB repo [synthetic/MT keyword] | yes |
| 3 | `Mehgoss/sa-languages-translation` | afr,nbl,nso,sot,ssw,tsn+4 | 1.07M | cc-by-4.0 (commercial_ok) | 10 SA languages x English (JW + Vukuzenzele + Autshumato), old build only | overlaps mt560 and Vukuzenzele (dedupe); JW religious domain [old build (confirmed)] | yes |
| 4 | `NaathNLP/Nuer_Dinka_Parallel_translation_pairs` | din,nus | 899.4k | mit (commercial_ok) | nus and din: the clean build has 0 and 7k pairs | templated look, provenance unstated, MIT label; sample first [old build (inferred)] | yes |
| 5 | `nwu-ctext/autshumato` | tsn,tso,zul | 833.5k | cc-by-2.5 (commercial_ok) | human government translation tsn/tso/zul | NOT streamable (loading script); CC-BY 2.5; partly inside Mehgoss [synthetic/MT keyword; old build (inferred)] | no |
| 6 | `Tamazight-NLP/Weblate-Translations` | aar,ach,aeb,afr,amh,arz+46 | 721.4k | other (unknown) | UI strings for about 50 African languages incl. weak ones (bam, ewe, ful, her, kab, ...) | short strings with placeholders; per-string licences (GPL etc.) mixed [old build (inferred)] | yes |
| 7 | `iamwille/igbo_language_translation_dataset` | ibo | 673.7k (+526k overlap) | mit (commercial_ok) | Igbo is at 21% of old tokens | same text as ccibeekeoc42/english_to_igbo (count once); card is an empty template [synthetic/MT keyword; old build (inferred)] | yes |
| 8 | `Tommy0201/JW300_Igbo_To_Eng` | ibo | 671.1k (5 repos) | unspecified (unknown) | public copies of the unrecoverable naija2 Igbo bundles (JW300, WebScraped, IGBONLP, Amazon): +672k ibo with the 4 siblings | JW300 + scraped news, no licence, siblings are duplicates of each other (igbo-to-english = igbo_to_english_split) [old build (inferred)] | yes |
| 9 | `kweku2088/twi_to_english` | twi | 561.2k | unspecified (unknown) | Twi is at 14% of old tokens | JW-style religious text, no licence [old build (inferred)] | yes |
| 10 | `Inioluwa/nigerianLanguageTranslator` | yor | 288.9k | cc (unknown) | Nupe (nup) is not in the build | English-NUPE (tagged yor in the catalogue); many word-level rows [old build (inferred)] | yes |
| 11 | `GiJoeHansFranz/Liseli` | bem,kqn,loz,lun,nya,toi | 277.3k | other (per-source) (unknown) | Zambian languages bem/nya/toi/loz/lun/kqn with English | word/phrase level, status=unverified, mixed licences [synthetic/MT keyword; old build (inferred)] | yes |
| 12 | `silvanosolutions/xhosa-nlp-dataset` | xho | 217.5k | other (unknown) | xho pair rows (xhosa/english columns) beyond the 1.39M already used | mixed with Glot500 monolingual rows (type column); no licence; xho is already the 3rd largest language [old build (inferred)] | yes |
| 13 | `Rogendo/English-Swahili-Sentence-Pairs` | swh | 210.5k | unspecified (unknown) | swh phrase pairs | very short phrases; swh already has 900k pairs: low value [old build (inferred)] | yes |
| 14 | `quranlab/quran` | aar,afr,amh,bam,dag,ful+15 | 174.6k | other (mixed-per-config) (unknown) | Quran translations in 21 African languages | translation licences (Tanzil-style) are non-commercial in the specialist build; verify [old build (inferred)] | yes |
| 15 | `DigitalUmuganda/NMT_Rwandan-Gazette_parallel_data_en_kin` | kin | 170.3k | cc (unknown) | kin legal/gazette domain | garbled English side in places; kin already 170k pairs [synthetic/MT keyword; old build (inferred)] | yes |
| 16 | `DigitalLearningGmbH/tatoeba_mt_parquet` | afr,ber,kab,run,swa | 150.4k | cc-by-2.0 (commercial_ok) | candidate to restore the Tatoeba-MT en-ber 119k / en-kab 26.6k / en-afr / en-run rows lost with the private bundles | ber_Latn configs duplicate ber; unclear whether this is train or test/validation Tatoeba-MT [synthetic/MT keyword; old build (inferred)] | yes |
| 17 | `dsfsi/govza-sa-cabinet-statements-sentence-aligned` | afr,nbl,nso,sot,ssw,tsn+4 | 145.0k | cc-by-4.0 (commercial_ok) | SA government statements for 10 SA languages; old build only (not in clean rebuild) | identical-line pairs, alignment-score threshold needed; overlaps Vukuzenzele-type text [synthetic/MT keyword; old build (confirmed)] | yes |
| 18 | `abdelhaqueidali/Kabyle-Latin-to-Tifinagh-Parallel-Corpus` | ber,kab | 135.1k | cc-by-2.0 (commercial_ok) | Tatoeba-like kab-en; kab has only 37k pairs now | Tifinagh side is rule-based transliteration; overlaps Tatoeba rows already in the specialist set [synthetic/MT keyword; old build (inferred)] | yes |
| 19 | `pkyoyetera/luganda_english_dataset` | lug | 128.2k (2 repos) | apache-2.0 (commercial_ok) | lug human-looking sentences (+ kambale/luganda-english-parallel-corpus 50k) | quality unchecked [old build (inferred)] | yes |
| 20 | `taqbaylit/kabyle-english-TM` | kab | 121.7k | cc-by-4.0 (commercial_ok) | kab-en localisation memory | UI strings, auto-pipeline [old build (inferred)] | yes |
| 21 | `FrancophonIA/Kabyle-French-pairs` | kab | 115.3k | unspecified (unknown) | fr-kab Tatoeba-like pairs (pivot via French only) | not English; no licence [old build (inferred)] | yes |
| 22 | `mteb/biblenlp-corpus-mmteb` | bba,ewe,fuh,hau,kik,lin+12 | 104.5k | cc-by-sa-4.0 (share_alike) | Bible en-X verse pairs for ewe/hau/lin/lug/luo/nya/som/swh/twi/wol/yor and more (public copy of the lost <HF_ORG>/biblenlp-corpus-mmteb) | Bible domain, CC-BY-SA 4.0 (share-alike); overlaps JW-style data in spirit, not in text [old build (inferred)] | yes |
| 23 | `galsenai/centralized_wolof_french_translation_data` | wol | 98.3k | unspecified (unknown) | wol-fr (old build had it as an unconverted source) | French pivot; collects other HF sets (dedupe); no licence [old build (confirmed)] | yes |
| 24 | `Helsinki-NLP/opus_infopankki` | som | 47.2k | cc-by-4.0 (commercial_ok) | en-so 47k, CC-BY 4.0, clean domain text | only one relevant config [synthetic/MT keyword; old build (inferred)] | yes |
| 25 | `dmatekenya/chichewa-machine-translation` | nya | 14.0k | apache-2.0 (commercial_ok) | human en-nya topic sentences (14k, CC-BY) | small [old build (inferred)] | yes |
| 26 | `michsethowusu/twi-english-parallel-synthetic-50m` | twi | 47.92M | cc-by-4.0 (commercial_ok) | LLM-paraphrase synthetic twi->en 47.9M | synthetic; the old build already took a QE-selected part; lowest priority [synthetic/MT keyword; old build (confirmed)] | yes |
| 27 | `cidjeu/french-moore-parallel` | mos | 3.00M | other (unknown) | fr->mos 3.0M web-crawl + Glosbe MT | French pivot and machine-translated target (synthetic); mos already has 181k human pairs [synthetic/MT keyword; old build (inferred)] | yes |
| 28 | `Mwanzau/Tumbuka_Instruction_Tuning` | tum | 2.76M (2 repos) | apache-2.0 (commercial_ok) | tum instruction data (2.0M), commercial_ok; plus Mwanzau/Tumbuka_language 739k | template-like prompts; check duplication; Tumbuka already 230k pairs [old build (inferred)] | yes |
| 29 | `akoksal/muri-it` | afr,amh,bam,din,ewe,fon+30 | 2.23M | apache-2.0 (commercial_ok) | multilingual reverse-instruction data in 36 African languages (Apache) | num_rows is whole repo; per-language split is already in the old build [synthetic/MT keyword; old build (inferred)] | yes |
| 30 | `oza75/bambara-lm-qa` | bam | 998.1k | unspecified (unknown) | Bambara task QA (bam is a weak language) | LM-style cloze/punctuation tasks; no licence [old build (inferred)] | yes |
| 31 | `YoseAli/amharic-llm-training-data` | amh | 846.1k | mit (commercial_ok) | Amharic instruction/chat mix | provenance of components unstated [old build (inferred)] | yes |
| 32 | `MaLA-LM/MassiveSumm_short` | afr,amh,bam,ful,hau,ibo+12 | 198.0k | unspecified (unknown) | summaries in 18 African languages | crawled news, no licence; copyright risk [old build (inferred)] | unknown |
| 33 | `yacdev/somali-100k-saaxiib-conversations-v2` | som | 100.0k | apache-2.0 (commercial_ok) | Somali conversational data | probably synthetic [old build (inferred)] | yes |
| 34 | `burtugeey/alpaca_somali_final` | som | 96.3k | apache-2.0 (commercial_ok) | Somali Alpaca | translated Alpaca; sibling Somali_alpaca is MT [old build (inferred)] | yes |
| 35 | `addisai/FineTome-single-turn-dedup-amharic` | amh | 83.3k | apache-2.0 (commercial_ok) | Amharic instruction (83k) | machine-translated FineTome [old build (inferred)] | yes |
| 36 | `sartifyllc/pawa-function-calling-60k` | swh | 60.0k | unspecified (unknown) | Swahili function-calling chats (matches the tool-call SFT) | no licence; format conversion needed [old build (inferred)] | yes |
| 37 | `masakhane/african-ultrachat` | amh,hau,ibo,kin,sna,som+5 | 55.0k | mit (commercial_ok) | chat in 11 African languages, commercial_ok | small (55k) [old build (inferred)] | yes |
| 38 | `africa-intelligence/alpaca-zulu` | zul | 52.0k | mit (commercial_ok) | Zulu Alpaca | translated Alpaca | yes |
| 39 | `ik/twi_alpaca-cleaned` | twi | 51.7k | unspecified (unknown) | Twi Alpaca (old build had ewe, not twi) | translated Alpaca, no licence [old build (inferred)] | yes |
| 40 | `vamboai/fikira` | aeb,amh,hau,ibo,kin,sna+4 | 50.0k | mit (commercial_ok) | reasoning/QA in 10 African languages | duplicate DKSibbs/fikira; in old r5 build [synthetic/MT keyword; old build (inferred)] | yes |

Notes on the table: (a) the 31 mt560 repos are the largest clean find: the clean build used 68 of the 99 mt560 repos because the other 31 languages are not in its language table, and the catalogue crawl missed them too (only egyptian-arabic is catalogued). (b) `Tommy0201/*` are public copies of private bundles that the rebuild lost (reachability table: JW300_Igbo_To_Eng, WebScraped_Igbo_To_Eng, IGBONLP, Amazon translator); Igbo is the language with the largest coverage gap after Twi/Fon (21% of old tokens). (c) `DigitalLearningGmbH/tatoeba_mt_parquet` is the only candidate that may restore part of the lost Tatoeba-MT train (10.9M rows in the old build): its eng-ber 119k and eng-kab 26.6k are large for those languages, but the repo may hold the Tatoeba-MT test/validation part, which would be evaluation data; check the split names before use.

Candidates never touched by any SFT build (old or clean), top 14 of 196:

| dataset | kind | languages | rows | licence | note |
|---|---|---|---:|---|---|
| `HPLT/DocHPLT` | parallel (en) | afr,swh,xho | 2.30M | commercial_ok | synthetic/MT keyword;  |
| `abdukuzi45/amharic-sft-mixed-cot-2026` | instruction/chat/QA | amh | 1.68M | unknown | chat+CoT mix, uncensored persona, check safety |
| `nileagi/swahili-language-exposure` | instruction/chat/QA | swh | 1.67M | commercial_ok | monolingual/QA exposure text, closer to pretraining |
| `nileagi/swahili-language-exposure-v2` | instruction/chat/QA | swh | 1.64M | commercial_ok | monolingual exposure text |
| `abdukuzi45/amharic-sft-combined-2026-with-cot` | instruction/chat/QA | amh | 1.52M | commercial_ok | chat+CoT mix, check safety/persona |
| `Mwanzau/Tumbuka_language` | instruction/chat/QA | tum | 739.5k | commercial_ok | Gemma-chat-formatted Tumbuka tasks |
| `MBZUAI-Paris/Egyptian-DPO-Mixture` | instruction/chat/QA | arz | 298.1k | unknown | synthetic/MT keyword;  |
| `google-research-datasets/tydiqa` | instruction/chat/QA | swh | 240.5k | commercial_ok | TyDi QA (swh part); benchmark-adjacent, check protected overlap |
| `copenlu/answerable_tydiqa` | instruction/chat/QA | swh | 129.4k | commercial_ok | TyDi QA derivative (swh) |
| `r1char9/detox-multilingual-dataset` | instruction/chat/QA | amh | 110.0k | commercial_ok |  |
| `Rabe3/egyptian-arabic-tts-diacritized` | instruction/chat/QA | arz | 97.2k | unknown |  |
| `willwade/AACConversations` | instruction/chat/QA | afr | 86.0k | commercial_ok | synthetic/MT keyword;  |
| `jvonrad/PolyFact` | instruction/chat/QA | swh | 58.8k | share_alike | multilingual fact MCQ (swh) |
| `africa-intelligence/alpaca-zulu` | instruction/chat/QA | zul | 52.0k | commercial_ok | translated Alpaca |

## 4. Lowest-resource languages: what new data exists

Languages with fewer than 20,000 pairs in the current plan (main `per_language.csv`, specialist `per_language_summary.csv`, Amendment-6 counts; ibb 998, nbl 188, men 1,621, knc 5k, ish 5k, nus 7k, tzm 7k, her 8k, kbp 10k, bin 11k, luy 13k, dyu 15k, dua 15k, shi 15k, kea 15k, bam 19k, dag 0, bbj 0), plus the next lowest (oke, pcm, ndc) and four absent languages with catalogue data (fat, gaa, nup, tig). In each cell: gross rows from verified per-language counts (`n/a(repo)` = the repo is multi-language and the per-language count is unknown; `*` = synthetic/MT keyword; `[SA]` share-alike; `[no-lic]` no licence declared). Instruction / task cells include only labelled-task and instruction repos (xP3x counts are the catalogue's repeated whole-repo figure, treat as unreliable). NC / NLLB / protected counts are repos that mention the language but are not usable.

| iso | language | current pairs | new en<->X pair candidates (usable licence) | other parallel (fr / X<->X) | instruction / task rows | notes |
|---|---|---:|---|---|---|---|
| nbl | South Ndebele | 188 | 175.7k total from 5: `Mehgoss/sa-languages-translation` 127.7k; `dsfsi/govza-sa-cabinet-statements-sentence-aligned` 48.0k*; `dsfsi/za-mafoko-dsac` n/a(repo) [no-lic]; ... | 55 total from 1: `Tamazight-NLP/Weblate-Translations` 55 [no-lic] | 0 total from 4: `Lindo20/lwazi-asr-corpus-compressed` n/a(repo); `ODYSSEYAILABS/odyssey-sa-voice-v0.4-sample` n/a(repo)*; ... | NC datasets 3; NLLB-excluded 0; protected 4 |
| ibb | Ibibio | 998 | none | none | none | NC datasets 0; NLLB-excluded 0; protected 2 |
| men | Mende | 1,621 | none | none | none | NC datasets 0; NLLB-excluded 0; protected 0 |
| knc | Central Kanuri | 5,073 | none | 0 total from 1: `Svngoku/xP3x-Kongo` n/a(repo)* | 5.7k total from 4: `lbourdois/VQA-Marsouuu-central-banks-and-monetary-authorities-reports-clean` 5.7k [no-lic]; `CohereLabs/xP3x` n/a(repo); ... | NC datasets 1; NLLB-excluded 2; protected 0 |
| ish | Esan | 5,085 | none | none | none | NC datasets 0; NLLB-excluded 0; protected 0 |
| nus | Nuer | 7,105 | 5.0k total from 2: `dayomtechnologies/English_Nuer_parallel_translation_open_corpus` 5.0k [no-lic]; `NaathNLP/Nuer_Dinka_Parallel_translation_pairs` n/a(repo) | 0 total from 1: `Svngoku/xP3x-Kongo` n/a(repo)* | 12.5k total from 3: `michsethowusu/nuer-sentiments-corpus` 12.5k*; `CohereLabs/xP3x` n/a(repo); ... | NC datasets 10; NLLB-excluded 37; protected 1 |
| tzm | Central Atlas Tamazight | 7,183 | 0 total from 1: `Tamazight-NLP/Beni-Mellal-Tamazight` n/a(repo) [no-lic] | 8.9k total from 5: `Tamazight-NLP/Weblate-Translations` 3.3k [no-lic]; `Himanshu9636/Weblate-Translations` 2.3k [no-lic]; ... | 0 total from 2: `CohereLabs/xP3x` n/a(repo); `Muennighoff/xP3x-sample` n/a(repo) | NC datasets 0; NLLB-excluded 0; protected 0 |
| her | Herero | 7,941 | none | 15 total from 3: `Tamazight-NLP/Weblate-Translations` 9 [no-lic]; `Himanshu9636/Weblate-Translations` 3 [no-lic]; ... | none | NC datasets 0; NLLB-excluded 0; protected 1 |
| kbp | Kabiyè | 9,970 | none | 0 total from 1: `Svngoku/xP3x-Kongo` n/a(repo)* | 51.4k total from 8: `michsethowusu/kabiye-emotions-corpus` 22.8k*; `michsethowusu/kabiye-sentiments-corpus` 15.1k*; ... | NC datasets 5; NLLB-excluded 0; protected 0 |
| bin | Edo (Bini) | 11,230 | none | 0 total from 2: `VelkroLM/african-languages-filtered` n/a(repo) [no-lic]; `rufatronics/african-languages-filtered` n/a(repo) [no-lic] | none | NC datasets 0; NLLB-excluded 0; protected 0 |
| luy | Luyia | 12,899 | 6.2k total from 1: `nassimjp/quran` 6.2k [no-lic] | none | 4.4k total from 1: `Digital-Divide-Data/Luhya-ASR-Data-subset-642H` 4.4k | NC datasets 1; NLLB-excluded 1; protected 1 |
| dyu | Dyula | 14,714 | none | 0 total from 1: `Svngoku/xP3x-Kongo` n/a(repo)* | 36.4k total from 4: `michsethowusu/dyula-sentiments-corpus` 36.4k*; `CohereLabs/xP3x` n/a(repo); ... | NC datasets 0; NLLB-excluded 40; protected 2 |
| dua | Duala | 14,907 | none | 7 total from 3: `Tamazight-NLP/Weblate-Translations` 3 [no-lic]; `Himanshu9636/Weblate-Translations` 2 [no-lic]; ... | none | NC datasets 0; NLLB-excluded 0; protected 0 |
| shi | Tachelhit | 15,224 | 44.1k total from 1: `abdelhaqueidali/Tashelhit-Tifinagh-Latin-Parallel-Tatoeba-Dataset` 44.1k* | 1.6k total from 2: `Tamazight-NLP/translatewiki` 1.6k; `abdelhaqueidali/Amazigh-Quran-Translation-Jouhadi` n/a(repo) [no-lic] | 1.8k total from 6: `zabr946/Chatbot-Url` 1.8k [SA]; `HackAI-2025/tts_dataset_speaker1` n/a(repo) [no-lic]; ... | NC datasets 1; NLLB-excluded 0; protected 0 |
| kea | Kabuverdianu | 15,266 | none | none | none | NC datasets 0; NLLB-excluded 0; protected 0 |
| bam | Bambara | 19,098 | 105.7k total from 2: `MALIBA-AI/bambara-mt-dataset` 93.2k*; `nassimjp/quran` 12.5k [no-lic] | 383.0k total from 19: `RobotsMali/kunkado` 118.9k* [SA]; `RobotsMaliAI/bayelemabaga` 94.0k* [SA]; ... | 1.16M total from 16: `oza75/bambara-lm-qa` 998.1k [no-lic]; `Makan09/Bambara-dataset_conversation` 90.1k*; ... | NC datasets 7; NLLB-excluded 42; protected 6 |
| dag | Dagbani | 0 | 6.2k total from 1: `nassimjp/quran` 6.2k [no-lic] | none | 64.7k total from 4: `ghananlpcommunity/ghana-speech-lfn-phonemized` 37.9k [no-lic]; `PrinceAlhassanNasamu/tekyerema-pa-asr` 16.8k; ... | NC datasets 13; NLLB-excluded 0; protected 1 |
| bbj | Ghomala | 0 | none | none | none | NC datasets 0; NLLB-excluded 0; protected 0 |
| oke | Okpe | 21,869 | none | none | none | NC datasets 0; NLLB-excluded 0; protected 0 |
| pcm | Nigerian Pidgin | 22,504 | 23.2k total from 3: `Okwu/african-language-parallel-corpus` 20.9k; `coderGit/Eng-PidginBioData` 2.3k*; `Jesujuwon/farma-agri-yoruba-pidgin` n/a(repo) [no-lic] | 0 total from 6: `<HF_ORG_LEGACY>/data-gen` n/a(repo)*; `DOVEXAI/XIMA-2122` n/a(repo); ... | 33.8k total from 15: `bltlab/open-ner-core-types` 11.1k; `masakhane/masakhaner2` 8.1k; ... | NC datasets 3; NLLB-excluded 2; protected 7 |
| ndc | Ndau | 29,544 | none | none | none | NC datasets 0; NLLB-excluded 0; protected 1 |
| fat | Fante | 0 | 3.5k total from 1: `Ghana-NLP/FANTE_ENGLISH_PARALLEL_TEXT` 3.5k* [no-lic] | none | 52.3k total from 2: `ghananlpcommunity/ghana-speech-lfn-phonemized` 51.2k [no-lic]; `zabr946/Chatbot-Url` 1.1k [SA] | NC datasets 8; NLLB-excluded 0; protected 1 |
| gaa | Ga | 0 | none | none | none | NC datasets 4; NLLB-excluded 0; protected 3 |
| nup | Nupe | 0 | 261 total from 1: `NupePilot/NupePilotV1.0` 261* | 0 total from 2: `VelkroLM/african-languages-filtered` n/a(repo) [no-lic]; `rufatronics/african-languages-filtered` n/a(repo) [no-lic] | none | NC datasets 0; NLLB-excluded 0; protected 0 |
| tig | Tigre | 0 | none | 76.0k total from 5: `BeitTigreAI/tigre-data-parallel-multilingual` 66.1k [no-lic]; `Tamazight-NLP/Weblate-Translations` 3.3k [no-lic]; ... | none | NC datasets 0; NLLB-excluded 0; protected 1 |

Take-aways: nbl gains about 176k pairs from the SA-government sets (Mehgoss, govza) but that is JW + government text; nus, din and nup are covered only by NaathNLP (templated, provenance unstated), `dayomtechnologies` (5k, no licence) and `Inioluwa` (Nupe); dag and bbj have **no commercial-clear pairs in the catalogue** (dag: only 6.2k Quran verses without licence; bbj: nothing; the Dagbani Bible set `narteybrown/sobolo-corpus-dagbani-bible` and the Ghomala Bible `stephanedonna/english_ghomala` from the 2026-10-29 hunt are not in the catalogue); ibb, men, ish, bin, kea, oke, ndc, her, dua have nothing new beyond the mt560/IbomMT data already used (only UI-string crumbs from Weblate); kab, bam and shi gain the most (Kabyle-English TM 122k, Tatoeba-style 135k, Tatoeba Tashelhit 44k, Bambara fr/en 105k+).

## 5. Non-commercial candidates (NOT usable in production; shown for completeness)

78 parallel, 84 instruction and 107 task/speech datasets are flagged `non_commercial` in the catalogue (catalogue row totals: parallel 35.8M, instruction 222.7M with Inkuba-instruct alone 211.6M (whole repo), task 12.1M). The `ghananlpcommunity` / `ghanaopenai` Twi sets (CC-BY-NC) and `NaathNLP` Nuer sets (CC-BY-NC-SA) dominate the parallel side. Already-used NC exceptions: IbomMT train (owner exception, `howard-nlp/ibom-mt`, 1,992 pairs); `fiifinketia/pristine-twi-english` is quarantined (upstream is NC). Top entries by catalogue rows:

| dataset | languages | rows (catalogue) | licence | kind | note |
|---|---|---:|---|---|---|
| `ghananlpcommunity/pristine-twi-english-parallel-sentences` | twi | 17.41M | cc-by-nc-4.0 | parallel |  |
| `ghanaopenai/english-twi-nouns` | twi | 2.43M | cc-by-nc-4.0 | parallel |  |
| `ghananlpcommunity/pristine-twi-english-bidirectional-multilen` | twi | 2.06M | cc-by-nc-4.0 | parallel |  |
| `ghananlpcommunity/english-twi-sentences-non-nouns` | twi | 1.76M | cc-by-nc-4.0 | parallel |  |
| `ghananlpcommunity/pristine-english-twi-direction-multilen-30-sentences-max` | twi | 1.12M | cc-by-nc-4.0 | parallel |  |
| `ghananlpcommunity/pristine-english-twi-direction-multilen-30-sentences-max-merged` | twi | 1.12M | cc-by-nc-4.0 | parallel |  |
| `ghananlpcommunity/pristine-twi-english_clean` | twi | 999.5k | cc-by-nc-4.0 | parallel |  |
| `ghananlpcommunity/pristine-english-twi-direction-multilen-50-sentences` | twi | 968.6k | cc-by-nc-4.0 | parallel |  |
| `NaathNLP/english_nuer_parallel_translations` | nus | 889.9k | cc-by-nc-sa-4.0 | parallel |  |
| `NaathNLP/multilingual-english-nuer-dinka-swahili-corpus` | din,nus,swh | 803.1k | cc-by-nc-sa-4.0 | parallel | whole-repo count |
| `ghanaopenai/english-twi-nouns-v2` | twi | 626.0k | cc-by-nc-4.0 | parallel |  |
| `AfriSpeech/african-transcribed-speech-monolingual` | amh,efi,fat,ibo+10 | 550.4k | cc-by-nc-4.0 | parallel |  |
| `lelapa/Inkuba-instruct` | hau,swh,xho,yor+1 | 211.56M | CC BY-NC 4.0 | instruction | whole-repo count |
| `ghananlpcommunity/ghana-qa` | ewe,gaa,twi | 3.55M | cc-by-nc-4.0 | instruction | whole-repo count |
| `ketanmore/LLaVA-Instruct-150K-13-African-Languages` | amh,ary,arz,hau+9 | 2.21M | cc-by-nc-4.0 | instruction | whole-repo count |
| `cis-lmu/m_lama` | afr | 843.1k | cc-by-nc-sa-4.0 | instruction |  |
| `ServiceNow-AI/M2Lingual` | amh,hau,ibo,mlg+7 | 289.8k | cc-by-nc-sa-4.0 | instruction | whole-repo count |
| `michsethowusu/ghana-chat-corpus-gaa` | gaa | 226.3k | cc-by-nc-4.0 | instruction |  |
| `michsethowusu/ghana-chat-corpus-ee` | ewe | 226.3k | cc-by-nc-4.0 | instruction |  |
| `michsethowusu/ghana-chat-corpus-ak` | twi | 226.3k | cc-by-nc-4.0 | instruction |  |
| `naijavoices/naijavoices-dataset-compressed` | hau,ibo,yor | 2.09M | cc-by-nc-sa-4.0 | task/other | whole-repo count |
| `disco-eth/WorldSpeech` | afr,amh,hau,ibo+18 | 1.85M | cc-by-nc-4.0 | task/other |  |
| `Centi234/WorldSpeech` | amh,hau,ibo,kau+17 | 1.83M | cc-by-nc-4.0 | task/other |  |
| `ERISLab/LisTAya-transcripts` | amh,hau,swh | 1.60M | cc-by-nc-4.0 | task/other | whole-repo count |
| `ghanaopenai/ghana-speech` | dag,dga,ewe,fat+6 | 930.5k | cc-by-nc-4.0 | task/other |  |
| `ghananlpcommunity/ghana-speech` | dag,dga,ewe,fat+6 | 930.5k | cc-by-nc-4.0 | task/other |  |
| `CLEAR-Global/Chichewa-Synthetic-ASR-Dataset` | nya | 374.6k | cc-by-nc-4.0 | task/other |  |
| `ghanaopenai/new-twi-tts-aligned-ipa` | twi | 161.4k | cc-by-nc-4.0 | task/other |  |

## 6. Excluded / protected / NLLB / mirror lists (counts)

| class | datasets | catalogue rows |
|---|---:|---:|
| nllb_excluded: `michsethowusu/<l>-<l>_sentence-pairs` NLLB/CCMatrix family | 729 | 332M |
| nllb_excluded: other (id says nllb/ccmatrix/translatepsy, Aya collection, suspected LASER-mined Svngoku sets, or the card/hints mention NLLB) | 63 | 1262M (513.8M x2 are the Aya collection copies) |
| protected (benchmarks / benchmark-derived / eval-looking names) | 47 | 1.08M |
| mirror_of_used (used, or a copy of a used set) | 105 | 54M |
| non_commercial | 269 | see section 5 |

NLLB/TranslatePsy-excluded, other than the 729 unsuffixed michsethowusu repos (63 datasets; 44 of them are keyword mentions that may be false positives, listed first-to-last by rows; every one is kept out on the strict side):

- `janavivekariya/aya_collection` (513.76M rows, commercial_ok): CohereLabs/aya_collection (re-upload); Aya excluded in clean build as NLLB/Aya-derived; re-upload of the Aya c
- `CohereLabs/aya_collection` (513.76M rows, commercial_ok): Aya collection: excluded in the clean build (aya-derived rule); Aya collection: translated subsets used NLLB; 
- `liboaccn/nmt-parallel-corpus` (141.68M rows, non_commercial): card/hints mention NLLB (keyword, may be mention-only; verify)
- `sentence-transformers/parallel-sentences-ccmatrix` (41.33M rows, unknown): NLLB/CCMatrix/CCAligned/TranslatePsy by id or michsethowusu unsuffixed pair family
- `zarnite/english-somali_sentence-pairs` (10.23M rows, unknown): NLLB/CCMatrix/CCAligned/TranslatePsy by id or michsethowusu unsuffixed pair family
- `Svngoku/central-africa-multilingual-translation` (7.05M rows, unknown): suspected NLLB/LASER-mined: columns score/similarity identical to the michsethowusu NLLB family and 7.05M rows
- `Svngoku/kongo-lingala-swahili-french-english-pairs-bible` (7.04M rows, unknown): suspected NLLB/LASER-mined: columns score/similarity identical to the michsethowusu NLLB family and 7.04M rows
- `qvac/TranslatePsy-AfriSLM-Synthetic-Mix` (5.67M rows, non_commercial): NLLB/CCMatrix/CCAligned/TranslatePsy by id or michsethowusu unsuffixed pair family
- `zarnite/english-oromo_sentence-pairs` (3.23M rows, unknown): NLLB/CCMatrix/CCAligned/TranslatePsy by id or michsethowusu unsuffixed pair family
- `Heng666/MultiCCAligned-TW-Corpus` (3.13M rows, commercial_ok): NLLB/CCMatrix/CCAligned/TranslatePsy by id or michsethowusu unsuffixed pair family
- `quickmt/quickmt-train.so-en` (2.88M rows, unknown): card/hints mention NLLB (keyword, may be mention-only; verify)
- `taqbaylit/nllb_en_kab` (2.79M rows, non_commercial): NLLB/CCMatrix/CCAligned/TranslatePsy by id or michsethowusu unsuffixed pair family
- `Emulated-Inc/parallel-translation-training-pool` (2.70M rows, commercial_ok): pool contains sources/nllb_sample and sources/smol (NLLB-derived and protected); African share tiny (sw only)
- `Imsidag-community/nllb_en_kab` (2.48M rows, unknown): NLLB/CCMatrix/CCAligned/TranslatePsy by id or michsethowusu unsuffixed pair family
- `agbalu/KabPunct` (1.32M rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `aiana94/xMINDlarge` (589.0k rows, non_commercial): card/hints mention NLLB (keyword, may be mention-only; verify)
- `ghananlpcommunity/twi-english-paragraph-dataset_news` (256.8k rows, non_commercial): card/hints mention NLLB (keyword, may be mention-only; verify)
- `BlakHasan/swa-augmented-collection` (233.2k rows, non_commercial): card/hints mention NLLB (keyword, may be mention-only; verify)
- `2ADT-Consulting/susu-parallel` (229.5k rows, unknown): card/hints mention NLLB (keyword, may be mention-only; verify)
- `Youssouf0123/bambara-french-model` (226.1k rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `BlakHasan/sheng-augmented-collection` (217.9k rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `EthioNLP/Amharic_LLAMA_MT` (204.0k rows, unknown): card/hints mention NLLB (keyword, may be mention-only; verify)
- `aiana94/xMINDsmall` (187.4k rows, non_commercial): card/hints mention NLLB (keyword, may be mention-only; verify)
- `EthioNLP/Amharic_Instruction_dataset` (154.0k rows, unknown): card/hints mention NLLB (keyword, may be mention-only; verify)
- `ymoslem/news-commentary-eng-arz` (83.7k rows, unknown): card/hints mention NLLB (keyword, may be mention-only; verify)
- `BlakHasan/luo-augmented-collection` (60.9k rows, non_commercial): card/hints mention NLLB (keyword, may be mention-only; verify)
- `2b2e/smoltalk-zulu-mtl` (53.6k rows, unknown): card/hints mention NLLB (keyword, may be mention-only; verify)
- `BidirLM/BidirLM-Contrastive` (52.5k rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `BlakHasan/kln-augmented-collection` (51.0k rows, non_commercial): card/hints mention NLLB (keyword, may be mention-only; verify)
- `alaminerca/sango-french-bible-parallel` (42.2k rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `yigagilbert/synthetic-parallel-sunbird` (35.8k rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `taqbaylit/kab-en-toponyms-sentences` (32.0k rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `BlakHasan/som-augmented-collection` (30.0k rows, non_commercial): card/hints mention NLLB (keyword, may be mention-only; verify)
- `agbalu/KabLiterary` (28.9k rows, share_alike): card/hints mention NLLB (keyword, may be mention-only; verify)
- `fotiecodes/Turaco-NLLB-mt-en-wes-cleaned-dataset` (27.8k rows, commercial_ok): NLLB/CCMatrix/CCAligned/TranslatePsy by id or michsethowusu unsuffixed pair family
- `Zyroxx66/Somali_English_NLLB_Corpus` (27.0k rows, unknown): NLLB/CCMatrix/CCAligned/TranslatePsy by id or michsethowusu unsuffixed pair family
- `regnant-io/swahili-instruction-22k` (22.5k rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `simpra/squad_xh` (21.6k rows, share_alike): card/hints mention NLLB (keyword, may be mention-only; verify)
- `kreasof-ai/tatoeba-eng-bem-backtranslation` (20.1k rows, unknown): card/hints mention NLLB (keyword, may be mention-only; verify)
- `Zyroxx66/Somali-Reasoning-Dataset` (18.4k rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `breakend/nllb-multi-domain` (17.6k rows, share_alike): NLLB/CCMatrix/CCAligned/TranslatePsy by id or michsethowusu unsuffixed pair family
- `coded-by-49/langbridge_ai_v3_igbo_datasets` (14.9k rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `BlakHasan/luhya-augmented-collection` (14.7k rows, non_commercial): card/hints mention NLLB (keyword, may be mention-only; verify)
- `yigagilbert/synthetic-parallel-salt` (11.4k rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `CohereLabs/aya_collection_language_split` (10.8k rows, commercial_ok): Aya collection: translated subsets used NLLB; the clean build excludes Aya-derived collections
- `jhu-clsp/kreyol-mt` (8.6k rows, unknown): card/hints mention NLLB (keyword, may be mention-only; verify)
- `NjeriKahoro/thiomi-multilingual-text` (6.8k rows, non_commercial): card/hints mention NLLB (keyword, may be mention-only; verify)
- `Tamazight-NLP/NLLB-Seed_Standard-Moroccan-Tamazight` (6.2k rows, share_alike): NLLB/CCMatrix/CCAligned/TranslatePsy by id or michsethowusu unsuffixed pair family; USED already: extra_source
- `chrismazii/kinycomet_dataset` (4.3k rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `m-a-d-i/wori-wolof-instructions` (3.7k rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `MEYNG/sango-vocabulary` (950 rows, share_alike): card/hints mention NLLB (keyword, may be mention-only; verify)
- `Semence/vaximere-qa-cg` (720 rows, non_commercial): card/hints mention NLLB (keyword, may be mention-only; verify)
- `glk360/fr-ewe-corpus` (60 rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `Ifihan/tiny-aya-base-blind-spots` (18 rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `kalilouisangare/corpus-parallel-en-fr-bam` (0 rows, unknown): card/hints mention NLLB (keyword, may be mention-only; verify)
- `Egonyu/ateso-english-bible-corpus` (0 rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `honourjesus/nllb-hausa-waec-translations` (0 rows, unknown): NLLB/CCMatrix/CCAligned/TranslatePsy by id or michsethowusu unsuffixed pair family
- `sagepond/english_luganda` (0 rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `Joahnne/fonassist-corpus` (0 rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `cheriftenga/tg-nlp-toolkit-fr-ewe-v0.3` (0 rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `burkimbia/nllb-mos-raw` (0 rows, commercial_ok): NLLB/CCMatrix/CCAligned/TranslatePsy by id or michsethowusu unsuffixed pair family
- `Makan09/Bambara_Translation_Corpus_FR-BM` (0 rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)
- `CIRAL/ciral-corpus` (0 rows, commercial_ok): card/hints mention NLLB (keyword, may be mention-only; verify)

Note: two sets that the owner has already approved as exceptions fall under this rule here and are marked: `michsethowusu/english-fulah_sentence-pairs` (NLLB-mined, used for ful with an NLLB tag) and `Tamazight-NLP/NLLB-Seed_Standard-Moroccan-Tamazight` (tzm, share-alike, NLLB-Seed provenance) are in the Amendment-6 extras, which contradicts the "nothing NLLB-derived" rule stated for this audit; please confirm or drop them.

Protected / benchmark-derived (47 catalogue repos; the catalogue's own `_excluded/` folder additionally holds 467 benchmark/eval repos and 4,077 dropped repos in total):

- `mabsoutwassim/smol` (commercial_ok, 237.9k rows): benchmark/eval or benchmark-derived by name
- `michaeldinzinger/webfaq-test` (unknown, 148.4k rows): benchmark-like name (eval/test/dev); heuristic
- `sbmaruf/forai_ml_masakhane_mafand` (non_commercial, 130.9k rows): benchmark/eval or benchmark-derived by name
- `cis-lmu/GlotStoryBook` (unknown, 80.1k rows): benchmark/eval or benchmark-derived by name
- `google/smol` (commercial_ok, 79.9k rows): benchmark/eval or benchmark-derived by name
- `McGill-NLP/NaijaS2ST` (commercial_ok, 73.5k rows): benchmark/eval or benchmark-derived by name
- `masakhane/masakhanews` (commercial_ok, 55.3k rows): benchmark/eval or benchmark-derived by name
- `Sunbird/salt` (share_alike, 46.9k rows): benchmark/eval or benchmark-derived by name
- `shanearora/CaLMQA` (commercial_ok, 38.6k rows): CaLMQA is an evaluation benchmark (culturally specific long-form QA); do not train
- `masakhane/masakhapos` (commercial_ok, 29.0k rows): benchmark/eval or benchmark-derived by name
- `masakhane/AfriDocMT` (non_commercial, 28.2k rows): benchmark/eval or benchmark-derived by name; also: clean specialist set (19-20 weak languages)
- `masakhane/afriqa` (share_alike, 24.3k rows): benchmark/eval or benchmark-derived by name
- `WueNLP/sib-fleurs` (share_alike, 17.9k rows): benchmark/eval or benchmark-derived by name
- `theophilusowiti/AfriQuAD` (commercial_ok, 14.4k rows): AfriQuAD looks like an evaluation set (hold-out QA); verify before any use
- `Lyte/moroccan-instruct-10k-test` (unknown, 10.0k rows): benchmark-like name (eval/test/dev); heuristic
- `galsenai/english-wolof-smol-translation` (unknown, 7.4k rows): benchmark/eval or benchmark-derived by name
- `howard-nlp/ibom-mt` (non_commercial, 6.0k rows): benchmark/eval or benchmark-derived by name; also: clean rebuild (rebuild/meta, per_source.csv)
- `synquid/wmt24pp` (commercial_ok, 6.0k rows): benchmark/eval or benchmark-derived by name
- `google/wmt24pp` (commercial_ok, 6.0k rows): benchmark/eval or benchmark-derived by name
- `Lulu19971017/wmt24pp` (commercial_ok, 6.0k rows): benchmark/eval or benchmark-derived by name
- `hadamard-2/fleurs-ethiopian-v2` (commercial_ok, 5.7k rows): benchmark/eval or benchmark-derived by name
- `MOH749/bocalantics-wolof-eval` (share_alike, 4.8k rows): benchmark-like name (eval/test/dev); heuristic
- `maanka2/fleurs-somali` (commercial_ok, 4.6k rows): benchmark/eval or benchmark-derived by name
- `badrex/anv-data-ke-kalenjin-eval` (unknown, 3.3k rows): benchmark-like name (eval/test/dev); heuristic
- `vonewman/fleurs-wolof-dataset` (unknown, 2.8k rows): benchmark/eval or benchmark-derived by name
- `masakhane/AfriMTE-WMT2024` (commercial_ok, 2.6k rows): benchmark/eval or benchmark-derived by name
- `hypaai/Hypa_Fleurs` (commercial_ok, 1.5k rows): benchmark/eval or benchmark-derived by name
- `KasuleTrevor/Lingala_bmd_test` (unknown, 1.3k rows): benchmark-like name (eval/test/dev); heuristic
- `hypaai/Hypa_Fleurs_v2` (commercial_ok, 1.0k rows): benchmark/eval or benchmark-derived by name
- `badrex/anv-data-ke-somali-test` (unknown, 805 rows): benchmark-like name (eval/test/dev); heuristic
- `zionia/isizulu-asr-test` (commercial_ok, 690 rows): benchmark-like name (eval/test/dev); heuristic
- `zionia/isixhosa-asr-test` (commercial_ok, 686 rows): benchmark-like name (eval/test/dev); heuristic
- `danielbyiringiro/sentiment_hausa_test` (unknown, 500 rows): benchmark-like name (eval/test/dev); heuristic
- `RobotsMali/nyana-eval` (commercial_ok, 45 rows): benchmark-like name (eval/test/dev); heuristic
- `waltertaya/kenyan-Edu-blindspot-eval` (commercial_ok, 30 rows): benchmark-like name (eval/test/dev); heuristic
- `eustacelee/fatima-techchallenge-blindspots-2026` (commercial_ok, 10 rows): benchmark-like name (eval/test/dev); heuristic
- `saaga/yoruba-cultural-reasoning-blindspots` (commercial_ok, 10 rows): benchmark-like name (eval/test/dev); heuristic
- `JibuMaroyi/frontier-model-blindspots` (commercial_ok, 10 rows): benchmark-like name (eval/test/dev); heuristic
- `BeitTigreAI/google-smol-tigre` (commercial_ok, 0 rows): benchmark/eval or benchmark-derived by name
- `masakhane/afriqa-gold-passages` (share_alike, 0 rows): benchmark/eval or benchmark-derived by name
- `djelia/bambara-asr-evaluation` (unknown, 0 rows): benchmark-like name (eval/test/dev); heuristic
- `muzzynine/test` (commercial_ok, 0 rows): benchmark-like name (eval/test/dev); heuristic
- `orionweller/masakhane-afriqa-gold-passages` (share_alike, 0 rows): benchmark/eval or benchmark-derived by name
- `amirkalateh/test` (unknown, 0 rows): benchmark-like name (eval/test/dev); heuristic
- `Theafricatechguy/afriqa` (share_alike, 0 rows): benchmark/eval or benchmark-derived by name
- `CraneAILabs/luganda-tokenizer-evaluation` (commercial_ok, 0 rows): benchmark-like name (eval/test/dev); heuristic
- `<HF_ORG_LEGACY>/africomet-translation` (unknown, 0 rows): benchmark/eval or benchmark-derived by name

Reasons in `_excluded/excluded_datasets.csv`:

| reason | repos |
|---|---:|
| mislabelled: country-statistics tables / jailbreak-prompt forks carrying African language tags | 2559 |
| mixed-scope collection; African part cannot be isolated (no per-language config or language column) | 590 |
| benchmark / evaluation / test set (owner: no benchmarks) | 467 |
| technical / domain-specific | 294 |
| empty placeholder repo (SEO mirror that only points to the original dataset) | 71 |
| no pretrain / sft / rl use detected | 66 |
| no free-text content (tabular / numeric) | 30 |

Used or copies of used sets (`mirror_of_used`, 105 datasets), largest first:

- `HuggingFaceFW/finetranslations` (9.95M rows): clean rebuild (rebuild/meta, per_source.csv)
- `<HF_ORG_LEGACY>/sft-new` (5.65M rows): clean rebuild
- `AfriNLP/AfricanFineTranslations-sentences` (4.95M rows): clean specialist set (19-20 weak languages)
- `nthakur/swim-ir-cross-lingual` (1.81M rows): sft-new/sft-all instruct-swim-ir (Swahili, Yoruba only)
- `Helsinki-NLP/opus-100` (1.77M rows): sentence-transformers/parallel-sentences-opus-100 (same OPUS-100 en-X pairs)
- `sentence-transformers/parallel-sentences-opus-100` (1.77M rows): clean rebuild (rebuild/meta, per_source.csv)
- `ds123DDD/opus-100` (1.76M rows): sentence-transformers/parallel-sentences-opus-100 (re-upload of OPUS-100)
- `KKamauWaithakaA/english-to-luo-opus` (1.49M rows): BrianMwangi/english-to-luo-opus (identical row count 1,489,107)
- `BrianMwangi/english-to-luo-opus` (1.49M rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-afrikaans_sentence-pairs_mt560` (1.16M rows): clean rebuild (rebuild/meta, per_source.csv)
- `ghananlpcommunity/pristine-twi-english` (999.5k rows): fiifinketia/pristine-twi-english (re-upload; quarantined as NC upstream)
- `fiifinketia/pristine-twi-english` (999.5k rows): clean rebuild (quarantine_nc; not in main output)
- `michsethowusu/english-xhosa_sentence-pairs_mt560` (993.7k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-zulu_sentence-pairs_mt560` (964.1k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-malagasy_sentence-pairs_mt560` (826.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-setswana_sentence-pairs_mt560` (780.8k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-chichewa_sentence-pairs_mt560` (778.1k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-shona_sentence-pairs_mt560` (763.5k rows): clean rebuild (rebuild/meta, per_source.csv)
- `takuM23/english-shona_sentence-pairs_mt560` (763.5k rows): michsethowusu/english-shona_sentence-pairs_mt560 (copy)
- `michsethowusu/english-tsonga_sentence-pairs_mt560` (757.9k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-amharic_sentence-pairs_mt560` (669.1k rows): clean rebuild (rebuild/meta, per_source.csv)
- `UBC-NLP/AfroLingu-MT` (620.6k rows): clean rebuild (rebuild/meta, per_source.csv)
- `KwekuAinoo/eng_twi` (561.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-northern-sotho_sentence-pairs_mt560` (555.8k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-lingala_sentence-pairs_mt560` (536.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `agentlans/finetranslations-200-weighted` (508.9k rows): HuggingFaceFW/finetranslations (re-weighted subset)
- `michsethowusu/english-igbo_sentence-pairs_mt560` (415.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-lozi_sentence-pairs_mt560` (411.9k rows): clean rebuild (rebuild/meta, per_source.csv)
- `JoshOrdin/english-yoruba_sentence-pairs_mt560` (411.5k rows): michsethowusu/english-yoruba_sentence-pairs_mt560 (copy)
- `michsethowusu/english-yoruba_sentence-pairs_mt560` (411.5k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-bemba_sentence-pairs_mt560` (381.3k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-rundi_sentence-pairs_mt560` (364.1k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-efik_sentence-pairs_mt560` (332.6k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-tigrigna_sentence-pairs_mt560` (320.9k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-hausa_sentence-pairs_mt560` (295.8k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-luba-kasai_sentence-pairs_mt560` (292.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-congo-swahili_sentence-pairs_mt560` (271.9k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-sango_sentence-pairs_mt560` (250.0k rows): clean rebuild (rebuild/meta, per_source.csv)
- `taqbaylit/tatoeba-en-kab` (245.0k rows): clean specialist set (19-20 weak languages)
- `ayymen/Pontoon-Translations` (237.9k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-tumbuka_sentence-pairs_mt560` (232.5k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-ganda_sentence-pairs_mt560` (224.7k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-kaonde_sentence-pairs_mt560` (219.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-isoko_sentence-pairs_mt560` (215.4k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-umbundu_sentence-pairs_mt560` (212.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-kongo_sentence-pairs_mt560` (206.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `CohereLabs/aya_dataset` (205.6k rows): sft-new/sft-all (aya_dataset, new-language rows only)
- `janavivekariya/aya_dataset` (205.6k rows): re-upload of CohereLabs/aya_dataset (used)
- `michsethowusu/english-venda_sentence-pairs_mt560` (204.4k rows): clean rebuild (rebuild/meta, per_source.csv)
- `Felladrin/ChatML-aya_dataset` (202.4k rows): re-format of CohereLabs/aya_dataset (used)
- `aiana94/polynews-parallel` (198.7k rows): clean specialist set (19-20 weak languages)
- `michsethowusu/english-moore_sentence-pairs_mt560` (186.4k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-tiv_sentence-pairs_mt560` (184.1k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-somali_sentence-pairs_mt560` (161.9k rows): clean rebuild (rebuild/meta, per_source.csv)
- `zarnite/english-somali_sentence-pairs_mt560` (161.9k rows): michsethowusu/english-somali_sentence-pairs_mt560 (copy)
- `michsethowusu/english-oromo_sentence-pairs_mt560` (155.1k rows): clean rebuild (rebuild/meta, per_source.csv)
- `yigagilbert/synthetic-parallel-external` (147.7k rows): clean rebuild (quarantine_nc; not in main output)
- `michsethowusu/english-dholuo_sentence-pairs_mt560` (136.6k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-lunda_sentence-pairs_mt560` (134.6k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-wolaytta_sentence-pairs_mt560` (120.6k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-swati_sentence-pairs_mt560` (116.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `masakhane/InjongoIntent` (104.8k rows): sft-new structured-tasks (intent)
- `michsethowusu/english-oshiwambo_sentence-pairs_mt560` (99.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `Tamazight-NLP/Pontoon-Translations` (95.3k rows): clean rebuild
- `michsethowusu/english-gikuyu_sentence-pairs_mt560` (94.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `nwu-ctext/nchlt` (87.8k rows): sft-new structured-tasks (NER)
- `sudoping01/english-wolof-translation` (84.7k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-tswa_sentence-pairs_mt560` (84.3k rows): clean rebuild (rebuild/meta, per_source.csv)
- `CLEAR-Global/Gamayun-kits` (75.3k rows): clean specialist set (19-20 weak languages)
- `michsethowusu/english-acholi_sentence-pairs_mt560` (73.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-makhuwa_sentence-pairs_mt560` (72.8k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-sena_sentence-pairs_mt560` (52.3k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-kamba_sentence-pairs_mt560` (51.1k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-nyankore_sentence-pairs_mt560` (50.4k rows): clean rebuild (rebuild/meta, per_source.csv)
- `openlanguagedata/oldi_seed` (49.5k rows): clean specialist set (19-20 weak languages)
- `michsethowusu/english-sidaama_sentence-pairs_mt560` (46.9k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-yao_sentence-pairs_mt560` (43.7k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-krio_sentence-pairs_mt560` (42.3k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-kabyle_sentence-pairs_mt560` (38.0k rows): clean rebuild (rebuild/meta, per_source.csv)
- `Makan09/bam-asr-conversational` (37.8k rows): RobotsMali/bam-asr-early (same row count)
- `michsethowusu/english-fon_sentence-pairs_mt560` (31.3k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-ndau_sentence-pairs_mt560` (30.4k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-urhobo_sentence-pairs_mt560` (29.3k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-cameroon-pidgin_sentence-pairs_mt560` (28.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-basaa_sentence-pairs_mt560` (27.8k rows): clean rebuild (rebuild/meta, per_source.csv)
- `universal-dependencies/universal_dependencies` (23.6k rows): sft-new structured-tasks (POS)
- `michsethowusu/english-okpe_sentence-pairs_mt560` (22.5k rows): clean rebuild (rebuild/meta, per_source.csv)
- `mamakobe/luhya-multilingual-dataset` (22.3k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-nigerian-pidgin_sentence-pairs_mt560` (22.0k rows): clean rebuild (rebuild/meta, per_source.csv)
- `beethogedeon/Helsinki-en-fon-opus-20k` (20.0k rows): clean rebuild (rebuild/meta, per_source.csv)
- `cibfaye/wolof-english-bible` (15.8k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-duala_sentence-pairs_mt560` (15.4k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-tachelhit_sentence-pairs_mt560` (15.3k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-wolof_sentence-pairs_mt560` (15.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-jula_sentence-pairs_mt560` (14.9k rows): clean rebuild (rebuild/meta, per_source.csv)
- `karim155/WolBanking77` (14.8k rows): sft-new structured-tasks (intent)
- `michsethowusu/english-edo_sentence-pairs_mt560` (11.6k rows): clean rebuild (rebuild/meta, per_source.csv)
- `michsethowusu/english-herero_sentence-pairs_mt560` (8.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `bltlab/lr-sum` (6.2k rows): sft-new structured-lrsum
- `michsethowusu/english-esan_sentence-pairs_mt560` (5.2k rows): clean rebuild (rebuild/meta, per_source.csv)
- `ARPRIM/pulaar_fulfulde` (1.7k rows): gemma_extra_ccby list (French-source, skipped)
- `michsethowusu/english-mende_sentence-pairs_mt560` (1.7k rows): clean rebuild (rebuild/meta, per_source.csv)
- `hbenayed/tunisian-msa-parallel-corpus-evaluated` (1.0k rows): tunis-ai/tunisian-msa-parallel-corpus
- `tunis-ai/tunisian-msa-parallel-corpus-evaluated` (1.0k rows): tunis-ai/tunisian-msa-parallel-corpus
- `ciatech-africa/kanuri-english-lexicon` (0 rows): considered (language_data_hunt), lexicon

## 7. What could not be verified, and what to check next

1. **Overlap with data already in the pairs.** All counts are gross. mt560, multi-open, Mehgoss, kweku2088, Tommy0201, iamwille and the Bible sets are largely JW/Bible text; the true new count can only be obtained by streaming a sample and hashing against the clean pairs (the `decon.py` + exact-dedupe machinery can do this on the instance).
2. **Old-build membership is inferred** for catalogue datasets with a `convert:*` catalog_plan verdict (r1-r3 plans were not saved). Confirmed old-build membership is only for datasets named in the final_v2 inventory, code, reachability table or sft-new card. The instruction/task bucket is therefore "not in the clean/Hub repos", not "never used".
3. **Whole-repo counts** for 90 multi-language repos: African share unknown (largest: `nordpolemil/biblenlp-corpus` 11.5M verses for 835 languages, `CohereLabs/xP3x` 85M, `akoksal/muri-it` 2.2M, `Svngoku/xP3x-Kongo`, `stanford-oval/ccnews`). They need a per-language count by streaming the language column.
4. **Only 7 repos had rows sampled (3 each).** The rest rely on card tags, column names and the catalogue's single sample row; 16 high-ranked entries were reclassified by hand after reading samples/cards (lexicons, CPT text, Taiwanese/Urdu mis-tags, Llama-4 outputs, an NLLB/SMOL pool) and 78 more were demoted by a name rule (emotion-label corpora, speech-text pairs, lexicons, dictionaries); long-tail pair counts (<2,000 rows) were classified by rule only.
5. **NLLB keyword hits (44) and the two suspected LASER-mined Svngoku sets** are unverified in both directions; the rule was applied on the strict side. 78 of the 273 new_parallel repos carry a synthetic/MT keyword in the card (precision unknown; cidjeu/louisbertson fr-mos is explicitly Glosbe MT).
6. **Licences:** "unknown" means no licence in the card, not "permissive"; Weblate/Liseli/Quran sets carry mixed per-string/per-source licences; Hub card licence differs from the catalogue for some repos (noted per row); gated repos (48) need access approval; 4 repos return 404 now (`MOH749/wolof-commerce-sft`, `gmahia/swahili-civic-nlp`, `voicedata/9jalingo-reviewed-igbo`, `voicedata/9jalingo-reviewed-yoruba`).
7. **Streamable = no** for 10 repos (loading scripts or unreachable): notably `nwu-ctext/autshumato` (script; check the `refs/convert/parquet` branch) and `sawadogosalif/MooreFRCollections`. Unknown (145): no data files seen in the first 400 listed files, gated, or viewer disabled.
8. **Catalogue gaps:** the 31 uncatalogued mt560 repos show the crawl missed languages outside its ISO table; other uncatalogued repos for languages not in the table are likely. Next step: list the Hub by the 31 language names and by uploader (michsethowusu, Svngoku, ghananlpcommunity, NaathNLP, taqbaylit, abdelhaqueidali) to find siblings.
9. **Protected overlap before fetch** cannot be measured: MALIBA-AI/bambara-mt-dataset contains GATITOS rows, BeitTigreAI contains SMOL rows, `Emulated-Inc` contains an NLLB sample and SMOL, `dsfsi/govza` and Mehgoss may contain sentences that are in FLORES+/SALT-type sets; run the protected-index filter on every fetched file.
10. **Not done:** no language-ID or alignment-quality check, no QE, no row-level dedupe; `rows_final` for "Hub config sizes" datasets sums config names that match African ISO codes/names, which can over- or under-count (e.g. Weblate 721k counts en-paired configs of ~50 languages).

Next checks in order: (1) stream 1k rows from each of the 31 mt560 repos and decide on tags; (2) stream-sample `DocHPLT` af-en/en-sw/en-xh, `Mehgoss`, `govza`, `kweku2088`, `Tommy0201/*`, `iamwille` and measure the share not already in the clean pairs; (3) read the splits of `tatoeba_mt_parquet`; (4) sample 20 rows of the NaathNLP Nuer-Dinka set and the `Inioluwa` Nupe set (new languages); (5) decide on the 3 NLLB-flavoured exceptions already in use.
