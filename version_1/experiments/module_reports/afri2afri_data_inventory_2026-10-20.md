<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Inter-African translation (X -> Y): data inventory (2026-10-20)

> **Update 2026-10-20 evening:** the full pair set (`full_pairs/`, 157.5 M rows), the 50 / 50 prompt mixture, the NLLB-mined sets in the SFT, the replay and the web-based RL pools are described in `afri2afri_addendum_2026-10-20_evening.md`; where numbers differ, the addendum is current (training dataset = `sft_v2`).

Deliverable 1 of D-46. Everything below was **measured on the old box** (RTX 5070 box, CPU only, `nice`d, outputs in `/workspace/afri2afri/`) from the data the en->X model was built from; numbers that are
recollection or not checked are labelled so. Raw tables: `reports/afri2afri_2026-10-20/` (csv / heat-maps), builders: `scripts/build_afri2afri_sft.py`, `scripts/afri2afri_report_tables.py`.

## 0. Summary

| question | answer (measured) |
|---|---|
| Do we hold true multi-way parallel text for several of our languages? | **Held-out evaluation sets only** (FLORES+ devtest 44 languages, NTREX 23, SmolSent / SmolDoc 45 each): 2,726 ordered directions with >= 100 shared items. They are the benchmarks and stay out of training. Training-grade multi-way text exists **only as English-pivoted** JW/Bible-style corpora (section 3). |
| Direct African<->African pairs in the SFT corpus (`final_v2`) | 3,164,404 translation rows with two African labels (1,836 labelled directions; 649,227 of them have the SAME language on both sides: mislabelled or monolingual, dropped), i.e. 2,515,177 cross-language rows; after dropping rows the converters flagged `unsure` (314,410), duplicates (every bitext appears in both orders) and non-SFT languages: **1,057,253 unique pairs over 616 language pairs** (79 pairs with >= 1k, 35 with >= 10k); 53 % of them are machine translations (`<HF_ORG>/tds`), 40 % government / curated text of the South African languages. |
| English-pivot join of the en->X pairs (the 25.57 M pairs of `pairs_final`) | 15.8 M distinct English sentences, **1,557,690 of them translated into >= 2 of our languages**; **102,902,360 ordered pairs** (51.5 M unordered) in 5,640 of the 5,852 directions; 4,086 directions >= 1k, 2,362 >= 10k, 128 >= 100k. **70.8 % of these pairs come from Bible / JW / OPUS-MT560 text**, 0.8 % from curated text, 3.0 % from machine translation. |
| What the first dataset contains (after LID, QE floor, UniMax, filters) | **6,683,317 train pairs + 77,328 validation pairs**, 5,618 ordered directions with data (3,854 >= 1k pairs), 1.34 B packed tokens (see the plan); the staged export has 6,628,491 + 76,866 pairs and 411,594 RL prompts with per-row `source_dataset` and declared `licence`. |
| Hard facts about gaps | 1,998 directions have < 1,000 pairs, 234 have none. Thin source languages: bam, nus, bbj, dag, shi, nbl, ibb, kin, sot, kab, som, wol, kon. |
| Not held, but would fill the most | the 528 `michsethowusu/<lang>-<lang>_sentence-pairs` NLLB-mined sets (110.9 M rows, 502 language pairs among 33 of our languages, licence unspecified, NC upstream): fills 278 of the 1,998 thin directions; **owner decision pending since 2026-10-11**. |

## 1. How the numbers were obtained

* `index` (stage 1): the 14 shards of `/workspace/sft_build/gemma/pairs_final` (25,573,645 rows, columns english / target / iso / source_dataset / licence / flipped / qe) -> 64-bit hash of the NFC, case-folded, punctuation-free English sentence (`afri2afri/text.py: pivot_key`), language id, dataset id (255 distinct `source_dataset` values), licence class, AfriCOMET-QE, lengths.
* `matrix` (stage 2): leak filter (section 7), then **one row per (English sentence, language)** (best provenance tier, then best QE), group by English hash, count ordered pairs per (tier, source, target) without materialising them (`pivot.pair_count_tensor`, tested against brute force).
* `direct` / `direct-clean`: the translation rows of `final_v2` (46,084,639 of 77,418,776 rows; 988 shards, 43 GB) in which neither side is English.
* Languages are the **77 SFT languages** (`languages.ALL_ISOS` minus nde fat knc gaa men luy tzm ful, D-38).
* **Provenance kinds are a name-pattern heuristic** (`afri2afri/provenance.py`), not an audit: `human_curated` (dsfsi, masakhane, mbaza, Kencorpus, ...), `religious_jw` (OPUS-MT560 sets of michsethowusu, Bible sets: mostly JW300 and Bible, some software strings), `mixed` (aggregate repos such as `<HF_ORG_LEGACY>/naija*`, CCMatrix, Tatoeba), `synthetic_mt` (tds, *synthetic*, topxgen, back-translation).

## 2. (a) Multi-way parallel sets (same sentences in several of our languages)

| set | where | size / languages (77 SFT languages only) | licence | status |
|---|---|---|---|---|
| **FLORES+** dev (997) / devtest (1,012) | Hub `openlanguagedata/flores_plus` (gated auto, ~200 languages). Box: devtest `/workspace/sft_build/bench/test_sets/flores_plus/devtest/<code>.jsonl` (44 of our languages); dev `/workspace/sft_build/ssa_cmp/flores/` (38 files) + `bench/bench_dev.jsonl` (17 languages) | 1,012 / 997 sentences per language | CC-BY-SA-4.0 | **EVAL, keep out.** dev = reward calibration split (D-24), devtest = benchmark. 33 of our 77 languages have no devtest file on the box |
| **NTREX-128** | `bench/test_sets/ntrex/NTREX-128/` (Hub `davidstap/NTREX`, `mteb/NTREX`) | 1,997 sentences, 23 of our languages | CC-BY-SA-4.0 | **EVAL, keep out** |
| **SMOL** (SmolSent, SmolDoc, GATITOS) | `bench/test_sets/{smol_sent,smol_doc,gatitos}/`, Hub `google/smol` (fdaff3a) | SmolSent 863 / language, SmolDoc 130-668 documents / language, 45 of our languages; GATITOS = ~4,000 words (a lexicon, not sentences) | CC-BY-4.0 | **EVAL, keep out** |
| IbomMT, FLORES-MOZ, GlotStoryBook, SALT test | `bench/test_sets/{ibom_mt,flores_moz,glot_story,salt}/` | one to a few languages each, English-source | CC-BY-NC-4.0 / CC-BY-SA-4.0 / per-row CC-BY(-NC) / CC-BY-SA-4.0 | **EVAL, keep out** (D-42) |
| MAFAND-MT | `bench/test_sets/mafand/en-*/test.json`, Hub `masakhane/mafand` | 15 languages, English (some French) -> X only, NOT multi-way | CC-BY-NC-4.0 | **EVAL (test) keep out**; dev rows only in `bench_dev.jsonl` |
| NLLB-Seed (OLDI) | not on the Hub under `openlanguagedata/seed` (RepositoryNotFound with our token) | ~6 k English sentences in ~40 languages (recollection, not checked) | CC-BY-SA (recollection) | not held; **candidate TRAINING multi-way set**, must be fetched from OLDI |
| TICO-19 | Hub `gmnlp/tico19` exists (80 files, no licence on the card) | English -> 38 languages incl. several African ones (recollection, not checked) | not on the card | not held; candidate |
| Bible corpora | Hub `nordpolemil/biblenlp-corpus` (cc-by-4.0 + other, 837 files, 833 languages, verse-aligned), `bible-nlp/biblenlp-corpus`, `Helsinki-NLP/bible_para` (cc0-1.0). Held: `michsethowusu/african-bible-parallel` (414,972 rows, 7 languages, en + fr), `AfriSpeech/africa-corpus`, `<HF_ORG>/biblenlp-corpus-mmteb*`, `cibfaye/wolof-english-bible`, JW pairs of the mt560 sets | multi-way by verse id in the Hub corpora; **held data is English-pivoted**: the join of section 3 recovers it | mixed per translation; Bible-society copyright not verified | in training (pivot) |
| JW300 / OPUS-MT560 | `michsethowusu/english-<language>_sentence-pairs_mt560` (99 sets in the catalogue; 43 of them are in the build: 7,874,800 translation rows, 118 to 1,375,800 per language), `AfriSpeech/african-corpus-jw`, `Tommy0201/JW300_Igbo_To_Eng` | 118 to 1.38 M English-X rows per language; the English side is the same JW300 English across languages: **this is the main multi-way source** | cards declare CC-BY-4.0 (upstream OPUS: jw.org text, copyright not cleared) | in training (pivot) |
| Tatoeba | `<HF_ORG_LEGACY>/naija2:tatoeba_mt_train_english_translation` (6.8 M en-X rows), `taqbaylit/tatoeba-en-kab` (225 k), Hub `Helsinki-NLP/tatoeba_mt` (CC-BY-2.0, gated auto) | short sentences, many languages | CC-BY-2.0 | in training (pivot) |
| OPUS-100 / other OPUS | `Helsinki-NLP/opus-100` (en-ha, en-ig, en-yo ...), `opus_xhosanavy`, `sentence-transformers/parallel-sentences-ccmatrix` (723 k, 3 languages) | en-X | mixed / unknown | in training (pivot) |
| AfroLingu-MT (AfriMT) | `UBC-NLP/AfroLingu-MT` (gated auto), 355,000 rows in the build, 73,098 African<->African | 29 languages | not on the card | in training (direct + pivot). `masakhane/afrimt` does not exist on the Hub |
| Lanfrica listings | not accessed (no API call made); the HF crawl catalogue `data-catalog/african_hf/rl/*.csv` (1,807 parallel-pair datasets) is our listing | | | see section 6 |

Measured coverage of the **held-out multi-way sets** (T3 below): they give 2,726 ordered directions with >= 100 shared items and 2,080 with >= 1,000, which is the inter-African benchmark pool (plan, section 7).

## 3. (b) Pivot-derived pairs (English as the pivot)

Method and numbers: section 1. Heat-maps with one character per cell for the full 77 x 77 matrix are in `reports/afri2afri_2026-10-20/heatmaps.md`; exact counts in the csv files of the same directory
(`pivot_en_all.csv`, `pivot_en_without_religious_or_synthetic.csv`, `pivot_en_human_curated_only.csv`).

| matrix (ordered directions, 77 languages) | >= 1 pair | >= 100 | >= 1k | >= 10k | >= 100k | total ordered pairs |
|---|---:|---:|---:|---:|---:|---:|
| pivot via English, all provenance | 5,640 | 4,806 | 4,086 | 2,362 | 128 | 102,902,360 |
| ... without Bible / JW and without machine translation | 3,672 | 2,272 | 1,100 | 478 | 44 | 27,003,532 |
| ... human-curated sources only | 846 | 280 | 124 | 18 | 0 | 785,854 |
| direct rows of the SFT corpus (both orders) | 1,232 | 198 | 158 | 70 | 4 | 2,114,506 |
| NOT HELD: NLLB-mined `michsethowusu` pair repos (catalogue numbers) | 1,002 | 1,002 | 1,002 | 994 | 544 | 221,736,070 |
| **built dataset (after all filters, UniMax budget 8 M)** | 5,618 | 4,636 | 3,854 | 0 | 0 | 13,366,634 |

* Pairs by provenance tier of the weaker side: religious_jw 72,811,118; mixed 26,217,678; synthetic_mt 3,087,710; human_curated 785,854.
* **French pivot**: 297,458 French sentences, only 19,603 translated into >= 2 languages, 125 language pairs, 143,313 pairs; most of them come from datasets whose language labels are inconsistent (e.g. afr-twi, bin-dag at 9 k each from one French source). Not used in the first dataset; not worth it before the labels are audited. No Arabic or Portuguese pivot data exists in the corpus (0 rows).
* Per language (en->X unique English sentences, how many sit in a multi-language group, number of partners, direct rows, monolingual shard size, benchmark sets, reward family): `reports/afri2afri_2026-10-20/language_table.md`.

#### Where the pivot pairs come from, and where the direct pairs come from

#### T1. Source repositories that produce the pivot pairs (share of the 102.9 M ordered pairs; `kind` = name-pattern heuristic)

| source repo | kind | rows in multi-language English groups | pair participations | share | languages |
|---|---|---:|---:|---:|---:|
| `michsethowusu/english-setswana_sentence-pairs_mt560` | religious_jw | 585,702 | 5,008,959 | 4.9% | 1 |
| `<HF_ORG_LEGACY>/naija` | mixed | 307,326 | 4,155,045 | 4.0% | 21 |
| `KwekuAinoo/eng_twi` | mixed | 410,508 | 4,137,342 | 4.0% | 1 |
| `michsethowusu/english-lozi_sentence-pairs_mt560` | religious_jw | 304,579 | 3,942,789 | 3.8% | 1 |
| `michsethowusu/english-luba-kasai_sentence-pairs_mt560` | religious_jw | 218,420 | 3,523,804 | 3.4% | 1 |
| `michsethowusu/english-ganda_sentence-pairs_mt560` | religious_jw | 182,782 | 3,374,480 | 3.3% | 1 |
| `michsethowusu/english-kaonde_sentence-pairs_mt560` | religious_jw | 163,029 | 3,059,870 | 3.0% | 1 |
| `michsethowusu/english-isoko_sentence-pairs_mt560` | religious_jw | 160,264 | 3,044,254 | 3.0% | 1 |
| `<HF_ORG_LEGACY>/naija2` | mixed | 602,372 | 2,659,270 | 2.6% | 22 |
| `michsethowusu/english-oromo_sentence-pairs_mt560` | religious_jw | 124,231 | 2,498,507 | 2.4% | 1 |
| `michsethowusu/english-dholuo_sentence-pairs_mt560` | religious_jw | 114,818 | 2,448,613 | 2.4% | 1 |
| `michsethowusu/english-lunda_sentence-pairs_mt560` | religious_jw | 99,358 | 2,144,508 | 2.1% | 1 |
| `michsethowusu/english-congo-swahili_sentence-pairs_mt560` | religious_jw | 167,550 | 1,932,888 | 1.9% | 1 |
| `michsethowusu/english-wolaytta_sentence-pairs_mt560` | religious_jw | 89,554 | 1,924,744 | 1.9% | 1 |
| `michsethowusu/english-venda_sentence-pairs_mt560` | religious_jw | 101,712 | 1,922,489 | 1.9% | 1 |
| `michsethowusu/english-tumbuka_sentence-pairs_mt560` | religious_jw | 100,264 | 1,902,482 | 1.8% | 1 |
| `michsethowusu/english-moore_sentence-pairs_mt560` | religious_jw | 99,268 | 1,884,901 | 1.8% | 1 |
| `michsethowusu/english-swati_sentence-pairs_mt560` | religious_jw | 80,453 | 1,790,888 | 1.7% | 1 |
| `michsethowusu/english-tiv_sentence-pairs_mt560` | religious_jw | 100,216 | 1,772,229 | 1.7% | 1 |
| `michsethowusu/english-gikuyu_sentence-pairs_mt560` | religious_jw | 72,244 | 1,740,163 | 1.7% | 1 |
| `michsethowusu/english-sango_sentence-pairs_mt560` | religious_jw | 99,331 | 1,732,628 | 1.7% | 1 |
| `michsethowusu/english-umbundu_sentence-pairs_mt560` | religious_jw | 99,732 | 1,722,020 | 1.7% | 1 |
| `michsethowusu/english-oshiwambo_sentence-pairs_mt560` | religious_jw | 74,366 | 1,704,951 | 1.7% | 1 |
| `michsethowusu/english-efik_sentence-pairs_mt560` | religious_jw | 109,136 | 1,647,935 | 1.6% | 1 |
| `michsethowusu/english-rundi_sentence-pairs_mt560` | religious_jw | 101,391 | 1,521,363 | 1.5% | 1 |
| `michsethowusu/english-tigrigna_sentence-pairs_mt560` | religious_jw | 98,259 | 1,501,270 | 1.5% | 1 |
| `michsethowusu/english-bemba_sentence-pairs_mt560` | religious_jw | 103,622 | 1,495,949 | 1.5% | 1 |
| `<HF_ORG>/tds` | synthetic_mt | 294,253 | 1,489,377 | 1.4% | 21 |
| `michsethowusu/english-hausa_sentence-pairs_mt560` | religious_jw | 89,138 | 1,440,198 | 1.4% | 1 |
| `michsethowusu/english-yoruba_sentence-pairs_mt560` | religious_jw | 113,080 | 1,434,804 | 1.4% | 1 |
| (198 more repos) | | | 32,343,640 | 31.4% | |


#### T2. Corpora of the SFT build that pair TWO AFRICAN languages directly (translation rows of `final_v2`, both orders counted)

| source repo | rows (all pairs) | African<->African rows | languages |
|---|---:|---:|---:|
| `<HF_ORG>/tds` | 10,210,565 | 1,187,011 | 22 |
| `dsfsi/govza-sa-cabinet-statements-sentence-aligned` | 672,940 | 548,086 | 10 |
| `bigscience/xP3mt` | 385,573 | 366,374 | 29 |
| `taresco/xP3x_african_subset` | 410,942 | 366,004 | 61 |
| `<HF_ORG_LEGACY>/naija2` | 12,057,194 | 198,773 | 22 |
| `dsfsi/vukuzenzele-sentence-aligned` | 193,041 | 170,062 | 10 |
| `UBC-NLP/AfroLingu-MT` | 355,000 | 73,098 | 29 |
| `Vambo-AI/Languages` | 43,250 | 43,246 | 32 |
| `<HF_ORG>/clean_data` | 366,255 | 27,809 | 14 |
| `mbazaNLP/Kinyarwanda_English_parallel_dataset` | 45,222 | 26,492 | 22 |
| `oza75/bambara-gec-synthetic` | 26,118 | 26,084 | 41 |
| `<HF_ORG_LEGACY>/naija` | 1,240,244 | 18,747 | 22 |
| `dsfsi/za-mafoko-dsac` | 21,868 | 18,174 | 10 |
| `gimmy256/ugandan_languages` | 27,986 | 9,562 | 2 |
| `Kencorpus/KenTrans` | 8,256 | 8,256 | 4 |
| `<HF_ORG>/clean_data2` | 296,514 | 7,040 | 8 |
| `RobotsMali/kunkado` | 6,544 | 6,484 | 37 |
| `taqbaylit/kabyle-english-TM` | 46,564 | 4,846 | 52 |
| `Rogendo/English-Swahili-Sentence-Pairs` | 140,518 | 4,498 | 46 |
| `JoshOrdin/english-yoruba_sentence-pairs_mt560` | 40,410 | 4,462 | 34 |
| `<HF_ORG_LEGACY>/Clean_translate` | 380,290 | 3,700 | 14 |
| `<HF_ORG>/gsm8k3` | 6,422 | 3,457 | 4 |



Reading T1: the multi-way mass is the JW/Bible English of the OPUS-MT560 sets (one repo per language), plus the old tagged repos `<HF_ORG_LEGACY>/naija*` (Tatoeba, JW, FLORES-era sets) and `KwekuAinoo/eng_twi`. **The domain of the pivot pairs is therefore religious / Jehovah's Witnesses text**; the FLORES+ domain (news, wiki, travel) is almost absent. This is the single most important caveat of the dataset.

Reading T2: the only large direct corpora are (i) `<HF_ORG>/tds` (machine-translated yor/hau/ibo, `synthetic_mt`, gated at AfriCOMET >= 0.55 by its builder), (ii) the South African government corpora `dsfsi/govza-*`, `dsfsi/vukuzenzele-*`, `dsfsi/za-mafoko-*` (**human, sentence-aligned, 10 South African languages, CC-BY-4.0 / other**: the best data we hold for zul xho ssw nbl sot nso tsn ven tso afr), (iii) small community corpora (Luo-Swahili, Ewe-Fon, Dyula-Bambara, Kinyarwanda-Pidgin, ...). Several "direct" rows in old converted repos have **wrong language labels** (e.g. `naija2` Igbo -> Igbo with an English answer); the build therefore LID-checks both sides (section 5).

## 4. (c) The en->X data the first model was trained on

| stage | location (old box) | count |
|---|---|---:|
| translation rows of the SFT corpus | `/workspace/sft_build/final_v2/<dataset>/part-*.parquet` (214 converters, 988 shards, 43 GB) | 46,084,639 of 77,418,776 rows |
| en->X pairs, raw | `gemma/pairs/` (52 files, 3.3 GB) | 26,635,338 |
| + extras, deduped, licence class | `gemma/merged/` (16 files, 3.9 GB) | 30,089,460 (licence: unspecified 20.79 M, other 4.72 M, cc-by 4.43 M, apache 64 k, mit 52 k, mpl 36 k, share_alike 6.7 k; non-commercial 0, already dropped) |
| QE-gated pairs (**the join input**) | `gemma/pairs_final/` (14 files, 3.1 GB) | 25,573,645 (per language QE >= max(0.20, P15)) |
| SFT windows | `gemma/windows_sft/` (29 files, 3.0 GB) | 25,821,421 pairs -> 11,442,172 windows (b64 / b256 / b512 / b1024 token-balanced) |
| packed training data | `gemma/packed_fwd/{train,val}` + `meta.json` | 11,329,709 samples, 2,204,187,720 tokens (1,318,270,369 target), seq_len 4096, uint32; bucket copy `remote:<S3_BUCKET>/gemma_mt/data/gemma_packed_fwd_sft_windows.tar.zst` (2.27 GB zstd) |
| QE scores | `gemma/scored*/` (3 dirs) | every pair scored by `masakhane/africomet-qe-stl-1.1` |
| English RL prompts | `gemma/rl_prompts_v2/` (77 language files), bucket `gemma_mt/data/gemma_rl_prompts_v2.tar.zst` (3.4 GB) | English windows from FineWeb-edu / Wikipedia |

Licence classes of these pairs are the weak point for any release: 81 % of the built pairs had class `unspecified` because the licence scan did not read the cards of the OPUS-MT560 repos (their cards declare CC-BY-4.0); the export stage re-reads the declared licences (plan, section 6).

## 5. (d) Monolingual African text (RL prompts)

| source | where | size | licence |
|---|---|---|---|
| pretraining text shards | `/workspace/tokdata/shards/<iso>/*.txt.zst`, 25 languages (afr amh efi ewe fon hau ibb ibo kin nya orm pcm sna som sot swh tir tiv tsn twi urh wol xho yor zul, + arb eng fuv ben); 13 GB zstd | one paragraph per line, no reliable document separator; file families `hf0-HuggingFaceFW_fineweb-2` (amh hau ibo pcm tir yor only: 20 files), `corpus-*` (Bible text), `training_cle-w*` (cleaned training chunks), `bm*` (task-formatted <HF_ORG_LEGACY> mixes: unusable as prose) | fineweb-2 = ODC-By-1.0; `training_cle` / `corpus` licences not vetted |
| bucket copy of the pretraining text | `remote:<S3_BUCKET>/datasets/pretrain/<iso>.tar.zst`, 73 objects, 95.9 GiB (afr 9.5, amh 10.0, xho 7.2, swh 6.7, hau 6.0, som 3.3, ibo 4.2, yor 4.0, zul 2.4, ... many languages < 0.1) | | mixed |
| held-out SFT-side texts | the African sides of `pairs_final` whose English sentence and text are in no selected SFT pair | all 77 languages | per-row licence class |
| HF catalogue crawl | `data-catalog/african_hf/pretrain/*.csv`: 7,049 dataset-language entries; every one of the 77 languages except kmb, bbj, kea has at least one | rows and commercial flag per entry in `language_table.csv` | per card |
| not fetched (v2) | Hub `wikimedia/wikipedia` (CC-BY-SA-3.0 + GFDL), `HuggingFaceFW/fineweb-2` (ODC-By), `cis-lmu/GlotCC-V1` (CC0), `HPLT/HPLT2.0_cleaned` (CC0), `allenai/c4` (ODC-By), `HuggingFaceFW/finetranslations` (ODC-By, `og_*` columns are the original African documents) | streamed, not stored | as listed (verified on the Hub 2026-10-20) |

## 6. Candidates that are NOT held (what would fill the thin directions)

* **`michsethowusu/<lang>-<lang>_sentence-pairs`** (NLLB-v1 mined, CCMatrix-style, `score` column; card tags NLLB / OPUS; no licence declared; the owner's rule of 2026-10-11 treats NLLB output as non-commercial): 528 repos with both languages among our 77 (562 resolved in total, 211 names unresolved because they involve English / French / languages outside the 77), **502 distinct unordered language pairs between 33 of our languages, 110,944,701 rows**, 498 of the 502 pairs with >= 10k rows. It would give a >= 1k direct count to 278 of the 1,998 directions that are thin in the built dataset (the 33 languages: afr amh bam bem dyu ewe fon hau ibo kam kik kin kmb kon lin nus nya orm run sna som ssw swh tir tsn tso tum twi umb wol xho yor zul; matched by language NAME in the repo id, so a spelling we did not map is missed: 211 names stayed unresolved) and a non-religious domain. Needs: owner decision on the licence; ~110 M rows ~ 22 GB.
* Other direct sets on the Hub (catalogue): `Svngoku/kongo-lingala-swahili-french-english-pairs-bible` (7.0 M rows, kon-lin-swh, licence unknown), `NaathNLP/Nuer_Dinka_Parallel_translation_pairs` (899 k, nus-din, MIT; Dinka is not among our 77), `abdelhaqueidali/Kabyle-Latin-to-Tifinagh-Parallel-Corpus` (script pair, not a language pair), `jake-anto/wiktionary` (lexicon), `gtak1/panlex-meanings` (lexicon).
* `CohereLabs/aya_collection` and relatives: NLLB-translated templates (owner rule: NC / held).
* Bible: `nordpolemil/biblenlp-corpus` would add verse-aligned text for any language of its 833 that is also ours (fetch ~ GBs; mostly the same domain as what we already have).

## 7. Held-out data and the leak policy

**Policy (D-46).** Nothing of the evaluation sets goes into any training or RL file of this project: FLORES+ dev and devtest, NTREX-128, SmolSent / SmolDoc / GATITOS, MAFAND-MT test (and dev), IbomMT, FLORES-MOZ, GlotStoryBook, SALT test, and the documents of the new long-document benchmark. The rule is applied to **every text of a candidate** (the English / pivot sentence, the source, the target), in any language, because these sets are multi-way: a Hausa training sentence that equals the FLORES+ Hausa line is a leak for every direction.

**Filter** (`afri2afri/contam.py`, `LeakIndex`): (a) *exact*: `pivot_key` (NFC, case-folded, punctuation-free; U+0329 and U+0323 unified) equal to any benchmark sentence of any language (benchmark entries with fewer than 3 words are never used for matching, so one-word GATITOS entries cannot delete the corpus); (b) *near*: >= 50 % of the text's word 8-grams occur in the benchmark 8-gram set (texts of >= 8 words).
The index holds **916,486 benchmark sentences** (FLORES+ 46,552 + 37,886 dev copies + 60,836 `bench_dev`, NTREX 49,924, MAFAND 40,014, SmolSent 77,666, SmolDoc 292,078, GATITOS 289,613, IbomMT 4,048, FLORES-MOZ 2,024, GlotStory 6,857, SALT 5,000, LID dev 3,988).

**Measured effect**: at the join, 29,953 en->X rows had their English sentence and 30,882 rows their African text in a benchmark (exact rule) -> removed before any pair is formed; at materialisation the text filters removed another 1,908 train / 59 validation pairs (exact or 8-gram; this run also indexes the 500 English documents of the long-document benchmark core set, 27,402 sentences, 568,225 8-grams: the first run had 451 / 3 and an independent re-check found 81 + 2 English pivot sentences of the staged data inside those documents, the second run finds 0 of 606,562); RL assemble (windows): 2,297 benchmark windows, 931 windows sharing a sentence with SFT texts, 797 LID failures, 16,720 duplicates, 4,283 basic/script failures; an independent re-check of 60,000 sampled RL prompts found 0 benchmark hits. **Residual risks, stated:** (1) the 8-gram rule needs >= 8 words, so a benchmark sentence of 5-7 words is only caught by the exact rule; (2) paraphrases / retranslations of benchmark sentences are not detectable; (3) a model initialised from the en->X checkpoint **has already seen** the English side of 91 % of SmolSent and the reference pair for bam 564, wol 630, ewe 524, yor 171, zul 77, hau 72, orm 58, ibo 31 sentences (D-42): for those languages the inter-African benchmark must report the `no_pair` subset next to the full number, or the model must start from the base Gemma.


## Appendix: measured tables

#### T3. Multi-way human benchmark sets we hold (restricted to the 77 SFT languages), measured on the old box

| set | languages | items per language (median) | ordered directions with shared items | directions with >= 500 shared items |
|---|---:|---:|---:|---:|
| flores_devtest | 44 | 1012 | 1892 | 1892 |
| ntrex | 23 | 1997 | 506 | 506 |
| smol_sent | 45 | 863 | 1980 | 1980 |
| smol_doc | 45 | 213 | 1980 | 12 |
| **union** | | | 2726 | >= 100 items: 2726; >= 1,000 items: 2080 |

Languages with NO multi-way human set at all (22): ibb, urh, bin, dag, oke, bbj, swc, kqn, lun, iso, loz, wal, tsc, kua, seh, her, yao, sid, ish, dua, bas, wes.


#### T4. Hub ids checked on 2026-10-20 with the read token (metadata only)

| Hub id | exists | gated | licence on the card |
|---|---|---|---|
| `openlanguagedata/flores_plus` | yes | auto | ['cc-by-sa-4.0'] |
| `openlanguagedata/seed` | no (RepositoryNotFoundError) | | |
| `facebook/flores` | yes | auto | ['cc-by-sa-4.0'] |
| `mteb/NTREX` | yes | False | cc-by-sa-4.0 |
| `mteb/ntrex-128` | no (RepositoryNotFoundError) | | |
| `davidstap/NTREX` | yes | False | ['cc-by-sa-4.0'] |
| `google/smol` | yes | False | cc-by-4.0 |
| `gmnlp/tico19` | yes | False | None |
| `tico-19/tico-19` | no (RepositoryNotFoundError) | | |
| `masakhane/mafand` | yes | False | ['cc-by-nc-4.0'] |
| `masakhane/afrimt` | no (RepositoryNotFoundError) | | |
| `Helsinki-NLP/tatoeba_mt` | yes | auto | ['cc-by-2.0'] |
| `Helsinki-NLP/tatoeba` | yes | False | ['cc-by-2.0'] |
| `nordpolemil/biblenlp-corpus` | yes | False | ['cc-by-4.0', 'other'] |
| `bible-nlp/biblenlp-corpus` | yes | False | ['cc-by-4.0', 'other'] |
| `Helsinki-NLP/bible_para` | yes | False | ['cc0-1.0'] |
| `Helsinki-NLP/jw300` | no (RepositoryNotFoundError) | | |
| `Helsinki-NLP/opus-100` | yes | False | ['unknown'] |
| `Helsinki-NLP/ccaligned_multilingual` | no (RepositoryNotFoundError) | | |
| `allenai/nllb` | yes | False | None |
| `facebook/nllb-seed` | no (RepositoryNotFoundError) | | |
| `UBC-NLP/AfroLingu-MT` | yes | auto | None |
| `McGill-NLP/SSA-MTE` | yes | False | ['cc-by-sa-4.0'] |
| `masakhane/afriqa` | yes | False | cc-by-sa-4.0 |
| `mteb/sib200` | yes | False | cc-by-sa-4.0 |
| `Davlan/sib200` | yes | False | ['cc-by-sa-4.0'] |
| `facebook/belebele` | yes | False | cc-by-sa-4.0 |
| `taresco/xP3x_african_subset` | yes | False | None |
| `CohereLabs/xP3x` | yes | False | ['apache-2.0'] |
| `michsethowusu/afrikaans-swahili_sentence-pairs` | yes | False | None |
| `michsethowusu/xhosa-zulu_sentence-pairs` | yes | False | None |
| `wmt/wmt22` | no (RepositoryNotFoundError) | | |
| `Helsinki-NLP/europarl` | yes | False | ['unknown'] |
| `ymoslem/Tatoeba-Translations` | yes | False | cc-by-2.0 |
| `HuggingFaceFW/finetranslations` | yes | False | odc-by |
| `mutiyama/alt` | yes | False | ['cc-by-4.0'] |
| `LinguaLift/IndicTrans` | no (RepositoryNotFoundError) | | |
| `Lacuna-Fund/ml_text` | no (RepositoryNotFoundError) | | |
| `masakhane/lafand-mt` | no (RepositoryNotFoundError) | | |
| `dsfsi/vukuzenzele-sentence-aligned` | yes | False | cc-by-4.0 |
| `dsfsi/govza-sa-cabinet-statements-sentence-aligned` | yes | False | cc-by-4.0 |
| `dsfsi/za-mafoko-dsac` | yes | False | other |
| `nwu-ctext/autshumato` | yes | False | ['cc-by-2.5'] |
| `wikimedia/wikipedia` | yes | False | ['cc-by-sa-3.0', 'gfdl'] |
| `allenai/c4` | yes | False | ['odc-by'] |
| `HPLT/HPLT2.0_cleaned` | yes | False | cc0-1.0 |
| `cis-lmu/GlotCC-V1` | yes | False | cc0-1.0 |
| `HuggingFaceFW/fineweb-2` | yes | False | odc-by |
| `Unbabel/TowerBlocks-v0.1` | yes | False | None |
| `sunbird/salt` | yes | auto | cc-by-sa-4.0 |
| `Sunbird/salt` | yes | auto | cc-by-sa-4.0 |
| `Sunbird/sunflower-translation` | no (RepositoryNotFoundError) | | |

Source repositories actually used by the staged export (203 repos, declared licence and rows touched): `reports/afri2afri_2026-10-20/source_datasets.csv`.
