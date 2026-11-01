---
license: other
license_name: per-row-source-licences
license_link: LICENSE
task_categories:
- translation
pretty_name: Inter-African language translation (X to Y), SFT pairs and RL prompts
language:
- yo
- ha
- ig
- ee
- wo
- sw
- rw
- so
- om
- am
- ti
- lg
- ln
- zu
- xh
- af
- sn
- ny
- st
- tn
- mg
- rn
- ve
- ts
- ki
- bm
size_categories:
- 100M<n<1B
configs:
- config_name: sft_pairs
  data_files:
  - split: train
    path: sft_pairs_train.parquet
  - split: validation
    path: sft_pairs_val.parquet
- config_name: rl_prompts
  data_files:
  - split: train
    path: rl_prompts.parquet
- config_name: full_pivot_en
  data_files:
  - split: train
    path: full_pairs/pivot_en/*/*.parquet
- config_name: full_direct
  data_files:
  - split: train
    path: full_pairs/direct/*.parquet
- config_name: full_nllb_mined
  data_files:
  - split: train
    path: full_pairs/nllb_mined/*.parquet
- config_name: full_pivot_en_sentences
  data_files:
  - split: train
    path: full_pairs/pivot_en_sentences.parquet
---

# interlaanguage-translation: African language X -> African language Y

Training data for a **direct** (no English pivot) inter-African-language translation model: sentence-pair windows between 77 African languages (SFT) and monolingual source windows with a target-language tag (RL prompts). The 77 languages are listed in `direction_counts.csv` (ISO 639-3 style tags in the `src_iso` / `tgt_iso` columns; the YAML header lists only those with a two-letter code). Built from the data of the English -> African translation model (`gemma_mt`, D-46 in the project's DECISIONS.md).

## IMPORTANT: licences

**Every row keeps the licence of the source it was taken from.** The `licence` column holds the licence *declared on the source dataset's card* (`unknown` when the card declares none; we never guess one). A pair that joins two sources shows both, `a + b`. **Some rows are non-commercial, share-alike or of unknown licence; the dataset as a whole is therefore NOT under one licence. Check the `licence` and `source_dataset` columns and filter before you use the data** (for commercial use keep only the licences you may use; for CC-BY-SA material keep the share-alike obligation). `licence_class` is the coarser class used by the build (`cc-by`, `share_alike`, `unspecified`, ...). Reported licence distribution of the SFT training pairs:

**33.7 % of the SFT training rows have at least one side whose licence is `unknown`, 3.4 % one side declared `other`.** No source in these rows declares a non-commercial licence (sources with NC licences were dropped by the en->X build, D-21), but the *original* texts behind some sources (JW / Bible material of the OPUS-MT560 sets) carry copyright of their own that the dataset cards do not clear.

| `licence` | rows | share |
|---|---:|---:|
| cc-by-4.0 | 4,287,634 | 62.66% |
| unknown | 1,229,762 | 17.97% |
| unknown + cc-by-4.0 | 873,345 | 12.76% |
| cc-by-4.0 + unknown | 141,267 | 2.06% |
| other + cc-by-4.0 | 135,954 | 1.99% |
| mit | 53,761 | 0.79% |
| unknown + other | 30,387 | 0.44% |
| other | 24,291 | 0.36% |
| cc-by-4.0 + other | 20,378 | 0.30% |
| other + unknown | 18,823 | 0.28% |
| mpl-2.0 | 9,216 | 0.13% |
| cc-by-sa-4.0 | 6,294 | 0.09% |
| mpl-2.0 + unknown | 4,946 | 0.07% |
| unknown + mit | 1,363 | 0.02% |
| apache-2.0 + cc-by-4.0 | 1,362 | 0.02% |
| apache-2.0 + unknown | 1,258 | 0.02% |
| unknown + apache-2.0 | 701 | 0.01% |
| unknown + mpl-2.0 | 379 | 0.01% |
| apache-2.0 + other | 267 | 0.00% |
| cc-by-4.0 + apache-2.0 | 239 | 0.00% |
| cc-by-sa-4.0 + unknown | 167 | 0.00% |
| mit + unknown | 139 | 0.00% |
| other + apache-2.0 | 78 | 0.00% |
| cc-by-sa-4.0 + cc-by-4.0 | 71 | 0.00% |
| mpl-2.0 + cc-by-4.0 | 58 | 0.00% |

## Files

| file | rows | content |
|---|---:|---|
| rl_prompts.parquet | 509,488 | RL prompts: source-language window + target tag |
| sft_pairs_train.parquet | 6,842,249 | SFT sentence pairs, one row per UNORDERED pair {src_iso, tgt_iso} (train a model on both directions) |
| sft_pairs_val.parquet | 81,760 | validation pairs (whole English sentences held out, never used with another partner in train) |

## What is in a row (SFT pairs)

`src_iso`, `tgt_iso` (ISO-639-3 style tags of `gemma_mt/languages.py`, `src_iso` sorts before `tgt_iso`), `src`, `tgt` (NFC text, windows of consecutive sentences are made at training time), `pivot_en` (the English sentence both sides translate, `null` for direct pairs), `origin` (`pivot_en` = two translations of the same English sentence; `direct` = a corpus that already pairs the two languages; `mined_nllb` = NLLB-v1 mined pair, filtered, **non-commercial upstream**), `provenance_kind` (`human_curated` | `mined_nllb` | `mixed` | `religious_jw` | `synthetic_mt`: a name-pattern heuristic, not an audit), `prompt_form`, `source_dataset`, `licence`, `licence_class`, `qe_src` / `qe_tgt` (AfriCOMET-QE of the English -> X translation of each side, NaN for direct pairs), `group_size` (number of languages that share the English sentence), `group_key`, `src_tok` / `tgt_tok` (Gemma tokens), `epochs` (how often the build planned to repeat the pair).

## Composition

* SFT train pairs: 6,842,249; ordered directions with data: 5622 of 5852; with >= 1,000 pairs: 4104; >= 100 pairs: 4792. Per-direction counts: `direction_counts.csv`.
* Sampling: UniMax over the unordered language pairs (uniform share of the budget, at most 3 passes over a pair's own data), so no rich pair dominates; the cap per language pair is a few thousand sentence pairs (<= 3.9 k).
* Domain warning: most multi-way material comes from Bible / JW-style parallel text (see `provenance_kind`); government text (South African languages), news and community corpora are a small share.
* The matching packed token dataset (not in this repo): 8,205,636 samples, 1,691,160,438 Gemma tokens, seq_len 4096; directions {'en2x': 1033766, 'x2en': 1033766, 'x2x': 6138104}; prompt forms {'explicit': 4101804, 'implicit': 4103832}.

## RL prompts

`rl_prompts.parquet`: 509,493 prompts staged (before the private-source exclusion), one source-language window with a target tag, 5852 ordered directions (median 116 prompts per direction, minimum 3). Columns: `source`, `src_iso`, `iso` (target), `language`, `src_language`, `src_afr_tokens` (AfroXLM-R tokens), `length_bin` (b64 / b256 / b512 / b1024), `n_sentences`, `source_dataset`, `licence`, `licence_class`, `src_qe_valid`, `tgt_qe_valid`. Sources: real documents (HuggingFaceFW/fineweb-2, ODC-By-1.0; amh hau ibo pcm tir yor) and, for every language, text of the en->X corpus that is held out of the SFT pairs (neither the text nor its English sentence is in any SFT pair). Thin pools (kea, her, ish, wes, dua, oke, dyu, ibb) have only a handful of windows each and are reused many times.

## `full_pairs/`: the complete pair set (not a sample)

The files above are the SAMPLED training set (UniMax, a few thousand pairs per language pair). `full_pairs/` holds **everything we could derive**, sharded parquet (zstd), every row with `source_dataset`, `licence`, `licence_class`; the held-out evaluation sets and the rows of private source repositories are excluded exactly as above. One row per UNORDERED pair `{src_iso, tgt_iso}` (`src_iso` sorts before `tgt_iso`); a model uses both directions, so the number of ordered pairs is twice the number of rows. Nothing is sampled or thresholded: the quality signals are columns (`qe_*`, `lid_*`, `provenance_kind`, `score`), filter on them.

| folder | rows (unordered pairs) | ordered pairs | content |
|---|---:|---:|---|
| `full_pairs/pivot_en/<src_iso>/part-*.parquet` | 50,824,383 | 101,648,766 | two translations of the same English sentence (join of the en->X corpus on the normalised English text); `group_key` links to `full_pairs/pivot_en_sentences.parquet` (1,552,726 English sentences); `in_sft` marks the pairs selected for the sampled SFT files |
| `full_pairs/direct/direct_pairs.parquet` | 1,053,147 | 2,106,294 | pairs of the SFT corpus that already pair two African languages (53 % machine translation, `provenance_kind` tells) |
| `full_pairs/nllb_mined/<src>-<tgt>.parquet` | 105,658,159 | 211,316,318 | NLLB-v1 mined pairs republished as `michsethowusu/<lang>-<lang>_sentence-pairs` (526 shards; a few language pairs have a second shard `__<repo>` from a second repo). **No licence is declared on those cards; treated as non-commercial upstream**; raw mined data with a margin `score`, NOT filtered for quality or language |

Held-out removal in `full_pairs`: 931 en->X rows (their pairs) matched an evaluation sentence by the 8-gram rule on top of the exact rule applied earlier; private-source rows removed: 53,449. Mined pairs dropped for a held-out match: 17,621. Mined rows with an empty side are dropped (one repo, swahili-zulu, ships an empty Zulu column and is absent; `akan-twi` pairs a language with itself and is absent).

Pivot pairs by provenance kind: synthetic_mt 1,537,968, mixed 12,533,012, religious_jw 36,360,612, human_curated 392,791.

Licence distribution of `full_pairs/pivot_en` (declared `licence`, rows):

| licence | rows |
|---|---:|
| cc-by-4.0 | 35,219,023 |
| unknown + cc-by-4.0 | 10,905,665 |
| unknown | 2,420,013 |
| cc-by-4.0 + unknown | 905,031 |
| other + cc-by-4.0 | 496,014 |
| unknown + other | 229,563 |
| mit | 205,319 |
| other | 151,262 |
| other + unknown | 122,174 |
| cc-by-4.0 + other | 66,498 |
| mpl-2.0 | 40,419 |
| cc-by-sa-4.0 | 21,897 |
| mpl-2.0 + unknown | 15,638 |
| unknown + mit | 10,716 |
| unknown + mpl-2.0 | 5,880 |

`full_pairs/direct` licences: unknown 968,491, cc-by-4.0 84,637, other 19.

Sizes on disk: pivot_en 4.12 GB, direct 0.14 GB, nllb_mined 7.77 GB, pivot_en_sentences 0.09 GB.

## RL prompts v2 (more real documents)

Source windows now also come from open web corpora streamed from the Hub (HuggingFaceFW/fineweb-2, cis-lmu/GlotCC-V1, HPLT/HPLT2.0_cleaned, wikimedia/wikipedia 20231101, allenai/c4 multilingual), filtered by GlotLID, de-duplicated and held out from the SFT pairs and the benchmarks; real documents are preferred over pseudo-paragraphs of translated sentences. Windows kept per source language (b64 + b256 + b512 + b1024) before -> after:

| language | before | after |
|---|---:|---:|
| yor | 6,000 | 6,000 |
| hau | 6,000 | 6,000 |
| ibo | 6,000 | 6,000 |
| pcm | 4,979 | 5,579 |
| efi | 5,298 | 6,000 |
| twi | 6,000 | 5,564 |
| ewe | 3,865 | 6,000 |
| fon | 5,348 | 5,568 |
| wol | 4,674 | 5,874 |
| swh | 4,307 | 6,000 |
| kin | 5,588 | 6,000 |
| som | 5,566 | 5,994 |
| orm | 5,466 | 6,000 |
| amh | 6,000 | 6,000 |
| tir | 6,000 | 6,000 |
| lug | 4,477 | 6,000 |
| lin | 3,841 | 5,907 |
| zul | 5,313 | 4,727 |
| xho | 5,287 | 6,000 |
| afr | 4,391 | 6,000 |
| sna | 5,322 | 2,220 |
| nya | 5,509 | 4,254 |
| sot | 5,685 | 5,768 |
| tsn | 5,307 | 6,000 |
| mlg | 5,259 | 5,887 |
| ibb | 150 | 3,511 |
| urh | 358 | 721 |
| bem | 3,839 | 3,306 |
| tum | 2,286 | 4,747 |
| run | 3,245 | 6,000 |
| ven | 2,417 | 4,691 |
| tso | 4,937 | 6,000 |
| nso | 5,419 | 6,000 |
| ssw | 1,693 | 5,024 |
| nbl | 5,179 | 5,911 |
| sag | 3,095 | 6,000 |
| tiv | 2,755 | 3,855 |
| umb | 2,300 | 3,250 |
| kmb | 1,726 | 2,678 |
| mos | 2,932 | 4,122 |
| kik | 1,071 | 3,758 |
| luo | 4,589 | 4,968 |
| kam | 388 | 1,291 |
| bin | 3,299 | 3,487 |
| dag | 3,521 | 4,273 |
| oke | 95 | 139 |
| shi | 617 | 946 |
| dyu | 102 | 2,781 |
| bam | 1,065 | 4,086 |
| kbp | 434 | 4,362 |
| bbj | 520 | 526 |
| kon | 139 | 2,013 |
| kea | 8 | 2,547 |
| kab | 4,101 | 6,000 |
| nus | 5,508 | 5,582 |
| swc | 4,740 | 6,000 |
| lua | 3,953 | 5,198 |
| kqn | 3,176 | 3,424 |
| lun | 1,579 | 1,580 |
| iso | 3,469 | 4,282 |
| loz | 5,359 | 5,833 |
| wal | 1,858 | 3,896 |
| ach | 1,033 | 2,294 |
| tsc | 1,313 | 1,537 |
| vmw | 1,315 | 1,871 |
| kua | 1,153 | 3,946 |
| seh | 695 | 1,363 |
| nyn | 708 | 1,747 |
| her | 22 | 1,063 |
| yao | 457 | 742 |
| sid | 249 | 931 |
| ndc | 407 | 909 |
| kri | 651 | 728 |
| ish | 24 | 68 |
| dua | 67 | 277 |
| bas | 271 | 474 |
| wes | 35 | 54 |

## Prompt forms (SFT)

`prompt_form` (`explicit` | `implicit`) is a deterministic function of the pair (`blake2b` of `group_key|src_iso|tgt_iso`, 50 % each): `explicit` = `Translate the text from {source} to {target}:  {text}`, `implicit` = `Translate the text to {target}:  {text}` (the en->X-compatible form). The answer is always `{target tag} {translation}`.

## Filters

English sentence must be English and the African side must pass GlotLID for its language (where LID is reliable); per-language floor on the AfriCOMET-QE of the English -> X translation (30th percentile on top of the build's 15 % gate); length-ratio window per language pair; no copy of the source (identical, or word-set Jaccard >= 0.9); script check (Ethiopic languages); URL / markup debris removed; de-duplication on the text pair; validation = whole English-sentence groups.

## Held-out evaluation data (NOT in this dataset)

Nothing from these sets is in any file, by exact match after normalisation (any language) and by word-8-gram overlap: FLORES+ dev and devtest, NTREX-128, Google SMOL (SmolSent, SmolDoc, GATITOS), MAFAND-MT test, IbomMT, FLORES-MOZ, GlotStoryBook, SALT test, and the documents of the project's long-document benchmark. They are the benchmarks of the models trained on this data.

## Private / personal data

No personal data is knowingly included. Rows taken from source repositories that are flagged *private* in the build's licence scan were **not** staged for this repository (45,024 rows).

## Sources (`source_dataset`, rows in the SFT training pairs)

| source_dataset | rows |
|---|---:|
| dsfsi/govza-sa-cabinet-statements-sentence-aligned | 101,976 |
| Tadesse/COMET_score | 53,761 |
| michsethowusu/african-bible-parallel | 43,785 |
| dsfsi/afriscience_mt | 35,610 |
| dsfsi/vukuzenzele-sentence-aligned | 27,571 |
| <HF_ORG_LEGACY>/sft-new | 24,272 |
| MaLA-LM/PolyWrite | 22,729 |
| UBC-NLP/AfroLingu-MT | 19,542 |
| <HF_ORG_LEGACY>/naija2 + <HF_ORG_LEGACY>/sft-new | 18,031 |
| Okwu/african-language-parallel-corpus | 9,528 |
| ayymen/Pontoon-Translations | 9,216 |
| Mehgoss/sa-languages-translation | 8,752 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-lozi_sentence-pairs_mt560 | 8,381 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-luba-kasai_sentence-pairs_mt560 | 8,164 |
| <HF_ORG_LEGACY>/naija2 | 7,842 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-isoko_sentence-pairs_mt560 | 7,813 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-kaonde_sentence-pairs_mt560 | 7,593 |
| <HF_ORG_LEGACY>/sft-new + BrianMwangi/english-to-luo-opus | 7,571 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-wolaytta_sentence-pairs_mt560 | 6,952 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-lunda_sentence-pairs_mt560 | 6,890 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-dholuo_sentence-pairs_mt560 | 6,788 |
| <HF_ORG_LEGACY>/sft-new + michsethowusu/english-lozi_sentence-pairs_mt560 | 6,714 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-sango_sentence-pairs_mt560 | 6,678 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-oshiwambo_sentence-pairs_mt560 | 6,645 |
| <HF_ORG_LEGACY>/naija2 + michsethowusu/english-luba-kasai_sentence-pairs_mt560 | 6,594 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-venda_sentence-pairs_mt560 | 6,571 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-moore_sentence-pairs_mt560 | 6,547 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-tiv_sentence-pairs_mt560 | 6,461 |
| <HF_ORG_LEGACY>/sft-new + michsethowusu/english-wolaytta_sentence-pairs_mt560 | 6,421 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-congo-swahili_sentence-pairs_mt560 | 6,395 |
| <HF_ORG_LEGACY>/naija2 + michsethowusu/english-lozi_sentence-pairs_mt560 | 6,340 |
| McGill-NLP/SSA-MTE | 6,294 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-sidaama_sentence-pairs_mt560 | 6,289 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-yao_sentence-pairs_mt560 | 6,257 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-northern-sotho_sentence-pairs_mt560 | 6,224 |
| <HF_ORG_LEGACY>/naija2 + michsethowusu/english-isoko_sentence-pairs_mt560 | 6,183 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-okpe_sentence-pairs_mt560 | 6,106 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-basaa_sentence-pairs_mt560 | 6,099 |
| <HF_ORG_LEGACY>/naija2 + michsethowusu/english-kaonde_sentence-pairs_mt560 | 5,941 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-acholi_sentence-pairs_mt560 | 5,915 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-tswa_sentence-pairs_mt560 | 5,890 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-swati_sentence-pairs_mt560 | 5,873 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-nyankore_sentence-pairs_mt560 | 5,873 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-tumbuka_sentence-pairs_mt560 | 5,822 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-makhuwa_sentence-pairs_mt560 | 5,800 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-sena_sentence-pairs_mt560 | 5,783 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-gikuyu_sentence-pairs_mt560 | 5,753 |
| <HF_ORG_LEGACY>/sft-new + michsethowusu/english-kaonde_sentence-pairs_mt560 | 5,715 |
| <HF_ORG_LEGACY>/naija2 + michsethowusu/english-congo-swahili_sentence-pairs_mt560 | 5,711 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-rundi_sentence-pairs_mt560 | 5,684 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-ganda_sentence-pairs_mt560 | 5,648 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-krio_sentence-pairs_mt560 | 5,600 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-umbundu_sentence-pairs_mt560 | 5,587 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-bemba_sentence-pairs_mt560 | 5,467 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-malagasy_sentence-pairs_mt560 | 5,434 |
| <HF_ORG_LEGACY>/sft-new + michsethowusu/english-congo-swahili_sentence-pairs_mt560 | 5,399 |
| <HF_ORG_LEGACY>/naija2 + michsethowusu/english-setswana_sentence-pairs_mt560 | 5,392 |
| <HF_ORG_LEGACY>/naija + michsethowusu/english-duala_sentence-pairs_mt560 | 5,344 |
| <HF_ORG_LEGACY>/naija2 + michsethowusu/english-venda_sentence-pairs_mt560 | 5,338 |
| <HF_ORG_LEGACY>/sft-new + michsethowusu/english-luba-kasai_sentence-pairs_mt560 | 5,258 |

The full list is in `MANIFEST.json`.

## Intended use and limits

Training and evaluating translation between African languages. Quality varies a lot by pair: pivot pairs inherit the noise of both English translations, `religious_jw` text is domain-narrow, `synthetic_mt` rows are machine translations (capped at 15 % per language pair in the build). No pair is human-verified at sentence level by us.
