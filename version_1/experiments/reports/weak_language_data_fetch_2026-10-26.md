# Fetch and assembly of new commercially usable English-parallel data for the weak languages (agent F1, 2026-10-26)

Output: `data/weak_parallel_2026-10-26/` (local, git-ignored, 0.6 GB) and the PRIVATE Hub dataset `<HF_ORG_LEGACY>/sabiyarn-weak-language-parallel-2026-10-26` (private=True verified after creation). Card: `README.md` in the folder; every source with licence: `SOURCES.md`; counts after every filter, tokens and sha256: `MANIFEST.json`. Scripts: `data/weak_parallel_2026-10-26/_work/code/` (raw caches in `_work/raw`, not uploaded; can be deleted).

## 1. Per-language result (after all filters; tokens = target side, Gemma-3 tokenizer)
| iso | permissive | share-alike | human total | target tok (human) | synthetic-EN | target tok (synth) | quarantine |
|---|---|---|---|---|---|---|---|
| amh | 2,920 | 2,683 | 5,603 | 0.35M | 40,000 | 15.71M | 0 |
| bam | 0 | 7,074 | 7,074 | 0.39M | 0 | 0.00M | 0 |
| dyu | 0 | 897 | 897 | 0.04M | 0 | 0.00M | 0 |
| efi | 415 | 0 | 415 | 0.01M | 0 | 0.00M | 0 |
| ewe | 1,137 | 1,728 | 2,865 | 0.12M | 0 | 0.00M | 0 |
| fon | 525 | 1,002 | 1,527 | 0.08M | 0 | 0.00M | 0 |
| ibb | 685 | 0 | 685 | 0.01M | 0 | 0.00M | 0 |
| kab | 59,474 | 1,943 | 61,417 | 0.94M | 20,000 | 7.46M | 0 |
| kam | 9 | 897 | 906 | 0.04M | 0 | 0.00M | 0 |
| kbp | 566 | 897 | 1,463 | 0.09M | 0 | 0.00M | 0 |
| kea | 6,025 | 897 | 6,922 | 0.14M | 0 | 0.00M | 1,901 |
| kmb | 0 | 897 | 897 | 0.04M | 0 | 0.00M | 0 |
| knc | 8,529 | 7,079 | 15,608 | 0.66M | 307 | 0.10M | 0 |
| knc_Arab | 0 | 7,069 | 7,069 | 0.64M | 0 | 0.00M | 0 |
| luo | 0 | 897 | 897 | 0.04M | 4,121 | 1.55M | 0 |
| mos | 4,594 | 897 | 5,491 | 0.17M | 0 | 0.00M | 0 |
| nus | 3,163 | 7,082 | 10,245 | 0.80M | 0 | 0.00M | 0 |
| run | 2,062 | 1,038 | 3,100 | 0.08M | 60,000 | 21.80M | 0 |
| seh | 0 | 897 | 897 | 0.04M | 0 | 0.00M | 0 |
| tir | 13,813 | 2,190 | 16,003 | 0.82M | 60,000 | 23.17M | 3,790 |
| tum | 1,051 | 2,531 | 3,582 | 0.17M | 0 | 0.00M | 0 |
| umb | 15 | 897 | 912 | 0.04M | 0 | 0.00M | 0 |
| vmw | 502 | 897 | 1,399 | 0.06M | 0 | 0.00M | 0 |
| wol | 1,528 | 1,090 | 2,618 | 0.08M | 0 | 0.00M | 0 |
| yor | 5,339 | 897 | 6,236 | 0.18M | 0 | 0.00M | 0 |

Totals: permissive 112,352 pairs (2.48M target tokens), share-alike 52,376 (3.56M), synthetic-English-side 184,428 (69.8M), quarantine 5,691 (0.16M). Human, train-ready (permissive + share-alike): 164,728 pairs, 6.0M tokens. Each language has at most the FLORES+ dev sample of 897 pairs (about 0.04M tokens) where nothing else exists: dyu, kmb, seh, umb (+15 Tatoeba), kam (+9), luo (+0 human). `knc_Arab` is the Ajami-script file. Included regardless of size, as the owner asked on 2026-10-26.

## 2. What each source gave (kept pairs)
Gamayun tig 4,994 / kau 4,993; TICO-19 12,193 (tir, nus, knc); Tatoeba 52,927 (kab 59k raw block, 47,540 kept pairs flagged "possible overlap with SFT"); translatewiki 37,245 (short UI strings); OLDI Seed knc Latn 6,178, knc Arab 6,173, bam 6,177, nus 6,185 (bam and nus were already built by R10); FLORES+ dev 20,629 (23 files, W1's 100 ids removed from each); Wikimedia 6,188; Travis Wikipedia block 846 (share-alike) and 3,790 (quarantine); frgr3618 kea 1,901 (quarantine, NLLB-derived); FineTranslations 184,428 (tir 60,000, run 60,000, amh 40,000, kab 20,000, luo 4,121, knc 307; jw.org documents removed).

## 3. Filters (all languages, per class; full per-language counts in MANIFEST.json)
| step | permissive | share-alike | synthetic | quarantine |
|---|---|---|---|---|
| raw candidates | 131,522 | 53,065 | 298,020 | 6,091 |
| target == source | 718 | 33 | 2,489 | 1 |
| length ratio outside 0.25-4 | 577 | 239 | 60 | 5 |
| over 400 words | 1 | 4 | 32 | 0 |
| MT artifact ("Here is ...") | 0 | 0 | 131 | 0 |
| empty | 0 | 0 | 25 | 0 |
| exact pair duplicate | 5,400 | 166 | 144 | 193 |
| near duplicate (same normalised target) | 11,950 | 163 | 4,671 | 146 |
| near duplicate (MinHash 0.8, targets of 12+ words) | 45 | 11 | 2,440 | 8 |
| English side exact match to an evaluation set | 379 | 45 | 4 | 40 |
| English side 8-gram overlap with an evaluation set | 100 | 28 | 2,044 | 7 |
| FineTranslations cap | - | - | 101,552 | - |
| **final** | **112,352** | **52,376** | **184,428** | **5,691** |

Removals by evaluation pool, all classes (exact / 8-gram): SmolDoc 260 / 724, MAFAND test 75 / 678, MAFAND dev 19 / 580, GlotStoryBook 93 / 59, AfriScience-MT test 0 / 43 (dev 0 / 3), NTREX-128 10 / 35, SSA-MTE 0 / 27, FLORES+ devtest 0 / 13, FLORES+ dev 8 / 12, SALT dev 2 / 0 (test 0 / 3), SmolSent 1 / 2. IbomMT, FLORES-MOZ, AfriMTE and W1's held-out dev text matched nothing outside the FLORES+ dev rows. Post-hoc check on the written files: 0 exact matches against any pool (354,847 records). Evaluation sets read: FLORES+ dev/devtest English (gated, token had access), NTREX-128, MAFAND-MT dev/test (all en-* and fr-* pairs), SmolSent, SmolDoc (all xx-en files), SALT text-all dev/test (gated, access OK), GlotStoryBook, FLORES-MOZ, IbomMT (efi, ibb dev/test), AfriScience-MT dev/test, SSA-MTE dev/test, AfriMTE-WMT2024 English sides. Nothing was unreadable. Also removed: pairs whose target equals a FLORES+ devtest target of the same language.

## 4. Findings that change the D1 picture
- **Travis Foundation Tigrinya: only 846 kept pairs are usable, not about 5,000.** The first ~5,000 pairs are a mix: pairs 0-908 are conversation prompts and English-Wikipedia snippets (the crowd-sourced block), from ~909 on they are TED-talk-like transcripts (TED is CC BY-NC-ND), app UI strings and other text. The 909-4,999 range (3,790 kept) is in `quarantine/tir.jsonl`.
- **Kabyle Tatoeba is probably already in the SFT.** No manifest on disk lists Tatoeba for kab, but `taqbaylit/tatoeba-en-kab` (245k rows) is in the original SFT build list, and 92.9% of OPUS Tatoeba en-kab pairs are identical to it (English 98.5%). The 47,540 pairs that survive are flagged per record; kab's 59,474 permissive pairs are therefore likely not new.
- FLORES+ dev English equals the English of IbomMT dev/test and FLORES-MOZ dev, so strict removal against those pools would erase all 897-pair samples. I kept the dev rows (owner allows dev) and exempted them from those three pools only; deleting `source_id == flores_plus_dev` removes them cleanly.
- luo: 1,010 of 2,692 FineTranslations documents are jw.org (not 15%), removed; 4,121 chunk pairs remain.
- R10's pool audit marks Tatoeba nus (85% used), kam Tatoeba and Wikimedia fon/tir as USED by the SFT; they are kept but carry the flag in `split_note`.

## 5. Still missing
No new data at all: urh, ish, bin, oke, tsc, tiv, bbj. Only the 897-pair FLORES+ dev sample (0.04M tokens): dyu, kmb, seh, umb, kam. Under 0.15M human tokens: kbp 0.09M, vmw 0.06M, efi 0.01M, ibb 0.01M, wol 0.08M, run 0.08M, fon 0.08M, ewe 0.12M, tum 0.17M, kea 0.14M (plus 1.9k quarantined). The only languages with 0.5M+ human tokens: tir 0.82M, kab 0.94M (probably in SFT), nus 0.80M, knc 0.66M (+0.64M Ajami). Volume beyond that exists only as synthetic-English FineTranslations (tir, run, amh, kab, luo, knc). Not fetched: TICO-19 amh (original release only), Wikimedia yor, pivot sets (Fon/Ewe-French, Kimbundu-Portuguese), and everything D1 marked unstated or NC.

## 6. Open decisions for the owner
1. FLORES+ dev rows in train_ready (20,629): keep (owner rule) or drop for strict dev exclusion.
2. FineTranslations: Gemma terms of use, unvetted web-text copyright and machine-made English side; also whether jw.org-free is enough.
3. Kab Tatoeba: confirm on the box whether `tatoeba_mt_parquet` / taqbaylit rows are in the pool; if so, drop.
4. Travis quarantine (3,790) and frgr3618 kea (1,901): stay quarantined unless the owner clears them.
5. translatewiki UI strings (37k, CC BY 3.0, upstream project licences differ): accept or move to quarantine.
