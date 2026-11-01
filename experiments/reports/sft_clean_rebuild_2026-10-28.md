# Clean English -> African SFT pairs: rebuild with a hard decontamination filter (agent D2, 2026-10-28)

Step 1 of the rerun (SFT -> RL -> science stage on the same sources, protected evaluation text removed). Nothing was trained, no GPU was used, no paid API. Work ran on box 2 (vast.ai, RTX 3060 box shared with the translation farm): 4 CPU cores pinned with `taskset`, `nice -n 19`, `ionice -c3`, processes of 2.5-5.5 GB resident memory each (most of it the shared read-only index), at most about 13 GB disk in `/workspace/work` at the peak (the 20 GB limit was not reached), sequential downloads. The farm, its queues, tmux sessions and the chess job were not touched. The Hub token was read from the farm environment inside my own processes only (never printed or written).

Scripts: `data_pipeline/sft_clean/` (also on the Hub in `scripts/`). Data tables: `reports/data/sft_clean_2026-10-28/`. Output repo (private): `<HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28`.

## 0. Headline

* **Done:** protected index (22 source datasets in 35 split-level sets, 45.6 M protected sentences added before dedupe, 2.99 M distinct sentence hashes, 42.7 M distinct word-8-gram hashes), the filter (unit-tested against the reference implementation), a rebuild from every source that is reachable with the owner's tokens, the exposure proof, the licence tables. **Not done:** a rebuild at the old coverage. The rebuild is **incomplete by construction** (section 1).
* **Output:** 23,406,354 unique en->X pairs in 76 language tags, estimated 1,171,995,571 target tokens (old build: 1,318,270,369 Gemma target tokens in 538,480 packs; not like for like, see 4). **268,020 unique pairs (1.13%) were removed by the filter** out of 23,674,374 that passed the old `ok()` filter.
* **Loss per language is small on the reachable sources: maximum 2.62% (Ewe), no language loses more than 20%.** Do not read this as the cost the old build would have had: the sources that carried FLORES+ into the old text (the private naija/tds/clean_data repositories) are not reachable, so their share of the old loss cannot be measured (section 1, section 6).
* **Languages the lead has to decide on** are the ones below the 2,000-pair minimum or with large coverage gaps (section 4): below 2,000 pairs after the filter: **ibb (998), men (1,621), nbl (188)** and no pairs at all for **nus, dag, bbj** (all three had old data); big coverage gaps against the old token counts: **pcm 5.4%, twi 14.0%, fon 14.5%, ibo 21.4%, bin 27.4%, ibb 29.5%, kab 35.7%, bam 37.3%, hau 41.2%, ewe 51.4%, yor 56.2%** (old Pidgin, Twi, Fon, Igbo, Hausa data sat in unreachable private repositories).
* **Exposure check (mandatory proof): 0 hits** for every protected set on both sides in a 1,001,020-pair sample (2,002,040 texts) of the output, by three methods including the audit's own `contamination.py` logic (section 7).
* **Licence:** the only deliberate non-commercial source is IbomMT train (1,992 pairs, `licence=non_commercial`, owner decision). Two sources were put in `quarantine_nc/`, not the main output (section 8).

## 1. Step 0 recon: what can be rebuilt (written before any large download)

The old chain was `final_v2` (60.7 M rows, 81.6 M source rows before dedupe in `provenance.json`) -> `gemma_sft_pairs.py` -> `gemma_merge_pairs.py` -> QE -> gate -> windows. `final_v2` itself is not on the Hub. The two public repositories that look like it are not: `<HF_ORG>/sft-new` (= `<HF_ORG_LEGACY>/sft-new`, 4.5 GB, 5.6 M rows) is only the first build (multi-open, finetranslations, opus100 translation groups and task data), `<HF_ORG_LEGACY>/sft-all` (1.8 GB) holds those same rows plus small task sets; neither holds the large translation carriers. I checked the 339 repositories named in the old licence scan, provenance and extras lists against the Hub with the owner's token (`source_access_2026-10-28.json`): **14,132,899 of 81,634,859 old source rows (17.3%) are in repositories that still resolve; 67,501,960 rows (82.7%) are in repositories that return 404 for both tokens** (older accounts, deleted or renamed). The largest:

| source repo | old-build rows | what it carried (old build step) |
|---|---:|---|
| <HF_ORG_LEGACY>/naija2 | 23,267,072 | naija2@Business_Knowledge_Dataset_Llama_3.1_Short_Token_Pairs_english_prompt-response|naij |
| <HF_ORG>/tds | 10,796,903 | tds |
| <HF_ORG_LEGACY>/naija_sft | 8,485,343 | old_eng_sft |
| <HF_ORG>/aya_interlanguage_translate | 5,874,377 | aya_interlanguage |
| <HF_ORG_LEGACY>/naija | 4,570,400 | naija@AfroLingu-MT_ibo_prompt-response|naija@Ehn-bible-bbc-gpt3.5_english_translation|naij |
| <HF_ORG_LEGACY>/eng_sft | 2,685,871 | old_eng_sft |
| <HF_ORG_LEGACY>/african_sft | 2,685,871 | old_african_sft |
| <HF_ORG>/processed_data | 1,372,380 | processed_data_tagged |
| <HF_ORG>/clean_data | 1,211,431 | clean_data_tagged |
| <HF_ORG>/african_languages_translation | 1,027,940 | african_languages_translation |
| <HF_ORG>/english_to_igbo | 812,058 | english_to_igbo |
| <HF_ORG>/clean_data2 | 767,801 | clean_data2_tagged |
| <HF_ORG>/translation | 756,361 | translation_tagged |
| <HF_ORG_LEGACY>/Clean_translate | 703,648 | clean_translate_tagged |

Full list: `source_reachability_old_vs_rebuilt.csv`. Substitutions made (public equivalents, NOT the same files): the 99 public `michsethowusu/english-<language>_sentence-pairs_mt560` repositories (cards: OPUS MT560, cc-by-4.0, no NLLB mention) stand in for the mt560 parts of the private bundles; plus the 46 CC-BY extras repositories of `gemma_extra_ccby.json` that are real English pair sets (the audio/ASR and French-source ones were skipped), KwekuAinoo/eng_twi, AfroLingu-MT train (gated, access worked), Pontoon, Vukuzenzele, the two Wolof sets, Helsinki-en-fon, and (owner additions) SSA-MTE train and IbomMT train. **Not recovered at all:** Tatoeba-MT train (10.9 M kept rows in the old build; the data are not on the Hub), `<HF_ORG>/tds`, `clean_data*`, `processed_data`, `translation`, `african_languages_translation`, `english_to_igbo`, `WebScraped_Igbo_To_Eng`, `Clean_translate`, the naija bundles (JW300 Igbo, Pidgin MTurk, mt560 for most languages), and the NLLB/Aya-derived sets that are excluded by rule anyway. Old per-language pair counts cannot be given (`pairs_final` was never backed up); the comparison below uses the old windows and Gemma target tokens of the training plan.

## 2. Protected index (heart of the task)

Built by `fetch_protected.py` (one file at a time: download -> index -> delete; benchmark text never stored in the repository, the Hub or a cache; a local normalised dump existed on the box only for the independent exposure check and was deleted afterwards). Every string cell of every selected file counts, so multi-way sets are covered on all sides. Version `protected-2026-10-28.1`, file sha256 `f094fd2b24d6a0e469d635abed042e4f05335f93a3e96400f66afe51db3919ad`. The sealed manifest (`protected_manifest.json`, also in `protected_index/` on the Hub) lists repos, commits and counts; the index file `protected_index.npz` (2.3 GB) and `protected_sentence_hashes.npz` (hashes only) are in `protected_index/` on the Hub; reusable for RL prompts, the science replay mix and calibration sets with `protected_index_loader.py` (`--seal`, `--verify`, a JSONL filter CLI, `load()`/`check_rows()` API; it refuses a file whose sha256 differs from the manifest).

| set | source | commit | files | sentences added (before dedupe) by split |
|---|---|---|---:|---|
| flores_plus | openlanguagedata/flores_plus | 1147e2dc59 | 461/461 | {'dev': 232298, 'devtest': 230736} |
| ibom_mt | howard-nlp/ibom-mt | 90d73874e2 | 8/8 | {'dev': 7973, 'test': 8096} |
| ibom_mt_efi_davlan | Davlan/ibom-mt-en-efi | 13ddfb293a | 2/2 | {'dev': 3998, 'test': 3990} |
| flores_moz | MOZNLP/FLORES_MOZ | 751dd12a6b | 10/10 | {'dev': 4985, 'devtest': 5060} |
| ntrex128 | davidstap/NTREX | fd20d54141 | 128/128 | {'all': 255397} |
| ntrex_african | masakhane/ntrex_african | 6a0e89ef2c | 32/32 | {'all': 127803} |
| smol | google/smol | fdaff3a1a0 | 867/867 | {'all': 3745884} |
| salt | Sunbird/salt | a0ffb52b49 | 2/2 | {'dev': 5456, 'test': 5500} |
| glot_story_book | cis-lmu/GlotStoryBook | ad913a24f4 | 2/2 | {'all': 469205} |
| afriscience_mt | dsfsi/afriscience_mt | 5e7b8238d5 | 2/2 | {'dev': 13105, 'test': 14971} |
| afrilongdoc | <HF_ORG>/AfriLongDoc | d8219803f3 | 99/99 | {'all': 1893039} |
| tico19 | gmnlp/tico19 | 55d70dc0b1 | 76/76 | {'dev': 158956, 'test': 342391} |
| ssa_mte | McGill-NLP/SSA-MTE | 504e077bf3 | 32/32 | {'dev': 41693, 'test': 161182} |
| afrimte_wmt2024 | masakhane/AfriMTE-WMT2024 | 27b3da428a | 26/26 | {'all': 5630} |
| wmt24pp | google/wmt24pp | fd7405c064 | 3/3 | {'all': 15315} |
| sib200 | Davlan/sib200 | 38977a667f | 410/410 | {'dev': 32185, 'test': 66215} |
| sib200_ibom | howard-nlp/sib200-ibom | b7a94ad9f7 | 10/10 | {'dev': 785, 'test': 1615} |
| xsimplusplus | jaygala24/xsimplusplus | 67d7396578 | 408/408 | {'dev': 18506079, 'devtest': 18904578} |
| afridocmt | masakhane/AfriDocMT | 8ae06a3f73 | 24/24 | {'dev': 64828, 'test': 128885} |
| ssa_mt | McGill-NLP/SSA-MT | 7db72bed29 | 0/0 | {} |
| naijas2st_news_mt | McGill-NLP/NaijaS2ST-news-MT | 4f2eca153e | 8/8 | {'test': 6000, 'validation': 6000} |
| mafand | github.com/masakhane-io/lafand-mt (Hub masakhane/mafand is a loading script that points there) | c1a4235007 | 42/42 | {'dev': 51222, 'test': 58212} |

Sets of the lead's list that are **not** in the index, and why: **WMT22 African** (`allenai/wmt22_african` is a loading script that points to external files; the validation/test text is FLORES-101, which FLORES+ covers except for the FLORES+ revisions), **AfroBench MT tasks** (no such repository on the Hub; its MT tasks reuse FLORES, MAFAND, NTREX and SALT, which are protected), **SSA-MT** (`McGill-NLP/SSA-MT` has no split-named files; nothing indexed). Added beyond the list: AfriDocMT, NaijaS2ST-news-MT, `ntrex_african`, `Davlan/ibom-mt-en-efi`, `howard-nlp/sib200-ibom`. **MAFAND** is a loading script on the Hub; the data were fetched from `github.com/masakhane-io/lafand-mt` (dev and test of every pair incl. French-source). FLORES+ includes all 461 dev/devtest files (all variants). xsim++ adds 37 M perturbed sentence strings (many repeated per language; 2.99 M distinct sentence hashes in total).

Filter = `decon.py`: rules R1 exact (normalised text or one of its sentence parts equals a protected sentence of at least 3 tokens), R2 contiguous (a protected sentence of at least 8 tokens occurs inside the text as a whole-token run), R3 shared word 8-gram (strict rule of the lead; the reference's "at least 50% of the 8-grams" is implied and also reported), each on two normal forms (NFKC + quote fold + casefold + non-alphanumerics -> space; and the same after dropping combining marks). Both sides of every pair are tested; any hit on either side drops the pair. Hashing: xxh64 of normalised sentences/tokens, 8-gram hash = position-mixed combination of token hashes, lookups by `np.searchsorted` on sorted uint64 arrays with set bitmasks (R2 is verified on the full token run of the rare first-gram hits). **Unit test** `data_pipeline/sft_clean/test_decon.py`: 623 texts (22 hand-planted cases: exact, sentence part of a paragraph, contained sentence, shared 8-gram only, NFD and under-dot variants, diacritic-free copy, quote variants, full-width, short exact-only texts, empty text, negatives; plus 600 random texts spliced from 300 protected random sentences) give **identical per-rule set masks and identical drop decisions** as `contam_audit_12_decontaminate.Protected.check`; the reference class's decisions were the oracle. The test passes (`all tests passed`). It also covers index save/load and `Builder.extend`. Speed on the box: about 1,000 (long passages) to 9,000 (short sentences) pairs per second with 3 workers.

## 3. Pipeline as run

`build_pairs.py`, 183 input units (one per input file), 181 finished, 2 failed (below). Per unit: read -> old `NOT_AFRICAN` / flip logic (only for the processed `sft-new` rows; raw sources are already English/target columns) -> source quarantine (benchmark-named sources dropped; none of the 183 units triggered it) -> canonical iso (`gemma_mt.languages.canonical_iso`; tags outside the table dropped) -> licence class (old `license_scan.json` logic; `non_commercial` dropped unless owner-allowed) -> NLLB/TranslatePsy/Aya exclusion by name and by card scan -> whitespace collapse + the old `ok()` filter (unchanged) -> decontamination -> shards. Global exact dedupe with the old key (English casefold, iso, target casefold; xxh64 instead of blake2b): 205,816 duplicates removed. Rows read 25,320,178; failing `ok()` 833,447.

**Deviations from the old chain (all would need work or a GPU):**
1. **No AfriCOMET-QE and no gate** (lowest 15% per language, QE >= 0.20, min 2,000 rows). QE needs the GPU. **Every one of the 23,406,354 output pairs would have needed QE scoring** (the old build scored the full set). The 2,000-row minimum is applied to the post-filter counts only to flag languages.
2. **No GlotLID / aligned-names / degenerate-text checks** that the old converters applied before the pairs existed (the raw sources went through `converters.py` in the old build; I did not port its fasttext dependency). The `sft-new` rows already carry those checks; the mt560 and other raw sources do not. English-side check only for SSA-MTE (cheap function-word test, because SSA-MTE has French and Portuguese originals).
3. No token-balanced windowing, no packing.
4. The mt560 set is a public equivalent of the private bundles; the old build also QE-vetted it (the old `mt560_qe` converter kept only pairs with QE >= 0.46 for fon/oromo/setswana). The converter docstring calls these repositories "NLLB-derived" while the repository cards say OPUS MT560; **owner to confirm before they are used** (section 9).
5. Failed units: `ibom_train:davlan_efi` (JSON layout not read; the howard-nlp efi train file was used), `pontoon:so` (no such file). The first run of the 58 GB `synthetic-parallel-external` repository was killed after it stalled and re-run with 1 MB range reads (15 minutes).

## 4. Result per language: old vs rebuilt before/after the filter

Old = training-plan windows and Gemma target tokens of the final SFT (after QE gate and windowing). Rebuilt = pairs and **estimated** target tokens (target characters divided by characters-per-token measured on 1,500 random output targets per language with the Gemma-3-270m tokenizer; `chars_per_token.json`). The rebuilt numbers are before QE and before windowing, so ratios above 100% mean "more raw pairs than the old gated text", not "better": for many languages the public mt560 repositories contain more pairs than the old gated text (the old QE gate and converters removed an unknown share). Languages with ratios far below 100% are the real coverage gaps. Flags: BELOW_2000_PAIRS, LOSS_GT_20PCT (none), COVERAGE_LT_20PCT_OF_OLD_TOKENS. CSV: `comparison_old_vs_rebuilt.csv`.

| iso | language | old windows | old target tokens | pairs before | pairs after | removed % | est. tokens after | after / old tokens % | flags |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| yor | Yoruba | 1,519,727 | 145,155,296 | 980,589 | 965,223 | 1.6 | 81,649,717 | 56.2 | |
| afr | Afrikaans | 298,234 | 33,190,969 | 2,139,436 | 2,127,231 | 0.6 | 80,566,155 | 242.7 | |
| zul | Zulu | 229,980 | 33,585,226 | 1,155,000 | 1,142,047 | 1.1 | 72,501,042 | 215.9 | |
| xho | Xhosa | 367,026 | 46,705,118 | 1,404,133 | 1,394,691 | 0.7 | 72,016,298 | 154.2 | |
| nya | Chichewa | 129,583 | 20,969,998 | 925,651 | 916,632 | 1.0 | 57,980,710 | 276.5 | |
| hau | Hausa | 1,155,980 | 135,024,789 | 693,168 | 680,211 | 1.9 | 55,647,149 | 41.2 | |
| ibo | Igbo | 2,964,219 | 260,311,732 | 674,824 | 664,283 | 1.6 | 55,821,715 | 21.4 | |
| swh | Swahili | 279,040 | 35,364,489 | 910,206 | 903,245 | 0.8 | 52,552,833 | 148.6 | |
| sna | Shona | 120,779 | 15,848,709 | 908,616 | 900,169 | 0.9 | 51,160,457 | 322.8 | |
| som | Somali | 193,714 | 37,763,040 | 374,378 | 371,791 | 0.7 | 50,481,207 | 133.7 | |
| mlg | Malagasy | 84,787 | 10,078,629 | 888,230 | 880,712 | 0.8 | 37,589,270 | 373.0 | |
| sot | Sesotho | 130,669 | 24,842,204 | 176,346 | 172,233 | 2.3 | 32,484,670 | 130.8 | |
| tsn | Setswana | 332,597 | 34,209,543 | 846,182 | 831,499 | 1.7 | 33,048,152 | 96.6 | |
| tso | Tsonga | 84,895 | 8,948,175 | 762,877 | 755,549 | 1.0 | 31,630,102 | 353.5 | |
| luo | Luo | 171,483 | 17,146,609 | 1,532,282 | 1,514,791 | 1.1 | 30,289,650 | 176.7 | |
| amh | Amharic | 59,035 | 8,289,388 | 664,987 | 657,412 | 1.1 | 26,859,262 | 324.0 | |
| kin | Kinyarwanda | 183,557 | 28,839,951 | 171,692 | 170,159 | 0.9 | 24,596,369 | 85.3 | |
| twi | Twi | 537,664 | 165,853,001 | 669,301 | 663,535 | 0.9 | 23,254,010 | 14.0 | COVERAGE_LT_20PCT_OF_OLD_TOKENS |
| nso | Sepedi | 100,637 | 10,464,494 | 554,615 | 546,806 | 1.4 | 22,339,715 | 213.5 | |
| lin | Lingala | 37,254 | 3,793,007 | 531,975 | 526,093 | 1.1 | 19,260,461 | 507.8 | |
| orm | Oromo | 111,026 | 14,483,322 | 226,738 | 221,086 | 2.5 | 16,610,522 | 114.7 | |
| tir | Tigrinya | 48,410 | 5,600,654 | 320,925 | 316,742 | 1.3 | 15,002,216 | 267.9 | |
| loz | Lozi | 116,310 | 12,490,309 | 408,643 | 404,054 | 1.1 | 14,995,930 | 120.1 | |
| efi | Efik | 168,862 | 15,626,749 | 330,317 | 324,959 | 1.6 | 13,720,167 | 87.8 | |
| run | Kirundi | 38,923 | 4,108,968 | 361,128 | 356,667 | 1.2 | 13,524,056 | 329.1 | |
| bem | Bemba | 46,881 | 5,023,191 | 383,475 | 378,752 | 1.2 | 13,302,880 | 264.8 | |
| kon | Kongo | 1,331 | 136,064 | 373,212 | 368,240 | 1.3 | 12,860,564 | 9451.8 | |
| lua | Luba-Kasai | 82,124 | 8,690,290 | 289,328 | 285,627 | 1.3 | 10,412,301 | 119.8 | |
| ven | Venda | 47,047 | 4,850,851 | 201,668 | 199,071 | 1.3 | 9,108,788 | 187.8 | |
| sag | Sango | 39,516 | 3,992,368 | 247,537 | 243,782 | 1.5 | 9,074,523 | 227.3 | |
| iso | Isoko | 73,114 | 7,347,545 | 212,389 | 209,624 | 1.3 | 8,800,281 | 119.8 | |
| swc | Congo Swahili | 53,316 | 6,757,824 | 269,765 | 267,715 | 0.8 | 8,471,421 | 125.4 | |
| lug | Luganda | 119,739 | 12,779,860 | 231,730 | 228,553 | 1.4 | 8,407,004 | 65.8 | |
| tum | Tumbuka | 35,350 | 3,819,053 | 230,162 | 227,076 | 1.3 | 8,257,192 | 216.2 | |
| mos | Mossi | 46,190 | 4,758,571 | 183,492 | 180,721 | 1.5 | 8,188,199 | 172.1 | |
| kqn | Kaonde | 58,468 | 6,355,468 | 217,149 | 214,340 | 1.3 | 7,536,562 | 118.6 | |
| umb | Umbundu | 34,376 | 3,675,942 | 209,249 | 206,584 | 1.3 | 7,318,875 | 199.1 | |
| tiv | Tiv | 40,656 | 4,270,512 | 181,933 | 179,529 | 1.3 | 7,059,503 | 165.3 | |
| wol | Wolof | 50,459 | 5,031,731 | 175,034 | 173,693 | 0.8 | 5,103,692 | 101.4 | |
| wal | Wolaytta | 35,349 | 3,773,484 | 128,592 | 127,145 | 1.1 | 5,087,877 | 134.8 | |
| fon | Fon | 375,361 | 34,028,294 | 127,540 | 125,446 | 1.6 | 4,938,555 | 14.5 | COVERAGE_LT_20PCT_OF_OLD_TOKENS |
| lun | Lunda | 34,525 | 3,768,742 | 132,652 | 130,982 | 1.3 | 4,593,625 | 121.9 | |
| kik | Kikuyu | 33,879 | 3,477,346 | 93,771 | 92,444 | 1.4 | 4,290,403 | 123.4 | |
| ssw | Swati | 31,467 | 3,565,671 | 117,631 | 116,351 | 1.1 | 4,220,839 | 118.4 | |
| kua | Kuanyama | 27,744 | 3,101,117 | 98,426 | 97,197 | 1.2 | 3,727,724 | 120.2 | |
| kmb | Kimbundu | 23,819 | 2,522,745 | 89,088 | 88,055 | 1.2 | 3,207,552 | 127.1 | |
| tsc | Tswa | 22,954 | 2,485,201 | 83,173 | 82,170 | 1.2 | 3,128,360 | 125.9 | |
| vmw | Makhuwa | 19,613 | 2,132,204 | 71,765 | 70,929 | 1.2 | 2,643,055 | 124.0 | |
| ach | Acholi | 17,534 | 1,861,626 | 81,870 | 80,529 | 1.6 | 2,427,140 | 130.4 | |
| kam | Kamba | 18,073 | 1,859,930 | 50,241 | 49,590 | 1.3 | 2,313,936 | 124.4 | |
| nyn | Nyankore | 13,297 | 1,439,508 | 59,644 | 58,659 | 1.7 | 1,990,504 | 138.3 | |
| seh | Sena | 12,821 | 1,426,424 | 51,604 | 51,044 | 1.1 | 1,845,181 | 129.4 | |
| sid | Sidamo | 12,467 | 1,363,870 | 45,806 | 45,220 | 1.3 | 1,675,143 | 122.8 | |
| kri | Krio | 12,902 | 1,313,946 | 41,647 | 41,162 | 1.2 | 1,665,689 | 126.8 | |
| yao | Yao | 12,176 | 1,342,259 | 42,981 | 42,443 | 1.3 | 1,640,603 | 122.2 | |
| ewe | Ewe | 30,182 | 3,101,519 | 30,466 | 29,668 | 2.6 | 1,593,309 | 51.4 | |
| kab | Kabyle | 39,105 | 3,885,266 | 37,993 | 37,704 | 0.8 | 1,386,424 | 35.7 | |
| shi | Tachelhit | 9,212 | 932,398 | 15,267 | 15,224 | 0.3 | 1,251,668 | 134.2 | |
| bas | Basaa | 7,843 | 815,088 | 32,417 | 32,034 | 1.2 | 1,231,730 | 151.1 | |
| urh | Urhobo | 12,528 | 1,223,228 | 28,802 | 28,456 | 1.2 | 1,143,305 | 93.5 | |
| ndc | Ndau | 7,511 | 826,303 | 29,827 | 29,544 | 0.9 | 1,005,870 | 121.7 | |
| oke | Okpe | 7,856 | 735,410 | 22,109 | 21,869 | 1.1 | 880,784 | 119.8 | |
| wes | Cameroon Pidgin | 4,170 | 428,204 | 27,891 | 27,573 | 1.1 | 715,658 | 167.1 | |
| dua | Duala | 5,027 | 532,596 | 15,063 | 14,907 | 1.0 | 695,035 | 130.5 | |
| pcm | Pidgin | 124,861 | 10,612,775 | 22,781 | 22,504 | 1.2 | 574,472 | 5.4 | COVERAGE_LT_20PCT_OF_OLD_TOKENS |
| dyu | Dyula | 4,504 | 481,760 | 14,438 | 14,278 | 1.1 | 547,678 | 113.7 | |
| kbp | Kabiye | 4,311 | 430,136 | 9,084 | 8,969 | 1.3 | 513,293 | 119.3 | |
| bin | Edo | 17,995 | 1,712,912 | 11,338 | 11,230 | 1.0 | 469,808 | 27.4 | |
| her | Herero | 2,170 | 234,739 | 8,013 | 7,941 | 0.9 | 293,583 | 125.1 | |
| ish | Esan | 1,697 | 163,946 | 5,130 | 5,085 | 0.9 | 205,420 | 125.3 | |
| luy | Luyia | - | 0 | 12,903 | 12,899 | 0.0 | 182,465 |  |  |
| kea | Kabuverdianu | 1,066 | 112,439 | 5,115 | 5,068 | 0.9 | 146,220 | 130.0 | |
| bam | Bambara | 3,736 | 377,272 | 8,110 | 8,101 | 0.1 | 140,806 | 37.3 | |
| men | Mende | - | 0 | 1,645 | 1,621 | 1.5 | 64,565 |  | BELOW_2000_PAIRS |
| ibb | Ibibio | 2,060 | 202,632 | 1,000 | 998 | 0.2 | 59,846 | 29.5 | BELOW_2000_PAIRS |
| nbl | South Ndebele | 37,195 | 4,141,565 | 189 | 188 | 0.5 | 13,826 | 0.3 | BELOW_2000_PAIRS; COVERAGE_LT_20PCT_OF_OLD_TOKENS |
| ful | Fulah | - | 0 | 0 | 0 | 0.0 | 0 |  | BELOW_2000_PAIRS |
| bbj | Ghomala | 3,540 | 375,594 | 0 | 0 | 0.0 | 0 | 0.0 | BELOW_2000_PAIRS; COVERAGE_LT_20PCT_OF_OLD_TOKENS |
| dag | Dagbani | 15,310 | 1,927,495 | 0 | 0 | 0.0 | 0 | 0.0 | BELOW_2000_PAIRS; COVERAGE_LT_20PCT_OF_OLD_TOKENS |
| tzm | Tamazight | - | 0 | 0 | 0 | 0.0 | 0 |  | BELOW_2000_PAIRS |
| fat | Fante | - | 0 | 0 | 0 | 0.0 | 0 |  | BELOW_2000_PAIRS |
| knc | Kanuri | - | 0 | 0 | 0 | 0.0 | 0 |  | BELOW_2000_PAIRS |
| nus | Nuer | 90,891 | 9,371,086 | 0 | 0 | 0.0 | 0 | 0.0 | BELOW_2000_PAIRS; COVERAGE_LT_20PCT_OF_OLD_TOKENS |
| gaa | Ga | - | 0 | 0 | 0 | 0.0 | 0 |  | BELOW_2000_PAIRS |
| nde | North Ndebele | - | 0 | 0 | 0 | 0.0 | 0 |  | BELOW_2000_PAIRS |


Languages with no old row (`luy`, `men`, `ful`, `tzm`, `fat`, `knc`, `gaa`, `nde`) are table entries outside the 77 SFT tags; `luy` (12,899 pairs, mamakobe Luhya) and `men` come from extras.

## 5. What the filter removed (rows removed per rule, per protected set, per source)

Unique pairs removed: **268,020** (1.13%). Sole reason shared 8-gram (no exact or contained sentence): **148,540 pairs (55%)**; so the strict R3 rule roughly doubles the removal compared with R1+R2 alone (R1 exact: 115,766 pairs, R2: 80,221). English side hit in 224,597 pairs, target side in 67,094, both in 23,671. Per protected set (rows before the global dedupe; a row can hit several sets; all in `removed_by_rule_set_side.csv` and `removed_by_protected_set.csv`):

| protected set | R1 exact, English side | R1 exact, target side | R2 contained (both) | R3 8-gram, English | R3 8-gram, target |
|---|---:|---:|---:|---:|---:|
| ibom_mt_efi_davlan:dev | 32,226 | 1,072 | 30,084 | 82,836 | 2,549 |
| ibom_mt_efi_davlan:test | 32,368 | 1,080 | 27,747 | 63,096 | 2,015 |
| smol:all | 18,174 | 4,024 | 2,416 | 13,006 | 10,433 |
| afrilongdoc:all | 14,873 | 1,088 | 348 | 13,031 | 8,124 |
| mafand:test | 4,538 | 6,146 | 8,280 | 7,625 | 11,445 |
| mafand:dev | 3,144 | 5,256 | 6,988 | 7,309 | 9,408 |
| tico19:test | 6,063 | 5,641 | 11,238 | 7,155 | 6,437 |
| glot_story_book:all | 1,626 | 574 | 281 | 3,060 | 4,683 |
| tico19:dev | 2,729 | 2,491 | 4,892 | 2,985 | 2,904 |
| afridocmt:test | 2,277 | 770 | 1,136 | 1,648 | 2,329 |
| afridocmt:dev | 1,017 | 289 | 499 | 1,502 | 1,635 |
| ntrex128:all | 49 | 89 | 10 | 155 | 2,831 |
| ntrex_african:all | 49 | 89 | 10 | 144 | 2,830 |
| ssa_mte:test | 608 | 39 | 648 | 745 | 2,163 |
| xsimplusplus:dev | 87 | 80 | 165 | 259 | 2,291 |
| flores_plus:dev | 87 | 75 | 160 | 223 | 2,281 |
| xsimplusplus:devtest | 72 | 74 | 145 | 259 | 1,627 |
| flores_plus:devtest | 72 | 65 | 136 | 250 | 1,625 |
| salt:test | 562 | 539 | 698 | 538 | 401 |
| salt:dev | 528 | 481 | 571 | 338 | 266 |
| sib200:test | 11 | 11 | 22 | 20 | 446 |
| ssa_mte:dev | 8 | 0 | 8 | 16 | 422 |
| afriscience_mt:dev | 1 | 13 | 0 | 36 | 386 |
| afriscience_mt:test | 1 | 0 | 0 | 94 | 292 |

Notes: the largest hits come from `Davlan/ibom-mt-en-efi` dev/test, which is Jehovah's-Witnesses (JW300-style) text, so about 1,000 English JW sentences are removed from the English side of every mt560 language file; SmolSent/SmolDoc, AfriLongDoc and TICO-19 hit through web-style English; FLORES+ itself hits only 1,600-2,300 target-side rows and about 250 English-side rows per split in the reachable sources (the old carriers are missing).

Sources that lost the most (`per_source.csv`):

| source repo | licence class | pairs before | removed | removed % |
|---|---|---:|---:|---:|
| African-Languages-Lab/multi-open | other | 3,783,794 | 34,830 | 0.92 |
| HuggingFaceFW/finetranslations | odc-by | 892,786 | 26,038 | 2.92 |
| BrianMwangi/english-to-luo-opus | cc-by | 1,396,947 | 15,705 | 1.12 |
| michsethowusu/english-setswana_sentence-pairs_mt560 | cc-by | 776,161 | 13,700 | 1.77 |
| michsethowusu/english-afrikaans_sentence-pairs_mt560 | cc-by | 1,153,804 | 8,433 | 0.73 |
| michsethowusu/english-northern-sotho_sentence-pairs_mt560 | cc-by | 552,037 | 7,773 | 1.41 |
| michsethowusu/english-amharic_sentence-pairs_mt560 | cc-by | 650,910 | 7,402 | 1.14 |
| michsethowusu/english-malagasy_sentence-pairs_mt560 | cc-by | 819,103 | 7,385 | 0.9 |
| michsethowusu/english-zulu_sentence-pairs_mt560 | cc-by | 958,022 | 7,326 | 0.76 |
| michsethowusu/english-tsonga_sentence-pairs_mt560 | cc-by | 753,199 | 7,291 | 0.97 |
| michsethowusu/english-chichewa_sentence-pairs_mt560 | cc-by | 773,113 | 7,105 | 0.92 |
| michsethowusu/english-xhosa_sentence-pairs_mt560 | cc-by | 968,985 | 7,059 | 0.73 |
| michsethowusu/english-shona_sentence-pairs_mt560 | cc-by | 758,842 | 6,577 | 0.87 |
| michsethowusu/english-lingala_sentence-pairs_mt560 | cc-by | 531,750 | 5,881 | 1.11 |

## 6. Cost summary and what this does not tell us

On the reachable sources the filter costs 1.13% of the pairs (0.3-2.6% per language; Ewe 2.62%, Orm 2.49%, Sot 2.33%, Hausa 1.87%, Tsn 1.74%, Yoruba 1.57%). It **cannot** tell what the old build would have lost: the audit showed 95.3% of FLORES+ devtest English in the old text through carriers that are unreachable. The honest statement is: the clean rebuild contains no protected text, and covers the old languages only in part. To close the gap the owner needs the unreachable repositories (or a copy of `gemma/pairs_final` / `gemma/merged`) and/or the GPU for QE. Languages whose only dev material is protected (efi/ibb/seh: their protected text comes from IbomMT and FLORES-MOZ) were filtered like all others; whether their dev text should stay out is an owner question (section 9).

## 7. Exposure check (mandatory)

`exposure_check.py`, output sample of **1,001,020 pairs of 23,406,354** (2,002,040 texts, both sides), 20 minutes on 2 cores:

| check | result |
|---|---|
| (A) hash index of this build, 35 protected-set groups | **0 texts with a hit** |
| (B) the audit's own flag logic, unchanged: `gemma_mt/eval/contamination.py` `BenchIndex` + `scan_windows` + the near >= 50% overlap of `flags_for_rows`, fed from a local dump of the normalised protected sentences (6 groups of at most 1 M sentences; the xsim++ perturbations were left out of this check for memory, they are covered by A) | **0 exact, 0 near, 0 shared 8-grams** in every group |
| (C) independent strict check with Python sets: any shared word 8-gram between a sampled text and any protected sentence | **0 texts** |
| (D) positive control: 6,000 protected sentences planted into the checkers | C found 6,000 of 6,000; B found 5,667 of 6,000 (the remaining 333 are probably multi-sentence protected strings shorter than 8 tokens, which the audit's sentence-splitting scan cannot match as one unit; not verified) |

Per-set tables: `exposure_check.json`. Protected sentences of fewer than 3 tokens are not indexed (rule R1 of the reference), so one- and two-word protected entries (GATITOS words) are not checked.

## 8. Licences, provenance, quarantine (owner's request)

Main output by licence class (pairs / estimated target tokens): cc-by 17,659,378 / 651.3 M; odc-by 866,748 / 343.7 M (HuggingFaceFW/finetranslations: model-written English sides, long web passages, 29% of the tokens); `other` 3,748,964 / 138.6 M (African-Languages-Lab/multi-open, licence "other" on the card: terms not stated); unspecified 981,048 / 35.6 M; apache 81,959 / 1.1 M; mpl 58,542 / 1.1 M (Pontoon); share_alike 7,723 / 0.41 M (SSA-MTE train, CC-BY-SA-4.0); **non_commercial 1,992 / 0.12 M (IbomMT train: efi 994, ibb 998; the only deliberate NC source, owner decision 2026-10-28; filter `licence != 'non_commercial'` for a commercial variant)**. Table per language and class (pairs / tokens; full CSV `licence_by_language.csv`, `licence_by_class.csv`):

| iso | unspecified pairs / tokens | non_commercial pairs / tokens | share_alike pairs / tokens | other pairs / tokens | odc-by pairs / tokens | cc-by/cc0/apache/mit/mpl tokens |
|---|---:|---:|---:|---:|---:|---:|
| yor | 8,304 / 423,122 | - | 683 / 44,728 | 507,869 / 38,857,994 | 43,296 / 20,611,953 | 21,711,918 |
| afr | 7,263 / 137,454 | - | - | 897,113 / 24,394,313 | 63,367 / 22,440,332 | 33,594,054 |
| zul | 8,511 / 269,068 | - | 706 / 32,592 | 92,496 / 2,653,038 | 83,941 / 34,042,124 | 35,504,218 |
| xho | 115,893 / 4,253,006 | - | - | 238,426 / 7,537,232 | 64,453 / 25,878,967 | 34,347,091 |
| nya | 967 / 46,530 | - | 1 / 65 | 82,682 / 2,324,175 | 66,720 / 26,963,001 | 28,646,937 |
| ibo | 10,227 / 410,644 | - | 562 / 32,629 | 177,518 / 9,960,757 | 69,336 / 28,025,561 | 17,392,122 |
| hau | 86,897 / 3,101,132 | - | 708 / 30,859 | 213,574 / 9,075,815 | 87,998 / 33,760,861 | 9,678,479 |
| swh | 9,132 / 405,071 | - | 756 / 36,575 | 809,255 / 22,838,947 | 78,198 / 29,147,095 | 125,143 |
| sna | - | - | - | 93,808 / 2,525,141 | 53,835 / 21,807,298 | 26,828,016 |
| som | 6,759 / 304,238 | - | - | 105,687 / 3,808,951 | 98,471 / 39,018,051 | 7,349,966 |
| mlg | 8,490 / 365,979 | - | - | - | - | 37,223,291 |
| tsn | 9,419 / 312,755 | - | - | 56,212 / 1,190,653 | - | 31,544,743 |
| sot | 7,853 / 44,073 | - | 711 / 33,807 | 86,350 / 2,273,721 | 73,912 / 29,964,430 | 168,636 |
| tso | 7,082 / 165,638 | - | - | - | - | 31,464,463 |
| luo | - | - | 705 / 30,522 | - | - | 30,259,128 |
| amh | 8,770 / 349,891 | - | 692 / 34,714 | - | - | 26,474,656 |
| kin | 20,420 / 510,005 | - | 703 / 34,844 | 92,420 / 2,425,326 | 56,197 / 21,616,517 | 9,675 |
| twi | 563,553 / 20,771,660 | - | 705 / 37,966 | 99,211 / 2,442,912 | - | 1,470 |
| nso | - | - | - | - | - | 22,339,715 |
| lin | 224 / 1,376 | - | - | - | - | 19,259,085 |
| orm | 9,283 / 388,104 | - | - | 37,669 / 978,786 | 23,534 / 9,134,952 | 6,108,678 |
| tir | 9,895 / 523,138 | - | - | - | - | 14,479,077 |
| loz | - | - | - | - | - | 14,995,930 |
| efi | - | 994 / 55,863 | - | - | - | 13,664,304 |
| run | - | - | - | - | - | 13,524,056 |
| bem | 4,964 / 205,681 | - | - | - | - | 13,097,198 |
| kon | 1,365 / 10,907 | - | - | - | - | 12,849,657 |
| lua | - | - | - | - | - | 10,412,301 |
| ven | - | - | - | - | - | 9,108,788 |
| sag | - | - | - | - | - | 9,074,523 |
| iso | - | - | - | - | - | 8,800,281 |
| swc | - | - | - | - | - | 8,471,421 |
| lug | 9,203 / 198,877 | - | - | - | - | 8,208,126 |
| tum | - | - | - | - | - | 8,257,192 |
| mos | - | - | - | - | - | 8,188,199 |
| kqn | - | - | - | - | - | 7,536,562 |
| umb | - | - | - | - | - | 7,318,875 |
| tiv | - | - | - | - | - | 7,059,503 |
| wol | 14,074 / 598,991 | - | - | 52,919 / 1,340,078 | 3,267 / 1,273,688 | 1,890,934 |
| wal | 9,939 / 510,982 | - | - | - | - | 4,576,894 |
| fon | - | - | - | 78,510 / 2,458,330 | - | 2,480,224 |
| lun | - | - | - | - | - | 4,593,625 |
| kik | - | - | 790 / 56,201 | - | - | 4,234,202 |
| ssw | 503 / 17,258 | - | - | - | - | 4,203,581 |
| kua | - | - | - | - | - | 3,727,724 |
| kmb | - | - | - | - | - | 3,207,552 |
| tsc | - | - | - | - | - | 3,128,360 |
| vmw | - | - | 1 / 80 | - | - | 2,642,974 |
| ach | 9,541 / 174,476 | - | - | - | - | 2,252,663 |
| kam | - | - | - | - | - | 2,313,936 |
| nyn | 9,572 / 208,239 | - | - | - | - | 1,782,265 |
| seh | - | - | - | - | - | 1,845,181 |
| sid | - | - | - | - | - | 1,675,143 |
| kri | - | - | - | - | - | 1,665,689 |
| yao | - | - | - | - | - | 1,640,603 |
| ewe | 90 / 4,857 | - | - | 27,245 / 1,549,004 | - | 39,447 |
| kab | - | - | - | - | - | 1,386,424 |
| shi | - | - | - | - | - | 1,251,668 |
| bas | 5,156 / 218,090 | - | - | - | - | 1,013,640 |
| urh | - | - | - | - | - | 1,143,305 |
| ndc | - | - | - | - | - | 1,005,870 |
| oke | - | - | - | - | - | 880,784 |
| wes | - | - | - | - | - | 715,658 |
| dua | - | - | - | - | - | 695,035 |
| pcm | 629 / 14,490 | - | - | - | 223 / 52,431 | 507,550 |
| dyu | - | - | - | - | - | 547,678 |
| kbp | 8,969 / 513,293 | - | - | - | - | 0 |
| bin | - | - | - | - | - | 469,808 |
| her | - | - | - | - | - | 293,583 |
| ish | - | - | - | - | - | 205,420 |
| luy | - | - | - | - | - | 182,465 |
| kea | - | - | - | - | - | 146,220 |
| bam | 8,101 / 140,806 | - | - | - | - | 0 |
| men | - | - | - | - | - | 64,565 |
| ibb | - | 998 / 59,846 | - | - | - | 0 |
| nbl | - | - | - | - | - | 13,826 |

**Sources with unspecified or unknown licence that are in the main output (used and flagged, as in the old build):** KwekuAinoo/eng_twi (551,358 pairs), sentence-transformers/parallel-sentences-opus-100 (210,479), UBC-NLP/AfroLingu-MT (207,796; gated), cibfaye/wolof-english-bible (11,415). **`other`:** African-Languages-Lab/multi-open (3,748,964; the card says `license: other`, no terms). **Provenance scan** of all 84 contributing repositories' cards (`provenance_scan.csv`): the only card that mentions NLLB is `yigagilbert/synthetic-parallel-external`; no card mentions TranslatePsy. **Quarantine (`quarantine_nc/` on the Hub, not in `data/`):** (1) `yigagilbert/synthetic-parallel-external` (Luganda, 139,993 pairs after the filter): the card mentions NLLB (reason in the `reason` column); the old build had used about 147,000 rows of it. (2) `fiifinketia/pristine-twi-english` (Twi, 411,089 pairs after the filter): the re-upload declares CC-BY-4.0, but the old licence scan classed its upstream `ghananlpcommunity/pristine-twi-english` as CC-BY-NC-4.0 (held); licence doubt, owner decides. Both were put through the same decontamination filter.

## 9. Open items for the owner (not decided by me)

1. Unreachable carriers (section 1): can the owner recover `<HF_ORG_LEGACY>/naija*`, `<HF_ORG>/tds`, `clean_data*` etc. or a copy of `gemma/merged`? Without them pcm, twi, fon, ibo, hau, bin and nus/dag/bbj/nbl stay thin.
2. mt560: "NLLB-derived" (old converter docstring) versus OPUS MT560 (cards). 69% of the output (16.2 M pairs) comes from the 80 mt560 repositories. If they must be treated as NLLB-derived, the main output shrinks to 7.2 M pairs and most small languages disappear.
3. `pristine-twi-english` licence (411,089 Twi pairs in quarantine), `multi-open` terms ("other", 3.7 M pairs), unspecified sources (above), and whether `synthetic-parallel-external` is NLLB-derived in fact (the card only mentions NLLB).
4. Whether FLORES+ dev should stay protected for efi/ibb/seh/pcm (their only human dev text); I protected it as instructed.
5. WMT22 African and AfroBench MT could not be fetched (section 2); MAFAND-MT train was **not** used as a source (licence of the GitHub data not checked; `masakhane/mafand` card header says Apache-2.0, the paper says CC-BY-NC).
6. QE scoring (GPU) and the old converter-level language checks are still to do; they will lower the counts of the raw sources. SSA-MTE train: Ewe, Lingala, Wolof (French source) and Nyanja, Emakhuwa (Portuguese source) yield no English pairs (1 pair each for nya, vmw); 7,723 pairs for the other languages.
7. The box disk may disappear on 2026-10-29: all outputs are on the Hub; scripts and reports are in this repository and in `research_archive_2026-10-23/local/`.

## 10. Files

* Scripts (`data_pipeline/sft_clean/`): `decon.py`, `test_decon.py`, `fetch_protected.py`, `protected_index_loader.py`, `build_pairs.py`, `finalize.py`, `tok_ratio.py`, `prov_scan.py`, `exposure_check.py`, `push_artifacts.py`, `make_report_tables.py`, `README_hub.md`.
* Data (`reports/data/sft_clean_2026-10-28/`): `protected_manifest.json`, `per_language.csv`, `per_language_source.csv`, `per_source.csv`, `comparison_old_vs_rebuilt.csv/.md`, `removed_by_rule_set_side.csv`, `removed_by_protected_set.csv`, `licence_by_language.csv`, `licence_by_class.csv`, `quarantine_nc.csv`, `provenance_scan.csv`, `source_reachability_old_vs_rebuilt.csv`, `source_access_2026-10-28.json`, `exposure_check.json`, `chars_per_token.json`, `summary.json`.
* Hub (`<HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28`, private): `data/`, `quarantine_nc/`, `rebuild/`, `protected_index/`, `stats/`, `scripts/`.


---
# Addendum 2026-10-28 (lead follow-up, steps A, B, C)

## STEP A: mt560 provenance (nothing deleted)

**Status A: done and pushed (2026-10-28). Verdict: NLLB-derived = NO (high confidence from dates and cards); underlying licence = MIXED / UNCLEAR, dominated by JW300 and Bible text whose licence is not established; the uploader's cc-by-4.0 is not backed by a primary source. `source_family` column added to `data/` (value `mt560` etc.); tokens per language with and without mt560 below.**

1. **Where "NLLB-derived" came from.** Only one place: `data_pipeline/sft_new/ext_translation.py` line 186 (identical in `old_box_snapshot_2026-10-21/sft_build/repo/data_pipeline/sft_new/ext_translation.py:186`): `@register("mt560_qe", "michsethowusu/english-{fon,oromo,setswana}_sentence-pairs_mt560 (NLLB-derived): AfriCOMET-QE >= 0.46 AND GlotLID-audited on both sides")`. It is a one-line registry description with no evidence attached. The other old converter that handles the same repositories, `ext_mich.py` lines 1-3, says the opposite: "the NLLB-v1 mined `<A>-<B>_sentence-pairs` sets (759) are NOT converted (NLLB-related, skipped for now)" and `mich_mt560` = "english-<lang>_sentence-pairs_mt560 (OPUS MT560)". `michsethowusu_families.json` (old build) classes the 759 un-suffixed repositories as NLLB (723 cards mention NLLB, CCMatrix) and the 99 `_mt560` ones as "Original Source: OPUS MT560" with 0 NLLB mentions. I conclude the docstring conflated the two michsethowusu families; I could not verify "NLLB-derived" for the `_mt560` repositories anywhere.
2. **Primary sources.** MT560 = Gowda, Zhang, Mattmann, May, *Many-to-English Machine Translation Tools, Data, and Pretrained Models*, ACL 2021 demos, arXiv 2104.00290v2 (1 Jul 2021; read from the PDF). Section 3.1 builds it with MTData from the sources of its Table 1: Europarl, KFTT, Indian6, **OPUS** (Tiedemann 2012: the whole OPUS collection as served by mtdata), UNPCv1, Tilde MODEL, TEDTalks, IITB Hi-En, ParaCrawl, WikiMatrix, **JW300**, PMIndia, OPUS100, WMT 13-20; for low-resource languages "all available datasets" were selected; records that overlap exactly with held-out sets were removed; 474 M sentences. The paper is from July 2021, before NLLB (2022), so the NLLB model, the OPUS NLLB mined corpus and NLLB-Seed cannot be in it. The OPUS MT560 page (opus.nlpl.eu/MT560, fetched today) lists OPUS, JW300, OPUS100, StatMT, NeuLab TEDTalks, WikiMatrix, Joshua Indian corpus, PMIndia, ParaCrawl, IITB, Tilde MODEL as notable sources, says "473M sentences after deduplication and cleaning, 560 source-side languages" and states **no licence**; its only rights text is "if you view this derived dataset as a violation of intellectual property rights, please let us know, so we will be happy to remove it from the corpus" and a request to cite all articles in `citations.bib`. The JW300 paper (Agic and Vulic, ACL 2019) abstract gives size and languages, no licence. I could not fetch an OPUS JW300 page (404 on the URLs tried), so the JW300 terms are not quoted here.
3. **Per-row source label: none.** The repositories have exactly two columns (`eng`, `<language>`; checked for yoruba, tsonga, kamba, gikuyu, wolof, igbo, hausa). The card says `Original Source: [OPUS MT560 Dataset]` and cc-by-4.0.
4. **What the text is.** Sample of 200 rows each for yor, tso, kam, kik, wol (kept rows) read by keyword class plus 5,000-row heuristic: Jehovah's-Witnesses (JW300) vocabulary 13-24% and Bible vocabulary 9-28% by keywords alone (lower bounds: many JW sentences carry no keyword: "Why must we train others to take up theocratic assignments ?"); no software/UI text (GNOME/Ubuntu/KDE) found (0-0.1%). Over all 68 mt560 files (16.3 M kept pairs, 20,000-row samples each) **95% of the English sides have the OPUS-tokenised spacing ("word , word .")** typical of JW300 and Bible-uedin, and 38% contain religious keywords. Wolof is mostly King James Bible English. Per-row examples are in `mt560_sample.json`; counts per file in `mt560_religion_share.json`.
* yor: 200-row sample by keyword classes {'other': 135, 'jw': 51, 'bible': 14}; keyword share on 5,000 rows {'other': 0.703, 'jw': 0.207, 'bible': 0.089, 'legal_tech': 0.0, 'ui': 0.0}
* tso: 200-row sample by keyword classes {'other': 151, 'jw': 29, 'bible': 20}; keyword share on 5,000 rows {'other': 0.736, 'jw': 0.177, 'bible': 0.085, 'legal_tech': 0.001, 'ui': 0.001}
* kam: 200-row sample by keyword classes {'other': 127, 'jw': 49, 'bible': 23, 'legal_tech': 1}; keyword share on 5,000 rows {'bible': 0.101, 'other': 0.664, 'jw': 0.234, 'legal_tech': 0.0, 'ui': 0.001}
* kik: 200-row sample by keyword classes {'jw': 58, 'other': 125, 'bible': 17}; keyword share on 5,000 rows {'jw': 0.237, 'other': 0.653, 'bible': 0.109, 'legal_tech': 0.001, 'ui': 0.0}
* wol: 200-row sample by keyword classes {'other': 118, 'bible': 54, 'jw': 28}; keyword share on 5,000 rows {'jw': 0.129, 'bible': 0.277, 'other': 0.594, 'ui': 0.0}
5. **Verdicts.** *NLLB-derived:* **no** (MT560 published July 2021, before NLLB; cards name OPUS MT560 and none mentions NLLB; the only contrary statement is an unsupported docstring). Caveat: michsethowusu re-packaged the data himself; the 560-language OPUS MT560 release has no per-row provenance, so row-level absence of NLLB output cannot be proven, only argued from the dates. *Underlying licence mix:* **unknown and mixed**: JW300 and Bible translations are crawled from jw.org / Bible publishers (copyright holders other than the uploader; no open licence stated by OPUS or the JW300 abstract), the rest of the OPUS mixture (Tatoeba, wikimedia, etc.) is mostly open or share-alike. Treat mt560 as **licence unknown, religious-domain-heavy**, not as CC-BY.
6. **Switch.** Column `source_family` is now in `data/pairs-*.parquet` (`mt560`, `multi-open`, `finetranslations`, `opus100`, `luo-opus`, ...). Without mt560 the main output is 7.2 M pairs / 623,135,791 of 1,171,995,571 estimated tokens come from mt560 (53.2%). Languages with fewer than 2,000 pairs without mt560: lin 224, loz 0, run 0, kon 1,365, efi 994, lua 0, sag 0, swc 0, ven 254, tum 0, kqn 0, umb 0, tiv 0, iso 0, mos 0, lun 0, kua 0, kik 790, tsc 0, kmb 0, vmw 1, seh 0, kam 0, sid 0, yao 0, kri 0, kab 0, ndc 0, wes 0, urh 0, shi 0, pcm 852, oke 0, dua 0, dyu 0, bin 0, her 0, ish 0, kea 0, men 0, ibb 998, nbl 188. Table (CSV `mt560_switch_per_language.csv`):

| iso | pairs total | pairs mt560 | pairs without | est. tokens total | est. tokens without mt560 | mt560 share of tokens % |
|---|---:|---:|---:|---:|---:|---:|
| afr | 2,127,231 | 1,145,371 | 981,860 | 80,566,155 | 47,236,887 | 41.4 |
| zul | 1,142,047 | 950,696 | 191,351 | 72,501,042 | 37,194,707 | 48.7 |
| xho | 1,394,691 | 961,926 | 432,765 | 72,016,298 | 38,014,508 | 47.2 |
| nya | 916,632 | 766,008 | 150,624 | 57,980,710 | 29,336,446 | 49.4 |
| hau | 680,211 | 287,951 | 392,260 | 55,647,149 | 46,014,495 | 17.3 |
| swh | 903,245 | 0 | 903,245 | 52,552,833 | 52,552,833 | 0.0 |
| yor | 965,223 | 400,753 | 564,470 | 81,649,717 | 60,043,757 | 26.5 |
| sna | 900,169 | 752,265 | 147,904 | 51,160,457 | 24,335,865 | 52.4 |
| som | 371,791 | 160,874 | 210,917 | 50,481,207 | 43,131,241 | 14.6 |
| ibo | 664,283 | 403,361 | 260,922 | 55,821,715 | 38,490,994 | 31.0 |
| mlg | 880,712 | 872,222 | 8,490 | 37,589,270 | 365,979 | 99.0 |
| tsn | 831,499 | 762,461 | 69,038 | 33,048,152 | 1,691,372 | 94.9 |
| sot | 172,233 | 0 | 172,233 | 32,484,670 | 32,484,670 | 0.0 |
| tso | 755,549 | 745,908 | 9,641 | 31,630,102 | 321,129 | 99.0 |
| luo | 1,514,791 | 132,844 | 1,381,947 | 30,289,650 | 25,963,372 | 14.3 |
| kin | 170,159 | 0 | 170,159 | 24,596,369 | 24,596,369 | 0.0 |
| nso | 546,806 | 544,264 | 2,542 | 22,339,715 | 150,493 | 99.3 |
| lin | 526,093 | 525,869 | 224 | 19,260,461 | 1,376 | 100.0 |
| twi | 663,535 | 0 | 663,535 | 23,254,010 | 23,254,010 | 0.0 |
| orm | 221,086 | 150,573 | 70,513 | 16,610,522 | 10,501,983 | 36.8 |
| amh | 657,412 | 643,508 | 13,904 | 26,859,262 | 519,961 | 98.1 |
| loz | 404,054 | 404,054 | 0 | 14,995,930 | 0 | 100.0 |
| bem | 378,752 | 373,788 | 4,964 | 13,302,880 | 205,681 | 98.5 |
| run | 356,667 | 356,667 | 0 | 13,524,056 | 0 | 100.0 |
| kon | 368,240 | 366,875 | 1,365 | 12,860,564 | 10,907 | 99.9 |
| efi | 324,959 | 323,965 | 994 | 13,720,167 | 55,863 | 99.6 |
| lua | 285,627 | 285,627 | 0 | 10,412,301 | 0 | 100.0 |
| sag | 243,782 | 243,782 | 0 | 9,074,523 | 0 | 100.0 |
| swc | 267,715 | 267,715 | 0 | 8,471,421 | 0 | 100.0 |
| lug | 228,553 | 219,350 | 9,203 | 8,407,004 | 198,877 | 97.6 |
| ven | 199,071 | 198,817 | 254 | 9,108,788 | 23,140 | 99.7 |
| tum | 227,076 | 227,076 | 0 | 8,257,192 | 0 | 100.0 |
| kqn | 214,340 | 214,340 | 0 | 7,536,562 | 0 | 100.0 |
| umb | 206,584 | 206,584 | 0 | 7,318,875 | 0 | 100.0 |
| tir | 316,742 | 306,847 | 9,895 | 15,002,216 | 523,138 | 96.5 |
| tiv | 179,529 | 179,529 | 0 | 7,059,503 | 0 | 100.0 |
| iso | 209,624 | 209,624 | 0 | 8,800,281 | 0 | 100.0 |
| mos | 180,721 | 180,721 | 0 | 8,188,199 | 0 | 100.0 |
| wal | 127,145 | 117,206 | 9,939 | 5,087,877 | 510,982 | 90.0 |
| wol | 173,693 | 15,172 | 158,521 | 5,103,692 | 4,427,280 | 13.3 |
| lun | 130,982 | 130,982 | 0 | 4,593,625 | 0 | 100.0 |
| ssw | 116,351 | 113,234 | 3,117 | 4,220,839 | 161,440 | 96.2 |
| kua | 97,197 | 97,197 | 0 | 3,727,724 | 0 | 100.0 |
| fon | 125,446 | 30,212 | 95,234 | 4,938,555 | 3,312,455 | 32.9 |
| kik | 92,444 | 91,654 | 790 | 4,290,403 | 56,201 | 98.7 |
| tsc | 82,170 | 82,170 | 0 | 3,128,360 | 0 | 100.0 |
| kmb | 88,055 | 88,055 | 0 | 3,207,552 | 0 | 100.0 |
| vmw | 70,929 | 70,928 | 1 | 2,643,055 | 80 | 100.0 |
| ach | 80,529 | 70,988 | 9,541 | 2,427,140 | 174,476 | 92.8 |
| nyn | 58,659 | 49,087 | 9,572 | 1,990,504 | 208,239 | 89.5 |
| seh | 51,044 | 51,044 | 0 | 1,845,181 | 0 | 100.0 |
| kam | 49,590 | 49,590 | 0 | 2,313,936 | 0 | 100.0 |
| sid | 45,220 | 45,220 | 0 | 1,675,143 | 0 | 100.0 |
| yao | 42,443 | 42,443 | 0 | 1,640,603 | 0 | 100.0 |
| kri | 41,162 | 41,162 | 0 | 1,665,689 | 0 | 100.0 |
| ewe | 29,668 | 0 | 29,668 | 1,593,309 | 1,593,309 | 0.0 |
| kab | 37,704 | 37,704 | 0 | 1,386,424 | 0 | 100.0 |
| bas | 32,034 | 26,878 | 5,156 | 1,231,730 | 218,090 | 82.3 |
| ndc | 29,544 | 29,544 | 0 | 1,005,870 | 0 | 100.0 |
| wes | 27,573 | 27,573 | 0 | 715,658 | 0 | 100.0 |
| urh | 28,456 | 28,456 | 0 | 1,143,305 | 0 | 100.0 |
| shi | 15,224 | 15,224 | 0 | 1,251,668 | 0 | 100.0 |
| pcm | 22,504 | 21,652 | 852 | 574,472 | 66,922 | 88.4 |
| oke | 21,869 | 21,869 | 0 | 880,784 | 0 | 100.0 |
| dua | 14,907 | 14,907 | 0 | 695,035 | 0 | 100.0 |
| dyu | 14,278 | 14,278 | 0 | 547,678 | 0 | 100.0 |
| bin | 11,230 | 11,230 | 0 | 469,808 | 0 | 100.0 |
| kbp | 8,969 | 0 | 8,969 | 513,293 | 513,293 | 0.0 |
| her | 7,941 | 7,941 | 0 | 293,583 | 0 | 100.0 |
| luy | 12,899 | 0 | 12,899 | 182,465 | 182,465 | 0.0 |
| ish | 5,085 | 5,085 | 0 | 205,420 | 0 | 100.0 |
| kea | 5,068 | 5,068 | 0 | 146,220 | 0 | 100.0 |
| bam | 8,101 | 0 | 8,101 | 140,806 | 140,806 | 0.0 |
| men | 1,621 | 1,621 | 0 | 64,565 | 0 | 100.0 |
| ibb | 998 | 0 | 998 | 59,846 | 59,846 | 0.0 |
| nbl | 188 | 0 | 188 | 13,826 | 13,826 | 0.0 |


## STEP B: FLORES+ dev train half for the weak languages (protected-2026-10-28.2)

**Status B: done and pushed (2026-10-28). Index version `protected-2026-10-28.2`; 8,308 train-half rows in `data_devhalf/`; exposure proof re-run: 0 hits for FLORES+ devtest and the held-out dev half on both sides (999,077-pair sample of the main output, the devhalf rows, three methods, controls pass).**

* **Rule (applied once to the 997 FLORES+ dev ids, all languages):** `held-out if int(sha256(str(id).encode()).hexdigest(), 16) % 2 == 1 or id in forced; train half otherwise`. Forced held-out: the 100 ids of `reports/data/weak_selection_2026-10-26/sample_ids.json`. Result: **557 held-out ids** (odd-hash ids plus the forced ones; sha256 of the sorted id list `3b620a92c5193105e1161dbaa4daecaedaf4ab8deb6c709ceb6e26041f0a5d45`) and **440 train-half ids** (`79ef4baaa6b73d09c5736e8559c918fad8c71a60ab1286fe7b8903fec50f09db`); forced list `44df645ba5d7a0f5bf5e937def63007b0b48326798902d33d9db66b377752b29`. All three lists are in `protected_manifest.json` (`devhalf`) and `devhalf_split.json`. FLORES+ commit `1147e2dc59`.
* **What is protected now.** FLORES+ **devtest**: unchanged, 100% protected in all languages. FLORES+ **dev held-out half**: protected in every language, both sides, together with every other protected set (xsim++, IbomMT, SIB-200 ... unchanged). The train half is exempt **only** as English and as target text of the 19 listed languages (amh tir fon dyu kbp nus mos kam bam umb kea kmb vmw kab seh wol ewe luo tum; all have a FLORES+ dev file, Sena = `seh_Latn_MZ`); train-half text of every other language stays protected.
* **Implementation (the v.1 index file is unchanged, sha256 `f094fd2b24d6a0e4...`):** a second hash file `protected_exemption_devhalf.npz` (sha256 `bb39466f7cae284c...`) holds the train-half sentences and 8-grams; `decon.Index.exempt` makes a hit on an exempt sentence or 8-gram not a hit; `protected_index_loader.load()` verifies both sha256 values and attaches it. The exemption is **trimmed** so that it can never unprotect other text: 213 8-grams that also occur in devtest or the held-out half (FLORES sentences of one article share phrases) were removed from it. This was found by my own independent check: the first version of the exemption let 32 train-half texts share an 8-gram with held-out/devtest text (32 texts, 100 distinct 8-grams, 4 near-duplicates); after trimming those rows are gone (8,340 -> 8,308 rows) and the proof is clean. `test_decon.py` has a new test for the exemption.
* **Rows.** `data_devhalf/pairs-0000.parquet`: columns english, target, iso, source_dataset (`openlanguagedata/flores_plus:dev:<code>`), licence = `share_alike` (CC-BY-SA-4.0), flipped, decon_flag, source_family = `flores-plus-dev-half`, `flores_id`, `split` = `dev_train_half`. After the old `ok()` filter and the v.2 filter: 8,308 pairs. A few train-half sentences are still blocked by other protected sets (English also in SmolDoc/AfriLongDoc/AfriDocMT, or sharing a phrase with devtest): per set {'flores_plus:dev': 104, 'ibom_mt:dev': 57, 'flores_moz:dev': 58, 'xsimplusplus:dev': 102, 'sib200:test': 37, 'sib200_ibom:test': 19, 'smol:all': 1, 'afriscience_mt:dev': 1, 'afrilongdoc:all': 1, 'afridocmt:dev': 1, 'afridocmt:test': 1, 'flores_plus:devtest': 5, 'xsimplusplus:devtest': 5, 'sib200:dev': 2}.

| iso | FLORES+ code | train-half ids with text | pass ok() | kept after v.2 filter |
|---|---|---:|---:|---:|
| amh | amh_Ethi | 440 | 440 | 439 |
| tir | tir_Ethi | 440 | 440 | 437 |
| fon | fon_Latn | 440 | 440 | 436 |
| dyu | dyu_Latn | 440 | 440 | 436 |
| kbp | kbp_Latn | 440 | 440 | 437 |
| nus | nus_Latn | 440 | 440 | 431 |
| mos | mos_Latn | 440 | 440 | 437 |
| kam | kam_Latn | 440 | 440 | 439 |
| bam | bam_Latn | 440 | 440 | 436 |
| umb | umb_Latn | 440 | 440 | 439 |
| kea | kea_Latn | 440 | 440 | 439 |
| kmb | kmb_Latn | 440 | 440 | 439 |
| vmw | vmw_Latn | 440 | 440 | 439 |
| kab | kab_Latn | 440 | 440 | 439 |
| seh | seh_Latn_MZ | 440 | 440 | 439 |
| wol | wol_Latn | 440 | 440 | 439 |
| ewe | ewe_Latn | 440 | 440 | 433 |
| luo | luo_Latn | 440 | 440 | 437 |
| tum | tum_Latn | 440 | 440 | 437 |

* **Exposure proof for .2** (`exposure_check_v2.json`; sample 999,077 of 23,406,354 main-output pairs plus the 8,308 devhalf rows, both sides, 352,187 protected FLORES+ sentences = devtest of all languages + held-out dev half of all languages, re-fetched into memory only): (A) v.2 hash index: **0 hits** in the main sample and in the devhalf rows; (B) the audit's `contamination.py` logic: **0 exact, 0 near, 0 shared 8-grams**; (C) independent strict 8-gram sets: **0 texts**. Controls: 1,000 of 1,000 planted held-out/devtest sentences caught by A and by B; train-half targets of the listed languages (amh tir fon dyu kbp nus) and train-half English: 0 flagged (exemption works); train-half text of **non-listed** languages (yor, hau, 300 each): 300 of 300 flagged (still protected).
* **Restore of main-output rows removed only by train-half dev matches: not done.** Their text was not stored (only the per-row meta), and a restore needs a re-run of the affected sources; the size is small (flores_plus:dev hits in the main output were about 90 rows with R1 and 220 with R3 on the English side before the global dedupe, about 45% of them train half), at most a few hundred pairs.
* Caveats for the lead: the held-out half has 557 ids (the forced 100 plus the odd-hash ones), a little more than half; FLORES+ dev English is shared by xsim++/IbomMT dev/SIB-200 and is exempt from those sets as English only for the train half; whether FLORES+ dev for efi/ibb/seh/pcm stays out is still the owner's call (the 19 listed languages include seh).

## STEP C: specialist dataset for the 19 weak languages

**Status C: done and pushed to `specialist/` in the same repo (2026-10-28).** 4,125,351 pairs after dedupe and the v.2 filter (candidates 4,167,435, 16,778 removed by the filter, duplicates removed before it). Files: `specialist/pairs-0000.parquet` (columns english, target, iso, source_dataset, licence, licence_raw, kind, source_family, flipped, tgt_tokens (exact Gemma-3 tokens of the target), decon_flag) and `specialist/stats/` (`per_language_summary.csv`, `per_language_class_kind.csv`, `per_language_source.csv`, `excluded.csv`, `removed_by_filter_and_duplicates.csv`; copies in `reports/data/sft_clean_2026-10-28/specialist/`).

* **Sources and priority for the exact dedupe** (first source keeps its class): (1) this build's main output; (2) weak repo `train_ready/permissive`; (3) R10 `sft_extra_v1/commercial_ok` (GoURMET CC0, GlobalVoices, Tatoeba, FineTranslations with synthetic English); (4) weak repo `train_ready/share_alike`; (5) R10 `share_alike` (OLDI Seed, Wikimedia); (6) `data_devhalf` (replaces the weak repo's `flores_plus_dev` rows, 17,043 rows excluded because they include the held-out half); (7) weak repo `synthetic_english_side/`; (8) R10 `NC_noncommercial`, kept **only with the tag** `non_commercial`; (9) R10 `owner_code_switch` = Gemma-4-31B synthetic targets, class `synthetic_gemma4`, licence `owner_private`. Every row has the v.2 decontamination filter (both sides, strict rule) and source/licence per row. The old `ok()` length filter was applied only to the main-output rows (the external builders had their own filters; FineTranslations document chunks exceed `ok()`).
* **Not taken, listed in `excluded.csv`:** weak repo `quarantine/` (tir 3,790, kea 1,901: licence doubt / NLLB-mined kea), R10 `EXCLUDED_unknown_licence` (CCAligned web-mined 70,805 rows, bible-uedin 48,307 rows: unknown licence class), the weak repo's `flores_plus_dev` rows (replaced). No row whose source mentions NLLB or TranslatePsy was found in these inputs (none was listed because none matched). **Gemma-4-31B synthetic kea/kri pairs: not found in the backups** (checked: R10 backup, final-backup file names, r11, r14, RL archive). What exists is `owner_code_switch` (Gemma-4 translations of English inside the African-Code-Switch data): 1,395 rows are in the specialist set for 11 of the 19 languages (amh 693, wol 594, tir 27, tum 27, luo 23 ...; kea none), class `synthetic_gemma4`; kea and kri have no such pairs. The final-backup `datagen/out/kri` files are seed documents, not pairs.
* **Largest removals by the filter** (`removed_by_filter_and_duplicates.csv`): TICO-19 rows of the weak repo (tir 6,099, nus 3,048: TICO-19 is protected, the weak repo had not excluded it), AfriDocMT dev/test rows (amh 2,969), MAFAND dev (amh 302), parts of FineTranslations tir/amh (8-gram overlaps with AfriLongDoc/SmolDoc web text).
* **Human vs synthetic.** `kind` = `human` (human translation or human-written parallel text, including JW/Bible mt560 and multi-open whose translation provenance is not stated), `synthetic_english_side` (real target-language web text with a machine-generated English side: FineTranslations), `synthetic_gemma4`.
* **Ge'ez exposure: amh 52,523,077 Gemma target tokens in 799,510 pairs (human 32.2 M, synthetic English side 20.2 M, Gemma-4 0.09 M); tir 43,333,968 tokens in 397,777 pairs (human 15.3 M, synthetic English side 28.0 M = 64.7%).** Their tokens by licence class are in the table below (amh: 48.8 M permissive/commercial, tir: 42.6 M).
* **Under 2,000 pairs: none of the 19 languages** (smallest: nus 7,105 pairs, kbp 9,970, dyu 14,714); several are thin in human text: nus 6,715 human pairs, bam 14,710, kbp 9,970, dyu 14,714.

Per language (CSV `specialist/per_language_summary.csv`):

| iso | pairs | Gemma target tokens | human pairs | human tokens | synthetic pairs | synthetic tokens | synthetic share of tokens % | permissive/commercial tokens | below 2,000 pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| amh | 799,510 | 52,429,918 | 747,125 | 32,180,263 | 52,385 | 20,249,655 | 38.6 | 48,823,932 | no |
| tir | 397,777 | 43,333,968 | 325,047 | 15,306,801 | 72,730 | 28,027,167 | 64.7 | 42,627,025 | no |
| fon | 126,514 | 4,982,690 | 126,507 | 4,981,775 | 7 | 915 | 0.0 | 2,406,951 | no |
| dyu | 14,714 | 570,208 | 14,714 | 570,208 | 0 | 0 | 0.0 | 549,651 | no |
| kbp | 9,970 | 560,605 | 9,970 | 560,605 | 0 | 0 | 0.0 | 12,047 | no |
| nus | 7,105 | 722,850 | 6,715 | 556,623 | 390 | 166,227 | 23.0 | 168,486 | no |
| mos | 185,742 | 8,330,956 | 185,737 | 8,330,148 | 5 | 808 | 0.0 | 8,307,477 | no |
| kam | 54,791 | 3,998,135 | 50,039 | 2,334,404 | 4,752 | 1,663,731 | 41.6 | 3,977,079 | no |
| bam | 19,098 | 1,922,999 | 14,710 | 505,670 | 4,388 | 1,417,329 | 73.7 | 1,416,329 | no |
| umb | 207,038 | 7,303,851 | 207,033 | 7,303,322 | 5 | 529 | 0.0 | 7,282,591 | no |
| kea | 15,266 | 1,665,882 | 11,529 | 265,265 | 3,737 | 1,400,617 | 84.1 | 1,648,398 | no |
| kmb | 88,499 | 3,223,354 | 88,494 | 3,222,710 | 5 | 644 | 0.0 | 3,201,769 | no |
| vmw | 71,868 | 2,673,742 | 71,868 | 2,673,742 | 0 | 0 | 0.0 | 2,650,177 | no |
| kab | 118,132 | 9,622,450 | 98,432 | 2,298,152 | 19,700 | 7,324,298 | 76.1 | 9,533,302 | no |
| seh | 51,483 | 1,858,595 | 51,483 | 1,858,595 | 0 | 0 | 0.0 | 1,837,464 | no |
| wol | 176,400 | 5,265,238 | 172,539 | 3,911,610 | 3,861 | 1,353,628 | 25.7 | 3,154,168 | no |
| ewe | 31,956 | 1,686,550 | 31,953 | 1,686,153 | 3 | 397 | 0.0 | 50,290 | no |
| luo | 1,519,271 | 31,721,781 | 1,515,228 | 30,212,133 | 4,043 | 1,509,648 | 4.8 | 31,671,958 | no |
| tum | 230,217 | 8,392,727 | 230,190 | 8,390,342 | 27 | 2,385 | 0.0 | 8,262,376 | no |

Gemma target tokens by licence class (`licence` column of the rows; `cc-by` includes the weak repo's CC-BY/Tatoeba/translatewiki rows, `other` = African-Languages-Lab/multi-open with unstated terms, `unspecified` = sources with no usable licence on the card, `non_commercial` = tagged NC rows of the R10 NC folder, Tanzil, TED, AfriDocMT, AfricanFineTranslations, PolyNews, and IbomMT where present):

| iso | cc0 | cc-by | odc-by | apache | mpl | share_alike | other | unspecified | non_commercial | owner_private | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| amh | 2,245,862 | 26,302,464 | 20,156,496 | 0 | 119,110 | 305,173 | 0 | 357,185 | 2,850,469 | 93,159 | 52,429,918 |
| tir | 0 | 14,602,835 | 28,024,190 | 0 | 0 | 166,140 | 0 | 537,826 | 0 | 2,977 | 43,333,968 |
| fon | 0 | 2,406,951 | 0 | 0 | 0 | 38,551 | 2,536,273 | 0 | 0 | 915 | 4,982,690 |
| dyu | 0 | 549,651 | 0 | 0 | 0 | 20,557 | 0 | 0 | 0 | 0 | 570,208 |
| kbp | 0 | 12,047 | 0 | 0 | 0 | 35,838 | 0 | 512,720 | 0 | 0 | 560,605 |
| nus | 0 | 2,259 | 166,227 | 0 | 0 | 554,364 | 0 | 0 | 0 | 0 | 722,850 |
| mos | 0 | 8,307,477 | 0 | 0 | 0 | 22,671 | 0 | 0 | 0 | 808 | 8,330,956 |
| kam | 0 | 2,313,348 | 1,663,731 | 0 | 0 | 21,056 | 0 | 0 | 0 | 0 | 3,998,135 |
| bam | 0 | 0 | 1,416,329 | 0 | 0 | 364,564 | 0 | 141,106 | 0 | 1,000 | 1,922,999 |
| umb | 0 | 7,282,591 | 0 | 0 | 0 | 20,731 | 0 | 0 | 0 | 529 | 7,303,851 |
| kea | 0 | 247,781 | 1,400,617 | 0 | 0 | 17,484 | 0 | 0 | 0 | 0 | 1,665,882 |
| kmb | 0 | 3,201,769 | 0 | 0 | 0 | 20,941 | 0 | 0 | 0 | 644 | 3,223,354 |
| vmw | 0 | 2,650,177 | 0 | 0 | 0 | 23,565 | 0 | 0 | 0 | 0 | 2,673,742 |
| kab | 0 | 2,209,004 | 7,324,298 | 0 | 0 | 89,148 | 0 | 0 | 0 | 0 | 9,622,450 |
| seh | 0 | 1,837,464 | 0 | 0 | 0 | 21,131 | 0 | 0 | 0 | 0 | 1,858,595 |
| wol | 0 | 686,363 | 1,259,074 | 1,124,541 | 84,190 | 34,401 | 1,379,019 | 603,096 | 0 | 94,554 | 5,265,238 |
| ewe | 0 | 15,586 | 0 | 0 | 34,704 | 77,541 | 1,554,188 | 4,134 | 0 | 397 | 1,686,550 |
| luo | 0 | 30,163,903 | 1,508,055 | 0 | 0 | 48,230 | 0 | 0 | 0 | 1,593 | 31,721,781 |
| tum | 0 | 8,262,376 | 0 | 0 | 0 | 127,966 | 0 | 0 | 0 | 2,385 | 8,392,727 |

Pairs by licence class:

| iso | cc0 | cc-by | odc-by | apache | mpl | share_alike | other | unspecified | non_commercial | owner_private | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| amh | 53,436 | 646,437 | 51,694 | 0 | 4,442 | 2,831 | 0 | 8,770 | 31,209 | 691 | 799,510 |
| tir | 0 | 313,429 | 72,703 | 0 | 0 | 1,723 | 0 | 9,895 | 0 | 27 | 397,777 |
| fon | 0 | 47,457 | 0 | 0 | 0 | 540 | 78,510 | 0 | 0 | 7 | 126,514 |
| dyu | 0 | 14,278 | 0 | 0 | 0 | 436 | 0 | 0 | 0 | 0 | 14,714 |
| kbp | 0 | 564 | 0 | 0 | 0 | 437 | 0 | 8,969 | 0 | 0 | 9,970 |
| nus | 0 | 115 | 390 | 0 | 0 | 6,600 | 0 | 0 | 0 | 0 | 7,105 |
| mos | 0 | 185,300 | 0 | 0 | 0 | 437 | 0 | 0 | 0 | 5 | 185,742 |
| kam | 0 | 49,600 | 4,752 | 0 | 0 | 439 | 0 | 0 | 0 | 0 | 54,791 |
| bam | 0 | 0 | 4,380 | 0 | 0 | 6,609 | 0 | 8,101 | 0 | 8 | 19,098 |
| umb | 0 | 206,594 | 0 | 0 | 0 | 439 | 0 | 0 | 0 | 5 | 207,038 |
| kea | 0 | 11,090 | 3,737 | 0 | 0 | 439 | 0 | 0 | 0 | 0 | 15,266 |
| kmb | 0 | 88,055 | 0 | 0 | 0 | 439 | 0 | 0 | 0 | 5 | 88,499 |
| vmw | 0 | 71,428 | 0 | 0 | 0 | 440 | 0 | 0 | 0 | 0 | 71,868 |
| kab | 0 | 96,959 | 19,700 | 0 | 0 | 1,473 | 0 | 0 | 0 | 0 | 118,132 |
| seh | 0 | 51,044 | 0 | 0 | 0 | 439 | 0 | 0 | 0 | 0 | 51,483 |
| wol | 0 | 16,654 | 3,267 | 81,959 | 6,302 | 631 | 52,919 | 14,074 | 0 | 594 | 176,400 |
| ewe | 0 | 1,122 | 0 | 0 | 2,333 | 1,163 | 27,245 | 90 | 0 | 3 | 31,956 |
| luo | 0 | 1,514,086 | 4,020 | 0 | 0 | 1,142 | 0 | 0 | 0 | 23 | 1,519,271 |
| tum | 0 | 228,124 | 0 | 0 | 0 | 2,066 | 0 | 0 | 0 | 27 | 230,217 |

Sources per language (top 7 by tokens; full table `per_language_source.csv`):

| iso | sources by Gemma target tokens (source / class / kind: tokens) |
|---|---|
| amh | mt560 / cc-by / human: 26,258,513; finetranslations_amh / odc-by / synthetic_english_side: 15,407,563; HuggingFaceFW/finetranslat / odc-by / synthetic_english_side: 4,748,933; opus / cc0 / human: 2,245,862; masakhane/AfriDocMT / non_commercial / human: 2,027,122; opus / non_commercial / human: 710,104; afrolingu-mt / unspecified / human: 357,185 |
| tir | finetranslations_tir / odc-by / synthetic_english_side: 22,409,033; mt560 / cc-by / human: 14,460,081; HuggingFaceFW/finetranslat / odc-by / synthetic_english_side: 5,615,157; afrolingu-mt / unspecified / human: 537,826; gamayun_kits_tig-eng / cc-by / human: 109,070; opus_wikimedia / share_alike / human: 81,444; travis_foundation_tigrinya / share_alike / human: 56,976 |
| fon | multi-open / other / human: 2,536,273; mt560 / cc-by / human: 1,566,754; fon-helsinki / cc-by / human: 831,038; flores-plus-dev-half / share_alike / human: 29,082; opus_wikimedia / share_alike / human: 9,469; opus_translatewiki / cc-by / human: 9,159; <HF_ORG>/African-Code- / owner_private / synthetic_gemma4: 915 |
| dyu | mt560 / cc-by / human: 549,651; flores-plus-dev-half / share_alike / human: 20,557 |
| kbp | afrolingu-mt / unspecified / human: 512,720; flores-plus-dev-half / share_alike / human: 35,838; opus_translatewiki / cc-by / human: 12,047 |
| nus | oldi_seed_nus_Latn / share_alike / human: 522,071; HuggingFaceFW/finetranslat / odc-by / synthetic_english_side: 166,227; flores-plus-dev-half / share_alike / human: 32,117; opus_tatoeba / cc-by / human: 2,162; openlanguagedata/oldi_seed / share_alike / human: 176; opus_translatewiki / cc-by / human: 97 |
| mos | mt560 / cc-by / human: 8,189,143; opus_translatewiki / cc-by / human: 118,334; flores-plus-dev-half / share_alike / human: 22,671; <HF_ORG>/African-Code- / owner_private / synthetic_gemma4: 808 |
| kam | mt560 / cc-by / human: 2,313,247; HuggingFaceFW/finetranslat / odc-by / synthetic_english_side: 1,663,731; flores-plus-dev-half / share_alike / human: 21,056; opus_tatoeba / cc-by / human: 97; opus / cc-by / human: 4 |
| bam | HuggingFaceFW/finetranslat / odc-by / synthetic_english_side: 1,416,329; oldi_seed_bam_Latn / share_alike / human: 340,773; afrolingu-mt / unspecified / human: 141,106; flores-plus-dev-half / share_alike / human: 23,288; <HF_ORG>/African-Code- / owner_private / synthetic_gemma4: 1,000; openlanguagedata/oldi_seed / share_alike / human: 503 |
| umb | mt560 / cc-by / human: 7,282,472; flores-plus-dev-half / share_alike / human: 20,731; <HF_ORG>/African-Code- / owner_private / synthetic_gemma4: 529; opus_tatoeba / cc-by / human: 119 |
| kea | HuggingFaceFW/finetranslat / odc-by / synthetic_english_side: 1,400,617; mt560 / cc-by / human: 147,146; opus_translatewiki / cc-by / human: 100,635; flores-plus-dev-half / share_alike / human: 17,484 |
| kmb | mt560 / cc-by / human: 3,201,769; flores-plus-dev-half / share_alike / human: 20,941; <HF_ORG>/African-Code- / owner_private / synthetic_gemma4: 644 |
| vmw | mt560 / cc-by / human: 2,642,273; flores-plus-dev-half / share_alike / human: 23,494; opus_translatewiki / cc-by / human: 7,904; ssa-mte / share_alike / human: 71 |
| kab | finetranslations_kab / odc-by / synthetic_english_side: 7,324,298; mt560 / cc-by / human: 1,385,774; opus_tatoeba / cc-by / human: 700,986; opus_translatewiki / cc-by / human: 122,244; opus_wikimedia / share_alike / human: 65,878; flores-plus-dev-half / share_alike / human: 23,270 |
| seh | mt560 / cc-by / human: 1,837,464; flores-plus-dev-half / share_alike / human: 21,131 |
| wol | multi-open / other / human: 1,379,019; finetranslations / odc-by / synthetic_english_side: 1,259,074; wolof-sudoping / apache / human: 1,124,541; mt560 / cc-by / human: 658,092; wolof-bible / unspecified / human: 493,022; afrolingu-mt / unspecified / human: 110,074; <HF_ORG>/African-Code- / owner_private / synthetic_gemma4: 94,554 |
| ewe | multi-open / other / human: 1,554,188; opus_wikimedia / share_alike / human: 53,142; pontoon / mpl / human: 34,704; flores-plus-dev-half / share_alike / human: 24,399; opus_translatewiki / cc-by / human: 15,520; afrolingu-mt / unspecified / human: 4,134; <HF_ORG>/African-Code- / owner_private / synthetic_gemma4: 397 |
| luo | luo-opus / cc-by / human: 25,965,078; mt560 / cc-by / human: 4,198,825; finetranslations_luo / odc-by / synthetic_english_side: 1,508,055; ssa-mte / share_alike / human: 28,866; flores-plus-dev-half / share_alike / human: 19,364; <HF_ORG>/African-Code- / owner_private / synthetic_gemma4: 1,593 |
| tum | mt560 / cc-by / human: 8,246,090; opus_wikimedia / share_alike / human: 102,645; flores-plus-dev-half / share_alike / human: 25,321; opus_translatewiki / cc-by / human: 15,929; <HF_ORG>/African-Code- / owner_private / synthetic_gemma4: 2,385; opus_tatoeba / cc-by / human: 357 |

Caveats: (1) mt560 (JW/Bible text, licence unknown, section STEP A) is the largest human source for amh, mos, umb, kmb, tum, vmw, seh, wol and luo (via luo-opus); switch it off with `source_family` where it is present in the main-output rows (`mt560`). (2) The permissive-licence column counts `odc-by` FineTranslations (synthetic English). (3) NC rows are in the file with `licence=non_commercial` so a commercial variant is `licence != 'non_commercial'` plus dropping `owner_private` if Gemma-4 terms matter. (4) The weak repo's translatewiki rows are short UI strings with software-project licences (flag in its `split_note`). (5) Duplicate Kabyle Tatoeba rows may overlap the old SFT (weak repo note). (6) No QE scoring was applied to any source (GPU).

## Open items after the follow-up
1. mt560 licence (JW300/Bible): decide whether to keep it in the main/specialist sets; the switch exists.
2. `multi-open` terms (licence "other"), `unspecified` and `non_commercial` rows are present with their class; owner decision on keeping them in a commercial run.
3. `pristine-twi-english` and `synthetic-parallel-external` stay in `quarantine_nc/`.
4. Gemma-4-31B synthetic kea/kri pairs were not found in the backups; the owner's copy (if any) is elsewhere.
5. The held-out dev half is 557 ids (55.9%); say if a different rule is wanted before reward calibration is fitted.
6. Restore of main-output rows removed by train-half dev matches was not done (a few hundred pairs at most).


---
# Addendum 3 (2026-10-28, lead package: yor in the specialist set, the post-GPU chain, projected bins)

## 1. yor added to the specialist set (pre-registration Amendment 1): protected-2026-10-28.3

**Status 1: done and pushed.** The FLORES+ dev train-half exemption now covers 20 languages (the 19 of step B plus yor), same id lists (held-out 557, train half 440; sha256 unchanged `3b620a92c5193105...` / `79ef4baaa6b73d09...`). The manifest had to change, so it is published as **protected-2026-10-28.3** (`protected_index/protected_manifest.json`, file sha256 of the unchanged index `f094fd2b24d6a0e4...`; new exemption file `protected_exemption_devhalf_v3.npz` sha256 `cc6f8f247e218732...`; the v.2 manifest and exemption stay as `protected_manifest_v2.json` / `protected_exemption_devhalf.npz`). The exemption was trimmed again (213 8-grams that also occur in devtest or the held-out half).
* **Exposure proof re-run for .3** (`exposure_check_v3.json`, same three methods and sample as before: 999,077 main-output pairs + 8,747 devhalf rows, both sides, 352,187 protected FLORES+ sentences = devtest of all languages + the held-out dev half of all languages): A (hash index) **0 hits** in the main sample and in the devhalf rows; B (the audit's contamination.py logic) **0 exact, 0 near, 0 shared 8-grams**; C (independent strict 8-gram sets) **0 texts**. Controls: 1,000 of 1,000 planted held-out/devtest sentences caught by A and B; train-half targets of all 20 listed languages including **yor** and train-half English: 0 flagged (exemption works); train-half text of **non-listed** languages (hau, swh, 300 each): 300 of 300 flagged (still protected; yor is no longer in the control list, hau and swh replace it). The `version` field of that json was corrected from .1 to .3 after the run (the sealing function had overwritten the version string; fixed in `protected_index_loader.py`; index and exemption files are the ones the manifest lists).
* `data_devhalf/pairs-0000.parquet`: now **8,747 rows for 20 languages** (yor train-half rows added). `specialist/`: yor added by running step C for yor alone (same sources and rules; `specialist.py --only-langs yor`, merged by `chain/merge_yor.py`): **yor 971,517 pairs, 81,976,070 Gemma target tokens** (human 927,682 pairs / 64.6 M tokens, synthetic English side 43,835 pairs / 17.3 M tokens = FineTranslations rows of the main output; permissive/commercial 41.3 M tokens; the rest `share_alike`, `other` (multi-open), `unspecified`). **The specialist set is now 20 languages, 5,096,868 pairs.** Updated per-language table (`specialist/per_language_summary.csv`):

| iso | pairs | Gemma target tokens | human pairs | human tokens | synthetic pairs | synthetic tokens | synthetic share % | permissive/commercial tokens | <2,000 pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| amh | 799,510 | 52,429,918 | 747,125 | 32,180,263 | 52,385 | 20,249,655 | 38.6 | 48,823,932 | no |
| tir | 397,777 | 43,333,968 | 325,047 | 15,306,801 | 72,730 | 28,027,167 | 64.7 | 42,627,025 | no |
| fon | 126,514 | 4,982,690 | 126,507 | 4,981,775 | 7 | 915 | 0.0 | 2,406,951 | no |
| dyu | 14,714 | 570,208 | 14,714 | 570,208 | 0 | 0 | 0.0 | 549,651 | no |
| kbp | 9,970 | 560,605 | 9,970 | 560,605 | 0 | 0 | 0.0 | 12,047 | no |
| nus | 7,105 | 722,850 | 6,715 | 556,623 | 390 | 166,227 | 23.0 | 168,486 | no |
| mos | 185,742 | 8,330,956 | 185,737 | 8,330,148 | 5 | 808 | 0.0 | 8,307,477 | no |
| kam | 54,791 | 3,998,135 | 50,039 | 2,334,404 | 4,752 | 1,663,731 | 41.6 | 3,977,079 | no |
| bam | 19,098 | 1,922,999 | 14,710 | 505,670 | 4,388 | 1,417,329 | 73.7 | 1,416,329 | no |
| umb | 207,038 | 7,303,851 | 207,033 | 7,303,322 | 5 | 529 | 0.0 | 7,282,591 | no |
| kea | 15,266 | 1,665,882 | 11,529 | 265,265 | 3,737 | 1,400,617 | 84.1 | 1,648,398 | no |
| kmb | 88,499 | 3,223,354 | 88,494 | 3,222,710 | 5 | 644 | 0.0 | 3,201,769 | no |
| vmw | 71,868 | 2,673,742 | 71,868 | 2,673,742 | 0 | 0 | 0.0 | 2,650,177 | no |
| kab | 118,132 | 9,622,450 | 98,432 | 2,298,152 | 19,700 | 7,324,298 | 76.1 | 9,533,302 | no |
| seh | 51,483 | 1,858,595 | 51,483 | 1,858,595 | 0 | 0 | 0.0 | 1,837,464 | no |
| wol | 176,400 | 5,265,238 | 172,539 | 3,911,610 | 3,861 | 1,353,628 | 25.7 | 3,154,168 | no |
| ewe | 31,956 | 1,686,550 | 31,953 | 1,686,153 | 3 | 397 | 0.0 | 50,290 | no |
| luo | 1,519,271 | 31,721,781 | 1,515,228 | 30,212,133 | 4,043 | 1,509,648 | 4.8 | 31,671,958 | no |
| tum | 230,217 | 8,392,727 | 230,190 | 8,390,342 | 27 | 2,385 | 0.0 | 8,262,376 | no |
| yor | 971,517 | 81,976,070 | 927,682 | 64,647,449 | 43,835 | 17,328,621 | 21.1 | 41,302,359 | no |

Gemma target tokens by licence class (20 languages):

| iso | cc0 | cc-by | odc-by | apache | mpl | share_alike | other | unspecified | non_commercial | owner_private | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| amh | 2,245,862 | 26,302,464 | 20,156,496 | 0 | 119,110 | 305,173 | 0 | 357,185 | 2,850,469 | 93,159 | 52,429,918 |
| tir | 0 | 14,602,835 | 28,024,190 | 0 | 0 | 166,140 | 0 | 537,826 | 0 | 2,977 | 43,333,968 |
| fon | 0 | 2,406,951 | 0 | 0 | 0 | 38,551 | 2,536,273 | 0 | 0 | 915 | 4,982,690 |
| dyu | 0 | 549,651 | 0 | 0 | 0 | 20,557 | 0 | 0 | 0 | 0 | 570,208 |
| kbp | 0 | 12,047 | 0 | 0 | 0 | 35,838 | 0 | 512,720 | 0 | 0 | 560,605 |
| nus | 0 | 2,259 | 166,227 | 0 | 0 | 554,364 | 0 | 0 | 0 | 0 | 722,850 |
| mos | 0 | 8,307,477 | 0 | 0 | 0 | 22,671 | 0 | 0 | 0 | 808 | 8,330,956 |
| kam | 0 | 2,313,348 | 1,663,731 | 0 | 0 | 21,056 | 0 | 0 | 0 | 0 | 3,998,135 |
| bam | 0 | 0 | 1,416,329 | 0 | 0 | 364,564 | 0 | 141,106 | 0 | 1,000 | 1,922,999 |
| umb | 0 | 7,282,591 | 0 | 0 | 0 | 20,731 | 0 | 0 | 0 | 529 | 7,303,851 |
| kea | 0 | 247,781 | 1,400,617 | 0 | 0 | 17,484 | 0 | 0 | 0 | 0 | 1,665,882 |
| kmb | 0 | 3,201,769 | 0 | 0 | 0 | 20,941 | 0 | 0 | 0 | 644 | 3,223,354 |
| vmw | 0 | 2,650,177 | 0 | 0 | 0 | 23,565 | 0 | 0 | 0 | 0 | 2,673,742 |
| kab | 0 | 2,209,004 | 7,324,298 | 0 | 0 | 89,148 | 0 | 0 | 0 | 0 | 9,622,450 |
| seh | 0 | 1,837,464 | 0 | 0 | 0 | 21,131 | 0 | 0 | 0 | 0 | 1,858,595 |
| wol | 0 | 686,363 | 1,259,074 | 1,124,541 | 84,190 | 34,401 | 1,379,019 | 603,096 | 0 | 94,554 | 5,265,238 |
| ewe | 0 | 15,586 | 0 | 0 | 34,704 | 77,541 | 1,554,188 | 4,134 | 0 | 397 | 1,686,550 |
| luo | 0 | 30,163,903 | 1,508,055 | 0 | 0 | 48,230 | 0 | 0 | 0 | 1,593 | 31,721,781 |
| tum | 0 | 8,262,376 | 0 | 0 | 0 | 127,966 | 0 | 0 | 0 | 2,385 | 8,392,727 |
| yor | 0 | 23,952,601 | 17,225,453 | 0 | 124,305 | 76,463 | 40,176,148 | 317,932 | 0 | 103,168 | 81,976,070 |

## 2. The chain that runs after the GPU is available (all CPU-tested)

**Status 2: done and pushed to `chain/` on the Hub and in `data_pipeline/sft_clean/chain/`** (qe_score.py, gate.py, hcal.py, windows.py, pack.py, sampler.py, project.py, merge_yor.py, make_sample.py, test_chain.py, README_chain.md with the exact command sequence for the GPU session). Tested on a 20k-pair sample of the clean output (75 languages, 170 source groups) on the box, 2 cores, plus the full-corpus windowing below.

| step | what was tested | result |
|---|---|---|
| (a) `qe_score.py` | stub scorer on 20,000 pairs (writes `scores_STUB/`, scorer column `STUB-NOT-REAL`, refused by `gate.py` unless `--allow-stub`, all gate outputs then prefixed `STUB_`); the REAL AfriCOMET-QE-STL-1.1 on CPU (torch cpu + unbabel-comet in a separate venv because comet pins numpy<2): 300-pair smoke test (mean QE 0.644, range 0.33-0.92) and a 2,000-pair sample of the 20 specialist languages | works; CPU speed 1.5-2.1 pairs/s on 2 cores (only a functional test); resumable per chunk, `--part i/N` for several processes, `--skip-keys-of` so that `specialist/` does not re-score main rows, prints pairs/s and ETA |
| (b) `gate.py` | `fit` computes BOTH rules side by side and drops nothing (`keep_rate_per_language.csv`, `thresholds.json`); `apply --rule old|cal` writes the keep-key list; unit test with synthetic scores (OLD keeps 85%; CAL respects the 5% floor and the 40% cap) | works |
| (c) `windows.py` | same definitions as the old `gemma_windows.py` (bins b64 24-64, b256 65-256, b512 257-512, b1024 513-1024 AfriXLM-R tokens, token-balanced bin choice, uniform budget in the bin, terminator rules, pcm x2); test: consecutive pairs stay consecutive and in order on both sides; dedupe across directories; keep-file gate; full run below | works |
| (d) `pack.py` | `prepare` = the exact old command (`--seq-len 4096 --max-sample-tokens 4096 --directions en2x --pack-by-language --val-fraction 0.002 --min-val-per-language 60 --decontaminate pcm`, then `audit_packed` train and val) on the real windows of the sample: 8,453 train + 948 val samples, 555 packs, **AUDIT OK**; `replay` (S2 re-tokenisation of existing packs) selftest: same tokenizer gives byte-identical tokens | works |
| (e) `sampler.py` | stage A/B schedules with the old functions `balanced_schedule` / `unimax_schedule`, from a real packed dataset and from a packs-per-language table; `arms` gives S1 = the A0 draws on specialist-language packs in A0's order (unit test: exact subsequence, every specialist language keeps A0's passes), S2 = same list | works |

### QE cost estimate (the question of GPU hours)
* **Pairs to score: about 23.77 M** (unique union of `data/` 23,406,354 + the 365,774 external specialist rows incl. devhalf; the 4,731,094 specialist rows that repeat main rows are scored once). Human references of H_cal: about 280 ids x 100 languages, negligible.
* **Throughput assumption:** the old measurements (`reports/gemma_rl_cost_estimate_2026-10-17.md`): AfriCOMET-QE fp16 on an RTX 5070, 290-350 pairs/s with one process, about 480 pairs/s aggregate with three processes. For an RTX PRO 6000 (Blackwell, assumed about 2.3x the 5070 on this encoder workload, **an assumption, not a measurement**): **1,100 pairs/s -> 6.0 GPU-hours**; at the old 480/s 13.8 h; at 1,500/s 4.4 h. The CPU smoke test (1.5-2.1 pairs/s on 2 cores) is not informative for the GPU. `qe_score.py --budget-pairs-per-s` prints the ETA at the assumed rate next to the measured one, so the first minutes of the GPU session replace the assumption by a measurement.
* The OLD rule needs all pairs scored (the 15th percentile alone could be estimated from a sample, but the pairs below it must be identified).

### Did a FLORES-performance-dependent threshold exist in the earlier build?
Searched `gemma_mt/DECISIONS.md`, the old `gemma_gate_pairs.py` and the old converters. **The SFT gate itself never used FLORES:** `gemma_gate_pairs.py` is the per-language 15th-percentile rule with the 0.20 minimum and 2,000-row minimum only (as the lead found). What does exist and is FLORES-dependent: (1) the **RL reward**: `rl/calibrate_shaping.py` sets `t_fail = max(floor, P90 of QE on mismatched references)`, `t_mid`/`t_great` = P50/P90 of QE(source, human reference) on the FLORES+ **dev** references per language, and `reward.fail_floor` moves t_fail for kam, lua, pcm (D-30 area, `DECISIONS.md` lines 167-170 and 477); (2) the old `mt560_qe` converter used a fixed QE >= 0.46 for fon/oromo/setswana (the AfriCOMET "failure line"). The new rule CAL imitates (1) for the SFT gate; it is new, not a restoration of an old SFT rule.

### Rule CAL as implemented (`gate.py`)
threshold_lang = max(0.20, 10th percentile of the QE of the language's FLORES+ dev **H_cal** human references); languages without a FLORES dev file use the pooled median offset d = median over languages with references of (P10(ref QE) - median(training QE)): threshold = max(0.20, median(training QE) + d); cap so that no language loses more than 40% (threshold <= P40 of its scored pairs), floor so that at least 5% are dropped (threshold >= P05); min 2,000 rows. **H_cal / H_sel:** the pre-registration gives no list, so it is fixed in `chain/hcal.py` and written to `protected_index/heldout_split_cal_sel.json`: the held-out ids already are the odd-hash ids plus the 100 forced ones, so the next sha256 bit is used (`(int(sha256(str(id)).hexdigest(),16) >> 1) & 1 == 0` -> H_cal): **H_cal = 282 ids (sha256 `baa4289330228c8c...`), H_sel = 275 ids (`d2254f1d3afd02a0...`)**; of the forced 100 ids 58 are in H_cal and 42 in H_sel. The H_cal human-reference pairs are benchmark text: `hcal.py refs` writes them locally on the GPU box only.

### Source order and what had to be assumed for the windows
The D2 output keeps the row order of every source file (dedupe keeps the first occurrence, one shard group per input file in the original order; the specialist set lists main rows first and then the external sources, each in file order), so consecutive pairs of one run (language, source, direction) are consecutive source rows with gaps where rows were removed. As in the old build (14 of 170 sources keep a weak document order) a window is a pseudo-paragraph: alignment sentence by sentence is exact, discourse coherence is not. Assumptions made: run key = (iso, full source_dataset, flipped) instead of the old base-name key (the specialist sources `weak-repo:<id>` would otherwise merge); English -> X only; cross-directory duplicates removed by key; the S1/S2 stream is defined on packs, and the trainer has no option to read a schedule file today (`train_sft.py` builds its schedule internally): **open item, a ~15-line change** (`balanced_schedule(...)` -> `np.load(schedule.npy)`).

## 3. Projected breakdown without the GPU (UNGATED, mt560 included; real figures need the QE scores)

**Status 3: done** (`reports/data/sft_clean_2026-10-28/bins/`, also on the Hub in `stats/bins/`). The windowing was run on the whole clean corpus: `data/` + `data_devhalf/` + `specialist/` (20 languages), deduplicated across directories: **23,794,632 pairs (duplicates across directories removed: 4,739,841; pcm second pass 22,504) -> 10,307,487 windows** (standalone one-sided pairs 1,244,285; full stops added to 3,543,683 pairs) in 4,395 s on 2 CPU cores. Old build for comparison: 11,442,172 windows, 538,480 packs, 2,204,187,720 tokens, 1,318,270,369 Gemma target tokens. English tokens are AfriXLM-R tokens (as the old table), target tokens are Gemma-3 tokens; the Gemma counts were measured exactly on a 3% sample of the pairs and scaled per language through the sampled Gemma/AfriXLM-R ratio (checked on the 20k test: total +1.9%, median per-language error 1.8%, max 3.2% at a 10% sample). Packs = ceil(sample tokens / (4,096 x 0.9994)); sample tokens = English + target Gemma tokens + the template overhead measured with the real chat format (19-22 tokens per sample); the formula over-predicted the sample tokens by 3.3% on the 20k test (some windows are dropped as duplicates or val overlaps). Packs per bin are the language's packs apportioned by the bin's share of sample tokens, because a pack mixes bins.

Per-bin totals, ungated (`per_bin_totals_ungated.csv`):

| bin | windows | English tokens (AfriXLM-R) | target tokens (Gemma) | packs (apportioned) |
|---|---:|---:|---:|---:|
| b64 | 6,293,941 | 167,469,741 | 249,585,722 | 127,740 |
| b256 | 2,550,926 | 183,689,792 | 300,974,389 | 125,020 |
| b512 | 1,120,492 | 287,987,557 | 461,730,887 | 179,857 |
| b1024 | 342,128 | 164,398,966 | 254,962,444 | 98,725 |
| total | 10,307,487 | 803,546,056 | 1,267,253,442 | 531,343 |

(old build, AfriCOMET-token bins: b64 7,150,630 / b256 2,750,979 / b512 967,519 / b1024 573,044 windows.) Total **531,343 packs** ungated (old 538,480).

Per language and bin, ungated: `windows_per_language_bin_ungated.csv` (all 77 languages; the table below lists the 12 largest and the 20 specialist languages; cells = windows / English tokens / target tokens (Gemma) / packs):

| iso | b64 win / Eng tok / Tgt tok / packs | b256 | b512 | b1024 |
|---|---|---|---|---|
| yor | 311,392 / 6,330,608 / 12,201,655 / 5,810 | 473,119 / 21,110,870 / 46,976,080 / 18,263 | 33,756 / 6,228,936 / 12,259,473 / 4,502 | 22,631 / 6,321,857 / 12,225,503 / 4,462 |
| afr | 530,021 / 16,648,385 / 20,355,352 / 11,234 | 136,554 / 16,490,247 / 20,649,686 / 9,269 | 50,743 / 16,144,290 / 20,712,380 / 8,774 | 24,576 / 16,673,434 / 20,712,599 / 8,754 |
| zul | 323,645 / 9,318,275 / 13,970,407 / 6,877 | 98,695 / 9,041,198 / 14,005,344 / 5,783 | 72,957 / 19,278,160 / 31,591,215 / 12,115 | 16,651 / 9,102,196 / 14,006,860 / 5,416 |
| xho | 396,232 / 10,858,480 / 16,451,050 / 8,433 | 118,069 / 10,486,832 / 16,513,682 / 6,880 | 55,701 / 14,245,161 / 23,932,591 / 9,152 | 19,614 / 10,582,526 / 16,524,213 / 6,376 |
| nya | 278,158 / 7,517,127 / 11,211,345 / 5,684 | 90,925 / 7,243,803 / 11,254,579 / 4,725 | 58,781 / 15,722,584 / 25,170,249 / 9,763 | 14,628 / 7,285,088 / 11,260,886 / 4,364 |
| ibo | 365,547 / 8,022,429 / 13,169,380 / 6,626 | 203,019 / 9,253,070 / 17,017,903 / 7,079 | 36,384 / 7,782,553 / 13,577,631 / 5,153 | 27,296 / 7,374,427 / 13,241,963 / 4,942 |
| hau | 219,215 / 6,369,408 / 9,020,930 / 4,647 | 71,680 / 6,223,106 / 9,065,267 / 3,911 | 69,191 / 19,794,822 / 29,274,342 / 11,779 | 11,158 / 6,290,274 / 9,093,177 / 3,640 |
| swh | 215,891 / 6,767,028 / 9,445,547 / 4,920 | 52,291 / 6,743,675 / 9,633,747 / 4,121 | 55,929 / 17,046,945 / 24,713,372 / 10,115 | 9,998 / 6,869,797 / 9,686,859 / 3,946 |
| amh | 179,328 / 5,839,094 / 8,621,817 / 4,251 | 64,607 / 7,265,223 / 11,603,939 / 4,697 | 50,377 / 14,495,735 / 24,047,672 / 9,191 | 8,721 / 5,979,974 / 9,141,423 / 3,540 |
| sna | 271,911 / 7,696,333 / 10,708,089 / 5,545 | 89,154 / 7,487,802 / 10,732,184 / 4,628 | 46,968 / 13,239,889 / 20,154,126 / 7,940 | 14,420 / 7,466,240 / 10,736,177 / 4,266 |
| som | 118,296 / 3,489,244 / 5,101,372 / 2,550 | 41,285 / 4,088,759 / 6,424,028 / 2,645 | 78,027 / 20,501,576 / 34,320,416 / 13,180 | 5,702 / 3,503,012 / 5,234,214 / 2,063 |
| tir | 92,668 / 2,252,002 / 3,964,126 / 1,928 | 46,646 / 3,759,976 / 7,163,071 / 2,797 | 64,560 / 13,997,155 / 27,741,598 / 10,115 | 6,152 / 2,588,662 / 4,962,323 / 1,800 |
| luo | 254,074 / 6,112,783 / 7,690,088 / 4,498 | 55,726 / 6,123,652 / 8,005,759 / 3,596 | 23,552 / 6,910,248 / 9,042,470 / 3,866 | 10,601 / 6,129,560 / 7,970,592 / 3,366 |
| kab | 16,500 / 362,582 / 622,879 / 314 | 10,540 / 798,658 / 1,384,185 / 570 | 15,517 / 3,950,491 / 6,378,510 / 2,525 | 2,176 / 798,890 / 1,367,628 / 525 |
| tum | 53,670 / 1,455,419 / 2,135,406 / 1,103 | 17,754 / 1,410,071 / 2,164,647 / 912 | 5,683 / 1,410,744 / 2,148,739 / 845 | 2,817 / 1,406,736 / 2,148,324 / 829 |
| mos | 52,359 / 1,227,822 / 2,123,618 / 1,025 | 22,672 / 1,105,266 / 2,132,409 / 858 | 5,898 / 1,138,451 / 2,133,502 / 783 | 2,944 / 1,132,799 / 2,133,032 / 768 |
| umb | 47,403 / 1,290,791 / 1,853,319 / 961 | 15,203 / 1,241,922 / 1,867,283 / 790 | 5,035 / 1,252,836 / 1,871,111 / 741 | 2,515 / 1,252,659 / 1,871,389 / 728 |
| wol | 35,818 / 996,076 / 1,318,409 / 714 | 11,181 / 994,953 / 1,387,284 / 611 | 3,405 / 909,738 / 1,357,546 / 548 | 1,883 / 953,843 / 1,322,848 / 541 |
| fon | 34,680 / 573,176 / 1,171,308 / 574 | 15,866 / 672,692 / 1,579,305 / 608 | 3,432 / 508,089 / 1,171,176 / 414 | 1,709 / 507,190 / 1,172,271 / 406 |
| kam | 14,508 / 326,541 / 608,072 / 292 | 7,550 / 413,106 / 803,893 / 323 | 5,093 / 992,515 / 1,943,108 / 714 | 983 / 343,747 / 700,470 / 250 |
| kmb | 20,970 / 530,049 / 821,963 / 424 | 7,172 / 507,859 / 824,668 / 346 | 2,289 / 508,890 / 824,475 / 320 | 1,119 / 511,064 / 825,187 / 314 |
| vmw | 17,153 / 442,863 / 680,653 / 352 | 5,847 / 416,310 / 683,494 / 285 | 1,813 / 420,243 / 683,179 / 264 | 902 / 418,439 / 683,857 / 259 |
| bam | 4,993 / 119,612 / 194,199 / 98 | 3,685 / 266,966 / 452,909 / 188 | 2,643 / 634,078 / 1,115,292 / 426 | 307 / 109,819 / 194,358 / 73 |
| seh | 11,724 / 308,725 / 474,246 / 235 | 3,737 / 298,146 / 475,698 / 196 | 1,237 / 298,898 / 476,133 / 184 | 612 / 296,871 / 476,236 / 181 |
| ewe | 8,697 / 223,644 / 365,829 / 182 | 6,372 / 343,124 / 625,610 / 261 | 1,042 / 201,949 / 366,915 / 140 | 501 / 202,750 / 365,205 / 137 |
| kea | 1,850 / 51,790 / 71,353 / 40 | 1,285 / 155,522 / 198,816 / 90 | 2,831 / 906,039 / 1,192,054 / 508 | 376 / 158,498 / 222,205 / 92 |
| nus | 1,826 / 38,582 / 84,969 / 38 | 4,102 / 167,540 / 429,009 / 161 | 306 / 65,900 / 134,866 / 48 | 134 / 37,966 / 85,314 / 30 |
| dyu | 3,641 / 89,541 / 144,765 / 72 | 1,285 / 83,038 / 145,749 / 59 | 399 / 84,666 / 146,268 / 55 | 198 / 84,521 / 145,364 / 54 |
| kbp | 2,881 / 57,205 / 114,932 / 56 | 2,495 / 91,773 / 226,370 / 89 | 334 / 49,231 / 115,316 / 40 | 167 / 49,767 / 115,510 / 40 |

**Gated projection (ESTIMATE).** The QE sample (CPU, 100 pairs per specialist language, 2,000 pairs) gives for rule OLD a keep rate of 85% for every specialist language by construction (the 15th percentile binds; the 0.20 floor never binds: all 15th percentiles are 0.28-0.64 and at most 2% of the sampled pairs are below 0.20):

| iso | n | median QE | 15th percentile | share < 0.20 |
|---|---:|---:|---:|---:|
| amh | 100 | 0.712 | 0.516 | 0.01 |
| bam | 100 | 0.423 | 0.298 | 0.02 |
| dyu | 100 | 0.659 | 0.536 | 0.0 |
| ewe | 100 | 0.584 | 0.467 | 0.0 |
| fon | 100 | 0.589 | 0.507 | 0.0 |
| kab | 100 | 0.491 | 0.383 | 0.01 |
| kam | 100 | 0.688 | 0.574 | 0.0 |
| kbp | 100 | 0.622 | 0.513 | 0.0 |
| kea | 100 | 0.416 | 0.312 | 0.01 |
| kmb | 100 | 0.682 | 0.563 | 0.0 |
| luo | 100 | 0.469 | 0.284 | 0.02 |
| mos | 100 | 0.707 | 0.631 | 0.0 |
| nus | 100 | 0.453 | 0.356 | 0.0 |
| seh | 100 | 0.679 | 0.534 | 0.0 |
| tir | 100 | 0.737 | 0.634 | 0.0 |
| tum | 100 | 0.702 | 0.589 | 0.0 |
| umb | 100 | 0.682 | 0.586 | 0.0 |
| vmw | 100 | 0.435 | 0.35 | 0.01 |
| wol | 100 | 0.547 | 0.442 | 0.0 |
| yor | 100 | 0.669 | 0.546 | 0.0 |

Keep rates of all other languages are the flat **15% drop (ESTIMATE)**; windows, tokens and packs are scaled by the keep rate (a real gate changes which pairs form windows, so the final counts will differ a little). Totals (`windows_per_language_bin_gated_ESTIMATE.csv`, `per_language_gated_ESTIMATE.csv`):

| bin | windows | English tokens | target tokens (Gemma) | packs |
|---|---:|---:|---:|---:|
| b64 | 5,349,815 | 142,349,241 | 212,147,827 | 108,580 |
| b256 | 2,168,253 | 156,136,289 | 255,828,193 | 106,267 |
| b512 | 952,379 | 244,789,390 | 392,471,219 | 152,878 |
| b1024 | 290,769 | 139,739,084 | 216,718,044 | 83,916 |
| total | 8,761,216 | 683,014,004 | 1,077,165,283 | 451,641 |

**Stage shares and effective epochs, old vs new, same columns as the 2026-10-14 training plan** (`stages_ungated.csv`, `stages_gated_ESTIMATE.csv`; stage A = old SFT: p ~ packs**0.5 floored at 0.2%, language retired after 4 passes, 32 packs/step, steps = one natural pass over all packs (16,827 in the old run; 16,604 ungated here); stage B = UniMax, 20 priority languages x4, min 1 / max 2 epochs, draws scaled to 1.36 x packs as in the old run; A0 = the 77-language stream, S1 = exactly the draws of A0 on specialist-language packs in A0's order, so S1's effective epochs equal A0's). Ungated, 12 largest languages and the 20 specialist languages:

| iso | old packs | new windows | new target tokens (Gemma) | new packs | A0 share % (old -> new) | A0 effective epochs (old -> new) | S1 share of S1 draws % | B share % / epochs (new) | val windows est |
|---|---:|---:|---:|---:|---|---|---:|---|---:|
| yor | 56,817 | 840,898 | 83,662,712 | 33,037 | 5.38 -> 3.654 | 0.5 -> 0.588 | 15.672 | 4.571 / 1.0 | 1681 |
| afr | 15,005 | 741,894 | 82,430,020 | 38,030 | 2.74 -> 3.868 | 1.0 -> 0.54 | - | 5.262 / 1.0 | 1483 |
| zul | 13,422 | 511,948 | 73,573,827 | 30,191 | 2.62 -> 3.51 | 1.0 -> 0.618 | - | 4.315 / 1.033 | 1023 |
| xho | 18,924 | 589,616 | 73,421,538 | 30,842 | 3.06 -> 3.524 | 0.9 -> 0.607 | - | 4.315 / 1.011 | 1179 |
| nya | 8,465 | 442,492 | 58,897,061 | 24,536 | 2.08 -> 3.13 | 1.3 -> 0.678 | - | 4.315 / 1.271 | 884 |
| ibo | 105,821 | 632,246 | 57,006,879 | 23,800 | 7.31 -> 3.09 | 0.4 -> 0.69 | - | 4.315 / 1.31 | 1264 |
| hau | 58,519 | 371,244 | 56,453,717 | 23,977 | 5.43 -> 3.123 | 0.5 -> 0.692 | - | 4.315 / 1.301 | 742 |
| swh | 15,347 | 334,109 | 53,479,527 | 23,102 | 2.79 -> 3.034 | 1.0 -> 0.698 | - | 4.315 / 1.35 | 668 |
| amh | 3,421 | 303,033 | 53,414,853 | 21,679 | 1.28 -> 2.97 | 2.0 -> 0.728 | 12.741 | 3.0 / 1.0 | 606 |
| sna | 6,657 | 422,453 | 52,330,578 | 22,379 | 1.81 -> 3.009 | 1.5 -> 0.714 | - | 4.315 / 1.393 | 844 |
| som | 14,873 | 243,310 | 51,080,031 | 20,438 | 2.75 -> 2.836 | 1.0 -> 0.737 | - | 4.315 / 1.526 | 486 |
| tir | 2,180 | 210,026 | 43,831,119 | 16,639 | 1.06 -> 2.586 | 2.6 -> 0.826 | 11.087 | 2.302 / 1.0 | 420 |
| luo | 7,559 | 343,953 | 32,708,911 | 15,325 | 1.95 -> 2.484 | 1.4 -> 0.861 | 10.652 | 2.121 / 1.0 | 687 |
| kab | 1,594 | 44,733 | 9,753,204 | 3,934 | 0.87 -> 1.248 | 2.9 -> 1.686 | 5.354 | 1.079 / 1.982 | 89 |
| tum | 1,592 | 79,924 | 8,597,117 | 3,690 | 0.89 -> 1.224 | 3.0 -> 1.762 | 5.25 | 1.021 / 2.0 | 159 |
| mos | 1,862 | 83,873 | 8,522,562 | 3,434 | 1.0 -> 1.176 | 2.9 -> 1.819 | 5.044 | 0.95 / 2.0 | 167 |
| umb | 1,541 | 70,156 | 7,463,103 | 3,221 | 0.89 -> 1.138 | 3.1 -> 1.877 | 4.881 | 0.891 / 2.0 | 140 |
| wol | 2,214 | 52,287 | 5,386,088 | 2,414 | 1.05 -> 0.979 | 2.5 -> 2.154 | 4.199 | 0.668 / 2.0 | 104 |
| fon | 13,172 | 55,687 | 5,094,062 | 2,003 | 2.55 -> 0.899 | 1.0 -> 2.385 | 3.857 | 0.554 / 2.0 | 111 |
| kam | 716 | 28,134 | 4,055,545 | 1,580 | 0.53 -> 0.778 | 4.0 -> 2.616 | 3.338 | 0.437 / 2.0 | 60 |
| kmb | 1,036 | 31,550 | 3,296,293 | 1,404 | 0.72 -> 0.764 | 3.7 -> 2.891 | 3.277 | 0.389 / 2.0 | 63 |
| vmw | 874 | 25,715 | 2,731,184 | 1,160 | 0.65 -> 0.676 | 4.0 -> 3.097 | 2.901 | 0.321 / 2.0 | 60 |
| bam | 154 | 11,628 | 1,956,760 | 785 | 0.11 -> 0.553 | 4.0 -> 3.745 | 2.374 | 0.217 / 2.0 | 60 |
| seh | 582 | 17,310 | 1,902,314 | 796 | 0.43 -> 0.558 | 4.0 -> 3.726 | 2.395 | 0.22 / 2.0 | 60 |
| ewe | 1,270 | 16,612 | 1,723,561 | 720 | 0.8 -> 0.535 | 3.4 -> 3.949 | 2.295 | 0.199 / 2.0 | 60 |
| kea | 50 | 6,342 | 1,684,430 | 730 | 0.04 -> 0.532 | 4.0 -> 3.873 | 2.282 | 0.202 / 2.0 | 60 |
| nus | 3,747 | 6,368 | 734,160 | 277 | 1.36 -> 0.209 | 2.0 -> 4.0 | 0.895 | 0.077 / 2.0 | 60 |
| dyu | 192 | 5,523 | 582,147 | 241 | 0.14 -> 0.181 | 4.0 -> 4.0 | 0.778 | 0.067 / 2.0 | 60 |
| kbp | 162 | 5,877 | 572,129 | 225 | 0.12 -> 0.169 | 4.0 -> 4.0 | 0.727 | 0.062 / 2.0 | 60 |

Reading: the new A0 gives the 20 specialist languages 23.3% of its draws; S1 gets 123,865 draws = 3,870 optimizer steps of 32 packs ungated (the 20 languages hold 113,294 packs, 1.09 draws per pack on average because the small languages are repeated) and 105,437 draws = 3,294 steps in the gated estimate (96,308 packs); A0 has 16,604 steps ungated. **The old amh/tir/nus/pcm effective epochs (2.0/2.6/2.0/1.6) do not carry over:** amh and tir now have 6x and 8x more packs (the specialist data) and are drawn 0.7-0.8 times; pcm (22.5k pairs) and nus (6.4k windows) are small and sit at the 4-pass cap (4.0). The per-language pass counts of A0 are what S1 inherits, so S1's weak small languages get up to 4 passes while the large specialist languages get less than one.
* **These are projections.** Final figures need the real QE scores (gate), which change pair counts, window composition, packs and the A0 pass counts that S1 inherits.

## Open items after addendum 3
1. The trainer needs a `schedule.npy` input for S1/S2 (currently internal sampler only).
2. Real QE throughput on the RTX PRO 6000 (assumed 1,100 pairs/s -> 6.0 GPU-h; first minutes of the run give the measurement); decide rule OLD vs CAL after `gate.py fit` (both keep rates are printed before anything is dropped).
3. pcm, nus, dag, bbj, nbl data gaps (unchanged): pcm has 22.5k pairs, nus 6.4k windows; they sit at the 4-pass cap.
4. The held-out/H_cal split rule is mine (next sha256 bit); the lead should confirm it before reward calibration.
5. Exposure json for .3 was patched for its version field (documented in the file); a rerun is cheap if the lead wants a clean file.


---
# Addendum 4 (2026-10-29): commercial-use rule in the chain, fresh-instance runbook, projections without NC rows

**Status: prepared on the laptop only (CPU, no instance reached; box 2 is lost, what was pushed is on the Hub). Nothing below was run on data; the projections scale the measured window statistics.**

## 1. Non-commercial rows are removed inside the chain (Amendment 4)
Files (new or changed, all in `data_pipeline/sft_clean/chain/`): `common.py` (`is_nc`, `assert_clean_src`), `windows.py` (drops NC rows BEFORE the cross-directory dedupe so that an NC copy cannot mask a commercial copy of the same pair; `stats.json` has `dropped_nc` and `dropped_nc_by_iso`; `--include-nc` is for tests only), `qe_score.py --exclude-nc` (NC rows are not scored), `gate.py --exclude-nc` (NC rows do not enter the thresholds or the keep list; also several `--scores` directories and `--refs-json-dir`), `nc_filter.py` (`report`: rows per language and source plus rows whose licence text looks NC under another class; `guard-windows`: fails if a window carries `non_commercial`; `write-clean`), `arms_extra.py` (G0 and the fallback A2 subset; DELETED in addendum 5: sampler.py select/subset replace it), `qe_crosscheck.py prepare --from-scores` (recovers the 500 cross-check pairs by key because the queue text was on the lost box), `project_nc.py`. Tests (`test_chain.py`, 7 passed, 2 skipped without real tokenizers): NC rows dropped before the dedupe, the guard fires when NC rows are kept, quarantine directories refused, the gate thresholds ignore NC rows, G0 = exact subsequence of A0's draws on amh/tir, A2 fallback = about 20% of every language's draws.
**Does anything else reintroduce NC rows?** Checked in the code and the build statistics: `data_devhalf/` = FLORES+ dev rows, class `share_alike` (20 languages, 8,747 rows); the specialist external NC rows are exactly the four amh sources below; `quarantine_nc/` is refused as an input; `rebuild/kept` and `meta` are refused; the windows carry the class and the guard checks it. What the class rule cannot see: rows with an unverified class whose upstream text is NC (e.g. AfroLingu-MT pairs of CC-BY-NC sets); `nc_filter.py report` lists rows whose licence TEXT looks NC but whose class is something else (to be read on the instance).

NC removal per language (measured from the build tables; efi/ibb tokens are estimates from characters, amh tokens are exact Gemma tokens):

| iso | NC pairs removed | target tokens removed | pairs before | share removed | sources |
|---|---:|---:|---:|---:|---|
| amh | 31,209 | 2,850,469 | 799,510 | 3.9% | masakhane/AfriDocMT (14,885); opus (12,063); AfriNLP/AfricanFineTranslations-sentences (4,259); aiana94/polynews-parallel (2) |
| ibb | 998 | 59,846 | 998 | 100.0% | howard-nlp/ibom-mt (main output) |
| efi | 994 | 55,863 | 324,959 | 0.31% | howard-nlp/ibom-mt (main output) |

(Total 33,201 pairs: 1,992 in the main set + 31,209 in the specialist set. ibb loses all its data; efi 0.31%; amh 3.9% of its pairs, 5.3% of its target tokens.)

## 2. Runbook and scripts for a fresh instance (`data_pipeline/sft_clean/fresh/`)
`README_RUNBOOK.md` (commands, storage plan with sizes per step, what to delete after the push, time estimates, risks), `env.example` (HF_TOKEN placeholder, nothing hardcoded), `sync_code.sh` (laptop: rsync of the chain code, the gemma_mt package and the v2 split; no data, no .env, no git), `bootstrap.sh` (venv_data: numpy 2, pyarrow, transformers 5; venv_comet: CUDA torch via `uv --torch-backend=auto`, unbabel-comet 2.2.7, numpy<2, pyarrow<20; optional venv_vllm 0.31.0 cu130), `run_chain.sh` (idempotent, `state/*.done`, `logs/*.log`, `logs/timing.tsv`, disk guard per step, `STOP_AFTER_FIT=1`, `FORCE=`, `STEPS=`), `disk_watch.sh`, `hub_get.py`, `push_packs.py`, `sanity_after_gate.py`. The run_chain skeleton was exercised on the laptop with synthetic data (skip/force/timing/NC report/sanity); the GPU and Hub steps were not run.
Order (resumable): preflight -> tokenizers -> CPU score shards -> pairs -> NC report -> **qe_crosscheck** (500 pairs, CPU fp32 vs GPU fp16, fails the chain below Pearson 0.999) -> H_cal references on the GPU (split v2, local text deleted after scoring) -> **qe_score** (GPU, `--skip-scored` with the CPU shards, `--exclude-nc`) -> **gate fit** (OLD and CAL side by side, nothing dropped; `STOP_AFTER_FIT=1` stops here) -> **apply CAL** -> sanity table -> windows -> NC guard -> pack -> schedules A0 (stage A), B0, S1/S2, G0, A2 -> private push. Agent T1's `sampler.py subset` was not in the repo when this was written: `run_chain.sh` uses it if the command exists and the fallback `arms_extra.py a2` otherwise.
Storage (estimates): peak about 37 GB (venvs 9.4 + pairs 3.2 + models 4.5 + scores 0.4 + windows 3.5 + pack 16.2), after the pack about 29 GB, final without venvs about 9 GB (packs 8.1 GB); request >= 50 GB free. Packed tokens: 2.03 B sample tokens x 4 B = 8.1 GB (commercial + CAL estimate; 8.7 GB ungated).

## 3. Projections without the NC rows, A0 / S1 / S2 / G0 / A2 (ESTIMATES: window statistics scaled by the removed token share and by the CAL keep rates of the partial CPU fit (about 155 pairs per language); the real numbers need windows.py + pack.py + the GPU gate)
| variant | windows | target tokens (Gemma) | packs | A0 steps | S1 / S2 steps | G0 steps | A2 steps | A0 draws | S1 draws | G0 draws | A2 draws |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ungated_with_NC(old projection) | 10,307,487 | 1,267,253,555 | 531,343 | 16,604 | 3,870 | 922 | 3,326 | 531,328 | 123,840 | 29,504 | 106,432 |
| commercial_ungated | 10,290,132 | 1,264,285,461 | 530,138 | 16,566 | 3,856 | 905 | 3,318 | 530,112 | 123,392 | 28,960 | 106,176 |
| commercial_CAL_ESTIMATE | 9,653,717 | 1,182,916,394 | 496,126 | 15,503 | 3,581 | 824 | 3,104 | 496,096 | 114,592 | 26,368 | 99,328 |
| commercial_OLD_ESTIMATE | 8,705,198 | 1,069,536,473 | 448,510 | 14,015 | 3,262 | 773 | 2,810 | 448,480 | 104,384 | 24,736 | 89,920 |

"commercial_CAL_ESTIMATE" is the expected production case: 9.65 M windows, 1.18 B Gemma target tokens, about 496,000 packs; A0 stage A = 15,503 steps of 32 packs (16,604 in the projection with NC rows and no gate); S1 and S2 (the same packs, S2 re-tokenised) 3,581 steps; G0 (amh, tir) 824 steps; the 20% stratified A2 subset 3,104 steps. The OLD rule would give 448,510 packs / 14,015 A0 steps (flat 85% keep). Per-bin totals of the CAL case (`commercial_CAL_ESTIMATE_per_bin_totals.csv`; the b64/b256/b512/b1024 split is scaled per language and bin):

| bin | windows | English tokens (AfriXLM-R) | target tokens (Gemma) | packs |
|---|---:|---:|---:|---:|
| b64 | 5,895,644 | 156,659,550 | 233,514,782 | 119,531 |
| b256 | 2,393,953 | 171,772,128 | 281,603,047 | 116,982 |
| b512 | 1,043,380 | 267,824,140 | 429,305,229 | 167,239 |
| b1024 | 320,776 | 153,738,967 | 238,493,266 | 92,339 |

Specialist languages in the CAL case (`commercial_CAL_ESTIMATE_arm_draws_per_language.csv`; effective epochs = A0 draws / packs, S1 inherits them):

| iso | packs | A0 draws | A0 effective epochs | S1 draws | G0 draws | A2 draws |
|---|---:|---:|---:|---:|---:|---:|
| amh | 17,647 | 13,688 | 0.776 | 13,682 | 13,683 | 2,723 |
| tir | 15,352 | 12,688 | 0.826 | 12,685 | 12,685 | 2,566 |
| yor | 31,343 | 18,251 | 0.582 | 18,248 | 0 | 3,615 |
| fon | 1,901 | 4,482 | 2.358 | 4,482 | 0 | 863 |
| dyu | 229 | 916 | 4.0 | 916 | 0 | 171 |
| kbp | 214 | 856 | 4.0 | 856 | 0 | 178 |
| nus | 252 | 1,008 | 4.0 | 1,008 | 0 | 200 |
| mos | 3,257 | 5,949 | 1.827 | 5,949 | 0 | 1,235 |
| kam | 1,500 | 3,932 | 2.621 | 3,932 | 0 | 748 |
| bam | 650 | 2,591 | 3.986 | 2,590 | 0 | 518 |
| umb | 3,056 | 5,707 | 1.867 | 5,706 | 0 | 1,189 |
| kea | 637 | 2,548 | 4.0 | 2,548 | 0 | 491 |
| kmb | 1,333 | 3,806 | 2.855 | 3,806 | 0 | 737 |
| vmw | 1,101 | 3,397 | 3.085 | 3,396 | 0 | 707 |
| kab | 3,734 | 6,149 | 1.647 | 6,146 | 0 | 1,216 |
| seh | 755 | 2,787 | 3.691 | 2,787 | 0 | 562 |
| wol | 2,291 | 4,752 | 2.074 | 4,751 | 0 | 895 |
| ewe | 684 | 2,624 | 3.836 | 2,624 | 0 | 549 |
| luo | 14,535 | 12,476 | 0.858 | 12,475 | 0 | 2,562 |
| tum | 3,500 | 6,008 | 1.717 | 6,005 | 0 | 1,151 |

amh loses 5.3% of its packs through the NC rule and a further 14% through the gate (CAL keep 0.86 for amh in the partial fit); efi 0.3%; ibb disappears.

## 4. Languages under 2,000 pairs after NC removal and the gate (ESTIMATE; the instance writes the real table to `work/gate/sanity_after_gate_under_2000.csv`)
| iso | pairs before | NC removed | after NC | after NC + CAL (est.) | after NC + OLD (est.) | status |
|---|---:|---:|---:|---:|---:|---|
| bbj | 0 | 0 | 0 | 0 | 0 | no data at all |
| dag | 0 | 0 | 0 | 0 | 0 | no data at all |
| fat | 0 | 0 | 0 | 0 | 0 | no data at all |
| ful | 0 | 0 | 0 | 0 | 0 | no data at all |
| gaa | 0 | 0 | 0 | 0 | 0 | no data at all |
| knc | 0 | 0 | 0 | 0 | 0 | no data at all |
| nde | 0 | 0 | 0 | 0 | 0 | no data at all |
| tzm | 0 | 0 | 0 | 0 | 0 | no data at all |
| nbl | 188 | 0 | 188 | 178 | 158 | below 2,000 |
| ibb | 998 | 998 | 0 | 0 | 0 | all rows were NC |
| men | 1,621 | 0 | 1,621 | 1,537 | 1,370 | below 2,000 |

nbl and men are source gaps (unchanged); ibb is empty because every ibb pair was IbomMT train; the eight languages without any data (bbj, dag, fat, gaa, knc, nde, tzm, ful) are not in the build either. **Ge'ez exposure after the rule:** amh keeps 768,301 of its 799,510 specialist-set pairs before the gate (about 660k at the CAL keep rate of the partial fit), tir is unchanged.

## 5. Risks (details in the runbook)
(1) unverified classes still contain NC-origin text (owner-accepted); (2) rule CAL thresholds come from about 155 pairs per language and 10 reference languages so far, final values need the GPU run and the H_cal references of every language with a FLORES+ dev file (the calibration agent's references on box 2 are lost with it; the chain recomputes them on the GPU); (3) the cross-check pairs are recovered by key from the Hub shards, the CPU shard q-000000 must be on the Hub; (4) two possible A2 subsets (T1's `subset` and the fallback) must not be mixed; (5) 16 GB extra disk during packing; (6) S2 replay needs the extended tokenizer; (7) the Hub `chain/` folder is stale (it predates the NC rule): the new instance pushes `code/` with `push_packs.py`.


# Addendum 5 (2026-10-30, laptop only, nothing run on the instance): IbomMT exception, hidden-NC filter, whole-GPU plan, A2 redesign

Owner updates of the evening of 2026-10-29 applied to the chain and to the fresh-instance runbook (`data_pipeline/sft_clean/fresh/README_RUNBOOK.md`, sections 6-9). No git operation, no write token used, nothing was started on the instance (SSH is still blocked). The Hub folder `chain/` is therefore stale; `sync_code.sh` copies the current code.

## 1. IbomMT owner exception
`--nc-allow howard-nlp/ibom-mt` (env `NC_ALLOW`, default that value; empty = no exception) is passed to `qe_score.py`, `gate.py fit/apply`, `windows.py` and `nc_filter.py report`. Rows of that source are scored, gated and windowed; every window carries the column `nc_exception=ibom-mt` and `nc_filter.py guard-windows` passes tagged windows and fails on UNTAGGED non-commercial windows (the exception can be undone by re-running step 50 without `NC_ALLOW`). `nc_filter.py report` writes `nc_kept_by_owner_exception.csv`. Effect: ibb is back with 998 pairs (about 947 after the CAL gate, still below 2,000), efi keeps its 994 (about 325 k pairs in total); the NC removal is now 31,209 amh pairs only (AfriDocMT 14,885; opus 12,063; AfricanFineTranslations 4,259; polynews 2). Tests: `chain/test_chain.py`, 11 passed / 2 skipped (no real tokenizer), including `test_allow_list_keeps_ibom_tagged`.

## 2. Hidden-NC filter (`chain/nc_hidden.py`, `chain/hidden_impact.py`)
Built as asked: it reports per language and source, BEFORE dropping, pairs and estimated target tokens; it never leaves a language below 2,000 pairs silently (`apply` refuses and names the languages, `--allow-below-min` / `HIDDEN_ALLOW_BELOW=1` is the explicit override); it drops by default only what is identifiable per row, and lists the unsplittable whole sources. Rules: R1 the row's own licence text carries an NC marker; R2 the pair is in a corpus whose licence is NC (index of hashes built by `build-index` from MAFAND-MT, all splits, GitHub `masakhane-io/lafand-mt`, and `fiifinketia/pristine-twi-english`); R3 whole sources with no per-row origin: only listed. Runs on the instance as steps 21-23 (CPU, beside the QE); `merge-keep` produces `keep_final.parquet` for the windows.

**What I could measure on the laptop (no pair data downloaded; the exact R1/R2 counts on the real pairs come from step 22):**
* AfroLingu-MT (`UBC-NLP/AfroLingu-MT`) is gated with the card text "I agree to use this model for non-commercial use ONLY"; it was built from 43 datasets and its rows carry only a language pair and an instruction, no source. Its train file (320,371 English-to-X rows read with a read token) compared with ALL splits of MAFAND-MT (141,390 pairs): only 1,315 exact pair matches (swh 1,046 = 10.5% of the swh rows, twi 178 = 1.8%, amh 71, yor 15, hau 5; kbp, bam and the other 25 languages 0) and 119 target-only matches. Scaled to the output, rule R2 would remove about 1,250 AfroLingu pairs (swh ~958, twi ~219, amh ~61, yor ~7, hau ~6; `hidden_nc/hidden_nc_R2_measured_afrolingu_vs_mafand.csv`), i.e. 0.6% of the source: MAFAND explains almost nothing of AfroLingu, so the source cannot be split per row and stays a whole-source decision.
* `KwekuAinoo/eng_twi` (561,215 rows with both sides, Bible/JW-style) vs the 999,497 pairs of `fiifinketia/pristine-twi-english` (NC upstream): 0 pair matches, 0 English-only, 0 Twi-only matches after NFKC/casefold/punctuation normalisation. No per-row evidence, so it is only listed.

**Impact table of the unsplittable whole sources (ESTIMATE from the build tables; language totals = unique pairs of data/, data_devhalf/ and specialist/; tokens = pairs x the language's mean Gemma target tokens per pair). Nothing is dropped.**
| whole source | pairs | est. Gemma target tokens | languages | share of the 23.4 M pairs | languages that would fall below 2,000 pairs only because of it |
|---|---:|---:|---:|---:|---|
| mt560 mixtures (`michsethowusu/english-*_mt560`) | 16,227,589 | 823 M | 66 | 69% | 34 languages: bin dua dyu efi her ish iso kik kmb kon kqn kri kua lin loz lua lun ndc oke pcm run sag seh shi sid swc tiv tsc umb urh ven vmw wes yao (for 41 of the 66 it is >= 99% of the language) |
| `African-Languages-Lab/multi-open` | 3,748,964 | 248 M | 18 | 16% | none (largest shares: ewe 92%, swh 90%, fon 63%, kin 54%, yor 53%, sot 50%) |
| `KwekuAinoo/eng_twi` | 551,358 | 19.6 M | 1 (twi) | 2.4% | none (twi would keep 112,177 pairs; 83% of twi) |
| `UBC-NLP/AfroLingu-MT` | 207,796 | 14.0 M | 32 | 0.9% | **kbp** (8,969 of its 9,970 pairs, 90%; 1,001 left) |

Per language and source (all 32 AfroLingu languages, the other three sources): `reports/data/sft_clean_2026-10-28/bins_commercial/hidden_nc/hidden_nc_whole_source_by_language_ESTIMATE.csv` (columns: pairs, tokens, share of the language, pairs left, flags), totals in `hidden_nc_whole_source_totals_ESTIMATE.csv`, "all four dropped" in `hidden_nc_all_whole_sources_dropped_ESTIMATE.csv` (44 of 73 languages below 2,000). Reading for the owner: dropping AfroLingu-MT alone costs 14.0 M tokens (1.2%) and one language (kbp); bam would keep 10,997 pairs (loses 42%), nyn 16%, bas 16%, ach 12%, wal 8%. Dropping mt560 (owner decision of Amendment 4: keep) would remove 69% of the pairs and 34 languages. Decision flag `HIDDEN_DROP_SOURCES="UBC-NLP/AfroLingu-MT"` implements the first if the owner wants it.

## 3. Using the whole GPU ($0.58/h)
* `chain/qe_probe.py` (step 25, 3-6 min): scores 3,000 DIFFERENT pairs per process with 1, 2 and 4 scoring processes and batch 128 and 256, polls `nvidia-smi` memory, aggregate rate = pairs / slowest process's scoring time (model load excluded; `qe_score.py` now prints `elapsed_s=` in its DONE line), a larger process count must be >= 5% faster to win, configurations beyond 80% VRAM are not tried; writes `work/qe_probe/qe_procs.json`, which `run_chain.sh` reads (`QE_PROCS=auto`, default). Tested with the stub scorer (`test_qe_probe_stub`); the real rates exist only on the GPU. `QE_PROCS=<n>` skips the probe.
* `run_chain.sh` v2: the bulk QE (32, 33) runs N processes with `nproc/N` loader workers each; the cross-check (30), the H_cal reference scoring (31), the CPU-only NC report and hidden-NC steps (20-23) run in background groups beside it, the chain waits for them before the gate; `logs/vram.log` samples GPU memory every 30 s.
* `fresh/gpu_jobs_runner.sh` (started by the chain): any `$ROOT/gpu_jobs.d/NAME.job` (header `# vram_gb: N`, optional `# after: <state>`) starts when N + 12 GB are free, runs once, logs to `logs/gpujob_NAME.log`; for the calibration GPU jobs (SSA-COMET, bigram table, NLLB pseudo-reference precompute), which therefore run next to the QE. The H_cal reference QE is built in.
* Two concurrent SFT jobs: `fresh/sft_concurrent.sh NAME=CONFIG[,k=v..] ...` (own out_dir and log per job, staggered start, VRAM check, optional MPS) and `fresh/sft_throughput_probe.sh [CONFIG] [STEPS]` + `sft_probe_summary.py`: runs the first STEPS (default 80) steps of A0 (`sampler.py head`) once with 1 job and once with 2 concurrent jobs, reads `step`/`time` from each `metrics.jsonl` (written by `train_sft.py` every `log_every`), skips the first 10 logged steps and prints combined steps/s and tokens/s; decision rule: two jobs if combined >= 1.15 x one job and the windows overlapped. The summary logic was tested on synthetic metrics only; nothing was timed on a GPU.

## 4. A2 redesign
`arms_extra.py` was deleted (no fallback mixing). `run_chain.sh` builds G0 with `sampler.py select --languages amh,tir --name G0` and ONE A2 file with `sampler.py subset --schedule A0 --frac 0.25 --stratify language --seed 1234 --name A2` (optional `A2_FLOOR`); A2-base, A2-up6, A2-up12 and Gemma-3-1B-it all point their `schedule_file` at that file (the sidecar's sha256 is the proof). The step fails clearly if a command is missing. `project_nc.py` uses the same calls (test `test_sampler_select_subset_commands`).

## 5. Updated projections (ESTIMATES, same method as Addendum 4; IbomMT back in, A2 at 25%; `bins_commercial/arms_summary.csv`)
| variant | windows | target tokens (Gemma) | packs | A0 steps | S1 / S2 steps | G0 steps | A2 steps (25%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ungated_with_NC (old) | 10,307,487 | 1,267,253,555 | 531,343 | 16,604 | 3,870 | 922 | 4,150 |
| commercial_ungated | 10,291,315 | 1,264,403,086 | 530,187 | 16,568 | 3,857 | 909 | 4,141 |
| **commercial_CAL_ESTIMATE** | 9,654,839 | 1,183,028,020 | 496,173 | 15,505 | 3,591 | 822 | 3,876 |
| commercial_OLD_ESTIMATE | 8,706,200 | 1,069,636,112 | 448,553 | 14,017 | 3,262 | 771 | 3,504 |
The CAL case is within 0.2% of Addendum 4 (IbomMT is 0.01% of the tokens). A2 steps rose from 3,104 (20%) to 3,876 (25%). The hidden-NC filter changes these numbers only by what rules R1/R2 remove (about 1,250 AfroLingu pairs by the MAFAND overlap; the real count comes from step 22); the whole-source scenarios of section 2 are not projected as arms.
Languages under 2,000 pairs after NC removal and the CAL gate (`languages_under_2000_pairs_ESTIMATE.csv`): nbl 188 (178 after CAL), ibb 998 (947), men 1,621 (1,537); no data: bbj dag fat gaa knc nde tzm ful. efi (994 IbomMT pairs) is no longer in the list (about 325 k pairs overall). Specialist-set check: after the owner exception amh is the only language that loses NC rows (31,209 pairs, 3.9% of its pairs; `nc_removed_per_language.csv`).

## 6. Risks and open points
1. The real R1/R2 counts of the hidden-NC filter and the real QE throughput are unmeasured: both come from the instance (steps 22 and 25). 2. The MAFAND index covers MAFAND-MT and pristine-twi only; other NC-origin corpora behind AfroLingu-MT (it lists 43) are not indexed. Add more with `build-index --hub-corpus 'repo|file_regex|col_en|col_tg|iso|label'` (`HIDDEN_CORPORA`). 3. mt560 (69% of the pairs) and multi-open stay in as the owner decided; if the owner later wants a whole source dropped, languages fall below 2,000 as tabulated, and the step stops instead of doing it silently. 4. The A2 floor (`--floor-draws`) is agent T1's choice; the default is none, which leaves the smallest languages with very few A2 draws (ibb 25, efi 1,893 in the CAL projection). 5. Several QE processes can be CPU-bound on a small instance; the probe shows it. 6. Two concurrent SFT jobs also double the data loader load and change each job's wall time but not its result (separate seeds, schedules, out_dirs). 7. `train_sft.py` and `rl/` were not touched.

Files changed or new (all under `data_pipeline/sft_clean/`): `chain/common.py`, `windows.py`, `qe_score.py`, `gate.py`, `nc_filter.py`, `nc_hidden.py` (new), `qe_probe.py` (new), `hidden_impact.py` (new), `project_nc.py`, `test_chain.py`, `arms_extra.py` (deleted); `fresh/run_chain.sh`, `gpu_jobs_runner.sh` (new), `sft_concurrent.sh` (new), `sft_throughput_probe.sh` (new), `sft_probe_summary.py` (new), `env.example`, `README_RUNBOOK.md`. Data: `reports/data/sft_clean_2026-10-28/bins_commercial/` (arms_summary.csv, per-variant csvs, `nc_owner_exception_kept.json`, `hidden_nc/`).


# Addendum 6 (2026-10-30, laptop only, nothing run on the instance): AfroLingu-MT kept, four new SFT languages (nde ful knc tzm)

Implements `reports/data/prereg_clean_2026-10-28/AMENDMENT_6.txt`. No git, no write token, nothing started on the instance. Tests: `chain/test_chain.py`, 18 passed with the real tokenizers (15 passed, 3 skipped without them).

## 1. AfroLingu-MT is kept everywhere
Nothing is dropped from it. Every window built from it carries `owner_exception=afrolingu-mt` (new column of `windows.py`, `common.owner_tag`; counted in `stats.json` `owner_exception_windows`; `nc_filter.py report` writes `owner_exception_rows.csv`). The hidden-NC filter keeps only its per-row rules R1 (own NC licence marker) and R2 (match to MAFAND-MT / pristine-twi); AfroLingu stays in the whole-source LIST only (its reason text now says "KEPT by owner decision"); `HIDDEN_DROP_SOURCES` stays empty. The measured MAFAND overlap (about 1,250 AfroLingu pairs by rule R2, Addendum 5) is still removed by R2 if step 22 finds it. kbp therefore keeps its 8,969 AfroLingu pairs.

## 2. The four new languages
* **Registry check:** `gemma_mt/languages.py` already contains nde, knc, tzm and ful as SFT-only rows (names North Ndebele, Kanuri, Tamazight, Fulah; ids nde 35, knc 55, tzm 83, ful 84 in the 85-language `ALL_ISOS`). **Nothing is inserted or shifted**, so packs (`iso_id.npy`), `token_iso`, schedules (`pack_lang.npy`), sampler, instruction/prompt templates, calibration files and RL tables keep their indexes; `format.py` and `prepare_sft.py` take the name from the registry and write the plain code `{iso} ` into the target (no tokenizer change). They are not `rl_eligible`, not in `_RL_*` lists: no RL pools, no reward calibration (the pooled prior would apply if they enter RL later). Only comments in `languages.py` ("nde fat knc gaa men luy tzm ful have no SFT training data") are stale; I did not edit that file. Tests: `test_new_languages_registered_and_ids_stable`.
* **`chain/extra_sources.py`** (run_chain step 15, before QE / NC steps / windows; QE of its pairs is step 34): downloads on the instance, standardises to the pair schema (`source_dataset` = `extra-<tag>:<detail>`, licence class, licence_raw, kind, source_family, key, aux), NFC / whitespace normalisation, the old builder's length-ratio-letter rules, placeholders dropped for translatewiki / Pontoon, exact dedupe by `key` in a fixed priority order, then the STRICT filter (protected-2026-10-28.3, both sides, R1/R2/R3; the 2.3 GB index is fetched from the Hub folder `protected_index/` and deleted afterwards). Output `hub/extra/` is a normal chain input directory (last in `$SRC`). Raw downloads are deleted as they are read, the 1.5 M-row Fulah file is streamed by row group with the similarity cut applied first (peak extra disk about 0.4 GB besides the index). Per-source stats and the removed-by-protected-set table go to `extra/stats/`. Failure of a source is fatal with its name (`EXTRA_ALLOW_MISSING` overrides).
* **S1/S2 train on 24 languages:** `common.SPECIALIST` = the 20 + `NEW_LANGS` appended; `sampler.py arms` picks it up (T1's file untouched); G0 (amh, tir) and A2 (25% stratified) are unchanged in definition.

## 3. Counts per language BEFORE the full filter (laptop run of `extra_sources.py --estimate`: all sources fetched, cleaned, deduped; the protected filter is an R1-only estimate = exact sentence match of either side against `protected_sentence_hashes.npz`; the real R2/R3 removal can only be larger; `reports/data/sft_clean_2026-10-28/bins_amend6/`)
| lang | source | raw rows | dropped by clean rules | duplicate | R1-estimated protected | kept |
|---|---|---:|---:|---:|---:|---:|
| nde | eBible nde + engwebp (verse aligned) | 31,095 | 4 | 197 | 23 | **30,871** |
| ful | OPUS wikimedia en-ff | 23,533 | 1,095 | 294 | 3 | 22,141 |
| ful | OPUS XLEnt en-ff | 7,132 | 3,491 | 19 | 1 | 3,621 |
| ful | stephanedonna/english_fulfulde (Adamawa Bible, 3 splits) | 30,802 | 256 | 135 | 0 | 30,411 |
| ful | NLLB-mined `english-fulah_sentence-pairs` (margin >= 1.06; 605,990 of 1,511,069 rows) | 605,990 | 31,438 | 2,982 | 1,320 | 570,250 |
| ful | **total** | 667,457 | 36,280 | 3,430 | 1,324 | **626,423** |
| knc | Gamayun kau-eng kit5k | 5,000 | 3 | 0 | 3 | 4,994 |
| knc | KanuriMT | 7,513 | 12 | 4,633 | 2,789 | 79 |
| knc | **total** | 12,513 | 15 | 4,633 | 2,792 | **5,073** |
| tzm | NLLB-Seed eng-tzm_Tfng | 6,193 | 1 | 1 | 2 | 6,189 |
| tzm | translatewiki en-tzm (placeholders dropped before counting) | 977 | 54 | 129 | 0 | 794 |
| tzm | Pontoon en-tzm (placeholders dropped before counting) | 226 | 2 | 24 | 0 | 200 |
| tzm | **total** | 7,396 | 57 | 154 | 2 | **7,183** |

Reading: nde 30.9 k (the full Bible, 197 repeated verses removed by the dedupe), ful 626 k of which 91% is the NLLB-mined set (clean human/permissive ful is only 26 k: wikimedia 22.1 k + XLEnt 3.6 k; plus 30.4 k Adamawa Bible), knc 5.1 k (KanuriMT is a 79-pair addition: 62% duplicates Gamayun, 37% overlaps a protected set, i.e. TICO-19), tzm 7.2 k. The mined-Fulah similarity cut is a choice: margin >= 1.05 keeps 1.51 M rows, >= 1.06 about 0.62 M, >= 1.08 about 0.15 M, >= 1.10 about 50 k (`FUL_MIN_SIM`). `nde_crosscheck.json`: only 25 of our 30,871 nde verses occur verbatim in `SalomonMetre13/nde_en`, so that set is a different (older) Ndebele Bible text, not a copy of eBible; it stays a cross-check only. Gemma target tokens (exact, windows.py --stats-only): ful 14.9 M, nde 1.45 M, tzm 0.78 M, knc 0.12 M.

## 4. Updated projections (ESTIMATES; the four languages enter with a flat keep rate of 0.85 because no QE exists for them; `bins_commercial/arms_summary.csv`)
| variant | windows | target tokens | packs | A0 steps | S1 / S2 steps | G0 steps | A2 steps (25%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| commercial CAL (Addendum 5, 77 languages) | 9,654,839 | 1,183,028,020 | 496,173 | 15,505 | 3,591 | 822 | 3,876 |
| **commercial CAL with nde ful knc tzm (81 languages)** | 9,888,455 | 1,197,677,832 | 503,024 | **15,719** | **3,914** | 819 | **3,929** |
| same, mined Fulah keeps only half at the gate | 9,797,984 | 1,192,469,330 | 500,534 | 15,641 | 3,854 | 821 | 3,910 |
| commercial OLD rule | 8,939,816 | 1,084,285,924 | 455,404 | 14,231 | 3,566 | 764 | 3,557 |
| commercial ungated | 10,566,160 | 1,281,638,162 | 538,244 | 16,820 | 4,219 | 896 | 4,204 |
A0 grows by 214 steps (+1.4%), S1/S2 by 323 steps (+9.0%, 20 -> 24 languages), A2 by 53 steps. In the CAL case (`commercial_CAL_ESTIMATE_arm_draws_per_language.csv`): nde 550 packs, 2,200 A0 draws (4.0 epochs, the cap), knc 43 packs / 173 draws, tzm 212 / 848, ful 6,046 packs / 7,912 draws (1.31 epochs); S1 repeats exactly A0's draws of the 24 languages; A2 (25%) gets nde 550, knc 43, tzm 212, ful 1,978 draws. QE adds about 0.67 M pairs of ful / nde / knc / tzm (roughly 10 minutes at the assumed 1,100 pairs/s). Languages still under 2,000 pairs: nbl 188, ibb 998 (947 after CAL), men 1,621; no data: bbj, dag, fat, gaa (fat and gaa are skipped by the owner); nde, ful, knc, tzm leave the "no data" list.

## 5. Disclosure tags (paper / model card, Amendment 6 item 5)
`owner_exception` values: `afrolingu-mt`, `nllb_mined` (ful), `nllb_seed` (tzm), `bible_derived` (nde; Adamawa Bible), `twb_gamayun_terms` (knc), `kanurimt_compilation` (knc); IbomMT keeps `nc_exception=ibom-mt`; mt560 stays unverified (list in `nc_report`); NLLB-200 as RL-only reward is outside this chain.

## 6. Risks and open points
1. The estimate is R1-only and ignores QE: the real strict filter (R2/R3 over 8-grams) and the CAL gate will remove more, mostly from the mined Fulah set (sensitivity above); the gate falls back to the pooled median offset for any of these languages without H_cal references (the registry maps a FLORES+ code only for knc, and FLORES+ knc / fuv dev and devtest are protected evaluation data; whether H_cal references exist for knc is decided by `hcal.py refs` on the instance).
2. ful is 91% NLLB-mined and 5% Adamawa Fulfulde Bible (a different dialect, KJV English): quality of `ful` will be mixed; the tag lets it be removed or down-weighted later. The mined set also dominates the A0 schedule (7.9 k of 503 k draws).
3. nde is Bible-only (31 k verse pairs, domain shift, repeated named entities); its 4.0 epochs are the cap, not a sign of a large set.
4. knc has 5.1 k pairs from one TWB source (Tatoeba sentences): below the 20-language specialist scale; Kanuri is Latin-script knc here and FLORES+ knc_Latn dev/devtest are protected (I did not check whether `data_devhalf/` holds any knc or ful rows; if it does they enter through that directory as for the other weak languages).
5. tzm is Tifinagh (NLLB-Seed, translatewiki) mixed with Latin-script Pontoon rows (200): one code, two scripts.
6. Source URLs: OPUS versions are pinned in `extra_sources.py` (wikimedia v20260327, XLEnt v1.2, translatewiki v2026-07-01); eBible files are fetched live (`nde_vpl.zip`, `engwebp_vpl.zip`, both 31,103 lines on 2026-10-30), so a later revision could change the nde alignment.
7. `languages.py` comments are stale (see section 2); packs, schedules and calibration are untouched by this addendum.


# Addendum 7 (2026-10-30, laptop only): QE cap per language and a faster scorer (instance = RTX 4500 Ada, 129 pairs/s measured)

Trigger: the probe on the real instance (RTX 4500 Ada 24 GB, 64 cores, 125 GB RAM, 39 GB disk) gave 129 pairs/s (1 process, batch 128, 18 GB VRAM; 2 processes 126/s), i.e. 52 h for all 24.4 M pairs. Nothing was run on the instance; everything below is laptop work (CPU tests with the real AfriCOMET checkpoint in a separate venv, fp32). 22 tests pass (`chain/test_chain.py` with the tokenizers; 18 pass + 4 skipped without torch / tokenizers).

## 1. `--cap-per-lang N` (monotone hash sample)
* Rule: a pair is scored when `(key & 0xFFFFFFFF) < ceil(N * 2^32 / count[language])`, with `count` = the language's unique scoreable pairs (NC rows other than owner exceptions are not scoreable), counted ONCE over all source directories by `qe_cap.py count` into `work/cap_counts.json` (run_chain step 16, never overwritten without `--force`). The threshold is monotone in N, so the pairs of a larger cap are a superset: raise `QE_CAP` later and `--skip-scored` scores only the new keys (tests: `test_cap_is_monotone_and_sized`, `test_qe_score_cap_and_resume_scores_only_new_keys`). Languages with <= N pairs are scored in full. Draw sizes are binomial (about N +- 2 sqrt(N)). The key is xxh64 of the pair, independent of language and source, so the sample is uniform within a language.
* `common.py`: `cap_thresholds`, `in_cap`, `cap_counts`; `qe_score.py --cap-per-lang --cap-counts`; `qe_cap.py count | table | check`.
* **Not scored = not kept.** `gate.py` already worked on the scored set (`collect` keeps only scored keys) and `windows.py --keep-file` drops every key that is not in the keep list; both are unchanged in behaviour. New reporting: `gate.py fit` writes `n_pairs_before_cap` next to `n_scored`; `gate.py apply` writes `gated_<rule>_summary.csv` with pairs_before_cap, scored_after_cap, kept_after_gate and the scored share; `fresh/sanity_after_gate.py --scores ...` writes pairs_before_cap / pairs_scored_after_cap / pairs_after_gate per language and the under-2,000 table (`test_gate_reports_before_cap_and_unscored_is_not_kept`). CPU-scored pairs from box 2 that lie outside the cap stay scored and therefore kept (a few thousand pairs).
* The specialist set and the extra sources (the four new languages) are capped the same way: step 16 counts `data data_devhalf specialist extra` together, steps 32/33/34 score each directory with `--skip-keys-of` the earlier ones.
* `run_chain.sh`: env `QE_CAP` (default 150000, 0 = no cap), step 16 (counts + table in `logs/qe_cap_table.txt`), `project_nc.py --cap N` for the projections.

## 2. Workload for caps 100k / 150k / 200k / 300k (laptop estimate from `queue_meta.json` + the four new languages: 24.41 M unique scoreable pairs in 81 languages; the lead's numbers 5.5 / 7.6 / 9.4 / 12.1 M are 2-3% lower because the four new languages are added here)
| cap | pairs to score | capped languages | h at 129/s | h at 210/s | h at 280/s |
|---|---:|---:|---:|---:|---:|
| 100,000 | 5,679,774 | 45 | 12.2 | 7.5 | 5.6 |
| 150,000 | 7,798,898 | 40 | 16.8 | 10.3 | 7.7 |
| 200,000 | 9,682,032 | 34 | 20.8 | 12.8 | 9.6 |
| 300,000 | 12,490,014 | 25 | 26.9 | 16.5 | 12.4 |
| none | 24,410,469 | 0 | 52.6 | 32.3 | 24.2 |
**What the cap does to the training set** (CAL ESTIMATE, flat 0.85 keep for languages without a fit; `bins_cap_<N>/arms_summary.csv`): cap 100k: 2.27 M windows, 283 M Gemma target tokens, 117,536 packs, A0 3,673 steps, S1/S2 1,033, G0 147, A2 (25%) 918. **Cap 150k: 3.14 M windows, 393 M tokens, 163,618 packs, A0 5,113, S1/S2 1,391, G0 215, A2 1,278.** Cap 200k: 3.91 M / 488 M / 203,068 / 6,345 / 1,691 / 280 / 1,586. Cap 300k: 5.07 M / 629 M / 261,813 / 8,181 / 2,120 / 398 / 2,045. Uncapped (Addendum 6): 9.89 M / 1.20 B / 503,024 / 15,719 / 3,914 / 819 / 3,929. So a cap of 150k keeps about a third of the tokens and the A0 run is a third as long; the 40 capped languages are the big ones (afr, luo, xho, zul, yor, nya, swh, sna, mlg, tsn, amh, tso, hau, ibo, twi, ful, ...), the thin and specialist languages are almost all untouched. Per-language tables (pairs before cap / after cap / after gate): `bins_cap_<N>/cap_language_table_CAL_ESTIMATE.csv`.

## 3. Scoring speed: what was found and what was implemented
AfriCOMET-QE-STL-1.1 is comet's `ReferencelessRegression` (XLM-R encoder, layer "mix", avg pooling, **two encoder passes**, MLP on [mt, src, mt*src, |mt-src|]); its encoder already runs SDPA attention (checked on the loaded model). Findings in comet 2.2.7 `predict`: length batching is ON, but it sorts by the length of the SOURCE only (`len(sample["src"])`), so the translation side of every batch of 128 is padded to its longest member; the translation is the longer side for African targets.
* **Implemented (`chain/qe_fast.py`, default in `qe_score.py`; `--no-fast` / `QE_FAST=0` to disable):** the sentence embedding of a text does not depend on its batch, so (a) source and translation are encoded in separate length-sorted passes, (b) batches are formed by a token budget (24,576 padded tokens, at most ~30% padding) instead of 128 sentences, (c) identical English sources are encoded once (unique per chunk plus a process-wide fp16 cache in CPU memory, 8 GB = 4 M sentences), (d) tokenisation once per chunk, batches padded with numpy, kernels queued without host sync. Same model, tokenizer, 510-token truncation, fp16 and estimator.
* **Equivalence (tested on the laptop with the real checkpoint, CPU fp32, 400 + 140 pairs): Pearson 1.0000000, max |diff| 1.3e-6**, also when the sources come from the cache; with a stand-in model the batcher matches per-pair scoring to 1e-5 (`test_fast_qe_matches_per_pair_scores_and_caches_sources`). On the GPU in fp16 only the rounding of padding differences can differ; it is verified there: run_chain step 24 runs `qe_fast.py check` (Pearson must be >= 0.9999, else `state/qe_fast_failed` and the chain uses `--no-fast`), and every `qe_score.py` run compares its first chunk (384 pairs) with `model.predict` and falls back by itself below 0.9999 (the check time is excluded from `elapsed_s`).
* **Expected gain (computed, not measured on a GPU):** padded tokens per real token on real pair samples (Fulah mined 100k, nde, tzm): comet predict 1.87 / 1.43 / 1.45 (translation side 2.23 / 1.57 / 1.47), fast scorer 1.01 / 1.02 / 1.11, i.e. 1.85 / 1.40 / 1.31 times fewer tokens through the encoder. If the GPU is compute-bound (it is at 18 GB VRAM and no gain from 2 processes) that is about **1.4-1.85 times, my central expectation 210-230 pairs/s** instead of 129. The English cache adds on top where English sources repeat across languages (0% inside one language of the new sources; mt560 / JW-style data should repeat across the 66 languages but with the cap each language contributes only a sample; the real hit rate is printed by every run). VRAM falls from 18 GB to a few GB, so the probe may now find a second process worthwhile.
* **Looked at and not done:** truncation to 256 tokens (0.14% of the Fulah pairs exceed it, 0.12% of the tokens saved; it would change those scores); bf16 (same tensor-core throughput as fp16, less precision); torch.compile (variable shapes, recompiles); a lower-level SDPA switch (already sdpa).

## 4. Bug found and fixed on the way
`qe_score.py` named chunk files `<shard>-<chunk>.parquet` and skipped writing when the file existed. After a restart with `--skip-scored` the chunk numbers begin at 0 again, so the first chunks of the new run collided with the files of the earlier run and were silently not written (those pairs would have stayed unscored). Chunk files now carry a per-process tag. Anyone resuming a GPU run with the old script would have lost chunks.

## 5. Risks
1. The cap removes about two thirds of the tokens of the big languages (see section 2); the QE gate then only applies to the capped sample. If more data is wanted later, raise `QE_CAP` (superset) rather than changing anything else. An option not implemented (it contradicts "not scored = not kept"): keep unscored pairs of high-keep languages ungated.
2. The speed-up is an expectation from padding statistics; the real rate must be read from step 24 and the first minutes of step 32 (`logs/qe_*.log`, the line "fast scoring: ..." at the end).
3. English-source cache memory: 8 GB per process by default (`--fast-cache-gb`); several `--part` processes do not share it.
4. The projection uses the old CAL keep rates (partial fit) and a flat 0.85 for the four new languages; the real keep rates need the scores.
