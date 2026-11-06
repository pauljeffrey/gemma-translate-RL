# Post-editor for the 385M Gemma English -> African translation model: plan and design (2026-11-03)

Idea (owner, decided): train a POST-EDITOR. Input = English source + a machine-translation DRAFT; target = the HUMAN reference. A small SFT of the 385M weights (new prompt, same tokenizer, single-token language tags) so the model learns to correct drafts. Test: draft translations of the benchmarks from our RL model go through the editor; chrF++ / AfriCOMET change per language and macro. If it helps, the owner tries RL for the editor (section 9; not built).

Status: the pipeline is built, CPU-tested (`gemma_mt/tests/test_post_edit.py`, 22 tests incl. an end-to-end smoke with a tiny random model) and the proof experiment was RUN on a dedicated 3090 Ti: results, timings and cost are in `reports/post_editor_run_2026-11-03.md`. **Result in one line: editor E1 (human targets, our-model drafts) applied to every sentence is flat on FLORES+ and negative (-0.4 to -0.9 chrF++, about -0.01 AfriCOMET) on NTREX / MAFAND / SmolSent, below beam-4 and MBR-16; it only repairs failed drafts (repair-only gate +0.2 to +0.7 chrF++, ~0 AfriCOMET).** Only E1 was trained (owner hold on further variants; E2 / E2q / E3 are built and CPU-tested). This file is the design record (sections 1-2 describe the final selection v4 and the full draft design; E1 used selection v2 and our models' drafts only). Related: `reports/experiments_after_stage_A_2026-11-03.md` (arms after RL stage A).

Code (all new, nothing existing was changed): `gemma_mt/gemma_mt/post_edit.py` (format, corruptions, MBR, backends), `gemma_mt/scripts/post_editor_select_data.py`, `post_editor_make_drafts.py`, `post_editor_mt_drafts.py` (NLLB / MADLAD drafts), `post_editor_bench_drafts.py`, `post_editor_build_sft.py`, `post_editor_eval.py`, `post_editor_comet.py`, config `gemma_mt/configs/post_editor/pe_sft.yaml`. The SFT uses the EXISTING trainer `gemma_mt.train_sft` on a packed dataset in the existing format (no new trainer).

## 1. Training data: human references only

### 1.1 What counts as human (policy table, `SOURCE_RULES` in the selection script; every unknown source is excluded)

Class `gen` = human translation of known, non-religious provenance; `rel` = human religious text (JW300 / Bible / Quran: all of MT560, Helsinki fon, luo-opus, Bible sets, Tanzil); `unv` = human but provenance unstated or aggregator (multi-open, OPUS-100, crowd sets of unknown origin; counts as non-religious text but only fills what `gen` cannot); `X` = never.

NEVER (class X, with the reason): NLLB-mined bitext (`nllb*`, 15.6M rows in `extra2k/nllb`), the Aya aligned part (quarantined as NLLB-derived), HPLT DocHPLT and ParaCrawl (crawl-mined), XLEnt (mined), FineTranslations (model-written English side), the Gemma-4 code-switch set (machine translation), AfroLingu-MT (contains MAFAND train pairs: owner rule), AfricanFineTranslations / polynews (translation provenance unclear), Igbo sets whose English does not translate the Igbo or comes from a machine translator, Nupe (Quran, licence unknown), and every benchmark dev / devtest / test text. TranslatePsy data: none present in the clean pair shards (the clean rebuild excluded it by card scan; nothing was added). MAFAND train: not used. FLORES+ DEV TRAIN HALF (440 ids, 20 specialist languages): allowed by the owner (2026-11-03) because it is in the SFT on purpose; devtest and the held-out dev half are not (data_devhalf holds the train half only; the policy rule matches `flores_plus:dev:` / `flores-plus-dev-half` and the unit test checks that `...:devtest:...` is excluded).

Sources of the selection (bucket `<S3_BUCKET>/clean_rerun_2026-11-01/backup/hub`: `data` main pairs, `data_devhalf`, `specialist`, `extra`, `extra2k`, `extra2_hold/mt560_new` = the OPUS-JW pairs of the new consensus languages) + AfriScience-MT train (34,007 pairs, professional translators, Apache-2.0; clean copy from `<HF_ORG_LEGACY>/sabiyarn-clean-rl-prompts-2026-10-28`). Not found as separate pair files: the Efik NT sheet (it exists only as packed windows `a40_backup/work/windows_efi`; Efik is covered by 323k MT560 pairs anyway), Tumbuka / Aya "human parts" (Tumbuka is MT560 + instruction sets, no pair file; the Aya aligned part is in the NLLB quarantine).

Eligible pairs per source family (after the length / ratio / English-looks-English / URL / copy filters; gross, before cross-source dedupe; mean English words show the register):

| class | register | source family | NC | eligible pairs | languages | mean English words |
|---|---|---|---|---:|---:|---:|
| rel | religious | mt560 |  | 17,282,815 | 66 | 19.2 |
| unv | unverified | multi-open |  | 4,085,368 | 18 | 17.8 |
| rel | religious | luo-opus |  | 2,400,492 | 1 | 11.5 |
| rel | religious | opus-mt560 |  | 2,379,363 | 31 | 19.1 |
| gen | gov_news | mehgoss-autshumato |  | 873,702 | 6 | 17.7 |
| unv | unverified | kweku-eng-twi |  | 509,198 | 1 | 17.8 |
| rel | religious | igbo-bundles |  | 282,540 | 1 | 20.5 |
| unv | unverified | opus100 |  | 189,694 | 6 | 17.5 |
| gen | salt_text | salt-train |  | 158,578 | 7 | 9.3 |
| unv | unverified | swh-rogendo |  | 152,106 | 1 | 9.8 |
| gen | gov_news | autshumato-autshumato-en-ts |  | 141,607 | 1 | 11.3 |
| gen | gov_news | kin-gazette |  | 138,857 | 1 | 20.2 |
| unv | unverified | igbo-bundles |  | 132,864 | 1 | 18.1 |
| unv | unverified | wolof-sudoping |  | 107,458 | 1 | 9.2 |
| unv | unverified | okwu |  | 83,688 | 3 | 17.6 |
| gen | crowd_short | tatoeba-abdelhaque |  | 66,923 | 2 | 6.8 |
| gen | gov_news | opus |  | 51,185 | 1 | 17.4 |
| gen | gov_news | autshumato-autshumato-en-tn |  | 44,308 | 1 | 19.0 |
| unv | unverified | lug-pkyoyetera |  | 40,228 | 1 | 19.6 |
| unv | unverified | xhosa-silvano |  | 39,585 | 1 | 17.0 |
| gen | gov_info | infopankki |  | 34,493 | 1 | 14.4 |
| gen | crowd_short | opus_tatoeba |  | 33,936 | 10 | 7.1 |
| sci | science | afriscience |  | 33,228 | 6 | 20.7 |
| gen | ui_localisation | weblate |  | 31,255 | 41 | 10.9 |
| rel | religious | ebible-nde |  | 29,765 | 1 | 23.6 |
| gen | ui_localisation | kabyle-tm |  | 29,659 | 1 | 10.9 |
| rel | religious | fub-bible |  | 28,891 | 1 | 24.4 |
| rel | religious | fon-helsinki |  | 24,062 | 1 | 16.2 |
| rel | religious | wolof-bible |  | 22,644 | 1 | 22.4 |
| gen | ui_localisation | pontoon |  | 22,039 | 17 | 11.6 |
| gen | gov_news | autshumato-autshumato-en-zu |  | 20,476 | 1 | 18.0 |
| gen | gov_news | vukuzenzele |  | 20,299 | 10 | 21.5 |
| gen | wiki | opus-wikimedia |  | 19,270 | 1 | 21.0 |
| gen | flores_dev_train | flores-plus-dev-half |  | 17,454 | 20 | 20.9 |
| gen | agri_health | chichewa-mt |  | 12,414 | 1 | 24.8 |
| gen | health_tech | masakhane/AfriDocMT | yes | 12,058 | 1 | 20.2 |
| gen | ui_localisation | opus_translatewiki |  | 10,542 | 13 | 10.5 |
| gen | news | ssa-mte |  | 9,791 | 13 | 20.4 |
| rel | religious | opus | yes | 9,524 | 1 | 24.5 |
| unv | unverified | luhya |  | 6,546 | 1 | 12.6 |
| gen | wiki | nllb-seed-tzm |  | 6,037 | 1 | 21.6 |
| gen | wiki | oldi_seed_bam_Latn |  | 6,021 | 1 | 21.6 |
| gen | wiki | oldi_seed_nus_Latn |  | 6,020 | 1 | 21.6 |
| gen | ui_localisation | gamayun-kau-eng |  | 4,577 | 1 | 10.4 |
| gen | gov_news | mehgoss-vukuzenzele |  | 4,377 | 10 | 21.5 |
| gen | wiki | opus_wikimedia |  | 3,661 | 7 | 23.1 |
| gen | ui_localisation | gamayun_kits_tig-eng |  | 3,541 | 1 | 10.5 |
| rel | religious | mehgoss-jworg |  | 3,513 | 1 | 23.4 |
| rel | religious | efik-nt-sheet |  | 3,482 | 1 | 23.3 |
| gen | gov_news | govza |  | 2,412 | 1 | 26.3 |
| gen | gov_news | autshumato-autshumato-en-ts-manual |  | 2,171 | 1 | 8.2 |
| gen | religious_mixed | ibom-mt | yes | 1,950 | 2 | 20.9 |
| gen | talks | opus | yes | 869 | 1 | 14.9 |
| gen | ui_localisation | travis_foundation_tigrinya |  | 618 | 1 | 16.6 |
| unv | unverified | kanurimt |  | 79 | 1 | 16.6 |
| gen | ui_localisation | opus-translatewiki |  | 25 | 1 | 6.5 |
| gen | crowd_short | opus |  | 13 | 2 | 11.7 |
| gen | wiki | opus |  | 12 | 1 | 17.2 |
| gen | ui_localisation | pontoon-tzm |  | 12 | 1 | 5.6 |
| gen | wiki | openlanguagedata/oldi_seed |  | 10 | 2 | 18.4 |
| unv | unverified | twi-kweku |  | 6 | 1 | 20.7 |

Licences: MT560 (cards say cc-by-4.0; underlying JW300 / Bible text has NO open licence, disclosed), Pontoon MPL, Weblate per-string licences (share-alike rows kept and tagged), SSA-MTE train CC-BY-SA, AfriScience Apache-2.0, FLORES+ dev CC-BY-SA, Vukuzenzele / Autshumato CC-BY, OLDI-seed / NLLB-Seed CC-BY-SA (human, professionally translated; the NAME contains NLLB, disclosed), unverified (`unspecified` / `other`): multi-open, kweku, OPUS-100, okwu, swh_rogendo, xhosa_silvano, lug_pky, wolof-sudoping. **Non-commercial rows (explicit NC): IbomMT train (efi, ibb), AfriDocMT sentence level, TED2020, Tanzil** = `988` pairs (1.7% of the set); every pair row carries `non_commercial` and the training rows carry `target_licence`; `--no-nc` drops them.

### 1.2 Register: how skewed is it against FLORES / news?

The human text we have is mostly religious: of 17.3M eligible MT560 pairs + 2.4M luo-opus + 2.4M `opus-mt560`, the lower bound of keyword-detectable JW / Bible text is 38% (D2 audit) and the OPUS-tokenised spacing ("word , word .") shows in 95% of the English sides: JW300 and Bible-uedin. FLORES+ / NTREX / MAFAND are news / Wikipedia / Wikinews / Wikibooks; Vukuzenzele / Autshumato / gazette / GoURMET / AfriScience are the closest non-religious sources; Pontoon / Weblate / translatewiki / Tatoeba are 5-8 word UI or crowd strings (kept <= 25% of a language's rows, English >= 5 words). The owner's rule (2026-11-03): **religious text only where nothing else exists**. Implemented as: a language with >= 100 non-religious human pairs takes religious pairs ONLY to complete its target and at most 25% of its final total (a language with enough non-religious text gets none); a language with fewer is `religious-only` / `religious-mostly`: all its non-religious pairs + religious pairs up to a small cap (`rel_only_cap`, lowered automatically until the overall religious share is <= 30%: it ended at 255). OPUS spacing is undone (`detok`) in religious / gov / crowd rows so the editor does not learn to write tokenised text. Result: overall religious share **26.5%** of 57,462 pairs; 28 of the 81 RL languages are religious-only (no non-religious pair at all) and 5 more are religious-mostly (> 50%): they are flagged in the table and the evaluation reports them as a separate stratum (section 5).

### 1.3 Coverage of the languages (owner: all 77, ideally the 81 current RL languages)

**All 81 RL languages are covered with human references** (81 of 81 with at least 250 pairs; none missing). The original 77 are a subset of the 81; of the 85 languages in `SUPPORTED_RL`, `arz` (Egyptian Arabic, 216 religious pairs; its FLORES+ reference exists), `ibb` (Ibibio, 633 pairs from IbomMT train, non-commercial) are covered, `dag` and `bbj` have no human pair in any clean source (the catalogue audit found none; dag / bbj have no commercial-clear pairs at all). Selection v4 (the one the plan's tables show; E1 was trained on the earlier v2): 57,462 pairs (54,990 train, 2,472 validation) in 116 languages, religious share 26.5% overall (cap 30%). **28 of the 81 RL languages are religious-only** (no non-religious pair exists in any clean source; ~255 religious pairs each, the cap was lowered automatically until the overall share was under 30%): bas, bin, crs, dua, guw, her, ish, iso, kon, kqn, kri, kua, loz, lua, lub, lue, lun, ndc, oke, sid, swc, tiv, tll, tsc, urh, wes, yao, zne; **5 more are religious-mostly (> 50%)**: bem, mfe, run, sag, wal. Languages of the 81 with NO benchmark reference at all (judged only by AfriCOMET-QE / SSA-COMET or on the held-out slice of the editor's own human data): urh, bin, oke, swc, iso, wal, yao, sid, ish, dua, bas, wes, guw, lue, tll, zne, lub, crs (18).

What changed between versions of the selection (RL languages): v1 (first cut, caps 50%, no religious limit) 55% religious overall, 35 of 81 religious-dominated; **v2** (owner rule: religious <= 25% where non-religious text exists, floor 250, overall <= 30%): 26.0% religious, 29 religious-only; **v3** (+ SALT train: ach, nyn, ttj, teo, lug, ibo, swh; + Infopankki som; + Chichewa nya; + Efik NT sheet as one more efi source): 28.7% religious, 28 religious-only (ach and nyn leave the religious-only list; the Efik NT sheet is not drawn: efi has 631 non-religious pairs, the religious pool is only used to complete a language); **v4** (+ AfriScience-MT train: cap 3,000 for amh hau lug nso yor zul with science at most 40% = 1,200 per language, about 130 shared English sentences per language held out as unseen in-domain validation; `domain=science` in pairs.jsonl): 26.5% religious, 28 religious-only, 0 thin languages. The sources added in v3 / v4 were searched for (SALT train, Infopankki, Chichewa, AfriScience, Efik NT); opus_gnome / ubuntu / kde4 / wikipedia have no African-English pair on the Hub, KDE4 is a loading script, MALIBA bambara-mt contains google/smol rows (benchmark text), the others of the catalogue audit are mined, machine-written or licence-restricted (see `post_editor_extra_sources.py`). Of the AfriScience languages, v2 gave 639 pairs each and v4 gives 2,970-2,999 (1,200 science each, validation 112-142 science pairs each; zul 2,999 / amh 2,970 / hau 2,972 / lug 2,983 / nso 2,999 / yor 2,973).

Targets are never QE-selected machine text (`target_type` human for every row); QE-high machine targets were not built (no language needs them). A moderate AfriCOMET-QE verification of the human targets is implemented (`post_editor_qe_filter.py`: the lowest 27% of the TRAIN pairs of each language with a validated QE calibration, never below 250 pairs, thin and uncalibrated languages untouched; scored on the v4 selection: 39 languages filtered, 8,826 of 33,479 train pairs flagged) but the filtered variant was not trained (owner hold).

Per-language table (chosen pairs; `(val)` = held-out validation pairs; "benchmark references" = which benchmarks have a human reference for the language, i.e. where the editor can be judged by chrF++; languages with none can only be judged with AfriCOMET-QE or on the held-out slice of the editor's own human data). Full CSV: `reports/data/post_editor_2026-11-03/selection/coverage_by_language.csv` (also the eligible counts per source, the excluded sources per language and the reason when a language is thin).

### 81 RL languages
| iso | language | benchmark references | chosen pairs (val) | human-general / religious / unverified | eligible pairs (gross, before dedupe) | register caveat |
|---|---|---|---:|---|---:|---|
| ach | Acholi | smol_sent, salt | 498 (20) | 498 / 0 / 0 | 90,422 |  |
| afr | Afrikaans | flores_plus, ntrex, smol_sent | 498 (20) | 498 / 0 / 0 | 1,852,030 |  |
| amh | Amharic | flores_plus, ntrex, mafand, smol_sent | 2970 (147) | 1770 / 0 / 0 | 1,248,413 | FLORES dev-train; 1200 AfriScience |
| bam | Bambara | flores_plus, smol_sent | 499 (20) | 499 / 0 / 0 | 6,900 | FLORES dev-train |
| bas | Basaa | none | 255 (10) | 0 / 255 / 0 | 25,657 | RELIGIOUS-ONLY (255 rows) |
| bem | Bemba | flores_plus, ntrex, smol_sent | 255 (10) | 1 / 254 / 0 | 349,911 | relig. 100% |
| bin | Edo | none | 255 (10) | 0 / 255 / 0 | 10,731 | RELIGIOUS-ONLY (255 rows) |
| crs | Seychellois Creole | none | 255 (10) | 0 / 255 / 0 | 173,632 | RELIGIOUS-ONLY (255 rows) |
| dua | Duala | none | 255 (10) | 0 / 255 / 0 | 14,196 | RELIGIOUS-ONLY (255 rows) |
| dyu | Dyula | flores_plus, smol_sent | 498 (20) | 435 / 63 / 0 | 28,052 | relig. 13%; FLORES dev-train |
| efi | Efik | smol_sent, ibom_mt | 493 (20) | 493 / 0 / 0 | 309,128 | 493 NC |
| ewe | Ewe | flores_plus, ntrex, smol_sent | 499 (20) | 499 / 0 / 0 | 54,268 | FLORES dev-train |
| fon | Fon | flores_plus, smol_sent | 497 (20) | 497 / 0 / 0 | 225,518 | FLORES dev-train |
| guw | Gun | none | 254 (10) | 0 / 254 / 0 | 263,552 | RELIGIOUS-ONLY (254 rows) |
| hau | Hausa | flores_plus, ntrex, mafand, smol_sent | 2972 (132) | 1772 / 0 / 0 | 543,772 | 1200 AfriScience |
| her | Herero | glot_story | 255 (10) | 0 / 255 / 0 | 7,529 | RELIGIOUS-ONLY (255 rows) |
| ibo | Igbo | flores_plus, ntrex, mafand, smol_sent, salt | 495 (20) | 495 / 0 / 0 | 978,457 |  |
| ish | Esan | none | 255 (10) | 0 / 255 / 0 | 4,872 | RELIGIOUS-ONLY (255 rows) |
| iso | Isoko | none | 255 (10) | 0 / 255 / 0 | 197,104 | RELIGIOUS-ONLY (255 rows) |
| kab | Kabyle | flores_plus, smol_sent | 499 (20) | 499 / 0 / 0 | 190,063 | FLORES dev-train |
| kam | Kamba | flores_plus | 495 (20) | 439 / 56 / 0 | 95,444 | relig. 11%; FLORES dev-train |
| kbp | Kabiye | flores_plus | 492 (20) | 492 / 0 / 0 | 1,004 | FLORES dev-train |
| kea | Kabuverdianu | flores_plus | 498 (20) | 498 / 0 / 0 | 12,726 | FLORES dev-train |
| kik | Kikuyu | flores_plus, smol_sent | 498 (20) | 498 / 0 / 0 | 87,716 |  |
| kin | Kinyarwanda | flores_plus, ntrex, mafand, smol_sent | 499 (20) | 499 / 0 / 0 | 236,140 |  |
| kmb | Kimbundu | flores_plus | 496 (20) | 436 / 60 / 0 | 168,366 | relig. 12%; FLORES dev-train |
| kon | Kongo | smol_sent | 255 (10) | 0 / 255 / 0 | 345,737 | RELIGIOUS-ONLY (255 rows) |
| kqn | Kaonde | glot_story | 255 (10) | 0 / 255 / 0 | 201,343 | RELIGIOUS-ONLY (255 rows) |
| kri | Krio | smol_sent | 255 (10) | 0 / 255 / 0 | 39,471 | RELIGIOUS-ONLY (255 rows) |
| kua | Kuanyama | glot_story | 255 (10) | 0 / 255 / 0 | 92,494 | RELIGIOUS-ONLY (255 rows) |
| lin | Lingala | flores_plus, ntrex, smol_sent | 498 (20) | 477 / 21 / 0 | 491,965 | relig. 4% |
| loz | Lozi | glot_story | 255 (10) | 0 / 255 / 0 | 377,670 | RELIGIOUS-ONLY (255 rows) |
| lua | Luba-Kasai | flores_plus | 255 (10) | 0 / 255 / 0 | 268,494 | RELIGIOUS-ONLY (255 rows) |
| lub | Luba-Katanga | none | 255 (10) | 0 / 255 / 0 | 180,756 | RELIGIOUS-ONLY (255 rows) |
| lue | Luvale | none | 254 (10) | 0 / 254 / 0 | 290,525 | RELIGIOUS-ONLY (254 rows) |
| lug | Luganda | flores_plus, ntrex, mafand, smol_sent, salt | 2983 (152) | 1785 / 0 / 0 | 269,403 | 1198 AfriScience |
| lun | Lunda | glot_story | 255 (10) | 0 / 255 / 0 | 123,443 | RELIGIOUS-ONLY (255 rows) |
| luo | Luo | flores_plus, mafand, smol_sent | 497 (20) | 497 / 0 / 0 | 2,653,700 | FLORES dev-train |
| mfe | Mauritian Creole | flores_plus | 255 (10) | 20 / 235 / 0 | 166,566 | relig. 92% |
| mlg | Malagasy | flores_plus, ntrex, smol_sent | 472 (19) | 353 / 119 / 0 | 790,985 | relig. 25% |
| mos | Mossi | flores_plus, smol_sent | 499 (20) | 499 / 0 / 0 | 342,122 | FLORES dev-train |
| nbl | South Ndebele | smol_sent, glot_story | 499 (20) | 499 / 0 / 0 | 108,370 |  |
| ndc | Ndau | smol_sent | 254 (10) | 0 / 254 / 0 | 28,315 | RELIGIOUS-ONLY (254 rows) |
| nso | Sepedi | flores_plus, ntrex, smol_sent | 2999 (160) | 1799 / 0 / 0 | 626,856 | 1200 AfriScience |
| nus | Nuer | flores_plus | 499 (20) | 499 / 0 / 0 | 6,960 | FLORES dev-train |
| nya | Chichewa | flores_plus, ntrex, mafand, smol_sent | 499 (20) | 499 / 0 / 0 | 800,322 |  |
| nyn | Nyankore | smol_sent, salt | 498 (20) | 498 / 0 / 0 | 69,905 |  |
| oke | Okpe | none | 255 (10) | 0 / 255 / 0 | 20,943 | RELIGIOUS-ONLY (255 rows) |
| orm | Oromo | flores_plus, ntrex, smol_sent | 499 (20) | 174 / 0 / 325 | 174,479 | 325 unverified-provenance pairs |
| pcm | Pidgin | mafand, smol_sent | 499 (20) | 0 / 0 / 499 | 49,100 | 499 unverified-provenance pairs |
| run | Kirundi | flores_plus, smol_sent | 255 (10) | 35 / 220 / 0 | 333,682 | relig. 86% |
| sag | Sango | flores_plus | 255 (10) | 35 / 220 / 0 | 228,892 | relig. 86% |
| seh | Sena | flores_plus, flores_moz | 494 (20) | 434 / 60 / 0 | 98,768 | relig. 12%; FLORES dev-train |
| shi | Tachelhit | ntrex | 499 (20) | 499 / 0 / 0 | 24,388 |  |
| sid | Sidamo | none | 255 (10) | 0 / 255 / 0 | 43,098 | RELIGIOUS-ONLY (255 rows) |
| sna | Shona | flores_plus, ntrex, mafand, smol_sent | 498 (20) | 244 / 0 / 254 | 786,756 | 254 unverified-provenance pairs |
| som | Somali | flores_plus, ntrex, smol_sent | 499 (20) | 499 / 0 / 0 | 271,048 |  |
| sot | Sesotho | flores_plus, ntrex, smol_sent | 499 (20) | 499 / 0 / 0 | 83,069 |  |
| ssw | Swati | flores_plus, ntrex, smol_sent | 499 (20) | 499 / 0 / 0 | 110,717 |  |
| swc | Congo Swahili | none | 255 (10) | 0 / 255 / 0 | 248,127 | RELIGIOUS-ONLY (255 rows) |
| swh | Swahili | flores_plus, ntrex, mafand, smol_sent, salt | 498 (20) | 498 / 0 / 0 | 970,114 |  |
| tir | Tigrinya | flores_plus, ntrex, smol_sent | 497 (20) | 497 / 0 / 0 | 578,110 | FLORES dev-train |
| tiv | Tiv | smol_sent | 255 (10) | 0 / 255 / 0 | 168,188 | RELIGIOUS-ONLY (255 rows) |
| tll | Tetela | none | 255 (10) | 0 / 255 / 0 | 203,881 | RELIGIOUS-ONLY (255 rows) |
| tsc | Tswa | glot_story | 253 (10) | 0 / 253 / 0 | 78,132 | RELIGIOUS-ONLY (253 rows) |
| tsn | Setswana | flores_plus, ntrex, mafand, smol_sent | 499 (20) | 499 / 0 / 0 | 1,004,939 |  |
| tso | Tsonga | flores_plus, smol_sent | 499 (20) | 499 / 0 / 0 | 987,322 |  |
| tum | Tumbuka | flores_plus, smol_sent | 496 (20) | 496 / 0 / 0 | 429,170 | FLORES dev-train |
| twi | Twi | flores_plus, mafand, smol_sent | 499 (20) | 499 / 0 / 0 | 603,540 |  |
| umb | Umbundu | flores_plus | 496 (20) | 437 / 59 / 0 | 389,397 | relig. 12%; FLORES dev-train |
| urh | Urhobo | none | 254 (10) | 0 / 254 / 0 | 27,224 | RELIGIOUS-ONLY (254 rows) |
| ven | Venda | ntrex, smol_sent | 428 (17) | 321 / 107 / 0 | 187,681 | relig. 25% |
| vmw | Makhuwa | flores_plus | 493 (20) | 493 / 0 / 0 | 136,195 | FLORES dev-train |
| wal | Wolaytta | none | 255 (10) | 2 / 253 / 0 | 110,990 | relig. 99% |
| wes | Cameroon Pidgin | none | 255 (10) | 0 / 255 / 0 | 26,492 | RELIGIOUS-ONLY (255 rows) |
| wol | Wolof | flores_plus, ntrex, smol_sent | 499 (20) | 499 / 0 / 0 | 262,473 | FLORES dev-train |
| xho | Xhosa | flores_plus, ntrex, mafand, smol_sent | 499 (20) | 499 / 0 / 0 | 1,305,975 |  |
| yao | Yao | none | 255 (10) | 0 / 255 / 0 | 40,392 | RELIGIOUS-ONLY (255 rows) |
| yor | Yoruba | flores_plus, ntrex, mafand, smol_sent | 2973 (150) | 1774 / 0 / 0 | 1,728,420 | FLORES dev-train; 1199 AfriScience |
| zne | Zande | none | 255 (10) | 0 / 255 / 0 | 173,973 | RELIGIOUS-ONLY (255 rows) |
| zul | Zulu | flores_plus, ntrex, mafand, smol_sent | 2999 (151) | 1799 / 0 / 0 | 1,196,952 | 1200 AfriScience |

### RL-list languages outside the 81 (arz dag bbj ibb)
| iso | language | benchmark references | chosen pairs (val) | human-general / religious / unverified | eligible pairs (gross, before dedupe) | register caveat |
|---|---|---|---:|---|---:|---|
| arz | Egyptian Arabic | flores_plus | 255 (10) | 0 / 255 / 0 | 43,326 | RELIGIOUS-ONLY (255 rows) |
| bbj | Ghomala | none | 0 (0) | 0 / 0 / 0 | 0 |  |
| dag | Dagbani | glot_story | 0 (0) | 0 / 0 / 0 | 0 |  |
| ibb | Ibibio | ibom_mt | 495 (20) | 495 / 0 / 0 | 977 | 495 NC |

### Other languages of the table that got pairs (not RL)
| iso | language | benchmark references | chosen pairs (val) | human-general / religious / unverified | eligible pairs (gross, before dedupe) | register caveat |
|---|---|---|---:|---|---:|---|
| aar | Afar | none | 78 (8) | 78 / 0 / 0 | 110 |  |
| ada | Adangme | none | 255 (10) | 0 / 255 / 0 | 58,795 | RELIGIOUS-ONLY (255 rows) |
| alz | Alur | none | 255 (10) | 0 / 255 / 0 | 6,998 | RELIGIOUS-ONLY (255 rows) |
| ber | Berber | none | 207 (8) | 207 / 0 / 0 | 351 |  |
| bum | Bulu | none | 255 (10) | 0 / 255 / 0 | 25,735 | RELIGIOUS-ONLY (255 rows) |
| cce | Chopi | none | 255 (10) | 0 / 255 / 0 | 16,818 | RELIGIOUS-ONLY (255 rows) |
| chw | Chuabo | none | 255 (10) | 0 / 255 / 0 | 16,890 | RELIGIOUS-ONLY (255 rows) |
| cjk | Chokwe | flores_plus | 255 (10) | 0 / 255 / 0 | 33,405 | RELIGIOUS-ONLY (255 rows) |
| cop | Coptic | none | 255 (10) | 0 / 255 / 0 | 15,222 | RELIGIOUS-ONLY (255 rows) |
| dik | Dinka | flores_plus | 255 (10) | 0 / 255 / 0 | 12,906 | RELIGIOUS-ONLY (255 rows) |
| dje | Zarma | none | 254 (10) | 0 / 254 / 0 | 55,033 | RELIGIOUS-ONLY (254 rows) |
| dop | Lelemi | none | 255 (10) | 0 / 255 / 0 | 15,231 | RELIGIOUS-ONLY (255 rows) |
| ful | Fulah | none | 499 (20) | 499 / 0 / 0 | 48,225 |  |
| knc | Kanuri | flores_plus | 499 (20) | 499 / 0 / 0 | 4,656 |  |
| koo | Konzo | none | 255 (10) | 0 / 255 / 0 | 33,781 | RELIGIOUS-ONLY (255 rows) |
| kwn | Kwangali | none | 255 (10) | 0 / 255 / 0 | 98,047 | RELIGIOUS-ONLY (255 rows) |
| lam | Lamba | none | 255 (10) | 0 / 255 / 0 | 13,656 | RELIGIOUS-ONLY (255 rows) |
| luy | Luyia | none | 499 (20) | 0 / 0 / 499 | 6,546 | 499 unverified-provenance pairs |
| men | Mende | none | 255 (10) | 0 / 255 / 0 | 1,557 | RELIGIOUS-ONLY (255 rows) |
| mgr | Mambwe | none | 255 (10) | 0 / 255 / 0 | 40,483 | RELIGIOUS-ONLY (255 rows) |
| nba | Nyemba | none | 253 (10) | 0 / 253 / 0 | 25,476 | RELIGIOUS-ONLY (253 rows) |
| nde | North Ndebele | ntrex | 254 (10) | 34 / 220 / 0 | 29,800 | relig. 87% |
| ndo | Ndonga | none | 255 (10) | 0 / 255 / 0 | 91,935 | RELIGIOUS-ONLY (255 rows) |
| nyk | Nyaneka | none | 255 (10) | 0 / 255 / 0 | 106,587 | RELIGIOUS-ONLY (255 rows) |
| nyu | Nyungwe | none | 255 (10) | 0 / 255 / 0 | 23,055 | RELIGIOUS-ONLY (255 rows) |
| rnd | Ruund | none | 255 (10) | 0 / 255 / 0 | 121,730 | RELIGIOUS-ONLY (255 rows) |
| sop | Songe | none | 254 (10) | 0 / 254 / 0 | 47,209 | RELIGIOUS-ONLY (254 rows) |
| taq | Tamasheq | flores_plus | 255 (10) | 0 / 255 / 0 | 5,062 | RELIGIOUS-ONLY (255 rows) |
| teo | Teso | none | 498 (20) | 498 / 0 / 0 | 22,993 |  |
| toi | Tonga | none | 255 (10) | 0 / 255 / 0 | 3,657 | RELIGIOUS-ONLY (255 rows) |
| ttj | Tooro | none | 498 (20) | 498 / 0 / 0 | 38,453 |  |
| tzm | Tamazight | none | 499 (20) | 499 / 0 / 0 | 6,363 |  |
| zgh | Standard Moroccan Tamazight | flores_plus | 211 (8) | 211 / 0 / 0 | 383 |  |

### 1.4 Size and selection rule
UniMax-style water-filling of a 48,000-pair budget over the languages after the class rules (cap 1,400 per language, a language with <= 250 eligible pairs keeps all of them), deterministic hash order, global English-sentence split: validation pairs = the smallest hash fractions per language (8-30 per language), no English sentence of a validation pair appears in training (any language). Within a pool, pairs that A0 and the stage-A checkpoint never saw (sources added after A0: stage-B-new; AfriScience) are taken first (`prefer_unseen`) because the SFT/RL model's drafts on text it has not seen are the realistic ones. Result: **57,462 pairs (54,990 train, 2,472 validation), 116 languages**. Benchmark exact-match check (`--bench-dir`: 191k normalised benchmark sentences, FLORES devtest, NTREX, MAFAND, SmolSent, SALT, GlotStory, IbomMT, FLORES-MOZ, either side) removed 0 further pairs; the protected index (protected-2026-10-28.3, both sides, rules R1-R3) removed 17: the shards had already passed it.

## 2. Draft generation and the copy-bias problem

The SFT/RL model saw most of these pairs, so its own drafts on them are better than on unseen text; an editor trained on such drafts learns "copy". Mitigations (all implemented, measured in the run report):

1. **Mixed-quality draft systems per pair** (`post_editor_make_drafts.py`, one engine per model): RL stage-A greedy, sampled T 0.7 (top-p 0.95) and T 1.0; SFT-final greedy and T 0.7; A0 270M step 5,625 greedy; RL beam-4 (HF batched, bf16) on a per-language subset and MBR-16 (chrF++ utility, 16 samples T 0.7 + greedy) for the languages where the decode study found MBR clearly ahead of beam (orm ewe lug tir). vLLM's own beam search is far too slow (more than 15 minutes of GPU time without a single batch finishing on 7.8k rows); the HF path does 7.8k rows in about 18 min.
2. **External systems as extra drafts** (owner decision for DRAFTS only; `post_editor_mt_drafts.py`): NLLB-200 (distilled 600M, 3.3B) and MADLAD-400 (3B, 7B) in greedy / beam-4 / sampling, on a shared sentence pool so every pool sentence has drafts from several systems. Every draft row carries `draft_system`, `draft_decoding`, `draft_licence` (NLLB: cc-by-nc-4.0 = non-commercial; MADLAD: apache-2.0 per the model card; ours: own); the share of NC-origin drafts is reported and `post_editor_build_sft.py --exclude-draft-licences nc` / `--exclude-systems` produce the ablations (drop NLLB; drop MADLAD = the held-out-system editor). TranslatePsy-AfriSLM-0.8B (`qvac/TranslatePsy-AfriSLM-0.8B`, public, downloaded) covers only its 19 languages and needs a chat-prompt path (vLLM, `e2_gen_chat.py` style): model downloaded and accessible, NOT run (time; its 19 languages are mostly high-resource ones already covered by NLLB / MADLAD). Per-language coverage of each external system: `mt_coverage.csv`.
3. **Synthetic corruption of the human reference** as extra drafts: drop 1-3 words, insert words, swap words, truncate to 35-75%, repeat a phrase, change a number / named entity, replace a clause by English source words (language switch), neighbour-sentence swap (a fluent draft of another sentence), plus `near` drafts (reference with 1-2 tiny edits). Real edit work even where the model draft is perfect.
4. **Leave-alone control**: 5-7% (7% drawn, 5.2% of the final examples) of all training examples have draft == reference (+ 3% (2.2%) near-reference); the false-edit rate on such rows is reported by the evaluation (`--copy-control`).
5. **Edit-rate matching ("quantile matching")**: every pair draws a desired sentence chrF++ from the empirical distribution of OUR model's draft quality on UNSEEN benchmark text of the same language (RL stage-A greedy drafts of flores_plus / ntrex, `--benchmark-predictions`) and receives the candidate draft whose chrF++ is closest (random among those within 2 chrF++, weighted by a prior per draft system so one system cannot take over). `matching_report.csv` prints both distributions per language (train matched drafts vs benchmark drafts: mean, p10 / p50 / p90). Result: On the training pairs the matched drafts average 36.0 sentence chrF++ against 30.6 for the RL drafts on the benchmarks (55 languages with a benchmark distribution; the rest use the pooled distribution): the matching removes most of the gap between our model's drafts on seen text (35.6) and unseen text but leaves a +5 chrF++ residual (candidates of low quality are scarcer than the target needs; the sampled T 1.0, A0 and corruption drafts are the tail). Sources of the matched drafts in E1: a0_greedy 10,293, corrupt 10,283, identity 3,180, near 1,394, rl_beam 1,086, rl_greedy 6,481, rl_mbr 328, rl_t07 6,760, rl_t10 8,055, sft_greedy 6,010, sft_t07 7,404. Draft == reference in 5.2% of the training examples..
6. **Exposure-aware drafts** (supported, optional): `--exposure-map pair_id,first_shard` + `--unseen-after-shard K` drafts only the pairs a checkpoint saved after shard K has never seen (the Hub holds `<HF_ORG>/gemma3-up6-g15-g1b-s1..s5` and `-g1a`; shards are random language-stratified partitions of the pack set, so a pair's shard needs a decode scan like `contam_audit_14_385m_scan.py` with the chosen pairs as the index: not run, 1-2 CPU-hours on the data box). The per-checkpoint metadata on the A100 (`runs/g1b_s6/last`) is empty (checkpoints removed); the bucket keeps only the final `g1b_s6/last` (3.75 GB).
7. Evaluation only on benchmark drafts (unseen) AND on drafts of a held-out slice of the editor's own sources (`post_editor_make_drafts.py --summarize --val-system S` writes `val_predictions.jsonl` in the evaluation format).

Not built (owner asked to note only): a variant that conditions on an explicit draft-quality hint (the draft's AfriCOMET-QE bucket, e.g. `Draft (quality 3/5):`): cheap to add (one scoring pass over the drafts, one extra prompt field, `PE_FORMAT_VERSION` 2), useful because the editor currently has to judge draft quality from the text alone; at inference the bucket comes from the same QE model, so it costs one scoring pass. Also not built: QE-high MACHINE targets (target_type `qe_high`). Reason: every one of the 81 RL languages already has human references in the selection, so no language needs them; the rule the owner gave (ensemble QE >= the language's human-reference anchor of `shaping_clean_v2.json` AND above the language's own 90th percentile, none for consensus-only languages) would apply to no row. The circularity caveat for any future use: selecting targets with AfriCOMET and judging with AfriCOMET alone is circular and AfriCOMET has a length / entity bias (`reports/data/africomet_bias_2026-11-02`), so the evaluation here stays reference-based (chrF++ against human references) with AfriCOMET only next to it.

## 3. Prompt format and SFT recipe

User turn (`gemma_mt.post_edit.build_pe_user_content`, the chat template of Gemma 3 around it):

```
Post-edit the draft translation to {language}:  {english}

Draft:  {draft}
```
Model turn: `{tag} {human reference}<end_of_turn>` with the SINGLE-TOKEN language tag (`<unusedK>`; tag mode `single_token`), i.e. byte-identical in shape to a translation so `format.strip_iso_prefix`, `rl.text.decode_completion` and the RL reward read it unchanged. Loss only on the model turn; drafts are cut to 192 tokens in training and in evaluation (same function). Training tokenizer = the ORIGINAL `google/gemma-3-270m-it` (the renamed RL tokenizer is not used: pack fingerprint). Optional KEEP marker token (`<unused6150>`, variant E2: target of a draft that is already right = tag + marker) is implemented and tested (`--keep-variant`, `--init-marker-rows`) but was not run: default format uses no new token.

Why not keep the literal "Translate the text to X:" prefix and append the draft: a model that can ignore the draft and translate is exactly the failure that makes an editor useless; a new verb makes the task explicit (~50k examples teach it). If the editor loses translation ability that would show on the draft-free control (not part of the first run).

SFT recipe (`configs/post_editor/pe_sft.yaml`, existing trainer, full fine-tune of all layers): weights = RL stage-A (`<HF_ORG>/gemma-0.4B-translate-rl@stage-a-b64`; the alternative is SFT-final: stated in the run report), packed 2,048-token sequences (one language per pack), Muon (2-D decoder matrices) + AdamW (embedding / norms) as stage B, LR 2e-5 cosine to 1e-5, 5% warm-up, 2 epochs over the set, 32 packs (65k tokens) per step, fp32 master weights with bf16 autocast, fused loss, per-language validation loss every 50 steps; no Hub push.

Hardware and time (measured on the 3090 Ti: E1 training 40 min for 2 epochs (278 steps x 65k tokens, micro batch 2 + gradient checkpointing, about 12 s per step, 6.5k tokens/s; validation loss plateaued after about 1.3 epochs); drafts for 47k pairs: 72 min of vLLM / HF generation for 8 draft systems; benchmark drafts 5 min greedy, 18 min beam-4, 19 min MBR-16 for 7.8k sentences; editor inference 5 min for 34.5k sentences; AfriCOMET about 20 min for E1; total GPU about 3.5 h. An A100 80 GB would be about 2-3x faster for training): a 24 GB GPU is enough for the training (micro batch 2-4 with gradient checkpointing, or 8 on an A100 80 GB); the draft generation dominates the GPU time.

## 4. Draft quality distributions
Sentence chrF++ of each draft system against the human reference on the E1 training pairs (47,423 pairs; per-language table: `reports/data/post_editor_2026-11-03/drafts/draft_quality_by_language_system.csv`; the matched E1 mix is in `matching_report_e1.csv`). The range the editor sees is 23 (A0 270M) to 100 (identity); the RL model's drafts on the benchmarks average 30.5 (greedy, FLORES+ / NTREX subset), 32.9 over the four benchmarks.

| draft system | n | mean sentence chrF++ | p10 | p50 | p90 |
|---|---:|---:|---:|---:|---:|
| a0_greedy | 47423 | 23.51 | 5.5 | 20.9 | 45.1 |
| corrupt | 47423 | 74.13 | 17.7 | 86.9 | 96.4 |
| identity | 47423 | 100.0 | 100.0 | 100.0 | 100.0 |
| near | 47423 | 89.03 | 80.3 | 90.5 | 95.6 |
| rl_beam | 8834 | 37.42 | 13.9 | 35.9 | 63.1 |
| rl_greedy | 47423 | 35.56 | 13.3 | 33.6 | 60.8 |
| rl_mbr | 2920 | 31.25 | 15.0 | 30.0 | 48.1 |
| rl_t07 | 47423 | 34.16 | 14.4 | 31.6 | 57.8 |
| rl_t10 | 47423 | 30.22 | 13.0 | 27.3 | 51.9 |
| sft_greedy | 47423 | 32.41 | 8.7 | 30.1 | 59.3 |
| sft_t07 | 47423 | 30.03 | 11.8 | 26.8 | 53.0 |

## 5. Evaluation protocol (`post_editor_eval.py`, `post_editor_comet.py`)
* Input = the evaluation's predictions.jsonl format (RL stage-A drafts: greedy on FLORES+ devtest, NTREX, MAFAND, SmolSent with 250 sentences per language and benchmark, deterministic subsample, the SAME sentences for draft and edit; beam-4 and MBR-16 drafts of the same model on a 100-sentence FLORES+ / NTREX subset). The editor runs greedy (beam-4 optional) on source + draft. Contamination flags from `reports/data/contamination_385m_2026-11-03` (`clean`, `no_pair` subsets).
* Scores per language and macro (equal weight per language): chrF++ and spBLEU with a paired sentence bootstrap against the draft (same resample for both, 1,000 resamples); AfriCOMET reference-based (STL-1.1) and QE for the 42 validated languages (`post_editor_comet.py`, paired bootstrap, macro equal weight per language). Restricted to benchmarks / languages that have references; languages without any benchmark reference are listed in the coverage table.
* Strata: religious-only / religious-mostly languages versus mixed-register languages (owner request), by benchmark; per-language wins / ties / losses (languages whose CI excludes 0), sentence-level wins / losses.
* What the editor did: share of drafts left unchanged, word-edit-rate distribution (0, 0-10%, 10-30%, 30-60%, > 60%), hard failures before / after (empty, loop by `rl.guards`, copy of the source, wrong language by GlotLID when the model file is available), false-edit rate on draft == reference rows.
* Baselines at similar compute on the same sentences: beam-4 and MBR-16 of the same RL model scored against its greedy draft (`--baseline`), and the decode-study table.
* Extra cuts: editing the benchmark drafts of NLLB-3.3B and MADLAD (same sentences); held-out-system editor (trained without MADLAD drafts) tested on MADLAD drafts; editor on drafts of a held-out slice of its own training sources.
* Caveats: one training seed; chrF++ uses one reference; AfriCOMET length / entity bias (shown next to chrF++, never instead).

## 6. Risks
1. **Copy bias** (section 2): measured by the false-edit rate, share unchanged, and the gain on the held-out-system test.
2. **Register skew**: even after the caps, 28 languages are religious-only (a few hundred rows each); benchmark text is news / Wikipedia. The strata in the evaluation show whether the gain differs.
3. **Alignment noise in the references**: AfriScience sentence alignments are sometimes shifted by a clause; MT560 JW sentences are paraphrastic; the editor's target is whatever the human text says, not the best translation of this English.
4. **Exposure**: references of training pairs may have been in the SFT / RL data of the draft model; the benchmark test is the unseen one.
5. **Circularity**: none for the main metric (human references); AfriCOMET only next to chrF++.
6. **Licences**: NC-origin drafts (NLLB) and NC targets (IbomMT, AfriDocMT, TED, Tanzil) make the editor weights research-only if they are in the set; rows are tagged for ablations.
7. **Tag handling**: single-token tags in training (original tokenizer) versus renamed tokens in the RL checkpoints (same ids): the evaluation prompts have no tag and the decoder accepts both.

## 7. Exact commands (3090 Ti box; env: `/workspace/pe/env.sh`)
```
# select (8 parallel slices, then merge with the protected index and the benchmark text)
python scripts/post_editor_select_data.py --input s3://<S3_BUCKET>/clean_rerun_2026-11-01/backup/hub --subdirs data data_devhalf specialist extra extra2k --afriscience afs_train.parquet --out SEL --tmp T --slice i/8 --save-state state/si.pkl
python scripts/post_editor_select_data.py --input none --out SEL --load-states state/*.pkl --bench-dir bench/test_sets --protected-index protected_index.npz --protected-manifest protected_manifest.json --chars-per-token cpt.json --profile-csv profile.csv --budget 48000 --cap-per-lang 1400
# benchmark drafts of the RL model, then the training-pair drafts of every system
python scripts/post_editor_bench_drafts.py --model RL_DIR --bench-root bench/test_sets --contam-dir contam --out BD --benchmarks flores_plus ntrex mafand smol_sent --per-language 250 --systems greedy
python scripts/post_editor_bench_drafts.py ... --benchmarks flores_plus ntrex --per-language 100 --systems beam4 --backend hfgpu      # and --systems mbr16 with vllm
python scripts/post_editor_make_drafts.py --pairs SEL/pairs.jsonl --out DR --model RL_DIR --plan "rl_greedy=greedy;rl_t07=sample,0.7,0.95;rl_t10=sample,1.0,1.0"
python scripts/post_editor_make_drafts.py ... --model SFT_DIR --plan "sft_greedy=greedy;sft_t07=sample,0.7,0.95"      # and A0
python scripts/post_editor_make_drafts.py ... --system corrupt --type corrupt     # identity, near
python scripts/post_editor_mt_drafts.py --pairs SEL/pairs.jsonl --out DR --plan "nllb3.3b:greedy,beam4;madlad7b:greedy;nllb600m:greedy,sample0.8;madlad3b:greedy" --per-language 160
# build, train, evaluate
python scripts/post_editor_build_sft.py --pairs SEL/pairs.jsonl --drafts DR --out packed_pe --benchmark-predictions BD/predictions_greedy_full.jsonl --benchmarks flores_plus ntrex
python -m gemma_mt.train_sft --config configs/post_editor/pe_sft.yaml model_name=RL_DIR data_dir=packed_pe out_dir=runs/pe micro_batch_size=4 grad_accum_steps=8
python scripts/post_editor_eval.py --predictions BD/predictions_greedy_full.jsonl BD/predictions_beam4.jsonl --labels greedy beam4 --model runs/pe/final --out EV --baseline beam4=BD/predictions_beam4.jsonl --baseline mbr16=BD/predictions_mbr16.jsonl --copy-control 600
python scripts/post_editor_comet.py --out CM --system draft=EV/greedy/edited_predictions.jsonl:draft --system edited=EV/greedy/edited_predictions.jsonl:hyp --pair edited:draft   # venv_comet
```

## 8. Reading the result
The editor "helps" only if the gain over the draft is positive with a CI above 0 on the CLEAN subsets in chrF++ AND AfriCOMET does not fall, the gain is not carried by a handful of languages (wins / losses), the false-edit rate on correct drafts is small, and it matches or beats beam-4 at lower cost. A gain that appears only on drafts from our own model and vanishes on NLLB / MADLAD drafts means the editor learned our model's habits, not editing.

## 9. What RL for the editor would need (owner: only note, do not build)
Reward = existing reward on the output + (reward(output) - reward(draft)), so copying the draft earns exactly the base reward minus itself = 0 improvement. Needs: (1) a prompt pool of (source, draft) built from a FROZEN draft system with a mixed-quality draft pool (the same mixture as here: greedy / sampled / weaker checkpoint / external MT / corrupted) refreshed per stage; (2) `rl/data.py` prompt rows carrying the draft and `format` rendering of the editor prompt (`post_edit.build_pe_prompt`); (3) the reward of the DRAFT computed once per prompt with the same scorer (one extra scoring call, cached; the draft reward uses the same guards) and the improvement term wired into `rl/reward.py` (clipped, with a minimum so a regression is penalised more than a gain is paid); (4) the KL reference = the SFT editor, not the translation model; (5) the same guards (finished, loop, LID, copy-of-source) plus a copy-of-draft detector so an unchanged output is neutral and logged; (6) rollout length: the prompt is about 2x longer, so `vllm_max_model_len` and the b64 / b256 bins refer to source + draft; (7) evaluation = this protocol on benchmark drafts, with reference-based chrF++ as the judge (the reward model is AfriCOMET-QE based: do not judge with it alone). Hardware: the stage-A recipe (A100 80 GB, about 20-30 s per step).

## 10. Open decisions for the owner
1. **Do not train more variants of the E1 type** (result in `post_editor_run_2026-11-03.md`): the editor applied to every sentence is flat on FLORES+ and negative on NTREX / MAFAND / SmolSent in chrF++ and AfriCOMET, and loses to beam-4 and MBR-16. E2 (external drafts), E2q (QE-filtered targets) and E3 (held-out system) are built and tested but untrained: the owner holds them.
2. **Repair-only gate**: using the E1 editor only on empty / unfinished / looping drafts (4% of sentences) is significantly positive in chrF++ (+0.2 to +0.7) with almost no AfriCOMET change. Worth keeping as a cheap post-processor only if beam is not available; beam-4 alone is better.
3. **If an editor is revisited**: add a quality signal to the prompt (QE bucket of the draft) and make the identity map optimal for good drafts, or go to editor-RL with the improvement-over-draft reward (section 9): SFT on references cannot teach 'leave it alone'.
4. Efik NT sheet, AfriScience train, SALT train were prepared for E2 but never trained on (selection v4 only).
5. TranslatePsy drafts were not generated; MADLAD-7B was abandoned (too slow on 24 GB); MADLAD-3B drafts were used for the held-out comparison.
