# Rare-language test data for the Gemma-3-270m English -> African model: SMOL, GATITOS and four more public sets (2026-10-19)

Owner task 2026-10-19: find benchmark / evaluation data for the rare languages of the Gemma-3-270m-it English -> African translation model, add it to the evaluation pipeline (D-34), run SFT and base on it. Code and decision: `gemma_mt/` (D-42 in `gemma_mt/DECISIONS.md`, tests `gemma_mt/tests/test_eval_smol.py`); raw result files of the three runs are in `reports/gemma_rare_language_benchmarks_2026-10-19/{gain,covered,other}/` (`results.json`, `results.csv`, `report.md`; `predictions.jsonl` stayed on the old box in `/workspace/sft_build/eval/{smol_gain,smol_covered,other_sets}/`); the evidence files behind sections 3-6 (GlotLID counts, token-coverage tables, row counts, contamination summaries, training-source counts) are in `.../meta/`.

## 0. Headline

* **Coverage.** Before: 47 of the 77 SFT target languages had a test set (FLORES+ / NTREX-128 / MAFAND-MT), 30 had none. **Now 18 of those 30 have at least one** (9 through SMOL / GATITOS, 13 through four more public sets, 4 through both); **12 remain without any test set**: `urh`, `bin`, `oke`, `bbj`, `swc`, `wal`, `yao`, `sid`, `ish`, `dua`, `bas`, `wes`. SmolSent covers 45 of the 77 languages, SmolDoc 45, GATITOS 35 (SMOL + GATITOS together 49).
* **Newly covered (no test set before):** SMOL / GATITOS: `efi`, `nbl`, `tiv`, `kon`, `iso`, `ach`, `nyn`, `ndc`, `kri` (`nyn` only through `cgg` = Rukiga, a proxy); IbomMT: `ibb` (+ `efi`); FLORES-MOZ: `seh`; GlotStoryBook: `kqn lun loz kua her dag tsc` (+ `nbl`); SALT: `nyn ach`.
* **Licences.** SMOL / SmolSent / SmolDoc / GATITOS: CC-BY-4.0 (attribution kept in `SOURCE.json` next to the data); IbomMT: **CC-BY-NC-4.0** (non-commercial); FLORES-MOZ: CC-BY-SA-4.0; GlotStoryBook: CC-BY / CC-BY-NC / CC BY-NC-SA per row (kept in the files); SALT: CC-BY-SA-4.0 (gated, auto-accept; the project READ token has access). Test-only use; none of these sets may be trained on.
* **Contamination (English side against the SFT training windows).** SmolSent 35413/38833 = 91.2% of the rows (787 of the 863 English sentences; clean = 76 per language), SmolDoc 58.5% (exact or >= 50% of the 8-grams; 22 clean documents per 130-document language), GATITOS 71.6% (entry equals a whole training sentence), IbomMT / FLORES-MOZ 95.3% (they use the FLORES+ English), SALT 84.4%, GlotStoryBook only 7.2%. The reference PAIR was seen only for the languages whose SMOL translations are in the training data (bam, wol, ewe, yor, zul, hau, orm, ibo; SALT-Luganda): for the 18 newly covered languages the pair-seen share is 0% on SmolSent / SmolDoc / IbomMT / FLORES-MOZ / GlotStoryBook / SALT-nyn / SALT-ach.
* **SFT vs base, macro chrF++ over the newly covered languages, all rows (clean rows) of SFT vs base `google/gemma-3-270m-it`**: smol_sent 20.2 (15.6 clean) vs 11.5 (11.7), 8 languages; smol_doc 24.8 (18.4 clean) vs 16.5 (16.0), 8 languages; ibom_mt 18.5 (16.6 clean) vs 13.9 (13.7), 2 languages; flores_moz 25.6 (25.0 clean) vs 13.2 (12.8), 1 language; glot_story 22.8 (22.9 clean) vs 12.1 (12.2), 8 languages; salt 25.6 (22.3 clean) vs 10.4 (10.3), 2 languages. The base model scores 5-15 chrF++ macro on every set (it copies or garbles the English), so almost every gap is positive; what matters are the small and the negative ones (7.1). The newly covered languages are far weaker than the old ones: smol_sent 29.77 (languages that had a test set) vs 20.23 (new); smol_doc 30.98 (languages that had a test set) vs 24.79 (new); gatitos 13.92 (languages that had a test set) vs 9.39 (new).
* **GATITOS (word level)** is a weak quality signal for this model: macro exact match 1.7% (new languages) / 4.2% (old) on all entries, 0.8% / 2.2% on the clean ones (base: 0.1-0.2%); the SFT data has almost no word-level pairs and the model answers a single word with a sentence, a copy or nothing (7.4).
* **Failure modes the new sets expose** (SFT, English -> X, greedy, 320 / 900 / 40 new tokens): runaway repetition. Share of completions that end inside the budget (macro over languages): SmolSent 84.2% (new languages) / 95.2% (old), SmolDoc 52.8% / 62.8%; worst `kon` (35% of the sentences and 1% of the documents finish: it loops on `muna muna ...`); the base model finishes 95-99% (it does not loop, it simply does not translate). On single words the SFT model often returns the tag and nothing else (empty translation): `bem` 75.2%, `nbl` 63.5%, `ndc` 61.6%, `kon` 54.9%, `ssw` 53.8% of the GATITOS prompts. The RL reward penalises truncated, degenerate and empty completions (failure classes of D-37 / D-38); these sets are a cheap way to watch them move on languages without a FLORES+ dev set.

## 1. What was done, where things are

* **Code** (laptop `gemma_mt/`, copy on both boxes in `/workspace/sft_build/gemma_mt_eval_candidate2/`): seven new benchmark names in `eval/benchmarks.py` (`smol_sent smol_doc gatitos ibom_mt flores_moz glot_story salt`), loaders and raw-file downloaders in `eval/fetch_data.py` (pinned revisions, no `datasets` dependency, `SOURCE.json` with licence next to every data directory), seven new columns in `languages.py` (same convention as `flores / ntrex / mafand`: one code per benchmark, `None` = not covered), word-level metrics (`exact`, multi-reference chrF++) in `eval/metrics.py` / `eval/eval_translation.py`, document / word rules in `eval/contamination.py`, per-benchmark generation budgets and batch sizes, `--bench-limit` (clean rows first), `--contamination-dir`, `release_cuda()` and `CometPair.release()` in `eval/sft_benchmarks.py` / `eval/runtime.py`, word-level and document tables in `eval/sft_report.py`, `BENCHMARKS=` / `CONTAM_DIR=` in `scripts/eval_sft_benchmarks.sh`. **The default behaviour is unchanged** (`--benchmarks` still defaults to flores_plus ntrex mafand; the report layout of a default run is identical).
* **Tests**: `cd gemma_mt && ../.venv-comet/bin/python tests/run_tests.py` -> **390 passed, 0 failed, 8 skipped** (334 / 7 when the task started; my new file `tests/test_eval_smol.py` adds 19 tests, the rest is from another session's work in the same tree)
* **Data on the old box** (`/workspace/sft_build/bench/test_sets/<benchmark>/`, 62 MB for the seven sets; contamination flags `bench/contamination_<benchmark>.json`; mapping evidence and counts in `bench/smol_meta/`; the full 271 MB mirror of the SMOL English-source files was deleted after the selection).
* **Staged on the rental** (<box>, nothing else touched, no GPU use): the seven data directories in `/workspace/sft_build/bench/test_sets/` (md5 of samples verified), their flag files in `/workspace/sft_build/bench/test_sets/_contamination/`, the updated evaluation code in the NEW directory `/workspace/sft_build/gemma_mt_eval_candidate2/`.
* **Runs** (old box RTX 5070, HF sampler, `flock /workspace/sft_build/.gpu.lock`, my GPU memory 2.3-4.1 GB; the card is shared with a foreign job that grew from 5.6 to 8.9 GB during the work). SFT = `<HF_ORG>/gemma3-270m-translate` main, commit 376a94f2314da6445878962fe22d59900ee2d2b9, training step 27638, macro validation loss 1.2654; base = `google/gemma-3-270m-it` `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`. Greedy, prompt `Translate the text to {language}:  {english}`, `max_new_tokens` 320 (sentences) / 900 (documents) / 40 (words), metrics on NFC text.
  * run A (`gain/`): the 9 languages that had no test set and are in SMOL / GATITOS (ach efi iso kon kri ndc nbl nyn tiv), full sets, no AfriCOMET (none of the nine is validated).
  * run B (`covered/`): the 37 mapped languages that already had a test set, SmolSent first 500 rows per language (the 76 clean rows first), SmolDoc 40 documents (clean first), GATITOS complete, AfriCOMET (validated languages, SmolSent only).
  * run C (`other/`): IbomMT, FLORES-MOZ, GlotStoryBook, SALT for 16 languages, complete sets, AfriCOMET for the validated ones.

## 2. SMOL and GATITOS: what they are, access, formats

Paper: Caswell et al., *SMOL: Professionally translated parallel data for 115 under-represented languages*, arXiv 2502.12301; GATITOS: Jones et al. 2023, arXiv 2303.15265.

| item | value (verified 2026-10-19) |
|---|---|
| location | Hub dataset **`google/smol`**, revision `fdaff3a1a019f89fa30a562a85d7c1d3e9150444` (lastModified 2026-08-06; the pin is in `eval/benchmarks.py`). The GitHub path `google-research-datasets/smol` returns 404 (API and web): the Hub is the only source I could verify. |
| access | open, **not gated**, no token needed (the READ token was not required) |
| licence | **CC-BY-4.0** (card metadata) |
| size | 600 MB for everything (gatitos 143 MB, smoldoc 370 MB, smolsent 83 MB, factuality ratings 0.4 MB); the English-source files we use are 62 MB |
| SmolSent | `smolsent/en_<code>.jsonl` (+ the reversed `<code>_en.jsonl`, same pairs): `{sl, tl, id, src, trg, is_src_orig}`; 863 English sentences per language (862 for `kab`, `som`), multi-way parallel (identical English in every file), professional translations; 114 language pairs |
| SmolDoc | `smoldoc/en_<code>.jsonl`: `{id, sl, tl, srcs[], trgs[], sep[], factuality, is_src_orig}`, sentence-aligned (`srcs` and `trgs` have equal length), 1-41 sentences, median 12 (about 190 English words, median 232 Gemma tokens, reference median 406 tokens, p98 1,024). 130 documents for the low-resource languages (after the April 2026 tier upgrade); the higher-resource ones have 207-668 (extra documents that are not multi-way parallel: the card names MediSMOL medical text, about 80 per language; the rest is not described). `sep` is the whitespace between sentences (space, newline, blank line, list bullets): documents are rebuilt with it. 130 language pairs |
| GATITOS | `gatitos/en_<code>.jsonl`: `{src, trgs[], sl, tl}`: about 4,000 English words / short phrases, each with ONE OR MORE accepted translations (6.6% of the entries have more than one); `en_<code>` = English source. 181 language pairs |
| direction | English -> X is the `en_<code>` file; `<code>_en` is the same pairs reversed (`is_src_orig` false). Some SMOL pairs have Russian or Chinese sources (`ru_ce`, `zh_yue` ...): not used |
| card changes | April 2026: more volunteer languages, MediSMOL documents, tier upgrades, `ff` renamed `fuc` ("neither appears to be very high quality"). The pin protects against further changes |

Why the numbers below are smaller than a first look at the card suggests: SMOL has 114 / 130 / 181 language pairs, but only **45 / 45 / 35 of our 77 targets** are in SmolSent / SmolDoc / GATITOS (49 in at least one). The rest of the languages are Asian, Middle-Eastern, Indigenous-American or European, or African languages we have no SFT data for (`nd`, `kr`, `ff`, `fuc`, `gaa`, `sus`, `dje`, `din`, `dov`, `alz`, `bci`, `tig`, `nqo`, `mfe`, ...).

## 3. Language-code mapping, verified on the sentences

SMOL uses its own codes (ISO 639-1 where one exists, `ber-Latn`, `ndc-ZW`, ...). Two checks per SmolSent file, 863 references each, both run on the old box: (1) **GlotLID** (`cis-lmu/glotlid` v3) top-1 label of every reference; (2) **token coverage**: the share of the reference tokens (NFC, lower-cased) that occur at least twice in the SFT training targets of each of our 77 languages (150k windows per language); the language with the highest coverage is shown next to the language it is mapped to. High-resource neighbours (ibo, hau) always get 0.4-0.6 from shared English words and punctuation, so only the ranking and the gap matter. GATITOS-only languages (no SmolSent) were checked on the entries.

| tag | SMOL code(s) | GlotLID top-2 (share) | best token coverage in our targets | verdict |
|---|---|---|---|---|
| `yor` | yo | yor_Latn 0.77, mcu_Latn 0.06 | yor 0.96 | ok |
| `hau` | ha | hau_Latn 1.00 | hau 0.99 | ok |
| `ibo` | ig | ibo_Latn 1.00, efi_Latn 0.00 | ibo 0.99 | ok |
| `pcm` | pcm | pcm_Latn 0.75, eng_Latn 0.16 | pcm 0.87 | ok; GlotLID calls 16% of the Pidgin 'English' (known), coverage 0.87 |
| `efi` | efi | efi_Latn 1.00 | efi 0.91 | ok |
| `twi` | ak | twi_Latn 1.00, fat_Latn 0.00 | twi 0.96 | ok; Akan = Twi (coverage 0.96); FLORES+ Twi is Akuapem |
| `ewe` | ee | ewe_Latn 1.00, umb_Latn 0.00 | ewe 0.93 | ok |
| `fon` | fon | fon_Latn 1.00, lem_Latn 0.00 | fon 0.92 | ok |
| `wol` | wo | wol_Latn 1.00, uzn_Latn 0.00 | wol 0.94 | ok |
| `swh` | sw | swh_Latn 0.86, swc_Latn 0.13 | swh 0.97 | ok; GlotLID swh 86% / swc 13% (Swahili general, not Congo Swahili) |
| `kin` | rw | kin_Latn 0.96, run_Latn 0.04 | kin 0.95 | ok |
| `som` | so | som_Latn 1.00, uzn_Latn 0.00 | som 0.95 | ok |
| `orm` | om | gaz_Latn 1.00 | orm 0.93 | ok |
| `amh` | am | amh_Ethi 1.00 | amh 0.84 | ok |
| `tir` | ti | tir_Ethi 0.89, tig_Ethi 0.11 | tir 0.72 | ok |
| `lug` | lg | lug_Latn 1.00, xho_Latn 0.00 | lug 0.91 | ok |
| `lin` | ln | lin_Latn 1.00 | lin 0.87 | ok |
| `zul` | zu | zul_Latn 0.96, xho_Latn 0.02 | zul 0.91 | ok |
| `xho` | xh | xho_Latn 0.99, zul_Latn 0.01 | xho 0.89 | ok |
| `afr` | af | afr_Latn 1.00, ind_Latn 0.00 | afr 0.96 | ok |
| `sna` | sn | sna_Latn 1.00, twx_Latn 0.00 | sna 0.87 | ok |
| `nya` | ny | nya_Latn 1.00, kzn_Latn 0.00 | nya 0.92 | ok |
| `sot` | st | sot_Latn 0.98, tsn_Latn 0.01 | sot 0.96 | ok |
| `tsn` | tn | tsn_Latn 1.00, nso_Latn 0.00 | tsn 0.97 | ok |
| `mlg` | mg | plt_Latn 0.99, xmv_Latn 0.00 | mlg 0.97 | ok |
| `bem` | bem | bem_Latn 0.98, lam_Latn 0.02 | bem 0.78 | ok |
| `tum` | tum | tum_Latn 0.96, tog_Latn 0.01 | tum 0.78 | ok |
| `run` | rn | run_Latn 0.92, kin_Latn 0.08 | run 0.80 | ok; GlotLID run 92% / kin 8% |
| `ven` | ve | ven_Latn 1.00, nya_Latn 0.00 | ven 0.89 | ok |
| `tso` | ts | tso_Latn 1.00, rng_Latn 0.00 | tso 0.96 | ok; SMOL `ts` = Xitsonga (not `tsc` Tswa) |
| `nso` | nso | nso_Latn 1.00, tsn_Latn 0.00 | nso 0.96 | ok |
| `ssw` | ss | ssw_Latn 0.99, zul_Latn 0.00 | ssw 0.66 | ok |
| `nbl` | nr | nbl_Latn 0.95, zul_Latn 0.02 | nbl 0.73 | ok |
| `sag` | sg | - | - | ok (GATITOS entries read: Hello -> Bara ala, I -> Mbi); GATITOS `sg` only |
| `tiv` | tiv | tiv_Latn 1.00, kac_Latn 0.00 | tiv 0.93 | ok |
| `mos` | mos | mos_Latn 0.96, biv_Latn 0.01 | mos 0.62 | ok |
| `kik` | ki | kik_Latn 0.99, kin_Latn 0.00 | kik 0.82 | ok |
| `luo` | luo | luo_Latn 1.00, kin_Latn 0.00 | luo 0.95 | ok |
| `dyu` | dyu | dyu_Latn 0.74, bam_Latn 0.25 | bam 0.78 (own 0.77) | ok; GlotLID: 25% of Dyula is called Bambara (known sibling confusion) |
| `bam` | bm | bam_Latn 0.88, dyu_Latn 0.11 | bam 0.83 | ok |
| `kon` | kg | kng_Latn 1.00, ktu_Latn 0.00 | kon 0.75 | ok; `kg` = Kikongo. `ktu` (Kituba) gets the same coverage (0.74) and is NOT mapped |
| `kab` | ber-Latn | kab_Latn 1.00 | kab 0.78 | ok; `ber-Latn`, not `ber` (Tifinagh) |
| `nus` | nus | - | - | ok (GATITOS entries read: Nuer -> Thok Nath) |
| `lua` | lua | - | - | ok (GATITOS entries read: Luba -> Tshiluba, Yes -> Eyowa) |
| `iso` | iso | - | - | ok (GATITOS entries read: Yes -> Ehe, Hello -> Uyere) |
| `ach` | ach | ach_Latn 0.97, adh_Latn 0.01 | ach 0.83 | ok |
| `nyn` | cgg | nyn_Latn 0.69, cgg_Latn 0.20 | nyn 0.47 | ok; **proxy**: `cgg` is Rukiga (the Runyankore-Rukiga cluster); GlotLID 69% nyn / 20% cgg; use SALT for the real Runyankore |
| `ndc` | ndc | sna_Latn 0.62, ndc_Latn 0.25 | sna 0.56 (own 0.24) | ok; **Shona-like text**: GlotLID sna 62% / ndc 25%, token coverage higher in our sna targets (0.56) than in our ndc targets (0.24); our ndc training data is small (22k pairs, one source). `ndc-ZW` is byte-identical and not used |
| `kri` | kri | kri_Latn 0.99, lin_Latn 0.00 | kri 0.74 | ok |

Codes that exist in SMOL and were **deliberately not mapped**:

| SMOL code | what it is | why not |
|---|---|---|
| `lu` | Luba-Katanga (GlotLID `lub` 98%; best coverage in any of our languages: kqn 0.51, i.e. none) | our `lua` is Luba-Kasai (Tshiluba); only GATITOS `lua` is Luba-Kasai |
| `ktu` | Kituba | same coverage as `kg` in our `kon` data; Kituba is a creole, `kon` stays `kg` |
| `ndc-ZW` | identical file to `ndc` (md5) | duplicate |
| `ber` | Tifinagh, GlotLID tzm_Tfng 84% | `tzm` has no SFT data |
| `nd` | North Ndebele (GlotLID nde 74% / zul 22%) | `nde` has no SFT data |
| `kr`, `ff`, `fuc`, `gaa` | Kanuri, Fulah, Pulaar, Ga | `knc`, `ful`, `gaa` have no SFT data |
| `dov`, `dje`, `din`, `alz`, `bci`, `sus`, `mfe`, `tig`, `nqo`, `aa` | Dombe (GlotLID toi), Zarma, Dinka, Alur, Baoule, Susu, Mauritian Creole, Tigre, N'Ko, Afar | not among our 77 targets |

## 4. Coverage table: the 77 SFT targets x benchmarks (English -> X rows)

Cells = number of English source rows per language (direction English -> X in every set; the reverse X -> English is the same data with the sides swapped, `--direction x2en` of `eval_translation`). FLORES+ = devtest 1,012 sentences, NTREX-128 = 1,997, MAFAND-MT = test split (the previous evaluation's counts). SmolSent = sentences, SmolDoc = **documents**, GATITOS = words / phrases, IbomMT = test split, FLORES-MOZ = devtest, GlotStory = aligned story sentences, SALT = `text-all` test. "test set" = a test set existed before (yes), exists only now (**new**), or still none (**none**). A language can be in several new sets; `nyn` in SMOL only through the proxy `cgg`.

Counts: **47 languages had a test set before, 30 had none; 18 of those 30 are now covered (9 by SMOL / GATITOS, 13 by the four other sets); 12 still have none.**

| tag | name | FLORES+ | NTREX | MAFAND | SmolSent | SmolDoc | GATITOS | IbomMT | FLORES-MOZ | GlotStory | SALT | test set |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `yor` | Yoruba | 1012 | 1997 | 1558 | 863 | 668 | - | - | - | - | - | yes |
| `hau` | Hausa | 1012 | 1997 | 1500 | 863 | 667 | - | - | - | - | - | yes |
| `ibo` | Igbo | 1012 | 1997 | 1500 | 863 | 478 | - | - | - | - | 500 | yes |
| `pcm` | Pidgin | - | - | 1564 | 863 | 213 | 4001 | - | - | - | - | yes |
| `efi` | Efik | - | - | - | 863 | 130 | 4001 | 1012 | - | - | - | **new** |
| `twi` | Twi | 1012 | - | 1500 | 863 | 347 | 4000 | - | - | - | - | yes |
| `ewe` | Ewe | 1012 | 1997 | - | 863 | 212 | 3990 | - | - | - | - | yes |
| `fon` | Fon | 1012 | - | - | 863 | 130 | 3998 | - | - | - | - | yes |
| `wol` | Wolof | 1012 | 1997 | - | 863 | 345 | 4001 | - | - | - | - | yes |
| `swh` | Swahili | 1012 | 1997 | 1835 | 863 | 664 | - | - | - | - | 500 | yes |
| `kin` | Kinyarwanda | 1012 | 1997 | 1006 | 863 | 345 | - | - | - | - | - | yes |
| `som` | Somali | 1012 | 1997 | - | 862 | 339 | - | - | - | - | - | yes |
| `orm` | Oromo | 1012 | 1997 | - | 863 | 477 | 3995 | - | - | - | - | yes |
| `amh` | Amharic | 1012 | 1997 | 1037 | 863 | 663 | - | - | - | - | - | yes |
| `tir` | Tigrinya | 1012 | 1997 | - | 863 | 339 | 4000 | - | - | - | - | yes |
| `lug` | Luganda | 1012 | - | 1500 | 863 | 343 | 3998 | - | - | - | 500 | yes |
| `lin` | Lingala | 1012 | - | - | 863 | 343 | 3994 | - | - | - | - | yes |
| `zul` | Zulu | 1012 | 1997 | 998 | 863 | 339 | - | - | - | - | - | yes |
| `xho` | Xhosa | 1012 | 1997 | 1002 | 863 | 344 | - | - | - | - | - | yes |
| `afr` | Afrikaans | 1012 | 1997 | - | 863 | 209 | - | - | - | - | - | yes |
| `sna` | Shona | 1012 | 1997 | 1005 | 863 | 345 | - | - | - | - | - | yes |
| `nya` | Chichewa | 1012 | 1997 | 1004 | 863 | 354 | - | - | - | - | - | yes |
| `sot` | Sesotho | 1012 | - | - | 863 | 341 | - | - | - | - | - | yes |
| `tsn` | Setswana | 1012 | 1997 | 1500 | 863 | 130 | 3995 | - | - | - | - | yes |
| `mlg` | Malagasy | 1012 | 1997 | - | 863 | 474 | - | - | - | - | - | yes |
| `ibb` | Ibibio | - | - | - | - | - | - | 1012 | - | - | - | **new** |
| `urh` | Urhobo | - | - | - | - | - | - | - | - | - | - | **none** |
| `bem` | Bemba | 1012 | 1997 | - | 863 | 130 | 3977 | - | - | - | - | yes |
| `tum` | Tumbuka | 1012 | - | - | 863 | 130 | 3983 | - | - | - | - | yes |
| `run` | Kirundi | 1012 | - | - | 863 | 130 | 3999 | - | - | - | - | yes |
| `ven` | Venda | - | 1997 | - | 863 | 130 | 3984 | - | - | - | - | yes |
| `tso` | Tsonga | 1012 | - | - | 863 | 130 | 3994 | - | - | - | - | yes |
| `nso` | Sepedi | 1012 | 1997 | - | 863 | 213 | 3996 | - | - | - | - | yes |
| `ssw` | Swati | 1012 | 1997 | - | 863 | 130 | 3924 | - | - | - | - | yes |
| `nbl` | South Ndebele | - | - | - | 863 | 130 | 4001 | - | - | 321 | - | **new** |
| `sag` | Sango | 1012 | - | - | - | - | 3994 | - | - | - | - | yes |
| `tiv` | Tiv | - | - | - | 863 | 130 | 4001 | - | - | - | - | **new** |
| `umb` | Umbundu | 1012 | - | - | - | - | - | - | - | - | - | yes |
| `kmb` | Kimbundu | 1012 | - | - | - | - | - | - | - | - | - | yes |
| `mos` | Mossi | 1012 | - | - | 863 | 130 | 4000 | - | - | - | - | yes |
| `kik` | Kikuyu | 1012 | - | - | 863 | 130 | 3994 | - | - | - | - | yes |
| `luo` | Luo | 1012 | - | 1500 | 863 | 342 | 3941 | - | - | - | - | yes |
| `kam` | Kamba | 1012 | - | - | - | - | - | - | - | - | - | yes |
| `bin` | Edo | - | - | - | - | - | - | - | - | - | - | **none** |
| `dag` | Dagbani | - | - | - | - | - | - | - | - | 135 | - | **new** |
| `oke` | Okpe | - | - | - | - | - | - | - | - | - | - | **none** |
| `shi` | Tachelhit | - | 1997 | - | - | - | - | - | - | - | - | yes |
| `dyu` | Dyula | 1012 | - | - | 863 | 130 | 3997 | - | - | - | - | yes |
| `bam` | Bambara | 1012 | - | - | 863 | 340 | 4000 | - | - | - | - | yes |
| `kbp` | Kabiye | 1012 | - | - | - | - | - | - | - | - | - | yes |
| `bbj` | Ghomala | - | - | - | - | - | - | - | - | - | - | **none** |
| `kon` | Kongo | - | - | - | 863 | 130 | 3998 | - | - | - | - | **new** |
| `kea` | Kabuverdianu | 1012 | - | - | - | - | - | - | - | - | - | yes |
| `kab` | Kabyle | 1012 | - | - | 862 | 208 | 4001 | - | - | - | - | yes |
| `nus` | Nuer | 1012 | - | - | - | - | 4001 | - | - | - | - | yes |
| `swc` | Congo Swahili | - | - | - | - | - | - | - | - | - | - | **none** |
| `lua` | Luba-Kasai | 1012 | - | - | - | - | 3998 | - | - | - | - | yes |
| `kqn` | Kaonde | - | - | - | - | - | - | - | - | 420 | - | **new** |
| `lun` | Lunda | - | - | - | - | - | - | - | - | 420 | - | **new** |
| `iso` | Isoko | - | - | - | - | - | 4001 | - | - | - | - | **new** |
| `loz` | Lozi | - | - | - | - | - | - | - | - | 960 | - | **new** |
| `wal` | Wolaytta | - | - | - | - | - | - | - | - | - | - | **none** |
| `ach` | Acholi | - | - | - | 863 | 130 | 4001 | - | - | - | 500 | **new** |
| `tsc` | Tswa | - | - | - | - | - | - | - | - | 101 | - | **new** |
| `vmw` | Makhuwa | 1012 | - | - | - | - | - | - | - | - | - | yes |
| `kua` | Kuanyama | - | - | - | - | - | - | - | - | 520 | - | **new** |
| `seh` | Sena | - | - | - | - | - | - | - | 1012 | - | - | **new** |
| `nyn` | Nyankore | - | - | - | 863 | 130 | 4001 | - | - | - | 500 | **new** |
| `her` | Herero | - | - | - | - | - | - | - | - | 549 | - | **new** |
| `yao` | Yao | - | - | - | - | - | - | - | - | - | - | **none** |
| `sid` | Sidamo | - | - | - | - | - | - | - | - | - | - | **none** |
| `ndc` | Ndau | - | - | - | 863 | 130 | 3991 | - | - | - | - | **new** |
| `kri` | Krio | - | - | - | 863 | 207 | 4020 | - | - | - | - | **new** |
| `ish` | Esan | - | - | - | - | - | - | - | - | - | - | **none** |
| `dua` | Duala | - | - | - | - | - | - | - | - | - | - | **none** |
| `bas` | Basaa | - | - | - | - | - | - | - | - | - | - | **none** |
| `wes` | Cameroon Pidgin | - | - | - | - | - | - | - | - | - | - | **none** |

## 5. Other public sources for the languages that still had no test set

Method: Hub search API for each language name / code, dataset-card and file-list reads, `datasets-server` /info and /first-rows, GitHub raw listings; nothing larger than a few rows was downloaded for the survey. A first pass was done by a research sub-agent and then re-checked by me (what I re-checked myself is marked **v**; the unmarked rows rest on the sub-agent's reading of cards and file lists and were not independently re-verified). "Usable" = held-out, English-aligned, human-translated, language verified on sample sentences.

| source (Hub id / URL) | languages of ours | rows (test) | licence | verdict |
|---|---|---|---|---|
| `howard-nlp/ibom-mt` (**v**, integrated as `ibom_mt`) | **ibb** (new), efi (second source) | 1,012 each (+1,000 train / 997 dev) | CC-BY-NC-4.0 | usable. Test = FLORES+ devtest English translated by 3 paid linguists (arXiv 2511.06531). Non-commercial licence. GlotLID calls Ibibio `efi` (sibling); token coverage in our `ibb` targets is the lowest of the mapped sets (0.42, versus 0.57 for ibo: the ibb training set has only 4.9k pairs). Also has Anaang (`anw`) and Oro (`orx`), no SFT data |
| `MOZNLP/FLORES_MOZ` (**v**, `flores_moz`) | **seh** (Sena) | 1,012 devtest (+997 dev) | CC-BY-SA-4.0 | usable, professional, FLORES+ English (95% contaminated, 48 clean sentences). Also Xichangana `tso_Latn_MZ` (NOT `tsc` Tswa; sibling of `tso`) and Mozambican Nyanja. Card: "evaluation benchmark, do not train on it" |
| `cis-lmu/GlotStoryBook` (**v**, `glot_story`) | **kqn 420, lun 420, loz 960, kua 520, her 549, dag 135, tsc 101**, nbl 321 (second source); kri 45, ach 8, nyn 16 too small | all rows (no split) | per row CC-BY / CC-BY-NC / CC BY-NC-SA (kept in the files) | usable with caveats: African Storybook Project children's stories, volunteer translations, short simple sentences; the only open English-aligned data for kqn lun kua her. `tsc` is 101 sentences and the language was not confirmed by a native speaker (GlotLID tsc 93%, token coverage tsc 0.73 vs tso 0.63). Only 7.2% of the English is in our training windows |
| `Sunbird/salt` `text-all` test (**v**, `salt`) | **nyn, ach** (real Runyankore / Acholi), lug, ibo, swh | 500 each | CC-BY-SA-4.0, gated (auto-accept) | usable. Agricultural domain, English source, multi-way. The project READ token has access (I accepted nothing). Contamination: English 84%, SALT-Luganda pair seen for 419 of 500 rows (it is in the training data), nyn / ach pair seen 0 |
| `google/smol` (SmolSent / SmolDoc / GATITOS) | see section 4 | see section 4 | CC-BY-4.0 | integrated |
| `CLEAR-Global/TICO-19-swc-speech` (**v** counts) | **swc** | test 2,124 rows (each sentence read by two voices: about half unique), dev 868; columns english, french, swc, audio | CC-BY-NC-4.0 (underlying TICO text CC0) | exists, **not integrated**: COVID / medical text only, professional (en -> fr -> swc), the parquet files carry audio (50 MB each), TICO rules forbade training on it. Usable later through the datasets-server rows API or after dropping the audio column |
| `masakhane/mafand` `fr-bbj` (**v** 1,430 test lines) | **bbj** (Ghomala) | 1,430 test | CC-BY-NC-4.0 | exists, **not integrated**: source is FRENCH (CamerounWeb news); an English -> Ghomala test would need machine-translating the French side |
| FLORES+ (`openlanguagedata/flores_plus`, **v**: 222 devtest codes) | none of the 30 | - | - | FLORES+ has `ktu_Latn` (Kituba) but no `kon` / `kng`, and none of the 12 remaining languages. `ktu` could serve as a proxy test for `kon` (GlotLID / coverage cannot separate Kituba from our Kongo data); not added because it is a different language |
| NTREX-128, OLDI-Seed v2 (`oldi_seed`), SIB-200, Belebele, TICO-19 zip, WMT22-25 African tracks, AfriDoc-MT, Tatoeba MT test | none of the 12 remaining (checked on the file lists / cards) | - | - | nothing new: NTREX-128 African = the files already used (D-34), OLDI-Seed African = bam dik fuv nqo nus taq zgh ary arz, Tatoeba MT test = afr hau kab run swa only, AfriDoc-MT = am ha sw yo zu, WMT22 African track = 24 FLORES-101 languages none of ours |
| `michsethowusu/*_sentence-pairs_mt560` (urhobo, edo, okpe, congo-swahili, kaonde, lunda, lozi, wolaytta, tswa, sena, herero, yao, sidaama, esan, duala, basaa, cameroon-pidgin, nyankore) | the 12 remaining and more | train split only | CC-BY-4.0 | **not a test set**: OPUS MT560 / JW-style religious text with no held-out split; this is what the SFT trained on |
| Bible-based (`bible-nlp/biblenlp-corpus`, eBible, `stephanedonna/english_ghomala` KJV -> Ghomala with test 792, OPUS `bible-uedin` en-wal 15.5k, michsethowusu African Bible) | wal, seh, yao, bbj, bin, dag ... | various | various | **rejected**: the Bible is in our training data (e.g. bin 21k, dag 22k, bbj 6.4k pairs come from `african-bible-parallel`), a Bible test split is contaminated by construction |
| OPUS `translatewiki`, `wikimedia`, `QED`; Dagbani word lists (`Wuninsu/english_to_dagbani`, `abdulhafis/Dagbani_English_Dataset`, `mohammednuruddin/dagbani`), ghananlpcommunity Dagbani sets; `grairudolf/kamtok-proverb-evaluation` (14 proverbs, no references); Mozilla Basaa-French (350 lines, NOODL-1.0 licence forbids generative-AI use); `stfotso/french-ghomala-bandjoun` (dictionary) | dag ish dua bas wes bbj wal | - | - | **rejected** (UI strings, word lists, mislabelled language, licence, no references, no split) |

Not verified (be careful): AfriMTE's language list; WMT23 / WMT24 African tracks; Lanfrica and Lacuna were searched only through the Hub; the GlotStoryBook licence split per language; the correctness of the GlotStoryBook `tsc` and `dag` translations beyond the language-ID and coverage checks; row counts of SMOL files other than the ones in section 4.

## 6. Contamination of the new sets by the SFT training windows

Same scanner as D-34 (`gemma/windows_sft`, 11,442,172 windows; NFKC + casefold + punctuation-free normalisation); **exact** = the English sentence is a whole window sentence or a contiguous run inside a window; **pair** = exact and the reference occurs in the same-language target of a window (the model saw the very pair). Documents: **exact OR near** (>= 50% of the 8-token n-grams in the training English); words (GATITOS): sentence rule only (an entry has fewer than 8 tokens, the contiguous-run rule cannot apply). Flag files: `contamination_<benchmark>.json`.

| benchmark | rows (all languages) | contaminated (headline) | sentence rule | near | reference pair seen | clean rows |
|---|---|---|---|---|---|---|
| smol_sent | 38833 | 35413 (91.2%) | 35053 | 0 | 2128 (5.5%) | 3420 (8.8%) |
| smol_doc | 12499 | 7317 (58.5%) | 0 | 1086 | 186 (1.5%) | 5182 (41.5%) |
| gatitos | 139770 | 100070 (71.6%) | 100070 | 0 | 2331 (1.7%) | 39700 (28.4%) |
| ibom_mt | 2024 | 1928 (95.3%) | 1700 | 0 | 0 (0.0%) | 96 (4.7%) |
| flores_moz | 1012 | 964 (95.3%) | 850 | 0 | 0 (0.0%) | 48 (4.7%) |
| glot_story | 3426 | 247 (7.2%) | 228 | 53 | 0 (0.0%) | 3179 (92.8%) |
| salt | 2500 | 2110 (84.4%) | 2110 | 0 | 419 (16.8%) | 390 (15.6%) |

Languages whose reference pair is in the training data (> 5% of the rows) per benchmark:

| benchmark | language pair-seen / rows |
|---|---|
| smol_sent | wol 630/863, bam 564/863, ewe 524/863, yor 171/863, zul 77/863, hau 72/863, orm 58/863 |
| smol_doc | bam 186/340 |
| gatitos | wol 711/4001, twi 258/4000 |
| salt | lug 419/500 |

Reading: (1) the training data contains SMOL-derived sets for some languages (`MALIBA-AI/bambara-mt-dataset`, `galsenai/english-wolof-smol-translation`, `<HF_ORG_LEGACY>/sft-new`, `<HF_ORG>/tds` carry 779 of the 863 SmolSent English sentences), so **for bam, wol, ewe, yor, zul, hau, orm, ibo the SMOL numbers must be read on the `no_pair` subset, never on `all`**; (2) for the 18 newly covered languages the pair was never seen, but the English side was, as for FLORES+ (the model has seen the sentence with other languages); that is why the report shows `all`, `clean` (English unseen: tiny, 76 sentences / 22 documents) and `no_pair` (the reference pair unseen: the leakage that matters most) side by side; (3) the previously planned registry source `google/smol:smoldoc__en_*` (efi, ee, fon, ha, ig, pcm, yo in `DATA_INVENTORY.md`) is NOT in the final windows (no window has a `source_dataset` containing `smol` except the Wolof one), which the pair counts confirm (SmolDoc pair seen: bam only).

## 7. Results: SFT vs base

`n` = rows of the evaluated sample; `contam.` = share of its English sentences in the training windows; `pair seen` = share whose reference pair was seen; chrF++ = sacrebleu word_order 2 on NFC text, corpus level per language; **clean** = English not in the training text, **no-pair** = reference pair not seen; the number in brackets is the number of rows. Languages marked `new` had no test set before. A `*` after n means a sub-sample (run B: SmolSent first 500 rows with the clean rows first, SmolDoc 40 documents). Δ = SFT minus base. Macros are unweighted means over languages (only languages with at least 20 rows in the subset). AfriCOMET only for the 42 validated languages and only on sentence-level sets.

### 7.1 Macro averages

| benchmark | languages | n | chrF++ all (SFT / base) | chrF++ clean | chrF++ no-pair-seen | exact match (words) or spBLEU all | AfriCOMET all (n) |
|---|---|---|---|---|---|---|---|
| **smol_sent** | languages with NO test set before | 8 | 20.23 / 11.53 (8) | 15.55 / 11.68 (8) | 20.23 / 11.53 (8) | 3.34 / 0.28 | - |
| **smol_sent** | languages that already had one | 37 | 29.77 / 10.62 (37) | 24.11 / 11.01 (37) | 29.54 / 10.62 (37) | 9.41 / 0.45 | 0.554 / 0.231 (35) |
| **smol_doc** | languages with NO test set before | 8 | 24.79 / 16.51 (8) | 18.41 / 16.03 (8) | 24.79 / 16.51 (8) | 5.54 / 1.91 | - |
| **smol_doc** | languages that already had one | 37 | 30.98 / 15.13 (37) | 29.55 / 14.98 (37) | 30.98 / 15.13 (37) | 10.44 / 2.34 | - |
| **gatitos** | languages with NO test set before | 9 | 9.39 / 5.30 (9) | 7.61 / 5.56 (9) | 9.37 / 5.30 (9) | 1.7% / 0.1% ; 0.8% / 0.1% ; 1.6% / 0.1% | - |
| **gatitos** | languages that already had one | 26 | 13.92 / 5.36 (26) | 10.40 / 5.55 (26) | 13.58 / 5.36 (26) | 4.2% / 0.1% ; 2.2% / 0.2% ; 3.8% / 0.1% | - |
| **ibom_mt** | languages with NO test set before | 2 | 18.49 / 13.89 (2) | 16.64 / 13.73 (2) | 18.49 / 13.89 (2) | 3.11 / 1.82 | 0.476 / 0.242 (1) |
| **flores_moz** | languages with NO test set before | 1 | 25.59 / 13.16 (1) | 25.02 / 12.82 (1) | 25.59 / 13.16 (1) | 4.02 / 2.00 | - |
| **glot_story** | languages with NO test set before | 8 | 22.85 / 12.12 (8) | 22.87 / 12.24 (8) | 22.85 / 12.12 (8) | 4.20 / 1.60 | - |
| **salt** | languages with NO test set before | 2 | 25.61 / 10.44 (2) | 22.34 / 10.28 (2) | 25.61 / 10.44 (2) | 5.07 / 0.21 | - |
| **salt** | languages that already had one | 3 | 46.11 / 7.49 (3) | 41.49 / 7.08 (3) | 43.39 / 7.41 (3) | 24.19 / 0.11 | 0.726 / 0.256 (3) |

Reading the tables:

* **smol_sent**: smallest SFT-minus-base gaps on the clean rows (chrF++): `kon` -5.5, `bam` -0.9, `kri` +0.3, `dyu` +0.6, `ndc` +1.1, `mos` +1.3; largest: `swh` +31.0, `afr` +26.9, `ibo` +26.3, `zul` +24.4. Lowest SFT finish rate (the completion ends inside the token budget): `kon` 35.3%, `bam` 82.8%, `tir` 83.0%, `kri` 83.5%, `kik` 83.8%.
* **smol_doc**: smallest SFT-minus-base gaps on the clean rows (chrF++): `kon` -7.6, `pcm` -5.8, `dyu` -1.4, `kri` -1.2, `tiv` -0.8, `mos` -0.3; largest: `som` +34.3, `swh` +34.0, `hau` +29.8, `ibo` +25.9. Lowest SFT finish rate (the completion ends inside the token budget): `kon` 0.8%, `dyu` 25.0%, `tir` 30.0%, `kik` 40.0%, `kri` 44.4%.
* **glot_story**: smallest SFT-minus-base gaps on the clean rows (chrF++): `nbl` +4.4, `tsc` +5.2, `her` +6.9, `dag` +8.8, `lun` +11.8, `kua` +13.5; largest: `loz` +19.3, `kqn` +15.1, `kua` +13.5, `lun` +11.8. Lowest SFT finish rate (the completion ends inside the token budget): `tsc` 77.2%, `her` 81.8%, `dag` 87.4%, `kqn` 88.6%, `kua` 92.1%.
* **salt**: smallest SFT-minus-base gaps on the clean rows (chrF++): `ach` +11.6, `nyn` +12.5, `lug` +29.6, `ibo` +29.8, `swh` +43.8; largest: `swh` +43.8, `ibo` +29.8, `lug` +29.6, `nyn` +12.5. Lowest SFT finish rate (the completion ends inside the token budget): `ach` 97.4%, `nyn` 98.2%, `ibo` 98.8%, `lug` 99.8%, `swh` 100.0%.
* AfriCOMET-QE (reference-free) is not a safe ranking signal against the base model on these sets: the base's English-copy / garbage output scores 0.36 on the clean SmolSent rows against 0.34 for the SFT model although the reference-based AfriCOMET says 0.21 vs 0.45; read AfriCOMET with the reference.
* Per-language clean numbers rest on 76 sentences (SmolSent), 22-40 documents (SmolDoc), 48 sentences (IbomMT, FLORES-MOZ), 100-960 stories sentences (GlotStoryBook) and are noisy; see section 10.

### 7.2 SmolSent (sentence level, 863 sentences per language)

| tag | language |  | n | contam. | pair seen | SFT chrF++ all | SFT clean (n) | SFT no-pair (n) | base all | base clean | base no-pair | Δ all | Δ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `yor` | Yoruba |  | 500* | 84.8% | 17.6% | 18.14 | 16.03 (76) | 17.47 (412) | 5.34 | 5.64 | 5.36 | 12.80 | 10.39 |
| `hau` | Hausa |  | 500* | 84.8% | 7.2% | 36.20 | 30.25 (76) | 35.98 (464) | 8.58 | 8.21 | 8.48 | 27.61 | 22.04 |
| `ibo` | Igbo |  | 500* | 84.8% | 2.6% | 43.29 | 34.01 (76) | 43.37 (487) | 7.37 | 7.66 | 7.39 | 35.91 | 26.35 |
| `pcm` | Pidgin |  | 500* | 84.8% | 0.0% | 48.73 | 54.08 (76) | 48.73 (500) | 37.97 | 44.35 | 37.97 | 10.77 | 9.73 |
| `efi` | Efik | new | 863 | 91.2% | 0.0% | 19.60 | 15.99 (76) | 19.60 (863) | 10.05 | 10.71 | 10.05 | 9.55 | 5.27 |
| `twi` | Twi |  | 500* | 84.8% | 0.0% | 28.64 | 24.56 (76) | 28.64 (500) | 10.79 | 11.58 | 10.79 | 17.85 | 12.98 |
| `ewe` | Ewe |  | 500* | 84.8% | 57.4% | 25.99 | 21.29 (76) | 22.88 (213) | 9.78 | 9.75 | 9.57 | 16.20 | 11.55 |
| `fon` | Fon |  | 500* | 84.8% | 0.2% | 15.49 | 14.76 (76) | 15.49 (499) | 6.65 | 6.61 | 6.66 | 8.84 | 8.14 |
| `wol` | Wolof |  | 500* | 84.8% | 66.6% | 22.53 | 19.49 (76) | 20.12 (167) | 10.78 | 10.76 | 10.73 | 11.75 | 8.73 |
| `swh` | Swahili |  | 500* | 84.8% | 0.0% | 51.29 | 41.52 (76) | 51.29 (500) | 10.73 | 10.56 | 10.73 | 40.57 | 30.96 |
| `kin` | Kinyarwanda |  | 500* | 84.8% | 0.0% | 29.87 | 22.24 (76) | 29.87 (500) | 11.44 | 11.42 | 11.44 | 18.43 | 10.82 |
| `som` | Somali |  | 500* | 84.8% | 0.0% | 32.74 | 27.57 (76) | 32.74 (500) | 8.63 | 8.38 | 8.63 | 24.11 | 19.19 |
| `orm` | Oromo |  | 500* | 84.8% | 8.4% | 22.38 | 15.10 (76) | 22.06 (458) | 8.05 | 8.80 | 8.05 | 14.33 | 6.30 |
| `amh` | Amharic |  | 500* | 84.8% | 0.0% | 13.27 | 10.26 (76) | 13.27 (500) | 0.07 | 0.07 | 0.07 | 13.20 | 10.19 |
| `tir` | Tigrinya |  | 500* | 84.8% | 0.0% | 8.92 | 6.16 (76) | 8.92 (500) | 0.10 | 0.09 | 0.10 | 8.83 | 6.07 |
| `lug` | Luganda |  | 500* | 84.8% | 0.0% | 30.02 | 22.25 (76) | 30.02 (500) | 9.88 | 10.26 | 9.88 | 20.14 | 11.99 |
| `lin` | Lingala |  | 500* | 84.8% | 0.0% | 33.74 | 24.24 (76) | 33.74 (500) | 11.56 | 12.05 | 11.56 | 22.18 | 12.19 |
| `zul` | Zulu |  | 500* | 84.8% | 8.8% | 42.45 | 34.26 (76) | 42.11 (456) | 9.39 | 9.83 | 9.36 | 33.06 | 24.43 |
| `xho` | Xhosa |  | 500* | 84.8% | 0.0% | 36.83 | 30.22 (76) | 36.83 (500) | 10.90 | 11.71 | 10.90 | 25.93 | 18.51 |
| `afr` | Afrikaans |  | 500* | 84.8% | 0.0% | 55.79 | 46.49 (76) | 55.79 (500) | 21.20 | 19.55 | 21.20 | 34.60 | 26.94 |
| `sna` | Shona |  | 500* | 84.8% | 0.0% | 32.92 | 26.49 (76) | 32.92 (500) | 7.17 | 7.59 | 7.17 | 25.75 | 18.90 |
| `nya` | Chichewa |  | 500* | 84.8% | 0.0% | 31.81 | 25.46 (76) | 31.81 (500) | 10.63 | 10.88 | 10.63 | 21.18 | 14.58 |
| `sot` | Sesotho |  | 500* | 84.8% | 0.0% | 37.79 | 29.59 (76) | 37.79 (500) | 11.97 | 12.34 | 11.97 | 25.82 | 17.25 |
| `tsn` | Setswana |  | 500* | 84.8% | 0.0% | 42.80 | 36.18 (76) | 42.80 (500) | 13.17 | 13.68 | 13.17 | 29.63 | 22.50 |
| `mlg` | Malagasy |  | 500* | 84.8% | 0.0% | 36.08 | 27.52 (76) | 36.08 (500) | 10.96 | 10.64 | 10.96 | 25.12 | 16.88 |
| `bem` | Bemba |  | 500* | 84.8% | 0.0% | 22.17 | 17.23 (76) | 22.17 (500) | 10.78 | 10.70 | 10.78 | 11.39 | 6.52 |
| `tum` | Tumbuka |  | 500* | 84.8% | 0.0% | 27.82 | 20.87 (76) | 27.82 (500) | 11.87 | 12.11 | 11.87 | 15.95 | 8.76 |
| `run` | Kirundi |  | 500* | 84.8% | 0.0% | 27.12 | 21.22 (76) | 27.12 (500) | 9.02 | 8.98 | 9.02 | 18.10 | 12.24 |
| `ven` | Venda |  | 500* | 84.8% | 0.0% | 29.83 | 21.79 (76) | 29.83 (500) | 9.92 | 10.09 | 9.92 | 19.91 | 11.69 |
| `tso` | Tsonga |  | 500* | 84.8% | 0.0% | 41.36 | 29.67 (76) | 41.36 (500) | 11.76 | 12.10 | 11.76 | 29.60 | 17.56 |
| `nso` | Sepedi |  | 500* | 84.8% | 0.0% | 36.18 | 32.92 (76) | 36.18 (500) | 11.64 | 12.31 | 11.64 | 24.54 | 20.61 |
| `ssw` | Swati |  | 500* | 84.8% | 0.0% | 31.54 | 25.24 (76) | 31.54 (500) | 11.87 | 11.98 | 11.87 | 19.67 | 13.26 |
| `nbl` | South Ndebele | new | 863 | 91.2% | 0.0% | 35.83 | 29.60 (76) | 35.83 (863) | 11.22 | 11.64 | 11.22 | 24.61 | 17.97 |
| `tiv` | Tiv | new | 863 | 91.2% | 0.0% | 21.39 | 15.99 (76) | 21.39 (863) | 12.15 | 12.10 | 12.15 | 9.24 | 3.89 |
| `mos` | Mossi |  | 500* | 84.8% | 0.0% | 13.18 | 11.45 (76) | 13.18 (500) | 9.44 | 10.14 | 9.44 | 3.75 | 1.31 |
| `kik` | Kikuyu |  | 500* | 84.8% | 0.0% | 23.78 | 16.72 (76) | 23.78 (500) | 10.27 | 10.74 | 10.27 | 13.52 | 5.98 |
| `luo` | Luo |  | 500* | 84.8% | 0.0% | 20.27 | 15.27 (76) | 20.27 (500) | 11.40 | 11.70 | 11.40 | 8.87 | 3.57 |
| `dyu` | Dyula |  | 500* | 84.8% | 0.0% | 14.71 | 11.35 (76) | 14.71 (500) | 10.13 | 10.74 | 10.13 | 4.57 | 0.60 |
| `bam` | Bambara |  | 500* | 84.8% | 60.4% | 14.80 | 10.73 (76) | 13.27 (198) | 10.60 | 11.67 | 11.18 | 4.20 | -0.94 |
| `kon` | Kongo | new | 863 | 91.2% | 0.0% | 6.11 | 5.00 (76) | 6.11 (863) | 10.94 | 10.46 | 10.94 | -4.83 | -5.46 |
| `kab` | Kabyle |  | 500* | 84.8% | 0.0% | 20.85 | 17.62 (76) | 20.85 (500) | 11.08 | 11.73 | 11.08 | 9.77 | 5.88 |
| `ach` | Acholi | new | 863 | 91.2% | 0.0% | 21.39 | 15.24 (76) | 21.39 (863) | 11.93 | 12.43 | 11.93 | 9.46 | 2.80 |
| `nyn` | Nyankore | new | 863 | 91.2% | 0.0% | 20.91 | 16.11 (76) | 20.91 (863) | 10.87 | 11.04 | 10.87 | 10.04 | 5.07 |
| `ndc` | Ndau | new | 863 | 91.2% | 0.0% | 18.04 | 13.11 (76) | 18.04 (863) | 11.98 | 11.99 | 11.98 | 6.06 | 1.12 |
| `kri` | Krio | new | 863 | 91.2% | 0.0% | 18.59 | 13.38 (76) | 18.59 (863) | 13.09 | 13.04 | 13.09 | 5.50 | 0.34 |

Other metrics, finish and format rates (SFT; the base model never writes the tag, so its `prefix ok` is 0 by construction and its whole output is scored):

| tag | SFT spBLEU all | SFT BLEU all | AfriCOMET all (SFT / base) | QE (SFT) | finished (SFT / base) | prefix ok (SFT) | empty (SFT) |
|---|---|---|---|---|---|---|---|
| `yor` | 5.07 | 2.19 | 0.577 / 0.181 | 0.490 | 95.8% / 90.4% | 100.0% | 0.0% |
| `hau` | 11.03 | 10.49 | 0.570 / 0.230 | 0.548 | 98.0% / 90.2% | 100.0% | 0.0% |
| `ibo` | 23.69 | 20.21 | 0.644 / 0.205 | 0.525 | 96.6% / 73.6% | 100.0% | 0.0% |
| `pcm` | 24.32 | 20.42 | 0.564 / 0.476 | 0.384 | 98.8% / 100.0% | 100.0% | 0.0% |
| `efi` | 3.05 | 1.28 | - | - | 91.3% / 90.2% | 100.0% | 0.0% |
| `twi` | 7.37 | 6.17 | 0.605 / 0.205 | 0.510 | 98.2% / 100.0% | 99.8% | 0.2% |
| `ewe` | 6.28 | 5.15 | 0.563 / 0.203 | 0.470 | 97.6% / 100.0% | 99.8% | 0.0% |
| `fon` | 1.70 | 1.00 | 0.557 / 0.188 | 0.462 | 98.6% / 96.2% | 99.8% | 0.0% |
| `wol` | 5.80 | 4.05 | 0.601 / 0.247 | 0.488 | 97.0% / 100.0% | 100.0% | 0.0% |
| `swh` | 29.48 | 25.64 | 0.627 / 0.277 | 0.598 | 100.0% / 86.2% | 100.0% | 0.0% |
| `kin` | 8.30 | 5.39 | 0.573 / 0.238 | 0.519 | 95.2% / 96.4% | 100.0% | 0.0% |
| `som` | 8.91 | 5.69 | 0.539 / 0.208 | 0.514 | 99.4% / 83.8% | 99.8% | 0.0% |
| `orm` | 2.82 | 1.01 | 0.510 / 0.179 | 0.463 | 84.2% / 69.0% | 100.0% | 0.0% |
| `amh` | 3.43 | 1.15 | 0.368 / 0.180 | 0.388 | 94.2% / 80.0% | 100.0% | 0.0% |
| `tir` | 1.33 | 0.41 | 0.461 / 0.236 | 0.397 | 83.0% / 98.2% | 100.0% | 0.0% |
| `lug` | 7.76 | 4.26 | 0.571 / 0.249 | 0.520 | 96.8% / 87.8% | 100.0% | 0.0% |
| `lin` | 6.20 | 5.68 | 0.519 / 0.245 | 0.450 | 94.8% / 93.6% | 100.0% | 0.0% |
| `zul` | 17.73 | 10.83 | 0.603 / 0.235 | 0.567 | 99.6% / 79.4% | 100.0% | 0.0% |
| `xho` | 11.89 | 5.76 | 0.568 / 0.242 | 0.542 | 98.8% / 93.4% | 100.0% | 0.0% |
| `afr` | 34.88 | 30.33 | 0.675 / 0.371 | 0.644 | 100.0% / 100.0% | 100.0% | 0.0% |
| `sna` | 8.20 | 4.87 | 0.558 / 0.193 | 0.512 | 98.0% / 57.4% | 100.0% | 0.0% |
| `nya` | 7.31 | 4.84 | 0.562 / 0.235 | 0.543 | 97.6% / 84.8% | 99.8% | 0.0% |
| `sot` | 14.44 | 13.22 | 0.588 / 0.223 | 0.544 | 98.8% / 90.0% | 99.8% | 0.0% |
| `tsn` | 19.51 | 17.57 | 0.646 / 0.247 | 0.590 | 98.2% / 100.0% | 100.0% | 0.0% |
| `mlg` | 11.21 | 7.18 | 0.486 / 0.248 | 0.482 | 97.4% / 95.6% | 100.0% | 0.0% |
| `bem` | 2.05 | 1.06 | 0.469 / 0.245 | 0.429 | 91.6% / 95.4% | 100.0% | 0.0% |
| `tum` | 5.91 | 3.95 | 0.529 / 0.230 | 0.491 | 94.4% / 100.0% | 100.0% | 0.0% |
| `run` | 4.24 | 2.30 | 0.577 / 0.220 | 0.530 | 97.0% / 78.6% | 100.0% | 0.0% |
| `ven` | 8.44 | 5.52 | - | - | 92.2% / 100.0% | 100.0% | 0.0% |
| `tso` | 15.71 | 14.67 | 0.600 / 0.223 | 0.522 | 97.0% / 100.0% | 100.0% | 0.0% |
| `nso` | 13.16 | 12.02 | 0.582 / 0.227 | 0.563 | 98.2% / 100.0% | 100.0% | 0.0% |
| `ssw` | 6.94 | 3.11 | 0.551 / 0.233 | 0.515 | 97.6% / 100.0% | 100.0% | 0.0% |
| `nbl` | 12.32 | 4.79 | - | - | 99.1% / 89.6% | 100.0% | 0.1% |
| `tiv` | 3.46 | 2.23 | - | - | 88.3% / 99.9% | 99.9% | 0.0% |
| `mos` | 0.48 | 0.38 | 0.459 / 0.170 | 0.473 | 94.6% / 100.0% | 100.0% | 0.0% |
| `kik` | 3.88 | 2.94 | 0.556 / 0.211 | 0.426 | 83.8% / 95.4% | 100.0% | 0.0% |
| `luo` | 1.98 | 1.32 | 0.557 / 0.232 | 0.503 | 92.8% / 99.4% | 100.0% | 0.0% |
| `dyu` | 0.81 | 0.67 | - | - | 88.8% / 100.0% | 100.0% | 0.0% |
| `bam` | 1.71 | 1.60 | 0.348 / 0.199 | 0.256 | 82.8% / 99.8% | 100.0% | 0.0% |
| `kon` | 0.09 | 0.03 | - | - | 35.3% / 85.3% | 100.0% | 0.0% |
| `kab` | 4.31 | 2.32 | 0.630 / 0.161 | 0.395 | 96.4% / 99.4% | 100.0% | 0.0% |
| `ach` | 2.29 | 1.60 | - | - | 92.1% / 99.8% | 100.0% | 0.0% |
| `nyn` | 1.52 | 0.60 | - | - | 93.4% / 97.9% | 100.0% | 0.0% |
| `ndc` | 0.77 | 0.41 | - | - | 90.7% / 99.7% | 97.3% | 0.0% |
| `kri` | 3.23 | 2.32 | - | - | 83.5% / 100.0% | 100.0% | 0.0% |

### 7.3 SmolDoc (document level)

**Domain caveat**: the 130-document languages (all `new` rows and several others) are evaluated on the multi-way parallel SmolDoc core documents (stories, dialogues, explanatory texts), of which 83% are in the training windows (22 clean); the languages with 200-670 documents also have extra documents (the card names MediSMOL medical text, about 80 per language; I could not tell what the rest are) that are NOT in the training data, and the clean-first sample of 40 documents is drawn from those. So the SmolDoc chrF++ of a high-resource language (e.g. yor, hau, swh in the 40-document rows) is on different documents from the one of a 130-document language: compare checkpoints on the same row, not languages with each other.

| tag | language |  | n | contam. | pair seen | SFT chrF++ all | SFT clean (n) | SFT no-pair (n) | base all | base clean | base no-pair | Δ all | Δ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `yor` | Yoruba |  | 40* | 0.0% | 0.0% | 24.33 | 24.33 (40) | 24.33 (40) | 8.04 | 8.04 | 8.04 | 16.29 | 16.29 |
| `hau` | Hausa |  | 40* | 0.0% | 0.0% | 39.05 | 39.05 (40) | 39.05 (40) | 9.26 | 9.26 | 9.26 | 29.79 | 29.79 |
| `ibo` | Igbo |  | 40* | 0.0% | 0.0% | 39.55 | 39.55 (40) | 39.55 (40) | 13.67 | 13.67 | 13.67 | 25.88 | 25.88 |
| `pcm` | Pidgin |  | 40* | 0.0% | 0.0% | 41.37 | 41.37 (40) | 41.37 (40) | 47.19 | 47.19 | 47.19 | -5.83 | -5.83 |
| `efi` | Efik | new | 130 | 83.1% | 0.0% | 26.21 | 17.13 (22) | 26.21 (130) | 15.84 | 13.74 | 15.84 | 10.38 | 3.40 |
| `twi` | Twi |  | 40* | 0.0% | 0.0% | 26.58 | 26.58 (40) | 26.58 (40) | 15.65 | 15.65 | 15.65 | 10.93 | 10.93 |
| `ewe` | Ewe |  | 40* | 0.0% | 0.0% | 20.02 | 20.02 (40) | 20.02 (40) | 15.44 | 15.44 | 15.44 | 4.59 | 4.59 |
| `fon` | Fon |  | 40* | 45.0% | 0.0% | 21.09 | 17.59 (22) | 21.09 (40) | 9.65 | 9.03 | 9.65 | 11.45 | 8.55 |
| `wol` | Wolof |  | 40* | 0.0% | 0.0% | 21.86 | 21.86 (40) | 21.86 (40) | 15.41 | 15.41 | 15.41 | 6.44 | 6.44 |
| `swh` | Swahili |  | 40* | 0.0% | 0.0% | 49.08 | 49.08 (40) | 49.08 (40) | 15.09 | 15.09 | 15.09 | 33.99 | 33.99 |
| `kin` | Kinyarwanda |  | 40* | 0.0% | 0.0% | 33.22 | 33.22 (40) | 33.22 (40) | 15.35 | 15.35 | 15.35 | 17.87 | 17.87 |
| `som` | Somali |  | 40* | 0.0% | 0.0% | 42.75 | 42.75 (40) | 42.75 (40) | 8.42 | 8.42 | 8.42 | 34.33 | 34.33 |
| `orm` | Oromo |  | 40* | 0.0% | 0.0% | 22.99 | 22.99 (40) | 22.99 (40) | 10.66 | 10.66 | 10.66 | 12.33 | 12.33 |
| `amh` | Amharic |  | 40* | 0.0% | 0.0% | 15.24 | 15.24 (40) | 15.24 (40) | 0.49 | 0.49 | 0.49 | 14.76 | 14.76 |
| `tir` | Tigrinya |  | 40* | 0.0% | 0.0% | 7.26 | 7.26 (40) | 7.26 (40) | 0.57 | 0.57 | 0.57 | 6.69 | 6.69 |
| `lug` | Luganda |  | 40* | 0.0% | 0.0% | 30.42 | 30.42 (40) | 30.42 (40) | 14.64 | 14.64 | 14.64 | 15.78 | 15.78 |
| `lin` | Lingala |  | 40* | 0.0% | 0.0% | 34.04 | 34.04 (40) | 34.04 (40) | 15.74 | 15.74 | 15.74 | 18.30 | 18.30 |
| `zul` | Zulu |  | 40* | 0.0% | 0.0% | 41.12 | 41.12 (40) | 41.12 (40) | 16.11 | 16.11 | 16.11 | 25.01 | 25.01 |
| `xho` | Xhosa |  | 40* | 0.0% | 0.0% | 38.90 | 38.90 (40) | 38.90 (40) | 15.80 | 15.80 | 15.80 | 23.10 | 23.10 |
| `afr` | Afrikaans |  | 40* | 0.0% | 0.0% | 47.26 | 47.26 (40) | 47.26 (40) | 28.71 | 28.71 | 28.71 | 18.55 | 18.55 |
| `sna` | Shona |  | 40* | 0.0% | 0.0% | 36.78 | 36.78 (40) | 36.78 (40) | 15.40 | 15.40 | 15.40 | 21.38 | 21.38 |
| `nya` | Chichewa |  | 40* | 0.0% | 0.0% | 37.87 | 37.87 (40) | 37.87 (40) | 16.66 | 16.66 | 16.66 | 21.21 | 21.21 |
| `sot` | Sesotho |  | 40* | 0.0% | 0.0% | 36.87 | 36.87 (40) | 36.87 (40) | 16.71 | 16.71 | 16.71 | 20.16 | 20.16 |
| `tsn` | Setswana |  | 40* | 45.0% | 0.0% | 46.68 | 39.98 (22) | 46.68 (40) | 17.78 | 18.21 | 17.78 | 28.90 | 21.77 |
| `mlg` | Malagasy |  | 40* | 0.0% | 0.0% | 37.37 | 37.37 (40) | 37.37 (40) | 12.84 | 12.84 | 12.84 | 24.53 | 24.53 |
| `bem` | Bemba |  | 40* | 45.0% | 0.0% | 29.06 | 24.57 (22) | 29.06 (40) | 16.67 | 16.40 | 16.67 | 12.39 | 8.17 |
| `tum` | Tumbuka |  | 40* | 45.0% | 0.0% | 31.12 | 24.46 (22) | 31.12 (40) | 17.22 | 16.61 | 17.22 | 13.89 | 7.85 |
| `run` | Kirundi |  | 40* | 45.0% | 0.0% | 30.54 | 27.01 (22) | 30.54 (40) | 14.64 | 13.97 | 14.64 | 15.90 | 13.04 |
| `ven` | Venda |  | 40* | 45.0% | 0.0% | 32.59 | 25.08 (22) | 32.59 (40) | 16.00 | 14.90 | 16.00 | 16.59 | 10.18 |
| `tso` | Tsonga |  | 40* | 45.0% | 0.0% | 42.10 | 36.25 (22) | 42.10 (40) | 16.58 | 15.80 | 16.58 | 25.52 | 20.46 |
| `nso` | Sepedi |  | 40* | 0.0% | 0.0% | 31.44 | 31.44 (40) | 31.44 (40) | 16.86 | 16.86 | 16.86 | 14.58 | 14.58 |
| `ssw` | Swati |  | 40* | 45.0% | 0.0% | 35.54 | 30.30 (22) | 35.54 (40) | 17.53 | 17.68 | 17.53 | 18.01 | 12.61 |
| `nbl` | South Ndebele | new | 130 | 83.1% | 0.0% | 40.56 | 33.20 (22) | 40.56 (130) | 16.98 | 16.42 | 16.98 | 23.58 | 16.78 |
| `tiv` | Tiv | new | 130 | 83.1% | 0.0% | 24.10 | 15.58 (22) | 24.10 (130) | 17.17 | 16.38 | 17.17 | 6.94 | -0.80 |
| `mos` | Mossi |  | 40* | 45.0% | 0.0% | 15.18 | 14.45 (22) | 15.18 (40) | 15.05 | 14.70 | 15.05 | 0.14 | -0.25 |
| `kik` | Kikuyu |  | 40* | 45.0% | 0.0% | 24.48 | 18.91 (22) | 24.48 (40) | 14.91 | 13.78 | 14.91 | 9.57 | 5.13 |
| `luo` | Luo |  | 40* | 0.0% | 0.0% | 27.81 | 27.81 (40) | 27.81 (40) | 12.77 | 12.77 | 12.77 | 15.03 | 15.03 |
| `dyu` | Dyula |  | 40* | 45.0% | 0.0% | 15.77 | 12.59 (22) | 15.77 (40) | 14.80 | 13.99 | 14.80 | 0.97 | -1.39 |
| `bam` | Bambara |  | 40* | 0.0% | 0.0% | 20.32 | 20.32 (40) | 20.32 (40) | 14.41 | 14.41 | 14.41 | 5.90 | 5.90 |
| `kon` | Kongo | new | 130 | 83.1% | 0.0% | 7.76 | 7.72 (22) | 7.76 (130) | 14.70 | 15.31 | 14.70 | -6.94 | -7.59 |
| `kab` | Kabyle |  | 40* | 0.0% | 0.0% | 18.59 | 18.59 (40) | 18.59 (40) | 17.23 | 17.23 | 17.23 | 1.36 | 1.36 |
| `ach` | Acholi | new | 130 | 83.1% | 0.0% | 27.76 | 19.03 (22) | 27.76 (130) | 18.21 | 18.16 | 18.21 | 9.55 | 0.86 |
| `nyn` | Nyankore | new | 130 | 83.1% | 0.0% | 28.18 | 20.40 (22) | 28.18 (130) | 15.55 | 14.55 | 15.55 | 12.62 | 5.85 |
| `ndc` | Ndau | new | 130 | 83.1% | 0.0% | 22.53 | 17.99 (22) | 22.53 (130) | 16.17 | 16.28 | 16.17 | 6.37 | 1.71 |
| `kri` | Krio | new | 207 | 52.2% | 0.0% | 21.18 | 16.19 (99) | 21.18 (207) | 17.47 | 17.37 | 17.47 | 3.72 | -1.18 |

Other metrics, finish and format rates (SFT; the base model never writes the tag, so its `prefix ok` is 0 by construction and its whole output is scored):

| tag | SFT spBLEU all | SFT BLEU all | AfriCOMET all (SFT / base) | QE (SFT) | finished (SFT / base) | prefix ok (SFT) | empty (SFT) |
|---|---|---|---|---|---|---|---|
| `yor` | 10.37 | 7.88 | - | - | 77.5% / 90.0% | 100.0% | 0.0% |
| `hau` | 13.13 | 12.69 | - | - | 75.0% / 90.0% | 100.0% | 0.0% |
| `ibo` | 21.77 | 18.96 | - | - | 75.0% / 90.0% | 100.0% | 0.0% |
| `pcm` | 22.41 | 20.66 | - | - | 90.0% / 100.0% | 100.0% | 0.0% |
| `efi` | 7.74 | 4.40 | - | - | 55.4% / 99.2% | 99.2% | 0.0% |
| `twi` | 6.42 | 4.90 | - | - | 45.0% / 97.5% | 100.0% | 0.0% |
| `ewe` | 4.58 | 3.94 | - | - | 75.0% / 100.0% | 100.0% | 0.0% |
| `fon` | 4.28 | 3.08 | - | - | 77.5% / 95.0% | 100.0% | 0.0% |
| `wol` | 6.08 | 5.18 | - | - | 57.5% / 95.0% | 100.0% | 0.0% |
| `swh` | 25.45 | 21.80 | - | - | 85.0% / 87.5% | 100.0% | 0.0% |
| `kin` | 9.69 | 7.60 | - | - | 65.0% / 97.5% | 100.0% | 0.0% |
| `som` | 17.34 | 13.63 | - | - | 75.0% / 87.5% | 100.0% | 0.0% |
| `orm` | 4.09 | 2.40 | - | - | 47.5% / 92.5% | 97.5% | 0.0% |
| `amh` | 3.82 | 1.00 | - | - | 67.5% / 70.0% | 100.0% | 0.0% |
| `tir` | 1.25 | 0.37 | - | - | 30.0% / 92.5% | 100.0% | 0.0% |
| `lug` | 8.44 | 4.70 | - | - | 57.5% / 100.0% | 100.0% | 0.0% |
| `lin` | 8.52 | 8.11 | - | - | 57.5% / 100.0% | 100.0% | 0.0% |
| `zul` | 13.68 | 8.46 | - | - | 67.5% / 92.5% | 100.0% | 0.0% |
| `xho` | 12.01 | 7.76 | - | - | 65.0% / 97.5% | 100.0% | 0.0% |
| `afr` | 29.04 | 26.93 | - | - | 72.5% / 100.0% | 100.0% | 0.0% |
| `sna` | 11.27 | 7.49 | - | - | 70.0% / 95.0% | 100.0% | 0.0% |
| `nya` | 9.45 | 7.90 | - | - | 65.0% / 95.0% | 100.0% | 0.0% |
| `sot` | 13.66 | 13.88 | - | - | 72.5% / 100.0% | 100.0% | 0.0% |
| `tsn` | 24.18 | 22.75 | - | - | 72.5% / 100.0% | 100.0% | 0.0% |
| `mlg` | 11.77 | 8.53 | - | - | 65.0% / 95.0% | 100.0% | 0.0% |
| `bem` | 5.30 | 3.46 | - | - | 70.0% / 97.5% | 100.0% | 0.0% |
| `tum` | 7.62 | 6.54 | - | - | 60.0% / 100.0% | 100.0% | 0.0% |
| `run` | 5.83 | 4.64 | - | - | 62.5% / 90.0% | 100.0% | 0.0% |
| `ven` | 12.37 | 9.43 | - | - | 62.5% / 97.5% | 97.5% | 0.0% |
| `tso` | 14.62 | 14.43 | - | - | 62.5% / 100.0% | 100.0% | 0.0% |
| `nso` | 13.50 | 13.19 | - | - | 60.0% / 100.0% | 100.0% | 0.0% |
| `ssw` | 8.61 | 4.74 | - | - | 62.5% / 100.0% | 97.5% | 0.0% |
| `nbl` | 13.29 | 6.09 | - | - | 83.8% / 95.4% | 100.0% | 0.0% |
| `tiv` | 5.10 | 3.82 | - | - | 52.3% / 100.0% | 100.0% | 0.0% |
| `mos` | 2.04 | 1.88 | - | - | 47.5% / 100.0% | 100.0% | 0.0% |
| `kik` | 5.15 | 4.15 | - | - | 40.0% / 95.0% | 100.0% | 0.0% |
| `luo` | 6.54 | 5.03 | - | - | 67.5% / 100.0% | 100.0% | 0.0% |
| `dyu` | 2.13 | 1.72 | - | - | 25.0% / 100.0% | 100.0% | 0.0% |
| `bam` | 5.75 | 5.46 | - | - | 45.0% / 100.0% | 100.0% | 0.0% |
| `kon` | 0.25 | 0.19 | - | - | 0.8% / 98.5% | 100.0% | 0.0% |
| `kab` | 4.10 | 2.48 | - | - | 50.0% / 100.0% | 100.0% | 0.0% |
| `ach` | 5.08 | 4.00 | - | - | 60.0% / 99.2% | 100.0% | 0.0% |
| `nyn` | 3.61 | 2.73 | - | - | 68.5% / 99.2% | 100.0% | 0.0% |
| `ndc` | 2.04 | 1.26 | - | - | 56.9% / 100.0% | 100.0% | 0.0% |
| `kri` | 7.20 | 6.05 | - | - | 44.4% / 100.0% | 100.0% | 0.0% |

### 7.4 GATITOS (word level)

| tag | language |  | n | contam. | pair seen | SFT chrF++ all | SFT clean (n) | SFT no-pair (n) | base all | base clean | base no-pair | SFT exact all | exact clean | exact no-pair | base exact all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pcm` | Pidgin |  | 4001 | 71.6% | 2.3% | 26.27 | 22.84 (1136) | 26.08 (3909) | 13.81 | 13.82 | 13.76 | 10.0% | 10.5% | 9.6% | 1.7% |
| `efi` | Efik | new | 4001 | 71.6% | 0.7% | 7.18 | 6.19 (1136) | 7.14 (3973) | 4.57 | 4.64 | 4.57 | 0.1% | 0.1% | 0.2% | 0.0% |
| `twi` | Twi |  | 4000 | 71.6% | 6.5% | 17.23 | 11.31 (1135) | 15.69 (3742) | 4.61 | 4.90 | 4.65 | 10.3% | 5.3% | 7.9% | 0.3% |
| `ewe` | Ewe |  | 3990 | 71.6% | 1.0% | 10.48 | 7.87 (1135) | 10.12 (3951) | 3.70 | 4.04 | 3.67 | 2.6% | 1.4% | 2.6% | 0.2% |
| `fon` | Fon |  | 3998 | 71.6% | 4.8% | 10.26 | 7.55 (1135) | 9.55 (3808) | 2.90 | 3.19 | 2.91 | 1.2% | 0.6% | 0.9% | 0.0% |
| `wol` | Wolof |  | 4001 | 71.6% | 17.8% | 14.20 | 9.57 (1136) | 12.12 (3290) | 4.10 | 4.13 | 4.05 | 7.2% | 4.1% | 6.1% | 0.0% |
| `orm` | Oromo |  | 3995 | 71.6% | 1.3% | 17.10 | 12.00 (1135) | 16.93 (3945) | 5.77 | 6.14 | 5.78 | 5.4% | 1.6% | 5.1% | 0.0% |
| `tir` | Tigrinya |  | 4000 | 71.6% | 0.4% | 5.15 | 2.68 (1135) | 5.11 (3986) | 0.01 | 0.00 | 0.01 | 0.9% | 0.0% | 0.9% | 0.0% |
| `lug` | Luganda |  | 3998 | 71.6% | 1.0% | 18.29 | 13.11 (1135) | 18.09 (3960) | 5.64 | 5.90 | 5.64 | 9.4% | 4.6% | 9.0% | 0.0% |
| `lin` | Lingala |  | 3994 | 71.7% | 0.8% | 19.21 | 14.47 (1132) | 19.03 (3964) | 6.07 | 6.27 | 6.08 | 6.8% | 3.6% | 6.5% | 0.0% |
| `tsn` | Setswana |  | 3995 | 71.6% | 3.3% | 24.55 | 18.31 (1136) | 23.80 (3864) | 6.26 | 6.47 | 6.28 | 7.5% | 2.3% | 6.6% | 0.0% |
| `bem` | Bemba |  | 3977 | 71.6% | 0.8% | 6.54 | 3.59 (1128) | 6.47 (3945) | 5.75 | 5.71 | 5.75 | 1.9% | 0.8% | 1.8% | 0.1% |
| `tum` | Tumbuka |  | 3983 | 71.7% | 0.5% | 10.27 | 9.01 (1129) | 10.24 (3963) | 6.43 | 6.78 | 6.43 | 1.6% | 1.1% | 1.6% | 0.1% |
| `run` | Kirundi |  | 3999 | 71.6% | 0.6% | 14.17 | 11.07 (1136) | 14.07 (3976) | 5.47 | 5.76 | 5.47 | 5.1% | 2.9% | 4.9% | 0.1% |
| `ven` | Venda |  | 3984 | 71.6% | 0.6% | 15.88 | 11.17 (1133) | 15.78 (3959) | 5.51 | 5.66 | 5.51 | 3.3% | 2.2% | 3.2% | 0.1% |
| `tso` | Tsonga |  | 3994 | 71.6% | 2.2% | 22.65 | 14.89 (1135) | 22.28 (3908) | 5.72 | 5.96 | 5.72 | 7.1% | 3.2% | 6.6% | 0.2% |
| `nso` | Sepedi |  | 3996 | 71.6% | 2.3% | 22.43 | 16.02 (1135) | 22.05 (3906) | 5.92 | 6.17 | 5.93 | 9.2% | 4.1% | 8.6% | 0.1% |
| `ssw` | Swati |  | 3924 | 71.7% | 0.5% | 14.35 | 9.32 (1109) | 14.23 (3906) | 5.92 | 6.14 | 5.93 | 5.4% | 1.8% | 5.1% | 0.0% |
| `nbl` | South Ndebele | new | 4001 | 71.6% | 1.3% | 11.50 | 7.44 (1136) | 11.48 (3947) | 5.87 | 6.06 | 5.87 | 3.5% | 0.9% | 3.5% | 0.0% |
| `sag` | Sango |  | 3994 | 71.6% | 0.6% | 16.89 | 13.81 (1135) | 16.84 (3971) | 7.07 | 7.33 | 7.07 | 2.1% | 1.3% | 2.0% | 0.0% |
| `tiv` | Tiv | new | 4001 | 71.6% | 0.4% | 12.06 | 9.89 (1136) | 12.02 (3983) | 5.22 | 5.45 | 5.23 | 1.4% | 1.1% | 1.3% | 0.1% |
| `mos` | Mossi |  | 4000 | 71.6% | 0.1% | 5.08 | 4.48 (1136) | 5.07 (3997) | 4.40 | 4.32 | 4.40 | 0.0% | 0.0% | 0.0% | 0.0% |
| `kik` | Kikuyu |  | 3994 | 71.6% | 0.4% | 12.61 | 10.63 (1133) | 12.59 (3980) | 5.12 | 5.26 | 5.13 | 1.7% | 1.1% | 1.6% | 0.0% |
| `luo` | Luo |  | 3941 | 71.7% | 1.9% | 11.76 | 8.79 (1116) | 11.58 (3867) | 5.61 | 5.91 | 5.62 | 2.9% | 1.8% | 2.6% | 0.0% |
| `dyu` | Dyula |  | 3997 | 71.6% | 0.1% | 7.59 | 6.94 (1135) | 7.58 (3993) | 4.98 | 5.11 | 4.98 | 0.1% | 0.0% | 0.1% | 0.0% |
| `bam` | Bambara |  | 4000 | 71.6% | 1.1% | 7.63 | 6.32 (1135) | 7.32 (3957) | 4.43 | 4.48 | 4.39 | 1.4% | 1.1% | 1.4% | 0.0% |
| `kon` | Kongo | new | 3998 | 71.6% | 0.1% | 3.79 | 3.34 (1136) | 3.78 (3994) | 6.02 | 6.12 | 6.02 | 0.9% | 0.9% | 0.9% | 0.1% |
| `kab` | Kabyle |  | 4001 | 71.6% | 3.2% | 10.82 | 6.69 (1136) | 10.04 (3874) | 4.63 | 4.58 | 4.64 | 3.7% | 0.9% | 3.1% | 0.0% |
| `nus` | Nuer |  | 4001 | 71.6% | 0.6% | 6.04 | 5.80 (1136) | 5.99 (3977) | 3.85 | 4.18 | 3.85 | 0.2% | 0.1% | 0.2% | 0.1% |
| `lua` | Luba-Kasai |  | 3998 | 71.6% | 0.6% | 14.58 | 12.05 (1136) | 14.53 (3976) | 5.80 | 5.97 | 5.81 | 2.2% | 0.5% | 1.9% | 0.0% |
| `iso` | Isoko | new | 4001 | 71.6% | 0.5% | 10.51 | 7.52 (1136) | 10.42 (3982) | 4.18 | 4.30 | 4.18 | 1.8% | 0.2% | 1.7% | 0.0% |
| `ach` | Acholi | new | 4001 | 71.6% | 0.3% | 8.99 | 6.94 (1136) | 8.95 (3990) | 5.14 | 5.46 | 5.14 | 1.7% | 1.5% | 1.6% | 0.0% |
| `nyn` | Nyankore | new | 4001 | 71.6% | 0.1% | 10.65 | 8.89 (1136) | 10.64 (3997) | 6.01 | 6.36 | 6.01 | 1.9% | 1.2% | 1.9% | 0.0% |
| `ndc` | Ndau | new | 3991 | 71.6% | 0.1% | 5.32 | 3.80 (1135) | 5.31 (3988) | 5.63 | 5.89 | 5.64 | 0.7% | 0.1% | 0.7% | 0.0% |
| `kri` | Krio | new | 4020 | 71.1% | 0.2% | 14.54 | 14.45 (1160) | 14.54 (4011) | 5.06 | 5.78 | 5.06 | 2.8% | 0.9% | 2.8% | 0.2% |

Other metrics, finish and format rates (SFT; the base model never writes the tag, so its `prefix ok` is 0 by construction and its whole output is scored):

| tag | SFT spBLEU all | SFT BLEU all | AfriCOMET all (SFT / base) | QE (SFT) | finished (SFT / base) | prefix ok (SFT) | empty (SFT) |
|---|---|---|---|---|---|---|---|
| `pcm` | - | - | - | - | 92.3% / 99.3% | 100.0% | 0.2% |
| `efi` | - | - | - | - | 95.4% / 85.5% | 100.0% | 14.8% |
| `twi` | - | - | - | - | 99.7% / 99.5% | 100.0% | 5.8% |
| `ewe` | - | - | - | - | 96.4% / 99.8% | 100.0% | 3.7% |
| `fon` | - | - | - | - | 97.0% / 99.0% | 100.0% | 33.6% |
| `wol` | - | - | - | - | 99.6% / 96.5% | 99.9% | 3.0% |
| `orm` | - | - | - | - | 94.3% / 97.4% | 100.0% | 3.2% |
| `tir` | - | - | - | - | 79.2% / 98.2% | 100.0% | 11.4% |
| `lug` | - | - | - | - | 99.8% / 97.0% | 100.0% | 15.3% |
| `lin` | - | - | - | - | 98.4% / 98.4% | 100.0% | 12.0% |
| `tsn` | - | - | - | - | 98.1% / 99.3% | 100.0% | 12.1% |
| `bem` | - | - | - | - | 99.6% / 98.4% | 100.0% | 75.2% |
| `tum` | - | - | - | - | 96.7% / 99.8% | 100.0% | 3.0% |
| `run` | - | - | - | - | 99.1% / 97.0% | 100.0% | 4.5% |
| `ven` | - | - | - | - | 94.8% / 99.3% | 100.0% | 8.9% |
| `tso` | - | - | - | - | 96.7% / 98.9% | 100.0% | 1.1% |
| `nso` | - | - | - | - | 95.5% / 99.7% | 100.0% | 8.3% |
| `ssw` | - | - | - | - | 99.1% / 99.8% | 100.0% | 53.8% |
| `nbl` | - | - | - | - | 99.7% / 95.9% | 100.0% | 63.5% |
| `sag` | - | - | - | - | 95.6% / 99.5% | 100.0% | 7.0% |
| `tiv` | - | - | - | - | 98.0% / 99.8% | 100.0% | 21.2% |
| `mos` | - | - | - | - | 79.8% / 99.9% | 100.0% | 10.8% |
| `kik` | - | - | - | - | 90.6% / 98.8% | 100.0% | 3.7% |
| `luo` | - | - | - | - | 99.2% / 99.4% | 100.0% | 24.6% |
| `dyu` | - | - | - | - | 92.9% / 99.5% | 100.0% | 9.0% |
| `bam` | - | - | - | - | 97.4% / 99.4% | 100.0% | 28.2% |
| `kon` | - | - | - | - | 92.0% / 97.3% | 100.0% | 54.9% |
| `kab` | - | - | - | - | 99.9% / 99.4% | 100.0% | 39.9% |
| `nus` | - | - | - | - | 95.2% / 99.2% | 100.0% | 7.3% |
| `lua` | - | - | - | - | 98.2% / 99.9% | 100.0% | 16.3% |
| `iso` | - | - | - | - | 97.2% / 99.3% | 100.0% | 0.3% |
| `ach` | - | - | - | - | 99.3% / 99.8% | 100.0% | 43.4% |
| `nyn` | - | - | - | - | 98.4% / 98.2% | 100.0% | 20.9% |
| `ndc` | - | - | - | - | 97.2% / 99.8% | 100.0% | 61.6% |
| `kri` | - | - | - | - | 93.9% / 99.9% | 100.0% | 0.0% |

### 7.5 IbomMT (Ibibio, Efik)

| tag | language |  | n | contam. | pair seen | SFT chrF++ all | SFT clean (n) | SFT no-pair (n) | base all | base clean | base no-pair | Δ all | Δ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `efi` | Efik | new | 1012 | 95.3% | 0.0% | 21.33 | 18.56 (48) | 21.33 (1012) | 14.59 | 14.84 | 14.59 | 6.74 | 3.72 |
| `ibb` | Ibibio | new | 1012 | 95.3% | 0.0% | 15.66 | 14.71 (48) | 15.66 (1012) | 13.19 | 12.62 | 13.19 | 2.47 | 2.09 |

Other metrics, finish and format rates (SFT; the base model never writes the tag, so its `prefix ok` is 0 by construction and its whole output is scored):

| tag | SFT spBLEU all | SFT BLEU all | AfriCOMET all (SFT / base) | QE (SFT) | finished (SFT / base) | prefix ok (SFT) | empty (SFT) |
|---|---|---|---|---|---|---|---|
| `efi` | 4.39 | 2.61 | 0.476 / 0.242 | 0.454 | 88.0% / 97.2% | 99.9% | 0.0% |
| `ibb` | 1.82 | 1.07 | - | - | 84.4% / 99.5% | 99.8% | 0.0% |

### 7.6 FLORES-MOZ (Sena)

| tag | language |  | n | contam. | pair seen | SFT chrF++ all | SFT clean (n) | SFT no-pair (n) | base all | base clean | base no-pair | Δ all | Δ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `seh` | Sena | new | 1012 | 95.3% | 0.0% | 25.59 | 25.02 (48) | 25.59 (1012) | 13.16 | 12.82 | 13.16 | 12.43 | 12.20 |

Other metrics, finish and format rates (SFT; the base model never writes the tag, so its `prefix ok` is 0 by construction and its whole output is scored):

| tag | SFT spBLEU all | SFT BLEU all | AfriCOMET all (SFT / base) | QE (SFT) | finished (SFT / base) | prefix ok (SFT) | empty (SFT) |
|---|---|---|---|---|---|---|---|
| `seh` | 4.02 | 2.86 | - | - | 93.6% / 100.0% | 99.9% | 0.0% |

### 7.7 GlotStoryBook (children's stories)

| tag | language |  | n | contam. | pair seen | SFT chrF++ all | SFT clean (n) | SFT no-pair (n) | base all | base clean | base no-pair | Δ all | Δ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `nbl` | South Ndebele | new | 321 | 5.0% | 0.0% | 16.10 | 16.20 (305) | 16.10 (321) | 11.52 | 11.76 | 11.52 | 4.59 | 4.44 |
| `dag` | Dagbani | new | 135 | 21.5% | 0.0% | 18.97 | 19.33 (106) | 18.97 (135) | 10.21 | 10.52 | 10.21 | 8.77 | 8.81 |
| `kqn` | Kaonde | new | 420 | 7.9% | 0.0% | 27.60 | 27.54 (387) | 27.60 (420) | 12.35 | 12.44 | 12.35 | 15.26 | 15.09 |
| `lun` | Lunda | new | 420 | 7.9% | 0.0% | 24.27 | 24.24 (387) | 24.27 (420) | 12.38 | 12.44 | 12.38 | 11.89 | 11.80 |
| `loz` | Lozi | new | 960 | 6.6% | 0.0% | 32.32 | 32.18 (897) | 32.32 (960) | 12.82 | 12.87 | 12.82 | 19.50 | 19.31 |
| `tsc` | Tswa | new | 101 | 1.0% | 0.0% | 14.42 | 14.43 (100) | 14.42 (101) | 9.18 | 9.20 | 9.18 | 5.23 | 5.23 |
| `kua` | Kuanyama | new | 520 | 6.3% | 0.0% | 28.28 | 28.26 (487) | 28.28 (520) | 14.64 | 14.72 | 14.64 | 13.64 | 13.54 |
| `her` | Herero | new | 549 | 7.1% | 0.0% | 20.82 | 20.81 (510) | 20.82 (549) | 13.82 | 13.94 | 13.82 | 7.00 | 6.87 |

Other metrics, finish and format rates (SFT; the base model never writes the tag, so its `prefix ok` is 0 by construction and its whole output is scored):

| tag | SFT spBLEU all | SFT BLEU all | AfriCOMET all (SFT / base) | QE (SFT) | finished (SFT / base) | prefix ok (SFT) | empty (SFT) |
|---|---|---|---|---|---|---|---|
| `nbl` | 3.19 | 1.67 | - | - | 94.4% / 95.3% | 100.0% | 0.0% |
| `dag` | 3.13 | 2.59 | - | - | 87.4% / 100.0% | 100.0% | 1.5% |
| `kqn` | 5.95 | 2.99 | - | - | 88.6% / 100.0% | 100.0% | 0.0% |
| `lun` | 4.68 | 2.27 | - | - | 92.1% / 99.8% | 100.0% | 0.0% |
| `loz` | 6.35 | 3.88 | - | - | 94.9% / 99.9% | 100.0% | 0.0% |
| `tsc` | 0.80 | 0.65 | - | - | 77.2% / 100.0% | 100.0% | 0.0% |
| `kua` | 5.96 | 3.35 | - | - | 92.1% / 99.8% | 100.0% | 0.0% |
| `her` | 3.54 | 2.36 | - | - | 81.8% / 98.7% | 100.0% | 0.0% |

### 7.8 SALT (agricultural sentences)

| tag | language |  | n | contam. | pair seen | SFT chrF++ all | SFT clean (n) | SFT no-pair (n) | base all | base clean | base no-pair | Δ all | Δ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ibo` | Igbo |  | 500 | 84.4% | 0.0% | 35.20 | 35.47 (78) | 35.20 (500) | 6.31 | 5.65 | 6.31 | 28.89 | 29.83 |
| `swh` | Swahili |  | 500 | 84.4% | 0.0% | 57.47 | 52.05 (78) | 57.47 (500) | 8.57 | 8.21 | 8.57 | 48.90 | 43.84 |
| `lug` | Luganda |  | 500 | 84.4% | 83.8% | 45.64 | 36.94 (78) | 37.49 (81) | 7.58 | 7.38 | 7.34 | 38.06 | 29.56 |
| `ach` | Acholi | new | 500 | 84.4% | 0.0% | 24.50 | 22.33 (78) | 24.50 (500) | 10.87 | 10.74 | 10.87 | 13.64 | 11.59 |
| `nyn` | Nyankore | new | 500 | 84.4% | 0.0% | 26.72 | 22.35 (78) | 26.72 (500) | 10.02 | 9.81 | 10.02 | 16.70 | 12.54 |

Other metrics, finish and format rates (SFT; the base model never writes the tag, so its `prefix ok` is 0 by construction and its whole output is scored):

| tag | SFT spBLEU all | SFT BLEU all | AfriCOMET all (SFT / base) | QE (SFT) | finished (SFT / base) | prefix ok (SFT) | empty (SFT) |
|---|---|---|---|---|---|---|---|
| `ibo` | 13.65 | 9.79 | 0.721 / 0.222 | 0.687 | 98.8% / 56.4% | 100.0% | 0.0% |
| `swh` | 34.55 | 29.47 | 0.758 / 0.292 | 0.759 | 100.0% / 85.6% | 100.0% | 0.0% |
| `lug` | 24.36 | 15.10 | 0.700 / 0.255 | 0.682 | 99.8% / 78.6% | 100.0% | 0.0% |
| `ach` | 4.82 | 2.94 | - | - | 97.4% / 98.8% | 100.0% | 0.0% |
| `nyn` | 5.32 | 2.39 | - | - | 98.2% / 96.8% | 100.0% | 0.0% |

## 8. Languages that still have no test set, and what could be done

| tag | name | training pairs (as packed in the SFT windows) | main training source (share) | mean QE of that source | public test data found |
|---|---|---|---|---|---|
| `urh` | Urhobo | 29,707 | english-urhobo_sentence-pairs_mt560 (76%) | 0.55 | none |
| `bin` | Edo | 31,562 | african-bible-parallel (68%) | 0.48 | Bible only |
| `oke` | Okpe | 17,943 | english-okpe_sentence-pairs_mt560 (100%) | 0.50 | none |
| `bbj` | Ghomala | 6,381 | african-bible-parallel (100%) | 0.32 | MAFAND fr-bbj (1,430 test, FRENCH source) |
| `swc` | Congo Swahili | 191,558 | english-congo-swahili_sentence-pairs_mt560 (100%) | 0.78 | TICO-19-swc-speech (CC-BY-NC, medical, 2,124 rows ~ 1,060 unique, audio columns) |
| `wal` | Wolaytta | 90,367 | english-wolaytta_sentence-pairs_mt560 (100%) | 0.64 | Bible only (eBible, OPUS bible-uedin): contaminated |
| `yao` | Yao | 33,016 | english-yao_sentence-pairs_mt560 (100%) | 0.48 | Bible NT only: contaminated |
| `sid` | Sidamo | 35,004 | english-sidaama_sentence-pairs_mt560 (100%) | 0.52 | none |
| `ish` | Esan | 3,926 | english-esan_sentence-pairs_mt560 (100%) | 0.43 | none |
| `dua` | Duala | 11,354 | english-duala_sentence-pairs_mt560 (100%) | 0.39 | none |
| `bas` | Basaa | 20,683 | english-basaa_sentence-pairs_mt560 (100%) | 0.41 | none usable (350 lines, NOODL licence) |
| `wes` | Cameroon Pidgin | 16,998 | Turaco-NLLB-mt-en-wes-cleaned-dataset (59%) | 0.39 | none (14 proverbs without references) |

The training data of these 12 languages is, for 11 of them, mostly one OPUS MT560 / JW-style religious-text set (`michsethowusu/english-<language>_sentence-pairs_mt560`) or the African Bible corpus (`bbj`, most of `bin`); `wes` is mostly an NLLB-translated set (`fotiecodes/Turaco-NLLB-mt-en-wes-cleaned-dataset`, machine output, QE 0.39). Consequences and options, best first:

1. **Commission a small professional test set** (the only leak-free option for the current checkpoints). IbomMT shows the price tag: 1,000 FLORES+ sentences per language by three paid linguists. FLORES+ English sentences are 95% in our training windows, so for a model-clean set use sentences from outside the pool (SALT-style domain text, or the 76 clean SmolSent sentences, which are professionally translated into 45 of our 77 languages already and would make a multi-way set with the new languages).
2. **Reference-free sanity metrics that need only English** (works today for all 12): translate the 1,012 FLORES+ devtest English sentences (or the clean SmolSent sentences) and report GlotLID language-ID rate in the accepted label set (`rl/lid.py`, D-37 gives the per-language thresholds), finish rate, repetition (the `degenerate` / `is_self_repetitive` guards) and length ratio. No reference needed; it catches the failures of section 7 (loops, empty answers, wrong language). AfriCOMET-QE is not validated for any of them: 10 are consensus-only languages (no QE call at all, D-38), `swc` and `wal` are only LISTED on the AfriCOMET card (QE blended with the consensus signal, never checked against human references).
3. **Hold out a slice of the training pairs - only for FUTURE training runs, with these caveats.** (a) The current SFT and the RL policies have trained on the whole pool, so a slice cut now measures fit to the training distribution, not generalisation; report it as "in-domain (religious / JW) held-in dev", never as a benchmark. (b) For a slice that is clean in later runs: deduplicate on the ENGLISH side across all 77 languages first (the same English sentence is in many languages' MT560 sets: 91% of SmolSent's English is already in the pool through other languages), take pairs from the middle-to-upper QE range and drop the extremes (very low = misaligned, very high = copies, numbers, names), 5-60 words, at most one pair per Bible chapter / document (neighbouring verses leak), 300-500 pairs per language, and remove those English sentences from ALL languages' windows before training. (c) Several of these sets are the same text type as the training text, so the score is optimistic by construction; compare it only between checkpoints, not between languages.
4. **Cheap partial sets**: swc - TICO-19-swc-speech after dropping the audio column (about 1,060 unique medical sentences, NC licence); seh was solved by FLORES-MOZ; bbj - the French-source MAFAND set can test a French -> Ghomala direction if one is ever trained.

## 9. Evaluating the RL checkpoints on these sets later

The flag files and data are already on the rental (`/workspace/sft_build/bench/test_sets/<set>/` and `/workspace/sft_build/bench/test_sets/_contamination/`, which also holds copies of the three default flag files so that one `CONTAM_DIR` works for every mix of benchmarks) and the updated code is in `/workspace/sft_build/gemma_mt_eval_candidate2/` (the training tree `gemma_mt/` was not touched). The model under test is always called `sft` by the driver; the untouched base is evaluated once (section 7) and does not need to be repeated: pass `--skip-base`.

```bash
# RL checkpoint (Hub repo <HF_ORG>/gemma-3-270m-translate-rl: main = best, branches latest / stage-*, tags rl-step-N; `stage-3` below is only an example), on the rental when its GPU is free
# (defaults of the script = the rental: venv_gemma, venv_vllm, vLLM sampler, GPU_MIN_FREE_GIB 24)
cd /workspace/sft_build/gemma_mt_eval_candidate2
OUT=/workspace/sft_build/eval/rare_rl_stage3 BENCHMARKS=smol_sent,smol_doc,gatitos,ibom_mt,flores_moz,glot_story,salt CONTAM_DIR=/workspace/sft_build/bench/test_sets/_contamination bash scripts/eval_sft_benchmarks.sh <HF_ORG>/gemma-3-270m-translate-rl --sft-revision stage-3 --skip-base      --languages ach efi iso kon kri ndc nbl nyn tiv ibb seh kqn lun loz kua her dag tsc   # the 18 newly covered languages; omit to run all 49
# add the default benchmarks to the same run:  BENCHMARKS=flores_plus,ntrex,mafand,smol_sent,...   (all ten names: BENCHMARKS=all)

# same on the shared old box with the HF sampler (what section 7 used)
cd /workspace/sft_build/gemma_mt_eval_candidate2 && source /workspace/sft_build/gemma_mt_eval_dev/old_env.sh
export GPU_LOCK_CMD="flock /workspace/sft_build/.gpu.lock" BUDGET=3600 GPU_MIN_FREE_GIB=1.5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/workspace/sft_build/eval/rare_rl_stage3 BENCHMARKS=smol_sent,smol_doc,gatitos,ibom_mt,flores_moz,glot_story,salt bash scripts/eval_sft_benchmarks.sh <HF_ORG>/gemma-3-270m-translate-rl --sft-revision stage-3 --skip-base      --comet-sequential --comet-batch-size 16 --hf-batch-size 128 --doc-batch-size 32 --word-batch-size 256 --unit-rows 2000 --min-subset-rows 20      --bench-limit smol_sent=500 smol_doc=40 --doc-max-new-tokens 900

# compare with the SFT numbers of this report (results.json of run A / B / C are in reports/gemma_rare_language_benchmarks_2026-10-19/*/)
python scripts/compare_eval_results.py reports/gemma_rare_language_benchmarks_2026-10-19/gain/results.json /workspace/sft_build/eval/rare_rl_stage3/results.json --subset no_pair
```
Notes: (1) a run with `smol_doc` needs a budget that fits one generation unit: `BUDGET` >= 3600 or `--unit-rows` small (a unit that does not finish inside one invocation is restarted from scratch - found the hard way). (2) The default `--skip-base` omits the base column of the report; the per-language tables then show `-` for it. (3) On the shared old box check `nvidia-smi` first: the foreign job took 5.6-8.9 GB, `--gpu-min-free-gib` must be below what is left. (4) The three flag files for flores_plus / ntrex / mafand are unchanged. (5) Do not use the rental GPU while its RL run is training.

## 10. Caveats and what was not verified

* The clean subsets are small by construction (76 SmolSent sentences, 22 SmolDoc documents per 130-document language, 48 FLORES-style sentences): a per-language clean chrF++ is noisy (+-3 chrF++ is plausible); use the macro over languages and the `no_pair` column next to it.
* `no_pair` removes only rows whose reference pair was seen; the English sentence was seen with other languages (91% of SmolSent), which lifts every language a little (the same effect as FLORES+ in the 2026-10-18 report). Compare checkpoints with each other on the same subset, not absolute numbers with other papers.
* Language-mapping caveats: nyn <- SMOL `cgg` is a proxy (the real test is SALT); ndc is Shona-like text; ibb is called `efi` by GlotLID (close sibling); tsc is not confirmed by a native speaker; SMOL `ak` is Akan (we treat it as Twi, FLORES+ is Akuapem); Congo Swahili `swc` is not in SMOL (`sw` is general Swahili).
* GATITOS measures a different skill (single words with several accepted answers, no context); the SFT data has almost no word-level pairs, so low scores say little about sentence translation. The "clean" subset for words only means "the English entry is not a whole training sentence".
* SmolDoc: the documents differ between language groups (core documents for the 130-document languages, additional documents for the rest, see 7.3), so SmolDoc scores are comparable between checkpoints, not between languages. A document is scored as one text per document (corpus chrF++ / BLEU over the joined document, not per sentence); a generation that hits the 900-token budget is scored as it is (`finished` rate in the second table); AfriCOMET is not run on documents (512-token encoder).
* Runs B and C used smaller batches than run A because the foreign job on the card grew; batch size does not change greedy results beyond bf16 noise, but I did not test that explicitly.
* The SFT model is the Hub `main` of `<HF_ORG>/gemma3-270m-translate` at commit 376a94f2314da6445878962fe22d59900ee2d2b9 (training step 27638); the earlier report (2026-10-18) used the <HF_ORG_LEGACY> repo at step 20,500.
* Not done: AfriCOMET-QE-only scoring of the languages without references; TICO-19 swc and MAFAND fr-bbj integration; SALT dev split; no push to the Hub or S3; nothing committed to git.
