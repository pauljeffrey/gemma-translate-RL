# Addendum to the long-context benchmark plan: 500 documents, general domains, trusted sources, release schema, two translators (2026-10-20)

> **Superseded in part (2026-10-20 evening):** six length bins, relaxed licences, hard split, the private Hub repo and the runner are in `longdoc_benchmark_plan_2026-10-20_addendum2_six_bins.md`.

Applies four owner decisions of 2026-10-20 on top of `longdoc_benchmark_plan_2026-10-20.md` (the earlier report stays valid for the metrics, the validation, the contamination method and the risks; where this addendum disagrees, **this addendum wins**: the 200/1000 core/extended split, the eight domains, the six-system translator list and the cost tables are superseded). Decision record: `gemma_mt/DECISIONS.md` D-44 (addendum). Nothing was run on the GPU; nothing was uploaded; the trainer and its supervisors were not touched.

## 1. What was decided and what changed

| owner decision | implementation |
|---|---|
| ONE shared English set of **500 documents** into all 77 languages (no core / extended split) + the **SmolDoc documents on top** as a clearly marked human-reference subset | `longdoc_select.py --n 500`; `subset = core` for the 500, `human_smoldoc` / `human_stitched` for the 45-language SmolDoc material (12,499 (document, language) rows from 2,424 English documents; 5,182 rows with English not in the SFT windows) |
| trusted, citable, redistributable sources; Gutenberg <= 10%; share-alike only as a separate split | pool rebuilt (section 2); Gutenberg + Britannica-1911 (a Gutenberg derivative) together capped at 10% (`--max-share gutenberg=0.10`, group `eb1911 -> gutenberg`), every other source <= 15%; Wikipedia / Wikivoyage / FLORES+ / NTREX English not in the 500 |
| contamination filter incl. the b2048 RL prompts | the scanner now reads `rl_prompts_v3` (154 parquet files incl. `zz_b2048/`; hard links to v2 counted once) and the selection drops any pool document whose URL is in `rl_prompts_b2048_doc_ids.txt` (20,644 URLs; 0 pool documents matched, the text check removes the rest) |
| **no medicine, no law** in the core | files `federalregister`, `fedhealth` (NIH), `pmc*`, `legislation*`, `billsum*`, `eurlex*`, `cdc*`, `medlineplus*`, `cp_usgpo*` are excluded, domains `health` and `law_government` are dropped after re-labelling GOV.UK speeches / press releases as `government_public` and OpenStax civics / business books as `education` (`longdoc_select.DOMAIN_REMAP`) |
| general domains | news, encyclopedia (explanatory), education / textbook, stories, science_explainer (science and technology), travel, government_public (non-legal public information), opinion_magazine |
| released record with translations and scores incl. the TOTAL RL reward | section 4: schema, `scripts/longdoc_score_reward.py`, datasheet, dataset card |
| **top-2 translators** | section 5: NLLB-200 3.3B (internal referee) + MADLAD-400 (released pseudo-reference) confirmed, one correction |
| hard domain split (medicine, law, maths-science) | section 7: design only |

## 2. The 500-document pool (final composition)

Funnel (measured, `SELECTION_REPORT_500.json`): 17,069 pool documents after the licence-class, file and domain policy -> contamination filter drops 1,500 contaminated + 1,604 weak (SFT stream, RL prompts v3 incl. b2048, existing benchmarks, duplicates between pool documents) -> cleaning guards drop 1,186 documents with more than 20% bad paragraphs and a few hundred more for length / sentence statistics -> **11,948 clean candidate documents**, of which 11,541 / 6,044 / 3,612 / 2,071 can serve the 512 / 1024 / 2048 / 4096 bin. **A document with more than 2% of its windows in our training data is not in the 500** (a document seen in training is "weak" or "contaminated" and is dropped).

| source | licence (as read on the licensor's page) | provenance / citation | why the research community trusts it | pool docs | clean candidates | in the 500 | share |
|---|---|---|---|---|---|---|---|
| Common Pile v0.1 `news` (CC BY outlets: 360info, Alt News, Balkan Diskurs, Factly, Freedom of the Press Foundation, Mekong Eye, Minority Africa, New Canadian Media, ZimFact ...) | CC-BY-4.0, per document (`license` field of the Common Pile record) | Kandpal et al., *The Common Pile v0.1*, arXiv:2506.05209; HF `common-pile/news`; licence list from opennewswire.org | EleutherAI / Toronto / Vector / HF consortium, per-document licence metadata, trains Comma v0.1; the paper itself warns about licence laundering (quote it in the release notes) | 1,153 | 842 | 75 | 15.0% |
| GOV.UK (speeches, news stories, guidance, FCDO travel advice, education and science pages) | Open Government Licence v3.0 (https://www.gov.uk/help/terms-conditions) | GOV.UK Search + Content API, Crown copyright, departments named per document | official UK government publisher; OGL is the standard open licence of public-sector information | 1,069 | 416 | 75 | 15.0% |
| Common Pile Pressbooks + LibreTexts open textbooks (<= 3 chapters per book) | CC-BY-4.0 / CC-BY-3.0 / public domain, per chapter | Kandpal et al. 2025 (as above); HF `common-pile/pressbooks`, `common-pile/libretexts` | named open-textbook publishers (BCcampus, eCampusOntario, OpenOregon, LibreTexts); chapters carry a licence statement checked by the Common Pile | 1,089 | 673 | 51 | 10.2% |
| Common Pile DOAB (English open-access books, one 4,600-token window per book) | CC-BY-4.0 / 3.0, statement in the book's front or back matter | Kandpal et al. 2025; Directory of Open Access Books (OAPEN Foundation) | DOAB is the standard index of peer-reviewed open-access books (OAPEN, Springer Nature OA, De Gruyter, Ubiquity Press); **scholarly register, 4096-bin only** | 263 | 140 | 42 | 8.4% |
| GovReport: GAO and CRS reports (SCROLLS packaging) | public domain (US government work); GAO copyright page quoted in `sources_v2.md`; **the CRS statement could not be read (403)** | Huang et al., NAACL 2021 (aclanthology 2021.naacl-main.112); Shaham et al. 2022 (SCROLLS), HF `tau/scrolls` | standard long-document benchmark (SCROLLS, ZeroSCROLLS, LongBench); expect it to be a known public dataset in other models' training data | 1,500 | 1,353 | 41 | 8.2% |
| Project Gutenberg novels, essays and travel books (authors died <= 1954) | public domain (US), per book | gutenberg.org catalog + `pg<id>.txt`; trademark text stripped | the canonical public-domain library; archaic English, hence the 10% cap (shared with EB1911) | 638 | 562 | 36 | 7.2% |
| Encyclopaedia Britannica 11th ed. (1910-11) via Project Gutenberg | public domain (US) | Project Gutenberg ebooks of the 29 volumes | classic reference work; Edwardian; **inside the 10% Gutenberg cap** | 755 | 648 | 14 | 2.8% |
| Wikinews (articles 2004-2026) | CC-BY-2.5 (to 2024-12-15), CC-BY-4.0 (after), public domain before 2005-09-25; per article | Wikimedia dump of 2026-10-20 | Wikimedia; the news side of FLORES (Goyal et al. 2022 take their English from Wikinews, Wikijunior and Wikivoyage; those exact articles are flagged and removed by the contamination filter) | 6,561 | 5,217 | 26 | 5.2% |
| Global Voices (citizen media, 2005-) | CC BY 3.0 (https://globalvoices.org/about/global-voices-attribution-policy/) | WordPress REST API; author bylines from the Common Pile `globalvoices` records | a standard OPUS corpus (GlobalVoices, Tiedemann) used in MT for years; credit file with author names is complete (1,299 of 1,299) | 1,299 | 388 | 21 | 4.2% |
| Voice of America English news | public domain: text produced exclusively by VOA (https://www.voanews.com/p/5338.html); wire-service articles dropped | VOA sitemaps | US government international broadcaster | 274 | 248 | 18 | 3.6% |
| NASA science and mission articles | US government work (https://www.nasa.gov/nasa-brand-center/images-and-media/) | WordPress REST API of nasa.gov / science.nasa.gov | official agency; credited ESA/JAXA/university text may be third-party | 1,598 | 832 | 20 | 4.0% |
| Our World in Data articles (health pages removed) | CC BY 4.0 (https://ourworldindata.org/faqs#can-i-use-our-content) | ourworldindata.org sitemap | widely cited research-and-data publication (Roser et al.); data and charts not included | 372 | 253 | 23 | 4.6% |
| OpenStax current books (CC BY 4.0 at repository head) | CC BY 4.0 (LICENSE of the book repository) | github.com/openstax/osbooks-*, CNXML | Rice University open-textbook publisher; the older CC-BY snapshot (relicensed NC-SA upstream) is excluded | 258 | 195 | 33 | 6.6% |
| National Park Service articles and history pages (`nps__pd`; the biomedical NIH files are excluded) | US federal government work (https://www.nps.gov/aboutus/disclaimer.htm) | NPS sitemap + HTML | official agency; photos and quotes may be third-party | 240 | 181 | 25 | 5.0% |
| **total** | pd 160 / permissive 340 | | | 17,069 | 11,948 | **500** | 100% |

Reading the table. Licence texts, the quoted evidence and the rejected sources (WMT news test sets, UN documents, the World Bank repository, Africa Is a Country, The Conversation, Mongabay, ... with the reason) are in `sources.md` and `sources_v2.md`. **WMT news**: the statmt.org terms say the data "can be freely used for research purposes" and "for other uses ... consult with original owners"; that is not permission to redistribute the publishers' articles in a public benchmark, and NTREX-128's English is the WMT19 set, so neither is used (WMT remains an external comparison, or ask the organisers). **GovReport**: the GAO half is clearly public; the CRS statement could not be verified from the rental (403) - the owner may drop the CRS half (no per-report flag is in the record yet; `extra.topic_guess` / report ids identify it). No source exceeds 15%; the largest are Common Pile news and GOV.UK (15.0% each).

## 3. Stratification of the 500

Plan (`quotas`, largest remainder): 512: 176, 1024: 152, 2048: 108, 4096: 64 (35.2 / 30.4 / 21.6 / 12.8%) over 8 domains of 62-63 each. Result after the shortfalls were filled (the thin domains cannot reach their quota; 159 documents were moved to the domain with the fewest documents, by source round-robin):

| domain | 512 | 1024 | 2048 | 4096 | total | clean candidates (all bins) |
|---|---|---|---|---|---|---|
| education | 33 | 26 | 14 | 8 | 81 | 716 |
| encyclopedia | 34 | 25 | 14 | 8 | 81 | 981 |
| government_public | 25 | 28 | 18 | 8 | 79 | 1,667 |
| news | 34 | 27 | 14 | 6 | 81 | 6,395 |
| opinion_magazine | 7 | 8 | 27 | 17 | 59 | 300 |
| science_explainer | 34 | 25 | 14 | 8 | 81 | 1,500 |
| stories | 7 | 6 | 4 | 6 | 23 | 270 |
| travel | 2 | 5 | 5 | 3 | 15 | 119 |
| **total** | 176 | 150 | 110 | 64 | 500 | 11,948 |

* Bins: 176 / 150 / 110 / 64 (the plan was 176 / 152 / 108 / 64).
* **Thin domains, honestly.** `stories` has 23 documents (quota 62): the only clean-licence stories are Gutenberg, whose 10% cap (50 documents) is shared with Britannica 1911 and with the Gutenberg encyclopedic books: encyclopedia takes 14 + 13 of them, stories the remaining 23 (the capped group may only fill `stories` and `encyclopedia`). `travel` has 15 (quota 62): 210 GOV.UK FCDO travel-advice pages are mostly near-duplicates (only 119 travel candidates survive the duplicate and contamination filters) and no modern travel writing has a clean licence (Wikivoyage is share-alike). `opinion_magazine` has 59 but 44 of them are 2048 / 4096-token features from the same Common Pile outlets. `encyclopedia` rests on Britannica 1911 (14), Gutenberg encyclopedic books (13), National Park Service pages and DOAB/Pressbooks chapters. Options in section 9 (D-1).
* **Private split**: 50 of the 500 (10%, same strata: 17 / 16 / 11 / 6 per bin) are `split = private`, written to `core_private/` and never into the public file; public 450.
* Files (rental, `longdoc_bench/source_set/`): `core/docs.jsonl` (all 500, master, with `split`), `core_public/`, `core_private/`, `ATTRIBUTION.jsonl` in each, `SELECTION_REPORT.json`, `core_examples.md` (three examples per domain, copy in `data/core500_examples.md`), `human/` (SmolDoc, stitched, FLORES+ passages with `contaminated` flags).

## 4. The public record, the reward script, the datasheet

* **Record** = one JSON line per (document, language, system), 52 columns (`gemma_mt/eval/longdoc_release.py::RECORD_COLUMNS`, rendered into the dataset card): English source and its licence / attribution, system id and checkpoint, the translation, `finished`, the references (list; human where available else pseudo with translator id and tier; a pseudo-reference of a non-permissive translator is **refused by the builder**), BLEU and chrF++ (document level, against the primary reference), AfriCOMET reference-based and QE (null when the language is not covered: `africomet_status`), `coverage_*`, `omitted_sents`, `copied_sents`, `loop`, `ends_cleanly`, `length_ratio`, and `reward_*` including `reward_total`, `reward_status`, content mode, chunks, failed chunks, GlotLID label, consensus chrF++, diacritic deficit, the reward range and the sha256 of the calibration file. Every record carries the canary string. `validate_record` checks types, ranges, tiers, the private split and the translator-licence rule; `scripts/longdoc_make_release.py build` assembles the public and the private files and exits non-zero on any problem.
* **`scripts/longdoc_score_reward.py`** (library `eval/longdoc_reward.py`): scores a system's per-document outputs with the REAL `TranslationReward`: each output is turned back into the policy's completion (`"{iso} {translation}"` + `<end_of_turn>`, or no stop id when `finished` is false), then the reward block of the RL yaml is applied unchanged (AfriCOMET-QE per chunk, sentence chunking with the language-aware omission detector, calibrated per-language shaping from `shaping_flores_dev_v6.json`, qe / blend / consensus modes, failure classes, length / lexical / diacritic penalties, optional GlotLID). It returns per document: reward, status, mean QE, content mode, chunks, failed chunks, weight, LID label, consensus chrF++, diacritic deficit. Two modes: every output alone (the trainer's greedy evaluation: blend languages score QE only, consensus-only languages get the rubric-only value) and `--group-systems` (the outputs of the several systems for one document form the consensus group). CPU: `CUDA_VISIBLE_DEVICES=""` with the worker scorer; GPU: leave the card visible. `--scorer fake --tokenizer fake` runs the whole path without any model (used by the tests). Example (rental): `python scripts/longdoc_score_reward.py --preds preds.jsonl --docs source_set/core/docs.jsonl --config /workspace/sft_build/gemma_mt_2c/configs/rl_rtxpro_phase2c.yaml --calibration /workspace/sft_build/gemma_mt_2c/calibration/shaping_flores_dev_v6.json --scorer worker --out out/reward_scores.jsonl`. Note the 2c yaml points at the v6 file; the D-43 efi variant (`shaping_flores_dev_v6_efi_consensus.json`) is a different calibration: record which one with the hash (done automatically).
* **Check with the real stack (CPU, measured)**: the script was run on the rental with the real config `rl_rtxpro_phase2c.yaml`, the real calibration `shaping_flores_dev_v6.json` (sha256 `fdaf9523...`), the real AfriCOMET-QE worker, GlotLID and the Gemma tokenizer, on 30 SmolDoc documents (6 each of yor hau ewe pcm kab) in three versions (`data/reward_check_summary.json`): the human reference scores a mean reward **0.68** (ewe 0.21, hau 1.11, kab 0.54, pcm 0.41, yor 1.14; all 30 `ok`), the same text cut after half of its sentences **-1.00** (omitted chunks count as failed chunks), and the same text flagged not finished **-2.00** (`no_end_of_turn`, the floor). 3 min 15 s for 90 outputs, mostly model loading. Reward range [-2.0, 2.25].
* **Datasheet draft**: `reports/longdoc_release_datasheet_draft_2026-10-20.md` (motivation, composition, collection, preprocessing, uses, limitations incl. the pseudo-reference caveat, distribution, maintenance; canary; private-half policy: ~10% held back). **Dataset card skeleton** with the YAML header and the column table: `reports/longdoc_release_dataset_card_skeleton_2026-10-20.md`.
* **Name and licence, two options (open decision R-1).** *Option 1 - permissive release*: name `AfriLongDoc`; data CC-BY-4.0 with per-document licences and an attribution file, code Apache-2.0; the share-alike material not included (or published separately later); pseudo-references only from Apache-2.0 translators. Simple, matches the owner's exclusion of share-alike from training, no copyleft on users. *Option 2 - full composite*: name `SabiLongDoc` (ties the benchmark to the model family); one CC-BY-SA-4.0 dataset that adds the Wikipedia / Wikivoyage / FLORES+ English split (fills encyclopedia and travel) - every derivative translation and score table must then be share-alike, which discourages some users and mixes in the most contaminated sources. Recommendation: option 1 with the SA split as an optional separate repository.

## 5. Translators: the two systems (owner proposal confirmed, with one correction)

The proposal - **NLLB-200 3.3B as the internal referee and MADLAD-400 as the released pseudo-reference system** - is confirmed as the main pair: they are the two best-known open MT systems that name African languages, they cover the same number of our languages (44 each) and between them 55 of the 77. One correction, from the published scores we collected (not measured by us): **MADLAD is much weaker than NLLB on a large part of the languages both name**, so with MADLAD as the only released system many languages will fall into tier C (reported, never ranked) unless the optional OPUS tc-bible supplement helps. The bake-off on FLORES+ dev decides per language; nothing below is a measurement of quality.

| | NLLB-200 3.3B (`facebook/nllb-200-3.3B`) | MADLAD-400 (`google/madlad400-7b-mt-bt` or `-3b-mt`) |
|---|---|---|
| role | internal referee, agreement system, quality ceiling; never published | released pseudo-reference (the only translator whose outputs are published), per-language choice with the OPUS supplement |
| languages of our 77 named | **44** (FLORES-200 list) | **44** (card language list and `<2xx>` vocabulary) |
| only this system | 11: bem kab kam kea kik lua luo mos nus tum umb | 11: bas ibb iso kri kua nbl seh tiv tsc ven wal |
| both | 33 | 33 |
| UNION | **55 of 77**; remainder **22**: ach bbj bin dag dua efi her ish kqn loz lun ndc nyn oke pcm shi sid swc urh vmw wes yao | |
| remainder covered by OPUS tc-bible-big (mul, Apache-2.0, 0.96 GB) | 8: ach efi her nyn pcm shi swc vmw -> 63 of 77 | |
| covered by none of the three | **14**: bbj bin dag dua ish kqn loz lun ndc oke sid urh wes yao | |
| paper | NLLB Team 2022, arXiv:2207.04672 (Nature 630, 2024); OpenAlex cited-by 370 for the arXiv record (2026-10-20; the Nature record is counted separately) | Kudugunta et al., NeurIPS 2023 Datasets & Benchmarks, arXiv:2309.04662; OpenAlex cited-by 19 for the NeurIPS record (low: the arXiv record is counted separately) |
| Hub usage, downloads in the last 30 days (2026-10-20, `usage_evidence.json`) | 3.3B 239,289; distilled-1.3B 480,413; distilled-600M 846,973; likes 483 / 192 / 996 | 3B 38,473 (240 likes); 7B-mt 2,450; 7B-mt-bt 648; 10B 3,418 (140 likes) |
| licence of the weights | CC-BY-NC-4.0 | Apache-2.0 |
| consequence for publishing outputs | The card is silent on outputs; whether the NC clause reaches generated text is untested. **Publish nothing generated by NLLB**: only derived statistics (the language's tier, agreement scores) - the decision rests with the owner's counsel. | No output restriction; outputs can be released as pseudo-references (CC-BY-4.0, attribution to the translator and to the source licence). |
| download | 17.58 GB (fp32 `pytorch_model` shards; distilled-1.3B 5.48 GB, distilled-600M 2.46 GB) | 7B-mt / 7B-mt-bt **33.19 GB** (fp32 safetensors; repo total 44.7 GB incl. duplicates: use `allow_patterns`), 3B 11.76 GB, 10B 42.85 GB |
| backends | HF, CTranslate2 (not vLLM) | HF, CTranslate2 (not vLLM) |

**Evidence on quality (published, not ours; MADLAD paper Table 17, chrF on FLORES-200 en->xx, MADLAD-7.2B vs NLLB-54B-MoE).** Of the 32 languages with published numbers for both, MADLAD-7.2B scores below 30 chrF on 14 (bam dyu ewe fon kbp kin kmb kon lin orm sag tir wol yor) and NLLB-54B reaches 30 or more on 9 of them (bam ewe kbp kin kon lin orm sag wol; e.g. Kinyarwanda 9.6 vs 53.8, Lingala 18.8 vs 50.8, Wolof 12.6 vs 31.8, Yoruba 20.3 vs 27.4). For the 11 NLLB-only languages MADLAD has no `<2xx>` token in its 3B/10B variants (about 2 chrF in the paper; the 7B variants 10-18). MADLAD-7B-bt adds +3.0 chrF on average over 7B-mt in the paper. The 3B model is the one the community actually downloads (16x the 7B-mt) but is typically 0.5-8 chrF below the 7.2B.
**What the supplement does.** The 240 M Apache-2.0 OPUS tc-bible-big models (`-mul` 59 of our languages, `-bnt` 31, `-afa` 8; 0.96 GB each) report FLORES-200 devtest chrF on the model cards of zu 53.3, xh 52.0, nso 48.6, tso 48.5, lin 45.6, sna 45.6, kin 41.5 - above MADLAD-7.2B on zu tso nso kin lin tsn ssw bem. They are Bible-heavy (domain mismatch) and their implementation of the metric may differ. As the owner asked they are **not a third main system**: they are an optional, zero-cost candidate for the released-reference slot, chosen per language by the same bake-off (and they are the only open source of a pseudo-reference for ach efi her nyn pcm shi swc vmw).
**Languages neither main system covers (22).** 8 are rescued by the supplement; for the 14 left (bbj bin dag dua ish kqn loz lun ndc oke sid urh wes yao): **ndc** has 130 SmolDoc documents (human reference, clean subset 22); **dag kqn loz lun** have sentence-aligned GlotStoryBook stories (a bake-off set, not documents); **bbj bin dua ish oke sid urh wes yao have no human reference of any kind** -> unscored; they stay in the model's training languages but cannot be in the benchmark unless somebody writes references.

### 5.1 Download plan for the bake-off (one system at a time, 40 GB free)

The "150 GB" of the first report was the sum of the six systems then recommended: Sunflower-Qwen3.8-27B 54.71 GB + MADLAD-7B 33.19 GB + TranslateGemma-12B 24.37 GB + AfriqueQwen3.5-9B 17.91 GB + NLLB-3.3B 17.58 GB + OPUS tc-bible mul/bnt/afa 2.88 GB = **150.7 GB** (Hub file metadata read 2026-10-20, `data/weight_sizes.json`). With two main systems:

| step | download | GB | resident peak | languages named after the step |
|---|---|---|---|---|
| 1 | OPUS tc-bible-mul (optional) | 0.96 | 1 GB | 59 |
| 2 | NLLB-200 3.3B (referee) | 17.58 | 17.6 GB | 44 |
| 3 | MADLAD-400 7B-mt-bt (`allow_patterns=['*.safetensors','*.json','*.model']`) | 33.19 | 33.2 GB | union 55; with the supplement 63 |
| 4 | OPUS tc-bible-bnt, -afa (optional) | 1.92 | 2 GB | 63 (better Bantu / Afro-Asiatic quality) |
| 5 | per-pair OPUS-MT `kqn lun loz` (optional, 0.3 GB each) | 0.9 | 1 GB | 66 (permissive named: kqn lun loz) |
| | **total transferred** | **54.5** (NLLB + MADLAD + all optional: 54.5; **NLLB + MADLAD only 50.8**) | **33.2 GB** (fits the 40 GB box; delete each system after its run; AfriCOMET is already cached) | |

Smaller variant, ~30 GB: NLLB-3.3B 17.58 + **MADLAD-3B 11.76** (the size the community uses) + OPUS 0.96-3.8 = 30.3-33.1 GB total, peak 17.6 GB. Not needed: TranslateGemma, Sunflower, AfriqueQwen, Gemma 4 (they only add quality options for the 31-47 languages they name, and Sunflower-27B at 54.7 GB does not fit on the box at all). CTranslate2 int8 conversion of either system (after the download) shrinks them to 3.5 GB and 8 GB for the translation run.

### 5.2 GPU hours with two systems

Bake-off on FLORES+ dev (997 sentences x 44 languages x 2 systems): 0.08 - 0.16 GPU-h of decoding plus AfriCOMET (1.77 M output tokens per system; the three-system variant with the OPUS supplement 0.09 - 0.20); the wall time is the 51 GB of downloads. The 500-document release costs 10.0 - 27.4 GPU-h in total for NLLB-3.3B + MADLAD-7B (8.4 - 23.4 with the QE pair cache; of which the two translators only 1.1 - 2.0); the dominant item is the generation and scoring of OUR model's 313 M tokens (table below, section 6).


## 6. Cost: 500 documents x 77 languages

Volumes are measured (705 k English tokens, 26,905 sentences per language, Gemma fertility 1.92); throughputs are the assumptions of section 5.2 of the main report (roofline x 25-45% for CTranslate2, 8-18% for HF; our model 10-40 k tokens/s; AfriCOMET x1.5-3 of the RTX 5070). Every translator is charged for all the languages it names (upper bound).

| item | volume | GPU hours (estimate) |
|---|---|---|
| English set | 500 documents x 77 languages; 705 k English tokens, 26905 sentences per language | |
| pseudo-references, NLLB-3.3B + MADLAD-7B: nllb200-3.3b (44 languages) | 44.8 M tokens | 0.3 - 0.5 |
| pseudo-references, NLLB-3.3B + MADLAD-7B: madlad400-7b-mt (44 languages) | 44.8 M tokens | 0.9 - 1.6 |
| **translators, NLLB-3.3B + MADLAD-7B** | | **1.1 - 2.0** |
| pseudo-references, NLLB-3.3B + MADLAD-3B: nllb200-3.3b (44 languages) | 44.8 M tokens | 0.3 - 0.5 |
| pseudo-references, NLLB-3.3B + MADLAD-3B: madlad400-3b-mt (44 languages) | 44.8 M tokens | 0.2 - 0.4 |
| **translators, NLLB-3.3B + MADLAD-3B** | | **0.5 - 0.9** |
| pseudo-references, optional OPUS tc-bible supplement: opus-mt-tc-bible-big-eng-mul (59 languages) | 60.0 M tokens | 0.4 - 0.9 |
| **translators, optional OPUS tc-bible supplement** | | **0.4 - 0.9** |
| our model, 3 checkpoints | 313 M generated tokens | 2.2 - 8.7 |
| AfriCOMET reference based (chunked) | 6.22 M triples | 2.5 - 6.0 |
| AfriCOMET QE (metric) | 6.22 M pairs | 1.6 - 4.0 |
| RL reward (QE per chunk + CPU rules) | 6.22 M pairs, 32 CPU core-hours | 1.6 - 4.0 (identical pairs as the QE metric: a pair cache removes one of the two) |
| human subset SmolDoc | 12,499 (document, language) rows, 3.65 M English tokens | model 0.1 - 0.6 + ref 0.1 - 0.3 + QE and reward 0.2 - 0.4 |
| human subset SmolDoc stitched to 1024 tokens | 2,744 (document, language) rows, 2.94 M English tokens | model 0.1 - 0.5 + ref 0.2 - 0.4 + QE and reward 0.2 - 0.5 |
| **TOTAL, NLLB-3.3B + MADLAD-7B** | | **10.0 - 27.4** (8.4 - 23.4 with the QE pair cache) |
| **TOTAL, NLLB-3.3B + MADLAD-3B** | | **9.4 - 26.2** |


Reading: the two translators cost only 1.1-2.0 GPU-h (MADLAD-7B; 0.5-0.9 with MADLAD-3B); with HF `generate` instead of CTranslate2 multiply by 2-4. The cost is our own model (3 checkpoints x 313 M generated tokens) and the three scoring passes; AfriCOMET-ref counts all 77 languages although only the 54 covered ones are scored (about 30% less); the RL reward and the AfriCOMET-QE metric score the same pairs, so a cache halves them. Bake-off (FLORES+ dev, 44 languages, 2 systems): 0.08 - 0.16 GPU-h of decoding plus AfriCOMET (1.77 M output tokens per system; the three-system variant with the OPUS supplement 0.09 - 0.20); the wall time is the 51 GB of downloads.

## 7. Design: an optional HARD / DOMAIN split (medicine, law, maths-science) - not built, not translated

Owner decision: the core has no medicine and no law. They become **a separate, optional dataset** ("hard / domain", working name `AfriLongDoc-Hard`), released and scored on its own, never mixed into the core macro average.

* **Size proposal.** 150 English documents (50 medicine, 50 law, 50 maths-science), bins 512 / 1024 / 2048 only (about 20 / 20 / 10 per domain; 4096-token legal and clinical texts are rarely coherent units), translated into the same 77 languages (or the tier A/B languages only). Translation volume is about 0.5x the core's (150 documents of about 1,000 tokens against 500 of about 1,000), i.e. well under 1 GPU-h (translators) + 2-4 GPU-h (model and scoring) GPU-h for the pseudo-references and the model runs. A 100-document variant (35 / 35 / 30) is the minimum for a domain-level CI of +-1-2 chrF++.
* **Sources and licences (all already partly in the pool, kept out of the core).**
  - *Medicine*: PubMed Central open-access articles under CC BY (intro / discussion prose; NLM / Europe PMC, per-article licence in the JATS XML; 263 documents in the pool), NIH institute pages (NIDDK / NIMH / NIDA, US federal text, 473 documents), CDC and NCI pages (US federal, "most of the information is not subject to copyright"; not yet fetched), GOV.UK health guidance (OGL, 154 documents). WHO is CC BY-NC-SA: excluded.
  - *Law*: UK Explanatory Notes and Acts (OGL v3, legislation.gov.uk; 224 documents fetched, the crawler needs about 2 h for more), EUR-Lex legal acts (EU reuse CC-BY 4.0; behind a WAF from the rental, Hub mirrors exist), US bills (BillSum, CC0 card, 500 in the pool), Federal Register rules and notices (US federal, 898 documents), US committee reports and hearings (GPO, public domain, 665 documents).
  - *Maths / science*: OpenStax current books (Statistics, Physics: CC BY 4.0) and LibreTexts / Pressbooks pages (Common Pile, per-chapter CC BY) **extracted with the formulas kept** (the core pool strips equation-heavy passages, so this needs a new extractor over CNXML / MathML / LaTeX), arXiv papers under CC BY (per-paper licence), NASA technical explainers (US federal).
* **Why separate.** Register (formulas, citations, clause numbering), entity density and error cost differ from general text; Bible-trained translators fail here in ways chrF++ hides, so the split needs its own checks and its own tiers (the FLORES+ bake-off says nothing about legal Zulu).
* **Metrics that suit it** (in addition to BLEU / chrF++ / AfriCOMET / coverage / loops of the core harness):
  1. **Number, unit and date preservation**: the multiset of numerals (after normalising digits, separators and the thousands/decimal convention) and of number+unit pairs (mg, kg, %, km, USD) in the source must be found in the translation; report precision, recall and F1, and the share of documents with any numeric error.
  2. **Entity and term preservation**: acronyms, chemical / drug / gene names, Latin names, statute and article references, URLs and citations should be copied or conventionally transliterated; a verbatim-copy check against a protected-token list extracted from the source, reported as recall.
  3. **Formula preservation**: LaTeX / MathML / inline formulas and symbols are masked in the source and must appear unchanged and in order in the translation (exact-match share, edit distance of the formula sequence).
  4. **Term consistency**: for English terms that occur at least 3 times in a document (key nouns, defined terms), the share of occurrences translated by the document's most frequent target rendering, measured on the sentence-aligned beads (a low value = the translation drifts, a high value on a wrong rendering is caught by the numeric / entity checks and the human spot check).
  5. **Omission / modality**: the coverage metrics of the core harness, plus negation and modal-verb checks on a sample (a dropped "not" or "must" is the worst legal / clinical error); sentence-level, human-audited on a small sample.
* **Caveat to print on the dataset card.** The translations and scores of this split are research artefacts: **medical and legal translations produced or scored here must never be used for clinical, diagnostic, treatment, legal or compliance purposes**; no output has been verified by a domain expert or a native speaker; the pseudo-references of rare languages are machine output of unknown accuracy.
* **Cost to build.** Source extraction with formulas and review of 150 documents: a few days of work; GPU cost negligible next to the core. Not started.


## 8. Files changed / added (laptop `gemma_mt/`, all synced to the rental `longdoc_bench/code/`)

* New: `gemma_mt/eval/longdoc_reward.py`, `gemma_mt/eval/longdoc_release.py`, `scripts/longdoc_score_reward.py`, `scripts/longdoc_make_release.py`, `tests/test_longdoc_reward.py` (6), `tests/test_longdoc_release.py` (5).
* Changed: `eval/longdoc_select.py` (single 500 set, private split, source / group caps, domain policy and re-labelling, b2048 URL exclusion, balanced fill, scarce-domain-first order), `eval/longdoc.py` (per-document BLEU and reference id in the rows), `eval/longdoc_costplan.py` (`plan500`, two-system bundles, reward and human-subset costs), `scripts/longdoc_contam_scan.py` + `longdoc_contam_run.sh` (RL prompts v3 incl. `zz_b2048`, fingerprint includes the file count), `tests/test_longdoc_pipeline.py` (domain policy, caps, private split, URL exclusion).
* From the pool agent: `scripts/longdoc_fetch_sources.py` (new `cp_news`, `cp_books`, `cp_doab`, `cp_usgpo`, `govreport`, `billsum`, `gv-authors`, `capacity`), `tests/test_longdoc_sources.py` (49), `parts/sources_v2.md`.
* Reports: this addendum, the datasheet draft, the dataset card skeleton; data in `reports/longdoc_benchmark_plan_2026-10-20_parts/data/` (`SELECTION_REPORT_500.json`, `core500_examples.md`, `cost_plan_500.json`, `bakeoff_cost_2sys.json`, `weight_sizes.json`, `usage_evidence.json`-derived numbers in section 5, `candidates_by_source.json`).
* Tests: **215 passed** on the rental with real pytest (`python -m pytest tests/test_longdoc*.py`, `venv_gemma`, 5.6 s) and 215 passed with `tests/run_tests.py` on the laptop `.venv-comet`: test_longdoc 22, test_longdoc_pipeline 24 (domain policy, caps, group domains, private split, URL exclusion, fake CTranslate2 / vLLM / HF backends), test_longdoc_reward 6, test_longdoc_release 5 (mine = 57), test_longdoc_systems 95, test_longdoc_contam 14, test_longdoc_sources 49. Real-stack CPU run of the reward script: see section 4.

## 9. Decisions needed from the owner

* **D-1 Thin domains.** `stories` (23 documents), `travel` (15) and, partly, `opinion_magazine` cannot reach 62 documents with trusted clean-licence sources. Options: (a) accept an unbalanced 500 as built (the shortfall went to the other domains); (b) raise the Gutenberg cap for `stories` only (e.g. 50 stories + 25 others = 15% in total); (c) merge stories into `opinion_magazine` as "narrative / features"; (d) add the share-alike Wikivoyage as a labelled travel split; (e) commission or find a CC-BY story / travel source (African Storybook is CC BY but under 400 tokens).
* **D-2 GovReport / CRS.** Keep both halves (public benchmark, CRS statement unverified) or only the GAO half? GovReport is a known benchmark and will be familiar to other models' training data.
* **D-3 Common Pile as the umbrella** for 34% of the set (news, textbooks, DOAB): it is citable, but the paper itself warns about licence laundering; keep, or require a per-document licence spot check before release?
* **D-4 Translators.** Confirm the two-system rule with the correction in section 5: MADLAD as the only released system leaves many languages in tier C by its own published scores; allow the OPUS tc-bible supplement to compete for the released-reference slot (bake-off decides per language)?
* **D-5 Release.** Name and licence option 1 or 2 (R-1); whether the SA split is released at all (R-2); canary + private 10% as proposed (R-3).
* **D-6 Hard split.** Build it, when, and with which domain experts for a spot check (section 7).
* **D-7 Downloads.** 50.8 GB for NLLB-3.3B + MADLAD-7B (one at a time, peak 33 GB) or 30 GB with MADLAD-3B; approval to start when the trainer is done.
