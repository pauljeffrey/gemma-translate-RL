---
license: other
license_name: tbd-owner-to-choose
pretty_name: SabiYarn-MT RL best-translation archive
language:
- en
- ach
- afr
- amh
- bam
- bas
- bem
- bin
- dag
- dua
- dyu
- efi
- ewe
- fon
- hau
- her
- ibb
- ibo
- ish
- iso
- kab
- kam
- kea
- kik
- kin
- kmb
- kon
- kqn
- kri
- kua
- lin
- loz
- lua
- lug
- lun
- luo
- mlg
- mos
- nbl
- ndc
- nso
- nus
- nya
- nyn
- oke
- orm
- pcm
- run
- sag
- seh
- sna
- som
- sot
- ssw
- swc
- swh
- tir
- tiv
- tsc
- tsn
- tso
- tum
- twi
- umb
- urh
- ven
- vmw
- wes
- wol
- xho
- yao
- yor
- zul
task_categories:
- translation
tags:
- machine-translation
- african-languages
- reinforcement-learning
- model-outputs
- best-of-n
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files: data/rl_archive.parquet
---

# SabiYarn-MT RL best-translation archive

**Status: PRIVATE draft, prepared 2026-10-24. The owner decides when (and whether) it becomes public. The licence is not chosen yet (see "Licence": TODO(owner)).**

The highest-scoring English-to-African-language translations that the **reinforcement-learning (GRPO) stages** of our Gemma-3-270m translation model saved during training: for each English source window
and language, the best sample seen so far whose reward passed a threshold. They are offered **as a dataset for study** (error analysis, metric research, best-of-n behaviour, candidate pools for
human review). They are **not** a gold standard.

> **Read this before using the data**
> * **Nothing here has been checked by a human.** The translations are samples from a 270M-parameter model that scored high under *learned* metrics. Spot-check before using them as training targets; some are wrong.
> * **The selection metric is the training reward.** For `reward_mode = qe` (and half of `blend`) the reward is AfriCOMET-QE (`masakhane/africomet-qe-stl-1.1`), the same metric the policy was optimised on, so a high score here is partly the result of optimisation against that metric
>   (reward hacking is possible: fluent-looking text that the metric likes). AfriCOMET-QE is *human-judgement-correlated* only on the languages where it was validated. `consensus` rows were selected by agreement (chrF++) with other samples of the same group, which rewards the policy's own mode, not correctness.
> * Rows are the *best* sample, selected from many: the archive is biased towards outputs that exploit the reward. Scores are **not** estimates of model quality.
> * The English sources are windows of web, encyclopaedic and scientific text; their licences differ (table below). Respect them.

## Contents
`data/rl_archive.parquet`: **6,357 rows, 72 languages**, from 2 runs:

| run | rows | langs | steps |
|---|---|---|---|
| final2 | 762 | 63 | 401 |
| rl2c | 5595 | 71 | 1433 |

Rows read from the two archives: {'rl2c': 6142, 'final2': 894}. Selection pipeline: (1) row filters (copy of the source, repetition loops, implausible length ratio) removed 638 rows
(`{'row_filter:repetition': 628, 'row_filter:copy_of_source': 10}`); (2) the same (language, English window) in both runs keeps the higher reward (`also_in_run` records the other run; 9 duplicates); (3) every row was joined with its prompt-corpus
metadata through the archive key `sha1(iso + NFC(source))[:16]` (no row without provenance, none with a non-commercial or unknown licence: 0 dropped for licence); (4) **decontamination**: rows whose English window matches an evaluation English side by exact sentence or
word 8-gram (23 rows; sets: FLORES+ dev and devtest, NTREX-128, MAFAND-MT, SmolSent/SmolDoc, SALT, GlotStoryBook, IbomMT, AfriScience-MT dev and test, AfriLongDoc core and hard documents, our dev sets and the RL-versus-SFT evaluation pool: 57,734 sentences and 973,277 8-grams)
or whose translation matches a benchmark human reference by 8-gram (9 rows) were removed.

### The runs
* `rl2c`: the RL stage that produced the checkpoint `final_1433` (called RL-final 2c in our reports; 1,433 steps; prompts `rl_prompts_v3` from FineWeb-Edu and Wikipedia windows). **No NLLB signal in the reward.**
* `final2`: the final science-stage RL run (401 steps; prompts `rl_prompts_sci_v2`: AfriScience-MT train sentences, PubMed / arXiv / LibreTexts science text, FineWeb-Edu and Wikipedia windows). Its reward could include a calibrated chrF++ term against an NLLB-200 pseudo-reference
  for `consensus` / `blend` languages (the pseudo-references were kept in a reward-only store; **the NLLB text itself is not in this dataset**). The flag `reward_used_nllb_signal` marks the rows for which this could have applied (language weight > 0 in that mode and at least one sentence of the window
  had a pseudo-reference); 3 rows are flagged. `nllb_pseudoref_sentence_coverage` is the share of the window's sentences with a pseudo-reference. NLLB-200 weights are CC BY-NC 4.0 and its outputs are not usable in production: if you need to exclude any row whose reward was touched by it, filter on that flag.

Rows by run and reward mode / by prompt length bin:

| run | reward_mode | rows |
|---|---|---|
| final2 | blend | 57 |
| final2 | consensus | 102 |
| final2 | qe | 603 |
| rl2c | blend | 650 |
| rl2c | consensus | 534 |
| rl2c | qe | 4411 |

| run | length_bin | rows |
|---|---|---|
| final2 | b1024 | 60 |
| final2 | b2048 | 2 |
| final2 | b256 | 257 |
| final2 | b512 | 138 |
| final2 | b64 | 305 |
| rl2c | b1024 | 159 |
| rl2c | b256 | 3435 |
| rl2c | b512 | 1619 |
| rl2c | b64 | 382 |

### Columns
`row_id` (archive key), `language`, `language_name`, `run`, `source_en` (English window; NFC), `translation` (exactly the text the reward scored: first stop cut, special strings removed, language tag stripped, NFC), `reward_mode` (`qe` = AfriCOMET-QE per chunk,
`blend` = half QE, half consensus, `consensus` = chrF++ against the other samples of the group), `reward` (the reward after length, lexical and diacritic penalties; range -2.0 to 2.25), `afri_qe_at_selection` (mean raw AfriCOMET-QE of its chunks; null in `consensus` mode),
`consensus_chrf_at_selection` (mean chrF++ against the valid siblings of its group; null for `qe` languages), `diacritic_deficit`, `lid_ok`, `lid_label` (GlotLID, when the text was long enough to be checked), `n_chunks`, `step_first`, `step_best` (training step at which the stored translation was found),
`times_seen`, `n_improved`, `length_bin` (b64 ... b2048), `stage` (curriculum stage), `domain`, `field` (prompt metadata, when present), `prompt_corpus`, `source_dataset`, `source_licence`, `source_licence_class`, `source_licence_description`, `source_doc_id`, `reward_used_nllb_signal`, `nllb_weight_language_mode`,
`nllb_pseudoref_sentence_coverage`, `also_in_run`, `afri_qe_is_reward_model` (true when AfriCOMET-QE was part of the reward for the row).

An entry is stored only if the sample had status `ok`, no failed chunk (omitted / extra / copied sentence), passed GlotLID where it is checked, had no diacritic deficit, and reached its mode's threshold (qe 1.0; blend 1.0 and consensus chrF++ >= 50; consensus 0.5, i.e. chrF++ >= about 59 against its siblings). A `qe` reward of 1.0 corresponds to the language's median human-reference QE score.

### Languages
| tag | language | rows | mean reward |
|---|---|---|---|
| yor | Yoruba | 476 | 1.529 |
| luo | Luo | 426 | 1.366 |
| lua | Luba-Kasai | 352 | 1.490 |
| twi | Twi | 263 | 1.496 |
| ibo | Igbo | 255 | 1.442 |
| lug | Luganda | 251 | 1.453 |
| orm | Oromo | 209 | 1.346 |
| kik | Kikuyu | 197 | 1.366 |
| xho | Xhosa | 179 | 1.352 |
| nso | Sepedi | 179 | 1.397 |
| wes | Cameroon Pidgin | 172 | 0.815 |
| afr | Afrikaans | 172 | 1.305 |
| sot | Sesotho | 170 | 1.321 |
| tsn | Setswana | 169 | 1.457 |
| nya | Chichewa | 166 | 1.381 |
| lin | Lingala | 147 | 1.351 |
| kmb | Kimbundu | 142 | 1.167 |
| sag | Sango | 132 | 1.194 |
| zul | Zulu | 127 | 1.309 |
| som | Somali | 125 | 1.351 |
| umb | Umbundu | 121 | 1.158 |
| pcm | Pidgin | 118 | 1.408 |
| hau | Hausa | 112 | 1.322 |
| wol | Wolof | 107 | 1.331 |
| ibb | Ibibio | 106 | 0.725 |
| swh | Swahili | 100 | 1.257 |
| tso | Tsonga | 98 | 1.298 |
| run | Kirundi | 98 | 1.345 |
| nbl | South Ndebele | 93 | 0.628 |
| sna | Shona | 88 | 1.316 |
| kin | Kinyarwanda | 85 | 1.318 |
| bem | Bemba | 83 | 1.386 |
| swc | Congo Swahili | 79 | 1.199 |
| ewe | Ewe | 73 | 1.283 |
| fon | Fon | 67 | 1.129 |
| ssw | Swati | 63 | 1.255 |
| tum | Tumbuka | 60 | 1.359 |
| kam | Kamba | 59 | 1.170 |
| seh | Sena | 57 | 0.713 |
| tsc | Tswa | 42 | 0.658 |
| loz | Lozi | 41 | 1.192 |
| tir | Tigrinya | 31 | 1.330 |
| mlg | Malagasy | 25 | 1.305 |
| vmw | Makhuwa | 24 | 0.640 |
| ach | Acholi | 22 | 0.674 |
| ndc | Ndau | 21 | 0.627 |
| yao | Yao | 17 | 0.636 |
| oke | Okpe | 16 | 0.683 |
| nyn | Nyankore | 15 | 0.649 |
| ven | Venda | 14 | 1.132 |
| kqn | Kaonde | 14 | 1.194 |
| kea | Kabuverdianu | 12 | 0.682 |
| mos | Mossi | 11 | 1.141 |
| amh | Amharic | 10 | 1.520 |
| bin | Edo | 9 | 0.711 |
| dag | Dagbani | 7 | 0.673 |
| kri | Krio | 7 | 0.793 |
| efi | Efik | 6 | 1.416 |
| dyu | Dyula | 5 | 1.131 |
| urh | Urhobo | 5 | 0.759 |
| dua | Duala | 5 | 0.672 |
| lun | Lunda | 4 | 1.221 |
| ish | Esan | 3 | 0.680 |
| bam | Bambara | 3 | 1.164 |
| bas | Basaa | 2 | 0.646 |
| iso | Isoko | 2 | 1.286 |
| tiv | Tiv | 2 | 1.248 |
| kua | Kuanyama | 2 | 1.110 |
| kon | Kongo | 1 | 1.145 |
| kab | Kabyle | 1 | 1.159 |
| her | Herero | 1 | 0.712 |
| nus | Nuer | 1 | 1.278 |

### Source licences of the English windows
| source_dataset | source_licence_description | source_licence_class | rows |
|---|---|---|---|
| HuggingFaceFW/fineweb-edu:sample-10BT | ODC-By 1.0 (dataset licence) + Common Crawl terms of use; copyright of the underlying web pages not cleared | odc_by | 3959 |
| wikimedia/wikipedia:20231101.en | CC BY-SA 4.0 (Wikipedia text, also GFDL); share-alike | share_alike | 2073 |
| dsfsi/afriscience_mt:train(en) | Apache-2.0 (AfriScience-MT dataset card); underlying papers' own licences not listed | permissive | 162 |
| common-pile/libretexts | CC BY (Common Pile; attribution to the original authors) | permissive | 64 |
| common-pile/pubmed | CC BY (Common Pile; attribution to the original authors) | permissive | 61 |
| HuggingFaceFW/finetranslations(en) | ODC-By 1.0 (dataset licence) + Common Crawl terms of use; copyright of the underlying web pages not cleared | odc_by | 15 |
| common-pile/arxiv_papers | CC BY (Common Pile; attribution to the original authors) | permissive | 10 |
| Sunbird/salt:text-all | CC BY-SA 4.0 (SALT, Sunbird AI); share-alike | share_alike | 4 |
| common-pile/arxiv_papers | CC0 | permissive | 3 |
| common-pile/pubmed | CC0 | permissive | 3 |
| gfissore/arxiv-abstracts-2021 | CC0 1.0 | permissive | 3 |

`odc_by` rows (FineWeb-Edu, FineTranslations) are licensed ODC-By 1.0 *as a dataset*, and carry the Common Crawl terms of use; the copyright of the underlying web pages is not cleared by that licence. If you cannot accept that risk, drop `source_licence_class == "odc_by"` (about 3,974 rows) before use. Wikipedia and SALT rows are CC BY-SA 4.0: translations of them are adaptations and must be shared under CC BY-SA.

## Limitations
Model errors and hallucination; metric bias (reward model and consensus); no human verification; coverage uneven across languages (72 of 77 trained languages have at least one row; the archive thresholds were not reached for the others); rows are a small, selection-biased sample;
windows are English web/science text, not a balanced domain mix; a restart of the trainer can double-count `times_seen`; evaluation rollouts were not archived; the NLLB flag is an upper bound at row level (chunk-level use may differ).

## Licence of this dataset: TODO(owner): choose licence

The front matter says `license: other` on purpose: **no licence has been chosen yet** and none is claimed.

TODO(owner): choose licence. Recommendation of the preparing agent (not a legal opinion):

* **CC BY-SA 4.0** for the whole repository **as is (it contains no non-commercial source)**. It is the safest single licence that is compatible
  with every other part: public-domain texts, CC BY / OGL-3.0 texts (attribution kept in every row), CC BY-SA texts (Wikipedia, Simple English Wikipedia, Wikivoyage, Wikijunior,
  NTREX-128, SALT: translations and other adaptations of them must be shared under CC BY-SA), CC BY 4.0 (SmolSent) and Apache-2.0 (AfriScience-MT) material. The machine translations
  and scores are our own contribution and can carry the same licence.
* **CC BY-NC-SA 4.0** for the whole repository **if the `noncommercial/` folders stay** (no non-commercial source is included in this repository).
* A permissive licence (CC BY 4.0, CC0, MIT) is **not possible** while any CC BY-SA or CC BY-NC source is included.
* Model outputs: the weights of the translating model are not an input licence issue here, but the owner should confirm the model's own licence (Gemma terms) before choosing.
* Pseudo-references from NLLB-200 / TranslatePsy are **not** part of this repository and were not used for any score; their licence (non-commercial / not usable in production) therefore does not apply to it.


## How to cite
TODO(owner): BibTeX of the SabiYarn-MT report. Cite AfriCOMET (Wang et al. 2024), SSA-COMET (Li et al. 2025), GRPO (Shao et al. 2024) as relevant, and the source corpora in the licence table.

Canary: this archive is derived from training prompts and must not be used to evaluate the model that produced it.
