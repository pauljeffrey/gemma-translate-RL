# Instruction / chat / QA intake for the clean SFT rerun (2026-10-31)

Agent: intake2_instr. Instance: vast $BOX_HOST (CPU only, GPU, `scores_*`, `work/gate`, `state/` and `logs/rescore_all.log` untouched, `nice -n 10`, at most 20 workers). Code: `data_pipeline/sft_clean/chain/intake2_instr.py` (+ `intake2_instr_sources.py`, `intake2_instr_report.py`, `intake2_aya.py`, `intake2_aya_emit.py`), the same files are on the instance in `/workspace/work/code/sft_clean/chain/`. Copy of this report: `/workspace/work/logs/intake2_instr_2026-10-31.md`; progress log `/workspace/work/logs/intake2_instr.progress`; per-source JSON reports `/workspace/work/instr/_state/<source>.report.json`.

## 1. Result in one paragraph

54 Hub repos (sources; the three `0xnu/*` repos were tested and dropped before the run, see section 5) were streamed (nothing raw kept), cleaned, decontaminated and deduplicated. **Main path `/workspace/work/instr/<source>/part-NNNN.parquet`: 2,422,836 conversations in 44 languages, about 838M Gemma-3 tokens (chat template of `google/gemma-3-270m-it`, estimate from a 2,000-row sample per language), of which only about 0.63M rows are not machine-translated / LLM-generated.** A further 0.84M synthetic rows (about 342M tokens) were moved to `/workspace/work/instr/_held_synthetic/<source>/` by the "synthetic only if it is the only source" rule (section 4); that move is reversible (files are only relocated). Of the 85 languages of `languages.py`, 37 have at least one row (48 have none, list in section 6); **new iso3 tags (not in `languages.py`)**: `arz` (Egyptian Arabic), `ary` (Moroccan Arabic), `aeb` (Tunisian Arabic), `bci` (Baoule), `din` (Dinka), `teo` (Ateso), `adh` (Dhopadhola).

The two headline finds of the audit are much smaller than their repo sizes suggest:
* `Mwanzau/Tumbuka_Instruction_Tuning` (2.8M): `train.parquet` (2.02M rows) is the **English original** (SuperNatural-Instructions style); only `translated_tumbuka_dataset.parquet` (705k rows) is Tumbuka, and it is a machine translation (system not stated; Tumbuka is an NLLB-200 language, so treat it as possibly NLLB-derived). Tumbuka had 0 instruction rows before; 627k now (400k cap of `tumbuka_instruction` + 227k of the near-identical `Mwanzau/Tumbuka_language`, the rest of that repo is a duplicate). **All flagged `synthetic=true`; the lead must decide whether Tumbuka MT is acceptable (it is the only source for `tum`).**
* `akoksal/muri-it` (2.2M): that is 200 languages; the 36 African language folders of `akoksal/muri-it-language-split` hold 203,112 train rows, 150,987 survive (22% were removed by the protected-index filter, mostly R1/R2 hits on Wikipedia-derived outputs). Zero overlap (by normalised user+assistant text) with the 1,062,579 non-translation rows of `<HF_ORG_LEGACY>/sft-all` + `<HF_ORG>/sft-new`. Outputs are human-written; the **instruction side is LLM-generated from an NLLB English translation and translated back (arXiv:2409.12958), so rows with `dataset_name == MRI` are flagged `synthetic=true` and `licence_raw` says so**; 38.7k of them were held back.

## 2. What was done to every row

1. Conversion to `messages` (JSON list of `{role, content}`, roles `user`/`assistant`, a first `system` message only where the source has one: `setswana_gpt4o`), `task_type` in instruction|chat|qa|other (no summarization source passed the licence/benchmark screen), `licence` (commercial_ok|share_alike|unspecified; no row is share_alike, the unspecified ones are repos without a declared licence), `licence_raw` (what the card says plus provenance remarks), `synthetic`, `n_chars`, `id` = `<source>:<file>:<row index>[:<native id>]`. Ethiopic/Arabic/Latin script check of the assistant text, English-stopword check of the assistant text (drop when >= 25% English function words), the declared language is the repo's or the folder/column's (muri, aya, webfaq, kakugo, fikira, kiya, ultrachat).
2. Cleaning: control characters removed, strip, `#VALUE!` spreadsheet artefacts removed, rows with API-error junk (`Error code: 400`, `invalid_request_error`, chat-template tokens) dropped; first user turn < 8 chars, any assistant turn < 2 chars, total > 12,000 chars dropped; roles repaired (alternation, leading assistant turns removed, trailing user turn removed); `<think>` blocks of the LLM chats (kakugo, amharic ChatML) removed.
3. Decontamination: `protected_index.npz` (version `protected-2026-10-28.3`, sha256 checked against the manifest) via `protected_index_loader.load` + `Index.check_texts`, the same call as `extra_sources.Filter`; **every turn (system, user and assistant) is tested and the row is dropped on any R1/R2/R3 hit**. 420k rows were removed this way (R3 shared 8-gram 289.7k, R1 exact sentence 87.0k, R2 contained sentence 43.3k): the strict 8-gram rule is what makes the numbers below small.
4. Exact dedupe on `xxh64(normalised user texts, normalised assistant texts)` within each source and across sources in priority order (aya_dataset, muri_it, tydiqa, webfaq, yoseali, ... , kakugo, fikira, kiya, bsc); the first source keeps the row (`alpaca_somali_b/c` are 99% copies of `alpaca_somali_a`, `kiya_sft` swh/yor/amh overlaps 121k rows with other sets). `muri_it` was also checked against the old build (0 hits).
5. Cap 400,000 rows per (source, language) (deterministic: lowest `xxh64(id)` kept): hit by `amharic_sft_combined` (1.39M candidates -> 400k) and `tumbuka_instruction` (593k -> 400k).
6. Shards of 100k rows, zstd, schema `id, source_dataset, iso, messages, task_type, licence, licence_raw, synthetic, n_chars`.

Reject counts over all sources (rows): 

| reason | rows |
|---|---:|
| rowfilter | 812499 |
| protected_R3 | 289711 |
| dup_in_file | 209102 |
| lang_not_in_scope | 96719 |
| protected_R1 | 86966 |
| not_african | 49025 |
| no_iso | 47126 |
| protected_R2 | 43327 |
| wrong_script | 17138 |
| assistant_english | 3676 |
| too_long | 2904 |
| short_assistant | 2889 |
| short_user | 1478 |
| empty_turns | 134 |
| junk_marker | 57 |

(`rowfilter` = rows deliberately not taken: tydiqa non-Swahili languages, webfaq pairs with similarity < 0.7, `yoseali` bare "Complete the following task" continuations, Kiya-SFT English rows, Baoule non-bci directions, AmaSquad unanswerable questions; `not_african`/`lang_not_in_scope` = eng/fra/other languages inside multilingual repos, mostly `aya_dataset`.)

## 3. Sources (priority order = dedupe order)

`synthetic` = machine-translated or LLM-generated on at least one side. "mixed" = muri_it (see section 1). `kept in total` includes the rows later held back.

| source | repo | licence class | synthetic | rows read | kept in total | held back (synthetic) | languages (rows kept) |
|---|---|---|---|---:|---:|---:|---|
| aya_dataset | `CohereLabs/aya_dataset` | commercial_ok | no | 202.4k | 44.6k | 0 | mlg 13.8k, yor 6.4k, som 5.7k, ary 5.3k, hau 3.4k, wol 2.7k, zul 1.6k, ibo 1.3k, sna 1.3k, amh 1.2k, nya 671, arz 525, swh 336, xho 332 ... |
| muri_it | `akoksal/muri-it-language-split` | commercial_ok | mixed | 203.1k | 151.0k | 38.7k | swh 16.0k, mlg 12.9k, afr 12.6k, amh 12.5k, ibo 11.0k, zul 10.2k, sna 9.9k, nso 8.5k, hau 7.7k, som 7.2k, kin 6.0k, lug 5.3k, xho 4.1k, yor 4.1k ... |
| tydiqa_swh | `google-research-datasets/tydiqa` | commercial_ok | no | 49.9k | 2.7k | 0 | swh 2.7k |
| webfaq_v2 | `PaDaS-Lab/webfaq-v2` | unspecified | no | 148.4k | 60.7k | 0 | afr 32.5k, swh 12.5k, mlg 7.8k, amh 7.6k, som 169, yor 40, arz 33 |
| yoseali_amh | `YoseAli/amharic-llm-training-data` | commercial_ok | no | 761.5k | 102.5k | 0 | amh 102.5k |
| african_ultrachat | `masakhane/african-ultrachat` | commercial_ok | yes | 53.9k | 51.6k | 4.8k | sna 4.9k, xho 4.9k, kin 4.8k, amh 4.8k, zul 4.8k, swh 4.8k, som 4.7k, sot 4.6k, yor 4.5k, hau 4.4k, ibo 4.4k |
| uganda_multilingual_qa | `BUAIR/Uganda-Multilingual-QA` | commercial_ok | yes | 17.0k | 17.0k | 0 | adh 4.3k, teo 4.3k, lug 4.3k, nyn 4.3k |
| twi_everyday_conv | `ik/twi_everyday_conversation` | commercial_ok | no | 1.7k | 1.6k | 0 | twi 1.6k |
| oromiffa_convo | `kkenna/oromiffa_convo_data` | unspecified | yes | 698 | 596 | 0 | orm 596 |
| pidgin_clean | `Ephraimmm/pidgin_clean` | commercial_ok | no | 2.1k | 1.4k | 0 | pcm 1.4k |
| lingala_alpaca | `Congo-digital-service/Lingala-Alpaca-Dataset-Augmented` | unspecified | no | 5.7k | 3.9k | 0 | lin 3.9k |
| kon_disolo_qa | `Svngoku/disolo-ya-afrika-qa` | commercial_ok | yes | 7.8k | 7.8k | 0 | kon 7.8k |
| tigqa | `Hailay/tigqa-extractive-qa` | commercial_ok | no | 1.7k | 1.7k | 0 | tir 1.7k |
| kin_alpaca_ukweli | `map-boy/ukwelitimes-kinyarwanda-alpaca` | unspecified | yes | 504 | 502 | 0 | kin 502 |
| kin_roleplay | `jojo-ai-mst/Roleplay-Kinyarwanda` | commercial_ok | yes | 1.9k | 1.9k | 0 | kin 1.9k |
| setswana_gpt4o | `OxxoCodes/gpt4o-setswana-instruct` | unspecified | yes | 1.6k | 1.4k | 0 | tsn 1.4k |
| luganda_fln | `CraneAILabs/luganda-fln-training-data` | commercial_ok | yes | 1.4k | 651 | 0 | lug 651 |
| ugandaedu_luganda | `gimmy256/UgandaEdu-5K` | commercial_ok | yes | 568 | 568 | 0 | lug 568 |
| bambara_lm_qa | `oza75/bambara-lm-qa` | unspecified | no | 824.5k | 381.0k | 0 | bam 381.0k |
| bambara_conv | `Makan09/Bambara-dataset_conversation` | commercial_ok | yes | 90.1k | 77.6k | 77.6k | bam 77.6k |
| som_conversations | `yacdev/somali-100k-saaxiib-conversations-v2` | commercial_ok | yes | 100.0k | 17.6k | 0 | som 17.6k |
| alpaca_twi | `ik/twi_alpaca-cleaned` | unspecified | yes | 51.7k | 50.1k | 0 | twi 50.1k |
| alpaca_swahili | `Mollel/alpaca-swahili` | unspecified | yes | 51.8k | 50.9k | 0 | swh 50.9k |
| alpaca_zulu | `africa-intelligence/alpaca-zulu` | commercial_ok | yes | 52.0k | 51.7k | 0 | zul 51.7k |
| alpaca_wolof | `ngia/alpaca-data-in-wolof` | commercial_ok | yes | 47.5k | 46.3k | 0 | wol 46.3k |
| alpaca_somali_a | `burtugeey/alpaca_somali_final` | commercial_ok | yes | 96.3k | 88.0k | 0 | som 88.0k |
| alpaca_somali_b | `burtugeey/Somali_alpaca` | commercial_ok | yes | 96.6k | 584 | 0 | som 584 |
| alpaca_somali_c | `burtugeey/Somali_dataset` | commercial_ok | yes | 44.8k | 83 | 0 | som 83 |
| math_malagasy | `Lo-Renz-O/grade-school-math-instructions-Malagasy` | commercial_ok | yes | 8.7k | 8.5k | 8.5k | mlg 8.5k |
| amharic_freedom_alpaca | `addisai/FreedomIntelligence-alpaca-amharic` | unspecified | yes | 50.0k | 48.8k | 48.8k | amh 48.8k |
| amharic_finetome | `addisai/FineTome-single-turn-dedup-amharic` | commercial_ok | yes | 83.3k | 79.9k | 79.9k | amh 79.9k |
| amharic_sft_combined | `abdukuzi45/amharic-sft-combined-2026-with-cot` | commercial_ok | yes | 1.52M | 400.0k | 400.0k | amh 400.0k |
| tumbuka_instruction | `Mwanzau/Tumbuka_Instruction_Tuning` | commercial_ok | yes | 704.8k | 400.0k | 0 | tum 400.0k |
| tumbuka_language | `Mwanzau/Tumbuka_language` | commercial_ok | yes | 739.5k | 227.6k | 0 | tum 227.6k |
| amasquad_amh | `nebhailema/AmaSquad` | commercial_ok | yes | 130.3k | 73.7k | 73.7k | amh 73.7k |
| baoule_alpaca | `Tree-AI-lab/fr_bci_4d_instruct` | commercial_ok | yes | 55.0k | 16.3k | 0 | bci 16.3k |
| kakugo_amh | `ptrdvn/kakugo-amh` | commercial_ok | yes | 36.9k | 32.9k | 32.9k | amh 32.9k |
| kakugo_arz | `ptrdvn/kakugo-arz` | commercial_ok | yes | 41.9k | 38.0k | 0 | arz 38.0k |
| kakugo_hau | `ptrdvn/kakugo-hau` | commercial_ok | yes | 29.4k | 28.4k | 0 | hau 28.4k |
| kakugo_ibo | `ptrdvn/kakugo-ibo` | commercial_ok | yes | 37.1k | 35.8k | 0 | ibo 35.8k |
| kakugo_kin | `ptrdvn/kakugo-kin` | commercial_ok | yes | 38.4k | 37.8k | 0 | kin 37.8k |
| kakugo_nya | `ptrdvn/kakugo-nya` | commercial_ok | yes | 38.2k | 37.6k | 0 | nya 37.6k |
| kakugo_plt | `ptrdvn/kakugo-plt` | commercial_ok | yes | 40.8k | 39.8k | 39.8k | mlg 39.8k |
| kakugo_run | `ptrdvn/kakugo-run` | commercial_ok | yes | 40.7k | 40.1k | 0 | run 40.1k |
| kakugo_sna | `ptrdvn/kakugo-sna` | commercial_ok | yes | 40.7k | 40.2k | 0 | sna 40.2k |
| kakugo_sot | `ptrdvn/kakugo-sot` | commercial_ok | yes | 38.8k | 37.9k | 0 | sot 37.9k |
| kakugo_swa | `ptrdvn/kakugo-swa` | commercial_ok | yes | 40.0k | 39.2k | 0 | swh 39.2k |
| kakugo_tir | `ptrdvn/kakugo-tir` | commercial_ok | yes | 36.6k | 33.1k | 0 | tir 33.1k |
| kakugo_xho | `ptrdvn/kakugo-xho` | commercial_ok | yes | 41.6k | 41.0k | 0 | xho 41.0k |
| kakugo_yor | `ptrdvn/kakugo-yor` | commercial_ok | yes | 41.1k | 40.0k | 0 | yor 40.0k |
| kakugo_zul | `ptrdvn/kakugo-zul` | commercial_ok | yes | 41.4k | 40.8k | 0 | zul 40.8k |
| fikira | `vamboai/fikira` | commercial_ok | yes | 50.0k | 49.7k | 5.0k | zul 5.0k, hau 5.0k, ibo 5.0k, swh 5.0k, amh 5.0k, xho 5.0k, aeb 5.0k, sna 5.0k, kin 4.9k, yor 4.9k |
| kiya_sft | `NaolBM/Kiya-SFT` | commercial_ok | yes | 747.3k | 153.1k | 31.1k | orm 51.6k, yor 51.5k, amh 31.1k, swh 18.9k |
| bsc_m_personas_swh | `BSC-LT/m-personas` | commercial_ok | yes | 27.9k | 27.5k | 0 | swh 27.5k |


## 4. The "synthetic only when it is the only source" rule

Applied per language after the run (`intake2_instr.py hold --threshold 20000`): synthetic rows of a language are moved to `_held_synthetic/` when that language has >= 20,000 non-synthetic rows over all sources. That holds for **afr** (32.5k, `webfaq_v2`), **amh** (111k), **bam** (381k, `bambara_lm_qa`, templated LM tasks) and **mlg** (21.6k). All other languages have fewer than 20k non-synthetic rows, so their synthetic rows stay in the main path flagged `synthetic=true` (the lead can filter on the column). Held rows: 840,774. Per (source, language): `/workspace/work/instr/_state/hold.json`. To undo: move the files back from `_held_synthetic/<source>/` (file names are unchanged).

## 5. Skipped sources and why

| source | reason |
|---|---|
| `CohereLabs/aya_collection`, `janavivekariya/aya_collection`, `vutuka/aya_african_alpaca`, `CohereLabs/aya_collection_language_split` (as an instruction source) | translated subsets are NLLB-3.3B output (owner rule: no NLLB-derived); templated subsets contain AfriQA / AfriSenti / MasakhaNews (protected benchmarks). The human-annotated `CohereLabs/aya_dataset` (Apache-2.0, not the collection) **is** included. The aligned en->X pairs are quarantined, see section 7. |
| `CohereLabs/xP3x`, `bigscience/xP3all`, `bigscience/xP3mt`, `Muennighoff/xP3x-sample`, `taresco/xP3x_african_subset` | prompt templates over many benchmark datasets (FLORES, AfriQA, MasakhaNews, SIB, XNLI ...) including their eval splits, cannot be separated without per-task split metadata; unsure -> skipped. |
| `nileagi/swahili-language-exposure`(-v2), `abdukuzi45/amharic-cpt-packed-text-2048-v9`, `stanford-oval/ccnews`, `olaverse/diacnet-1.1-train`, `bitter-aloe/trc-testimonies`, `zabr946/Chatbot-Url` (Wikipedia dumps, 72 GB) | pretraining-style raw text, not instruction/chat/QA. |
| `OxxoCodes/pulabert-dataset`, `lightblue/reranker_continuous_filt_max7_train` | `source-category: opus-nllb` (NLLB-derived) / retrieval-reranker pairs. |
| `abdukuzi45/amharic-sft-mixed-cot-2026`, `abdukuzi45/Kuzi-Amharic-Uncensored-Dataset` | "uncensored AI assistant" persona system prompts, no licence declared, safety review needed. |
| `0xnu/hausa`, `0xnu/igbo`, `0xnu/yoruba` | tested on 1,000 rows each: LLM-generated sentence-building tasks with word-salad outputs and 36-42% exact duplicates; dropped as low quality (not pipeline errors). |
| `amphora/Open-R1-Mulitlingual-SFT` | multilingual maths with English reasoning traces; no African-language content. |
| `anakin87/tulu-3-sft-mixture-with-language` | in the sampled shard no African language reaches the top 40 language tags (African rows negligible). |
| `amidblue/AfriQuAD` | AfriQA / KenSwQuAD derived (ids `sw_...`, has test split): benchmark-derived. |
| `MaLA-LM/MassiveSumm_short/long` | gated, no licence, copyright risk. `reuben256/alpaca-luganda-cleaned` is gated (403). |
| `rasyosef/xlsum_amharic_summarization` | XL-Sum is CC BY-NC-SA (non-commercial). `lelapa/Inkuba-instruct`, `ghananlpcommunity/*`, `NaathNLP/*`, `ServiceNow-AI/M2Lingual` ... are NC (audit section 5). |
| `Mikecyane/Kinyarwanda_chat` (a private WhatsApp export), `martinturuta/safi-kinyarwanda-conversations` (audio annotations), `sartifyllc/pawa-function-calling-60k` (tool-call format, no licence), `SIMBA9657/tigrinya-haddas-dataset` and `Shads229/french-fongbe-corpus` (translation tasks whose answer is English / whose prompt is French: belong to the pair stream) | not suitable as African-language instruction data. |
| `MBZUAI-Paris/Egyptian-*`, `wghezaiel/*Tunisian*`, `safouaneb/arywiki-instruct`, `Rabe3/*` | Arabic dialects, not in the 85-language table, unknown licences and mostly synthetic; Arabic-dialect rows enter only through `aya_dataset`, `kakugo_arz`, `fikira` and `webfaq_v2` as the new tags above. |
| long tail | about 250 further catalogue `new_instruction` repos with < 40k rows each (many are Arabic-dialect, Taiwanese/Urdu mis-tags, ASR transcripts or `aar` mis-tags) were not processed; the largest-first order was followed down to the 20-minute-per-source limit and the languages the stream is short of. Languages with no instruction source at all in the catalogue: see table 6 (everything of the 85 not listed there). |

## 6. Rows and estimated Gemma-3 tokens per language

Estimate = rows x mean tokens per row of a random 2,000-row sample per language (or all rows when fewer), `tokenizer.apply_chat_template` of `google/gemma-3-270m-it` incl. BOS and turn markers; mean tokens are computed per (source, language) from the sample, falling back to the language mean when a source has fewer than 20 sampled rows. Long-form data dominates: `african_ultrachat` ~1,100 tokens/row, `tumbuka` ~260, `amharic` ~800 (Ethiopic script tokenises poorly).

| iso | main-path rows | est. Gemma tokens | synthetic rows | synthetic tokens | held-back synthetic rows | held-back tokens | main-path sources (rows) |
|---|---:|---:|---:|---:|---:|---:|---|
| amh | 111.3k | 89.61M | 0 | 0 | 688.7k | 272.76M | yoseali_amh 102.5k, webfaq_v2 7.6k, aya_dataset 1.2k |
| tum | 627.7k | 164.22M | 627.6k | 164.18M | 0 | 0 | tumbuka_instruction 400.0k, tumbuka_language 227.6k, muri_it 127 |
| swh | 177.8k | 62.30M | 159.1k | 58.96M | 0 | 0 | alpaca_swahili 50.9k, kakugo_swa 39.2k, bsc_m_personas_swh 27.5k, kiya_sft 18.9k, muri_it 16.0k, webfaq_v2 12.5k |
| bam | 381.1k | 34.43M | 0 | 0 | 78.2k | 22.16M | bambara_lm_qa 381.0k, muri_it 116 |
| yor | 111.5k | 55.60M | 101.1k | 50.95M | 0 | 0 | kiya_sft 51.5k, kakugo_yor 40.0k, aya_dataset 6.4k, fikira 4.9k, african_ultrachat 4.5k, muri_it 4.1k |
| zul | 114.0k | 45.93M | 109.9k | 44.82M | 0 | 0 | alpaca_zulu 51.7k, kakugo_zul 40.8k, muri_it 10.2k, fikira 5.0k, african_ultrachat 4.8k, aya_dataset 1.6k |
| mlg | 21.6k | 2.54M | 0 | 0 | 61.2k | 36.80M | aya_dataset 13.8k, webfaq_v2 7.8k |
| ibo | 57.5k | 37.27M | 53.4k | 35.35M | 0 | 0 | kakugo_ibo 35.8k, muri_it 11.0k, fikira 5.0k, african_ultrachat 4.4k, aya_dataset 1.3k |
| kin | 55.9k | 35.31M | 53.4k | 34.60M | 0 | 0 | kakugo_kin 37.8k, muri_it 6.0k, fikira 4.9k, african_ultrachat 4.8k, kin_roleplay 1.9k, kin_alpaca_ukweli 502 |
| sna | 61.3k | 34.20M | 57.5k | 33.30M | 0 | 0 | kakugo_sna 40.2k, muri_it 9.9k, fikira 5.0k, african_ultrachat 4.9k, aya_dataset 1.3k |
| xho | 55.3k | 34.05M | 52.2k | 33.38M | 0 | 0 | kakugo_xho 41.0k, fikira 5.0k, african_ultrachat 4.9k, muri_it 4.1k, aya_dataset 332 |
| som | 124.0k | 33.65M | 117.9k | 32.16M | 0 | 0 | alpaca_somali_a 88.0k, som_conversations 17.6k, muri_it 7.2k, aya_dataset 5.7k, african_ultrachat 4.7k, alpaca_somali_b 584 |
| sot | 43.1k | 30.02M | 43.1k | 30.02M | 0 | 0 | kakugo_sot 37.9k, african_ultrachat 4.6k, muri_it 641 |
| hau | 48.9k | 27.58M | 45.5k | 27.13M | 0 | 0 | kakugo_hau 28.4k, muri_it 7.7k, fikira 5.0k, african_ultrachat 4.4k, aya_dataset 3.4k |
| nya | 41.7k | 24.75M | 38.5k | 24.13M | 0 | 0 | kakugo_nya 37.6k, muri_it 3.4k, aya_dataset 671 |
| run | 40.7k | 24.56M | 40.2k | 24.23M | 0 | 0 | kakugo_run 40.1k, muri_it 649 |
| tir | 34.9k | 18.41M | 33.2k | 16.17M | 0 | 0 | kakugo_tir 33.1k, tigqa 1.7k, muri_it 147 |
| arz (new) | 38.6k | 17.55M | 38.0k | 17.45M | 0 | 0 | kakugo_arz 38.0k, aya_dataset 525, webfaq_v2 33 |
| orm | 53.6k | 17.25M | 53.6k | 17.25M | 0 | 0 | kiya_sft 51.6k, muri_it 1.4k, oromiffa_convo 596 |
| twi | 51.7k | 16.62M | 50.1k | 16.57M | 0 | 0 | alpaca_twi 50.1k, twi_everyday_conv 1.6k |
| wol | 51.0k | 13.86M | 47.1k | 13.29M | 0 | 0 | alpaca_wolof 46.3k, aya_dataset 2.7k, muri_it 1.9k |
| afr | 32.5k | 2.81M | 0 | 0 | 12.6k | 10.63M | webfaq_v2 32.5k |
| bci (new) | 16.3k | 2.82M | 16.3k | 2.82M | 0 | 0 | baoule_alpaca 16.3k |
| tsn | 4.9k | 2.43M | 2.4k | 1.90M | 0 | 0 | muri_it 3.5k, setswana_gpt4o 1.4k |
| lug | 10.7k | 2.34M | 8.2k | 1.56M | 0 | 0 | muri_it 5.3k, uganda_multilingual_qa 4.3k, luganda_fln 651, ugandaedu_luganda 568 |
| kbp | 1.4k | 1.28M | 1.4k | 1.28M | 0 | 0 | muri_it 1.4k |
| aeb (new) | 5.0k | 1.02M | 5.0k | 1.02M | 0 | 0 | fikira 5.0k |
| nso | 8.6k | 926.7k | 6.0k | 632.9k | 0 | 0 | muri_it 8.5k, aya_dataset 135 |
| kon | 8.1k | 720.0k | 8.1k | 720.0k | 0 | 0 | kon_disolo_qa 7.8k, muri_it 336 |
| lin | 5.5k | 608.7k | 602 | 66.4k | 0 | 0 | lingala_alpaca 3.9k, muri_it 1.6k |
| ful | 1.4k | 408.3k | 1.4k | 408.3k | 0 | 0 | muri_it 1.4k |
| tso | 3.1k | 407.3k | 560 | 74.4k | 0 | 0 | muri_it 3.1k |
| nyn | 4.3k | 351.2k | 4.3k | 351.2k | 0 | 0 | uganda_multilingual_qa 4.3k |
| teo (new) | 4.3k | 317.7k | 4.3k | 317.7k | 0 | 0 | uganda_multilingual_qa 4.3k |
| adh (new) | 4.3k | 301.3k | 4.3k | 301.3k | 0 | 0 | uganda_multilingual_qa 4.3k |
| ewe | 841 | 271.3k | 841 | 271.3k | 0 | 0 | muri_it 841 |
| ary (new) | 5.3k | 212.5k | 0 | 0 | 0 | 0 | aya_dataset 5.3k |
| ssw | 398 | 194.4k | 398 | 194.4k | 0 | 0 | muri_it 398 |
| din (new) | 255 | 176.5k | 255 | 176.5k | 0 | 0 | muri_it 255 |
| pcm | 1.4k | 151.8k | 0 | 0 | 0 | 0 | pidgin_clean 1.4k |
| fon | 422 | 124.9k | 293 | 86.7k | 0 | 0 | muri_it 422 |
| ven | 581 | 97.4k | 581 | 97.4k | 0 | 0 | muri_it 581 |
| sag | 94 | 9.4k | 94 | 9.4k | 0 | 0 | muri_it 94 |
| kik | 7 | 872 | 0 | 0 | 0 | 0 | muri_it 7 |
| **total** | 2.42M | 837.64M | 1.79M | 686.23M | 840.8k | 342.35M | |

Languages of `languages.py` with **no** instruction/chat/QA row from this intake (48): `efi ibb urh bem nbl nde tiv umb kmb mos luo kam bin dag oke shi dyu bbj kea fat knc gaa kab nus swc lua kqn lun iso loz wal ach tsc vmw kua seh her yao sid ndc men kri luy ish dua bas wes tzm`; the catalogue has no commercial-clear instruction data for them (their only text in this build is the pair stream). Tags present but tiny (< 1.5k rows): `kik ssw sag ven fon ewe pcm ful`.

## 7. Aya collection alignment verdict (200-row check requested; done on every row of the first files)

Method (`intake2_aya.py`, result `/workspace/work/instr/_work/aya_alignment_v2.json`, copy in `logs/`): for each African split of `CohereLabs/aya_collection_language_split` read `dataset_name` and `id` of the first ~3.6-4.0M train rows and compare position by position with the English split; then 200 aligned positions per split were read in full (content check) plus 60 rows at fixed offsets for hausa / xhosa / swahili (proper-noun overlap English vs X on aligned rows: 0.43-0.45 in Flan-Coqa passages against 0.00 for shuffled pairs; other datasets contain few capitalised words, so the id equality is the evidence there).

**Verdict: PARTLY ALIGNED. Row k of the large African splits is the translation of row k of the English split for contiguous blocks only: about 64% of the first 3.6M positions (20 of 36 blocks of 100k rows are 100% aligned; 8 blocks 0%; the rest partial), then the alignment is lost for good after ~3.1M. The owner's "first 1-3M rows are direct translations" is therefore true for the translated datasets inside the aligned blocks but not uniformly: position-based pairing would create wrong pairs in the misaligned blocks, so the pairs must be built by joining on (dataset_name, sub_dataset_name, id, template_id).** The six small splits (bemba, fon, kinyarwanda, nyanja, twi, wolof; 231-7k rows) are native / templated data (AfriQA-inst, AfriSenti-inst, Aya-Dataset) and are not translations at all. Kanuri aligns only 35%.

| Aya split | iso | rows in first train file(s) | 100k-blocks fully aligned (same dataset_name + id at the same position) | blocks with no alignment | aligned share of rows | verdict |
|---|---|---:|---:|---:|---:|---|
| afrikaans | afr | 3,577,285 | 20 of 36 | 8 | 0.644 | positionally aligned in contiguous blocks; join by key |
| amharic | amh | 3,589,993 | 20 of 36 | 8 | 0.642 | positionally aligned in contiguous blocks; join by key |
| bemba | bem | 231 | 0 of 1 | 1 | 0.000 | NOT a translation of English (small native/templated split: AfriQA / AfriSenti / Aya-Dataset inst.) |
| central_kanuri | knc | 4,000,000 | 12 of 40 | 23 | 0.353 | positionally aligned in contiguous blocks; join by key |
| fon | fon | 250 | 0 of 1 | 1 | 0.000 | NOT a translation of English (small native/templated split: AfriQA / AfriSenti / Aya-Dataset inst.) |
| hausa | hau | 3,608,883 | 20 of 37 | 9 | 0.639 | positionally aligned in contiguous blocks; join by key |
| igbo | ibo | 3,597,292 | 20 of 36 | 8 | 0.641 | positionally aligned in contiguous blocks; join by key |
| kinyarwanda | kin | 6,859 | 0 of 1 | 1 | 0.000 | NOT a translation of English (small native/templated split: AfriQA / AfriSenti / Aya-Dataset inst.) |
| northern_sotho | nso | 3,572,506 | 20 of 36 | 8 | 0.645 | positionally aligned in contiguous blocks; join by key |
| nyanja | nya | 688 | 0 of 1 | 1 | 0.000 | NOT a translation of English (small native/templated split: AfriQA / AfriSenti / Aya-Dataset inst.) |
| plateau_malagasy | mlg | 3,586,962 | 20 of 36 | 8 | 0.643 | positionally aligned in contiguous blocks; join by key |
| shona | sna | 3,576,309 | 20 of 36 | 8 | 0.645 | positionally aligned in contiguous blocks; join by key |
| somali | som | 3,582,111 | 20 of 36 | 8 | 0.644 | positionally aligned in contiguous blocks; join by key |
| southern_sotho | sot | 3,572,365 | 20 of 36 | 8 | 0.645 | positionally aligned in contiguous blocks; join by key |
| swahili | swh | 3,580,061 | 20 of 36 | 8 | 0.644 | positionally aligned in contiguous blocks; join by key |
| tamasheq | tmh | 3,572,365 | 20 of 36 | 8 | 0.645 | positionally aligned in contiguous blocks; join by key |
| twi | twi | 7,320 | 0 of 1 | 1 | 0.000 | NOT a translation of English (small native/templated split: AfriQA / AfriSenti / Aya-Dataset inst.) |
| wolof | wol | 3,146 | 0 of 1 | 1 | 0.000 | NOT a translation of English (small native/templated split: AfriQA / AfriSenti / Aya-Dataset inst.) |
| xhosa | xho | 3,574,806 | 20 of 36 | 8 | 0.645 | positionally aligned in contiguous blocks; join by key |

Pairs: built by the key join (`intake2_aya_emit.py`, 400k translated `(T)` rows per language, inputs and targets as separate pairs, exact dedupe, strict protected filter on both sides), 11 languages (afr amh hau ibo mlg nso sna som sot swh xho), 6.19M pairs after filtering (about 545-566k per language), chain pair schema, `key = common.key_of(english, iso, target)`: **written to `/workspace/work/hub/extra2/aya_aligned_QUARANTINE_nllb/` (1.3 GB) and not to `hub/extra2/aya_aligned/`**, because (a) the translated subsets of the Aya collection are NLLB-3.3B machine translations (owner rule: no NLLB-derived data; `common.assert_clean_src` refuses any directory with "quarantine" in its name, so it cannot enter the chain by accident) and (b) quality is poor: numbered answers are mistranslated ("1. the girls 2. no" -> Hausa "1. 'yan mata A'a 2. Na uku ..."), many targets are one-word Flan answers. `licence`=unspecified, `kind` = `aya_translated_inputs` / `aya_translated_targets`. To use them anyway: `mv` the directory to `hub/extra2/aya_aligned` (the lead's decision).

## 8. Caveats for the lead

* Provenance flags that need a decision: `tumbuka_*` (MT, possibly NLLB), `muri_it` MRI rows (instruction side NLLB-translated), `african_ultrachat`, `kiya_sft`, `alpaca_*`, `amharic_*` (all machine-translated; systems not stated). The `synthetic` column is the switch; `licence_raw` carries the remark.
* `bambara_lm_qa` (381k, 83% of Bambara) is templated LM tasks (punctuation repair, sentence completion), `task_type=other`, no licence declared.
* `webfaq_v2` rows are web FAQ pairs (no licence declared; `unspecified`), similarity >= 0.7 only.
* Rows of `setswana_gpt4o` carry a long English system prompt (kept as the `system` message).
* The `iso` of `kiya_sft`/`aya_dataset` comes from the repo's language column; Arabic dialect codes `arz/ary/aeb` are new tags.
* The decontamination is strict (any turn, any rule); expect 10-25% of Wikipedia-derived outputs (muri) to be removed.
* Nothing was written outside `/workspace/work/instr`, `/workspace/work/hub/extra2/aya_aligned_QUARANTINE_nllb`, `/workspace/work/logs` and the code folder; temp files (`instr/_tmp`, `_cand`) are empty/removable.

## 9. Reproduce

```
cd /workspace/work/code/sft_clean/chain; source /workspace/work/env.sh
python intake2_instr.py oldkeys                    # hashes of the old build (sft-all, sft-new) for the muri check
python intake2_instr.py test --n 1000 --sources muri_it,tumbuka_instruction     # 1,000-row adapter test
python intake2_instr.py run --workers 20           # stage 1 (download, convert, clean, decon) + stage 2 (dedupe, cap, shards); `merge` resumes stage 2
python intake2_instr.py hold --threshold 20000     # synthetic rows of well-covered languages -> _held_synthetic
python intake2_instr.py tokens                     # Gemma token estimates -> _state/tokens.csv
python intake2_aya.py; python intake2_aya_emit.py --langs afrikaans,... --max-per-lang 400000
```
