<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Inter-African translation (African X -> African Y, no English pivot): data and design plan (2026-10-20)

> **Update 2026-10-20 evening:** the full pair set (`full_pairs/`, 157.5 M rows), the 50 / 50 prompt mixture, the NLLB-mined sets in the SFT, the replay and the web-based RL pools are described in `afri2afri_addendum_2026-10-20_evening.md`; where numbers differ, the addendum is current (training dataset = `sft_v2`).

D-46. Companion of `afri2afri_data_inventory_2026-10-20.md` (what we hold). Everything here was built **on the old box, CPU only, `nice`d**, in new directories under `/workspace/afri2afri/` (nothing of
other sessions was touched; the rental was not used). Code: `gemma_mt/afri2afri/` (library), `scripts/build_afri2afri_sft.py`, `scripts/build_afri2afri_rl.py`, `scripts/afri2afri_reward_probe.py`,
`scripts/afri2afri_report_tables.py`, `scripts/afri2afri_language_table.py`, `scripts/afri2afri_dataset_card.py`; tests `tests/test_afri2afri_*.py`.

## 1. What was built (measured)

| artefact | path (old box) | size | content |
|---|---|---:|---|
| pair index | `/workspace/afri2afri/sft/index/`, `best_rows.npz`, `lid.npz` | 0.7 GB + 0.3 GB + 0.07 GB | 25,573,645 en->X rows as hashes; 8,718,777 rows in multi-language English groups with GlotLID labels of both sides |
| SFT pairs | `sft/pairs_train.parquet`, `pairs_val.parquet` | 859 MB, 6 MB | **6,683,317 train + 77,328 validation unordered pairs**, with provenance + licence columns (below) |
| windows | `sft/windows_train.parquet`, `windows_val.parquet` | 537 MB, 6 MB | 3,120,000 train / 21,194 validation multi-sentence windows (2.53 pairs per window; token share b64 23.6 % / b256 28.2 % / b512 24.5 % / b1024 23.7 %) |
| **packed SFT data** | `sft/packed/{train,val}` + `meta.json` | **5.2 GB** (uint32) | **train 6,240,000 samples in 326,982 packs of 4,096 tokens, 1,338,498,793 tokens (609,713,583 target), packing efficiency 0.999; val 42,388 samples, 12,490,230 tokens**; same tokenizer fingerprint as `packed_fwd` (`69f7f26e...`), `PackedDataset` loads it |
| RL prompts | `/workspace/afri2afri/rl/rl_prompts_x2x/<target>.parquet` (77 files), `pool_docs/`, `pool_pairs/` | see section 5 | source windows + target tag |
| reward probe | `sft/reward_probe_{comet,labse,v2}.json`, `reward_probe_v2_emb.pt` | small | section 6 |
| staged export | `sft/export/{stage,sample,private_sources}`, `README.md` (the card), `MANIFEST.json`, `source_datasets.csv` | 1.08 GB | pairs / prompts with `source_dataset` + declared `licence` columns; **nothing uploaded** (section 8) |
| bucket staging | `sft/bucket_stage/afri2afri_packed_x2x_sft.tar.zst` + `.sha256` | 2.06 GB | the packed dataset, zstd -3, sha256 `47fff4fe...`; **not uploaded** |

Row columns of `pairs_*.parquet`: `src_iso`, `tgt_iso` (`src_iso` < `tgt_iso` in `ALL_ISOS` order; the trainer uses both directions), `src`, `tgt`, `pivot_en` (the English sentence both sides translate; null for direct pairs), `origin` (`pivot_en` 6,491,020 / `direct` 192,297), `provenance_kind`, `src_dataset`, `tgt_dataset`, `licence` (class), `qe_src`, `qe_tgt` (AfriCOMET-QE of the en->X translation of each side), `group_size`, `group_key`, `src_tok`, `tgt_tok` (Gemma tokens), `epochs` (planned repetition).

Composition of the 6,683,317 train pairs: `religious_jw` 4,928,453 (73.7 %), `mixed` 1,406,093 (21.0 %), `human_curated` 252,322 (3.8 %), `synthetic_mt` 96,449 (1.4 %); 87.7 % of the pairs come from English groups shared by >= 10 languages (Bible / JW multi-way); median group size 27.

## 2. Prompt / tag format

**Proposal (implemented in `afri2afri/prompt.py`, nothing existing was edited):**

```
<bos><start_of_turn>user
Translate the text from Yoruba to Hausa:  Ẹ kaaro<end_of_turn>
<start_of_turn>model
hau Ina kwana<end_of_turn>
```

* It is the literal en->X template (`Translate the text to {language}:  {text}`, two spaces after the colon) with `from {source language} ` inserted; the **answer is unchanged**: the TARGET tag, a space, the translation, `<end_of_turn>`. Therefore the loss mask, the stop ids (106, 1), `format.strip_iso_prefix`, the RL reward's sanitisation chain (cut at the first stop id, NFC, strip the tag) and the GlotLID guard keep working without change; the tokenizer needs **no new special token** (language names are ordinary text with the same fixed names as the en->X prompt, D-02).
* `English` is a legal source (`Translate the text from English to Yoruba:`) and `source=None` gives the old implicit form (`build_user_content_x2x(text, None, tgt)` is identical to `format.build_user_content`, pinned by a test), so ONE model can be trained on the explicit form for all three families (en->X, X->en, X->Y) or keep the old prompts for replay.
* **Why name the source** (the en->X model infers it from the text): near-identical bags of n-grams (Nguni zul/xho/ssw/nbl, Sotho-Tswana sot/tsn/nso, Manding bam/dyu, Akan twi/fat, Kinyarwanda/Kirundi, swh/swc) make the source ambiguous exactly where pairs are thin; the RL prompt file needs `src_iso` anyway; and the reward's pivot checks and the per-direction calibration need to know the source language. Cost: 2-4 extra prompt tokens per sample (measured with the Gemma tokenizer on seven pairs: +2, +2, +4, +3, +3, +3, +3).
* Alternatives considered: (1) `Translate yor to hau:` style tags (breaks the literal template, mixes the tag with the language name); (2) a chat-template variant with a system turn (the Gemma template folds it into the user turn; adds tokens, no information); (3) leave the source implicit (cheapest, risks wrong-source reading in related-language pairs; kept as the `source=None` option for replay).
* Packed layout = `gemma_mt.data.prepare_sft` (files and dtypes identical) + `src_iso_id.npy` (uint16; `len(ALL_ISOS)` = English) and `direction.npy` = 2 for X->Y (0 = en2x, 1 = x2en). `meta.json` keeps `format_version` 1 and adds `format_x2x_version` 2, `instruction_template_from`, `direction_codes`.

**Trainer changes needed (NOT made, the trainer is running elsewhere):** `train_sft.evaluate` sizes its per-language validation vector `2 * n_langs` (en2x, x2en) and `PackedDataset.token_iso = iso + n_langs * direction`: with direction 2 the index reaches `3 * n_langs`. Fix: `n_dir = max(2, int(ds.direction.max()) + 1)`, allocate `n_dir * n_l` and report `val_loss_x2x`; optionally read `src_iso_id.npy` for per-source breakdowns. Two datasets cannot be mixed by the current loader: for replay of en->X / X->en either (a) pack them together (`pack` accepts any window file; extend it with the `windows_sft` windows rendered with `source="eng"`) or (b) add a mixture sampler to the trainer. Owner decision 2 below.

## 3. Sampling policy over the 77 x 76 directions

* **Unit = the unordered language pair** (2,813 pairs have data; both directions use the same sentence pairs, like the en->X / X->en samples of one pair), so the dataset is symmetric by construction.
* **UniMax** (`policy.unimax`, Chung et al. 2023): spend a budget **uniformly** over the pairs, at most 3 passes over a pair's own data; what small pairs cannot absorb goes to the larger ones. Budget of the first dataset: **8,000,000 sentence-pair draws** (`--budget`). Result (measured): median pair has 3,746 available pairs and gets 3,855 draws; **no pair exceeds 3,856 draws**, 1,419 pairs are repeated (mean 1.18 passes, 20.3 % of the pairs sit in repeated pairs); 1,932 pairs have >= 1,000 distinct pairs, 881 pairs < 1,000 (489 < 100).
  Why UniMax and not temperature sampling: with 5,852 directions the goal is the same quality for every direction; proportional sampling would put 28 % of the data on the 100 largest of the 2,820 pairs (the 20 largest alone 10 %, measured on the join) and starve the 880 pairs with < 1,000 pairs. The price is that rich pairs are not exploited: **the cap is the knob** (`--budget 24000000` triples it; re-run `select -> pack`, ~35 min CPU).
* **Composition inside a pair** (`policy.compose`): fill order human_curated -> mixed -> religious_jw -> synthetic_mt; caps `religious_jw` 60 % and `synthetic_mt` 15 % of the pair's draws, relaxed (and the shortfall reported) only when nothing else exists for that pair. Share of English-pivot-derived vs direct pairs: pivot_en 6,491,020 (97.1 %), direct 192,297 (2.9 %): **true direct data is 3 % of the dataset because we hold little of it**; direct pairs win every duplicate against a pivot pair of the same texts.
* **Back-translation**: no back-translated text is added by the build. Rows from repos that are back-translations / machine translations are labelled `synthetic_mt` and capped at 15 % per pair (1.4 % overall); tds-derived pairs still enter through the pivot join when tds is the English partner.
* **Validation**: whole English groups (2 % by hash) are validation, so a validation sentence is never trained on with another partner; at most 40 pairs per language pair (77,328 validation pairs, 2,442 language pairs).
* **Direction balance** (measured on the packed data): 5,618 ordered directions have samples; median 1,334 samples = 295,190 tokens per direction; 3,878 directions >= 1,000 samples; none >= 10,000 samples (flat by design); minimum 3 samples. Language involvement (pairs as source or target): lowest nus 4,246, bam 7,270, bbj 15,523, shi 23,485, dag 29,709, nbl 34,850, kab 38,494, kin 40,569; highest ibo 262,192, yor 254,512, twi 251,411.

## 4. Filters (in order) and what each removed (measured)

1. **Exact + 8-gram leak filter** against the 916,486 benchmark sentences (inventory section 7): 29,953 + 30,882 en->X rows at the join; 1,908 train / 59 validation pairs at materialisation (the 500 long-document benchmark documents are indexed too).
2. **GlotLID of both sides** (the measured per-language labels / minimum lengths / enabled flags of the reward calibration, D-37; pcm and kon are not checked because their LID is unreliable): English side must be `eng_Latn` (checked from 30 characters): 51,041 rows rejected; African side must carry an accepted label: **20,280 rows rejected (0.23 %)**, 8,124,124 confirmed, 574,373 not checkable (short text / LID off). Worst languages: zul 4,983, ibo 2,983, tsn 1,606, nso 1,081, xho 1,064.
3. **Per-language floor on the AfriCOMET-QE of the en->X translation of each side** (the 30th percentile of that language's rows, on top of the build's own P15 gate): 8,718,777 -> **6,056,478 rows** survive 1-3 (a pivot pair is as noisy as both of its sides together; the min of the two side scores is implicit in applying the floor to both).
4. **Text filters per pair** (`filters.py`): letters / length / URL / markup (`basic`), script check for Ethiopic languages, **copy of the source** (identical or word-Jaccard >= 0.9) and **length-ratio window** per language pair (log ratio within 0.8 of the ratio of the two languages' median "characters per English character", +0.4 for direct pairs). Removed from the 6,759,707 selected train pairs: length ratio 69,841, copy 4,322, benchmark 1,908, basic 282, script 37 -> **6,683,317 kept (98.9 %)**; validation 78,438 -> 77,328.
5. **De-duplication** on the text pair (direct beats pivot; better provenance class beats worse): 121,794 duplicate pairs among the 32.5 M candidate records.
6. *Optional, not run:* AfriCOMET-QE directly on (X text, Y text). Section 6 shows why: out-of-distribution and weakest exactly for the closest languages; the English-pivot QE of step 3 is the in-distribution substitute (it is the QE of each side against the same English sentence).

## 5. RL prompts

Format: `rl_prompts_v2` family, one parquet per **target** language, `english` renamed `source`, new `src_iso` / `src_language`, plus `src_qe_valid` / `tgt_qe_valid` (reward-family flags); columns, bins (AfroXLM-R tokens, b64 [24,64] ... b1024 [513,1024]) and loader notes in `afri2afri/rl_prompts.py`. **Source texts are monolingual African windows**, never English.

Pools: (a) real documents: the 20 `hf0-HuggingFaceFW_fineweb-2` shards of `/workspace/tokdata/shards` (amh hau ibo pcm tir yor; ODC-By), blocks of 60 consecutive paragraph lines -> sentences -> windows (the `gemma_rl_windows` rules); (b) for all 77 languages, windows from the African side of en->X rows that are **held out of the SFT**: neither the text nor the English sentence of any SFT train / validation pair (503,756 groups, 3.64 M texts excluded), synthetic sources and the lowest-QE third dropped, up to 60,000 rows per language, pseudo-paragraphs of consecutive rows of one source.
Assemble: dedupe, `basic_ok` + script check, **benchmark leak filter** (windows), the SFT-text exclusion on document windows, GlotLID guard, then per (source, bin) up to 1,500 windows spread over the 76 target languages with a cyclic walk (each window used equally often, a (window, target) pair never repeated), quotas = 24 prompts per direction and bin x the **UniMax weight of the pair in the SFT** (mean 1, clipped to [0.25, 2]): the RL data is shaped by the same allocation as the SFT, and a window serves at most 6 targets.

**Result (measured):** 77 files (one per target language), **411,814 prompts** (411,594 staged after the private-source exclusion), 215,563 distinct source windows, **5,776 of 5,852 ordered directions have prompts** (per direction: median 72, minimum 1, maximum 132; per target file 1,356 to 6,654 rows). Bins: b64 126,473 / b256 117,368 / b512 90,413 / b1024 77,560 (AfroXLM-R tokens; the b64 minimum is 7 tokens: short standalone sentences). Sources: held-out SFT-side text 385,636 rows, fineweb-2 document windows 26,178 rows (6.4 %). Dropped on the way: 16,720 duplicate windows, 4,283 basic / script, 2,297 benchmark leaks, 931 sharing a sentence with SFT texts, 797 failing GlotLID. **Thin source pools:** ibb (86 / 48 / 10 / 6 windows in b64 / b256 / b512 / b1024), urh, kam, oke, dyu, kon, kbp, bbj, shi, bam, kik, sid and above all **kea (4 windows), her (12), ish (10), wes (24), dua (27)**: their directions repeat the same few windows and are not usable for RL as they stand; the fix is the v2 pools of inventory section 5 (Wikipedia / fineweb-2 / GlotCC / bucket `datasets/pretrain/*.tar.zst`). Real documents exist only for amh hau ibo pcm tir yor; every other language's prompts are pseudo-paragraphs of translated sentences (religious / news mix; `source_dataset` tells which). Independent re-check: 0 of 60,000 sampled prompts match a benchmark sentence.



## 6. Reward design for X -> Y GRPO

### 6.1 What the en->X reward is made of and what transfers

| component (D-17 ... D-45) | needs the source? | transfers to X->Y? |
|---|---|---|
| sanitisation chain (cut at the first stop id, NFC, strip the target tag), failure classes: no `<end_of_turn>`, missing / wrong tag, empty, post-eos text, degenerate repetition, self-repetition | no | **yes, unchanged** (the answer format is unchanged, section 2) |
| copy-of-source (normalised equality or word-set Jaccard >= 0.9) | yes (the source text) | **yes**; measured: on 132 human FLORES+ dev directions among 12 languages **no true pair reaches Jaccard 0.9**, the closest (sot-tsn) averages 0.19, zul-xho 0.07 |
| length-ratio window per language (calibrated vs English) | yes | needs a **per-pair expectation**: use the ratio of the two languages' calibrated medians (what `filters.expected_log_ratio` does); measured human char ratios tgt/src: p05 >= 0.42, p95 <= 2.40 over the panel, inside the old [0.2, 5] limits |
| omission / extra-sentence detector (Gale-Church alignment of source and output sentences), lexical-fidelity (shared stop-words, built for Pidgin / English) and diacritics blocks | yes | omission: **yes** (language-agnostic once both sides are split into sentences; the cost constants were fitted on English sources and need re-fitting per family); lexical-fidelity: **no** (it exploits English <-> Pidgin similarity; drop it); diacritics: yes (target-side) |
| GlotLID guard on the output | no | **yes** (target language; also makes a copy of an English or third-language text fail) |
| content = AfriCOMET-QE per chunk, shaped by per-language anchors (P10 / P50 / P90 of QE on human references) | yes | **partly**: section 6.2 |
| consensus chrF++ against the other valid samples of the group (QE-free, D-38) | no | **yes, unchanged**; its validation (Spearman 0.38 with reference chrF++ for efi, 0.27-0.38 for ach / nyn / ibb) was done on en->X samples: **not re-measured for X->Y** (needs the model) |

### 6.2 Is AfriCOMET-QE valid when the source is an African language? (honest answer)

* **Documentation.** The model card (`masakhane/africomet-qe-stl-1.1`) lists the encoder's 76 languages and its example is Yoruba source with an English translation; the paper (arXiv 2311.09828, abstract read; the full text was not re-read) trains on direct-assessment data of well-resourced languages plus 13 African languages with an African-centric encoder. **Nothing documents African-source -> African-hypothesis scoring**; English (and French) are the other side in everything the card shows. So source = African is in distribution only when the translation is English; X -> Y is a transfer case.
* **Measurement** (`scripts/afri2afri_reward_probe.py`, FLORES+ **dev**, 120 random sentences, 12 languages yor hau ibo swh zul xho afr amh som kin sot tsn, all 132 ordered directions, CPU, the AfriCOMET encoder embeds each unique text once; contrasts: *mismatch* = the translation of another sentence, *neighbour* = of a sentence 1-3 positions away (topic kept), *half-omitted* = first half of the words; AUC = P(score of the human translation > score of the contrast)):

| scorer / orientation | directions | AUC mismatch | AUC neighbour | AUC half-omitted | copy of the source scores >= human |
|---|---:|---:|---:|---:|---:|
| **AfriCOMET-QE, X -> Y** (African -> African) | 132 | **0.856** | 0.841 | 0.878 | 1.1 % |
| ... same family (zul-xho, sot-tsn, swh-kin pairs) | 6 | 0.788 | 0.760 | 0.823 | 0 % |
| ... different families | 126 | 0.859 | 0.845 | 0.881 | 1.1 % |
| ... pairs with Amharic (other script) | 24 | 0.900 | 0.885 | 0.904 | 0.9 % |
| AfriCOMET-QE, **English -> Y** (the en->X reward setting, same sentences) | 12 | 0.914 | 0.900 | 0.888 | 5.6 % |
| AfriCOMET-QE, **X -> English** (the documented orientation) | 12 | 0.952 | 0.937 | 0.968 | 3.7 % |
| SSA-COMET-QE (McGill), X -> Y (first panel, 10 languages) | 90 | 0.838 | 0.824 | - | 2.1 % |
| cosine of the AfriCOMET sentence embeddings, X -> Y (10 languages) | 90 | 0.953 | 0.945 | - | **100 %** |
| LaBSE cosine, X -> Y (10 languages, all supported by LaBSE) | 90 | **0.999** | 0.998 | - | **100 %** |

  Worst directions: sot->tsn 0.607, tsn->sot 0.670, yor->xho 0.729, kin->yor 0.736, yor->amh 0.736, yor->ibo 0.741 (mismatch AUC); the closest related pairs (Sotho-Tswana) are barely above the 0.65 "unusable" line of D-17, the old reliability bar of 0.70 fails for them. Median QE of the human translations is **lower in X -> Y than in English -> Y by 0.055 on average (range -0.009 for yor to -0.097 for amh) and varies with the source (standard deviation 0.03-0.07 across sources)**: the en->X anchors (`t_fail / t_mid / t_great`) cannot be reused. Small-sample caveat: 120 sentences per direction, standard error of an AUC ~0.03; single FLORES sentences only (no windows).
* **Reading.** AfriCOMET-QE keeps most of its discriminating power on African sources (0.856 vs 0.914 for English sources, a loss of 0.06 AUC) and still rejects copies, so it is **usable as a content term at reduced weight**, not as the sole signal for close language pairs. Embedding similarity (LaBSE, AfriCOMET cosine) is excellent at "same meaning" (0.95-1.00) but **cannot see copies** (a copy has cosine 1) and knows nothing about fluency or target-language correctness; LaBSE covers only 15 of the 77 languages (afr amh hau ibo mlg nya kin sna som sot swh wol xho yor zul; coded from the model's published language list, not re-verified) - it is a candidate for an *adequacy gate* on those, behind the copy / LID guards, not a reward by itself.

### 6.3 Proposed reward for X -> Y (a design; the numbers above are the only measurements)

1. **Rubric first** (unchanged failure classes, GlotLID target guard, copy guard, omission detector with per-family cost) -> the shaped floor.
2. **Content = blend**, per direction family: `reward = (1 - b) x shaped_QE_Y|X + b x consensus`, with **b = 0.5 by default and 0.7 for the families whose measured mismatch AUC is < 0.80** (Sotho-Tswana; check Nguni, Manding, Akan, Kinyarwanda-Kirundi on the full calibration), QE weight (D-17 AUC rule) 0 below 0.65. Consensus is untouched.
3. **English-pivot QE term (new, cheap, in distribution):** for every prompt keep an English rendering `E_src` of the source (for the pairs-pool prompts a human English sentence exists in the index: store it as `english_ref`; for document windows translate once, offline, with our own X->en direction or NLLB). Score a sample `h` as `QE(src = h, mt = E_src)`: African text as source, English as translation - the documented orientation (AUC 0.952 above, measured with the human English as `E_src`; **not measured with a machine `E_src`**, which is the open number). No extra generation per sample: `E_src` is fixed per prompt.
4. **Round-trip / pivot consistency by our own model** (translate `h` back to X, or to English, compare with the source by chrF++): costs one extra greedy pass per sample (x2 inference); keep it for evaluation and for a small audit subset, not for the loop. Not measured (needs the model).
5. **LaBSE adequacy gate** for the 15 supported languages: `LaBSE(source, h) >= tau` (tau from FLORES dev, e.g. the 5th percentile of true pairs) else the sample is capped at the floor - it removes the "fluent but unrelated" outputs QE can miss, while the copy / LID guards remove what LaBSE cannot see.
6. **Calibration per direction family from FLORES+ dev** (997 sentences; dev files held: 38 in `ssa_cmp/flores` + 17 in `bench_dev.jsonl`, the Hub has the rest): families (Nguni, Sotho-Tswana, Eastern Bantu, Western Bantu / Bantoid, Kwa-Gbe, Yoruboid-Igboid, Chadic, Cushitic, Ethio-Semitic, Malagasy, Afrikaans, other) x target language -> anchors P10 / P50 / P90, mismatch / copy / half-omitted AUC and the omission cost, written in the format of `calibration/shaping_flores_dev_v6*.json` with a `src_family` key. Languages without dev data (33 of 77 have no FLORES+ devtest file here; dev: SmolSent / IbomMT / the 150-sentence reference split of D-43) calibrate on those sets or run consensus-only exactly as in D-38. `scripts/afri2afri_reward_probe.py` already computes everything but the anchors' file format; embeddings are saved (`reward_probe_v2_emb.pt`) so calibration of more families costs only the estimator pass.
7. **What can go wrong (watch in the first 50 steps):** the policy learning to copy the source with a few word changes (QE barely sees it for related languages: Jaccard guard at 0.9 is too loose for sot-tsn, whose true pairs average 0.19 - tighten to 0.6 for the closest families); mode collapse of consensus; QE reward hacking by an English-like output (GlotLID guard); translationese from Bible-heavy SFT (the SFT domain is 73.7 % religious: RL prompts are therefore taken from non-Bible text where possible, section 5).

## 7. Benchmark plan (short)

* **Pool** (measured, section 2 / T3 of the inventory): the held-out multi-way sets give **2,726 ordered directions with >= 100 shared items, 2,080 with >= 1,000**: FLORES+ devtest 44 languages -> 1,892 directions x 1,012 sentences; NTREX 23 languages -> 506 directions x 1,997; SmolSent 45 languages -> 1,980 directions x 863; SmolDoc 45 languages -> 1,980 directions, all with >= 100 shared documents (median 213 documents per language, 130-668); the long-document benchmark's human tracks (SmolDoc, FLORES+ passages, stitched documents: 17,989 documents, 54 languages) give the document level. 22 of the 77 languages (ibb urh bin dag oke bbj swc kqn lun iso loz wal tsc kua seh her yao sid ish dua bas wes) have **no** multi-way human set: not benchmarkable, report them as "unevaluated" or with pseudo references only.
* **Design.** (1) *Core*: FLORES+ devtest, all 1,892 directions among the 44 languages, 1,012 sentences (or a fixed 250-sentence subset per direction = 473 k sentences if decoding time matters; decoding time not measured), chrF++ / spBLEU, AfriCOMET-ref only for the 42 QE-validated languages; headline = macro over directions, plus macro by source family, by target family and the 12 x 12 full matrix. (2) *Wide*: SmolSent + NTREX for the languages FLORES+ does not cover (45 / 23 languages). (3) *Documents*: SmolDoc and stitched documents, X -> Y among the best-covered 20 languages (380 directions), one document = one segment, with the omission / loop / finished metrics of the long-document harness. (4) Always report `clean` vs `no_pair`/`seen` subsets: a model that starts from the en->X checkpoint saw the English side of many items and the reference pair of bam / wol / ewe (inventory, section 7).
* **Pseudo-references** (long-document approach): for X -> Y where no human document exists in Y, take the human document in X as source and a **pseudo reference = the best open translator's X -> Y output** (NLLB-200 directly, or MADLAD-400 with the `<2yy>` tag; for the eight languages no system names - urh bin oke bbj yao sid ish dua - no pseudo reference exists), with the tier logic of the long-document benchmark (A / B / C from agreement with human references on FLORES dev). Two ways to build `ref_Y`: direct X -> Y, or X -> English -> Y; keep the direct one as primary and use the pivoted one as a second system - their disagreement is the confidence flag. Where BOTH languages have human documents (FLORES+, SmolDoc) use the human text and do not mix the tracks in one number. Circularity warning (from the long-document plan): NLLB / MADLAD outputs circulate in public African datasets; part of our own pairs may be NLLB-derived (`tds`, `aya`), so a pseudo-reference score rewards resemblance to those systems: pseudo tracks are reported separately and never ranked against human tracks.

## 8. Hub destination, sizes, and what has (not) been pushed

**Nothing has been uploaded anywhere** (no Hub write, no bucket write) by this session. Everything is staged on the old box and the push is ONE command per step.

* **Destination named by the coordinator for the owner: `<HF_ORG_LEGACY>/interlaanguage-translation` (dataset, public).** The coordinator's two messages (first: redistributable licences only plus a restricted bucket folder; then: everything, whatever the licence, every row with `source_dataset` + `licence`, held-out sets and private data out) arrived through the coordinator, not from the owner in chat. A *public* upload of rows that are non-commercial / share-alike / of unknown licence is an irreversible publication, so the push is **prepared but waits for an explicit confirmation in the owner's own words**. After the confirmation:
  `python scripts/afri2afri_push_hub.py --stage-dir /workspace/afri2afri/sft/export --repo-id <HF_ORG_LEGACY>/interlaanguage-translation --step sample --public --yes` (card + manifest + `direction_counts.csv` + 3 samples of 300 rows), check the repo, then `--step full --public --yes` (3 parquet files, 1.08 GB). The script is a dry run without `--yes`, reads `HF_WRITE_TOKEN` (then `HF_TOKEN`) from the environment and never prints it, uploads one file per commit and skips parquet files that are already there with the same size (resume).
* **Staged content** (`/workspace/afri2afri/sft/export/`, `MANIFEST.json` has sha256 and row counts):

| file | rows | size |
|---|---:|---:|
| `stage/sft_pairs_train.parquet` | 6,628,491 | 891 MB |
| `stage/sft_pairs_val.parquet` | 76,866 | 6.4 MB |
| `stage/rl_prompts.parquet` | 411,594 | 185 MB |
| `sample/sample_*.parquet` (300 random rows each) | 300 x 3 | 0.3 MB |
| `README.md` (the dataset card, copy in `reports/afri2afri_2026-10-20/hub_dataset_card_*.md`), `MANIFEST.json`, `direction_counts.csv`, `source_datasets.csv` | | 1 MB |
| `private_sources/*` (**NOT for the repo**: rows whose source repo is flagged private in the licence scan; 54,826 + 462 + 220 rows) | | 7 MB |

  Every row carries `source_dataset` (repo id; `a + b` for a pivot pair) and `licence` (the licence **declared on the source card**; `unknown` when none; the 99 `michsethowusu/english-*_mt560` cards declare CC-BY-4.0 and the scan had missed it, so this is read from the card audit; the upstream of those sets is OPUS MT560 = JW300 + Bible + others, copyright of the original texts not cleared) plus `licence_class`. **Declared-licence distribution of the SFT train rows** (6,628,491): cc-by-4.0 4,740,618 (71.5 %); `unknown + cc-by-4.0` 999,028 (15.1 %); unknown 392,007 (5.9 %); `cc-by-4.0 + unknown` 153,494; `other + cc-by-4.0` 144,574; mit 61,076; others ... 33 distinct strings, full list in the card; **RL prompts**: cc-by-4.0 210,499; unknown 113,587; other 59,196; odc-by-1.0 26,178; apache-2.0 1,331; mit 749; mpl-2.0 50; cc-by-sa-4.0 4. Share-alike rows are few (cc-by-sa-4.0 6,782 pairs); non-commercial licences are not declared by any source in these rows (the build had already dropped them, D-21), so the card says "some rows may be non-commercial or share-alike: check `licence`" and means it for the rows with an unknown or `other` side: **24.3 % of the SFT train rows have at least one side of unknown declared licence, 3.8 % an `other`, 27.6 % / 14.4 % of the RL prompts**.
  203 source repos (`source_datasets.csv`): 124 with unknown declared licence, 72 cc-by-4.0, 2 other, and one each mit / mpl-2.0 / cc-by-sa-4.0 / apache-2.0 / odc-by-1.0.
* **Held out and absent** (stated in the card): FLORES+ dev / devtest, NTREX-128, SMOL (SmolSent, SmolDoc, GATITOS), MAFAND test, IbomMT, FLORES-MOZ, GlotStoryBook, SALT test and the 500 English documents of the long-document benchmark core set (their *human-reference documents* are SmolDoc / FLORES+ passages, already in the list). The long-document `extended` source set was not on the rental when this was run (only `core`, `core_private`, `core_public` exist): if it is rebuilt, re-run `select -> pack` with its docs as `--extra-bench`.
* **Not in the repo:** the packed token dataset (5.2 GB raw; `remote:<S3_BUCKET>/afri2afri/` is the established place: staged as `bucket_stage/afri2afri_packed_x2x_sft.tar.zst`, 2.06 GB zstd -3, sha256 `47fff4fe173a...`), the windows parquet (537 MB) and the withheld private-source rows. Public repos do not use the private quota (the <HF_ORG_LEGACY> private quota is full, D-40; I did not test it, the earlier idea of a tiny upload to `<HF_ORG_LEGACY>/afri2afri-sft` was dropped when the destination changed).
* **Names proposed for later** (owner decides, none created): model repos `<HF_ORG>/gemma3-270m-afri2afri-sft` and `...-rl` (the org is full: confirm space; mirrors `gemma3-270m-translate` / `gemma-3-270m-translate-rl`); benchmark `<HF_ORG_LEGACY>/interlaanguage-translation-bench` (public, the held-out sets are public already).

## 9. Reproduction

```
cd gemma_mt   # code is synced to /workspace/afri2afri/code/gemma_mt_repo on the old box
python scripts/build_afri2afri_sft.py index  --pairs gemma/pairs_final --out OUT           # 3 min
python scripts/build_afri2afri_sft.py matrix --out OUT --bench bench/test_sets --extra-bench <flores dev files> <longdoc docs.jsonl>     # 1 min
python scripts/build_afri2afri_sft.py lid    --out OUT --pairs gemma/pairs_final --glotlid model_v3.bin                                 # 23 min, 10 procs
python scripts/build_afri2afri_sft.py direct --final-v2 final_v2 --out OUT ; ... direct-clean --glotlid ...                              # 1 + 4 min
python scripts/build_afri2afri_sft.py select --out OUT --calibration calibration/shaping_flores_dev_v6_efi_consensus.json --budget 8000000   # 1 min
python scripts/build_afri2afri_sft.py materialize --out OUT --pairs gemma/pairs_final --bench ... --tokenizer <gemma-3-270m-it dir>       # 9 min
python scripts/build_afri2afri_sft.py windows --out OUT ; ... pack --out OUT --tokenizer <dir>                                         # 1 + 6 min
python scripts/build_afri2afri_rl.py pool-docs / pool-pairs / assemble ...                                                             # 10 + 3 + 35 min
python scripts/build_afri2afri_sft.py export --out OUT --rl-dir rl/rl_prompts_x2x --license-scan report/license_scan.json              # 2 min
python scripts/afri2afri_dataset_card.py ... ; python scripts/afri2afri_push_hub.py ... (dry run by default)
python scripts/afri2afri_reward_probe.py --flores ... --out probe.json ...                                                              # 8 min, 5 threads
```

Tests (old box, real pytest 9 in `/workspace/afri2afri/pylibs`, `nice`): `tests/test_afri2afri_core.py` (23), `test_afri2afri_prompt.py` (6), `test_afri2afri_pipeline.py` (5, needs pyarrow), `test_afri2afri_rl_builder.py` (1): see section 11 for the final run.

## 10. Owner decisions

1. **Publish**: confirm in your own words that the staged data may be pushed PUBLICLY to `<HF_ORG_LEGACY>/interlaanguage-translation` with the rows as they are (24 % of the SFT rows have a side of unknown declared licence, upstream JW/Bible copyright not cleared, CC-BY-SA rows flagged); the private-source rows (55,508) stay out, unless you say otherwise (`--keep-private-sources`).
2. **Initialisation and replay**: start from base Gemma or from the en->X checkpoint? Mix en->X / X->en replay (explicit-source form) into the same packed dataset? Requires the trainer patch of section 2 (validation vector for direction 2) and either a replay option in `pack` or a mixture sampler.
3. **Budget / flatness**: 8 M draws gives ~3.9 k pairs per language pair and no direction above 10 k samples (1.34 B tokens). Raise to 16-24 M for the rich pairs? (cost: ~linear in tokens; re-run `select -> pack`).
4. **Prompt format**: explicit `from X to Y` always, or a mixture with the implicit en->X form (and what share)?
5. **NLLB-mined `michsethowusu` pair sets** (110.9 M rows, NC upstream): use, skip, or use only for RL prompts / evaluation of thin directions? Pending since 2026-10-11.
6. **Domain**: 73.7 % of the SFT pairs are religious. Fetch non-religious multi-way data (NLLB-Seed, TICO-19, Wikipedia-aligned, the Bible-free part of the catalogue) before training, or accept it and rely on RL prompts from news / wiki text?
7. **RL sources**: vet and use the `training_cle-*` / `corpus-*` shards (licence unknown) and the bucket `datasets/pretrain` text for the 50 languages with thin or no document pools (kea her ish wes dua oke dyu ibb ...); or fetch Wikipedia / fineweb-2 / GlotCC / HPLT (verified open licences).
8. **Reward**: approve the blend with higher consensus weight for close pairs, the English-pivot QE term (needs an English rendering per prompt: store `english_ref` for the pairs-pool prompts, translate the document windows offline), the LaBSE adequacy gate for 15 languages; and the copy-guard Jaccard of 0.6 for close families.
9. **Benchmark scope**: core = FLORES+ devtest 44 languages, 1,892 directions (subsample of 250 sentences if time-bound); documents among the best 20 languages; the 22 languages without any multi-way human set reported as unevaluated.
10. **French pivot** (143 k pairs) and Arabic / Portuguese: audit the labels or drop (currently dropped).
11. **Licence wording**: the `mt560` cards say CC-BY-4.0 while the content is JW300/Bible-derived: keep the declared licence in `licence` (current) or flag these rows as `cc-by-4.0 (upstream: OPUS MT560)`.

## 11. Test log (old box CPU, real pytest 9, `nice`, 2026-10-20)

* New tests: `tests/test_afri2afri_core.py` 23 (text normalisation incl. the Yoruba U+0329/U+0323 unification, provenance, join vs brute force, `pick_by_quota`, UniMax properties, compose caps, LID verdicts, length-ratio window, copy / script / basic filters, leak index exact + 8-gram, every benchmark file format incl. long-document `sents`, windows alignment, RL windows / assign_targets / direction weights), `test_afri2afri_prompt.py` 6 (template pinned equal to the en->X template + `from X `, implicit form identical to `format.build_example`, loss mask, `strip_iso_prefix`), `test_afri2afri_pipeline.py` 5 (fast batch encoder == `build_example_x2x`, packed layout loadable by `PackedDataset`, parquet index + join, end-to-end `select_records` properties incl. "validation sentence never trained on with another partner"), `test_afri2afri_rl_builder.py` 1 (assemble on a synthetic pool: one file per target, new columns, leak dropped, no repeated (window, target)): **35 passed**.
* Whole suite of the module on the same box (`pytest tests`, gated integration test excluded): **721 passed, 4 skipped, 1 failed** - `tests/test_stratified.py::test_pusher_is_disabled_without_repo_or_token`, the failure that D-43 already records for this environment (not touched by D-46).
* The laptop runner (`python tests/run_tests.py`) passes the 29 numpy-only ones and skips the pyarrow ones.
* Not tested / not run: the trainer changes, any GPU path, a real RL step on the new prompts, the hub push (dry run only).
