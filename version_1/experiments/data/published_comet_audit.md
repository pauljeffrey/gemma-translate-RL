# Published COMET-family numbers for the baseline models: audit (2026-10-22, agent E2-lit)

Companion of `published_comet_audit.csv` (122 rows; every one of the 43 models of `published_baselines.csv` has at least one row).
Method: full text of each source searched (PDF text or arXiv HTML) for COMET / AfriCOMET / SSA-COMET / MetricX / BLEURT; model cards opened; numbers
re-read from the arXiv HTML tables where possible (AfriqueLLM Tables 20-23 and TranslatePsy Tables 24/25 were parsed from the HTML source, not from a summary).
`verified=not_found` means "searched, absent", not "could not open". The only unreadable item is the AfroBench leaderboard Space (sleeping).

## 10-line summary

1. **The owner's belief is wrong for almost every model.** AfriCOMET numbers (any `africomet-*` checkpoint) are published for exactly **two** of the 43 models:
   NLLB-600M and AfriNLLB-600M (AfriNLLB paper, arXiv 2602.09373, Table 7: `masakhane/africomet-mtl` v1.0, FLORES-200 devtest, en->xx, 9 African languages, scale 0-100).
2. **SSA-COMET numbers are published for 34 of the 43** (not AfriCOMET; 25 via TranslatePsy, 9 more only via AfriqueLLM Table 20): TranslatePsy-AfriSLM Tables 3/24 (FLORES-200 devtest, 19 languages, `McGill-NLP/ssa-comet-mtl`, direction not stated, probably both averaged) for NLLB 600M/1.3B/3.3B, AfriNLLB, Qwen3 4B/8B, Qwen3.5 0.8B-122B, Apertus, Hunyuan-MT, TranslateGemma 4B/12B/27B, AfriqueLLM, and the three TranslatePsy models; AfriqueLLM Table 20 (FLORES-200, en->xx, 24 languages, 21 of them ours, checkpoint unnamed) for 17 models incl. Llama3.1-8B, Lugha-Llama, Gemma3-4B/12B, Qwen3-4B/8B/14B, Qwen3.5-4B.
3. **COMET-22 and MetricX-24** are published for the same 25 models in TranslatePsy Table 25/24, and as 55-pair WMT24++ aggregates for TranslateGemma (MetricX per-pair for sw_KE, sw_TZ, zu_ZA only). Qwen3.5-4B has one XCOMET-XXL aggregate (WMT24++, card).
4. **No COMET-family number anywhere (verified absent): NLLB-200-distilled-1.3B, NLLB-200-MoE-54.5B, Aya-101, GPT-4o, Gemini 1.5 Pro, Gemini 2.5 Flash, Gemma 2 27B, Llama 3.1 70B** (AfroBench and LingBench++ use chrF; NLLB release files are chrF++ only), plus MADLAD-400 and InkubaLM (not in the csv). **Absent from the models' own papers (but present via a third-party paper):** NLLB 600M/1.3B/3.3B, Lugha-Llama, Apertus, Qwen3/3.5, Hunyuan-MT (no African language at all), Gemma 3, Llama 3.1 8B. (The csv alias `NLLB 3.3B` is the same checkpoint as `NLLB-200-3.3B`.)
5. NLLB-200 1.3B/3.3B (and 600M) get a COMET-family number only through TranslatePsy (SSA-COMET 0.5878 / 0.5944 / 0.5718; C22 0.7471 / 0.7525 / 0.7340; 19-language macro); NLLB-200-distilled-1.3B and MoE-54.5B have none.
6. **Nothing matches our protocol exactly** (FLORES+ devtest, en->xx, `africomet-stl-1.1`, reference-based). Closest: AfriNLLB (AfriCOMET family, FLORES-200 devtest, en->xx) = `partial`. Everything else is a different metric (`no`).
7. AfriCOMET v1.0 `africomet-mtl` does not cover Wolof or Lingala (its AfroXLMR has 17 African languages), so the AfriNLLB wol value (17.27 / 22.65) is not meaningful, and AfriNLLB's `africomet-mtl` is not comparable in scale to `stl-1.1`.
8. Tiny Aya (3.35B) uses AfriCOMET only to pick a data translator, not as a score. TranslatePsy uses `masakhane/africomet-qe-stl-1.1` only as a data filter (their Table 1/21 ablation).
9. The SSA-COMET paper (EMNLP 2025) shows SSA-COMET correlates better with human judgements than AfriCOMET v1.1 (avg Spearman 0.639 for SSA-COMET-MTL vs 0.500 for AfriCOMET-v1.1-STL / 0.550 for v1.1-MTL on SSA-MTE), which explains why 2026 papers (AfriqueLLM, TranslatePsy) report SSA-COMET and not AfriCOMET.
10. **Consequence for the single table:** a chrF++ + AfriCOMET table over all 43 models is impossible from published numbers. Options: (a) re-score outputs of the open models ourselves with `africomet-stl-1.1` (needs the models' FLORES+ devtest en->xx translations); (b) add SSA-COMET as a second column using the published TranslatePsy/AfriqueLLM numbers (clearly labelled, different languages and directions); (c) AfriCOMET only for ours + AfriNLLB/NLLB-600M with the checkpoint caveat.

## Which paper introduced which metric

| metric | paper | what it is |
|---|---|---|
| AfriMTE + AfriCOMET | Wang et al., "AfriMTE and AfriCOMET: Enhancing COMET to Embrace Under-resourced African Languages", NAACL 2024, arXiv 2311.09828 | 13 language pairs of human DA/MQM-style annotation (AfriMTE); COMET models trained on WMT 2017-2020 DA + MLQE-PE (no African training data), African-enhanced encoder AfroXLM-R-L; STL and MTL (multi-task), plus QE variants |
| AfriCOMET-STL-1.1, QE-STL-1.1 | model cards `masakhane/africomet-stl-1.1`, `masakhane/africomet-qe-stl-1.1` (repo `masakhane-io/africomet`) | same family, encoder replaced by `Davlan/afro-xlmr-large-76L`; "verified in the WMT 2024 Metrics Shared Task"; Apache-2.0; training data not stated on the card |
| SSA-MTE + SSA-COMET | Li et al., "SSA-COMET: Do LLMs Outperform Learned Metrics in Evaluating MT for Under-Resourced African Languages?", EMNLP 2025 main, arXiv 2506.04557 | SSA-MTE news-domain human annotations (arXiv v2: 14 pairs, 73k annotations; EMNLP camera-ready: 13 pairs, 63k); SSA-COMET-STL/MTL (reference-based, MTL also does QE) and SSA-COMET-QE; paper says encoder AfroXLMR-114L |

## AfriMTE and the "17 languages" (correction)

* **AfriMTE has 13 language pairs, not 17** (paper Section 2.1; repo README lists the 13 of the WMT 2024 challenge set): ary-fra (Darija->French), eng-arz, eng-fra (control), eng-hau, eng-ibo, eng-kik, eng-luo, eng-som, eng-swh, eng-twi, eng-xho, eng-yor, yor-eng (+ four extra eng-yor domain sets: IT, movie, news, TED). African languages: **ary arz hau ibo kik luo som swh twi xho yor** (11). In our 77: hau ibo kik luo som swh twi xho yor (9); ary and arz are not ours.
* **"17" is the original AfroXLM-R encoder** (Alabi et al. 2022, `Davlan/afro-xlmr-large`): 17 African languages **afr amh hau ibo mlg nya orm pcm kin run sna som sot swh xho yor zul** (+ eng fra ara). All 17 are among our 77. That encoder (adapted XLM-R-large) is what AfriCOMET-STL/MTL-1.0 build on; `masakhane/africomet-mtl` card tags 21 but prints 20 (the 17 + eng fra ara).
* AfriCOMET-1.0 models are **zero-shot for African languages**: the paper says "WMT Others" training data contains no African language except English; AfriMTE is only validation (dev) and test (devtest).

## Language coverage of the 1.1 models (AfriCOMET and SSA-COMET) - ISO codes

### AfriCOMET-STL-1.1 and AfriCOMET-QE-STL-1.1 (encoder `afro-xlmr-large-76L`)
Card text says "76 languages" (header 77) but **prints 75 entries**; both cards print the same list:

`eng amh ara som swa por afr fra zul mlg hau sna arz nya ibo xho yor sot kin tir tso orm run nso ewe lin twi pcm gaa loz lug guw bem efi lue lua toi ven tum tll iso kqn zne umb mos tiv lub fuv kwy bci rnd luo wal ssw lun wol nyk kua kik fon bam cjk dik dyu kab kam kbp knc kmb kon nus sag taq tzm nqo`

* The list is the encoder's language set; the metric itself never saw African training labels (see above), so "covered" means "in the encoder", and validity per language is an empirical question (our D-32 study measured it).
* Tswana (`tsn`) is **not** printed although our repo treats it as covered (the 76th language is plausibly tsn; unverified).
* Intersection with our 77 (card names 52): yor hau ibo pcm efi twi ewe fon wol swh kin som orm amh tir lug lin zul xho afr sna nya sot mlg bem tum run ven tso nso ssw sag tiv umb kmb mos kik luo kam dyu bam kbp kon kab nus lua kqn lun iso loz wal kua. With tsn (repo-verified) and swc (Swahili) this is the 54 of `QE_LANGUAGES`.
* Not on the card (25 of our 77): tsn ibb urh nbl bin dag oke shi bbj kea swc ach tsc vmw seh nyn her yao sid ndc kri ish dua bas wes (tsn and swc are covered de facto).
* Card languages that are not among our 77: gaa guw lue toi tll zne lub fuv kwy bci rnd nyk cjk dik knc taq tzm nqo (+ eng ara por fra arz).
* `africomet-stl` / `qe-stl` v1.0 and `africomet-mtl` use the 17-language AfroXLM-R (card lists 20 languages: afr ara amh eng fra hau ibo mlg nya orm pcm kin run sna som sot swa xho yor zul).

### SSA-COMET
* Training/fine-tuning languages = SSA-MTE pairs. arXiv v2: **14 pairs**: eng-{amh hau ibo kik kin luo twi yor zul} (9), fra-{ewe lin wol} (3), por-{vmw nya} (2). `ssa-comet-mtl` card says fine-tuned on 12 languages (amh hau ibo kik kin luo twi yor zul ewe lin wol); `ssa-comet-mtl-final` card says 14 (adds nya, Emakhuwa). EMNLP camera-ready abstract says 13 pairs / 63k annotations. Test source: Global Voices news (40 documents), vmw from Voice of America. Training data also includes the non-African WMT DA data of AfriCOMET and AfriMTE dev; Igbo, Swahili, Sesotho were annotated but excluded (low overlap agreement).
* Language set in our 77: yor hau ibo twi ewe wol kin amh lin zul nya kik luo vmw (14 of 14 pairs' targets are ours).
* Encoder: paper Section 4.2/5.1 says **AfroXLMR-114L** (110 African languages + eng fra ara por; includes Emakhuwa; 76L does not); the `ssa-comet-mtl` and `ssa-comet-mtl-final` model cards print **the same 75-entry list as the AfriCOMET card** (copy of the 76L card; the -final card says `afro-xlmr-large-114L`; the 114L card itself also prints only the 75-entry list). I could not obtain a source that enumerates the 110 African languages of the 114L encoder; treat the card list as a lower bound.
* Variants: SSA-COMET-STL, SSA-COMET-MTL (reference-based; also usable as QE with src+mt), SSA-COMET-QE (reference-free). Released as `McGill-NLP/ssa-comet-{stl,mtl,qe}` (+ `-final`). TranslatePsy uses `ssa-comet-mtl`; AfriqueLLM does not name a checkpoint.
* Results in the SSA-COMET paper (avg Spearman on SSA-MTE test): AfriCOMET v1.1 STL 0.500, v1.0 MTL 0.513, v1.1 MTL 0.550, SSA-COMET STL 0.586, SSA-COMET MTL 0.639, Gemini-2.5-Pro prompting 0.633; QE: AfriCOMET v1.1 MTL (QE mode) 0.490, SSA-COMET-QE 0.547, SSA-COMET-MTL (QE mode) 0.595.

## Per-family notes

* **NLLB-200** (paper arXiv 2207.04672): Section 7.1 discusses COMET and BLEURT and then does **not** use them (chrF++, spBLEU, XSTS human). Release `metrics.csv` has a single `chrf++` column (header verified for dense 3.3B). COMET-family values exist only in TranslatePsy (600M, 1.3B, 3.3B) and AfriNLLB (600M).
* **AfriNLLB** (arXiv 2602.09373): FLORES-200 **devtest**; BLEU, chrF++ (sacreBLEU), AfriCOMET `africomet-mtl` for African languages and `wmt22-comet-da` for Arabic and European languages; per-direction table (Table 7) with COMET for each of afr amh arz hau som swh wol yor zul en->xx and the reverse. Whether arz was scored with AfriCOMET or wmt22-comet-da is ambiguous in the paper text ("Arabic").
* **AfriqueLLM** (arXiv 2601.06395): SSA-COMET replaces chrF++ in the main text; Table 20 (en->xx) / 21 (xx->en) per language, Tables 22/23 chrF++. Checkpoint, ref/QE and FLORES split not stated. Scale 0-100.
* **TranslatePsy-AfriSLM** (arXiv 2608.18655 v2, HF card `qvac/TranslatePsy-afriSLM-4B`): checkpoints named in footnotes (`Unbabel/wmt22-comet-da`, `McGill-NLP/ssa-comet-mtl`, `google/metricx-24-hybrid-xl-v2p6`; QE filtering used `Masakhane/africomet-qe-stl-1.1` and `ssa-comet-qe`, `metricx-24-hybrid-xl-v2p6withoutreference`). Evaluation sets FLORES-200 devtest, BOUQuET test, Smol smolsent. The model card prints SSA-COMET only (0.6143 FLORES-200 for the 4B). Footnote 8 prints 20 language names (adds pcm) while the text and the card say 19; pcm is not in the 19 of the card.
* **TranslateGemma** (arXiv 2601.09012): MetricX-24 and Comet22 on WMT24++ (55 pairs), Vistra images; the text lists MetricX-24-XXL-QE as an RL reward. No chrF++ and no African per-pair COMET22.
* **Hunyuan-MT**: XCOMET-XXL and CometKiwi on FLORES-200 / WMT24pp / Mandarin-minority; no African language in the report.
* **Qwen3 / Qwen3.5**: Qwen3 report has no COMET; the Qwen3.5-4B card gives one XCOMET-XXL WMT24++ aggregate (66.6).
* **AfroBench** (arXiv 2311.07978): chrF for MT (Table 4 pooled), BERTScore column for other tasks; AfriCOMET and COMET appear only in the Limitations paragraph ("promising ... however, limited language coverage ... we rely on chrF++"). Leaderboard Space unreadable.
* **LingBench++** (arXiv 2507.16809): Gemini-2.5-Flash, chrF (not chrF++), first 10 FLORES-200 sentences of dev+devtest per language; no COMET.
* **Aya-101, MADLAD-400, Lugha-Llama, InkubaLM, Apertus, GPT-4o, Gemini 1.5 Pro, Gemma 2/3 reports**: no COMET-family number for African languages in the papers (text search), see csv.

## Appendix: AfriqueLLM Table 20, SSA-COMET en->xx (FLORES-200, 0-100), our 21 languages
Mean over the 21 languages that are in our 77 (afr amh ewe hau ibo kin lin lug nya orm sna som sot swa tir tsn twi wol xho yor zul); the 24-language average of the paper (which includes aeb ary arz) is in the csv.

| model | mean over our 21 | paper avg (24) |
|---|---|---|
| Llama3.1-8B | 21.59 | 25.73 |
| Lugha-Llama-8B-wura | 43.51 | 44.55 |
| AfriqueLlama-8B | 55.39 | 55.94 |
| Gemma3-4B | 32.67 | 34.01 |
| AfriqueGemma-4B | 51.26 | 48.91 |
| Gemma3-12B | 40.45 | 41.44 |
| AfriqueGemma-12B | 58.59 | 58.93 |
| Qwen3-4B | 15.85 | 20.47 |
| AfriqueQwen-4B | 51.76 | 52.38 |
| Qwen3.5-4B | 26.21 | 30.01 |
| AfriqueQwen3.5-4B | 55.17 | 55.78 |
| AfriqueQwen3.5-4B-ExtendedCM | 55.19 | 55.76 |
| AfriqueQwen3.5-4B-50Langs | 62.86 | 62.85 |
| Qwen3-8B | 16.00 | 21.12 |
| AfriqueQwen-8B | 53.88 | 54.52 |
| Qwen3-14B | 18.86 | 23.69 |
| AfriqueQwen-14B | 55.65 | 56.19 |

(The per-language rows are in the arXiv HTML `https://arxiv.org/html/2601.06395`, Table 20. AfriqueGemma-4B is the only row whose 21-language mean is higher than its 24-language average by more than 1.5 points, i.e. its aeb/ary/arz values are low; this is what the table prints and the average was recomputed from the row and matches.)

## Uncertainties
* Several reads were first obtained through a summarising fetch tool; the numbers in the csv for AfriqueLLM Tables 20-23 and TranslatePsy Tables 24/25 were re-read from the HTML source. TranslateGemma Table 1/4 numbers were read from the PDF text. AfriNLLB Tables 4/7 were read from the PDF text. The Qwen3.5-4B card value (66.6) and the SSA-COMET paper numbers (Table 3/4 averages) come from a summarising fetch only.
* The TranslatePsy direction (both averaged) is an inference shared with `published_baselines_2026-10-21.md`; the captions do not say.
* AfriNLLB: which checkpoint (`africomet-mtl` unversioned = v1.0 as listed in the repo README) and ref-vs-QE mode are not stated in the paper.
* The TranslateGemma HF card prints "Comet 81.6" for 4B, which disagrees with Table 1 of the report (80.1 for TranslateGemma 4B, 77.2 for Gemma 3 4B); the csv follows the report.
