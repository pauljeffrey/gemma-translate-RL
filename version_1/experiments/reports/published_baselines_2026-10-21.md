# Published English -> African translation results next to ours: baselines collected from papers, model cards and repos (2026-10-21)

Agent A3, web research only (no GPU, nothing pushed, nothing re-run). Decision D-51. Machine-readable table: `reports/data/published_baselines.csv`
(685 rows; columns `model, version, iso, flores_code, benchmark, direction, chrF++, BLEU_spBLEU, bleu_kind, metric_note, source_url, source_loc, lang_set, flag`).
Join with our per-language csv: `gemma_mt/scripts/join_published_vs_ours.py` -> `reports/comparison_with_published_2026-10-21.md`.
A first run of the join against the 2026-10-18 SFT results (stand-in, NOT the final model) is `reports/comparison_with_published_2026-10-21_PRELIMINARY_sft_2026-10-18.md`.
`NA` = not published / not covered. Nothing below is estimated from a plot.

## 1. Headline (read this first)

1. **Only two sources publish per-language en->X chrF++ tables that cover our languages**: the NLLB-200 release (`metrics.csv`, 44 of our 77 languages, chrF++ only) and the AfriqueLLM paper (Appendix Table 22, 21 of our 77, chrF++ only; covers AfriqueLlama-8B, AfriqueGemma-12B, AfriqueQwen-14B and 14 more rows). Nobody publishes a per-language **BLEU** for these systems; the only per-language spBLEU is NLLB-200 **54.5B MoE** (not the 1.3B / 3.3B dense models).
2. **TranslatePsy-AfriSLM-4B: 0 of 77 languages with a per-language number.** Its model card, GitHub repo and blog give 19-language macros only (SSA-COMET on the card; chrF++ / spBLEU in paper Table 25). Per-language scores exist only as figures (Fig. 11-13), which we did not read.
3. **The TranslatePsy paper's macros (Tables 24/25) are NOT en->X.** They are averages over both directions (en->X and X->en); we verified this by recomputation (section 4). Do not put them next to our en->X numbers.
4. **TranslateGemma 4B/27B: no chrF++ and no BLEU anywhere** in the technical report or the model card (MetricX-24 and COMET-22 on WMT24++ only; African pairs en-sw_KE, en-sw_TZ, en-zu_ZA). The only chrF++/spBLEU for TranslateGemma are the 19-language bidirectional macros of the TranslatePsy paper.
5. **Gemini: no per-language benchmark table of a Gemini version was found in the papers named in the brief** (TranslatePsy and AfriqueLLM contain no Gemini evaluation; TranslateGemma only used Gemini 2.5 Flash to generate synthetic training data). The only per-language Gemini en->X numbers found are in the LingBench++ paper (Gemini 2.5 Flash, chrF not chrF++, **10 sentences per language**), included flagged `low_n`. AfroBench gives pooled Gemini 1.5 Pro / GPT-4o / Aya-101 chrF averages (section 5).

Coverage of our 77 languages by a published per-language en->X chrF++:

| system | languages (of our 77) | BLEU / spBLEU |
|---|---|---|
| NLLB-200 dense 1.3B (and distilled 1.3B) | 44 | NA |
| NLLB-200 dense 3.3B | 44 | NA |
| NLLB-200 MoE 54.5B (extra) | 44 | spBLEU (SPM-200) for the same 44 |
| AfriqueLlama-8B | 21 | NA |
| AfriqueGemma-12B | 21 | NA |
| AfriqueQwen-14B | 21 | NA |
| Gemini 2.5 Flash (LingBench++, n=10, chrF) | 42 (flagged `low_n`, excluded from macros by default) | NA |
| TranslateGemma 4B / 27B | 0 (MetricX only for swh, zul) | NA |
| TranslatePsy-AfriSLM-4B | 0 (19 languages trained and evaluated, macro only, bidirectional) | NA |

The union of the NLLB and Afrique languages is the same 44 (Afrique's 21 are a subset of NLLB's 44). **33 of our 77 languages have no published number from any system we cover** (pcm efi ibb urh ven nbl tiv bin dag oke shi bbj swc kqn lun iso loz wal ach tsc vmw kua seh nyn her yao sid ndc kri ish dua bas wes).

## 2. Sources, exactly

| system | benchmark | direction | what is published | source |
|---|---|---|---|---|
| NLLB-200 dense 3.3B, dense 1.3B, distilled 1.3B | FLORES-200 devtest (NLLB paper section 8.3; the csv does not name the split) | en->X, 1 row per direction | chrF++ only, 1 decimal | `https://dl.fbaipublicfiles.com/large_objects/nllb/models/nllb_200_dense_3b/metrics.csv` (`.../nllb_200_dense_1b/...`, `.../nllb_200_dense_distill_1b/...`); linked from the HF model cards via tinyurl.com/nllb200dense3bmetrics etc.; fairseq `nllb` branch `examples/nllb/modeling/README.md` ("only chrF++ ... 402 English-centric + 200 sampled directions") |
| NLLB-200 MoE 54.5B (extra) | same | all 40,602 directions | chrF++, spBLEU (SPM-200), spBLEU (SPM-100) | `https://tinyurl.com/nllb200moe54bmetrics` (1.3 MB csv) |
| AfriqueLlama-8B, AfriqueGemma-4B/12B, AfriqueQwen-4B/8B/14B + bases | FLORES-200 through the AfroBench-Lite lm-eval task (split not stated) | en->X (also X->en, Table 23) | chrF++ per language (Table 22) and SSA-COMET (Table 20); 24 FLORES languages, 21 of them ours | arXiv 2601.06395 (ACL 2026), HTML v3, Appendix Tables 20-23; code `github.com/McGill-NLP/AfriqueLLM` (`eval/`), models `huggingface.co/collections/McGill-NLP/afriquellm` |
| TranslatePsy-AfriSLM 0.8B/2B/4B, plus 22 baselines | FLORES-200 devtest 1,012 sent., BOUQuET test 854, Smol smolsent 863 | **both directions averaged (inferred, section 4)** | 19-language macros of C22, SSA-COMET, MetricX, chrF++, spBLEU (Tables 24, 25) | arXiv 2608.18655 v2 (EMNLP 2026), `huggingface.co/qvac/TranslatePsy-afriSLM-4B`, blog `huggingface.co/blog/qvac/translate-psy-afrislm`, code `github.com/tether-ai-research/qvac-translatepsy-afri-slm` (data processing and chat only, no result files) |
| TranslateGemma 4B/12B/27B | WMT24++ (55 pairs), WMT25 human MQM (10), Vistra (4) | en->X | MetricX-24 and COMET-22 (per language pair MetricX in Appendix A Table 4) | arXiv 2601.09012, HF `google/translategemma-*-it` |
| Gemini 2.5 Flash | first 10 sentences of FLORES-200 dev+devtest per language | en->X and X->en | chrF (NOT chrF++) per language, Appendix G Table 9 | arXiv 2507.16809 (LingBench++) |
| Gemini 1.5 Pro, GPT-4o (Aug), Aya-101, NLLB 3.3B, ... | AfroBench MT (Flores + MAFAND + NTREX pooled) | en/fr->X | pooled chrF only (Table 4); per-language heat-maps only | arXiv 2311.07978 |

Metric implementations stated by the sources:
* TranslatePsy paper: chrF++ = sacreBLEU, char order 6, word order 2, beta 2, case-sensitive, whitespace excluded, no epsilon smoothing; spBLEU = sacreBLEU with the Flores-200 tokenizer (SPM-200). Same as our `chrF++ (word_order=2)` / `spBLEU (flores200)`.
* NLLB: `sacrebleu -m chrf --chrf-word-order 2` (flores repo `flores200/README.md`), spBLEU with the NLLB SPM-200 tokenizer; decoding of the release csv is beam search (not stated per file).
* AfriqueLLM: "chrF++ from the official AfroBench"; implementation / sacreBLEU version not stated; base models 5-shot, `do_sample=False`, vLLM or HF.
* Decoding of the TranslatePsy baselines (Table 15): general LLMs and TranslateGemma greedy, `max_new_tokens=256`; NLLB greedy with forced target BOS; Afrique 5-shot with random FLORES **dev** examples, greedy, 256 new tokens. Prompt: system "You are a professional SOURCE_LANG to TARGET_LANG translator ..." + "Please translate the following SOURCE_LANG text into TARGET_LANG".

## 3. Macro averages

### 3.1 Published systems over their own coverage of our 77 languages (en->X chrF++, no model of ours involved)

| system | languages | macro chrF++ |
|---|---|---|
| NLLB-200 1.3B | 44 | 38.3 |
| NLLB-200 distilled 1.3B | 44 | 38.5 |
| NLLB-200 3.3B | 44 | 39.0 |
| NLLB-200 MoE 54.5B | 44 | 39.2 |
| AfriqueLlama-8B | 21 | 36.7 |
| AfriqueGemma-12B | 21 | 38.6 |
| AfriqueQwen-14B | 21 | 36.7 |
| Gemini 2.5 Flash (n=10, chrF) | 42 | 38.7 (noisy, different metric) |

On the 21 languages Afrique covers (afr amh ewe hau ibo kin lin lug nya orm sna som sot swh tir tsn twi wol xho yor zul), NLLB-200 1.3B / 3.3B / MoE reach 42.3 / 43.3 / 43.7, against 36.7 / 38.6 / 36.7 for AfriqueLlama-8B / AfriqueGemma-12B / AfriqueQwen-14B (the Afrique models are weak on ewe, lin, lug, twi, wol: 6-26 chrF++).

### 3.2 Against OUR model: PRELIMINARY, with the 2026-10-18 SFT checkpoint as a stand-in

The final-model csv `reports/data/final_model_per_language.csv` did not exist yet. As a stand-in the join was run on `reports/gemma_sft_benchmarks_2026-10-18/full_flores/results.csv` (checkpoint `sft`, FLORES+ devtest, 45 languages). Treat these as placeholders; they change when the final model arrives.

| published system | shared languages | macro ours, all rows | macro ours, clean rows (48 sentences / language) | macro system |
|---|---|---|---|---|
| NLLB-200 1.3B | 43 | 30.9 | 27.7 | 38.1 |
| NLLB-200 3.3B | 43 | 30.9 | 27.7 | 38.8 |
| NLLB-200 MoE 54.5B | 43 | 30.9 | 27.7 | 39.1 |
| AfriqueLlama-8B | 21 | 37.1 | 33.1 | 36.7 |
| AfriqueGemma-12B | 21 | 37.1 | 33.1 | 38.6 |
| AfriqueQwen-14B | 21 | 37.1 | 33.1 | 36.7 |

Our SFT stand-in is below NLLB-200 3.3B and 54.5B on all 43 shared languages (all rows; below 1.3B on 41) and above each Afrique model on 5 of the 21 shared languages. The "all rows" macro is inflated by contamination (95% of the English devtest side was in SFT). spBLEU (SFT 10-18 vs NLLB 54.5B on 43 languages): 10.8 vs 17.9. Our macro over all 45 evaluated languages: 30.2 (all) / 27.0 (clean).

### 3.3 The 19-language bidirectional macros of the TranslatePsy paper (FLORES-200 devtest)

Languages: afr amh hau ibo kin lin lug mlg nya orm sna som sot swh tsn wol xho yor zul (the paper's footnote lists 20 with pcm; pcm is not in FLORES-200, so 19). Averages of en->X and X->en (section 4); **not comparable with an en->X-only number**.

| system | chrF++ | spBLEU |
|---|---|---|
| TranslatePsy-AfriSLM-4B | 51.13 | 30.28 |
| TranslatePsy-AfriSLM-2B | 50.00 | 28.95 |
| TranslatePsy-AfriSLM-0.8B | 48.30 | 27.00 |
| NLLB-200 3.3B | 48.89 | 28.06 |
| NLLB-200 1.3B | 47.77 | 26.80 |
| AfriqueGemma-12B | 47.84 | 28.48 |
| AfriqueQwen-14B | 46.16 | 26.82 |
| AfriqueLlama-8B | 45.04 | 25.57 |
| Qwen3.5-122B-A10B | 44.42 | 21.89 |
| TranslateGemma-27B | 43.33 | 21.13 |
| TranslateGemma-12B | 39.48 | 16.66 |
| TranslateGemma-4B | 24.78 | 10.21 |

SSA-COMET (model card / Table 3, same bidirectional 19 languages): TranslatePsy-4B 0.6143, NLLB-3.3B 0.5944, TranslateGemma-27B 0.5455, TranslateGemma-4B 0.3913, AfriqueGemma-12B 0.5655. BOUQuET and Smol columns are in the csv (`benchmark` column).
For an en->X comparison on the same 19 languages, the comparable en->X-only macros are: NLLB-200 1.3B 44.2, 3.3B 45.2 (release csv), AfriqueLlama-8B 40.6, AfriqueGemma-12B 42.5, AfriqueQwen-14B 40.7 (Afrique Table 22, 18 languages, no mlg); SFT 10-18 stand-in 39.4 (all rows) / 35.2 (clean).

## 4. Why Tables 24/25 of the TranslatePsy paper are bidirectional

The table captions do not name a direction (Table 21, the QE ablation, says "English-African", Fig. 6/11 say "averaged over both translation directions"). Recomputation from other published sources, same 19 languages (18 for Afrique, no mlg):

| system | en->X from source | X->en from source | mean of the two | TranslatePsy Table 25 |
|---|---|---|---|---|
| NLLB-200 3.3B chrF++ | 45.2 (release csv) | 54.4 (release csv) | 49.8 | 48.89 |
| NLLB-200 54.5B spBLEU | 23.4 | 36.9 | 30.2 | (3.3B: 28.06) |
| AfriqueGemma-12B chrF++ | 42.5 (Afrique T22) | 51.9 (T23) | 47.2 | 47.84 |
| AfriqueLlama-8B chrF++ | 40.6 | 49.2 | 44.9 | 45.04 |
| AfriqueQwen-14B chrF++ | 40.7 | 51.3 | 46.0 | 46.16 |

The en->X-only values are 3.7-5.5 points below the paper's numbers for every system, the mean of both directions is within 1 point. Greedy decoding explains the NLLB gap (beam vs greedy). This is an inference, not a statement by the authors: ask them or check Appendix B / the code if it matters.

## 5. Other baselines met (not chased)

* AfroBench (arXiv 2311.07978, Table 4, column MT en/fr->xx, pooled Flores + MAFAND + NTREX, best of 3 prompts, 0-shot chrF): Gemini 1.5 Pro 37.6 (Table 6, 0-shot: 35.4), GPT-4o (Aug) 35.1 (Table 6: 31.5), NLLB 3.3B 40.4, Aya-101 13B 23.4, Gemma 2 27B 27.9, LLaMa 3.1 70B 25.1 (rows in the csv with `iso = MACRO_AFROBENCH`, `flag = aggregate`). Per-language values only as heat-maps (Figs 24-29).
* Gemini 2.5 Flash, LingBench++ (`temperature 0.1`, thinking budget 0, zero-shot, 10 sentences per language, chrF word order 0): e.g. yor 25.9, hau 61.9, ibo 50.6, zul 58.6, swh 75.2, amh 35.9, fon 7.6. English->target output was empty for up to 7 of 10 sentences in some low-resource languages (Kabiye 3 of 10 non-empty, Fon 5 of 10; the csv benchmark field gives the non-empty count). Use only as a rough guide.
* MADLAD-400 (Hub `google/madlad400-*`), Aya-23, TranslateGemma-12B: mentioned in other papers; per-language en->X tables for our languages were not collected. MADLAD per-language Flores tables exist in the MADLAD-400 paper (not read).

## 6. Per-language published matrix (en->X chrF++, our 77 languages)

Cells are chrF++ (Gemini: chrF, n=10) or `-` (not covered / no number). `NLLB 54.5B spBLEU` is the only per-language BLEU-family number. `TranslateGemma`: `T` = the language appears in the SFT language list of the technical report (Appendix C; no score), `MetricX only` = a WMT24++ MetricX number exists. `TranslatePsy`: `T, NA` = one of its 19 trained and evaluated languages, per-language number not published in a table. FLORES code = NLLB / FLORES-200 code used for the lookup (Oromo `gaz_Latn`, Malagasy `plt_Latn`; ours Twi is Akuapem, theirs generic `twi_Latn`).

| iso | language | FLORES code | NLLB 1.3B | NLLB 3.3B | AfriqueLlama-8B | AfriqueGemma-12B | AfriqueQwen-14B | Gemini 2.5 Flash (n=10, chrF) | NLLB 54.5B spBLEU | TranslateGemma | TranslatePsy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yor | Yoruba | yor_Latn | 24.0 | 24.9 | 27.4 | 28.6 | 27.4 | 25.9 | 10.5 | T | T, NA |
| hau | Hausa | hau_Latn | 51.1 | 52.5 | 49.6 | 51.4 | 49.8 | 61.9 | 31.4 | T | T, NA |
| ibo | Igbo | ibo_Latn | 40.7 | 41.1 | 40.8 | 42.0 | 41.1 | 50.6 | 20.6 | T | T, NA |
| pcm | Pidgin | - | - | - | - | - | - | - | - | T | T, NA |
| efi | Efik | - | - | - | - | - | - | - | - | T | - |
| twi | Twi | twi_Latn | 37.2 | 38.3 | 13.9 | 15.6 | 12.8 | 40.1 | 15.2 | T | - |
| ewe | Ewe | ewe_Latn | 37.3 | 38.2 | 6.8 | 8.5 | 6.2 | 36.7 | 17.2 | T | - |
| fon | Fon | fon_Latn | 18.5 | 19.3 | - | - | - | 7.6 | 6.4 | T | - |
| wol | Wolof | wol_Latn | 26.2 | 28.1 | 8.9 | 7.1 | 8.4 | 27.2 | 9.6 | T | T, NA |
| swh | Swahili | swh_Latn | 59.2 | 60.0 | 59.5 | 62.1 | 59.5 | 75.2 | 36.8 | T, MetricX only | T, NA |
| kin | Kinyarwanda | kin_Latn | 46.3 | 48.4 | 46.4 | 49.6 | 46.5 | 59.1 | 27.2 | T | T, NA |
| som | Somali | som_Latn | 42.3 | 42.8 | 42.0 | 42.1 | 41.6 | 48.8 | 18.4 | T | T, NA |
| orm | Oromo | gaz_Latn | 34.0 | 35.7 | 35.5 | 37.0 | 36.4 | 43.9 | 12.6 | T | T, NA |
| amh | Amharic | amh_Ethi | 36.3 | 37.9 | 32.1 | 36.1 | 33.5 | - | 31.6 | T | T, NA |
| tir | Tigrinya | tir_Ethi | 23.7 | 24.8 | 19.8 | 20.9 | 18.4 | - | 17.8 | T | - |
| lug | Luganda | lug_Latn | 37.1 | 38.5 | 21.0 | 26.6 | 21.2 | 42.9 | 14 | T | T, NA |
| lin | Lingala | lin_Latn | 47.7 | 48.0 | 14.2 | 19.6 | 14.7 | 48.4 | 21.9 | T | T, NA |
| zul | Zulu | zul_Latn | 52.5 | 53.5 | 49.7 | 51.8 | 49.8 | 58.6 | 31.4 | MetricX only | T, NA |
| xho | Xhosa | xho_Latn | 48.4 | 48.6 | 47.4 | 49.2 | 46.5 | 51.6 | 25.4 | T | T, NA |
| afr | Afrikaans | afr_Latn | 63.5 | 64.7 | 67.5 | 69.0 | 67.6 | 73.9 | 44.4 | T | T, NA |
| sna | Shona | sna_Latn | 42.7 | 42.8 | 42.9 | 43.9 | 42.9 | 50.0 | 19.7 | T | T, NA |
| nya | Chichewa | nya_Latn | 44.6 | 45.3 | 45.6 | 46.9 | 45.9 | 57.3 | 17.7 | T | T, NA |
| sot | Sesotho | sot_Latn | 46.2 | 46.5 | 53.8 | 55.5 | 53.5 | 55.4 | 20.7 | T | T, NA |
| tsn | Setswana | tsn_Latn | 47.8 | 48.6 | 46.1 | 46.6 | 46.1 | 45.3 | 25.6 | T | T, NA |
| mlg | Malagasy | plt_Latn | 49.4 | 50.2 | - | - | - | 54.3 | 25.3 | T | T, NA |
| ibb | Ibibio | - | - | - | - | - | - | - | - | - | - |
| urh | Urhobo | - | - | - | - | - | - | - | - | - | - |
| bem | Bemba | bem_Latn | 36.4 | 37.5 | - | - | - | 43.0 | 13.6 | T | - |
| tum | Tumbuka | tum_Latn | 36.1 | 36.6 | - | - | - | 48.3 | 13.3 | T | - |
| run | Kirundi | run_Latn | 40.9 | 41.9 | - | - | - | 46.2 | 19.6 | T | - |
| ven | Venda | - | - | - | - | - | - | - | - | T | - |
| tso | Tsonga | tso_Latn | 50.0 | 50.2 | - | - | - | 53.8 | 26.7 | T | - |
| nso | Sepedi | nso_Latn | 49.8 | 50.3 | - | - | - | 58.9 | 26.5 | T | - |
| ssw | Swati | ssw_Latn | 43.3 | 43.5 | - | - | - | 47.5 | 19.9 | T | - |
| nbl | South Ndebele | - | - | - | - | - | - | - | - | T | - |
| sag | Sango | sag_Latn | 35.6 | 36.5 | - | - | - | 18.3 | 10.5 | T | - |
| tiv | Tiv | - | - | - | - | - | - | - | - | T | - |
| umb | Umbundu | umb_Latn | 25.5 | 23.6 | - | - | - | 19.9 | 4.1 | - | - |
| kmb | Kimbundu | kmb_Latn | 25.8 | 24.4 | - | - | - | 21.4 | 4.5 | - | - |
| mos | Mossi | mos_Latn | 23.3 | 24.1 | - | - | - | 15.5 | 6.8 | T | - |
| kik | Kikuyu | kik_Latn | 36.3 | 37.3 | - | - | - | 5.6 | 15.4 | T | - |
| luo | Luo | luo_Latn | 38.5 | 39.0 | - | - | - | 37.9 | 15.2 | T | - |
| kam | Kamba | kam_Latn | 25.7 | 23.6 | - | - | - | 24.9 | 6.1 | - | - |
| bin | Edo | - | - | - | - | - | - | - | - | - | - |
| dag | Dagbani | - | - | - | - | - | - | - | - | - | - |
| oke | Okpe | - | - | - | - | - | - | - | - | - | - |
| shi | Tachelhit | - | - | - | - | - | - | - | - | - | - |
| dyu | Dyula | dyu_Latn | 17.5 | 16.2 | - | - | - | 16.3 | 2.7 | T | - |
| bam | Bambara | bam_Latn | 30.7 | 31.4 | - | - | - | 2.1 | 9.3 | T | - |
| kbp | Kabiye | kbp_Latn | 27.5 | 29.5 | - | - | - | 0.4 | 11.3 | - | - |
| bbj | Ghomala | - | - | - | - | - | - | - | - | - | - |
| kon | Kongo | kon_Latn | 45.9 | 45.9 | - | - | - | 24.6 | 18.9 | T | - |
| kea | Kabuverdianu | kea_Latn | 45.6 | 47.9 | - | - | - | 58.0 | 22.5 | - | - |
| kab | Kabyle | kab_Latn | 29.7 | 32.0 | - | - | - | 32.0 | 16.9 | - | - |
| nus | Nuer | nus_Latn | 28.1 | 28.9 | - | - | - | 6.7 | 14.4 | T | - |
| swc | Congo Swahili | - | - | - | - | - | - | - | - | - | - |
| lua | Luba-Kasai | lua_Latn | 34.8 | 35.2 | - | - | - | 30.0 | 9.8 | - | - |
| kqn | Kaonde | - | - | - | - | - | - | - | - | - | - |
| lun | Lunda | - | - | - | - | - | - | - | - | - | - |
| iso | Isoko | - | - | - | - | - | - | - | - | T | - |
| loz | Lozi | - | - | - | - | - | - | - | - | - | - |
| wal | Wolaytta | - | - | - | - | - | - | - | - | - | - |
| ach | Acholi | - | - | - | - | - | - | - | - | T | - |
| tsc | Tswa | - | - | - | - | - | - | - | - | - | - |
| vmw | Makhuwa | - | - | - | - | - | - | - | - | - | - |
| kua | Kuanyama | - | - | - | - | - | - | - | - | - | - |
| seh | Sena | - | - | - | - | - | - | - | - | - | - |
| nyn | Nyankore | - | - | - | - | - | - | - | - | - | - |
| her | Herero | - | - | - | - | - | - | - | - | - | - |
| yao | Yao | - | - | - | - | - | - | - | - | - | - |
| sid | Sidamo | - | - | - | - | - | - | - | - | - | - |
| ndc | Ndau | - | - | - | - | - | - | - | - | T | - |
| kri | Krio | - | - | - | - | - | - | - | - | T | - |
| ish | Esan | - | - | - | - | - | - | - | - | - | - |
| dua | Duala | - | - | - | - | - | - | - | - | - | - |
| bas | Basaa | - | - | - | - | - | - | - | - | - | - |
| wes | Cameroon Pidgin | - | - | - | - | - | - | - | - | - | - |

## 7. Comparability caveats

1. **FLORES-200 vs FLORES+**: ours is FLORES+ devtest (same 1,012 English sentences, corrected references in some languages, Akuapem Twi). NLLB reports FLORES-200 devtest; the Afrique paper does not name the split; the TranslatePsy paper uses FLORES-200 devtest. LingBench++ takes the first 10 sentences of the combined dev + devtest.
2. **Contamination on our side**: our SFT saw the English side of 95% of FLORES devtest sentences (the reference pair only for 7 languages), so our "all rows" numbers are optimistic; the clean subset has 48 sentences per language. The published systems have no such advantage (NLLB used FLORES dev for model selection only).
3. **Prompting and decoding differ**: NLLB beam search with forced target token; Afrique base models 5-shot with random FLORES dev examples, greedy; TranslatePsy baselines greedy 256 tokens with a chat prompt; ours greedy with `Translate the text to {language}:` and a language tag. LLM numbers depend on the prompt (the TranslatePsy paper measures up to 1.4 chrF++ between prompt variants for Qwen3.5-27B / TranslateGemma-27B, Table 20).
4. **Metric implementations**: only the TranslatePsy paper spells out sacreBLEU settings; the NLLB csv is rounded to 1 decimal; the Afrique chrF++ implementation is not described; LingBench++ is chrF without word n-grams.
5. **Language sets differ** (44 vs 21 vs 19), so macros are only comparable over the same languages; the join script computes shared-language macros and a common-set macro.
6. **Direction**: everything in the per-language tables is English -> X. The TranslatePsy 19-language macros (both directions) and the AfroBench pooled averages (en/fr->X, three datasets) are kept in the csv with `flag = aggregate` and are excluded from per-language ranks.
7. **Noise**: the Gemini rows rest on 10 sentences; the clean subset of ours on 48.
8. The baselines in the TranslatePsy paper are single runs from the authors; Afrique numbers are from the Afrique authors; none was re-run by us.

## 8. What could not be found

* Per-language chrF++ / spBLEU / BLEU of **TranslatePsy-AfriSLM** (0.8B, 2B, 4B) on FLORES-200: only 19-language macros (Tables 24/25) and per-language figures for 2B vs Qwen3.5-2B (Fig. 11 BOUQuET, Fig. 12-13 OOD). The repo has no result files. A direct request to the authors (qvac / Tether research) is the only route.
* **en->X-only macro** of TranslatePsy and of TranslateGemma on FLORES-200 (the paper averages both directions).
* **Any chrF++ or BLEU for TranslateGemma** (the technical report and the HF card show MetricX / COMET / MQM only); WMT24++ has chrF as a heat-map only (Fig. 11 of arXiv 2502.12404).
* **Per-language Gemini / GPT-4o en->X tables** on a full FLORES devtest for African languages (only AfroBench pooled averages and the 10-sentence LingBench++ table).
* **Per-language BLEU** for NLLB-200 1.3B / 3.3B (release csv has chrF++ only); spBLEU exists only for the 54.5B MoE.
* The split (dev vs devtest) used by the AfroBench FLORES task of the Afrique paper, and the exact chrF++ implementation there.
* None of our 33 uncovered languages has any published en->X number (see section 1).
* The Afrique paper's X->en tables are in the csv source but not used.

## 9. Reproduction

All numbers were read from the files above on 2026-10-21 (arXiv HTML v2/v3, NLLB csv from dl.fbaipublicfiles.com, HF model cards). The csv was assembled by a throw-away script from those files; every row carries `source_url` and `source_loc` (table / column / row). Join: `python gemma_mt/scripts/join_published_vs_ours.py --ours reports/data/final_model_per_language.csv` (add `--subset clean`, `--models all`, `--include-low-n`). The script was tested on a synthetic final-model csv (44 shared languages, ranks, macros, spBLEU table, aggregates table) and on the 2026-10-18 SFT results.
