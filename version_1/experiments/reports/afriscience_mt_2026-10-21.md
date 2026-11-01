# AfriScience-MT: what it is, how it is scored, what the published systems reach (research for the science RL run, D-52)

Written 2026-10-21 by agent A4. Everything below is from the paper, the dataset card and the released per-run metrics (cited); numbers marked "ours" were measured on the rental box. Nothing here was verified by a human translator.

## 1. Sources

| what | where |
|---|---|
| paper | Idris Abdulmumin, Tajuddeen Gwadabe, Shamsuddeen Hassan Muhammad, David Ifeoluwa Adelani, Nomonde Khalo, Ibrahim Said Ahmad, Abiodun Modupe, Anina Mumm, Sibusiso Biyela, Michelle Rabie, Johanna Havemann, Marek Rei, Jade Abbott, Vukosi Marivate: *AfriScience-MT: Towards Decolonizing Science in Africa through Text Translation*, arXiv:2605.29741 (cs.CL, 28 May 2026, v1), https://arxiv.org/abs/2605.29741 (paper licence CC BY 4.0). Affiliations: Data Science for Social Impact (U. Pretoria), Masakhane, Imperial College London, Mila / McGill, U. Cape Town, Lelapa AI and others. |
| dataset (canonical) | https://huggingface.co/datasets/dsfsi/afriscience_mt (mirror: masakhane/afriscience_mt), **Apache-2.0**, not gated, created 2026-06-08, parquet. Configs: `corpus` (train / dev / test), `glossary`, `predictions` (1,982,558 per-sentence model outputs, 260 MB), `metrics` (2,934 per-run rows: BLEU, chrF, SSA-COMET, validation metrics). |
| code and models | CLI https://github.com/dsfsi/afriscience_mt (the page returned 404 to an unauthenticated fetch on 2026-10-21; the dataset card links it); fine-tuned models: Hub collection https://huggingface.co/collections/dsfsi/afriscience-mt-6a2136f66dafdf67456103b4 (NLLB / M2M100 / Gemma-2 / Gemma-3 / Tiny-Aya / Llama LoRA adapters per direction; the project README says the old `AfriScience-MT/...` links redirect to `dsfsi/...`). |

## 2. The data

* **Languages**: English plus six African languages: Amharic (amh), Hausa (hau), Luganda (lug), Northern Sotho / Sepedi (nso), Yoruba (yor), isiZulu (zul). All six are in our 77 RL languages. Multi-parallel: every English sentence is translated into all six (checked: the English side of `eng-hau` and `eng-zul` is identical in train, dev and test).
* **Directions**: the primary use is **English -> African** ("our experiments focus on English->Target"). The paper also reports Target -> English and a zero-shot African <-> African probe. For us only English -> X matters.
* **Domains**: 230 open-access English-language papers (AfricArXiv, Zenodo, Lens.org) in 11 domains: agriculture, biochemistry, biology, chemistry, computer science, engineering, geography, health, indigenous knowledge, sociology, statistics. Health dominates (2,621 of 5,792 train sentences). 70 % of the selection is core natural science; papers with African first authors were prioritised.
* **What is translated** (not the full papers: "scientific translation required significantly more time", up to 10x): a domain-expert science communicator wrote a 250-350 word **plain-language ("lay") summary** of each paper plus a glossary; a professional translator then translated the lay summary, the abstract, the terminology and the definitions; the lead translator per language reviewed everything. So the text is *simplified science communication*, abstract-style, not dense article body text.
* **Size**: 7,605 English sentences after cleaning (untranslated and single-word sentences removed; English sentences de-duplicated for the sentence-level data, repetitions kept in the document-level data). Splits are made per **paper** within each domain (80/10/10, every split has every domain):

| split | papers | sentences per language | English AfriXLM-R tokens per paper (ours) |
|---|---|---|---|
| train | 177 | 5,792 | median 986 (min 233, max 1,795); 97 papers in b1024, 77 in b2048 |
| dev | 25 | 843 | median 981 (542-1,346); 16 b1024, 9 b2048 |
| test | 28 | 970 | median 1,104 (623-1,495); 12 b1024, 16 b2048 |

  Sentences are short (median 30 AfriXLM-R tokens, p90 49, max 139). Glossaries: amh 333, hau 130, lug 181, nso 263, yor 587, zul 385 terms. Token counts of the target sides differ a lot (Northern Sotho 256k tokens for the whole corpus against 139k for Amharic and isiZulu and 180k for English; paper section 3.4).
* **Licence**: the dataset card says Apache-2.0. The source papers are open-access (the paper does not list a licence per paper); the translations are new work. The paper commits to maintaining the artefacts for five years and warns about commercial exploitation and low-quality re-deployment.
* **Document level**: a "document" is one paper's sentences concatenated (the card: "aligned at both the sentence and document level"). Our rebuild from the released sentence-level parquet (sentences joined by single spaces, in `sentence_id` order) gives the 28 test documents above; because the release is the de-duplicated sentence-level data, a rebuilt document can be slightly shorter than the paper's own document-level data (which kept repetitions). We did not find a separate document-level file.

## 3. How the test set is scored

* **Metrics** (paper section 4 and appendix C): **SSA-COMET** (Li et al. 2025, the African-tuned COMET; the *primary* metric, called "COMET" in the paper), **chrF** (Popovic 2015) as a surface check, and BLEU "computed for every prediction set but not reported" (it is in the released `metrics` config). Document level: additionally two LLM judges (GPT-5.5 and Gemini-3.1-Flash-Lite) for fluency, accuracy errors, lexical and grammatical mistakes. A large COMET minus chrF gap is used as a degeneration flag (fluent but off-target output).
* **Sentence level**: metrics "on individual sentence pairs" (the released `metrics` rows carry one corpus-level chrF per run; we report both the corpus chrF and the mean of per-sentence chrF). **Document level**: metrics on the concatenated document translations.
* The implementation of the chrF (which sacrebleu settings, word order) is not stated in the text we could read; the released CLI would settle it. We therefore report plain chrF (word order 0) as the paper's number and chrF++ next to it; do not compare chrF++ to the paper's chrF.
* **Models and settings**: seq2seq M2M100-418M/1.2B and NLLB-600M/1.3B fine-tuned per direction; Llama3-8B, Gemma2-9B, AfriqueLlama-8B, AfriqueQwen-8B, Tiny-Aya-Earth/Global fine-tuned with QLoRA (r in {4, 64}); TranslateGemma-12B zero/few-shot only; closed GPT-4o, Gemini-1.5-Flash, GPT-5.4, Gemini-3.1-Flash-Lite zero/few-shot. Sentence level: template 5, 10-shot, T=0; document level: 0- and 1-shot.
* **Headline results** (paper abstract and section 5): closed models best, GPT-5.4 and Gemini-3.1-Flash-Lite lead at sentence level (average SSA-COMET 68.3 and 68.0), tie at document level (48.3); fine-tuned NLLB-1.3B averages 67.3 SSA-COMET over 12 directions and is the best open system at sentence level; TranslateGemma-12B 1-shot reaches 44.0 at document level; fine-tuning on AfriScience-MT beats prompting any open LLM; English -> X is 1-3 COMET points harder than X -> English. Three open LLMs (AfriqueLlama-8B, AfriqueQwen-8B, TranslateGemma-12B at the sentence template) degenerate (COMET far above chrF).

## 4. Published chrF / BLEU / SSA-COMET, English -> X, test split

From the released `metrics` config (dataset `afriscience_mt`, split test; mean over the six languages is an unweighted average computed by us). Sentence-level systems first, then document-level ("doc:"). SSA-COMET is on a 0-1 scale here (the paper multiplies by 100). The sentence-level 10-shot GPT-5.4 / Gemini rows are flagged `is_ablation` in the release although the paper's Table 3 uses that configuration; we took them for that reason. Full csv: `reports/afriscience_science_run_2026-10-21/published_chrf_test_eng2x.csv`.

| system | amh | hau | lug | nso | yor | zul | mean chrF | mean BLEU | mean SSA-COMET |
|---|---|---|---|---|---|---|---|---|---|
| NLLB-200-1.3B fine-tuned | 45.5 | 67.7 | 51.5 | 64.3 | 17.7 | 67.2 | **52.3** | 29.2 | 0.672 |
| NLLB-200-distilled-600M fine-tuned | 44.4 | 66.9 | 50.6 | 64.2 | 17.6 | 66.3 | 51.7 | 28.4 | 0.667 |
| M2M100-1.2B fine-tuned | 39.0 | 63.1 | 49.5 | 62.4 | 16.6 | 58.4 | 48.2 | 24.4 | 0.633 |
| NLLB-200-1.3B zero-shot | 37.4 | 59.4 | 43.2 | 51.4 | 13.0 | 60.1 | 44.1 | 20.1 | 0.652 |
| Gemma2-9B LoRA r64 fine-tuned | 39.7 | 58.8 | 50.1 | 61.6 | 41.6 | 50.2 | 50.3 | 24.1 | 0.562 |
| GPT-5.4 10-shot | 43.8 | 63.0 | 56.4 | 59.5 | 28.1 | 68.9 | 53.3 | 22.3 | 0.678 |
| Gemini-3.1-Flash-Lite 10-shot | 41.6 | 61.1 | 49.5 | 55.9 | 22.9 | 61.8 | 48.8 | 21.3 | 0.678 |
| doc: GPT-5.4 1-shot | 42.8 | 66.8 | 59.5 | 64.9 | 28.7 | 68.6 | 55.2 | 22.0 | 0.448 |
| doc: GPT-5.4 0-shot | 26.0 | 66.1 | 59.0 | 63.3 | 19.8 | 66.1 | 50.0 | 19.7 | 0.454 |
| doc: Gemini-3.1-Flash-Lite 1-shot | 45.1 | 64.3 | 50.8 | 55.8 | 17.5 | 62.9 | 49.4 | 21.6 | 0.450 |
| doc: TranslateGemma-12B 1-shot | 20.4 | 33.3 | 15.6 | 21.5 | 7.0 | 32.7 | 21.8 | 4.3 | 0.382 |
| doc: Gemma2-9B-IT 1-shot | 3.8 | 30.9 | 8.2 | 10.4 | 2.7 | 12.6 | 11.4 | 1.6 | 0.324 |

Reading it: Yoruba is the outlier everywhere (chrF 13-29 for the systems that do not fine-tune the LLM with LoRA; only the Gemma2-9B and Tiny-Aya LoRA fine-tunes reach 37-42 there; we did not investigate why); Amharic is 20-45 chrF because of the Ge'ez script and its character n-grams.

## 5. Consequences for the science run

* The benchmark is small (970 test sentences per language; 28 papers) and lay-summary style: the English text is easy, short, number-poor, so the **science anchor** (numbers, units, formulas) will be less decisive on the benchmark than on the arXiv / PMC / LibreTexts windows used for training. The long documents (600-1,500 AfriXLM-R tokens) are in our b1024 and b2048 bins; sentence level is our b64 bin.
* Sentence-level scoring matches what published numbers cover; document-level scoring is the more interesting test for an RL run on long text and has a thin published baseline (GPT-5.4 1-shot 55.2 chrF, 22.0 BLEU).
* SSA-COMET is not available in our reward (we use AfriCOMET-QE / consensus); we report chrF, chrF++, BLEU, spBLEU only. A comparison with the paper on SSA-COMET would need the SSA-COMET checkpoint (not run here).
* 2026-10-21 caveat: the released train split is de-duplicated at sentence level and the "documents" we rebuild are the de-duplicated ones.

## 6. Where our final general model stands (ours, measured 2026-10-21)

`<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c@latest` (step 1433), greedy, English -> X, test split (`scripts/eval_afriscience.py`, `reports/afriscience_science_run_2026-10-21/baseline_final_1433/`):

| level | amh | hau | lug | nso | yor | zul | macro chrF | macro chrF++ | macro BLEU | finished |
|---|---|---|---|---|---|---|---|---|---|---|
| sentence (970 / language) | 25.3 | 59.4 | 46.2 | 55.7 | 37.4 | 63.4 | **47.9** | 44.9 | 20.0 | 100 % |
| whole paper (28 / language) | 22.1 | 49.1 | 30.8 | 34.2 | 21.8 | 40.8 | **33.1** | 29.3 | 5.0 | 73 % |

Against the table in section 4: sentence level is 4.4 chrF below fine-tuned NLLB-1.3B (52.3) and 5.4 below GPT-5.4 10-shot (53.3), above Gemini-3.1-Flash-Lite 10-shot's Yoruba by 14.5 and the NLLB fine-tune's Yoruba by 19.7, 20 points below it on Amharic; whole-paper level is 22 chrF below GPT-5.4 1-shot (55.2) and 16 below Gemini-3.1-Flash-Lite 1-shot (49.4) but 11 above TranslateGemma-12B 1-shot (21.8); 27 % of the papers hit the 7,168-token budget in a repetition loop. These numbers are chrF, not SSA-COMET.
