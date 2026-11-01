# African-NLP literature notes (compiled 2026-10-22)

Scope: African half of the lit review for "GRPO with a QE reward ensemble on Gemma-3-270m-it, EN->77 African languages, extended to documents up to 2,048 tokens".
Every entry in `refs_africa.bib` (78 entries) was checked against a fetched arXiv abs page (citation meta tags), an ACL Anthology .bib, or a fetched page. Numbers below were seen in a source I fetched, or in a web-search summary of the abstract. Items marked (S) rest only on a search summary and should be re-read before quoting.

Process warning: several arXiv ids I first guessed from memory were wrong (they resolved to unrelated astronomy and math papers). All ids below were re-checked by fetching the abs page.

---

## 1. NOVELTY CHECK

### Verdict
The claim "nobody has trained a ~270M model with RL to translate into low-resource African languages, extended to 2,048-token documents" holds as far as I can find. Each half has close neighbours, so wording matters. Safe wording: "to our knowledge the first RL-trained sub-1B model for EN->African translation covering both sentence and document level". Do not claim first RL for African MT, first small model for African MT, or first African document-level MT.

### Evidence FOR novelty (nothing found that does the combination)
- Document-level African MT work exists only as benchmarks and large-model studies. AfriDoc-MT: 5 African languages, models 1.2B to 13B, no RL, no sub-1B (Alabi et al., arXiv 2501.06374, 2025-01-10; EMNLP 2025). https://arxiv.org/abs/2501.06374
- The smallest African-oriented LLM MT work is TranslatePsy-AfriSLM: 0.8B / 2B / 4B, from Qwen3.5, full-parameter SFT only, no RL, sentence-level benchmarks only (FLORES-200, BOUQuET, Smol) (Gritta et al., arXiv 2608.18655, 2026-08-19; EMNLP 2026 main). https://arxiv.org/abs/2608.18655
- TranslateGemma uses SFT then RL with a reward-model ensemble (MetricX-QE, AutoMQM), but only at 4B/12B/27B, and its abstract names neither African languages nor documents (Finkelstein et al., arXiv 2601.09012, 2026-01-13). https://arxiv.org/abs/2601.09012
- RL-for-MT papers I found use models of 600M and up. Nothing at 270M. WALAR, MT-R1-Zero and the MiLMMT papers are sentence-level and do not target African languages (abstracts).
- No paper found that uses AfriCOMET or SSA-COMET as an RL reward. (The only "AfriCOMET as RL reward" artefact I saw was the owner's own HF dataset `<HF_ORG_LEGACY>/africomet-translation`; do not cite it as external evidence.)
- The Gemma-3-270m-it model itself: Google's blog (2025-08-14) positions it for task-specific fine-tuning; the Gemma 3 report (2025-03-25) does not cover 270M. https://developers.googleblog.com/en/introducing-gemma-3-270m/
- Searches for "GRPO document-level MT" returned only sentence-level GRPO papers and a pre-GRPO document-level RL paper (arXiv 2010.04314, title seen in a search result, not verified, not in bib).

### Evidence AGAINST (neighbours that limit the claim; cite and differentiate)
1. **RL + African languages + small models already touched.** Attia & Aji, "Improving Low-Resource MT via Round-Trip RL" (arXiv 2601.12535, 2026-01-18): GRPO on NLLB-600M and 1.3B, reward chrF++/BLEU on the reconstructed English, languages include Wolof and Dyula (African), plus Aymara, Friulian, Bhojpuri, Russian. Sentence-level, not Gemma, not QE reward. https://arxiv.org/abs/2601.12535
2. **Reference-free GRPO on NLLB.** Garcia-Estrada et al. (arXiv 2605.15976, 2026-05-15): NLLB 600M/1.3B, reward LaBSE + COMET-Kiwi (a QE-type reward), 13 languages; a web-search summary says Swahili and Yoruba are included, so it counts as RL on African targets with a small model (S for the language list). https://arxiv.org/abs/2605.15976
3. **Sunflower** (Sunbird AI, arXiv 2510.07203, 2025-10-08): Qwen3 14B/32B, 31 Ugandan languages, post-training with IRPO (a DPO variant described as RL). No GRPO in the paper body. The Sunbird model page mentions a GRPO-tuned 14B checkpoint (search summary only; unverified). https://arxiv.org/abs/2510.07203
4. **Small African MT models exist:** TranslatePsy-AfriSLM (0.8B, beats TranslateGemma-27B and Qwen3.5-122B on its SSA-COMET table; vendor-reported), AfriNLLB (pruned NLLB-600M, arXiv 2602.09373, 2026-02-10), InkubaLM (0.4B, arXiv 2408.17024, 2024-08-30).
5. **Long-context African MT exists at larger scale.** AfriqueLLM (arXiv 2601.06395, 2026-01-10; ACL 2026 main) trains 4B to 14B Llama/Gemma/Qwen with 16K context and reports AfriDoc-MT document translation: AfriqueGemma-12B reaches d-chrF 60.2 EN->African vs 47.8 for a Llama 3.1 8B SFT10 baseline. https://arxiv.org/abs/2601.06395 AfriScience-MT (arXiv 2605.29741, 2026-05-28) reports document-level COMET for 11 LLMs on 6 African languages. https://arxiv.org/abs/2605.29741
6. **Sub-1B / Gemma-based RL translation generally:** TranslateGemma is the direct methodological ancestor (Gemma 3 + SFT + RL with QE-like reward ensemble), and MiLMMT-46 (arXiv 2602.11961, 2608.10812) is Gemma-3-based with a two-QE-model RL reward. The reward-ensemble idea is therefore not new in itself; the novelty is the combination (270M, African targets, doc-level).
7. **Gaps in my search:** OLDI/WMT 2026 shared tasks and anything posted after mid-Oct 2026 are not covered. A re-search before submission is advised (search "GRPO" + "African" + "translation", "270M translation", "document-level" + "Gemma 3").

### Document length context for the 2,048-token claim
- AfriDoc-MT documents average about 30 to 37 sentences and about 650 whitespace tokens of English (fetched HTML body). In an African-language target a 270M Gemma tokenizer may need several times more tokens per word, which is where 2,048 tokens could become tight. I did not measure that.
- NLLB model cards state training inputs were capped at 512 tokens and the models are "not intended to be used for document translation" (fetched for 1.3B; 600M via search result). That supports the point that standard African MT baselines are sentence-bound.
- AfriDoc-MT finding useful for motivation: sentence-by-sentence beats pseudo-document translation into African languages for most models (GPT-4o is the exception); SFT on 10-sentence pseudo-documents generalizes better to long inputs than sentence SFT; d-chrF falls as document length grows; open LLMs show off-target output, under-generation (<70% of reference length) and repetition.
- WMT24++ is paragraph-level (998 segments from 171 documents, average 189 tokens per document); the fetched table listed only ar_EG, sw_KE, sw_TZ, zu_ZA as African/Arabic-in-Africa (the summariser may have truncated the list). SMOLDOC (584 LLM-generated documents) and BOUQuET (paragraph-level) are the other document-flavoured eval sources.

---

## 2. Entries (key: what it is / fact useful to us)

### MT data, benchmarks, evaluation
- **adelani2022mafand** (NAACL 2022; arXiv 2205.02022, 2022-05-04). MAFAND-MT: news-domain corpus for 16 African languages, 8 of them absent from any prior evaluation set (abstract). Finding: fine-tuning large pre-trained models on a few thousand high-quality sentence pairs is the best way to add languages and domains. Our sentence-level training/eval data likely overlaps this.
- **reid2021afromt** (EMNLP 2021; arXiv 2109.04715). AfroMT: benchmark for 8 African languages plus word-alignment and pseudo-monolingual augmentation for pretraining mT5/mBART-style models; up to +2 BLEU over strong baselines, up to +12 BLEU in data-constrained settings (abstract).
- **nekoto2020participatory** (Findings EMNLP 2020; arXiv 2010.02353). Masakhane: participatory MT research; benchmarks for 30+ African languages, human evaluation for a third (abstract). I did not verify its data-source breakdown, so do not cite it for the JW-heavy claim.
- **wang2024africomet** (NAACL 2024; arXiv 2311.09828). AfriMTE human data (simplified MQM and DA) for 13 languages, and AfriCOMET built on AfroXLM-R; best Spearman 0.441 with human judgments (abstract). One of our reward components if used.
- **li2025ssacomet** (EMNLP 2025; arXiv 2506.04557, 2025-06-05). SSA-MTE: 14 African pairs, >73k sentence annotations (abstract; the EMNLP version says 13 pairs / >63k per search summary, so check version). SSA-COMET and SSA-COMET-QE significantly beat AfriCOMET and are competitive with Gemini 2.5 Pro as judge, notably on Twi, Luo, Yoruba. This is the best-justified QE reward for African targets.
- **ojo2025afrobench** (Findings ACL 2025; arXiv 2311.07978). 64 languages, 15 tasks, 22 datasets; prompted LLMs trail fine-tuned baselines and the gap to English is large, smaller for GPT-4o/Gemini-1.5 Pro (S, abstract summary). Early version found GPT-4 very poor on MT.
- **adelani2025irokobench** (NAACL 2025; arXiv 2406.03368). AfriXNLI, AfriMGSM, AfriMMLU over 17 languages; best open model (Gemma 2 27B) reaches about 63% of GPT-4o (S, latest version). Not MT, but cite for the "LLMs weak on African languages" framing.
- **goyal2022flores101** (TACL 2022; arXiv 2106.03193). FLORES-101: many-to-many benchmark, 101 languages, including many African ones.
- **nllb2022nllb** (arXiv 2207.04672, 2022-07-11) and **nllb2024nature** (Nature 630:841-846, 5 June 2024). NLLB-200 and FLORES-200; Nature abstract: "average of 44% improvement ... as measured by BLEU"; main text: "+12.5 chrF++" gain from mined data and back-translation for very low-resource languages into English (fetched PMC). Source for chrF++ as the reporting metric via model cards.
- **popovic2017chrfpp** (WMT 2017). chrF++ definition, needed wherever we report chrF++.
- **oldi2025floresplus** (HF dataset card v4.6). FLORES+: 230 language varieties, 997 dev and 1,012 devtest sentences each, CC BY-SA 4.0, maintained by OLDI; card says it "should not be used as training data". Check our training-data decontamination against it.
- **burchell2024oldi** (WMT 2024) and **dale2025oldi** (WMT 2025). OLDI shared-task findings: community extension of FLORES+ and the seed dataset (S for content).
- **deutsch2025wmt24pp** (arXiv 2502.12404). 55 languages, paragraph-level segments, 998 segments from 171 documents (fetched HTML); LLMs best in all 55 languages by automatic metrics. African coverage: see caveat in the novelty section.
- **caswell2025smol** (arXiv 2502.12301). SMOLSENT (863 sentences) + SMOLDOC (584 documents) professionally translated for 115-124 under-resourced languages; reports chrF gains when prompting or fine-tuning LLMs with it.
- **omt2025bouquet** (arXiv 2502.04314). Paragraph-level MT benchmark, 8 source languages translated into English and 266 others; test split 854 sentences / 198 paragraphs (S). TranslatePsy-AfriSLM evaluates on it.
- **omt2026omnilingual** (arXiv 2603.16309). 1,600-language MT; 1B-8B models match or exceed a 70B baseline (abstract). Body (S) says GRPO with chrF++/BLEU was tried and curation mattered.
- **bapna2022thousand** (arXiv 2205.03983). Google's recipe for 1000+ languages: semi-supervised LID and filtering, monolingual+parallel models; discusses metric limits and error modes.
- **kreutzer2022quality** (TACL 2022; arXiv 2103.12028). Audit of web-crawled corpora (CCAligned, ParaCrawl, WikiMatrix, OSCAR, mC4; also JW300 language codes); at least 15 corpora had no usable text and many had <50% acceptable sentences (S). Use for data-quality arguments.
- **agic2019jw300** (ACL 2019). JW300: >300 languages, about 100k pairs per pair, jw.org source (S). Basis of the "JW-heavy / Bible-domain" caveat.
- **adelani2021menyo** (MT Summit 2021; arXiv 2103.08647). MENYO-20k; shows domain and diacritic effects on Yoruba MT; noisy-diacritic web data hurts (S). Evidence that JW300-trained models transfer poorly across domains.
- **akerman2023ebible** (arXiv 2304.09919). eBible corpus: 1,009 Bible translations in 833 languages (S). Evidence on how much open parallel data is religious.
- **google2024translate110** (blog, 27 June 2024). 110 new Google Translate languages, about a quarter African, "largest expansion of African languages to date" (S via search). Cite for the commercial-coverage baseline; Google gives no per-language quality data there.
- **lacuna2021awards** (website/news). Lacuna Fund funded African NLP datasets including "Masakhane MT: Decolonizing Scientific Writing for Africa" (HapaKenya, 29 April 2021, fetched). Funding context only.

### Ugandan / Sunbird
- **akera2022salt** (AfricaNLP @ ICLR 2022). SALT corpus for 5 Ugandan languages (Luganda, Runyankole, Acholi, Lugbara, Ateso); mean BLEU 26.2 to English, 19.9 from English (abstract).
- **nsumba2026salt31** (AfricaNLP 2026). SALT-31: 31 Ugandan languages, scenario-based mini-dialogues, supports paragraph-level eval (S). A rare discourse-level African MT benchmark.
- **akera2025sunflower** (arXiv 2510.07203). Qwen3 14B/32B; CPT on about 0.75B characters, LoRA SFT on 2,347 translation + 726 conversation examples, IRPO. Average eng->xx chrF: Sunflower-32B 0.357 vs Gemini 2.5 Pro 0.301 vs GPT-4o 0.235; NLLB-3.3B higher on Kinyarwanda (fetched HTML). Test items are single sentences, no document-level.

### African LMs and adaptation (for context; none are MT-RL)
- **tonja2024inkubalm** (arXiv 2408.17024). 0.4B model for 5 African languages plus English/French, trained on about 2.4B tokens (S, Lelapa blog). The closest "tiny African LM" precedent, not specialised for MT.
- **adebara2023serengeti** (Findings ACL 2023). 517 African languages/varieties, AfroNLU 20 datasets (S).
- **dossou2022afrolm** (SustaiNLP 2022). 23 African languages, self-active learning, corpus about 14x smaller than baselines (S).
- **alabi2022afroxlmr** (COLING 2022). The AfroXLMR paper: multilingual adaptive fine-tuning on 17 African languages plus 3 high-resource; AfroXLM-R is the encoder under AfriCOMET.
- **ogueji2021afriberta** (MRL 2021). AfriBERTa: <1 GB, 11 languages (S).
- **ogundepo2022afriteva** (DeepLo 2022). AfriTeVA: T5-family, 10 African languages (S).
- **oladipo2023afriteva2** (EMNLP 2023). Wura corpus and AfriTeVa V2 (no arXiv paper titled "AfriTeVa V2" exists; this EMNLP paper is the reference). Authors audited mC4 for 13 languages and pretrained a T5 model on cleaner data (S).
- **buzaaba2025lugha** (arXiv 2504.06536). Lugha-Llama: Llama-3.1-8B continued pretraining on WURA plus English educational text; the content of the English data, not its language, drives gains (S).
- **jacaranda2023uliza** (HF model card). UlizaLlama 7B: Llama 2 + 321,530,045 Swahili tokens CPT, Swahili/English instruction tuning, CC BY-NC-SA. No paper.
- **adebara2024cheetah** (ACL 2024). Cheetah: NLG for 517 African languages (S).
- **elmadany2024toucan** (arXiv 2407.04796). Toucan, fine-tuned from Cheetah-1.2B/3.7B, plus AfroLingu-MT benchmark and spBLEU-1K metric covering 614 African languages (abstract).
- **emezue2021mmtafrica** (WMT 2021). MMTAfrica: many-to-many for 6 African languages with BT&REC objective (abstract).
- **adebara2025sahara** (ACL 2025). Sahara benchmark; a few African languages perform reasonably, many remain marginalised (abstract page).
- **adelani2025thesis** (arXiv 2507.00297). PhD thesis on African NLP; content not read.
- **issaka2025allab** (arXiv 2510.05644). African Languages Lab: 40 languages, 19B tokens; fine-tuning gains of +23.69 chrF++ on average across 31 languages (abstract).
- **alhanai2025bridging** (AAAI 2025). Eight languages, about 1M human-translated words of benchmark, >400 fine-tuned models (abstract).

### Task benchmarks (for "if relevant")
- **adelani2021masakhaner**, **adelani2022masakhaner2** (20 languages), **adelani2023masakhanews** (16 languages (S)), **muhammad2023afrisenti** (>110k tweets, 14 languages (S)). Cite once as examples of the Masakhane-style community benchmark ecosystem; not needed for MT claims.

### Multilingual / Gemma family
- **ustun2024aya** (arXiv 2402.07827). Aya 101: 13B, 101 languages, more than half lower-resourced (S).
- **singh2024ayadataset** (ACL 2024). Aya Dataset: human-curated instructions in 65 languages; 513M-instance collection over 114 languages (S).
- **dang2024ayaexpanse** (arXiv 2412.04261). Aya Expanse 8B/32B, 23 languages (S).
- **salamanca2026tinyaya** (arXiv 2603.11510, 2026-03-12). Tiny Aya 3.35B, 70 languages, regional variants incl. one covering African languages (fetched abstract; variant details from HF card via search). Another small multilingual model; claims SOTA translation for its size.
- **gemma2025gemma3** (arXiv 2503.19786). Gemma 3 report; 1B-27B, at least 128K context for the larger models (abstract). 270M not covered.
- **google2025gemma270m** (blog). 270M = 170M embedding + 100M transformer parameters, 256k vocabulary (fetched). Our base model; the large embedding share matters for tokenisation of African scripts.
- **finkelstein2026translategemma** (arXiv 2601.09012). See novelty. SFT on synthetic + human parallel data, then RL with MetricX-QE and AutoMQM reward ensemble; WMT24++ 55 pairs, WMT25 human eval 10 pairs; smaller TranslateGemma often matches larger baselines.
- **shang2026milmmt** (arXiv 2602.11961). Gemma-3-based MiLMMT-46, 46 languages (abstract); a search summary says its FLORES+ 8-shot table includes a Gemma 270M that trails larger models (S).
- **song2025slmlrl** (arXiv 2503.24102). Small LMs for low-resource MT; Llama-3.2-3B distillation for Luxembourgish; no African language named (abstract).
- **kocmi2025wmt25** (WMT 2025). General MT findings, 30 language pairs (S); language list not checked.
- **kudugunta2023madlad** (arXiv 2309.04662). MADLAD-400: 419 languages, 3T tokens, 10.7B MT model over 450+ languages; "document-level" in the title refers to the corpus. AfriDoc-MT evaluates MADLAD-400 3B.
- **nllb600m2022card**, **nllb13b2022card**. 512-token training cap; "not intended to be used for document translation"; metrics BLEU, spBLEU, chrF++; CC-BY-NC.

### Document-level / recent African MT with LLMs
- **alabi2025afridocmt** (EMNLP 2025; arXiv 2501.06374). The only dedicated African document-level MT corpus I found: EN + Amharic, Hausa, Swahili, Yoruba, Zulu; 334 health + 271 IT news documents; evaluated Toucan 1.2B, NLLB 1.3B/3.3B, MADLAD 3B, Aya-101, Gemma2-9B, Llama3.1-8B, LLaMAX3-8B, GPT-3.5, GPT-4o; fine-tuned NLLB-1.3B and two 8B LLMs. GPT-4o d-chrF 63.5 (health) and 62.8 (tech). No RL. Directly relevant; cite as the nearest document-level baseline and consider evaluating on it.
- **yu2026afriquellm** (arXiv 2601.06395; ACL 2026 main). AfriqueLLM: 20 languages, 26B CPT tokens, five base models; document-level AfriDoc-MT 3-shot d-chrF 60.2 (AfriqueGemma-12B), 59.2 (AfriqueQwen-14B), 58.5 (AfriqueLlama-8B) EN->African; 16K sequence length; smallest model is 4B (fetched HTML).
- **abdulmumin2026afriscience** (arXiv 2605.29741). Six languages (Amharic, Hausa, Luganda, Northern Sotho, Yoruba, isiZulu), 11 scientific domains; sentence-level COMET GPT-5.4 68.3 and Gemini-3.1-Flash-Lite 68.0, document-level 48.3 for both; best open: TranslateGemma-12B 44.0 doc-level (1-shot), fine-tuned NLLB-1.3B 67.3 sentence-level (abstract). Shows a large sentence-to-document quality drop.
- **adjovi2026hausafongbe** (arXiv 2606.22269). LLM EN->Hausa/Fongbe: human 4.0-4.5/5 Hausa vs 1.0-2.2/5 Fongbe; BERTScore collapses; recommends >= 2,500 sentences for stable rankings (abstract). Supports multi-metric evaluation.
- **gritta2026translatepsy** (arXiv 2608.18655). Fetched HTML: Qwen3.5-based 0.8B/2B/4B, full SFT 1 epoch, parallel data about 427M raw pairs capped at 5M per pair, QE filtering removes 96% of tokens (44.93B to 1.76B) with SSA-COMET 0.530 vs 0.528, final synthetic mix 32.37B tokens. SSA-COMET Flores-200/BOUQuET/Smol: 0.8B 0.5944/0.6223/0.4973; NLLB-3.3B 0.5944/0.6178/0.4909; TranslateGemma-27B 0.5455/0.5677/0.4608. Language list names 20 languages though text says 19. Strongest competitor; any 270M result should be compared to their 0.8B row.
- **moslem2026afrinllb** (AfricaNLP 2026; arXiv 2602.09373). Layer-pruned and quantised NLLB-600M with distillation, 15 pairs (S).

### RL / QE reward neighbours
- **attia2026roundtrip**, **garciaestrada2026reffree**, **su2026semanticrl**, **han2026milmmtrl**, **feng2025mtr1zero**, **liu2026walar**, **hu2026rlunseen**, **he2024qereward**, **mosquera2025wayuunaiki**: see novelty section; one-liners: round-trip GRPO on NLLB with chrF++/BLEU; reference-free GRPO on NLLB with LaBSE + COMET-Kiwi; Tibetan semantic-reward GRPO (Qwen3-4B (S)); two-QE-model RL reward gated by language ID (46 languages); R1-Zero-style MT with BLEU + COMET-Kiwi (S); WALAR reward-hacking mitigation via word and language alignment (101 languages); chrF-reward GRPO with in-context dictionaries for unseen languages (S); QE as a reward model with reward-hacking caveats (S); dictionary-guided FT + RL for Spanish-Wayuunaiki (title only).
- **wang2026doc2frc** (arXiv 2609.12674, WMT 2026). Fixed-range chunking for length-consistent document MT; non-African; abstract not read. Possibly relevant to how we chunk 2,048-token inputs.

### Extra item requested by the other agent
- **lian2025lingbenchpp** (arXiv 2507.16809). Exists; but is a Linguistics-Olympiad reasoning benchmark over 90+ low-resource languages, not MT and not African-focused (abstract fetched). Low relevance; likely a mis-cited id in the internal notes.

---

## 3. Unverified / not found

- **AfriqueLLM, TranslatePsy-AfriSLM, TranslateGemma, AfriScience-MT**: all verified (ids confirmed). The ids requested by the other agent for these four were correct. Note AfriqueLLM's authors are Hao Yu et al. with D. I. Adelani last; the abstract page does not name McGill-NLP (affiliation not checked).
- **OLDI Seed** has no standalone paper that I could find. The seed data originates as the NLLB Seed Dataset (cite nllb2022nllb); OLDI shared-task findings cited instead. Third-party OLDI-seed language papers exist (e.g. French and Kyrgyz seeds, WMT25) but were not verified individually.
- **AfriTeVa V2**: no arXiv paper with that title found; the EMNLP 2023 Wura paper is the reference.
- **UlizaLlama**: no paper; model card only.
- **Lugha-Llama, AfroXLMR, Aya 101 publication venues**: Lugha-Llama arXiv only; Aya 101 ACL 2024 venue not confirmed here.
- **"Gemini Urhobo" low-resource translation work**: not found (searches returned only general Gemini tech report and adjacent LRL papers). Left out.
- **NLLB-200 3.3B model card**: not fetched; only 600M (search result) and 1.3B (fetched) are cited.
- **MasakhaNEWS final venue**; **MADLAD-400 NeurIPS venue**; **WMT24++ Findings of ACL 2025 and SMOL WMT 2025 acceptance** (search summaries only); **BOUQuET EMNLP 2025** (search summary only).
- **WMT24++ full African language list**: the fetched summary showed only sw_KE, sw_TZ, zu_ZA (+ ar_EG); not verified as complete.
- **Google Translate African coverage numbers**: only the "about a quarter of 110" statement, from search summaries of the Google blog; official page body not fetched. Per-language quality data not found.
- **An AfriCOMET/SSA-COMET reward used for GRPO**: not found anywhere in the literature.
- **Document-level RL for MT** (arXiv 2010.04314, "Dynamic Context Selection for Document-level NMT via RL"): seen in search results only, not verified, not in bib.
- **Sunbird GRPO-tuned Sunflower-14B checkpoint**: claimed on a model page (search summary), absent from the paper; unverified.
- **Not searched at all**: AfriMed (not MT), Masakhane MT GitHub repo as a citable artefact, JW300 domain-bias evaluations specific to African MT beyond MENYO-20k, WMT 2025 low-resource African tracks (Maasai search returned nothing).
