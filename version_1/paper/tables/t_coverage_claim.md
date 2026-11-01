### Table A-M4. African-language coverage of other systems (evidence; reports/data/language_coverage_claim_check.md)

| system | params | open | African languages supported (claimed/trained) | evaluated en->X | verified |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SabiYarn Gemma-MT (ours) | 270M (Gemma-3-270m based) | owner decision (not checked he | 77 (gemma_mt/languages.py SUPPORTED_RL; 44 are in FLORES-200, 33 are not; 54 nam | 44 on FLORES+ devtest; 64 of 77 have some public en->xx benchmark in l | yes |
| NLLB-200 (54.5B MoE, 3.3B, 1.3B, 600M, distilled 600M/1.3B) | 0.6B / 1.3B / 3.3B dense, 54.5B MoE | yes (weights released; MoE via | 55 per the paper (51 sub-Saharan/Berber/Malagasy FLORES-200 languages + aeb ary  | all 55 (FLORES-200 devtest, chrF++/spBLEU per direction) | yes |
| MADLAD-400 MT (3B / 7.2B / 10.7B) and 8B LM | 3B, 7.2B, 10.7B (MT), 8B (LM) | yes (checkpoints released per  | about 85 African-language entries among the 419 languages (my tally of the Appen | FLORES-200 and WMT only as aggregates by resource group (Table 5); per | partial |
| Aya-101 | 13B (mT5-based) | yes | 13 (afr amh hau ibo mlg nya sna som sot swa xho yor zul); Wolof only in the Aya  | AfroBench MT pooled chrF (Flores+MAFAND+NTREX), per-language only as h | yes |
| Aya Expanse (8B/32B) and Aya-23 (8B/35B) | 8B / 32B (Expanse); 8B / 35B (Aya-23) | yes (research license, not che | 0 sub-Saharan (23 languages; Arabic is the only African-spoken language) | n/a | partial |
| Tiny Aya (Base/Global/Earth/Fire/Water) | 3.35B | yes (open weights per Cohere;  | 10 (amh hau ibo mlg sna swh wol xho yor zul) of 70 | 10 (FLORES en->xx chrF over 66 languages incl. these 10; WMT24++ 55) | yes |
| Google Translate (cloud MT) | closed | no | 2022: +10 African (Bambara Ewe Krio Lingala Luganda Oromo Sepedi Tigrinya Tsonga | no public per-language en->xx scores | partial |
| Gemini 2.5 Flash (API) | closed | no | claims 100+ languages; African list not published | FLORES-200 all African variants incl. Fon, Nuer, Tamasheq, Kabiye...,  | yes |
| GPT-4o | closed | no | no per-language claim | AfroBench MT pooled chrF; per-language only heat-maps | partial |
| AfriqueLLM (Llama-8B, Gemma-4B/12B, Qwen-4B/8B/14B, Qwen3.5- | 4B-14B | yes | 20 African (afr swh ary som amh arz hau kin zul ibo plt xho sna yor nya sot tir  | 24 FLORES-200 languages en->xx and xx->en (SSA-COMET Tables 20/21, chr | yes |
| TranslatePsy-AfriSLM (0.8B, 2B, 4B) | 0.8B / 2B / 4B (Qwen3.5 base) | yes (Apache-2.0) | 19 African (afr amh hau ibo kin lin lug mlg nya orm sna som sot swh tsn wol xho  | 19 (FLORES-200 devtest, BOUQuET, Smol; en->xx and xx->en, macro only); | yes |
| Apertus (8B, 70B) | 8B / 70B | yes (fully open) | pretraining 1,811 languages (FineWeb-2; African share not enumerated); post-trai | no African en->xx MT evaluation in the report (translation eval is Rom | yes |
| Hunyuan-MT-7B / Chimera-7B | 7B | yes | 0 (33 languages + 5 Chinese minority languages; no African term in the report te | 0 | yes |
| AfriNLLB (pruned NLLB-600M, 548M) | 548M-600M | yes (HF AfriNLP; model page ga | 10 African: afr amh arz hau som swh wol yor zul + lin (via French) | 9 en<->xx FLORES-200 devtest pairs + fr<->lin, fr<->wol (BLEU, chrF++, | yes |
| InkubaLM-0.4B | 0.4B | yes | 5 (Swahili, Yoruba, isiXhosa, Hausa, isiZulu) | AfroBench-style tasks incl. MT (not extracted) | partial |
| Lugha-Llama-8B-wura | 8B | yes | 16 African (WURA) + en fr ar pt | AfroBench/IrokoBench; csv has 21 languages from AfriqueLLM Table 22 | yes |
| SERENGETI (E250, E110, ...) | encoders (mBERT-size) | yes | 517 African languages and varieties (+ top-10 world languages) | NLU only (AfroNLU: 20 datasets, 28 languages); not a translator | partial |
| AfroXLM-R (large 17L, 76L, 114L) | ~560M encoder | yes | 17 / 76 (card prints 75) / 114 (110 African per SSA-COMET paper) | NLU; used as the AfriCOMET/SSA-COMET encoder | yes |
| AfroLM | encoder | yes | 23 African languages | NER/classification/sentiment | partial |
| AfriBERTa | encoder | yes | 11 African languages | NER/classification | partial |
| mT5 / ByT5 / mT0 | up to 13B | yes | 13 African in mT5's 101 languages (afr amh hau ibo mlg nya sna som sot swa xho y | n/a (base models) | partial |
| BLOOM / BLOOMZ | up to 176B | yes (RAIL) | 21 Niger-Congo languages listed in the card (each 0.00002%-0.02% of data; Swahil | xP3 / FLORES few-shot (not extracted) | yes |
| mBART-50 many-to-many | 0.6B | yes | 3 (af_ZA sw_KE xh_ZA) | FLORES-101 style evaluations (not extracted) | yes |
| Qwen3 (0.6B-235B) | 0.6B-235B | yes | 2 sub-Saharan (Swahili, Kabuverdianu) + 8 Arabic varieties of 119 | Belebele (kea, swh, afr in Table 36) only; no African MT | partial |
| Qwen3.5 (0.8B-122B-A10B) | 0.8B-122B | yes | card claims 201 languages and dialects; no list | TranslatePsy: SSA-COMET 0.30-0.55 on 19 languages | partial |
| Llama 3.1 | 8B / 70B / 405B | yes | 0 officially (8 languages: en de fr it pt hi es th) | AfroBench / IrokoBench (non-official) | yes |
| TranslateGemma (4B, 12B, 27B) | 4B / 12B / 27B | yes (Gemma terms) | 55 evaluated languages claimed on the card; the SFT data lists in the report app | 2 African languages (Swahili Kenya/Tanzania, Zulu) on WMT24++ with Met | yes |
| Gemma 3 (1B, 4B, 12B, 27B; the 270M card was not opened) | 1B-27B | yes | claims 'over 140 languages', no list; African subset not published | AfriqueLLM Tables 20/22: Gemma3-4B/12B SSA-COMET 34.0/41.4 (24 langs) | partial |
| SMOL + GATITOS (data, Google) | dataset | yes (CC-BY-4.0) | 115 languages; 66 'Africa' rows in the paper's language table (includes Arabic d | n/a (human translated test/train data, SmolSent/SmolDoc/GATITOS) | yes |
| Omnilingual ASR / MMS (Meta) | speech models | yes (Apache-2.0 for Omnilingua | ASR 1,600+ languages (not MT) | n/a | partial |
| Omnilingual MT (OMT-LLaMA 1B/3B/8B, OMT-NLLB 3B) | 1B-8B | unverified (leaderboard/eval s | 1,600+ languages as source, ~1,200 as target; African count not extracted | Bible partition 1,560 languages (xCOMET, chrF++); BOUQuET; African exa | partial |
| SeamlessM4T v2 large | 2.3B | yes (CC-BY-NC) | 15 African text languages (afr amh ary arz fuv gaz ibo lug luo nya sna som swh y | FLORES-style evaluation in the Seamless paper (not extracted) | partial |
| Toucan / AfroLingu-MT (UBC) | 1.2B / 3.7B | yes (HF UBC-NLP/toucan-*) | 46 languages = 43 African + ar en fr (secondary source) | AfroLingu-MT, 156 pairs | partial |
| Sunflower-14B (Sunbird) | 14B (Qwen3-14B base) | yes (HF Sunbird/Sunflower-14B  | 31 Ugandan languages (+67 African planned per Sunbird) | SALT-31 (31 languages, 5-sentence dialogues) | partial |
| Lelapa Vulavula / Lelapa-X-GLOT | closed API | no | 11-13 (Afrikaans, Sesotho, Sepedi, Setswana, isiXhosa, isiZulu, Swahili, Hausa,  | none public | partial |
