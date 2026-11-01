# Candidate translators for the long-document benchmark (English -> 77 African languages)

Written 2026-10-20. Companion files: `gemma_mt/gemma_mt/eval/longdoc_systems.py` (registry, 29 systems), `gemma_mt/tests/test_longdoc_systems.py`, `reports/longdoc_benchmark_plan_2026-10-20_parts/system_cards.json` (evidence: card URL, retrieval date, licence string as written, language-list source, pinned revision; copy on the rental in `longdoc_bench/sysinfo/`).
Every licence and language list below was read from the public model card, paper or config on 2026-10-20 (retrieval date for all URLs). Quality numbers are **published by the model authors, not measured by us**. "Explicit" = the language is named in the system's own language list (card / paper / tokenizer); naming a language is not a quality claim. "Proxy" = a close variety is supported, shown as `code(p)` and never counted as support.

## TL;DR

* **68 of the 77 languages are named by at least one open-weight system; 67 by at least one system with a permissive licence** (Apache-2.0 / CC-BY-4.0). **No open system names these 8**: urh (Urhobo), bin (Edo), oke (Okpe), bbj (Ghomala), yao (Yao), sid (Sidamo), ish (Esan), dua (Duala); wes (Cameroon Pidgin) only has the Nigerian-Pidgin proxy; **ndc (Ndau) is named only by TranslateGemma (Gemma Terms)**.
* Broadest **permissive** options: OPUS-MT tc-bible-big eng->mul (59 of ours, Apache-2.0, 240M params, Bible-heavy), AfriqueQwen3.5-9B-50Langs-Instruct (41, CC-BY-4.0), MADLAD-400 (44, Apache-2.0), Sunflower-Qwen3.8-27B / 3.5-9B (31, Apache-2.0; best self-reported quality on the languages it names). Broadest overall: NLLB-200 (44, **CC-BY-NC-4.0**) and TranslateGemma (47 named in its SFT data, **Gemma Terms**; only sw and zu actually evaluated by Google).
* **Published FLORES-200 en->xx chrF (MADLAD paper, Table 17) says MADLAD is far behind NLLB-54B on several of our languages** (e.g. rw 9.6 vs 53.8, ln 18.8 vs 50.8, wo 12.6 vs 31.8, yo 20.3 vs 27.4, lg 30.3 vs 44.7, om 29.3 vs 43.3, zu 48.6 vs 58.8; MADLAD-7.2B vs NLLB-54B-MoE, published). The tiny Apache-2.0 OPUS-MT tc-bible-big Bantu model reports FLORES-200 devtest chrF above MADLAD-7.2B on zu, tso, nso, kin, lin, tsn, ssw, bem (card numbers; same metric family but not guaranteed same implementation/split).
* **Gemma 4 (2026) is Apache-2.0** (unlike Gemma 1-3 / TranslateGemma) but publishes **no language list** and no African-language evaluation; it is a candidate only if the bake-off shows it works.
* **vLLM 0.31 cannot serve NLLB / MADLAD / Marian / Seamless / Aya-101 / mT0** (encoder-decoder archs not in its registry); the Gemma / Qwen / Llama / Bloom families are registered. `ctranslate2` is **not installed** on the rental (no GPU smoke test done); HF `generate` works for everything.
* Meta's Omnilingual MT (OMT-LLaMA / OMT-NLLB, arXiv 2603.16309, 2026-03-17, "1,600+ languages") has **no public weights that I could find** (Hub search of `facebook`, GitHub `facebookresearch`, paper page: only eval data/leaderboard are released).

## (a) Systems

| system | params (B) | licence (class) | what it says about outputs | backends | VRAM fp16 / int8 (GB, weights only) | explicit / proxy of 77 |
|---|---|---|---|---|---|---|
| `nllb-200-distilled-600m` | 0.615 | CC-BY-NC-4.0 (noncommercial) | CC-BY-NC; card: research model, not for production or document translation; outputs not addressed | hf, ctranslate2 | 1.2 / 0.6 | 44 / 6 |
| `nllb-200-distilled-1.3b` | 1.37 | CC-BY-NC-4.0 (noncommercial) | CC-BY-NC; card: research model, not for production or document translation; outputs not addressed | hf, ctranslate2 | 2.7 / 1.4 | 44 / 6 |
| `nllb-200-1.3b` | 1.37 | CC-BY-NC-4.0 (noncommercial) | CC-BY-NC; card: research model, not for production or document translation; outputs not addressed | hf, ctranslate2 | 2.7 / 1.4 | 44 / 6 |
| `nllb-200-3.3b` | 3.34 | CC-BY-NC-4.0 (noncommercial) | CC-BY-NC; card: research model, not for production or document translation; outputs not addressed | hf, ctranslate2 | 6.7 / 3.3 | 44 / 6 |
| `madlad400-3b-mt` | 2.94 | Apache-2.0 (permissive) | Apache-2.0; no output restriction | hf, ctranslate2 | 5.9 / 2.9 | 44 / 7 |
| `madlad400-7b-mt` | 8.3 | Apache-2.0 (permissive) | Apache-2.0; no output restriction | hf, ctranslate2 | 16.6 / 8.3 | 44 / 7 |
| `madlad400-7b-mt-bt` | 8.3 | Apache-2.0 (permissive) | Apache-2.0; no output restriction | hf, ctranslate2 | 16.6 / 8.3 | 44 / 7 |
| `madlad400-10b-mt` | 10.71 | Apache-2.0 (permissive) | Apache-2.0; no output restriction | hf, ctranslate2 | 21.4 / 10.7 | 44 / 7 |
| `translategemma-4b-it` | 4.97 | gemma (gemma_terms) | Gemma ToU: Google claims no rights in outputs, but training a model on them = 'Model Derivative' (s1.1(e)) carrying the terms | hf, vllm | 9.9 / 5 | 47 / 8 |
| `translategemma-12b-it` | 13.19 | gemma (gemma_terms) | Gemma ToU: Google claims no rights in outputs, but training a model on them = 'Model Derivative' (s1.1(e)) carrying the terms | hf, vllm | 26.4 / 13.2 | 47 / 8 |
| `translategemma-27b-it` | 28.84 | gemma (gemma_terms) | Gemma ToU: Google claims no rights in outputs, but training a model on them = 'Model Derivative' (s1.1(e)) carrying the terms | hf, vllm | 57.7 / 28.8 | 47 / 8 |
| `gemma-3-27b-it` | 27.43 | gemma (gemma_terms) | Gemma ToU: Google claims no rights in outputs, but training a model on them = 'Model Derivative' (s1.1(e)) carrying the terms | hf, vllm | 54.9 / 27.4 | 2 / 0 |
| `gemma-4-31b-it` | 31.27 | Apache-2.0 (permissive) | Apache-2.0; no output restriction | hf, vllm | 62.5 / 31.3 | 0 / 0 |
| `qwen3-32b` | 32.76 | Apache-2.0 (permissive) | Apache-2.0; no output restriction | hf, vllm | 65.5 / 32.8 | 3 / 0 |
| `aya-101` | 12.92 | Apache-2.0 (permissive) | Apache-2.0; no output restriction | hf | 25.8 / 12.9 | 15 / 4 |
| `tiny-aya-earth` | 3.35 | CC-BY-NC-4.0 + Cohere Labs Acceptable Use Policy (noncommercial) | CC-BY-NC + Cohere AUP; outputs not addressed | hf | 6.7 / 3.4 | 10 / 3 |
| `tiny-aya-global` | 3.35 | CC-BY-NC-4.0 + Cohere Labs Acceptable Use Policy (noncommercial) | CC-BY-NC + Cohere AUP; outputs not addressed | hf | 6.7 / 3.4 | 10 / 3 |
| `seamless-m4t-v2-large` | 2.31 | CC-BY-NC-4.0 (noncommercial) | CC-BY-NC; outputs not addressed | hf | 4.6 / 2.3 | 12 / 3 |
| `mt0-xxl-mt` | 13.95 | Apache-2.0 (permissive) | Apache-2.0; no output restriction | hf | 27.9 / 13.9 | 13 / 4 |
| `bloomz-7b1-mt` | 7.07 | bigscience-bloom-rail-1.0 (other) | BLOOM RAIL use restrictions (text not re-read) | hf, vllm | 14.1 / 7.1 | 21 / 5 |
| `opus-mt-en-x` | 0.077 | Apache-2.0 (permissive) | Apache-2.0; no output restriction (OPUS/Bible-heavy data) | hf, ctranslate2 | 0.2 / 0.1 | 35 / 3 |
| `opus-mt-tc-bible-big-eng-mul` | 0.241 | Apache-2.0 (permissive) | Apache-2.0; no output restriction (OPUS/Bible-heavy data) | hf, ctranslate2 | 0.5 / 0.2 | 59 / 5 |
| `opus-mt-tc-bible-big-eng-bnt` | 0.24 | Apache-2.0 (permissive) | Apache-2.0; no output restriction (OPUS/Bible-heavy data) | hf, ctranslate2 | 0.5 / 0.2 | 31 / 3 |
| `opus-mt-tc-bible-big-eng-afa` | 0.24 | Apache-2.0 (permissive) | Apache-2.0; no output restriction (OPUS/Bible-heavy data) | hf, ctranslate2 | 0.5 / 0.2 | 8 / 0 |
| `sunflower-qwen3.8-27b` | 27.36 | Apache-2.0 (permissive) | Apache-2.0; no output restriction (training data undocumented) | hf, vllm | 54.7 / 27.4 | 31 / 6 |
| `sunflower-qwen3.5-9b` | 9.41 | Apache-2.0 (permissive) | Apache-2.0; no output restriction (training data undocumented) | hf, vllm | 18.8 / 9.4 | 31 / 6 |
| `afriqueqwen3.5-9b-50langs-instruct` | 8.95 | CC-BY-4.0 (permissive) | CC-BY-4.0 (attribution); base Qwen Apache-2.0; CPT used GPT-4.1 synthetic translations | hf, vllm | 17.9 / 8.9 | 41 / 6 |
| `llamax3-8b-alpaca` | 8.03 | Llama-3 Community License (Hub tag: MIT) (llama) | Llama 3 licence cl. 1.b.v: outputs may not be used to improve another LLM | hf, vllm | 16.1 / 8 | 20 / 3 |
| `n-atlas-llm` | 8.03 | N-ATLaS Open-Source Research and Innovation License (custom, 1000-user cap) (other) | custom: <=1000 end-users without a commercial licence | hf, vllm | 16.1 / 8 | 3 / 0 |

Notes on the columns.
* **Backends** = known to work for the architecture, verified only as follows (2026-10-20, CPU-only import checks, nothing loaded to a GPU): `hf` = class present in transformers 5.19.0 (venv_gemma; 5.17.0 in venv_vllm); `vllm` = architecture present in `vllm 0.31.0` `ModelRegistry` (387 archs) and for encoder-decoder families absent: T5/MT5/M2M100/Marian/Seamless/Bart are **not** registered, consistent with vLLM having dropped encoder-decoder support (T5 PR #11901 closed unmerged 2025-10-02; Whisper native, BART/Florence-2 only via an external plugin); `ctranslate2` = converter listed in CTranslate2's `docs/guides/transformers.md` (master): M2M100/NLLB, T5, MarianMT, Gemma 2/3/4 (text only), Qwen 2.5/3, Llama, BLOOM; **not** SeamlessM4T, **not** Qwen3.5, mT5 only as UMT5. Community CT2 conversions of MADLAD-3B/10B exist on the Hub. Whether CT2 has an sm_120 (Blackwell) build is unconfirmed: smoke-test first.
* **VRAM** is weights only (2 B/param fp16/bf16, 1 B int8) from the Hub safetensors parameter counts; add 10-25% for activations/KV. TranslateGemma sizes include the SigLIP vision tower (about 0.4B). Seamless 2.3B is the whole multitask checkpoint.
* **Gated repos** (free but need the form accepted by the account whose token is used): all Gemma-3/TranslateGemma, Tiny Aya, Sunflower, N-ATLaS. NLLB, MADLAD, Gemma 4, Aya-101, mT0, Opus-MT, Seamless, AfriqueQwen, Qwen3 are open.
* The `marian` prompt style covers two cases: per-pair models (`hf_id_template="Helsinki-NLP/opus-mt-en-{code}"`) and the multi-target tc-bible models (`hf_id` set, `template=">>{tgt_code}<< {text}"`, `tgt_code` = ISO-639-3 label).

Licence paraphrases (sources in `system_cards.json`):
* **NLLB-200**: CC-BY-NC-4.0 (https://creativecommons.org/licenses/by-nc/4.0/). The card calls it a research model, not for production deployment and not intended for document translation (trained on inputs up to 512 tokens). Nothing is said about generated translations; whether the NC clause reaches outputs is untested, so releasing NLLB outputs as a public pseudo-reference is a risk that needs a legal decision.
* **MADLAD-400**: Apache-2.0 on the cards; no output restriction. Card names the research community as primary users.
* **Gemma 1-3 / TranslateGemma**: Gemma Terms of Use, page marked last modified 2026-04-01 (https://ai.google.dev/gemma/terms): Google claims no rights in Outputs (s3.3); a model trained on synthetic Outputs of Gemma is a "Model Derivative" (s1.1(e)) and inherits the terms and use restrictions; Outputs themselves are not Model Derivatives; the s3.1 duties (pass the use restrictions on, give recipients the terms) are not clearly triggered by distributing a pure-output dataset, but anyone who then trains on it may fall under the Model Derivative definition: ambiguous, needs legal review. TranslateGemma itself was SFT'd on Gemini-2.5-Flash synthetic translations (report, section 2).
* **Gemma 4**: Apache-2.0 (card YAML + licence link to the Gemma 4 "Apache License 2.0" page); no Model-Derivative clause.
* **Llama 3 (LLaMAX3, N-ATLaS)**: Llama 3 Community License (2024-04-18; read from github.com/meta-llama/llama3): clause 1.b.v forbids using the Llama Materials *or any output or results* to improve another LLM, plus "Built with Meta Llama 3" attribution. LLaMAX's Hub tag says MIT, which does not remove this. N-ATLaS adds a custom 1,000-end-user cap. Both unsuitable for a permissive public benchmark.
* **Cohere (Tiny Aya)**: CC-BY-NC-4.0 plus Cohere Labs Acceptable Use Policy; non-commercial acknowledgement at download; outputs not addressed.
* **Seamless v2**: CC-BY-NC-4.0, outputs not addressed. **mT0 / Aya-101 / Qwen3 / Sunflower / OPUS-MT**: Apache-2.0 on the cards. **AfriqueQwen**: CC-BY-4.0 on the card (attribution; base Apache-2.0). **BLOOMZ**: BigScience BLOOM RAIL 1.0 (licence text not re-read).

## (b) Coverage matrix (77 languages)

Cell = the system's own code; `code(p)` = close-variety proxy (not support); `-` = not named. Identical columns are collapsed: **nllb-200-3.3b** stands for all four NLLB checkpoints, **madlad400-7b-mt-bt** for all four MADLAD checkpoints (explicit lists are identical; quality differs), **translategemma-27b-it** for the 4B/12B/27B, **sunflower-qwen3.8-27b** for the 3.5-9B, **tiny-aya-earth** for tiny-aya-global. `google` / `microsoft` columns are the closed-system reference (section g), not candidates. FLORES-200 codes are in `FLORES200_CODE` (44 of the 77 are in FLORES-200; 33 are not).

| iso | language | nllb-200-3.3b | madlad400-7b-mt-bt | translategemma-27b-it | aya-101 | tiny-aya-earth | seamless-m4t-v2-large | mt0-xxl-mt | bloomz-7b1-mt | opus-mt-en-x | opus-mt-tc-bible-big-eng-mul | opus-mt-tc-bible-big-eng-bnt | opus-mt-tc-bible-big-eng-afa | sunflower-qwen3.8-27b | afriqueqwen3.5-9b-50langs-instruct | llamax3-8b-alpaca | gemma-3-27b-it | qwen3-32b | n-atlas-llm | google | microsoft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yor | Yoruba | yor_Latn | yo | yo | yor | yo | yor | yo | yo | - | yor | - | - | yor | yor_Latn | yo | - | - | yor | yo | yo |
| hau | Hausa | hau_Latn | ha | ha | hau | ha | - | ha | - | ha | hau | - | hau | hau | hau_Latn | ha | - | - | hau | ha | ha |
| ibo | Igbo | ibo_Latn | ig | ig | ibo | ig | ibo | ig | ig | ig | ibo | - | - | ibo | ibo_Latn | ig | - | - | ibo | ig | ig |
| pcm | Pidgin | - | - | pcm | - | - | - | - | - | - | pcm | - | - | pcm | - | - | - | - | - | kri(p) | - |
| efi | Efik | - | ibb(p) | efi | - | - | - | - | - | efi | efi | - | - | - | - | - | - | - | - | - | - |
| twi | Twi | twi_Latn | ak | ak | twi | - | - | - | tw | tw | twi | - | - | aka(p) | twi_Latn | - | - | - | - | ak | - |
| ewe | Ewe | ewe_Latn | ee | ee | - | - | - | - | - | ee | ewe | - | - | ewe | ewe_Latn | - | - | - | - | ee | - |
| fon | Fon | fon_Latn | fon | fon | - | - | - | - | fon | - | fon | - | - | - | fon_Latn | - | - | - | - | - | - |
| wol | Wolof | wol_Latn | wo | wo | - | wo | - | - | wo | - | wol | - | - | wol | wol_Latn | wo | - | - | - | - | - |
| swh | Swahili | swh_Latn | sw | sw | swa | sw | swh | sw | sw | sw | swh | swh | - | swa | swh_Latn | sw | sw | swh | - | sw | sw |
| kin | Kinyarwanda | kin_Latn | rw | rw | - | - | - | - | rw | rw | kin | kin | - | kin | kin_Latn | - | - | - | - | rw | rw |
| som | Somali | som_Latn | so | so | som | - | som | so | - | - | som | - | som | som | som_Latn | so | - | - | - | so | so |
| orm | Oromo | gaz_Latn | om | om | - | - | gaz | - | - | om | orm | - | orm | orm | gaz_Latn | om | - | - | - | om | - |
| amh | Amharic | amh_Ethi | am | am | amh | am | amh | am | - | - | amh | - | amh | amh | amh_Ethi | am | - | - | - | am | am |
| tir | Tigrinya | tir_Ethi | ti | ti | - | - | - | - | - | ti | tir | - | tir | - | tir_Ethi | - | - | - | - | ti | ti |
| lug | Luganda | lug_Latn | lg | lg | - | - | lug | - | lg | lg | lug | lug | - | lug | lug_Latn | lg | - | - | - | lg | lug |
| lin | Lingala | lin_Latn | ln | ln | - | - | - | - | ln | ln | lin | lin | - | lin | lin_Latn | ln | - | - | - | ln | ln |
| zul | Zulu | zul_Latn | zu | zu | zul | zu | zul | zu | zu | - | zul | zul | - | zul | zul_Latn | zu | zu | - | - | zu | zu |
| xho | Xhosa | xho_Latn | xh | xh | xho | xh | - | xh | xh | xh | xho | xho | - | xho | xho_Latn | xh | - | - | - | xh | xh |
| afr | Afrikaans | afr_Latn | af | af | afr | - | afr | af | - | af | afr | - | - | afr | afr_Latn | af | - | afr | - | af | af |
| sna | Shona | sna_Latn | sn | sn | sna | sn | sna | sn | sn | sn | sna | sna | - | sna | sna_Latn | sn | - | - | - | sn | sn |
| nya | Chichewa | nya_Latn | ny | ny | nya | - | nya | ny | ny | ny | nya | nya | - | nya | nya_Latn | ny | - | - | - | ny | nya |
| sot | Sesotho | sot_Latn | st | st | sot | - | - | st | st | st | sot | sot | - | sot | sot_Latn | - | - | - | - | st | st |
| tsn | Setswana | tsn_Latn | tn | tn | - | - | - | - | tn | tn | tsn | tsn | - | tsn | tsn_Latn | - | - | - | - | tn | tn |
| mlg | Malagasy | plt_Latn | mg | mg | mlg | mg | - | mg | - | mg | mlg | - | - | mlg | plt_Latn | - | - | - | - | mg | mg |
| ibb | Ibibio | - | ibb | efi(p) | - | - | - | - | - | efi(p) | efi(p) | - | - | - | - | - | - | - | - | - | - |
| urh | Urhobo | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| bem | Bemba | bem_Latn | - | bem | - | - | - | - | - | bem | bem | bem | - | bem | bem_Latn | - | - | - | - | bem | - |
| tum | Tumbuka | tum_Latn | - | tum | - | - | - | - | tum | - | tum | tum | - | - | - | - | - | - | - | - | - |
| run | Kirundi | run_Latn | rn | rn | - | - | - | - | rn | rn | run | run | - | run | run_Latn | - | - | - | - | rn | run |
| ven | Venda | - | ve | ve | - | - | - | - | - | - | ven | ven | - | - | - | - | - | - | - | - | - |
| tso | Tsonga | tso_Latn | ts | ts | - | - | - | - | ts | ts | tso | tso | - | - | tso_Latn | - | - | - | - | ts | - |
| nso | Sepedi | nso_Latn | nso | nso | nso | - | - | - | nso | nso | nso | nso | - | - | nso_Latn | ns | - | - | - | nso | nso |
| ssw | Swati | ssw_Latn | ss | ss | - | - | - | - | - | ss | ssw | ssw | - | - | ssw_Latn | - | - | - | - | ss | - |
| nbl | South Ndebele | zul_Latn(p) | nr | nr | zul(p) | zu(p) | zul(p) | zu(p) | zu(p) | - | nbl | nbl | - | nbl | zul_Latn(p) | zu(p) | - | - | - | nr | - |
| sag | Sango | sag_Latn | sg | sg | - | - | - | - | - | sg | sag | - | - | - | sag_Latn | - | - | - | - | sg | - |
| tiv | Tiv | - | tiv | tiv | - | - | - | - | - | tiv | - | - | - | - | - | - | - | - | - | - | - |
| umb | Umbundu | umb_Latn | - | - | - | - | - | - | - | umb | umb | umb | - | - | - | umb | - | - | - | - | - |
| kmb | Kimbundu | kmb_Latn | kmb | - | - | - | - | - | - | - | kmb | kmb | - | - | kmb_Latn | - | - | - | - | - | - |
| mos | Mossi | mos_Latn | - | mos | - | - | - | - | - | mos | mos | - | - | - | mos_Latn | - | - | - | - | - | - |
| kik | Kikuyu | kik_Latn | - | ki | - | - | - | - | ki | - | kik | kik | - | kik | kik_Latn | - | - | - | - | - | - |
| luo | Luo | luo_Latn | - | luo | - | - | luo | - | - | luo | luo | - | - | luo | luo_Latn | luo | - | - | - | luo | - |
| kam | Kamba | kam_Latn | - | - | - | - | - | - | - | - | kam | kam | - | - | kam_Latn | kam | - | - | - | - | - |
| bin | Edo | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| dag | Dagbani | - | - | - | - | - | - | - | - | - | - | - | - | dag | - | - | - | - | - | - | - |
| oke | Okpe | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| shi | Tachelhit | tzm_Tfng(p) | ber(p) | ber(p) | - | - | - | - | - | - | shi | - | shi | ber(p) | tzm_Tfng(p) | - | - | - | - | - | - |
| dyu | Dyula | dyu_Latn | dyu | dyu | - | - | - | - | - | - | dyu | - | - | - | dyu_Latn | - | - | - | - | - | - |
| bam | Bambara | bam_Latn | bm | bm | - | - | - | - | bm | - | bam | - | - | bam | bam_Latn | - | - | - | - | bm | - |
| kbp | Kabiye | kbp_Latn | kbp | - | - | - | - | - | - | - | kbp | - | - | - | kbp_Latn | - | - | - | - | - | - |
| bbj | Ghomala | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| kon | Kongo | kon_Latn | kg | kg | - | - | - | - | - | kg | kon | kon | - | - | kon_Latn | - | - | - | - | ktu(p) | - |
| kea | Kabuverdianu | kea_Latn | - | - | - | - | - | - | - | - | kea | - | - | - | kea_Latn | kea | - | kea | - | - | - |
| kab | Kabyle | kab_Latn | ber_Latn(p) | ber-Latn(p) | - | - | - | - | - | - | kab | - | kab | kab | kab_Latn | - | - | - | - | - | - |
| nus | Nuer | nus_Latn | - | nus | - | - | - | - | - | - | nus | - | - | - | - | - | - | - | - | nus | - |
| swc | Congo Swahili | swh_Latn(p) | sw(p) | sw(p) | swa(p) | sw(p) | swh(p) | sw(p) | sw(p) | swc | swc | swc | - | swa(p) | swh_Latn(p) | sw(p) | - | - | - | sw(p) | sw(p) |
| lua | Luba-Kasai | lua_Latn | lu(p) | lua | - | - | - | - | - | lua | lua | lua | - | - | lua_Latn | - | - | - | - | - | - |
| kqn | Kaonde | - | - | - | - | - | - | - | - | kqn | - | - | - | - | - | - | - | - | - | - | - |
| lun | Lunda | - | - | - | - | - | - | - | - | lun | - | - | - | - | - | - | - | - | - | - | - |
| iso | Isoko | - | iso | iso | - | - | - | - | - | iso | - | - | - | - | - | - | - | - | - | - | - |
| loz | Lozi | sot_Latn(p) | st(p) | st(p) | sot(p) | - | - | st(p) | st(p) | loz | sot(p) | sot(p) | - | sot(p) | sot_Latn(p) | - | - | - | - | - | - |
| wal | Wolaytta | - | wal | - | - | - | - | - | - | - | wal | - | wal | - | - | - | - | - | - | - | - |
| ach | Acholi | - | - | ach | - | - | - | - | - | - | ach | - | - | ach | - | - | - | - | - | ach | - |
| tsc | Tswa | tso_Latn(p) | tsc | ts(p) | - | - | - | - | ts(p) | ts(p) | tso(p) | tso(p) | - | - | tso_Latn(p) | - | - | - | - | - | - |
| vmw | Makhuwa | - | - | - | - | - | - | - | - | - | vmw | vmw | - | - | - | - | - | - | - | - | - |
| kua | Kuanyama | - | kj | - | - | - | - | - | - | - | kua | kua | - | - | - | - | - | - | - | - | - |
| seh | Sena | - | seh | - | - | - | - | - | - | - | seh | seh | - | - | - | - | - | - | - | - | - |
| nyn | Nyankore | - | - | cgg(p) | - | - | - | - | - | - | nyn | nyn | - | nyn | - | - | - | - | - | cgg(p) | - |
| her | Herero | - | - | - | - | - | - | - | - | - | her | her | - | - | - | - | - | - | - | - | - |
| yao | Yao | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| sid | Sidamo | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| ndc | Ndau | sna_Latn(p) | sn(p) | ndc-ZW | sna(p) | sn(p) | sna(p) | sn(p) | sn(p) | sn(p) | sna(p) | sna(p) | - | sna(p) | sna_Latn(p) | sn(p) | - | - | - | - | - |
| kri | Krio | - | kri | kri | - | - | - | - | - | - | kri | - | - | - | - | - | - | - | - | kri | - |
| ish | Esan | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| dua | Duala | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| bas | Basaa | - | bas | - | - | - | - | - | - | - | bas | bas | - | - | - | - | - | - | - | - | - |
| wes | Cameroon Pidgin | - | - | pcm(p) | - | - | - | - | - | - | pcm(p) | - | - | pcm(p) | - | - | - | - | - | - | - |

Look-alike decisions worth knowing: MADLAD `ts` = Tsonga (our tso), `tsc` = Tswa, `ndc_ZW` = Ndau (vocab token only, not in the README list: not counted), `kg` = Kongo (our kon), `lu` = Luba-Katanga (proxy only for lua), `ach` is in the README list but has **no `<2ach>` vocab token** (not counted), `ak` = "Twi" in the paper table; NLLB has both `aka_Latn` and `twi_Latn` (ours: twi), `gaz_Latn` = West Central Oromo (our orm), `plt_Latn` = Plateau Malagasy (mlg), `kon_Latn`/`lua_Latn`/`kab_Latn` explicit, `tzm_Tfng` only a proxy for Tachelhit; TranslateGemma names Kiga (`cgg`, proxy for nyn), `lu` = Kiluba, `ber`/`ber-Latn` = Tamazight (proxies for shi/kab), `ak` = Twi, `ndc-ZW`, `lua` = Tshiluba (Luba-Lulua = Luba-Kasai, en->lua only); Sunflower's `swa`/`aka`/`ful`/`ber` are macro-language codes.
TranslateGemma caveat: its prompt template only accepts codes in a built-in name table; 17 of our 47 are **not** in it (pcm efi fon bem tum nso tiv mos luo dyu kg nus lua iso ach ndc-ZW kri), so those prompts must be built by hand (details in the registry notes). The template was read from an ungated mirror (mlx-community/translategemma-4b-it-4bit) because the google repos are gated.

## (c) Languages with no open system, or only a non-permissive one

* **Named by no open system (8)**: urh Urhobo, bin Edo, oke Okpe, bbj Ghomala, yao Yao, sid Sidamo, ish Esan, dua Duala. Plus **wes** (Cameroon Pidgin): only a proxy via Nigerian Pidgin (pcm: Sunflower, TranslateGemma, OPUS tc-bible-mul). These should be human-reference-only (if SmolDoc/FLORES+ has them) or dropped from the translation-based benchmark. Google and Microsoft do not list any of the 8 either (section g).
* **Only a non-permissive system**: **ndc (Ndau)**: TranslateGemma only (Gemma Terms; `ndc-ZW`, template-invalid). NLLB-style CC-BY-NC does not name it either. (No language is covered *only* by a CC-BY-NC system: every NLLB language is also named by at least one permissive system, though sometimes only by the OPUS tc-bible-mul model.)
* **Permissive coverage rests on a single weak-evidence family (OPUS-MT, Bible/OPUS data, no or old published numbers)**: efi, tum, umb, shi, nus, swc, kqn, lun, loz, vmw, her (OPUS only); ibb, ven, tiv, iso, wal, tsc, kua, seh, kri, bas (OPUS or MADLAD only). For kqn, lun, loz only the 2020 per-pair `opus-mt-en-*` models exist; nus, vmw, her only the 1,196-label tc-bible-mul model with **no per-language published score**. Treat these as "pseudo-reference quality unknown / likely poor" and plan human-reference-only or a native-speaker spot check.
* **Licence-sensitive (NLLB names them, MADLAD does not)**: bem, kab, kam, kea, kik, lua, luo, mos, nus, tum, umb. Their permissive alternatives are OPUS tc-bible / Afrique / Sunflower only (nus and tum: OPUS tc-bible only; umb: OPUS only). MADLAD scores ~2 chrF on these (3B/10B) in its own paper because they have no `<2xx>` token.

## (d) Recommendation for the bake-off (FLORES+ dev)

Permissive first, then the best non-permissive as a quality ceiling (suggest 4 systems, 5th/6th if cheap):

1. **`sunflower-qwen3.8-27b`** (Apache-2.0, gated repo with a free access form, 27B, 55 GB bf16 or ~28 GB fp8; HF + vLLM). Covers 31 of ours (all the high-demand ones plus ach, nyn, dag, luo, bem, bam, kik, nbl, kab). Published by Sunbird: chrF eng->xx 41.6 mean over its 68 languages (Gemini 3.1 Pro 41.8, GPT-5.1 33.3, AfriqueQwen3.5-9B 28.8), FLORES-200 14-language mean 53.5; per-language numbers in the table below (salt-69 test set, 100 sentences/language: **not comparable with FLORES chrF**). Smaller sibling `sunflower-qwen3.5-9b` (card: within 0.4 chrF) is the cheap alternative. Risks: training data undocumented; `Translate to {language}: {text}` prompt takes English language *names*.
2. **`madlad400-7b-mt-bt`** (Apache-2.0; 8.3B, 17 GB fp16; HF, CT2; **no vLLM**). 44 of ours; the paper reports +3.0 chrF en->xx for the -bt variant. Include as the main permissive MT-system baseline (in Table 17 the 3B variant is typically 0.5-8 chrF below 7.2B, e.g. sot 34.7 vs 43.1, zu 44.7 vs 48.6; 10.7B is within about +-2 of 7.2B on most languages, so skip 10B). Expect it to lose to NLLB and Sunflower on many languages (table below).
3. **`opus-mt-tc-bible-big-eng-bnt` / `-afa` / `-mul`** (Apache-2.0; 240M each, trivially cheap on CPU or GPU; HF / CT2). The only permissive coverage for 11 of our languages (efi tum umb shi nus swc kqn lun loz vmw her) and one of only two permissive options for 10 more (ibb ven tiv iso wal tsc kua seh kri bas), with a surprisingly competitive published chrF on Bantu FLORES devtest (zu 53.3, xho 52.0, nso 48.6, tso 48.5, tsn 45.2, sot 45.3, sna 45.6, nya 45.1, lin 45.6, kin 41.5, ssw 43.6, bem 36.4, kon 37.8, run 38.8, lug 34.2 per the card). Use `-bnt` for Bantu, `-afa` for hau/som/amh/tir/orm/kab/shi/wal, `-mul` for the rest (no published per-language scores for `-mul`). Bible-heavy training data: expect domain mismatch (section f).
4. **`nllb-200-3.3b`** (CC-BY-NC-4.0; quality ceiling; 3.3B; HF/CT2). 44 languages; NLLB-54B-MoE is the best published column in Table 17 for all our languages except dyu (the 3.3B dense is somewhat lower; the NLLB paper is the place to check, I did not extract it). If the licence stops NLLB outputs from being used as released pseudo-references, NLLB remains useful as an *internal* referee (e.g. to choose between permissive systems per language).
5. Optional: **`translategemma-12b-it`** (Gemma Terms) to test whether Google's Gemini-distilled model beats the above on the rare languages it was SFT'd on (47 named; only sw/zu evaluated by Google: en->sw MetricX 4.45 (27B), en->zu 6.99; both published, lower is better) and **`afriqueqwen3.5-9b-50langs-instruct`** (CC-BY-4.0, 41 of ours, adds an LLM option for rare languages such as kea, kbp, mos, kmb, kam, dyu, lua, kon, where the other permissive choices are MADLAD (variable) or Bible-trained Marian; published FLORES SSA-COMET 63.8 vs Sunflower-9B 62.3 on the AfriqueQwen card, but Sunbird's own table puts it 13 chrF below Sunflower). **`gemma-4-31b-it`** (Apache-2.0) only if there is GPU time to test coverage empirically: no list exists.
Left out on purpose: Aya-101 (15, mT5-13B), mT0/BLOOMZ (baselines of little value), Seamless (12, CC-BY-NC), Tiny Aya (10, CC-BY-NC), LLaMAX3 and N-ATLaS (Llama licence), Qwen3-32B (3 languages evaluated), Hy-MT2 / Command-A-Translate / North-Small-Translate / GemmaX2 (no African language).

### Published en->xx scores (NOT measured by us)

chrF (sacrebleu), FLORES-200, English source. Columns 3-6: Kudugunta et al. 2023, arXiv:2309.04662 Table 17 (the NLLB column is the 54B MoE, not the released dense checkpoints; the table gives `<bleu> / <chrf>`, I take chrF; split and chrF-vs-chrF++ variant not stated). OPUS tc-bible: model cards' `benchmark_results.txt`, flores200-devtest, chrF x 100 (Bantu model for Bantu languages, Afro-Asiatic model for hau/som/amh/tir/kab; `swh` 24.5 is an unexplained outlier). Sunflower: Sunbird README, salt-69 test set (100 sentences/language), chrF eng->xx, **different test set**.

| iso | language | NLLB-54B-MoE | MADLAD-3B | MADLAD-7.2B | MADLAD-10.7B | OPUS tc-bible bnt/afa | Sunflower-27B (other test set) |
|---|---|---|---|---|---|---|---|
| yor | Yoruba | 27.4 | 19.7 | 20.3 | 20.7 | - | 51.5 |
| hau | Hausa | 56.0 | 44.4 | 45.3 | 47.1 | 49.0 | 57.7 |
| ibo | Igbo | 43.6 | 34.6 | 38.7 | 38.5 | - | 61.1 |
| pcm | Pidgin | - | - | - | - | - | 57.1 |
| ewe | Ewe | 41.9 | 26.9 | 29.2 | 28.4 | - | 52.0 |
| fon | Fon | 23.5 | 15.6 | 16.0 | 15.9 | - | - |
| wol | Wolof | 31.8 | 13.2 | 12.6 | 15.4 | - | 34.4 |
| swh | Swahili | 61.2 | 57.6 | 58.1 | 58.5 | 24.5 | 68.1 |
| kin | Kinyarwanda | 53.8 | 14.6 | 9.6 | 11.9 | 41.5 | 59.6 |
| som | Somali | 47.3 | 41.8 | 43.4 | 43.0 | 42.8 | 59.7 |
| orm | Oromo | 43.3 | 22.8 | 29.3 | 29.1 | - | 46.5 |
| amh | Amharic | 42.2 | 31.2 | 32.2 | 32.5 | 32.2 | 50.3 |
| tir | Tigrinya | 27.9 | 15.7 | 18.6 | 18.9 | 15.6 | - |
| lug | Luganda | 44.7 | 27.0 | 30.3 | 29.6 | 34.2 | 58.5 |
| lin | Lingala | 50.8 | 18.1 | 18.8 | 20.8 | 45.6 | 60.8 |
| zul | Zulu | 58.8 | 44.7 | 48.6 | 47.6 | 53.3 | 68.4 |
| xho | Xhosa | 54.7 | 47.4 | 50.1 | 49.2 | 52.0 | 65.1 |
| afr | Afrikaans | 66.6 | 66.1 | 67.6 | 67.3 | - | 82.9 |
| sna | Shona | 48.8 | 38.2 | 42.3 | 41.8 | 45.6 | 52.5 |
| nya | Chichewa | 48.7 | 38.6 | 44.9 | 43.5 | 45.1 | 69.4 |
| sot | Sesotho | 49.2 | 34.7 | 43.1 | 41.6 | 45.3 | 61.4 |
| tsn | Setswana | 51.0 | 34.8 | 38.6 | 37.7 | 45.2 | 57.2 |
| mlg | Malagasy | 54.2 | 50.8 | 53.1 | 52.8 | - | 61.1 |
| bem | Bemba | 42.3 | 2.4 | 17.7 | 2.7 | 36.4 | 46.3 |
| tum | Tumbuka | 38.5 | 2.2 | 15.3 | 2.5 | 12.0 | - |
| run | Kirundi | 47.3 | 31.8 | 32.6 | 30.7 | 38.8 | 50.6 |
| tso | Tsonga | 53.2 | 34.7 | 38.1 | 37.0 | 48.5 | - |
| nso | Sepedi | 53.3 | 37.7 | 41.1 | 40.5 | 48.6 | - |
| ssw | Swati | 49.2 | 33.4 | 34.7 | 36.0 | 43.6 | - |
| nbl | South Ndebele | - | - | - | - | - | 47.4 |
| sag | Sango | 38.0 | 30.1 | 29.2 | 30.7 | - | - |
| umb | Umbundu | 31.2 | 2.9 | 10.5 | 2.5 | 25.2 | - |
| kmb | Kimbundu | 28.3 | 19.8 | 19.1 | 19.4 | 22.7 | - |
| mos | Mossi | 26.4 | 2.3 | 16.6 | 2.2 | - | - |
| kik | Kikuyu | 40.1 | 2.2 | 15.5 | 1.8 | 25.5 | 40.2 |
| luo | Luo | 41.7 | 2.7 | 18.3 | 2.7 | - | 52.2 |
| kam | Kamba | 28.8 | 2.3 | 17.7 | 3.6 | 21.1 | - |
| dag | Dagbani | - | - | - | - | - | 32.4 |
| dyu | Dyula | 19.8 | 21.0 | 21.1 | 22.1 | - | - |
| bam | Bambara | 32.9 | 27.8 | 26.8 | 30.1 | - | 39.3 |
| kbp | Kabiye | 30.4 | 14.7 | 16.4 | 16.9 | - | - |
| kon | Kongo | 48.2 | 14.7 | 16.3 | 19.0 | 37.8 | - |
| kea | Kabuverdianu | 44.9 | 3.1 | 10.0 | 3.1 | - | - |
| kab | Kabyle | 37.9 | 1.4 | 12.4 | 1.9 | 24.8 | 33.4 |
| nus | Nuer | 31.5 | 29.3 | 27.7 | 30.2 | - | - |
| lua | Luba-Kasai | 39.4 | 2.2 | 12.0 | 2.6 | 18.2 | - |
| ach | Acholi | - | - | - | - | - | 51.2 |
| nyn | Nyankore | - | - | - | - | - | 49.5 |

How these were treated: the registry never ranks systems; the bake-off should measure all candidates on FLORES+ dev with the same chrF++/COMET settings used elsewhere in the project. FLORES is public and may sit in training data of the LLM-type systems: see risks.

## (e) Throughput facts for GPU inference (sourced; see `_throughput_raw.md` for all rows)

Hardware: RTX PRO 6000 Blackwell = 96 GB GDDR7, **1,792 GB/s** (NVIDIA spec). Comparison bandwidths: A100-80GB 1,935-2,039 GB/s, H100 SXM 3.35 TB/s.
* **vLLM, bf16, 1x RTX PRO 6000, 50 concurrent requests, 100 in / 600 out** (hosting-blog measurement, databasemart.com): gemma-3-12b-it 1,560 output tok/s; Llama-3.1-8B / Qwen3-8B ~2.7k; Qwen3-14B 1,744; 32B-class ~0.8k. That is 40-60% of the weight-bandwidth bound (derived by the research agent: bound = 1,792 GB/s / weight GB x concurrency). Same blog: at 300 concurrent the PRO 6000 is at or above A100-80GB and H100 in total tok/s for 8B-32B dense models; on A100, 50 -> 300 concurrency gives 2.3x (8B), 2.9x (14B), 1.25x (32B). No first-party throughput exists for TranslateGemma, Gemma 3/4 or Sunflower; the vLLM recipe page for TranslateGemma gives only serving flags (optimised for ~2K context).
* **NLLB / CTranslate2**: the only published figure is the AfriNLLB paper (arXiv 2602.09373): NLLB-200-distilled-600M, CT2, beam 3, A40: 1,468 output tok/s on average (en->xx 1,530); the pruned 548M model in fp16 3,532 tok/s (en->xx 3,733), about 2.4x. No A100/H100 number, no HF baseline, no int8, no sentences/s. CT2's own OPUS-MT (Marian) benchmark on an A10G: HF 1,023 tok/s vs CT2 fp16 9,297 tok/s (about 9x); int8 slower than fp16 on that GPU. Meta's NLLB paper: 3.3B dense decode 246 words/s on one GPU (old, unspecified).
* **MADLAD-400**: no throughput numbers from any backend; vLLM does not serve it (T5). Options: CT2 (community conversions), HF, llama.cpp GGUF (exists), TensorRT-LLM (unverified).
* **African-language token fertility**: only a Gemma 4 tokenizer measurement exists (tokens/word: English 1.23, Hausa 1.85, Igbo 2.27, Amharic 3.04; community dataset CipherSenseAI/afri-fertility-results); not verified for the Gemma 3 / TranslateGemma tokenizer.
* **ESTIMATES (arithmetic, not measurements)** for planning on this card, assuming ~50 output tokens per translated African-language sentence (25 English tokens x 1.5-2.5 fertility): Gemma-family 4B / 12B / 27B on vLLM at high concurrency ~100-200 / 60-100 / 20-50 sentences/s, i.e. ~1.4-2.8 / 2.8-4.6 / 5.6-14 h per million sentence-translations; NLLB-600M-class on CT2 fp16 ~70 sentences/s per A40-class GPU (scaled from the AfriNLLB figure; 3.3B pessimistic floor ~13 sentences/s; unmeasured). 27B Sunflower is in the 27B class (fp8 halves weight reads). **Pilot before committing**; the per-pair Marian models (75M-240M) should be 10-100x cheaper than anything above.
* Batching: use length-sorted batches with token budgets for CT2 (`batch_type="tokens"`), `max_num_batched_tokens` above 8192 and 256+ concurrent sequences for short sentences in vLLM (docs); Anyscale (2023, OPT-13B) reports 8x for continuous vs static batching and up to 23x for vLLM, an upper bound for HF-pipeline baselines.

## (f) Risks

1. **Training-data overlap with the benchmark and with us.** MADLAD-400 (CommonCrawl-derived, also the monolingual base of AfriqueLLM), NLLB (mined web bitext, seed and public bitext incl. religious text, per the NLLB paper, not re-read today), OPUS-MT (OPUS corpora; the tc-bible checkpoint names carry 'jhubc' = JHU Bible corpus; the 2020 per-pair cards cite GlobalVoices etc.), Sunflower (training data undocumented on the card; its test set is Sunbird/salt-69), TranslateGemma (Gemini-2.5-Flash synthetic translations of monolingual web text plus public parallel documents) all saw web text that may contain our English source documents or their human translations (SmolDoc, FLORES+ passages). Sunbird itself reports that Claude Sonnet 4.5 reproduced the public FLORES English references verbatim (contamination probe), so FLORES-based numbers (all the published ones above) are an optimistic upper bound for LLM-type systems. Run the existing contamination checks (`eval/contamination.py`) on each source set against each translator's known corpora where available, and keep FLORES+ *dev* out of pseudo-reference selection if devtest is used for reporting.
2. **Pseudo-reference circularity.** If any of our SFT/RL data or reward models (AfriCOMET/SSA-COMET) were built from these translators (NLLB, MADLAD, TranslateGemma, Gemini), a Gemma-MT model evaluated against their outputs is partly graded by its own teachers. The registry should be cross-checked against the data-generation lineage before the benchmark is called independent.
3. **Bible/religious data and domain mismatch.** OPUS-MT tc-bible, NLLB and MADLAD (and Aya/mT5-family) rely on religious parallel text for many rare languages (bem, tum, umb, kmb, kam, kik, kon, lua, seh, vmw, tiv, iso, loz, lun, kqn, her, kua...). The outputs will lean on biblical vocabulary and register and may be fluent-sounding but wrong for news/science/long-form text; FLORES chrF cannot reveal this. Prefer systems with non-religious in-domain evaluation and have native speakers spot-check a few documents per rare language.
4. **Sentence-level only.** NLLB (trained up to 512 tokens, card says not for document translation), MADLAD (training length 256), Marian models and Seamless are sentence-level; the pseudo-reference of a document is a concatenation of independently translated sentences (no discourse, pronoun/terminology consistency, no paragraph structure). LLM-based systems (Sunflower, Afrique, TranslateGemma, Gemma 4) could translate paragraphs but were mostly tuned and evaluated on single sentences (Sunflower eval uses a 100-token budget). Segmentation and sentence-alignment errors (merge/split, skipped sentences, repetition loops for LLMs, hallucination on very short inputs) add noise: needs length-ratio and LID filters per language.
5. **Licence risks of releasing outputs.** NLLB / Seamless / Tiny Aya (CC-BY-NC, Cohere AUP); Gemma Terms Model-Derivative clause for TranslateGemma and Gemma 3 outputs; Llama 3 clause for LLaMAX3/N-ATLaS; Afrique's CPT data includes GPT-4.1-translated text (OpenAI terms on that training data); Sunflower and TranslateGemma teacher/data terms are not documented on the card. Apache-2.0 systems (MADLAD, OPUS-MT, Sunflower, Gemma 4) are the only clean path. None of this is legal advice.
6. **Coverage and access risks.** Sunflower / TranslateGemma / Tiny Aya need gated-repo acceptance; OMT (1,600 languages) has no weights; vLLM does not run the best classical MT systems; the ctranslate2 package and its Blackwell build are untested here. Google's closed-system language list was read through a page-extraction tool (not raw HTML), and its currency is unconfirmed.
7. **Unevaluated claims.** TranslateGemma's 47 named languages and Gemma 4 / Qwen3 / Gemma 3 "140+/119 languages" are claims of training coverage, not measured quality for these directions; OPUS tc-bible-mul has no per-language scores; MADLAD `ach`/`ndc_ZW`/`ber_Latn` are ambiguous edge cases.

## (g) Closed systems as a reference ("closed upper bound")

Source (retrieved 2026-10-20, read through a page-extraction tool, not raw HTML; no scraping): Google Cloud Translation language list https://cloud.google.com/translate/docs/languages (no "last updated" date shown; the Translation LLM list adds no African language that NMT lacks; am ha ig so are marked Experimental there) and Microsoft Translator language support https://learn.microsoft.com/en-us/azure/ai-services/translator/language-support (page metadata dated 2026-06-06, updated 2026-08-23; no African language flagged preview). Presence on a list says nothing about quality. Per-language cells are the `google` / `microsoft` columns of the matrix in (b); the full table with notes is in `_closed_systems_raw.md`.

| system | explicit | proxy only | not supported |
|---|---|---|---|
| Google Translate | 33 | 4 (pcm->kri Krio, kon->ktu Kituba, swc->sw, nyn->cgg Kiga) | 40 |
| Microsoft Translator | 20 (all also on Google's list) | 1 (swc->sw) | 56 |

Supported by neither (40): efi fon wol ibb urh tum ven tiv umb kmb mos kik kam bin dag oke shi dyu kbp bbj kea kab lua kqn lun iso loz wal tsc vmw kua seh her yao sid ndc ish dua bas wes. A "closed upper bound" could therefore cover at most 33 languages (Google); all 33 are also named by at least one permissive open system. Caveat: wol, fon and dyu are absent from this extraction of Google's list; verify on the live page before quoting.

## (h) Hand-off notes for the bake-off script

* Prompt styles: `nllb` (tokenizer `src_lang="eng_Latn"`, `forced_bos_token_id`), `madlad` (`"<2xx> "+text`), `chat` (apply the model's chat template to `template.format(src_lang="English", tgt_lang=LANGUAGES[iso].name, tgt_code=codes[iso], text=...)`; TranslateGemma needs a hand-built turn for template-invalid codes; Qwen-based systems: `enable_thinking=False`), `t5prompt` (plain string; LLaMAX uses an Alpaca wrapper), `seamless` (`SeamlessM4Tv2ForTextToText`, `tgt_lang=<3-letter>`), `marian` (per-pair `hf_id_template` or `>>xxx<< ` target label).
* Fields not obtainable offline or not verified: revisions of the per-pair Marian repos (per-code SHAs are in `system_cards.json`), TranslateGemma `config.json` / `chat_template` from the gated repos (read from mirrors), Tiny Aya / Sunflower `config.json` (gated), CT2 on sm_120, the BLOOMZ licence text, Llama-4 / Seed-X cards (not read).
* The test for the registry runs with `python tests/run_tests.py tests/test_longdoc_systems.py` on the laptop; on the rental (real pytest installed) use `python -m pytest tests/test_longdoc_systems.py` because `run_tests.py` does not expand `@parametrize` when pytest is present (95 passed there).
