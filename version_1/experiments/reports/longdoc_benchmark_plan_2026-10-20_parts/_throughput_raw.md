# Throughput raw facts for single-GPU (RTX PRO 6000 Blackwell 96 GB) African-language MT

Researched 2026-10-20, web only. Every number below was read from the cited page in this session unless flagged "snippet only". Anything marked ESTIMATE is my own arithmetic, with the arithmetic shown. "not found" means searched and nothing usable surfaced.

## 1. Measured / published numbers

| # | Model | Backend | Hardware | Number & unit | Source URL | Caveat |
|---|-------|---------|----------|---------------|------------|--------|
| 1 | NLLB-200-distilled-600M (baseline, unquantized; precision not stated in paper, presumably fp32) | CTranslate2 (inference), beam 3, batch 1024 tokens | 1x A40 48GB | avg 1,468 output tok/s over all 30 directions (Table 7); per-pair range 676-1,780 tok/s; en-xx avg 1,530 tok/s | https://arxiv.org/html/2602.09373 (AfriNLLB, Moslem et al.; AfricaNLP 2026) | Published by the paper authors. Tokens/s only; sentences/s NOT reported. "Time (s)" column undefined. Test-set sizes unknown. |
| 2 | AfriNLLB 548M (12 enc / 8 dec layers, pruned + fine-tuned) | CTranslate2, beam 3, batch 1024 tokens, fp32 | 1x A40 48GB | avg 1,834 tok/s (+~24.9% vs baseline); en-xx 1,947 tok/s | same | Same caveats. Pruned model is not the stock 600M. |
| 3 | AfriNLLB 548M, FP16 | CTranslate2, beam 3, batch 1024 tokens | 1x A40 48GB | avg 3,532 tok/s (~2.4x the 600M baseline); en-xx 3,733 tok/s; per-pair range 1,130-4,222 tok/s | same | Paper text says "57% faster" for FP16 but its own tables show ~2.4x; paper is internally inconsistent. No int8 numbers reported. |
| 4 | OPUS-MT (Marian-sized En-De), NOT NLLB | HF Transformers 4.26.1 (PyTorch 1.13.1) | AWS g5.xlarge, 1x A10G | 1,022.9 tok/s | https://github.com/OpenNMT/CTranslate2 (Benchmarks section) | Published by CTranslate2 project, En->De newstest2014. Batch size not stated. Old versions (CT2 3.6.0, 2023). |
| 5 | same OPUS-MT | CTranslate2 3.6.0 fp32 / int8 / float16 / int8+float16 | 1x A10G | 5,876 / 7,522 / 9,297 / 8,363 tok/s | same | float16 is fastest, int8 and int8+float16 are slower than float16 on this GPU. float16 vs HF = 9,297/1,023 = 9.1x (my division). Small Marian model, not NLLB. |
| 6 | same OPUS-MT | HF Transformers vs CTranslate2 | AWS c5.2xlarge CPU, 4 threads | HF 147.3 tok/s; CT2 525.0 (fp32), 696.1 (int8) | same | CPU only, for completeness. |
| 7 | NLLB-200 3.3B dense | fairseq-style (Meta), batch "4k" tokens, 1 GPU | GPU model not stated for that row; text says pruned MoE "requires a single 32GB V100" | 246 words/s (86 s per FLORES-valid decode, averaged over 29 languages -> English) | https://arxiv.org/abs/2212.09811 (Table 5, read from PDF text) | Published by Meta authors. Words/s not tokens/s; V100 attribution is inferred from surrounding text. Teacher-forcing for some steps is mentioned elsewhere in the paper; this row is a decode benchmark. Old hardware. |
| 8 | NLLB-200 3.3B | CTranslate2 2.34 FP16 | RTX 5060 Ti (per vendor) | ~350 output tok/s | https://gigagpu.com/?p=32017 (search snippet ONLY, page not fetched) | Hardware-vendor blog, loosely described, unverified. Treat as anecdote. |
| 9 | NLLB-200 600M/1.3B/3.3B, HF Transformers vs CT2 on A100/H100 | - | - | not found | - | See "not found". |
| 10 | MADLAD-400 3B/7B/10B | any backend | any | not found | - | Community CT2 conversions exist (https://huggingface.co/santhosh/madlad400-3b-ct2, https://huggingface.co/LifeAi-dev/madlad400-3b-mt-ct2-int8) but model cards contain no speed numbers. |
| 11 | gemma-3-12b-it (bf16) | vLLM, `benchmark_serving.py`, 50 prompts, request-rate inf, 100 in / 600 out, max-model-len 4096, TP1 | 1x RTX PRO 6000 (96 GB) | 1,560 output tok/s (1,817 total tok/s) | https://www.databasemart.com/blog/vllm-gpu-benchmark-pro6000 | Hosting-provider blog measurement. vLLM version not stated. Only 50 requests x 600 tokens (30,000 generated tokens) so ramp-up/tail depress the figure. |
| 12 | Llama-3.1-8B bf16 / Qwen3-8B bf16 / DS-R1-Distill-Llama-8B bf16 | vLLM, same settings as row 11 | 1x RTX PRO 6000 | 2,744 / 2,658 / 2,755 output tok/s (3,196 / 3,101 / 3,210 total) | same | Same caveats. |
| 13 | Qwen3-14B bf16 / DS-R1-Distill-Qwen-14B bf16 | vLLM, same settings | 1x RTX PRO 6000 | 1,744 / 1,608 output tok/s (2,034 / 1,873 total) | same | Same caveats. |
| 14 | DS-R1-Distill-Qwen-32B bf16 / Qwen3-VL-32B bf16 | vLLM, same settings | 1x RTX PRO 6000 | 829 / 796 output tok/s (966 / 929 total) | same | Same caveats. No 27B/30B dense row on this page. |
| 15 | gpt-oss-20b / gpt-oss-120b (page says 8-bit in table, 4-bit in text; unresolved) | vLLM, same settings | 1x RTX PRO 6000 | 3,753 / 1,525 output tok/s | same | MoE, not comparable to dense. |
| 16 | DS-R1-Distill-Llama-8B / -Qwen-14B / -Qwen-32B bf16 | vLLM, 300 concurrent | 1x RTX PRO 6000 | 8,990 / 5,160 / 1,655 TOTAL tok/s (in+out, output-only not given) | same | Same page also gives A100-80GB 6,842 / 4,187 / 721 and H100 5,934 / 4,821 / 1,482 total tok/s at 300 concurrent. Sample is the only same-page PRO 6000 vs H100 vs A100 comparison I found. |
| 17 | DS-R1-Distill-Llama-8B / -Qwen-14B / -Qwen-32B bf16 | vLLM, 50 vs 300 concurrent, 100 in / 600 out | 1x A100 80GB | OUTPUT tok/s: 2,552 -> 5,791 (2.27x); 1,134 -> 3,335 (2.94x); 472 -> 591 (1.25x) | https://www.databasemart.com/blog/vllm-gpu-benchmark-a100-80gb | Hosting-provider blog. Only source I found with same-unit concurrency scaling. gemma-2-9b-it (not gemma-3): 328 -> 512 output tok/s at 50 -> 300; gemma-2-27b-it: 302 -> 915. Gemma-2 numbers look odd (low 9B figure); distrust. |
| 18 | gemma-3-27b-it bf16 | vLLM TP=2 | 2x A100 40GB | 1,012 output tok/s (1,232 total), 50 concurrent, 100 in / 600 out | https://www.databasemart.com/blog/vllm-gpu-benchmark-dual-a100-40gb (from search summary; page not fully read) | Hosting-provider blog. Two cards, so not single GPU. |
| 19 | Qwen3-Coder-30B (MoE, AWQ int4 weights, FP8 KV cache) | vLLM, 8K context | 1x RTX PRO 6000 | 8,425 tok/s (RTX 5090: 4,570; RTX 4090: 2,259) | https://www.cloudrift.ai/gpu-benchmarks | CloudRift (GPU cloud). Concurrency, input/output lengths, vLLM version NOT stated. MoE with ~3B active params, so not representative of dense 8B-30B. |
| 20 | GLM-4.5-Air (4-bit AWQ, MoE) | vLLM, 1000 in / 1000 out, concurrency 256-512, fp8 KV | 1x RTX PRO 6000 vs 1x H100 | 3,140 vs 2,987 tok/s | https://www.cloudrift.ai/blog/benchmarking-rtx6000-vs-datacenter-gpus | CloudRift. Shows PRO 6000 roughly equal to H100 on a 4-bit MoE at high concurrency; rows per-run concurrency not given. MoE, so not directly transferable. |
| 21 | Llama 3.1 8B | Ollama (not vLLM), single user | RTX PRO 6000 | ~227 tok/s (RTX 5090 ~264) | https://vrlatech.com/gpu-benchmark-ai-llm-2026/ | Vendor article; batch-1 decode, quoted for bandwidth sanity only. |
| 22 | nvidia/Gemma-4-26B-A4B-NVFP4 (MoE) | vLLM, 16-64 concurrent | RTX PRO 6000 | 12,614 tok/s (median of 7 runs) | https://www.trooper.ai/benchmarks-compare-RTX-Pro-5000-Blackwell-with-RTX-Pro-6000-Blackwell | GPU-rental provider fleet data. Qwen3-8B on vLLM listed as ~5,988 tok/s vs 1,102 on RTX 4090 in search summary (settings not read). MoE + NVFP4: not comparable to bf16 dense. |
| 23 | TranslateGemma 4B/12B/27B on vLLM | vLLM | H100/A100/RTX 6000-class | not found | - | Only a serving recipe, no throughput: https://docs.vllm.ai/projects/recipes/en/latest/Google/TranslateGemma.html (image vllm/vllm-openai:v0.14.1-cu130, Infomaniak-AI/vllm-translategemma-27b-it, `--gpu-memory-utilization 0.8`; "verified for 4B/27B"; model optimised for ~2K context). |
| 24 | translategemma-4b-it 4-bit | oMLX (Apple MLX), NOT vLLM | Apple M5 Max 40-core GPU, 128 GB | TG 160 tok/s at batch 1 -> 448 tok/s at batch 8 (2.8x); 1k context | https://omlx.ai/benchmarks/9n62v98a | Community benchmark, different hardware and engine. Only shows that batching helps; no sentence lengths. |
| 25 | Gemma 4 31B dense / 26B-A4B MoE | vLLM, concurrency 1 | 1x H100 | ~40 / ~177 tok/s per stream baseline | https://markaicode.com/benchmarks/hugging-face-gemma-4-h100-throughput-benchmark/ (search summary only) | Low-quality benchmark blog (the same site admits an earlier unverified harness). Concurrency 1 only. Distrust. |
| 26 | OPT-13B | HF Pipelines (static batching) vs continuous batching vs vLLM | 1x A100 40GB | HF static: as low as 81 tok/s at highest length variance; continuous batching "8x" over naive; vLLM "up to 23x" | https://www.anyscale.com/blog/continuous-batching-llm-inference | Anyscale, June 2023. 1,000 requests, 512 input tokens, generation lengths exponential mean 128, burst arrival. Old; upper bound; baseline is HF *pipelines* static batching, not modern HF generate with sorted batches. Per-run numbers only in images. |

## 2. vLLM and encoder-decoder (T5 / MADLAD / NLLB) status

| Fact | Source | Caveat |
|------|--------|--------|
| vLLM issue #7366 (encoder/decoder RFC): T5 never supported in V0 (needed attention bias); on 2025-05-19 maintainers said enc-dec "limited to V0 only"; Sept 2025 maintainer said support was pulled because it "adds too much to code complexity"; Dec 2025 reply confirmed support dropped incl. T5Gemma; issue auto-closed "not planned" 2025-09-18. | https://github.com/vllm-project/vllm/issues/7366 | Read via WebFetch summary. |
| T5 PR "[Model] Add T5 model (2/2)" #11901 (opened 2025-01-09) was closed NOT merged on 2025-10-02: "Closing as vLLM no longer supports encoder-decoder models." | https://github.com/vllm-project/vllm/issues/11901 | |
| Current vLLM docs (latest, fetched 2026-10-20): T5ForConditionalGeneration, MADLAD, NLLB/M2M100, TranslateGemma are not in the supported-models list. Whisper is native in V1; BART and Florence-2 are supported only via an external plugin (vllm-project/bart-plugin, needs vLLM >= 0.24.0, no benchmarks given; T5 and NLLB/M2M100 not mentioned there, MBart listed as TODO). | https://docs.vllm.ai/en/latest/models/supported_models/ , https://docs.vllm.ai/en/latest/usage/v1_guide.html , https://github.com/vllm-project/bart-plugin | Conclusion: as of these pages, vLLM does NOT serve MADLAD-400 (T5) or NLLB (M2M100). I could not fetch release notes to date the plugin precisely. |
| llama.cpp supports T5/FLAN-T5 (PR #5763: llama_encode etc.) and its converter lists MT5/UMT5ForConditionalGeneration. MADLAD GGUF exists (e.g. https://huggingface.co/mtsdurica/madlad400-3b-mt-Q4_0-GGUF). No speed numbers found. | https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/807b0c49ff7071094f97ebc3a0a8e2b9e274f503 | MADLAD-specific support not confirmed by a source; PR merge date not confirmed. |
| TensorRT-LLM documents T5/BART/mBART/Whisper on its PyTorch backend. | https://nvidia.github.io/TensorRT-LLM/latest/models/encoder-decoder.html | Not verified for MADLAD or NLLB; no throughput given. |
| CTranslate2 on Blackwell (sm_120): only a secondhand lead (Whishper issue says CTranslate2 >= 4.5.0 with CUDA 12.8 works for RTX 50-series). Not confirmed in CT2 release notes. | https://github.com/pluja/whishper/issues/172 | Verify with a smoke test before planning around CT2 on this card. |

## 3. Hardware specs (published)

| GPU | Memory | Bandwidth | Source |
|-----|--------|-----------|--------|
| RTX PRO 6000 Blackwell Workstation Edition | 96 GB GDDR7 ECC | 1,792 GB/s (NVIDIA spec; marketed as 1.8 TB/s), 600 W max | https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/ |
| H100 SXM | 80 GB HBM3 | 3.35 TB/s | https://www.nvidia.com/en-us/data-center/h100/ |
| H100 NVL | 94 GB | 3.9 TB/s | same |
| A100 80GB PCIe / SXM | 80 GB | 1,935 / 2,039 GB/s | https://www.nvidia.com/en-us/data-center/a100/ |

Observed (databasemart, rows 16): at 300 concurrent, PRO 6000 is above H100 and A100-80GB for 8B/14B/32B bf16 total tok/s in that blog, despite lower bandwidth. That is the source's measurement; I do not know why (vLLM versions / settings not stated) and would not generalise.

## 4. Tokens per African-language sentence

- Gemma 4 tokenizer (vocab 262,144) fertility in tokens/word, from a community dataset: English 1.229, Hausa 1.851, Igbo 2.274, Amharic 3.035; Yoruba/Swahili/Zulu rows were not readable. Source: https://huggingface.co/datasets/CipherSenseAI/afri-fertility-results . Ratios vs English (my division): Hausa 1.51x, Igbo 1.85x, Amharic 2.47x.
- "The African Language Tax" (arXiv 2606.24460): Gemma 4 reduces mean African premium from 3.31x (cl100k_base) to 2.38x; Ethiopic and N'Ko scripts 7-9x on GPT-5-style tokenizers; Amharic latency multiplier 7.4x. https://arxiv.org/abs/2606.24460 (abstract only read).
- Gemma 3 / TranslateGemma tokenizer fertility on African languages: not found (Gemma 3 and Gemma 4 both list 262k vocab, but no source says the tokenizers are identical, so Gemma 4 numbers may not transfer).

## 5. Batch size / continuous batching statements

- Anyscale (row 26): continuous batching ~8x over static, vLLM up to 23x, with high output-length variance. OPT-13B/A100-40GB, 2023.
- vLLM optimization docs: "set `max_num_batched_tokens` > 8192 especially for smaller models on large GPUs" for throughput; chunked prefill default in V1; lowering `max_num_seqs` only recommended to reduce preemption. https://docs.vllm.ai/en/stable/configuration/optimization
- CTranslate2 performance doc: "use a larger batch size whenever possible"; prefer `batch_type="tokens"`; inputs sorted by length inside a batch; default beam 2, `beam_size=1` suggested for speed; Tensor Core GPU (CC >= 7.0). https://github.com/OpenNMT/CTranslate2/blob/master/docs/performance.md
- Concurrency scaling data point (row 17): A100-80GB, 50 -> 300 concurrent: 8B 2.27x, 14B 2.94x, 32B 1.25x (output tok/s). Small models benefit most from more concurrency.
- Batch-scaling data point (row 24): TranslateGemma-4B on M5 Max, batch 1 -> 8 = 2.8x.

## 6. NOT FOUND

- NLLB-200 (600M / 1.3B / 3.3B) sentences/s or tokens/s with HF Transformers vs CTranslate2 on A100/H100/consumer GPUs, fp16 vs int8: not found (only OPUS-MT proxy in rows 4-6; AfriNLLB reports CT2 only, no HF run, no int8).
- OpenNMT forum NLLB speed posts: forum.opennmt.net could not be fetched from this environment (DNS failure); a search summary says the tutorial post reports quality (TICO-19) but no speed. Not verified.
- Any sentences/s figure for NLLB, MADLAD or TranslateGemma on any GPU.
- MADLAD-400 3B/7B/10B throughput on HF, CT2, llama.cpp (any hardware): not found.
- vLLM T5/MADLAD support: not supported per sources above (not found as supported in any version).
- TranslateGemma 4B/12B/27B vLLM tokens/s on H100/A100/RTX 6000-class: not found. Same for Unsloth, Red Hat/Neural Magic, NVIDIA, Google blogs (no throughput published that I could find).
- Gemma-3 27B/12B/4B offline-batch numbers on H100: only low-quality blogs; no first-party number.
- Dense 27B-30B (non-MoE) vLLM figures on RTX PRO 6000: not found (only 32B rows in row 14 and MoE rows).
- Published "maximum useful batch size" for short-sentence MT in vLLM vs HF generate: not found.
- AfriNLLB int8 results: none in paper. AfriNLLB hardware is A40, not A100/H100.
- Fujitsu whitepaper (vLLM 0.19.0, RTX PRO 6000 Server Edition) returned HTTP 403; not read. https://sp.ts.fujitsu.com/dmsp/Publications/public/wp-llm-inference-benchmark-en.pdf

## 7. ESTIMATES for RTX PRO 6000 Blackwell (all derived, not sourced)

Arithmetic basis. Weights bf16 = 2 bytes x nominal params: 4B ~8 GB, 12B ~24 GB, 27B ~54 GB; fp8 halves. Decode is bandwidth-bound: weight-read steps/s <= 1,792 GB/s / weight GB; aggregate tok/s <= steps/s x concurrent sequences (ignores KV-cache reads, compute limits and scheduler overhead, so it is an upper bound).

1. Bandwidth ceilings (ESTIMATE): 4B bf16 = 1,792/8 = 224 steps/s; 12B bf16 = 1,792/24 = 74.7; 27B bf16 = 1,792/54 = 33.2; 27B fp8 = 1,792/27 = 66.4. Per sequence at batch 1 these are the max single-stream tok/s. For comparison A100-80GB (2,039 GB/s) is 1.14x and H100 SXM (3,350 GB/s) 1.87x the PRO 6000 on this bound.
2. Calibration from row 11-14 at 50 concurrent (ESTIMATE of efficiency): 8B bf16 (16 GB): bound 112 x 50 = 5,600 vs measured 2,744 = 49%; 14B (28 GB): 64 x 50 = 3,200 vs 1,744 = 54%; gemma-3-12b (24 GB): 74.7 x 50 = 3,733 vs 1,560 = 42%; 32B (64 GB): 28 x 50 = 1,400 vs 829 = 59%. So realistic vLLM output rate at ~50 concurrent is 40-60% of the bandwidth bound, and higher concurrency should move it up (A100 data: 2.3-2.9x for 8B-14B going 50 -> 300).
3. Tokens per sentence (ESTIMATE): assume ~25 tokens per English sentence (assumption, not sourced); Gemma 4 fertility ratios 1.5x (Hausa) to 2.5x (Amharic) give ~40-65 output tokens per translated sentence, central ~50, worse for Ethiopic/N'Ko-type scripts. Gemma 3 tokenizer not measured.
4. Gemma-class LLM translators on vLLM, bf16, high concurrency (ESTIMATE, extrapolated from rows 11-14, not measured): 4B ~5,000-10,000 output tok/s (8B measured 2,744 at conc 50, x2 for half the weight bytes, x1-2 for higher concurrency); 12B ~3,000-5,000 (measured 1,560 at conc 50, x2-3 for concurrency); 27B bf16 ~1,000-2,500 (32B measured ~800 at conc 50, x1.19 for fewer weight bytes, up to x2 with concurrency; fp8 could add up to ~1.5x if bandwidth-bound, unverified). At ~50 tokens/sentence: 4B ~100-200 sent/s (1.4-2.8 h per 1M sentence-translations); 12B ~60-100 sent/s (2.8-4.6 h per 1M); 27B ~20-50 sent/s (5.6-14 h per 1M). Treat as order-of-magnitude, wide error bars.
5. NLLB-600M-class on CTranslate2 fp16 (ESTIMATE from row 3): A40 gave ~3,530 tok/s at beam 3 -> at ~50 tokens/sentence = ~70 sent/s per A40-class GPU; a PRO 6000 should do at least comparable, but small-model CT2 speed is often launch/compute-limited rather than bandwidth-limited, so do not assume it scales with bandwidth. 1.3B/3.3B CT2 and MADLAD-400 3B/7B/10B: no published figure; only fallback is weight-ratio scaling (3.3B ~5.5x the 600M's 1.2 GB fp16 weights -> ~13 sent/s per A40-class GPU, a pessimistic floor). Need a pilot run: use `batch_type="tokens"`, large batches, sort by length, and check the sm_120 CT2 build first. Since vLLM does not serve T5/NLLB, MADLAD options are CT2, HF generate, llama.cpp or TensorRT-LLM, all unbenchmarked in the sources found.

## 8. Five-line summary of what can be assumed

1. Hardware: RTX PRO 6000 Blackwell = 96 GB, 1,792 GB/s (NVIDIA), about 54% of an H100 SXM (3,350 GB/s) and 88% of an A100-80GB PCIe/SXM (1,935-2,039 GB/s); community vLLM runs put it level with or above H100/A100 on bf16 dense models at 300 concurrent.
2. Measured vLLM bf16 output on this card at 50 concurrent: 8B ~2.7k tok/s, 12B (gemma-3) ~1.56k, 14B ~1.6-1.7k, 32B ~0.8k tok/s (one hosting-blog source, short runs); that is 40-60% of the bandwidth bound, so use `bound x ~0.5` as the starting assumption and push concurrency to 256+ for short sentences.
3. ESTIMATE: at ~50 output tokens per African-language sentence (Gemma 4 fertility ratios 1.5-2.5x English; Gemma 3 unmeasured), 4B / 12B / 27B Gemma-family on vLLM is roughly 100-200 / 60-100 / 20-50 sentences/s, i.e. ~1.4-2.8 / 2.8-4.6 / 5.6-14 h per million sentence-translations.
4. NLLB-600M-class on CTranslate2 fp16 has one published figure (AfriNLLB, A40, beam 3, ~3.5k tok/s, ~70 sent/s at 50 tok/s per sentence); no published CT2 numbers for NLLB 1.3B/3.3B or MADLAD, none vs HF for NLLB, and no int8 data, so those need a pilot measurement.
5. vLLM cannot serve MADLAD-400 (T5) or NLLB (M2M100): encoder-decoder support was removed in 2025 (T5 PR #11901 closed 2025-10-02, only Whisper native and BART/Florence-2 via plugin); plan MADLAD/NLLB on CTranslate2 (verify sm_120 build), HF, llama.cpp or TensorRT-LLM, and use vLLM only for Gemma/TranslateGemma.
