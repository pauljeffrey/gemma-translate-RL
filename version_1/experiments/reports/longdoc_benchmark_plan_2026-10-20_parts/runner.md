# Pseudo-reference runner (NLLB-200 3.3B + MADLAD-400) for the RTX 3090 box - 2026-10-20

Files (laptop `gemma_mt/`, synced to the rental `longdoc_bench/code/`):

| file | what |
|---|---|
| `scripts/longdoc_translate.py` | CLI (thin wrapper; REPLACES the old single-process script, whose library `eval/longdoc_pseudo.py` is untouched and still importable) |
| `gemma_mt/eval/longdoc_runner.py` | library: engines (stub / HF / CTranslate2), conversion helper, resumable `Runner`, progress, `--dry-run` |
| `tests/test_longdoc_runner.py` | 24 tests (laptop `tests/run_tests.py` and `python -m pytest`) |

Everything below marked **ESTIMATE** / **ASSUMED** was not measured. What was measured is listed in section 9.

## 1. What it does (one paragraph)

Reads the source set(s) (`docs.jsonl`: `{doc_id, text, sents?, para_starts?, bin, ...}`; several files allowed after `--docs`, the English short items too), and for every selected system and every language the system names (registry `codes`), translates every sentence of every document ON ITS OWN, then writes the sentence alignment per document. The units are the document's own `sents` / `para_starts` (else `canonical_sentences` / `paragraph_starts`), so the result re-joins exactly like `longdoc.translate_documents` (tested equal on the stub). A sentence longer than 700 characters is cut by `split_long_sentence`; a piece longer than the model's INPUT limit in tokens (NLLB 512, MADLAD 256: registry `max_input_tokens`, counted with the model tokenizer incl. special tokens and the `<2xx>` prefix) is cut again, smaller. Documents are processed in windows of 100 (`--window-docs`); inside a window the unique units are sorted by token length (longest first), batched by padded-token budget, translated, scattered back. Nothing needs the network at run time except the model download (`HF_HOME` respected; revisions pinned in the registry).

## 2. Usage on the RTX 3090 box (Python 3.12, `/venv/main` exists)

```bash
# --- 0. environment: a SEPARATE venv is recommended (keeps the torch of /venv/main untouched)
python3.12 -m venv /venv/ct2 && . /venv/ct2/bin/activate
pip install -U pip
pip install "ctranslate2==4.8.3" "sentencepiece==0.2.2" "transformers==5.20.2" "huggingface_hub==2.2.0" "tokenizers==0.23.3" "protobuf==7.36.2" "numpy==2.5.4" \
            "nvidia-cublas-cu12"                       # libcublas.so.12 for the PyPI ctranslate2 wheel (section 3)
pip install torch                                      # a CUDA build matching the driver (cu128 / cu130 index); >= 2.6 is REQUIRED to read NLLB's .bin shards
export HF_HOME=/data/hf PYTHONPATH=/path/to/gemma_mt   # the code dir (the folder that contains gemma_mt/ and scripts/)
# read-only Hub access only if the repos need it; both are public. Never write the token into logs.

# --- 1. download (optional: the runner / converter download on first use; this makes the 30-50 GB transfer visible and skips the GGUF files)
python - <<'E'
from huggingface_hub import snapshot_download
snapshot_download("facebook/nllb-200-3.3B", revision="1a07f7d195896b2114afcb79b7b57ab512e7b43e")                       # 17.58 GB (3 fp32 .bin shards)
snapshot_download("google/madlad400-3b-mt", revision="fa184c675da0b5c9e1c8694fccd4e12e2d422094",
                  allow_patterns=["*.safetensors", "*.json", "*.model"])                                             # 11.76 GB (skips model-q*.gguf, 3.9 GB)
# snapshot_download("google/madlad400-7b-mt-bt", revision="1259294dcefa2a4d8e0167ad418fdb47cfaeeca3", allow_patterns=["*.safetensors", "*.json", "*.model"])  # 33.19 GB
E

# --- 2. convert to CTranslate2 (int8_float16) - or let the runner do it on the fly into --ct2-cache; --convert-only does just this and exits
export CT2_CACHE=/data/ct2
python scripts/longdoc_translate.py --docs source_set/core/docs.jsonl --out refs --system nllb200-3.3b --system madlad400-3b-mt --convert-only

# --- 3. dry run: support table (system x language), counts, token estimates, time ASSUMPTIONS; writes nothing
python scripts/longdoc_translate.py --docs source_set/core/docs.jsonl source_set/short/items_b64.jsonl source_set/short/items_b256.jsonl \
        --out refs --system nllb200-3.3b --system madlad400-3b-mt --dry-run

# --- 4. first a small real check on the GPU (3 languages, 20 documents), look at tok/s and the outputs, then the full run
python scripts/longdoc_translate.py --docs source_set/core/docs.jsonl --out refs_smoke --system madlad400-3b-mt --langs yor hau ibo --limit-docs 20 --device cuda
nohup python scripts/longdoc_translate.py --docs source_set/core/docs.jsonl source_set/short/*.jsonl --out refs \
      --system nllb200-3.3b --system madlad400-3b-mt --backend auto --device cuda > refs.stdout 2>&1 &
tail -f refs/progress.log                       # one line every 20 s

# --- 5. stop / resume: Ctrl-C or `kill -TERM <pid of the runner>` finishes the current batch, flushes, exits 0. Run the SAME command again to resume.
#     Finished languages are skipped ("already done"), finished documents are not repeated, translated sentences come from the cache.
# --- 6. other systems / options
#   --system madlad400-7b-mt-bt        (33 GB download; 8.3 B parameters: fits the 24 GB card only quantised: ct2 int8 ~8.5 GB, or HF bf16 ~16.6 GB + small batches)
#   --hf-id Helsinki-NLP/opus-mt-en-ig --prompt-style marian --code ibo=ibo --langs ibo        (any seq2seq model; nllb|madlad|marian styles)
#   --backend hf --dtype bf16          (transformers fallback)      --greedy / --num-beams N       --allow-proxy      --exclude-langs ...
#   --ct2-dir /path/to/converted       (SYSTEM=PATH when several systems)     --revision SHA | SYSTEM=SHA      --retry-errors
```

Exit code is 0 after a completed or a gracefully stopped run, 2 on a usage error; an unexpected exception propagates (non-zero) and is resumable too.

## 3. pip packages, versions, CUDA 13

| package | verified here | recommended pin (Py 3.12) | needed for |
|---|---|---|---|
| ctranslate2 | **4.8.3** (PyPI, released 2026-10-13; wheel `cp312-manylinux_2_27_x86_64`, 40 MB) | `==4.8.3` | CT2 backend + converter |
| sentencepiece | 0.2.2 | `==0.2.2` | MADLAD / NLLB slow-tokenizer conversion, Marian |
| transformers | **5.20.2** (CT2 CPU + converter), 5.19.0 (HF CPU), 4.57.x (laptop, HF tiny model) | `==5.20.2` | tokenizers, HF backend, converter |
| torch | 2.14.1+cpu (venv_ct2), 2.14.0 (venv_gemma), 2.14 (laptop) | a CUDA build matching the box (>= 2.6) | converter, HF fallback; `.bin` loading of NLLB (transformers refuses `torch.load` below torch 2.6) |
| huggingface_hub | 2.2.0 (venv_ct2), 1.33.0 (venv_gemma) | `==2.2.0` | download |
| tokenizers | 0.23.3 / 0.23.2 | `==0.23.3` | fast tokenizers |
| protobuf | 7.36.2 | `==7.36.2` | sentencepiece-model conversion |
| numpy | 2.5.4 / 2.5.3 | `==2.5.4` | ctranslate2 dependency (also `pyyaml<7,>=5.3`) |
| accelerate | not installed, not needed (no `device_map`) | - | - |
| nvidia-cublas-cu12 | not installed in the venvs (CPU) | latest 12.x | libcublas.so.12 for the ctranslate2 wheel |

The runner itself imports only the repo code (`gemma_mt.eval.longdoc`, `longdoc_backends`, `longdoc_systems`, `languages`) plus the standard library; sacrebleu is NOT needed (all pytests of the runner pass without it).

**Does the PyPI CT2 wheel work with a CUDA 13 driver? Facts checked in the wheel (no GPU used): `libctranslate2-*.so` statically contains the CUDA runtime (cudart) and loads only `libcuda.so.1` (the driver) and `libcublas.so.12` at run time (`strings`/`readelf`; no cuDNN reference). It carries native code for sm_53, 60, 61, 70, 75, 80, **86** (+ PTX sm_86) (`cuobjdump --list-elf`), so the 3090 (sm_86) runs native kernels; there is no sm_90 / sm_100 / sm_120 SASS (those cards would JIT the PTX). The upstream docs say "install CUDA 12.x".** A CUDA 13 driver (R580+) runs CUDA-12 binaries (the driver is backward compatible), so the driver is not the obstacle; the missing piece on a CUDA-13-only machine is `libcublas.so.12`, which the wheel does not ship: `pip install nvidia-cublas-cu12` provides it. The runner then loads it itself from `site-packages/nvidia/cublas/lib` (`_preload_cuda12_blas`, RTLD_GLOBAL) and imports torch before ctranslate2 (a CUDA-12 torch exposes the same library), so no `LD_LIBRARY_PATH` is needed; if it still fails use `export LD_LIBRARY_PATH=$(python -c "import nvidia.cublas,os;print(os.path.join(os.path.dirname(nvidia.cublas.__file__),'lib'))")`. **I could not run a GPU, so "works on the 3090 with the CUDA 13 driver" is NOT verified**; the symptoms of a problem would be `Library libcublas.so.12 is not found` (fix above) or `CUDA driver version is insufficient for CUDA runtime version` (driver older than R525). **Fallback**: `--backend hf` (transformers, fp16 NLLB / bf16 MADLAD; any torch build for the box) works without ctranslate2; `--backend auto` falls back to it by itself when ctranslate2 cannot be imported or the engine cannot be created (log line `warn  CTranslate2 unavailable ...`). Slower (assumed 2-4x), same outputs format.

## 4. Models, pinned revisions, files (Hub file metadata read 2026-10-20; the registry revisions equal the current `main` heads)

| system (`--system`) | repo | pinned revision | files to download | size |
|---|---|---|---|---|
| `nllb200-3.3b` (= `nllb-200-3.3b`) | facebook/nllb-200-3.3B | `1a07f7d195896b2114afcb79b7b57ab512e7b43e` | `pytorch_model-0000{1,2,3}-of-00003.bin` (6.93 + 8.55 + 2.10 GB), `pytorch_model.bin.index.json`, `config.json`, `generation_config.json`, `sentencepiece.bpe.model` (4.9 MB), `tokenizer.json` (17 MB), `tokenizer_config.json`, `special_tokens_map.json` (no duplicate weight formats in the repo; only `.bin`, fp32) | 17.58 GB |
| `madlad400-3b-mt` | google/madlad400-3b-mt | `fa184c675da0b5c9e1c8694fccd4e12e2d422094` | `model.safetensors` (11.76 GB fp32), `spiece.model`, `tokenizer.json` (17 MB), `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, `config.json`, `generation_config.json`; SKIP `model-q2k/q3k/q4k.gguf` (3.9 GB) via `allow_patterns=["*.safetensors","*.json","*.model"]` | 11.76 GB |
| `madlad400-7b-mt` | google/madlad400-7b-mt | `1ff63ce7ddd571a69d45cad9672c553428419671` | 7 `model-0000N-of-00007.safetensors` (5.0 x6 + 3.3 GB) + `model.safetensors.index.json` + the tokenizer/config files; SKIP `model-q4k/q6k.gguf` (11.5 GB) | 33.19 GB |
| `madlad400-7b-mt-bt` | google/madlad400-7b-mt-bt | `1259294dcefa2a4d8e0167ad418fdb47cfaeeca3` | same layout as 7b-mt | 33.19 GB |

(`from_pretrained` / the converter fetch only the weights they use, so GGUF files are not downloaded even without `allow_patterns`; the explicit `snapshot_download` just makes it visible.) The revision is passed to the converter (`--revision`), to the HF backend and recorded in every output row (`revision`); override with `--revision SHA` or `SYSTEM=SHA`.

**CT2 conversion commands** (what `--convert-only` / the on-the-fly step runs; output `<CT2_CACHE>/<system>-<rev10>-int8_float16/`, written to `<dir>.tmp` and renamed when finished, `convert_info.json` marks it complete; a failure of `--copy_files` is retried once without it, the tokenizer is then read from the Hub):
```bash
ct2-transformers-converter --model facebook/nllb-200-3.3B --revision 1a07f7d195896b2114afcb79b7b57ab512e7b43e --quantization int8_float16 --low_cpu_mem_usage \
   --output_dir /data/ct2/nllb-200-3.3b-1a07f7d195-int8_float16 --copy_files tokenizer.json sentencepiece.bpe.model special_tokens_map.json tokenizer_config.json
ct2-transformers-converter --model google/madlad400-3b-mt --revision fa184c675da0b5c9e1c8694fccd4e12e2d422094 --quantization int8_float16 --low_cpu_mem_usage \
   --output_dir /data/ct2/madlad400-3b-mt-fa184c675d-int8_float16 --copy_files tokenizer.json spiece.model special_tokens_map.json tokenizer_config.json added_tokens.json
```
Resulting size ESTIMATE: about 1 byte per parameter: NLLB ~3.5 GB, MADLAD-3B ~3.2 GB, MADLAD-7B ~8 GB (the main plan uses 3.5 / 8 GB). The conversion runs on the CPU and loads the full fp32 checkpoint: ESTIMATE 20-30 GB RAM peak for NLLB-3.3B, 15-25 GB for MADLAD-3B, 40-50 GB for 7B (convert on a machine with the RAM, then copy the directory and pass `--ct2-dir`). The MADLAD config has `decoder_start_token_id=0` (`<unk>`) and `pad_token_id=1` (`<s>`); the CT2 T5 loader takes the decoder start from `config.decoder_start_token_id` (read in its source), and a tiny T5 with exactly this token layout gives the same greedy tokens in CT2 and HF (test), but the real MADLAD conversion was NOT run.

## 5. The exact CT2 calls and the tokenisation rules (`CT2Engine`, docstring in the code)

```python
translator = ctranslate2.Translator(model_dir, device="cuda", device_index=0, compute_type="int8_float16", inter_threads=1, intra_threads=0)   # cpu: int8 / float32
ids     = hf_tokenizer(inputs, truncation=True, max_length=512 | 256)["input_ids"]   # SentencePiece via the HF tokenizer, WITH special tokens
source  = [hf_tokenizer.convert_ids_to_tokens(i) for i in ids]                        # CT2 takes token strings
results = translator.translate_batch(source, target_prefix=[[tgt_code]] * n   # NLLB only (None for MADLAD / Marian)
              , beam_size=4, max_batch_size=n * longest_source_len, batch_type="tokens",   # = this batch's padded tokens, so CT2 does not re-split it
              max_decoding_length=min(512, 3 * longest + 16) (+1 for the NLLB prefix), num_hypotheses=1, return_scores=False,
              [repetition_penalty=..., no_repeat_ngram_size=...])                     # optional flags --repetition-penalty / --no-repeat-ngram-size
hyp  = results[i].hypotheses[0];  NLLB: hyp = hyp[1:]                                  # drop the forced target-language token
text = hf_tokenizer.decode(hf_tokenizer.convert_tokens_to_ids(hyp), skip_special_tokens=True)   # whitespace-collapsed
```
* **NLLB**: `AutoTokenizer(..., src_lang="eng_Latn")` gives `[eng_Latn] tokens </s>`; the target code (registry `codes`, FLORES-200 codes e.g. `yor_Latn`, `gaz_Latn` for `orm`, `plt_Latn` for `mlg`) is forced as target prefix (CT2) / `forced_bos_token_id` (HF). Input limit 512 tokens (model card: trained on <= 512), generation cap 512.
* **MADLAD**: the target is selected by the text prefix `"<2yo> "` (`format_inputs`: `<2{code}> {text}`); no target prefix; T5 decoder start token from the config; input limit 256 tokens (the prefix and `</s>` count), so 700-character pieces are re-split when they exceed 256 tokens; the registry `codes` are MADLAD's language tokens (`yo`, `ha`, `ig`, `rw`, ...; 44 of our 77).
* Special tokens are always removed from the output (`skip_special_tokens`); empty/whitespace units are not sent to the model (written empty, flagged `empty`).
* Over-long inputs: `split_long_sentence` at 700 characters, then token-aware re-splitting (up to 8 rounds, pieces halve), and as the last resort tokenizer truncation (counted as `truncated_units` in the summary). A translated sentence is the `norm_text` join of its pieces.
* Repetition/degeneration protection: beam 4, cap `3 x input + 16` tokens (<= 512), `sentence_flag` marks `loop` / `copy` / `short` / `long` / `empty` per sentence, documents with > 5% flagged sentences are `valid=false` (`--max-flagged`). `--repetition-penalty` / `--no-repeat-ngram-size` are available but OFF by default (they change the output; decide before the run).
* HF backend: `AutoModelForSeq2SeqLM`, `num_beams=4`, `dtype` auto = bf16 for MADLAD (T5 overflows in fp16), fp16 for NLLB / Marian, fp32 on the CPU; `forced_bos_token_id` for NLLB; the generation config's `max_length` is cleared and `max_new_tokens` set.

## 6. Batch budgets and throughput (ESTIMATES, to be replaced by the first minutes of the smoke run)

`--max-batch-tokens` = padded input tokens per batch (`n x longest`), `--max-batch-sents` 128. Defaults when 0 (ESTIMATES for 24 GB, beam 4): CT2 3B-class 12288, CT2 7B-class 6144, HF 3B-class 4096, HF 7B-class 1536. Memory ~ n x (source + 4 beams x target length); the units are processed longest first, so the first batch is the hardest: a CUDA/host out-of-memory error there halves the budget (log `oom  [...] batch budget halved to N`), the failing batch is split and retried, the budget grows back 25% after 100 clean batches (never above the size that failed). NLLB-3.3B int8_float16 weights ~3.5 GB leave ~20 GB; MADLAD-7B int8 ~8.5 GB.
Throughput ASSUMPTIONS used by `--dry-run` only (RTX 3090, beam 4, GENERATED tokens/s): CT2 3B-class 8,000, CT2 7B-class 4,000, HF 3B-class 2,500, HF 7B-class 1,200 (`--assume-tok-s` overrides; these are guesses, the true value can be 3x lower or 3x higher). Volume (real source set `source_set/core/docs.jsonl` on the rental, 500 documents): 26,905 sentences per language (26,485 unique), ~40.1 M estimated source tokens per system over its 44 languages (characters/4 + 2 per sentence; the main plan counts 44.8 M), generated ~1.5x that. At the assumed rates that is 2.1 h per 3B-class system and 4.2 h for a 7B (about 8.4 h for NLLB + MADLAD-3B + 7B-bt together) - the main plan's 0.3-1.6 GPU-h per system is the optimistic end. The English short items (b64 / b256) add little.

## 7. Output, progress, resume

`<out>/<system>/` (system = registry name, e.g. `nllb-200-3.3b`; the alias `nllb200-3.3b` is accepted on the command line):
* `<iso>.jsonl`: one row per document: `doc_id, iso, system, code, bin, src_sents, tgt_sents, para_starts, flags (sentence_flag or 'error'), flagged_share, valid, n_src_tokens, n_tgt_tokens (generated, per unit summed), backend (ct2|hf|stub), revision, ref_kind "pseudo", created`. Appended per window while running, rewritten in document order (atomic rename) when the language completes.
* `<iso>.refs.jsonl`: `{doc_id, iso, ref_kind "pseudo", ref_system, ref_sents, flags, flagged_share, valid}` = exactly what `longdoc.load_refdocs(docs.jsonl, refs, only_valid=...)` reads (tested). `<iso>.jsonl` has `tgt_sents` instead of `ref_sents`.
* `<iso>.summary.json` (atomic): docs, valid / invalid docs, sentences, flag counts, flagged_share, errors, units translated THIS run, tokens, seconds, avg tok/s, OOM halvings, truncated units, backend, revision.
* `sentence_cache.jsonl` (one per system): `{"i": iso, "k": sha1(system \0 iso \0 unit text), "t": translation, "s": src tokens, "g": generated tokens, "e": error}`; appended + flushed + fsync'ed every `--flush-sentences` units (1000) or `--flush-seconds` (60). Only the languages still to do are loaded into memory.
* `<out>/progress.log`, `<out>/progress.json`.

**progress.log** (flushed per line, also on stdout; interval `--progress-every 20`):
```
17:29:12 load  [madlad3b] madlad400-3b-mt: backend ct2, revision fa184c675d, input limit 256 tokens
17:29:44 [madlad3b] hau  doc  412/500  sent 18432/26110  1531 tok/s  lang 71%  all 38%  ETA 02:41
17:51:03 [madlad3b] hau  DONE  500 docs  26110 sent  flagged 1.8%  invalid docs 3  21 min  avg 1480 tok/s
17:29:00 skip  nllb-200-3.3b pcm  not supported (no code)
17:35:10 oom  [madlad3b] hau: batch of 61 x 200 tokens (12200 padded) failed; batch budget halved to 6100
17:36:02 error  [madlad3b] hau: sentence failed (RuntimeError: ...): 'The ...'
```
`doc` = documents finished in THIS language (a document counts when its window ends, 100 documents by default), `sent` = sentence slots whose units are translated (cache hits count at once), `tok/s` = generated tokens over the last interval, `lang`/`all` percent of sentence slots (language / all selected languages of all systems), `ETA hh:mm` = remaining source characters of all pending work / a smoothed (EMA 0.3) rate of the characters actually computed (cache hits and model loading excluded); it prints `--:--` until two intervals were measured; a system not yet started is scaled by the parameter ratio to the measured one (rough). The table at the start of the log (lines starting `# `) is the same as `--dry-run` plus the plan totals. `progress.json` (atomic rewrite at every update): `{updated, started, elapsed_s, state: running|done|stopped, current:{system,iso}, totals:{docs_done, docs_total, sent_done, sent_total, pct, eta_s, eta, tokens_generated, ...}, systems:{<system>:{<iso>:{status: pending|running|done|skipped, docs_done, docs_total, sent_done, sent_total, tok_s, started, finished, flagged_share}}}}`.

**Resume semantics.** A document is done when its row is in `<iso>.jsonl` (and its `src_sents` equal the current source set's: a changed document is translated again, only its changed sentences cost model calls). Partial documents resume from the cache. Torn last lines (kill -9) in either file are ignored/repaired. A finished language (summary present and rows match) is skipped. `--retry-errors` re-translates sentences that failed before. Stopping (SIGINT/SIGTERM) finishes the current batch (a 7B batch can take seconds), flushes and exits 0; a second signal raises immediately. Failure isolation: OOM -> halve and retry; other exception -> the batch is retried sentence by sentence; a sentence that still fails is written empty with flag `error` (document `valid=false` if > 5%), counted in the summary. Deterministic: units are ordered by (tokens desc, text), beam search has no seed. Deviation from the spec worth knowing: the length-sorted pool is the 100-document WINDOW (not all 500 documents) so that documents complete (and are appended to the file) continuously; `--window-docs 0` gives one pool per language; identical sentences in other windows are served from the cache.

## 8. Languages
`--langs` (space or comma separated; default the 77 in registry order), `--exclude-langs`, `--allow-proxy` (adds registry `proxy_codes`: not counted as support), `--limit-docs N` (first N documents), `--limit-sentences N` (whole documents until N sentences are reached). Unsupported languages are logged `skip  <system> <iso>  not supported (no code)`. Counts from the registry: NLLB-3.3B names 44, MADLAD 3B / 7B(-bt) name 44 each; the real dry-run output on the rental (500 documents) is:
```
system                         backend  selected  supported  skipped  documents  sentences  est. source tokens  est. generated tokens  est. GPU time (ASSUMED rate)
nllb-200-3.3b                  ct2            77         44       33      22000    1183820          40,126,812             60,190,218    2.09 h @ 8000 tok/s
madlad400-3b-mt                ct2            77         44       33      22000    1183820          40,126,812             60,190,218    2.09 h @ 8000 tok/s
madlad400-7b-mt-bt             ct2            77         44       33      22000    1183820          40,126,812             60,190,218    4.18 h @ 4000 tok/s
TOTAL                                                                     66000    3551460         120,380,436            180,570,654    8.36 h
```

## 9. Tests (what was actually run)

`tests/test_longdoc_runner.py`, 24 tests. Laptop (`../.venv-comet/bin/python tests/run_tests.py`, py3.11, torch 2.14, transformers 4.57, no ctranslate2): 20 passed, 4 skipped (opus-mt-en-ig not cached + 3 CT2). Rental, `/workspace/venv_gemma` (py3.12, transformers 5.19, no ctranslate2) `python -m pytest`: 21 passed, 3 skipped (the three real CT2 tests). Rental, the new CPU venv `longdoc_bench/venv_ct2` (ctranslate2 4.8.3, transformers 5.20.2, torch 2.14.1+cpu): **24 passed, 0 skipped**, including real CTranslate2 CPU runs. Coverage: stub end to end (paragraphs, a 1.9k-character sentence, equal to `translate_documents`, `load_refdocs` reads the refs), token-limit re-splitting, scatter correctness with length-sorted batches, longest-first and budget, resume after a simulated crash (identical output, no unit translated twice, third start does no model call), graceful stop and resume, torn lines, changed source, progress.log regexes, progress.json schema, ETA helper, unsupported-language skipping (+ `--allow-proxy`), OOM halving with a fake CUDA-OOM, failure isolation + `--retry-errors`, dry-run table, CLI, CT2 call shape (fake translator), a real SIGTERM to a child process (exit 0, resumable), HF CPU on a tiny random Marian (batch composition does not change beam-search results), **HF CPU with the real `Helsinki-NLP/opus-mt-en-ig` on 10 FLORES+ devtest sentences**, and real CTranslate2 CPU runs of converted tiny models: Marian (int8), a T5 with MADLAD's special-token layout and an M2M100 with NLLB's `[eng_Latn] ... </s>` layout and language-code prefix (float32, greedy): identical output to HF, and identical when batched differently. Also run by hand on the rental: the CLI with the real opus-mt-en-ig on 8 documents (FLORES sentences, one with two paragraphs) with `--backend hf` and with `--backend ct2` after `--convert-only` (28 s, fallback without `--copy_files` exercised): same shapes; the int8 CT2 translations differ from the fp32 HF ones in 11 of 14 sentences (quantisation; not a bug). NOT tested: any GPU, the real NLLB-3.3B / MADLAD weights, their conversion, fp16/bf16 behaviour, real throughput, the sm_86 kernels.

## 10. Open questions
1. **MADLAD with `int8_float16`**: T5 models can overflow in fp16 (that is why the HF default is bf16). If MADLAD output is empty/garbage or all `loop`-flagged in the smoke run, re-run with `--ct2-compute-type int8_bfloat16` (supported on Ampere) or `--backend hf`. Decide after the 20-document smoke run (look at flagged_share per language in `progress.log` DONE lines).
2. Real throughput and the batch budgets: unmeasured. The first `tok/s` lines tell; adjust `--max-batch-tokens` (OOM halving corrects only downwards at first).
3. **CUDA 13 + ctranslate2 wheel**: needs `libcublas.so.12` (`nvidia-cublas-cu12`); not verified on the box. Fallback HF.
4. RAM of the box for the conversion (estimates above); alternatively convert elsewhere and pass `--ct2-dir`.
5. 3B or 7B(-bt) for MADLAD (the main plan's cost table assumes either; 3B is the size the community uses). The runner treats them identically (`--system madlad400-3b-mt` or `madlad400-7b-mt-bt`).
6. Repetition control (`--no-repeat-ngram-size`, `--repetition-penalty`) is OFF; flagged sentences are recorded, never re-translated (the old `retry_fn` second pass of `longdoc_pseudo.py` is not part of this runner). Say if a second pass with different decoding for `loop` / `copy` flags is wanted.
7. `valid` threshold 5% flagged sentences per document (`--max-flagged`), unchanged from `longdoc_pseudo.py`.
8. NLLB outputs may not be published (licence question in the addendum section 5): the runner writes them to the `nllb-200-3.3b/` directory only; keep that directory out of any release build.
9. The `.bin` checkpoint of NLLB-3.3B needs `torch >= 2.6` for transformers to load it (conversion and HF backend); `/venv/main`'s torch version was not inspected.
10. Short English items: pass them as additional `--docs` files; `doc_id`s must be unique across the files (the runner refuses duplicates).
