# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pseudo-reference runner of the long-document benchmark (D-44): English source set -> NLLB-200 3.3B / MADLAD-400 translations, sentence by sentence, for up to
77 African languages, on ONE GPU box (RTX 3090, 24 GB), resumable, with a `tail -f`-able progress log.  CLI: `scripts/longdoc_translate.py` (see `--help`).

    python scripts/longdoc_translate.py --docs source_set/core/docs.jsonl --out refs_pseudo --system nllb200-3.3b --system madlad400-3b-mt --dry-run
    python scripts/longdoc_translate.py --docs source_set/core/docs.jsonl --out refs_pseudo --system madlad400-3b-mt --backend auto --device cuda

Design
  * Units.  Each document's sentences are `longdoc.canonical_sentences(text)` (or the document's own `sents` / `para_starts` when present).  A sentence longer than
    `--max-sentence-chars` (700) is cut by `longdoc.split_long_sentence` into pieces; a piece longer than the model's input limit in TOKENS (NLLB 512, MADLAD 256,
    from the registry `max_input_tokens`) is cut again, smaller.  The pieces are the translation units; a sentence is the joined (`norm_text`) translations of its
    pieces, exactly as `longdoc.translate_documents` does it (tested equal).
  * Throughput.  The documents of a language are processed in WINDOWS of `--window-docs` documents (default 100; 0 = the whole language).  Inside a window the unique
    units are sorted by token length (longest first, so an out-of-memory error shows up at once), cut into batches of at most `--max-batch-tokens` padded input tokens
    and `--max-batch-sents` sentences, translated, and scattered back.  After a window the finished documents are appended to `<out>/<system>/<iso>.jsonl`.
  * Resume.  Every translated unit goes into `<out>/<system>/sentence_cache.jsonl` (key = sha1(system, iso, unit text); appended + flushed + fsync'ed every
    `--flush-sentences` units or `--flush-seconds`).  On restart the cache and the finished documents are read, nothing already translated is translated again, a
    document is "done" when its row is in `<iso>.jsonl`.  When a language is complete `<iso>.jsonl` is rewritten in document order (atomic rename), plus
    `<iso>.refs.jsonl` (the format of `longdoc.load_refdocs`) and `<iso>.summary.json`.
  * Failures.  CUDA / host out-of-memory halves the batch budget and retries; any other error of a batch is retried sentence by sentence; a sentence that still fails
    is written as empty with the flag 'error' (counted).  SIGINT / SIGTERM stop gracefully after the current batch (flush, exit 0, resumable).
  * Backends.  CTranslate2 (preferred; int8_float16 on CUDA; converted on the fly with `ct2-transformers-converter` or given with --ct2-dir) or HF transformers.
    The exact CTranslate2 calls are documented at `CT2Engine`.
Heavy libraries (torch, transformers, ctranslate2) are imported inside the engines: this module imports and its tests run without them (stub engine).
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from gemma_mt.eval import longdoc as L
from gemma_mt.eval.longdoc_backends import format_inputs, target_code
from gemma_mt.languages import canonical_iso

ISO77 = ("yor hau ibo pcm efi twi ewe fon wol swh kin som orm amh tir lug lin zul xho afr sna nya sot tsn mlg ibb urh bem tum run ven tso nso ssw nbl sag tiv umb kmb "
         "mos kik luo kam bin dag oke shi dyu bam kbp bbj kon kea kab nus swc lua kqn lun iso loz wal ach tsc vmw kua seh nyn her yao sid ndc kri ish dua bas wes").split()

# registry-name aliases (the coordinator's spelling) and the short names used in the progress log
ALIASES = {"nllb200-3.3b": "nllb-200-3.3b", "nllb200-1.3b": "nllb-200-1.3b", "nllb200-distilled-600m": "nllb-200-distilled-600m", "nllb200-distilled-1.3b": "nllb-200-distilled-1.3b"}
SHORT_NAMES = {"nllb-200-3.3b": "nllb3b", "nllb-200-1.3b": "nllb1b", "nllb-200-distilled-1.3b": "nllbd1b", "nllb-200-distilled-600m": "nllbd600m", "madlad400-3b-mt": "madlad3b",
               "madlad400-7b-mt": "madlad7b", "madlad400-7b-mt-bt": "madlad7bbt", "madlad400-10b-mt": "madlad10b", "stub": "stub",
               "opus-mt-tc-bible-big-eng-mul": "opusmul", "opus-mt-tc-bible-big-eng-bnt": "opusbnt", "opus-mt-tc-bible-big-eng-afa": "opusafa"}
# files copied next to a converted CTranslate2 model so that the tokenizer can be loaded from the converted directory
CT2_COPY_FILES = {"nllb": ["tokenizer.json", "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer_config.json"],
                  "madlad": ["tokenizer.json", "spiece.model", "special_tokens_map.json", "tokenizer_config.json", "added_tokens.json"],
                  "marian": ["source.spm", "target.spm", "vocab.json", "tokenizer_config.json", "special_tokens_map.json"]}
# starting budgets (padded INPUT tokens per batch) for a 24 GB card - ESTIMATES, not measured; OOM halving corrects them at run time
DEFAULT_BATCH_TOKENS = {("ct2", "small"): 12288, ("ct2", "large"): 6144, ("hf", "small"): 4096, ("hf", "large"): 1536, ("stub", "small"): 4096, ("stub", "large"): 4096}
# assumed GENERATED tokens/s on an RTX 3090, beam 4 (ESTIMATES used only by --dry-run and the first ETA of nothing measured)
ASSUMED_TOK_S = {("ct2", "small"): 8000.0, ("ct2", "large"): 4000.0, ("hf", "small"): 2500.0, ("hf", "large"): 1200.0, ("stub", "small"): 1e6, ("stub", "large"): 1e6}


# ------------------------------------------------------------------------------------------------ small helpers
def now_hms(clock: Callable[[], float] = time.time) -> str:
    return time.strftime("%H:%M:%S", time.localtime(clock()))


def fmt_eta(sec: Optional[float]) -> str:
    if sec is None or sec != sec or sec < 0:
        return "--:--"
    m = int(round(sec / 60.0))
    return f"{m // 60:02d}:{m % 60:02d}"


def unit_key(system: str, iso: str, text: str) -> str:
    return hashlib.sha1(f"{system}\x00{iso}\x00{text}".encode("utf-8")).hexdigest()


def is_oom(exc: BaseException) -> bool:
    """CUDA / host out-of-memory in any of the spellings torch, CTranslate2 and the allocator use."""
    s = (type(exc).__name__ + " " + str(exc)).lower()
    return any(k in s for k in ("out of memory", "outofmemory", "cublas_status_alloc_failed", "bad_alloc", "alloc failed", "failed to allocate"))


def empty_cuda_cache() -> None:
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def fsync_flush(fh) -> None:
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except OSError:
        pass


def atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        fsync_flush(f)
    os.replace(tmp, path)


def atomic_write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
        fsync_flush(f)
    os.replace(tmp, path)
    return n


def read_jsonl_tolerant(path: str, repair: bool = False) -> Tuple[List[Dict[str, Any]], int]:
    """Rows of a JSONL file that may end in a torn line (a crash in the middle of a write). Returns (rows, number of unreadable lines).
    `repair`: truncate the file after the last good line, so that appending resumes on a clean line."""
    rows: List[Dict[str, Any]] = []
    bad = 0
    good_end = 0
    if not os.path.isfile(path):
        return rows, 0
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    while pos < len(data):
        nl = data.find(b"\n", pos)
        end = len(data) if nl < 0 else nl + 1
        raw = data[pos:end]
        if raw.strip():
            try:
                rows.append(json.loads(raw.decode("utf-8")))
                if nl >= 0:
                    good_end = end
                else:
                    good_end = -1                                   # complete JSON but no newline: keep, add the newline when appending
            except Exception:
                bad += 1
        elif nl >= 0:
            good_end = end
        pos = end
    if repair and bad and good_end >= 0:
        with open(path, "r+b") as f:
            f.truncate(good_end)
    return rows, bad


class AppendFile:
    """Append-only JSONL writer: write -> flush -> fsync on `flush()`. A missing final newline (torn write) is repaired when opening."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                last = f.read(1)
            if last != b"\n":
                with open(path, "ab") as f:
                    f.write(b"\n")
        self.fh = open(path, "a", encoding="utf-8")

    def write(self, row: Dict[str, Any]) -> None:
        self.fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        fsync_flush(self.fh)

    def close(self) -> None:
        if self.fh:
            try:
                self.flush()
            finally:
                self.fh.close()
                self.fh = None


# ------------------------------------------------------------------------------------------------ systems
def resolve_system(name: str, hf_id: str = "", prompt_style: str = "", codes: Optional[Dict[str, str]] = None, template: str = "", src_code: str = "",
                   adhoc_name: str = "", max_input_tokens: int = 512, revision: str = "", params_b: float = 0.0) -> Any:
    """A registry spec (`longdoc_systems.SYSTEMS`, aliases accepted), the stub, or an ad hoc spec for any seq2seq model (`hf_id` + `prompt_style` nllb|madlad|marian)."""
    if hf_id:
        if prompt_style not in ("nllb", "madlad", "marian"):
            raise ValueError("--hf-id needs --prompt-style nllb|madlad|marian")
        nm = adhoc_name or re.sub(r"[^A-Za-z0-9._-]+", "-", hf_id.split("/")[-1]).lower()
        return SimpleNamespace(name=nm, hf_id=hf_id, revision=revision, family=prompt_style, prompt_style=prompt_style, template=template, hf_id_template="",
                               backends=("hf", "ctranslate2"), params_b=params_b, max_input_tokens=max_input_tokens,
                               src_code=src_code or {"nllb": "eng_Latn", "madlad": "en", "marian": "en"}[prompt_style], codes=dict(codes or {}), proxy_codes={},
                               weights_licence="unknown", licence_class="unknown", adhoc=True)
    if name == "stub":
        return SimpleNamespace(name="stub", hf_id="", revision="", family="stub", prompt_style="nllb", template="", hf_id_template="", backends=("stub",), params_b=0.0,
                               max_input_tokens=512, src_code="eng_Latn", codes={i: i for i in ISO77}, proxy_codes={}, weights_licence="n/a", licence_class="permissive")
    from gemma_mt.eval import longdoc_systems as S

    name = ALIASES.get(name, name)
    if name not in S.SYSTEMS:
        raise KeyError(f"unknown system {name!r}; registered: {', '.join(sorted(S.SYSTEMS))}")
    return S.SYSTEMS[name]


def short_name(spec: Any) -> str:
    n = getattr(spec, "name", "x")
    return SHORT_NAMES.get(n, re.sub(r"[^a-z0-9]", "", n.lower())[:10] or "sys")


def size_class(spec: Any) -> str:
    return "large" if float(getattr(spec, "params_b", 0) or 0) > 5.0 else "small"


# ------------------------------------------------------------------------------------------------ engines
class Engine:
    """One loaded model of one system. `format` builds the model input strings, `count_tokens` their length, `translate_inputs` translates ONE batch."""

    backend = "?"

    def __init__(self, spec: Any, allow_proxy: bool = False):
        self.spec, self.allow_proxy = spec, allow_proxy
        self.name = spec.name
        self.max_input_tokens = int(getattr(spec, "max_input_tokens", 512) or 512)
        self.revision = getattr(spec, "revision", "") or ""

    def format(self, texts: Sequence[str], iso: str) -> List[str]:
        return format_inputs(self.spec, list(texts), iso, self.allow_proxy)

    def count_tokens(self, inputs: Sequence[str]) -> List[int]:             # pragma: no cover - abstract
        raise NotImplementedError

    def translate_inputs(self, inputs: Sequence[str], iso: str, budget_tokens: int) -> List[Tuple[str, int]]:   # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:
        pass


class StubEngine(Engine):
    """No model: every word gets the language code as suffix and the order is reversed (deterministic; never a copy of the English). `on_batch(inputs, iso)` is a hook
    (tests: record calls, raise a simulated crash, raise a CUDA-OOM-like error). Tokens = whitespace words (+1)."""

    backend = "stub"

    def __init__(self, spec: Any, allow_proxy: bool = False, on_batch: Optional[Callable[[List[str], str], None]] = None, max_input_tokens: Optional[int] = None):
        super().__init__(spec, allow_proxy)
        self.on_batch = on_batch
        if max_input_tokens:
            self.max_input_tokens = max_input_tokens
        self.calls: List[List[str]] = []

    def count_tokens(self, inputs):
        return [len(s.split()) + 1 for s in inputs]

    def translate_inputs(self, inputs, iso, budget_tokens):
        inputs = list(inputs)
        if self.on_batch:
            self.on_batch(inputs, iso)
        self.calls.append(inputs)
        out = []
        for s in inputs:
            words = s.split()
            out.append((" ".join(f"{w}{iso}" for w in reversed(words)), len(words) + 1))
        return out


def _preload_cuda12_blas() -> None:
    """The PyPI ctranslate2 wheel loads libcublas.so.12 at run time (cudart is linked in; the driver comes from the machine). With a CUDA-13-only environment the 12.x
    library is missing: `pip install nvidia-cublas-cu12` provides it, and this function loads it from site-packages/nvidia/cublas/lib so that no LD_LIBRARY_PATH is needed."""
    for base in list(sys.path):
        for lib in sorted(glob.glob(os.path.join(base, "nvidia", "cublas", "lib", "libcublas*.so.12"))):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _tokenizer_kwargs(spec: Any) -> Dict[str, Any]:
    return {"src_lang": spec.src_code} if spec.prompt_style == "nllb" else {}


class HFEngine(Engine):
    """transformers seq2seq (`AutoModelForSeq2SeqLM`): NLLB (tokenizer src_lang=eng_Latn, forced_bos_token_id = target code), MADLAD (T5, `<2xx> ` in the text), Marian."""

    backend = "hf"

    def __init__(self, spec: Any, device: str = "cuda", dtype: str = "auto", num_beams: int = 4, revision: str = "", allow_proxy: bool = False, max_new_tokens: int = 512,
                 out_len_factor: float = 3.0, repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0, model_path: str = ""):
        super().__init__(spec, allow_proxy)
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if spec.prompt_style not in ("nllb", "madlad", "marian"):
            raise ValueError(f"HFEngine: unsupported prompt style {spec.prompt_style!r}")
        self.torch, self.device, self.beams = torch, device, num_beams
        self.cap, self.factor, self.rep, self.nrep = max_new_tokens, out_len_factor, repetition_penalty, no_repeat_ngram_size
        self.revision = revision or self.revision
        src = model_path or spec.hf_id
        rev = None if model_path else (self.revision or None)
        if dtype == "auto":                                    # T5 (MADLAD) overflows in fp16: bf16 on a GPU; NLLB / Marian fp16 on a GPU; fp32 on the CPU
            dtype = "fp32" if device == "cpu" else ("bf16" if spec.prompt_style == "madlad" else "fp16")
        self.dtype_name = dtype
        td = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        self.tok = AutoTokenizer.from_pretrained(src, revision=rev, **_tokenizer_kwargs(spec))
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(src, revision=rev, dtype=td)
        except TypeError:                                      # transformers < 4.56
            self.model = AutoModelForSeq2SeqLM.from_pretrained(src, revision=rev, torch_dtype=td)
        self.model.eval().to(device)
        try:
            self.model.generation_config.max_length = None             # we always pass max_new_tokens: avoids the 'both max_new_tokens and max_length' notice on every batch
        except Exception:
            pass

    def count_tokens(self, inputs):
        return [len(x) for x in self.tok(list(inputs), add_special_tokens=True)["input_ids"]]

    def translate_inputs(self, inputs, iso, budget_tokens):
        torch = self.torch
        enc = self.tok(list(inputs), return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_tokens, return_token_type_ids=False).to(self.device)
        cap = min(self.cap, int(enc["input_ids"].shape[1] * self.factor) + 16)
        kw: Dict[str, Any] = {"num_beams": self.beams, "max_new_tokens": cap + (1 if self.spec.prompt_style == "nllb" else 0), "do_sample": False}   # +1: the forced language token
        if self.spec.prompt_style == "nllb":
            kw["forced_bos_token_id"] = self.tok.convert_tokens_to_ids(target_code(self.spec, iso, self.allow_proxy))
        if self.rep and self.rep != 1.0:
            kw["repetition_penalty"] = self.rep
        if self.nrep:
            kw["no_repeat_ngram_size"] = self.nrep
        with torch.no_grad():
            gen = self.model.generate(**enc, **kw)
        pad = self.tok.pad_token_id
        ntok = (gen[:, 1:] != pad).sum(dim=1).tolist()          # generated tokens without the decoder start token and the padding
        return [(" ".join(t.split()), int(n)) for t, n in zip(self.tok.batch_decode(gen, skip_special_tokens=True), ntok)]

    def close(self):
        self.model = None
        empty_cuda_cache()


class CT2Engine(Engine):
    """CTranslate2 (`pip install ctranslate2`; the PyPI wheel needs libcublas.so.12 for a GPU). `model_dir` = a directory made by `ct2-transformers-converter`.

    Calls, exactly as made here (one `translate_inputs` call = one batch chosen by the runner):
        translator = ctranslate2.Translator(model_dir, device="cuda", device_index=0, compute_type="int8_float16", inter_threads=1, intra_threads=0)
        ids        = hf_tokenizer(inputs, truncation=True, max_length=<512 NLLB | 256 MADLAD>)["input_ids"]     # SentencePiece through the HF tokenizer; adds the
                                                                                                         # special tokens: NLLB [eng_Latn] ... </s>, T5/MADLAD ... </s>
        source     = [hf_tokenizer.convert_ids_to_tokens(i) for i in ids]                                 # CTranslate2 takes token STRINGS
        results    = translator.translate_batch(source, target_prefix=[[<tgt code>]] * n  (NLLB only; None for MADLAD / Marian), beam_size=4,
                                                max_batch_size=n * longest_source (padded tokens of THIS batch: CTranslate2 does not split it further),
                                                batch_type="tokens", max_decoding_length=min(512, 3 * longest_source + 16 (+1 for the prefix)),
                                                num_hypotheses=1, return_scores=False, repetition_penalty=<opt>, no_repeat_ngram_size=<opt>)
        hyp        = results[i].hypotheses[0]; NLLB: hyp = hyp[1:]  (drop the target-language token that was forced as prefix)
        text       = hf_tokenizer.decode(hf_tokenizer.convert_tokens_to_ids(hyp), skip_special_tokens=True)
    MADLAD: the `<2xx> ` prefix is part of the input text (`format_inputs`); the T5 decoder starts with the model's decoder_start_token (`<unk>`, id 0; the converter reads
    it from config.json). Input limits: NLLB 512 tokens, MADLAD 256 tokens (registry `max_input_tokens`); over-long sentences are split before this point."""

    backend = "ct2"

    def __init__(self, spec: Any, model_dir: str, device: str = "cuda", compute_type: str = "int8_float16", num_beams: int = 4, revision: str = "", allow_proxy: bool = False,
                 max_new_tokens: int = 512, out_len_factor: float = 3.0, repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0, tokenizer_src: str = "",
                 device_index: int = 0):
        super().__init__(spec, allow_proxy)
        if device == "cuda":
            _preload_cuda12_blas()
        try:
            import torch  # noqa: F401  (imported first: it makes libcublas.so.12 of a CUDA-12 torch visible to ctranslate2)
        except Exception:
            pass
        import ctranslate2
        from transformers import AutoTokenizer

        if device == "cpu" and ("float16" in compute_type or "bfloat16" in compute_type):          # no half-precision kernels on the CPU
            compute_type = "int8" if compute_type.startswith("int8") else "float32"
        self.model_dir, self.beams, self.cap, self.factor = model_dir, num_beams, max_new_tokens, out_len_factor
        self.rep, self.nrep, self.compute_type = repetition_penalty, no_repeat_ngram_size, compute_type
        self.revision = revision or self.revision
        self.translator = ctranslate2.Translator(model_dir, device=device, device_index=device_index, compute_type=compute_type, inter_threads=1, intra_threads=0)
        src = tokenizer_src or (model_dir if os.path.isfile(os.path.join(model_dir, "tokenizer_config.json")) else spec.hf_id)
        rev = None if src == model_dir else (self.revision or None)
        self.tok = AutoTokenizer.from_pretrained(src, revision=rev, **_tokenizer_kwargs(spec))
        self.version = ctranslate2.__version__

    def count_tokens(self, inputs):
        return [len(x) for x in self.tok(list(inputs), add_special_tokens=True)["input_ids"]]

    def translate_inputs(self, inputs, iso, budget_tokens):
        ids = self.tok(list(inputs), truncation=True, max_length=self.max_input_tokens)["input_ids"]
        source = [self.tok.convert_ids_to_tokens(i) for i in ids]
        longest = max(len(s) for s in source)
        nllb = self.spec.prompt_style == "nllb"
        prefix = [[target_code(self.spec, iso, self.allow_proxy)]] * len(source) if nllb else None
        kw: Dict[str, Any] = {}
        if self.rep and self.rep != 1.0:
            kw["repetition_penalty"] = self.rep
        if self.nrep:
            kw["no_repeat_ngram_size"] = self.nrep
        res = self.translator.translate_batch(source, target_prefix=prefix, beam_size=self.beams, max_batch_size=max(1, longest * len(source)), batch_type="tokens",
                                              max_decoding_length=min(self.cap, int(self.factor * longest) + 16) + (1 if nllb else 0), num_hypotheses=1, return_scores=False, **kw)
        out = []
        for r in res:
            hyp = r.hypotheses[0]
            if nllb:
                hyp = hyp[1:]
            out.append((" ".join(self.tok.decode(self.tok.convert_tokens_to_ids(hyp), skip_special_tokens=True).split()), len(hyp)))
        return out

    def close(self):
        self.translator = None
        empty_cuda_cache()


def ct2_cache_default() -> str:
    return os.environ.get("CT2_CACHE") or os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "ct2_models")


def ct2_dir_for(spec: Any, revision: str, quantization: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{spec.name}-{(revision or 'main')[:10]}-{quantization}")


def ensure_ct2_model(spec: Any, revision: str, quantization: str, cache_dir: str, log: Callable[[str], None] = print, low_cpu_mem: bool = True) -> str:
    """Convert `spec.hf_id@revision` to CTranslate2 (`ct2-transformers-converter --quantization <q> --copy_files <tokenizer files>`) into `cache_dir` unless it is already
    there (a `convert_info.json` marks a finished conversion; conversion goes into `<dir>.tmp` and is renamed at the end). Returns the model directory. Needs torch +
    transformers (the converter loads the checkpoint on the CPU: ~2x the fp32 size in RAM for the .bin / safetensors checkpoints without --low_cpu_mem_usage)."""
    out = ct2_dir_for(spec, revision, quantization, cache_dir)
    marker = os.path.join(out, "convert_info.json")
    if os.path.isfile(marker):
        log(f"convert  {spec.name}: using the converted model {out}")
        return out
    os.makedirs(cache_dir, exist_ok=True)
    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    cmd = [sys.executable, "-c", "from ctranslate2.converters.transformers import main; main()", "--model", spec.hf_id, "--output_dir", tmp, "--quantization", quantization, "--force"]
    if revision:
        cmd += ["--revision", revision]
    if low_cpu_mem:
        cmd += ["--low_cpu_mem_usage"]
    files = CT2_COPY_FILES.get(spec.prompt_style, [])
    if files:
        cmd += ["--copy_files"] + files
    log(f"convert  {spec.name}: {spec.hf_id}@{(revision or 'main')[:10]} -> {out} ({quantization}); this takes minutes and reads the full checkpoint: "
        "ct2-transformers-converter " + " ".join(cmd[3:]))
    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0 and files:
        # a tokenizer file that does not exist in the repo makes --copy_files fail: retry once without it (the tokenizer is then loaded from the Hub)
        log("convert  failed with --copy_files; retrying without it: " + p.stdout.strip().splitlines()[-1][:200])
        shutil.rmtree(tmp, ignore_errors=True)
        p = subprocess.run(cmd[: cmd.index("--copy_files")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError("ct2 conversion failed:\n" + p.stdout[-3000:])
    with open(os.path.join(tmp, "convert_info.json"), "w", encoding="utf-8") as f:
        json.dump({"source": spec.hf_id, "revision": revision, "quantization": quantization, "seconds": round(time.time() - t0, 1),
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    log(f"convert  {spec.name}: done in {time.time() - t0:.0f} s -> {out}")
    return out


# ------------------------------------------------------------------------------------------------ progress
class Eta:
    """Smoothed work rate (source characters per second of the computed work, cache hits excluded). No ETA before `min_samples` measured intervals."""

    def __init__(self, alpha: float = 0.3, min_samples: int = 2):
        self.alpha, self.min_samples, self.rate, self.n = alpha, min_samples, None, 0

    def update(self, work: float, dt: float) -> None:
        if dt > 0 and work > 0:
            r = work / dt
            self.rate = r if self.rate is None else self.alpha * r + (1 - self.alpha) * self.rate
            self.n += 1

    def eta(self, remaining: float, scale: float = 1.0) -> Optional[float]:
        if self.rate is None or self.n < self.min_samples or self.rate <= 0:
            return None
        return remaining / (self.rate * scale)


class Progress:
    """`progress.log` (one line per update, flushed; also echoed) and `progress.json` (atomic rewrite)."""

    def __init__(self, out_dir: Optional[str], every: float = 20.0, clock: Callable[[], float] = time.time, echo: Optional[Callable[[str], None]] = print):
        self.out_dir, self.every, self.clock, self.echo = out_dir, every, clock, echo
        self.log_fh = None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            self.log_fh = open(os.path.join(out_dir, "progress.log"), "a", encoding="utf-8")
        self.state: Dict[str, Dict[str, Dict[str, Any]]] = {}       # system -> iso -> state
        self.meta: Dict[str, Dict[str, Any]] = {}                    # system -> {"short", "params_b"}
        self.eta = Eta()
        self.t_start = clock()
        self.last_tick = clock()
        self.last_tick_tokens = 0
        self.last_tick_work = 0.0
        self.tokens_total = 0
        self.computed_chars = 0.0
        self.current: Tuple[str, str] = ("", "")
        self.run_state = "running"
        self.lines: List[str] = []

    # --- plain log lines
    def log(self, msg: str) -> None:
        line = f"{now_hms(self.clock)} {msg}"
        self.lines.append(line)
        if self.log_fh:
            self.log_fh.write(line + "\n")
            self.log_fh.flush()
        if self.echo:
            self.echo(line)

    def raw(self, text: str) -> None:
        """Unprefixed lines (plan table)."""
        for ln in text.splitlines():
            if self.log_fh:
                self.log_fh.write("# " + ln + "\n")
            if self.echo:
                self.echo(ln)
        if self.log_fh:
            self.log_fh.flush()

    # --- state
    def register(self, system: str, iso: str, docs_total: int, sent_total: int, chars_total: int, status: str = "pending") -> None:
        self.state.setdefault(system, {})[iso] = {"status": status, "docs_done": 0, "docs_total": docs_total, "sent_done": 0, "sent_total": sent_total, "tok_s": 0.0,
                                                  "started": None, "finished": None, "flagged_share": None, "chars_total": chars_total, "chars_done": 0.0, "tokens": 0}

    def totals(self) -> Dict[str, Any]:
        active = [s for sy in self.state.values() for s in sy.values() if s["status"] != "skipped"]
        sd, st = sum(min(int(s["sent_done"] + 1e-6), s["sent_total"]) for s in active), sum(s["sent_total"] for s in active)
        dd, dt = sum(s["docs_done"] for s in active), sum(s["docs_total"] for s in active)
        return {"docs_done": dd, "docs_total": dt, "sent_done": sd, "sent_total": st, "pct": (100.0 * sd / st) if st else 100.0}

    def remaining_eta(self) -> Optional[float]:
        cur_sys = self.current[0]
        cur_params = float(self.meta.get(cur_sys, {}).get("params_b") or 0)
        total = 0.0
        for sy, d in self.state.items():
            pb = float(self.meta.get(sy, {}).get("params_b") or 0)
            scale = (cur_params / pb) if (cur_params and pb and sy != cur_sys) else 1.0     # a bigger model than the one measured is slower (rough: ratio of parameters)
            for s in d.values():
                if s["status"] in ("pending", "running"):
                    total += max(s["chars_total"] - s["chars_done"], 0.0) / max(scale, 1e-6)
        return self.eta.eta(total)

    def line(self, system: str, iso: str, tok_s: float) -> str:
        s = self.state[system][iso]
        tt = self.totals()
        sdone = min(int(s["sent_done"] + 1e-6), s["sent_total"])
        lp = 100.0 * sdone / s["sent_total"] if s["sent_total"] else 100.0
        return (f"[{self.meta[system]['short']}] {iso}  doc  {s['docs_done']:>3}/{s['docs_total']}  sent {sdone}/{s['sent_total']}  {tok_s:.0f} tok/s  "
                f"lang {lp:.0f}%  all {tt['pct']:.0f}%  ETA {fmt_eta(self.remaining_eta())}")

    def tick(self, system: str, iso: str, force: bool = False) -> None:
        t = self.clock()
        dt = t - self.last_tick
        if not force and dt < self.every:
            return
        dtok = self.tokens_total - self.last_tick_tokens
        dwork = self.computed_chars - self.last_tick_work
        self.eta.update(dwork, dt)
        tok_s = dtok / dt if dt > 0 else 0.0
        self.state[system][iso]["tok_s"] = round(tok_s, 1)
        self.last_tick, self.last_tick_tokens, self.last_tick_work = t, self.tokens_total, self.computed_chars
        self.log(self.line(system, iso, tok_s))
        self.write_json()

    def write_json(self) -> None:
        if not self.out_dir:
            return
        tt = self.totals()
        eta = self.remaining_eta()
        doc = {"updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.clock())), "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t_start)),
               "elapsed_s": round(self.clock() - self.t_start, 1), "state": self.run_state, "current": {"system": self.current[0], "iso": self.current[1]},
               "totals": {**tt, "eta_s": None if eta is None else round(eta, 1), "eta": fmt_eta(eta), "tokens_generated": self.tokens_total,
                          "tok_s_smoothed_chars_per_s": None if self.eta.rate is None else round(self.eta.rate, 1)},
               "systems": {sy: {iso: {k: (min(int(v + 1e-6), s["sent_total"]) if k == "sent_done" else v) for k, v in s.items() if k not in ("chars_total", "chars_done", "tokens")}
                                  for iso, s in d.items()} for sy, d in self.state.items()}}
        atomic_write_text(os.path.join(self.out_dir, "progress.json"), json.dumps(doc, ensure_ascii=False, indent=1))

    def close(self) -> None:
        if self.log_fh:
            self.log_fh.close()
            self.log_fh = None


# ------------------------------------------------------------------------------------------------ planning
def load_source_docs(paths: Sequence[str], limit_docs: int = 0, limit_sentences: int = 0) -> List[Dict[str, Any]]:
    """Rows of one or several source-set JSONL files -> [{doc_id, text, sents, para_starts, bin, domain}]. A document brings its own `sents` / `para_starts` when it has
    them (else `canonical_sentences` / `paragraph_starts`, as `longdoc.translate_documents`). `limit_docs`: the first N documents; `limit_sentences`: whole documents until
    the sentence count reaches N (documents are never cut, the alignment unit is the document)."""
    docs: List[Dict[str, Any]] = []
    seen = set()
    for p in paths:
        for d in L.read_jsonl(p):
            if d["doc_id"] in seen:
                raise ValueError(f"duplicate doc_id {d['doc_id']!r} in {p}")
            seen.add(d["doc_id"])
            sents = list(d["sents"]) if d.get("sents") else L.canonical_sentences(d["text"])
            starts = list(d.get("para_starts") or L.paragraph_starts(d["text"], sents))
            docs.append({"doc_id": d["doc_id"], "text": d["text"], "sents": sents, "para_starts": starts, "bin": d.get("bin", 0), "domain": d.get("domain", "")})
    if limit_docs:
        docs = docs[:limit_docs]
    if limit_sentences:
        out, n = [], 0
        for d in docs:
            if n >= limit_sentences:
                break
            out.append(d)
            n += len(d["sents"])
        docs = out
    return docs


def doc_stats(docs: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    slots = sum(len(d["sents"]) for d in docs)
    uniq = {s for d in docs for s in d["sents"]}
    chars = sum(len(s) for d in docs for s in d["sents"])
    return {"docs": len(docs), "sentences": slots, "unique_sentences": len(uniq), "chars": chars, "est_src_tokens": int(chars / 4.0 + 2 * slots)}


def parse_map_arg(values: Optional[Sequence[str]], what: str) -> Dict[str, str]:
    """`SYSTEM=VALUE` list -> dict; a bare VALUE is stored under the key ''."""
    out: Dict[str, str] = {}
    for v in values or []:
        if "=" in v:
            k, val = v.split("=", 1)
            out[ALIASES.get(k, k)] = val
        else:
            out[""] = v
    return out


def split_langs(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not values:
        return None
    out = []
    for v in values:
        for x in re.split(r"[,\s]+", v.strip()):
            if x:
                out.append(canonical_iso(x) or x)
    return out


def support_table(specs: Sequence[Any], langs: Sequence[str], docs_stats: Dict[str, int], allow_proxy: bool, assume_tok_s: Dict[str, float], backends: Dict[str, str]) -> str:
    rows = ["system                         backend  selected  supported  skipped  documents  sentences  est. source tokens  est. generated tokens  est. GPU time (ASSUMED rate)"]
    tot_docs = tot_sent = tot_tok = 0
    tot_s = 0.0
    for sp in specs:
        sup = [i for i in langs if target_code(sp, i, allow_proxy) is not None]
        n = len(sup)
        src_tok = docs_stats["est_src_tokens"] * n
        gen_tok = int(src_tok * 1.5)
        rate = assume_tok_s.get(sp.name, 1000.0)
        sec = gen_tok / rate if rate else 0
        tot_docs += docs_stats["docs"] * n
        tot_sent += docs_stats["sentences"] * n
        tot_tok += src_tok
        tot_s += sec
        rows.append(f"{sp.name:<30} {backends.get(sp.name, '?'):<8} {len(langs):>8}  {n:>9}  {len(langs) - n:>7}  {docs_stats['docs'] * n:>9}  {docs_stats['sentences'] * n:>9}  {src_tok:>18,}  "
                    f"{gen_tok:>21,}  {sec / 3600:>6.2f} h @ {rate:.0f} tok/s")
    rows.append(f"{'TOTAL':<30} {'':<8} {'':>8}  {'':>9}  {'':>7}  {tot_docs:>9}  {tot_sent:>9}  {tot_tok:>18,}  {int(tot_tok * 1.5):>21,}  {tot_s / 3600:>6.2f} h")
    return "\n".join(rows)


# ------------------------------------------------------------------------------------------------ the runner
class StopRequested(Exception):
    pass


class LangRun:
    """State of one (system, language) while it is processed."""

    def __init__(self, system: str, iso: str, code: str, docs: List[Dict[str, Any]]):
        self.system, self.iso, self.code, self.docs = system, iso, code, docs
        self.cache: Dict[str, Tuple[str, int, int, int]] = {}          # key -> (text, n_src_tokens, n_tgt_tokens, error)
        self.done_rows: Dict[str, Dict[str, Any]] = {}                  # doc_id -> row
        self.tokens = 0
        self.seconds = 0.0
        self.n_errors = 0
        self.n_oom = 0
        self.n_truncated = 0
        self.n_computed = 0                                             # units translated in this run (cache hits excluded)
        self.seconds_batches = 0.0                                      # wall time inside batches (this run)


class Runner:
    def __init__(self, docs: Sequence[Dict[str, Any]], out_dir: str, specs: Sequence[Any], *, engine_factory: Callable[[Any], Engine], langs: Optional[Sequence[str]] = None,
                 exclude_langs: Sequence[str] = (), allow_proxy: bool = False, window_docs: int = 100, max_batch_tokens: int = 0, max_batch_sents: int = 128,
                 max_sentence_chars: int = 700, flush_sentences: int = 1000, flush_seconds: float = 60.0, progress_every: float = 20.0, max_flagged: float = 0.05,
                 retry_errors: bool = False, clock: Callable[[], float] = time.time, echo: Optional[Callable[[str], None]] = print, install_signals: bool = True,
                 backend_label: Optional[Dict[str, str]] = None, min_budget: int = 128, size_class_of: Optional[Callable[[Any], str]] = None):
        self.docs = list(docs)
        self.out_dir = out_dir
        self.specs = list(specs)
        self.engine_factory = engine_factory
        self.langs = list(langs) if langs else list(ISO77)
        self.exclude = {canonical_iso(x) or x for x in exclude_langs}
        self.allow_proxy = allow_proxy
        self.window_docs = window_docs
        self.budget_cfg = max_batch_tokens
        self.max_items = max_batch_sents
        self.max_chars = max_sentence_chars
        self.flush_n, self.flush_s = flush_sentences, flush_seconds
        self.max_flagged = max_flagged
        self.retry_errors = retry_errors
        self.clock = clock
        self.install_signals = install_signals
        self.min_budget = min_budget
        self.backend_label = backend_label or {}
        self.size_class_of = size_class_of or size_class
        self.prog = Progress(out_dir, progress_every, clock, echo)
        self.stop = False
        self._old_handlers: Dict[int, Any] = {}
        self.stats = doc_stats(self.docs)
        self.budget = 0
        self.budget_ceiling = 0
        self.ok_streak = 0

    # --- signals
    def _on_signal(self, signum, frame):
        if self.stop:                                                   # second signal: stop at once
            raise KeyboardInterrupt
        self.stop = True
        self.prog.log(f"stop  signal {signum} received: finishing the current batch, flushing, exiting (resumable)")

    def _install(self) -> None:
        if not self.install_signals or threading.current_thread() is not threading.main_thread():
            return
        for s in (signal.SIGINT, signal.SIGTERM):
            self._old_handlers[s] = signal.signal(s, self._on_signal)

    def _restore(self) -> None:
        for s, h in self._old_handlers.items():
            signal.signal(s, h)
        self._old_handlers = {}

    # --- planning
    def planned_langs(self, spec: Any) -> Tuple[List[str], List[str]]:
        sup, skip = [], []
        for iso in self.langs:
            if iso in self.exclude:
                continue
            (sup if target_code(spec, iso, self.allow_proxy) is not None else skip).append(iso)
        return sup, skip

    # --- main loop
    def run(self) -> int:
        p = self.prog
        os.makedirs(self.out_dir, exist_ok=True)
        plan: List[Tuple[Any, List[str]]] = []
        for spec in self.specs:
            sup, skip = self.planned_langs(spec)
            p.meta[spec.name] = {"short": short_name(spec), "params_b": getattr(spec, "params_b", 0.0)}
            for iso in skip:
                p.register(spec.name, iso, 0, 0, 0, status="skipped")
                p.log(f"skip  {spec.name} {iso}  not supported (no code)")
            for iso in sup:
                p.register(spec.name, iso, len(self.docs), self.stats["sentences"], self.stats["chars"])
            plan.append((spec, sup))
        p.write_json()
        self._install()
        try:
            for spec, sup in plan:
                if self.stop or not sup:
                    continue
                self._run_system(spec, sup)
        finally:
            self._restore()
        p.run_state = "stopped" if self.stop else "done"
        p.write_json()
        p.log(f"{'STOPPED (resumable: run the same command again)' if self.stop else 'ALL DONE'}  {p.totals()['docs_done']}/{p.totals()['docs_total']} documents")
        p.close()
        return 0

    def _run_system(self, spec: Any, langs: List[str]) -> None:
        p = self.prog
        sdir = os.path.join(self.out_dir, spec.name)
        os.makedirs(sdir, exist_ok=True)
        # which languages are already complete?
        todo = []
        for iso in langs:
            if self._language_complete(sdir, spec, iso):
                st = p.state[spec.name][iso]
                st.update(status="done", docs_done=st["docs_total"], sent_done=st["sent_total"], chars_done=st["chars_total"])
                s = self._read_summary(sdir, iso)
                st["flagged_share"] = s.get("flagged_share") if s else None
                p.log(f"skip  {spec.name} {iso}  already done ({st['docs_total']} docs)")
            else:
                todo.append(iso)
        p.write_json()
        if not todo:
            return
        cache_path = os.path.join(sdir, "sentence_cache.jsonl")
        engine = self.engine_factory(spec)
        engine.allow_proxy = self.allow_proxy
        p.log(f"load  [{p.meta[spec.name]['short']}] {spec.name}: backend {engine.backend}, revision {engine.revision[:10] or '-'}, input limit {engine.max_input_tokens} tokens")
        sc = self.size_class_of(spec)
        self.budget_cfg_eff = self.budget_cfg or DEFAULT_BATCH_TOKENS.get((engine.backend, sc), 4096)
        self.budget = self.budget_cfg_eff
        self.budget_ceiling = self.budget
        cache_w = AppendFile(cache_path)
        try:
            for iso in todo:
                if self.stop:
                    break
                p.current = (spec.name, iso)
                self._run_language(spec, engine, iso, sdir, cache_path, cache_w)
        finally:
            cache_w.close()
            engine.close()

    # --- files of one language
    @staticmethod
    def _read_summary(sdir: str, iso: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(sdir, f"{iso}.summary.json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _language_complete(self, sdir: str, spec: Any, iso: str) -> bool:
        s = self._read_summary(sdir, iso)
        if not s or s.get("docs") != len(self.docs):
            return False
        rows, _ = read_jsonl_tolerant(os.path.join(sdir, f"{iso}.jsonl"))
        if len(rows) != len(self.docs):
            return False
        if self.retry_errors and any("error" in (r.get("flags") or []) for r in rows):
            return False
        return all(r.get("doc_id") == d["doc_id"] and r.get("src_sents") == d["sents"] and r.get("system") == spec.name for r, d in zip(rows, self.docs))

    def _load_cache(self, cache_path: str, iso: str, lr: LangRun) -> None:
        if not os.path.isfile(cache_path):
            return
        prefix = f'{{"i": "{iso}",'
        bad = 0
        with open(cache_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith(prefix):
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                if r.get("e") and self.retry_errors:
                    continue
                lr.cache[r["k"]] = (r["t"], int(r.get("s", 0)), int(r.get("g", 0)), int(r.get("e", 0)))
        if bad:
            self.prog.log(f"cache  {lr.system} {iso}: {bad} unreadable line(s) ignored (torn write)")

    # --- one language
    def _run_language(self, spec: Any, engine: Engine, iso: str, sdir: str, cache_path: str, cache_w: AppendFile) -> None:
        p = self.prog
        st = p.state[spec.name][iso]
        code = target_code(spec, iso, self.allow_proxy)
        lr = LangRun(spec.name, iso, code, self.docs)
        rows_path = os.path.join(sdir, f"{iso}.jsonl")
        rows, bad = read_jsonl_tolerant(rows_path, repair=True)
        if bad:
            p.log(f"resume  {spec.name} {iso}: {bad} torn line(s) removed from {os.path.basename(rows_path)}")
        by_id = {d["doc_id"]: d for d in self.docs}
        for r in rows:
            d = by_id.get(r.get("doc_id"))
            if d is None or r.get("src_sents") != d["sents"] or r.get("system") != spec.name or (self.retry_errors and "error" in (r.get("flags") or [])):
                continue                                               # stale / other source set / errors to retry: translated again (units come from the cache)
            lr.done_rows[r["doc_id"]] = r
        self._load_cache(cache_path, iso, lr)
        rows_w = AppendFile(rows_path)
        if len(lr.done_rows) != len(rows):                            # rewrite the file without the dropped rows
            rows_w.close()
            atomic_write_jsonl(rows_path, lr.done_rows.values())
            rows_w = AppendFile(rows_path)
        st["status"] = "running"
        st["started"] = time.strftime("%H:%M:%S", time.localtime(self.clock()))
        t_lang0 = self.clock()
        slots_by_doc = {d["doc_id"]: len(d["sents"]) for d in self.docs}
        chars_by_doc = {d["doc_id"]: sum(len(s) for s in d["sents"]) for d in self.docs}
        st["docs_done"] = len(lr.done_rows)
        st["sent_done"] = sum(slots_by_doc[i] for i in lr.done_rows)
        st["chars_done"] = float(sum(chars_by_doc[i] for i in lr.done_rows))
        if lr.done_rows:
            p.log(f"resume  [{p.meta[spec.name]['short']}] {iso}: {len(lr.done_rows)}/{len(self.docs)} documents already done, {len(lr.cache)} cached units")
        pending = [d for d in self.docs if d["doc_id"] not in lr.done_rows]
        p.tick(spec.name, iso, force=True)                              # first line of the language (also opens the measuring interval)
        win = self.window_docs if self.window_docs and self.window_docs > 0 else max(len(pending), 1)
        try:
            for w0 in range(0, len(pending), win):
                if self.stop:
                    break
                self._run_window(spec, engine, lr, pending[w0: w0 + win], cache_w, rows_w, st)
        finally:
            cache_w.flush()
            rows_w.close()
        lr.seconds += self.clock() - t_lang0
        if len(lr.done_rows) == len(self.docs):
            self._finalize(spec, engine, lr, sdir, st, rows_path)
        else:
            p.tick(spec.name, iso, force=True)

    def _run_window(self, spec: Any, engine: Engine, lr: LangRun, wdocs: List[Dict[str, Any]], cache_w: AppendFile, rows_w: AppendFile, st: Dict[str, Any]) -> None:
        p = self.prog
        iso = lr.iso
        # 1. pieces of every unique sentence (token-fitted)
        sents = list(dict.fromkeys(s for d in wdocs for s in d["sents"]))
        pieces_of, in_tok = self._fit_pieces(engine, iso, sents, lr)
        units = list(dict.fromkeys(u for s in sents for u in pieces_of[s]))
        mult: Dict[str, int] = collections.Counter(u for d in wdocs for s in d["sents"] for u in pieces_of[s])              # occurrences of a unit
        slot_w: Dict[str, float] = collections.defaultdict(float)                                                            # sentence slots a unit completes (1/pieces each)
        for d in wdocs:
            for s in d["sents"]:
                for u in pieces_of[s]:
                    slot_w[u] += 1.0 / len(pieces_of[s])
        unit_chars = {u: len(u) for u in units}
        keys = {u: unit_key(lr.system, iso, u) for u in units}
        # 2. empty units need no model call; cached units are credited at once
        todo = []
        for u in units:
            k = keys[u]
            if k in lr.cache:
                st["sent_done"] += slot_w[u]
                st["chars_done"] += unit_chars[u] * mult[u]
            elif not u.strip():
                lr.cache[k] = ("", 0, 0, 0)
                st["sent_done"] += slot_w[u]
            else:
                todo.append(u)
        inputs_of = dict(zip(todo, engine.format(todo, iso))) if todo else {}
        todo.sort(key=lambda u: (-in_tok[u], u))                            # longest first; ties by text: deterministic
        lens = [in_tok[u] for u in todo]
        pos, n = 0, len(todo)
        pending_rows: List[Dict[str, Any]] = []
        last_flush = self.clock()
        while pos < n:
            if self.stop:
                break
            per = max(1, min(self.max_items, self.budget // max(lens[pos], 1)))
            batch = todo[pos: pos + per]
            t0 = self.clock()
            results = self._run_batch(engine, iso, lr, batch, [inputs_of[u] for u in batch], [in_tok[u] for u in batch])
            lr.seconds_batches += self.clock() - t0
            for u, (text, ntok, err) in zip(batch, results):
                k = keys[u]
                lr.cache[k] = (text, in_tok[u], ntok, err)
                pending_rows.append({"i": iso, "k": k, "t": text, "s": in_tok[u], "g": ntok, "e": err})
                st["sent_done"] += slot_w[u]
                st["chars_done"] += unit_chars[u] * mult[u]
                p.computed_chars += unit_chars[u] * mult[u]
                lr.tokens += ntok
                p.tokens_total += ntok
                lr.n_computed += 1
                lr.n_errors += err
            pos += len(batch)
            if len(pending_rows) >= self.flush_n or self.clock() - last_flush >= self.flush_s:
                for r in pending_rows:
                    cache_w.write(r)
                cache_w.flush()
                pending_rows, last_flush = [], self.clock()
            p.tick(lr.system, iso)
        for r in pending_rows:
            cache_w.write(r)
        cache_w.flush()
        # 3. finished documents of the window -> rows
        n_new = 0
        for d in wdocs:
            row = self._doc_row(spec, engine, lr, d, pieces_of, keys_fn=lambda u: unit_key(lr.system, iso, u))
            if row is None:
                continue
            lr.done_rows[d["doc_id"]] = row
            rows_w.write(row)
            n_new += 1
        rows_w.flush()
        st["docs_done"] = len(lr.done_rows)
        # slots of incomplete documents of the window stay counted as they are (units done); the progress is by units
        p.tick(lr.system, iso)

    def _fit_pieces(self, engine: Engine, iso: str, sents: List[str], lr: LangRun) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """pieces of every sentence (<= max_chars characters AND <= the model's input limit in tokens) and the input-token length of every piece."""
        pieces_of = {s: L.split_long_sentence(s, self.max_chars) for s in sents}
        limit = engine.max_input_tokens
        in_tok: Dict[str, int] = {}
        for it in range(8):
            need = list(dict.fromkeys(u for ps in pieces_of.values() for u in ps if u not in in_tok))
            if need:
                for u, t in zip(need, engine.count_tokens(engine.format(need, iso))):
                    in_tok[u] = t
            over = {u for ps in pieces_of.values() for u in ps if in_tok[u] > limit and len(u) > 40}
            if not over:
                break
            for s, ps in list(pieces_of.items()):
                if any(u in over for u in ps):
                    new = []
                    for u in ps:
                        new += L.split_long_sentence(u, max(30, int(len(u) * 0.55))) if u in over else [u]
                    pieces_of[s] = new
        lr.n_truncated += sum(1 for ps in pieces_of.values() for u in ps if in_tok.get(u, 0) > limit)
        return pieces_of, in_tok

    def _run_batch(self, engine: Engine, iso: str, lr: LangRun, units: List[str], inputs: List[str], lens: List[int]) -> List[Tuple[str, int, int]]:
        """Translate one batch -> [(text, generated tokens, error)] with OOM halving and sentence-by-sentence isolation of failures."""
        p = self.prog
        try:
            res = engine.translate_inputs(inputs, iso, self.budget)
            if len(res) != len(inputs):
                raise RuntimeError(f"engine returned {len(res)} outputs for {len(inputs)} inputs")
            self.ok_streak += 1
            if self.ok_streak >= 100 and self.budget < self.budget_ceiling:
                self.budget = min(max(self.budget + 1, int(self.budget * 1.25)), self.budget_ceiling)
                self.ok_streak = 0
            return [(t, n, 0) for t, n in res]
        except Exception as e:                                           # BaseException (KeyboardInterrupt, simulated crash) propagates
            if is_oom(e):
                self.ok_streak = 0
                lr.n_oom += 1
                padded = max(lens) * len(lens)
                if len(inputs) > 1:
                    self.budget_ceiling = max(1, min(self.budget_ceiling, padded - 1))
                    self.budget = max(1, min(self.budget, padded) // 2)             # strictly smaller batches, so the retry cannot loop
                    p.log(f"oom  [{p.meta[lr.system]['short']}] {iso}: batch of {len(inputs)} x {max(lens)} tokens ({padded} padded) failed; batch budget halved to {self.budget}")
                    empty_cuda_cache()
                    out: List[Tuple[str, int, int]] = []
                    i = 0
                    while i < len(inputs):
                        per = max(1, min(self.max_items, self.budget // max(lens[i], 1)))
                        out += self._run_batch(engine, iso, lr, units[i: i + per], inputs[i: i + per], lens[i: i + per])
                        i += per
                    return out
                p.log(f"oom  [{p.meta[lr.system]['short']}] {iso}: a single sentence ({lens[0]} tokens) ran out of memory")
                empty_cuda_cache()
            if len(inputs) > 1:
                p.log(f"retry  [{p.meta[lr.system]['short']}] {iso}: batch of {len(inputs)} failed ({type(e).__name__}: {str(e)[:100]}); retrying sentence by sentence")
                out = []
                for i in range(len(inputs)):
                    out += self._run_batch(engine, iso, lr, units[i: i + 1], inputs[i: i + 1], lens[i: i + 1])
                return out
            p.log(f"error  [{p.meta[lr.system]['short']}] {iso}: sentence failed ({type(e).__name__}: {str(e)[:100]}): {units[0][:60]!r}")
            return [("", 0, 1)]

    def _doc_row(self, spec: Any, engine: Engine, lr: LangRun, d: Dict[str, Any], pieces_of: Dict[str, List[str]], keys_fn) -> Optional[Dict[str, Any]]:
        tgt, flags, ns, ng = [], [], 0, 0
        for s in d["sents"]:
            parts, err = [], 0
            for u in pieces_of[s]:
                c = lr.cache.get(keys_fn(u))
                if c is None:
                    return None
                parts.append(c[0])
                ns += c[1]
                ng += c[2]
                err |= c[3]
            t = L.norm_text(" ".join(x.strip() for x in parts))
            tgt.append(t)
            flags.append("error" if err else L.sentence_flag(s, t))
        n = max(len(flags), 1)
        share = sum(1 for f in flags if f) / n
        return {"doc_id": d["doc_id"], "iso": lr.iso, "system": spec.name, "code": lr.code, "bin": d.get("bin", 0), "src_sents": d["sents"], "tgt_sents": tgt,
                "para_starts": d["para_starts"], "flags": flags, "flagged_share": round(share, 4), "valid": share <= self.max_flagged, "n_src_tokens": ns, "n_tgt_tokens": ng,
                "backend": engine.backend, "revision": engine.revision, "ref_kind": "pseudo", "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.clock()))}

    def _finalize(self, spec: Any, engine: Engine, lr: LangRun, sdir: str, st: Dict[str, Any], rows_path: str) -> None:
        p = self.prog
        iso = lr.iso
        rows = [lr.done_rows[d["doc_id"]] for d in self.docs]
        atomic_write_jsonl(rows_path, rows)                                                   # document order, atomic
        refs = [{"doc_id": r["doc_id"], "iso": iso, "ref_kind": "pseudo", "ref_system": spec.name, "ref_sents": r["tgt_sents"], "flags": r["flags"],
                 "flagged_share": r["flagged_share"], "valid": r["valid"]} for r in rows]
        atomic_write_jsonl(os.path.join(sdir, f"{iso}.refs.jsonl"), refs)
        n_sent = sum(len(r["tgt_sents"]) for r in rows)
        flag_counts: Dict[str, int] = collections.Counter(f for r in rows for f in r["flags"] if f)
        flagged = sum(flag_counts.values())
        invalid = sum(1 for r in rows if not r["valid"])
        secs = max(lr.seconds_batches, 1e-9)
        avg = lr.tokens / secs if lr.n_computed else 0.0
        summary = {"system": spec.name, "iso": iso, "code": lr.code, "docs": len(rows), "valid_docs": len(rows) - invalid, "invalid_docs": invalid, "sentences": n_sent,
                   "flags": dict(flag_counts), "flagged_share": flagged / max(n_sent, 1), "errors": flag_counts.get("error", 0), "units_translated_this_run": lr.n_computed,
                   "tokens_generated_this_run": lr.tokens, "seconds_this_run": round(lr.seconds, 1), "avg_tok_s_this_run": round(avg, 1), "oom_halvings": lr.n_oom,
                   "truncated_units": lr.n_truncated, "backend": engine.backend, "revision": engine.revision, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        atomic_write_text(os.path.join(sdir, f"{iso}.summary.json"), json.dumps(summary, ensure_ascii=False, indent=1))
        st.update(status="done", finished=time.strftime("%H:%M:%S", time.localtime(self.clock())), docs_done=len(rows), sent_done=st["sent_total"], flagged_share=round(summary["flagged_share"], 4),
                  tok_s=round(avg, 1), chars_done=st["chars_total"])
        p.log(f"[{p.meta[lr.system]['short']}] {iso}  DONE  {len(rows)} docs  {n_sent} sent  flagged {100 * summary['flagged_share']:.1f}%  invalid docs {invalid}  "
              f"{lr.seconds / 60:.0f} min  avg {avg:.0f} tok/s")
        p.write_json()
        lr.cache.clear()


# ------------------------------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", nargs="+", required=True, help="source-set JSONL file(s): rows {doc_id, text, sents?, para_starts?, bin?, ...} (the short English items too)")
    ap.add_argument("--out", required=True, help="output directory: <out>/<system>/<iso>.jsonl, progress.log, progress.json")
    ap.add_argument("--system", action="append", default=None, help="registry name (nllb200-3.3b, madlad400-3b-mt, madlad400-7b-mt, madlad400-7b-mt-bt, opus-mt-tc-bible-big-eng-mul, ...) "
                    "or 'stub'; repeat for several systems (run one after the other)")
    ap.add_argument("--hf-id", default=None, help="ad hoc seq2seq model instead of a registry system (needs --prompt-style and --code)")
    ap.add_argument("--prompt-style", default=None, choices=["nllb", "madlad", "marian"])
    ap.add_argument("--adhoc-name", default="", help="name of the ad hoc system (output directory); default: the repo name")
    ap.add_argument("--code", action="append", default=[], help="ad hoc codes iso=code (e.g. ibo=ibo for opus-mt-en-ig, yor=yor_Latn for NLLB-style); repeatable")
    ap.add_argument("--template", default="", help="ad hoc input template, e.g. '>>{tgt_code}<< {text}' for multi-target Marian")
    ap.add_argument("--backend", default="auto", choices=["auto", "ct2", "hf"])
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda when torch sees one, else cpu)")
    ap.add_argument("--ct2-dir", action="append", default=[], help="converted CTranslate2 model dir: PATH (one system) or SYSTEM=PATH; default: converted on the fly into --ct2-cache")
    ap.add_argument("--ct2-cache", default=None, help="where on-the-fly conversions go (default $CT2_CACHE or $HF_HOME/ct2_models)")
    ap.add_argument("--ct2-quantization", default="int8_float16", help="weight quantisation of the conversion")
    ap.add_argument("--ct2-compute-type", default=None, help="compute type at load time (default: the quantisation of the conversion; int8_bfloat16 / bfloat16 if int8_float16 misbehaves on T5)")
    ap.add_argument("--convert-only", action="store_true", help="convert the selected systems to CTranslate2 (and tell where), then exit")
    ap.add_argument("--revision", action="append", default=[], help="override the registry's pinned revision: SHA (one system) or SYSTEM=SHA")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"], help="HF backend (auto: bf16 for MADLAD / T5, fp16 for NLLB / Marian on a GPU)")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--greedy", action="store_true", help="beam size 1")
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--max-batch-tokens", type=int, default=0, help="padded input tokens per batch (0: backend / model size default for 24 GB, an ESTIMATE; OOM halves it)")
    ap.add_argument("--max-batch-sents", type=int, default=128)
    ap.add_argument("--max-sentence-chars", type=int, default=700)
    ap.add_argument("--max-new-tokens", type=int, default=512, help="cap of generated tokens per sentence")
    ap.add_argument("--window-docs", type=int, default=100, help="documents per processing window (0 = a whole language at once)")
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--exclude-langs", nargs="*", default=None)
    ap.add_argument("--allow-proxy", action="store_true", help="also translate languages for which the system only has a proxy code (registry proxy_codes)")
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--limit-sentences", type=int, default=0, help="whole documents until this many sentences are reached (smoke tests)")
    ap.add_argument("--flush-sentences", type=int, default=1000)
    ap.add_argument("--flush-seconds", type=float, default=60.0)
    ap.add_argument("--progress-every", type=float, default=20.0)
    ap.add_argument("--max-flagged", type=float, default=0.05, help="a document with a larger share of flagged sentences is valid=false")
    ap.add_argument("--retry-errors", action="store_true", help="translate sentences again that failed in an earlier run (flag 'error')")
    ap.add_argument("--dry-run", action="store_true", help="print the plan (support table, counts, token and time estimates) and exit")
    ap.add_argument("--assume-tok-s", type=float, default=0.0, help="dry-run only: assumed generated tokens/s for the time estimate (default: a guess per backend and model size)")
    return ap


def _pick(m: Dict[str, str], name: str, n_systems: int) -> str:
    if name in m:
        return m[name]
    if "" in m and n_systems == 1:
        return m[""]
    return ""


def main(argv: Optional[List[str]] = None, engine_factory: Optional[Callable[[Any], Engine]] = None, echo: Optional[Callable[[str], None]] = print) -> int:
    a = build_parser().parse_args(argv)
    if a.hf_id:
        codes = dict(x.split("=", 1) for x in a.code)
        specs = [resolve_system("", hf_id=a.hf_id, prompt_style=a.prompt_style or "", codes=codes, template=a.template, adhoc_name=a.adhoc_name)]
    else:
        specs = [resolve_system(n) for n in (a.system or [])]
    if not specs:
        print("give --system NAME (repeatable) or --hf-id", file=sys.stderr)
        return 2
    docs = load_source_docs(a.docs, a.limit_docs, a.limit_sentences)
    revs, ct2dirs = parse_map_arg(a.revision, "revision"), parse_map_arg(a.ct2_dir, "ct2-dir")
    langs = split_langs(a.langs) or list(ISO77)
    excl = split_langs(a.exclude_langs) or []
    # backend choice per system
    try:
        import ctranslate2  # noqa: F401
        have_ct2 = True
    except Exception:
        have_ct2 = False
    device = a.device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    backends: Dict[str, str] = {}
    for sp in specs:
        if sp.name == "stub":
            backends[sp.name] = "stub"
        elif a.backend == "ct2" or (a.backend == "auto" and have_ct2 and ("ctranslate2" in sp.backends or _pick(ct2dirs, sp.name, len(specs)))):
            backends[sp.name] = "ct2"
        else:
            backends[sp.name] = "hf"
    cache_dir = a.ct2_cache or ct2_cache_default()
    stats = doc_stats(docs)
    assume = {sp.name: a.assume_tok_s or ASSUMED_TOK_S.get((backends[sp.name], size_class(sp)), 1000.0) for sp in specs}
    runner_langs = [i for i in langs if i not in {canonical_iso(x) or x for x in excl}]
    if a.dry_run:
        out = [f"DRY RUN  {len(docs)} documents from {', '.join(a.docs)}; languages selected: {len(runner_langs)}; allow-proxy: {a.allow_proxy}; device {device}; beams {1 if a.greedy else a.num_beams}",
               support_table(specs, runner_langs, stats, a.allow_proxy, assume, backends),
               "est. source tokens: characters/4 + 2 per sentence (English; the real count comes from the tokenizer at run time). Generated tokens ASSUMED 1.5 x source tokens "
               "(African-language fertility); the tok/s per system are ASSUMED guesses (RTX 3090, beam 4), NOT measured: the real rate appears in progress.log after the first minute "
               "(the true time can be 3x lower or 3x higher).",
               "unique sentences per language: %d of %d (identical sentences are translated once)" % (stats["unique_sentences"], stats["sentences"])]
        for sp in specs:
            sup = [i for i in runner_langs if target_code(sp, i, a.allow_proxy) is not None]
            skip = [i for i in runner_langs if i not in sup]
            out.append(f"{sp.name}: supported ({len(sup)}): {' '.join(sup)}")
            out.append(f"{sp.name}: skipped, no code ({len(skip)}): {' '.join(skip)}")
        for ln in out:
            print(ln)
        return 0
    if a.convert_only:
        for sp in specs:
            if sp.name == "stub":
                continue
            rev = _pick(revs, sp.name, len(specs)) or sp.revision
            ensure_ct2_model(sp, rev, a.ct2_quantization, cache_dir, log=print)
        return 0

    def factory(spec: Any) -> Engine:
        beams = 1 if a.greedy else a.num_beams
        rev = _pick(revs, spec.name, len(specs)) or getattr(spec, "revision", "")
        if spec.name == "stub":
            return StubEngine(spec)
        kw = dict(allow_proxy=a.allow_proxy, num_beams=beams, revision=rev, max_new_tokens=a.max_new_tokens, repetition_penalty=a.repetition_penalty,
                  no_repeat_ngram_size=a.no_repeat_ngram_size)
        be = backends[spec.name]
        if be == "ct2":
            try:
                d = _pick(ct2dirs, spec.name, len(specs))
                if not d:
                    d = ensure_ct2_model(spec, rev, a.ct2_quantization, cache_dir, log=runner.prog.log)
                return CT2Engine(spec, d, device=device, compute_type=a.ct2_compute_type or a.ct2_quantization, **kw)
            except Exception as e:
                if a.backend == "ct2":
                    raise
                runner.prog.log(f"warn  CTranslate2 unavailable for {spec.name} ({type(e).__name__}: {str(e)[:200]}); falling back to the HF backend")
                backends[spec.name] = "hf"
        return HFEngine(spec, device=device, dtype=a.dtype, **kw)

    runner = Runner(docs, a.out, specs, engine_factory=engine_factory or factory, langs=langs, exclude_langs=excl, allow_proxy=a.allow_proxy, window_docs=a.window_docs,
                    max_batch_tokens=a.max_batch_tokens, max_batch_sents=a.max_batch_sents, max_sentence_chars=a.max_sentence_chars, flush_sentences=a.flush_sentences,
                    flush_seconds=a.flush_seconds, progress_every=a.progress_every, max_flagged=a.max_flagged, retry_errors=a.retry_errors, echo=echo)
    runner.prog.raw("PLAN\n" + support_table(specs, runner_langs, stats, a.allow_proxy, assume, backends))
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
