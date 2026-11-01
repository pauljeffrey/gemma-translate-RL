# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model-agnostic SENTENCE translators for the long-document benchmark (D-44): the bake-off and the pseudo-reference translation use the same objects.

    tr = make_translator(spec, backend="hf" | "vllm" | "ctranslate2" | "stub", device="cpu", ...)
    tr.translate(["An English sentence.", ...], "yor")  ->  ["...", ...]
    fn = tr.fn_for("yor")                                ->  callable for longdoc.translate_documents

`spec` is a `longdoc_systems.SystemSpec` (or any object with the same attributes): `prompt_style` selects how the input is built
  nllb      tokenizer(src_lang=spec.src_code) + forced BOS = the target code token             (NLLB-200; HF seq2seq, CTranslate2)
  madlad    "<2{code}> {text}" as the input of a T5 model                                       (MADLAD-400; HF seq2seq, CTranslate2)
  chat      the model's chat template around `spec.template` (placeholders {src_lang} {tgt_lang} {tgt_code} {text})  (Gemma, TranslateGemma, Aya, Qwen)
  t5prompt  `spec.template` as a plain prompt for a seq2seq / causal model                    (Aya-101, mT0)
  seamless  SeamlessM4Tv2ForTextToText with tgt_lang=code                                       (SeamlessM4T v2)
  marian    one Marian model per language, hf_id_template.format(code=code)                    (Opus-MT)
Language codes come from `spec.codes[iso]` (explicit support) or `spec.proxy_codes[iso]` when `allow_proxy=True`. Heavy libraries (torch, transformers, vllm,
ctranslate2) are imported inside the classes, so this module imports on a machine without them; the input builders are pure functions and unit-tested.
Decoding: beam search for seq2seq models (default 4 beams, as the NLLB / MADLAD papers), greedy for LLMs; `max_new_tokens` follows the input length.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from gemma_mt.languages import LANGUAGES, canonical_iso


# ------------------------------------------------------------------------------------------------ input builders (pure)
def target_code(spec: Any, iso: str, allow_proxy: bool = False) -> Optional[str]:
    """The system's code for `iso`, or None when the system does not (explicitly) support the language."""
    iso = canonical_iso(iso) or iso
    codes = getattr(spec, "codes", {}) or {}
    if iso in codes:
        return codes[iso]
    if allow_proxy:
        return (getattr(spec, "proxy_codes", {}) or {}).get(iso)
    return None


# language names written into LLM prompts when our short fixed name is ambiguous for a general model (the benchmark's instruction names stay in languages.py)
PROMPT_LANG_NAMES = {"pcm": "Nigerian Pidgin", "wes": "Cameroon Pidgin", "swc": "Congo Swahili", "twi": "Twi", "mlg": "Malagasy", "orm": "Oromo", "kon": "Kikongo",
                     "nso": "Northern Sotho (Sepedi)", "run": "Kirundi", "luo": "Dholuo", "mos": "Mossi (Moore)", "kab": "Kabyle", "kea": "Kabuverdianu (Cape Verdean Creole)",
                     "kri": "Krio", "tsn": "Setswana", "ven": "Venda (Tshivenda)", "tso": "Xitsonga", "ssw": "siSwati", "nbl": "South Ndebele (isiNdebele)", "sag": "Sango"}


def prompt_language_name(iso: str) -> str:
    c = canonical_iso(iso)
    return PROMPT_LANG_NAMES.get(c, LANGUAGES[c].name) if c else iso


def is_encoder_decoder(spec: Any) -> bool:
    """T5-style (MADLAD, mT0, Aya-101), NLLB, Seamless, Marian: seq2seq. BLOOMZ, LLaMAX, Qwen, Gemma, Aya-23-style chat models: decoder-only."""
    name = (str(getattr(spec, "name", "")) + " " + str(getattr(spec, "hf_id", ""))).lower()
    if spec.prompt_style in ("nllb", "madlad", "seamless", "marian"):
        return True
    return spec.prompt_style == "t5prompt" and any(k in name for k in ("mt0", "aya-101", "mt5", "flan-t5"))


def build_prompt(spec: Any, text: str, iso: str, code: str) -> str:
    """The user message / plain prompt of the `chat` and `t5prompt` styles."""
    tpl = getattr(spec, "template", "") or "Translate the following text from {src_lang} into {tgt_lang}. Output only the translation.\n\n{text}"
    name = prompt_language_name(iso)
    return tpl.format(src_lang="English", tgt_lang=name, tgt_code=code, src_code=getattr(spec, "src_code", "en"), text=text)


def madlad_input(text: str, code: str) -> str:
    return f"<2{code}> {text}"


def format_inputs(spec: Any, sentences: Sequence[str], iso: str, allow_proxy: bool = False) -> List[str]:
    """Model inputs (strings) for `sentences`, per prompt style. For `nllb` / `seamless` the target code is passed to generate(), so the strings are the plain sentences."""
    code = target_code(spec, iso, allow_proxy)
    if code is None:
        raise KeyError(f"{getattr(spec, 'name', spec)} does not support {iso}")
    style = spec.prompt_style
    if style == "madlad":
        return [madlad_input(s, code) for s in sentences]
    if style in ("chat", "t5prompt"):
        return [build_prompt(spec, s, iso, code) for s in sentences]
    if style == "marian":
        tpl = getattr(spec, "template", "") or ""
        return [tpl.format(tgt_code=code, text=s) if tpl else s for s in sentences]
    if style in ("nllb", "seamless"):
        return list(sentences)
    raise ValueError(f"unknown prompt_style {style!r}")


_STRIP_LEAD = re.compile(r"^\s*(?:translation|translated text|\w+ translation)\s*:\s*", re.IGNORECASE)


def clean_llm_output(text: str) -> str:
    """Chat models sometimes wrap the answer: code fences, a leading 'Translation:'. The first paragraph is kept when several are generated for ONE sentence."""
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", t).strip()
    t = _STRIP_LEAD.sub("", t)
    if "\n\n" in t:
        t = t.split("\n\n")[0]
    return " ".join(t.split())


def length_order(items: Sequence[str]) -> List[int]:
    """Indices that sort `items` by length (batches of similar length waste fewer pad tokens)."""
    return sorted(range(len(items)), key=lambda i: len(items[i]))


def batches_by_budget(lengths: Sequence[int], max_items: int, max_tokens: int) -> List[List[int]]:
    """Group positions 0..n-1 (already sorted by length) into batches of at most `max_items` items and `max_tokens` padded tokens."""
    out, cur, cur_max = [], [], 0
    for i, ln in enumerate(lengths):
        new_max = max(cur_max, ln)
        if cur and (len(cur) + 1 > max_items or new_max * (len(cur) + 1) > max_tokens):
            out.append(cur)
            cur, new_max = [], ln
        cur.append(i)
        cur_max = new_max
    if cur:
        out.append(cur)
    return out


class Translator:
    """Base class: subclasses implement `_translate(sentences, iso) -> list` (any order of work inside, same order out)."""

    name = "translator"

    def __init__(self, spec: Any = None, allow_proxy: bool = False, cache_path: Optional[str] = None):
        self.spec = spec
        self.allow_proxy = allow_proxy
        self._cache: Dict[str, str] = {}
        self._cache_fh = None
        self.cache_path = cache_path
        self.n_calls = 0
        self.n_sentences = 0
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            if os.path.isfile(cache_path):
                with open(cache_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            k, v = json.loads(line)
                            self._cache[k] = v
                        except Exception:
                            continue
            self._cache_fh = open(cache_path, "a", encoding="utf-8")

    def _key(self, iso: str, s: str) -> str:
        return hashlib.sha1(f"{self.name}\x00{iso}\x00{s}".encode("utf-8")).hexdigest()

    def supports(self, iso: str) -> bool:
        return self.spec is None or target_code(self.spec, iso, self.allow_proxy) is not None

    def translate(self, sentences: Sequence[str], iso: str) -> List[str]:
        iso = canonical_iso(iso) or iso
        keys = [self._key(iso, s) for s in sentences]
        todo = [i for i, k in enumerate(keys) if k not in self._cache]
        if todo:
            uniq = list(dict.fromkeys(sentences[i] for i in todo))
            res = self._translate(uniq, iso)
            if len(res) != len(uniq):
                raise RuntimeError(f"{self.name} returned {len(res)} outputs for {len(uniq)} inputs")
            self.n_calls += 1
            self.n_sentences += len(uniq)
            for s, t in zip(uniq, res):
                k = self._key(iso, s)
                self._cache[k] = t
                if self._cache_fh:
                    self._cache_fh.write(json.dumps([k, t], ensure_ascii=False) + "\n")
            if self._cache_fh:
                self._cache_fh.flush()
        return [self._cache[k] for k in keys]

    def fn_for(self, iso: str) -> Callable[[List[str]], List[str]]:
        return lambda sents: self.translate(sents, iso)

    def _translate(self, sentences: List[str], iso: str) -> List[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:
        if self._cache_fh:
            self._cache_fh.close()
            self._cache_fh = None


# ------------------------------------------------------------------------------------------------ stub (plumbing tests, dry runs)
class StubTranslator(Translator):
    """No model. With `lookup` (English -> reference) it returns the reference with a fraction `drop` of the words removed (a controllable fake system whose
    score against the reference is known); without it every word gets a `iso` suffix and the order is reversed (never a copy of the English). Deterministic."""

    name = "stub"

    def __init__(self, lookup: Optional[Dict[str, str]] = None, drop: float = 0.0, seed: int = 0, **kw):
        super().__init__(**kw)
        self.lookup, self.drop, self.seed = lookup or {}, drop, seed

    def _translate(self, sentences, iso):
        out = []
        for s in sentences:
            ref = self.lookup.get(s)
            if ref is None:
                out.append(" ".join(f"{w}{iso}" for w in reversed(s.split())))
                continue
            if self.drop <= 0:
                out.append(ref)
                continue
            rng = random.Random(f"{self.seed}|{s}")
            words = ref.split()
            kept = [w for w in words if rng.random() >= self.drop] or words[:1]
            out.append(" ".join(kept))
        return out


# ------------------------------------------------------------------------------------------------ Hugging Face transformers
def _dtype(torch, name: str):
    return {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16}[name]


class HFSeq2SeqTranslator(Translator):
    """NLLB / MADLAD / SeamlessM4T-v2 (text) / Marian / mT5-style prompts with transformers seq2seq models."""

    name = "hf-seq2seq"

    def __init__(self, spec, device: str = "cpu", dtype: str = "fp32", num_beams: int = 4, batch_items: int = 32, batch_tokens: int = 8192,
                 max_new_tokens_cap: int = 512, model_id: Optional[str] = None, revision: Optional[str] = None, **kw):
        super().__init__(spec, **kw)
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch, self.device, self.num_beams = torch, device, num_beams
        self.batch_items, self.batch_tokens, self.cap = batch_items, batch_tokens, max_new_tokens_cap
        self.model_id = model_id or spec.hf_id
        self.revision = revision or (getattr(spec, "revision", "") or None)
        self.name = getattr(spec, "name", self.model_id)
        self._models: Dict[str, Tuple[Any, Any]] = {}
        self.dtype = dtype
        if self.model_id:                                      # per-language Marian models (hf_id_template) are loaded on first use
            self._load(self.model_id)

    def _load(self, model_id: str):
        if model_id in self._models:
            return self._models[model_id]
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        style = self.spec.prompt_style
        if style == "seamless":
            from transformers import SeamlessM4Tv2ForTextToText as M

            model = M.from_pretrained(model_id, revision=self.revision, torch_dtype=_dtype(self.torch, self.dtype))
            tok = AutoTokenizer.from_pretrained(model_id, revision=self.revision)
        else:
            kw = {}
            if style == "nllb":
                kw["src_lang"] = self.spec.src_code
            tok = AutoTokenizer.from_pretrained(model_id, revision=self.revision, **kw)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=self.revision, torch_dtype=_dtype(self.torch, self.dtype))
        model.eval().to(self.device)
        self._models[model_id] = (tok, model)
        return tok, model

    def _translate(self, sentences, iso):
        torch = self.torch
        style = self.spec.prompt_style
        code = target_code(self.spec, iso, self.allow_proxy)
        if style == "marian" and getattr(self.spec, "hf_id_template", ""):
            tok, model = self._load(self.spec.hf_id_template.format(code=code))
        else:
            tok, model = self._load(self.model_id)
        inputs = format_inputs(self.spec, sentences, iso, self.allow_proxy)
        order = length_order(inputs)
        lens = [len(inputs[i]) // 3 + 2 for i in order]
        out = [""] * len(inputs)
        gen_kw: Dict[str, Any] = {"num_beams": self.num_beams}
        if style == "nllb":
            gen_kw["forced_bos_token_id"] = tok.convert_tokens_to_ids(code)
        if style == "seamless":
            gen_kw["tgt_lang"] = code
        for b in batches_by_budget(lens, self.batch_items, self.batch_tokens):
            idx = [order[j] for j in b]
            enc = tok([inputs[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            cap = min(self.cap, int(enc["input_ids"].shape[1] * 3) + 16)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=cap, **gen_kw)
            for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = " ".join(t.split())
        return out


class HFCausalTranslator(Translator):
    """Chat / plain-prompt decoder-only models with transformers (greedy). The registry's `template` is the user message (TranslateGemma's card prompt included)."""

    name = "hf-causal"

    def __init__(self, spec, device: str = "cpu", dtype: str = "bf16", batch_items: int = 16, batch_tokens: int = 8192, max_new_tokens_cap: int = 768,
                 model_id: Optional[str] = None, revision: Optional[str] = None, **kw):
        super().__init__(spec, **kw)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.device = torch, device
        self.batch_items, self.batch_tokens, self.cap = batch_items, batch_tokens, max_new_tokens_cap
        mid = model_id or spec.hf_id
        rev = revision or (getattr(spec, "revision", "") or None)
        self.name = getattr(spec, "name", mid)
        self.tok = AutoTokenizer.from_pretrained(mid, revision=rev, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(mid, revision=rev, torch_dtype=_dtype(torch, dtype)).eval().to(device)

    def _render(self, prompt: str) -> str:
        if self.spec.prompt_style == "chat":
            msgs = [{"role": "user", "content": prompt}]
            try:                                               # Qwen3-style templates (Sunflower, AfriqueQwen) think unless told not to; others ignore the flag
                return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return prompt

    def _translate(self, sentences, iso):
        torch = self.torch
        code = target_code(self.spec, iso, self.allow_proxy)
        prompts = format_inputs(self.spec, sentences, iso, self.allow_proxy)
        texts = [self._render(p) for p in prompts]
        order = length_order(texts)
        lens = [len(texts[i]) // 3 + 2 for i in order]
        out = [""] * len(texts)
        for b in batches_by_budget(lens, self.batch_items, self.batch_tokens):
            idx = [order[j] for j in b]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            cap = min(self.cap, int(enc["input_ids"].shape[1] * 2) + 32)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=cap, do_sample=False)
            new = gen[:, enc["input_ids"].shape[1]:]
            for i, t in zip(idx, self.tok.batch_decode(new, skip_special_tokens=True)):
                out[i] = clean_llm_output(t)
        return out


class VLLMTranslator(Translator):
    """Decoder-only chat / plain-prompt models through vLLM (GPU; greedy, temperature 0). NOT used while the GPU is busy; written for the translation run."""

    name = "vllm"

    def __init__(self, spec, gpu_memory_utilization: float = 0.5, max_model_len: int = 4096, dtype: str = "bfloat16", max_tokens: int = 768, **kw):
        super().__init__(spec, **kw)
        from vllm import LLM, SamplingParams

        self.name = getattr(spec, "name", spec.hf_id)
        self.llm = LLM(model=spec.hf_id, revision=(getattr(spec, "revision", "") or None), dtype=dtype, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len)
        self.sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def _translate(self, sentences, iso):
        prompts = format_inputs(self.spec, sentences, iso, self.allow_proxy)
        tok = self.llm.get_tokenizer()
        texts = []
        for p in prompts:
            if self.spec.prompt_style == "chat":
                try:
                    texts.append(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
                except TypeError:
                    texts.append(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True))
            else:
                texts.append(p)
        res = self.llm.generate(texts, self.sp)
        return [clean_llm_output(r.outputs[0].text) for r in res]


class CT2Translator(Translator):
    """CTranslate2 for NLLB / MADLAD (T5): `model_dir` is a converted checkpoint (ct2-transformers-converter ...). Written against the documented API; not
    exercised here (ctranslate2 is not installed on the rental)."""

    name = "ctranslate2"

    def __init__(self, spec, model_dir: str, device: str = "cuda", compute_type: str = "int8_float16", beam_size: int = 4, max_batch_size: int = 64,
                 tokenizer_id: Optional[str] = None, **kw):
        super().__init__(spec, **kw)
        import ctranslate2
        from transformers import AutoTokenizer

        self.name = getattr(spec, "name", model_dir)
        self.tr = ctranslate2.Translator(model_dir, device=device, compute_type=compute_type)
        kwargs = {"src_lang": spec.src_code} if spec.prompt_style == "nllb" else {}
        self.tok = AutoTokenizer.from_pretrained(tokenizer_id or spec.hf_id, **kwargs)
        self.beam, self.maxb = beam_size, max_batch_size

    def _translate(self, sentences, iso):
        code = target_code(self.spec, iso, self.allow_proxy)
        inputs = format_inputs(self.spec, sentences, iso, self.allow_proxy)
        toks = [self.tok.convert_ids_to_tokens(self.tok.encode(s)) for s in inputs]
        prefix = [[code]] * len(toks) if self.spec.prompt_style == "nllb" else None
        res = self.tr.translate_batch(toks, target_prefix=prefix, beam_size=self.beam, max_batch_size=self.maxb, batch_type="tokens" if self.maxb > 256 else "examples")
        outs = []
        for r in res:
            hyp = r.hypotheses[0]
            if prefix:
                hyp = hyp[1:]
            outs.append(" ".join(self.tok.decode(self.tok.convert_tokens_to_ids(hyp), skip_special_tokens=True).split()))
        return outs


# ------------------------------------------------------------------------------------------------ factory
def make_translator(spec: Any, backend: str = "hf", **kw) -> Translator:
    """`backend`: stub | hf | vllm | ctranslate2. `hf` picks the seq2seq or the causal class from the spec's prompt style."""
    if backend == "stub":
        return StubTranslator(**{k: v for k, v in kw.items() if k in ("lookup", "drop", "seed", "allow_proxy", "cache_path")})
    if spec is None:
        raise ValueError("a SystemSpec is required for the real backends")
    if backend == "hf":
        return (HFSeq2SeqTranslator if is_encoder_decoder(spec) else HFCausalTranslator)(spec, **kw)
    if backend == "vllm":
        return VLLMTranslator(spec, **kw)
    if backend in ("ctranslate2", "ct2"):
        return CT2Translator(spec, **kw)
    raise ValueError(f"unknown backend {backend!r}")
