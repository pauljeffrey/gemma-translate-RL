# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-editor (English source + draft translation -> corrected translation): prompt format, training examples, draft-quality helpers, MBR, generation backends.

Why this file exists: the editor is a small SFT of the translation model (same tokenizer, same single-token language tags, same chat template, same loss mask) on a NEW prompt.
Everything that defines that prompt lives here (one place, covered by tests/test_post_edit.py), exactly as `gemma_mt.format` does for translation.

    user turn   :  Post-edit the draft translation to {african_language}:  {english_text}

                   Draft:  {draft_text}
    model turn  :  {tag} {corrected_translation}<end_of_turn>                      (default, "rewrite" target: same output shape as a translation)
                   {tag} <unused6150><end_of_turn>                                 (variant "keep": the dedicated KEEP marker means "the draft is already right")

* The model turn of the default target is byte-identical in shape to a translation (`{tag} {text}<end_of_turn>`), so `format.strip_iso_prefix`, `rl.text.decode_completion`, every
  scorer and the RL reward read the editor's output unchanged.
* Loss covers only the model turn (tag + translation + <end_of_turn>), as in `format.build_example`.
* The prompt starts with the verb "Post-edit" and not "Translate": it is a new task, trained with ~50k examples. The alternative that keeps the literal translation prompt and appends the draft was
  considered and rejected (a model that can ignore the draft and translate is exactly the failure that makes an editor useless; the plan, section 3, explains).
* A draft is cut to `MAX_DRAFT_TOKENS` Gemma tokens (training AND evaluation, same function) so that a looping greedy draft (hundreds of tokens) cannot blow the sequence length.
* Optional marker tokens use free `<unusedK>` slots at K = 6150.. (below the tag slots 6241 - (n_languages - 99), which start at 6216 for the 125 languages of the table and reach 6150 only after
  91 more languages are appended; `assert_marker_free` checks it). They need their embedding row initialised (`init_marker_rows` in scripts/post_editor_build_sft.py); the default format uses none.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import random
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from gemma_mt import format as fmt
from gemma_mt import tags as _tags
from gemma_mt.languages import ALL_ISOS, target_tag_and_name

PE_FORMAT_VERSION = 1
PE_INSTRUCTION_TEMPLATE = "Post-edit the draft translation to {african_language}:  {english_text}\n\nDraft:  {draft_text}"
DRAFT_SEP = "\n\nDraft:  "
MAX_DRAFT_TOKENS = 192

KEEP_K = 6150                       # <unused6150>: "the draft is already correct" (variant "keep")
MARKER_SLOTS = {"keep": KEEP_K}
KEEP_MARKER = f"<unused{KEEP_K}>"


# ------------------------------------------------------------------------------------------------ markers
def assert_marker_free(k: int = KEEP_K) -> None:
    """The marker slot must not be the tag slot of any language of the table (tags take K = 0..98 and 6241, 6240, ... downward)."""
    slot = _tags.slot_from_unused_index(k)
    if 0 <= slot < len(ALL_ISOS):
        raise ValueError(f"<unused{k}> is the language tag of {ALL_ISOS[slot]!r}; pick another marker slot")


# ------------------------------------------------------------------------------------------------ text helpers
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def truncate_draft(draft: str, tokenizer: Any = None, max_tokens: int = MAX_DRAFT_TOKENS) -> str:
    """Cut a draft to `max_tokens` tokens (the same rule in training and evaluation). Without a tokenizer: `max_tokens * 4` characters."""
    draft = (draft or "").strip()
    if tokenizer is None:
        return draft[: max_tokens * 4]
    ids = tokenizer(draft, add_special_tokens=False)
    ids = ids["input_ids"] if not isinstance(ids, list) else ids
    if len(ids) <= max_tokens:
        return draft
    return tokenizer.decode(list(ids[:max_tokens]), skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()


# ------------------------------------------------------------------------------------------------ messages
def build_pe_user_content(english: str, draft: str, iso: str) -> str:
    english = (english or "").strip()
    if not english:
        raise ValueError("empty source text")
    return PE_INSTRUCTION_TEMPLATE.format(african_language=target_tag_and_name(iso)[1], english_text=english, draft_text=(draft or "").strip())


def parse_pe_user_content(text: str) -> Tuple[str, str, str]:
    """-> (language name, english, draft) of a user turn built by `build_pe_user_content` (the FIRST draft separator splits; a source that itself contains the separator is not supported)."""
    head, sep = "Post-edit the draft translation to ", ":  "
    if not text.startswith(head):
        raise ValueError("not a post-editor prompt")
    rest = text[len(head):]
    name, _, rest = rest.partition(sep)
    if DRAFT_SEP not in rest:
        if rest.endswith("\n\nDraft:"):               # an empty draft: the chat template trims the trailing spaces
            return name, rest[: -len("\n\nDraft:")], ""
        raise ValueError("no draft separator")
    english, _, draft = rest.partition(DRAFT_SEP)
    return name, english, draft


def build_pe_model_content(iso: str, text: str, *, keep: bool = False, mode: Optional[str] = None, tokenizer: Any = None, keep_marker: str = KEEP_MARKER) -> str:
    """`{tag} {text}`; with keep=True the text is the KEEP marker (the target of a draft that should be left alone)."""
    if keep:
        tag = target_tag_and_name(iso)[0]
        mode = mode or fmt.get_tag_mode()
        if mode == "single_token" and tag != fmt.PIVOT_ISO:
            tag = _tags.tag_string(tag, tokenizer)
        return f"{tag} {keep_marker}"
    return fmt.build_model_content(iso, text, mode, tokenizer)


def build_pe_messages(english: str, draft: str, iso: str, reference: Optional[str] = None, *, keep: bool = False, mode: Optional[str] = None,
                      tokenizer: Any = None, keep_marker: str = KEEP_MARKER) -> List[Dict[str, str]]:
    msgs = [{"role": "user", "content": build_pe_user_content(english, draft, iso)}]
    if reference is not None or keep:
        msgs.append({"role": "assistant", "content": build_pe_model_content(iso, reference or "", keep=keep, mode=mode, tokenizer=tokenizer, keep_marker=keep_marker)})
    return msgs


def pe_prompt_text(tokenizer: Any, english: str, draft: str, iso: str) -> str:
    return fmt.render_chat(tokenizer, build_pe_messages(english, draft, iso), add_generation_prompt=True)


def build_pe_prompt(english: str, draft: str, iso: str, tokenizer: Any, max_draft_tokens: int = MAX_DRAFT_TOKENS) -> List[int]:
    """Token ids to feed `generate()`: ends with the generation prompt (the model's first token is the tag). The draft is cut with `truncate_draft`."""
    draft = truncate_draft(draft, tokenizer, max_draft_tokens)
    ids = fmt._encode(tokenizer, pe_prompt_text(tokenizer, english, draft, iso))
    fmt._check_single_bos(tokenizer, ids)
    return ids


def build_pe_example(english: str, draft: str, iso: str, reference: str, tokenizer: Any, *, keep: bool = False, max_draft_tokens: int = MAX_DRAFT_TOKENS,
                     keep_marker: str = KEEP_MARKER) -> fmt.Example:
    """Training example: loss ONLY on `{tag} {reference}<end_of_turn>` (or `{tag} <keep marker><end_of_turn>`); same construction as `format.build_example`."""
    draft = truncate_draft(draft, tokenizer, max_draft_tokens)
    msgs = build_pe_messages(english, draft, iso, reference, keep=keep, tokenizer=tokenizer, keep_marker=keep_marker)
    prompt = fmt.render_chat(tokenizer, msgs[:1], add_generation_prompt=True)
    full = fmt.render_chat(tokenizer, msgs, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise ValueError(f"chat template: the full conversation does not start with the generation prompt (prompt={prompt!r}, full={full!r})")
    suffix = full[len(prompt):]
    cut = suffix.rfind(fmt.END_OF_TURN)
    if cut < 0:
        raise ValueError(f"no {fmt.END_OF_TURN} in the rendered model turn: {suffix!r}")
    target_str, tail_str = suffix[: cut + len(fmt.END_OF_TURN)], suffix[cut + len(fmt.END_OF_TURN):]
    p_ids, t_ids = fmt._encode(tokenizer, prompt), fmt._encode(tokenizer, target_str)
    tail_ids = fmt._encode(tokenizer, tail_str) if tail_str else []
    fmt._check_single_bos(tokenizer, p_ids)
    eot = tokenizer.convert_tokens_to_ids(fmt.END_OF_TURN)
    if not t_ids or t_ids[-1] != eot:
        raise ValueError(f"the target does not end with the {fmt.END_OF_TURN} token id ({eot}): {t_ids[-3:]}")
    tag = target_tag_and_name(iso)[0]
    if fmt.get_tag_mode() == "single_token" and tag != fmt.PIVOT_ISO:
        want = tokenizer.convert_tokens_to_ids(_tags.tag_string(tag, tokenizer))
        if t_ids[0] != want:
            raise ValueError(f"tag mode single_token: the target of {iso!r} must start with the tag token id {want}, it starts with {t_ids[:3]}")
    if keep:
        mk = tokenizer.convert_tokens_to_ids(keep_marker)
        unk = getattr(tokenizer, "unk_token_id", None)
        if mk is None or mk == unk or mk not in t_ids:
            raise ValueError(f"the keep marker {keep_marker!r} is not a single token of this tokenizer")
    ids = p_ids + t_ids + tail_ids
    labels = [fmt.IGNORE_INDEX] * len(p_ids) + list(t_ids) + [fmt.IGNORE_INDEX] * len(tail_ids)
    return fmt.Example(ids, labels, len(p_ids), len(t_ids), len(tail_ids), text=full)


# ------------------------------------------------------------------------------------------------ outputs
def edited_text(completion: str, iso: str, draft: str, keep_marker: str = KEEP_MARKER) -> Tuple[str, str]:
    """(final text, status) of a decoded editor completion. status: ok | missing | wrong | empty | keep (the KEEP marker: the draft is returned unchanged)."""
    r = fmt.strip_iso_prefix(completion, iso)
    t = r.text.strip()
    if t == keep_marker:
        return (draft or "").strip(), "keep"
    return t, r.status


# ------------------------------------------------------------------------------------------------ metrics
def sentence_chrf(hyp: str, ref: str) -> float:
    """Sentence chrF++ (word_order 2, NFC, 0..100) of one hypothesis against one reference; an empty hypothesis scores 0."""
    from sacrebleu.metrics import CHRF

    hyp, ref = nfc(hyp).strip(), nfc(ref).strip()
    if not hyp or not ref:
        return 0.0
    return float(CHRF(word_order=2).sentence_score(hyp, [ref]).score)


def word_edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance between two token sequences (rolling rows)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def edit_rate(draft: str, edited: str) -> float:
    """Word-level edit distance divided by the length (in words) of the longer text; 0 = identical, 1 = nothing shared."""
    a, b = nfc(draft).split(), nfc(edited).split()
    n = max(len(a), len(b))
    return 0.0 if n == 0 else word_edit_distance(a, b) / n


def mbr_select(candidates: Sequence[str], utility: Optional[Callable[[str, str], float]] = None) -> int:
    """Index of the candidate with the highest MEAN utility against all the OTHERS (the decode study's rule: chrF++ utility, empty candidates lose). Ties -> the earliest candidate,
    so put the greedy output first."""
    util = utility or (lambda h, r: sentence_chrf(h, r))
    n = len(candidates)
    if n == 1:
        return 0
    best, best_s = 0, -1.0
    for i, c in enumerate(candidates):
        s = (sum(util(c, d) for j, d in enumerate(candidates) if j != i) / (n - 1)) if c.strip() else -1.0
        if s > best_s:
            best, best_s = i, s
    return best


# ------------------------------------------------------------------------------------------------ deterministic helpers
def stable_hash(*parts: Any) -> int:
    return int.from_bytes(hashlib.blake2b("\x1f".join(map(str, parts)).encode("utf-8"), digest_size=8).digest(), "big")


def stable_unit(*parts: Any) -> float:
    """Deterministic uniform number in [0, 1) from the parts."""
    return stable_hash(*parts) / 2.0 ** 64


def unimax_water_fill(counts: Dict[str, int], budget: int, cap: int, floor: int = 0) -> Dict[str, int]:
    """UniMax-style allocation of `budget` items over languages: every language gets min(available, cap, level) with the water level chosen so that the total hits the budget; a language with
    fewer than `floor` items available keeps ALL of them (tail flooring). Deterministic; the budget is met up to the available mass (when every language is exhausted the total is smaller)."""
    avail = {k: min(int(v), cap) for k, v in counts.items() if v > 0}
    out = {k: 0 for k in avail}
    left = dict(avail)
    remaining = int(budget)
    # tail flooring first: small languages keep everything they have
    for k, v in list(left.items()):
        if v <= floor and remaining >= v:
            out[k] = v; remaining -= v; del left[k]
    while left and remaining > 0:
        share = remaining // len(left)
        if share == 0:                                   # fewer items than languages: one each in a deterministic order
            for k in sorted(left, key=lambda k: (-left[k], k))[:remaining]:
                out[k] += 1
            remaining = 0
            break
        small = [k for k, v in left.items() if v <= share]
        if not small:
            for k in left:
                out[k] += share
            remaining -= share * len(left)
            # distribute the remainder one by one to the languages that still have room
            for k in sorted(left, key=lambda k: (-(left[k] - out[k]), k)):
                if remaining <= 0:
                    break
                if out[k] < left[k]:
                    out[k] += 1; remaining -= 1
            break
        for k in small:
            out[k] += left[k]; remaining -= left[k]; del left[k]
    return {k: min(v, avail[k]) for k, v in out.items() if v > 0}


# ------------------------------------------------------------------------------------------------ synthetic "good draft" noise
def perturb_light(reference: str, seed_parts: Sequence[Any]) -> str:
    """A NEAR-reference draft: the reference with one or two tiny, deterministic edits (a changed final punctuation mark, a dropped or swapped word, a dropped diacritic-free letter).
    Never returns the reference unchanged unless the text is a single character."""
    words = reference.split()
    rng = random.Random(stable_hash("perturb", *seed_parts))
    ops = ["punct", "swap", "drop", "case"]
    rng.shuffle(ops)
    out = list(words)
    n_ops = 1 if len(words) < 8 else 2
    for op in ops[:n_ops]:
        if op == "punct" and out:
            w = out[-1]
            out[-1] = w[:-1] if w and w[-1] in ".!?;,:" else w + "."
        elif op == "swap" and len(out) >= 4:
            i = rng.randrange(len(out) - 1); out[i], out[i + 1] = out[i + 1], out[i]
        elif op == "drop" and len(out) >= 6:
            del out[rng.randrange(len(out))]
        elif op == "case" and out:
            i = rng.randrange(len(out)); w = out[i]
            out[i] = (w[0].lower() + w[1:]) if w[:1].isupper() else (w[:1].upper() + w[1:])
    res = " ".join(out)
    return res if res != reference else reference[:-1] if len(reference) > 1 else reference


# ------------------------------------------------------------------------------------------------ synthetic corruption of the REFERENCE (extra drafts with real edit work)
CORRUPTIONS = ("drop", "insert", "swap", "truncate", "repeat", "number_entity", "switch_en", "neighbour")


def corrupt_reference(reference: str, english: str, kind: str, seed_parts: Sequence[Any], neighbour: Optional[str] = None) -> str:
    """A draft made by damaging the human reference (never a model output). Deterministic in (kind, seed_parts).
    drop: remove 1-3 words | insert: add 1-3 words copied from elsewhere in the reference or the English | swap: exchange two neighbouring (or distant) words |
    truncate: keep the first 35-75% of the words | repeat: repeat a 2-4 word phrase | number_entity: change a number, else replace a capitalised word, else a random word |
    switch_en: replace a 2-5 word span by a span of the ENGLISH source (a clause left untranslated) | neighbour: the reference of ANOTHER sentence (`neighbour`), i.e. a fluent but wrong draft.
    The result differs from the reference whenever the text has enough words; very short texts fall back to a different operation, then to appending/removing a final mark."""
    if kind not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {kind!r}")
    reference = (reference or "").strip()
    rng = random.Random(stable_hash("corrupt", kind, *seed_parts))
    w, ew = reference.split(), (english or "").split()
    out = list(w)
    if kind == "neighbour":
        return (neighbour or "").strip() or perturb_light(reference, seed_parts)
    if kind == "drop" and len(out) >= 3:
        for _ in range(min(rng.randint(1, 3), len(out) - 1)):
            del out[rng.randrange(len(out))]
    elif kind == "insert" and out:
        pool = out + ew
        for _ in range(rng.randint(1, 3)):
            out.insert(rng.randrange(len(out) + 1), rng.choice(pool))
    elif kind == "swap" and len(out) >= 3:
        i = rng.randrange(len(out) - 1)
        j = i + 1 if rng.random() < 0.6 or len(out) < 5 else min(len(out) - 1, i + rng.randint(2, 3))
        out[i], out[j] = out[j], out[i]
    elif kind == "truncate" and len(out) >= 4:
        out = out[: max(1, int(round(len(out) * rng.uniform(0.35, 0.75))))]
    elif kind == "repeat" and out:
        n = min(rng.randint(2, 4), len(out)); i = rng.randrange(len(out) - n + 1)
        out[i + n:i + n] = out[i:i + n] * rng.randint(1, 2)
    elif kind == "number_entity" and out:
        digit_idx = [i for i, t in enumerate(out) if any(c.isdigit() for c in t)]
        cap_idx = [i for i, t in enumerate(out) if i > 0 and t[:1].isupper()]
        if digit_idx:
            i = rng.choice(digit_idx)
            out[i] = "".join(str((int(c) + rng.randint(1, 8)) % 10) if c.isdigit() else c for c in out[i])
        elif cap_idx and len(out) > 1:
            i = rng.choice(cap_idx); others = [t for t in out + ew if t[:1].isupper() and t != out[i]]
            out[i] = rng.choice(others) if others else out[i][::-1]
        else:
            i = rng.randrange(len(out)); out[i] = rng.choice(out + ew)
    elif kind == "switch_en" and out and ew:
        n = min(rng.randint(2, 5), len(out)); m = min(n, len(ew))
        i = rng.randrange(len(out) - n + 1); k = rng.randrange(len(ew) - m + 1)
        out[i:i + n] = ew[k:k + m]
    res = " ".join(out)
    if res == " ".join(w):                                    # too short for the operation: fall back so the draft is never the reference
        for alt in ("insert", "swap", "number_entity"):
            if alt != kind:
                r2 = corrupt_reference(reference, english, alt, list(seed_parts) + ["fb"], neighbour)
                if r2 != reference:
                    return r2
        return perturb_light(reference, seed_parts)
    return res


def sample_desired_chrf(dist: Optional[Sequence[float]], rng: random.Random, mean: float = 30.0, sd: float = 18.0) -> float:
    """A target sentence chrF++ for one training pair: a draw from the empirical distribution of the benchmark drafts' sentence chrF++ of that language (`dist`), else a clipped normal."""
    if dist:
        return float(dist[rng.randrange(len(dist))])
    return float(min(100.0, max(0.0, rng.gauss(mean, sd))))


def pick_candidate(chrfs: Sequence[float], desired: float, rng: random.Random, weights: Optional[Sequence[float]] = None) -> int:
    """Index of the candidate draft whose sentence chrF++ is closest to `desired` (edit-rate matching, section 'quantile matching' of the plan). Ties and near-ties (within 2 chrF) are broken
    randomly, optionally by `weights` (a prior per candidate system, so that the mix of draft SOURCES does not collapse onto one system)."""
    d = [abs(c - desired) for c in chrfs]
    best = min(d)
    near = [i for i, x in enumerate(d) if x <= best + 2.0]
    if weights is None:
        return rng.choice(near)
    ws = [max(1e-9, weights[i]) for i in near]
    return rng.choices(near, ws)[0]


# ------------------------------------------------------------------------------------------------ generation backends (vLLM on the GPU box, HF for CPU tests)
@dataclass
class GenOut:
    ids: List[int]          # generated token ids (the stop token included when the sequence finished)
    finished: bool


class HFBackend:
    """Reference backend on transformers `generate` (CPU smoke tests, tiny models, or a quick GPU check). Same interface as VLLMBackend."""

    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu", stop_ids: Optional[Sequence[int]] = None, batch_tokens: int = 0, dtype: Any = None, max_batch: int = 160):
        """batch_tokens > 0: batched generation (prompts sorted by length, left-padded, a batch holds at most that many prompt tokens x beams); 0: one prompt at a time (CPU tests)."""
        import torch

        self.torch, self.tok, self.device, self.batch_tokens, self.max_batch = torch, tokenizer, device, batch_tokens, max_batch
        self.model = (model.to(dtype) if dtype is not None else model).to(device).eval()
        self.stop = list(stop_ids) if stop_ids is not None else fmt.resolve_stop_tokens(tokenizer).ids

    def _gen(self, prompts: Sequence[Sequence[int]], max_new: int, **kw) -> List[List[List[int]]]:
        torch = self.torch
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        res: List[Any] = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        width = max(1, int(kw.get("num_beams", 1)), int(kw.get("num_return_sequences", 1)))
        i = 0
        while i < len(order):
            L = len(prompts[order[i]])
            bs = 1 if not self.batch_tokens else max(1, min(self.max_batch, self.batch_tokens // max(1, L * width)))
            idx = order[i:i + bs]; i += bs
            lens = [len(prompts[j]) for j in idx]; Lm = max(lens)
            x = torch.tensor([[pad] * (Lm - len(prompts[j])) + list(prompts[j]) for j in idx], device=self.device)
            m = torch.tensor([[0] * (Lm - len(prompts[j])) + [1] * len(prompts[j]) for j in idx], device=self.device)
            with torch.no_grad():
                out = self.model.generate(input_ids=x, attention_mask=m, max_new_tokens=max_new, eos_token_id=self.stop, pad_token_id=pad, **kw)
            nret = out.shape[0] // len(idx)
            for k, j in enumerate(idx):
                res[j] = [out[k * nret + r][Lm:].tolist() for r in range(nret)]
        return res

    @staticmethod
    def _finish(g: List[int], stop: Sequence[int]) -> GenOut:
        for i, t in enumerate(g):
            if t in stop:
                return GenOut(g[: i + 1], True)
        return GenOut(g, False)

    def greedy(self, prompts, max_new: int = 256) -> List[GenOut]:
        return [self._finish(o[0], self.stop) for o in self._gen(prompts, max_new, do_sample=False, num_beams=1)]

    def beam(self, prompts, width: int = 4, max_new: int = 256, length_penalty: float = 1.0) -> List[GenOut]:
        return [self._finish(o[0], self.stop) for o in self._gen(prompts, max_new, do_sample=False, num_beams=width, length_penalty=length_penalty, early_stopping=True)]

    def sample(self, prompts, n: int = 1, temperature: float = 0.7, top_p: float = 0.95, max_new: int = 256, seed: int = 0) -> List[List[GenOut]]:
        self.torch.manual_seed(seed)
        return [[self._finish(g, self.stop) for g in o] for o in self._gen(prompts, max_new, do_sample=True, temperature=temperature, top_p=top_p, top_k=0, num_return_sequences=n)]


class VLLMBackend:
    """vLLM backend (the GPU box: /workspace/venv_vllm). `model` is an HF-format directory; prompts are token-id lists built with the ORIGINAL tokenizer."""

    def __init__(self, model: str, dtype: str = "bfloat16", gpu_util: float = 0.2, max_model_len: int = 2048, seed: int = 0, stop_ids: Sequence[int] = (1, 106), enforce_eager: bool = False, max_num_seqs: int = 512):
        from vllm import LLM

        self.llm = LLM(model=model, dtype=dtype, gpu_memory_utilization=gpu_util, max_model_len=max_model_len, enforce_eager=enforce_eager, seed=seed, max_num_seqs=max_num_seqs)
        self.stop = list(stop_ids)

    def _finish(self, toks: Sequence[int]) -> GenOut:
        return HFBackend._finish([int(t) for t in toks], self.stop)

    def greedy(self, prompts, max_new: int = 256) -> List[GenOut]:
        from vllm import SamplingParams

        outs = self.llm.generate([{"prompt_token_ids": list(p)} for p in prompts], SamplingParams(temperature=0.0, max_tokens=max_new, stop_token_ids=self.stop))
        return [self._finish(o.outputs[0].token_ids) for o in outs]

    def sample(self, prompts, n: int = 1, temperature: float = 0.7, top_p: float = 0.95, max_new: int = 256, seed: int = 0) -> List[List[GenOut]]:
        from vllm import SamplingParams

        outs = self.llm.generate([{"prompt_token_ids": list(p)} for p in prompts],
                                 SamplingParams(temperature=temperature, top_p=top_p, n=n, max_tokens=max_new, stop_token_ids=self.stop, seed=seed))
        return [[self._finish(c.token_ids) for c in o.outputs] for o in outs]

    def beam(self, prompts, width: int = 4, max_new: int = 256, length_penalty: float = 1.0) -> List[GenOut]:
        from vllm.sampling_params import BeamSearchParams

        P = [{"prompt_token_ids": list(p)} for p in prompts]
        outs = self.llm.beam_search(P, BeamSearchParams(beam_width=width, max_tokens=max_new, length_penalty=length_penalty))
        res = []
        for p, o in zip(prompts, outs):
            toks = [int(t) for t in o.sequences[0].tokens[len(p):]]
            res.append(self._finish(toks))
        return res


def decode_gen(tokenizer: Any, g: GenOut, iso: str, stop_ids: Sequence[int] = (1, 106)):
    """-> (translation text without the tag, finished, tag status) of one generation, through the evaluation's own decoder."""
    from gemma_mt.rl.text import decode_completion, translation_from

    d = decode_completion(tokenizer, g.ids, stop_ids)
    r = translation_from(d, iso)
    return r.text, bool(d.finished), r.status


def make_backend(kind: str, model: str, tok: Any, dtype: str = "bfloat16", gpu_util: float = 0.8, max_model_len: int = 2048, batch_tokens: int = 100000, seed: int = 0):
    """kind: vllm (GPU, fast, but its beam search is slow) | hfgpu (transformers generate, batched, bf16) | hf (CPU, one prompt at a time: tests)."""
    if kind == "vllm":
        return VLLMBackend(model, dtype=dtype, gpu_util=gpu_util, max_model_len=max_model_len, seed=seed)
    import torch
    from gemma_mt.modeling import load_model
    if kind == "hfgpu":
        return HFBackend(load_model(model, dtype=getattr(torch, dtype)), tok, device="cuda", batch_tokens=batch_tokens)
    if kind == "hf":
        return HFBackend(load_model(model), tok, device="cpu")
    raise ValueError(kind)
