"""Shared machinery of the second-round converters (ext_*.py).

  * language handling: every DECLARED language is audited with GlotLID (langaudit.py); a row whose text is confidently another language is dropped
    (translation pairs) or relabelled (single-text rows, when the detected language is one we carry);
  * AfriCOMET-QE scores computed offline (twi50m_filter.py) are looked up by row position;
  * generic row shapes: aligned pairs, labelled texts, instruction/input/output triples, chat message lists, function-calling conversations.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from africomet_translation.instructions import make_instruction
from sft_new import hubio
from sft_new import langdetect as L
from sft_new import templates as T
from sft_new import ext_pools                        # noqa: F401  (registers the pools)
from sft_new.compose import A, Rec, Skip
from sft_new.langaudit import audit, code_of, judge, register_extra_languages
from sft_new.textutil import aligned, blank, clean, degenerate, length_ratio_ok, rep_ratio

register_extra_languages()

QE_DIR = Path(os.environ.get("SFT_QE_DIR", "/workspace/sft_build/out/qe"))
from sft_new.langaudit import EXISTING_TAG_LANGS as _EXISTING
EXTRA_GLOT = {"fra_Latn": "fra", "amh_Ethi": "amh", "tir_Ethi": "ti", "dag_Latn": "dag", "bin_Latn": "bin", "ssw_Latn": "ssw", "nso_Latn": "nso", "tso_Latn": "tso",
              "ven_Latn": "ven", "kik_Latn": "kik", "lug_Latn": "lug", "lin_Latn": "lin", "luo_Latn": "luo", "bem_Latn": "bem", "nbl_Latn": "nbl", "kln_Latn": "kln", "llg_Latn": "llg", "lch_Latn": "lch"}
OUR = frozenset(L.SUPPORTED) | {"fra", "amh", "ti", "dag", "bin", "ssw", "nso", "tso", "ven", "kik", "lug", "lin", "luo", "bem", "nbl", "kln", "llg", "lch"}


for _c in _EXISTING:
    EXTRA_GLOT.setdefault(f"{'bam' if _c == 'bm' else _c}_Latn", _c)
OUR = frozenset(OUR) | set(_EXISTING)


def label_code(label: str | None) -> str | None:
    """A raw GlotLID label -> one of our language codes (None when it is not a language we carry)."""
    return L.GLOT_TO_CODE.get(label) or EXTRA_GLOT.get(label) if label else None


def il_for(rng: random.Random, prefer: str | None = None) -> str:
    from africomet_translation.instructions import instruction_languages
    have = instruction_languages()
    if prefer in have and rng.random() < 0.3:
        return prefer
    return "eng" if rng.random() < 0.5 else rng.choice(have)


def batched(it: Iterable, n: int) -> Iterator[list]:
    import itertools
    it = iter(it)
    while True:
        b = list(itertools.islice(it, n))
        if not b:
            return
        yield b


def translation_recs(a: str, b: str, la: str, lb: str, dataset: str, rng: random.Random, ua=False, ub=False, both=True, meta=None) -> list:
    out = []
    dirs = ((a, b, la, lb, ua, ub, "forward"), (b, a, lb, la, ub, ua, "backward")) if both else ((a, b, la, lb, ua, ub, "forward"),)
    for x, y, s, t, u1, u2, direction in dirs:
        ins = make_instruction(il_for(rng, s), s, t, rng)
        out.append(Rec("translation", dataset, s, t, y, input=x, instruction=ins.text, position=ins.position, plan=["<translate>"],
                       unsure_src=u1, unsure_tgt=u2, meta={**(meta or {}), "instruction_language": ins.language, "direction": direction}))
    return out


# ------------------------------------------------------------------ QE scores computed offline

def qe_scores(slug: str):
    """float16 array of AfriCOMET-QE scores for the repo `slug` (NaN = prefiltered), indexed by global row position, or None."""
    p = QE_DIR / slug / "scores_float16.npy"
    return np.load(p, mmap_mode="r") if p.exists() else None


def require_qe(slug: str):
    s = qe_scores(slug)
    if s is None:
        raise RuntimeError(f"AfriCOMET-QE scores for {slug} are missing ({QE_DIR / slug}); run `python -m sft_new.twi50m_filter --repo ... --out {QE_DIR / slug}` first")
    return s


# ------------------------------------------------------------------ pairs

def _get(r: dict, k):
    return k(r) if callable(k) else r.get(k)


def pairs(rows: Iterable[dict], ka, kb, la: str, lb: str, dataset: str, rng: random.Random, *, limit: int | None = None, scores=None,
          min_score: float | None = None, max_chars: int = 3600, both: bool = True, meta: dict | None = None, row_ok=None, min_chars: int = 8,
          audit_langs: bool = True) -> Iterator:
    """Aligned two-text rows whose languages are DECLARED by the repo (`la`, `lb` are our codes). Both sides are audited with GlotLID: a confident
    verdict for another language drops the pair (`lid_mismatch:<side>:<declared>><detected>`); an unconfirmed side is kept and marked unsure.
    `scores` (AfriCOMET-QE, one per row in iteration order) gates on `min_score`. Both directions are emitted unless both=False."""
    for code in (la, lb):
        if code not in A.VALID_LANGS or code not in __import__("africomet_translation.instructions", fromlist=["x"]).DISPLAY_NAMES:
            n = sum(1 for _ in rows)                                       # the language has no tag in the tokenizer (yet): count the rows, write none
            for _ in range(n):
                yield Skip(f"language_has_no_tag:{code}")
            return
    kept, idx, nrows = 0, -1, 0
    for chunk in batched(rows, 256):
        cand = []
        for r in chunk:
            idx += 1
            nrows += 1
            if row_ok is not None and not row_ok(r):
                yield Skip("row_filter")
                continue
            a, b = clean(_get(r, ka)), clean(_get(r, kb))
            if scores is not None:
                s = float(scores[idx]) if idx < len(scores) else float("nan")
                if s != s:
                    yield Skip("qe_prefiltered")
                    continue
                if min_score is not None and s < min_score:
                    yield Skip("qe_below_threshold")
                    continue
            if not a or not b:
                yield Skip("empty")
            elif a.casefold() == b.casefold():
                yield Skip("identical_sides")
            elif len(a) > max_chars or len(b) > max_chars or len(a) < min_chars or len(b) < min_chars:
                yield Skip("length")
            elif not length_ratio_ok(a, b):
                yield Skip("length_ratio")
            elif degenerate(a) or degenerate(b):
                yield Skip("repetitive")
            elif not aligned(a, b):
                yield Skip("misaligned_names_numbers")
            else:
                cand.append((a, b, scores[idx] if scores is not None else None, r))
        if not cand:
            continue
        if audit_langs:
            va, vb = audit([c[0] for c in cand], la), audit([c[1] for c in cand], lb)
        else:
            from sft_new.langaudit import Verdict
            va = vb = [Verdict("unconfirmed")] * len(cand)
        for (a, b, s, r), x, y in zip(cand, va, vb):
            if not x.ok:
                yield Skip(f"lid_mismatch:src:{la}>{(x.detected or '?')[:3]}")
                continue
            if not y.ok:
                yield Skip(f"lid_mismatch:tgt:{lb}>{(y.detected or '?')[:3]}")
                continue
            m = dict(meta or {})
            if s is not None:
                m["qe"] = round(float(s), 3)
            for rec in translation_recs(a, b, la, lb, dataset, rng, x.unsure, y.unsure, both=both, meta=m):
                yield rec
            kept += 1
        if limit and kept >= limit:
            break
    if scores is not None and not limit and nrows != len(scores):
        raise RuntimeError(f"row count {nrows} differs from the QE score array ({len(scores)}): the scores do not belong to this iteration order")


# ------------------------------------------------------------------ single texts with a declared language

def resolve(texts: list, declared: str | None) -> list:
    """For each text -> (code | None, unsure). With a declared language it is audited: confirmed/unconfirmed keep it (unsure flag), a CONFIDENT other
    language that we carry replaces it, a confident language we do not carry gives None. Without one the detector decides (None when it cannot)."""
    ds = L.detect_many([t[:1500] for t in texts])
    out = []
    for t, d in zip(texts, ds):
        if declared:
            v = judge(d, declared, text=t)
            if v.ok:
                out.append((declared, v.unsure))
            elif len(t.split()) <= 8 and label_code(d.label) is None:      # "Molo" is not Toba Batak: short chat turns are too short for a confident verdict
                out.append((declared, True))
            else:
                out.append((label_code(d.label), False))
        elif d.code and d.level != "none":
            out.append((d.code, d.level != "certain"))
        else:
            c = label_code(d.label) if d.prob >= 0.55 else None
            out.append((c, False) if c else (None, False))
    return out


def classify_rec(text: str, label: str, lang: str, unsure: bool, dataset: str, rng: random.Random, task: str, instr_task: str, plan: list, label_token: str | None,
                 meta: dict | None = None, slots: dict | None = None, answer_lang: str = "eng") -> Rec:
    ins = T.make(instr_task, rng, target=answer_lang, source=lang, prefer=lang if lang in T.languages(instr_task) else None, **(slots or {}))
    return Rec(task, dataset, lang, answer_lang, label, input=text, instruction=ins.text, position=ins.position, plan=list(plan), label=label_token,
               unsure_src=unsure, meta={**(meta or {}), "instruction_language": ins.language})


def label_stream(rows: Iterable[dict], text_key, label_of, dataset: str, rng: random.Random, *, task: str, instr_task: str, plan: list, label_token: str | None,
                 declared=None, limit: int | None = None, min_chars: int = 4, max_chars: int = 6000, meta: dict | None = None, slots=None, row_ok=None) -> Iterator:
    """Texts with a class label. `declared` is a language code, or a callable row -> code (None = detect)."""
    kept = 0
    for chunk in batched(rows, 256):
        items = []
        for r in chunk:
            if row_ok is not None and not row_ok(r):
                yield Skip("row_filter")
                continue
            t = clean(_get(r, text_key))
            lab = label_of(r)
            if not t or lab is None or not str(lab).strip():
                yield Skip("empty_text_or_label")
            elif len(t) < min_chars:
                yield Skip("too_short")
            elif len(t) > max_chars or degenerate(t):
                yield Skip("too_long_or_repetitive")
            else:
                items.append((t, str(lab).strip(), r))
        if not items:
            continue
        by_decl: dict = {}
        for i, (_, _, r) in enumerate(items):
            by_decl.setdefault(declared(r) if callable(declared) else declared, []).append(i)
        res = [None] * len(items)
        for decl, ids in by_decl.items():
            for i, v in zip(ids, resolve([items[i][0] for i in ids], decl)):
                res[i] = v
        for (t, lab, r), (code, unsure) in zip(items, res):
            if not code:
                yield Skip("language_unsupported_or_mislabelled")
                continue
            if code not in A.VALID_LANGS or code not in __import__("africomet_translation.instructions", fromlist=["x"]).DISPLAY_NAMES:
                yield Skip(f"language_has_no_tag:{code}")
                continue
            yield classify_rec(t, lab, code, unsure, dataset, rng, task, instr_task, plan, label_token, meta=meta, slots=slots)
            kept += 1
        if limit and kept >= limit:
            return


def read_tsv(repo: str, path: str, columns=None, header=True) -> Iterator[dict]:
    """Rows of a tab-separated file on the Hub, streamed (quotes are not special)."""
    import csv
    import io
    with hubio.fs().open(f"datasets/{repo}/{path}", "rb") as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="")
        rd = csv.reader(text, delimiter="\t", quoting=csv.QUOTE_NONE)
        head = next(rd) if header else (columns or [])
        head = [h.strip() for h in head]
        for row in rd:
            if len(row) < len(head):
                continue
            yield dict(zip(head, row))


def files_of(repo: str, patterns=(".parquet", ".jsonl", ".json", ".csv", ".tsv")) -> list:
    from huggingface_hub import HfApi
    info = HfApi(token=hubio.token()).dataset_info(repo, files_metadata=True)
    return sorted((s.rfilename, s.size or 0) for s in info.siblings if s.rfilename.endswith(patterns) and not s.rfilename.startswith("."))


def iter_any(repo: str, path: str, columns=None) -> Iterator[dict]:
    if path.endswith(".tsv"):
        yield from read_tsv(repo, path)
    else:
        yield from hubio.iter_file(repo, path, columns)


def iter_files(repo: str, paths: list, columns=None) -> Iterator[dict]:
    for p in paths:
        for r in iter_any(repo, p, columns):
            r["__file"] = p
            yield r


def stem_language(path: str, depth: int = 0) -> str | None:
    """'afr-eng/train-0000.parquet' etc. are handled per repo; this is for 'data/yor/train.parquet' / 'yor_Latn/...' style folder names."""
    parts = path.split("/")
    seg = parts[-2 - depth] if len(parts) > 1 + depth else parts[0]
    return code_of(seg.split("_")[0])


_ANSWER = re.compile(r"\s*<answer>\s*(.*?)\s*</answer>\s*", re.S | re.I)


def unwrap_markup(text: str) -> str:
    """Math answers wrapped in light markup: `<answer>X</answer>` becomes a final `#### X` line (the GSM8K convention used elsewhere in the mix);
    <sub>/<sup> become _x / ^x; <br> a newline; purely presentational tags (<b>, <i>, <p>, <solution_step>...) are removed. Not applied to text
    where the markup is content (software strings)."""
    t = text or ""
    t = _ANSWER.sub(lambda m: "\n\n#### " + m.group(1).strip() + "\n", t)
    t = re.sub(r"<sub>(.*?)</sub>", lambda m: "_" + m.group(1), t, flags=re.S | re.I)
    t = re.sub(r"<sup>(.*?)</sup>", lambda m: "^" + m.group(1), t, flags=re.S | re.I)
    t = re.sub(r"</?br\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"</?(?:b|i|u|em|strong|p|span|div|solution_step|step|solution|explanation|think|reasoning)\s*[^<>]*>", "", t, flags=re.I)
    return t.strip()


def iter_arrow(repo: str, path: str, columns=None, batch_rows: int = 2000):
    """Rows of a Hugging Face `datasets` .arrow file (IPC stream), read through the Hub filesystem without caching it."""
    import pyarrow as pa
    with hubio.fs().open(f"datasets/{repo}/{path}", "rb") as fh:
        try:
            rd = pa.ipc.open_stream(fh)
            batches = iter(rd)
        except pa.ArrowInvalid:
            fh.seek(0)
            rd = pa.ipc.open_file(fh)
            batches = (rd.get_batch(i) for i in range(rd.num_record_batches))
        for b in batches:
            names = b.schema.names
            cols = [c for c in (columns or names) if c in names]
            d = {c: b.column(names.index(c)).to_pylist() for c in cols}
            for i in range(b.num_rows):
                yield {c: d[c][i] for c in cols}


def iter_json_array(repo: str, path: str, chunk: int = 1 << 23):
    """Objects of a (possibly multi-GB) JSON array file on the Hub, one at a time, never holding more than one chunk and one object."""
    import codecs
    dec = json.JSONDecoder()
    inc = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf, pos, started, eof = "", 0, False, False
    with hubio.fs().open(f"datasets/{repo}/{path}", "rb") as fh:
        while True:
            if pos > (1 << 24):                               # drop what has been consumed
                buf, pos = buf[pos:], 0
            while True:
                # skip separators
                while pos < len(buf) and buf[pos] in " \n\r\t,":
                    pos += 1
                if pos < len(buf) and not started and buf[pos] == "[":
                    started, pos = True, pos + 1
                    continue
                if pos < len(buf) and buf[pos] == "]":
                    return
                if pos >= len(buf):
                    break
                try:
                    obj, end = dec.raw_decode(buf, pos)
                except ValueError:
                    break                                      # incomplete object: read more
                yield obj
                pos = end
            if eof:
                return
            data = fh.read(chunk)
            if not data:
                eof = True
                buf += inc.decode(b"", final=True)
                continue
            buf += inc.decode(data)


ISO3 = re.compile(r"^[a-z]{3}$")


def lang(name) -> str | None:
    """Our code for a language name / ISO code; an unknown 3-letter code is kept as it is (the build then reports it as a language without a tag)."""
    if name is None:
        return None
    s = str(name).strip()
    c = code_of(s)
    if c:
        return c
    s = s.lower().replace("-", "_").split("_")[0]
    return s if ISO3.match(s) else None


_THINK_PATTERNS = [
    re.compile(r"^\s*<think>\s*(?P<t>.*?)\s*</think>\s*(?P<a>.*?)\s*$", re.S | re.I),
    re.compile(r"^\s*<thinking>\s*(?P<t>.*?)\s*</thinking>\s*(?P<a>.*?)\s*$", re.S | re.I),
    re.compile(r"^\s*<reasoning>\s*(?P<t>.*?)\s*</reasoning>\s*(?P<a>.*?)\s*$", re.S | re.I),
    re.compile(r"^\s*<scratchpad>\s*(?P<t>.*?)\s*</scratchpad>\s*(?P<a>.*?)\s*$", re.S | re.I),
    re.compile(r"^\s*<reflection>\s*(?P<t>.*?)\s*</reflection>\s*(?P<a>.*?)\s*$", re.S | re.I),
    re.compile(r"^\s*<\|begin_of_thought\|>\s*(?P<t>.*?)\s*<\|end_of_thought\|>\s*(?:<\|begin_of_solution\|>)?\s*(?P<a>.*?)\s*(?:<\|end_of_solution\|>)?\s*$", re.S),
    re.compile(r"^\s*<\|channel\|>analysis<\|message\|>\s*(?P<t>.*?)\s*<\|end\|>\s*(?:<\|start\|>assistant)?(?:<\|channel\|>final<\|message\|>)?\s*(?P<a>.*?)\s*(?:<\|return\|>)?\s*$", re.S),
]


def split_think(text: str):
    """Reasoning markup of other families (<think>, <thinking>, <reasoning>, <scratchpad>, <reflection>, DeepSeek-style begin_of_thought, gpt-oss analysis channel)
    -> (reasoning, answer) so it can be written as OUR <think> field; (None, text) when there is none or it is unbalanced."""
    t = text or ""
    for rx in _THINK_PATTERNS:
        m = rx.match(t)
        if m and m.group("a") and m.group("t"):
            return m.group("t").strip(), m.group("a").strip()
    return None, t
