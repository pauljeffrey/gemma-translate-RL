"""Converters for the repos of the owner's HF accounts that crawl_plan.py marked `convert:<shape>` (one registered step per account).

Rules as everywhere: licence / NLLB / Llama-4 / RL / coding / evaluation verdicts were taken by the planner; here every repo is read in a streaming way,
capped, language-identified per row (languages we carry only), tasks come from the columns, `enhanced_*` / reasoning-trace / checklist columns (RL material)
are never used. A repo whose first rows are not in a language we carry is dropped after 60 rows (counted as `repo_language_not_carried`).
"""
from __future__ import annotations

import json
import os
import re

from sft_new import hubio
from sft_new import langdetect as L
from sft_new import templates as T
from sft_new.compose import A, Rec, Skip
from sft_new.converters import register
from sft_new import ext_instruct as EI
from sft_new.ext_common import batched, code_of, files_of, iter_any, pairs, resolve
from sft_new.ext_common import lang
from sft_new.textutil import blank, clean, degenerate

PLAN = os.environ.get("SFT_CRAWL_PLAN", "/workspace/sft_build/crawl_plan.json")
REPO_CAP = 25_000
SPLIT_SKIP = re.compile(r"(^|[/_.-])(test|dev|valid|validation|eval|val)([/_.-]|$)", re.I)


def _plan(account: str) -> list:
    try:
        return [p for p in json.load(open(PLAN))["convert"] if p["account"] == account]
    except OSError:
        return []


def _name_lang(repo: str) -> str | None:
    for tok in re.split(r"[-_./]", repo.split("/", 1)[1].lower()):
        c = lang(tok) if len(tok) > 2 else None
        if c and c != "eng" and (c in L.SUPPORTED_SET or c in A.VALID_LANGS):
            return c
    return None


def repo_rows(repo: str, cap: int):
    paths = [p for p, _ in files_of(repo, patterns=(".parquet", ".jsonl", ".json", ".csv", ".tsv")) if not SPLIT_SKIP.search(p)]
    paths.sort(key=lambda p: ("train" not in p, p))
    n = 0
    for p in paths:
        try:
            for r in iter_any(repo, p):
                yield r
                n += 1
                if n >= cap:
                    return
        except Exception:
            continue


def _first(r: dict, names, text=True):
    for k in names:
        if k in r and r[k] is not None:
            v = r[k]
            if isinstance(v, dict) and "text" in v:
                v = v["text"]
            if isinstance(v, (list, tuple)):
                v = v[0] if v and isinstance(v[0], str) else ""
            if isinstance(v, str) and blank(v):
                return v
    return ""


def _gate(rows: list, texts) -> bool:
    """True when at least a third of the sample is in a language we carry."""
    sample = [t[:500] for t in (texts(r) for r in rows) if t][:60]
    if not sample:
        return False
    ok = sum(1 for c, _ in resolve(sample, None) if c)
    return ok >= max(3, len(sample) // 3)


def convert_repo(entry: dict, rng, cap: int):
    repo, shape = entry["repo"], entry["shape"]
    it = repo_rows(repo, cap)
    head = []
    for r in it:
        head.append(r)
        if len(head) >= 60:
            break
    if not head:
        yield Skip("no_rows")
        return
    keys = set(head[0])
    import itertools
    rows = itertools.chain(head, it)
    meta = {"crawl_account": entry["account"]}
    if shape == "chat":
        col = next(k for k in ("messages", "conversations", "conversation") if k in keys)
        def getm(r):
            m = r.get(col)
            if isinstance(m, str):
                try:
                    m = json.loads(m)
                except ValueError:
                    return None
            return m
        if not _gate([{"t": " ".join(str(x.get("content") or x.get("value") or "") for x in (getm(r) or [])[:3])} for r in head], lambda r: r["t"]):
            yield Skip("repo_language_not_carried")
            return
        for r in rows:
            m = getm(r)
            yield EI.chat_convo(m, repo, rng, declared=None, meta=meta) if m else Skip("shape")
    elif shape == "instruction":
        ik = next((k for k in ("instruction", "prompt", "inputs") if k in keys), None)
        ok = next((k for k in ("output", "response", "completion", "targets", "answer") if k in keys), None)
        ink = next((k for k in ("input", "context") if k in keys and k != ik), None)
        if not ik or not ok:
            yield Skip("no_instruction_columns")
            return
        decl = _name_lang(repo)
        if not decl and not _gate(head, lambda r: _first(r, (ik,)) or _first(r, (ok,))):
            yield Skip("repo_language_not_carried")
            return
        taco = "taco" in repo.lower()
        if taco:
            for chunk in batched(rows, 128):
                items = []
                for r in chunk:
                    out = blank(r.get(ok)) or blank(r.get("text"))
                    m = EI.TACO.search(out or "")
                    ins, inp = blank(r.get(ik)), blank(r.get(ink)) if ink else ""
                    if not ins or not m:
                        yield Skip("no_taco_structure")
                        continue
                    items.append((ins, inp, EI.clean_keep_lines(m.group("ie")), EI.clean_keep_lines(m.group("re")), EI.clean_keep_lines(m.group("ans"))))
                if not items:
                    continue
                ci = resolve([i[0][:500] for i in items], decl)
                ca = resolve([i[4][:500] for i in items], decl)
                for (ins, inp, ie, re_, ans), (c1, u1), (c2, u2) in zip(items, ci, ca):
                    if not c1 or not c2:
                        yield Skip("language_unsupported_or_undetermined")
                        continue
                    from sft_new.taskguess import guess
                    task, plan = guess(ie, inp)
                    yield Rec(task, repo, c1, c2, ans, input="", prompt_override=EI.join_prompt(ins, inp), plan=plan, think=f"Instruction in English: {ie}\nResponse in English: {re_}",
                              input_langs=[c1], unsure_src=u1, unsure_tgt=u2, meta={**meta, "pivot": "english"})
        else:
            yield from EI.instruct_rows(rows, ik, ink, ok, repo, rng, declared=decl, meta=meta)
    elif shape == "qa":
        qk = next((k for k in ("question", "question_text", "query") if k in keys), None)
        ak = next((k for k in ("answer", "answers", "explanation", "response", "output") if k in keys), None)
        ck = next((k for k in ("context", "passage", "document", "supporting_details", "original_context") if k in keys), None)
        if not qk or not ak:
            yield Skip("no_qa_columns")
            return
        decl = _name_lang(repo)
        if not decl and not _gate(head, lambda r: _first(r, (qk,))):
            yield Skip("repo_language_not_carried")
            return
        for chunk in batched(rows, 128):
            items = []
            for r in chunk:
                q, a = clean(_first(r, (qk,))), EI.clean_keep_lines(_first(r, (ak,)))
                c = EI.clean_keep_lines(_first(r, (ck,))) if ck else ""
                if len(q) < 6 or not a or len(a) > 6000 or len(c) > 7000 or degenerate(a):
                    yield Skip("empty_or_long")
                    continue
                items.append((q, a, c))
            if not items:
                continue
            for (q, a, c), (code, unsure) in zip(items, resolve([q[:300] for q, _, _ in items], decl)):
                if not code:
                    yield Skip("language_unsupported_or_undetermined")
                    continue
                if c and len(c) > 60 and rng.random() < 0.75:               # owner 2026-10-11: context QA is modelled as RAG through function calls
                    from sft_new.rag_tools import rag_convo
                    yield rag_convo(q, c, a, code, repo, rng, unsure=unsure, extra_meta=meta)
                    continue
                if c and len(c) > 60:
                    ins = T.make("qa_context", rng, target=code, source=code, prefer=code if code in T.languages("qa_context") else None)
                    body = f"{ins.text}\n\n{c}\n\n{q}" if rng.random() < 0.5 else f"{ins.text}\n\n{q}\n\n{c}"
                    yield Rec("reading_comprehension", repo, code, code, a, input="", prompt_override=body, plan=["<qa>"], label="<answer>",
                              input_langs=list(dict.fromkeys([ins.language, code]))[:3], unsure_src=unsure, unsure_tgt=unsure, meta={**meta, "instruction_language": ins.language})
                else:
                    yield Rec("question_answering", repo, code, code, a, input="", prompt_override=q, plan=["<qa>"], unsure_src=unsure, unsure_tgt=unsure, meta=meta)
    elif shape == "translation":
        sk = next((k for k in ("source", "src") if k in keys), None)
        tk = next((k for k in ("target", "tgt") if k in keys), None)
        if "translation" in keys and not sk:
            rows = ({"source": (r.get("translation") or {}).get("en") if isinstance(r.get("translation"), dict) else None,
                     "target": next((v for k, v in (r.get("translation") or {}).items() if k != "en"), None)} for r in rows)
            sk, tk = "source", "target"
        if not sk or not tk:
            yield Skip("no_translation_columns")
            return
        for chunk in batched(rows, 256):
            items = [(clean(_first(r, (sk,))), clean(_first(r, (tk,)))) for r in chunk]
            items = [(a, b) for a, b in items if len(a) > 6 and len(b) > 6]
            if not items:
                continue
            da, db = resolve([a[:500] for a, _ in items], None), resolve([b[:500] for _, b in items], None)
            by: dict = {}
            for (a, b), (ca, _), (cb, _) in zip(items, da, db):
                if ca and cb and ca != cb:
                    by.setdefault((ca, cb), []).append({"a": a, "b": b})
                else:
                    yield Skip("language_pair_not_carried")
            for (ca, cb), rs in by.items():
                yield from pairs(iter(rs), "a", "b", ca, cb, repo, rng, meta=meta)


def _make(account: str, label: str):
    def fn(rng, limit=None):
        for entry in _plan(account):
            try:
                yield from convert_repo(entry, rng, limit or REPO_CAP)
            except Exception as e:                                         # one broken repo must not end the step
                yield Skip(f"repo_error:{entry['repo'].split('/')[1][:30]}:{type(e).__name__}")
    fn.__name__ = f"crawl_{account}"
    register(f"crawl_{account.lower().replace('-', '_')}", f"{label}: every repo of the account that crawl_plan.py marked convert (capped {REPO_CAP:,} rows each)")(fn)


for _acct, _label in (("saillab", "saillab"), ("Svngoku", "Svngoku"), ("McGill-NLP", "McGill-NLP"), ("ghananlpcommunity", "ghananlpcommunity"),
                      ("ghanaopenai", "ghanaopenai"), ("Kppwdfgu1", "Kppwdfgu1"), ("michaeldinzinger", "michaeldinzinger"), ("FrancophonIA", "FrancophonIA")):
    _make(_acct, _label)


@register("francophonia_pairs", "FrancophonIA/Kabyle-French-pairs (tsv) and bambara-french (json): two text columns, languages from GlotLID -> translation (the PDF dictionaries of the account are not parsed)")
def francophonia_pairs(rng, limit=None):
    import itertools
    for repo in ("FrancophonIA/Kabyle-French-pairs", "FrancophonIA/bambara-french"):
        paths = [p for p, _ in files_of(repo, patterns=(".tsv", ".json", ".csv", ".jsonl"))]
        for p in paths:
            rows = list(itertools.islice(iter_any(repo, p), 200_000))
            if not rows:
                continue
            texty = [k for k in rows[0] if isinstance(rows[0][k], str) and k.lower() not in ("id", "idx", "index")]
            if len(texty) < 2:
                yield Skip("no_two_text_columns")
                continue
            ka, kb = texty[0], texty[1]
            for chunk in batched(iter(rows), 256):
                items = [(clean(r.get(ka)), clean(r.get(kb))) for r in chunk]
                items = [(a, b) for a, b in items if len(a) > 3 and len(b) > 3]
                if not items:
                    continue
                da, db = resolve([a[:500] for a, _ in items], None), resolve([b[:500] for _, b in items], None)
                by: dict = {}
                for (a, b), (ca, _), (cb, _) in zip(items, da, db):
                    if ca and cb and ca != cb:
                        by.setdefault((ca, cb), []).append({"a": a, "b": b})
                    else:
                        yield Skip("language_pair_not_carried")
                for (ca, cb), rs in by.items():
                    yield from pairs(iter(rs), "a", "b", ca, cb, repo, rng, meta={"crawl_account": "FrancophonIA"}, min_chars=3)
