"""Converters for the datasets of the catalog worker's `sft/` folder that `catalog_plan.py` marked `convert:<shape>` (owner 2026-10-11).

One registered step per shape (translation is split into four so the queue workers can share it). Every repo is streamed and capped, its language comes
from the catalog (single African language) or GlotLID per row; rows in languages without a tag are dropped and counted. Labels that are bare class ids are
used only when the repo's README names the classes. Licence `unknown` / unspecified is used and logged (the plan carries `licence_flag`).
"""
from __future__ import annotations

import itertools
import json
import os
import re
import zlib

from sft_new import hubio
from sft_new import langdetect as L
from sft_new import templates as T
from sft_new.compose import A, Rec, Skip
from sft_new.converters import register
from sft_new import ext_instruct as EI
from sft_new.ext_common import batched, label_stream, lang, pairs, resolve
from sft_new.ext_crawl import _first, _gate, _name_lang, repo_rows
from sft_new.rag_tools import rag_convo
from sft_new.textutil import blank, clean, degenerate

PLAN = os.environ.get("SFT_CATALOG_PLAN", "/workspace/sft_build/catalog_plan.json")
CAP = {"translation": 150_000, "label": 60_000, "chat": 50_000, "instruction": 60_000, "qa": 60_000, "summ": 50_000, "tags": 40_000, "retrieval": 60_000}
KIND = {"sentiment": ("sentiment", "sentiment", "<sentiment>"), "emotion": ("emotion", "emotion", "<answer>"), "topic": ("topic", "topic", "<topic>"),
        "hate": ("hate", "hate", "<answer>"), "intent": ("intent", "intent", "<intent>")}
PIVOT = {"eng", "fra"}


def _entries(shape: str, shard: int = 0, nshards: int = 1, rnd: str = "") -> list:
    path = PLAN if not rnd else PLAN.replace(".json", f"_{rnd}.json")
    try:
        es = [e for e in json.load(open(path))["convert"] if e["shape"] == shape]
    except OSError:
        return []
    return [e for e in es if zlib.crc32(e["repo"].encode()) % nshards == shard]


def _decl(entry) -> str | None:
    """The single African language of a repo, when the catalog names exactly one (pivot languages aside)."""
    langs = [lang(x) for x in entry.get("langs") or []]
    langs = [c for c in langs if c and c not in PIVOT]
    return langs[0] if len(set(langs)) == 1 else None


_CL_CACHE: dict = {}


def class_names(repo: str, col: str):
    """ClassLabel names of `col` from the README's dataset_info (dict id -> name), or None."""
    key = (repo, col)
    if key in _CL_CACHE:
        return _CL_CACHE[key]
    names = None
    try:
        import yaml
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo, "README.md", repo_type="dataset", token=hubio.token(), cache_dir=os.path.join(os.environ.get("HF_HOME", "/tmp"), "tmp_readme"))
        front = open(p, encoding="utf-8").read().split("---", 2)[1]
        info = yaml.safe_load(front).get("dataset_info")
        info = info[0] if isinstance(info, list) else info
        for f in info.get("features", []):
            if f.get("name") != col:
                continue
            cl = f.get("dtype", {}) if isinstance(f.get("dtype"), dict) else {}
            cl = cl.get("class_label") or f.get("class_label") or (f.get("list") or {}).get("class_label") or (f.get("sequence") or {}).get("class_label")
            if isinstance(cl, dict) and cl.get("names"):
                n = cl["names"]
                names = {int(k): str(v) for k, v in n.items()} if isinstance(n, dict) else {i: str(v) for i, v in enumerate(n)}
    except Exception:
        names = None
    _CL_CACHE[key] = names
    return names


def _convert(entry: dict, rng, cap: int):
    repo, shape, cols = entry["repo"], entry["shape"], entry.get("cols") or []
    meta = {"catalog": True, "licence": entry.get("licence"), "licence_flag": entry.get("licence_flag")}
    decl = _decl(entry)
    it = repo_rows(repo, cap)
    head = list(itertools.islice(it, 60))
    if not head:
        yield Skip("no_rows")
        return
    rows = itertools.chain(head, it)
    keys = set(head[0])
    if shape == "chat":
        col = cols[0] if cols and cols[0] in keys else next((k for k in ("messages", "conversations", "conversation", "text") if k in keys), None)
        if not col:
            yield Skip("no_chat_column")
            return
        if "tools" in keys and any('"function_call"' in json.dumps(r.get(col), ensure_ascii=False)[:2000] for r in head[:5]):
            yield from _tool_calls(entry, rows, col, rng, meta, decl)
            return

        def getm(r):
            m = r.get(col)
            if isinstance(m, str):
                try:
                    m = json.loads(m)
                except ValueError:
                    return None
            return m if isinstance(m, list) else None
        if not decl and not _gate([{"t": " ".join(str(x.get("content") or x.get("value") or "") for x in (getm(r) or [])[:3] if isinstance(x, dict))} for r in head], lambda r: r["t"]):
            yield Skip("repo_language_not_carried")
            return
        for r in rows:
            m = getm(r)
            yield EI.chat_convo(m, repo, rng, declared=decl, meta=meta) if m else Skip("shape")
    elif shape == "instruction":
        ik, ok, ink = (cols + ["", "", ""])[:3]
        ink = ink or None
        if ik not in keys or ok not in keys:
            yield Skip("no_instruction_columns")
            return
        if not decl and not _gate(head, lambda r: _first(r, (ik,)) or _first(r, (ok,))):
            yield Skip("repo_language_not_carried")
            return
        yield from EI.instruct_rows(rows, ik, ink, ok, repo, rng, declared=decl, meta=meta)
    elif shape == "qa":
        qk = next((k for k in ("question", "question_text", "query") if k in keys), None)
        ak = next((k for k in ("answer", "answers", "explanation", "response", "output") if k in keys), None)
        ck = next((k for k in ("context", "passage", "document", "supporting_details", "original_context", "article") if k in keys), None)
        if not qk or not ak:
            yield Skip("no_qa_columns")
            return
        if not decl and not _gate(head, lambda r: _first(r, (qk,))):
            yield Skip("repo_language_not_carried")
            return
        for chunk in batched(rows, 128):
            items = []
            for r in chunk:
                q, a = clean(_first(r, (qk,))), EI.clean_keep_lines(_first(r, (ak,)))
                a = re.sub(r"<<[^<>]*>>", "", a)                            # GSM8K calculator annotations
                c = EI.clean_keep_lines(_first(r, (ck,))) if ck else ""
                if len(q) < 6 or not a or len(a) > 6000 or len(c) > 7000 or degenerate(a):
                    yield Skip("empty_or_long")
                    continue
                url = next((r[k] for k in ("url", "link", "source_url", "page_url", "article_url", "source", "doc_url") if isinstance(r.get(k), str) and r[k].startswith("http")), None)
                items.append((q, a, c, url))
            if not items:
                continue
            for (q, a, c, url), (code, unsure) in zip(items, resolve([q[:300] for q, _, _, _ in items], decl)):
                if not code:
                    yield Skip("language_unsupported_or_undetermined")
                    continue
                if c and len(c) > 60 and rng.random() < 0.75:               # owner 2026-10-11: context QA is modelled as RAG through function calls
                    rec = rag_convo(q, c, a, code, repo, rng, url=url, unsure=unsure, extra_meta=meta)
                    yield rec
                    continue
                if c and len(c) > 60:
                    ins = T.make("qa_context", rng, target=code, source=code, prefer=code if code in T.languages("qa_context") else None)
                    body = f"{ins.text}\n\n{c}\n\n{q}" if rng.random() < 0.5 else f"{ins.text}\n\n{q}\n\n{c}"
                    yield Rec("reading_comprehension", repo, code, code, a, input="", prompt_override=body, plan=["<qa>"], label="<answer>",
                              input_langs=list(dict.fromkeys([ins.language, code]))[:3], unsure_src=unsure, unsure_tgt=unsure, meta={**meta, "instruction_language": ins.language})
                else:
                    yield Rec("question_answering", repo, code, code, a, input="", prompt_override=q, plan=["<qa>"], unsure_src=unsure, unsure_tgt=unsure, meta=meta)
    elif shape == "translation":
        yield from _translation(repo, cols, rows, keys, rng, meta, decl)
    elif shape == "label":
        yield from _label(entry, rows, keys, rng, meta, decl, cap)
    elif shape == "summ":
        body, summ, title = (cols + ["", "", ""])[:3]
        if body not in keys:
            yield Skip("no_text_column")
            return
        for chunk in batched(rows, 64):
            items = []
            for r in chunk:
                t = EI.clean_keep_lines(blank(r.get(body)))
                su = clean(r.get(summ)) if summ else ""
                ti = clean(r.get(title)) if title else ""
                if len(t) < 200 or len(t) > 12000 or not (su or ti):
                    yield Skip("no_text_or_target")
                    continue
                items.append((t, su, ti))
            if not items:
                continue
            for (t, su, ti), (code, unsure) in zip(items, resolve([t[:500] for t, _, _ in items], decl)):
                if not code:
                    yield Skip("language_unsupported_or_undetermined")
                    continue
                if su and 20 <= len(su) < len(t) * 0.8 and not degenerate(su):
                    ins = T.make("summarize", rng, target=code, source=code, prefer=code if code in T.languages("summarize") else None)
                    yield Rec("summarization", repo, code, code, su, input=t[:10000], instruction=ins.text, position=ins.position, plan=["<summarize>"],
                              unsure_src=unsure, unsure_tgt=unsure, meta={**meta, "instruction_language": ins.language})
                if ti and 8 <= len(ti) <= 300 and not degenerate(ti):
                    ih = T.make("headline", rng, target=code, source=code, prefer=code if code in T.languages("headline") else None)
                    yield Rec("headline", repo, code, code, ti, input=t[:6000], instruction=ih.text, position=ih.position, plan=["<summarize>"], label="<headline>",
                              unsure_src=unsure, unsure_tgt=unsure, meta={**meta, "instruction_language": ih.language})
    elif shape == "tags":
        yield from _tags(entry, rows, keys, rng, meta, decl)
    elif shape == "retrieval":
        yield from _retrieval(entry, rows, keys, rng, meta, decl)


def _tool_calls(entry, rows, col, rng, meta, decl):
    """Single-turn function calling: [human, function_call] + a `tools` column (JSON list of {"type": "function", "function": {...}}).  The row ends at the call
    (meta.call_only): the model learns to pick a function and write its arguments; one call per row, the called function must be in `tools`."""
    from sft_new.converters import tools_with_distractors
    from sft_new.ext_instruct import call_plan
    repo = entry["repo"]
    system = {"role": "system", "content": "You are a helpful assistant. When you need information or an action you cannot produce yourself, call one of the functions listed below."}
    for chunk in batched(rows, 128):
        items = []
        for r in chunk:
            m, tools = r.get(col), r.get("tools")
            try:
                m = json.loads(m) if isinstance(m, str) else m
                tools = json.loads(tools) if isinstance(tools, str) else tools
            except ValueError:
                yield Skip("bad_json")
                continue
            if not isinstance(m, list) or len(m) != 2 or str(m[0].get("from") or m[0].get("role")).lower() not in ("human", "user") or m[1].get("from") != "function_call":
                yield Skip("shape")
                continue
            try:
                calls = json.loads(m[1].get("value") or "")
            except ValueError:
                yield Skip("bad_call_json")
                continue
            if isinstance(calls, dict):
                calls = [calls]
            if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
                yield Skip("multi_call")                                   # sequential single calls are the supported pattern (TOOL_CALL_FORMAT.md section 4)
                continue
            name, args = calls[0].get("name"), calls[0].get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    yield Skip("bad_arguments")
                    continue
            if not isinstance(tools, list) or not name or not isinstance(args, dict):
                yield Skip("no_tools_or_call")
                continue
            tl = [t for t in tools if isinstance(t, dict) and isinstance(t.get("function"), dict)]
            by_name = {t["function"].get("name"): t for t in tl}
            text = clean(m[0].get("value") or m[0].get("content"))
            if name not in by_name or len(text) < 6:
                yield Skip("call_not_in_tools_or_empty_prompt")
                continue
            items.append((text, name, args, tl))
        if not items:
            continue
        for (text, name, args, tl), (code, unsure) in zip(items, resolve([t[:400] for t, _, _, _ in items], decl)):
            if not code:
                yield Skip("language_unsupported_or_undetermined")
                continue
            used = {t["function"]["name"] for t in tl}
            extra = [t for t in tools_with_distractors(next(t for t in tl if t["function"]["name"] == name), rng, 1, 3) if t["function"]["name"] not in used]
            allt = tl + extra[: max(0, 5 - len(tl))]
            rng.shuffle(allt)
            try:
                msg = A.assistant_message({"input_lang": [code], "task_plan": call_plan(name), "tool_call": {"name": name, "arguments": args}})
            except Exception:                                             # noqa: BLE001
                yield Skip("assembly_failed")
                continue
            yield Rec("function_calling", repo, code, code, "", input=text, convo=[system, {"role": "user", "content": text}, msg], tools=allt,
                      unsure_src=unsure, meta={**meta, "call_only": True, "calls": 1})


def _retrieval(entry, rows, keys, rng, meta, decl):
    """(query, positive passage[, link, negative passages]) -> a RAG conversation; the answer is the opening of the passage, as for the Wikipedia RAG rows."""
    from sft_new.ext_rag import sentences
    repo = entry["repo"]
    qk, pk, lk, nk = (entry.get("cols") or ["query", "passage", "", ""])[:4]
    if qk not in keys or pk not in keys:
        yield Skip("no_retrieval_columns")
        return
    for chunk in batched(rows, 64):
        items = []
        for r in chunk:
            q, p = clean(_first(r, (qk,))), EI.clean_keep_lines(_first(r, (pk,)))
            if len(q) < 4 or len(p) < 150 or degenerate(p):
                yield Skip("empty_or_short")
                continue
            sents = sentences(p)
            ans = " ".join(sents[: rng.randint(2, 5)]) if len(sents) > 2 else p
            if len(ans) > 1500:
                ans = ans[:1500].rsplit(" ", 1)[0]
            negs = r.get(nk) if nk else None
            negs = [clean(x) for x in negs if isinstance(x, str) and len(x) > 100][:3] if isinstance(negs, (list, tuple)) else []
            items.append((q, p, ans, r.get(lk) if lk else None, negs))
        if not items:
            continue
        for (q, p, ans, url, negs), (code, unsure) in zip(items, resolve([q[:300] for q, *_ in items], decl)):
            if not code:
                yield Skip("language_unsupported_or_undetermined")
                continue
            yield rag_convo(q, p, ans, code, repo, rng, url=url if isinstance(url, str) else None, unsure=unsure, negatives=negs, extra_meta={**meta, "answer": "passage opening"})


def _translation(repo, cols, rows, keys, rng, meta, decl):
    """Two text columns, or a `translation` dict column ({lang: text})."""
    if len(cols) == 1:                                                     # dict column
        col = cols[0]
        by: dict = {}
        for r in rows:
            d = r.get(col)
            if isinstance(d, str):
                try:
                    d = json.loads(d)
                except ValueError:
                    d = None
            if not isinstance(d, dict) or len(d) < 2:
                continue
            items = [(lang(k), clean(v)) for k, v in d.items() if isinstance(v, str)]
            items = [(c, v) for c, v in items if c and len(v) > 3]
            if len(items) < 2:
                continue
            (ca, a), (cb, b) = items[0], items[1]
            if ca != cb:
                by.setdefault((ca, cb), []).append({"a": a, "b": b})
        for (ca, cb), rs in by.items():
            yield from pairs(iter(rs), "a", "b", ca, cb, repo, rng, meta=meta, min_chars=3)
        return
    ka, kb = cols[:2]
    if ka not in keys or kb not in keys:
        yield Skip("no_translation_columns")
        return
    for chunk in batched(rows, 256):
        items = [(clean(_first(r, (ka,))), clean(_first(r, (kb,)))) for r in chunk]
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
            yield from pairs(iter(rs), "a", "b", ca, cb, repo, rng, meta=meta, min_chars=3)


def _label(entry, rows, keys, rng, meta, decl, cap):
    repo = entry["repo"]
    tk, lk = entry["cols"][:2]
    kind = entry.get("kind") or "topic"
    task, instr, token = KIND.get(kind, KIND["topic"])
    if tk not in keys or lk not in keys:
        yield Skip("no_label_columns")
        return
    names = class_names(repo, lk)
    head_rows = []
    for r in rows:
        head_rows.append(r)
        if len(head_rows) >= 60:
            break
    rows = itertools.chain(head_rows, rows)
    vals = []
    for r in head_rows:
        v = r.get(lk)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and names:
            v = names.get(int(v))
        if isinstance(v, str):
            vals.append(v.strip().lower())
    distinct = set(vals)
    if not vals or len(distinct) < 2 or len(distinct) > 40 or all(re.fullmatch(r"[-+]?\d+(\.\d+)?", v) for v in distinct) and not names:
        yield Skip("label_vocabulary_unsuitable")
        return
    if distinct <= {"positive", "negative", "neutral", "pos", "neg", "neu", "mixed", "very positive", "very negative"}:
        kind = "sentiment"                                                 # the values say it, whatever the column / task name
    elif distinct <= {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral", "love", "happy", "sad", "angry"}:
        kind = "emotion"
    task, instr, token = KIND.get(kind, KIND["topic"])

    def lab(r):
        v = r.get(lk)
        if isinstance(v, bool):
            v = "positive" if v else "negative"
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if names is None:
                return None                                                # bare ids without class names are not a usable label
            v = names.get(int(v))
        if not isinstance(v, str):
            return None
        v = clean(v).replace("_", " ").strip()
        if not v or len(v) > 40:
            return None
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", v) and not names:
            return None
        if kind == "sentiment":
            v = {"pos": "positive", "neg": "negative", "neu": "neutral"}.get(v.lower(), v).capitalize()
        return v
    slots = {"labels": "anger, disgust, fear, joy, sadness, surprise, neutral"} if kind == "emotion" else None
    yield from label_stream(rows, tk, lab, repo, rng, task=task, instr_task=instr, plan=["<classify>"], label_token=token, declared=decl, limit=cap,
                            meta={**meta, "labels": "from the dataset"}, slots=slots)


def _tags(entry, rows, keys, rng, meta, decl):
    repo = entry["repo"]
    tk, gk = entry["cols"][:2]
    kind = entry.get("kind") or "ner"
    if tk not in keys or gk not in keys:
        yield Skip("no_tag_columns")
        return
    names = class_names(repo, gk)
    code = decl or _name_lang(repo)
    if not code or code not in A.VALID_LANGS:
        yield Skip("language_has_no_tag")
        return
    from sft_new.ext_tasks import ner_rec
    for r in rows:
        words, tags = r.get(tk), r.get(gk)
        if not isinstance(words, (list, tuple)) or not isinstance(tags, (list, tuple)) or len(words) != len(tags) or not words:
            yield Skip("shape")
            continue
        tg = []
        for t in tags:
            if isinstance(t, int):
                t = (names or {}).get(t)
            if not isinstance(t, str):
                tg = None
                break
            tg.append("O" if t.strip().upper() in ("OUT", "OUTSIDE", "0", "O") else t.strip())
        if tg is None:
            yield Skip("label_ids_without_names")
            continue
        ws = [clean(w) for w in words]
        if kind == "ner":
            rec = ner_rec(ws, tg, code, True, repo, rng, meta=meta)
        else:
            ins = T.make("pos", rng, target=code, source=code)
            rec = Rec("pos", repo, code, code, " ".join(f"{w} <tag> {t}" for w, t in zip(ws, tg)), input=" ".join(ws), instruction=ins.text, position=ins.position,
                      plan=["<POS>"], sequence_labels=True, unsure_src=True, unsure_tgt=True, meta={**meta, "instruction_language": ins.language}) if len(ws) <= 200 else None
        yield rec if rec else Skip("too_long")


def _make(shape: str, shard: int = 0, nshards: int = 1, suffix: str = "", rnd: str = ""):
    def fn(rng, limit=None):
        for entry in _entries(shape, shard, nshards, rnd):
            try:
                yield from _convert(entry, rng, limit or CAP[shape])
            except Exception as e:                                         # one broken repo must not end the step
                yield Skip(f"repo_error:{entry['repo'].split('/')[1][:30]}:{type(e).__name__}")
    pre = f"catalog_{rnd}_" if rnd else "catalog_"
    fn.__name__ = f"{pre}{shape}{suffix}"
    register(f"{pre}{shape}{suffix}", f"catalog worker's sft/ datasets planned as `{shape}` ({'shard ' + str(shard) + '/' + str(nshards) if nshards > 1 else 'all'}"
              f"{', round ' + rnd if rnd else ''}), capped per repo")(fn)


# round "" is the first plan (catalog_plan.json); later hourly checks write catalog_plan_r2.json, _r3.json ... with only the datasets that are new
for _rnd in ("", "r2", "r3", "r4", "r5", "r6"):
    for _s in ("chat", "instruction", "qa", "label", "summ", "tags", "retrieval"):
        _make(_s, rnd=_rnd)
    for _i in range(4):
        _make("translation", _i, 4, f"_{_i}", rnd=_rnd)
