"""Second-round retrieval / tool-use / reranking sources.

Wikipedia (per language), the Wikipedia summary set and the Cohere passages all become the same shape as wiki_yo: the user asks about a topic, the
assistant calls `search_wikipedia` (among distractor tools), the tool returns the article text with its URL, and the assistant answers from it. Roughly
a third of the rows are the plain encyclopedic answer with no call, so the model also learns to answer from its weights.
"""

from __future__ import annotations

import re

from sft_new import hubio
from sft_new import langdetect as L
from sft_new import templates as T
from sft_new.compose import A, Rec, Skip
from sft_new.converters import SEARCH_TOOLS, WIKI_SYSTEM, register, search_tool_def, tools_with_distractors
from sft_new.ext_common import batched, files_of, iter_arrow, resolve
from sft_new.textutil import blank, clean, degenerate, rep_ratio

ISO1 = {"afr": "af", "amh": "am", "ewe": "ee", "hau": "ha", "ibo": "ig", "kin": "rw", "nya": "ny", "orm": "om", "sna": "sn", "som": "so", "sot": "st",
        "swh": "sw", "tsn": "tn", "twi": "tw", "wol": "wo", "xho": "xh", "yor": "yo", "zul": "zu", "eng": "en", "fon": "fon", "pcm": "pcm", "ti": "ti",
        "fra": "fr"}
CAP = {"eng": 60_000, "swh": 40_000, "afr": 40_000, "fra": 0, "default": 30_000}


def sentences(passage: str) -> list:
    return [s for s in re.split(r"(?<=[.!?።])\s+", passage) if s.strip()]


def rag_rec(title: str, passage: str, url: str | None, lang: str, unsure: bool, dataset: str, rng, iso: str, kind: str = "wikipedia") -> "Rec | Skip":
    ins = T.make("wiki_ask", rng, target=lang, source=lang, prefer=lang if lang in T.languages("wiki_ask") else None, position="prepend", title=title)
    qlang = ins.language
    sents = sentences(passage)
    answer = " ".join(sents[:rng.randint(3, 7)]) if len(sents) > 3 else passage
    if rng.random() < 0.35:                                          # no tool: an encyclopedic answer
        return Rec("encyclopedic_qa", dataset, qlang, lang, passage[:5000], instruction=ins.text, position="prepend", plan=["<|explain|>"],
                   input_langs=list(dict.fromkeys([qlang, lang])), unsure_tgt=unsure, meta={"instruction_language": qlang, "title": title, "source": kind})
    name, arg = rng.choice(SEARCH_TOOLS)
    shown = passage[:rng.choice([1200, 2000, 3000])]
    result = f"{title}\n{url}\n\n{shown}" if url else f"{title}\n\n{shown}"
    try:
        call = A.assistant_message({"input_lang": [qlang], "task_plan": ["<|RAG|>"], "tool_call": {"name": name, "arguments": {arg: title, "language": ISO1.get(lang, lang)}}})
        final = A.assistant_content(input_lang=[qlang], target_lang=lang, task_plan=["<|RAG|>", "<|explain|>"], response=answer, unsure_target=unsure)
    except A.AssemblyError as e:
        return Skip("assembly:" + str(e)[:40])
    convo = [{"role": "system", "content": WIKI_SYSTEM}, {"role": "user", "content": ins.text}, call, {"role": "tool", "content": result},
             {"role": "assistant", "content": final}]
    return Rec("rag_tool_call", dataset, qlang, lang, answer, input=title, instruction=ins.text, convo=convo, tools=tools_with_distractors(search_tool_def(name, arg), rng),
               meta={"instruction_language": qlang, "tool": name, "source": kind, "url": bool(url)})


@register("wikipedia_rag", "wikimedia/wikipedia 20231101 for the languages we carry (article -> search_wikipedia tool call / encyclopedic answer)")
def wikipedia_rag(rng, limit=None):
    ds = "wikimedia/wikipedia"
    have = {p.split("/")[0]: p for p, _ in files_of(ds) if p.startswith("20231101.")}
    for code, iso in ISO1.items():
        cap = CAP.get(code, CAP["default"])
        key = f"20231101.{iso}"
        if cap == 0 or key not in have:
            if key not in have:
                yield Skip(f"no_wikipedia_edition:{code}")
            continue
        n = 0
        for chunk in batched(hubio.iter_file(ds, have[key]), 128):
            rows = []
            for r in chunk:
                title, text, url = clean(r.get("title")), blank(r.get("text")), blank(r.get("url"))
                body = text[len(title):].strip() if text.startswith(title) else text
                if len(body) < 300 or not title or len(title) > 120 or rep_ratio(body) > 0.4 or rng.random() > 0.5:
                    yield Skip("short_empty_or_not_sampled")
                    continue
                rows.append((title, body, url))
            if not rows:
                continue
            for (title, body, url), (c2, unsure) in zip(rows, resolve([b[:900] for _, b, _ in rows], code)):
                if c2 != code:
                    yield Skip(f"lid_mismatch:{code}>{c2}")
                    continue
                rec = rag_rec(title, body, url, code, unsure, ds, rng, iso)
                yield rec
                n += isinstance(rec, Rec)
            if n >= cap or (limit and n >= limit):
                break


@register("wikipedia_summary_rag", "54rt1n/wikipedia-summary-dataset (English; topic -> summary) as a search tool call / encyclopedic answer")
def wikipedia_summary_rag(rng, limit=None):
    ds = "54rt1n/wikipedia-summary-dataset"
    n, cap = 0, limit or 100_000
    for r in hubio.iter_repo(ds):
        topic, summary = clean(r.get("topic")), blank(r.get("summary"))
        if len(summary) < 200 or not topic or len(topic) > 120 or rng.random() > 0.05:
            yield Skip("short_empty_or_not_sampled")
            continue
        rec = rag_rec(topic, summary, None, "eng", False, ds, rng, "en", kind="wikipedia_summary")
        yield rec
        n += isinstance(rec, Rec)
        if n >= cap:
            return


@register("cohere_wiki_rag", "CohereLabs/wikipedia-2023-11-embed-multilingual-v3 (title, text, url; no embeddings read) for our languages -> search tool call with the passage")
def cohere_wiki_rag(rng, limit=None):
    ds = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"
    fl = files_of(ds)
    for code, iso in ISO1.items():
        if code in ("eng", "fra"):
            continue
        paths = [p for p, _ in fl if p.startswith(f"{iso}/")][:1]
        n, cap = 0, limit or 10_000
        for p in paths:
            for chunk in batched(hubio.iter_file(ds, p, columns=["title", "text", "url"]), 128):
                rows = [(clean(r.get("title")), blank(r.get("text")), blank(r.get("url"))) for r in chunk]
                rows = [x for x in rows if x[0] and len(x[1]) >= 250 and rep_ratio(x[1]) <= 0.4 and rng.random() < 0.2]
                if not rows:
                    continue
                for (title, body, url), (c2, unsure) in zip(rows, resolve([b[:900] for _, b, _ in rows], code)):
                    if c2 != code:
                        yield Skip(f"lid_mismatch:{code}>{c2}")
                        continue
                    rec = rag_rec(title, body, url, code, unsure, ds, rng, iso, kind="cohere_wikipedia")
                    yield rec
                    n += isinstance(rec, Rec)
                if n >= cap:
                    break


@register("dear_cot_rerank", "abdoelsayed/DeAR-COT (listwise reranking with a reasoning trace in <think>) -> <|analyze|> with think + final ranking")
def dear_cot(rng, limit=None):
    ds = "abdoelsayed/DeAR-COT"
    n = 0
    for r in hubio.iter_file(ds, "DeAR-COT.json"):
        system, ins, inp, out = blank(r.get("system")), blank(r.get("instruction")), blank(r.get("input")), blank(r.get("output"))
        m = re.match(r"^\s*<think>\s*(.*?)\s*</think>\s*(.*)$", out, re.S)
        if not ins or not m or not re.search(r"\[\d+\]", m.group(2)):
            yield Skip("no_think_or_ranking")
            continue
        think, ranking = re.sub(r"</?think>", "", m.group(1)).strip(), re.sub(r"</?think>", "", m.group(2)).strip()
        if len(ins) > 24_000:
            yield Skip("too_long")
            continue
        if not ranking.lower().startswith("###"):
            ranking = "### Final Reranking: " + ranking
        yield Rec("rerank", ds, "eng", "eng", ranking, input="", prompt_override=ins + (("\n\n" + inp) if inp else ""), system=system or None, think=think,
                  plan=["<|analyze|>"], input_langs=["eng"], meta={"passages": len(re.findall(r"^\[\d+\]", ins, re.M))})
        n += 1
        if limit and n >= limit:
            return


CTGA_PLAN = {
    "sentiment": ("sentiment", ["<classify>"], "<sentiment>"),
    "topic": ("topic", ["<classify>"], "<topic>"),
    "natural language inference": ("nli", ["<nli>"], "<answer>"),
    "summarization": ("summarization", ["<summarize>"], None),
    "multiple-choice": ("multiple_choice", ["<qa>"], "<answer>"),
    "question answering": ("qa", ["<qa>"], None),
    "question generation": ("question_generation", ["<|generate|>"], None),
    "text generation": ("generation", ["<|generate|>"], None),
    "paraphrase": ("paraphrase", ["<classify>"], "<answer>"),
    "title generation": ("headline", ["<summarize>"], "<headline>"),
}
CTGA_CAP = 25_000


@register("ctga_tasks", "BatsResearch/ctga-v1 (task_type mapped to our verbs; extractive QA becomes a RAG tool call; per-task cap)")
def ctga(rng, limit=None):
    ds = "BatsResearch/ctga-v1"
    per: dict = {}
    paths = sorted(p for p, _ in files_of(ds, patterns=(".arrow",)) if p.startswith("train/"))
    for p in paths:
        for r in iter_arrow(ds, p, columns=["context", "task_input", "task_output", "task_type", "dataset"]):
            tt = str(r.get("task_type") or "").lower()
            if per.get(tt, 0) >= (limit or CTGA_CAP):
                continue
            if rng.random() > 0.15:
                continue
            ctx, ti, to = blank(r.get("context")), blank(r.get("task_input")), blank(r.get("task_output"))
            prompt = ti.replace("{{context}}", ctx) if "{{context}}" in ti else ti
            out = to.replace("{{context}}", ctx) if "{{context}}" in to else to
            if len(prompt) < 12 or not out or len(prompt) > 9000 or len(out) > 6000 or degenerate(out):
                yield Skip("empty_or_long")
                continue
            if "extractive question answering" in tt and ctx:
                q = ti.replace("{{context}}", "").strip()
                if len(q) < 8 or len(q) > 400:
                    yield Skip("no_standalone_question")
                    continue
                name, arg = rng.choice(SEARCH_TOOLS)
                try:
                    call = A.assistant_message({"input_lang": ["eng"], "task_plan": ["<|RAG|>"], "tool_call": {"name": name, "arguments": {arg: q[:200], "language": "en"}}})
                    final = A.assistant_content(input_lang=["eng"], target_lang="eng", task_plan=["<|RAG|>", "<qa>"], response=out, label_token="<answer>")
                except A.AssemblyError as e:
                    yield Skip("assembly:" + str(e)[:40])
                    continue
                convo = [{"role": "system", "content": WIKI_SYSTEM}, {"role": "user", "content": q}, call, {"role": "tool", "content": ctx[:3000]},
                         {"role": "assistant", "content": final}]
                yield Rec("rag_tool_call", ds, "eng", "eng", out, input=q, convo=convo, tools=tools_with_distractors(search_tool_def(name, arg), rng),
                          meta={"ctga_task_type": tt, "ctga_dataset": r.get("dataset"), "tool": name})
            else:
                task, plan, label = next((v for k, v in CTGA_PLAN.items() if k in tt), ("instruction_following", ["<|chat|>"], None))
                yield Rec(task, ds, "eng", "eng", out, input="", prompt_override=prompt, plan=plan, label=label, input_langs=["eng"],
                          meta={"ctga_task_type": tt, "ctga_dataset": r.get("dataset")})
            per[tt] = per.get(tt, 0) + 1


RERANK_SETS = ["nq", "triviaqa", "squad", "web_questions", "curatedtrec", "hotpotqa", "wiki_qa"]
RERANK_PER_SET = 30_000


@register("rerank_light_rag", "abdoelsayed/reranking-datasets-light bm25/<set>/train.json + psgs_w100.tsv: question -> search_wikipedia call returning 3 retrieved passages -> short answer (English Wikipedia; NOT an African-language source)")
def rerank_light_rag(rng, limit=None):
    from sft_new.ext_common import iter_json_array
    ds, corpus = "abdoelsayed/reranking-datasets-light", "abdoelsayed/reranking-datasets"
    picked = []                                                    # (set, question, answers, [ids in result order], positive_id)
    need = set()
    for name in RERANK_SETS:
        n = 0
        for obj in iter_json_array(ds, f"bm25/{name}/train.json"):
            q, answers, ctxs = clean(obj.get("question")), [clean(a) for a in (obj.get("answers") or []) if clean(a)], obj.get("ctxs") or []
            pos = [c for c in ctxs[:20] if c.get("has_answer")]
            if not q or not answers or not pos or len(q) > 300 or len(answers[0]) > 120:
                continue
            others = [c for c in ctxs[:10] if not c.get("has_answer")][:2]
            chosen = [pos[0]] + others
            rng.shuffle(chosen)
            ids = [int(c["id"]) for c in chosen]
            picked.append((name, q, answers, ids, int(pos[0]["id"])))
            need.update(ids)
            n += 1
            if n >= (limit or RERANK_PER_SET):
                break
    # one streaming pass over the passage corpus for the ids we need
    text: dict = {}
    import io
    with hubio.fs().open(f"datasets/{corpus}/psgs_w100/psgs_w100.tsv", "rb") as fh:
        for i, line in enumerate(io.TextIOWrapper(fh, encoding="utf-8", errors="replace")):
            if i == 0:
                continue
            cut = line.find("\t")
            if cut < 0:
                continue
            try:
                pid = int(line[:cut])
            except ValueError:
                continue
            if pid in need:
                parts = line.rstrip("\n").split("\t")
                text[pid] = (parts[2].strip('"') if len(parts) > 2 else "", parts[1].strip('"'))
                if len(text) == len(need):
                    break
    for name, q, answers, ids, pos_id in picked:
        if any(i not in text for i in ids):
            yield Skip("passage_missing")
            continue
        ins_lang = "eng"
        results = "\n\n".join(f"[{k + 1}] {text[i][0]}\n{text[i][1]}" for k, i in enumerate(ids))
        arg_name, arg = rng.choice(SEARCH_TOOLS)
        try:
            call = A.assistant_message({"input_lang": ["eng"], "task_plan": ["<|RAG|>"], "tool_call": {"name": arg_name, "arguments": {arg: q, "language": "en"}}})
            final = A.assistant_content(input_lang=["eng"], target_lang="eng", task_plan=["<|RAG|>", "<qa>"], response=answers[0], label_token="<answer>")
        except A.AssemblyError as e:
            yield Skip("assembly:" + str(e)[:40])
            continue
        convo = [{"role": "system", "content": WIKI_SYSTEM}, {"role": "user", "content": q}, call, {"role": "tool", "content": results},
                 {"role": "assistant", "content": final}]
        yield Rec("rag_tool_call", ds, "eng", "eng", answers[0], input=q, convo=convo, tools=tools_with_distractors(search_tool_def(arg_name, arg), rng),
                  meta={"retrieval_set": name, "tool": arg_name, "positive_rank": ids.index(pos_id) + 1})
