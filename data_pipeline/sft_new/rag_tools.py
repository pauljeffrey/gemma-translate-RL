"""Retrieval-augmented answers as FUNCTION CALLS (owner 2026-10-11): a (question, context[, url], answer) row becomes a tool-using conversation in the
chat template v2 format of docs/TOOL_CALL_FORMAT.md.  The context is what the tool RETURNS; the model has to decide to call it and then answer from it.

Variants (one per row, drawn at random):
  kb      search_documents / retrieve_context / ...   one call, the tool result is the passage                              (document store)
  web     web_search / internet_search / ...          one call, the result is "title, url, page text"                      (mimics the internet)
  fetch   fetch_url / open_url / read_webpage         the USER message carries a link; one call to fetch it; result = page text
  chain   web_search -> fetch_url                     two sequential calls in ONE assistant turn: search returns the link, fetch returns the text

The URL is the dataset's own when it has one, otherwise a make-shift link built from the question (`https://<site>/<section>/<slug>-<hash>`).  Every
conversation attaches the function definitions it uses plus 1-4 distractors to the system turn (the contract requires the called function to be listed).
The answer is the dataset's answer, untouched; the passage is cut at a sentence end when it is longer than `max_chars`.
"""
from __future__ import annotations

import hashlib
import random
import re
import unicodedata

from sft_new.compose import A, Rec, Skip

ISO1 = {"afr": "af", "amh": "am", "ewe": "ee", "hau": "ha", "ibo": "ig", "kin": "rw", "nya": "ny", "orm": "om", "sna": "sn", "som": "so", "sot": "st",
        "swh": "sw", "tsn": "tn", "twi": "tw", "wol": "wo", "xho": "xh", "yor": "yo", "zul": "zu", "eng": "en", "fon": "fon", "pcm": "pcm", "ti": "ti",
        "fra": "fr", "lin": "ln", "lug": "lg", "mlg": "mg", "kik": "ki", "luo": "luo", "bem": "bem", "nso": "nso", "tso": "ts", "ven": "ve", "ssw": "ss", "nbl": "nr"}

KB_TOOLS = [
    ("search_documents", "Search the document collection and return the passage that best matches the query."),
    ("retrieve_context", "Retrieve the most relevant document passage for a question from the knowledge base."),
    ("knowledge_base_search", "Search the knowledge base and return the best matching document text."),
    ("lookup_documents", "Look up documents that answer a query and return the top passage."),
    ("find_passages", "Find the passage in the library that is most relevant to the query."),
]
WEB_TOOLS = [
    ("web_search", "Search the web and return the top result with its URL and page text."),
    ("internet_search", "Search the internet for a query and return the best page (title, URL, text)."),
    ("search_the_web", "Run a web search and return the most relevant page with its link."),
    ("online_search", "Search online sources and return the top matching page and its address."),
]
FETCH_TOOLS = [
    ("fetch_url", "Download the web page at a URL and return its text."),
    ("open_url", "Open a link and return the text of the page."),
    ("read_webpage", "Read the web page at the given address and return its main text."),
    ("get_page_text", "Get the plain text of the page behind a URL."),
]
SITES = ["library.sabi-docs.org", "kb.afrisearch.net", "docs.naijaknowledge.org", "wiki.africaedu.org", "articles.mashariki-info.org", "reference.kowledgebase.africa",
         "pages.ilmu-hub.org", "archive.sahel-library.org", "learn.nyumbani-docs.org", "read.jollofpedia.org"]
SECTIONS = ["articles", "docs", "pages", "reference", "library", "topics", "entries"]
SYSTEMS = [
    "You are a helpful assistant. When you need facts you do not already have, call one of the functions listed below, then answer from what it returns.",
    "You are a helpful assistant with access to the functions listed below. If a question depends on information you do not have, call the right function first and base your answer on its result.",
    "You answer questions using the tools provided. Do not guess: call a function to get the source text, then answer only from that text.",
    "You are a research assistant. Use the available functions to find or open the relevant source, then give a short, accurate answer in the language of the question.",
    "Answer the user's question. The functions below can search or fetch documents; use them whenever the answer must come from a source.",
]
_ABS = re.compile(r"^https?://\S+$", re.I)


def _slug(text: str, n: int = 6) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    words = re.findall(r"[a-z0-9]+", t)[:n]
    return "-".join(words) or "article"


def makeshift_url(question: str, rng: random.Random, seed: str = "") -> str:
    h = hashlib.sha1((question + seed).encode("utf-8")).hexdigest()[:8]
    return f"https://{rng.choice(SITES)}/{rng.choice(SECTIONS)}/{_slug(question)}-{h}"


def clean_url(u) -> str | None:
    u = (u or "").strip() if isinstance(u, str) else ""
    return u if _ABS.match(u) and len(u) < 400 else None


def cut_passage(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    m = list(re.finditer(r"[.!?።؟]\s", cut))
    return cut[: m[-1].end()].strip() if m and m[-1].end() > max_chars * 0.6 else cut.rstrip()


def _tool_def(name: str, desc: str, props: dict, required: list) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": required}}}


def _search_def(name: str, desc: str) -> dict:
    return _tool_def(name, desc, {"query": {"type": "string", "description": "The question or search terms."},
                                  "language": {"type": "string", "description": "ISO 639-1 code of the language of the documents, e.g. sw."}}, ["query"])


def _fetch_def(name: str, desc: str) -> dict:
    return _tool_def(name, desc, {"url": {"type": "string", "description": "The full URL of the page."}}, ["url"])


def _ranked(passage: str, negatives: list, rng: random.Random, max_chars: int, with_urls: bool, real_url: str | None, question: str) -> str:
    """Several retrieved passages with the right one at a random rank: the model has to pick the passage that answers the question."""
    k = 1 + min(len(negatives), rng.randint(1, 2))
    each = max(600, max_chars // k)
    docs = [(passage, real_url)] + [(cut_passage(n, each), None) for n in negatives[: k - 1]]
    docs[0] = (cut_passage(docs[0][0], each), docs[0][1])
    rng.shuffle(docs)
    out = []
    for i, (text, u) in enumerate(docs, 1):
        head = f"[{i}] {u or makeshift_url(question, rng, text[:40])}\n" if with_urls else f"[{i}] "
        out.append(f"{head}{text}")
    return "\n\n".join(out)


def rag_convo(question: str, context: str, answer: str, lang: str, dataset: str, rng: random.Random, *, url: str | None = None, unsure: bool = False,
              title: str | None = None, max_chars: int = 5000, answer_lang: str | None = None, qlang: str | None = None, extra_meta: dict | None = None,
              negatives: list | None = None, variants=("kb", "web", "fetch", "chain"), weights=(0.38, 0.27, 0.2, 0.15)) -> "Rec | Skip":
    """A tool-call conversation whose tool result is `context`.  `lang` = language of the question and context, `answer_lang` = language of the answer.
    `negatives` (other, non-answering passages) turn a kb / web result into a ranked list in which `context` sits at a random rank."""
    from sft_new.converters import tools_with_distractors
    answer_lang = answer_lang or lang
    qlang = qlang or lang
    question, answer, context = question.strip(), answer.strip(), context.strip()
    if not question or not answer or len(context) < 40:
        return Skip("empty_question_answer_or_context")
    passage = cut_passage(context, max_chars)
    real_url = clean_url(url)
    variant = rng.choices(list(variants), weights=list(weights))[0]
    page_url = real_url or makeshift_url(question, rng, dataset)
    iso = ISO1.get(lang, lang)
    sys_msg = {"role": "system", "content": rng.choice(SYSTEMS)}
    user_text = question
    defs: list = []
    steps: list = []                                               # [(name, arguments, tool_result)]
    kb_name, kb_desc = rng.choice(KB_TOOLS)
    web_name, web_desc = rng.choice(WEB_TOOLS)
    fe_name, fe_desc = rng.choice(FETCH_TOOLS)
    head = f"{title}\n" if title else ""
    if variant == "kb":
        defs = [_search_def(kb_name, kb_desc)]
        res = _ranked(passage, negatives, rng, max_chars, False, None, question) if negatives else (f"{head}{passage}" if head else passage)
        steps = [(kb_name, {"query": question[:240], "language": iso}, res)]
    elif variant == "web":
        defs = [_search_def(web_name, web_desc)]
        res = (_ranked(passage, negatives, rng, max_chars, True, real_url, question) if negatives
               else f"{title or _slug(question, 8).replace('-', ' ').capitalize()}\n{page_url}\n\n{passage}")
        steps = [(web_name, {"query": question[:240], "language": iso}, res)]
    elif variant == "fetch":
        defs = [_fetch_def(fe_name, fe_desc)]
        user_text = f"{question}\n{page_url}" if rng.random() < 0.5 else f"{page_url}\n\n{question}"
        steps = [(fe_name, {"url": page_url}, passage)]
    else:                                                          # chain: search -> fetch
        defs = [_search_def(web_name, web_desc), _fetch_def(fe_name, fe_desc)]
        first = re.split(r"(?<=[.!?።؟])\s+", passage, maxsplit=1)[0][:240]
        listing = f"1. {title or _slug(question, 8).replace('-', ' ').capitalize()}\n   {page_url}\n   {first}"
        steps = [(web_name, {"query": question[:240], "language": iso}, listing), (fe_name, {"url": page_url}, passage)]
    plan = ["<|RAG|>"]
    convo = [sys_msg, {"role": "user", "content": user_text}]
    try:
        for name, args, result in steps:
            convo.append(A.assistant_message({"input_lang": [qlang], "task_plan": plan, "tool_call": {"name": name, "arguments": args}}))
            convo.append({"role": "tool", "content": result})
        final = A.assistant_content(input_lang=[qlang], target_lang=answer_lang, task_plan=["<|RAG|>", "<qa>"], response=answer, label_token="<answer>",
                                    unsure_input=unsure)
    except A.AssemblyError as e:
        return Skip("assembly:" + str(e)[:40])
    convo.append({"role": "assistant", "content": final})
    tools = tools_with_distractors(defs[0], rng, 1, 3)
    for d in defs[1:]:
        tools.append(d)
    names = set()
    tools = [t for t in tools if not (t["function"]["name"] in names or names.add(t["function"]["name"]))]
    rng.shuffle(tools)
    return Rec("rag_tool_call", dataset, qlang, answer_lang, answer, input=question, convo=convo, tools=tools,
               meta={**(extra_meta or {}), "rag_variant": variant, "url_source": "dataset" if real_url else "makeshift", "calls": len(steps)})
