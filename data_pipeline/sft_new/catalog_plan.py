"""Plan the conversion of the datasets of the HF catalog worker's `sft/` folder (owner 2026-10-11). Pure metadata: nothing is downloaded.

    python -m sft_new.catalog_plan --catalog data-catalog/african_hf --covered covered_repos.json --out catalog_plan.json

The catalog (`sft/<iso3>_<language>.csv`, one row per dataset x language) carries per dataset: tasks, columns (`name:dtype; ...`), useful_columns, licence,
commercial_use, scope, provenance / synthetic hints and a sample row. A dataset appears once per language, so rows are merged by `dataset_id` first.

Verdicts (first match wins; the reason is written so the owner can audit it):
  skip:covered           already built / held / excluded by an earlier step of this build
  deferred:michsethowusu the owner will go through that account interactively at the end of the work
  skip:nc_licence        commercial_use == non_commercial (rows logged with the size)
  skip:no_sft_task       the catalog lists only pretraining / speech / vision tasks for it
  skip:rl_material       preference / rlhf / reward material (owner: no RL-type data for now)
  skip:coding            programming data (owner: none)
  skip:nllb              the card / provenance hints name NLLB (owner: no NLLB-derived data for now)
  skip:llama4            generated with Llama 4 (its licence forbids using the outputs)
  hold:arabic_general    only Arabic varieties (owner: general Arabic is held until told)
  skip:no_usable_columns no column layout one of the converters can use
  skip:evaluation_or_analysis name says eval / benchmark / blind-spot / retrieval-with-negatives / analysis (owner removed evals)
  skip:jailbreak_prompts ChatGPT-jailbreak prompt collections (English, not what the model should learn)
  convert:<shape>        chat | instruction | qa | translation | label | summ | tags | retrieval
Licence `unknown` / unspecified is used and logged (same rule as earlier batches); share-alike is used and flagged.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import re
from pathlib import Path

csv.field_size_limit(10 ** 9)

ARABIC = {"ary", "arz", "aeb", "acm", "apc", "arb", "ar", "ars", "apd", "ayl", "shu", "ayn", "afb", "acw", "aao", "abv", "acq", "acx", "adf", "ajp", "avl", "pga", "ssh"}
CODE = re.compile(r"(code|coding|program|leetcode|python|sql|compiler|swe[-_]|humaneval|codeforces|debug|javascript|stackoverflow)", re.I)
EVALISH = re.compile(r"(^|[-_ /.])(eval|evals|evaluation|benchmark|bench|blind|spots?|leaderboard|testset|test|dev)([-_ .]|$)|blindspot|retrieval|negatives|analysis|[-_]mlm|embedding|rerank", re.I)
NLLB = re.compile(r"nllb|no language left behind", re.I)
LLAMA4 = re.compile(r"llama[- ]?4", re.I)
TEXT_NAMES = ("text", "sentence", "content", "tweet", "review", "comment", "message", "body", "document", "article", "input", "utterance", "query", "post", "headline", "title")
LABEL_NAMES = ("label", "labels", "sentiment", "emotion", "topic", "category", "class", "intent", "polarity", "tag", "target", "toxicity", "hate", "category_name")
SUMM_NAMES = ("summary", "highlights", "abstract", "short_summary", "summarization")
TITLE_NAMES = ("title", "headline")
BODY_NAMES = ("text", "article", "document", "content", "body", "passage", "story", "full_text", "source")
SKIP_COLS = re.compile(r"(^id$|_id$|^idx$|^index$|url|link|audio|image|path|file|hash|^lang|language|^split$|timestamp|date|duration|speaker|^source$|^dataset$)", re.I)


def parse_columns(s: str) -> dict:
    out = {}
    for part in (s or "").split(";"):
        part = part.strip()
        if ":" in part:
            n, t = part.split(":", 1)
            out[n.strip()] = t.strip()
    return out


def kind_of(tasks: list, label_col: str = "") -> str | None:
    """Only explicit kinds: the label column's own name first, then a task name that says what is classified (a bare `text classification` is not used)."""
    lc = (label_col or "").lower()
    for key, kind in (("sentiment", "sentiment"), ("polarity", "sentiment"), ("emotion", "emotion"), ("hate", "hate"), ("toxic", "hate"), ("offens", "hate"),
                      ("intent", "intent"), ("topic", "topic"), ("categor", "topic")):
        if key in lc:
            return kind
    t = " ".join(tasks).lower()
    for key, kind in (("sentiment", "sentiment"), ("emotion", "emotion"), ("hate", "hate"), ("toxicity", "hate"), ("intent", "intent"), ("topic", "topic")):
        if key in t:
            return kind
    return None


def is_text(dtype: str) -> bool:
    return dtype.startswith("string") or dtype.startswith("large_string") or dtype in ("str", "text")


def decide(d: dict) -> dict:
    """d: merged dataset record -> {"verdict": ..., plus the fields a converter needs}."""
    cols = d["columns"]
    lowered = {c.lower(): c for c in cols}
    texty = [c for c, t in cols.items() if is_text(t) and not SKIP_COLS.search(c)]
    useful = [c.strip() for c in d["useful_columns"].split(";") if c.strip() and c.strip() in cols]
    tasks = d["tasks"]
    tl = " ".join(tasks).lower()

    def first(names, pool=None):
        for n in names:
            if n in lowered and (pool is None or lowered[n] in pool):
                return lowered[n]
        return None

    # chat
    for n in ("messages", "conversations", "conversation"):
        if n in lowered:
            return {"shape": "chat", "cols": [lowered[n]]}
    if "text" in lowered and cols[lowered["text"]].startswith("list<struct") and re.search(r"chat|instruction|question", tl):
        return {"shape": "chat", "cols": [lowered["text"]]}
    ik = first(("instruction", "prompt", "inputs", "question_prompt", "user"))
    ok = first(("output", "response", "completion", "targets", "answer", "answers", "assistant"))
    if ik and ok and ik != ok:
        return {"shape": "instruction", "cols": [ik, ok, first(("input", "context")) or ""]}
    if "input" in lowered and "output" in lowered:
        return {"shape": "instruction", "cols": [lowered["input"], lowered["output"], ""]}
    qk, ak = first(("question", "question_text", "query")), first(("answer", "answers", "explanation", "response", "output"))
    if qk and ak and qk != ak:
        return {"shape": "qa"}
    # retrieval corpora: query + positive passage (+ link, + negative passages) -> RAG with a ranked tool result
    if re.search(r"\brag\b|retriev", tl) and "query" in lowered and "passage" in lowered:
        return {"shape": "retrieval", "cols": [lowered["query"], lowered["passage"], first(("link", "url", "source_url")) or "",
                                               first(("negative_passages", "negatives", "hard_negatives")) or ""]}
    # token-level tags: NER / POS with a ClassLabel (or string) tag column
    if re.search(r"\bner\b|pos tagging", tl):
        tk = first(("tokens", "words", "sentence_tokens"))
        gk = first(("ner_tags", "pos_tags", "upos", "tags", "ner", "pos", "labels"))
        if tk and gk:
            return {"shape": "tags", "cols": [tk, gk], "kind": "ner" if "ner" in tl and "pos" not in lowered.get(gk, gk).lower() else "pos"}
    if "translation" in tl or "inter-african" in tl:
        a = first(("source", "src"))
        b = first(("target", "tgt"))
        if a and b:
            return {"shape": "translation", "cols": [a, b]}
        if "translation" in lowered and re.match(r"translation\(|struct|dict|\{", cols[lowered["translation"]]):
            return {"shape": "translation", "cols": [lowered["translation"]]}
        pool = [c for c in useful if c in texty]
        if len(pool) < 2:
            pool = texty
        if len(pool) >= 2:
            return {"shape": "translation", "cols": pool[:2]}
    if re.search(r"summar|headline|title generation", tl):
        body = first(BODY_NAMES)
        summ = first(SUMM_NAMES)
        title = first(TITLE_NAMES)
        if body and (summ or title) and body not in (summ, title):
            return {"shape": "summ", "cols": [body, summ or "", title or ""]}
    if re.search(r"classification", tl):
        lab = first(LABEL_NAMES)
        kind = kind_of(tasks, lab or "")
        txt = first(TEXT_NAMES, texty) or next((c for c in useful if c in texty and c != lab), None) or next((c for c in texty if c != lab), None)
        if kind and lab and txt and lab != txt:
            return {"shape": "label", "cols": [txt, lab], "kind": kind}
    return {"shape": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, required=True, help="the african_hf folder (contains sft/*.csv)")
    ap.add_argument("--covered", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bucket", default="sft", choices=["sft", "rl", "pretrain"], help="which folder of the catalog to read (rl/ lists translation-pair sources usable for fine-tuning too)")
    a = ap.parse_args()
    covered = json.loads(a.covered.read_text())
    D: dict = {}
    for f in sorted(glob.glob(str(a.catalog / a.bucket / "*.csv"))):
        with open(f, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                d = D.setdefault(r["dataset_id"], {"repo": r["dataset_id"], "langs": set(), "rows": 0, "pivot": set(), "meta": r})
                d["langs"] |= {x for x in re.split(r"[,;]\s*", r["language_iso3"]) if x}
                d["langs"] |= {x for x in re.split(r"[,;]\s*", r.get("all_african_languages") or "") if x}
                try:
                    d["rows"] = max(d["rows"], int(float(r["num_rows"] or 0)))
                except ValueError:
                    pass
    convert, skipped = [], []
    for repo, d in D.items():
        m = d["meta"]
        rec = {"repo": repo, "langs": sorted(d["langs"]), "rows": d["rows"], "licence": m["license"], "commercial_use": m["commercial_use"], "scope": m["scope"],
               "tasks": [t.strip() for t in m["tasks"].split(";") if t.strip()], "columns": parse_columns(m["columns"]), "useful_columns": m["useful_columns"],
               "author": m["author"], "synthetic_or_mt_hints": m.get("synthetic_or_mt_hints", ""), "provenance_hints": m.get("provenance_hints", ""),
               "gated": m.get("gated")}
        sft_tasks = [t for t in rec["tasks"] if not re.match(r"pretrain|speech|tts|vision|unknown", t, re.I)]
        why = None
        if repo in covered:
            why = "skip:covered"
        elif m["author"] == "michsethowusu":
            why = "deferred:michsethowusu"
        elif m["commercial_use"] == "non_commercial":
            why = "skip:nc_licence"
        elif not sft_tasks:
            why = "skip:no_sft_task"
        elif any(re.search(r"preference|rlhf|reward", t, re.I) for t in sft_tasks) and not any(re.search(r"chat|instruction|translation|question|summar", t, re.I) for t in sft_tasks):
            why = "skip:rl_material"
        elif re.search(r"jailbreak", repo, re.I):
            why = "skip:jailbreak_prompts"
        elif EVALISH.search(repo.split("/", 1)[1]):
            why = "skip:evaluation_or_analysis"
        elif CODE.search(repo.split("/", 1)[1]):
            why = "skip:coding"
        elif NLLB.search(repo) or NLLB.search(rec["synthetic_or_mt_hints"] + " " + rec["provenance_hints"] + " " + m.get("readme_summary", "")):
            why = "skip:nllb"
        elif LLAMA4.search(rec["synthetic_or_mt_hints"] + " " + rec["provenance_hints"]):
            why = "skip:llama4"
        elif rec["langs"] and set(rec["langs"]) <= ARABIC:
            why = "hold:arabic_general"
        if why:
            skipped.append({**rec, "verdict": why})
            continue
        rec["tasks"] = sft_tasks
        dec = decide({"columns": rec["columns"], "useful_columns": rec["useful_columns"], "tasks": sft_tasks})
        if dec["shape"] is None:
            skipped.append({**rec, "verdict": "skip:no_usable_columns"})
        else:
            convert.append({**rec, **dec, "verdict": f"convert:{dec['shape']}",
                            "licence_flag": "share_alike" if m["commercial_use"] == "share_alike" else ("unspecified" if m["commercial_use"] == "unknown" else "ok")})
    a.out.write_text(json.dumps({"convert": convert, "skipped": skipped}, indent=1, ensure_ascii=False))
    print("datasets:", len(D))
    print("convert:", len(convert), dict(collections.Counter(c["shape"] for c in convert)), "rows:", sum(c["rows"] for c in convert))
    print("skipped:", dict(collections.Counter(s["verdict"] for s in skipped)))
    nc = [s for s in skipped if s["verdict"] == "skip:nc_licence"]
    print("nc rows logged:", sum(s["rows"] for s in nc))


if __name__ == "__main__":
    main()
