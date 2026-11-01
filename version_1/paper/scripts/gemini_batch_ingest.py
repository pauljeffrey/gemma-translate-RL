#!/usr/bin/env python
"""Ingest Gemini Batch API result JSONL(s) into pseudo_gemini.jsonl + an actual-cost report.  No API call, no key.

Usage:
  python gemini_batch_ingest.py --results res_01.jsonl [res_02.jsonl ...] --index gem_req/requests_index.jsonl \
      --prices reports/data/longctx/gemini_prices_2026-10-23.json --model gemini-2.5-flash-lite --tier batch --out-dir gem_out

Result line (batch page): {"key": ..., "response": <GenerateContentResponse>}  or  {"key": ..., "error": {...}}.
GenerateContentResponse fields read (camelCase as in REST, snake_case also accepted): candidates[0].content.parts[].text (parts with
thought=true are skipped), candidates[0].finishReason, promptFeedback.blockReason, usageMetadata.{promptTokenCount,
candidatesTokenCount, thoughtsTokenCount, totalTokenCount}.  Billed output = candidates + thoughts tokens.

Output pseudo_gemini.jsonl row: {iid, hyp_text, hyp_sents (chunked only), finish, status, flags, n_chunks, usage{prompt,candidates,thoughts,total}}
status: ok | truncated | blocked | empty | error | missing | line_mismatch   (worst chunk wins; priority error>blocked>empty>missing>truncated>line_mismatch>ok)
Only status ok (and optionally line_mismatch/truncated) rows should be used as references; the scorer must drop the rest.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemini_batch_common as C  # noqa: E402

PRIORITY = ["error", "blocked", "empty", "missing", "truncated", "line_mismatch", "ok"]
BLOCK_FINISH = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY", "LANGUAGE", "OTHER"}
PREFACE = re.compile(r"^\s*(?:\*\*)?(?:sure|certainly|of course|okay|ok|here(?:'s| is| are)|the (?:following )?translation|translation|below is|this is)\b[^\n]{0,120}[:：]\s*(?:\*\*)?\s*$", re.I)
FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*$")


def g(d, *names, default=None):
    """camelCase / snake_case tolerant getter."""
    for n in names:
        if isinstance(d, dict) and n in d and d[n] is not None:
            return d[n]
        snake = re.sub(r"([A-Z])", lambda m: "_" + m.group(1).lower(), n)
        if isinstance(d, dict) and snake in d and d[snake] is not None:
            return d[snake]
    return default


def clean_text(t):
    """Strip code fences, markdown bold wrapper lines and a one-line preface ("Here is the translation:")."""
    lines = t.replace("\r\n", "\n").split("\n")
    lines = [l for l in lines if not FENCE.match(l)]
    while lines and not lines[0].strip():
        lines.pop(0)
    if len(lines) > 1 and PREFACE.match(lines[0]):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(l.rstrip() for l in lines)


def parse_line(obj):
    """-> dict(status, finish, text, usage)"""
    usage = dict(prompt=0, candidates=0, thoughts=0, total=0)
    if obj.get("error") and not obj.get("response"):
        e = obj["error"]
        return dict(status="error", finish="ERROR:%s" % (g(e, "code", default="") if isinstance(e, dict) else e), text="", usage=usage,
                    detail=(g(e, "message", default="") if isinstance(e, dict) else str(e))[:300])
    resp = obj.get("response") or {}
    um = g(resp, "usageMetadata", default={}) or {}
    usage = dict(prompt=int(g(um, "promptTokenCount", default=0) or 0), candidates=int(g(um, "candidatesTokenCount", default=0) or 0),
                 thoughts=int(g(um, "thoughtsTokenCount", default=0) or 0), total=int(g(um, "totalTokenCount", default=0) or 0))
    cands = resp.get("candidates") or []
    block = g(g(resp, "promptFeedback", default={}) or {}, "blockReason")
    if not cands:
        return dict(status="blocked" if block else "empty", finish="BLOCK:%s" % block if block else "NO_CANDIDATES", text="", usage=usage)
    c = cands[0]
    finish = g(c, "finishReason", default="UNKNOWN") or "UNKNOWN"
    parts = (g(c, "content", default={}) or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
    text = clean_text(text)
    if finish in BLOCK_FINISH and not text.strip():
        return dict(status="blocked", finish=finish, text="", usage=usage)
    if not text.strip():
        return dict(status="truncated" if finish == "MAX_TOKENS" else "empty", finish=finish, text="", usage=usage)
    if finish == "MAX_TOKENS":
        return dict(status="truncated", finish=finish, text=text, usage=usage)
    return dict(status="ok", finish=finish, text=text, usage=usage)


def worst(statuses):
    return min(statuses, key=PRIORITY.index) if statuses else "missing"


def add_usage(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in ("prompt", "candidates", "thoughts", "total")}


def split_chunk_text(text, n_expected):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines, len(lines) == n_expected


def ingest(result_paths, index_path, out_dir, prices_path=None, model=None, tier="batch", phase="current"):
    results = {}
    dup = 0
    for p in result_paths:
        with open(p, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    results.setdefault("__bad_line__%s:%d" % (os.path.basename(p), ln), dict(status="error", finish="BAD_JSON", text="", usage={}))
                    continue
                k = obj.get("key")
                if k in results:
                    dup += 1
                results[k] = parse_line(obj)
    # expected structure
    expected = {}      # iid -> list of index rows
    if index_path:
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    expected.setdefault(r["iid"], []).append(r)
    else:
        for k in results:
            if k.startswith("__bad_line__"):
                continue
            iid, ci = C.split_key(k)
            expected.setdefault(iid, []).append(dict(key=k, iid=iid, chunk=ci, n_chunks=None, lo=None, hi=None))
    rows = []
    tot = dict(prompt=0, candidates=0, thoughts=0, total=0)
    counts = {}
    for iid in sorted(expected):
        ex = sorted(expected[iid], key=lambda r: -1 if r.get("chunk") is None else r["chunk"])
        chunked = ex[0].get("chunk") is not None
        sts, usage, sents, texts, flags, finishes = [], dict(prompt=0, candidates=0, thoughts=0, total=0), [], [], [], []
        for r in ex:
            res = results.get(r["key"])
            if res is None:
                sts.append("missing")
                finishes.append("MISSING")
                if chunked and r.get("hi") is not None:
                    sents += [""] * (r["hi"] - r["lo"])
                continue
            st = res["status"]
            usage = add_usage(usage, res["usage"])
            finishes.append(res["finish"])
            if chunked:
                n = (r["hi"] - r["lo"]) if r.get("hi") is not None else None
                lines, okn = split_chunk_text(res["text"], n) if n else (res["text"].split("\n"), True)
                if not okn:
                    if st == "ok":
                        st = "line_mismatch"
                        flags.append("chunk%s: %d lines for %d sentences" % (r["chunk"], len(lines), n))
                    lines = (lines + [""] * n)[:n] if len(lines) <= n else lines[:n - 1] + [" ".join(lines[n - 1:])]
                if n and st in ("blocked", "empty", "error"):
                    lines = [""] * n
                sents += lines
            else:
                texts.append(res["text"])
            sts.append(st)
        status = worst(sts)
        if chunked:
            # rebuild the document: paragraph break before sentences that start a paragraph
            pb = []
            for r in ex:
                pb += r.get("para_break_before") or [False] * ((r.get("hi") or 0) - (r.get("lo") or 0))
            out = ""
            for i, s in enumerate(sents):
                if i:
                    out += ("\n\n" if (i < len(pb) and pb[i]) else " ")
                out += s
            hyp_text = out.strip()
        else:
            hyp_text = texts[0] if texts else ""
        bad = next((f for s_, f in zip(sts, finishes) if s_ == status), finishes[0] if finishes else "MISSING")
        row = dict(iid=iid, hyp_text=hyp_text, finish=bad if status != "ok" else "STOP", status=status, flags=flags, n_chunks=len(ex), usage=usage)
        if chunked:
            row["hyp_sents"] = sents
        rows.append(row)
        counts[status] = counts.get(status, 0) + 1
        tot = add_usage(tot, usage)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "pseudo_gemini.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    unexpected = sorted(k for k in results if not k.startswith("__bad_line__") and C.split_key(k)[0] not in expected)
    report = dict(items=len(rows), status_counts=counts, duplicate_result_keys=dup, unexpected_keys=unexpected[:20], usage_total=tot,
                  results_lines=len(results))
    if prices_path and model:
        pr = json.load(open(prices_path, encoding="utf-8"))["models"][model]
        key = tier + ("_later" if phase == "later" and (tier + "_later") in pr else "")
        p = pr[key]
        usd = (tot["prompt"] * p["in"] + (tot["candidates"] + tot["thoughts"]) * p["out"]) / 1e6
        report.update(model=model, tier=tier, price_per_1M=p, actual_usd=round(usd, 4),
                      usd_per_1000_items=round(usd / max(1, len(rows)) * 1000, 3),
                      hidden_thinking_share_of_output=round(tot["thoughts"] / max(1, tot["thoughts"] + tot["candidates"]), 4),
                      note="billed output = candidatesTokenCount + thoughtsTokenCount; failed/blocked requests may still bill input tokens (UNVERIFIED)")
    with open(os.path.join(out_dir, "gemini_ingest_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    return rows, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--index", default="", help="requests_index.jsonl written by gemini_batch_requests.py (recommended)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prices", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--tier", choices=["batch", "standard"], default="batch")
    ap.add_argument("--phase", choices=["current", "later"], default="current")
    a = ap.parse_args(argv)
    rows, rep = ingest(a.results, a.index or None, a.out_dir, a.prices or None, a.model or None, a.tier, a.phase)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return rows, rep


if __name__ == "__main__":
    main()
