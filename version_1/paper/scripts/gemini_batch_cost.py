#!/usr/bin/env python
"""Gemini (Batch API) cost model for the LC1 long-document pseudo-reference benchmark.

A pure function of
  * the per-item counts CSV written on the box by gemini_count_tokens.py  (reports/data/longctx/gemini_counts_items.csv,
    gemini_counts_meta.csv), and
  * the prices JSON (reports/data/longctx/gemini_prices_2026-10-23.json).
No hand-typed totals: rerun this script to regenerate
  reports/data/longctx/gemini_cost_table.csv   (long form, every number)
  paper/tables/t_longctx_gemini_cost.md        (markdown)

Model of one request:
  input  tokens = instruction (measured, per request) + framing + source text tokens
  output tokens = multiplier x English source tokens (visible)  +  hidden thinking tokens per request (scenario)
  $ = (input x p_in + output x p_out) / 1e6, p_out includes thinking tokens (pricing page).
Source tokens: --src-basis chars4 (default, conservative: ~4 chars/token rule of thumb, ~18% above the Gemma-3 count) or gemma.
Multipliers 2.0 / 2.75 / 3.5 = assumed Gemini-token ratio of African-language output to English input
(Amharic / Ge'ez script may be higher still; 3.5 is NOT guaranteed to cover it).
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gemini_batch_common as C  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LONGCTX = os.path.join(REPO, "reports", "data", "longctx")
MULTS = [2.0, 2.75, 3.5]
SELECTIONS = ["full", "reduced_core", "reduced_sci_dev", "reduced"]
SEL_LABEL = {
    "full": "FULL set (all items)",
    "reduced_core": "REDUCED core (8 lang x 5 docs x b512/b1024/b2048)",
    "reduced_sci_dev": "REDUCED sci dev (25 papers x 6 lang)",
    "reduced": "REDUCED total (core + sci dev)",
}
LOW_THINK = 500
MODES = ["whole", "chunk1", "chunk8"]
PILOT_SENTS = 50
PILOT_LANGS = C.HUMAN_LANGS + [l for l in C.SCI_LANGS if l not in C.HUMAN_LANGS]


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_items(path):
    rows = read_csv(path)
    for r in rows:
        for k in r:
            if k not in ("iid", "set", "bin", "doc_id", "iso"):
                r[k] = int(r[k]) if r[k] not in ("", "None") else 0
    return rows


def load_meta(path):
    return {r["key"]: float(r["value"]) for r in read_csv(path)}


def thinking_scenarios(model):
    sc = []
    if model.get("can_disable_thinking", True):
        sc.append(("off", 0))
    sc.append(("low", LOW_THINK))
    d = model.get("thinking_default_hidden_tokens", 2000)
    if all(d != v for _, v in sc):
        sc.append(("default", d))
    elif model.get("thinking_default_hidden_tokens", 0) == 0:
        sc[0] = ("off/default", 0)
    return sc


def price(model, tier, phase):
    key = tier + ("_later" if phase == "later" and (tier + "_later") in model else "")
    return model[key]


def aggregate(rows, mode, meta, basis):
    """Totals for a set of item rows in one request mode.  Returns dict(requests, items, docs, in_src, out_src)."""
    ratio = (lambda r: r["chars4"] / r["src_tok"]) if basis == "chars4" else (lambda r: 1.0)
    items = len(rows)
    docs = len({r["doc_id"] for r in rows})
    if mode == "whole":
        req = items
        src = sum(r["chars4"] if basis == "chars4" else r["src_tok"] for r in rows)
        instr = meta["instr_tokens_whole_mean"]
    else:
        n = int(mode[len("chunk"):])
        req = sum(r["n_chunks_%d" % n] for r in rows)
        src = sum(r["sent_tok_sum"] * ratio(r) for r in rows)
        instr = meta["instr_tokens_chunk_mean"]
    overhead = req * (instr + meta["framing_tokens_assumed"])
    return dict(requests=req, items=items, docs=docs, src_tokens=src, in_tokens=src + overhead, overhead_tokens=overhead)


def cost_usd(agg, mult, hidden_per_req, p):
    out_vis = agg["src_tokens"] * mult
    out_hid = agg["requests"] * hidden_per_req
    usd = (agg["in_tokens"] * p["in"] + (out_vis + out_hid) * p["out"]) / 1e6
    return usd, out_vis, out_hid


def fmt(x):
    return "$%.3f" % x if x < 10 else ("$%.2f" % x if x < 1000 else "${:,.0f}".format(x))


def pilot_aggregate(meta, basis, n_langs, sents, chunk):
    tok = meta["sent_tok_mean_sm"] * (meta["chars4_over_gemma"] if basis == "chars4" else 1.0)
    req_per_lang = -(-sents // chunk)
    req = req_per_lang * n_langs
    src = sents * n_langs * tok
    instr = meta["instr_tokens_chunk_mean"] if chunk > 1 else meta["instr_tokens_chunk_mean"]
    return dict(requests=req, items=sents * n_langs, docs=sents, src_tokens=src,
                in_tokens=src + req * (instr + meta["framing_tokens_assumed"]), overhead_tokens=req * (instr + meta["framing_tokens_assumed"]))


def build(items, meta, prices, phase, basis):
    long_rows = []
    sel_rows = {s: C.select_items(items, s) for s in SELECTIONS}
    # items rows carry iid/doc_id/iso/set/bin keys, so select_items works directly on CSV rows
    for s in SELECTIONS:
        for mode in MODES:
            agg = aggregate(sel_rows[s], mode, meta, basis)
            for mid, m in prices["models"].items():
                for tier in ("standard", "batch"):
                    p = price(m, tier, phase)
                    for tname, hid in thinking_scenarios(m):
                        for mult in MULTS:
                            usd, ov, oh = cost_usd(agg, mult, hid, p)
                            long_rows.append(dict(kind="main", selection=s, mode=mode, model=mid, tier=tier, thinking=tname, hidden_per_request=hid,
                                                  mult=mult, documents=agg["docs"], items=agg["items"], requests=agg["requests"],
                                                  input_tokens=round(agg["in_tokens"]), output_visible_tokens=round(ov), output_hidden_tokens=round(oh),
                                                  usd_total=round(usd, 4), usd_per_1000_items=round(usd / agg["items"] * 1000, 3)))
    for chunk in (1, 8):
        agg = pilot_aggregate(meta, basis, len(PILOT_LANGS), PILOT_SENTS, chunk)
        for mid, m in prices["models"].items():
            for tier in ("standard", "batch"):
                p = price(m, tier, phase)
                for tname, hid in thinking_scenarios(m):
                    usd, ov, oh = cost_usd(agg, 2.75, hid, p)
                    long_rows.append(dict(kind="pilot", selection="pilot_%dsent_x_%dlang" % (PILOT_SENTS, len(PILOT_LANGS)), mode="chunk%d" % chunk, model=mid, tier=tier,
                                          thinking=tname, hidden_per_request=hid, mult=2.75, documents=0, items=agg["items"], requests=agg["requests"],
                                          input_tokens=round(agg["in_tokens"]), output_visible_tokens=round(ov), output_hidden_tokens=round(oh),
                                          usd_total=round(usd, 4), usd_per_1000_items=round(usd / agg["items"] * 1000, 3)))
    return long_rows, sel_rows


def render_md(rows, sel_rows, meta, prices, phase, basis):
    idx = {}
    for r in rows:
        idx[(r["kind"], r["selection"], r["mode"], r["model"], r["tier"], r["thinking"], r["mult"])] = r
    models = prices["models"]
    L = []
    L.append("# Gemini Batch cost table for the long-document pseudo-reference benchmark")
    L.append("")
    L.append("Generated by `paper/scripts/gemini_batch_cost.py` from `reports/data/longctx/gemini_counts_items.csv` + `gemini_counts_meta.csv` + "
             "`gemini_prices_2026-10-23.json` (prices read from https://ai.google.dev/gemini-api/docs/pricing on %s; page says last updated 2026-10-22). "
             "Price phase: **%s**; source-token basis: **%s**. Do not edit by hand." % (prices["read_date"], phase, basis))
    L.append("")
    L.append("Assumptions: multiplier = Gemini output tokens / English source tokens (2.0 / 2.75 / 3.5, NOT measured, Amharic may exceed 3.5); thinking rows "
             "add hidden output tokens PER REQUEST (off +0, low +%d, default = model default, 2000 for default-on models); instruction ~%.0f tokens whole-doc / "
             "~%.0f chunked + %d framing tokens per request; no retries. `$/1k` = dollars per 1,000 translated items (document x language)."
             % (LOW_THINK, meta["instr_tokens_whole_mean"], meta["instr_tokens_chunk_mean"], meta["framing_tokens_assumed"]))
    L.append("")
    L.append("## 1. What is being translated")
    L.append("")
    L.append("| selection | documents | items | source tokens (basis) | requests whole | requests chunk-1 | requests chunk-8 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in SELECTIONS:
        a = {m: aggregate(sel_rows[s], m, meta, basis) for m in MODES}
        L.append("| %s | %d | %d | %s | %s | %s | %s |" % (SEL_LABEL[s], a["whole"]["docs"], a["whole"]["items"], "{:,.0f}".format(a["whole"]["src_tokens"]),
                 "{:,}".format(a["whole"]["requests"]), "{:,}".format(a["chunk1"]["requests"]), "{:,}".format(a["chunk8"]["requests"])))
    L.append("")
    L.append("## 2. Whole-document requests: total $ (Batch at 2.0 / 2.75 / 3.5, Standard at 2.75) and Batch $ per 1,000 items at 2.75")
    L.append("")
    for s in SELECTIONS:
        L.append("### %s" % SEL_LABEL[s])
        L.append("")
        L.append("| model | thinking | Batch x2.0 | Batch x2.75 | Batch x3.5 | Standard x2.75 | Batch $/1k items x2.75 |")
        L.append("|---|---|---:|---:|---:|---:|---:|")
        for mid, m in models.items():
            for tname, _ in thinking_scenarios(m):
                g = lambda t, x: idx[("main", s, "whole", mid, t, tname, x)]
                L.append("| %s | %s | %s | %s | %s | %s | %s |" % (m["label"], tname, fmt(g("batch", 2.0)["usd_total"]), fmt(g("batch", 2.75)["usd_total"]),
                         fmt(g("batch", 3.5)["usd_total"]), fmt(g("standard", 2.75)["usd_total"]), fmt(g("batch", 2.75)["usd_per_1000_items"])))
        L.append("")
    L.append("## 3. Whole-document vs chunked requests (Batch, multiplier 2.75): total $ and ratio to whole-document")
    L.append("")
    L.append("Chunked mode repeats the instruction in every request and pays the per-request thinking overhead every time; requests = ceil(sentences / N) per item.")
    L.append("")
    for s in SELECTIONS:
        L.append("### %s" % SEL_LABEL[s])
        L.append("")
        L.append("| model | thinking | whole | chunk-8 | x whole | chunk-1 | x whole | requests whole / c8 / c1 |")
        L.append("|---|---|---:|---:|---:|---:|---:|---|")
        for mid, m in models.items():
            for tname, _ in thinking_scenarios(m):
                w, c8, c1 = (idx[("main", s, md, mid, "batch", tname, 2.75)] for md in ("whole", "chunk8", "chunk1"))
                L.append("| %s | %s | %s | %s | %.2f | %s | %.2f | %s / %s / %s |" % (m["label"], tname, fmt(w["usd_total"]), fmt(c8["usd_total"]),
                         c8["usd_total"] / w["usd_total"], fmt(c1["usd_total"]), c1["usd_total"] / w["usd_total"],
                         "{:,}".format(w["requests"]), "{:,}".format(c8["requests"]), "{:,}".format(c1["requests"])))
        L.append("")
    L.append("## 4. 50-sentence pilot (%d sentences x %d candidate languages, multiplier 2.75)" % (PILOT_SENTS, len(PILOT_LANGS)))
    L.append("")
    L.append("Languages: %s. Mean SmolDoc sentence length %.1f Gemma tokens. Pilot sentences are translated one per request (chunk-1, 700 requests) or "
             "8 per request (chunk-8, 7 requests per language)." % (", ".join(PILOT_LANGS), meta["sent_tok_mean_sm"]))
    L.append("")
    L.append("| model | thinking | Batch chunk-1 | Batch chunk-8 | Standard chunk-1 | Standard chunk-8 |")
    L.append("|---|---|---:|---:|---:|---:|")
    sel = "pilot_%dsent_x_%dlang" % (PILOT_SENTS, len(PILOT_LANGS))
    for mid, m in models.items():
        for tname, _ in thinking_scenarios(m):
            g = lambda md, t: idx[("pilot", sel, md, mid, t, tname, 2.75)]["usd_total"]
            L.append("| %s | %s | %s | %s | %s | %s |" % (m["label"], tname, fmt(g("chunk1", "batch")), fmt(g("chunk8", "batch")), fmt(g("chunk1", "standard")), fmt(g("chunk8", "standard"))))
    L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", default=LONGCTX)
    ap.add_argument("--prices", default=os.path.join(LONGCTX, "gemini_prices_2026-10-23.json"))
    ap.add_argument("--out-csv", default=os.path.join(LONGCTX, "gemini_cost_table.csv"))
    ap.add_argument("--out-md", default=os.path.join(REPO, "paper", "tables", "t_longctx_gemini_cost.md"))
    ap.add_argument("--phase", choices=["current", "later"], default="current", help="'later' = scheduled price rises (3.x Flash from 2027-01-01, Gemini 4 from 2026-12-01)")
    ap.add_argument("--src-basis", choices=["chars4", "gemma"], default="chars4")
    a = ap.parse_args()
    items = load_items(os.path.join(a.counts_dir, "gemini_counts_items.csv"))
    meta = load_meta(os.path.join(a.counts_dir, "gemini_counts_meta.csv"))
    prices = json.load(open(a.prices, encoding="utf-8"))
    rows, sel_rows = build(items, meta, prices, a.phase, a.src_basis)
    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.makedirs(os.path.dirname(a.out_md), exist_ok=True)
    with open(a.out_md, "w", encoding="utf-8") as f:
        f.write(render_md(rows, sel_rows, meta, prices, a.phase, a.src_basis))
    print("wrote", a.out_csv, len(rows), "rows;", a.out_md)


if __name__ == "__main__":
    main()
