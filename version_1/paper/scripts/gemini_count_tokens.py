#!/usr/bin/env python
"""Box-side counting for the Gemini Batch cost model.  CPU only; writes small CSVs.

Run on the box:
    source /workspace/jobs/lc1/env_lc.sh
    cd /workspace/jobs/lc1/gemini      # contains gemini_batch_common.py + this script
    nice -n 19 $PYG gemini_count_tokens.py --work /workspace/jobs/lc1/work --out /workspace/jobs/lc1/gemini/counts

English tokens are counted with the Gemma-3 tokenizer (stand-in for the Gemini tokenizer, which cannot be called
offline); a chars/4 estimate is written next to it as a sanity check.  Outputs:
  gemini_counts_items.csv    one row per benchmark item (document x language)
  gemini_counts_summary.csv  per (set, bin, iso): documents, items, tokens, sentences, ...
  gemini_counts_meta.csv     key,value (instruction overhead in tokens per request, pilot sentence statistics)
"""
import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemini_batch_common as C  # noqa: E402


def rd(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/jobs/lc1/work")
    ap.add_argument("--out", default="counts")
    ap.add_argument("--tok", default="/workspace/sft_build/runs/final_1433")
    ap.add_argument("--chunk-sizes", default="1,8")
    ap.add_argument("--no-tokenizer", action="store_true", help="use chars/4 only (tests)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sizes = [int(x) for x in a.chunk_sizes.split(",")]

    if a.no_tokenizer:
        def ntok(texts):
            return [C.est_tokens_chars4(t) for t in texts]
    else:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(a.tok)

        def ntok(texts):
            return [len(x) for x in tk(list(texts), add_special_tokens=False)["input_ids"]]

    docs = {d["doc_id"]: d for d in rd(os.path.join(a.work, "docs.jsonl"))}
    items = rd(os.path.join(a.work, "items.jsonl"))

    # per-document stats (computed once)
    st = {}
    ids = sorted(docs)
    dtok = ntok([docs[i]["text"] for i in ids])
    for i, t in zip(ids, dtok):
        d = docs[i]
        sents = d["sents"] if isinstance(d["sents"], list) else []
        stok = ntok(sents) if sents else []
        st[i] = dict(src_tok=t, src_tok_stored=d.get("src_tokens"), chars=len(d["text"]),
                     chars4=C.est_tokens_chars4(d["text"]), n_sents=len(sents), n_paras=len(d.get("para_starts", [])) or 1,
                     sent_tok=sum(stok), sent_toks=stok)

    cols = ["iid", "set", "bin", "doc_id", "iso", "src_tok", "src_tok_stored", "src_chars", "chars4", "n_sents", "n_paras",
            "sent_tok_sum"] + ["n_chunks_%d" % n for n in sizes]
    summ = {}
    with open(os.path.join(a.out, "gemini_counts_items.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for it in items:
            s = st[it["doc_id"]]
            w.writerow([it["iid"], it["set"], it["bin"], it["doc_id"], it["iso"], s["src_tok"], s["src_tok_stored"], s["chars"],
                        s["chars4"], s["n_sents"], s["n_paras"], s["sent_tok"]] + [len(C.chunk_spans(s["n_sents"], n)) for n in sizes])
            k = (it["set"], it["bin"], it["iso"])
            g = summ.setdefault(k, dict(docs=set(), items=0, src_tok=0, chars4=0, n_sents=0, sent_tok_sum=0))
            g["docs"].add(it["doc_id"])
            g["items"] += 1
            g["src_tok"] += s["src_tok"]
            g["chars4"] += s["chars4"]
            g["n_sents"] += s["n_sents"]
            g["sent_tok_sum"] += s["sent_tok"]
    with open(os.path.join(a.out, "gemini_counts_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "bin", "iso", "documents", "items", "src_tok_gemma", "src_tok_chars4", "sentences", "sent_tok_sum"])
        for k in sorted(summ):
            g = summ[k]
            w.writerow(list(k) + [len(g["docs"]), g["items"], g["src_tok"], g["chars4"], g["n_sents"], g["sent_tok_sum"]])

    # instruction overhead (token counts of the exact system instructions, per request) -------------------------
    meta = [("framing_tokens_assumed", C.FRAMING_TOKENS)]
    langs = sorted(C.LANG_NAMES)
    wt = ntok([C.instruction(l, "whole") for l in langs])
    ct = ntok([C.instruction(l, "chunk", 8) for l in langs])
    meta += [("instr_tokens_whole_mean", round(statistics.mean(wt), 1)), ("instr_tokens_whole_max", max(wt)),
             ("instr_tokens_chunk_mean", round(statistics.mean(ct), 1)), ("instr_tokens_chunk_max", max(ct))]
    # pilot sentence statistics: all sentences of SmolDoc `sm` docs (have human references) and of all core docs
    for name, pred in (("sm", lambda d: d["bin"] == "sm"), ("core", lambda d: d["set"] == "core"), ("sci", lambda d: d["set"] == "sci")):
        toks = [x for i, d in docs.items() if pred(d) for x in st[i]["sent_toks"]]
        if toks:
            meta += [("sent_tok_mean_%s" % name, round(statistics.mean(toks), 2)), ("sent_tok_median_%s" % name, statistics.median(toks)),
                     ("sent_count_%s" % name, len(toks))]
    # documents / tokens ratio sanity
    tot_gemma = sum(s["src_tok"] for s in st.values())
    tot_c4 = sum(s["chars4"] for s in st.values())
    meta += [("docs_total", len(docs)), ("items_total", len(items)), ("doc_tok_gemma_total", tot_gemma), ("doc_tok_chars4_total", tot_c4),
             ("chars4_over_gemma", round(tot_c4 / max(1, tot_gemma), 4))]
    with open(os.path.join(a.out, "gemini_counts_meta.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(meta)
    print("wrote", a.out, "items", len(items), "docs", len(docs))


if __name__ == "__main__":
    main()
