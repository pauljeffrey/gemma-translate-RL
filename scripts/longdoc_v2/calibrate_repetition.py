# SPDX-License-Identifier: AGPL-3.0-or-later
"""Repetition limits of HUMAN documents per length bucket (R8, diagnosis (c) and the calibration of reward v2).

  python calibrate_repetition.py --out calibration/longdoc_v2.json --report rep_fp.json

Human text: SmolDoc references stitched to the median source length of each bin (45 languages that have SmolDoc references) and the NLLB reference translations of the core
set (pseudo-human, the b4096 bucket has no human text otherwise). For each bucket: quantiles of every repetition level, the false-positive rate of the v1 guards
(is_repetitive and its three rules, is_self_repetitive) and of the v2 ramp with the calibrated limits."""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from gemma_mt.rl import guards  # noqa: E402
from gemma_mt.rl import longdoc as LD  # noqa: E402
from gemma_mt.rl.segments import split_sentences  # noqa: E402


def v1_flags(text: str):
    words = text.split()
    run = 1
    maxrun = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        maxrun = max(maxrun, run)
    import zlib

    raw = text.encode()
    return {"run": maxrun > 4,
            "unique": len(words) >= 8 and len(set(w.lower() for w in words)) / len(words) < 0.3,
            "zlib": len(raw) >= 80 and len(zlib.compress(raw)) / len(raw) < 0.18,
            "selfrep": guards.is_self_repetitive(text)}


def q(vals, p):
    v = sorted(vals)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--per-cell", type=int, default=6, help="stitched documents per language and bin")
    ap.add_argument("--quantile", type=float, default=0.995)
    ap.add_argument("--margin", type=float, default=1.15)
    ap.add_argument("--langs-min-docs", type=int, default=150)
    a = ap.parse_args()
    edges = LD.DEFAULT_BUCKET_EDGES
    rows = []          # (bucket, kind, iso, profile, v1 flags, n_words, src_words)
    # --- human (SmolDoc stitched)
    langs = sorted(os.path.basename(p)[:-6] for p in glob.glob(C.CORE + "/human_smoldoc/refs/*.jsonl"))
    for iso in langs:
        pool = C.smol_equal_docs(iso)
        if len(pool) < a.langs_min_docs:
            continue
        for b, tw in C.BIN_WORDS.items():
            for k, (src, ref) in enumerate(C.stitched_human(iso, tw, a.per_cell, seed=7, pool=pool)):
                text = " ".join(ref)
                sw = sum(len(x.split()) for x in src)
                prof = LD.repetition_profile(text)
                rows.append((bisect(edges, sw), "human", iso, prof, v1_flags(text), prof.n_words, sw, b, LD.repetition_profile(" ".join(src))))
    print("human docs", len(rows), flush=True)
    # --- NLLB references of the core set (all bins)
    docs = C.core_docs()
    _sp = {}

    def src_prof(did, d):
        if did not in _sp:
            _sp[did] = LD.repetition_profile(d["text"])
        return _sp[did]

    nl_langs = sorted(os.path.basename(p).split(".")[0] for p in glob.glob(C.NLLB + "/*.refs.jsonl"))
    for iso in nl_langs:
        for did, r in C.nllb_refs(iso).items():
            d = docs.get(did)
            if d is None or not r.get("valid", True):
                continue
            text = " ".join(r["ref_sents"])
            sw = len(d["text"].split())
            prof = LD.repetition_profile(text)
            rows.append((bisect(edges, sw), "nllb", iso, prof, v1_flags(text), prof.n_words, sw, d["bin"], src_prof(did, d)))
    print("all docs", len(rows), flush=True)
    levels = list(LD.LEVELS)
    rep = {"buckets": []}

    def exc(r, lv):
        return max(0.0, r[3].levels[lv] - r[8].levels[lv])

    t0 = []
    for bk in range(len(edges) + 1):
        for kind in ("human", "nllb"):
            sub = [r for r in rows if r[0] == bk and r[1] == kind]
            if not sub:
                continue
            ent = {"bucket": bk, "kind": kind, "n": len(sub), "levels": {}, "v1_fp": {}}
            for lv in levels:
                v = [exc(r, lv) for r in sub]
                ent["levels"][lv] = {"p50": q(v, .5), "p95": q(v, .95), "p99": q(v, .99), "p995": q(v, .995), "max": max(v), "raw_p995": q([r[3].levels[lv] for r in sub], .995)}
            for fl in ("run", "unique", "zlib", "selfrep"):
                ent["v1_fp"][fl] = sum(1 for r in sub if r[4][fl]) / len(sub)
            ent["v1_fp"]["any_is_repetitive"] = sum(1 for r in sub if r[4]["run"] or r[4]["unique"] or r[4]["zlib"]) / len(sub)
            rep["buckets"].append(ent)
        # calibrated limit: HUMAN documents only (NLLB sentence translations do loop, so they are a test, not a reference), excess over the source, quantile * margin, with a floor
        sub = [r for r in rows if r[0] == bk and r[1] == "human"]
        if not sub:                                    # no human text in this bucket (the shortest one): the nearest bucket that has some
            sub = [r for r in rows if r[1] == "human" and r[0] == min({x[0] for x in rows if x[1] == "human"}, key=lambda b: abs(b - bk))]
        lim = {}
        for lv in levels:
            v = [exc(r, lv) for r in sub]
            lim[lv] = round(max(q(v, a.quantile) * a.margin, LD.LEVEL_FLOOR[lv]), 4)
        t0.append(lim)
    cal = {"edges": list(edges), "t0": t0, "t1": LD.DEFAULT_T1, "meta": {"quantile": a.quantile, "margin": a.margin, "n_docs": len(rows),
                                                                      "source": "SmolDoc human references stitched to the median source length of the bins (45 languages) + NLLB-200 3.3B references of the core set (16 languages)"}}
    # false positives of the calibrated ramp (any severity > 0) per bucket and kind, and the share with severity >= 0.5
    c = LD.LongdocCalibration(tuple(edges), t0, dict(LD.DEFAULT_T1))
    fp = defaultdict(lambda: [0, 0, 0])
    for bk, kind, iso, prof, f1, nw, sw, b, sp in rows:
        s, _ = c.severity(prof, sw, sp)
        for key in ((bk, kind), (bk, "all")):
            fp[key][0] += 1
            fp[key][1] += s > 0
            fp[key][2] += s >= 0.5
    rep["v2_fp"] = {f"{k[0]}:{k[1]}": {"n": v[0], "any": v[1] / v[0], "ge_half": v[2] / v[0]} for k, v in sorted(fp.items())}
    by_lang = defaultdict(lambda: [0, 0])
    for bk, kind, iso, prof, f1, nw, sw, b, sp in rows:
        s, _ = c.severity(prof, sw, sp)
        by_lang[iso][0] += 1
        by_lang[iso][1] += s > 0
    rep["v2_fp_by_language"] = {k: {"n": v[0], "any": v[1] / v[0]} for k, v in sorted(by_lang.items(), key=lambda kv: -kv[1][1] / kv[1][0])[:12]}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(cal, open(a.out, "w"), indent=1)
    json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps(cal["t0"], indent=0))
    for e in rep["buckets"]:
        print(e["bucket"], e["kind"], e["n"], "v1 FP", {k: round(v, 3) for k, v in e["v1_fp"].items()}, "p995", {k: round(v["p995"], 3) for k, v in e["levels"].items()})
    print(json.dumps(rep["v2_fp"]))
    print(json.dumps(rep["v2_fp_by_language"]))


def bisect(edges, x):
    import bisect as _b

    return _b.bisect_left(list(edges), x)


if __name__ == "__main__":
    main()
