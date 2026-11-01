# SPDX-License-Identifier: AGPL-3.0-or-later
"""Aggregation of sentence-level QE into a document reward (R8, diagnosis (b)): mean vs min vs p25 vs CVaR vs mean - lambda * spread, for AfriCOMET-QE, SSA-COMET-MTL and the ensemble.

  python agg_study.py --chunks chunks_main.pkl --out agg_study.json

Input: chunk_study.py records (per item and alignment mode: plan items [(segment | None, weight)], shaped per-chunk rewards r1 / r2 / rens). A failed chunk counts as -2.
Tables:
  paired AUC      P(score(sentfix) > score(corrupted sentfix)) over the same (document, language)   (sensitivity to a localised failure)
  robust AUC      P(score(NLLB or window translation of the same document) > score(corrupted sentfix))  (the corruption must also fall below another GOOD system, i.e. below the noise of good output)
  damage rho      Spearman between the score and the known damage over the 8 graded versions of each document (none, drop1, drop3, drop10, trunc80, trunc50, repl10, repl30)
  spread          sd of the score of the good systems (sentfix, nllb, windows) inside a bin: the noise floor"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
from collections import defaultdict

PEN = -2.0
DAMAGE = {"sf": 0.0, "C:drop1": 0.02, "C:drop3": 0.05, "C:drop10": 0.10, "C:trunc80": 0.20, "C:repl10": 0.10, "C:repl30": 0.30, "C:trunc50": 0.50}
CORR = ["C:drop1", "C:drop3", "C:drop10", "C:dup1", "C:dup3", "C:loop_tail_sent", "C:loop_tail_word", "C:trunc80", "C:trunc50", "C:repl10", "C:repl30"]


def wq(vals, ws, q):
    tot = sum(ws)
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    acc = 0.0
    for i in order:
        acc += ws[i]
        if acc >= q * tot - 1e-12:
            return vals[i]
    return vals[order[-1]]


def cvar(vals, ws, q):
    tot = sum(ws)
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    budget, acc, s = q * tot, 0.0, 0.0
    for i in order:
        t = min(ws[i], budget - acc)
        if t <= 0:
            break
        s += t * vals[i]
        acc += t
    return s / max(acc, 1e-12)


def aggs(vals, ws):
    tot = sum(ws)
    m = sum(v * w for v, w in zip(vals, ws)) / tot
    sd = math.sqrt(sum(w * (v - m) ** 2 for v, w in zip(vals, ws)) / tot)
    return {"mean": m, "min": min(vals), "p25": wq(vals, ws, 0.25), "p10": wq(vals, ws, 0.10), "cvar25": cvar(vals, ws, 0.25),
            "mean+cvar25": 0.5 * m + 0.5 * cvar(vals, ws, 0.25), "mean+p25": 0.5 * m + 0.5 * wq(vals, ws, 0.25), "mean-0.5sd": m - 0.5 * sd, "mean-1sd": m - sd}


def doc_scores(rec, chan):
    items = rec["items"]
    if not items:
        return {k: PEN for k in aggs([PEN], [1.0])}
    vals = [PEN if seg is None else rec[chan][seg] for seg, _ in items]
    ws = [w for _, w in items]
    return aggs(vals, ws)


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    # rank-sum with ties
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j + 2) / 2.0
        i = j + 1
    rp = sum(r for r, (v, l) in zip(ranks, allv) if l == 1)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def spearman(x, y):
    def rk(a):
        o = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and a[o[j + 1]] == a[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rk(x), rk(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", nargs="*", default=["v1", "v2"])
    a = ap.parse_args()
    d = pickle.load(open(a.chunks, "rb"))
    recs = d["records"]
    keys = defaultdict(dict)
    for (doc, iso, cond, mode), rec in recs.items():
        keys[(doc, iso, mode)][cond] = rec
    bin_of = {(it["doc_id"], it["iso"]): it["bin"] for it in d["items"]}
    res = {}
    for mode in a.modes:
        for chan in ("r1", "r2", "rens"):
            tables = defaultdict(lambda: defaultdict(list))             # (bin, kind) -> aggregator -> list
            pair = defaultdict(lambda: defaultdict(list))               # (bin, corruption) -> aggregator -> [(sf, corrupt, nllb, win)]
            damage = defaultdict(lambda: defaultdict(list))
            for (doc, iso, m), conds in keys.items():
                if m != mode or "sf" not in conds:
                    continue
                b = bin_of[(doc, iso)]
                sc = {c: doc_scores(r, chan) for c, r in conds.items()}
                for g in ("sf", "nllb", "S:winfix"):
                    if g in sc:
                        for k, v in sc[g].items():
                            tables[(b, g)][k].append(v)
                for c in CORR:
                    if c in sc:
                        for k in sc["sf"]:
                            others = [sc[g][k] for g in ("nllb", "S:winfix") if g in sc]
                            pair[(b, c)][k].append((sc["sf"][k], sc[c][k], others))
                if all(c in sc for c in DAMAGE):
                    for k in sc["sf"]:
                        damage[b][k].append(spearman([sc[c][k] for c in DAMAGE], [DAMAGE[c] * -1 for c in DAMAGE]))
            out = {"paired_auc": {}, "robust_auc": {}, "damage_rho": {}, "good_sd": {}, "good_mean": {}}
            for (b, c), per in pair.items():
                for k, lst in per.items():
                    paired = [1.0 if s > x else (0.5 if s == x else 0.0) for s, x, _ in lst]
                    rob = [1.0 if o > x else (0.5 if o == x else 0.0) for _, x, os_ in lst for o in os_]
                    out["paired_auc"].setdefault(k, {})[f"b{b}:{c}"] = sum(paired) / len(paired)
                    out["robust_auc"].setdefault(k, {})[f"b{b}:{c}"] = sum(rob) / max(len(rob), 1)
            for b, per in damage.items():
                for k, v in per.items():
                    out["damage_rho"].setdefault(k, {})[f"b{b}"] = sum(v) / len(v)
            for (b, g), per in tables.items():
                for k, v in per.items():
                    out["good_sd"].setdefault(k, {})[f"b{b}:{g}"] = statistics.pstdev(v)
                    out["good_mean"].setdefault(k, {})[f"b{b}:{g}"] = statistics.mean(v)
            res[f"{mode}:{chan}"] = out
    json.dump(res, open(a.out, "w"), indent=1)
    # compact print: mean over bins >= 1024 of the paired / robust AUC per corruption group and aggregator, ensemble channel
    for mode in a.modes:
        for chan in ("rens", "r1", "r2"):
            o = res[f"{mode}:{chan}"]
            print(f"\n== {mode} {chan}: paired / robust AUC averaged over bins b1024-b4096; damage rho (b1024-b4096)")
            hdr = ["mean", "min", "p25", "cvar25", "mean+cvar25", "mean+p25", "mean-0.5sd", "mean-1sd"]
            print("corruption".ljust(18) + "".join(h.ljust(14) for h in hdr))
            for c in CORR:
                line = c[2:].ljust(18)
                for h in hdr:
                    pa = [o["paired_auc"][h][f"b{b}:{c}"] for b in (1024, 2048, 4096) if f"b{b}:{c}" in o["paired_auc"][h]]
                    ro = [o["robust_auc"][h][f"b{b}:{c}"] for b in (1024, 2048, 4096) if f"b{b}:{c}" in o["robust_auc"][h]]
                    line += f"{sum(pa)/max(len(pa),1):.2f}/{sum(ro)/max(len(ro),1):.2f}".ljust(14)
                print(line)
            line = "damage rho".ljust(18)
            for h in hdr:
                v = [o["damage_rho"][h][f"b{b}"] for b in (1024, 2048, 4096) if f"b{b}" in o["damage_rho"][h]]
                line += f"{sum(v)/max(len(v),1):.2f}".ljust(14)
            print(line)
            line = "sd good sf b2048".ljust(18)
            for h in hdr:
                line += f"{o['good_sd'][h].get('b2048:sf', float('nan')):.3f}".ljust(14)
            print(line)


if __name__ == "__main__":
    main()
