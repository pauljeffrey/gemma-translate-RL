#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Markdown tables from the JSON of scripts/eval_anchors_v2.py (D-61 report)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

DET_ORDER = ["copy", "func", "pad", "repeat", "len", "wordratio", "lexical", "lexicon", "gate_copy", "gate_loop", "gate_len"]
SAMPLE_COLS = DET_ORDER


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--weights", default="func=0.3,pad=1.0,repeat=1.0,len=1.0,wordratio=1.0,lexical=1.0,lexicon=0.2", help="detector=weight used for the combined penalty (len/wordratio/lexical are already weighted in eval)")
    ap.add_argument("--only", default=None)
    ap.add_argument("--cal", default=None, help="calibration json: the pad AUC column is then split into languages with / without a bigram table")
    a = ap.parse_args()
    E = json.load(open(a.eval))
    L = E["languages"]
    w = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(","))}
    isos = sorted(L) if not a.only else a.only.split(",")
    print(f"## false positives on held-out human references (fold {E['fold']}), mean over {len(isos)} languages\n")
    print("| detector | judged languages | mean FP share | max FP share | languages > 2 % |\n|---|---|---|---|---|")
    for d in DET_ORDER:
        v = [L[i]["fp"][d]["share"] for i in isos if d in L[i]["fp"]]
        if v:
            print(f"| {d} | {len(v)} | {np.mean(v):.3f} | {np.max(v):.3f} | {sum(1 for x in v if x > 0.02)} |")
    attacks = sorted({n for i in isos for n in L[i]["auc"]})
    print("\n## AUC of the uncapped raw score, human vs attack (mean over languages with the detector on; [n languages])\n")
    bg = set()
    if a.cal:
        bg = {i for i, b in json.load(open(a.cal))["languages"].items() if (b.get("pad") or {}).get("has_bigrams") and (b["pad"].get("enabled", True))}
    print("| attack | copy | func | pad (bigram-gain languages) | pad (type/token ratio only) |\n|---|---|---|---|---|")
    for n in attacks:
        cells = []
        for d, sel in (("copy", None), ("func", None), ("pad", True), ("pad", False)):
            v = [L[i]["auc"][n][d] for i in isos if n in L[i]["auc"] and d in L[i]["auc"][n] and not np.isnan(L[i]["auc"][n][d]) and (sel is None or ((i in bg) == sel))]
            cells.append(f"{np.mean(v):.2f} [{len(v)}]" if v else "-")
        print(f"| {n} | " + " | ".join(cells) + " |")
    print("\n## share of attacks with deficit > 0 / mean deficit (all judged languages)\n")
    print("| attack | " + " | ".join(DET_ORDER) + " |\n|---|" + "---|" * len(DET_ORDER))
    for n in attacks:
        cells = []
        for d in DET_ORDER:
            v = [L[i]["det"][n][d] for i in isos if n in L[i]["det"] and d in L[i]["det"][n]]
            cells.append(f"{np.mean([x['share'] for x in v]):.2f}/{np.mean([x['mean'] for x in v]):.2f}" if v else "-")
        print(f"| {n} | " + " | ".join(cells) + " |")
    hv = [d for d in DET_ORDER if any(d in L[i]["fp"] for i in isos)]
    cells = []
    for d in DET_ORDER:
        v = [L[i]["fp"][d] for i in isos if d in L[i]["fp"]]
        cells.append(f"{np.mean([x['share'] for x in v]):.2f}/{np.mean([x['mean'] for x in v]):.2f}" if v else "-")
    print("| **human** | " + " | ".join(cells) + " |")
    # combined penalty (sum of weighted deficits, floor at 2) per sample
    print("\n## combined penalty (min(2, sum of weight x deficit)); copy enters as 2 x fraction; gates = 2\n")
    print("| attack | mean | share > human p95 | share >= 1.0 |\n|---|---|---|---|")

    def comb(row):
        copy, func, pad, rep, ln, wr, lx, lex, g1, g2, g3 = row
        p = 0.0
        p += w.get("func", 0) * (func or 0) + w.get("pad", 0) * (pad or 0) + w.get("repeat", 0) * (rep or 0) + (ln or 0) + (wr or 0) + (lx or 0)
        p += w.get("lexicon", 0) * (lex or 0) + 2.0 * (copy or 0)
        if g1 or g2 or g3:
            p = 2.0
        return min(2.0, p)

    names = ["human"] + attacks
    for n in names:
        per_lang = []
        for i in isos:
            s = L[i]["samples"].get(n)
            h = L[i]["samples"].get("human")
            if not s or not h:
                continue
            ph = np.array([comb(r) for r in h])
            ps = np.array([comb(r) for r in s])
            per_lang.append((ps.mean(), float((ps > np.quantile(ph, 0.95)).mean()), float((ps >= 1.0).mean())))
        if per_lang:
            m = np.mean(per_lang, axis=0)
            print(f"| {n} | {m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f} |")
    print("\n## corruption classes of agent A5 (FLORES+ dev, fold B): share with deficit > 0\n")
    cls = sorted(E.get("hack", {}))
    print("| class | copy | func | pad | repeat | gate_copy | gate_loop | gate_len |\n|---|---|---|---|---|---|---|---|")
    for c in cls:
        rows = [r for i, v in E["hack"][c].items() if i in isos for r in v]
        cells = []
        for j in range(7):
            v = [r[j] for r in rows if r[j] is not None]
            cells.append(f"{np.mean(np.asarray(v) > 0):.2f}" if v else "-")
        print(f"| {c} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
