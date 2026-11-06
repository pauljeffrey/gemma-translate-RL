#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fluency model v2 on HUMAN reference text (FLORES+ dev): char 5-gram + word-order component, three training sets, hard sibling negatives.

    python scripts/fluency_eval_flores.py --flores flores_dev.tsv --train-old texts_train.tsv --train-human mono_train.tsv --siblings ft_per_language.csv --out DIR

Per language with FLORES dev text (test = FLORES sentences, never in the models): training sets A = translation targets, B = human monolingual (web + Wikipedia), C = A + B. For each, AUC of
human vs damaged: shuffle (word order), strip (diacritics), garble (15 % characters), foreign (random other language), sibling (the most confused neighbour language: hard negative).
Scores: char (mean log-prob per char), word (word-order PMI), comb (z-scored char + z-scored word, z from 300 held-out training sentences of the same language).
"""
import argparse, collections, csv, math, os, random, sys
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fluency_lm as F


def load(p):
    by = collections.defaultdict(list)
    for l in open(p):
        if "\t" in l:
            i, t = l.rstrip("\n").split("\t", 1); by[i].append(t)
    return by


G = {}


def work(iso):
    rng = random.Random(7)
    flo = G["flores"][iso][:400]
    sib = G["sib"].get(iso)
    pool_sib = (G["flores"].get(sib) or G["mono_test"].get(sib) or [])[:400]
    pool_all = [t for i, ts in G["flores"].items() if i != iso for t in ts[:5]]
    out = []
    for name, tr in (("A", G["old"].get(iso, [])[:3000]), ("B", G["human"].get(iso, [])[:12000]), ("C", (G["old"].get(iso, [])[:3000] + G["human"].get(iso, [])[:9000]))):
        if len(tr) < 500: continue
        hold, fit = tr[:300], tr[300:]
        cm = F.CharLM(5).fit(fit).trim(2); wm = F.WordPMI().fit(fit)
        hc = np.array([cm.score(t) for t in hold]); hw = np.array([x for x in (wm.score(t) for t in hold) if x is not None])
        mu_c, sd_c = hc.mean(), hc.std() + 1e-6; mu_w, sd_w = (hw.mean(), hw.std() + 1e-6) if len(hw) > 20 else (0.0, 1.0)

        def sc(t):
            c = cm.score(t); w = wm.score(t)
            return c, w, (c - mu_c) / sd_c + (((w - mu_w) / sd_w) if w is not None else 0.0)
        H = [sc(t) for t in flo]
        chars = sorted(set("".join(fit[:300])))
        dmg = {"shuffle": [], "strip": [], "garble": [], "foreign": [], "sibling": []}
        for t in flo:
            w = t.split()
            if len(w) >= 4:
                w2 = w[:]; rng.shuffle(w2)
                if w2 != w: dmg["shuffle"].append(" ".join(w2))
            s = F.strip_marks(t)
            if s != t: dmg["strip"].append(s)
            dmg["garble"].append("".join(rng.choice(chars) if rng.random() < 0.15 else c for c in t))
            dmg["foreign"].append(rng.choice(pool_all))
            if pool_sib: dmg["sibling"].append(rng.choice(pool_sib))
        row = dict(iso=iso, train=name, n_flores=len(flo))
        for k, v in dmg.items():
            if len(v) < 20: continue
            D = [sc(x) for x in v]
            row[f"{k}_char"] = round(F.auc([h[0] for h in H], [d[0] for d in D]), 3)
            ww = [(h[1]) for h in H if h[1] is not None]; dw = [d[1] for d in D if d[1] is not None]
            row[f"{k}_word"] = round(F.auc(ww, dw), 3) if len(dw) >= 20 else float("nan")
            row[f"{k}_comb"] = round(F.auc([h[2] for h in H], [d[2] for d in D]), 3)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--flores", required=True); ap.add_argument("--train-old", required=True); ap.add_argument("--train-human", required=True)
    ap.add_argument("--mono-test"); ap.add_argument("--siblings"); ap.add_argument("--out", required=True); ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    G["flores"] = load(a.flores); G["old"] = load(a.train_old); G["human"] = load(a.train_human); G["mono_test"] = load(a.mono_test) if a.mono_test else {}
    G["sib"] = {}
    if a.siblings:
        for r in csv.DictReader(open(a.siblings)):
            tc = r["top_confusions"].split(" ")[0].split(":")[0] if r["top_confusions"] else ""
            if tc: G["sib"][r["iso"]] = tc
    isos = sorted(i for i in G["flores"] if len(G["flores"][i]) >= 200)
    with mp.get_context("fork").Pool(a.procs) as p:
        rows = [r for rs in p.imap_unordered(work, isos) for r in rs]
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("iso", "train", "n_flores"), k))
    with open(os.path.join(a.out, "fluency_eval_flores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(sorted(rows, key=lambda r: (r["iso"], r["train"])))
    print("languages", len(isos))
    for tr in "ABC":
        rs = [r for r in rows if r["train"] == tr]
        line = []
        for k in ("shuffle_char", "shuffle_word", "shuffle_comb", "strip_char", "garble_comb", "foreign_comb", "sibling_comb"):
            v = [r[k] for r in rs if k in r and not math.isnan(r[k])]
            line.append(f"{k} {np.mean(v):.3f} ({sum(x >= 0.8 for x in v)}/{len(v)})")
        print(f"train {tr} n={len(rs)}:", " | ".join(line))


if __name__ == "__main__":
    main()
