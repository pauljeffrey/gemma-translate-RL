#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-language CHARACTER n-gram fluency model for the RL reward (reference-free, CPU only, a few MB per language).

    python scripts/fluency_lm.py validate --train texts_train.tsv --test texts_test.tsv --out reports/data/fluency_validation_2026-11-01 [--order 5]

Model: interpolated Witten-Bell character n-gram (order 5 by default) trained on the monolingual text of ONE language (here: the target side of the training pairs, tab separated `iso<TAB>text`).
Score of a sentence = mean log-probability per character (natural log, with a start/end marker). It measures how much the sentence looks like text of that language (spelling, diacritics, word shapes), NOT
whether it means the source: fluency, not adequacy. For the RL reward it is used only as a PENALTY below a per-language threshold (z-score against human text), never as a bonus.

`validate` is the validity test of the AfriCOMET kind: for every language, human test sentences against three kinds of damage, AUC = P(score(human) > score(damaged)):
    shuffle   words in a random order (sentences with >= 4 words)
    strip     diacritics / non-ASCII letters removed or flattened (only sentences that change)
    garble    15 % of the characters replaced by random characters of the same language
    foreign   a random sentence of another language of the pool
Output: fluency_validation.csv (per language: n, mean score, p10, std, AUCs), summary.txt.
"""
import argparse, collections, csv, math, os, pickle, random, sys, unicodedata
import numpy as np

BOS, EOS = "\x02", "\x03"


class CharLM:
    def __init__(self, order=5):
        self.n = order
        self.cnt = [collections.Counter() for _ in range(order + 1)]       # cnt[k][context+char] for context length k-1
        self.ctx = [collections.Counter() for _ in range(order + 1)]       # total count of a context
        self.typ = [collections.Counter() for _ in range(order + 1)]       # number of distinct followers of a context
        self.V = 1

    def fit(self, texts):
        chars = set()
        for t in texts:
            s = BOS * (self.n - 1) + t + EOS
            chars.update(t)
            for i in range(self.n - 1, len(s)):
                for k in range(1, self.n + 1):
                    g = s[i - k + 1:i + 1]
                    if self.cnt[k][g] == 0: self.typ[k][g[:-1]] += 1
                    self.cnt[k][g] += 1; self.ctx[k][g[:-1]] += 1
        self.V = len(chars) + 2
        return self

    def prob(self, hist, c):
        p = 1.0 / self.V
        for k in range(1, self.n + 1):
            h = hist[len(hist) - (k - 1):] if k > 1 else ""
            tot = self.ctx[k].get(h, 0)
            if tot == 0: continue
            lam = tot / (tot + self.typ[k].get(h, 0))
            p = lam * self.cnt[k].get(h + c, 0) / tot + (1 - lam) * p
        return p

    def score(self, t):
        s = BOS * (self.n - 1) + t + EOS; lp = 0.0
        for i in range(self.n - 1, len(s)):
            lp += math.log(self.prob(s[max(0, i - self.n + 1):i], s[i]))
        return lp / (len(t) + 1)

    def trim(self, min_count=2):
        for k in range(1, self.n + 1):
            keep = {g for g, c in self.cnt[k].items() if c >= min_count}
            self.cnt[k] = collections.Counter({g: self.cnt[k][g] for g in keep})
        return self


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    allv = np.concatenate([pos, neg]); order = allv.argsort(kind="mergesort"); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks); ranks = (sums / cnts)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def strip_marks(t):
    d = unicodedata.normalize("NFD", t); d = "".join(c for c in d if not unicodedata.combining(c))
    return "".join(c if ord(c) < 128 or not c.isalpha() else "?" for c in d) if False else d


def load(p):
    by = collections.defaultdict(list)
    for l in open(p):
        if "\t" in l:
            i, t = l.rstrip("\n").split("\t", 1); by[i].append(t)
    return by


def validate(a):
    rng = random.Random(3)
    train, test = load(a.train), load(a.test); os.makedirs(a.out, exist_ok=True)
    pool = [(i, t) for i, ts in test.items() for t in ts[:50]]
    rows = []; models = {}
    for iso in sorted(set(train) & set(test)):
        if len(train[iso]) < 200: continue
        m = CharLM(a.order).fit(train[iso]).trim(2); models[iso] = m
        hs = [t for t in test[iso] if len(t) >= 20][:a.n_test]
        h = np.array([m.score(t) for t in hs])
        chars = sorted(set("".join(train[iso][:300])))
        dmg = {"shuffle": [], "strip": [], "garble": [], "foreign": []}
        for t in hs:
            w = t.split()
            if len(w) >= 4:
                w2 = w[:]; rng.shuffle(w2)
                if w2 != w: dmg["shuffle"].append(" ".join(w2))
            s = strip_marks(t)
            if s != t: dmg["strip"].append(s)
            dmg["garble"].append("".join(rng.choice(chars) if rng.random() < 0.15 else c for c in t))
            j, f = rng.choice(pool)
            if j != iso: dmg["foreign"].append(f)
        res = {k: auc(h, [m.score(x) for x in v]) if v else float("nan") for k, v in dmg.items()}
        rows.append(dict(iso=iso, n_train=len(train[iso]), n_test=len(hs), mean=round(float(h.mean()), 3), p10=round(float(np.percentile(h, 10)), 3), std=round(float(h.std()), 3),
                         **{f"auc_{k}": round(v, 3) for k, v in res.items()}, n_strip=len(dmg["strip"])))
        print(iso, rows[-1]["auc_shuffle"], rows[-1]["auc_strip"], rows[-1]["auc_garble"], rows[-1]["auc_foreign"], flush=True)
    with open(os.path.join(a.out, "fluency_validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        for k in ("shuffle", "strip", "garble", "foreign"):
            v = [r[f"auc_{k}"] for r in rows if not math.isnan(r[f"auc_{k}"])]
            f.write(f"{k}: mean AUC {np.mean(v):.3f}, min {np.min(v):.3f}, languages with AUC >= 0.8: {sum(x >= 0.8 for x in v)}/{len(v)}\n")
        weak = [(r["iso"], r["auc_shuffle"], r["auc_garble"], r["auc_foreign"]) for r in rows if min(r["auc_garble"], r["auc_foreign"]) < 0.8]
        f.write(f"weak (garble or foreign AUC < 0.8): {weak}\n")
    pickle.dump({k: m for k, m in models.items()}, open(os.path.join(a.out, "models.pkl"), "wb"))
    print(open(os.path.join(a.out, "summary.txt")).read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd")
    p = sp.add_parser("validate"); p.add_argument("--train", required=True); p.add_argument("--test", required=True); p.add_argument("--out", required=True)
    p.add_argument("--order", type=int, default=5); p.add_argument("--n-test", type=int, default=400); p.set_defaults(fn=validate)
    a = ap.parse_args(); a.fn(a)
