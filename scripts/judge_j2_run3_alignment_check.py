"""Run 3 diagnostic (free, no API): is a low judge score of a whole-document unit a translation failure or a misalignment?

For each judged unit (system, document, source sentences [start, end)) the unit's output text is compared (character n-gram F-score, n = 1..6, beta = 2, a
self-contained chrF without word n-grams; diagnostic only) with the NLLB-3.3B pseudo-reference of (a) its own source block and (b) the same-size blocks shifted by
-3..+3 sentences. If a shifted block matches clearly better than the own block, the unit is flagged "possibly misaligned" (best shifted score >= 30 and >= own + 15).
Reports, per system: share of flagged units, mean own-block chrF, mean judge score of flagged / unflagged units, and the Spearman correlation judge vs own-block chrF.

    python judge_j2_run3_alignment_check.py --lc1 <extracted lc1 dir> --dir <reports/data/judge_j2_2026-10-25>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ngr(s, n):
    s = "".join(s.split())
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(h, r, beta=2.0):
    ps, rs = [], []
    for n in range(1, 7):
        a, b = ngr(h, n), ngr(r, n)
        if not a or not b:
            continue
        m = sum((a & b).values())
        ps.append(m / sum(a.values()))
        rs.append(m / sum(b.values()))
    if not ps:
        return 0.0
    p, r_ = np.mean(ps), np.mean(rs)
    return 0.0 if p + r_ == 0 else 100 * (1 + beta ** 2) * p * r_ / (beta ** 2 * p + r_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc1", required=True)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    D, lc = Path(a.dir), Path(a.lc1)
    refs = {}
    for fn in ("refs_nllb_core.jsonl",):
        for l in open(lc / "work" / fn, encoding="utf-8"):
            r = json.loads(l)
            refs[r["iid"]] = r["ref_sents"]
    import gzip
    pr = {}
    for l in gzip.open(D / "prompts_run3_longdoc_qe.jsonl.gz", "rt", encoding="utf-8"):
        d = json.loads(l)
        pr[d["id"]] = d["prompt"]
    u = pd.read_csv(D / "run3_units_scored.csv")
    rows = []
    for _, r in u.iterrows():
        rs = refs.get(f"core|{r.doc_id}|{r.lang}")
        hyp = pr[r.id].rsplit("Translation: ", 1)[1].rsplit(" Score: ", 1)[0]
        k = int(r.block_sents)
        own = chrf(hyp, " ".join(rs[int(r.start):int(r.end)]))
        best = 0.0
        for sh in (-3, -2, -1, 1, 2, 3):
            s0 = int(r.start) + sh
            if s0 < 0 or s0 + k > len(rs):
                continue
            best = max(best, chrf(hyp, " ".join(rs[s0:s0 + k])))
        rows.append(dict(id=r.id, system=r.system, lang=r.lang, bin=r.bin, block_sents=k, score=r.score, chrf_own=own, chrf_best_shift=best,
                         possibly_misaligned=int(best >= 30 and best >= own + 15)))
    T = pd.DataFrame(rows)
    T.to_csv(D / "run3_alignment_check_units.csv", index=False)
    out = []
    for s, g in T.groupby("system"):
        f = g[g.possibly_misaligned == 1]
        nf = g[g.possibly_misaligned == 0]
        out.append(dict(system=s, units=len(g), share_possibly_misaligned=len(f) / len(g), mean_chrf_own_vs_nllb_ref=g.chrf_own.mean(),
                        judge_mean_flagged=f.score.mean() if len(f) else np.nan, judge_mean_unflagged=nf.score.mean(),
                        spearman_judge_vs_chrf_own=spearmanr(g.score, g.chrf_own)[0]))
    O = pd.DataFrame(out)
    O.to_csv(D / "run3_alignment_check.csv", index=False)
    print(O.round(3).to_string())


if __name__ == "__main__":
    main()
