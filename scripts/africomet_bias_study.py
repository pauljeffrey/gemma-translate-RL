#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Is AfriCOMET-QE biased against human ratings WITHIN a source? (owner question 2026-11-02; offline, CPU, numpy only)

Data: SSA-MTE test (human DA z-scores, ~6 systems per source, 11 pairs) with the stored scores of AfriCOMET-QE-STL-1.1 and SSA-COMET-MTL (reference-free) from the D-53 study (box_snapshot_2026-10-21/jobs/a5).
GRPO standardises the reward inside the group of samples for ONE prompt, so only the ranking within a source matters: everything is demeaned within (language pair, source).
1. within-source agreement of each metric and of their mean with the human z;
2. which cheap features of the output explain the human-minus-metric residual (length ratio, repetition, non-ASCII share, copy overlap, short output);
3. leave-one-language-pair-out ridge correction on features only (never language identity): does it raise the held-out within-source correlation?
"""
import json, os, sys, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import qe_choice_study as Q

A5 = os.environ.get("A5", os.path.join(HERE, "..", "..", "box_snapshot_2026-10-21", "jobs", "a5"))
OUT = os.environ.get("OUT", os.path.join(HERE, "..", "..", "reports", "data", "africomet_bias_2026-11-02"))


def rows_ssamte():
    rows = []
    for cfg, stem in Q.SSAMTE_FILES.items():
        f = os.path.join(A5, "ssamte", "test", (stem if stem else cfg) + ".json")
        if not os.path.exists(f): f = os.path.join(HERE, "..", "data_pipeline", "code-switch", "work", "ssa_mte_full", "test_" + stem + ".json")     # Kikuyu is not in the box snapshot
        for i, l in enumerate(open(f, encoding="utf-8")):
            r = json.loads(l)
            rows.append(dict(lp=cfg, src=r["src_sent"], mt=r["tgt_sent"], z=float(r["z_score"])))
    return rows


def feats(src, mt):
    sw, mw = src.split(), mt.split()
    lr = math.log((len(mt) + 5) / (len(src) + 5))
    rep = 1.0 - len(set(mw)) / max(1, len(mw))
    nonascii = sum(1 for c in mt if c.isalpha() and ord(c) > 127) / max(1, sum(c.isalpha() for c in mt))
    jac = len(set(w.lower() for w in sw) & set(w.lower() for w in mw)) / max(1, len(set(w.lower() for w in sw) | set(w.lower() for w in mw)))
    short = 1.0 if len(mw) < 0.5 * len(sw) else 0.0
    return [lr, rep, nonascii, jac, short]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0: return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    rows = rows_ssamte()
    A = np.array(json.load(open(os.path.join(A5, "ssamte_scores", "africomet_qe_stl11.json"))))
    S = np.array(json.load(open(os.path.join(A5, "ssamte_scores", "ssa_mtl_qe.json"))))
    assert len(rows) == len(A) == len(S), (len(rows), len(A), len(S))
    zA = (A - A.mean()) / A.std(); zS = (S - S.mean()) / S.std(); E = (zA + zS) / 2
    H = np.array([r["z"] for r in rows]); F = np.array([feats(r["src"], r["mt"]) for r in rows]); lp = np.array([r["lp"] for r in rows])
    g = {}
    for i, r in enumerate(rows): g.setdefault((r["lp"], r["src"]), []).append(i)
    groups = [np.array(v) for v in g.values() if len(v) >= 3]
    print("rows", len(rows), "groups with >= 3 systems", len(groups))
    def dm(x):
        y = np.array(x, dtype=float).copy()
        for v in groups: y[v] = x[v] - x[v].mean()
        return y
    inG = np.concatenate(groups)
    Hd, Ad, Sd, Ed, Fd = dm(H), dm(zA), dm(zS), dm(E), np.stack([dm(F[:, j]) for j in range(F.shape[1])], 1)
    def within_sp(score, idx_groups):
        v = [spearman(score[x], H[x]) for x in idx_groups]; v = [t for t in v if not math.isnan(t)]
        return float(np.mean(v)), len(v)
    print("\nwithin-source Spearman with human z (mean over sources):")
    res = {}
    for name, sc in (("AfriCOMET-QE", zA), ("SSA-COMET-MTL", zS), ("mean of both", E)):
        res[name] = within_sp(sc, groups); print(f"  {name:15s} {res[name][0]:.3f} over {res[name][1]} sources")
    # residual explained by features (pooled, within-source)
    X = np.concatenate([Ad[:, None], Fd], 1)[inG]; y = Hd[inG]
    names = ["AfriCOMET (z)", "log length ratio", "repetition", "non-ASCII letter share", "word overlap with source", "very short output"]
    sd = X.std(0) + 1e-9; Xs = X / sd
    beta = np.linalg.lstsq(Xs, y, rcond=None)[0]
    r2_base = 1 - ((y - Xs[:, :1] @ beta[:1] * 0 - np.linalg.lstsq(Xs[:, :1], y, rcond=None)[0][0] * Xs[:, 0]) ** 2).sum() / (y ** 2).sum()
    r2_all = 1 - ((y - Xs @ beta) ** 2).sum() / (y ** 2).sum()
    print(f"\nwithin-source R^2 of human z: AfriCOMET alone {r2_base:.3f}, with 5 features {r2_all:.3f}")
    print("standardised coefficients (human z ~ AfriCOMET + features, within source):")
    for n, b in zip(names, beta): print(f"  {n:26s} {b:+.3f}")
    # leave-one-pair-out ridge correction (features + both metrics, no language identity)
    pairs = sorted(set(lp)); gains = []
    base_all, corr_all = [], []
    for held in pairs:
        tr = np.concatenate([v for v in groups if lp[v[0]] != held]); te_g = [v for v in groups if lp[v[0]] == held]
        if not te_g: continue
        Xt = np.concatenate([Ad[:, None], Sd[:, None], Fd], 1); mu = Xt[tr].std(0) + 1e-9
        lam = 50.0; Xtr = Xt[tr] / mu; w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Hd[tr])
        pred = np.zeros(len(rows)); idx = np.concatenate(te_g); pred[idx] = (Xt[idx] / mu) @ w
        b = within_sp(E, te_g)[0]; c = within_sp(pred, te_g)[0]
        gains.append((held, b, c, c - b)); print(f"  held-out {held:10s} mean-of-both {b:.3f} -> corrected {c:.3f} ({c - b:+.3f})")
    print("mean gain over held-out pairs: %+.3f (min %+.3f, max %+.3f)" % (np.mean([x[3] for x in gains]), min(x[3] for x in gains), max(x[3] for x in gains)))
    json.dump(dict(within_source_spearman={k: v[0] for k, v in res.items()}, r2_alone=r2_base, r2_features=r2_all, coef=dict(zip(names, map(float, beta))),
                   held_out=[dict(pair=h, base=b, corrected=c) for h, b, c, _ in gains]), open(os.path.join(OUT, "bias_study.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
