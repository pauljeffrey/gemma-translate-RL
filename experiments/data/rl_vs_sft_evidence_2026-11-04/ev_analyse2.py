#!/usr/bin/env python
"""Paired bootstrap of RL against SFT from the per-sentence statistics written by ev_stats.py (CPU only).

    python ev_analyse.py EVID_DIR CONTAM_DIR GROUPS_DIR OUT_DIR [--boot 1000] [--seed 20261104]

Scoring is sacrebleu's: corpus chrF++ (word_order 2) and plain chrF (word_order 0, the first 6 orders) from the summed statistics of a cell
(benchmark x language); macro = mean over cells. AfriCOMET = masakhane/africomet-stl-1.1 (reference-based) mean per cell over the stored
sentence scores (AfriCOMET-supported languages only).
Bootstrap: Poisson(1) weights per source sentence, SHARED by the systems being compared (paired) and, for the parallel benchmarks (FLORES+, NTREX),
by all languages (the same English sentence has the same weight in every language, so macro intervals carry the cross-language correlation).
MAFAND is not parallel across languages: independent weights per (language, row).
Rows with a contamination flag (exact / sentence / near hit of the benchmark English in the 385M SFT text) are removed everywhere.
"""
import argparse
import json
import os

import numpy as np

PAR = {"flores_plus", "ntrex"}
BENCH = ["flores_plus", "ntrex", "mafand"]
SYS = ["sft", "stageA", "b1", "mixed"]          # overridden by --systems
LABEL = {"sft": "SFT", "stageA": "RL stage A", "b1": "RL stage B1 best", "mixed": "RL mixed best (final RL model)"}


def fscore(S, n_orders):
    """sacrebleu CHRF._compute_f_score, vectorised over leading axes. S[..., 3*n_orders]: [hyp, ref, match] per order; beta 2."""
    h = S[..., 0:3 * n_orders:3].astype(np.float64)
    r = S[..., 1:3 * n_orders:3].astype(np.float64)
    m = S[..., 2:3 * n_orders:3].astype(np.float64)
    ok = (h > 0) & (r > 0)
    prec = np.where(ok, m / np.where(h > 0, h, 1), 0.0)
    rec = np.where(ok, m / np.where(r > 0, r, 1), 0.0)
    eff = ok.sum(-1)
    ap = np.where(eff > 0, prec.sum(-1) / np.maximum(eff, 1), 0.0)
    ar = np.where(eff > 0, rec.sum(-1) / np.maximum(eff, 1), 0.0)
    den = 4 * ap + ar
    return np.where(den > 0, 100 * 5 * ap * ar / np.where(den > 0, den, 1), 0.0)


def load(evid):
    D = {}
    for s in SYS:
        z = np.load(os.path.join(evid, s + ".npz"))
        D[s] = {k: z[k] for k in z.files}
        D[s]["key"] = {(b, i, r): n for n, (b, i, r) in enumerate(zip(D[s]["bench"], D[s]["iso"], D[s]["rid"]))}
    return D


def contam_mask(contam_dir, bench, iso, rid):
    bad = np.zeros(len(iso), dtype=bool)
    for b in BENCH:
        d = json.load(open(os.path.join(contam_dir, f"contamination_{b}.json")))
        flag = {}
        for part in ("exact", "sentence", "near"):
            for k, v in d[part].items():
                if v:
                    flag[k] = True
        sel = np.where(bench == b)[0]
        for n in sel:
            if f"{iso[n]}:{rid[n]}" in flag:
                bad[n] = True
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evid"); ap.add_argument("contam"); ap.add_argument("groups"); ap.add_argument("out")
    ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--systems", default="sft,stageA,b1,mixed"); ap.add_argument("--pairs", default="sft:mixed", help="first pair = primary (variants, groups); the others get the all-rows macro only")
    ap.add_argument("--benchmarks", default="flores_plus,ntrex,mafand"); ap.add_argument("--languages", default="", help="restrict to these isos (comma list)"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global SYS, BENCH
    SYS = a.systems.split(","); BENCH = a.benchmarks.split(","); PAIRS = [tuple(p.split(":")) for p in a.pairs.split(",")]
    LANGS = set(a.languages.split(",")) if a.languages else None
    os.makedirs(a.out, exist_ok=True)
    D = load(a.evid)
    out_pairs = a.pairs
    base = D[SYS[0]]
    # common rows: benchmark in BENCH and present in all four systems
    keep = [n for n, k in enumerate(zip(base["bench"], base["iso"], base["rid"])) if k[0] in BENCH and (LANGS is None or k[1] in LANGS) and all(k in D[s]["key"] for s in SYS)]
    keep = np.array(keep)
    bench, iso, rid = base["bench"][keep], base["iso"][keep], base["rid"][keep]
    A = {}
    for s in SYS:
        ix = np.array([D[s]["key"][(b, i, r)] for b, i, r in zip(bench, iso, rid)])
        A[s] = {k: D[s][k][ix] for k in ("stats", "tstats", "finished", "loop", "comet", "hyp_chars", "ref_chars")}
    bad = contam_mask(a.contam, bench, iso, rid)
    print("rows", len(keep), "contamination-flagged and removed:", int(bad.sum()), {b: int(bad[bench == b].sum()) for b in BENCH})
    groups = {g: json.load(open(os.path.join(a.groups, f"groups_{g}.json"))) for g in ("family", "weak", "a0new", "target8")}
    rng = np.random.default_rng(a.seed)
    B = a.boot
    # Poisson weights per (benchmark-unit, row-id)
    cells = {}
    for b in BENCH:
        for l in sorted(set(iso[bench == b])):
            cells[(b, l)] = np.where((bench == b) & (iso == l))[0]
    W = {}
    for b in BENCH:
        if b in PAR:
            ids = sorted(set(rid[bench == b]))
            ix = {r: n for n, r in enumerate(ids)}
            w = rng.poisson(1.0, size=(B, len(ids))).astype(np.float32)
            for (bb, l), rows in cells.items():
                if bb == b:
                    W[(bb, l)] = w[:, [ix[r] for r in rid[rows]]]
        else:
            for (bb, l), rows in cells.items():
                if bb == b:
                    W[(bb, l)] = rng.poisson(1.0, size=(B, len(rows))).astype(np.float32)

    def cell_vals(sysname, rows, use_t, mask, w):
        """point + bootstrap chrF++, chrF, comet-mean, for one system on selected rows (mask indexes into rows)."""
        st = A[sysname]["tstats" if use_t else "stats"][rows][mask].astype(np.float64)
        ww = w[:, mask]
        out = {}
        tot = st.sum(0)
        bt = ww @ st
        out["pp"] = (fscore(tot[None], 8)[0], fscore(bt, 8))
        out["c"] = (fscore(tot[None, :18], 6)[0], fscore(bt[:, :18], 6))
        return out

    def run(pair, variant):
        """-> dict cell -> {metric: (A_point, B_point, diff_point, diff_boot[B])} ; metrics pp, c, com"""
        sa, sb = pair
        res = {}
        for (b, l), rows in cells.items():
            ok = ~bad[rows]
            fa, la = A[sa]["finished"][rows], A[sa]["loop"][rows]
            fb, lb = A[sb]["finished"][rows], A[sb]["loop"][rows]
            ta = tb = False
            if variant == "all":
                m = ok
            elif variant == "paired_good":
                m = ok & fa & ~la & fb & ~lb
            elif variant == "sft_bad_rows":
                m = ok & (~fa | la)
            elif variant == "sft_ok_rows":
                m = ok & fa & ~la
            elif variant == "sft_trunc":
                m = ok; ta = True
            elif variant == "both_trunc":
                m = ok; ta = tb = True
            else:
                raise ValueError(variant)
            if m.sum() < 20:
                continue
            w = W[(b, l)]
            va = cell_vals(sa, rows, ta, m, w); vb = cell_vals(sb, rows, tb, m, w)
            r = {}
            for k in ("pp", "c"):
                r[k] = (va[k][0], vb[k][0], vb[k][0] - va[k][0], vb[k][1] - va[k][1])
            if not (ta or tb):
                ca = A[sa]["comet"][rows][m]; cb = A[sb]["comet"][rows][m]
                both = np.isfinite(ca) & np.isfinite(cb)
                if both.sum() >= 20:
                    ww = w[:, m][:, both]
                    den = ww.sum(1)
                    ba = (ww @ ca[both]) / den; bb_ = (ww @ cb[both]) / den
                    r["com"] = (ca[both].mean(), cb[both].mean(), cb[both].mean() - ca[both].mean(), bb_ - ba)
                    r["n_com"] = int(both.sum())
            r["n"] = int(m.sum())
            res[(b, l)] = r
        return res

    def agg(res, cell_filter):
        """macro over cells passing cell_filter -> {metric: (A, B, diff, lo, hi, n_cells)}"""
        out = {}
        for k in ("pp", "c", "com"):
            sel = [v[k] for c, v in res.items() if cell_filter(c) and k in v]
            if not sel:
                continue
            pa = np.mean([s[0] for s in sel]); pb = np.mean([s[1] for s in sel]); d = pb - pa
            bd = np.mean([s[3] for s in sel], axis=0)
            lo, hi = np.percentile(bd, [2.5, 97.5])
            out[k] = (float(pa), float(pb), float(d), float(lo), float(hi), len(sel))
        return out

    # ----- language groups
    fl = {l: None for (b, l) in cells if b == "flores_plus"}
    sft_fl = {}
    for (b, l), rows in cells.items():
        if b == "flores_plus":
            m = ~bad[rows]
            sft_fl[l] = fscore(A["sft"]["stats"][rows][m].sum(0)[None].astype(np.float64), 8)[0]
    order = sorted(sft_fl, key=lambda x: sft_fl[x])
    terc = {l: ("T1 weakest third of SFT chrF++" if i < len(order) / 3 else "T2 middle third" if i < 2 * len(order) / 3 else "T3 strongest third") for i, l in enumerate(order)}
    G = {
        "family": lambda l: groups["family"].get(l, "?"),
        "weak": lambda l: groups["weak"].get(l, "?"),
        "a0new": lambda l: groups["a0new"].get(l, "?"),
        "sft_tercile": lambda l: terc.get(l, "not on FLORES+"),
    }
    out = {"meta": {"boot": B, "seed": a.seed, "removed_contaminated": int(bad.sum()), "rows": int(len(keep))}, "sft_flores_chrfpp": sft_fl}

    # ----- flags per system (clean rows, macro over FLORES+/NTREX/MAFAND cells)
    flags = {}
    for s in SYS:
        r = {}
        for b in BENCH:
            rows = np.where((bench == b) & ~bad)[0]
            fin = A[s]["finished"][rows]; lp = A[s]["loop"][rows]
            per = {}
            for l in sorted(set(iso[rows])):
                rr = np.where(iso[rows] == l)[0]
                per[l] = (fin[rr].mean(), lp[rr].mean())
            r[b] = {"finished": float(np.mean([v[0] for v in per.values()])), "loop": float(np.mean([v[1] for v in per.values()])), "finished_pooled": float(fin.mean()), "loop_pooled": float(lp.mean()), "n": int(len(rows))}
            # length ratio hyp/ref chars (pooled per language then mean)
            lr = []
            for l in sorted(set(iso[rows])):
                rr = rows[iso[rows] == l]
                lr.append(A[s]["hyp_chars"][rr].sum() / max(1, A[s]["ref_chars"][rr].sum()))
            r[b]["len_ratio"] = float(np.mean(lr))
        flags[s] = r
    out["flags"] = flags

    variants = ["all", "paired_good", "sft_trunc", "both_trunc", "sft_bad_rows", "sft_ok_rows"]
    out["macro"] = {}
    percell = {}
    for v in variants:
        res = run(PAIRS[0], v)
        percell[v] = res
        out["macro"][v] = {b: agg(res, lambda c, b=b: c[0] == b) for b in BENCH}
        out["macro"][v]["all_cells"] = agg(res, lambda c: True)
    # stage-wise decomposition (all rows)
    out["stages"] = {}
    for pair in PAIRS[1:]:
        res = run(pair, "all")
        out["stages"]["%s->%s" % pair] = {b: agg(res, lambda c, b=b: c[0] == b) for b in BENCH}
    # groups (variants all, paired_good, sft_trunc)
    out["groups"] = {}
    for v in ["all", "paired_good", "sft_trunc"]:
        res = percell[v]
        gg = {}
        for gname, f in G.items():
            labels = sorted(set(f(l) for (b, l) in res))
            gg[gname] = {}
            for lab in labels:
                gg[gname][lab] = {b: agg(res, lambda c, lab=lab, b=b: c[0] == b and f(c[1]) == lab) for b in BENCH}
                gg[gname][lab]["langs_flores"] = sorted(l for (b, l) in res if b == "flores_plus" and f(l) == lab)
        out["groups"][v] = gg
    # per cell table (all rows) and counts of languages where RL is worse
    cells_out = {}
    for v in ["all", "paired_good", "sft_trunc"]:
        t = {}
        for (b, l), r in percell[v].items():
            e = {"n": r["n"]}
            for k in ("pp", "c", "com"):
                if k in r:
                    lo, hi = np.percentile(r[k][3], [2.5, 97.5])
                    e[k] = {"sft": float(r[k][0]), "rl": float(r[k][1]), "diff": float(r[k][2]), "lo": float(lo), "hi": float(hi)}
            if "n_com" in r:
                e["n_com"] = r["n_com"]
            t[f"{b}:{l}"] = e
        cells_out[v] = t
    out["cells"] = cells_out
    json.dump(out, open(os.path.join(a.out, "evidence%s.json" % a.tag), "w"), indent=1)
    print("done", a.out)


if __name__ == "__main__":
    main()
