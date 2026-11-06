#!/usr/bin/env python
"""Paired bootstrap comparison of systems from per-sentence npz files (ev_stats.py layout; e02_mknpz.py writes the same layout for the decoding study). CPU only.

    python ecompare.py --sys NAME=file.npz ... [--pair A:B ...] [--langs yor,hau,...] [--contam DIR] [--out PREFIX] [--boot 1000] [--seed 20261104] [--per-language]

Scoring: sacrebleu corpus chrF++ (word_order 2) and chrF (word_order 0) from summed n-gram statistics of a cell (benchmark x language); spBLEU (flores200 tokenizer) when `bstats` is present;
macro = mean over cells. AfriCOMET = masakhane/africomet-stl-1.1 (reference-based) mean over the cached sentence scores of the cell (cells without scores are left out of that macro). finished = share of outputs with an
end token before the cap; loop = the reward family's repetition flag (guards.is_repetitive / is_self_repetitive); iloop = the independent detector of E-08 (when present).
Bootstrap: Poisson(1) weights per source sentence, shared by all systems (paired) and, for the parallel benchmarks (flores_plus, ntrex), by all languages; mafand: independent per row. Percentile 95% intervals of
the macro and of every difference. Rows flagged in the contamination files (exact / sentence / near hit) are removed everywhere when --contam is given.
"""
import argparse, json, os
import numpy as np

PAR = {"flores_plus", "ntrex"}


def fscore(S, n_orders):
    h = S[..., 0:3 * n_orders:3].astype(np.float64); r = S[..., 1:3 * n_orders:3].astype(np.float64); m = S[..., 2:3 * n_orders:3].astype(np.float64)
    ok = (h > 0) & (r > 0)
    prec = np.where(ok, m / np.where(h > 0, h, 1), 0.0); rec = np.where(ok, m / np.where(r > 0, r, 1), 0.0)
    eff = ok.sum(-1)
    ap = np.where(eff > 0, prec.sum(-1) / np.maximum(eff, 1), 0.0); ar = np.where(eff > 0, rec.sum(-1) / np.maximum(eff, 1), 0.0)
    den = 4 * ap + ar
    return np.where(den > 0, 100 * 5 * ap * ar / np.where(den > 0, den, 1), 0.0)


def bleu_from_stats(S):
    """S[..., 10] = [hyp_len, ref_len, correct1..4, total1..4] -> corpus BLEU (sacrebleu: exp smoothing only when a count is 0, brevity penalty)."""
    S = S.astype(np.float64); hl, rl = S[..., 0], S[..., 1]; c = S[..., 2:6]; t = S[..., 6:10]
    p = np.where(t > 0, c / np.where(t > 0, t, 1), 0.0)
    # sacrebleu 'exp' smoothing: a zero n-gram precision becomes 1 / (2^k * total) with k counting the zeros
    out = np.zeros(S.shape[:-1])
    flat_c = c.reshape(-1, 4); flat_t = t.reshape(-1, 4); flat_p = p.reshape(-1, 4); flat_hl = hl.reshape(-1); flat_rl = rl.reshape(-1)
    res = np.zeros(flat_c.shape[0])
    for i in range(flat_c.shape[0]):
        if flat_c[i, 0] == 0 or flat_hl[i] == 0:
            continue
        smooth = 1.0; ps = []
        for n in range(4):
            if flat_t[i, n] == 0: ps.append(0.0); continue
            if flat_c[i, n] == 0:
                smooth *= 2; ps.append(100.0 / (smooth * flat_t[i, n]))
            else:
                ps.append(100.0 * flat_c[i, n] / flat_t[i, n])
        if min(ps) <= 0: continue
        bp = 1.0 if flat_hl[i] > flat_rl[i] else np.exp(1 - flat_rl[i] / flat_hl[i])
        res[i] = bp * np.exp(np.mean(np.log(ps)))
    return res.reshape(S.shape[:-1])


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def contam_mask(contam_dir, bench, iso, rid):
    bad = np.zeros(len(iso), dtype=bool)
    for b in sorted(set(bench)):
        p = os.path.join(contam_dir, f"contamination_{b}.json")
        if not os.path.exists(p): continue
        d = json.load(open(p)); flag = {}
        for part in ("exact", "sentence", "near"):
            for k, v in d.get(part, {}).items():
                if v: flag[k] = True
        for n in np.where(bench == b)[0]:
            if f"{iso[n]}:{rid[n]}" in flag: bad[n] = True
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", nargs="+", required=True); ap.add_argument("--pair", nargs="*", default=[]); ap.add_argument("--langs", default=None)
    ap.add_argument("--contam", default="/workspace/eval_385m/bench"); ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=20261104); ap.add_argument("--per-language", action="store_true")
    ap.add_argument("--benchmarks", default="flores_plus,ntrex,mafand"); ap.add_argument("--comet-langs-from", default=None, help="npz whose finite-comet languages define the AfriCOMET scope")
    a = ap.parse_args()
    names, D = [], {}
    for s in a.sys:
        n, p = s.split("="); names.append(n); d = load(p)
        d["key"] = {(b, i, str(r)): k for k, (b, i, r) in enumerate(zip(d["bench"], d["iso"], d["rid"]))}; D[n] = d
    langs = set(a.langs.split(",")) if a.langs else None; benches = set(a.benchmarks.split(","))
    base = D[names[0]]
    keep = [k for k, key in enumerate(zip(base["bench"], base["iso"], base["rid"].astype(str))) if key[0] in benches and (langs is None or key[1] in langs)
            and all((key[0], key[1], key[2]) in D[n]["key"] for n in names)]
    keep = np.array(keep); bench, iso, rid = base["bench"][keep], base["iso"][keep], base["rid"][keep].astype(str)
    A = {}
    for n in names:
        ix = np.array([D[n]["key"][(b, i, r)] for b, i, r in zip(bench, iso, rid)])
        A[n] = {k: D[n][k][ix] for k in D[n] if k != "key" and D[n][k].shape[:1] == D[n]["bench"].shape}
    bad = contam_mask(a.contam, bench, iso, rid) if a.contam else np.zeros(len(keep), bool)
    print("common rows", len(keep), "contamination-flagged and removed", int(bad.sum()))
    ok = ~bad; bench, iso, rid = bench[ok], iso[ok], rid[ok]
    for n in names: A[n] = {k: v[ok] for k, v in A[n].items()}
    cells = {}
    for b in sorted(set(bench)):
        for l in sorted(set(iso[bench == b])): cells[(b, l)] = np.where((bench == b) & (iso == l))[0]
    rng = np.random.default_rng(a.seed); B = a.boot; W = {}
    for b in sorted(set(bench)):
        if b in PAR:
            ids = sorted(set(rid[bench == b])); ix = {r: k for k, r in enumerate(ids)}; w = rng.poisson(1.0, size=(B, len(ids))).astype(np.float32)
            for (bb, l), rows in cells.items():
                if bb == b: W[(bb, l)] = w[:, [ix[r] for r in rid[rows]]]
        else:
            for (bb, l), rows in cells.items():
                if bb == b: W[(bb, l)] = rng.poisson(1.0, size=(B, len(rows))).astype(np.float32)
    comet_langs = None
    if a.comet_langs_from:
        z = load(a.comet_langs_from); comet_langs = set(i for i in set(z["iso"]) if np.isfinite(z["comet"][z["iso"] == i]).any())
    METRICS = ["chrfpp", "chrf", "comet", "finished", "loop", "iloop", "spbleu", "len_ratio"]
    # per system, per cell: point and bootstrap arrays [metric] -> (pt, boot[B])
    R = {n: {} for n in names}
    for n in names:
        for cell, rows in cells.items():
            w = W[cell]; st = A[n]["stats"][rows].astype(np.float64); r = {}
            tot = st.sum(0); bt = w @ st
            r["chrfpp"] = (fscore(tot[None], 8)[0], fscore(bt, 8)); r["chrf"] = (fscore(tot[None, :18], 6)[0], fscore(bt[:, :18], 6))
            c = A[n]["comet"][rows]; m = np.isfinite(c)
            if m.any() and (comet_langs is None or cell[1] in comet_langs):
                r["comet"] = (c[m].mean(), (w[:, m] @ c[m]) / np.maximum(w[:, m].sum(1), 1e-9))
            for key, arr in (("finished", A[n]["finished"][rows].astype(float)), ("loop", A[n]["loop"][rows].astype(float))) + ((("iloop", A[n]["iloop"][rows].astype(float)),) if "iloop" in A[n] else ()):
                r[key] = (arr.mean(), (w @ arr) / np.maximum(w.sum(1), 1e-9))
            hc = A[n]["hyp_chars"][rows].astype(float); rc = np.maximum(A[n]["ref_chars"][rows].astype(float), 1)
            r["len_ratio"] = (hc.sum() / rc.sum(), (w @ hc) / np.maximum(w @ rc, 1e-9))
            if "bstats" in A[n]:
                bs = A[n]["bstats"][rows].astype(np.float64); r["spbleu"] = (bleu_from_stats(bs.sum(0)[None])[0], bleu_from_stats(w @ bs))
            R[n][cell] = r

    def macro(n, metric, cs):
        cs = [c for c in cs if metric in R[n][c]]
        if not cs: return None
        return (np.mean([R[n][c][metric][0] for c in cs]), np.mean([R[n][c][metric][1] for c in cs], axis=0))
    scopes = {"all": list(cells)}
    for b in sorted(set(bench)): scopes[b] = [c for c in cells if c[0] == b]
    ci = lambda x: np.percentile(x, [2.5, 97.5])
    out = {"n_rows": int(len(bench)), "boot": B, "seed": a.seed, "systems": {}, "pairs": {}}
    lines = ["# systems (macro over cells; 95% bootstrap interval)\n", "| system | scope | n cells | chrF++ | chrF | spBLEU | AfriCOMET | finished | loop (reward flag) | loop (independent) | length ratio |", "|---|---|---:|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, d=2: "-" if v is None else f"{v[0]:.{d}f} [{ci(v[1])[0]:.{d}f}, {ci(v[1])[1]:.{d}f}]"
    for n in names:
        out["systems"][n] = {}
        for sc, cs in scopes.items():
            vals = {m: macro(n, m, cs) for m in METRICS}
            out["systems"][n][sc] = {m: (None if v is None else {"point": float(v[0]), "lo": float(ci(v[1])[0]), "hi": float(ci(v[1])[1])}) for m, v in vals.items()}
            lines.append(f"| {n} | {sc} | {len(cs)} | {fmt(vals['chrfpp'])} | {fmt(vals['chrf'])} | {fmt(vals['spbleu'])} | {fmt(vals['comet'],4)} | {fmt(vals['finished'],4)} | {fmt(vals['loop'],4)} | {fmt(vals['iloop'],4)} | {fmt(vals['len_ratio'],3)} |")
    lines += ["\n# paired differences (first minus second; macro over cells; 95% interval)\n", "| pair | scope | chrF++ | chrF | spBLEU | AfriCOMET | finished | loop (reward flag) | loop (independent) |", "|---|---|---|---|---|---|---|---|---|"]
    perlang = []
    for pr in a.pair:
        if pr.count(":") == 3:      # difference of differences (A - B) - (C - D): the interaction of two factors
            a_, b_, c_, d_ = pr.split(":"); out["pairs"][pr] = {}
            for sc, cs in scopes.items():
                row = {}
                for m in METRICS:
                    v = [macro(n_, m, cs) for n_ in (a_, b_, c_, d_)]
                    if any(t is None for t in v): row[m] = None; continue
                    d = (v[0][1] - v[1][1]) - (v[2][1] - v[3][1]); row[m] = {"point": float((v[0][0] - v[1][0]) - (v[2][0] - v[3][0])), "lo": float(ci(d)[0]), "hi": float(ci(d)[1])}
                out["pairs"][pr][sc] = row
                f = lambda m, dd=2: "-" if row[m] is None else f"{row[m]['point']:+.{dd}f} [{row[m]['lo']:+.{dd}f}, {row[m]['hi']:+.{dd}f}]"
                lines.append(f"| ({a_} - {b_}) - ({c_} - {d_}) | {sc} | {f('chrfpp')} | {f('chrf')} | {f('spbleu')} | {f('comet',4)} | {f('finished',4)} | {f('loop',4)} | {f('iloop',4)} |")
            continue
        x, y = pr.split(":"); out["pairs"][pr] = {}
        for sc, cs in scopes.items():
            row = {}
            for m in METRICS:
                vx, vy = macro(x, m, cs), macro(y, m, cs)
                if vx is None or vy is None: row[m] = None; continue
                d = vx[1] - vy[1]; row[m] = {"point": float(vx[0] - vy[0]), "lo": float(ci(d)[0]), "hi": float(ci(d)[1])}
            out["pairs"][pr][sc] = row
            f = lambda m, dd=2: "-" if row[m] is None else f"{row[m]['point']:+.{dd}f} [{row[m]['lo']:+.{dd}f}, {row[m]['hi']:+.{dd}f}]"
            lines.append(f"| {pr} | {sc} | {f('chrfpp')} | {f('chrf')} | {f('spbleu')} | {f('comet',4)} | {f('finished',4)} | {f('loop',4)} | {f('iloop',4)} |")
        if a.per_language:
            for cell in cells:
                rx, ry = R[x][cell], R[y][cell]
                d = rx["chrfpp"][1] - ry["chrfpp"][1]; lo, hi = ci(d)
                dc = (rx["comet"][1] - ry["comet"][1]) if "comet" in rx and "comet" in ry else None
                perlang.append([pr, cell[0], cell[1], len(cells[cell]), rx["chrfpp"][0], ry["chrfpp"][0], rx["chrfpp"][0] - ry["chrfpp"][0], lo, hi,
                                "" if dc is None else rx["comet"][0] - ry["comet"][0], "" if dc is None else ci(dc)[0], "" if dc is None else ci(dc)[1]])
    open(a.out + ".md", "w").write("\n".join(lines) + "\n"); json.dump(out, open(a.out + ".json", "w"), indent=1)
    if perlang:
        import csv
        with open(a.out + "_per_language.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pair", "bench", "iso", "n", "chrfpp_first", "chrfpp_second", "diff", "lo", "hi", "comet_diff", "comet_lo", "comet_hi"]); w.writerows(perlang)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
