#!/usr/bin/env python3
"""AfriCOMET (reference, STL-1.1) and AfriCOMET-QE per language and macro for two systems from the on-disk score caches of the evaluation run, with a paired bootstrap of the
macro difference. CPU only.

    python scripts/comet_rows.py --system m385=OUT_m385 --system a0=OUT_a0 --pair m385:a0 --out DIR [--subsets all,clean,no_pair] [--boot 1000]
OUT_x = the evaluation directory (needs predictions*.jsonl [--pred-name] and work/comet_cache/comet_{ref,qe}.jsonl). Key of a cache entry = sha1(english NUL hyp NUL reference) (QE: reference empty).
Rows whose key is not in the cache (language not covered, or a text that was changed) get no score and are reported as `n_scored`.
"""
import argparse, csv, hashlib, json, os, sys
from collections import defaultdict
import numpy as np

SUBSETS = {"all": lambda r: True, "clean": lambda r: r.get("contaminated") is False, "no_pair": lambda r: r.get("pair_seen") is False}
key = lambda s, m, r="": hashlib.sha1(f"{s}\x00{m}\x00{r}".encode("utf-8")).hexdigest()


def load_cache(p):
    d = {}
    for l in open(p, encoding="utf-8"):
        try:
            k, v = json.loads(l); d[k] = float(v)
        except Exception:
            pass
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--system", action="append", required=True); ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--out", required=True); ap.add_argument("--subsets", default="all"); ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--pred-name", default="predictions.jsonl")
    ap.add_argument("--min-n", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); subs = a.subsets.split(",")
    S = {}
    for s in a.system:
        name, d = s.split("=", 1)
        ref, qe = load_cache(os.path.join(d, "work/comet_cache/comet_ref.jsonl")), load_cache(os.path.join(d, "work/comet_cache/comet_qe.jsonl"))
        rows = {}
        miss = 0
        for l in open(os.path.join(d, a.pred_name), encoding="utf-8"):
            r = json.loads(l)
            kr = ref.get(key(r["english"], r["hyp"], r["reference"])); kq = qe.get(key(r["english"], r["hyp"]))
            r["_c"], r["_q"] = kr, kq
            rows[(r["benchmark"], r["iso"], str(r["id"]))] = r
        S[name] = rows; print(name, len(rows), "rows; scored", sum(r["_c"] is not None for r in rows.values()), flush=True)
    # per language
    with open(os.path.join(a.out, "comet_by_language.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["system", "benchmark", "subset", "iso", "n", "n_scored", "comet", "comet_qe"])
        for name, rows in S.items():
            cell = defaultdict(list)
            for (b, iso, i), r in rows.items():
                for sub in subs:
                    if SUBSETS[sub](r): cell[(b, sub, iso)].append(r)
            for (b, sub, iso), rs in sorted(cell.items()):
                c = [r["_c"] for r in rs if r["_c"] is not None]; q = [r["_q"] for r in rs if r["_q"] is not None]
                if c: w.writerow([name, b, sub, iso, len(rs), len(c), float(np.mean(c)), float(np.mean(q)) if q else ""])
    # macro per system (own language set with >= min_n scored) and paired
    names = list(S)
    out_m = open(os.path.join(a.out, "comet_macro.csv"), "w", newline=""); wm = csv.writer(out_m)
    wm.writerow(["pair_or_system", "benchmark", "subset", "n_languages", "n_sentences", "comet_b", "comet_a", "d_comet", "d_lo", "d_hi", "qe_b", "qe_a", "d_qe", "dq_lo", "dq_hi"])
    wl = open(os.path.join(a.out, "comet_pair_by_language.csv"), "w", newline=""); wl2 = csv.writer(wl)
    wl2.writerow(["pair", "benchmark", "subset", "iso", "n", "comet_b", "comet_a", "d_comet", "d_lo", "d_hi"])
    rng = np.random.default_rng(a.seed)
    for pr in a.pair:
        b_, a_ = pr.split(":")
        benchmarks = sorted({k[0] for k in S[b_]})
        for bench in benchmarks:
            for sub in subs:
                per = defaultdict(list)
                for (b, iso, i), rb in S[b_].items():
                    if b != bench or not SUBSETS[sub](rb): continue
                    ra = S[a_].get((b, iso, i))
                    if ra is None or rb["_c"] is None or ra["_c"] is None: continue
                    per[iso].append((rb["_c"], ra["_c"], rb["_q"], ra["_q"]))
                per = {k: np.array([[x if x is not None else np.nan for x in t] for t in v]) for k, v in per.items() if len(v) >= a.min_n}
                if not per: continue
                B = a.boot; dc = np.zeros(B); dq = np.zeros(B); cb = []; ca = []; qb = []; qa = []; tot = 0
                for iso, arr in sorted(per.items()):
                    n = len(arr); cnt = rng.multinomial(n, np.full(n, 1.0 / n), size=B) / n
                    d = arr[:, 0] - arr[:, 1]; bs = cnt @ d
                    dc += bs; cb.append(arr[:, 0].mean()); ca.append(arr[:, 1].mean()); tot += n
                    lo, hi = np.percentile(bs, [2.5, 97.5]); wl2.writerow([pr, bench, sub, iso, n, cb[-1], ca[-1], cb[-1] - ca[-1], lo, hi])
                    ok = ~np.isnan(arr[:, 2]) & ~np.isnan(arr[:, 3])
                    if ok.all():
                        dqq = arr[:, 2] - arr[:, 3]; dq += cnt @ dqq; qb.append(arr[:, 2].mean()); qa.append(arr[:, 3].mean())
                L = len(per); dc /= L; lo, hi = np.percentile(dc, [2.5, 97.5])
                if len(qb) == L:
                    dq /= L; qlo, qhi = np.percentile(dq, [2.5, 97.5]); qbm, qam = float(np.mean(qb)), float(np.mean(qa))
                else: qlo = qhi = qbm = qam = ""
                wm.writerow([pr, bench, sub, L, tot, float(np.mean(cb)), float(np.mean(ca)), float(np.mean(cb) - np.mean(ca)), lo, hi, qbm, qam, (qbm - qam) if qbm != "" else "", qlo, qhi])
    out_m.close(); wl.close(); print("done")


if __name__ == "__main__":
    main()
