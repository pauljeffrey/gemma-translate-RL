#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""G1 (box, CPU only): pairwise learned-metric comparison of the final checkpoint (final2, step 401) with RL-final (and SFT-final), the Section 6.9 protocol.

Per-row scores
  final2 : Hub dataset <HF_ORG_LEGACY>/SabiYarn-MT-benchmark-translations (private) eval_benchmarks/*.parquet, written by R20/HF1 (columns afri_ref, ssa_ref, afri_qe, ssa_qe, chrf)
  RL-final, SFT-final : /workspace/jobs/e1/sys_{rl,sft}.jsonl.gz (chrF++ sentence level, flags), AfriCOMET-ref / SSA-COMET-ref from /workspace/jobs/e1/neural/{afri_ref,ssa_ref}.jsonl
                        (E1: all `clean` rows + a hash sample of <= 400 rows per language) and AfriCOMET-ref from /workspace/jobs/t1/comet_extra/afri_<benchmark>.jsonl (T1: all `no_pair` rows of
                        NTREX / MAFAND / SmolSent / GlotStoryBook / SALT in the AfriCOMET languages).  Joined by (benchmark, language, row id); a row enters a metric's comparison only when BOTH
                        systems have that metric.  AfriCOMET-QE is the RL reward model and is NOT used here.
Conventions of pairwise_stats.py (E1): win of B over A when delta > threshold (chrF++ 1.0; COMET family 0.01), loss when delta < -threshold, tie otherwise; `preferred` = wins / (wins + losses)
(micro, sentences pooled over languages); macro = mean over languages with >= MINN sentences, every language weighs equally; paired bootstrap B=1000, the sentences of every language are resampled
independently (same indices for A and B), then pooled / averaged.  Subset no_pair (reference pair not in the SFT text) = headline.
usage: box_g1_pairwise_final2.py PARQUET_DIR OUTDIR
"""
import csv, gzip, json, os, sys, zlib, unicodedata
from collections import defaultdict
import numpy as np
import pandas as pd

PQ, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
B, MINN = 1000, 20
E1, T1 = "/workspace/jobs/e1", "/workspace/jobs/t1/comet_extra"
FILES = {"flores_plus": "flores_plus", "ntrex": "ntrex", "mafand": "noncommercial/mafand", "smol_sent": "smol_sent"}
THR = {"chrf": 1.0, "afri_ref": 0.01, "ssa_ref": 0.01}


def load_sys(name):
    d = {}
    with gzip.open(f"{E1}/sys_{name}.jsonl.gz", "rt", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r["b"] in FILES:
                d[(r["b"], r["iso"], str(r["id"]))] = dict(chrf=r["chrf"], pair=r["pair_seen"], cont=r["contaminated"], fin=bool(r["fin"]))
    return d


def add_neural(S, name):
    for m in ("afri_ref", "ssa_ref"):
        for l in open(f"{E1}/neural/{m}.jsonl", encoding="utf-8"):
            j = json.loads(l)
            b, iso = j["chunk"].split("|")
            for i, sc in j["scores"].items():
                if name in sc and (b, iso, str(i)) in S:
                    S[(b, iso, str(i))][m] = sc[name]
    for b in FILES:
        p = f"{T1}/afri_{b}.jsonl"
        if not os.path.exists(p):
            continue
        for l in open(p, encoding="utf-8"):
            j = json.loads(l)
            if j["model"] == name:
                k = (j["b"], j["iso"], str(j["id"]))
                if k in S and "afri_ref" not in S[k]:
                    S[k]["afri_ref"] = j["score"]


def load_final2():
    d = {}
    for b, f in FILES.items():
        df = pd.read_parquet(f"{PQ}/{f}.parquet")
        for r in df.itertuples():
            d[(b, r.language, str(r.row_id))] = dict(chrf=r.chrf, afri_ref=None if pd.isna(r.afri_ref) else float(r.afri_ref), ssa_ref=None if pd.isna(r.ssa_ref) else float(r.ssa_ref),
                                                    pair=bool(r.pair_seen_in_training), cont=bool(r.en_seen_in_training), fin=bool(r.finished))
    return d


def compare(A, F, bench_list, label):
    """A = baseline system dict, F = final2 dict.  Returns per-language, macro and pooled rows."""
    keys = sorted(k for k in F if k in A and A[k]["pair"] is False and F[k]["pair"] is False)
    res_lang, res = [], []
    for scope in bench_list:
        bs = scope if isinstance(scope, tuple) else (scope,)
        name = "+".join(bs)
        for metric in ("chrf", "chrf_sub", "afri_ref", "ssa_ref"):
            col = "chrf" if metric.startswith("chrf") else metric
            thr = THR[col]
            by = defaultdict(list)
            for k in keys:
                if k[0] not in bs:
                    continue
                a, f = A[k].get(col), F[k].get(col)
                if a is None or f is None or a != a or f != f:
                    continue
                if metric == "chrf_sub" and (A[k].get("afri_ref") is None or F[k].get("afri_ref") is None):
                    continue  # chrF++ restricted to the sentences AfriCOMET-ref scored (same rows as the learned-metric comparison)
                by[(k[0], k[1])].append((a, f))
            # languages: pool benchmarks of this scope per language
            lang = defaultdict(list)
            for (b, iso), v in by.items():
                lang[iso].extend(v)
            lang = {i: np.array(v, float) for i, v in lang.items() if len(v) >= MINN}
            if not lang:
                continue
            rng = np.random.default_rng(zlib.crc32(f"{label}|{name}|{metric}".encode()))
            M, ws, ls, ns = [], [], [], []
            boots = []
            for iso, v in sorted(lang.items()):
                n = len(v); d = v[:, 1] - v[:, 0]
                ix = rng.integers(0, n, size=(B, n))
                dm = d[ix].mean(1)
                w = (d > thr); l = (d < -thr)
                bw, bl = w[ix].sum(1), l[ix].sum(1)
                boots.append(dict(dm=dm, am=v[:, 0][ix].mean(1), fm=v[:, 1][ix].mean(1), bw=bw, bl=bl, n=n))
                lo, hi = np.percentile(dm, [2.5, 97.5])
                res_lang.append(dict(baseline=label, scope=name, metric=metric, iso=iso, n=n, mean_base=round(float(v[:, 0].mean()), 5), mean_final2=round(float(v[:, 1].mean()), 5),
                                     delta=round(float(d.mean()), 5), lo=round(float(lo), 5), hi=round(float(hi), 5), win=round(float(w.mean()), 4), tie=round(float(1 - w.mean() - l.mean()), 4),
                                     loss=round(float(l.mean()), 4)))
            L = len(boots)
            nn = np.array([b["n"] for b in boots])
            mdm = np.mean([b["dm"] for b in boots], 0)
            mam = np.mean([b["am"] for b in boots], 0); mfm = np.mean([b["fm"] for b in boots], 0)
            mw = np.mean([b["bw"] / b["n"] for b in boots], 0); ml = np.mean([b["bl"] / b["n"] for b in boots], 0)
            BW = np.sum([b["bw"] for b in boots], 0); BL = np.sum([b["bl"] for b in boots], 0)
            pref = BW / np.maximum(1, BW + BL)
            rl = [r for r in res_lang if r["baseline"] == label and r["scope"] == name and r["metric"] == metric]
            pt = lambda f: float(np.mean([f(r) for r in rl]))
            allw = sum(r["win"] * r["n"] for r in rl); alll = sum(r["loss"] * r["n"] for r in rl); N = sum(r["n"] for r in rl)
            ci = lambda x: tuple(round(float(y), 5) for y in np.percentile(x, [2.5, 97.5]))
            res.append(dict(baseline=label, scope=name, metric=metric, langs=L, n=int(N), mean_base=round(pt(lambda r: r["mean_base"]), 5), mean_final2=round(pt(lambda r: r["mean_final2"]), 5),
                            delta=round(pt(lambda r: r["delta"]), 5), delta_lo=ci(mdm)[0], delta_hi=ci(mdm)[1],
                            macro_win=round(pt(lambda r: r["win"]), 4), macro_tie=round(pt(lambda r: r["tie"]), 4), macro_loss=round(pt(lambda r: r["loss"]), 4),
                            micro_win=round(allw / N, 4), micro_tie=round(1 - allw / N - alll / N, 4), micro_loss=round(alll / N, 4),
                            pref=round(allw / max(1, allw + alll), 4), pref_lo=ci(pref)[0], pref_hi=ci(pref)[1],
                            langs_up=sum(r["delta"] > 0 for r in rl), langs_clearly_better=sum(r["lo"] > 0 for r in rl), langs_clearly_worse=sum(r["hi"] < 0 for r in rl)))
    return res_lang, res


def main():
    F = load_final2(); print("final2", len(F), flush=True)
    out_l, out_m = [], []
    for name in ("rl", "sft"):
        A = load_sys(name); add_neural(A, name)
        for m in ("afri_ref", "ssa_ref"):
            print(name, m, sum(1 for v in A.values() if m in v), flush=True)
        for v in A.values():
            for m in ("afri_ref", "ssa_ref"):
                v.setdefault(m, None)
        scopes = [("ntrex",), ("mafand",), ("smol_sent",), ("flores_plus",), ("ntrex", "mafand", "smol_sent")]
        rl_, r_ = compare(A, F, scopes, name)
        out_l += rl_; out_m += r_
    for fn, rows in (("g1_pairwise_final2_per_language.csv", out_l), ("g1_pairwise_final2_summary.csv", out_m)):
        with open(f"{OUT}/{fn}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("done", len(out_l), len(out_m))


if __name__ == "__main__":
    main()
