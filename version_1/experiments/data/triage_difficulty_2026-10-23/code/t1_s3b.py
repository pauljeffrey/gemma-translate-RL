"""Bin-vs-reference: language-relative quality of each score bin minus that of the reference bin (score <= 3.0 = owner's 'very easy'), cluster bootstrap over English chunks."""
from t1load import *
A = pd.read_pickle(f"{DATA}/A.pkl"); P = A[(A.n_pair_seen == 0) & ~A.efi].copy()
def dm(d, c): return d[c] - d.groupby(["b", "k", "iso"])[c].transform("mean")
for m in ["rl", "sft", "nllb3b3"]:
    for met in ("chrf", "comet"): P[f"res_{met}_{m}"] = dm(P, f"{met}_{m}")
for met in ("chrf", "comet"): P[f"res_gap_{met}"] = dm(P.assign(g=P[f"{met}_rl"] - P[f"{met}_nllb3b3"]), "g")
BINS = [(3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.0), (5.0, 9.0)]
rows = []
rng = np.random.default_rng(11)
sets = [("FLORES+ devtest", 1), ("NTREX-128", 1), ("MAFAND-MT", 1), ("SmolSent", 1), ("pooled (all benchmarks)", 1), ("pooled (all benchmarks)", 3), ("pooled (all benchmarks)", 5), ("NTREX-128", 5), ("MAFAND-MT", 5)]
inv = {v: k for k, v in BNAME.items()}
for name, k in sets:
    d0 = P[P.k == k] if name.startswith("pooled") else P[(P.b == inv[name]) & (P.k == k)]
    for g in ["all", "strong", "mid", "weak"]:
        d1 = d0 if g == "all" else d0[d0.group == g]
        for outc, col in [("RL AfriCOMET", "res_comet_rl"), ("NLLB-3.3B AfriCOMET", "res_comet_nllb3b3"), ("RL chrF++", "res_chrf_rl"), ("NLLB-3.3B chrF++", "res_chrf_nllb3b3"), ("RL-NLLB AfriCOMET", "res_gap_comet"), ("RL-NLLB chrF++", "res_gap_chrf")]:
            dd = d1.dropna(subset=[col]); 
            gtab = dd.groupby("ekey").agg(x=("score", "first"), S=(col, "sum"), N=(col, "size"))
            x, S, N = gtab.x.values, gtab.S.values, gtab.N.values; C = len(x)
            ref = x <= 3.0
            if ref.sum() < 20: continue
            idxs = [rng.integers(0, C, C) for _ in range(400)]
            def diff(mask, idx=None):
                if idx is None: idx = np.arange(C)
                a = S[idx][mask[idx]].sum() / max(N[idx][mask[idx]].sum(), 1); r = S[idx][ref[idx]].sum() / max(N[idx][ref[idx]].sum(), 1); return a - r
            for lo, hi in BINS:
                mk = (x > lo) & (x <= hi)
                if mk.sum() < 20: continue
                e = diff(mk); bs = np.array([diff(mk, i) for i in idxs])
                rows.append({"dataset": name, "k": k, "group": g, "outcome": outc, "bin": f"({lo},{hi if hi < 9 else 'max'}]", "n_chunks_bin": int(mk.sum()), "n_chunks_ref": int(ref.sum()), "diff_vs_very_easy": e, "lo": np.quantile(bs, .025), "hi": np.quantile(bs, .975),
                             "significant_drop": bool(np.quantile(bs, .975) < 0)})
pd.DataFrame(rows).to_csv(f"{OUT}/04c_bin_vs_very_easy_reference.csv", index=False, float_format="%.4g")
t = pd.DataFrame(rows)
print(t[(t.group == "all") & t.outcome.isin(["RL AfriCOMET", "NLLB-3.3B AfriCOMET", "RL-NLLB AfriCOMET", "RL chrF++", "RL-NLLB chrF++"])].pivot_table(index=["dataset", "k", "outcome"], columns="bin", values="diff_vs_very_easy").round(3).to_string())
