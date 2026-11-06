"""E-03: loop / degeneration rates of the control (C1 step 6,000) with BOTH detectors, next to the stored values of SFT, stage A, B1 best, mixed best.
Reward-family flag (rw_loop: guards.is_repetitive or is_self_repetitive) and the independent detector of E-08 (lzma residual / char 8-gram / word 3-gram, thresholds from the
0.1% tail of the 122,585 human references), unfinished rate, language macro per benchmark. No LID. Usage: python c1_degeneration.py PREDICTIONS.jsonl OUT.csv"""
import os, sys, json
import numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
AUD = os.path.join(ROOT, "reports/data/degeneration_audit_independent_2026-11-04")
sys.path.insert(0, AUD)
import compute_features as CF
def main():
    pred, out = sys.argv[1], sys.argv[2]
    R = pd.read_parquet(os.path.join(AUD, "refs_features.parquet"))
    R["r_logb"] = np.log(R["r_bytes"].clip(lower=1)); R["r_logc"] = np.log(R["r_lz_bytes"].clip(lower=1))
    coef = {}
    for (b, i), g in R.groupby(["benchmark", "iso"]):
        g = g[g["r_bytes"] >= 20]
        coef[(b, i)] = np.polyfit(g["r_logb"], g["r_logc"], 1) if len(g) >= 100 and g["r_logb"].std() > 0 else None
    g = R[R.r_bytes >= 20]; pooled = np.polyfit(g["r_logb"], g["r_logc"], 1)
    THR = json.load(open(os.path.join(AUD, "thresholds.json")))["thresholds_from_references_p0.1_or_p99.9"]
    rows = CF.load([pred])
    with Pool(4) as p:
        feats = p.map(CF.row_feats, [(r["hyp"], r["reference"], r["english"], r["finished"], r["prefix_status"]) for r in rows], chunksize=2000)
    df = pd.concat([pd.DataFrame({"benchmark": [r["benchmark"] for r in rows], "iso": [r["iso"] for r in rows], "id": [str(r["id"]) for r in rows]}), pd.DataFrame(feats)], axis=1)
    lb = np.log(df["h_bytes"].clip(lower=1).values); lc = np.log(df["h_lz_bytes"].clip(lower=1).values); z = np.full(len(df), np.nan)
    for (b, i), idx in df.groupby(["benchmark", "iso"]).indices.items():
        c = coef.get((b, i)); c = pooled if c is None else c
        z[idx] = lc[idx] - (c[0] * lb[idx] + c[1])
    z[df["h_bytes"].values < 20] = np.nan
    df["ind_loop"] = (z < THR["z"]) | (df["h_c8_rep"] > THR["c8"]) | (df["h_w3_rep"] > THR["w3"])
    df["unfinished"] = ~df["finished"].astype(bool)
    res = []
    for b, d in df.groupby("benchmark"):
        pl = d.groupby("iso")[["rw_loop", "ind_loop", "unfinished"]].mean()
        res.append(dict(model="c1_s6000", benchmark=b, n_rows=len(d), n_lang=len(pl), rw_loop=pl.rw_loop.mean(), ind_loop=pl.ind_loop.mean(), unfinished=pl.unfinished.mean()))
    r = pd.DataFrame(res); r.to_csv(out, index=False); print(r.to_string())
    S = pd.read_csv(os.path.join(AUD, "summary_by_model_benchmark.csv"))
    print(S[S.model.isin(["sft", "stageA", "b1best", "mixedbest"])][["model", "benchmark", "n_rows", "n_lang", "rw_loop", "ind_loop", "unfinished"]].to_string())


if __name__ == '__main__':
    main()
