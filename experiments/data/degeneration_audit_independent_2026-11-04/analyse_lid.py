"""E-08 step 4: reward LID (GlotLID v3, accepted labels of gemma_mt.languages, min 15 chars) versus the independent NLLB-218 LID on the same rows.
Needs features_*.parquet (compute_features.py), glot_*.parquet (compute_glotlid.py), lid_nllb218_reference_precision_recall.csv (analyse.py).
Writes lid_comparison.csv, lid_disagreement_by_language.csv, lid_glot_reference_control.csv.
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gemma_mt"))
import gemma_mt.languages as GL
KEY = ["benchmark", "iso", "id"]; BENCH = ["flores_plus", "ntrex", "mafand"]
MODELS = ["sft", "stageA", "b1best", "mixedbest"]
pr = pd.read_csv(os.path.join(HERE, "lid_nllb218_reference_precision_recall.csv"))
RELIABLE = set(pr[pr.reliable.fillna(False)].iso)
cov = json_cov = None
# NLLB accepted labels (as analyse.py): reuse its function through the csv column 'labels'
ACC = {r.iso: set(str(r.labels).split()) for r in pr.itertuples() if r.covered}
R = pd.read_parquet(os.path.join(HERE, "refs_features.parquet"))[KEY + ["lid_label", "r_chars"]].rename(columns={"lid_label": "n_label"})
G = pd.read_parquet(os.path.join(HERE, "glot_refs.parquet")).rename(columns={"glot_label": "g_label", "glot_ok": "g_ok"})
Rr = R.merge(G[KEY + ["g_label", "g_ok"]], on=KEY)
Rr["n_ok"] = [l in ACC.get(i, set()) for l, i in zip(Rr.n_label, Rr.iso)]
Rr["checked_g"] = Rr.r_chars >= 15
# reward LID on references: recall per language (GlotLID accepted labels)
ctrl = Rr[Rr.checked_g].groupby("iso").agg(glot_ref_pass=("g_ok", "mean"), nllb_ref_pass=("n_ok", "mean"), n=("g_ok", "size")).reset_index()
ctrl["nllb_covered"] = ctrl.iso.isin(ACC); ctrl["nllb_reliable"] = ctrl.iso.isin(RELIABLE)
ctrl.to_csv(os.path.join(HERE, "lid_glot_reference_control.csv"), index=False)
rows, dis = [], []
for m in MODELS:
    F = pd.read_parquet(os.path.join(HERE, f"features_{m}.parquet"))[KEY + ["lid_label", "h_chars", "finished", "rw_loop"]].rename(columns={"lid_label": "n_label"})
    Gm = pd.read_parquet(os.path.join(HERE, f"glot_{m}.parquet"))[KEY + ["glot_label", "glot_ok"]]
    d = F.merge(Gm, on=KEY)
    d["n_ok"] = [l in ACC.get(i, set()) for l, i in zip(d.n_label, d.iso)]
    d = d[(d.h_chars >= 15) & d.iso.isin(RELIABLE)]      # the languages where BOTH LIDs are measured reliable (NLLB recall >= 0.90 on references; GlotLID labels exist for all)
    for b in BENCH:
        g = d[d.benchmark == b]
        both_fail = (~g.glot_ok & ~g.n_ok).groupby(g.iso).mean().mean()
        rows.append(dict(model=m, benchmark=b, n_lang=g.iso.nunique(), n_rows=len(g), glot_pass=g.groupby("iso").glot_ok.mean().mean(), nllb_pass=g.groupby("iso").n_ok.mean().mean(),
                         both_fail=both_fail, glot_fail_only=(~g.glot_ok & g.n_ok).groupby(g.iso).mean().mean(), nllb_fail_only=(g.glot_ok & ~g.n_ok).groupby(g.iso).mean().mean(),
                         agree_rate=((g.glot_ok == g.n_ok)).mean(),
                         nllb_fail_given_glot_pass=((~g.n_ok) & g.glot_ok).sum() / max(1, g.glot_ok.sum())))
        if m in ("mixedbest", "sft"):
            t = g.groupby("iso").agg(n=("n_ok", "size"), glot_pass=("glot_ok", "mean"), nllb_pass=("n_ok", "mean")).reset_index()
            t["model"] = m; t["benchmark"] = b; dis.append(t)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "lid_comparison.csv"), index=False)
pd.concat(dis).to_csv(os.path.join(HERE, "lid_pass_by_language_glot_vs_nllb.csv"), index=False)
print(pd.DataFrame(rows).round(4).to_string())
print(ctrl.describe().round(3))
