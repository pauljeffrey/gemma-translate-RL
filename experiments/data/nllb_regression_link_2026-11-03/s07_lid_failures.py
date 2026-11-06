#!/usr/bin/env python3
"""Step 7: what are the NLLB-mined rows that GlotLID does not accept as the language? (top wrong labels, examples) and the link to the 385M's OUTPUT language-ID drop vs A0.
Outputs: nllb_lid_failure_labels.csv, nllb_lid_failure_examples.csv, lid_mechanism_corr.csv
"""
import glob, os, sys, collections
import numpy as np, pandas as pd
from scipy import stats
import fasttext
H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gemma_mt"))
from gemma_mt.languages import LANGUAGES
fasttext.FastText.eprint = lambda *a, **k: None
m = fasttext.load_model(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--cis-lmu--glotlid/snapshots/*/model_v3.bin"))[0])
smp = pd.read_parquet(os.path.join(H, "token_sample.parquet"))
rng = np.random.default_rng(5)
rows, ex = [], []
for iso, g in smp[smp.grp == "nllb_mined"].groupby("iso"):
    if iso not in LANGUAGES or len(g) < 60:
        continue
    acc = set(LANGUAGES[iso].glotlid)
    g = g.sample(min(len(g), 500), random_state=3)
    labs = []
    for t, e_ in zip(g.target, g.english):
        t1 = " ".join(t.split())[:600]
        lab, p = m.predict(t1, k=1) if t1 else (("__label__",), (0,))
        labs.append((lab[0].replace("__label__", ""), float(p[0]), t, e_))
    fail = [x for x in labs if x[0] not in acc]
    c = collections.Counter(x[0] for x in fail)
    rows.append(dict(iso=iso, n=len(labs), fail=len(fail) / len(labs), top_wrong="; ".join(f"{k}:{v/len(labs):.2f}" for k, v in c.most_common(4)),
                     mean_p_wrong=float(np.mean([x[1] for x in fail])) if fail else np.nan))
    for x in list(rng.permutation(len(fail)))[:3]:
        ex.append(dict(iso=iso, lid=fail[x][0], p=round(fail[x][1], 2), english=fail[x][3][:160], target=fail[x][2][:160]))
r = pd.DataFrame(rows)
r.to_csv(os.path.join(H, "nllb_lid_failure_labels.csv"), index=False)
pd.DataFrame(ex).to_csv(os.path.join(H, "nllb_lid_failure_examples.csv"), index=False)
pd.set_option("display.width", 250)
print(r.sort_values("fail", ascending=False).round(2).to_string())

# ---- mechanism check: NLLB target-LID failure rate vs the change of the model's OUTPUT LID pass (385M SFT - A0), FLORES devtest, all rows
f = pd.read_csv(os.path.join(H, "..", "eval_385m_results_2026-11-03", "flores_per_language_385m_full.csv"))
f = f[(f.subset == "all") & (f.benchmark == "flores_plus")][["iso", "lid_ok_a0", "lid_ok_m385", "chrf_a0", "chrf_m385"]]
f["dlid"] = f.lid_ok_m385 - f.lid_ok_a0
q = pd.read_csv(os.path.join(H, "nllb_quality_by_language.csv"))[["iso", "lid_target_ok_nllb_mined", "lid_target_ok_other"]]
sh = pd.read_csv(os.path.join(H, "nllb_share_by_language.csv"))[["iso", "share_nllb_mined", "instr_tokens", "tumbuka_mt", "tokens_B"]]
d = f.merge(q, on="iso").merge(sh, on="iso")
d = d[~d.iso.isin(["ful", "kon", "tzm", "bam"])]            # GlotLID label mismatch (ful/kon/tzm) or LID unreliable on the references (bam: 0.60 for non-NLLB too)
d["fail_nllb"] = 1 - d.lid_target_ok_nllb_mined
d["fail_excess"] = (d.lid_target_ok_other - d.lid_target_ok_nllb_mined)
d["nllb_fail_tokens_share"] = d.fail_excess.clip(lower=0) * d.share_nllb_mined
out = []
for x, y in [("fail_nllb", "dlid"), ("fail_excess", "dlid"), ("nllb_fail_tokens_share", "dlid"), ("share_nllb_mined", "dlid"), ("fail_excess", "chrf_m385"), ("fail_excess", "chrf_a0")]:
    rho = stats.spearmanr(d[x], d[y])
    bs = [stats.spearmanr(d.iloc[i][x], d.iloc[i][y]).statistic for i in (rng.integers(0, len(d), len(d)) for _ in range(3000))]
    out.append(dict(x=x, y=y, n=len(d), rho=rho.statistic, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5), p=rho.pvalue))
c = pd.DataFrame(out)
c.to_csv(os.path.join(H, "lid_mechanism_corr.csv"), index=False)
print(c.round(3).to_string())
d.to_csv(os.path.join(H, "lid_mechanism_table.csv"), index=False)
