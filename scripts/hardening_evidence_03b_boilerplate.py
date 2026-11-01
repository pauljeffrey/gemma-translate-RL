#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""H1 step 3b: cross-prompt boilerplate, the symptom of the consensus attractor (paper Section 7.6, anchors `boiler`): share of the word 5-grams of an output (those not copied from its
source) that occur in the outputs of >= 3 OTHER prompts of the same language (prompts whose English sources share a word 6-gram with this one do not count as "other"). In the
run the buffer holds the last 600 outputs; here the pool is every output of the same system in the same (benchmark, language), so the number of prompts is the same for every system
being compared. The human references of the same prompts are the baseline (the calibrated limit is 3% of 5-grams). Reuses gemma_mt.rl.anchors.prepare.
Output $HE_OUT/boilerplate_by_system.csv (per system, benchmark, language: mean boilerplate share, share of outputs above the 3% limit) and boilerplate_paired.csv (final2 - RL-final).
"""
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hardening_evidence_common as C  # noqa: E402
import hardening_evidence_detectors as D  # noqa: E402
from gemma_mt.rl import anchors as AN  # noqa: E402

warnings.filterwarnings("ignore")
MIN_OTHER = 3
LIMIT = 0.03


def cell_boiler(srcs, hyps):
    """per-output boilerplate share for one pool of prompts (list of English sources, list of outputs)."""
    n = len(srcs)
    prep = [AN.prepare("x", s, h, key=i, with_grams=True) for i, (s, h) in enumerate(zip(srcs, hyps))]
    gram_prompts = defaultdict(set)
    for i, p in enumerate(prep):
        for g in p.grams.tolist():
            gram_prompts[g].add(i)
    src_idx = defaultdict(set)                                  # English 6-gram -> prompts
    for i, p in enumerate(prep):
        for g in p.src_hashes.tolist():
            src_idx[g].add(i)
    out = np.full(n, np.nan)
    for i, p in enumerate(prep):
        if p.grams.size == 0:
            continue
        rel = set()
        for g in p.src_hashes.tolist():
            rel |= src_idx[g]
        rel.add(i)
        hit = 0
        for g in p.grams.tolist():
            if len(gram_prompts[g] - rel) >= MIN_OTHER:
                hit += 1
        out[i] = hit / p.grams.size
    return out


def main():
    rows = pd.read_parquet(os.path.join(C.WORK, "rows.parquet"), columns=["system", "bench", "iso", "id", "src", "ref", "hyp", "pair_seen"])
    rows = rows[~rows.pair_seen & rows.iso.isin(D.rl77())].drop_duplicates(["system", "bench", "iso", "id"])
    groups = pd.read_csv(os.path.join(C.OUT, "language_groups.csv")).set_index("iso")
    res = []
    for (bench, iso), g in rows.groupby(["bench", "iso"]):
        if bench.startswith("afs") or len(g) < 80:
            continue
        # same prompts for every system: the intersection of ids
        systems = sorted(g.system.unique())
        ids = None
        for s in systems:
            x = set(g[g.system == s].id)
            ids = x if ids is None else ids & x
        ids = sorted(ids)
        if len(ids) < 80:
            continue
        base = g[g.system == systems[0]].set_index("id").loc[ids]
        pools = {s: g[g.system == s].set_index("id").loc[ids] for s in systems}
        pools["human reference"] = base.assign(hyp=base.ref)
        for s, pdf in pools.items():
            b = cell_boiler(pdf.src.tolist(), pdf.hyp.tolist())
            for i, v in zip(ids, b):
                res.append((s, bench, iso, i, v))
        print(bench, iso, len(ids), flush=True)
    r = pd.DataFrame(res, columns=["system", "bench", "iso", "id", "boiler"])
    r.to_parquet(os.path.join(C.WORK, "boiler_rows.parquet"))
    r["above"] = (r.boiler > LIMIT).astype(float)
    r["tier"] = r.iso.map(groups["tier"])
    c = r.groupby(["system", "bench", "iso", "tier"]).agg(boiler=("boiler", "mean"), above=("above", "mean"), n=("boiler", "size")).reset_index()
    out = []
    for scope in ("ALL",) + tuple(sorted(r.bench.unique())):
        cc = c if scope == "ALL" else c[c.bench == scope]
        for tier in ("all",) + tuple(sorted(cc.tier.dropna().unique())):
            x = cc if tier == "all" else cc[cc.tier == tier]
            per_lang = x.groupby(["system", "iso"])[["boiler", "above"]].mean().reset_index()
            for s, y in per_lang.groupby("system"):
                out.append(dict(scope=scope, tier=tier, system=s, n_lang=y.iso.nunique(), boiler=y.boiler.mean(), above_limit=y.above.mean()))
    pd.DataFrame(out).to_csv(os.path.join(C.OUT, "boilerplate_by_system.csv"), index=False, float_format="%.5g")
    # paired final2 - rl, rows resampled within language (macro over languages)
    a = r[r.system == "rl"].set_index(["bench", "iso", "id"])
    b = r[r.system == "final2"].set_index(["bench", "iso", "id"])
    idx = a.index.intersection(b.index)
    d = pd.DataFrame({"boiler": b.loc[idx, "boiler"].values - a.loc[idx, "boiler"].values, "above": b.loc[idx, "above"].values - a.loc[idx, "above"].values,
                      "bench": idx.get_level_values(0), "iso": idx.get_level_values(1)})
    d["tier"] = d.iso.map(groups["tier"])
    rows_out = []
    for scope in ("ALL",) + tuple(sorted(d.bench.unique())):
        for tier in ("all",) + tuple(sorted(d.tier.dropna().unique())):
            x = d if scope == "ALL" else d[d.bench == scope]
            x = x if tier == "all" else x[x.tier == tier]
            if not len(x):
                continue
            rec = dict(scope=scope, tier=tier, n_rows=len(x), n_lang=x.iso.nunique())
            for col in ("boiler", "above"):
                diffs = {l: v[col].dropna().values for l, v in x.groupby("iso")}
                pt, lo, hi = C.macro_paired_bootstrap(diffs, B=1000)
                rec[col + "_diff"], rec[col + "_lo"], rec[col + "_hi"] = pt, lo, hi
            rows_out.append(rec)
    pd.DataFrame(rows_out).to_csv(os.path.join(C.OUT, "boilerplate_paired_final2_minus_rl.csv"), index=False, float_format="%.5g")
    print(pd.DataFrame(out).query("scope=='ALL'").round(4).to_string())
    print(pd.DataFrame(rows_out).query("scope=='ALL'").round(4).to_string())


if __name__ == "__main__":
    main()
