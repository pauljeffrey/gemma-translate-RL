#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""H1 step 2: run the final run's detectors (anchors v3 etc.) and GlotLID on every stored sentence-level output and on the human references.
Reads $HE_WORK/rows.parquet (step 1), writes $HE_WORK/indicators.parquet (one row per system / bench / iso / id, no text) and $HE_WORK/ref_indicators.parquet (human references).
    HE_WORK=... python hardening_evidence_02_indicators.py [--procs 8]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hardening_evidence_common as C  # noqa: E402
import hardening_evidence_detectors as D  # noqa: E402


def work(args):
    return [D.sentence_row(i, s, h) for i, s, h in args]


def run(df, text_col, procs):
    items = list(zip(df["iso"].tolist(), df["src"].tolist(), df[text_col].tolist()))
    chunks = [items[i:i + 2000] for i in range(0, len(items), 2000)]
    out = []
    with Pool(procs, initializer=D.init) as pool:
        for k, r in enumerate(pool.imap(work, chunks)):
            out += r
            if k % 20 == 0:
                print(f"  {len(out)}/{len(items)}", flush=True)
    return pd.DataFrame(out)


def add_lid(ind, texts, isos):
    lid = D.Lid()
    uniq = list(dict.fromkeys(texts))
    t0 = time.time()
    tops = dict(zip(uniq, lid.top1(uniq)))
    print("glotlid", len(uniq), "texts", round(time.time() - t0), "s", flush=True)
    res = [lid.verdict(i, t, tops[t]) for i, t in zip(isos, texts)]
    ind["lid_judged"] = [r[0] for r in res]
    ind["lid_wrong"] = [r[1] for r in res]
    ind["lid_top1"] = [tops[t][0] for t in texts]
    ind["lid_prob"] = [tops[t][1] for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    df = pd.read_parquet(os.path.join(C.WORK, "rows.parquet"))
    df = df.drop_duplicates(["system", "bench", "iso", "id"], keep="first").reset_index(drop=True)
    rl77 = set(D.rl77())
    df = df[df["iso"].isin(rl77)].reset_index(drop=True)                      # the 77 RL languages: the others (8 zero-shot tags) have no calibration block
    print("rows", len(df), df["system"].value_counts().to_dict())
    t0 = time.time()
    ind = run(df, "hyp", a.procs)
    print("detectors", round(time.time() - t0), "s")
    add_lid(ind, df["hyp"].tolist(), df["iso"].tolist())
    for c in ("system", "bench", "iso", "id", "pair_seen", "contaminated", "finished", "chrf", "afri_qe", "ssa_qe", "afri_ref", "ssa_ref"):
        ind[c] = df[c].values
    ind["unfinished"] = (~ind["finished"].astype(bool)).astype(float)
    ind["content_mode"] = ind["iso"].map(D.mode_of)
    ind.to_parquet(os.path.join(C.WORK, "indicators.parquet"))
    # human references: one per (bench, iso, id)
    refs = df.drop_duplicates(["bench", "iso", "id"])[["bench", "iso", "id", "src", "ref", "pair_seen"]].reset_index(drop=True)
    refs = refs[refs["ref"].str.strip().ne("")].reset_index(drop=True)
    r = run(refs, "ref", a.procs)
    add_lid(r, refs["ref"].tolist(), refs["iso"].tolist())
    for c in ("bench", "iso", "id", "pair_seen"):
        r[c] = refs[c].values
    r["content_mode"] = r["iso"].map(D.mode_of)
    r.to_parquet(os.path.join(C.WORK, "ref_indicators.parquet"))
    print("done", len(ind), len(r), round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
