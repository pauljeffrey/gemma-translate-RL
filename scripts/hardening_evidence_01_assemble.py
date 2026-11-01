#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""H1 step 1: assemble ONE table of stored sentence-level outputs: SFT-final, RL-final, final2 (step 401), final-run step 250 (FLORES+ only) and A2W alpha 0.7 (FLORES+ and
AfriScience only), keyed by (system, bench, iso, id). Writes $HE_WORK/rows.parquet (text included; NOT copied to the repository: the FLORES+ terms ask us not to re-host the text).

Sources (all stored, nothing is regenerated):
  sft, rl      box/jobs/e1/sys_{sft,rl}.jsonl.gz  (E1 pipeline: English source, reference, hypothesis, chrF++, finished / loop / lid flags; all benchmarks, all rows)
  final2       Hub dataset <HF_ORG_LEGACY>/SabiYarn-MT-benchmark-translations eval_benchmarks/*.parquet (hypothesis + AfriCOMET-QE, SSA-COMET-QE, with-reference scores, finished, lid)
               (source / reference text come from the E1 rows, which hold the same benchmark rows; AfriScience from the *_text.parquet files)
  live250, a2w tarball e2/out_{live250,a2w}/predictions.jsonl (FLORES+ only)
  AfriScience  box/jobs/a4/eval_final{1433,2_step401}, box/jobs/r7/eval/A2W{_a0.7_test,/alpha_0.70}: sentence level (same row order as the Hub files)
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hardening_evidence_common as C  # noqa: E402


def load_e1(name):
    rows = C.read_jsonl(os.path.join(C.JOBS, "e1", f"sys_{name}.jsonl.gz"))
    df = pd.DataFrame(rows)
    df["system"] = name
    df = df.rename(columns={"b": "bench", "en": "src", "fin": "finished"})
    for c in ("contaminated", "pair_seen"):
        df[c] = df[c].astype(str).str.lower().eq("true")
    df["id"] = df["id"].astype(str)
    return df


def main():
    hub = C.ensure_hub(["eval_benchmarks/*", "stats/*"])
    tar = C.ensure_tar_members(["e2/out_a2w/predictions.jsonl", "e2/out_live250/predictions.jsonl"])
    parts = []
    base = {}
    for n in ("sft", "rl"):
        d = load_e1(n)
        parts.append(d)
        base[n] = d
    b = base["rl"].set_index(["bench", "iso", "id"])
    # ---- final2 from the Hub files
    f2 = []
    hubfiles = {"flores_plus": ["flores_plus.parquet"], "ntrex": ["ntrex.parquet"], "smol_sent": ["smol_sent.parquet"], "salt": ["salt.parquet"],
                "mafand": ["noncommercial/mafand.parquet"], "glot_story": ["glot_story.parquet", "noncommercial/glot_story.parquet"]}
    for bench, fl in hubfiles.items():
        for f in fl:
            h = pd.read_parquet(os.path.join(hub, "eval_benchmarks", f))
            h = h.rename(columns={"language": "iso", "hypothesis": "hyp", "row_id": "id"})
            h["bench"] = bench
            f2.append(h)
    f2 = pd.concat(f2, ignore_index=True)
    f2["id"] = f2["id"].astype(str)
    f2 = f2.join(b[["src", "ref", "contaminated", "pair_seen"]], on=["bench", "iso", "id"], how="left")
    miss = f2["src"].isna().sum()
    print("final2 sentence rows", len(f2), "without E1 source/reference:", miss)
    f2 = f2[f2["src"].notna()].copy()
    f2["system"] = "final2"
    parts.append(f2[["system", "bench", "iso", "id", "src", "ref", "hyp", "pair_seen", "contaminated", "finished", "chrf", "afri_qe", "ssa_qe", "afri_ref", "ssa_ref", "lid_top", "lid_ok"]])
    # ---- a2w / live250 (FLORES+)
    for name, d in (("live250", "out_live250"), ("a2w", "out_a2w")):
        rows = C.read_jsonl(os.path.join(tar, "e2", d, "predictions.jsonl"))
        x = pd.DataFrame(rows)
        x["id"] = x["id"].astype(str)
        x = x[x["benchmark"] == "flores_plus"].rename(columns={"benchmark": "bench"})
        x["finished"] = x["finished"].astype(str).str.lower().eq("true")
        x = x.join(b[["src", "ref"]], on=["bench", "iso", "id"], how="left")
        x["pair_seen"] = x["pair_seen"].astype(str).str.lower().eq("true")
        x["contaminated"] = x["contaminated"].astype(str).str.lower().eq("true")
        x["system"] = name
        parts.append(x[["system", "bench", "iso", "id", "src", "ref", "hyp", "pair_seen", "contaminated", "finished"]])
        print(name, len(x), "missing src", int(x["src"].isna().sum()))
    # ---- AfriScience sentence level
    afs = []
    srcs = {"dev": {"rl": os.path.join(C.JOBS, "a4", "eval_final1433", "predictions_dev.json.gz"), "a2w": os.path.join(C.JOBS, "r7", "eval", "A2W", "alpha_0.70", "afs_predictions_dev.json.gz"),
                    "final2": os.path.join(C.JOBS, "a4", "eval_final2_step401", "predictions_dev.json.gz")},
            "test": {"rl": os.path.join(C.JOBS, "a4", "eval_final1433", "predictions_test.json.gz"), "a2w": os.path.join(C.JOBS, "r7", "eval", "A2W_a0.7_test", "afs_predictions_test.json.gz"),
                     "final2": os.path.join(C.JOBS, "a4", "eval_final2_step401", "predictions_test.json.gz")}}
    for split, m in srcs.items():
        h = pd.read_parquet(os.path.join(hub, "eval_benchmarks", f"afriscience_mt_{split}.parquet"))
        t = pd.read_parquet(os.path.join(hub, "eval_benchmarks", f"afriscience_mt_{split}_text.parquet"))
        h = h[h["level"] == "sentence"]
        t = t[t["level"] == "sentence"].set_index(["language", "row_id"])
        for sysname, path in m.items():
            d = json.load(gzip.open(path, "rt", encoding="utf-8"))["sentence"]
            for iso, lst in d.items():
                hh = h[h["language"] == iso]
                assert len(hh) == len(lst), (split, sysname, iso, len(hh), len(lst))
                for (rid, row), e in zip(hh.set_index("row_id").iterrows(), lst):
                    tt = t.loc[(iso, rid)]
                    afs.append((sysname, f"afs_{split}", iso, rid, tt["source_en"], tt["reference"], e["hyp"], False, False, bool(e["finished"]), np.nan,
                                row["afri_qe"] if sysname == "final2" else np.nan, row["ssa_qe"] if sysname == "final2" else np.nan,
                                row["afri_ref"] if sysname == "final2" else np.nan, row["ssa_ref"] if sysname == "final2" else np.nan))
            # sanity: the stored final2 hyps of the Hub equal the a4 predictions
    afs = pd.DataFrame(afs, columns=["system", "bench", "iso", "id", "src", "ref", "hyp", "pair_seen", "contaminated", "finished", "chrf", "afri_qe", "ssa_qe", "afri_ref", "ssa_ref"])
    parts.append(afs)
    df = pd.concat(parts, ignore_index=True)
    df["hyp"] = df["hyp"].fillna("").map(C.nfc)
    df["src"] = df["src"].fillna("").map(C.nfc)
    df["ref"] = df["ref"].fillna("").map(C.nfc)
    out = os.path.join(C.WORK, "rows.parquet")
    df.to_parquet(out)
    print(df.groupby(["system", "bench"]).size().unstack(0))
    print("saved", out, len(df))


if __name__ == "__main__":
    main()
