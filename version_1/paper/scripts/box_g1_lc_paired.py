#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""G1 (box, CPU): paired document-bootstrap differences final2 minus {live, A2W, RL-final} in LongCtx-MT (same documents, same draws; B=1000), macro over the common languages
(SmolDoc amh hau ibo swh yor zul; AfriScience amh hau lug yor zul; core: the same six).  Re-uses the loading and bootstrap code of /workspace/jobs/lc1/lc_agg.py (the part before the
'cells' section), so the point estimates equal macro_common.csv.  usage (cwd /workspace/jobs/lc1, venv_gemma): box_g1_lc_paired.py OUT.csv"""
import sys, csv, zlib
out = sys.argv[1]
sys.argv = ["lc_agg.py", "--only", "final2,live,a2w,rl", "--boot", "1000", "--langs", "amh,hau,ibo,swh,yor,zul,lug"]
src = open("/workspace/jobs/lc1/lc_agg.py").read().split("# ---------------------------------------------------------------- cells")[0]
exec(compile(src, "lc_agg_head", "exec"))
rows = []
CASES = [("human", "st512", "human"), ("human", "st1024", "human"), ("human", "st2048", "human"), ("sci", "sci_dev", "human"), ("sci", "sci_test", "human"),
         ("core", "b512", "nllb"), ("core", "b1024", "nllb"), ("core", "b2048", "nllb")]
for (setn, binn, ref) in CASES:
    for mode in ("whole", "sentfix"):
        la = {}
        for m in ("final2", "live", "a2w", "rl"):
            n = f"{m}__{mode}"
            if n in CELLS and (setn, binn) in CELLS[n]:
                la[m] = lang_arrays(CELLS[n][(setn, binn)], ref, sorted(CELLS[n][(setn, binn)]))
        for other in ("live", "a2w", "rl"):
            if "final2" not in la or other not in la:
                continue
            isos = sorted(set(la["final2"]) & set(la[other]))
            sub = {"final2": {i: la["final2"][i] for i in isos}, other: {i: la[other][i] for i in isos}}
            assert all(sub["final2"][i]["ids"] == sub[other][i]["ids"] for i in isos), "document lists differ"
            res, isos = macro_boot(sub, rng_seed=zlib.crc32(f"{setn}{binn}".encode()) % 1000, nb=1000)
            a, b = res["final2"], res[other]
            for met in ["chrf", "finished", "loop", "cov_sents", "length_ratio"]:
                d = a["draws"][met] - b["draws"][met]
                lo, hi = ci(d)
                rows.append(dict(set=setn, bin=binn, ref=ref, mode=mode, other=other, metric=met, n_lang=len(isos), final2=a[met], other_value=b[met], delta=a[met] - b[met], lo=lo, hi=hi))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("rows", len(rows))
