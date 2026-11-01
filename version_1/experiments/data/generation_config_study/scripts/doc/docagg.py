"""R12 document-level aggregation (venv_gemma, in doc/). usage: docagg.py <out.json> <base> <sel|sel2> name [name ...]
For each core bin (256..2048; chrF++ vs NLLB, macro over B8, documents bootstrapped, shared resample) and each clean human bin (SmolDoc refs, 6 languages) the configs are compared on the
documents present in ALL configs (paired). sel = all selected documents, sel2 = the unit-level subset."""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, "/workspace/jobs/r12/doc")
from common import *
import agg
B8 = set("hau yor swh amh zul wol kea sag".split())
out_p, base, selname = sys.argv[1:4]; names = sys.argv[4:]
if base not in names: names = [base] + names
SEL = set(json.load(open(MYWORK + ("/sel_iids.json" if selname == "sel" else "/sel2_iids.json"))))
rows = {}
for n in names:
    p = f"{ROOT}/sc/rl2c/{n}.pkl"
    rows[n] = [r for r in pickle.load(open(p, "rb")) if r["iid"] in SEL and r["iso"] in B8]
common = set.intersection(*[{r["iid"] for r in rows[n]} for n in names])
print("common items", len(common), {n: len(rows[n]) for n in names})
rows = {n: [r for r in rows[n] if r["iid"] in common] for n in names}
res = {"names": names, "base": base, "n_common": len(common), "core": {}, "human": {}, "fail_core": {}, "fail_human": {}}
bn = {n: agg.build_core(rows[n], "nllb") for n in names}
for b in (256, 512, 1024, 2048):
    if all(b in bn[n] for n in names):
        res["core"][b] = agg.macro_boot({n: bn[n][b] for n in names}, 500)
        res["core"][b].pop("_pairs", None)
for hb in ("sm", "st512", "st1024", "st2048"):
    r = agg.human_macro(rows, names, [hb], False, 500)
    if r: res["human"][hb] = r
for n in names:
    fc = {}
    for b, d in agg.failure_rows([r for r in rows[n] if r["set"] == "core"], "core").items(): fc[b] = d
    res["fail_core"][n] = fc
    res["fail_human"][n] = agg.failure_rows([r for r in rows[n] if r["set"] == "human"], "human")
json.dump(res, open(out_p, "w"), default=lambda o: float(o) if hasattr(o, "item") else str(o))
for b, t in res["core"].items():
    print("core", b); [print(f"  {n:28s} {t[n]['chrf']:.1f} [{t[n]['chrf_lo']:.1f},{t[n]['chrf_hi']:.1f}] d {t[n]['d_chrf_vs_base']:+.1f} [{t[n]['d_lo']:+.1f},{t[n]['d_hi']:+.1f}] fin {res['fail_core'][n][b]['finished']:.2f} loop {res['fail_core'][n][b]['loop']:.2f} len {res['fail_core'][n][b]['length_ratio_med']:.2f}") for n in names]
