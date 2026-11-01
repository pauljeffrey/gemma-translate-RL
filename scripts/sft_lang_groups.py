#!/usr/bin/env python
"""Per-language / group deltas of a candidate vs a baseline (weak suite chrF++, general suite chrF++, AfriScience dev sentence chrF++ and paper chrF++), bootstrap CI over languages."""
import json, os, sys, argparse
sys.path.insert(0, "/workspace/jobs/r7/gemma_mt_run/scripts")
from sft_stage_eval import _boot_ci, _load
ap = argparse.ArgumentParser(); ap.add_argument("--baseline"); ap.add_argument("--run"); ap.add_argument("--name", default="run"); ap.add_argument("--base-name", default="base")
a = ap.parse_args()
B, R = _load(a.baseline), _load(a.run)
NLLB = "kbp nus kea bam tir kon amh kab sag luo ewe bem orm run tum kik zul xho lug kin mlg sna tsn lin ssw tso".split()
FOCUS = ["amh", "tir", "kea", "kab", "nus", "bam", "kbp"]
def get(r, suite, iso, key="chrf_pp"):
    return r["suites"].get(suite, {}).get("per_language", {}).get(iso, {}).get(key)
def afs(r, level, iso, key):
    return r["suites"]["afs"]["dev"][level]["per_language"].get(iso, {}).get(key)
rows = []
print(f"| language | weak-dev chrF++ {a.base_name} | {a.name} | delta | general-dev delta | afs sentence delta | afs paper chrF++ delta |")
print("|---|---|---|---|---|---|---|")
for iso in FOCUS:
    b, r = get(B, "weak", iso), get(R, "weak", iso)
    g = (get(R, "general", iso) - get(B, "general", iso)) if get(B, "general", iso) is not None and get(R, "general", iso) is not None else None
    s = (afs(R, "sentence", iso, "chrf_pp") - afs(B, "sentence", iso, "chrf_pp")) if afs(B, "sentence", iso, "chrf_pp") is not None else None
    d = (afs(R, "document", iso, "chrf_pp") - afs(B, "document", iso, "chrf_pp")) if afs(B, "document", iso, "chrf_pp") is not None else None
    f = lambda x: "-" if x is None else f"{x:+.2f}"
    print(f"| {iso} | {b if b is None else round(b,2)} | {r if r is None else round(r,2)} | {f(None if b is None or r is None else r-b)} | {f(g)} | {f(s)} | {f(d)} |")
def grp(name, langs, suite):
    ds = [get(R, suite, i) - get(B, suite, i) for i in langs if get(B, suite, i) is not None and get(R, suite, i) is not None]
    if not ds: return
    lo, hi = _boot_ci(ds)
    print(f"| {name} ({suite}) | {len(ds)} | {sum(ds)/len(ds):+.2f} [{lo:+.2f}, {hi:+.2f}] |")
print(); print("| group | n languages | mean delta chrF++ [95% CI over languages] |"); print("|---|---|---|")
grp("NLLB-covered teacher languages", NLLB, "weak"); grp("NLLB-covered teacher languages", NLLB, "general")
notcov = [i for i in B["suites"]["weak"]["per_language"] if i not in NLLB]
grp("NOT NLLB-covered", notcov, "weak")
ge = ["amh", "tir"]; grp("Ge'ez (amh, tir)", ge, "weak")
grp("consensus-only/blend focus (kea kab nus bam kbp)", ["kea", "kab", "nus", "bam", "kbp"], "weak")
