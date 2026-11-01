#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-59: guardrail table of an SFT-stage sweep. Rule (lead, 2026-10-22): among the alphas, take the best sentence-level dev chrF++ (AfriScience-MT dev, 6 languages) subject to
(a) general-dev macro chrF++ >= baseline - 0.5, (b) EACH weak-dev group (science6, weak-consensus, weak-blend, control, other) >= baseline - 0.5, (c) paper-level finished rate >= baseline.

  python scripts/sft_guardrails.py --baseline eval/final_1433 --runs eval/B_v1/alpha_0.30 eval/B_v1/alpha_0.50 ... [--md out.md] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_stage_eval import SCI6, _boot_ci, _load, group_of  # noqa: E402

GROUPS = ["science6", "weak-consensus", "weak-blend", "control", "other"]


def metrics(r, base):
    afs = r["suites"]["afs"]["dev"]
    sent = sum(afs["sentence"]["per_language"][i]["chrf_pp"] for i in SCI6) / len(SCI6)
    doc_chrf = sum(afs["document"]["per_language"][i]["chrf"] for i in SCI6) / len(SCI6)
    fin = 100 * sum(afs["document"]["per_language"][i]["finished_rate"] for i in SCI6) / len(SCI6)
    out = {"sent_chrfpp": sent, "paper_chrf": doc_chrf, "paper_finished": fin}
    gen_b, gen_r = base["suites"]["general"]["per_language"], r["suites"]["general"]["per_language"]
    ls = [i for i in gen_b if i in gen_r]
    out["general_all_delta"] = sum(gen_r[i]["chrf_pp"] - gen_b[i]["chrf_pp"] for i in ls) / len(ls)
    wb, wr = base["suites"]["weak"]["per_language"], r["suites"]["weak"]["per_language"]
    for g in GROUPS + ["ALL"]:
        ls = [i for i in wb if i in wr and (g == "ALL" or group_of(i) == g)]
        if not ls:
            continue
        d = [wr[i]["chrf_pp"] - wb[i]["chrf_pp"] for i in ls]
        lo, hi = _boot_ci(d)
        out[f"weak_{g}"] = (sum(d) / len(d), lo, hi, len(ls))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--md", default=None); ap.add_argument("--json", default=None)
    ap.add_argument("--tol", type=float, default=0.5)
    a = ap.parse_args()
    base = _load(a.baseline)
    bm = metrics(base, base)
    rows, res = [], {}
    L = ["| model | sent. chrF++ | paper chrF | paper finished % | general ALL d | " + " | ".join(f"weak {g} d [CI]" for g in GROUPS) + " | weak ALL d | guardrails |", "|---|---|---|---|---|" + "---|" * (len(GROUPS) + 2)]
    L.append(f"| final_1433 | {bm['sent_chrfpp']:.2f} | {bm['paper_chrf']:.2f} | {bm['paper_finished']:.1f} | 0 | " + " | ".join("0" for _ in GROUPS) + " | 0 | (baseline) |")
    best = None
    for d in a.runs:
        r = _load(d)
        m = metrics(r, base)
        fails = []
        if m["general_all_delta"] < -a.tol:
            fails.append("general")
        for g in GROUPS:
            if f"weak_{g}" in m and m[f"weak_{g}"][0] < -a.tol:
                fails.append(g)
        if m["paper_finished"] < bm["paper_finished"] - 1e-9:
            fails.append("finished")
        name = os.path.basename(os.path.normpath(d)) if "alpha" in d else d
        res[d] = {**m, "fails": fails}
        cells = " | ".join(f"{m[f'weak_{g}'][0]:+.2f} [{m[f'weak_{g}'][1]:+.2f},{m[f'weak_{g}'][2]:+.2f}]" if f"weak_{g}" in m else "-" for g in GROUPS)
        L.append(f"| {d.rstrip('/').split('/')[-2]}/{name} | {m['sent_chrfpp']:.2f} | {m['paper_chrf']:.2f} | {m['paper_finished']:.1f} | {m['general_all_delta']:+.2f} | {cells} | {m['weak_ALL'][0]:+.2f} | {'PASS' if not fails else 'FAIL: ' + ','.join(fails)} |")
        if not fails and (best is None or m["sent_chrfpp"] > res[best]["sent_chrfpp"]):
            best = d
    L.append("")
    L.append(f"best passing: {best if best else 'NONE -> keep final_1433 (alpha 0)'}")
    txt = "\n".join(L)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")
    if a.json:
        json.dump({"baseline": bm, "runs": res, "best": best}, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
