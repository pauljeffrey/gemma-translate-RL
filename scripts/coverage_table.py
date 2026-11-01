#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-61: per weak language (consensus / blend): which guards cover it. Markdown on stdout.
    python scripts/coverage_table.py --cal calibration/anchors_v2.json --shaping calibration/shaping_flores_dev_v8.json [--lexicon-dir lexicon] [--docs-eval eval_v2_docs10.json]"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--cal", required=True)
ap.add_argument("--shaping", required=True)
ap.add_argument("--lexicon-dir", default=None)
ap.add_argument("--docs-eval", default=None)
a = ap.parse_args()
cal = json.load(open(a.cal))
sh = json.load(open(a.shaping))
L = cal["languages"]
ev = json.load(open(a.docs_eval))["languages"] if a.docs_eval else {}
rows = []
for iso, b in sh.items():
    c = (b or {}).get("content") if isinstance(b, dict) else None
    if not c or c.get("mode") not in ("consensus", "blend"):
        continue
    rows.append((c["mode"], iso))
rows.sort()
print("| lang | mode | entity num / name | copy | stop-word profile (doc-level AUC stuff) | padding / bigram order (doc-level AUC stuff) | lexicon weight | stuffing covered by |\n|---|---|---|---|---|---|---|---|")
for mode, iso in rows:
    b = L.get(iso) or {}
    e = b.get("entity2") or {}
    def ch(c):
        x = e.get(c)
        return "-" if not x else (f"{x['w']:.2f}" if x["enabled"] else "off")
    cp = b.get("copy")
    cps = "-" if not cp else (f"t {cp['t_w']:.2f}" if cp["enabled"] else "off")
    fu = b.get("func")
    au = (ev.get(iso) or {}).get("auc") or {}
    fs = "-" if not fu else (f"on ({au.get('stuff', {}).get('func', float('nan')):.2f})" if fu["enabled"] else "off")
    pd = b.get("pad")
    gain_ok = bool(pd and pd.get("has_bigrams") and pd["enabled"] and (pd["t_gn"][-1] or 0) > 0)
    ps = "-" if not pd else (f"gain on ({au.get('stuff', {}).get('pad', float('nan')):.2f})" if gain_ok else ("ttr only" if pd["enabled"] else "off"))
    lw = "-"
    if a.lexicon_dir and (b.get("lexicon") or {}).get("enabled"):
        lw = f"{min(0.25, json.load(open(os.path.join(a.lexicon_dir, iso + '.json')))['meta'].get('weight', 0)):.2f}"
    cov = []
    if fu and fu["enabled"]:
        cov.append("stop-word")
    if gain_ok:
        cov.append("bigram order")
    if lw != "-":
        cov.append("lexicon density")
    print(f"| {iso} | {mode} | {ch('num')} / {ch('name')} | {cps} | {fs} | {ps} | {lw} | {', '.join(cov) or 'nothing (QE / consensus only)'} |")
