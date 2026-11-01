#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-language enable table of calibration/anchors_v2.json (D-61 report): entity weights, copy thresholds, stop-word / padding blocks, lexicon weights. Markdown on stdout."""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--cal", required=True)
ap.add_argument("--lex-cal", default=None, help="anchors_v1_lex.json (lexicon blocks)")
ap.add_argument("--lexicon-dir", default=None)
ap.add_argument("--eval", default=None, help="eval_anchors_v2.json (fold-B FP / AUC columns)")
a = ap.parse_args()
cal = json.load(open(a.cal))
lex = json.load(open(a.lex_cal))["languages"] if a.lex_cal else {}
ev = json.load(open(a.eval))["languages"] if a.eval else {}
weak = set((cal.get("_meta") or {}).get("weak_languages") or [])
L = cal["languages"]


def f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


print("| lang | weak | entity num d / w | entity name d / w | copy t_w / t_c (FP cv, tail25 / tail50) | func (FP cv, held-out FP, AUC stuff / EN-copy / other-lang) | pad (FP cv, bigram table, AUC stuff / pad) | lexicon weight |")
print("|---|---|---|---|---|---|---|---|")
for iso in sorted(L):
    b = L[iso]
    e = b.get("entity2") or {}

    def ch(c):
        x = e.get(c)
        if not x:
            return "-"
        return f"{x['d']:.2f} / {x['w']:.2f}" + ("" if x["enabled"] else " OFF") + f" (n={x['n']})"
    c = b.get("copy")
    cs = "-" if not c else f"{c['t_w']:.2f} / {c['t_c']:.2f} ({c['fp_cv']:.3f}, {c['detect']['tail25']:.2f} / {c['detect']['tail50']:.2f})" + ("" if c["enabled"] else " OFF")
    fu = b.get("func")
    r = ev.get(iso) or {}
    au = r.get("auc") or {}
    fs = "-"
    if fu:
        fs = f"{fu['fp_cv']:.3f}, " + f(((r.get('fp') or {}).get('func') or {}).get('share'), 3) + ", " + " / ".join(f(au.get(n, {}).get("func")) for n in ("stuff", "en_copy", "other_lang")) + ("" if fu.get("enabled") else " OFF")
    pd = b.get("pad")
    ps = "-"
    if pd:
        ps = f"{pd['fp_cv']:.3f}, {pd.get('n_bigrams', 0)}{'' if pd.get('has_bigrams') else ' (ttr only)'}, " + " / ".join(f(au.get(n, {}).get("pad")) for n in ("stuff", "pad")) + ("" if pd.get("enabled") else " OFF")
    lw = "-"
    if iso in lex and (lex[iso].get("lexicon") or {}).get("enabled") and a.lexicon_dir:
        m = json.load(open(os.path.join(a.lexicon_dir, iso + ".json")))["meta"]
        lw = f"{min(0.25, m.get('weight', 0)):.2f}"
    print(f"| {iso} | {'w' if iso in weak else ''} | {ch('num')} | {ch('name')} | {cs} | {fs} | {ps} | {lw} |")
