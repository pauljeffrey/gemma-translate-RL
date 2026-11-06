#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-editor: WHERE does the editor help or hurt? Sentence-level chrF++ change by draft-quality bin, by source length and by draft failure type (loop / unfinished / wrong-length).

    python scripts/post_editor_analyze.py --edited EV/greedy/edited_predictions.jsonl --out EV/where.csv
"""
import argparse, csv, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--edited", required=True); ap.add_argument("--out", required=True); ap.add_argument("--benchmarks", nargs="*", default=None)
    a = ap.parse_args()
    import numpy as np
    from gemma_mt.post_edit import sentence_chrf
    from gemma_mt.rl.guards import is_repetitive, is_self_repetitive
    rows = [json.loads(l) for l in open(a.edited, encoding="utf-8") if l.strip()]
    if a.benchmarks: rows = [r for r in rows if r["benchmark"] in a.benchmarks]
    for r in rows:
        r["_d"] = sentence_chrf(r["draft"], r["reference"]); r["_e"] = sentence_chrf(r["hyp"], r["reference"])
        r["_loop"] = bool(r["draft"].strip()) and (is_repetitive(r["draft"]) or is_self_repetitive(r["draft"]))
        r["_len"] = len(r["english"].split())
    groups = {
        "draft chrF 0-10": lambda r: r["_d"] < 10, "draft chrF 10-20": lambda r: 10 <= r["_d"] < 20, "draft chrF 20-30": lambda r: 20 <= r["_d"] < 30, "draft chrF 30-40": lambda r: 30 <= r["_d"] < 40,
        "draft chrF 40-50": lambda r: 40 <= r["_d"] < 50, "draft chrF 50-60": lambda r: 50 <= r["_d"] < 60, "draft chrF 60-80": lambda r: 60 <= r["_d"] < 80, "draft chrF 80-100": lambda r: r["_d"] >= 80,
        "draft is a loop": lambda r: r["_loop"], "draft unfinished": lambda r: not r.get("finished_draft", True),
        "source <= 10 words": lambda r: r["_len"] <= 10, "source 11-25 words": lambda r: 11 <= r["_len"] <= 25, "source > 25 words": lambda r: r["_len"] > 25, "all": lambda r: True}
    out = []
    for name, f in groups.items():
        g = [r for r in rows if f(r)]
        if not g: continue
        d = np.array([r["_e"] - r["_d"] for r in g]); ch = np.array([r["edit_rate"] > 0 for r in g])
        out.append({"group": name, "n": len(g), "share_changed": round(float(ch.mean()), 3), "mean_draft_chrf": round(float(np.mean([r["_d"] for r in g])), 2), "mean_edited_chrf": round(float(np.mean([r["_e"] for r in g])), 2),
                    "mean_delta": round(float(d.mean()), 2), "share_better": round(float((d > 0.5).mean()), 3), "share_worse": round(float((d < -0.5).mean()), 3),
                    "mean_delta_when_changed": round(float(d[ch].mean()), 2) if ch.any() else ""})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    for o in out: print(o)


if __name__ == "__main__":
    main()
