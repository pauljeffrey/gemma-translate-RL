# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pivot of align_study.json: f1 (recall / precision) per corruption x aligner, pooled over the three sets, for the given bins (0 = all bins pooled)."""
import json, sys
from collections import defaultdict

rows = json.load(open(sys.argv[1]))
bins = [int(x) for x in sys.argv[2:]] or [0]
for b in bins:
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    for r in rows:
        if r["bin"] != b:
            continue
        a = acc[(r["corruption"], r["aligner"])]
        w = r["n"]
        a[0] += r["recall"] * w; a[1] += r["precision"] * w; a[2] += r["f1"] * w; a[3] += w; a[4] += r["ms"] * w
    cors = sorted({k[0] for k in acc}, key=lambda c: ["none", "drop1", "drop3", "drop10", "dup1", "dup3", "loop_tail_sent", "loop_tail_word", "trunc80", "trunc50", "repl10", "repl30"].index(c))
    als = sorted({k[1] for k in acc})
    print(f"\nbin {b or 'all'}: f1 (recall/precision), ms/doc")
    print("corruption".ljust(16) + "".join(a.ljust(30) for a in als))
    for c in cors:
        line = c.ljust(16)
        for a in als:
            v = acc.get((c, a))
            line += (f"{v[2]/v[3]:.3f} ({v[0]/v[3]:.2f}/{v[1]/v[3]:.2f}) {v[4]/v[3]:5.1f}ms".ljust(30) if v else "-".ljust(30))
        print(line)
    for a in als:
        vs = [acc[(c, a)] for c in cors if (c, a) in acc]
        print(f"  mean f1 {a}: {sum(v[2]/v[3] for v in vs)/len(vs):.3f}")
