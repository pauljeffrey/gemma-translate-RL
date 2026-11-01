# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ranking validation of a reward variant from reward_run.py outputs (R8, task 3).

  python validate_v2.py --run v1=rw_v1_items.jsonl v2=rw_v2_items.jsonl --out validate.json [--human v1=rw_v1_human.jsonl v2=...]

Classes (per document x language x system), from R6's flags:
  A  sentfix (sentence-by-sentence + repair)        B  whole-document output that finished and has no loop (R6 loop detector)
  C  whole-document output that finished WITH a loop   D  whole-document output that did not finish (truncated / ran to the token limit)
  N  NLLB sentence-by-sentence   W  256-token windows (winfix)
For each bin and each ordered pair of classes: P(reward(X) > reward(Y)) (ties 0.5) over all item pairs, 95% CI by bootstrapping (document, language) clusters (400 resamples).
Also the mean reward per class and bin, and the share of each class with status != ok."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict


def cls(r):
    c = r["cond"]
    if c == "sf":
        return "A"
    if c == "nllb":
        return "N"
    if c.startswith("S:"):
        return "W" if c == "S:winfix" else None
    if c.startswith("W:"):
        if not r.get("finished", True):
            return "D"
        return "C" if r.get("r6_loop") else "B"
    if c.startswith("C:"):
        return c
    return c


def auc_cluster(rows_x, rows_y, n_boot=400, seed=0):
    """rows_*: [(cluster, reward)]. Point estimate and bootstrap CI over clusters."""
    def point(X, Y):
        if not X or not Y:
            return float("nan")
        s = 0.0
        ys = sorted(v for _, v in Y)
        import bisect

        for _, x in X:
            lo = bisect.bisect_left(ys, x)
            hi = bisect.bisect_right(ys, x)
            s += lo + 0.5 * (hi - lo)
        return s / (len(X) * len(Y))
    p = point(rows_x, rows_y)
    cl = sorted({c for c, _ in rows_x} | {c for c, _ in rows_y})
    bx, by = defaultdict(list), defaultdict(list)
    for c, v in rows_x:
        bx[c].append((c, v))
    for c, v in rows_y:
        by[c].append((c, v))
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        pick = [rng.choice(cl) for _ in cl]
        X = [t for c in pick for t in bx.get(c, [])]
        Y = [t for c in pick for t in by.get(c, [])]
        v = point(X, Y)
        if v == v:
            vals.append(v)
    vals.sort()
    ci = (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if vals else (float("nan"), float("nan"))
    return p, ci


PAIRS = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "C"), ("A", "D"), ("B", "D"), ("A", "N"), ("A", "W")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="name=path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins", nargs="*", type=int, default=[256, 512, 1024, 2048, 4096])
    a = ap.parse_args()
    res = {}
    for spec in a.run:
        name, path = spec.split("=", 1)
        rows = [json.loads(l) for l in open(path)]
        by = defaultdict(lambda: defaultdict(list))            # bin -> class -> [(cluster, reward)]
        stat = defaultdict(lambda: defaultdict(list))
        for r in rows:
            c = cls(r)
            if c is None:
                continue
            cluster = f"{r['doc_id']}|{r['iso']}"
            by[r["bin"]][c].append((cluster, r["reward"]))
            stat[r["bin"]][c].append(r["status"] != "ok")
        out = {"mean": {}, "not_ok": {}, "pairs": {}}
        for b in a.bins:
            out["mean"][b] = {c: sum(v for _, v in l) / len(l) for c, l in by[b].items() if l}
            out["not_ok"][b] = {c: sum(l) / len(l) for c, l in stat[b].items() if l}
            for x, y in PAIRS:
                if by[b].get(x) and by[b].get(y):
                    out["pairs"][f"b{b}:{x}>{y}"] = auc_cluster(by[b][x], by[b][y])
        res[name] = out
    json.dump(res, open(a.out, "w"), indent=1)
    for name, out in res.items():
        print(f"\n=== {name}: mean reward per class")
        cl = ["A", "N", "W", "B", "C", "D"]
        print("bin   " + "".join(c.rjust(8) for c in cl))
        for b, m in out["mean"].items():
            print(f"b{b:<5d}" + "".join((f"{m[c]:8.2f}" if c in m else "       -") for c in cl))
        print("share not ok:")
        for b, m in out["not_ok"].items():
            print(f"b{b:<5d}" + "".join((f"{m[c]:8.2f}" if c in m else "       -") for c in cl))
        print("P(X > Y) [95% CI]")
        for k, (p, ci) in out["pairs"].items():
            print(f"  {k:12s} {p:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")


if __name__ == "__main__":
    main()
