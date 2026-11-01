# SPDX-License-Identifier: AGPL-3.0-or-later
"""Aggregator comparison on the chunk rewards that reward_run.py kept (ensemble channel, penalties included): paired / robust AUC of each aggregator on the corruptions of sentfix.
python agg_from_rw.py rw_items.jsonl [--g0 0.9]"""
import json, math, sys, statistics
from collections import defaultdict

sys.path.insert(0, "/workspace/jobs/r8/gemma_mt_r8")
from gemma_mt.rl.longdoc import weighted_tail_mean, weighted_quantile  # noqa: E402

CORR = ["C:drop1", "C:drop3", "C:drop10", "C:dup1", "C:dup3", "C:loop_tail_sent", "C:loop_tail_word", "C:trunc80", "C:trunc50", "C:repl10", "C:repl30"]
DAMAGE = {"sf": 0.0, "C:drop1": 0.02, "C:drop3": 0.05, "C:drop10": 0.10, "C:trunc80": 0.20, "C:repl10": 0.10, "C:repl30": 0.30, "C:trunc50": 0.50}


def variants(ch, g0):
    v = [c[0] for c in ch]
    w = [c[1] for c in ch]
    tot = sum(w)
    m = sum(a * b for a, b in zip(v, w)) / tot
    out = {"mean": m}
    if len(v) < 8:
        for k in ("cvar_blend", "ex_cvar_a1", "ex_cvar_a2", "ex_p25_a1", "ex_min_a0.1", "ex_p10_a1"):
            out[k] = m
        return out
    cv = weighted_tail_mean(v, w, 0.25)
    out["cvar_blend"] = 0.5 * m + 0.5 * cv
    out["ex_cvar_a1"] = m - 1.0 * max(0.0, (m - cv) - g0)
    out["ex_cvar_a2"] = m - 2.0 * max(0.0, (m - cv) - g0)
    p25 = weighted_quantile(v, w, 0.25)
    out["ex_p25_a1"] = m - 1.0 * max(0.0, (m - p25) - g0)
    p10 = weighted_quantile(v, w, 0.10)
    out["ex_p10_a1"] = m - 1.0 * max(0.0, (m - p10) - (g0 + 0.4))
    out["ex_min_a0.1"] = m - 0.1 * max(0.0, (m - min(v)) - (g0 + 1.5))
    return out


def auc(pairs):
    return sum(1.0 if a > b else (0.5 if a == b else 0.0) for a, b in pairs) / max(len(pairs), 1)


def main():
    fn = sys.argv[1]
    g0 = float(sys.argv[sys.argv.index("--g0") + 1]) if "--g0" in sys.argv else 0.9
    rows = [json.loads(l) for l in open(fn)]
    D = defaultdict(dict)
    for r in rows:
        if r["chunks"] and r["status"] == "ok":
            D[(r["doc_id"], r["iso"], r["bin"])][r["cond"]] = variants(r["chunks"], g0)
        elif r["status"] != "ok":
            D[(r["doc_id"], r["iso"], r["bin"])][r["cond"]] = {k: -2.0 for k in variants([(0.0, 1.0)] * 9, g0)}
    names = list(variants([(0.0, 1.0)] * 9, g0))
    print("g0", g0, "| paired AUC / robust AUC (vs nllb & winfix of the same document), bins b1024-b4096 pooled")
    print("corruption".ljust(18) + "".join(n.ljust(15) for n in names))
    for c in CORR:
        line = c[2:].ljust(18)
        for n in names:
            pa, ro = [], []
            for (doc, iso, b), cs in D.items():
                if b < 1024 or "sf" not in cs or c not in cs:
                    continue
                pa.append((cs["sf"][n], cs[c][n]))
                for g in ("nllb", "S:winfix"):
                    if g in cs:
                        ro.append((cs[g][n], cs[c][n]))
            line += f"{auc(pa):.2f}/{auc(ro):.2f}".ljust(15)
        print(line)
    # human-like: does the aggregate move good systems? mean shift of nllb/winfix/sf relative to mean
    print("shift of the good systems vs mean aggregate (b1024+):")
    for g in ("sf", "nllb", "S:winfix"):
        line = g.ljust(18)
        for n in names:
            d = [cs[g][n] - cs[g]["mean"] for (doc, iso, b), cs in D.items() if b >= 1024 and g in cs]
            line += f"{statistics.mean(d):+.2f}".ljust(15)
        print(line)
    # damage rho
    def spear(x, y):
        def rk(a):
            o = sorted(range(len(a)), key=lambda i: a[i]); r = [0.0] * len(a); i = 0
            while i < len(o):
                j = i
                while j + 1 < len(o) and a[o[j + 1]] == a[o[i]]:
                    j += 1
                for k in range(i, j + 1):
                    r[o[k]] = (i + j) / 2.0
                i = j + 1
            return r
        rx, ry = rk(x), rk(y)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / den if den else float("nan")
    line = "damage rho".ljust(18)
    for n in names:
        v = [spear([cs[c][n] for c in DAMAGE], [-DAMAGE[c] for c in DAMAGE]) for (doc, iso, b), cs in D.items() if b >= 1024 and all(c in cs for c in DAMAGE)]
        line += f"{statistics.mean(v):.2f}".ljust(15)
    print(line)


main()
