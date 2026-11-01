# SPDX-License-Identifier: AGPL-3.0-or-later
"""GRPO groups (15 sampled T=1 + greedy LAST, here G-1 sampled + 1 greedy) on long documents: what the reward does to the advantages (R8, greedy-in-group assessment).

  python group_study.py --run v1=rw_v1_groups.jsonl v2=rw_v2_groups.jsonl --out group_study.json

Per bin and reward version: finished / loop shares of the sampled members and of the greedy member, reward mean / sd, the std inflation caused by the greedy outlier
(std over all members / std over the sampled ones), the greedy member's advantage before and after greedy_adv_scale, its share of the token-weighted advantage mass
sum |A_i| * n_tokens_i (loss_agg token) for scales 1 / 0.3 / 0.1 / 0, and the within-group Spearman between reward and completeness of the sampled members."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict

EPS = 1e-4


def spearman(x, y):
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
    if len(x) < 3:
        return None
    rx, ry = rk(x), rk(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / den if den else None


def completeness(r):
    ch = r.get("chunks") or []
    tot = sum(w for _, w in ch)
    if not ch or tot <= 0:
        return 0.0
    return sum(w for rw, w in ch if rw > -2.0 + 1e-9) / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for spec in a.run:
        name, path = spec.split("=", 1)
        groups = defaultdict(list)
        for l in open(path):
            r = json.loads(l)
            groups[r["rid"]].append(r)
        agg = defaultdict(lambda: defaultdict(list))
        for rid, mem in groups.items():
            g = [m for m in mem if m["kind"] == "greedy"]
            s = [m for m in mem if m["kind"] != "greedy"]
            if len(g) != 1 or len(s) < 2:
                continue
            g = g[0]
            b = g["bin"]
            A = agg[b]
            allr = [m["reward"] for m in s] + [g["reward"]]
            mean = sum(allr) / len(allr)
            sd_all = statistics.pstdev(allr)
            sd_s = statistics.pstdev([m["reward"] for m in s])
            A["s_fin"].append(sum(1 for m in s if m["status"] != "no_end_of_turn") / len(s))
            A["g_fin"].append(1.0 if g["status"] != "no_end_of_turn" else 0.0)
            A["g_bad"].append(1.0 if g["status"] != "ok" else 0.0)
            A["s_bad"].append(sum(1 for m in s if m["status"] != "ok") / len(s))
            A["s_mean"].append(sum(m["reward"] for m in s) / len(s))
            A["g_reward"].append(g["reward"])
            A["s_sd"].append(sd_s)
            A["sd_infl"].append(sd_all / (sd_s + 1e-9))
            A["s_tok"].append(sum(m["n_tokens"] for m in s) / len(s))
            A["g_tok"].append(g["n_tokens"])
            A["flat_all"].append(1.0 if sd_all < 1e-3 else 0.0)
            adv_g = (g["reward"] - mean) / (sd_all + EPS)
            A["adv_g_raw"].append(adv_g)
            adv_s = [(m["reward"] - mean) / (sd_all + EPS) for m in s]
            A["adv_s_abs"].append(sum(abs(x) for x in adv_s) / len(adv_s))
            for sc in (1.0, 0.3, 0.1, 0.0):
                mg = abs(adv_g) * sc * g["n_tokens"]
                ms = sum(abs(x) * m["n_tokens"] for x, m in zip(adv_s, s))
                A[f"share_g_{sc}"].append(mg / (mg + ms) if mg + ms > 0 else 0.0)
            rho = spearman([m["reward"] for m in s], [completeness(m) for m in s])
            if rho is not None:
                A["rho_complete"].append(rho)
            # advantage of the sampled members if the group statistics ignored the greedy member
            if sd_s > 1e-3:
                mean_s = sum(m["reward"] for m in s) / len(s)
                A["adv_s_abs_excl"].append(sum(abs((m["reward"] - mean_s) / (sd_s + EPS)) for m in s) / len(s))
            A["n"].append(1)
        out = {}
        for b, A in sorted(agg.items()):
            out[b] = {k: (sum(v) / len(v) if v else None) for k, v in A.items() if k != "n"}
            out[b]["n_groups"] = len(A["n"])
        res[name] = out
    json.dump(res, open(a.out, "w"), indent=1)
    keys = ["n_groups", "s_fin", "g_fin", "s_bad", "g_bad", "s_mean", "g_reward", "s_sd", "sd_infl", "s_tok", "g_tok", "adv_g_raw", "adv_s_abs", "adv_s_abs_excl",
            "share_g_1.0", "share_g_0.3", "share_g_0.1", "share_g_0.0", "rho_complete", "flat_all"]
    for name, out in res.items():
        print(f"\n=== {name}")
        print("metric".ljust(16) + "".join(f"b{b}".rjust(10) for b in out))
        for k in keys:
            print(k.ljust(16) + "".join((f"{out[b][k]:10.3f}" if out[b].get(k) is not None else "         -") for b in out))


if __name__ == "__main__":
    main()
