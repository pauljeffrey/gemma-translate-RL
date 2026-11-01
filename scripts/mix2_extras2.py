#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Variant B_v2: pick R10's extra windows per licence-class group under explicit per-language TOKEN BUDGETS (no window used twice, sources balanced inside a language).

    python scripts/mix2_extras2.py --plan plan.json --out extras.jsonl [--r10 /workspace/jobs/r10/sft_extra_v1]
plan.json = {"commercial_ok": {"tir": 2000000, ...}, "share_alike": {...}, "owner_code_switch": {...}, "NC_noncommercial": {...}}   (budget = prompt + answer Gemma tokens)
Reads <r10>/<group>/windows.parquet (TRAIN split rows only); `EXCLUDED_unknown_licence` is refused. Output rows: english, target, iso, n_tokens, source_dataset, licence, licence_class, synthetic, origin = "r10/<group>"."""
import argparse, collections, json, os, random
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--r10", default="/workspace/jobs/r10/sft_extra_v1")
ap.add_argument("--max-window-tokens", type=int, default=8192)
ap.add_argument("--seed", type=int, default=20261022)
a = ap.parse_args()
plan = json.load(open(a.plan))
rng = random.Random(a.seed)
out, summ = [], {}
for group, budgets in plan.items():
    if "unknown" in group.lower():
        raise SystemExit("EXCLUDED_unknown_licence must never be merged")
    rows = pq.read_table(os.path.join(a.r10, group, "windows.parquet")).to_pylist()
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("split", "train") not in ("train", None):
            continue
        n = int(r["total_tokens"])
        if 0 < n <= a.max_window_tokens and r["iso"] in budgets:
            by[r["iso"]].append({"iso": r["iso"], "english": r["english"], "target": r["target"], "n_tokens": n, "source_dataset": str(r.get("source_dataset")), "licence": str(r.get("licence")),
                                 "licence_class": r.get("licence_class") or "unknown", "synthetic": bool(r.get("synthetic_en")), "origin": f"r10/{group}"})
    for iso, bud in budgets.items():
        pool = by.get(iso, [])
        bysrc = collections.defaultdict(list)
        rng.shuffle(pool)
        for r in pool:
            bysrc[r["source_dataset"]].append(r)
        chosen, used = [], 0
        while any(bysrc.values()) and used < bud:
            for s in sorted(bysrc):
                if bysrc[s]:
                    r = bysrc[s].pop()
                    if used + r["n_tokens"] <= bud:
                        chosen.append(r); used += r["n_tokens"]
        out += chosen
        summ.setdefault(group, {})[iso] = {"budget": bud, "available": sum(r["n_tokens"] for r in pool), "chosen_windows": len(chosen), "chosen_tokens": used,
                                          "synthetic_tokens": sum(r["n_tokens"] for r in chosen if r["synthetic"]),
                                          "by_source": dict(collections.Counter({s: sum(r["n_tokens"] for r in chosen if r["source_dataset"] == s) for s in {r["source_dataset"] for r in chosen}}))}
with open(a.out, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(summ, open(a.out.replace(".jsonl", "_summary.json"), "w"), indent=1)
print(len(out), "windows,", sum(r["n_tokens"] for r in out), "tokens")
