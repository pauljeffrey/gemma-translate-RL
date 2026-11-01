#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mixture v2, variant B: choose the extra windows of R10 (human / real-text data by licence class) and R14 (NLLB teacher distillation) under per-language TIER CAPS.

    python scripts/mix2_extras.py --r10 /workspace/jobs/r10/sft_extra_v1 --groups commercial_ok [--r14 /workspace/jobs/r14/<dir>] \
        --base /workspace/jobs/r11/mix_v2A --out /workspace/jobs/r11/work/extras_B.jsonl

Reads windows.parquet of every requested R10 group (columns iso, english, target, total_tokens, source_dataset, licence, licence_class, synthetic_en, split ...),
only TRAIN-split windows. Per language the token cap depends on the TIER (data-driven: the weak languages have ~150 k human-reference tokens in mix_v1, the strong ones 150-300 k, science languages 1.0-1.3 M):
    weak (blend + consensus-only languages, work/weak_langs.txt)   --cap-weak   tokens
    Ge'ez (amh, tir)                                               --cap-geez
    other                                                           --cap-other
inside a language: sources are balanced (no source above --max-source-share of the language's extra tokens), windows are drawn at random (seeded); windows whose English side is
SYNTHETIC (`synthetic_en`: real African text, machine-translated English) or whose target is a teacher output are limited to --max-synth-share of the language's TOTAL tokens (base + extras).
Output: jsonl rows {english, target, iso, source_dataset, licence, licence_class, synthetic, n_tokens} + a per-language summary json next to it.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random

import pyarrow.parquet as pq


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r10", default="/workspace/jobs/r10/sft_extra_v1")
    ap.add_argument("--groups", default="commercial_ok")
    ap.add_argument("--r14", default="", help="directory / parquet / jsonl files of R14 teacher data (columns iso, english, target ...)")
    ap.add_argument("--base", required=True, help="mix dir whose mixture_tables_v2.json gives the base tokens per language")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weak", default="/workspace/jobs/r11/work/weak_langs.txt")
    ap.add_argument("--cap-weak", type=int, default=300_000)
    ap.add_argument("--cap-geez", type=int, default=500_000)
    ap.add_argument("--cap-other", type=int, default=150_000)
    ap.add_argument("--cap-teacher", type=int, default=150_000, help="per language cap of R14 teacher tokens")
    ap.add_argument("--max-source-share", type=float, default=0.6)
    ap.add_argument("--max-synth-share", type=float, default=0.30)
    ap.add_argument("--max-window-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args(argv)
    weak = set(open(a.weak).read().strip().split(","))
    geez = {"amh", "tir"}
    tab = json.load(open(os.path.join(a.base, "mixture_tables_v2.json")))
    base = collections.Counter()
    for s, d in tab.items():
        for iso, v in d.items():
            base[iso] += v["tokens"]
    rows = []
    for g in [x for x in a.groups.split(",") if x]:
        p = os.path.join(a.r10, g, "windows.parquet")
        t = pq.read_table(p).to_pylist()
        for r in t:
            if r.get("split", "train") not in ("train", None):
                continue
            rows.append({"iso": r["iso"], "english": r["english"], "target": r["target"], "n_tokens": int(r["total_tokens"]), "source_dataset": r.get("source_dataset") if isinstance(r.get("source_dataset"), str) else str(r.get("source_dataset")),
                         "licence": r.get("licence") if isinstance(r.get("licence"), str) else str(r.get("licence")), "licence_class": r.get("licence_class") or g, "synthetic": bool(r.get("synthetic_en")), "origin": f"r10/{g}"})
    if a.r14:
        files = [a.r14] if os.path.isfile(a.r14) else sorted(glob.glob(os.path.join(a.r14, "*.parquet")) + glob.glob(os.path.join(a.r14, "*.jsonl")))
        for p in files:
            t = pq.read_table(p).to_pylist() if p.endswith(".parquet") else [json.loads(l) for l in open(p, encoding="utf-8")]
            for r in t:
                rows.append({"iso": r["iso"], "english": r["english"], "target": r["target"], "n_tokens": int(r.get("total_tokens") or r.get("n_tokens") or 0), "source_dataset": str(r.get("source_dataset", "r14")),
                             "licence": str(r.get("licence", "nllb-600m teacher output (CC-BY-NC model)")), "licence_class": r.get("licence_class", "nc_teacher"), "synthetic": True, "origin": "r14"})
    by = collections.defaultdict(list)
    for r in rows:
        if 0 < r["n_tokens"] <= a.max_window_tokens:
            by[r["iso"]].append(r)
    rng = random.Random(a.seed)
    out, summ = [], {}
    for iso, rs in sorted(by.items()):
        cap = a.cap_geez if iso in geez else (a.cap_weak if iso in weak else a.cap_other)
        teacher = [r for r in rs if r["origin"] == "r14"]
        real = [r for r in rs if r["origin"] != "r14" and not r["synthetic"]]
        synth = [r for r in rs if r["origin"] != "r14" and r["synthetic"]]
        chosen, used, per_src = [], 0, collections.Counter()

        def take(pool, budget, src_share=True):
            nonlocal used
            rng.shuffle(pool)
            got = 0
            # round-robin over the sources so that no source dominates
            bysrc = collections.defaultdict(list)
            for r in pool:
                bysrc[r["source_dataset"]].append(r)
            order = []
            while any(bysrc.values()):
                for s in sorted(bysrc):
                    if bysrc[s]:
                        order.append(bysrc[s].pop())
            for r in order:
                if got + r["n_tokens"] > budget:
                    continue
                if src_share and per_src[r["source_dataset"]] + r["n_tokens"] > a.max_source_share * cap and len(by_src_all(rs)) > 1:
                    continue
                chosen.append(r)
                got += r["n_tokens"]
                per_src[r["source_dataset"]] += r["n_tokens"]
            used += got
            return got

        def by_src_all(x):
            return {r["source_dataset"] for r in x}

        take(real, cap)
        # synthetic-English windows: at most max_synth_share of the language's total tokens (base + everything chosen)
        room = int(a.max_synth_share * (base.get(iso, 0) + used) / (1 - a.max_synth_share)) - 0
        take(synth, max(0, min(cap - used, room)))
        tcap = a.cap_teacher
        take(teacher, min(tcap, max(0, int(a.max_synth_share * (base.get(iso, 0) + used) / (1 - a.max_synth_share)) - sum(r["n_tokens"] for r in chosen if r["synthetic"]))), src_share=False)
        out += chosen
        summ[iso] = {"tier": "geez" if iso in geez else ("weak" if iso in weak else "other"), "cap": cap, "available_tokens": sum(r["n_tokens"] for r in rs), "chosen_windows": len(chosen),
                     "chosen_tokens": sum(r["n_tokens"] for r in chosen), "synthetic_tokens": sum(r["n_tokens"] for r in chosen if r["synthetic"]), "base_tokens": base.get(iso, 0),
                     "by_source": {k: v for k, v in collections.Counter({s: sum(r["n_tokens"] for r in chosen if r["source_dataset"] == s) for s in {r["source_dataset"] for r in chosen}}.items())},
                     "by_class": dict(collections.Counter({c: sum(r["n_tokens"] for r in chosen if r["licence_class"] == c) for c in {r["licence_class"] for r in chosen}}))}
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(summ, open(a.out.replace(".jsonl", "_summary.json"), "w"), indent=1)
    print(f"{len(out)} windows, {sum(r['n_tokens'] for r in out):,} tokens, {len(summ)} languages")


if __name__ == "__main__":
    main()
