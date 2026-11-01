# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-65 experiment plan: matched language pairs for a within-run treated / control comparison, with a pre-registered balanced randomisation.

Candidates: RL languages that keep draw weight 1.0 in the final run and have FLORES+ devtest rows (the primary outcome set). Pairs: identical content mode and script, SFT chrF++ within 4 points
(greedy nearest). Assignment: all 2^P treated / control assignments are enumerated; the ones whose mean difference in PAST SFT->RL gain is <= 0.25 and in SFT level <= 0.5 chrF++ are kept (the
randomisation is restricted to balanced allocations: with 9 pairs an unrestricted coin flip gave a 1.5 chrF++ head start to the treated side in a trial draw), and ONE of them is chosen by the
smallest sha1('lw-exp-v1|' + the assignment string). Nothing here depends on future data.

    python scripts/lang_weight_experiment_pairs.py --evidence reports/adaptive_language_weights_2026-10-22 --out reports/adaptive_language_weights_2026-10-22/experiment_pairs.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys

import numpy as np

FINAL_BOOST = {"ish", "bin", "bbj", "tsc", "kon", "nus", "dag", "oke", "tir", "amh", "kbp", "kam", "bam", "mos", "umb", "kmb", "dyu", "kea", "luo", "kik", "orm", "fon", "yor", "vmw", "efi",
               "hau", "lug", "nso", "zul"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(__file__))
    from lang_weight_experiment_power import matched_pairs

    rows = list(csv.DictReader(open(os.path.join(a.evidence, "per_language.csv"))))
    for r in rows:
        for k in ("sft", "gain", "se"):
            r[k] = float(r[k])
    cand = [r for r in rows if r["iso"] not in FINAL_BOOST and "flores_plus" in r["benches"].split(",")]
    pairs = [tuple(sorted(p, key=lambda r: r["iso"])) for p in matched_pairs(cand, tol=4.0)]
    P = len(pairs)
    good = []
    for bits in itertools.product((0, 1), repeat=P):
        t = [p[b] for p, b in zip(pairs, bits)]
        c = [p[1 - b] for p, b in zip(pairs, bits)]
        dg = np.mean([x["gain"] for x in t]) - np.mean([x["gain"] for x in c])
        ds = np.mean([x["sft"] for x in t]) - np.mean([x["sft"] for x in c])
        if abs(dg) <= 0.25 and abs(ds) <= 0.5:
            good.append((hashlib.sha1(("lw-exp-v1|" + "".join(map(str, bits))).encode()).hexdigest(), bits, dg, ds))
    good.sort()
    h, bits, dg, ds = good[0]
    out = []
    for p, b in zip(pairs, bits):
        t, c = p[b], p[1 - b]
        out.append({"treated": t["iso"], "control": c["iso"], "mode": t["mode"], "sft": [round(t["sft"], 1), round(c["sft"], 1)], "gain_sft_to_rl_2c": [round(t["gain"], 1), round(c["gain"], 1)]})
    rms = math.sqrt(np.mean([(p[0]["gain"] - p[1]["gain"]) ** 2 for p in pairs]))
    res = {"rule": __doc__.split("\n\n")[1].strip(), "n_candidates": len(cand), "n_pairs": P, "balanced_assignments": len(good), "past_gain_diff_treated_minus_control": round(float(dg), 3),
           "sft_diff": round(float(ds), 3), "rms_pair_gain_difference": round(rms, 3), "pairs": out}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rule"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
