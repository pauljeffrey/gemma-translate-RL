# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-65 experiment plan arithmetic: what difference in chrF++ between a static-weight and an adaptive-weight language can a run actually detect?

Inputs: per_language.csv + regression.json of scripts/lang_weight_evidence.py. Everything is empirical: the spread of SFT->RL gains between SIMILAR languages (matched on content mode and SFT level) is the
noise floor of any between-language comparison; the paired-bootstrap spread of a chrF++ delta on n sentences is the noise floor of any within-language comparison.

    python scripts/lang_weight_experiment_power.py --evidence reports/data/adaptive_language_weights_2026-10-22 [--out .../power.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

Z = 2.8          # (1.96 + 0.84): two-sided alpha 0.05, power 0.80


def matched_pairs(rows, tol=4.0):
    """Greedy pairs: same content mode and script, SFT chrF++ within `tol`, closest first."""
    used, pairs = set(), []
    cand = sorted(rows, key=lambda r: r["sft"])
    for i, a in enumerate(cand):
        if a["iso"] in used:
            continue
        best = None
        for b in cand[i + 1:]:
            if b["iso"] in used or b["mode"] != a["mode"] or b["script"] != a["script"]:
                continue
            d = abs(b["sft"] - a["sft"])
            if d <= tol and (best is None or d < best[0]):
                best = (d, b)
        if best:
            used.update([a["iso"], best[1]["iso"]])
            pairs.append((a, best[1]))
    return pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = list(csv.DictReader(open(os.path.join(a.evidence, "per_language.csv"))))
    for r in rows:
        for k in ("sft", "gain", "se"):
            r[k] = float(r[k])
    reg = json.load(open(os.path.join(a.evidence, "regression.json")))
    sd_sent = reg["paired_delta_sd_per_sentence"]["median"]
    out = {"paired_delta_sd_per_sentence": sd_sent}
    # (1) within-language: smallest detectable change of one language's chrF++ between two checkpoints / two arms on n dev sentences
    out["mde_one_language_two_models"] = {str(n): round(Z * sd_sent / math.sqrt(n), 2) for n in (40, 60, 100, 200, 400, 843, 1012, 1997)}
    out["sentences_needed_for_delta"] = {str(d): int(math.ceil((Z * sd_sent / d) ** 2)) for d in (0.5, 1.0, 2.0, 3.0, 5.0)}
    # (2) between matched languages in ONE run (treated = adaptive, control = static)
    pairs = matched_pairs(rows)
    diffs = np.array([p[0]["gain"] - p[1]["gain"] for p in pairs])
    sd_d = float(np.std(diffs, ddof=1)) if len(diffs) > 2 else float("nan")
    out["matched_pairs"] = {"n_pairs": len(pairs), "sd_of_gain_difference_within_pair": sd_d, "mean_abs_sft_gap": float(np.mean([abs(p[0]["sft"] - p[1]["sft"]) for p in pairs])),
                            "pairs": [[p[0]["iso"], p[1]["iso"], round(p[0]["sft"], 1), round(p[1]["sft"], 1), p[0]["mode"]] for p in pairs]}
    out["mde_matched_pairs"] = {str(P): round(Z * sd_d / math.sqrt(P), 2) for P in (6, 8, 10, 12, 16, 20, 30)}
    # (3) what effect is plausible? the pooled (confounded) estimate says ~1.4 chrF++ per doubling of RL samples (CI includes 0): a +-25 % weight change is 0.32 doublings
    est = reg["language_level"]["C_full_2c"]["ols"]["coef"]["log2_c2c"]
    out["plausible_effects"] = {"pooled_coef_per_doubling": est, "per_weight_change": {str(f): {"doublings": round(math.log2(f), 3), "chrf_at_estimate": round(est["est"] * math.log2(f), 2),
                                                                                                    "chrf_at_upper_ci": round(est["hi"] * math.log2(f), 2)} for f in (0.75, 1.25, 1.5, 2.0, 4.0)}}
    # (4) pairs needed to detect an effect (80 % power)
    out["pairs_needed"] = {str(d): int(math.ceil((Z * sd_d / d) ** 2)) for d in (0.5, 1.0, 1.5, 2.0, 3.0)}
    # (5) cost: one 800-step run at the measured 209 s/step
    out["cost"] = {"hours_per_800_steps": round(800 * 209 / 3600.0, 1), "hours_per_200_steps": round(200 * 209 / 3600.0, 1)}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
