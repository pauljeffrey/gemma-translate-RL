#!/usr/bin/env python3
"""Scale the SFT-window sample hits to the full SFT text PER LANGUAGE (the sample is ~780-2,600 windows per language, not proportional to size).
Inputs: reports/data/contamination_audit_2026-10-26/sft_pool_rescan_windows.csv (hits), r10 pool counts (`_counts.json`, read from the Hub backup),
reports/gemma_sft_training_plan_2026-10-14.md (table `train samples` per language). Poisson 95% interval by the exact (Garwood) method.
Output: sft_pool_scaled_estimate.csv
"""
import csv, json, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from contam_audit_common import OUT, REPO, hf_token
from huggingface_hub import HfFileSystem
from math import exp, lgamma, log


def poisson_ci(k, alpha=0.05):
    # Garwood via bisection on the Poisson cdf
    def cdf(kk, mu):
        s = 0.0
        for i in range(kk + 1):
            s += exp(-mu + i * log(mu) - lgamma(i + 1)) if mu > 0 else (1.0 if i == 0 else 0.0)
        return s
    lo = 0.0
    if k > 0:
        a, b = 0.0, k + 10 * (k ** 0.5) + 10
        for _ in range(100):
            m = (a + b) / 2
            (a, b) = (m, b) if 1 - cdf(k - 1, m) < alpha / 2 else (a, m)
        lo = (a + b) / 2
    a, b = 0.0, k + 10 * (k ** 0.5) + 10
    for _ in range(100):
        m = (a + b) / 2
        (a, b) = (m, b) if cdf(k, m) > alpha / 2 else (a, m)
    return lo, (a + b) / 2


plan = open(os.path.join(REPO, "reports", "gemma_sft_training_plan_2026-10-14.md"), encoding="utf-8").read()
train = {}
for m in re.finditer(r"^\| (\w{3}) \| [^|]+ \| [\d,]+ / [\d,]+ / [\d,]+ / [\d,]+ \| ([\d,]+) \|", plan, re.M):
    train[m.group(1)] = int(m.group(2).replace(",", ""))
fs = HfFileSystem(token=hf_token())
counts = json.loads(fs.cat(("datasets/" + os.environ['HF_ORG_LEGACY'] + "/sabiyarn-box-backup-r10-weak-language-data/r10/pool/_counts.json")))
hits = Counter(r["iso"] for r in csv.DictReader(open(os.path.join(OUT, "sft_pool_rescan_windows.csv"))))
rows = []
tot = 0.0
for iso, n in counts.items():
    if iso not in train:
        continue
    scale = train[iso] / n
    k = hits.get(iso, 0)
    lo, hi = poisson_ci(k)
    rows.append({"iso": iso, "sft_train_windows(plan table, samples)": train[iso], "pool_windows": n, "scale": round(scale, 1), "hit_windows": k,
                 "est_windows": round(k * scale), "ci_lo": round(lo * scale), "ci_hi": round(hi * scale)})
    tot += k * scale
rows.sort(key=lambda r: -r["est_windows"])
with open(os.path.join(OUT, "sft_pool_scaled_estimate.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("languages with a plan row:", len(rows), "total est windows with FLORES English:", round(tot))
for r in rows[:8]:
    print(r)
