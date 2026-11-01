#!/usr/bin/env python3
"""R17: calibration v2 of the NLLB pseudo-reference signal = w_analyze.py's output + (1) a cap for languages where the human direct-assessment study (SSA-MTE, nllb_widening_blend_da.csv)
shows that a mixing weight above 0.25 stops helping, (2) the double-pull discount by the share of NLLB teacher tokens each language has in the C3 SFT mixture (reports/sft_mixture_v2_2026-10-22/C3/c3_vs_a2w.md).
usage: w_finalize_calibration.py IN_v2.json BLEND_DA.csv C3_TABLE.md OUT.json  (stdlib only)"""
import csv, json, re, sys
src, da_csv, c3_md, out = sys.argv[1:5]
cal = json.load(open(src))
# human-DA evidence: weight above which the blend with the QE ensemble no longer improves Spearman with SSA-MTE z (CI of the change includes 0 or below)
da_cap = {}
for r in csv.DictReader(open(da_csv)):
    ok = [w for w in (0.15, 0.25, 0.35, 0.5) if float(r[f"d_sp_w{w}_lo"]) > 0]
    da_cap[r["iso"]] = max(ok) if ok else 0.0
share = {}
for l in open(c3_md):
    m = re.match(r"\| ([a-z]{3}) \|.*\| ([\d.]+)% \| \+?[\d,]+ \|$", l.strip())
    if m: share[m.group(1)] = float(m.group(2)) / 100
CAP_WHEN_DA_WEAK = 0.25
for iso, v in cal.items():
    n = v["nllb"]; p = n["pseudoref"]
    w0 = n["weight"]; n["weight_v2_uncapped"] = w0
    if n["use"] and iso in da_cap and da_cap[iso] < 0.35:
        w0 = min(w0, CAP_WHEN_DA_WEAK); n["da_cap"] = CAP_WHEN_DA_WEAK
        n["da_note"] = f"human DA (SSA-MTE): blend gain over the QE ensemble is significant only up to w={da_cap[iso]}"
    sh = share.get(iso, 0.0)
    f_c3 = max(0.4, min(1.0, 1.0 - 0.7 * sh)) if sh > 0 else 1.0           # double pull: the more of the language's SFT tokens are NLLB output, the less the reward should pull toward NLLB again
    n["weight"] = round(w0, 3)
    n["teacher_share_C3"] = sh; n["double_pull_factor_C3"] = round(f_c3, 3); n["weight_C3"] = round(w0 * f_c3, 3)
    n["weight_if_teacher_sft_in_mixture"] = n["weight_C3"]
    rel = float(p["reliability"])
    if n["use"] and rel > 0: p["weight_factor"] = w0 / (0.35 * rel)         # keeps NllbSignal.weight() = 0.35 x reliability x weight_factor = w0 without any override
json.dump(cal, open(out, "w"), indent=1, ensure_ascii=False)
use = sorted(k for k, v in cal.items() if v["nllb"]["use"])
print(len(use), "languages in use"); print({k: (cal[k]["nllb"]["weight"], cal[k]["nllb"]["weight_C3"]) for k in use})
