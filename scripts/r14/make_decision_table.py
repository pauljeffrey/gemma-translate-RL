#!/usr/bin/env python3
"""R14: decision table NLLB (3.3B / 600M) vs our final_1433, from reports/data/nllb600m_per_language_benchmark.csv -> nllb600m_decision_table.csv + coverage table. Stdlib only."""
import csv, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = ROOT + "/reports/data"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT + "/gemma_mt")
from nllb_langmap import NLLB_MAP
from gemma_mt import languages as L
BENCH_ORDER = ["flores_plus", "ntrex", "mafand", "smol_sent", "salt"]
rows = list(csv.DictReader(open(DATA + "/nllb600m_per_language_benchmark.csv")))
f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else None
def cat(d, lo, hi):
    if d is None: return ""
    if d >= 5: c = "NLLB much better (>=+5)"
    elif d >= 2: c = "NLLB better (+2..+5)"
    elif d > -2: c = "similar (-2..+2)"
    else: c = "NLLB worse (<-2)"
    sig = "" if lo is None else (" [CI>0]" if lo > 0 else " [CI<0]" if hi < 0 else " [CI incl. 0]")
    return c + sig
by = {}
for r in rows:
    if r["subset"] in ("no_pair", "all"): by.setdefault((r["iso"], r["benchmark"], r["subset"]), r)
out = []
for iso in L.ALL_ISOS:
    code, kind, note = NLLB_MAP[iso]
    l = L.LANGUAGES[iso]
    grp = "qe_validated" if iso in L.QE_VALIDATED else "consensus_only" if l.rl_consensus else "blend" if l.rl_eligible else "sft_only"
    rec = {"iso": iso, "language": l.name, "group": grp, "nllb_code": code or "", "nllb_coverage": kind, "note": note}
    pick = None
    for b in BENCH_ORDER:
        r = by.get((iso, b, "no_pair")) or by.get((iso, b, "all"))
        if r and f(r, "nllb_n33_beam4_chrf") is not None: pick = (b, r); break
    if pick:
        b, r = pick
        rec.update({"benchmark": b, "subset": r["subset"], "n": r["n"], "ours_chrf": f(r, "ours_chrf"), "nllb33_chrf": f(r, "nllb_n33_beam4_chrf"), "nllb600_chrf": f(r, "nllb_n600_beam4_chrf"),
                    "d33": f(r, "nllb_n33_beam4_d_chrf"), "d33_lo": f(r, "nllb_n33_beam4_d_chrf_lo"), "d33_hi": f(r, "nllb_n33_beam4_d_chrf_hi"),
                    "d600": f(r, "nllb_n600_beam4_d_chrf"), "d600_lo": f(r, "nllb_n600_beam4_d_chrf_lo"), "d600_hi": f(r, "nllb_n600_beam4_d_chrf_hi"),
                    "d33_spbleu": f(r, "nllb_n33_beam4_d_spbleu"), "d600_spbleu": f(r, "nllb_n600_beam4_d_spbleu"),
                    "ours_spbleu": f(r, "ours_spbleu"), "nllb33_spbleu": f(r, "nllb_n33_beam4_spbleu"), "nllb600_spbleu": f(r, "nllb_n600_beam4_spbleu"),
                    "nllb33_greedy_chrf": f(r, "nllb_n33_greedy_chrf"), "nllb600_greedy_chrf": f(r, "nllb_n600_greedy_chrf")})
        rec["category_3.3B"] = cat(rec["d33"], rec["d33_lo"], rec["d33_hi"])
        rec["category_600M"] = cat(rec["d600"], rec["d600_lo"], rec["d600_hi"])
        d, lo = rec["d33"], rec["d33_lo"]
        rec["useful_as_reward_signal_3.3B"] = "yes" if d >= 2 and lo > 0 else "no" if d <= -2 or (d < 0 and rec["d33_hi"] < 0) else "weak/uncertain"
        d6, lo6 = rec["d600"], rec["d600_lo"]
        rec["useful_as_reward_signal_600M"] = "not run" if d6 is None else "yes" if d6 >= 2 and lo6 > 0 else "no" if d6 <= -2 or (d6 < 0 and rec["d600_hi"] < 0) else "weak/uncertain"
    else:
        rec["category_3.3B"] = rec["category_600M"] = "no benchmark" if kind != "none" else "NOT COVERED by NLLB"
        rec["useful_as_reward_signal_3.3B"] = rec["useful_as_reward_signal_600M"] = "unknown (no human reference)" if kind != "none" else "no (not covered)"
    out.append(rec)
fields = ["iso", "language", "group", "nllb_code", "nllb_coverage", "benchmark", "subset", "n", "ours_chrf", "nllb33_chrf", "nllb600_chrf", "d33", "d33_lo", "d33_hi", "d600", "d600_lo", "d600_hi",
          "ours_spbleu", "nllb33_spbleu", "nllb600_spbleu", "d33_spbleu", "d600_spbleu", "nllb33_greedy_chrf", "nllb600_greedy_chrf", "category_3.3B", "category_600M",
          "useful_as_reward_signal_3.3B", "useful_as_reward_signal_600M", "note"]
with open(DATA + "/nllb600m_decision_table.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
    for r in out: w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k in fields})
print(len(out), "languages;", sum(1 for r in out if r["nllb_coverage"] == "exact"), "exact,", sum(1 for r in out if r["nllb_coverage"] == "proxy"), "proxy,", sum(1 for r in out if r["nllb_coverage"] == "none"), "none")
