"""R12: markdown tables of the document-level results (local, stdlib). usage: doc_tables.py <whole.json> <repair.json> <unit_repair.json> <out.json>"""
import json, sys
w = json.load(open(sys.argv[1])); r = json.load(open(sys.argv[2])) if sys.argv[2] != "-" else None
u = json.load(open(sys.argv[3])) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
LAB = {"greedy": "greedy, whole document", "t03": "T 0.3 (R6)", "t05": "T 0.5 (R6)", "t07": "T 0.7 (R6)", "t10": "T 1.0 (R6)", "bon4": "best-of-4 T 0.7, chrF++ consensus (R6)", "rep105": "greedy + repetition_penalty 1.05 (R6)", "freq005": "greedy + frequency_penalty 0.05 (R6)",
       "sentfix": "sentence-by-sentence, greedy + R6 repair (best-of-4 T 0.7)", "greedy_sent": "sentence-by-sentence, greedy, no repair (R6)", "greedy_sent_r12": "sentence-by-sentence, greedy, no repair", "sentfix_r12": "R6 repair re-assembled (best-of-4 T 0.7)"}
def md(rows, header): return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(str(x) for x in rr) + " |" for rr in rows])
def lab(n): return LAB.get(n, f"`{n}`")
def core_tables(t):
    names = t["names"]; base = t["base"]; out = {}
    rows = []
    for n in names:
        row = [lab(n)]
        for b in ("256", "512", "1024", "2048"):
            c = t["core"].get(b, {}).get(n)
            if not c: row.append("-"); continue
            d = c["d_chrf_vs_base"]; sig = "*" if (c["d_lo"] > 0 or c["d_hi"] < 0) else ""
            row.append(f"{c['chrf']:.1f} ({d:+.1f}{sig})" if n != base else f"{c['chrf']:.1f}")
        rows.append(row)
    out["chrf"] = md(rows, ["config", "b256", "b512", "b1024", "b2048"])
    rows = []
    for n in names:
        row = [lab(n)]
        for b in ("512", "1024", "2048"):
            f = t["fail_core"][n].get(b) or t["fail_core"][n].get(int(b)) if isinstance(t["fail_core"][n], dict) else None
            if not f: row += ["-", "-", "-", "-"]; continue
            row += [f"{f['finished']*100:.0f}", f"{f['loop']*100:.0f}", f"{f['cov_sents']*100:.0f}", f"{f['length_ratio_med']:.2f}"]
        rows.append(row)
    out["fail"] = md(rows, ["config"] + [f"{b} {k}" for b in ("b512", "b1024", "b2048") for k in ("fin %", "loop %", "cov %", "len")])
    rows = []
    for n in names:
        row = [lab(n)]
        for hb in ("sm", "st512", "st1024", "st2048"):
            c = t["human"].get(hb, {}).get(n)
            if not c: row.append("-"); continue
            d = c["d_chrf_vs_base"]; sig = "*" if (c["d_lo"] > 0 or c["d_hi"] < 0) else ""
            row.append(f"{c['chrf']:.1f} ({d:+.1f}{sig})" if n != base else f"{c['chrf']:.1f}")
        rows.append(row)
    out["human"] = md(rows, ["config", "sm (<=680 tok)", "st512", "st1024", "st2048"])
    return out
res = {}
c = core_tables(w); res["doc_whole"] = c["chrf"] + "\n\nFailure metrics (finished = end token generated, loop = repeated word cycle the reference does not have, cov = sentence coverage, len = median hypothesis/reference length):\n\n" + c["fail"]; res["doc_human"] = c["human"]
if r:
    keep = ["greedy_sent_r12", "sentfix_r12", "fix_T0.5_n4", "fix_T0.7_n8", "fix_T0.7_mp0.05_n8", "fix_T1.0_mp0.05_n4", "fix_T1.0_mp0.05_n8", "fix_T1.0_n4", "fix_T0_rp1.1", "greedy"]
    r["names"] = [n for n in r["names"] if n in keep or n == r["base"]]
    c2 = core_tables(r); res["doc_repair"] = c2["chrf"] + "\n\nFailure metrics:\n\n" + c2["fail"] + "\n\nHuman references:\n\n" + c2["human"]
if u:
    rows = [[k, v["n"], v.get("units_covered", ""), f"{v['finished']*100:.1f}", f"{v['loop_or_unfinished']*100:.1f}", "-" if v["chrf_vs_nllb"] is None else f"{v['chrf_vs_nllb']:.1f}", "-" if v["len_ratio"] is None else f"{v['len_ratio']:.2f}", f"{v.get('any_candidate_finished', float('nan'))*100:.0f}" if "any_candidate_finished" in v else "-", f"{v.get('mean_cand_finished', float('nan'))*100:.0f}" if "mean_cand_finished" in v else "-"] for k, v in u.items()]
    res["doc_units"] = md(rows, ["repair of the unfinished units", "units", "re-decoded", "finished %", "loop or unfinished %", "chrF++ vs NLLB (units with a reference)", "len ratio", "units with >= 1 finished candidate %", "candidates finished %"])
json.dump(res, open(sys.argv[-1], "w"))
