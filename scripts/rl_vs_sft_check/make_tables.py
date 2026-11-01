"""python make_tables.py results.json PRED_DIR OUT_tables.md   -> markdown tables + examples"""
import sys, json, os, unicodedata
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_lib import sent_chrf
R = json.load(open(sys.argv[1])); pdir = sys.argv[2]; out = sys.argv[3]
cks = R["checkpoints"]; LB = {"sft": "SFT", "rl1": "RL1", "b64": "2b-b64", "b256": "b256", "b512": "b512", "b1024": "b1024", "b2048": "b2048", "c2c": "2c-latest"}
lab = lambda n: LB.get(n, n)
L = []
def f1(x): return "-" if x is None or x != x else f"{x:.1f}"
def pc(x): return "-" if x is None or x != x else f"{100*x:.1f}"
def cell(m, ci, fmt=f1): return f"{fmt(m)} [{fmt(ci[0])}, {fmt(ci[1])}]"
GN = {"consensus": "Consensus-only languages (reward = agreement of the samples + guards, no QE)", "blend": "Blend languages (QE named on the card but unvalidated, 50% consensus)", "control": "Control languages (validated QE reward)"}
for g in ["consensus", "blend", "control"]:
    G = R["groups"].get(g)
    if not G: continue
    L.append(f"### {GN[g]}\n")
    for sub in ["all", "no_pair", "clean"]:
        if sub not in G: continue
        S = G[sub]
        L.append(f"chrF++ ({sub}), group mean over {S['n_langs']} languages: " + ", ".join(S["langs"]) + "\n")
        L.append("| checkpoint | mean chrF++ [95% CI] | delta vs SFT [95% CI] |\n|---|---|---|")
        for n in cks:
            d = "-" if n == cks[0] else cell(S["delta_vs_first"][n]["mean"], S["delta_vs_first"][n]["ci"])
            L.append(f"| {lab(n)} | {cell(S['mean'][n], S['ci'][n])} | {d} |")
        L.append("")
    M = G.get("metrics", {})
    L.append("Other metrics (group mean over languages, %; 95% CI by bootstrap over languages; human = same metric on the human references):\n")
    L.append("| metric | n lang | human | " + " | ".join(lab(n) for n in cks) + " |\n|---|---|---|" + "---|" * len(cks))
    names = [("finished", "finished (ended inside 320 tokens)", True), ("degenerate", "degenerate (unfinished / empty / loop)", True), ("copy", "copy of source (chrF++ vs source >= 75)", True),
             ("lid_strict", "GlotLID = target (strict)", True), ("lid_sib", "GlotLID = target or sibling", True), ("lid_en", "GlotLID = English", True),
             ("anchor_recall", "source-anchor recall (numbers+names)", True), ("boiler", "5-gram boilerplate (100 outputs)", True), ("boiler40", "5-gram boilerplate (40-source sample)", True)]
    for k, nm, p in names:
        if k not in M: continue
        e = M[k]; h = e.get("human_mean")
        L.append(f"| {nm} | {e['n_langs']} | {pc(h) if h is not None else '-'} | " + " | ".join(f"{pc(e['mean'][n])} [{pc(e['ci'][n][0])}, {pc(e['ci'][n][1])}]" for n in cks) + " |")
    L.append("")
    L.append("Source dependence (40 sources per language; mean sentence chrF++): own = output vs its own reference; cross-ref = output vs the reference of a DIFFERENT source; cross-hyp = output vs the output for a DIFFERENT source.\n")
    L.append("| quantity | n lang | " + " | ".join(lab(n) for n in cks) + " |\n|---|---|" + "---|" * len(cks))
    for k, nm in [("own", "own reference"), ("cross_ref", "different source's reference"), ("cross_hyp", "different source's output")]:
        if k in M: L.append(f"| {nm} | {M[k]['n_langs']} | " + " | ".join(f"{M[k]['mean'][n]:.1f} [{M[k]['ci'][n][0]:.1f}, {M[k]['ci'][n][1]:.1f}]" for n in cks) + " |")
    hum = [R["langs"][i]["human"].get("cross_ref_human") for i in R["langs"] if R["langs"][i]["group"] == g and R["langs"][i]["human"].get("cross_ref_human") is not None]
    if hum: L.append(f"\n(human reference vs the reference of a different source: {np.mean(hum):.1f} chrF++, {len(hum)} languages)")
    L.append("")
# per-language table
last = cks[-1]
L.append("## Per-language table\n")
for g in ["consensus", "blend", "control"]:
    ls = [i for i, v in R["langs"].items() if v["group"] == g]
    L.append(f"### {g}\n")
    hdr = "| lang | n (ref / no_pair / clean) | sets | " + " | ".join(f"chrF++ {lab(n)}" for n in cks) + f" | d({lab(last)}-SFT) all [CI] | d no_pair [CI] | LID% SFT -> {lab(last)} (human) | anchor% SFT -> {lab(last)} (human) | boiler% SFT -> {lab(last)} (human) | degen% SFT -> {lab(last)} |"
    L.append(hdr); L.append("|" + "---|" * (hdr.count("|") - 1))
    def key(i):
        d = R["langs"][i].get("delta", {}).get("all", {}).get(last)
        return d["mean"] if d else 999
    for i in sorted(ls, key=key):
        v = R["langs"][i]; m = v["models"]; h = v["human"]
        da = v.get("delta", {}).get("all", {}).get(last); dn = v.get("delta", {}).get("no_pair", {}).get(last)
        fmtd = lambda d: "-" if not d else f"{d['mean']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}]"
        sets = ",".join(f"{b}:{n}" for b, n in v["benches"].items())
        L.append(f"| {i} | {v['n']} / {v['n_no_pair']} / {v['n_clean']} | {sets} | " + " | ".join(f1(m[n].get("chrf_all")) for n in cks) +
                 f" | {fmtd(da)} | {fmtd(dn)} | {pc(m[cks[0]]['lid_sib'])} -> {pc(m[last]['lid_sib'])} ({pc(h.get('lid_sib'))}) | "
                 f"{pc(m[cks[0]]['anchor_recall'])} -> {pc(m[last]['anchor_recall'])} ({pc(h.get('anchor_recall'))}) | "
                 f"{pc(m[cks[0]]['boiler'])} -> {pc(m[last]['boiler'])} ({pc(h.get('boiler'))}) | {pc(m[cks[0]]['degenerate'])} -> {pc(m[last]['degenerate'])} |")
    L.append("")
open(out, "w", encoding="utf-8").write("\n".join(L))
print("tables written", out)
