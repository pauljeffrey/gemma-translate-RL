"""E-03: tables from evidence_c1/*.json (ev_analyse2.py output on the box, paired bootstrap) -> verdict_tables.md / verdict_tables.json"""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); E = H + "/evidence_c1/"
L = lambda f: json.load(open(E + f))
def fmt(t, k=2): return f"{t[2]:+.{k}f} [{t[3]:+.{k}f}, {t[4]:+.{k}f}]"
out = {}; md = []
main = L("evidence_mixed_vs_c6000.json")
pairs = {"mixed best": main, "stage A": L("evidence_stageA_vs_c6000.json"), "B1 best": L("evidence_b1_vs_c6000.json")}
md.append("## Headline (RL minus control, clean rows; chrF++ and AfriCOMET macro over cells; 95% paired bootstrap)\n")
md.append("| RL model | cells | control chrF++ | RL chrF++ | RL - control chrF++ | RL - control chrF (plain) | AfriCOMET cells | control AfriCOMET | RL AfriCOMET | RL - control AfriCOMET |\n|---|---|---|---|---|---|---|---|---|---|")
for n, d in pairs.items():
    a = d["macro"]["all"]["all_cells"]; out[n] = a
    md.append(f"| {n} | {a['pp'][5]} | {a['pp'][0]:.2f} | {a['pp'][1]:.2f} | {fmt(a['pp'])} | {fmt(a['c'])} | {a['com'][5]} | {a['com'][0]:.4f} | {a['com'][1]:.4f} | {fmt(a['com'],4)} |")
md.append("\n## By benchmark (mixed best minus control)\n\n| benchmark | cells | control | RL | RL - control chrF++ | AfriCOMET control | RL | diff |\n|---|---|---|---|---|---|---|---|")
for b, a in main["macro"]["all"].items():
    if b == "all_cells": continue
    md.append(f"| {b} | {a['pp'][5]} | {a['pp'][0]:.2f} | {a['pp'][1]:.2f} | {fmt(a['pp'])} | {a['com'][0]:.4f} | {a['com'][1]:.4f} | {fmt(a['com'],4)} |")
md.append("\n## Language panels (the 19 TranslatePsy languages vs the others)\n\n| RL model | panel | cells | control chrF++ | RL chrF++ | RL - control | AfriCOMET diff |\n|---|---|---|---|---|---|---|")
for n, f in (("mixed best", "mixed"), ("stage A", "stageA")):
    for pn in ("TP19", "OTHERS"):
        d = L(f"evidence_{f}_vs_c6000_{pn}.json"); a = d["macro"]["all"]["all_cells"]; out[f"{n} {pn}"] = a
        md.append(f"| {n} | {pn} | {a['pp'][5]} | {a['pp'][0]:.2f} | {a['pp'][1]:.2f} | {fmt(a['pp'])} | {fmt(a['com'],4)} (n={a['com'][5]}) |")
md.append("\n## SFT to control and SFT to RL (language-macro chrF++ per benchmark)\n\n| benchmark | SFT | control | control - SFT | mixed | mixed - SFT |\n|---|---|---|---|---|---|")
s_c = main["stages"]["sft->c1_s6000"]; s_m = main["stages"]["sft->mixed"]
tot = {"c": 0, "m": 0, "n": 0, "sft": 0}
for b in ("flores_plus", "ntrex", "mafand"):
    c = s_c[b]["pp"]; m = s_m[b]["pp"]; n = c[5]
    md.append(f"| {b} | {c[0]:.2f} | {c[1]:.2f} | {fmt(c)} | {m[1]:.2f} | {fmt(m)} |")
    tot["c"] += c[1] * n; tot["m"] += m[1] * n; tot["sft"] += c[0] * n; tot["n"] += n
sft, ctl, mix = tot["sft"] / tot["n"], tot["c"] / tot["n"], tot["m"] / tot["n"]
md.append(f"| all 93 cells | {sft:.2f} | {ctl:.2f} | {ctl-sft:+.2f} | {mix:.2f} | {mix-sft:+.2f} |")
out["gap_closed_chrfpp"] = (ctl - sft) / (mix - sft)
md.append(f"\nShare of the SFT-to-RL (mixed best) chrF++ gap closed by the control: {(ctl-sft)/(mix-sft):.1%} (arithmetic of the macros above; cell-count weighted).")
md.append("\n## Finished and loop rates (reward-family loop flag; clean rows; language macro)\n\n| system | FLORES+ finished | FLORES+ loop | NTREX finished | NTREX loop | MAFAND finished | MAFAND loop | FLORES+ length ratio |\n|---|---|---|---|---|---|---|---|")
for s, lab in (("sft", "SFT"), ("c1_s6000", "control C1 (6,000)"), ("stageA", "stage A"), ("b1", "B1 best"), ("mixed", "mixed best")):
    f = main["flags"][s]
    md.append(f"| {lab} | {f['flores_plus']['finished']:.3f} | {f['flores_plus']['loop']:.3f} | {f['ntrex']['finished']:.3f} | {f['ntrex']['loop']:.3f} | {f['mafand']['finished']:.3f} | {f['mafand']['loop']:.3f} | {f['flores_plus']['len_ratio']:.2f} |")
# decision rule
a = main["macro"]["all"]["all_cells"]; tp = L("evidence_mixed_vs_c6000_TP19.json")["macro"]["all"]["all_cells"]; ot = L("evidence_mixed_vs_c6000_OTHERS.json")["macro"]["all"]["all_cells"]
conds = {"lower 95% bound of d > +1.0": a["pp"][3] > 1.0, "AfriCOMET lower bound > 0": a["com"][3] > 0, "d > 0 on the 19 TranslatePsy languages": tp["pp"][2] > 0, "d > 0 on the other languages": ot["pp"][2] > 0}
better = all(conds.values()); comparable = (a["pp"][3] > -1.0 and a["pp"][4] < 1.0 and a["com"][3] <= 0 <= a["com"][4]); ctl_better = a["pp"][4] < -1.0
verdict = "RL is a better use of the compute" if better else "comparable" if comparable else "the control is better" if ctl_better else "inconclusive (registered rule: anything else)"
md.append("\n## Pre-registered decision rule applied to mixed best vs C1 at 6,000 steps\n\n| condition | value | met |\n|---|---|---|")
md.append(f"| lower 95% bound of d (chrF++, 93 cells) > +1.0 | d = {fmt(a['pp'])} | {conds['lower 95% bound of d > +1.0']} |")
md.append(f"| lower bound of the AfriCOMET difference > 0 | {fmt(a['com'],4)} | {conds['AfriCOMET lower bound > 0']} |")
md.append(f"| d > 0 on the 19 TranslatePsy languages | d = {fmt(tp['pp'])} (AfriCOMET {fmt(tp['com'],4)}) | {conds['d > 0 on the 19 TranslatePsy languages']} |")
md.append(f"| d > 0 on the other languages | d = {fmt(ot['pp'])} | {conds['d > 0 on the other languages']} |")
md.append(f"\n**Verdict by the rule as registered: {verdict}.**")
out["verdict"] = verdict; out["conditions"] = conds
open(H + "/verdict_tables.md", "w").write("\n".join(md) + "\n"); json.dump(out, open(H + "/verdict_tables.json", "w"), indent=1)
print("\n".join(md))
