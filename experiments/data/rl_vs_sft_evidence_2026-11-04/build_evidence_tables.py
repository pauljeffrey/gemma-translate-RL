#!/usr/bin/env python
"""Markdown tables of reports/rl_vs_sft_evidence_2026-11-04.md from evidence.json + extra.json.   python build_evidence_tables.py > tables.md"""
import json

d = json.load(open("evidence.json")); x = json.load(open("extra.json"))
fam = json.load(open("groups_family.json")); weak = json.load(open("groups_weak.json")); a0 = json.load(open("groups_a0new.json"))
B = ["flores_plus", "ntrex", "mafand"]; BN = {"flores_plus": "FLORES+ devtest", "ntrex": "NTREX", "mafand": "MAFAND"}
out = []
P = out.append


def ci(v, nd=2, sign=True):
    f = f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}"
    return f"{f.format(v[2])} [{f.format(v[3])}, {f.format(v[4])}]"


def row(m, k, nd=2):
    return f"{m[k][0]:.{nd}f} | {m[k][1]:.{nd}f} | {ci(m[k], nd)}" if k in m else "- | - | -"


P("## Table 1. Macro gain of the final RL model over SFT (mean over languages of corpus scores; 95 % paired bootstrap CI, 1000 resamples)\n")
VN = {"all": "all clean rows", "paired_good": "only rows where BOTH models finished and neither loops", "sft_trunc": "SFT output cut at its first loop (RL as is)",
      "both_trunc": "both outputs cut at their first loop", "sft_bad_rows": "only rows where SFT did not finish or looped", "sft_ok_rows": "only rows where SFT finished and did not loop"}
for k, nm in (("pp", "chrF++"), ("c", "chrF (word order 0)"), ("com", "AfriCOMET-STL-1.1 (reference-based, held out)")):
    nd = 4 if k == "com" else 2
    P(f"### {nm}\n")
    P("| rows | benchmark | languages | SFT | RL | RL - SFT [95 % CI] |\n|---|---|---:|---:|---:|---|")
    for v in ("all", "paired_good", "sft_trunc", "both_trunc", "sft_bad_rows", "sft_ok_rows"):
        for b in B + ["all_cells"]:
            m = d["macro"][v].get(b, {})
            if k not in m:
                continue
            P(f"| {VN[v]} | {BN.get(b, 'all 93 benchmark-language cells')} | {m[k][5]} | {m[k][0]:.{nd}f} | {m[k][1]:.{nd}f} | {ci(m[k], nd)} |")
    P("")

P("## Table 2. Finishing and looping (clean rows; share of rows, language-macro)\n")
P("| system | " + " | ".join(f"{BN[b]} finished / loop / hyp-to-ref length" for b in B) + " |\n|---|---|---|---|")
for s, nm in (("sft", "SFT"), ("stageA", "RL stage A"), ("b1", "RL B1 best"), ("mixed", "RL mixed best")):
    P(f"| {nm} | " + " | ".join(f"{d['flags'][s][b]['finished']:.3f} / {d['flags'][s][b]['loop']:.3f} / {d['flags'][s][b]['len_ratio']:.2f}" for b in B) + " |")
tot = {b: {k: sum(v[k] for kk, v in x.items() if kk.startswith(b)) for k in ("n", "good", "ref_flag")} for b in B}
P("\nShare of rows kept by the paired subset (both finished, no loop): " + ", ".join(f"{BN[b]} {tot[b]['good'] / tot[b]['n']:.3f}" for b in B)
  + ". Loop detector (`is_repetitive` or `is_self_repetitive` of the RL reward guards) on the human references: " + ", ".join(f"{BN[b]} {tot[b]['ref_flag'] / tot[b]['n']:.4f}" for b in B) + " of rows.\n")

P("## Table 3. Where the gain accrued along the RL chain (all clean rows, macro; paired bootstrap)\n")
P("| step | benchmark | chrF++ before | after | Δ chrF++ [CI] | Δ chrF [CI] | Δ AfriCOMET [CI] |\n|---|---|---:|---:|---|---|---|")
for p, nm in (("sft->stageA", "SFT -> stage A (400 steps)"), ("stageA->b1", "stage A -> B1 best (135 steps run, best checkpoint)"), ("b1->mixed", "B1 best -> mixed best (60 steps)"), ("sft->b1", "SFT -> B1 best")):
    for b in B:
        m = d["stages"][p][b]
        P(f"| {nm} | {BN[b]} | {m['pp'][0]:.2f} | {m['pp'][1]:.2f} | {ci(m['pp'])} | {ci(m['c'])} | {ci(m['com'], 4) if 'com' in m else '-'} |")
P("")

P("## Table 4. By language group (language-macro Δ, final RL minus SFT; FLORES+ devtest, then NTREX where the group is present)\n")
GL = {"family": "language family (`groups_family.json`)", "weak": "weak list (the x2.5 RL list, reports/weak_language_selection_2026-10-26.md)", "a0new": "A0 language (first 85) vs language added later (40)", "sft_tercile": "terciles of the SFT's own FLORES+ chrF++ (a resource-level proxy: no external resource labels exist for the 54 languages)"}
for v in ("all", "paired_good", "sft_trunc"):
    P(f"### {VN[v]}\n")
    P("| grouping | group | FLORES+ languages | Δ chrF++ [CI] | Δ chrF [CI] | Δ AfriCOMET [CI] | NTREX Δ chrF++ [CI] |\n|---|---|---:|---|---|---|---|")
    for g, gg in d["groups"][v].items():
        for lab, m in gg.items():
            f = m["flores_plus"]; n = m["ntrex"]
            if not f and not n:
                continue
            P(f"| {GL[g].split(' (')[0]} | {lab} | {f['pp'][5] if f else 0} | {ci(f['pp']) if f else '-'} | {ci(f['c']) if f else '-'} | {ci(f['com'], 4) if f and 'com' in f else '-'} | {ci(n['pp']) if n and 'pp' in n else '-'} |")
    P("")

P("## Table 5. Languages where the RL model is WORSE than the SFT (point estimate; ‘sig’ = 95 % CI entirely below zero)\n")
P("| rows | benchmark | metric | cells | worse | sig. worse | sig. better |\n|---|---|---|---:|---:|---:|---:|")
for v in ("all", "paired_good", "sft_trunc"):
    for b in B:
        for k, nm in (("pp", "chrF++"), ("c", "chrF"), ("com", "AfriCOMET")):
            xs = [(c.split(":")[1], e[k]) for c, e in d["cells"][v].items() if c.startswith(b + ":") and k in e]
            if not xs:
                continue
            w = [l for l, e in xs if e["diff"] < 0]; sw = [l for l, e in xs if e["hi"] < 0]; sb = [l for l, e in xs if e["lo"] > 0]
            P(f"| {VN[v]} | {BN[b]} | {nm} | {len(xs)} | {len(w)}{(' (' + ', '.join(w) + ')') if w else ''} | {len(sw)}{(' (' + ', '.join(sw) + ')') if sw else ''} | {len(sb)} |")
P("")

P("## Table 6. Per language, FLORES+ devtest (54 languages; chrF++ unless stated)\n")
P("| iso | family | weak list | SFT | RL | Δ all rows [CI] | Δ chrF [CI] | Δ AfriCOMET [CI] | SFT finished / loop | RL finished / loop | rows kept (paired) | Δ paired rows [CI] | Δ SFT cut at loop [CI] |\n|---|---|---|---:|---:|---|---|---|---|---|---:|---|---|")
for k in sorted(d["cells"]["all"], key=lambda c: -d["cells"]["all"][c]["pp"]["diff"]):
    if not k.startswith("flores_plus:"):
        continue
    l = k.split(":")[1]; e = d["cells"]["all"][k]; p = d["cells"]["paired_good"].get(k); t = d["cells"]["sft_trunc"].get(k); xx = x[k]
    f = lambda z, nd=2: f"{z['diff']:+.{nd}f} [{z['lo']:+.{nd}f}, {z['hi']:+.{nd}f}]"
    P(f"| {l} | {fam.get(l, '?')} | {'yes' if weak.get(l, '').startswith('weak') else ''} | {e['pp']['sft']:.2f} | {e['pp']['rl']:.2f} | {f(e['pp'])} | {f(e['c'])} | {f(e['com'], 3) if 'com' in e else '-'} | "
      f"{xx['sft_fin'] / xx['n']:.2f} / {xx['sft_loop'] / xx['n']:.2f} | {xx['rl_fin'] / xx['n']:.2f} / {xx['rl_loop'] / xx['n']:.2f} | {xx['good'] / xx['n']:.2f} | {f(p['pp']) if p else '-'} | {f(t['pp']) if t else '-'} |")
P("")
for b in ("ntrex", "mafand"):
    P(f"## Table 7{'a' if b == 'ntrex' else 'b'}. Per language, {BN[b]} (clean rows)\n")
    P("| iso | n | SFT chrF++ | RL chrF++ | Δ all rows [CI] | Δ chrF [CI] | Δ AfriCOMET [CI] | Δ paired rows [CI] | rows kept |\n|---|---:|---:|---:|---|---|---|---|---:|")
    for k in sorted(d["cells"]["all"], key=lambda c: -d["cells"]["all"][c]["pp"]["diff"]):
        if not k.startswith(b + ":"):
            continue
        l = k.split(":")[1]; e = d["cells"]["all"][k]; p = d["cells"]["paired_good"].get(k); xx = x[k]
        f = lambda z, nd=2: f"{z['diff']:+.{nd}f} [{z['lo']:+.{nd}f}, {z['hi']:+.{nd}f}]"
        P(f"| {l} | {e['n']} | {e['pp']['sft']:.2f} | {e['pp']['rl']:.2f} | {f(e['pp'])} | {f(e['c'])} | {f(e['com'], 3) if 'com' in e else '-'} | {f(p['pp']) if p else '-'} | {xx['good'] / xx['n']:.2f} |")
    P("")
print("\n".join(out))
