#!/usr/bin/env python
"""Tables of reports/eval_science_rl_385m_2026-11-04.md.  python build_sci_rl_tables.py > tables.md   (inputs in this directory)"""
import json
H = "."
L = ["amh", "hau", "lug", "nso", "yor", "zul"]
afs = {n: json.load(open(f"afsci_{n}.json")) for n in ("sci_a0", "sci_a0.5", "sci_rl_best", "sci_rl_final")}
na = json.load(open("neural_alpha.json")); sn = json.load(open("sent_neural_rl.json")); sp = json.load(open("sci_pairs.json"))
LAB = {"sci_a0": "385M mixed best = science-SFT alpha 0 (science RL's grandparent)", "sci_a0.5": "385M science SFT, alpha 0.5 (**the science RL's start**)", "sci_rl_best": "385M science RL, best/ (step 75)", "sci_rl_final": "385M science RL, final/ (step 160)"}
mean = lambda xs: sum(xs) / len(xs)
out = []; P = out.append
def neu(n, sp_, key):
    if n in ("sci_a0", "sci_a0.5"):
        a = "0" if n == "sci_a0" else "0.5"; d = na.get(f"{a}/{sp_}", {})
        return mean([d[l][key] for l in L]) if all(l in d and key in d[l] for l in L) else None
    d = sn.get(f"{n}/{sp_}", {}); k = {"africomet": "africomet", "ssa_comet_mtl": "ssa_comet_mtl", "ssa_qe": "ssa_qe"}[key]
    return mean([d[l][k] for l in L]) if all(l in d for l in L) else None
print_ = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"
for s in ("test", "dev"):
    P(f"## Table S1{'a' if s == 'test' else 'b'}. AfriScience-MT {s.upper()}, sentence level, greedy, macro over the 6 languages (amh hau lug nso yor zul)\n")
    P("| model | chrF (word order 0) | chrF++ | BLEU | finished | length ratio | AfriCOMET-STL (ref, held out) | SSA-COMET-QE (reward family) | SSA-COMET-MTL ref (reward family) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in afs:
        m = afs[n]["splits"][s]["sentence"]["macro"]
        qe = {"sci_a0": sp["pairs"]["alpha0->alpha05/" + s]["qe"]["a"], "sci_a0.5": sp["pairs"]["alpha05->rl_best/" + s]["qe"]["a"]}.get(n)
        P(f"| {LAB[n]} | {m['chrf']:.2f} | {m['chrf_pp']:.2f} | {m['bleu']:.2f} | {m['finished_rate']:.4f} | {m['length_ratio']:.3f} | {print_(neu(n, s, 'africomet'), 4)} | {print_(qe if qe is not None else neu(n, s, 'ssa_qe'), 4)} | {print_(neu(n, s, 'ssa_comet_mtl'), 4)} |")
    P("")
P("## Table S2. Paired differences, AfriScience-MT sentence level (paired Poisson bootstrap over sentences, 1000 resamples, macro over the 6 languages; 95 % CI)\n")
P("| comparison | split | chrF | AfriCOMET-STL (ref, held out) | SSA-COMET-QE (reward family) |\n|---|---|---|---|---|")
NAME = {"alpha05->rl_best": "science RL best minus science SFT alpha 0.5", "alpha05->rl_final": "science RL final minus science SFT alpha 0.5", "alpha0->rl_best": "science RL best minus alpha 0 (mixed best)",
        "alpha0->alpha05": "science SFT alpha 0.5 minus alpha 0 (for scale)", "rl_best->rl_final": "science RL final minus best"}
for k in NAME:
    for s in ("test", "dev"):
        v = sp["pairs"][f"{k}/{s}"]; f = lambda m, nd: f"{v[m]['diff']:+.{nd}f} [{v[m]['lo']:+.{nd}f}, {v[m]['hi']:+.{nd}f}]"
        P(f"| {NAME[k]} | {s} | {f('chrf', 2)} | {f('afri', 4)} | {f('qe', 4)} |")
P("")
P("## Table S3. Per language, TEST, science RL final minus science SFT alpha 0.5\n")
P("| iso | chrF alpha 0.5 | chrF RL final | diff [CI] | AfriCOMET alpha 0.5 | AfriCOMET RL final | diff [CI] |\n|---|---:|---:|---|---:|---:|---|")
v = sp["pairs"]["alpha05->rl_final/test"]
for l in L:
    c, a = v["chrf"]["per_language"][l], v["afri"]["per_language"][l]
    P(f"| {l} | {c['a']:.2f} | {c['b']:.2f} | {c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] | {a['a']:.4f} | {a['b']:.4f} | {a['diff']:+.4f} [{a['lo']:+.4f}, {a['hi']:+.4f}] |")
P("")
print("\n".join(out))
