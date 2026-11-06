import json
d = json.load(open("e05_all.json"))
L = ["amh", "hau", "lug", "nso", "yor", "zul"]
LAB = {"sft": "SFT (385M, start of RL)", "stageA": "RL stage A (final)", "b1best": "RL B1 best", "mixedbest": "RL mixed best (= science SFT alpha 0)", "weakfinal": "weak-group run final (not in the main chain)",
       "sci_a0": "science-SFT alpha 0", "sci_a0.3": "science-SFT alpha 0.3", "sci_a0.5": "science-SFT alpha 0.5 (science RL start)", "sci_a0.7": "science-SFT alpha 0.7", "sci_a1.0": "science-SFT alpha 1.0",
       "sci_rl_best": "science RL best (step 75)", "sci_rl_final": "science RL final (step 160)", "long_rl_best": "long-document RL best (step 50)", "long_rl_final": "long-document RL final (step 80)"}
FAM = {"sft": "none", "stageA": "RL (sentence reward)", "b1best": "RL (sentence reward)", "mixedbest": "RL (sentence reward)", "weakfinal": "RL (sentence reward)", "sci_a0": "RL (sentence reward)", "sci_a0.3": "RL + science SFT", "sci_a0.5": "RL + science SFT",
       "sci_a0.7": "RL + science SFT", "sci_a1.0": "RL + science SFT", "sci_rl_best": "RL (science reward incl. ssa-comet-mtl ref)", "sci_rl_final": "RL (science reward incl. ssa-comet-mtl ref)", "long_rl_best": "RL (+ long-document reward)", "long_rl_final": "RL (+ long-document reward)"}
mean = lambda xs: sum(xs) / len(xs)
# long_rl_final was not yet evaluated when E-05 ran: add it from the post-run evaluation (same scorers: sent_neural_long.json, afsci_long_rl_final.json)
sn = json.load(open("sent_neural_long.json")); af = json.load(open("afsci_long_rl_final.json"))
d["long_rl_final"] = {}
for sp in ("dev", "test"):
    pl = af["splits"][sp]["sentence"]["per_language"]
    d["long_rl_final"][sp] = {l: {"chrf": pl[l]["chrf"], "chrfpp": pl[l]["chrf_pp"], "ssa_qe": sn[f"long_rl_final/{sp}"][l]["ssa_qe"], "africomet_stl": sn[f"long_rl_final/{sp}"][l]["africomet"], "ssa_mtl_ref": sn[f"long_rl_final/{sp}"][l]["ssa_comet_mtl"]} for l in pl}
out = []; P = out.append
for sp in ("test", "dev"):
    P(f"## AfriScience-MT {sp.upper()}, sentence level, macro over the 6 languages (amh hau lug nso yor zul)\n")
    P("| stage | training with | chrF | chrF++ | **SSA-COMET-QE** | AfriCOMET-STL (held out) | SSA-COMET-MTL ref |\n|---|---|---:|---:|---:|---:|---:|")
    for k in LAB:
        if k not in d: continue
        r = d[k][sp]
        if not all(l in r for l in L): continue
        g = lambda m: mean([r[l][m] for l in L])
        P(f"| {LAB[k]} | {FAM[k]} | {g('chrf'):.2f} | {g('chrfpp'):.2f} | **{g('ssa_qe'):.4f}** | {g('africomet_stl'):.4f} | {g('ssa_mtl_ref'):.4f} |")
    P("")
print("\n".join(out))
