"""finished / loop rates (reward-family loop flag) of SFT, control C1 (6,000) and mixed best on the 19 TranslatePsy languages vs the other languages (language macro per benchmark, clean rows). usage: python panel_finished_loop.py SFT.npz MIXED.npz C1.npz"""
import sys, json, numpy as np
TP = set("afr amh hau ibo kin lin lug mlg nya orm sna som sot swh tsn wol xho yor zul".split())
Z = {n: np.load(p, allow_pickle=True) for n, p in zip(("sft", "mixed", "c1"), sys.argv[1:4])}
keys = {n: {(b, i, r): k for k, (b, i, r) in enumerate(zip(z["bench"], z["iso"], z["rid"]))} for n, z in Z.items()}
common = [k for k in keys["sft"] if k in keys["mixed"] and k[0] in ("flores_plus", "ntrex", "mafand") and k in keys["c1"]]
out = {}
for n, z in Z.items():
    ix = np.array([keys[n][k] for k in common]); b = np.array([k[0] for k in common]); i = np.array([k[1] for k in common])
    fin = z["finished"][ix].astype(float); lp = z["loop"][ix].astype(float)
    for panel, sel in (("TP19", np.isin(i, list(TP))), ("OTHERS", ~np.isin(i, list(TP)))):
        for bm in ("flores_plus", "ntrex", "mafand"):
            m = sel & (b == bm); langs = sorted(set(i[m]))
            if not langs: continue
            f = np.mean([fin[m & (i == l)].mean() for l in langs]); l_ = np.mean([lp[m & (i == l)].mean() for l in langs])
            out[f"{n}|{panel}|{bm}"] = (round(float(f), 3), round(float(l_), 3), len(langs))
for panel in ("TP19", "OTHERS"):
    print(panel)
    for bm in ("flores_plus", "ntrex", "mafand"):
        print(" ", bm, {n: out.get(f"{n}|{panel}|{bm}") for n in Z})
json.dump(out, open("panel_finished_loop.json", "w"), indent=1)
