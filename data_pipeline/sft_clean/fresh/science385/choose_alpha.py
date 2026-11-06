#!/usr/bin/env python3
"""Choice of the interpolation weight of the science SFT (old-stage rule, 2026-10-21, adapted): the alpha with the best AfriScience DEV sentence chrF (macro of the 6 languages)
subject to NTREX and MAFAND (macro chrF++ over their languages, all rows) not lower than the start model (alpha 0) by more than 0.5. Reads eval dirs out_sci_a<alpha>_x (NTREX/MAFAND results.csv) and
out_afsci_sci_a<alpha>/afriscience_eval.json. Writes choice.json.
    python choose_alpha.py --root /workspace/eval_385m --alphas 0 0.3 0.5 0.7 1.0 --out choice_alpha.json
"""
import argparse, csv, json, os, statistics as st
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--alphas", nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
rows = {}
for al in a.alphas:
    tag = f"a{al}"
    d = json.load(open(f"{a.root}/out_afsci_sci_{tag}/afriscience_eval.json"))
    dev = d["splits"]["dev"]["sentence"]["macro"]["chrf"]
    res = {}
    for r in csv.DictReader(open(f"{a.root}/out_sci_{tag}_x/results.csv")):
        if r["subset"] == "all" and r["chrf"]: res.setdefault(r["benchmark"], []).append(float(r["chrf"]))
    rows[al] = dict(afsci_dev_chrf=dev, ntrex=st.mean(res["ntrex"]), mafand=st.mean(res["mafand"]))
base = rows[a.alphas[0]]
ok = {al: r for al, r in rows.items() if r["ntrex"] >= base["ntrex"] - 0.5 and r["mafand"] >= base["mafand"] - 0.5}
best = max(ok, key=lambda al: ok[al]["afsci_dev_chrf"])
json.dump(dict(chosen_alpha=best, rows=rows, rule="best AfriScience dev chrF subject to NTREX and MAFAND >= start - 0.5"), open(a.out, "w"), indent=1); print(best, json.dumps(rows, indent=1))
