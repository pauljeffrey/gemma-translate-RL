#!/usr/bin/env python
"""E-01: the pre-registered choice. Among the guard configurations (greedy excluded) take the one with the highest dev macro chrF++ of the SFT (rows of FLORES+ dev held-out half); a tie within 0.05
chrF++ goes to the weaker penalty. The RL grid is only reported.   python guard_choose.py SFT_GRID.json RL_GRID.json OUT.json"""
import json, sys
ORDER = ["greedy", "freq002", "freq005", "freq010", "rep102", "rep105", "freq005_rep102"]
sft = json.load(open(sys.argv[1]))["models"]; sft = sft.get("sft") or next(iter(sft.values()))
grid = {g["name"]: g for g in json.load(open(sys.argv[1]))["grid"]}
best = max([n for n in ORDER if n != "greedy"], key=lambda n: (round(sft[n]["chrfpp_macro"] / 0.05), -ORDER.index(n)))
res = {"rule": "highest dev macro chrF++ of the SFT among the guard configurations (greedy excluded); ties within 0.05 -> weaker penalty; dev = FLORES+ dev held-out half (assets/dev/bench_dev_hsel.jsonl), never devtest",
       "chosen": {"name": best, "frequency_penalty": grid[best]["frequency_penalty"], "repetition_penalty": grid[best]["repetition_penalty"],
                  "sft_dev_chrfpp": sft[best]["chrfpp_macro"], "sft_dev_greedy_chrfpp": sft["greedy"]["chrfpp_macro"]},
       "sft_grid": {n: {k: v for k, v in sft[n].items() if k != "per_language"} for n in ORDER}}
try:
    rl = json.load(open(sys.argv[2]))["models"]; rl = rl.get("rl") or next(iter(rl.values()))
    res["rl_grid"] = {n: {k: v for k, v in rl[n].items() if k != "per_language"} for n in ORDER}
except Exception as e:
    res["rl_grid"] = f"not available: {e}"
json.dump(res, open(sys.argv[3], "w"), indent=1); print("CHOSEN", json.dumps(res["chosen"]))
