import json, sys
g = json.load(open("/workspace/jobs/r12/res/grid.json")); p = json.load(open("/workspace/jobs/r12/res/pools.json"))
allr = {**g, **p}
cfgs = sys.argv[1:]
isos = list(g["greedy"]["per_lang_chrf"])
print("config".ljust(26) + " ".join(i.rjust(5) for i in isos))
base = g["greedy"]["per_lang_chrf"]
print("greedy".ljust(26) + " ".join(f"{base[i]:5.1f}" for i in isos))
for c in cfgs:
    pl = allr[c]["per_lang_chrf"]; print(c.ljust(26) + " ".join(f"{pl[i]-base[i]:+5.1f}" for i in isos))
