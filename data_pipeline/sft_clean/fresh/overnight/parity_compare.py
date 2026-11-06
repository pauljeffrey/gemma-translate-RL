#!/usr/bin/env python
"""E-09 summary: identical-output share between engines and chrF++ of each (240 sentences), per model.   python parity_compare.py OUT.json NAME=prefix ..."""
import json, sys, itertools, unicodedata, os
import sacrebleu
out = sys.argv[1]; res = {}
for spec in sys.argv[2:]:
    name, pre = spec.split("=", 1); runs = {}
    for m in ("hf", "hf32", "vllm", "vllm32"):
        p = f"{pre}_{m}.json"
        if os.path.exists(p): runs[m] = json.load(open(p))
    if not runs: continue
    ref = [unicodedata.normalize("NFC", it[2]) for it in next(iter(runs.values()))["items"]]
    nfc = lambda x: unicodedata.normalize("NFC", x)
    r = {"chrfpp": {m: round(sacrebleu.corpus_chrf([nfc(h) for h in d["hyp"]], [ref], word_order=2).score, 2) for m, d in runs.items()}, "identical": {}}
    for a, b in itertools.combinations(sorted(runs), 2):
        r["identical"][f"{a} vs {b}"] = round(sum(x == y for x, y in zip(runs[a]["hyp"], runs[b]["hyp"])) / len(ref), 3)
    res[name] = r; print(name, json.dumps(r))
json.dump(res, open(out, "w"), indent=1)
