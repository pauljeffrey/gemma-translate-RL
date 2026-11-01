"""Sentence-pair-level corpus chrF++ for sentence-by-sentence hypotheses on the AfriScience items (comparable with the published / eval_afriscience.py sentence-level numbers),
next to the document-level pooled chrF++ used everywhere else.  usage: lc_sentlevel.py MODEL__MODE ...   -> res/sentlevel.csv"""
import sys, csv, collections, os
from lc_common import *
from sacrebleu.metrics import CHRF
chrf = CHRF(word_order=2)
items = {i["iid"]: i for i in load_items()}
refs = load_refs("human")
rows = []
for name in sys.argv[1:]:
    hyps = {r["iid"]: r for r in rd(f"{HYP}/{name}.jsonl")}
    by = collections.defaultdict(lambda: ([], []))
    for iid, it in items.items():
        if it["set"] != "sci" or iid not in hyps or not hyps[iid].get("hyp_sents"):
            continue
        h, r = hyps[iid]["hyp_sents"], refs[iid]["ref_sents"]
        if len(h) != len(r):
            continue
        by[(it["bin"], it["iso"])][0].extend(h); by[(it["bin"], it["iso"])][1].extend(r)
    for (b, iso), (h, r) in sorted(by.items()):
        rows.append({"name": name, "bin": b, "lang": iso, "n_sent": len(h), "chrf_pp_sentence_level": chrf.corpus_score(h, [r]).score})
p = RES + "/sentlevel.csv"
new = not os.path.exists(p)
with open(p, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "bin", "lang", "n_sent", "chrf_pp_sentence_level"])
    if new: w.writeheader()
    w.writerows(rows)
print(len(rows))
