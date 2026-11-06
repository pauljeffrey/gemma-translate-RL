#!/usr/bin/env python
"""E-02, AfriCOMET-STL-1.1 (reference-based) of every output of decode_study2.py json files, plus BLEU/spBLEU-free corpus BLEU per language.   python decode_comet.py OUT.json IN1.json IN2.json ..."""
import json, sys, unicodedata
sys.path.insert(0, "/workspace/eval_385m/gemma_mt_tree")
import torch
torch.cuda.set_per_process_memory_fraction(12 / 79.0)
import sacrebleu
from gemma_mt.rl.comet import CometScorer, REF_MODEL_ID
sc = CometScorer(REF_MODEL_ID, batch_size=32, reference_based=True)
out = {}
for p in sys.argv[2:]:
    d = json.load(open(p)); items = d["items"]; r = {}
    for sysn, hs in d["hyp"].items():
        h = [x["hyp"] for x in hs]
        v = [float(x) for x in sc([i[1] for i in items], h, [i[2] for i in items])]
        per, bleu = {}, {}
        for iso in sorted(set(i[0] for i in items)):
            ii = [k for k, i in enumerate(items) if i[0] == iso]
            per[iso] = sum(v[k] for k in ii) / len(ii)
            bleu[iso] = sacrebleu.corpus_bleu([unicodedata.normalize("NFC", h[k]) for k in ii], [[unicodedata.normalize("NFC", items[k][2]) for k in ii]]).score
        r[sysn] = {"africomet_per_language": per, "africomet_macro": sum(per.values()) / len(per), "bleu_macro": sum(bleu.values()) / len(bleu), "bleu_per_language": bleu}
        print(p, sysn, round(r[sysn]["africomet_macro"], 4), flush=True)
    out[p] = r
json.dump(out, open(sys.argv[1], "w"), indent=1)
