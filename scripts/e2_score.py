"""E2: per-sentence chrF++ / spBLEU statistics for one system on FLORES+ devtest en->xx.
usage: e2_score.py NAME KIND PATH     KIND = nllb (iso,english,hyp) | pipe (predictions.jsonl of sft_benchmarks, benchmark flores_plus) | iid (iso,id,hyp)
Output scores/NAME.pkl: {iso: {ids, hyp, chrf_st (n x k), bleu_st (n x k)}}.  chrF++ = sacrebleu CHRF(word_order=2) on NFC text; spBLEU = BLEU(tokenize=flores200) on NFC."""
import sys, json, pickle, unicodedata
import numpy as np
sys.path.append("/workspace/sft_build/pylibs")
from sacrebleu.metrics import CHRF, BLEU
name, kind, path = sys.argv[1:4]
nfc = lambda s: unicodedata.normalize("NFC", s)
refs = {}
by_en = {}
for l in open("/workspace/jobs/e2/refs_flores_plus.jsonl"):
    d = json.loads(l); refs.setdefault(d["iso"], []).append(d); by_en[(d["iso"], d["english"])] = d
hyps = {}
for l in open(path, encoding="utf-8"):
    d = json.loads(l)
    if kind == "nllb":
        r = by_en.get((d["iso"], d["english"]))
        if r is None: continue
        hyps[(d["iso"], str(r["id"]))] = d["hyp"]
    elif kind == "pipe":
        if d.get("benchmark") != "flores_plus": continue
        hyps[(d["iso"], str(d["id"]))] = d["hyp"].split("</think>")[-1].strip()
    else:
        hyps[(d["iso"], str(d["id"]))] = d["hyp"].split("</think>")[-1].strip()
chrf, bleu = CHRF(word_order=2), BLEU(tokenize="flores200")
out = {}
for iso, rows in sorted(refs.items()):
    rows = [r for r in rows if (iso, str(r["id"])) in hyps]
    if not rows: continue
    if len(rows) != 1012: print("WARNING", iso, len(rows), flush=True)
    h = [nfc(hyps[(iso, str(r["id"]))]) for r in rows]; rf = [nfc(r["reference"]) for r in rows]
    out[iso] = {"ids": [str(r["id"]) for r in rows], "hyp": h, "src": [r["english"] for r in rows], "ref": [r["reference"] for r in rows],
                "contaminated": [r["contaminated"] for r in rows], "pair_seen": [r["pair_seen"] for r in rows],
                "chrf_st": np.array(chrf._extract_corpus_statistics(h, [rf]), dtype=float), "bleu_st": np.array(bleu._extract_corpus_statistics(h, [rf]), dtype=float)}
    print(iso, len(rows), round(chrf._compute_score_from_stats(list(out[iso]["chrf_st"].sum(0))).score, 2), flush=True)
pickle.dump(out, open(f"/workspace/jobs/e2/scores/{name}.pkl", "wb"))
print("saved", name, len(out), "languages")
