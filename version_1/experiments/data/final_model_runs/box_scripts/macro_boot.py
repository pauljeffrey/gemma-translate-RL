"""a1: per (benchmark, language, subset): B=1000 paired bootstrap replicates of chrF++ and spBLEU for SFT and RL (same resampled sentences), saved so that macro CIs can be formed on any language group."""
import json, sys, unicodedata, pickle
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/workspace/sft_build/pylibs")
from sacrebleu.metrics import CHRF, BLEU
OUT = "/workspace/jobs/a1"
SUBS = {"all": lambda r: True, "clean": lambda r: r.get("contaminated") is False, "no_pair": lambda r: r.get("pair_seen") is False}
nfc = lambda s: unicodedata.normalize("NFC", s)
B = 1000
def work(item):
    key, a, b = item
    chrf, bleu = CHRF(word_order=2), BLEU(tokenize="flores200")
    refs = [nfc(r) for _, r in a]
    rng = np.random.default_rng(0); ix = rng.integers(0, len(a), size=(B, len(a)))
    res = {}
    for name, m in (("chrf", chrf), ("spbleu", bleu)):
        sa = np.array(m._extract_corpus_statistics([nfc(h) for h, _ in a], [refs]), dtype=float)
        sb = np.array(m._extract_corpus_statistics([nfc(h) for h, _ in b], [refs]), dtype=float)
        f = lambda s: m._compute_score_from_stats(list(s)).score
        res[name + "_sft"] = np.array([f(sa[i].sum(0)) for i in ix]); res[name + "_rl"] = np.array([f(sb[i].sum(0)) for i in ix])
        res[name + "_sft_pt"] = f(sa.sum(0)); res[name + "_rl_pt"] = f(sb.sum(0))
    return key, len(a), res
def main():
    P = {}
    for s in ("sft", "rl"):
        for l in open(f"{OUT}/out_{s}/predictions.jsonl", encoding="utf-8"):
            r = json.loads(l); P[(s, r["benchmark"], r["iso"], str(r["id"]))] = r
    keys = defaultdict(list)
    for (s, b, i, id_), r in P.items():
        if s == "rl" and ("sft", b, i, id_) in P and b not in ("gatitos", "smol_doc"):
            for sub, f in SUBS.items():
                if f(r): keys[(b, i, sub)].append(id_)
    items = []
    for k, ids in sorted(keys.items()):
        ids = sorted(ids)
        items.append((k, [(P[("sft",) + k[:2] + (x,)]["hyp"], P[("sft",) + k[:2] + (x,)]["reference"]) for x in ids], [(P[("rl",) + k[:2] + (x,)]["hyp"], P[("rl",) + k[:2] + (x,)]["reference"]) for x in ids]))
    out = {}
    with Pool(4) as p:
        for k, n, res in p.imap(work, items, chunksize=2): out[k] = (n, res)
    pickle.dump(out, open(f"{OUT}/macro_boot.pkl", "wb"))
    print("done", len(out))
main()
