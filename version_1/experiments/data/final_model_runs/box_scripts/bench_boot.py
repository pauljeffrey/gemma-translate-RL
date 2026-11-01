"""a1: paired sentence-level bootstrap (500 resamples) of RL - SFT per (benchmark, language, subset) for chrF++ and spBLEU, from the two predictions.jsonl."""
import csv, json, sys, unicodedata
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/workspace/sft_build/pylibs")
from sacrebleu.metrics import CHRF, BLEU
OUT = "/workspace/jobs/a1"
SUBS = {"all": lambda r: True, "clean": lambda r: r.get("contaminated") is False, "no_pair": lambda r: r.get("pair_seen") is False}
nfc = lambda s: unicodedata.normalize("NFC", s)
B = 500

def work(item):
    key, a, b = item        # a, b: lists of (hyp, ref) aligned
    chrf, bleu = CHRF(word_order=2), BLEU(tokenize="flores200")
    refs = [nfc(r) for _, r in a]
    res = {}
    rng = np.random.default_rng(abs(hash(key[:3])) % (2 ** 32) if False else 0)
    ix = rng.integers(0, len(a), size=(B, len(a)))
    for name, m, f in (("chrf", chrf, lambda s: chrf._compute_score_from_stats(list(s)).score), ("spbleu", bleu, lambda s: bleu._compute_score_from_stats(list(s)).score)):
        sa = np.array(m._extract_corpus_statistics([nfc(h) for h, _ in a], [refs]), dtype=float)
        sb = np.array(m._extract_corpus_statistics([nfc(h) for h, _ in b], [refs]), dtype=float)
        pt = f(sb.sum(0)) - f(sa.sum(0))
        d = np.array([f(sb[i].sum(0)) - f(sa[i].sum(0)) for i in ix])
        lo, hi = np.percentile(d, [2.5, 97.5])
        res[name] = (pt, lo, hi)
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
        a = [(P[("sft",) + k[:2] + (x,)]["hyp"], P[("sft",) + k[:2] + (x,)]["reference"]) for x in ids]
        b = [(P[("rl",) + k[:2] + (x,)]["hyp"], P[("rl",) + k[:2] + (x,)]["reference"]) for x in ids]
        items.append((k, a, b))
    with Pool(4) as p, open(f"{OUT}/bench_boot.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["benchmark", "iso", "subset", "n", "d_chrf", "d_chrf_lo", "d_chrf_hi", "d_spbleu", "d_spbleu_lo", "d_spbleu_hi"])
        for k, n, res in p.imap(work, items, chunksize=2):
            w.writerow([*k, n, *res["chrf"], *res["spbleu"]])
    print("done", len(items))
main()
