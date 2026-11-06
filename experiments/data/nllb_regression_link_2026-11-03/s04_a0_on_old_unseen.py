#!/usr/bin/env python3
"""Step 4: A0 (the 270M clean-rerun SFT the 385M was grown from) scored on the SAME old-unseen sentences as the old 270M models and the 385M SFT.

The A0 predictions (research_archive_2026-11-03/eval_385m/out_a0/predictions.jsonl) carry no flags; the OLD contamination flags (eval_385m/old_flags/contamination_<bench>.json,
`exact` = English side seen by the old SFT) are attached by key '<iso>:<id>'. 'clean' = exact is False (the scorer's rule). chrF++ = sacrebleu CHRF(word_order=2) on NFC text, corpus level.
The 385M SFT predictions are re-scored the same way as a check against vs_old270m_same_sentences_per_language.csv.
"""
import json, os, sys, unicodedata
from collections import defaultdict
from multiprocessing import Pool
import pandas as pd

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(H, "..", "..", ".."))
EV = os.path.join(ROOT, "research_archive_2026-11-03", "eval_385m")
nfc = lambda s: unicodedata.normalize("NFC", s)


def score(job):
    from sacrebleu.metrics import CHRF
    key, hyps, refs = job
    return key, CHRF(word_order=2).corpus_score([nfc(h) for h in hyps], [[nfc(r) for r in refs]]).score, len(hyps)


def main():
    flags = {}
    for fn in os.listdir(os.path.join(EV, "old_flags")):
        b = fn[len("contamination_"):-5]
        flags[b] = json.load(open(os.path.join(EV, "old_flags", fn)))["exact"]
    jobs = []
    for sysname, path in (("a0", os.path.join(EV, "out_a0", "predictions.jsonl")), ("m385", os.path.join(EV, "out_m385", "predictions.jsonl"))):
        cells = defaultdict(lambda: ([], []))
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            b = r["benchmark"]
            if b not in flags:
                continue
            k = f"{r['iso']}:{r['id']}"
            fl = flags[b].get(k)
            if fl is None:
                continue
            for sub, ok in (("clean", fl is False), ("all", True)):
                if ok:
                    c = cells[(sysname, b, sub, r["iso"])]
                    c[0].append(r["hyp"]); c[1].append(r["reference"])
        jobs += [(k, h, rf) for k, (h, rf) in cells.items()]
        print(sysname, "cells", len(cells), flush=True)
    with Pool(8) as p:
        out = p.map(score, jobs, chunksize=4)
    df = pd.DataFrame([(*k, s, n) for k, s, n in out], columns=["system", "bm", "subset", "iso", "chrf", "n"])
    df.to_csv(os.path.join(H, "a0_m385_rescored_old_unseen.csv"), index=False)
    print(df.groupby(["system", "subset"]).chrf.mean())


if __name__ == "__main__":
    main()
