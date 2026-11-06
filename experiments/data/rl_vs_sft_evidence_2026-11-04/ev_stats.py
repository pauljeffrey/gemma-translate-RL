#!/usr/bin/env python
"""Per-sentence chrF / chrF++ n-gram statistics, finish / loop flags, loop-truncated statistics and the AfriCOMET join for one system.

Run on the A100 box with the venv_comet python, CPU only (no torch, no GPU):
    nice -n 19 python ev_stats.py SYSTEM PREDICTIONS.jsonl OUT.npz [COMET_CACHE.jsonl ...]

Per row (key = benchmark, iso, id; the one duplicated key of the 385M SFT file keeps its first row):
  stats[24]      sacrebleu CHRF(word_order=2) statistics of NFC(hyp) vs NFC(ref): 6 char orders then 2 word orders, [hyp, ref, match] each.
                 chrF++ = f_score(sum of all 24), plain chrF = f_score(sum of the first 18) (the evaluation's definition, word_order 0)
  tstats[24]     the same for the hypothesis truncated at the first loop onset (see truncate_loop); equal to stats when the row has no loop
  finished, loop flags; comet = AfriCOMET-STL-1.1 (reference-based) per-sentence score from the stored cache, NaN when the sentence is not cached
"""
import hashlib
import importlib.util
import json
import sys
import unicodedata

import numpy as np
from sacrebleu.metrics import CHRF

GUARDS = "/workspace/clean_rerun/tree/gemma_mt/gemma_mt/rl/guards.py"
spec = importlib.util.spec_from_file_location("guards_standalone", GUARDS)
guards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guards)


def truncate_loop(text, max_period=20, min_reps=3):
    """Cut a looping hypothesis at its first loop onset.

    1. earliest word position p with a block of k <= max_period words repeated >= min_reps times in a row from p: keep words[:p + k]
       (one occurrence of the block);
    2. otherwise (the detector fired on a low-diversity text without an exact period): cut at the start of the second occurrence of the first
       4-gram that occurs three times.
    Returns the text unchanged when no onset is found. Truncation only shortens the text, which raises precision and lowers recall."""
    w = text.split()
    n = len(w)
    for p in range(n):
        for k in range(1, max_period + 1):
            if p + k * min_reps > n:
                break
            blk = w[p:p + k]
            if all(w[p + j * k:p + (j + 1) * k] == blk for j in range(1, min_reps)):
                return " ".join(w[:p + k])
    seen = {}
    for i in range(n - 3):
        g = tuple(w[i:i + 4])
        seen.setdefault(g, []).append(i)
        if len(seen[g]) == 3:
            return " ".join(w[:seen[g][1]])
    return text


def nfc(s):
    return unicodedata.normalize("NFC", s)


def asbool(v):
    return v is True or str(v).lower() == "true"


def main():
    system, path, out = sys.argv[1:4]
    caches = sys.argv[4:]
    comet = {}
    for c in caches:
        for line in open(c):
            k, v = json.loads(line)
            comet[k] = v
    chrf = CHRF(word_order=2)
    keys, rows = [], []
    seen = set()
    for line in open(path):
        r = json.loads(line)
        key = (r["benchmark"], r["iso"], str(r["id"]))
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
        rows.append(r)
    n = len(rows)
    hyps = [nfc(r["hyp"]) for r in rows]
    refs = [nfc(r["reference"]) for r in rows]
    stats = np.array(chrf._extract_corpus_statistics(hyps, [refs]), dtype=np.int32)
    loop = np.array([bool(guards.is_repetitive(r["hyp"]) or guards.is_self_repetitive(r["hyp"])) for r in rows])
    thyps = [nfc(truncate_loop(r["hyp"])) if loop[i] else hyps[i] for i, r in enumerate(rows)]
    tstats = stats.copy()
    idx = np.where(loop)[0]
    if len(idx):
        tstats[idx] = np.array(chrf._extract_corpus_statistics([thyps[i] for i in idx], [[refs[i] for i in idx]]), dtype=np.int32)
    finished = np.array([asbool(r.get("finished")) for r in rows])
    cm = np.array([comet.get(hashlib.sha1(f"{r['english']}\x00{r['hyp']}\x00{r['reference']}".encode()).hexdigest(), np.nan) for r in rows], dtype=np.float64)
    np.savez_compressed(out, system=system, bench=np.array([k[0] for k in keys]), iso=np.array([k[1] for k in keys]), rid=np.array([k[2] for k in keys]),
                        stats=stats, tstats=tstats, finished=finished, loop=loop, comet=cm,
                        hyp_chars=np.array([len(r["hyp"]) for r in rows], dtype=np.int32), ref_chars=np.array([len(r["reference"]) for r in rows], dtype=np.int32))
    print(system, n, "rows; finished", finished.mean().round(4), "loop", loop.mean().round(4), "comet cached", np.isfinite(cm).mean().round(4),
          "corpus chrF++ all", round(chrf._compute_f_score(stats.sum(0)), 3))


if __name__ == "__main__":
    main()
