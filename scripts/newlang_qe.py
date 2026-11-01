#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AfriCOMET-QE of the probe outputs and of the human references (D-47), CPU. Run it with the COMET interpreter of the old box (transformers 4.x):
    /workspace/sft_build/comet_python.sh scripts/newlang_qe.py --in-dir out/rl2c --out out/qe.csv --keys fuv gaa ... [--kinds greedy sample human mismatch] [--threads 6]

For every probe file of a language that the AfriCOMET-QE-1.1 card names (or a trained yardstick) it scores (English source, text) for
    greedy / sample    the model's completions (NFC text, tag stripped)
    human              the human reference of the same sentence
    mismatch           the human reference of ANOTHER sentence (the validity test of D-29: AUC of human vs mismatch must be >= 0.70)
and writes one row per scored pair. `gemma_mt.rl.comet.CometScorer` is the production scorer (the QE model of the reward).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--sets-dir", default=None, help="score only the human / mismatch rows of the reference sets (key__set.jsonl of newlang_probe_sets.py): the validity test, no model outputs")
    ap.add_argument("--samples-csv", default=None, help="score the completions of a metrics samples.csv instead of raw probe files (English / reference looked up by position in --sets-dir; the first 60 of a 150-set equal the 60-set)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--kinds", nargs="+", default=["greedy", "sample", "human", "mismatch"])
    ap.add_argument("--conds", nargs="+", default=["trained", "prefill"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import torch

    from gemma_mt.rl.comet import CometScorer

    torch.set_num_threads(a.threads)
    scorer = CometScorer("masakhane/africomet-qe-stl-1.1", batch_size=a.batch_size, device="cpu", half=False)
    rng = random.Random(a.seed)
    rows = []
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out, encoding="utf-8")):
            done.add((r["label"], r["key"], r["set"], r["cond"]))
        rows = list(csv.DictReader(open(a.out, encoding="utf-8")))
    nfc = lambda s: unicodedata.normalize("NFC", s)             # noqa: E731
    items = []
    if a.in_dir:
        for path in sorted(glob.glob(os.path.join(a.in_dir, "*.jsonl"))):
            with open(path, encoding="utf-8") as f:
                meta = json.loads(f.readline())["meta"]
                recs = [json.loads(l) for l in f if l.strip()]
            items.append((meta, recs))
    if a.samples_csv:
        import collections

        grp = collections.defaultdict(list)
        for r in csv.DictReader(open(a.samples_csv, encoding="utf-8")):
            grp[(r["label"], r["key"], r["set"], r["cond"])].append({"kind": r["kind"], "pi": int(r["pi"]), "k": int(r["k"]), "text": r["text"]})
        for (label, key, sname, cond), recs in grp.items():
            sp = os.path.join(a.sets_dir, f"{key}__{sname}.jsonl")
            if not os.path.exists(sp):
                continue
            srows = [json.loads(l) for l in open(sp, encoding="utf-8") if l.strip()]
            items.append(({"key": key, "set": sname, "cond": cond, "label": label, "english": [r["english"] for r in srows], "reference": [r["reference"] for r in srows]}, recs))
    if a.sets_dir and not a.samples_csv:
        for path in sorted(glob.glob(os.path.join(a.sets_dir, "*__*.jsonl"))):
            key, sname = os.path.basename(path)[:-6].split("__", 1)
            srows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            items.append(({"key": key, "set": sname, "cond": "refs", "label": "human", "english": [r["english"] for r in srows], "reference": [r["reference"] for r in srows]}, []))
    for meta, recs in items:
        key, sname, cond, label = meta["key"], meta["set"], meta["cond"], meta["label"]
        if (a.keys and key not in a.keys) or (cond not in a.conds and cond != "refs") or (label, key, sname, cond) in done:
            continue
        en, ref = meta["english"], meta["reference"]
        todo = []                                           # (kind, pi, k, src, mt)
        for r in recs:
            if r["kind"] in a.kinds and r["text"].strip():
                todo.append((r["kind"], r["pi"], r["k"], en[r["pi"]], nfc(r["text"])))
        if "human" in a.kinds:
            todo += [("human", i, 0, en[i], nfc(ref[i])) for i in range(len(en))]
        if "mismatch" in a.kinds:
            for i in range(len(en)):
                j = rng.randrange(len(en))
                while len(en) > 1 and j == i:
                    j = rng.randrange(len(en))
                todo.append(("mismatch", i, 0, en[i], nfc(ref[j])))
        sc = scorer([t[3] for t in todo], [t[4] for t in todo])
        for t, s in zip(todo, sc):
            rows.append({"label": label, "key": key, "set": sname, "cond": cond, "kind": t[0], "pi": t[1], "k": t[2], "qe": round(float(s), 4)})
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["label", "key", "set", "cond", "kind", "pi", "k", "qe"])
            w.writeheader()
            w.writerows(rows)
        print("qe", label, key, sname, cond, len(todo), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
