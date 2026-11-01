# SPDX-License-Identifier: AGPL-3.0-or-later
"""Alignment accuracy of the reward's sentence aligners on documents with KNOWN alignment (R8, diagnosis (a)).

  python align_study.py --cal <shaping json> --out align_study.json [--aligners legacy lm anchored ...]

Sets: core sentence-by-sentence outputs (sentfix) and NLLB references (units = sentences), human SmolDoc documents stitched to 6/20/41/83/177 sentences
(the sentence counts of b256..b4096). Corruptions: corrupt.CORRUPTIONS. Metric (pairs = (source sentence, output sentence) inside one bead):
recall = true pairs found / true pairs, precision = true pairs found / pairs proposed, f1."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import corrupt as K  # noqa: E402

from gemma_mt.rl.segments import LengthModel, gale_church, split_sentences  # noqa: E402
from gemma_mt.rl.reward_shaping import ContentShaper  # noqa: E402


def beads_prod(src: List[str], out: List[str], lm, mode: str):
    n, m = len(src), len(out)
    if n == 0 or m == 0:
        return [(tuple(range(n)), tuple(range(m)))]
    if n < 2:
        return [((0,), tuple(range(m)))]
    if n == m:
        return [((i,), (i,)) for i in range(n)]
    if m <= 1:
        return [(tuple(range(n)), tuple(range(m)))]
    return gale_church(src, out, lm if mode == "lm" else None)


def pair_set(beads) -> set:
    return {(i, j) for si, oi in beads for i in si for j in oi}


def true_pairs(labels) -> set:
    return {(i, j) for j, lab in enumerate(labels) for i in lab}


def score_pairs(pred: set, truth: set) -> Tuple[float, float]:
    tp = len(pred & truth)
    return tp, len(pred), len(truth)


def build_sets(per_bin: int, n_human: int, langs, shaper):
    docs = C.core_docs()
    sel = C.select_doc_ids(docs, per_bin, [256, 512, 1024, 2048, 4096])
    sf = C.hyp_rows("sentfix")
    items = []   # (set, bin, iso, src_sents, units, other_pool)
    for iso in langs:
        nl = C.nllb_refs(iso)
        pool_other = []
        for b, ids in sel.items():
            for did in ids:
                d = docs[did]
                src = split_sentences(d["text"])
                r = sf.get((did, iso))
                if r is not None and len(r["hyp_sents"]) == len(src):
                    items.append(("sentfix", b, iso, src, K.units_from(r["hyp_sents"]), None))
                    pool_other.append((did, r["hyp_sents"]))
                x = nl.get(did)
                if x is not None and len(x["ref_sents"]) == len(src):
                    items.append(("nllb", b, iso, src, K.units_from(x["ref_sents"]), None))
        # other-document pools for the replace corruption (same language, different document)
        for k, it in enumerate(items):
            if it[2] == iso and it[5] is None:
                others = [s for did, hs in pool_other for s in hs]
                items[k] = it[:5] + (others,)
        try:
            pool = C.smol_equal_docs(iso)
        except FileNotFoundError:
            continue                                     # no human SmolDoc references for this language
        for b in (256, 512, 1024, 2048, 4096):
            for src, ref in C.stitched_human(iso, C.BIN_WORDS[b], n_human, seed=1, pool=pool):
                others = [s for _, _, rr in pool[:40] for s in rr]
                items.append(("human", b, iso, src, K.units_from(ref), others))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bin", type=int, default=8)
    ap.add_argument("--n-human", type=int, default=4)
    ap.add_argument("--langs", nargs="*", default=C.L6)
    ap.add_argument("--aligners", nargs="*", default=["legacy", "lm"])
    ap.add_argument("--corruptions", nargs="*", default=K.CORRUPTIONS)
    a = ap.parse_args()
    shaper = ContentShaper.from_file(a.cal, reliability_from="same_source")
    items = build_sets(a.per_bin, a.n_human, a.langs, shaper)
    print(len(items), "items", flush=True)
    aligners: Dict[str, Callable] = {"legacy": lambda s, o, lm, ctx: beads_prod(s, o, lm, "legacy"), "lm": lambda s, o, lm, ctx: beads_prod(s, o, lm, "lm")}
    for name in a.aligners:
        if name not in aligners:
            from gemma_mt.rl import longdoc as LD  # noqa: E402  (new module; aligners beyond the two baselines)

            aligners[name] = LD.make_study_aligner(name)
    acc: Dict[Tuple[str, str, int, str], List[float]] = defaultdict(lambda: [0, 0, 0, 0, 0.0])   # (aligner, corruption, set|bin) -> tp, pred, truth, n, seconds
    for k, (st, b, iso, src, units, others) in enumerate(items):
        lm = shaper.length_model(iso, None, None)
        for cname in a.corruptions:
            rng = random.Random(f"{cname}-{k}")
            cu = K.apply(cname, units, rng, others or [u[0] for u in units])
            text, out, labels = K.sentence_labels(cu)
            truth = true_pairs(labels)
            for an in a.aligners:
                t0 = time.time()
                ctx = {"iso": iso, "src_text": " ".join(src), "out_text": text}
                beads = aligners[an](src, out, lm, ctx)
                dt = time.time() - t0
                tp, npred, ntrue = score_pairs(pair_set(beads), truth)
                for key in ((an, cname, st, b), (an, cname, st, 0)):
                    r = acc[key]
                    r[0] += tp; r[1] += npred; r[2] += ntrue; r[3] += 1; r[4] += dt
    rows = []
    for (an, cname, st, b), (tp, npred, ntrue, n, dt) in sorted(acc.items()):
        rec = tp / max(ntrue, 1)
        prec = tp / max(npred, 1)
        rows.append({"aligner": an, "corruption": cname, "set": st, "bin": b, "n": n, "recall": rec, "precision": prec,
                     "f1": 2 * rec * prec / max(rec + prec, 1e-9), "ms": 1000 * dt / max(n, 1)})
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    # compact printout: f1 per aligner x corruption, pooled over sets and bins >= 1024
    print("aligner corruption set bin: f1 (recall/precision)")
    for r in rows:
        if r["bin"] in (0, 1024, 4096):
            print(f"{r['aligner']:8s} {r['corruption']:14s} {r['set']:8s} b{r['bin']:<5d} n={r['n']:<4d} f1={r['f1']:.3f} ({r['recall']:.3f}/{r['precision']:.3f}) {r['ms']:.1f} ms")


if __name__ == "__main__":
    main()
