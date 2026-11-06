#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-editor: MODERATE AfriCOMET-QE verification of the human targets (owner request 2026-11-03; runs in the AfriCOMET environment, GPU).

    python scripts/post_editor_qe_filter.py --pairs SEL/pairs.jsonl --out SEL/pairs_qe.jsonl [--drop-share 0.27] [--floor 250] [--min-train 300] [--ssa]

Every pair is scored reference-free with AfriCOMET-QE-STL-1.1 (src = English, mt = the HUMAN reference, i.e. an alignment / adequacy check of the target) and optionally SSA-COMET-MTL (--ssa, the mean of
the two per-language z-scores is used). Rule (a MODERATE filter, not best-of-the-best): only for languages with a validated QE calibration (`gemma_mt.languages.QE_VALIDATED`, 42 languages), only TRAIN
pairs, the lowest `--drop-share` (default 27%) of a language's train pairs are flagged `qe_drop`, never below `--floor` (250) train pairs left, and languages with fewer than `--min-train` train pairs
keep everything; consensus-only / uncalibrated languages get NO filter (`qe_applied` false). Reason for not taking the top: AfriCOMET prefers certain lengths and copies entities (africomet_bias_study.py),
selecting only its favourites would bias the editor towards what the metric likes, shrink the diversity, and gut the languages where QE is unreliable. The evaluation stays reference-based (chrF++).
Writes pairs_qe.jsonl (fields qe, qe_applied, qe_drop added) and qe_filter_by_language.csv (threshold, dropped, kept per language).
"""
import argparse, csv, hashlib, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))


def apply_rule(rows, scores, valid, drop_share=0.27, floor=250, min_train=300):
    """Pure function (unit-tested): rows with pair_id / iso / split, scores {pair_id: z}. Sets qe, qe_applied, qe_drop; returns a per-language report."""
    by = defaultdict(list)
    for r in rows:
        r["qe"] = scores.get(r["pair_id"]); r["qe_applied"] = False; r["qe_drop"] = False
        if r["split"] == "train" and r["qe"] is not None: by[r["iso"]].append(r)
    rep = []
    for iso, rs in sorted(by.items()):
        n = len(rs); drop = 0; thr = None
        if iso in valid and n >= min_train:
            drop = max(0, min(int(drop_share * n), n - floor))
            order = sorted(rs, key=lambda r: (r["qe"], r["pair_id"]))
            for r in order[:drop]: r["qe_drop"] = True
            for r in rs: r["qe_applied"] = True
            thr = order[drop - 1]["qe"] if drop else None
        rep.append({"iso": iso, "n_train": n, "qe_applied": int(iso in valid and n >= min_train), "dropped": drop, "kept": n - drop, "threshold": thr})
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--drop-share", type=float, default=0.27); ap.add_argument("--floor", type=int, default=250)
    ap.add_argument("--min-train", type=int, default=300); ap.add_argument("--ssa", action="store_true"); ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--cache", default=None); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    import numpy as np
    from gemma_mt.languages import QE_VALIDATED
    from gemma_mt.rl.comet import QE_MODEL_ID, SSA_MTL_MODEL_ID, CometScorer
    rows = [json.loads(l) for l in open(a.pairs, encoding="utf-8") if l.strip()]
    valid = set(QE_VALIDATED)
    cache_path = a.cache or a.out + ".cache.jsonl"; cache = {}
    if os.path.exists(cache_path):
        for l in open(cache_path):
            k, m, v = json.loads(l); cache[(k, m)] = v
    metrics = [("qe", QE_MODEL_ID)] + ([("ssa", SSA_MTL_MODEL_ID)] if a.ssa else [])
    key = lambda r: hashlib.sha1(f"{r['english']}\x00{r['reference']}".encode()).hexdigest()
    for m, mid in metrics:
        todo = [r for r in rows if r["iso"] in valid and (key(r), m) not in cache]
        print(f"[{m}] scoring {len(todo)} pairs", flush=True)
        if todo:
            sc = CometScorer(mid, batch_size=a.batch_size, device="cpu" if a.cpu else None, half=not a.cpu)
            with open(cache_path, "a") as f:
                for i in range(0, len(todo), 4000):
                    part = todo[i:i + 4000]; out = sc([r["english"] for r in part], [r["reference"] for r in part], None)
                    for r, x in zip(part, out): cache[(key(r), m)] = float(x); f.write(json.dumps([key(r), m, float(x)]) + "\n")
                    f.flush(); print(f"  {min(i + 4000, len(todo))}/{len(todo)}", flush=True)
            del sc
    # per-language z-scores (the scales differ by language) and their mean over the metrics
    z = {}
    by = defaultdict(list)
    for r in rows:
        if r["iso"] in valid: by[r["iso"]].append(r)
    for iso, rs in by.items():
        zs = []
        for m, _ in metrics:
            v = np.array([cache[(key(r), m)] for r in rs]); zs.append((v - v.mean()) / (v.std() + 1e-9))
        for r, val in zip(rs, np.mean(zs, axis=0)): z[r["pair_id"]] = float(val)
    rep = apply_rule(rows, z, valid, a.drop_share, a.floor, a.min_train)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.splitext(a.out)[0] + "_by_language.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rep[0])); w.writeheader(); w.writerows(rep)
    print(json.dumps({"languages_filtered": sum(r["qe_applied"] for r in rep), "dropped": sum(r["dropped"] for r in rep), "train_pairs": sum(r["n_train"] for r in rep)}))


if __name__ == "__main__":
    main()
