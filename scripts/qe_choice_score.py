#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Score a JSONL of (src, mt[, ref]) rows with one or more COMET models (D-53 QE-metric choice study).

    PYTHONPATH=<pylibs_tf4>:<pylibs>:<repo> python qe_choice_score.py --rows rows.jsonl --out-dir scores/ \
        --spec africomet_qe=masakhane/africomet-qe-stl-1.1:qe --spec ssa_mtl_ref=McGill-NLP/ssa-comet-mtl:ref ...

A spec is NAME=MODEL:MODE, MODE = qe (the reference column is NOT passed) | ref (it is). Output: <out-dir>/<NAME>.json = list of floats aligned
with the input rows (a finished NAME is skipped, so the script can be restarted). Identical (src, mt, ref) triples are scored once. Same code path as
the RL reward (gemma_mt.rl.comet.CometScorer, fp16 on GPU), so the numbers are what the reward would see. Needs transformers 4.x (unbabel-comet 2.2.7).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def unified_call(sc, srcs, mts, refs):
    """CometScorer for UnifiedMetric checkpoints (SSA-COMET-MTL, AfriCOMET-MTL): prepare_sample returns a TUPLE of encoded inputs and the score
    comes from predict_step (src+mt only -> ONE forward over the (src, mt) view; with a reference -> the mean of the three views, as the model
    card's predict does)."""
    import torch
    from gemma_mt.rl.comet import batched_in_length_order

    rows = [{"src": s, "mt": m, **({"ref": r} if refs is not None else {})} for s, m, r in zip(srcs, mts, refs if refs is not None else [None] * len(srcs))]

    def run(batch):
        with torch.no_grad():
            inputs = sc.model.prepare_sample(batch, stage="predict")
            inputs = tuple({k: v.to(sc.device) for k, v in d.items()} for d in inputs)
            return [float(x) for x in sc.model.predict_step(inputs).scores.flatten().float().tolist()]

    return batched_in_length_order(rows, lambda r: len(r["src"]) + len(r["mt"]), run, sc.batch_size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--spec", action="append", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    from gemma_mt.rl.comet import CometScorer

    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8")]
    os.makedirs(args.out_dir, exist_ok=True)
    for spec in args.spec:
        name, rest = spec.split("=", 1)
        model, mode = rest.rsplit(":", 1)
        out = os.path.join(args.out_dir, name + ".json")
        if os.path.exists(out):
            print(f"skip {name} (exists)", flush=True)
            continue
        use_ref = mode == "ref"
        t0 = time.time()
        sc = CometScorer(model, batch_size=args.batch_size, half=not args.fp32, reference_based=use_ref)
        keys = {}
        uniq = []
        for r in rows:
            k = (r["src"], r["mt"], r.get("ref", "") if use_ref else "")
            if k not in keys:
                keys[k] = len(uniq)
                uniq.append(k)
        call = unified_call if type(sc.model).__name__ == "UnifiedMetric" else (lambda c, s_, m_, r_: c(s_, m_, r_))
        scores = call(sc, [u[0] for u in uniq], [u[1] for u in uniq], [u[2] for u in uniq] if use_ref else None)
        vals = [scores[keys[(r["src"], r["mt"], r.get("ref", "") if use_ref else "")]] for r in rows]
        json.dump(vals, open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
        print(f"{name}: {len(uniq)} unique / {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)
        del sc
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
