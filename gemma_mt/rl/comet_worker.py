#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AfriCOMET scoring worker (ported from the repo's rl/comet_worker.py).

    python -m gemma_mt.rl.comet_worker --model masakhane/africomet-qe-stl-1.1 [--model McGill-NLP/ssa-comet-mtl ...] [--fake]

`--model` may be repeated (D-55, the QE ensemble): every checkpoint is loaded into this process, a request picks one with "model": k (default 0). AfriCOMET-STL
checkpoints and UnifiedMetric checkpoints (SSA-COMET-MTL, run reference-free) are both handled by CometScorer.

Protocol, one JSON object per line:
    in : {"data": [{"src": ..., "mt": ..., "ref": ...}, ...], "batch_size": 32, "model": 0}
    out: {"scores": [..], "stats": {..}}  or  {"error": "..."}      (first line out: {"ready": true})
Library chatter goes to stderr; only protocol lines reach the real stdout. The reference column is used only
when the loaded model is reference-based.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--reference-based", action="store_true")
    ap.add_argument("--max-batch-chars", type=int, default=None, help="batches of long chunks are cut so that batch size x longest (src + mt) chars <= this (default: fixed-size batches)")
    args = ap.parse_args()
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    models = args.model or ["masakhane/africomet-qe-stl-1.1"]
    multi = len(models) > 1
    if multi:       # D-55: two checkpoints share this process and the GPU next to the policy: expandable segments (less fragmentation) and the cache is returned after every request
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.fake:
        from gemma_mt.rl.comet import FakeScorer

        scorers = [FakeScorer() for _ in models]
    else:
        from gemma_mt.rl.comet import CometScorer

        scorers = [CometScorer(m, reference_based=args.reference_based, max_batch_chars=args.max_batch_chars) for m in models]
    proto.write(json.dumps({"ready": True}) + "\n")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            data = req["data"]
            scorer = scorers[int(req.get("model", 0))]
            if "batch_size" in req and hasattr(scorer, "batch_size"):
                scorer.batch_size = int(req["batch_size"])
            scores = scorer([d["src"] for d in data], [d["mt"] for d in data],
                            [d.get("ref", "") for d in data] if args.reference_based else None)
            proto.write(json.dumps({"scores": [float(s) for s in scores], "stats": getattr(scorer, "last_stats", {}) or {}}) + "\n")
            if multi and not args.fake:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as exc:
            proto.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
