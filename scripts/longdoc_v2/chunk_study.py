# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chunk plans + per-chunk QE of every study item (R8): the raw material of the aggregation study.

  python chunk_study.py --items items_main.jsonl --out chunks_main.pkl --cache qe_cache --modes v1 v2 [--no-gpu] [--procs 6]

For every item and alignment mode (v1 = production plan: language-aware Gale-Church + omission detector; v2 = anchored, redundant copies removed) the chunk plan of
rl/chunk_plan.plan_chunks is built, the scored chunks go through AfriCOMET-QE-STL-1.1 and SSA-COMET-MTL (cached on disk), and the per-chunk record keeps
plan items (segment index | None = failed chunk, weight), raw scores c1 / c2, shaped rewards r1 / r2 / ensemble, the plan statistics. No texts are kept."""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import qe_cache  # noqa: E402

from gemma_mt.rl import longdoc as LD  # noqa: E402
from gemma_mt.rl.chunk_plan import plan_chunks  # noqa: E402
from gemma_mt.rl.qe_ensemble import QEEnsemble  # noqa: E402
from gemma_mt.rl.reward_shaping import ContentShaper  # noqa: E402

CAL = os.environ.get("R8_CAL", "/workspace/jobs/r8/gemma_mt_r8/calibration/shaping_flores_dev_v8.json")
_SH = None


def shaper():
    global _SH
    if _SH is None:
        _SH = ContentShaper.from_file(CAL, reliability_from="same_source", mode="calibrated", preset="repo", neg_scale=2.0, fail_floor=0.0)
    return _SH


def plan_item(args):
    it, mode, lexicon = args
    sh = shaper()
    iso = it["iso"]
    lm = sh.length_model(iso, None, None)
    w_hi = (sh.len_window.get(iso) or (None, None))[1]
    om = {"length_model": lm, "short_frac": 0.7, "short_min": 0.15} if lm is not None else {}
    if mode == "v2":
        om = dict(om, beads_fn=LD.beads_fn(lm, iso if lexicon else None))
    t0 = time.time()
    cp = plan_chunks(it["src_text"], it["out"], chunking="sentences", chunk_align="gale_church", chunk_agg="source_length", extra_output_penalty=True,
                     extra_ratio_limit=1.8, length_slack=0.15, copy_jaccard=0.9, w_hi=w_hi, **om)
    dt = time.time() - t0
    return {"segments": cp.segments, "items": cp.items, "how": cp.how, "n_failed": cp.n_failed, "extra": cp.extra, "n_short": cp.n_short, "seconds": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--modes", nargs="*", default=["v1", "v2"])
    ap.add_argument("--lexicon", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--skip-unfinished", action="store_true", default=True)
    a = ap.parse_args()
    docs = C.core_docs()
    items = C.rd(a.items)
    for it in items:
        it["src_text"] = docs[it["doc_id"]]["text"]
    work = [it for it in items if it.get("finished", True)]      # an unfinished output is a failure class (no_end_of_turn) whatever its content
    print(len(items), "items,", len(work), "finished", flush=True)
    sh = shaper()
    ens = QEEnsemble.from_file(CAL, sh, delta=1.0, prior_langs=True, reliability="ensemble", reliability_from="same_source", preset="repo", neg_scale=2.0, fail_floor=0.0)
    scorer = qe_cache.build_scorer(a.cache, gpu=not a.no_gpu)
    out = {"modes": a.modes, "records": {}}
    for mode in a.modes:
        t0 = time.time()
        with Pool(a.procs) as pool:
            plans = pool.map(plan_item, [(it, mode, a.lexicon) for it in work], chunksize=8)
        print(f"[{mode}] planned {len(plans)} in {time.time() - t0:.0f}s (mean {1000 * sum(p['seconds'] for p in plans) / len(plans):.1f} ms/plan)", flush=True)
        S, T = [], []
        for p in plans:
            for s, t in p["segments"]:
                S.append(s); T.append(t)
        t0 = time.time()
        c1 = scorer(S, T)
        c2 = scorer.score_secondary(S, T)
        print(f"[{mode}] scored {len(S)} segments in {time.time() - t0:.0f}s; cache hits {scorer.hits} misses {scorer.misses}", flush=True)
        k = 0
        for it, p in zip(work, plans):
            n = len(p["segments"])
            iso = it["iso"]
            a1, a2 = c1[k: k + n], c2[k: k + n]
            k += n
            r1 = [float(sh(x, iso)) for x in a1]
            r2 = [float(ens.shaper2(x, iso)) for x in a2]
            rr = [ens.shape(x, y, iso) for x, y in zip(a1, a2)]
            rec = {"items": p["items"], "c1": a1, "c2": a2, "r1": r1, "r2": r2, "rens": rr, "how": p["how"], "n_failed": p["n_failed"], "extra": p["extra"],
                   "n_short": p["n_short"], "seconds": p["seconds"], "seg_src_chars": [len(s) for s, _ in p["segments"]], "seg_mt_chars": [len(t) for _, t in p["segments"]]}
            out["records"][(it["doc_id"], it["iso"], it["cond"], mode)] = rec
        ens.reset()
    scorer.close()
    out["items"] = [{k: v for k, v in it.items() if k not in ("out", "src_text")} for it in items]
    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    print("saved", a.out, len(out["records"]))


if __name__ == "__main__":
    main()
