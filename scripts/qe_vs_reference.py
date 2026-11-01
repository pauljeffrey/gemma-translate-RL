#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Companion of validate_consensus.py (D-43): add the AfriCOMET-QE score to the samples of a validation run and report the same within-prompt Spearman numbers
for QE alone and for the 50/50 blend (group z-scores of QE and consensus), so the content modes qe / blend / consensus can be compared on the same samples.

    CUDA_VISIBLE_DEVICES= python scripts/qe_vs_reference.py --val val_efi_0.json val_efi_1.json --out qe_efi.json
CPU only (the QE model runs in the child process of gemma_mt.rl.comet.WorkerScorer, transformers 4.x on its PYTHONPATH).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

from gemma_mt.rl import calibrate_refs as C


def zs(v):
    m = statistics.mean(v)
    s = statistics.pstdev(v)
    return [(x - m) / s if s > 0 else 0.0 for x in v]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default="/workspace/venv_gemma/bin/python")
    ap.add_argument("--pythonpath", default="/workspace/sft_build/pylibs_tf4:/workspace/sft_build/pylibs")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args(argv)
    rows = []
    for p in a.val:
        rows += json.load(open(p, encoding="utf-8"))["rows"]
    ok = [r for r in rows if r["status"] == "ok" and r["consensus"] is not None and r["ref_chrf"] is not None]
    from gemma_mt.rl.comet import WorkerScorer

    sc = WorkerScorer(python=a.python, model_id="masakhane/africomet-qe-stl-1.1", batch_size=a.batch_size, pythonpath=a.pythonpath)
    qe = sc([r["english"] for r in ok], [r["text"] for r in ok])
    sc.close()
    for r, q in zip(ok, qe):
        r["qe"] = q
    groups = {}
    for r in ok:
        groups.setdefault(r["english"], []).append(r)
    res = {}
    for name, f in (("consensus", lambda g: [x["consensus"] for x in g]), ("qe", lambda g: [x["qe"] for x in g]),
                    ("blend50", lambda g: [0.5 * a_ + 0.5 * b_ for a_, b_ in zip(zs([x["qe"] for x in g]), zs([x["consensus"] for x in g]))])):
        rhos, cx, cy = [], [], []
        for g in groups.values():
            if len(g) < 3:
                continue
            v, y = f(g), [x["ref_chrf"] for x in g]
            r_ = C.spearman(v, y)
            if r_ is not None:
                rhos.append(r_)
            mv, my = statistics.mean(v), statistics.mean(y)
            cx += [t - mv for t in v]
            cy += [t - my for t in y]
        res[name] = {"within_prompt_mean": statistics.mean(rhos) if rhos else None, "within_prompt_se": statistics.pstdev(rhos) / len(rhos) ** 0.5 if len(rhos) > 1 else None,
                     "groups": len(rhos), "centered": C.spearman(cx, cy)}
    allq = C.spearman([r["qe"] for r in ok], [r["ref_chrf"] for r in ok])
    res["qe_overall"] = allq
    res["qe_vs_consensus_within_prompt"] = statistics.mean([x for x in (C.spearman([y["qe"] for y in g], [y["consensus"] for y in g]) for g in groups.values() if len(g) >= 3) if x is not None])
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
