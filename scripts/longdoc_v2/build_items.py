# SPDX-License-Identifier: AGPL-3.0-or-later
"""Documents x conditions of the long-document reward study (R8): R6's saved outputs and corruptions of the sentence-by-sentence translation.

  python build_items.py --out items.jsonl [--per-bin 12] [--langs hau yor swh amh zul wol]

Conditions (cond):
  sf        sentfix (sentence-by-sentence + repair), assumed to stop (finished True)
  nllb      NLLB-200 3.3B sentence-by-sentence translation (finished True)
  W:<name>  R6 whole-document output of system <name> (greedy rep105 rep110 freq005 t03 t05 t07 t10 bon4), finished / loop flags of R6
  wins      greedy_win / winfix (256-token windows): chunked, no loops, 'good' like sf
  C:<name>  corruption of sf (corrupt.CORRUPTIONS) with a known alignment"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import corrupt as K  # noqa: E402

from gemma_mt.rl.segments import split_sentences  # noqa: E402

WHOLE = ["greedy", "rep105", "rep110", "freq005", "t03", "t05", "t07", "t10", "bon4"]
WINDOWS = ["greedy_win", "winfix"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bin", type=int, default=12)
    ap.add_argument("--bins", nargs="*", type=int, default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--langs", nargs="*", default=C.L6)
    a = ap.parse_args()
    docs = C.core_docs()
    sel = C.select_doc_ids(docs, a.per_bin, a.bins)
    sf = C.hyp_rows("sentfix")
    whole = {s: (C.hyp_rows(s), C.sc_rows(s)) for s in WHOLE + WINDOWS}
    rows = []
    for iso in a.langs:
        nl = C.nllb_refs(iso)
        pool = {b: [(did, sf[(did, iso)]["hyp_sents"]) for did in ids if (did, iso) in sf] for b, ids in sel.items()}
        for b, ids in sel.items():
            others_all = [s for b2, lst in pool.items() for did, hs in lst for s in hs]
            for did in ids:
                d = docs[did]
                src_text = d["text"]
                src = split_sentences(src_text)
                base = dict(doc_id=did, iso=iso, bin=b, src_words=len(src_text.split()), n_src_sents=len(src))
                r = sf.get((did, iso))
                if r is None or len(r["hyp_sents"]) != len(src):
                    continue
                units = K.units_from(r["hyp_sents"])
                rows.append({**base, "cond": "sf", "out": K.join_units(units), "finished": True})
                x = nl.get(did)
                if x is not None and len(x["ref_sents"]) == len(src):
                    rows.append({**base, "cond": "nllb", "out": " ".join(" ".join(s.split()) for s in x["ref_sents"]), "finished": True})
                for sname in WHOLE + WINDOWS:
                    hr, sc = whole[sname]
                    rr = hr.get((did, iso))
                    if rr is None:
                        continue
                    s = sc.get((did, iso)) or {}
                    rows.append({**base, "cond": ("W:" if sname in WHOLE else "S:") + sname, "out": rr["hyp"], "finished": bool(rr["finished"]),
                                 "r6_loop": bool(s.get("loop")), "r6_loop_tail": bool(s.get("loop_tail")), "r6_cov": s.get("cov_sents"), "r6_len_ratio": s.get("length_ratio")})
                others = [t for t in others_all if t not in r["hyp_sents"]] or others_all
                for cname in K.CORRUPTIONS:
                    if cname == "none":
                        continue
                    rng = random.Random(f"{cname}-{did}-{iso}")
                    cu = K.apply(cname, units, rng, others)
                    rows.append({**base, "cond": "C:" + cname, "out": K.join_units(cu), "finished": True,
                                 "n_kept": sum(1 for _, i in cu if i is not None)})
    C.wr(a.out, rows)
    from collections import Counter

    print(len(rows), "items;", Counter(r["cond"] for r in rows))


if __name__ == "__main__":
    main()
