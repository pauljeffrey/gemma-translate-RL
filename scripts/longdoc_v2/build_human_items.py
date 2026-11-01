# SPDX-License-Identifier: AGPL-3.0-or-later
"""Human long documents for the false-positive rate of the reward (R8): equal-count SmolDoc documents stitched to the median source length of each bin (known 1:1 alignment),
the human reference as the 'output', the stitched English as the source.   python build_human_items.py --out items_human.jsonl [--per-cell 6]"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--per-cell", type=int, default=6)
ap.add_argument("--langs", nargs="*", default=C.L8)
a = ap.parse_args()
rows = []
for iso in a.langs:
    pool = C.smol_equal_docs(iso)
    for b in (256, 512, 1024, 2048, 4096):
        for k, (src, ref) in enumerate(C.stitched_human(iso, C.BIN_WORDS[b], a.per_cell, seed=11, pool=pool)):
            rows.append({"doc_id": f"human-{iso}-{b}-{k}", "iso": iso, "bin": b, "cond": "human", "src_text": " ".join(src), "out": " ".join(ref), "finished": True,
                         "src_words": sum(len(x.split()) for x in src), "n_src_sents": len(src)})
C.wr(a.out, rows)
print(len(rows), "human documents")
