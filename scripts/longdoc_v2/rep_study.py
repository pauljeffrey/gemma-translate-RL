# SPDX-License-Identifier: AGPL-3.0-or-later
"""Repetition detectors on the study items (R8, diagnosis (c)): which loop types the v1 guards miss or wrongly flag at long length, and what the v2 severity does.

  python rep_study.py --items items_main.jsonl --cal calibration/longdoc_v2.json --out rep_study.json"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from gemma_mt.rl import guards  # noqa: E402
from gemma_mt.rl import longdoc as LD  # noqa: E402

CAL = None
SRC = {}


def one(it):
    text = it["out"]
    sw = it["src_words"]
    prof = LD.repetition_profile(text)
    sp = SRC[it["doc_id"]]
    sev, lvl = CAL.severity(prof, sw, sp)
    words = text.split()
    run = 1
    maxrun = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        maxrun = max(maxrun, run)
    raw = text.encode()
    v1 = {"run": maxrun > 4, "unique": len(words) >= 8 and len(set(w.lower() for w in words)) / len(words) < 0.3,
          "zlib": len(raw) >= 80 and len(zlib.compress(raw)) / len(raw) < 0.18, "selfrep": guards.is_self_repetitive(text)}
    loop = LD.__dict__.get("_r6_loop")
    return {"cond": it["cond"], "bin": it["bin"], "iso": it["iso"], "doc_id": it["doc_id"], "finished": it.get("finished", True), "r6_loop": it.get("r6_loop"),
            "levels": prof.levels, "src_levels": sp.levels, "sev": sev, "lvl": lvl, "v1": v1, "v1_any": v1["run"] or v1["unique"] or v1["zlib"] or v1["selfrep"],
            "v1_repetitive": v1["run"] or v1["unique"] or v1["zlib"]}


def init(cal_path, docs):
    global CAL, SRC
    CAL = LD.LongdocCalibration.from_file(cal_path)
    SRC = {k: LD.repetition_profile(v["text"]) for k, v in docs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--cal", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    docs = C.core_docs()
    items = C.rd(a.items)
    with Pool(a.procs, initializer=init, initargs=(a.cal, docs)) as pool:
        rows = pool.map(one, items, chunksize=16)
    json.dump(rows, open(a.out, "w"))
    # table: per condition x bin: share flagged by v1 (is_repetitive | selfrep), v2 severity >= 0.5 and mean severity
    agg = defaultdict(lambda: [0, 0, 0, 0.0, 0, 0])
    for r in rows:
        for key in ((r["cond"], r["bin"]), (r["cond"], 0)):
            g = agg[key]
            g[0] += 1
            g[1] += bool(r["v1_any"])
            g[2] += r["sev"] >= 0.5
            g[3] += r["sev"]
            g[4] += r["sev"] > 0
            g[5] += bool(r["r6_loop"])
    print("cond                 bin  n    v1_flag  v2_sev>=.5  v2_sev>0  mean_sev  r6_loop")
    for (c, b), g in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if b in (0, 1024, 2048, 4096):
            print(f"{c:20s} {b:5d} {g[0]:4d} {g[1]/g[0]:7.2f} {g[2]/g[0]:9.2f} {g[4]/g[0]:9.2f} {g[3]/g[0]:9.3f} {g[5]/g[0]:7.2f}")


if __name__ == "__main__":
    main()
