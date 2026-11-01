#!/usr/bin/env python3
"""SMOKE helper (agent C2): writes a SYNTHETIC archive.sqlite (gemma_mt.rl.archive writer) so that s4_archive.py can be exercised before the new RL run exists.
The English sides are real windows of the clean prompt pools; the 'translations' are target-language text of the replay file chosen by length: they are NOT translations of the source.

    python s4_smoke_archive.py --pool-v3 clean_v3 --pool-sci clean_sci_v2 --bank replay_train.jsonl --out archive.sqlite --langs yor,hau,amh,kea --per-lang 12
"""
import argparse
import collections
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s4_common as K  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool-v3", required=True)
ap.add_argument("--pool-sci", required=True)
ap.add_argument("--bank", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--langs", default="yor,hau,amh,kea")
ap.add_argument("--per-lang", type=int, default=12)
a = ap.parse_args()
K.gemma_pkg()
import pyarrow.parquet as pq
from gemma_mt.rl.archive import TranslationArchive

bank = collections.defaultdict(list)
for r in K.read_jsonl(a.bank):
    bank[r["iso"]].append(r["target"])
if os.path.exists(a.out):
    os.remove(a.out)
arch = TranslationArchive(a.out)
rng = random.Random(1)
for iso in a.langs.split(","):
    rows = []
    for pool in (a.pool_v3, a.pool_sci):
        p = os.path.join(pool, f"{iso}.parquet")
        if not os.path.exists(p):
            continue
        t = pq.read_table(p, columns=["english", "length_bin"]).to_pydict()
        idx = [i for i, b in enumerate(t["length_bin"]) if b in ("b256", "b512", "b1024")]
        rng.shuffle(idx)
        for i in idx[:a.per_lang // 2]:
            en = t["english"][i]
            tr = ""
            while len(tr) < 1.2 * len(en) and bank[iso]:
                tr += " " + rng.choice(bank[iso])
            tr = tr.strip()[: int(1.4 * len(en))]
            mode = "consensus" if iso == "kea" else "qe"
            rows.append((iso, en, mode, tr, rng.uniform(1.5, 2.2) if mode == "qe" else 0.9, "ok", 0.7, 75.0 if mode == "consensus" else None, 0.0, f"{iso}_Latn", 4, 0, t["length_bin"][i]))
    arch.submit(1, "smoke", rows)
arch.flush()
print(arch.close())
