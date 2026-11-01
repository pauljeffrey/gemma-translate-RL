#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One row per target language: what we hold for it (en->X pairs, pivot partners, direct rows, monolingual text, benchmark sets, reward family). numpy/json only.

    python scripts/afri2afri_language_table.py --matrix-report pair_matrix_report.json --tensor pair_count_tensor.npy --bench bench_multiway_coverage.json \
        --direct direct_pair_counts.json --tokdata tokdata_kb.txt --catalog-mono catalog_mono.json --out-dir reports/afri2afri_2026-10-20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gemma_mt.languages import ALL_ISOS, LANGUAGES  # noqa: E402

SFT77 = [i for i in ALL_ISOS if i not in ("nde", "fat", "knc", "gaa", "men", "luy", "tzm", "ful")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-report", required=True)
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--direct", required=True)
    ap.add_argument("--tokdata", required=True, help="lines 'iso kilobytes' of /workspace/tokdata/shards")
    ap.add_argument("--catalog-mono", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    rep = json.load(open(a.matrix_report))["per_language"]
    c = np.load(a.tensor).sum(0)
    bench = json.load(open(a.bench))["summary"]
    direct = json.load(open(a.direct))["direct_pairs"]
    tok = {}
    for line in open(a.tokdata):
        k, kb = line.split()
        tok[k] = int(kb) / 1024.0
    cat = json.load(open(a.catalog_mono))
    d_by = {}
    for k, v in direct.items():
        x, y = k.split("-")
        d_by[x] = d_by.get(x, 0) + v
        d_by[y] = d_by.get(y, 0) + v
    tokmap = {"swh": "swh", "orm": "orm", "mlg": "mlg", "pcm": "pcm", "twi": "twi", "tir": "tir"}
    rows = []
    for iso in SFT77:
        i = ALL_ISOS.index(iso)
        L = LANGUAGES[iso]
        partners = int(((c[i] > 0)).sum())
        bsets = [n for n in ("flores_devtest", "ntrex", "smol_sent", "smol_doc") if iso in bench[n]["languages"]]
        mode = "qe" if (L.africomet_status == "verified" and L.africomet_code) else ("blend/qe-listed" if L.qe_covered else "consensus")
        rows.append({"iso": iso, "name": L.name, "en_pairs_unique_english": rep.get(iso, {}).get("unique_english_sentences", 0),
                     "in_multilanguage_groups": rep.get(iso, {}).get("in_multi_language_group", 0), "pivot_partners_ge1": partners,
                     "pivot_pairs_out": int(c[i].sum()), "direct_rows": d_by.get(iso, 0), "mono_shard_mb": round(tok.get(iso, 0.0), 1),
                     "catalog_pretrain_datasets": cat.get(iso, {}).get("datasets", 0), "catalog_commercial_ok": cat.get(iso, {}).get("commercial_ok", 0),
                     "multiway_benchmarks": "+".join(bsets), "africomet": L.africomet_status, "reward_family_en2x": mode})
    os.makedirs(a.out_dir, exist_ok=True)
    cols = list(rows[0])
    with open(os.path.join(a.out_dir, "language_table.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in cols) + "\n")
    lines = ["| iso | name | en->X unique English | in multi-lang groups | pivot partners | pivot pairs (out) | direct rows | mono shards MB | multi-way benchmarks | reward family (en->X) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['iso']} | {r['name']} | {r['en_pairs_unique_english']:,} | {r['in_multilanguage_groups']:,} | {r['pivot_partners_ge1']} | {r['pivot_pairs_out']:,} | {r['direct_rows']:,} | "
                     f"{r['mono_shard_mb']:,.0f} | {r['multiway_benchmarks'] or '-'} | {r['reward_family_en2x']} |")
    open(os.path.join(a.out_dir, "language_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(len(rows), "languages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
