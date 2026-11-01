#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The COMPLETE inter-African pair set as sharded parquet (D-46, owner 2026-10-20): every pivot-derived pair, every direct pair of the SFT corpus; the NLLB-mined pairs are written by
`afri2afri_fetch_mined.py` into the same folder tree.

    python scripts/build_afri2afri_full.py pivot  --sft OUT_SFT --pairs gemma/pairs_final --out OUT/full_pairs --bench ... --license-scan ...
    python scripts/build_afri2afri_full.py direct --sft OUT_SFT --out OUT/full_pairs --bench ... --license-scan ...

Layout: pivot_en/<src_iso>/part-NNNN.parquet (<= 1.5 M rows per file), pivot_en_sentences.parquet (group_key -> English sentence), direct/direct_pairs.parquet.
ONE row per UNORDERED pair {src_iso, tgt_iso} with src_iso < tgt_iso in `languages.ALL_ISOS` order: the ordered pair count is twice the row count, a model is trained on both directions
(storing both orders would double the text). Rows containing a text that matches a held-out evaluation set (exact or word-8-gram, incl. the long-document benchmark documents) and rows
whose source repo is flagged private are NOT written. Nothing is sampled, no quality threshold is applied: the quality signals are columns (qe_src / qe_tgt, lid_*, provenance_kind).
Columns of pivot_en: src_iso tgt_iso src tgt group_key group_size source_dataset licence licence_class provenance_kind qe_src qe_tgt lid_src lid_tgt lid_pivot_en_ok in_sft
(lid_*: 1 = GlotLID agrees, 0 = disagrees, 2 = not checkable; in_sft: 0 = not selected, 1 = selected for the SFT train sample, 2 = for the validation sample, before the text filters of `materialize`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import build_afri2afri_sft as B                                              # noqa: E402
from gemma_mt.afri2afri import contam, filters as F, pivot as pv, provenance as prov   # noqa: E402
from gemma_mt.afri2afri.text import norm_text                                 # noqa: E402
from gemma_mt.languages import ALL_ISOS, LANGUAGES                            # noqa: E402

PART_ROWS = 1_500_000
_W = {}


def declared_info(scan, name: str):
    """(repo, declared licence, private) of a source_dataset name (same rules as the SFT export)."""
    repo = prov.repo_of(name)
    d = scan.get(repo) or {}
    decl = d.get("declared") or B.KNOWN_DECLARED.get(repo) or ("cc-by-4.0" if re.match(B.MT560_RX, repo) else "unknown")
    return repo, decl, bool(d.get("private"))


def _init(bench, extra):
    li = contam.LeakIndex()
    li.add_all(contam.iter_benchmark_texts(bench, extra))
    _W["leak"] = li


def _gather(args):
    path, rows = args
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["english", "target"]).take(pa.array(rows, pa.int64()))
    en = [norm_text(x) for x in t["english"].to_pylist()]
    tg = [norm_text(x) for x in t["target"].to_pylist()]
    leak = _W["leak"]
    flags = [bool(leak.is_leak(a) or leak.is_leak(b)) for a, b in zip(en, tg)]
    return en, tg, flags


def _dict_array(pa, codes, strings):
    return pa.DictionaryArray.from_arrays(pa.array(codes, pa.int32()), pa.array(strings))


def cmd_pivot(a) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from multiprocessing import Pool

    t0 = time.time()
    scan = json.load(open(a.license_scan))["datasets"]
    z = dict(np.load(os.path.join(a.sft, "best_rows.npz")))
    L = np.load(os.path.join(a.sft, "lid.npz"))
    labels = json.load(open(os.path.join(a.sft, "lid_labels.json")))
    names = json.load(open(os.path.join(a.sft, "index", "datasets.json"), encoding="utf-8"))
    cal = json.load(open(a.calibration))
    pol = F.lid_policy(cal, fallback={k: v.glotlid for k, v in LANGUAGES.items()})
    n = len(z["key"])
    ver = B.row_verdicts(z["iso"], z["enlen"], z["tlen"], L["en_label"], L["tg_label"], labels, pol)
    verdict, en_ok = ver["verdict"], ver["en_ok"]
    # LID verdict codes of F: LID_OK 1, LID_BAD 0, LID_SKIP 2 (written as is)
    info = [declared_info(scan, nm) for nm in names]
    priv = np.array([x[2] for x in info], bool)[z["ds"]]
    files = sorted(__import__("glob").glob(os.path.join(a.pairs, "pairs-*.parquet")))
    jobs = []
    pos_of = []
    st0, sz0 = pv.group_bounds(z["group"])
    rlimit = int(st0[min(len(st0), a.max_groups) - 1] + sz0[min(len(st0), a.max_groups) - 1]) if a.max_groups else n          # test runs only
    for si in np.unique(z["shard"]):
        pos = np.flatnonzero(z["shard"] == si)
        pos = pos[pos < rlimit]
        if len(pos) == 0:
            continue
        for s in range(0, len(pos), 500_000):
            part = pos[s:s + 500_000]
            jobs.append((files[int(si)], [int(r) for r in z["row"][part]]))
            pos_of.append(part)
    en_text = np.empty(n, dtype=object)
    tg_text = np.empty(n, dtype=object)
    leak = np.zeros(n, bool)
    with Pool(a.workers, initializer=_init, initargs=(a.bench, a.extra_bench)) as pool:
        for k, (part, (en, tg, fl)) in enumerate(zip(pos_of, pool.imap(_gather, jobs))):
            en_text[part], tg_text[part], leak[part] = en, tg, fl
            print(f"gathered {k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    excl = leak | priv
    print("rows", n, "held-out hits", int(leak.sum()), "private-source rows", int(priv.sum()), "excluded", int(excl.sum()), flush=True)
    # SFT membership
    N = np.int64(n)
    sel = {}
    for tag, f in ((1, "selection_train.npz"), (2, "selection_val.npz")):
        s = np.load(os.path.join(a.sft, f))
        m = s["origin"] == 0
        sel[tag] = np.sort(s["ref_a"][m].astype(np.int64) * N + s["ref_b"][m].astype(np.int64))
    # per-row lookup tables
    ds_repo = [x[0] for x in info]
    ds_lic = [x[1] for x in info]
    lic_names = B.ix.LICENCE_CLASSES
    kind_names = [prov.KINDS[t] for t in range(4)]
    starts, sizes = pv.group_bounds(z["group"])
    out_root = os.path.join(a.out, "pivot_en")
    os.makedirs(out_root, exist_ok=True)
    writers = {}
    stats = {"rows_by_src": Counter(), "rows_by_kind": Counter(), "licence_rows": Counter(), "source_rows": Counter(), "in_sft": Counter(), "rows": 0}
    schema = None
    sentences = {"group_key": [], "group_size": [], "pivot_en": []}

    def writer_for(iso_id, table):
        w = writers.get(iso_id)
        if w is None or w["rows"] >= PART_ROWS:
            if w is not None:
                w["w"].close()
            d = os.path.join(out_root, ALL_ISOS[iso_id])
            os.makedirs(d, exist_ok=True)
            part = (w["part"] + 1) if w else 0
            w = {"w": pq.ParquetWriter(os.path.join(d, f"part-{part:04d}.parquet"), table.schema, compression="zstd"), "rows": 0, "part": part}
            writers[iso_id] = w
        return w

    G = min(len(starts), a.max_groups) if a.max_groups else len(starts)
    step = a.chunk_groups
    for g0 in range(0, G, step):
        g1 = min(G, g0 + step)
        r0 = int(starts[g0])
        r1 = int(starts[g1 - 1] + sizes[g1 - 1])
        gl, ra, rb = pv.enumerate_pairs(starts[g0:g1] - r0, sizes[g0:g1])
        ra = ra.astype(np.int64) + r0
        rb = rb.astype(np.int64) + r0
        swap = z["iso"][ra] > z["iso"][rb]
        ra, rb = np.where(swap, rb, ra), np.where(swap, ra, rb)
        ok = ~excl[ra] & ~excl[rb]
        gl, ra, rb = gl[ok], ra[ok], rb[ok]
        if len(ra) == 0:
            continue
        gkeys = np.array([format(int(k), "016x") for k in z["key"][starts[g0:g1]]], dtype=object)
        # English sentences of the groups that keep at least one pair
        for g in np.unique(gl):
            gg = g0 + int(g)
            sentences["group_key"].append(gkeys[g]); sentences["group_size"].append(int(sizes[gg])); sentences["pivot_en"].append(en_text[starts[gg]])
        sa, sb = z["ds"][ra].astype(np.int64), z["ds"][rb].astype(np.int64)
        combo = sa * len(names) + sb
        ucombo, cinv = np.unique(combo, return_inverse=True)
        src_ds = [ds_repo[c // len(names)] if ds_repo[c // len(names)] == ds_repo[c % len(names)] else f"{ds_repo[c // len(names)]} + {ds_repo[c % len(names)]}" for c in ucombo]
        lic_s = [ds_lic[c // len(names)] if ds_lic[c // len(names)] == ds_lic[c % len(names)] else f"{ds_lic[c // len(names)]} + {ds_lic[c % len(names)]}" for c in ucombo]
        lc = z["lic"][ra].astype(np.int64) * 16 + z["lic"][rb].astype(np.int64)
        ulc, linv = np.unique(lc, return_inverse=True)
        lic_class = [prov.combine_licences(lic_names[c // 16], lic_names[c % 16]) for c in ulc]
        tier = np.minimum(z["tier"][ra], z["tier"][rb]).astype(np.int64)
        key = ra.astype(np.int64) * N + rb.astype(np.int64)
        in_sft = np.zeros(len(ra), np.int8)
        for tag in (1, 2):
            pos = np.searchsorted(sel[tag], key)
            hit = (pos < len(sel[tag])) & (sel[tag][np.minimum(pos, len(sel[tag]) - 1)] == key)
            in_sft[hit] = tag
        tbl = pa.table({
            "src_iso": pa.DictionaryArray.from_arrays(pa.array(z["iso"][ra].astype(np.int8)), pa.array(list(ALL_ISOS))),
            "tgt_iso": pa.DictionaryArray.from_arrays(pa.array(z["iso"][rb].astype(np.int8)), pa.array(list(ALL_ISOS))),
            "src": pa.array(tg_text[ra].tolist(), pa.string()), "tgt": pa.array(tg_text[rb].tolist(), pa.string()),
            "group_key": pa.array(gkeys[gl].tolist(), pa.string()), "group_size": pa.array(sizes[g0 + gl].astype(np.int16)),
            "source_dataset": _dict_array(pa, cinv, src_ds), "licence": _dict_array(pa, cinv, lic_s), "licence_class": _dict_array(pa, linv, lic_class),
            "provenance_kind": _dict_array(pa, tier, kind_names), "qe_src": pa.array(z["qe"][ra]), "qe_tgt": pa.array(z["qe"][rb]),
            "lid_src": pa.array(verdict[ra].astype(np.int8)), "lid_tgt": pa.array(verdict[rb].astype(np.int8)),
            "lid_pivot_en_ok": pa.array(en_ok[ra] & en_ok[rb]), "in_sft": pa.array(in_sft)})
        siso = z["iso"][ra]
        for iso_id in np.unique(siso):
            m = np.flatnonzero(siso == iso_id)
            sub = tbl.take(pa.array(m))
            w = writer_for(int(iso_id), sub)
            w["w"].write_table(sub)
            w["rows"] += len(m)
            stats["rows_by_src"][ALL_ISOS[int(iso_id)]] += len(m)
        stats["rows"] += len(ra)
        for c, k in zip(*np.unique(tier, return_counts=True)):
            stats["rows_by_kind"][kind_names[int(c)]] += int(k)
        for c, k in zip(*np.unique(cinv, return_counts=True)):
            stats["licence_rows"][lic_s[int(c)]] += int(k)
            stats["source_rows"][src_ds[int(c)]] += int(k)
        for c, k in zip(*np.unique(in_sft, return_counts=True)):
            stats["in_sft"][int(c)] += int(k)
        print(f"groups {g1}/{G} pairs so far {stats['rows']:,} {time.time() - t0:.0f}s", flush=True)
    for w in writers.values():
        w["w"].close()
    pq.write_table(pa.table({"group_key": sentences["group_key"], "group_size": pa.array(sentences["group_size"], pa.int16()), "pivot_en": sentences["pivot_en"]}),
                   os.path.join(a.out, "pivot_en_sentences.parquet"), compression="zstd")
    rep = {"unordered_pairs_written": stats["rows"], "ordered_pairs": 2 * stats["rows"], "held_out_rows_removed": int(leak.sum()), "private_source_rows_removed": int(priv.sum()),
           "rows_removed_total": int(excl.sum()), "english_sentences": len(sentences["group_key"]), "rows_by_src_iso": dict(stats["rows_by_src"]), "rows_by_provenance_kind": dict(stats["rows_by_kind"]),
           "licence_rows": dict(stats["licence_rows"].most_common()), "source_dataset_rows": dict(stats["source_rows"].most_common()), "in_sft": {str(k): v for k, v in stats["in_sft"].items()},
           "seconds": round(time.time() - t0)}
    json.dump(rep, open(os.path.join(a.out, "_pivot_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if not isinstance(v, dict)}, indent=1))
    return 0


def cmd_direct(a) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = time.time()
    scan = json.load(open(a.license_scan))["datasets"]
    cal = json.load(open(a.calibration))
    pol = F.lid_policy(cal, fallback={k: v.glotlid for k, v in LANGUAGES.items()})
    d = pq.read_table(os.path.join(a.sft, "direct_clean.parquet")).to_pydict()
    n = len(d["src"])
    li = contam.LeakIndex()
    li.add_all(contam.iter_benchmark_texts(a.bench, a.extra_bench))
    cols = {k: [] for k in ("src_iso", "tgt_iso", "src", "tgt", "source_dataset", "licence", "licence_class", "provenance_kind", "lid_src", "lid_tgt")}
    removed = Counter()
    cache = {}
    for i in range(n):
        s_iso, t_iso, s, t, ds = d["src_iso"][i], d["tgt_iso"][i], d["src"][i], d["tgt"][i], d["source_dataset"][i]
        if ds not in cache:
            repo, decl, priv = declared_info(scan, ds)
            lc = prov.licence_class(decl if decl != "unknown" else None)
            cache[ds] = (repo, decl, priv, lc, prov.classify_repo(ds))
        repo, decl, priv, lc, kind = cache[ds]
        if priv:
            removed["private_source"] += 1
            continue
        if li.is_leak(s) or li.is_leak(t):
            removed["held_out"] += 1
            continue
        if ALL_ISOS.index(s_iso) > ALL_ISOS.index(t_iso):
            s_iso, t_iso, s, t, ls, lt = t_iso, s_iso, t, s, d["lid_tgt"][i], d["lid_src"][i]
        else:
            ls, lt = d["lid_src"][i], d["lid_tgt"][i]
        cols["src_iso"].append(s_iso); cols["tgt_iso"].append(t_iso); cols["src"].append(s); cols["tgt"].append(t)
        cols["source_dataset"].append(repo); cols["licence"].append(decl); cols["licence_class"].append(lc); cols["provenance_kind"].append(kind)
        cols["lid_src"].append(F.lid_verdict(s_iso, len(s), ls, pol)); cols["lid_tgt"].append(F.lid_verdict(t_iso, len(t), lt, pol))
    out = os.path.join(a.out, "direct")
    os.makedirs(out, exist_ok=True)
    tbl = pa.table({k: (pa.array(v, pa.int8()) if k.startswith("lid_") else pa.array(v).dictionary_encode() if k in ("src_iso", "tgt_iso", "source_dataset", "licence", "licence_class", "provenance_kind") else pa.array(v))
                    for k, v in cols.items()})
    pq.write_table(tbl, os.path.join(out, "direct_pairs.parquet"), compression="zstd")
    rep = {"input_rows": n, "written": len(cols["src"]), "removed": dict(removed), "licence_rows": dict(Counter(cols["licence"]).most_common()),
           "source_dataset_rows": dict(Counter(cols["source_dataset"]).most_common()), "provenance_kind_rows": dict(Counter(cols["provenance_kind"]).most_common()), "seconds": round(time.time() - t0)}
    json.dump(rep, open(os.path.join(a.out, "_direct_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k in ("input_rows", "written", "removed")}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("pivot", cmd_pivot), ("direct", cmd_direct)):
        p = sp.add_parser(name)
        p.add_argument("--sft", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--bench", required=True)
        p.add_argument("--extra-bench", nargs="*", default=[])
        p.add_argument("--license-scan", required=True)
        p.add_argument("--calibration", required=True)
        p.add_argument("--pairs", default=None)
        p.add_argument("--workers", type=int, default=10)
        p.add_argument("--chunk-groups", type=int, default=40000)
        p.add_argument("--max-groups", type=int, default=0, help="test runs: only the first N English groups")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
