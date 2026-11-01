#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inter-African SFT builder (D-46): X -> Y training windows from the data of the en->X build. CPU only, runs `nice`d on the old box.

    python scripts/build_afri2afri_sft.py index   --pairs /workspace/sft_build/gemma/pairs_final --out /workspace/afri2afri/sft
    python scripts/build_afri2afri_sft.py matrix  --out /workspace/afri2afri/sft --bench /workspace/sft_build/bench/test_sets
    python scripts/build_afri2afri_sft.py direct  --final-v2 /workspace/sft_build/final_v2 --out /workspace/afri2afri/sft
    python scripts/build_afri2afri_sft.py select  ...     (policy -> chosen pairs)           [see the sections below]
    python scripts/build_afri2afri_sft.py pack    ...     (windows -> packed uint32 dataset)

Stages are resumable: each writes into `--out` and the next one reads from there. Nothing here downloads or pushes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from gemma_mt.afri2afri import contam, index as ix, pivot as pv, prompt as PF, provenance as prov      # noqa: E402
from gemma_mt.languages import ALL_ISOS, LANGUAGES                                  # noqa: E402

SFT77 = tuple(i for i in ALL_ISOS if i not in ("nde", "fat", "knc", "gaa", "men", "luy", "tzm", "ful"))      # D-38: the 85 table rows minus the 8 without SFT data


def _index_worker(args):
    path, = args
    table: Dict[str, int] = {}
    arr = ix.index_shard(path, table)
    return path, arr, table


def cmd_index(a) -> int:
    from multiprocessing import Pool

    files = sorted(glob.glob(os.path.join(a.pairs, "pairs-*.parquet")))
    os.makedirs(os.path.join(a.out, "index"), exist_ok=True)
    t0 = time.time()
    results = []
    with Pool(a.workers) as pool:
        for path, arr, table in pool.imap(_index_worker, [(f,) for f in files]):
            results.append((path, arr, table))
            print(f"indexed {os.path.basename(path)} rows={len(arr['key']):,} {time.time() - t0:.0f}s", flush=True)
    merged, remaps = ix.merge_ds_tables([r[2] for r in results])
    for (path, arr, _), rm in zip(results, remaps):
        arr["ds"] = rm[arr["ds"]]
        np.savez(os.path.join(a.out, "index", os.path.basename(path).replace(".parquet", ".npz")), **arr)
    ix.save_json(os.path.join(a.out, "index", "datasets.json"), [k for k, _ in sorted(merged.items(), key=lambda kv: kv[1])])
    print("datasets", len(merged), "rows", sum(len(r[1]["key"]) for r in results), f"{time.time() - t0:.0f}s")
    return 0


def load_index(out: str):
    files = sorted(glob.glob(os.path.join(out, "index", "pairs-*.npz")))
    cols: Dict[str, List[np.ndarray]] = {}
    shard, row = [], []
    for si, f in enumerate(files):
        z = np.load(f)
        for k in z.files:
            cols.setdefault(k, []).append(z[k])
        n = len(z["key"])
        shard.append(np.full(n, si, np.uint8))
        row.append(np.arange(n, dtype=np.uint32))
    arr = {k: np.concatenate(v) for k, v in cols.items()}
    arr["shard"], arr["row"] = np.concatenate(shard), np.concatenate(row)
    names = json.load(open(os.path.join(out, "index", "datasets.json"), encoding="utf-8"))
    return arr, names, files


def build_leak(bench: str, extra: List[str]) -> contam.LeakIndex:
    li = contam.LeakIndex()
    li.add_all(contam.iter_benchmark_texts(bench, extra))
    return li


def cmd_matrix(a) -> int:
    t0 = time.time()
    arr, names, _ = load_index(a.out)
    n = len(arr["key"])
    tier_of_ds = np.array([prov.tier_of(x) for x in names], np.uint8)
    kind_names = [prov.classify_repo(x) for x in names]
    tier = tier_of_ds[arr["ds"]]
    report: Dict = {"rows": int(n)}
    ok = (arr["key"] != 0) & (arr["iso"] != ix.NO_LANG)
    report["rows_with_usable_key_and_language"] = int(ok.sum())
    extra = [p for p in (a.extra_bench or [])]
    li = build_leak(a.bench, extra)
    ex = np.fromiter(li.exact, dtype=np.uint64, count=len(li.exact))
    leak_en = np.isin(arr["key"], ex)
    leak_t = np.isin(arr["tkey"], ex)
    report["benchmark_sentences"] = li.count
    report["benchmark_sentences_by_set"] = li.by_bench
    report["rows_english_in_benchmark"] = int((leak_en & ok).sum())
    report["rows_target_in_benchmark"] = int((leak_t & ok).sum())
    keep = ok & ~leak_en & ~leak_t
    report["rows_kept_after_exact_leak_filter"] = int(keep.sum())
    idx = np.flatnonzero(keep)
    key, iso, qe, tr = arr["key"][idx], arr["iso"][idx], arr["qe"][idx], tier[idx]
    best = pv.best_per_key_iso(key, iso, tr.astype(np.float64) * 10 + qe)
    sel = idx[best]
    report["unique_english_x_language"] = int(len(sel))
    skey, siso, stier = arr["key"][sel], arr["iso"][sel], tier[sel]
    starts, sizes = pv.group_bounds(skey)
    report["unique_english_sentences"] = int(len(starts))
    hist = np.bincount(np.minimum(sizes, 90))
    report["group_size_histogram"] = {int(k): int(v) for k, v in enumerate(hist) if v}
    report["groups_with_2_or_more_languages"] = int((sizes >= 2).sum())
    cnt = pv.pair_count_tensor(siso, stier, starts, sizes, len(ALL_ISOS))
    report["ordered_pairs_total"] = int(cnt.sum())
    report["ordered_pairs_by_tier"] = {prov.KINDS[t]: int(cnt[t].sum()) for t in range(4)}
    np.save(os.path.join(a.out, "pair_count_tensor.npy"), cnt)
    # language table: rows per language before/after the join
    lang_rows = np.bincount(siso, minlength=len(ALL_ISOS))
    in_groups = np.zeros(len(skey), bool)
    gid = np.repeat(np.arange(len(starts)), sizes)
    in_groups = sizes[gid] >= 2
    lang_joinable = np.bincount(siso[in_groups], minlength=len(ALL_ISOS))
    report["per_language"] = {ALL_ISOS[i]: {"unique_english_sentences": int(lang_rows[i]), "in_multi_language_group": int(lang_joinable[i])}
                              for i in range(len(ALL_ISOS)) if lang_rows[i]}
    multi = np.flatnonzero(in_groups)
    gsize = sizes[gid]
    np.savez(os.path.join(a.out, "best_rows.npz"), shard=arr["shard"][sel][multi], row=arr["row"][sel][multi], key=skey[multi], iso=siso[multi],
             ds=arr["ds"][sel][multi], tier=stier[multi], qe=arr["qe"][sel][multi], tkey=arr["tkey"][sel][multi], enlen=arr["enlen"][sel][multi],
             tlen=arr["tlen"][sel][multi], lic=arr["lic"][sel][multi], group=gid[multi].astype(np.uint32), gsize=gsize[multi].astype(np.uint8))
    kinds = {}
    for t in range(4):
        kinds[prov.KINDS[t]] = int((stier == t).sum())
    report["unique_english_x_language_by_tier"] = kinds
    report["seconds"] = round(time.time() - t0, 1)
    json.dump(report, open(os.path.join(a.out, "pair_matrix_report.json"), "w"), indent=1, ensure_ascii=False)
    # csv matrix (all tiers)
    tot = cnt.sum(0)
    with open(os.path.join(a.out, "pair_matrix_pivot_en.csv"), "w", encoding="utf-8") as f:
        f.write("src\\tgt," + ",".join(SFT77) + "\n")
        for s in SFT77:
            f.write(s + "," + ",".join(str(int(tot[ALL_ISOS.index(s), ALL_ISOS.index(t)])) for t in SFT77) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=1))
    return 0


NOT_AFRICAN = {"eng", "fra", "spa", "por", "deu", "ita", "rus", "ara", "arb", "ary", "arz", "aeb", "acm", "apc", "afb", "ayl", "und", "hin", "ben", "tir_dup"}


def _direct_worker(path: str):
    """Rows of one final_v2 shard that are African <-> African (direct) or African <-> French (pivot), as lists of dicts."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from gemma_mt.afri2afri.text import norm_text
    from gemma_mt.languages import canonical_iso

    light = pq.read_table(path, columns=["task", "source_lang", "targ_lang"])
    sl, tl = light["source_lang"].to_pylist(), light["targ_lang"].to_pylist()
    tk = light["task"].to_pylist()
    cache: Dict[str, object] = {}

    def cano(x):
        if x not in cache:
            cache[x] = "fra" if x == "fra" else ("eng" if x == "eng" else canonical_iso(x))
        return cache[x]

    take = []
    for i, (s, g, t) in enumerate(zip(sl, tl, tk)):
        if t != "translation" or s == g:
            continue
        cs, cg = cano(s), cano(g)
        if cs is None or cg is None or cs == "eng" or cg == "eng":
            continue
        if cs == "fra" and cg == "fra":
            continue
        take.append(i)
    if not take:
        return path, []
    full = pq.read_table(path, columns=["source_dataset", "source_lang", "targ_lang", "input", "response", "metadata"]).take(take).to_pylist()
    out = []
    for r in full:
        cs, cg = cano(r["source_lang"]), cano(r["targ_lang"])
        a, b = norm_text(r["input"] or ""), norm_text(r["response"] or "")
        if not a or not b:
            continue
        unsure = False
        try:
            m = json.loads(r["metadata"] or "{}")
            unsure = bool(m.get("unsure_src") or m.get("unsure_tgt"))
        except Exception:
            pass
        out.append({"src_iso": cs, "tgt_iso": cg, "src": a, "tgt": b, "source_dataset": r["source_dataset"] or "", "unsure": unsure})
    return path, out


def cmd_direct(a) -> int:
    """Direct African<->African rows and African<->French rows of the SFT corpus (final_v2) -> direct.parquet (labels as given; LID checks come later)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from multiprocessing import Pool

    files = sorted(glob.glob(os.path.join(a.final_v2, "*", "part-*.parquet")))
    rows: List[dict] = []
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (path, out) in enumerate(pool.imap_unordered(_direct_worker, files, chunksize=4)):
            rows.extend(out)
            if k % 100 == 0:
                print(f"{k}/{len(files)} shards, {len(rows):,} rows, {time.time() - t0:.0f}s", flush=True)
    cols = {c: [r[c] for r in rows] for c in ("src_iso", "tgt_iso", "src", "tgt", "source_dataset", "unsure")}
    os.makedirs(a.out, exist_ok=True)
    pq.write_table(pa.table(cols), os.path.join(a.out, "direct_raw.parquet"), compression="zstd")
    print("direct rows", len(rows), f"{time.time() - t0:.0f}s")
    return 0



def _lid_worker(args):
    shard_path, rows, model_path = args
    import pyarrow.parquet as pq

    from gemma_mt.rl.lid import GlotLIDClassifier

    lid = GlotLIDClassifier(path=model_path, cache_size=1)
    t = pq.read_table(shard_path, columns=["english", "target"])
    en = t["english"].take(rows).to_pylist()
    tg = t["target"].take(rows).to_pylist()
    r_en = [x[0] for x in lid._predict_many(en, 1)]
    r_tg = [x[0] for x in lid._predict_many(tg, 1)]
    return shard_path, rows, [x[0] for x in r_en], [x[1] for x in r_en], [x[0] for x in r_tg], [x[1] for x in r_tg]


def cmd_lid(a) -> int:
    """GlotLID top-1 of the English and of the African text of every row that sits in a multi-language English group (best_rows.npz)."""
    from multiprocessing import Pool

    z = np.load(os.path.join(a.out, "best_rows.npz"))
    shard, row = z["shard"], z["row"]
    files = sorted(glob.glob(os.path.join(a.pairs, "pairs-*.parquet")))
    labels: List[str] = ["und"]
    lab_id: Dict[str, int] = {"und": 0}
    n = len(shard)
    en_lab, tg_lab = np.zeros(n, np.uint16), np.zeros(n, np.uint16)
    en_p, tg_p = np.zeros(n, np.float16), np.zeros(n, np.float16)
    jobs = []
    for si in np.unique(shard):
        pos = np.flatnonzero(shard == si)
        for s in range(0, len(pos), 400_000):
            part = pos[s:s + 400_000]
            jobs.append((part, (files[int(si)], [int(x) for x in row[part]], a.glotlid)))
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (res, (part, _)) in enumerate(zip(pool.imap(_lid_worker, [j[1] for j in jobs]), jobs)):
            _, _, el, ep, tl, tp = res
            for name_list, dst in ((el, en_lab), (tl, tg_lab)):
                ids = np.empty(len(name_list), np.uint16)
                for i, nm in enumerate(name_list):
                    v = lab_id.get(nm)
                    if v is None:
                        v = lab_id[nm] = len(labels)
                        labels.append(nm)
                    ids[i] = v
                dst[part] = ids
            en_p[part], tg_p[part] = ep, tp
            print(f"lid chunk {k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(a.out, "lid.npz"), en_label=en_lab, en_prob=en_p, tg_label=tg_lab, tg_prob=tg_p)
    json.dump(labels, open(os.path.join(a.out, "lid_labels.json"), "w"))
    print("lid done", n, f"{time.time() - t0:.0f}s")
    return 0



def _dclean_worker(args):
    """LID both sides of a chunk of direct rows."""
    chunk, model_path = args
    from gemma_mt.rl.lid import GlotLIDClassifier

    lid = GlotLIDClassifier(path=model_path, cache_size=1)
    sa = [x[0][0] for x in lid._predict_many([c[0] for c in chunk], 1)]
    sb = [x[0][0] for x in lid._predict_many([c[1] for c in chunk], 1)]
    return sa, sb


def cmd_direct_clean(a) -> int:
    """direct_raw.parquet -> direct_clean.parquet: SFT-77 languages, 'unsure' labels dropped, unordered-pair de-duplication, GlotLID of both sides."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from multiprocessing import Pool

    from gemma_mt.afri2afri.text import hash64, pivot_key

    t = pq.read_table(os.path.join(a.out, "direct_raw.parquet")).to_pydict()
    n = len(t["src"])
    seen = set()
    keep = []
    for i in range(n):
        s, g = t["src_iso"][i], t["tgt_iso"][i]
        if t["unsure"][i] or s == "fra" or g == "fra" or s not in SFT77 or g not in SFT77 or s == g:
            continue
        ha, hb = (s, hash64(pivot_key(t["src"][i]))), (g, hash64(pivot_key(t["tgt"][i])))
        k = (ha, hb) if ha <= hb else (hb, ha)
        if k in seen:
            continue
        seen.add(k)
        keep.append(i)
    print("direct rows kept for LID", len(keep), "of", n, flush=True)
    chunks = [[(t["src"][i], t["tgt"][i]) for i in keep[s:s + 20000]] for s in range(0, len(keep), 20000)]
    la, lb = [], []
    with Pool(a.workers) as pool:
        for sa, sb in pool.imap(_dclean_worker, [(c, a.glotlid) for c in chunks]):
            la.extend(sa)
            lb.extend(sb)
    cols = {"src_iso": [t["src_iso"][i] for i in keep], "tgt_iso": [t["tgt_iso"][i] for i in keep], "src": [t["src"][i] for i in keep],
            "tgt": [t["tgt"][i] for i in keep], "source_dataset": [t["source_dataset"][i] for i in keep], "lid_src": la, "lid_tgt": lb}
    pq.write_table(pa.table(cols), os.path.join(a.out, "direct_clean.parquet"), compression="zstd")
    print("direct_clean rows", len(keep))
    return 0



def _lid_ids(labels: List[str], wanted) -> np.ndarray:
    return np.array([i for i, l in enumerate(labels) if l in wanted], dtype=np.int64)


def row_verdicts(iso: np.ndarray, enlen: np.ndarray, tlen: np.ndarray, en_label: np.ndarray, tg_label: np.ndarray, labels: List[str], pol) -> Dict[str, np.ndarray]:
    """Per-row LID outcome: English side must be eng_Latn (checked from 30 chars), African side must be an accepted label of its language (LID_BAD fails)."""
    from gemma_mt.afri2afri import filters as F

    eng_ids = _lid_ids(labels, {"eng_Latn"})
    en_ok = np.isin(en_label, eng_ids) | (enlen < 30)
    verdict = np.full(len(iso), F.LID_SKIP, np.uint8)
    for i, name in enumerate(ALL_ISOS):
        m = iso == i
        if not m.any() or name not in pol:
            continue
        accepted, min_chars, enabled = pol[name]
        if not enabled:
            continue
        ids = _lid_ids(labels, accepted)
        ok = np.isin(tg_label[m], ids)
        long_enough = tlen[m] >= min_chars
        verdict[m] = np.where(~long_enough, F.LID_SKIP, np.where(ok, F.LID_OK, F.LID_BAD))
    return {"en_ok": en_ok, "verdict": verdict}


def cmd_select(a) -> int:
    from gemma_mt.afri2afri import filters as F, policy as pol_mod, select_pairs as SP
    from gemma_mt.languages import LANGUAGES

    t0 = time.time()
    z = dict(np.load(os.path.join(a.out, "best_rows.npz")))
    L = np.load(os.path.join(a.out, "lid.npz"))
    labels = json.load(open(os.path.join(a.out, "lid_labels.json")))
    cal = json.load(open(a.calibration))
    pol = F.lid_policy(cal, fallback={k: v.glotlid for k, v in LANGUAGES.items()})
    v = row_verdicts(z["iso"], z["enlen"], z["tlen"], L["en_label"], L["tg_label"], labels, pol)
    n = len(z["key"])
    valid = v["en_ok"] & (v["verdict"] != F.LID_BAD) & (z["enlen"] >= a.min_chars)
    report: Dict = {"best_rows_in_multi_language_groups": int(n), "english_not_english": int((~v["en_ok"]).sum()),
                    "african_side_lid_bad": int((v["verdict"] == F.LID_BAD).sum()), "african_side_lid_ok": int((v["verdict"] == F.LID_OK).sum()),
                    "african_side_lid_skipped": int((v["verdict"] == F.LID_SKIP).sum())}
    # per-language side-QE floor on top of the en->X gate (pivot noise adds up)
    qe = z["qe"]
    thr = np.zeros(len(ALL_ISOS), np.float32)
    for i in range(len(ALL_ISOS)):
        m = (z["iso"] == i) & valid
        if m.sum() >= 200:
            thr[i] = np.quantile(qe[m], a.min_side_qe_quantile)
    valid &= qe >= thr[z["iso"]]
    report["rows_valid_after_all_row_filters"] = int(valid.sum())
    report["per_language_lid_bad"] = {ALL_ISOS[i]: int(((z["iso"] == i) & (v["verdict"] == F.LID_BAD)).sum()) for i in range(len(ALL_ISOS)) if ((z["iso"] == i) & (v["verdict"] == F.LID_BAD)).any()}
    # records
    in_sft = np.isin(z["iso"], [ALL_ISOS.index(i) for i in SFT77])
    valid &= in_sft
    rec_piv = SP.records_from_groups(z["group"], z["iso"], z["tier"], z["tkey"], z["key"], valid)
    parts = [rec_piv]
    report["pivot_records_before_dedupe"] = int(len(rec_piv["iso_a"]))
    # direct records
    dpath = os.path.join(a.out, "direct_clean.parquet")
    if os.path.exists(dpath) and not a.no_direct:
        import pyarrow.parquet as pq
        from gemma_mt.afri2afri.text import hash64, pivot_key

        d = pq.read_table(dpath).to_pydict()
        nd = len(d["src"])
        iso_a = np.array([ALL_ISOS.index(x) for x in d["src_iso"]], np.uint8)
        iso_b = np.array([ALL_ISOS.index(x) for x in d["tgt_iso"]], np.uint8)
        ok = np.ones(nd, bool)
        for side, isos, lab in (("src", iso_a, d["lid_src"]), ("tgt", iso_b, d["lid_tgt"])):
            for i in range(nd):
                name = ALL_ISOS[isos[i]]
                entry = pol.get(name)
                if entry and entry[2] and len(d[side][i]) >= entry[1] and lab[i] not in entry[0]:
                    ok[i] = False
        report["direct_rows_after_lid"] = int(ok.sum())
        report["direct_rows_lid_failed"] = int((~ok).sum())
        tier = np.array([prov.tier_of(x) for x in d["source_dataset"]], np.uint8)
        ta = np.array([hash64(pivot_key(x)) for x in d["src"]], np.uint64)
        tb = np.array([hash64(pivot_key(x)) for x in d["tgt"]], np.uint64)
        swap = iso_a > iso_b
        ia, ib = np.where(swap, iso_b, iso_a), np.where(swap, iso_a, iso_b)
        ta2, tb2 = np.where(swap, tb, ta), np.where(swap, ta, tb)
        idx = np.flatnonzero(ok)
        rd = {"iso_a": ia[idx], "iso_b": ib[idx], "cls": SP.CLASS_OF_TIER[tier[idx]].astype(np.int8), "h": SP.mix64(ta2[idx]) * np.uint64(3) + SP.mix64(tb2[idx]),
              "gkey": SP.mix64(ta2[idx] ^ tb2[idx], 99), "ref_a": idx.astype(np.int64), "ref_b": np.full(len(idx), -1, np.int64), "origin": np.ones(len(idx), np.uint8)}
        parts.append(rd)
    mpath = os.path.join(a.out, "mined_candidates.parquet")
    if a.use_mined and os.path.exists(mpath):
        import pyarrow.parquet as pq
        from gemma_mt.afri2afri.text import hash64, pivot_key

        m = pq.read_table(mpath, columns=["src_iso", "tgt_iso", "src", "tgt"]).to_pydict()
        nm = len(m["src"])
        ta = np.array([hash64(pivot_key(x)) for x in m["src"]], np.uint64)
        tb = np.array([hash64(pivot_key(x)) for x in m["tgt"]], np.uint64)
        rm = {"iso_a": np.array([ALL_ISOS.index(x) for x in m["src_iso"]], np.uint8), "iso_b": np.array([ALL_ISOS.index(x) for x in m["tgt_iso"]], np.uint8),
              "cls": np.full(nm, SP.MINED_CLASS, np.int8), "h": SP.mix64(ta) * np.uint64(3) + SP.mix64(tb), "gkey": SP.mix64(ta ^ tb, 99), "ref_a": np.arange(nm, dtype=np.int64),
              "ref_b": np.full(nm, -1, np.int64), "origin": np.full(nm, 2, np.uint8)}
        parts.append(rm)
        report["mined_candidate_records"] = nm
    rec = SP.concat_records(*parts)
    report["records_total"] = int(len(rec["iso_a"]))
    res = SP.select_records(rec, len(ALL_ISOS), a.budget, a.max_epochs, a.val_frac, a.val_cap, a.seed)
    tr, va = res["train"], res["val"]
    report["selected_train_pairs"] = int(len(tr))
    report["selected_val_pairs"] = int(len(va))
    report["dropped"] = res["dropped"]
    pid = res["pid"]
    keep = {k: rec[k][tr] for k in ("iso_a", "iso_b", "cls", "gkey", "ref_a", "ref_b", "origin")}
    keepv = {k: rec[k][va] for k in ("iso_a", "iso_b", "cls", "gkey", "ref_a", "ref_b", "origin")}
    epochs_tr = np.array([res["epochs"].get(int(p), 1.0) for p in pid[tr]], np.float32)
    np.savez(os.path.join(a.out, "selection_train.npz"), epochs=epochs_tr, **keep)
    np.savez(os.path.join(a.out, "selection_val.npz"), **keepv)
    cpe = F.chars_per_english_char(z["enlen"][valid], z["tlen"][valid], z["iso"][valid], len(ALL_ISOS))
    json.dump([None if np.isnan(x) else float(x) for x in cpe], open(os.path.join(a.out, "chars_per_english_char.json"), "w"))
    isos = ALL_ISOS
    info = {f"{isos[p // len(isos)]}-{isos[p % len(isos)]}": dict(v, epochs=res["epochs"].get(p, 0)) for p, v in res["pid_info"].items()}
    json.dump(info, open(os.path.join(a.out, "selection_pairs.json"), "w"))
    report["language_pairs_with_data"] = len(info)
    report["language_pairs_repeated_more_than_once"] = int(sum(1 for v in info.values() if v["epochs"] > 1.0001))
    report["seconds"] = round(time.time() - t0, 1)
    json.dump(report, open(os.path.join(a.out, "select_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))
    return 0



# ----------------------------------------------------------------------------------------------------------------------- materialize
_MW = {}


def _gather_worker(args):
    shard_path, rows = args
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pq.read_table(shard_path, columns=["english", "target"]).take(pa.array(rows, pa.int64()))
    return t["english"].to_pylist(), t["target"].to_pylist()


def _filter_init(bench, extra, cpe, tokenizer_path):
    from gemma_mt.afri2afri import contam as C

    _MW["leak"] = build_leak(bench, extra)
    _MW["cpe"] = np.array([np.nan if x is None else x for x in cpe])
    if tokenizer_path:
        from transformers import AutoTokenizer

        _MW["tok"] = AutoTokenizer.from_pretrained(tokenizer_path)


def _filter_worker(args):
    """Text-level filters of a chunk of pairs; returns (reason or '', src_tokens, tgt_tokens) per pair."""
    from gemma_mt.afri2afri import filters as F

    chunk, tol = args
    leak, cpe, tok = _MW["leak"], _MW["cpe"], _MW.get("tok")
    out = []
    for iso_a, iso_b, ta, tb, en, origin in chunk:
        why = ""
        if not (F.basic_ok(ta) and F.basic_ok(tb)):
            why = "basic"
        elif not (F.script_ok(ta, ALL_ISOS[iso_a]) and F.script_ok(tb, ALL_ISOS[iso_b])):
            why = "script"
        elif F.is_copy(ta, tb):
            why = "copy"
        elif not F.ratio_ok(len(ta), len(tb), F.expected_log_ratio(cpe, iso_a, iso_b), tol if origin == 0 else tol + 0.4):
            why = "length_ratio"
        else:
            for txt in (ta, tb, en):
                if txt and leak.is_leak(txt):
                    why = "benchmark_leak"
                    break
        out.append(why)
    toks_a = toks_b = None
    if tok is not None:
        toks_a = [len(x) for x in tok([c[2] for c in chunk], add_special_tokens=False)["input_ids"]]
        toks_b = [len(x) for x in tok([c[3] for c in chunk], add_special_tokens=False)["input_ids"]]
    return out, toks_a, toks_b


def cmd_materialize(a) -> int:
    """selection_{train,val}.npz -> pairs_{train,val}.parquet: read the texts, apply the text filters, count Gemma tokens."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from multiprocessing import Pool

    from gemma_mt.afri2afri.text import norm_text

    t0 = time.time()
    z = dict(np.load(os.path.join(a.out, "best_rows.npz")))
    names = json.load(open(os.path.join(a.out, "index", "datasets.json"), encoding="utf-8"))
    files = sorted(glob.glob(os.path.join(a.pairs, "pairs-*.parquet")))
    cpe = json.load(open(os.path.join(a.out, "chars_per_english_char.json")))
    d = pq.read_table(os.path.join(a.out, "direct_clean.parquet")).to_pydict() if os.path.exists(os.path.join(a.out, "direct_clean.parquet")) else None
    scan = json.load(open(a.license_scan))["datasets"] if a.license_scan and os.path.exists(a.license_scan) else {}
    lic_names = ix.LICENCE_CLASSES
    pool = Pool(a.workers, initializer=_filter_init, initargs=(a.bench, a.extra_bench, cpe, a.tokenizer))
    summary = {}
    for split in ("train", "val"):
        sel = dict(np.load(os.path.join(a.out, f"selection_{split}.npz")))
        n = len(sel["iso_a"])
        ta, tb, en = [None] * n, [None] * n, [None] * n
        piv = np.flatnonzero(sel["origin"] == 0)
        # gather the pivot texts shard by shard
        for side, ref in (("a", sel["ref_a"]), ("b", sel["ref_b"])):
            sh, rw = z["shard"][ref[piv]], z["row"][ref[piv]]
            jobs, where = [], []
            for si in np.unique(sh):
                m = np.flatnonzero(sh == si)
                jobs.append((files[int(si)], [int(x) for x in rw[m]]))
                where.append(m)
            for m, (e_list, t_list) in zip(where, pool.imap(_gather_worker, jobs)):
                for k, e, t in zip(m, e_list, t_list):
                    i = piv[k]
                    (ta if side == "a" else tb)[i] = norm_text(t)
                    if side == "a":
                        en[i] = norm_text(e)
        md = None
        mi = np.flatnonzero(sel["origin"] == 2)
        if len(mi):
            md = pq.read_table(os.path.join(a.out, "mined_candidates.parquet")).to_pydict()
            for i in mi:
                r = int(sel["ref_a"][i])
                ta[i], tb[i] = md["src"][r], md["tgt"][r]               # stored oriented src_iso < tgt_iso, like the records
        dir_idx = np.flatnonzero(sel["origin"] == 1)
        for i in dir_idx:
            r = int(sel["ref_a"][i])
            s_iso, t_iso = d["src_iso"][r], d["tgt_iso"][r]
            if s_iso == ALL_ISOS[sel["iso_a"][i]]:
                ta[i], tb[i] = d["src"][r], d["tgt"][r]
            else:
                ta[i], tb[i] = d["tgt"][r], d["src"][r]
        chunks = [[(int(sel["iso_a"][i]), int(sel["iso_b"][i]), ta[i], tb[i], en[i], int(sel["origin"][i])) for i in range(s, min(n, s + 20000))] for s in range(0, n, 20000)]
        reasons, tka, tkb = [], [], []
        for out, xa, xb in pool.imap(_filter_worker, [(c, a.ratio_tol) for c in chunks]):
            reasons.extend(out)
            tka.extend(xa or [0] * len(out))
            tkb.extend(xb or [0] * len(out))
        reasons = np.array(reasons)
        from collections import Counter

        summary[split] = {"selected": int(n), "dropped_by_text_filters": dict(Counter(r for r in reasons.tolist() if r)), "kept": int((reasons == "").sum())}
        ok = np.flatnonzero(reasons == "")
        from gemma_mt.afri2afri import policy as _pol

        cls_names = list(_pol.CLASSES)
        cols = {k: [] for k in ("src_iso", "tgt_iso", "src", "tgt", "pivot_en", "origin", "provenance_kind", "src_dataset", "tgt_dataset", "licence", "qe_src", "qe_tgt",
                                "group_size", "group_key", "src_tok", "tgt_tok", "epochs", "prompt_form")}
        ep = sel.get("epochs")
        for i in ok:
            ia, ib = int(sel["iso_a"][i]), int(sel["iso_b"][i])
            cols["src_iso"].append(ALL_ISOS[ia]); cols["tgt_iso"].append(ALL_ISOS[ib]); cols["src"].append(ta[i]); cols["tgt"].append(tb[i])
            cols["src_tok"].append(tka[i]); cols["tgt_tok"].append(tkb[i]); cols["group_key"].append(format(int(sel["gkey"][i]), "016x"))
            cols["provenance_kind"].append(cls_names[int(sel["cls"][i])])
            cols["epochs"].append(float(ep[i]) if ep is not None else 1.0)
            cols["prompt_form"].append(PF.prompt_form_of(PF.example_id_of(cols["group_key"][-1], cols["src_iso"][-1], cols["tgt_iso"][-1])))
            if sel["origin"][i] == 2:
                r = int(sel["ref_a"][i])
                cols["pivot_en"].append(None); cols["origin"].append("mined_nllb")
                cols["src_dataset"].append(md["source_dataset"][r]); cols["tgt_dataset"].append(md["source_dataset"][r])
                cols["licence"].append("unspecified")
                cols["qe_src"].append(float("nan")); cols["qe_tgt"].append(float("nan")); cols["group_size"].append(2)
            elif sel["origin"][i] == 0:
                ra, rb = int(sel["ref_a"][i]), int(sel["ref_b"][i])
                cols["pivot_en"].append(en[i]); cols["origin"].append("pivot_en")
                cols["src_dataset"].append(names[int(z["ds"][ra])]); cols["tgt_dataset"].append(names[int(z["ds"][rb])])
                cols["licence"].append(prov.combine_licences(lic_names[int(z["lic"][ra])], lic_names[int(z["lic"][rb])]))
                cols["qe_src"].append(float(z["qe"][ra])); cols["qe_tgt"].append(float(z["qe"][rb])); cols["group_size"].append(int(z["gsize"][ra]))
            else:
                r = int(sel["ref_a"][i])
                dsn = d["source_dataset"][r]
                cols["pivot_en"].append(None); cols["origin"].append("direct")
                cols["src_dataset"].append(dsn); cols["tgt_dataset"].append(dsn)
                lc = scan.get(prov.repo_of(dsn))
                cols["licence"].append(prov.licence_class(lc.get("declared")) if lc else "unspecified")
                cols["qe_src"].append(float("nan")); cols["qe_tgt"].append(float("nan")); cols["group_size"].append(2)
        pq.write_table(pa.table({**cols, "qe_src": pa.array(cols["qe_src"], pa.float32()), "qe_tgt": pa.array(cols["qe_tgt"], pa.float32()),
                                 "group_size": pa.array(cols["group_size"], pa.int16()), "src_tok": pa.array(cols["src_tok"], pa.int32()),
                                 "tgt_tok": pa.array(cols["tgt_tok"], pa.int32()), "epochs": pa.array(cols["epochs"], pa.float32())}),
                       os.path.join(a.out, f"pairs_{split}.parquet"), compression="zstd")
        print(split, summary[split], f"{time.time() - t0:.0f}s", flush=True)
    pool.close()
    json.dump(summary, open(os.path.join(a.out, "materialize_report.json"), "w"), indent=1)
    return 0



# ----------------------------------------------------------------------------------------------------------------------- windows
def cmd_windows(a) -> int:
    """pairs_{train,val}.parquet -> windows_{train,val}.parquet: multi-sentence windows per language pair, repeated `epochs` times with new groupings."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from gemma_mt.afri2afri import windows as W
    from gemma_mt.afri2afri.provenance import combine_licences

    t0 = time.time()
    for split in ("train", "val"):
        d = pq.read_table(os.path.join(a.out, f"pairs_{split}.parquet")).to_pydict()
        n = len(d["src"])
        rng = np.random.default_rng(a.seed + (0 if split == "train" else 1))
        buckets: Dict[tuple, List[int]] = {}
        for i in range(n):
            buckets.setdefault((d["src_iso"][i], d["tgt_iso"][i]), []).append(i)
        cols = {k: [] for k in ("src_iso", "tgt_iso", "src", "tgt", "n_pairs", "src_tokens", "tgt_tokens", "bin", "punct_added", "provenance_kind", "origin", "licence", "pass_", "prompt_form")}
        for (sa, sb), idxs in sorted(buckets.items()):
            idxs = np.array(idxs)
            e = float(d["epochs"][int(idxs[0])]) if split == "train" else 1.0
            passes = int(np.floor(e + 1e-6))
            frac = e - passes
            plan = [(k, 1.0) for k in range(max(passes, 1))] + ([(passes, frac)] if frac > 0.05 and passes >= 1 else [])
            used = {b: 0 for b in W.CYCLE}
            for k, share in plan:
                order = rng.permutation(len(idxs))
                if share < 1.0:
                    order = order[: max(1, int(round(share * len(idxs))))]
                sel = idxs[order]
                # runs: same origin and provenance kind stay together (a window never mixes a Bible verse with a government sentence)
                runs: Dict[tuple, List[int]] = {}
                for i in sel:
                    runs.setdefault((d["origin"][i], d["provenance_kind"][i], d["prompt_form"][i]), []).append(int(i))
                for (origin, kind, pform), ii in runs.items():
                    ws = W.build_windows([d["src"][i] for i in ii], [d["tgt"][i] for i in ii], [d["src_tok"][i] for i in ii], [d["tgt_tok"][i] for i in ii], rng, used)
                    lic = "cc0"
                    for i in ii:
                        lic = combine_licences(lic, d["licence"][i])
                    for w in ws:
                        for key in ("src", "tgt", "n_pairs", "src_tokens", "tgt_tokens", "bin", "punct_added"):
                            cols[key].append(w[key])
                        cols["src_iso"].append(sa); cols["tgt_iso"].append(sb); cols["provenance_kind"].append(kind); cols["origin"].append(origin)
                        cols["licence"].append(lic); cols["pass_"].append(k); cols["prompt_form"].append(pform)
        pq.write_table(pa.table(cols), os.path.join(a.out, f"windows_{split}.parquet"), compression="zstd")
        print(split, "pairs", n, "language pairs", len(buckets), "windows", len(cols["src"]), f"{time.time() - t0:.0f}s", flush=True)
    return 0


# ----------------------------------------------------------------------------------------------------------------------- pack
_PK = {}


def _pack_init(tok_path):
    from transformers import AutoTokenizer

    _PK["tok"] = AutoTokenizer.from_pretrained(tok_path)


def _pack_worker(items):
    from gemma_mt.afri2afri import packing as PK

    return PK.encode_batch(_PK["tok"], items)


def cmd_pack(a) -> int:
    import pyarrow.parquet as pq
    from multiprocessing import Pool
    from transformers import AutoTokenizer

    from gemma_mt import format as fmt
    from gemma_mt.afri2afri import packing as PK, prompt as P
    from gemma_mt.data.prepare_sft import tokenizer_fingerprint

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    fmt.assert_template_matches_reference(tok)
    stop = fmt.resolve_stop_tokens(tok)
    os.makedirs(os.path.join(a.out, "packed"), exist_ok=True)
    meta_splits = {}
    dropped: Dict[str, int] = {}
    pool = Pool(a.workers, initializer=_pack_init, initargs=(a.tokenizer,))
    for split in ("train", "val"):
        d = pq.read_table(os.path.join(a.out, f"windows_{split}.parquet")).to_pydict()
        n = len(d["src"])
        items, ids_meta = [], []
        for i in range(n):
            sa, sb = d["src_iso"][i], d["tgt_iso"][i]
            explicit = d["prompt_form"][i] == "explicit"          # one form per window, both directions (the form is a function of the pair id)
            items.append((d["src"][i], sa if explicit else None, sb, d["tgt"][i]))
            ids_meta.append((ALL_ISOS.index(sb), ALL_ISOS.index(sa), 0 if explicit else 1))
            items.append((d["tgt"][i], sb if explicit else None, sa, d["src"][i]))
            ids_meta.append((ALL_ISOS.index(sa), ALL_ISOS.index(sb), 0 if explicit else 1))
        dir_meta = [P.DIRECTION_X2X] * len(items)
        rp = os.path.join(a.out, f"replay_windows_{split}.parquet")
        if a.with_replay and os.path.exists(rp):
            r = pq.read_table(rp).to_pydict()
            eng = len(ALL_ISOS)
            for i in range(len(r["iso"])):
                iso = r["iso"][i]
                ii = ALL_ISOS.index(iso)
                explicit = r["prompt_form"][i] == "explicit"
                items.append((r["english"][i], "eng" if explicit else None, iso, r["target"][i]))      # English -> X (implicit form = the original en->X prompt)
                ids_meta.append((ii, eng, 0 if explicit else 1)); dir_meta.append(0)
                items.append((r["target"][i], iso if explicit else None, "eng", r["english"][i]))      # X -> English
                ids_meta.append((ii, ii, 0 if explicit else 1)); dir_meta.append(1)
        store = PK.RawStore(os.path.join(a.out, "packed", f"_tmp_{split}"))
        chunks = [items[s:s + 4000] for s in range(0, len(items), 4000)]
        k = 0
        for res in pool.imap(_pack_worker, chunks):
            for (ids, pl, tl) in res:
                if len(ids) > a.max_sample_tokens:
                    dropped[f"too_long_{split}"] = dropped.get(f"too_long_{split}", 0) + 1
                    k += 1
                    continue
                store.add(ids, pl, tl, *ids_meta[k], direction=dir_meta[k])
                k += 1
        meta_splits[split] = PK.write_split(store, os.path.join(a.out, "packed", split), a.seq_len, a.buffer_size, a.seed + (0 if split == "train" else 1))
        try:
            os.rmdir(os.path.join(a.out, "packed", f"_tmp_{split}"))
        except OSError:
            pass
        m = meta_splits[split]
        print(split, f"{m['n_samples']:,} samples {m['n_packs']:,} packs {m['n_tokens']:,} tokens ({m['target_tokens']:,} target) efficiency {m['packing_efficiency']:.3f} {time.time() - t0:.0f}s", flush=True)
    pool.close()
    meta = {"format_version": 1, "format_x2x_version": P.FORMAT_VERSION_X2X, "dtype": "uint32", "seq_len": a.seq_len,
            "instruction_template": fmt.INSTRUCTION_TEMPLATE, "instruction_template_from": P.INSTRUCTION_TEMPLATE_FROM, "end_of_turn": fmt.END_OF_TURN,
            "ignore_index": fmt.IGNORE_INDEX,
            "tokenizer": {"name": getattr(tok, "name_or_path", a.tokenizer), "requested": "google/gemma-3-270m-it", "revision": None, "fingerprint": tokenizer_fingerprint(tok),
                          "vocab_size": len(tok), "bos_id": tok.bos_token_id, "eos_id": tok.eos_token_id, "pad_id": tok.pad_token_id, "end_of_turn_id": stop.end_of_turn_id},
            "iso_table": list(ALL_ISOS), "english_source_id": len(ALL_ISOS), "directions": (["x2x", "en2x", "x2en"] if a.with_replay else ["x2x"]), "direction_codes": {"en2x": 0, "x2en": 1, "x2x": 2}, "prompt_mixture": {"explicit": "Translate the text from {source} to {target}:  {text}", "implicit": "Translate the text to {target}:  {text}", "explicit_share": 0.5, "rule": "prompt.prompt_form_of(group_key|src_iso|tgt_iso)", "codes": {"explicit": 0, "implicit": 1}},
            "pivot_tag": fmt.PIVOT_ISO, "dropped": dropped, "splits": meta_splits, "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(a.out, "packed", "meta.json"), "w"), indent=2, ensure_ascii=False)
    return 0



# ----------------------------------------------------------------------------------------------------------------------- export
KNOWN_DECLARED = {"HuggingFaceFW/fineweb-2": "odc-by-1.0", "HPLT/HPLT2.0_cleaned": "cc0-1.0", "allenai/c4": "odc-by-1.0"}      # as declared on the Hub cards (checked 2026-10-20)
# the 99 `michsethowusu/english-<language>_sentence-pairs_mt560` cards all declare cc-by-4.0 (audit: data_pipeline/sft_new/audit_mich.py, michsethowusu_families.json),
# but `license_scan.json` did not read it from the card metadata; the upstream is OPUS MT560 (JW300, Bible, ... : the card names no per-corpus licence)
MT560_RX = "michsethowusu/english-.*_sentence-pairs_mt560$"


def _sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cmd_export(a) -> int:
    """Stage the files for the Hub: every row keeps `source_dataset` (repo id(s)) and `licence` (the licence DECLARED on the source's card, 'unknown' when none; a pivot pair
    joins the two sides' licences with ' + '), plus the coarse `licence_class` of the build. Rows whose source repo is flagged private in the licence scan are written to
    export/private_sources/ instead (not staged for the public repo) unless --keep-private-sources. Nothing is uploaded here."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    scan = json.load(open(a.license_scan))["datasets"]

    def info(name: str):
        repo = prov.repo_of(name)
        d = scan.get(repo)
        import re

        decl = (d or {}).get("declared") or KNOWN_DECLARED.get(repo) or ("cc-by-4.0" if re.match(MT560_RX, repo) else "unknown")
        return repo, decl, bool((d or {}).get("private"))

    stage = os.path.join(a.out, "export", "stage")
    priv = os.path.join(a.out, "export", "private_sources")
    os.makedirs(stage, exist_ok=True)
    os.makedirs(priv, exist_ok=True)
    manifest: Dict = {"files": {}, "private_source_rows": {}, "licence_rows": {}, "source_dataset_rows": {}}
    cache: Dict[tuple, tuple] = {}

    def lookup(x: str, y: str):
        k = (x, y)
        if k not in cache:
            rx, lx, px = info(x)
            ry, ly, py = info(y) if y else (rx, lx, px)
            repo = rx if rx == ry else f"{rx} + {ry}"
            lic = lx if (lx == ly or not y) else f"{lx} + {ly}"
            cache[k] = (repo, lic, px or py)
        return cache[k]

    def emit(table, name, kind):
        n = table.num_rows
        if kind == "sft":
            a_, b_ = table["src_dataset"].to_pylist(), table["tgt_dataset"].to_pylist()
            res = [lookup(x, y) for x, y in zip(a_, b_)]
            table = table.drop(["src_dataset", "tgt_dataset"])
        else:
            a_ = table["source_dataset"].to_pylist()
            res = [lookup(x, "") for x in a_]
            table = table.drop(["source_dataset"])
        table = table.rename_columns([("licence_class" if c == "licence" else c) for c in table.column_names])
        table = table.append_column("source_dataset", pa.array([r[0] for r in res])).append_column("licence", pa.array([r[1] for r in res]))
        is_priv = pa.array([r[2] for r in res])
        keep = pc.invert(is_priv) if not a.keep_private_sources else pa.array([True] * n)
        for tbl, d, tag in ((table.filter(keep), stage, "stage"), (table.filter(pc.invert(keep)), priv, "private_sources")):
            if tbl.num_rows:
                path = os.path.join(d, name)
                pq.write_table(tbl, path, compression="zstd")
                manifest["files"][f"{tag}/{name}"] = {"rows": tbl.num_rows, "bytes": os.path.getsize(path), "sha256": _sha256(path)}
                if tag == "stage":
                    lic = pc.value_counts(tbl["licence"]).to_pylist()
                    manifest["licence_rows"][name] = {x["values"]: x["counts"] for x in sorted(lic, key=lambda x: -x["counts"])}
                    sd = pc.value_counts(tbl["source_dataset"]).to_pylist()
                    manifest["source_dataset_rows"][name] = {x["values"]: x["counts"] for x in sorted(sd, key=lambda x: -x["counts"])}
                else:
                    sd = pc.value_counts(tbl["source_dataset"]).to_pylist()
                    manifest["private_source_rows"][name] = {x["values"]: x["counts"] for x in sorted(sd, key=lambda x: -x["counts"])}

    for split_name in ("train", "val"):
        p = os.path.join(a.out, f"pairs_{split_name}.parquet")
        if os.path.exists(p):
            emit(pq.read_table(p), f"sft_pairs_{split_name}.parquet", "sft")
    if a.rl_dir and os.path.isdir(a.rl_dir):
        tabs = [pq.read_table(f) for f in sorted(glob.glob(os.path.join(a.rl_dir, "*.parquet")))]
        if tabs:
            emit(pa.concat_tables(tabs), "rl_prompts.parquet", "rl")
    # small random samples of every staged file (the first thing to push, to check that the repo works)
    sdir = os.path.join(a.out, "export", "sample")
    os.makedirs(sdir, exist_ok=True)
    rng = np.random.default_rng(0)
    for fn in sorted(os.listdir(stage)):
        if fn.endswith(".parquet"):
            t = pq.read_table(os.path.join(stage, fn))
            idx = np.sort(rng.choice(t.num_rows, min(a.sample_rows, t.num_rows), replace=False))
            pq.write_table(t.take(pa.array(idx)), os.path.join(sdir, "sample_" + fn), compression="zstd")
    repos = {}
    for name, d in manifest["source_dataset_rows"].items():
        for k, n in d.items():
            for r in k.split(" + "):
                repos.setdefault(name, {})
                repos[name][r] = repos[name].get(r, 0) + n
    manifest["source_repos_rows_touched"] = {k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in repos.items()}
    json.dump(manifest, open(os.path.join(a.out, "export", "MANIFEST.json"), "w"), indent=1)
    print(json.dumps({"files": manifest["files"], "licence_rows": {k: dict(list(v.items())[:12]) for k, v in manifest["licence_rows"].items()}}, indent=1))
    return 0



def cmd_add_form(a) -> int:
    """Add (or recompute) the `prompt_form` column of pairs_{train,val}.parquet without redoing `materialize` (same function as materialize uses)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    for split in ("train", "val"):
        path = os.path.join(a.out, f"pairs_{split}.parquet")
        t = pq.read_table(path)
        if "prompt_form" in t.column_names:
            t = t.drop(["prompt_form"])
        gk, si, ti = t["group_key"].to_pylist(), t["src_iso"].to_pylist(), t["tgt_iso"].to_pylist()
        form = [PF.prompt_form_of(PF.example_id_of(g, s_, t_), a.explicit_share) for g, s_, t_ in zip(gk, si, ti)]
        t = t.append_column("prompt_form", pa.array(form).dictionary_encode())
        pq.write_table(t, path + ".tmp", compression="zstd")
        os.replace(path + ".tmp", path)
        n_exp = sum(1 for f in form if f == "explicit")
        print(split, len(form), "explicit", n_exp, f"{100 * n_exp / max(1, len(form)):.2f}%")
    return 0



# ----------------------------------------------------------------------------------------------------------------------- replay
def cmd_replay(a) -> int:
    """en->X / X->en REPLAY windows for the same packed dataset (owner 2026-10-20: 20-30 % replay). Streams `gemma/windows_sft`, keeps a window with the probability that makes the number of windows per
    language follow UniMax over the language window counts (budget = replay_share / (1 - replay_share) x the X->Y samples / 2), drops windows that match a held-out evaluation sentence, and writes
    replay_windows_{train,val}.parquet (columns iso, english, target, prompt_form, licence, source_dataset). Each window gives TWO samples at pack time: English -> X and X -> English."""
    import glob as _glob
    import pyarrow as pa
    import pyarrow.parquet as pq

    from gemma_mt.afri2afri import policy
    from gemma_mt.afri2afri.text import split_key_fraction

    t0 = time.time()
    files = sorted(_glob.glob(os.path.join(a.windows, "windows-*.parquet")))
    counts: Dict[str, int] = {}
    for f in files:
        for iso, k in zip(*np.unique(np.array(pq.read_table(f, columns=["iso"])["iso"].to_pylist()), return_counts=True)):
            counts[iso] = counts.get(iso, 0) + int(k)
    langs = [l for l in SFT77 if counts.get(l)]
    meta = json.load(open(os.path.join(a.out, "packed", "meta.json")))
    n_x2x = meta["splits"]["train"]["n_samples"]
    budget = a.replay_share / (1.0 - a.replay_share) * n_x2x / 2.0
    alloc = dict(zip(langs, policy.unimax([counts[l] for l in langs], budget, a.max_epochs)))
    keep_p = {l: min(1.0, alloc[l] / counts[l]) for l in langs}
    print("replay budget windows", int(budget), "languages", len(langs), "total available", sum(counts.values()), flush=True)
    li = contam.LeakIndex()
    li.add_all(contam.iter_benchmark_texts(a.bench, a.extra_bench))
    cols = {"iso": [], "english": [], "target": [], "prompt_form": [], "licence": [], "source_dataset": []}
    cols_v = {k: [] for k in cols}
    drop = Counter()
    for f in files:
        t = pq.read_table(f, columns=["english", "target", "iso", "source_dataset", "licence"]).to_pydict()
        for en, tg, iso, ds, lic in zip(t["english"], t["target"], t["iso"], t["source_dataset"], t["licence"]):
            p_keep = keep_p.get(iso)
            if p_keep is None:
                drop["language_not_in_sft77"] += 1
                continue
            u = split_key_fraction(en + "\x1f" + iso, a.seed, "replay-select-v1")
            if u >= p_keep:
                continue
            if li.is_leak(en) or li.is_leak(tg):
                drop["held_out"] += 1
                continue
            val = split_key_fraction(en, a.seed, "replay-val-v1") < a.val_frac
            d = cols_v if val else cols
            d["iso"].append(iso); d["english"].append(en); d["target"].append(tg); d["licence"].append(lic); d["source_dataset"].append(ds)
            d["prompt_form"].append(PF.prompt_form_of(f"replay|{iso}|{en[:200]}"))
        print(os.path.basename(f), len(cols["iso"]), f"{time.time() - t0:.0f}s", flush=True)
    for name, d in (("train", cols), ("val", cols_v)):
        pq.write_table(pa.table(d), os.path.join(a.out, f"replay_windows_{name}.parquet"), compression="zstd")
    json.dump({"replay_windows_train": len(cols["iso"]), "replay_windows_val": len(cols_v["iso"]), "dropped": dict(drop), "per_language_available": counts,
               "per_language_alloc": {k: round(v) for k, v in alloc.items()}}, open(os.path.join(a.out, "replay_report.json"), "w"), indent=1)
    print("replay windows", len(cols["iso"]), len(cols_v["iso"]), dict(drop))
    return 0



# ----------------------------------------------------------------------------------------------------------------------- mined-sample
def _mined_worker(args):
    path, k_max, q, cpe, model_path, calib, seed = args
    import pyarrow as pa
    import pyarrow.parquet as pq

    from gemma_mt.afri2afri import filters as F
    from gemma_mt.rl.lid import GlotLIDClassifier

    t = pq.read_table(path).to_pydict()
    n = len(t["src"])
    sc = np.asarray(t["score"], np.float32)
    thr = float(np.quantile(sc, q)) if n else 0.0
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(sc >= thr)
    rng.shuffle(idx)
    cal = json.load(open(calib))
    pol = F.lid_policy(cal, fallback={k: v.glotlid for k, v in LANGUAGES.items()})
    a_iso, b_iso = t["src_iso"][0], t["tgt_iso"][0]
    ia, ib = ALL_ISOS.index(a_iso), ALL_ISOS.index(b_iso)
    cpe_arr = np.array([np.nan if x is None else x for x in cpe])
    exp = F.expected_log_ratio(cpe_arr, ia, ib)
    cand, seen = [], set()
    for i in idx:
        s_, t_ = t["src"][i], t["tgt"][i]
        if (s_, t_) in seen or len(s_) < 8 or len(t_) < 8:
            continue
        seen.add((s_, t_))
        if not (F.basic_ok(s_) and F.basic_ok(t_)) or not (F.script_ok(s_, a_iso) and F.script_ok(t_, b_iso)) or F.is_copy(s_, t_) or not F.ratio_ok(len(s_), len(t_), exp, 1.2):
            continue
        cand.append(int(i))
        if len(cand) >= int(k_max * 1.25):
            break
    lid = GlotLIDClassifier(path=model_path, cache_size=1)
    la = [x[0][0] for x in lid._predict_many([t["src"][i] for i in cand], 1)]
    lb = [x[0][0] for x in lid._predict_many([t["tgt"][i] for i in cand], 1)]
    keep = []
    for i, x, y in zip(cand, la, lb):
        if F.lid_verdict(a_iso, len(t["src"][i]), x, pol) == F.LID_BAD or F.lid_verdict(b_iso, len(t["tgt"][i]), y, pol) == F.LID_BAD:
            continue
        keep.append(i)
        if len(keep) >= k_max:
            break
    return {"src_iso": [a_iso] * len(keep), "tgt_iso": [b_iso] * len(keep), "src": [t["src"][i] for i in keep], "tgt": [t["tgt"][i] for i in keep],
            "score": [float(sc[i]) for i in keep], "source_dataset": [t["source_dataset"][i] for i in keep]}, n, len(cand), len(keep)


def cmd_mined_sample(a) -> int:
    """Candidates for the SFT from the NLLB-mined pair shards: top `score_quantile` of every shard by the margin score, the usual text filters, GlotLID of both sides, at most `per_pair` rows per shard."""
    import glob as _glob
    import pyarrow as pa
    import pyarrow.parquet as pq
    from multiprocessing import Pool

    cpe = json.load(open(os.path.join(a.out, "chars_per_english_char.json")))
    files = [f for f in sorted(_glob.glob(os.path.join(a.mined_dir, "*.parquet"))) if os.path.basename(f).split("-")[0] != os.path.basename(f).split("-")[1].split("__")[0].split(".")[0]]
    jobs = [(f, a.per_pair, a.score_quantile, cpe, a.glotlid, a.calibration, a.seed + i) for i, f in enumerate(files)]
    t0 = time.time()
    cols = {k: [] for k in ("src_iso", "tgt_iso", "src", "tgt", "score", "source_dataset")}
    stats = {}
    with Pool(a.workers) as pool:
        for k, (d, n, nc, nk) in enumerate(pool.imap_unordered(_mined_worker, jobs)):
            for c in cols:
                cols[c].extend(d[c])
            if d["src_iso"]:
                stats[f"{d['src_iso'][0]}-{d['tgt_iso'][0]}"] = {"rows": n, "candidates": nc, "kept": nk}
            if k % 25 == 0:
                print(f"{k + 1}/{len(files)} shards, {len(cols['src']):,} candidates {time.time() - t0:.0f}s", flush=True)
    pq.write_table(pa.table(cols), os.path.join(a.out, "mined_candidates.parquet"), compression="zstd")
    json.dump(stats, open(os.path.join(a.out, "mined_candidates_report.json"), "w"))
    print("mined candidates", len(cols["src"]), "from", len(files), "shards")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("index")
    p.add_argument("--pairs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_index)
    p = sp.add_parser("matrix")
    p.add_argument("--out", required=True)
    p.add_argument("--bench", required=True)
    p.add_argument("--extra-bench", nargs="*", default=[])
    p.set_defaults(fn=cmd_matrix)
    p = sp.add_parser("lid")
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--glotlid", required=True)
    p.add_argument("--workers", type=int, default=10)
    p.set_defaults(fn=cmd_lid)
    p = sp.add_parser("export")
    p.add_argument("--out", required=True)
    p.add_argument("--rl-dir", default=None, help="directory of the RL prompt parquet files (rl_prompts_x2x)")
    p.add_argument("--license-scan", required=True)
    p.add_argument("--keep-private-sources", action="store_true")
    p.add_argument("--sample-rows", type=int, default=300)
    p.set_defaults(fn=cmd_export)
    p = sp.add_parser("add-form")
    p.add_argument("--out", required=True)
    p.add_argument("--explicit-share", type=float, default=0.5)
    p.set_defaults(fn=cmd_add_form)
    p = sp.add_parser("mined-sample")
    p.add_argument("--out", required=True)
    p.add_argument("--mined-dir", required=True)
    p.add_argument("--glotlid", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--per-pair", type=int, default=8000)
    p.add_argument("--score-quantile", type=float, default=0.8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_mined_sample)
    p = sp.add_parser("replay")
    p.add_argument("--out", required=True)
    p.add_argument("--windows", default="/workspace/sft_build/gemma/windows_sft")
    p.add_argument("--bench", required=True)
    p.add_argument("--extra-bench", nargs="*", default=[])
    p.add_argument("--replay-share", type=float, default=0.25)
    p.add_argument("--max-epochs", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_replay)
    p = sp.add_parser("windows")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_windows)
    p = sp.add_parser("pack")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--max-sample-tokens", type=int, default=4096)
    p.add_argument("--with-replay", action="store_true", help="also pack replay_windows_{train,val}.parquet (en->X and X->en samples, direction codes 0 / 1)")
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_pack)
    p = sp.add_parser("materialize")
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--bench", required=True)
    p.add_argument("--extra-bench", nargs="*", default=[])
    p.add_argument("--tokenizer", default=None, help="path of google/gemma-3-270m-it (for the Gemma token counts)")
    p.add_argument("--license-scan", default=None)
    p.add_argument("--ratio-tol", type=float, default=0.8)
    p.add_argument("--workers", type=int, default=10)
    p.set_defaults(fn=cmd_materialize)
    p = sp.add_parser("select")
    p.add_argument("--out", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--budget", type=float, default=8_000_000, help="sentence-pair draws over all language pairs (UniMax); each draw gives two directional samples")
    p.add_argument("--max-epochs", type=float, default=3.0)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--val-cap", type=int, default=40)
    p.add_argument("--min-side-qe-quantile", type=float, default=0.30)
    p.add_argument("--min-chars", type=int, default=8)
    p.add_argument("--no-direct", action="store_true")
    p.add_argument("--use-mined", action="store_true", help="add mined_candidates.parquet (class mined_nllb, cap 40 % of a pair's draws)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_select)
    p = sp.add_parser("direct-clean")
    p.add_argument("--out", required=True)
    p.add_argument("--glotlid", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.set_defaults(fn=cmd_direct_clean)
    p = sp.add_parser("direct")
    p.add_argument("--final-v2", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.set_defaults(fn=cmd_direct)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
