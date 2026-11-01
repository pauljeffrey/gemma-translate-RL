#!/usr/bin/env python3
"""STEP (c): token-balanced SFT windows, exactly the definitions of the old gemma_windows.py (same bins, budgets, bin choice, terminator rules, pcm oversampling).

    python windows.py --src data data_devhalf specialist --out windows_sft [--scores scores_dir] [--stats-only] [--limit-pairs N]

Input: the clean pair shards (columns english, target, iso, source_dataset, licence, flipped [, key]). Several directories are united in the given order and exactly
deduplicated by `key` (a pair that is in data/ and in specialist/ is used once, with the licence of the first). If --scores is given the QE score of each pair (matched by key) is
carried into the window (mean over the window's pairs) and pairs flagged by --keep-file are dropped (the gate); without scores `qe` is NaN (the UNGATED projection).

Differences to the old script (all forced by the D2 output; each is listed in README_chain.md):
  * the run key is (iso, FULL source_dataset, flipped) instead of (iso, source_dataset.split(":")[0], flipped): the specialist set has sources such as `weak-repo:<id>` whose
    base name would merge unrelated sources;
  * source order: the D2 shards keep the row order of each source file (dedupe keeps the first occurrence), so consecutive pairs of one run are consecutive source rows with gaps
    where rows were removed; sft-new rows that came in both directions were unified by the dedupe (first direction kept, `flipped` records it). No document order is added or assumed;
  * tokenisation: `tokenizers` JSON of Davlan/afro-xlmr-large-76L (same vocabulary as the old transformers tokenizer; counts = ids without the two specials, +2 added as before).
--stats-only writes per-window counts (no text) plus Gemma-3 token counts (--gemma-tokenizer) for the projection.
"""
import argparse, collections, glob, json, os, sys, time
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BINS, CYCLE, TERM, bin_of, key_of, shard_files, is_nc, assert_clean_src, parse_nc_allow, nc_exception_tag, nc_blocked, owner_tag


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores", nargs="*", default=None, help="score directories (CPU and GPU shards; first score per key wins). Optional: only needed for the per-window mean QE column; the gate decision comes from --keep-file")
    ap.add_argument("--keep-file", default=None, help="npy/parquet with the keys to KEEP (gate output); default: keep everything")
    ap.add_argument("--afro-tokenizer", default="/workspace/work/tok_afro/tokenizer.json")
    ap.add_argument("--gemma-tokenizer", default=None)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--shard-rows", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oversample", default="pcm:2")
    ap.add_argument("--nc-allow", nargs="*", default=[], help="owner exceptions: NC sources that stay in training, repo[=tag] (e.g. howard-nlp/ibom-mt); their windows carry nc_exception=<tag>")
    ap.add_argument("--include-nc", action="store_true", help="TEST ONLY: keep rows of licence class non_commercial (default: they are removed before dedupe and windowing, Amendment 4)")
    ap.add_argument("--limit-pairs", type=int, default=0, help="test: stop after this many pairs")
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--gemma-sample", type=float, default=1.0, help="share of the pairs (by key hash) whose Gemma tokens are counted exactly; the rest are estimated per language from the sampled ratio gemma/afro (written to gemma_ratio.json). 1.0 = all pairs")
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    tka = Tokenizer.from_file(a.afro_tokenizer)
    tkg = Tokenizer.from_file(a.gemma_tokenizer) if a.gemma_tokenizer else None
    assert_clean_src(a.src)
    allow = parse_nc_allow(a.nc_allow)
    os.makedirs(a.out, exist_ok=True)
    for f in glob.glob(f"{a.out}/*.parquet"):
        os.remove(f)
    rng = np.random.default_rng(a.seed)
    used = collections.defaultdict(collections.Counter)
    cur = {}
    buf = collections.defaultdict(list)
    shard = 0
    stats = collections.Counter()
    by_iso_bin = collections.defaultdict(collections.Counter)
    seen = set()
    nc_by_iso = collections.Counter(); exc_pairs = collections.Counter(); owner_win = collections.Counter()
    ratio = collections.defaultdict(lambda: [0, 0, 0, 0, 0])     # iso -> [afro_en, gemma_en, afro_tg, gemma_tg, n] on the Gemma-counted sample
    qe_map = None
    if a.scores:
        import pyarrow.dataset as ds
        qe_map = {}
        for sd in a.scores:
            t = ds.dataset(sd, format="parquet").to_table(columns=["key", "qe"])
            for k_, q_ in zip(t["key"].to_pylist(), t["qe"].to_pylist()):
                qe_map.setdefault(k_, q_)
    keep = None
    if a.keep_file:
        keep = set(np.load(a.keep_file).tolist()) if a.keep_file.endswith(".npy") else set(pq.read_table(a.keep_file)["key"].to_pylist())

    def new_budget(iso):
        b = min(CYCLE, key=lambda x: (used[iso][x], CYCLE.index(x)))
        lo, hi = BINS[b]
        budget = int(rng.integers(lo, hi))
        used[iso][b] += budget
        return budget, b

    def flush():
        nonlocal shard
        if buf["iso"]:
            t = pa.table({k: v for k, v in buf.items()})
            pq.write_table(t, f"{a.out}/windows-{shard:04d}.parquet", compression="zstd")
            shard += 1
            buf.clear()

    def emit(key, w):
        iso, src, flipped = key
        n = max(w["te"], w["tt"]) + 2
        b = bin_of(n)
        used[iso][w["bin"]] -= w["budget"]
        used[iso][b] += n
        if not a.stats_only:
            buf["english"].append(" ".join(w["en"])); buf["target"].append(" ".join(w["tg"]))
        buf["iso"].append(iso); buf["source_dataset"].append(src); buf["licence"].append(w["lic"]); buf["flipped"].append(flipped)
        buf["qe"].append(float(np.mean(w["qe"])) if w["qe"] else float("nan")); buf["length_bin"].append(b); buf["n_pairs"].append(len(w["te_l"]))
        buf["src_afr_tokens"].append(w["te"] + 2); buf["tgt_afr_tokens"].append(w["tt"] + 2); buf["punct_added"].append(w["pa"])
        buf["src_gemma_tokens"].append(w["ge"]); buf["tgt_gemma_tokens"].append(w["gt"])
        buf["nc_exception"].append(nc_exception_tag(src, allow) if is_nc(w["lic"]) else "")
        buf["owner_exception"].append(owner_tag(src)); owner_win[(owner_tag(src), iso)] += 1
        by_iso_bin[iso][b] += 1
        stats["windows"] += 1
        if len(buf["iso"]) >= a.shard_rows:
            flush()

    over = {k: int(v) for k, v in (x.split(":") for x in a.oversample.split(",") if x)}
    files = shard_files(a.src)
    t0 = time.time()

    def pairs_of(rep):
        """yield batches (english, target, iso, source, flipped, licence, key) of one pass; rep 1 = all rows (deduped), rep > 1 = oversampled languages shuffled"""
        for f in files:
            pf = pq.ParquetFile(f)
            names = pf.schema_arrow.names
            for rb in pf.iter_batches(batch_size=a.batch):
                d = rb.to_pydict()
                n = len(d["english"])
                keys = d["key"] if "key" in d else [key_of(d["english"][i], d["iso"][i], d["target"][i]) for i in range(n)]
                fl = d["flipped"] if "flipped" in d else [False] * n
                idx = []
                lic_col = d["licence"] if "licence" in d else [None] * n
                for i in range(n):
                    if is_nc(lic_col[i]) and nc_exception_tag(d["source_dataset"][i], allow):
                        if rep == 1:
                            exc_pairs[(nc_exception_tag(d["source_dataset"][i], allow), d["iso"][i])] += 1
                    elif not a.include_nc and is_nc(lic_col[i]):          # BEFORE the dedupe: an NC copy must not mask a commercial copy of the same pair
                        if rep == 1:
                            stats["dropped_nc"] += 1; nc_by_iso[d["iso"][i]] += 1
                        continue
                    if rep == 1:
                        k = keys[i]
                        if k in seen:
                            stats["duplicate_across_dirs"] += 1
                            continue
                        seen.add(k)
                    elif over.get(d["iso"][i], 1) < rep:
                        continue
                    if keep is not None and keys[i] not in keep:
                        stats["dropped_by_gate"] += 1
                        continue
                    idx.append(i)
                if rep > 1:
                    rng.shuffle(idx)
                if idx:
                    yield ([d["english"][i] for i in idx], [d["target"][i] for i in idx], [d["iso"][i] for i in idx], [d["source_dataset"][i] for i in idx],
                           [bool(fl[i]) for i in idx], [d["licence"][i] for i in idx], [keys[i] for i in idx], f)

    done = False
    for rep in range(1, max(over.values(), default=1) + 1):
        for en_l, tg_l, iso_l, src_l, fl_l, lic_l, key_l, f in pairs_of(rep):
            te = [len(e.ids) for e in tka.encode_batch(en_l, add_special_tokens=False)]
            tt = [len(e.ids) for e in tka.encode_batch(tg_l, add_special_tokens=False)]
            if tkg and a.gemma_sample >= 1.0:
                ge = [len(e.ids) for e in tkg.encode_batch(en_l, add_special_tokens=False)]
                gt = [len(e.ids) for e in tkg.encode_batch(tg_l, add_special_tokens=False)]
            else:
                ge = [0] * len(en_l); gt = [0] * len(en_l)
                if tkg:
                    sel = [i for i, k in enumerate(key_l) if (k % 100000) < int(a.gemma_sample * 100000)]
                    if sel:
                        sg = tkg.encode_batch([en_l[i] for i in sel], add_special_tokens=False); st_ = tkg.encode_batch([tg_l[i] for i in sel], add_special_tokens=False)
                        for i, x, y in zip(sel, sg, st_):
                            r_ = ratio[iso_l[i]]; r_[0] += te[i]; r_[1] += len(x.ids); r_[2] += tt[i]; r_[3] += len(y.ids); r_[4] += 1
            for i in range(len(en_l)):
                en, tg, iso, src, fl, lic = en_l[i], tg_l[i], iso_l[i], src_l[i], fl_l[i], lic_l[i]
                qe = qe_map.get(key_l[i]) if qe_map is not None else None
                key = (iso, src, fl)
                a_term, b_term = bool(TERM.search(en)), bool(TERM.search(tg))
                stats["pairs"] += 1
                if a_term != b_term:
                    w = {"en": [en], "tg": [tg], "te": te[i], "tt": tt[i], "te_l": [1], "qe": [qe] if qe is not None else [], "lic": lic, "pa": False, "ge": ge[i], "gt": gt[i], "budget": 0, "bin": "b64"}
                    stats["standalone_one_sided"] += 1
                    emit(key, w)
                    continue
                added = False
                if not a_term:
                    en, tg, added = en.rstrip() + ".", tg.rstrip() + ".", True
                    te_i, tt_i = te[i] + 1, tt[i] + 1
                    stats["punct_added"] += 1
                else:
                    te_i, tt_i = te[i], tt[i]
                w = cur.get(key)
                if w is not None and (w["te"] + te_i + 2 > w["budget"] or w["tt"] + tt_i + 2 > w["budget"]):
                    emit(key, w)
                    w = None
                if w is None:
                    bud, bn = new_budget(iso)
                    w = cur[key] = {"en": [], "tg": [], "te": 0, "tt": 0, "te_l": [], "qe": [], "lic": lic, "pa": False, "budget": bud, "bin": bn, "ge": 0, "gt": 0}
                w["en"].append(en); w["tg"].append(tg); w["te"] += te_i; w["tt"] += tt_i; w["te_l"].append(1)
                if qe is not None:
                    w["qe"].append(qe)
                w["pa"] = w["pa"] or added; w["ge"] += ge[i]; w["gt"] += gt[i]
                if a.limit_pairs and stats["pairs"] >= a.limit_pairs:
                    done = True
                    break
            if rep > 1:
                stats[f"oversampled_pass{rep}"] += len(en_l)
            if done:
                break
        print(f"pass {rep}", "pairs", stats["pairs"], "windows", stats["windows"], f"{time.time() - t0:.0f}s", flush=True)
        if done:
            break
    for key, w in list(cur.items()):
        emit(key, w)
    flush()
    rep_ = {"pairs": stats["pairs"], "windows": stats["windows"], "standalone_one_sided": stats["standalone_one_sided"], "punct_added": stats["punct_added"],
            "duplicate_across_dirs": stats["duplicate_across_dirs"], "dropped_nc": stats["dropped_nc"], "nc_allow": allow, "nc_exception_pairs": {f"{k[0]}|{k[1]}": v for k, v in exc_pairs.items()}, "owner_exception_windows": {f"{k[0]}|{k[1]}": v for k, v in owner_win.items() if k[0]}, "dropped_nc_by_iso": dict(nc_by_iso), "include_nc": a.include_nc, "dropped_by_gate": stats["dropped_by_gate"], "scores_used": bool(a.scores),
            "oversampled": {k: v for k, v in stats.items() if k.startswith("oversampled")}, "by_language": {k: dict(v) for k, v in by_iso_bin.items()}, "seed": a.seed}
    json.dump(rep_, open(f"{a.out}/stats.json", "w"), indent=1)
    json.dump({k: {"en": v[1] / max(v[0], 1), "tg": v[3] / max(v[2], 1), "n": v[4]} for k, v in ratio.items()}, open(f"{a.out}/gemma_ratio.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep_.items() if k != "by_language"}))


if __name__ == "__main__":
    main()
