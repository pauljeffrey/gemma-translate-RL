#!/usr/bin/env python
"""Non-destructive source filter of the long-document RL pool (2026-11-04, owner: LibreTexts is EXCLUDED until its licence is resolved).

    python filter_long_pool_source.py --pool /workspace/clean_rerun/prompts/long_rl --out /workspace/clean_rerun/prompts/long_rl_filtered --exclude libretexts [--log OUT.json]

`--out` is a hard-linked copy of `--pool` (cp -al: no extra disk) in which every parquet file with a `source_dataset` matching an excluded substring (case-insensitive) is rewritten
without those rows; empty files are not written. The original pool is left untouched. Row counts per directory and source before / after are logged, and the result is verified:
no excluded source remains, and every kept row is a row of the original (by `window_id`/`prov_en_xxh64` + iso, or by the English text), i.e. the filtered pool is a subset of the pool that
passed the protected-index scan (R1-R3, 0 hits)."""
import argparse, collections, glob, json, os, subprocess

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True); ap.add_argument("--out", required=True); ap.add_argument("--exclude", nargs="+", default=["libretexts"]); ap.add_argument("--log", default=None)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: refusing to overwrite")
    subprocess.run(["cp", "-al", a.pool, a.out], check=True)
    before, after, removed = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter), collections.Counter()
    n_files_rewritten = n_files_dropped = 0
    for f in sorted(glob.glob(os.path.join(a.out, "**", "*.parquet"), recursive=True)):
        d = os.path.relpath(os.path.dirname(f), a.out)
        t = pq.read_table(f)
        if "source_dataset" not in t.column_names:
            for s in ["(no source column)"]: before[d][s] += t.num_rows; after[d][s] += t.num_rows
            continue
        src = t.column("source_dataset")
        bad = None
        for ex in a.exclude:
            m = pc.match_substring(pc.utf8_lower(src), ex.lower())
            bad = m if bad is None else pc.or_(bad, m)
        bad = pc.fill_null(bad, False)
        for s, n in collections.Counter(src.to_pylist()).items(): before[d][s] += n
        if pc.any(bad).as_py():
            keep = t.filter(pc.invert(bad))
            removed[d] += t.num_rows - keep.num_rows
            os.unlink(f)                                          # drops the hard link only; the original file keeps its inode
            if keep.num_rows:
                pq.write_table(keep, f, compression="zstd"); n_files_rewritten += 1
            else:
                n_files_dropped += 1
            t = keep
        for s, n in collections.Counter(t.column("source_dataset").to_pylist()).items(): after[d][s] += n
    # verification
    left = {d: {s: n for s, n in c.items() if any(ex.lower() in s.lower() for ex in a.exclude)} for d, c in after.items()}
    left = {d: c for d, c in left.items() if c}
    key = lambda t: set(zip(t.column("iso").to_pylist(), t.column("english").to_pylist()))
    orig, filt = set(), set()
    for f in glob.glob(os.path.join(a.pool, "**", "*.parquet"), recursive=True):
        orig |= key(pq.read_table(f, columns=["iso", "english"]))
    for f in glob.glob(os.path.join(a.out, "**", "*.parquet"), recursive=True):
        filt |= key(pq.read_table(f, columns=["iso", "english"]))
    res = {"pool": a.pool, "out": a.out, "exclude": a.exclude, "rows_removed_by_dir": dict(removed), "files_rewritten": n_files_rewritten, "files_dropped_empty": n_files_dropped,
           "rows_before_total": sum(sum(c.values()) for c in before.values()), "rows_after_total": sum(sum(c.values()) for c in after.values()),
           "sources_before": {d: dict(c.most_common()) for d, c in before.items()}, "sources_after": {d: dict(c.most_common()) for d, c in after.items()},
           "excluded_sources_left": left, "distinct_prompts_original": len(orig), "distinct_prompts_filtered": len(filt), "filtered_subset_of_original": filt <= orig}
    if a.log:
        json.dump(res, open(a.log, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("sources_")}, indent=1))
    print("sources_after (science_b2048):", json.dumps(res["sources_after"].get("science_b2048")))
    if left or not res["filtered_subset_of_original"]:
        raise SystemExit("VERIFICATION FAILED")


if __name__ == "__main__":
    main()
