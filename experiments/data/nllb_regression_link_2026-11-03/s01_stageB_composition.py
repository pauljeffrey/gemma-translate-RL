#!/usr/bin/env python3
"""Step 1: composition of the 385M's stage-B TRANSLATION pairs per language and source (laptop, CPU only, reads the databox backup).

Universe of translation pairs seen by the 385M SFT = keys of work/gate_old/keep_final.parquet (A0's data, 7.1 M) UNION work/gate_all/keep_delta.parquet (the new
pairs appended for stage B, 35.2 M). The two sets are disjoint (checked). Rows are read from hub/{data,data_devhalf,specialist,extra,extra2k/*} in the order the
chain used (finish_all.sh SRC order), the first occurrence of a key wins (windows.py dedupe). `data*` and `specialist` shards carry no `key` column: it is recomputed
with the chain's own definition xxh64(english.casefold() 0x01 iso 0x01 target.casefold()).

Outputs (this directory):
  composition_rows_by_lang_source.csv   iso, source_group, source_dataset_base, family, kind, rows, chars
  token_sample.parquet                  1/64 hash sample of the kept pairs (text) for tokens-per-character estimates

    python s01_stageB_composition.py --root <databox_backup>
"""
import argparse, glob, os, sys, time
import multiprocessing as mp
import numpy as np
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
import xxhash

HERE = os.path.dirname(os.path.abspath(__file__))
U = None


def key_of(en, iso, tg):
    return xxhash.xxh64_intdigest((en.casefold() + "\x01" + iso + "\x01" + tg.casefold()).encode("utf-8", "surrogatepass"))


def group_of(sd):
    s = sd or ""
    if s.startswith("nllb-mined") or s.startswith("extra-nllb_mined"):
        return "nllb_mined"
    if s.startswith("extra-nllb_seed"):
        return "nllb_seed"
    return "other"


def work(args):
    idx, f, tmp = args
    t = pq.read_table(f)
    n = t.num_rows
    iso = t["iso"].to_pylist()
    if "key" in t.column_names:
        key = np.asarray(t["key"].to_numpy(zero_copy_only=False), dtype=np.uint64)
    else:
        en, tg = t["english"].to_pylist(), t["target"].to_pylist()
        key = np.fromiter((key_of(e, i, g) for e, i, g in zip(en, iso, tg)), dtype=np.uint64, count=n)
    pos = np.clip(np.searchsorted(U, key), 0, len(U) - 1)
    ok = U[pos] == key
    if not ok.any():
        return idx, 0
    sd = t["source_dataset"].to_pylist()
    fam = t["source_family"].to_pylist() if "source_family" in t.column_names else [""] * n
    kind = t["kind"].to_pylist() if "kind" in t.column_names else [""] * n
    lic = t["licence"].to_pylist() if "licence" in t.column_names else [""] * n
    chars = np.asarray(pc.add(pc.utf8_length(t["english"]), pc.utf8_length(t["target"])).to_numpy(zero_copy_only=False), dtype=np.int64)
    m = np.flatnonzero(ok)
    out = pa.table({
        "file": pa.array(np.full(len(m), idx, np.int32)), "row": pa.array(m.astype(np.int64)), "key": pa.array(key[m]),
        "iso": pa.array([iso[i] for i in m]), "sd": pa.array([sd[i] for i in m]),
        "sdb": pa.array([(sd[i] or "").split(":")[0] for i in m]), "family": pa.array([fam[i] or "" for i in m]),
        "kind": pa.array([kind[i] or "" for i in m]), "licence": pa.array([lic[i] or "" for i in m]),
        "grp": pa.array([group_of(sd[i]) for i in m]), "chars": pa.array(chars[m]),
    })
    pq.write_table(out, os.path.join(tmp, f"{idx:05d}.parquet"), compression="zstd")
    smp = [i for i in m if int(key[i]) % 64 == 0]
    if smp:
        sub = t.take(pa.array(smp))
        pq.write_table(pa.table({"key": pa.array(key[smp]), "iso": sub["iso"], "grp": pa.array([group_of(sd[i]) for i in smp]), "english": sub["english"], "target": sub["target"],
                                 "file": pa.array(np.full(len(smp), idx, np.int32)), "row": pa.array(np.array(smp, np.int64))}), os.path.join(tmp, f"smp_{idx:05d}.parquet"), compression="zstd")
    return idx, len(m)


def main():
    global U
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--hold-only", action="store_true", help="process only hub/extra2_hold (languages that were not in the registry when extra2 was split; they ARE in the stage-B windows, finish_all.sh SRC) with file ids from 1000, then finalize")
    ap.add_argument("--finalize-only", action="store_true")
    a = ap.parse_args()
    R = a.root
    os.makedirs(a.tmp, exist_ok=True)
    old = pq.read_table(f"{R}/work/gate_old/keep_final.parquet")["key"].to_numpy()
    dlt = pq.read_table(f"{R}/work/gate_all/keep_delta.parquet")["key"].to_numpy()
    U = np.unique(np.concatenate([old, dlt]))
    print("old", len(old), "delta", len(dlt), "union", len(U), "overlap", len(old) + len(dlt) - len(U), flush=True)
    files = []
    off = 0
    if a.hold_only:
        off = 1000
        for dd in sorted(glob.glob(f"{R}/hub/extra2_hold/*")):
            files += sorted(glob.glob(f"{dd}/**/*.parquet", recursive=True))
    else:
        for d in ["data", "data_devhalf", "specialist", "extra"]:
            files += sorted(glob.glob(f"{R}/hub/{d}/**/*.parquet", recursive=True))
        for dd in sorted(glob.glob(f"{R}/hub/extra2k/*")):
            files += sorted(glob.glob(f"{dd}/**/*.parquet", recursive=True))
    with open(os.path.join(a.tmp, "files%s.txt" % ("_hold" if a.hold_only else "")), "w") as fh:
        fh.write("\n".join(files))
    print(len(files), "files", flush=True)
    t0 = time.time()
    if not a.finalize_only:
        with mp.get_context("fork").Pool(a.procs) as pool:
            tot = 0
            for idx, n in pool.imap_unordered(work, [(off + i, f, a.tmp) for i, f in enumerate(files)]):
                tot += n
                print(f"{idx}/{len(files)} kept-in-universe {n} total {tot} {time.time()-t0:.0f}s", flush=True)
    import duckdb
    con = duckdb.connect()
    con.execute(f"""create table k as select * from read_parquet('{a.tmp}/[0-9]*.parquet')""")
    print("rows in universe before dedupe", con.execute("select count(*) from k").fetchone()[0])
    con.execute("create table kd as select * from (select *, row_number() over (partition by key order by file, row) rn from k) where rn=1")
    print("after dedupe", con.execute("select count(*) from kd").fetchone()[0])
    con.execute(f"""copy (select iso, grp as source_group, sdb as source_dataset_base, family, kind, licence, count(*) rows, sum(chars) chars from kd group by all order by iso, rows desc)
                    to '{HERE}/composition_rows_by_lang_source.csv' (header, delimiter ',')""")
    # token sample restricted to rows that survived the dedupe
    con.execute(f"""copy (select s.* from read_parquet('{a.tmp}/smp_*.parquet') s semi join kd using (key, file, row)) to '{HERE}/token_sample.parquet' (format parquet, compression zstd)""")
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
