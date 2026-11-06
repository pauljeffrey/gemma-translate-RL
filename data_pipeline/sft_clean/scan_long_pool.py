#!/usr/bin/env python
"""Decontamination scan of the b2048 rows that earlier pools dropped unscanned (2026-11-04, long-document RL stage).

Rows of `length_bin == b2048` of (a) the ORIGINAL science pool `sci_v2` and (b) the v3 `zz_b2048/` files are checked on their English side against the full
protected index (R1 exact sentence, R2 contained sentence, R3 shared word 8-gram, strict, no FLORES+ dev train-half exemption: AfriScience-MT dev and test papers,
AfriLongDoc, SmolDoc, AfriDocMT, TICO-19, NTREX, MAFAND, FLORES+ ... every set of protected_manifest.json). A prompt with any hit is dropped, never repaired. Distinct
English texts are checked once. Output: <out>/<name>/<iso>.parquet with the original columns + prov_index, and <out>/<name>_stats.json.

    python scan_long_pool.py --index protected_index.npz --manifest protected_manifest.json --in RAW_DIR/jobs/r7/assets2/rl_prompts_sci_v2 --name sci_v2_b2048 --out OUT [--nproc 4]
    python scan_long_pool.py ... --in RAW_DIR/sft_build/gemma/rl_prompts_v3/zz_b2048 --name v3_b2048 --out OUT
"""
import argparse, collections, glob, json, os, sys, time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import s3_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--out", required=True); ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--bins", default="b2048"); a = ap.parse_args()
    bins = set(a.bins.split(","))
    ix, ex, man = C.load_strict(a.index, a.manifest)
    print(C.now(), "index", man["version"], man["index_file_sha256"][:16], flush=True)
    files = sorted(glob.glob(os.path.join(a.inp, "*.parquet")))
    tabs = {}
    texts = {}
    for f in files:
        t = pq.read_table(f)
        if "length_bin" in t.column_names:
            t = t.filter(pc.is_in(t.column("length_bin"), value_set=pa.array(sorted(bins))))
        if t.num_rows == 0:
            continue
        tabs[os.path.basename(f)] = t
        for s in t.column("english").to_pylist():
            texts.setdefault(s, None)
    uniq = list(texts)
    print(C.now(), f"{len(tabs)} files, {sum(t.num_rows for t in tabs.values())} rows, {len(uniq)} distinct English texts", flush=True)
    chk = C.Checker(ix, None, None, nproc=a.nproc)
    hit = {}
    by_set, by_rule = collections.Counter(), collections.Counter()
    t0 = time.time(); step = 2000
    for i in range(0, len(uniq), step):
        res = chk.run(uniq[i:i + step])
        for j, r in enumerate(res):
            m = r[0] | r[1] | r[2]
            if m:
                hit[uniq[i + j]] = True
                for nm in ix.names(m):
                    by_set[nm] += 1
                for rule, mm in zip(C.RULES, r[:3]):
                    if mm:
                        by_rule[rule] += 1
        print(C.now(), f"{min(len(uniq), i + step)}/{len(uniq)} flagged {len(hit)} {time.time() - t0:.0f}s", flush=True)
    chk.close()
    od = os.path.join(a.out, a.name); os.makedirs(od, exist_ok=True)
    n_in = n_keep = 0; by_lang = {}
    for fn, t in tabs.items():
        keep = [s not in hit for s in t.column("english").to_pylist()]
        kt = t.filter(pa.array(keep))
        n_in += t.num_rows; n_keep += kt.num_rows
        by_lang[fn[:-8]] = [t.num_rows, kt.num_rows]
        if kt.num_rows:
            kt = kt.append_column("prov_index", pa.array([man["version"]] * kt.num_rows, pa.string()))
            pq.write_table(kt, os.path.join(od, fn), compression="zstd")
    st = dict(index_version=man["version"], index_sha256=man["index_file_sha256"], rule="R1 exact sentence, R2 contained sentence, R3 shared word 8-gram (strict, both normal forms), English side only, no FLORES+ dev train-half exemption",
              input=a.inp, bins=sorted(bins), rows_in=n_in, rows_kept=n_keep, distinct_texts=len(uniq), distinct_flagged=len(hit),
              flagged_by_set_any_rule=dict(by_set.most_common()), flagged_by_rule=dict(by_rule), rows_by_language=by_lang)
    json.dump(st, open(os.path.join(a.out, a.name + "_stats.json"), "w"), indent=1)
    print(C.now(), json.dumps({k: v for k, v in st.items() if k != "rows_by_language"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
