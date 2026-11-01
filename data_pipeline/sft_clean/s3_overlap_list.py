#!/usr/bin/env python3
"""Per-prompt overlap lists of the S3 prompt filter (agent S3, 2026-10-28). Hash + provenance only, no text.

  overlap_removed_<set>.csv   prompts REMOVED because they overlap one of: afriscience_mt:dev, afriscience_mt:test, afrilongdoc:all   (from out/removed_log/*.parquet)
  flag_ssa_mte_train.csv      prompts that were KEPT but overlap the SSA-MTE TRAIN split (reward-model training / T2 calibration data): flag only   (from out/clean_*/ *.parquet)

    python s3_overlap_list.py --out-root out --dest reports/data/clean_rl_science_2026-10-28
"""
import argparse
import csv
import glob
import os

import pyarrow.parquet as pq

SETS = ("afriscience_mt:dev", "afriscience_mt:test", "afrilongdoc:all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    writers, files = {}, {}
    n = dict.fromkeys(SETS, 0)
    for f in sorted(glob.glob(os.path.join(a.out_root, "removed_log", "*.parquet"))):
        pool, iso = os.path.basename(f)[:-8].split("__")
        t = pq.read_table(f).to_pydict()
        for i in range(len(t["prov_row"])):
            sets = t["protected_sets"][i].split(";")
            for s in SETS:
                if s in sets:
                    if s not in writers:
                        fh = open(os.path.join(a.dest, "overlap_removed_" + s.replace(":", "_") + ".csv"), "w", newline="")
                        files[s] = fh
                        writers[s] = csv.writer(fh)
                        writers[s].writerow(["pool", "iso", "prov_row", "prov_en_xxh64", "length_bin", "source_dataset", "domain", "rules", "all_protected_sets"])
                    writers[s].writerow([pool, iso, t["prov_row"][i], t["prov_en_xxh64"][i], t["length_bin"][i], t["source_dataset"][i], t["domain"][i], t["rules"][i], t["protected_sets"][i]])
                    n[s] += 1
    for fh in files.values():
        fh.close()
    m = 0
    with open(os.path.join(a.dest, "flag_ssa_mte_train.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pool", "iso", "prov_row", "prov_en_xxh64", "length_bin", "source_dataset", "domain"])
        for f in sorted(glob.glob(os.path.join(a.out_root, "clean_*", "*.parquet"))):
            pool = os.path.basename(os.path.dirname(f)).replace("clean_", "")
            cols = ["prov_row", "prov_en_xxh64", "length_bin", "source_dataset", "flag_ssa_mte_train"] + (["domain"] if "domain" in pq.ParquetFile(f).schema_arrow.names else [])
            t = pq.read_table(f, columns=cols).to_pydict()
            doms = t.get("domain")
            for i, fl in enumerate(t["flag_ssa_mte_train"]):
                if fl:
                    w.writerow([pool, os.path.basename(f)[:-8], t["prov_row"][i], t["prov_en_xxh64"][i], t["length_bin"][i], t["source_dataset"][i], (doms[i] if doms else None) or "general"])
                    m += 1
    print("removed-overlap rows:", n, "| kept SSA-MTE-train-flagged rows:", m)


if __name__ == "__main__":
    main()
