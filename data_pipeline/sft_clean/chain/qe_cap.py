#!/usr/bin/env python3
"""Per-language QE workload cap (see common.cap_thresholds): the counts file, the workload table, and a self-check of the monotone sampler.

    python qe_cap.py count --src hub/data hub/data_devhalf hub/specialist hub/extra --out work/cap_counts.json [--nc-allow howard-nlp/ibom-mt]
        counts the unique scoreable pairs per language ONCE; every qe_score.py run of the chain reads this file (--cap-counts), so that raising the cap later only adds keys
    python qe_cap.py table --counts work/cap_counts.json --caps 100000 150000 200000 300000 [--rate 129] [--scored-keys N]
        pairs to score for each cap, the hours at --rate pairs/s, the languages that are capped
    python qe_cap.py check --counts work/cap_counts.json --src ... --cap 150000       # drawn sample sizes vs the cap per language (needs the pair shards)
"""
import argparse, collections, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cap_counts, cap_thresholds, in_cap, shard_files, parse_nc_allow


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count"); c.add_argument("--src", nargs="+", required=True); c.add_argument("--out", required=True); c.add_argument("--nc-allow", nargs="*", default=[]); c.add_argument("--force", action="store_true", help="overwrite an existing counts file (changes the thresholds of already scored languages: a smaller sample would no longer be a subset)")
    t = sub.add_parser("table"); t.add_argument("--counts", required=True); t.add_argument("--caps", nargs="+", type=int, default=[100000, 150000, 200000, 300000]); t.add_argument("--rate", type=float, default=129.0)
    k = sub.add_parser("check"); k.add_argument("--counts", required=True); k.add_argument("--src", nargs="+", required=True); k.add_argument("--cap", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "count":
        if os.path.exists(a.out) and not a.force:
            print(f"{a.out} exists: kept (the counts are fixed once so that a larger cap scores a superset); --force to recount"); return
        t0 = time.time(); allow = set(parse_nc_allow(a.nc_allow))
        counts = cap_counts(a.src, True, allow)
        json.dump({"counts": counts, "total": sum(counts.values()), "src": a.src, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "nc_allow": sorted(allow)}, open(a.out, "w"), indent=1, sort_keys=True)
        print(f"{sum(counts.values()):,} unique scoreable pairs in {len(counts)} languages -> {a.out} ({time.time() - t0:.0f}s)")
    elif a.cmd == "table":
        counts = json.load(open(a.counts))["counts"]
        print(f"{'cap':>9} {'pairs to score':>15} {'capped languages':>17} {'hours @%g/s' % a.rate:>12}")
        for n in a.caps:
            s = sum(min(c, n) for c in counts.values())
            print(f"{n:>9,} {s:>15,} {sum(1 for c in counts.values() if c > n):>17} {s / a.rate / 3600:>12.1f}")
        print(f"{'no cap':>9} {sum(counts.values()):>15,} {0:>17} {sum(counts.values()) / a.rate / 3600:>12.1f}")
    else:
        counts = json.load(open(a.counts))["counts"]; thr = cap_thresholds(counts, a.cap)
        import pyarrow.parquet as pq
        from common import key_of
        got = collections.Counter(); seen = set()
        for f in shard_files(a.src):
            tb = pq.read_table(f, columns=[c for c in ("key", "english", "target", "iso") if c in pq.ParquetFile(f).schema_arrow.names]).to_pydict()
            for i, iso in enumerate(tb["iso"]):
                kk = tb["key"][i] if "key" in tb else key_of(tb["english"][i], iso, tb["target"][i])
                if kk in seen: continue
                seen.add(kk)
                if in_cap(kk, iso, thr): got[iso] += 1
        for iso, c in sorted(counts.items(), key=lambda kv: -kv[1])[:30]:
            print(f"{iso:5s} pairs {c:>9,}  in cap {got[iso]:>8,}  (target {min(c, a.cap):,})")


if __name__ == "__main__":
    main()
