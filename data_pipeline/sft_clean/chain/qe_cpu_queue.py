#!/usr/bin/env python3
"""Priority queue for the CPU QE job (the GPU run continues from the same score directory).

    python qe_cpu_queue.py --out qe_cpu/queue --src out/final out/data_devhalf out/specialist [--seed 20261028] [--p1a 500] [--p1 2000] [--p2-cap 50000] [--refs]

Stable pair id = key = xxh64(english.casefold, iso, target.casefold) (the `key` column of out/kept and the id qe_score.py / gate.py / windows.py use).
Deterministic random order inside every language: u = xxh64(key, seed) / 2^64, smallest u first, so ANY prefix of a language's list is an unbiased random sample and the P1 sample is
reproducible. Queue shards (500 pairs, columns english, target, iso, key) are named q-<seq>.parquet in processing order; queue_manifest.json maps shard -> stage:
    P1a  ranks 0..p1a-1 of every language, round-robin across languages (so progress is balanced)
    REF  FLORES+ dev H_cal human references of the 20 specialist languages (BENCHMARK TEXT: stays in this local queue, never pushed; only key+qe scores may be pushed) [--refs]
    P2a  the thinnest specialist languages nus kbp dyu bam kea: all remaining pairs (up to p2-cap)
    P1b  ranks p1a..p1-1 of every other language, round-robin
    P2b  the other priority languages in order (fon ewe wol seh kmb vmw kam, then ascending by pair count through the rest of the 20 specialist languages): ranks p1..p2-cap
Shard q-000000 (the first 500 pairs, mixed languages) is the cross-check set for the GPU session.
"""
import argparse, collections, csv, heapq, json, os, sys, time
import pyarrow as pa, pyarrow.parquet as pq
import xxhash
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import key_of, shard_files

SPEC = "amh tir fon dyu kbp nus mos kam bam umb kea kmb vmw kab seh wol ewe luo tum yor".split()
FIRST = ["nus", "kbp", "dyu", "bam", "kea", "fon", "ewe", "wol", "seh", "kmb", "vmw", "kam"]
P2A = FIRST[:5]


def u01(key, seed):
    return xxhash.xxh64_intdigest(int(key).to_bytes(8, "little"), seed) / 2 ** 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=20261028); ap.add_argument("--p1a", type=int, default=500); ap.add_argument("--p1", type=int, default=2000); ap.add_argument("--p2-cap", type=int, default=50000)
    ap.add_argument("--shard", type=int, default=500)
    ap.add_argument("--refs", action="store_true"); ap.add_argument("--hcal-split", default="/workspace/work/heldout_split_cal_sel.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    heaps = collections.defaultdict(list)           # iso -> max-heap of (-u, key, english, target)
    counts = collections.Counter(); seen = set()
    cap = lambda iso: a.p2_cap if iso in FIRST or iso in SPEC else a.p1
    for f in shard_files(a.src):
        pf = pq.ParquetFile(f)
        for rb in pf.iter_batches(batch_size=50000, columns=["english", "target", "iso"]):
            d = rb.to_pydict()
            for en, tg, iso in zip(d["english"], d["target"], d["iso"]):
                k = key_of(en, iso, tg)
                if k in seen:
                    continue
                seen.add(k); counts[iso] += 1
                u = u01(k, a.seed); h = heaps[iso]
                if len(h) < cap(iso):
                    heapq.heappush(h, (-u, k, en, tg))
                elif u < -h[0][0]:
                    heapq.heapreplace(h, (-u, k, en, tg))
        print(os.path.basename(os.path.dirname(f)), os.path.basename(f), f"{len(seen):,} unique pairs, {time.time() - t0:.0f}s", flush=True)
    L = {iso: sorted((-x[0], x[1], x[2], x[3]) for x in h) for iso, h in heaps.items()}          # ascending u
    langs = sorted(L)
    rest = sorted([i for i in SPEC if i not in FIRST], key=lambda i: counts[i])
    prio = FIRST + rest
    order = []                                          # (stage, key, en, tg, iso)
    def rr(stage, lo, hi, isos):
        for r in range(lo, hi):
            for iso in isos:
                if r < len(L.get(iso, ())):
                    u, k, en, tg = L[iso][r]; order.append((stage, k, en, tg, iso))
    rr("P1a", 0, a.p1a, langs)
    if a.refs:
        import hcal
        sp = json.load(open(a.hcal_split)); os.makedirs(a.out + "/_refs", exist_ok=True)
        hcal.refs(argparse.Namespace(split=a.hcal_split, out=a.out + "/_refs", tmp="/tmp/hcal_dl_qe", isos=SPEC))
        t = pq.read_table(a.out + "/_refs/pairs-0000.parquet").to_pydict()
        for en, tg, iso, k in zip(t["english"], t["target"], t["iso"], t["key"]):
            order.append(("REF", k, en, tg, iso))
    for iso in P2A:
        for r in range(a.p1a, len(L.get(iso, ()))):
            u, k, en, tg = L[iso][r]; order.append(("P2a", k, en, tg, iso))
    rr("P1b", a.p1a, a.p1, [i for i in langs if i not in P2A])
    for iso in prio:
        if iso in P2A: continue
        for r in range(a.p1, len(L.get(iso, ()))):
            u, k, en, tg = L[iso][r]; order.append(("P2b", k, en, tg, iso))
    manifest = {}; n = 0
    for i in range(0, len(order), a.shard):
        ch = order[i:i + a.shard]; name = f"q-{i // a.shard:06d}.parquet"
        stages = collections.Counter(x[0] for x in ch)
        pq.write_table(pa.table({"english": [x[2] for x in ch], "target": [x[3] for x in ch], "iso": [x[4] for x in ch], "key": pa.array([x[1] for x in ch], pa.uint64())}), f"{a.out}/{name}", compression="zstd")
        manifest[name] = {"stage": stages.most_common(1)[0][0], "n": len(ch), "stages": dict(stages)}
    json.dump(manifest, open(a.out + "/queue_manifest.json", "w"), indent=0)
    json.dump({"seed": a.seed, "p1a": a.p1a, "p1": a.p1, "p2_cap": a.p2_cap, "crosscheck_shard": "q-000000.parquet", "priority_order": prio, "pairs_by_language": dict(counts), "queue_pairs": len(order),
               "queue_by_stage": dict(collections.Counter(x[0] for x in order))}, open(a.out + "/queue_meta.json", "w"), indent=1)
    with open(a.out + "/language_counts.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "unique_pairs_union"]); [w.writerow([k, v]) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    print("queue:", len(order), "pairs in", len(manifest), "shards;", dict(collections.Counter(x[0] for x in order)), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
