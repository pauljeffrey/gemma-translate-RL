#!/usr/bin/env python3
"""500-pair cross-check CPU fp32 vs GPU fp16 (run once at the start of the GPU session, before the bulk run).
    python qe_crosscheck.py prepare --queue qe_cpu/queue --out qe_cpu/crosscheck_pairs            # the 500 pairs of queue shard q-000000 (already scored on CPU)
    python qe_score.py --src qe_cpu/crosscheck_pairs --out qe_cpu/scores_gpu_check --device cuda --half --batch 64 --chunk 500          # NO --skip-scored: score them again on the GPU
    python qe_crosscheck.py compare --cpu qe_cpu/scores_cpu --gpu qe_cpu/scores_gpu_check
Pass: Pearson >= 0.999 and mean |difference| <= 0.003 (the old fp16-vs-fp32 check on 6,000 pairs: Pearson 0.99998, mean |diff| 0.0004). If it fails, do NOT mix CPU and GPU shards: re-score the CPU part on the GPU.
"""
import argparse, os, shutil, sys
import numpy as np
import pyarrow.dataset as pads, pyarrow.parquet as pq


def prepare(a):
    """The 500 pairs of queue shard q-000000. The queue (pair text) lived on the lost box 2; the Hub holds only the CPU scores (keys). With --from-scores the text is recovered by key from the pair shards."""
    import pyarrow as pa
    os.makedirs(a.out, exist_ok=True)
    if a.from_scores:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from common import key_of, shard_files
        want = set(pq.read_table(os.path.join(a.from_scores, "queue__q-000000-0000.parquet"), columns=["key"])["key"].to_pylist())
        rows = {"english": [], "target": [], "iso": [], "key": []}
        for f in shard_files(a.pairs):
            t = pq.read_table(f, columns=["english", "target", "iso"]).to_pydict()
            for e, g, i in zip(t["english"], t["target"], t["iso"]):
                k = key_of(e, i, g)
                if k in want:
                    want.discard(k); rows["english"].append(e); rows["target"].append(g); rows["iso"].append(i); rows["key"].append(k)
            if not want: break
        pq.write_table(pa.table({"english": rows["english"], "target": rows["target"], "iso": rows["iso"], "key": pa.array(rows["key"], pa.uint64())}), os.path.join(a.out, "q-000000.parquet"))
        print(len(rows["key"]), "cross-check pairs recovered by key;", len(want), "not found ->", a.out); return
    shutil.copy(os.path.join(a.queue, "q-000000.parquet"), os.path.join(a.out, "q-000000.parquet")); print("500-pair cross-check set ->", a.out)


def compare(a):
    c = pads.dataset(a.cpu, format="parquet").to_table(columns=["key", "qe", "device", "dtype"]).to_pydict()
    g = pads.dataset(a.gpu, format="parquet").to_table(columns=["key", "qe", "device", "dtype"]).to_pydict()
    cm = {k: q for k, q, d in zip(c["key"], c["qe"], c["device"]) if d == "cpu"}
    gm = {k: q for k, q in zip(g["key"], g["qe"])}
    ks = sorted(set(cm) & set(gm))
    x = np.array([cm[k] for k in ks]); y = np.array([gm[k] for k in ks])
    r = float(np.corrcoef(x, y)[0, 1]); mad = float(np.abs(x - y).mean())
    print(f"{len(ks)} common pairs (GPU device/dtype: {sorted(set(zip(g['device'], g['dtype'])))}): Pearson {r:.5f}, mean |diff| {mad:.5f}, max |diff| {np.abs(x - y).max():.5f}, mean bias (GPU-CPU) {float((y - x).mean()):+.5f}")
    ok = len(ks) >= 400 and r >= 0.999 and mad <= 0.003
    print("CROSS-CHECK", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--queue", default=None); p.add_argument("--from-scores", default=None, help="dir with the CPU score shards (queue__q-000000-0000.parquet): recover the pairs by key from --pairs"); p.add_argument("--pairs", nargs="*", default=[]); p.add_argument("--out", required=True)
    c = sub.add_parser("compare"); c.add_argument("--cpu", required=True); c.add_argument("--gpu", required=True)
    a = ap.parse_args(); (prepare if a.cmd == "prepare" else compare)(a)
