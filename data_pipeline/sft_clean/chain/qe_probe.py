#!/usr/bin/env python3
"""Throughput probe for the one GPU: pairs/s for 1, 2, 4 (8) scoring processes (and optionally two batch sizes), measured in the first minutes, then picks the best.

    python qe_probe.py --pairs hub/data --out work/qe_probe --python venv_comet/bin/python --procs 1 2 4 [--batches 128 256] [--n 3000]
Each configuration scores P x n DIFFERENT pairs (P processes at once, one shard each, fp16, CUDA); the aggregate rate = P*n / the slowest process's scoring time (model load excluded, parsed from
the `elapsed_s=` field of qe_score.py's DONE line); the GPU memory peak is polled with nvidia-smi. Output: <out>/qe_procs.json {best_procs, best_batch, table, vram_peak_mib, vram_total_mib}
and a line `BEST procs=N batch=B` (run_chain.sh reads the json). A larger process count must be at least 5% faster than the next smaller one to win (fewer processes = less CPU/RAM load).
"""
import argparse, json, os, re, subprocess, sys, threading, time
import pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import shard_files


def vram():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True).splitlines()[0].split(",")
        return int(o[0]), int(o[1])
    except Exception:
        return 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--procs", nargs="+", type=int, default=[1, 2, 4]); ap.add_argument("--batches", nargs="+", type=int, default=[128]); ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--stub", action="store_true", help="test mode: stub scorer (no model, no GPU)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    need = max(a.procs) * a.n * len(a.batches) * len(a.procs)
    rows = {"english": [], "target": [], "iso": []}
    for f in shard_files(a.pairs):
        t = pq.read_table(f, columns=["english", "target", "iso"]).to_pydict()
        for k in rows: rows[k] += t[k]
        if len(rows["iso"]) >= need: break
    qe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qe_score.py")
    table, cursor = [], 0
    for batch in a.batches:
        for P in a.procs:
            dirs = []
            for p in range(P):
                d = os.path.join(a.out, f"probe_b{batch}_p{P}_{p}"); os.makedirs(d, exist_ok=True)
                sl = slice(cursor, cursor + a.n); cursor += a.n
                pq.write_table(pa.table({k: v[sl] for k, v in rows.items()}), os.path.join(d, "pairs.parquet")); dirs.append(d)
            peak = [0]; stop = threading.Event()
            def poll():
                while not stop.is_set():
                    peak[0] = max(peak[0], vram()[0]); time.sleep(1)
            th = threading.Thread(target=poll, daemon=True); th.start()
            t0 = time.time(); procs = []
            for d in dirs:
                cmd = [a.python, qe, "--src", d, "--out", d + "_scores", "--batch", str(batch), "--workers", str(a.workers), "--chunk", str(a.n), "--device", "cuda", "--half"] + (["--stub"] if a.stub else [])
                procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
            outs = [p.communicate()[0] for p in procs]; stop.set(); th.join()
            el = [float(m.group(1)) for o in outs for m in [re.search(r"elapsed_s=([0-9.]+)", o)] if m]
            ok = len(el) == P and all(p.returncode == 0 for p in procs)
            rate = (P * a.n / max(el)) if ok else 0.0
            table.append({"procs": P, "batch": batch, "pairs_per_s": round(rate, 1), "ok": ok, "wall_s": round(time.time() - t0, 1), "vram_peak_mib": peak[0]})
            print(table[-1], flush=True)
            tot = vram()[1]
            if tot and peak[0] > 0.8 * tot:
                print(f"VRAM peak {peak[0]} of {tot} MiB > 80%: larger configurations are not tried", flush=True); break
    best = None
    for batch in a.batches:
        cand = sorted([r for r in table if r["batch"] == batch and r["ok"]], key=lambda r: r["procs"])
        for r in cand:
            if best is None or (r["pairs_per_s"] > best["pairs_per_s"] * (1.05 if r["batch"] == best["batch"] else 1.0)):
                best = r
    tot = vram()[1]
    res = {"best_procs": best["procs"] if best else 1, "best_batch": best["batch"] if best else a.batches[0], "table": table, "vram_peak_mib": max((r["vram_peak_mib"] for r in table), default=0), "vram_total_mib": tot}
    json.dump(res, open(os.path.join(a.out, "qe_procs.json"), "w"), indent=1)
    print(f"BEST procs={res['best_procs']} batch={res['best_batch']}  (VRAM peak {res['vram_peak_mib']} of {tot} MiB)")


if __name__ == "__main__":
    main()
