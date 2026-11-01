#!/usr/bin/env python3
"""Live status of the CPU QE job: python qe_cpu_status.py  (on box 2, /workspace/work). Writes qe_cpu/status.md and prints it."""
import json, os, time, glob
import pyarrow.dataset as pads
R = "/workspace/work/qe_cpu"
pgs = [json.load(open(f)) for f in sorted(glob.glob(R + "/progress_*.json"))] or ([json.load(open(R + "/progress.json"))] if os.path.exists(R + "/progress.json") else [])
pg = {"stage_done": {}, "stage_total": {}, "pairs_per_s_recent": 0.0, "pairs_per_s_run_average": 0.0}
for x in pgs:
    for k in ("stage_done",):
        for st, v in x.get(k, {}).items(): pg[k][st] = pg[k].get(st, 0) + v
    pg["stage_total"] = x.get("stage_total", pg["stage_total"])
    pg["pairs_per_s_recent"] += x.get("pairs_per_s_recent", 0); pg["pairs_per_s_run_average"] += x.get("pairs_per_s_run_average", 0)
    for k in ("device", "dtype", "model_sha"): pg[k] = x.get(k)
tot_all = 23_772_128
n_scored = 0
if glob.glob(R + "/scores_cpu/*.parquet"):
    n_scored = pads.dataset(R + "/scores_cpu", format="parquet").count_rows()
# stage totals/done from the score shards on disk (survives restarts); manifest maps queue shard -> stage
man = json.load(open(R + "/queue/queue_manifest.json")) if os.path.exists(R + "/queue/queue_manifest.json") else {}
pg["stage_total"] = {}; pg["stage_done"] = {}
for name, v in man.items():
    pg["stage_total"][v["stage"]] = pg["stage_total"].get(v["stage"], 0) + v["n"]
for f in glob.glob(R + "/scores_cpu/queue__q-*-0000.parquet"):
    nm = os.path.basename(f)[len("queue__"):-len("-0000.parquet")] + ".parquet"
    if nm in man:
        pg["stage_done"][man[nm]["stage"]] = pg["stage_done"].get(man[nm]["stage"], 0) + man[nm]["n"]
rate = pg.get("pairs_per_s_recent") or pg.get("pairs_per_s_run_average") or 0
lines = [f"# CPU QE status (updated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})", "",
         f"scored on CPU so far: {n_scored:,} pairs; measured speed: {rate:.2f} pairs/s recent, {pg.get('pairs_per_s_run_average', 0):.2f} run average; device {pg.get('device')} {pg.get('dtype')}; model_sha {pg.get('model_sha')}",
         f"the GPU still has to score: {tot_all - n_scored:,} of {tot_all:,} unique pairs", "", "| stage | done | total | share | ETA at the recent speed |", "|---|---:|---:|---:|---:|"]
cum = 0
for st in ("P1a", "REF", "P2a", "P1b", "P2b"):
    d, t = pg.get("stage_done", {}).get(st, 0), pg.get("stage_total", {}).get(st, 0)
    if t:
        eta = (t - d) / rate / 3600 if rate else float("nan")
        lines.append(f"| {st} | {d:,} | {t:,} | {100 * d / t:.1f}% | {eta:.1f} h |")
lines += ["", "stages (queue order): P1a = 500 random pairs per language (round-robin), REF = FLORES+ dev H_cal human references of the 20 specialist languages, P2a = all pairs (cap 50k) of nus kbp dyu bam kea, P1b = ranks 500-1999 of every language, P2b = the other priority languages."]
txt = "\n".join(lines)
open(R + "/status.md", "w").write(txt + "\n"); print(txt)
