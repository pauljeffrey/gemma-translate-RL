"""Rebuild per-row FLORES+ predictions from the generation units of a sft_benchmarks run (units = consecutive rows of the benchmark in loader order). usage: e2_units.py OUTDIR OUT.jsonl"""
import sys, json, glob
d, outp = sys.argv[1:3]
rows = [json.loads(l) for l in open("/workspace/jobs/e2/refs_flores_plus.jsonl")]
hyps = []
for f in sorted(glob.glob(f"{d}/work/*/gen_flores_plus_*.jsonl")):
    L = [json.loads(l) for l in open(f)]
    hyps += [{"hyp": o["hyp"], "raw": o["raw"]} for o in L[1:]]
assert len(hyps) == len(rows), (len(hyps), len(rows))
with open(outp, "w") as o:
    for r, h in zip(rows, hyps):
        o.write(json.dumps({"benchmark": "flores_plus", "iso": r["iso"], "id": r["id"], "hyp": h["hyp"]}, ensure_ascii=False) + "\n")
print("ok", len(rows))
