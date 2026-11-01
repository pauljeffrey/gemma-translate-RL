import sys, json
sys.path.insert(0, "/workspace/sft_build/gemma_mt_2c"); sys.path.append("/workspace/sft_build/pylibs")
from gemma_mt.eval import benchmarks as B, contamination as C
rows = B.load_benchmark("flores_plus", path="/workspace/sft_build/bench/test_sets/flores_plus")
ff = {"flores_plus": "/workspace/sft_build/bench/test_sets/_contamination/contamination_flores_plus.json"}
print(C.apply_flags(rows, ff))
with open("refs_flores_plus.jsonl", "w") as f:
    for r in rows: f.write(json.dumps({k: r.get(k) for k in ("iso","id","english","reference","contaminated","pair_seen")}, ensure_ascii=False) + "\n")
print(len(rows))
