"""R14: dump the human-reference benchmark rows (with contamination flags) for the NLLB-covered languages -> jsonl. Runs on the box with the gemma_mt tree of sft_build."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nllb_langmap import EXACT
from gemma_mt.eval import benchmarks as B, contamination as C
ROOT = "/workspace/sft_build/bench"
TS = ROOT + "/test_sets"
isos = sorted(EXACT)
out = []
for b in ["flores_plus", "ntrex", "mafand", "smol_sent", "salt"]:
    rows = B.load_benchmark(b, path=os.path.join(TS, b), isos=isos)
    files = C.default_flag_files(TS + "/_contamination", [b])
    miss = C.apply_flags(rows, files)
    print(b, len(rows), sorted({r["iso"] for r in rows}).__len__(), "langs; rows without flag", miss, flush=True)
    out += rows
snap = [p for p in __import__("glob").glob("/workspace/sft_build/hf/hub/datasets--openlanguagedata--flores_plus/snapshots/*")][0]
rows = B.load_flores_plus(isos, "dev", snap)
for r in rows:
    r["benchmark"] = "flores_plus_dev"; r["contaminated"] = r["pair_seen"] = None
print("flores_plus_dev", len(rows), len({r["iso"] for r in rows}), "langs", flush=True)
out += rows
with open("/workspace/jobs/r14/out/bench_rows.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps({k: r.get(k) for k in ("benchmark", "iso", "id", "english", "reference", "contaminated", "pair_seen")}, ensure_ascii=False) + "\n")
print("total", len(out))
