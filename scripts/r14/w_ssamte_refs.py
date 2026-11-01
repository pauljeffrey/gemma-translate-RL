"""R14 widening: NLLB-3.3B translations of the English sources of SSA-MTE (human DA scores) for the candidate languages -> /workspace/jobs/r14/s3/ssamte_refs_3b3.json."""
import json, sys
from gemma_mt.rl.nllb_signal import NllbCT2
m = NllbCT2("/workspace/jobs/r14/models/nllb3p3b-ct2-int8", "/workspace/jobs/r14/models/nllb3p3b-ct2-int8", compute_type="int8_float16")
by = {}
for l in open("/workspace/jobs/a5/ssamte_rows.jsonl"):
    r = json.loads(l)
    if r["lp"].startswith("eng-"): by.setdefault(r["lp"][4:], set()).add(r["src"])
out = {}
for iso in ("amh", "kik", "kin", "luo", "hau", "yor", "twi"):
    u = sorted(by.get(iso, [])); 
    if u: out[iso] = dict(zip(u, m.translate(u, iso, max_batch_tokens=3072))); print(iso, len(u), flush=True)
json.dump(out, open("/workspace/jobs/r14/s3/ssamte_refs_3b3.json", "w"), ensure_ascii=False)
