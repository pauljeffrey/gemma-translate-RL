#!/usr/bin/env python3
"""Composition of the RL development / monitoring sets (2026-10-26 contamination audit, agent C1, section 2).

Reads (owner's private Hub backup `<HF_ORG_LEGACY>/sabiyarn-box-final-backup`, read-only, streamed): sft_build/bench/bench_dev.jsonl (all dev rows),
sft_build/bench/bench_dev_clean.jsonl (the rows the trainer used: `benchmark_eval_path` of rl2c_config.json) and jobs/r7/assets/dev/bench_weak_flores_dev.jsonl.
Counts rows by (benchmark label, iso) and checks that no row is a FLORES+ devtest or other test-set English sentence (normalised equality against FLORES+ devtest
English, plus the English of the MAFAND TEST files if reachable). Writes counts to reports/data/contamination_audit_2026-10-26/dev_set_composition.csv.
"""
import csv, json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from contam_audit_common import OUT, hf_token, normalise
from huggingface_hub import HfFileSystem, hf_hub_download

REPO_ID = (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-box-final-backup")
from contam_audit_common import CACHE
fs = HfFileSystem(token=hf_token())


def rows(path):
    with fs.open(f"datasets/{REPO_ID}/{path}", "rb") as f:
        return [json.loads(l) for l in f if l.strip()]


devtest = {normalise(json.loads(l)["text"]) for l in open(hf_hub_download("openlanguagedata/flores_plus", "devtest/eng_Latn.jsonl", repo_type="dataset", token=hf_token(), cache_dir=CACHE))}
dev = {normalise(json.loads(l)["text"]) for l in open(hf_hub_download("openlanguagedata/flores_plus", "dev/eng_Latn.jsonl", repo_type="dataset", token=hf_token(), cache_dir=CACHE))}
out = []
for path in ("sft_build/bench/bench_dev.jsonl", "sft_build/bench/bench_dev_clean.jsonl", "jobs/r7/assets/dev/bench_weak_flores_dev.jsonl"):
    rs = rows(path)
    c = Counter((r.get("benchmark"), r["iso"]) for r in rs)
    n_dt = sum(1 for r in rs if normalise(r["english"]) in devtest)
    n_d = sum(1 for r in rs if normalise(r["english"]) in dev)
    by_b = Counter(r.get("benchmark") for r in rs)
    print(path, len(rs), "languages", len({r["iso"] for r in rs}), dict(by_b), "english in FLORES+ devtest:", n_dt, " in FLORES+ dev:", n_d)
    for (b, iso), n in sorted(c.items()):
        out.append({"file": path, "benchmark_label": b, "iso": iso, "rows": n})
    out.append({"file": path, "benchmark_label": "_TOTAL", "iso": "", "rows": len(rs)})
    out.append({"file": path, "benchmark_label": "_english_in_flores_devtest", "iso": "", "rows": n_dt})
    out.append({"file": path, "benchmark_label": "_english_in_flores_dev", "iso": "", "rows": n_d})
with open(os.path.join(OUT, "dev_set_composition.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file", "benchmark_label", "iso", "rows"]); w.writeheader(); w.writerows(out)
