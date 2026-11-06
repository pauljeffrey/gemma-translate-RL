#!/usr/bin/env python
"""Sanity scan of a SAMPLE of the replay pool (stage-B shard 6 packs in the object-storage bucket, read-only ranged reads) against the protected index, with the same machinery as
tp_prepare.py. The replay text itself was scanned in full by the 385M contamination audit (reports/contamination_audit_385m_2026-11-03.md: s1-s6 and g0, 25,911,095 samples);
this script re-checks a stratified sample of packs with the INTAKE filter's rules (R1-R3, both normal forms, strings < 3 tokens never indexed) so that the control's replay half
is covered by the same rule as its TranslatePsy half.

    AK=... SK=... python replay_sample_scan.py --meta-dir s6_meta --tokenizer DIR --index I.npz --manifest M.json --decon-dir D --n-packs 600 --out report.json
"""
import argparse, collections, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--meta-dir", required=True); ap.add_argument("--tokenizer", required=True); ap.add_argument("--index", required=True); ap.add_argument("--manifest", required=True)
ap.add_argument("--decon-dir", required=True); ap.add_argument("--n-packs", type=int, default=600); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--prefix", default="clean_rerun_2026-11-01/shards/s6/train/tokens.u32.bin")
a = ap.parse_args()
sys.path.insert(0, a.decon_dir); sys.path.insert(0, "gemma_mt")
import boto3
from botocore.config import Config
from transformers import AutoTokenizer
import protected_index_loader as L
from gemma_mt.languages import ALL_ISOS
T = {n: np.load(f"{a.meta_dir}/train/{n}.npy") for n in ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id", "direction", "pack_start")}
ps = T["pack_start"].astype(np.int64); so = T["sample_offset"].astype(np.int64); sl = T["sample_len"].astype(np.int64)
plang = T["iso_id"][ps[:-1]].astype(np.int64)
rng = np.random.default_rng(a.seed)
chosen = []                                   # stratified: the same number of packs from every language that has packs (capped by what it has)
per = max(1, a.n_packs // len(np.unique(plang)))
for l in np.unique(plang):
    m = np.flatnonzero(plang == l); chosen += list(rng.choice(m, min(per, len(m)), replace=False))
tok = AutoTokenizer.from_pretrained(a.tokenizer)
s3 = boto3.client("s3", endpoint_url=os.environ['S3_ENDPOINT_URL'], aws_access_key_id=os.environ["AK"], aws_secret_access_key=os.environ["SK"], region_name=os.environ.get("S3_REGION", "us-east-1"),
                  config=Config(retries={"max_attempts": 5}, max_pool_connections=8))
ix, man = L.load(a.index, a.manifest)
rules = ("R1_exact", "R2_contiguous", "R3_8gram")
rep = {"index_version": man["version"], "packs": len(chosen), "samples": 0, "texts_scanned": 0, "rows_with_hit": 0, "rows_by_set": collections.Counter(), "hits_by_rule": collections.Counter(), "kind": collections.Counter()}
texts, meta_ = [], []
for k, p in enumerate(chosen):
    lo, hi = so[ps[p]], so[ps[p + 1] - 1] + sl[ps[p + 1] - 1]
    buf = np.frombuffer(s3.get_object(Bucket=os.environ['S3_BUCKET'], Key=a.prefix, Range=f"bytes={int(lo) * 4}-{int(hi) * 4 - 1}")["Body"].read(), np.uint32)
    for s in range(ps[p], ps[p + 1]):
        o = so[s] - lo; n = sl[s]; pl = int(T["prompt_len"][s]); tl = int(T["target_len"][s])
        ids = buf[o:o + n].tolist()
        pr = tok.decode(ids[:pl]); tg = tok.decode(ids[pl:pl + tl])
        if "Translate the text to" in pr:
            pr = pr.split(":  ", 1)[-1].replace("<end_of_turn>\n<start_of_turn>model\n", ""); rep["kind"]["translation"] += 1
        else:
            rep["kind"]["other"] += 1
        texts.append((pr, tg)); rep["samples"] += 1
for i in range(0, len(texts), 2000):
    ch = texts[i:i + 2000]
    rs = ix.check_texts([c[0] for c in ch]); rt = ix.check_texts([c[1] for c in ch])
    for x, y in zip(rs, rt):
        rep["texts_scanned"] += 2
        if any(x[:3]) or any(y[:3]): rep["rows_with_hit"] += 1
        for r in (x, y):
            for rule, m in zip(rules, r[:3]):
                if m: rep["hits_by_rule"][rule] += 1
            for nm in ix.names(r[0] | r[1] | r[2]): rep["rows_by_set"][nm] += 1
for k in ("rows_by_set", "hits_by_rule", "kind"): rep[k] = dict(rep[k])
json.dump(rep, open(a.out, "w"), indent=1); print(json.dumps(rep, indent=1)[:1500])
