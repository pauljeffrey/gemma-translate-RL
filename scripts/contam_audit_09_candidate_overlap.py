#!/usr/bin/env python3
"""Overlap of candidate FRESH English->African test sets with what our models saw (2026-10-26 audit, agent C1, section 5).

Candidates (English side only, public, read-only): WMT24++ en-sw_KE / en-sw_TZ / en-zu_ZA (google/wmt24pp), SSA-MTE English sources of all splits (McGill-NLP/SSA-MTE),
Gamayun kits core v1 English (CLEAR-Global/Gamayun-kits), AfriMTE-WMT2024 English sources, NTREX is not re-tested (already flagged).
Reference texts it is matched against (everything that is still reachable):
  rl_v3   sft_build/gemma/rl_prompts_v3/*.parquet (5.94 M RL prompt windows = every phase of RL-final and the general 30% of final2) -- private Hub backup, streamed
  rl_sci  r10/rl_prompts_sci_v2 (science pools of the final run)
  sft_pool the 87,498-window random sample of the SFT text (r10/pool)
  flores  FLORES+ dev + devtest English, and the other paper benchmarks' English (sys_rl.jsonl.gz of the evaluation)
Rule (same as gemma_mt/eval/contamination.py): normalised sentence equality, plus the share of 8-token n-grams of the candidate sentence found in the pool.
Writes counts to reports/data/contamination_audit_2026-10-26/candidate_overlap.csv (no candidate text).
"""
import csv, gzip, json, os, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from contam_audit_common import ARCHIVE, OUT, hf_token, normalise
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

from contam_audit_common import CACHE
N = 8
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def jl(repo, path):
    return [json.loads(l) for l in open(hf_hub_download(repo, path, repo_type="dataset", token=hf_token(), cache_dir=CACHE), encoding="utf-8") if l.strip()]


cands = defaultdict(set)
for cfg in ("en-sw_KE", "en-sw_TZ", "en-zu_ZA"):
    for r in jl("google/wmt24pp", cfg + ".jsonl"):
        cands["wmt24pp"].add(normalise(r["source"]))
for sp in ("train", "dev", "test"):
    for lang in ("Amharic", "Hausa", "Igbo", "Kikuyu", "Kinyarwanda", "Luo", "Twi", "Yoruba", "Zulu", "Swahili", "Sesotho"):
        for r in jl("McGill-NLP/SSA-MTE", f"{sp}/{lang}.json"):
            cands["ssa_mte_" + sp].add(normalise(r["src_sent"]))
for l in open(hf_hub_download("CLEAR-Global/Gamayun-kits", "core/v1/en/kit5k.tsv", repo_type="dataset", token=hf_token(), cache_dir=CACHE), encoding="utf-8"):
    p = l.rstrip("\n").split("\t")
    if len(p) > 1:
        cands["gamayun_kit5k"].add(normalise(p[-1]))
for f in ("data/all/test.jsonl",):
    try:
        for r in jl("masakhane/AfriMTE-WMT2024", f):
            if r.get("source_language", "").startswith("en"):
                cands["afrimte_wmt24"].add(normalise(r["source"]))
    except Exception as e:
        print("afrimte skip", str(e)[:60])
bench = defaultdict(set)
for sp, repo in (("devtest", "openlanguagedata/flores_plus"), ("dev", "openlanguagedata/flores_plus")):
    for r in jl(repo, f"{sp}/eng_Latn.jsonl"):
        bench["flores_" + sp].add(normalise(r["text"]))
with gzip.open(os.path.join(ARCHIVE, "box", "jobs", "e1", "sys_rl.jsonl.gz"), "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        bench["bench_" + r["b"]].add(normalise(r["en"]))
cands = {k: {s for s in v if len(s.split()) >= 3} for k, v in cands.items()}
print({k: len(v) for k, v in cands.items()}, flush=True)


def grams(s):
    t = s.split()
    return {hash(tuple(t[i:i + N])) for i in range(len(t) - N + 1)}


allc = {}
for k, v in cands.items():
    for s in v:
        allc.setdefault(s, set()).add(k)
cg = {s: grams(s) for s in allc}
g2s = defaultdict(set)
for s, gs in cg.items():
    for g in gs:
        g2s[g].add(s)
hit_exact = defaultdict(set)      # source -> candidate sentences found as a sentence
hit_gram = defaultdict(lambda: defaultdict(set))     # source -> sentence -> grams found


def scan_text(src, text):
    for s in SPLIT.split(text):
        n = normalise(s)
        if n in allc:
            hit_exact[src].add(n)
    toks = normalise(text).split()
    for i in range(len(toks) - N + 1):
        g = hash(tuple(toks[i:i + N]))
        c = g2s.get(g)
        if c:
            for s in c:
                hit_gram[src][s].add(g)


fs = HfFileSystem(token=hf_token())
import pyarrow.parquet as pq
for src, repo, prefix in (("rl_v3", (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-box-final-backup"), "sft_build/gemma/rl_prompts_v3"), ("rl_sci", (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-box-backup-r10-weak-language-data"), "r10/rl_prompts_sci_v2")):
    seen = set()
    files = sorted(p for p in fs.find(f"datasets/{repo}/{prefix}") if p.endswith(".parquet"))
    for i, p in enumerate(files):
        with fs.open(p, "rb") as f:
            col = pq.ParquetFile(f).read(columns=["english"]).column("english").to_pylist()
        for t in col:
            h = hash(t)
            if h in seen:
                continue
            seen.add(h)
            scan_text(src, t)
        print(src, i + 1, "/", len(files), len(seen), flush=True)
with fs.open(("datasets/" + os.environ['HF_ORG_LEGACY'] + "/sabiyarn-box-backup-r10-weak-language-data/r10/pool/_counts.json"), "rb") as f:
    isos = list(json.loads(f.read()).keys())
for iso in isos:
    with fs.open(f"datasets/{os.environ['HF_ORG_LEGACY']}/sabiyarn-box-backup-r10-weak-language-data/r10/pool/{iso}.jsonl", "rb") as f:
        for line in f:
            scan_text("sft_pool", json.loads(line)["english"])
rows = []
for k, v in {**cands, **bench}.items():
    if k in bench:
        continue
    r = {"candidate": k, "sentences": len(v)}
    for src in ("rl_v3", "rl_sci", "sft_pool"):
        ex = sum(1 for s in v if s in hit_exact[src])
        gr = [len(hit_gram[src][s]) / max(1, len(cg[s])) for s in v if cg[s]]
        r[f"{src}_exact"] = ex
        r[f"{src}_exact_pct"] = round(100 * ex / len(v), 2)
        r[f"{src}_any8gram_pct"] = round(100 * sum(1 for s in v if hit_gram[src].get(s)) / len(v), 2)
        r[f"{src}_ge50pct8gram_pct"] = round(100 * sum(1 for x, s in zip(gr, [s for s in v if cg[s]]) if x >= 0.5) / len(v), 2)
    rows.append(r)
    print(r)
with open(os.path.join(OUT, "candidate_overlap.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
