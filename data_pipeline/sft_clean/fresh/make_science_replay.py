#!/usr/bin/env python3
"""General-SFT replay slice for the science SFT stage of the 385M model (packs cut out of the stage-B shards / packed_g0 with ranged reads; no decoding, no re-tokenising).

    python make_science_replay.py --out /workspace/science_sft/replay_train --tokens 11800000 --science-langs amh,hau,lug,nso,yor,zul --split train
    python make_science_replay.py --out /workspace/science_sft/replay_val --per-language-packs 3 --split val --prefixes <S3_PREFIX>/g0/packed_g0/

Selection (A2W recipe, adapted): translation packs only (first sample direction 0 = en2x) of the source packed datasets; UniMax water-filling over the languages: every language gets
the same token budget c (a language with less data gives all of it), the --science-langs get --sci-weight (2) x c and the --weak-langs get --weak-scale (0.5) x c; packs are drawn at random
inside a language without replacement; the budget is solved so that the total is --tokens. The output is a packed dataset (train/ arrays + tokens.u32.bin + meta.json of the source,
so the tokenizer fingerprint, seq_len and iso table are those of the shards) that chain/extend_packed.py can append to the science packs. Reads the bucket (keys from /root/.bucket_env).
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

NAMES = ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id", "direction", "pack_start")
DEFAULT_TRAIN = [f"{os.environ['S3_PREFIX']}/shards/s{i}/" for i in (2, 3, 4, 5, 6)] + [os.environ['S3_PREFIX'] + "/g0/packed_g0/"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--tokens", type=int, default=11_800_000); ap.add_argument("--per-language-packs", type=int, default=0, help="val: this many packs per language instead of a token budget")
    ap.add_argument("--prefixes", nargs="*", default=None); ap.add_argument("--science-langs", default="amh,hau,lug,nso,yor,zul"); ap.add_argument("--sci-weight", type=float, default=2.0)
    ap.add_argument("--weak-langs", default=""); ap.add_argument("--weak-scale", type=float, default=0.5); ap.add_argument("--exclude-langs", default="arz,zgh,shi,dik,aeb,ary,taq,tzm")
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    for l in open("/root/.bucket_env"):
        k, v = l.strip().split("=", 1); os.environ[k] = v
    import boto3
    from botocore.config import Config
    s3 = boto3.client("s3", endpoint_url=os.environ['S3_ENDPOINT_URL'], aws_access_key_id=os.environ["AK"], aws_secret_access_key=os.environ["SK"], region_name=os.environ.get("S3_REGION", "us-east-1"),
                      config=Config(max_pool_connections=a.threads + 4, retries={"max_attempts": 8}))
    B = os.environ['S3_BUCKET']; prefixes = a.prefixes or (DEFAULT_TRAIN if a.split == "train" else [os.environ['S3_PREFIX'] + "/g0/packed_g0/"])
    tmp = "/tmp/sci_replay_arrays"; os.makedirs(tmp, exist_ok=True)
    rng = np.random.default_rng(a.seed); cand = []                       # (source index, pack index, language id, tokens)
    sources = []
    for si, pre in enumerate(prefixes):
        d = f"{tmp}/{si}"; os.makedirs(d, exist_ok=True)
        for n in NAMES: s3.download_file(B, f"{pre}{a.split}/{n}.npy", f"{d}/{n}.npy")
        s3.download_file(B, pre + "meta.json", f"{d}/meta.json")
        A = {n: np.load(f"{d}/{n}.npy") for n in NAMES}; meta = json.load(open(f"{d}/meta.json"))
        ps = A["pack_start"].astype(np.int64); so = A["sample_offset"].astype(np.int64); sl = A["sample_len"].astype(np.int64)
        if a.split == "train" and si > 0 and meta["tokenizer"]["fingerprint"] != sources[0][1]["tokenizer"]["fingerprint"]: raise SystemExit("tokenizer differs between sources")
        sources.append((pre, meta, A, ps, so, sl))
        first = ps[:-1]; lang = A["iso_id"][first].astype(np.int64); dire = A["direction"][first]
        ntok = (so[ps[1:] - 1] + sl[ps[1:] - 1]) - so[first]
        for p in np.flatnonzero(dire == 0): cand.append((si, int(p), int(lang[p]), int(ntok[p])))
        print("source", pre, a.split, "packs", len(first), "translation packs", int((dire == 0).sum()), flush=True)
    table = sources[0][1]["iso_table"]
    excl = {table.index(x) for x in a.exclude_langs.split(",") if x in table}
    sci = {table.index(x) for x in a.science_langs.split(",") if x in table}; weak = {table.index(x) for x in a.weak_langs.split(",") if x in table}
    by = {}
    for c in cand:
        if c[2] not in excl: by.setdefault(c[2], []).append(c)
    avail = {l: sum(c[3] for c in v) for l, v in by.items()}
    wt = {l: (a.sci_weight if l in sci else a.weak_scale if l in weak else 1.0) for l in by}
    chosen = []
    if a.split == "val" or a.per_language_packs:
        for l, v in by.items():
            idx = rng.permutation(len(v))[: a.per_language_packs or 3]; chosen += [v[i] for i in idx]
    else:
        lo, hi = 0.0, 1e9
        for _ in range(80):
            mid = (lo + hi) / 2
            if sum(min(avail[l], wt[l] * mid) for l in by) < a.tokens: lo = mid
            else: hi = mid
        c_tok = hi
        for l, v in by.items():
            budget = min(avail[l], wt[l] * c_tok); got = 0
            for i in rng.permutation(len(v)):
                if got >= budget: break
                chosen.append(v[i]); got += v[i][3]
        print(f"water level c = {c_tok:,.0f} tokens per language; languages {len(by)}; chosen packs {len(chosen)}", flush=True)
    chosen.sort(key=lambda c: (c[0], c[1]))
    ntok_all = sum(c[3] for c in chosen); print("replay tokens", ntok_all, "packs", len(chosen), flush=True)
    new = {n: [] for n in NAMES if n != "pack_start"}; pack_sizes = []; buf = np.empty(ntok_all, np.uint32); pos = 0; jobs = []
    for si, pi, l, nt in chosen:
        pre, meta, A, ps, so, sl = sources[si]
        lo_, hi_ = int(so[ps[pi]]), int(so[ps[pi + 1] - 1] + sl[ps[pi + 1] - 1]); jobs.append((si, pi, lo_, hi_, pos)); pos += hi_ - lo_
    def get(j):
        si, pi, lo_, hi_, p0 = j; pre = sources[si][0]
        r = s3.get_object(Bucket=B, Key=pre + f"{a.split}/tokens.u32.bin", Range=f"bytes={lo_ * 4}-{hi_ * 4 - 1}")["Body"].read()
        buf[p0:p0 + (hi_ - lo_)] = np.frombuffer(r, np.uint32)
    with ThreadPoolExecutor(a.threads) as ex: list(ex.map(get, jobs))
    off = 0; out = {n: [] for n in NAMES if n != "pack_start"}; starts = [0]
    for (si, pi, lo_, hi_, p0) in jobs:
        pre, meta, A, ps, so, sl = sources[si]; s0, s1 = int(ps[pi]), int(ps[pi + 1])
        for n in ("sample_len", "prompt_len", "target_len", "iso_id", "direction"): out[n].append(A[n][s0:s1])
        out["sample_offset"].append((so[s0:s1] - lo_ + p0).astype(np.uint64)); starts.append(starts[-1] + (s1 - s0))
    os.makedirs(a.out + f"/{a.split}", exist_ok=True)
    buf.tofile(a.out + f"/{a.split}/tokens.u32.bin")
    ref = sources[0][2]
    for n, v in out.items(): np.save(f"{a.out}/{a.split}/{n}.npy", np.concatenate(v).astype(ref[n].dtype))
    np.save(f"{a.out}/{a.split}/pack_start.npy", np.asarray(starts, dtype=ref["pack_start"].dtype))
    meta = dict(sources[0][1]); meta.pop("extension", None); meta["replay_of"] = prefixes; meta["replay_split"] = a.split
    json.dump(meta, open(a.out + "/meta.json", "w"), indent=1)
    per_lang = {}
    for si, pi, l, nt in chosen: per_lang[table[l]] = per_lang.get(table[l], 0) + nt
    json.dump({"tokens": ntok_all, "packs": len(chosen), "tokens_by_language": per_lang}, open(a.out + "/replay_stats.json", "w"), indent=1)
    print("wrote", a.out, "languages", len(per_lang), flush=True)


if __name__ == "__main__":
    main()
