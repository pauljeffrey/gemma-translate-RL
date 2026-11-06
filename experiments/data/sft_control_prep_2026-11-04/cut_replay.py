#!/usr/bin/env python
"""Replay half of the continued-SFT control: cut N packs out of the EXISTING stage-B packs (read-only ranged reads from the object-storage bucket, nothing else downloaded) into a small
compact packed dataset (train/ + meta.json), language-balanced by UniMax.

Pool = the packs the model saw in its LAST SFT shard:
    s6   clean_rerun_2026-11-01/shards/s6/train           154,421 packs (stage-B shard 6 as uploaded)
    tail clean_rerun_2026-11-01/g0/packed_g0/train        30,627 old A0 translation packs of the 40 languages with < 2,000 old packs (the tail-language replay that the box appended
                                                          to shards 2..6 with make_replay.py; the selection rule is copied from make_replay.py: first n_old packs, language count < max_packs)
Selection: unimax_allocation(pool packs per language, budget = N, min_epochs 0, max_epochs 1) -> per-language draw counts (every pack at most once; tail languages replayed in full,
large languages cut to the common water level), then a seeded random choice of that many packs inside each language. Tokens are copied verbatim, so the replay packs are
byte-identical to what shard 6 trained on (tag mode single_token, 4,096-token one-language packs).

    AK=.. SK=.. python cut_replay.py --s6-meta DIR --g0-meta DIR --n-packs 96000 --out replay_sel [--seed 1234] [--threads 32]
    # laptop test: --n-packs 600 (about 10 MB)
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.environ.get("GEMMA_MT_PKG"), os.path.join(HERE, "..", "..", "..", "gemma_mt")):
    if p and os.path.isdir(os.path.join(p, "gemma_mt")):
        sys.path.insert(0, p); break
from gemma_mt.data.packed_dataset import unimax_allocation
from gemma_mt.languages import ALL_ISOS

NAMES = ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id", "direction", "pack_start")


def load(d):
    return {n: np.load(f"{d}/train/{n}.npy") for n in NAMES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s6-meta", required=True); ap.add_argument("--g0-meta", required=True); ap.add_argument("--n-packs", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-packs", type=int, default=2000); ap.add_argument("--n-old", type=int, default=138873)
    ap.add_argument("--s6-key", default="clean_rerun_2026-11-01/shards/s6/train/tokens.u32.bin"); ap.add_argument("--g0-key", default="clean_rerun_2026-11-01/g0/packed_g0/train/tokens.u32.bin")
    ap.add_argument("--dry-plan", action="store_true", help="print the per-language plan and stop (no bucket access)")
    a = ap.parse_args()
    S, G = load(a.s6_meta), load(a.g0_meta)
    pl6 = S["iso_id"][S["pack_start"][:-1].astype(np.int64)].astype(np.int64)
    ps_g = G["pack_start"].astype(np.int64); plg = G["iso_id"][ps_g[:-1]].astype(np.int64)
    cnt = np.bincount(plg[:a.n_old], minlength=len(ALL_ISOS))
    sel_lang = np.flatnonzero((cnt > 0) & (cnt < a.max_packs))
    tail_ids = np.flatnonzero(np.isin(plg[:a.n_old], sel_lang))                 # indices into g0 packs
    pool_lang = np.concatenate([pl6, plg[tail_ids]]); n6 = len(pl6)
    langs, counts = np.unique(pool_lang, return_counts=True)
    alloc = unimax_allocation(counts, a.n_packs, min_epochs=0.0, max_epochs=1.0)
    rng = np.random.default_rng([a.seed, 99])
    chosen = np.concatenate([rng.choice(np.flatnonzero(pool_lang == l), int(k), replace=False) for l, k in zip(langs, alloc)])
    chosen = np.sort(chosen)                                                    # pool order (s6 first, then tail): ranged reads are sequential
    plan = {ALL_ISOS[l]: {"pool_packs": int(c), "selected": int(k)} for l, c, k in zip(langs, counts, alloc)}
    print(f"pool {len(pool_lang)} packs ({n6} s6 + {len(tail_ids)} tail) in {len(langs)} languages; selecting {len(chosen)}; languages fully replayed: {int((alloc >= counts).sum())}", flush=True)
    os.makedirs(a.out, exist_ok=True); json.dump(plan, open(f"{a.out}/replay_plan.json", "w"), indent=1)
    if a.dry_plan: return
    import boto3
    from botocore.config import Config
    s3 = boto3.client("s3", endpoint_url=os.environ['S3_ENDPOINT_URL'], aws_access_key_id=os.environ["AK"], aws_secret_access_key=os.environ["SK"], region_name=os.environ.get("S3_REGION", "us-east-1"),
                      config=Config(max_pool_connections=a.threads + 4, retries={"max_attempts": 8}))
    B = os.environ['S3_BUCKET']
    parts = []                                                                  # (source arrays, key, pack index inside that source)
    for c in chosen:
        parts.append((S, a.s6_key, int(c)) if c < n6 else (G, a.g0_key, int(tail_ids[c - n6])))
    new = {n: [] for n in ("sample_len", "prompt_len", "target_len", "iso_id", "direction")}; counts_per_pack = []; tok_chunks = [None] * len(parts)

    def get(i):
        A, key, p = parts[i]
        ps = A["pack_start"].astype(np.int64); lo_s, hi_s = ps[p], ps[p + 1]
        so, sl = A["sample_offset"].astype(np.int64), A["sample_len"].astype(np.int64)
        lo, hi = so[lo_s], so[hi_s - 1] + sl[hi_s - 1]
        r = s3.get_object(Bucket=B, Key=key, Range=f"bytes={int(lo) * 4}-{int(hi) * 4 - 1}")["Body"].read()
        tok_chunks[i] = np.frombuffer(r, np.uint32)
    with ThreadPoolExecutor(a.threads) as ex:
        for j, _ in enumerate(ex.map(get, range(len(parts)))):
            if j % 10000 == 0: print("fetched", j, flush=True)
    for (A, key, p) in parts:
        ps = A["pack_start"].astype(np.int64); sl_ = slice(ps[p], ps[p + 1])
        for n in new: new[n].append(A[n][sl_])
        counts_per_pack.append(ps[p + 1] - ps[p])
    os.makedirs(f"{a.out}/train", exist_ok=True)
    toks = np.concatenate(tok_chunks); toks.tofile(f"{a.out}/train/tokens.u32.bin")
    arr = {n: np.concatenate(v) for n, v in new.items()}
    assert int(arr["sample_len"].astype(np.int64).sum()) == len(toks), "token count != sum of sample lengths"
    arr["pack_start"] = np.concatenate([[0], np.cumsum(counts_per_pack)]).astype(S["pack_start"].dtype)
    arr["sample_offset"] = np.concatenate([[0], np.cumsum(arr["sample_len"].astype(np.int64))[:-1]]).astype(S["sample_offset"].dtype)
    for n, v in arr.items(): np.save(f"{a.out}/train/{n}.npy", v.astype(S[n].dtype))
    meta = json.load(open(f"{a.s6_meta}/meta.json")); meta.pop("extension", None)
    meta["replay_of"] = {"s6": a.s6_key, "g0": a.g0_key, "n_packs": int(len(chosen)), "seed": a.seed, "rule": "unimax min 0 max 1 over the s6 + tail pool"}
    os.makedirs(f"{a.out}/val", exist_ok=True)                                  # validation = the stage-B shard-6 validation split (78 languages; the SFT's own 1.3412 macro / 1.2846 micro are on it)
    for n in NAMES + ("tokens.u32",):
        fn = f"{n}.npy" if n in NAMES else "tokens.u32.bin"
        s3.download_file(B, a.s6_key.replace("train/tokens.u32.bin", "val/" + fn), f"{a.out}/val/{fn}")
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    print(f"wrote {a.out}: {len(chosen)} packs, {len(toks) / 1e6:.1f} M tokens, {int(arr['pack_start'][-1])} samples")


if __name__ == "__main__":
    main()
