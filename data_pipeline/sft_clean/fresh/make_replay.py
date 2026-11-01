#!/usr/bin/env python3
"""Replay set for the tail languages (growth 1, stage B): the old A0 translation packs (retagged) of every language that has fewer than --max-packs of them.

Stage B draws each pack once, so a language that has few packs and few new ones disappears from the mix for hours (measured: macro validation loss 1.785 -> 2.03 after the unfreeze, worst on
tiny languages). The replay set is cut out of the packed_g0 object in the bucket with ranged reads (only the selected packs are downloaded, not the 7.3 GB) and written as a packed dataset
(train/ + meta.json) that `extend_packed.py --inplace` can append to a shard.

    python make_replay.py --out /workspace/work/run/replay --max-packs 2000 --n-old 138873
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

NAMES = ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id", "direction", "pack_start")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--max-packs", type=int, default=2000); ap.add_argument("--n-old", type=int, default=138873)
    ap.add_argument("--prefix", default=(os.environ['S3_PREFIX'] + "/")); ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    for l in open(os.environ.get("ENV_FILE", ".env")):
        k, v = l.strip().split("=", 1); os.environ[k] = v
    sys.path.insert(0, "/workspace/s3lib")
    import boto3
    from botocore.config import Config
    s3 = boto3.client("s3", endpoint_url=os.environ['S3_ENDPOINT_URL'], aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"], region_name=os.environ.get("S3_REGION", "us-east-1"),
                      config=Config(max_pool_connections=a.threads + 4, retries={"max_attempts": 8}))
    B = os.environ['S3_BUCKET']; tmp = "/tmp/g0a_replay"; os.makedirs(tmp + "/train", exist_ok=True)
    for n in NAMES:
        s3.download_file(B, a.prefix + f"train/{n}.npy", f"{tmp}/train/{n}.npy")
    s3.download_file(B, a.prefix + "meta.json", tmp + "/meta.json")
    A = {n: np.load(f"{tmp}/train/{n}.npy") for n in NAMES}
    ps = A["pack_start"].astype(np.int64); so = A["sample_offset"].astype(np.int64); sl = A["sample_len"].astype(np.int64)
    plang = A["iso_id"][ps[:-1]].astype(np.int64)
    cnt = np.bincount(plang[:a.n_old], minlength=int(plang.max()) + 1)
    sel_lang = np.flatnonzero((cnt > 0) & (cnt < a.max_packs))
    P = np.flatnonzero(np.isin(plang[:a.n_old], sel_lang))
    print(f"{len(sel_lang)} languages with < {a.max_packs} old packs -> {len(P)} packs", flush=True)
    key = a.prefix + "train/tokens.u32.bin"
    lo = so[ps[P]]; hi = so[ps[P + 1] - 1] + sl[ps[P + 1] - 1]
    ntok = int((hi - lo).sum()); print("tokens", ntok, f"({ntok * 4 / 1e9:.2f} GB)", flush=True)
    buf = np.empty(ntok, np.uint32); pos = np.concatenate([[0], np.cumsum(hi - lo)])

    def get(i):
        r = s3.get_object(Bucket=B, Key=key, Range=f"bytes={int(lo[i]) * 4}-{int(hi[i]) * 4 - 1}")["Body"].read()
        buf[pos[i]:pos[i + 1]] = np.frombuffer(r, np.uint32)
    with ThreadPoolExecutor(a.threads) as ex:
        for j, _ in enumerate(ex.map(get, range(len(P)))):
            if j % 5000 == 0: print("fetched", j, flush=True)
    cntp = ps[P + 1] - ps[P]
    first = np.repeat(ps[P] - np.concatenate([[0], np.cumsum(cntp)[:-1]]), cntp) + np.arange(int(cntp.sum()))
    new = {n: A[n][first] for n in ("sample_len", "prompt_len", "target_len", "iso_id", "direction")}
    new["pack_start"] = np.concatenate([[0], np.cumsum(cntp)]).astype(A["pack_start"].dtype)
    new["sample_offset"] = np.concatenate([[0], np.cumsum(sl[first])[:-1]]).astype(A["sample_offset"].dtype)
    assert int(sl[first].sum()) == ntok
    os.makedirs(a.out + "/train", exist_ok=True)
    buf.tofile(a.out + "/train/tokens.u32.bin")
    for n, v in new.items(): np.save(f"{a.out}/train/{n}.npy", v)
    meta = json.load(open(tmp + "/meta.json")); meta.pop("extension", None); meta["replay_of"] = a.prefix; meta["replay_max_packs"] = a.max_packs
    json.dump(meta, open(a.out + "/meta.json", "w"), indent=1)
    table = meta["iso_table"]
    print("replay languages:", sorted((table[l], int(cnt[l])) for l in sel_lang)[:60])
    print(f"wrote {a.out}: {len(P)} packs, {ntok / 1e6:.0f}M tokens")


if __name__ == "__main__":
    main()
