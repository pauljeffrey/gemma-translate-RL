#!/usr/bin/env python3
"""Cut the stage-B pack set into K SHARDS that fit the A100's disk (owner 2026-11-01: "shard stage B if it doesn't fit").

    python make_shards.py --packed work/packed_g --stage-a work/sched_g/A --k 5 --out work/shards [--seed 1234]

Stage B = every pack of packed_g = [A0 old packs retagged | instruction packs | new translation packs retagged] that stage A did not draw, each drawn ONCE. The packs are split into K
shards, language by language (a seeded shuffle, round-robin), so every shard holds about 1/K of every language: each shard is a small copy of the whole mix. Shard k becomes its own packed
dataset work/shards/s<k>/ (train/ with renumbered packs, val/ copied from packed_g, meta.json) and its own schedule work/shards/s<k>/sched/ (every pack once, languages spread over the shard
with the UniMax trick; sidecar pinning the shard's pack structure). A shard is complete when work/shards/s<k>/READY exists.
"""
import argparse, json, os, shutil, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "chain"))
import sampler as S
from make_stage_schedules import finish, sub_schedule, load

NAMES = ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id", "direction", "pack_start")


def write_shard(src, ps, out, P):
    """dataset with only the packs P (sorted ascending) of src (the loaded train arrays), renumbered 0..len(P)-1"""
    A = {n: np.load(os.path.join(src, "train", n + ".npy"), mmap_mode="r") for n in NAMES}
    tok = np.memmap(os.path.join(src, "train", "tokens.u32.bin"), dtype=np.uint32, mode="r")
    ps = A["pack_start"].astype(np.int64)
    cnt = ps[P + 1] - ps[P]
    first = np.repeat(ps[P] - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt) + np.arange(int(cnt.sum()))     # sample indices of the selected packs, in order
    sl_src = np.asarray(A["sample_len"]).astype(np.int64)
    sl = sl_src[first]
    so_src = np.asarray(A["sample_offset"]).astype(np.int64)
    os.makedirs(os.path.join(out, "train"), exist_ok=True)
    new = {"sample_len": np.asarray(A["sample_len"])[first], "prompt_len": np.asarray(A["prompt_len"])[first], "target_len": np.asarray(A["target_len"])[first],
           "iso_id": np.asarray(A["iso_id"])[first], "direction": np.asarray(A["direction"])[first],
           "pack_start": np.concatenate([[0], np.cumsum(cnt)]).astype(A["pack_start"].dtype),
           "sample_offset": np.concatenate([[0], np.cumsum(sl)[:-1]]).astype(A["sample_offset"].dtype)}
    with open(os.path.join(out, "train", "tokens.u32.bin"), "wb") as f:
        for p in P:                                                                  # a pack's tokens are contiguous in the source stream
            a, b = ps[p], ps[p + 1]
            lo, hi = so_src[a], so_src[b - 1] + sl_src[b - 1]
            tok[lo:hi].tofile(f)
    assert os.path.getsize(os.path.join(out, "train", "tokens.u32.bin")) // 4 == int(sl.sum())
    for n, v in new.items():
        np.save(os.path.join(out, "train", n + ".npy"), v)
    if not os.path.exists(os.path.join(out, "val")):
        shutil.copytree(os.path.join(src, "val"), os.path.join(out, "val"))
    meta = json.load(open(os.path.join(src, "meta.json"))); meta.pop("extension", None); meta["shard_of"] = os.path.abspath(src)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--packed", required=True); ap.add_argument("--stage-a", required=True); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--per-step", type=int, default=32)
    a = ap.parse_args(); Pn = a.per_step
    PD, ALL, ds, pack_lang = load(a.packed); n = len(pack_lang)
    A = np.load(os.path.join(a.stage_a, "schedule.npy")).reshape(-1)
    drawn = np.zeros(n, bool); drawn[A] = True
    rest = np.flatnonzero(~drawn)
    rng = np.random.default_rng(a.seed)
    shard_of = np.full(n, -1, np.int64)
    for l in np.unique(pack_lang[rest]):
        idx = rest[pack_lang[rest] == l]; idx = idx[rng.permutation(len(idx))]
        shard_of[idx] = (np.arange(len(idx)) + int(rng.integers(0, a.k))) % a.k                  # round-robin with a random start so the extra pack of small languages is spread
    os.makedirs(a.out, exist_ok=True)
    plan = {"k": a.k, "packs_not_in_stage_a": int(len(rest)), "shards": []}
    for k in range(a.k):
        P = np.flatnonzero(shard_of == k); d = os.path.join(a.out, f"s{k + 1}")
        shutil.rmtree(d, ignore_errors=True); t0 = time.time()
        write_shard(a.packed, ds, d, P)
        _, _, dk, pl_k = load(d)
        steps = -(-len(P) // Pn)
        sched, _ = PD.unimax_schedule(pl_k, steps, Pn, min_epochs=1.0, max_epochs=2.0, weights=None, seed=a.seed + 10 + k)
        finish(os.path.join(d, "sched"), np.asarray(sched, dtype=np.int64), pl_k, ALL, dk, d, f"growth-1-stage-B-shard-{k + 1}", a.seed, Pn)
        tot = np.bincount(np.asarray(sched).reshape(-1), minlength=len(P)); assert tot.min() >= 1 and tot.max() <= 2
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(d) for f in fs)
        plan["shards"].append({"shard": k + 1, "packs": int(len(P)), "steps": int(steps), "bytes": int(size), "languages": int(len(np.unique(pl_k)))})
        open(os.path.join(d, "READY"), "w").write(time.strftime("%FT%TZ"))
        print(f"shard {k + 1}/{a.k}: {len(P)} packs, {steps} steps, {size / 1e9:.2f} GB, {time.time() - t0:.0f}s", flush=True)
    json.dump(plan, open(os.path.join(a.out, "plan.json"), "w"), indent=1)
    print("total steps", sum(s["steps"] for s in plan["shards"]))


if __name__ == "__main__":
    main()
