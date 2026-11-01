#!/usr/bin/env python3
"""Add the tail-language replay packs to a stage-B shard (growth 1) and rebuild its schedule.

    python add_replay_to_shard.py --shard run/shards/s2 --replay run/replay --frac 0.5 --seed 2 [--sha run/shards/s2.sha256]

The replay packs are appended to the shard IN PLACE (chain/extend_packed.py --inplace: shard pack indices unchanged, replay pack j becomes pack n_shard + j). The new schedule is the shard's old
one (every shard pack exactly once, same order) interleaved with draws of a random --frac of the replay packs (the keys (k + u) / draws of the UniMax trick keep the mixture constant over the run);
it is padded with a few more replay draws so that it fills whole steps of 32. The sidecar is rewritten for the extended pack structure and the shard's checksum file is regenerated.
Idempotent: a shard that already has sched/REPLAY_ADDED is left alone.
"""
import argparse, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "chain"))
import sampler as S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", required=True); ap.add_argument("--replay", required=True); ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--sha"); ap.add_argument("--per-step", type=int, default=32)
    a = ap.parse_args(); P = a.per_step
    if os.path.exists(os.path.join(a.shard, "sched", "REPLAY_ADDED")):
        print("replay already added to", a.shard); return
    n_shard = len(np.load(os.path.join(a.shard, "train", "pack_start.npy"))) - 1
    n_rep = len(np.load(os.path.join(a.replay, "train", "pack_start.npy"))) - 1
    old = np.load(os.path.join(a.shard, "sched", "schedule.npy")).astype(np.int64)
    D = old.reshape(-1)
    ext = os.path.join(HERE, "..", "chain", "extend_packed.py")
    subprocess.run([sys.executable, ext, "--base", a.shard, "--add", a.replay, "--out", a.shard, "--inplace"], check=True)
    rng = np.random.default_rng([a.seed, 424242])
    k = int(round(a.frac * n_rep)); pad = (-(len(D) + k)) % P
    chosen = rng.choice(n_rep, size=min(n_rep, k), replace=False)
    if pad: chosen = np.concatenate([chosen, rng.choice(n_rep, size=pad, replace=True)])
    R = n_shard + chosen
    keys = np.concatenate([(np.arange(len(D)) + rng.random()) / len(D), (np.arange(len(R)) + rng.random()) / len(R)])
    mixed = np.concatenate([D, R])[np.argsort(keys, kind="stable")]
    assert len(mixed) % P == 0
    sched = np.ascontiguousarray(mixed.reshape(-1, P), dtype=np.int64)
    PD, ALL = S._pd()
    ds = PD.PackedDataset(a.shard, "train", pad=False); pack_lang = np.asarray(ds.pack_languages(), dtype=np.int64)
    assert len(pack_lang) == n_shard + n_rep and int(sched.max()) < len(pack_lang)
    tot = np.bincount(sched.reshape(-1), minlength=len(pack_lang))
    assert tot[:n_shard].min() >= 1 and tot[:n_shard].max() <= 2, "a shard pack lost or repeated"
    out = os.path.join(a.shard, "sched"); np.save(out + "/schedule.npy", sched); np.save(out + "/pack_lang.npy", pack_lang)
    S.write_sidecar(out, sched, pack_lang, ALL, stage="growth-1-stage-B-shard+replay", seed=a.seed, source=a.shard, projected=False, packed_structure_sha256=S.packed_structure_sha256(ds))
    json.dump({"steps": int(sched.shape[0]), "per_step": P, "n_shard_packs": n_shard, "n_replay_packs": n_rep, "replay_draws": int(len(R)), "replay_frac": a.frac},
              open(out + "/meta.json", "w"), indent=1)
    open(out + "/REPLAY_ADDED", "w").write("ok")
    if a.sha:
        subprocess.run(f"cd {a.shard} && find . -type f ! -name READY ! -name READY_LOCAL | sort | xargs sha256sum > {os.path.abspath(a.sha)}", shell=True, check=True)
    print(f"{a.shard}: {n_shard} shard packs + {len(R)} replay draws ({len(R) / len(mixed) * 100:.1f}% of draws) -> {sched.shape[0]} steps")


if __name__ == "__main__":
    main()
