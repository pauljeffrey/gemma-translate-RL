#!/usr/bin/env python3
"""Growth-1 schedules over the append-only pack set  packed_g = [A0 old packs (retagged) | instruction packs | new translation packs (retagged)].

    python make_stage_schedules.py A --packed work/packed_g0 --n-old N_OLD --out work/sched_g/A [--steps 1500]
    python make_stage_schedules.py B --packed work/packed_g  --stage-a work/sched_g/A --out work/sched_g/B

Stage A (new layers + 125 tag rows trainable, everything else frozen): half of the draws are INSTRUCTION packs no model has seen, half are a replay of the old translation packs, both balanced over
languages with the UniMax allocation (every language gets draws in proportion to min(n, lambda); at most one pass over its packs), so each of the 125 tag rows is trained.
Stage B (everything trainable): ONE epoch of every pack NOT drawn in stage A (low-resource languages had 4 epochs in A0; new data is seen once); the < 32 packs that do not fill the last step are
drawn twice. The sidecar of each schedule pins the pack structure of the dataset it is drawn on (the append keeps indices of the earlier part).
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "chain"))
import sampler as S


def load(packed):
    PD, ALL = S._pd()
    ds = PD.PackedDataset(packed, "train", pad=False)             # dataset root + split, as the trainer opens it
    return PD, ALL, ds, np.asarray(ds.pack_languages(), dtype=np.int64)


def sub_schedule(PD, pack_lang, idx, steps, per_step, min_e, max_e, seed):
    """unimax_schedule over the packs `idx` only; returns the draws as indices into the full set"""
    sched, _ = PD.unimax_schedule(pack_lang[idx], steps, per_step, min_epochs=min_e, max_epochs=max_e, weights=None, seed=seed)
    return idx[np.asarray(sched, dtype=np.int64)]


def finish(out, sched, pack_lang, ALL, ds, packed, stage, seed, per_step):
    sched = np.ascontiguousarray(sched, dtype=np.int64)
    os.makedirs(out, exist_ok=True)
    np.save(out + "/schedule.npy", sched); np.save(out + "/pack_lang.npy", pack_lang)
    sc = S.write_sidecar(out, sched, pack_lang, ALL, stage=stage, seed=seed, source=packed, projected=False, packed_structure_sha256=S.packed_structure_sha256(ds))
    tot = np.bincount(sched.reshape(-1), minlength=len(pack_lang))
    json.dump({"steps": int(sched.shape[0]), "per_step": per_step, "n_packs": int(len(pack_lang)), "packs_drawn": int((tot > 0).sum()), "max_draws_per_pack": int(tot.max()),
               "schedule_sha256": sc["schedule_sha256"]}, open(out + "/meta.json", "w"), indent=1)
    print(f"{stage}: {sched.shape[0]} steps x {per_step}, {int((tot > 0).sum())} distinct packs, max draws per pack {int(tot.max())}, ~{sched.size * 4096 * 0.9994 / 1e9:.2f}B tokens")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["A", "B"]); ap.add_argument("--packed", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-old", type=int); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--stage-a"); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--per-step", type=int, default=32)
    a = ap.parse_args(); P = a.per_step
    PD, ALL, ds, pack_lang = load(a.packed); n = len(pack_lang)
    if a.stage == "A":
        assert a.n_old and a.n_old < n
        half = a.steps // 2
        old = np.arange(0, a.n_old); ins = np.arange(a.n_old, n)
        # the instruction packs sit after the old packs; in packed_g0 they are the last ones (nothing was appended yet)
        s_old = sub_schedule(PD, pack_lang, old, half, P, 0.0, 1.0, a.seed)
        s_ins = sub_schedule(PD, pack_lang, ins, a.steps - half, P, 0.0, 1.0, a.seed + 1)
        T, I = s_old.reshape(-1), s_ins.reshape(-1)
        rng = np.random.default_rng([a.seed, 7919])
        keys = np.concatenate([(np.arange(len(T)) + rng.random()) / len(T), (np.arange(len(I)) + rng.random()) / len(I)])
        mixed = np.concatenate([T, I])[np.argsort(keys, kind="stable")].reshape(a.steps, P)
        finish(a.out, mixed, pack_lang, ALL, ds, a.packed, "growth-1-stage-A", a.seed, P)
    else:
        A = np.load(os.path.join(a.stage_a, "schedule.npy")).reshape(-1)
        drawn = np.zeros(n, bool); drawn[A] = True
        idx = np.flatnonzero(~drawn); steps = -(-len(idx) // P)
        sched = sub_schedule(PD, pack_lang, idx, steps, P, 1.0, 2.0, a.seed + 2)
        tot = np.bincount(sched.reshape(-1), minlength=n)
        assert tot[idx].min() >= 1 and tot.max() <= 2 and tot[drawn].sum() == 0
        print(f"stage A drew {int(drawn.sum())} distinct packs; stage B covers the other {len(idx)}")
        finish(a.out, sched, pack_lang, ALL, ds, a.packed, "growth-1-stage-B", a.seed, P)


if __name__ == "__main__":
    main()
