#!/usr/bin/env python3
"""Growth-1 schedule: ONE epoch over every pack of the extended pack set (old A0 packs, new translation packs, instruction packs): a pack is drawn exactly once (owner 2026-11-01: low-resource
languages had 4 epochs in A0, so growth stages do not repeat them; new data is seen once). Packs are mixed by the UniMax trick of gemma_mt.data.packed_dataset.unimax_schedule (every language's
draws spread over the whole run), whole steps of 32 packs; the < 32 packs that would not fill the last step are drawn a second time (the smallest languages first), never more.

    python make_growth_schedule.py --packed work/packed_g --out work/sched_g/ALL --seed 1234
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "chain"))
import sampler as S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--packed", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--per-step", type=int, default=32)
    a = ap.parse_args()
    PD, ALL_ISOS = S._pd()
    ds = PD.PackedDataset(a.packed, "train", pad=False)
    pack_lang = np.asarray(ds.pack_languages(), dtype=np.int64); n = len(pack_lang)
    steps = -(-n // a.per_step)                                                   # ceil: every pack at least once
    sched, stats = PD.unimax_schedule(pack_lang, steps, a.per_step, min_epochs=1.0, max_epochs=2.0, weights=None, seed=a.seed)
    sched = np.ascontiguousarray(sched, dtype=np.int64)
    tot = np.bincount(sched.reshape(-1), minlength=n)
    assert tot.min() >= 1 and tot.max() <= 2, (int(tot.min()), int(tot.max()))
    extra = int((tot == 2).sum()); assert extra < a.per_step + 1, f"{extra} packs drawn twice"
    os.makedirs(a.out, exist_ok=True)
    np.save(a.out + "/schedule.npy", sched); np.save(a.out + "/pack_lang.npy", pack_lang)
    sc = S.write_sidecar(a.out, sched, pack_lang, ALL_ISOS, stage="growth-1-one-epoch", seed=a.seed, source=a.packed, projected=False, packed_structure_sha256=S.packed_structure_sha256(ds))
    json.dump({"steps": int(sched.shape[0]), "per_step": a.per_step, "n_packs": n, "packs_drawn_twice": extra, "schedule_sha256": sc["schedule_sha256"]}, open(a.out + "/meta.json", "w"), indent=1)
    print(f"growth schedule: {sched.shape[0]} steps x {a.per_step} over {n} packs, {extra} packs drawn twice, tokens ~{sched.size * 4096 * 0.9994 / 1e9:.2f}B")


if __name__ == "__main__":
    main()
