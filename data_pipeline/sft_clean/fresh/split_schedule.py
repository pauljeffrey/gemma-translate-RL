#!/usr/bin/env python3
"""Cut ONE schedule (growth 1: every pack of packed_g drawn once) into the two phases of the staged unfreezing, each with a sidecar the trainer accepts.

    python split_schedule.py --sched work/sched_g/ALL --out work/sched_g --stage-a-steps 1500 [--packed work/packed_g]

Writes <out>/G1A/ (rows [0, A)) and <out>/G1B/ (rows [A, steps)): schedule.npy, pack_lang.npy and schedule.json (sampler.write_sidecar: sha256, steps, per_step, n_packs, language of every
pack, pack-structure hash inherited from the parent, plus `split_from`). Stage B follows stage A directly: A + B == the parent, row for row. The pack language array comes from
<sched>/pack_lang.npy, or from --packed when the parent has none.
"""
import argparse, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "chain"))
import sampler as S


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sched", required=True, help="directory with schedule.npy (+ schedule.json)"); ap.add_argument("--out", required=True)
    ap.add_argument("--stage-a-steps", type=int, default=1500); ap.add_argument("--packed", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args(argv)
    PD, ALL_ISOS = S._pd()
    sched = np.ascontiguousarray(np.load(os.path.join(a.sched, "schedule.npy")), dtype=np.int64)
    steps, per_step = sched.shape
    if not 0 < a.stage_a_steps < steps:
        raise SystemExit(f"--stage-a-steps must be inside (0, {steps})")
    parent = json.load(open(os.path.join(a.sched, "schedule.json"))) if os.path.exists(os.path.join(a.sched, "schedule.json")) else {}
    if os.path.exists(os.path.join(a.sched, "pack_lang.npy")):
        pack_lang = np.load(os.path.join(a.sched, "pack_lang.npy")).astype(np.int64)
    elif a.packed:
        d = a.packed
        ds = PD.PackedDataset(d, "train", pad=False) if os.path.exists(os.path.join(d, "train", "tokens.u32.bin")) else PD.PackedDataset(d, None, pad=False)
        pack_lang = np.asarray(ds.pack_languages(), dtype=np.int64)
    else:
        raise SystemExit("no pack_lang.npy next to the schedule: pass --packed")
    n_packs = len(pack_lang)
    if int(sched.max()) >= n_packs or int(sched.min()) < 0:
        raise SystemExit("schedule indexes packs outside the pack set")
    if parent.get("n_packs") not in (None, n_packs):
        raise SystemExit(f"parent sidecar says n_packs={parent['n_packs']} but pack_lang has {n_packs}")
    psha = hashlib.sha256(sched.tobytes()).hexdigest()
    if parent.get("schedule_sha256") not in (None, psha):
        raise SystemExit("parent sidecar sha256 does not match its schedule.npy")
    for name, (lo, hi) in (("G1A", (0, a.stage_a_steps)), ("G1B", (a.stage_a_steps, steps))):
        d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
        part = np.ascontiguousarray(sched[lo:hi])
        np.save(os.path.join(d, "schedule.npy"), part); np.save(os.path.join(d, "pack_lang.npy"), pack_lang)
        extra = dict(stage=name, seed=a.seed, source=parent.get("source"), projected=False, packed_structure_sha256=parent.get("packed_structure_sha256"),
                     split_from={"schedule_sha256": psha, "rows": [int(lo), int(hi)], "parent_steps": int(steps)})
        S.write_sidecar(d, part, pack_lang, ALL_ISOS, **extra)
        print(f"{name}: rows [{lo}, {hi}) = {part.shape[0]} steps x {per_step}, sha256 {hashlib.sha256(part.tobytes()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
