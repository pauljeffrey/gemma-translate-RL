#!/usr/bin/env python3
"""Final schedule of the A0 continuation over translation + instruction packs.

The first --prefix-steps rows are EXACTLY the stitched translation schedule's (already trained by A0, sha256 pinned); the rest is the translation tail interleaved with the instruction
stream by the keys (k + u) / draws of each stream, so the mixture stays constant over the run. Instruction pack indices are offset by the number of translation train packs (the order of
`extend_packed.py --base packed_full --add packed_instr`). Writes schedule.npy + the sidecar schedule.json the trainer checks (same fields as sampler.py stitch).

    python finalize_schedule.py --packed-all work/packed_all --trans work/sched_full/A0F --instr work/sched_instr --n-trans-packs N --prefix-steps 5625 --out work/sched_all/A0F
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "chain"))
import sampler as S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--packed-all", required=True); ap.add_argument("--trans", required=True); ap.add_argument("--instr", required=True)
    ap.add_argument("--n-trans-packs", type=int, required=True); ap.add_argument("--prefix-steps", type=int, default=5625); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--max-epochs", type=int, default=4)
    a = ap.parse_args()
    PD, ALL_ISOS = S._pd()
    ds = PD.PackedDataset(a.packed_all, None, pad=False) if os.path.exists(os.path.join(a.packed_all, "tokens.u32.bin")) else PD.PackedDataset(os.path.dirname(a.packed_all), os.path.basename(a.packed_all), pad=False)
    pack_lang = np.asarray(ds.pack_languages(), dtype=np.int64); n_packs = len(pack_lang)
    T = np.ascontiguousarray(np.load(a.trans + "/schedule.npy"), dtype=np.int64); per_step = T.shape[1]; P = a.prefix_steps
    prefix, tail = T[:P], T[P:].reshape(-1)
    I = np.load(a.instr + "/schedule.npy").reshape(-1).astype(np.int64) + a.n_trans_packs
    assert int(T.max()) < a.n_trans_packs and int(I.min()) >= a.n_trans_packs and int(I.max()) < n_packs, "pack index ranges do not fit the extended pack set"
    rng = np.random.default_rng([a.seed, 7919])
    keys = np.concatenate([(np.arange(len(tail)) + rng.random()) / len(tail), (np.arange(len(I)) + rng.random()) / len(I)])
    mixed = np.concatenate([tail, I])[np.argsort(keys, kind="stable")]
    steps_tail = len(mixed) // per_step; dropped = len(mixed) - steps_tail * per_step
    sched = np.concatenate([prefix, mixed[: steps_tail * per_step].reshape(steps_tail, per_step)]).astype(np.int64)
    tot = np.bincount(sched.reshape(-1), minlength=n_packs)
    # the dropped tail draws can leave a pack unseen only if it had a single draw; report instead of hiding it
    unseen = int((tot == 0).sum()); over = int((tot > a.max_epochs).sum())
    assert over == 0, f"{over} packs drawn more than {a.max_epochs} times"
    os.makedirs(a.out, exist_ok=True)
    np.save(a.out + "/schedule.npy", sched); np.save(a.out + "/pack_lang.npy", pack_lang)
    tm = json.load(open(a.trans + "/meta.json")) if os.path.exists(a.trans + "/meta.json") else {}
    stitched = tm.get("stitched_from")
    sc = S.write_sidecar(a.out, sched, pack_lang, ALL_ISOS, stage="B-stitched+instr", seed=a.seed, source=a.packed_all, projected=False,
                         packed_structure_sha256=S.packed_structure_sha256(ds), stitched_from=stitched, n_trans_packs=a.n_trans_packs, n_instr_draws=int(len(I)), dropped_tail_draws=int(dropped))
    json.dump({"steps": int(sched.shape[0]), "per_step": per_step, "n_packs": n_packs, "unseen_packs": unseen, "dropped_tail_draws": int(dropped), "schedule_sha256": sc["schedule_sha256"],
               "translation_draws": int(len(tail)) + P * per_step, "instruction_draws": int(len(I))}, open(a.out + "/meta.json", "w"), indent=1)
    print(f"final schedule {sched.shape[0]} steps x {per_step}; prefix {P} unchanged; translation tail {len(tail)} + instruction {len(I)} draws; unseen packs {unseen}; dropped tail draws {dropped}")
    assert unseen == 0, "a pack is never drawn"


if __name__ == "__main__":
    main()
