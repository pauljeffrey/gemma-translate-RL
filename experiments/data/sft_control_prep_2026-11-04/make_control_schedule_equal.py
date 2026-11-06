#!/usr/bin/env python
"""Schedule of the E-03 control with an EQUAL, FIXED number of TranslatePsy samples per language (every TranslatePsy pack is drawn exactly once).

Combined pack index space: [0, n_replay) replay packs (UniMax cut, cut_replay.py), [n_replay, n_all) TranslatePsy packs (one language per pack; equal samples per language, so the
number of packs differs between languages). Every step has 32 packs. TP packs are spread evenly over the run per language (position (j + 0.5) / n_l), so every prefix has the same
language mix; P = number of TP packs; steps with 17 (P > 16 x steps) or 15 (P < 16 x steps) TP packs are spread evenly; the replay half fills the rest: UniMax over the replay pool
(`unimax_schedule`, min 0, max 1 pass, no replay pack twice).

    python make_control_schedule_equal.py --packed-dir combined --n-replay 96600 --steps 6000 --out sched_ctl
"""
import argparse, hashlib, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.environ.get("GEMMA_MT_PKG"), os.path.join(HERE, "..", "..", "..", "gemma_mt")):
    if p and os.path.isdir(os.path.join(p, "gemma_mt")):
        sys.path.insert(0, p); break
from gemma_mt.data.packed_dataset import unimax_schedule, PackedDataset
from gemma_mt.languages import ALL_ISOS


def sha(a): return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()
def structure_sha(ds):
    h = hashlib.sha256()
    for name in ("pack_start", "iso_id", "direction"):
        x = np.ascontiguousarray(np.asarray(getattr(ds, name)), dtype=np.int64)
        h.update(f"{name}:{x.shape[0]}\n".encode()); h.update(x.tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-dir", required=True); ap.add_argument("--n-replay", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True); ap.add_argument("--per-step", type=int, default=32); ap.add_argument("--tp-base", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ds = PackedDataset(a.packed_dir, "train", pad=False) if os.path.isdir(os.path.join(a.packed_dir, "train")) else PackedDataset(os.path.dirname(a.packed_dir.rstrip("/")), "train", pad=False)
    lang = np.asarray(ds.pack_languages(), dtype=np.int64); s_hash = structure_sha(ds)
    n_rep = a.n_replay; rep_l = lang[:n_rep]; tp_l = lang[n_rep:]; P = len(tp_l)
    S, base = a.steps, a.tp_base
    k = P - base * S                                   # >0: k steps carry 17 TP packs; <0: |k| steps carry 15
    assert abs(k) <= S, k
    rng = np.random.default_rng([a.seed, 11])
    # 1. order of the TP packs: per language evenly spaced positions, then global sort
    pos = np.empty(P); idx_by_lang = {}
    for l in np.unique(tp_l):
        ii = np.where(tp_l == l)[0]; ii = ii[rng.permutation(len(ii))]
        pos[ii] = (np.arange(len(ii)) + 0.5) / len(ii) + rng.uniform(-0.25, 0.25, len(ii)) / len(ii); idx_by_lang[int(l)] = len(ii)
    order = np.argsort(pos, kind="stable")
    # 2. TP packs per step
    sizes = np.full(S, base, dtype=np.int64)
    if k != 0:
        sel = np.unique(np.floor((np.arange(abs(k)) + 0.5) * S / abs(k)).astype(np.int64)); assert len(sel) == abs(k)
        sizes[sel] += 1 if k > 0 else -1
    assert sizes.sum() == P
    # 3. replay: UniMax, base replay per step = per_step - base, then trim / extend per step
    rep_per = a.per_step - base
    rep, rep_stats = unimax_schedule(rep_l, S, rep_per, min_epochs=0.0, max_epochs=1.0, seed=a.seed)
    sched = np.zeros((S, a.per_step), dtype=np.int64); c = 0
    for t in range(S):
        s_tp = int(sizes[t]); r = a.per_step - s_tp
        assert r <= rep_per, "need more replay draws per step than the UniMax schedule gives; choose n so that P >= 16 x steps"
        sched[t, :s_tp] = order[c:c + s_tp] + n_rep; c += s_tp
        sched[t, s_tp:] = rep[t][:r]
        sched[t] = sched[t][rng.permutation(a.per_step)]
    assert c == P
    assert sched.min() >= 0 and sched.max() < len(lang)
    flat = sched.reshape(-1); is_tp = flat >= n_rep
    assert np.bincount(flat[is_tp] - n_rep, minlength=P).max() == 1 and is_tp.sum() == P, "every TP pack must be drawn exactly once"
    assert np.bincount(flat[~is_tp]).max() == 1, "replay pack repeated"
    np.save(f"{a.out}/schedule.npy", np.ascontiguousarray(sched))
    have = np.bincount(lang, minlength=len(ALL_ISOS)); d_tp = np.bincount(lang[flat[is_tp]], minlength=len(ALL_ISOS)); d_rep = np.bincount(lang[flat[~is_tp]], minlength=len(ALL_ISOS))
    per_lang = {ALL_ISOS[i]: {"packs": int(have[i]), "draws_replay": int(d_rep[i]), "draws_translatepsy": int(d_tp[i])} for i in range(len(ALL_ISOS)) if have[i]}
    sc = {"format": "sabiyarn-schedule-1", "schedule_sha256": sha(sched), "steps": int(S), "per_step": int(a.per_step), "n_packs": int(len(lang)), "pack_lang_sha256": sha(lang),
          "packed_structure_sha256": s_hash, "projected": False, "stage": "control-E03-continued-sft-equal-samples", "seed": a.seed, "n_replay_packs": int(n_rep), "n_tp_packs": int(P),
          "tp_packs_per_step_min_max": [int(sizes.min()), int(sizes.max())], "draws_per_language": per_lang, "max_draws_per_pack": 1}
    json.dump(sc, open(f"{a.out}/schedule.json", "w"), indent=1)
    print(f"steps {S} x {a.per_step}; TP packs {P} (steps with {base + 1 if k > 0 else base - 1} TP packs: {abs(k)}); replay draws {int((~is_tp).sum())}; sha {sc['schedule_sha256'][:16]}")
    print("TP draws per language:", {ALL_ISOS[i]: int(d_tp[i]) for i in range(len(ALL_ISOS)) if d_tp[i]})


if __name__ == "__main__":
    main()
