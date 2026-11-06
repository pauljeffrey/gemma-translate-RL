#!/usr/bin/env python
"""Schedule file for the continued-SFT control (E-03): every optimizer step = 32 packs = 16 REPLAY packs + 16 TRANSLATEPSY packs (tp-share 0.5), in the format the trainer reads
(`schedule_file`: int64 [steps, 32] + sidecar schedule.json with sha256 / n_packs / pack-language and pack-structure hashes; gemma_mt.train_sft.load_schedule).

Pack index space of the combined packed dataset (built with extend_packed.py, see build_control_data.sh):
    [0, n_s6)                       stage-B shard 6 packs                         (bucket clean_rerun_2026-11-01/shards/s6, 154,421 packs)
    [n_s6, n_s6 + n_tail)           tail-language replay packs cut from g0         (make_replay.py, 30,627 packs, appended in place to s6 in the original run)
    [n_s6 + n_tail, n_all)          TranslatePsy en->xx packs                      (prepare_sft --pack-by-language on tp_clean)

Replay half: UniMax over the 111 languages of the replay pool (min 0, max 1 pass, so no replay pack is drawn twice and the tail languages are replayed in full);
TP half: UniMax over the 19 TranslatePsy languages (min 0, max 1 pass: no TranslatePsy pack is ever repeated). Both use gemma_mt.data.packed_dataset.unimax_schedule (the
function the 385M SFT used), which spreads every language's draws evenly over the run, so ANY PREFIX of the schedule has the same language mix: the stage-A-matched arm is the
step-<N> checkpoint of the same run (constant learning rate).

    # on the box (real dataset)
    python make_control_schedule.py --packed-dir combined/train --n-replay 185048 --steps 6000 --out sched_ctl
    # dry run on the laptop (no token files): arrays only
    python make_control_schedule.py --replay-pack-lang replay_pack_lang.npy --tp-pack-lang tp_pack_lang.npy --steps 6000 --out dry
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
        a = np.ascontiguousarray(np.asarray(getattr(ds, name)), dtype=np.int64)
        h.update(f"{name}:{a.shape[0]}\n".encode()); h.update(a.tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-dir"); ap.add_argument("--n-replay", type=int, help="number of replay packs at the START of the combined dataset (rest = TranslatePsy)")
    ap.add_argument("--replay-pack-lang"); ap.add_argument("--tp-pack-lang")
    ap.add_argument("--steps", type=int, required=True); ap.add_argument("--per-step", type=int, default=32); ap.add_argument("--tp-share", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.packed_dir:
        ds = PackedDataset(a.packed_dir, "train") if os.path.isdir(os.path.join(a.packed_dir, "train")) else PackedDataset(os.path.dirname(a.packed_dir.rstrip("/")), "train")
        lang = np.asarray(ds.pack_languages(), dtype=np.int64); s_hash = structure_sha(ds)
        rep_l, tp_l = lang[:a.n_replay], lang[a.n_replay:]
    else:
        rep_l = np.load(a.replay_pack_lang).astype(np.int64); tp_l = np.load(a.tp_pack_lang).astype(np.int64)
        lang = np.concatenate([rep_l, tp_l]); s_hash = None
    n_rep = len(rep_l); k_tp = int(round(a.per_step * a.tp_share)); k_rep = a.per_step - k_tp
    rep, rep_stats = unimax_schedule(rep_l, a.steps, k_rep, min_epochs=0.0, max_epochs=1.0, seed=a.seed)
    tp, tp_stats = unimax_schedule(tp_l, a.steps, k_tp, min_epochs=0.0, max_epochs=1.0, seed=a.seed + 1)
    tp = tp + n_rep                                                    # global index
    rng = np.random.default_rng([a.seed, 7])
    sched = np.concatenate([rep, tp], axis=1)
    for i in range(sched.shape[0]): sched[i] = sched[i][rng.permutation(a.per_step)]     # mix replay and TP inside every step (micro-batches see both)
    sched = np.ascontiguousarray(sched, dtype=np.int64)
    assert sched.min() >= 0 and sched.max() < len(lang)
    np.save(f"{a.out}/schedule.npy", sched)
    flat = sched.reshape(-1); drawn = np.bincount(lang[flat], minlength=len(ALL_ISOS)); have = np.bincount(lang, minlength=len(ALL_ISOS))
    is_tp = flat >= n_rep; drawn_tp = np.bincount(lang[flat[is_tp]], minlength=len(ALL_ISOS)); drawn_rep = np.bincount(lang[flat[~is_tp]], minlength=len(ALL_ISOS))
    per_lang = {ALL_ISOS[i]: {"packs": int(have[i]), "draws": int(drawn[i]), "draws_replay": int(drawn_rep[i]), "draws_translatepsy": int(drawn_tp[i])}
                for i in range(len(ALL_ISOS)) if have[i] or drawn[i]}
    sc = {"format": "sabiyarn-schedule-1", "schedule_sha256": sha(sched), "steps": int(sched.shape[0]), "per_step": int(sched.shape[1]), "n_packs": int(len(lang)),
          "pack_lang_sha256": sha(lang), "packed_structure_sha256": s_hash, "projected": False, "stage": "control-E03-continued-sft", "seed": a.seed,
          "tp_share": a.tp_share, "n_replay_packs": int(n_rep), "n_tp_packs": int(len(tp_l)), "packs_drawn": int(np.unique(flat).size), "max_draws_per_pack": int(np.bincount(flat).max()),
          "draws_per_language": per_lang, "note": "dry run: packed_structure_sha256 not computed (no token files)" if s_hash is None else ""}
    json.dump(sc, open(f"{a.out}/schedule.json", "w"), indent=1)
    print(f"steps {sched.shape[0]} x {sched.shape[1]}; replay draws {int((~is_tp).sum())}, TP draws {int(is_tp.sum())}; unique packs {sc['packs_drawn']}; max draws/pack {sc['max_draws_per_pack']}; sha {sc['schedule_sha256'][:16]}")
    print("TP draws per language:", {ALL_ISOS[i]: int(drawn_tp[i]) for i in range(len(ALL_ISOS)) if drawn_tp[i]})


if __name__ == "__main__":
    main()
