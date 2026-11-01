#!/usr/bin/env python3
"""STEP (e): stage samplers and the arm streams of the pre-registration. Uses the OLD functions unchanged (gemma_mt.data.packed_dataset.balanced_schedule / unimax_schedule).

    # from a packed dataset (GPU session) or from a projected packs-per-language table (planning)
    python sampler.py plan --stage A --packed-dir packed/train        --out sched/A0 [--seed 1234] [--steps N]
    python sampler.py plan --stage A --packs-csv packs_per_language.csv --out sched/A0_projected
    python sampler.py plan --stage B --packed-dir packed/train --budget-tokens 3000000000 --out sched/B0
    python sampler.py arms --schedule sched/A0 --out sched            # S1 = the draws of A0 whose pack belongs to a specialist language, same order; S2 = the same file

Stage A (old SFT run, plan section 3): p(language) ~ packs**0.5 floored at 0.2%, a language retired after 4 passes, 32 packs per step, number of steps = one natural pass over all packs
(16,827 = 538,480 // 32 in the old run). Stage B (sft_unimax.yaml): UniMax, budget 3 B tokens = 22,888 steps x 131,072 tokens, min 1 / max 2 epochs, 20 priority languages x4.
Schedules are pure functions of (pack_lang, steps, seed): the same seeded sampler log feeds A0 and S1.
Outputs: schedule.npy (int64 [steps, per_step] pack indices), draws_per_language.csv (packs, draws, share, epochs, retired), meta.json (arguments, sha256 of the schedule),
schedule.json (SIDECAR, added 2026-10-28: sha256, steps, per_step, n_packs, sha256 of the language of every pack, sha256 of the pack structure of the packed dataset; the trainer
`gemma_mt.train_sft` with `schedule_file: .../schedule.npy` refuses a schedule whose sidecar disagrees with the packed dataset it is given).
`select --languages amh,tir --name G0` (G0 arm) and `subset --frac 0.20 --stratify language` (A2 arms) derive further schedules from the A0 schedule in the same way (see their docstrings).
`arms` writes S1/schedule.npy = the A0 draws of specialist-language packs in A0's order, regrouped into steps of 32 (a final incomplete step is dropped and reported); every specialist
language therefore gets exactly the passes it has in A0. S2 uses the same pack-membership list (pack.py replay re-tokenises each pack with the extended tokenizer).
"""
import argparse, csv, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import STAGE_A, STAGE_B, SPECIALIST, SEQ_LEN


def _pd():
    for p in (os.environ.get("GEMMA_MT_PKG"), "/workspace/work/gemma_mt_pkg", "/path/to/repo/gemma_mt"):
        if p and os.path.isdir(os.path.join(p, "gemma_mt")):
            sys.path.insert(0, p); break
    from gemma_mt.data import packed_dataset as PD
    from gemma_mt.languages import ALL_ISOS
    return PD, ALL_ISOS


def packed_structure_sha256(ds):
    """Same value as gemma_mt.train_sft.packed_structure_sha256 (a test pins the equality): which samples form which pack, independent of the token ids."""
    h = hashlib.sha256()
    for name in ("pack_start", "iso_id", "direction"):
        a = np.ascontiguousarray(np.asarray(getattr(ds, name)), dtype=np.int64)
        h.update(f"{name}:{a.shape[0]}\n".encode()); h.update(a.tobytes())
    return h.hexdigest()


def pack_lang_sha256(pack_lang):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(pack_lang), dtype=np.int64).tobytes()).hexdigest()


def write_sidecar(out_dir, sched, pack_lang, ALL_ISOS, **extra):
    """schedule.json next to schedule.npy: what the trainer checks before it spends GPU time (see gemma_mt.train_sft.load_schedule)."""
    sched = np.ascontiguousarray(sched, dtype=np.int64)
    draws = np.bincount(np.asarray(pack_lang)[sched.reshape(-1)], minlength=len(ALL_ISOS)); have = np.bincount(np.asarray(pack_lang), minlength=len(ALL_ISOS))
    sc = {"format": "sabiyarn-schedule-1", "schedule_sha256": hashlib.sha256(sched.tobytes()).hexdigest(), "steps": int(sched.shape[0]), "per_step": int(sched.shape[1]),
          "n_packs": int(len(pack_lang)), "pack_lang_sha256": pack_lang_sha256(pack_lang), "packed_structure_sha256": None, "projected": False,
          "draws_per_language": {ALL_ISOS[i]: {"packs": int(have[i]), "draws": int(draws[i])} for i in range(len(ALL_ISOS)) if have[i] or draws[i]}}
    sc.update(extra)
    json.dump(sc, open(os.path.join(out_dir, "schedule.json"), "w"), indent=1)
    return sc


def pack_lang_from_csv(path, ALL_ISOS):
    rows = list(csv.DictReader(open(path)))
    pl = []
    for r in rows:
        pl += [ALL_ISOS.index(r["iso"])] * int(r["packs"])
    return np.array(pl, np.int64)


def plan(a):
    PD, ALL_ISOS = _pd()
    ds = None
    if a.packed_dir:
        ds = PD.PackedDataset(a.packed_dir, None, pad=False) if os.path.exists(os.path.join(a.packed_dir, "tokens.u32.bin")) else PD.PackedDataset(os.path.dirname(a.packed_dir), os.path.basename(a.packed_dir), pad=False)
        pack_lang = ds.pack_languages()
    else:
        pack_lang = pack_lang_from_csv(a.packs_csv, ALL_ISOS)
    n_packs = len(pack_lang)
    per_step = a.per_step or (STAGE_A["per_step"] if a.stage == "A" else STAGE_B["per_step"])
    if a.stage == "A":
        steps = a.steps or n_packs // per_step
        sched, stats = PD.balanced_schedule(pack_lang, steps, per_step, alpha=a.alpha, min_share=STAGE_A["min_share"], max_epochs=STAGE_A["max_epochs"], seed=a.seed)
    else:
        steps = a.steps or (a.budget_draws // per_step if getattr(a, "budget_draws", 0) else int(a.budget_tokens // STAGE_B["step_tokens"]))
        w = {} if getattr(a, "unweighted", False) else {ALL_ISOS.index(i): STAGE_B["priority_weight"] for i in STAGE_B["priority"] if i in ALL_ISOS}
        min_e = a.min_epochs if getattr(a, "min_epochs", None) is not None else STAGE_B["min_epochs"]; max_e = a.max_epochs if getattr(a, "max_epochs", None) is not None else STAGE_B["max_epochs"]
        seen = None
        if getattr(a, "seen_from", None):                                          # packs an earlier schedule (A0) already used come LAST in each language's first pass (the old run's unimax_unseen_first)
            seen = np.zeros(n_packs, dtype=bool); seen[np.load(os.path.join(a.seen_from, "schedule.npy")).reshape(-1)] = True
        sched, stats = PD.unimax_schedule(pack_lang, steps, per_step, min_epochs=min_e, max_epochs=max_e, weights=w, seed=a.seed, seen=seen)
    os.makedirs(a.out, exist_ok=True)
    np.save(a.out + "/schedule.npy", sched)
    np.save(a.out + "/pack_lang.npy", pack_lang)
    with open(a.out + "/draws_per_language.csv", "w", newline="") as f:
        w_ = csv.writer(f); w_.writerow(["iso", "packs", "draws", "share", "effective_epochs"])
        for lid, s in sorted(stats.items(), key=lambda kv: -kv[1]["draws"]):
            w_.writerow([ALL_ISOS[lid], s["packs"], s["draws"], round(s["share"], 6), round(s["epochs"], 3)])
    meta = {"stage": a.stage, "steps": int(steps), "per_step": per_step, "n_packs": int(n_packs), "seed": a.seed, "alpha": a.alpha if a.stage == "A" else None,
            "schedule_sha256": hashlib.sha256(sched.tobytes()).hexdigest(), "source": a.packed_dir or a.packs_csv}
    json.dump(meta, open(a.out + "/meta.json", "w"), indent=1)
    write_sidecar(a.out, sched, pack_lang, ALL_ISOS, stage=a.stage, seed=a.seed, source=a.packed_dir or a.packs_csv, projected=ds is None,
                  packed_structure_sha256=packed_structure_sha256(ds) if ds is not None else None)
    print(meta)


def waterfill(c, r, rng):
    """extra draws x_p (sum r) for packs with current counts c, filling the least-seen packs first (unseen packs first); ties random"""
    c = np.asarray(c, dtype=np.int64)
    if r <= 0:
        return np.zeros_like(c)
    lo, hi = int(c.min()), int(c.max()) + int(r) + 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if int(np.maximum(0, mid - c).sum()) <= r:
            lo = mid
        else:
            hi = mid
    x = np.maximum(0, lo - c); rem = int(r) - int(x.sum())
    if rem > 0:
        cand = np.flatnonzero(c + x == lo); x[rng.choice(cand, size=rem, replace=False)] += 1
    return x


def stitch(a):
    """CONTINUATION schedule: the first --prefix-steps rows are EXACTLY the old schedule's (already trained); the rest is rebuilt on the (possibly extended) pack set so that the whole run
    follows the UniMax allocation for --budget-draws (floor/cap epochs per language), every pack is drawn at least once overall and the least-seen packs come first. Total steps stay budget // per_step."""
    PD, ALL_ISOS = _pd()
    ds = PD.PackedDataset(a.packed_dir, None, pad=False) if os.path.exists(os.path.join(a.packed_dir, "tokens.u32.bin")) else PD.PackedDataset(os.path.dirname(a.packed_dir), os.path.basename(a.packed_dir), pad=False)
    pack_lang = np.asarray(ds.pack_languages(), dtype=np.int64); n_packs = len(pack_lang)
    old = np.ascontiguousarray(np.load(a.old_schedule + "/schedule.npy"), dtype=np.int64); per_step = old.shape[1]; P = a.prefix_steps
    assert 0 < P <= old.shape[0], "prefix longer than the old schedule"
    first = old[:P]; assert int(first.max()) < n_packs
    total_steps = a.budget_draws // per_step; R = (total_steps - P) * per_step
    counts = np.bincount(pack_lang, minlength=len(ALL_ISOS)); live = np.flatnonzero(counts > 0)
    alloc = np.asarray(PD.unimax_allocation(counts[live].astype(np.int64), int(total_steps * per_step), min_epochs=a.min_epochs, max_epochs=a.max_epochs)).astype(np.int64)
    c_all = np.bincount(first.reshape(-1), minlength=n_packs)
    rng = np.random.default_rng(a.seed)
    idx = {int(l): np.flatnonzero(pack_lang == l) for l in live}
    d = np.array([int(c_all[idx[int(l)]].sum()) for l in live]); unseen = np.array([int((c_all[idx[int(l)]] == 0).sum()) for l in live])
    r = np.maximum(alloc - d, unseen)                                    # never fewer extra draws than there are unseen packs (every pack is seen at least once)
    capx = a.max_epochs * counts[live] - (d + r)                         # room under the per-language cap
    diff = int(R - r.sum())
    while diff != 0:                                                     # make the continuation exactly R draws
        if diff > 0:
            cand = np.flatnonzero(capx > 0)
            k = rng.choice(cand, size=min(diff, len(cand)), replace=False); r[k] += 1; capx[k] -= 1; diff -= len(k)
        else:
            cand = np.flatnonzero(r > unseen)
            k = rng.choice(cand, size=min(-diff, len(cand)), replace=False); r[k] -= 1; capx[k] += 1; diff += len(k)
    draws = []
    for j, l in enumerate(live):
        ids = idx[int(l)]; x = waterfill(c_all[ids], int(r[j]), rng); draws.append(np.repeat(ids, x))
    cont = np.concatenate(draws); rng.shuffle(cont); cont = cont.reshape(-1, per_step)
    sched = np.concatenate([first, cont]).astype(np.int64); assert sched.shape[0] == total_steps
    tot = np.bincount(sched.reshape(-1), minlength=n_packs)
    assert tot.min() >= 1, "a pack is never drawn"; assert tot.max() <= int(np.ceil(a.max_epochs)), "a pack is drawn more often than the cap"
    ep = {int(l): tot[idx[int(l)]].sum() / len(idx[int(l)]) for l in live}
    os.makedirs(a.out, exist_ok=True)
    np.save(a.out + "/schedule.npy", sched); np.save(a.out + "/pack_lang.npy", pack_lang)
    with open(a.out + "/draws_per_language.csv", "w", newline="") as f:
        w_ = csv.writer(f); w_.writerow(["iso", "packs", "draws", "effective_epochs"])
        for l in sorted(live, key=lambda l: -len(idx[int(l)])):
            w_.writerow([ALL_ISOS[int(l)], len(idx[int(l)]), int(tot[idx[int(l)]].sum()), round(ep[int(l)], 3)])
    old_sha = hashlib.sha256(old.tobytes()).hexdigest()
    stitched = {"schedule_sha256": old_sha, "prefix_steps": int(P), "prefix_sha256": hashlib.sha256(np.ascontiguousarray(first).tobytes()).hexdigest()}
    meta = {"stage": "B-stitched", "steps": int(total_steps), "per_step": per_step, "n_packs": int(n_packs), "seed": a.seed, "schedule_sha256": hashlib.sha256(sched.tobytes()).hexdigest(),
            "source": a.packed_dir, "stitched_from": stitched, "min_epochs": a.min_epochs, "max_epochs": a.max_epochs}
    json.dump(meta, open(a.out + "/meta.json", "w"), indent=1)
    write_sidecar(a.out, sched, pack_lang, ALL_ISOS, stage="B-stitched", seed=a.seed, source=a.packed_dir, projected=False, packed_structure_sha256=packed_structure_sha256(ds), stitched_from=stitched)
    print(meta); print("effective epochs min/max:", round(min(ep.values()), 3), round(max(ep.values()), 3), "| continuation draws", int(R), "| per-pack total draws min/max", int(tot.min()), int(tot.max()))


def arms(a):
    PD, ALL_ISOS = _pd()
    sched = np.load(a.schedule + "/schedule.npy"); pack_lang = np.load(a.schedule + "/pack_lang.npy")
    per_step = sched.shape[1]
    langs = [x for x in a.languages.split(",") if x] if getattr(a, "languages", None) else SPECIALIST
    spec_ids = np.array([ALL_ISOS.index(i) for i in langs if i in ALL_ISOS])
    flat = sched.reshape(-1)
    m = np.isin(pack_lang[flat], spec_ids)
    s1 = flat[m]
    n_full = len(s1) // per_step
    dropped = len(s1) - n_full * per_step
    s1 = s1[:n_full * per_step].reshape(n_full, per_step)
    a0sc = {}
    for cand in ("schedule.json", "meta.json"):                                   # sidecar of A0: its pack-structure hash is inherited (S2's replay keeps the packs)
        if os.path.exists(os.path.join(a.schedule, cand)):
            a0sc = json.load(open(os.path.join(a.schedule, cand))); break
    a0_projected = bool(a0sc.get("projected")) or str(a0sc.get("source", "")).endswith(".csv")
    for arm in ("S1", "S2"):
        d = os.path.join(a.out, arm); os.makedirs(d, exist_ok=True)
        np.save(d + "/schedule.npy", s1); np.save(d + "/pack_lang.npy", pack_lang)
        np.save(d + "/a0_positions.npy", np.flatnonzero(m)[:n_full * per_step])
        write_sidecar(d, s1, pack_lang, ALL_ISOS, arm=arm, source=a0sc.get("source"), projected=a0_projected, packed_structure_sha256=a0sc.get("packed_structure_sha256"),
                      a0_schedule=os.path.abspath(a.schedule), a0_schedule_sha256=hashlib.sha256(np.ascontiguousarray(sched, dtype=np.int64).tobytes()).hexdigest(),
                      note="same pack indices as A0; S2 is trained on the replayed (re-tokenised) packs, whose pack structure is identical")
    json.dump({"a0_draws": int(len(flat)), "s1_draws": int(m.sum()), "s1_steps": int(n_full), "dropped_incomplete_step_draws": int(dropped), "specialist_languages_in_packs": sorted({ALL_ISOS[i] for i in np.unique(pack_lang[flat[m]])}),
               "note": "S2 = same pack list as S1; pack.py replay re-tokenises every pack with the extended tokenizer"}, open(a.out + "/arms.json", "w"), indent=1)
    # per-language passes must be identical in A0 and S1: check
    import collections
    ca = collections.Counter(pack_lang[flat[m]].tolist()); cs = collections.Counter(pack_lang[s1.reshape(-1)].tolist())
    for k in ca:
        assert cs[k] <= ca[k] and ca[k] - cs[k] <= per_step, (k, ca[k], cs[k])
    print({"s1_steps": n_full, "s1_draws": int(m.sum()), "dropped": int(dropped)})


# ---------------------------------------------------------------------------------------------------- derived schedules (2026-10-29, Amendment 4)
def _a0_context(schedule_dir):
    sched = np.load(schedule_dir + "/schedule.npy"); pack_lang = np.load(schedule_dir + "/pack_lang.npy")
    a0sc = {}
    for cand in ("schedule.json", "meta.json"):
        if os.path.exists(os.path.join(schedule_dir, cand)):
            a0sc = json.load(open(os.path.join(schedule_dir, cand))); break
    return sched, pack_lang, a0sc


def _emit(out_dir, name, flat_draws, positions, per_step, pack_lang, ALL_ISOS, parent_sched, parent_dir, a0sc, **extra):
    """Regroup `flat_draws` into steps of per_step (the incomplete last step is dropped and reported) and write schedule.npy, pack_lang.npy, a0_positions.npy,
    schedule.json (sidecar with the parent's pack-structure hash and sha256) and subset.json (per-language draws before / after)."""
    n_full = len(flat_draws) // per_step
    keep = n_full * per_step
    sch = np.ascontiguousarray(flat_draws[:keep].reshape(n_full, per_step), dtype=np.int64)
    os.makedirs(out_dir, exist_ok=True)
    np.save(out_dir + "/schedule.npy", sch); np.save(out_dir + "/pack_lang.npy", pack_lang); np.save(out_dir + "/a0_positions.npy", np.asarray(positions)[:keep])
    projected = bool(a0sc.get("projected")) or str(a0sc.get("source", "")).endswith(".csv")
    sc = write_sidecar(out_dir, sch, pack_lang, ALL_ISOS, arm=name, source=a0sc.get("source"), projected=projected,
                       packed_structure_sha256=a0sc.get("packed_structure_sha256"), a0_schedule=os.path.abspath(parent_dir),
                       a0_schedule_sha256=hashlib.sha256(np.ascontiguousarray(parent_sched, dtype=np.int64).tobytes()).hexdigest(), **extra)
    pl = np.asarray(pack_lang)
    before = np.bincount(pl[parent_sched.reshape(-1)], minlength=len(ALL_ISOS)); after = np.bincount(pl[sch.reshape(-1)], minlength=len(ALL_ISOS))
    rep_ = {"name": name, "parent_draws": int(parent_sched.size), "draws": int(sch.size), "steps": int(n_full), "per_step": int(per_step), "dropped_incomplete_step_draws": int(len(flat_draws) - keep),
            "per_language": {ALL_ISOS[i]: {"parent_draws": int(before[i]), "draws": int(after[i]), "fraction": round(float(after[i]) / max(1, int(before[i])), 4)}
                             for i in range(len(ALL_ISOS)) if before[i] or after[i]}, **extra}
    json.dump(rep_, open(out_dir + "/subset.json", "w"), indent=1)
    return sc, rep_


def select(a):
    """G0 (and any language list): the draws of the A0 schedule whose pack belongs to one of --languages, in A0's order (per-language exposure and passes identical to A0)."""
    PD, ALL_ISOS = _pd()
    sched, pack_lang, a0sc = _a0_context(a.schedule)
    ids = [ALL_ISOS.index(i) for i in a.languages.split(",") if i]
    flat = sched.reshape(-1)
    m = np.isin(pack_lang[flat], ids)
    if not m.any():
        raise SystemExit(f"no draw of {a.languages} in the schedule")
    sc, rep_ = _emit(a.out, a.name, flat[m], np.flatnonzero(m), sched.shape[1], pack_lang, ALL_ISOS, sched, a.schedule, a0sc, languages=a.languages)
    print({k: rep_[k] for k in ("name", "parent_draws", "draws", "steps", "dropped_incomplete_step_draws")})
    return rep_


def head(a):
    """The first --steps steps of a schedule (the 300-step AdamW-vs-Muon sanity pair uses the first 300 steps of A0's schedule). Same sidecar logic."""
    PD, ALL_ISOS = _pd()
    sched, pack_lang, a0sc = _a0_context(a.schedule)
    n = min(int(a.steps), sched.shape[0])
    flat = sched[:n].reshape(-1)
    sc, rep_ = _emit(a.out, a.name, flat, np.arange(flat.size), sched.shape[1], pack_lang, ALL_ISOS, sched, a.schedule, a0sc, head_steps=n)
    print({k: rep_[k] for k in ("name", "parent_draws", "draws", "steps")})
    return rep_


def subset(a):
    """A2 schedule: a seeded subset of A0's draws, in A0's order, built ONCE and shared by every A2 arm (same file, same sha256 in the sidecar).
    --stratify language keeps  max(min(n_l, floor_draws), round(frac * n_l))  draws of EVERY language (n_l = A0's draws of that language), i.e. A0's language mix at
    --frac with a per-language floor so that small languages stay measurable. --method systematic (default) takes evenly spaced draws of each language's A0 sequence with a seeded random
    offset (every pass over the language's packs contributes in proportion, and the draws are spread over the whole run); --method random takes a uniform random subset.
    --stratify none draws round(frac * total) draws from the whole schedule (no floor). --languages first restricts to a language list.
    subset.json lists, per language, the A0 draws, the kept draws and whether the floor applied, plus languages / draws / steps / padded tokens."""
    PD, ALL_ISOS = _pd()
    sched, pack_lang, a0sc = _a0_context(a.schedule)
    flat = sched.reshape(-1)
    pos = np.arange(len(flat))
    if getattr(a, "languages", None):
        ids = [ALL_ISOS.index(i) for i in a.languages.split(",") if i]
        pos = pos[np.isin(pack_lang[flat], ids)]
    if not 0 < a.frac <= 1:
        raise SystemExit("--frac must be in (0, 1]")
    floor = int(getattr(a, "floor_draws", 0) or 0)
    method = getattr(a, "method", "systematic")
    rng = np.random.default_rng([a.seed, 31337])
    lang_of = pack_lang[flat[pos]]
    floored = []
    if a.stratify == "language":
        chosen = []
        for l in np.unique(lang_of):
            idx = np.flatnonzero(lang_of == l)
            n = len(idx)
            k = max(1, int(round(a.frac * n)))
            kf = max(k, min(n, floor))
            if kf > k:
                floored.append(ALL_ISOS[l])
            k = min(kf, n)
            if method == "random":
                pick = np.sort(rng.choice(n, size=k, replace=False))
            else:
                pick = np.minimum(((np.arange(k) + rng.random()) * n / k).astype(np.int64), n - 1)
            chosen.append(idx[pick])
        chosen = np.sort(np.concatenate(chosen))
    else:
        chosen = np.sort(rng.choice(len(pos), size=max(1, int(round(a.frac * len(pos)))), replace=False))
    sel = pos[chosen]
    sc, rep_ = _emit(a.out, a.name, flat[sel], sel, sched.shape[1], pack_lang, ALL_ISOS, sched, a.schedule, a0sc,
                     frac=a.frac, floor_draws=floor, method=method, stratify=a.stratify, seed=a.seed, languages=getattr(a, "languages", None))
    for iso in rep_["per_language"]:
        rep_["per_language"][iso]["floored"] = iso in floored
    rep_["n_languages"] = sum(1 for v in rep_["per_language"].values() if v["draws"] > 0)
    rep_["floored_languages"] = sorted(floored)
    rep_["padded_tokens"] = int(rep_["steps"]) * int(rep_["per_step"]) * SEQ_LEN
    json.dump(rep_, open(a.out + "/subset.json", "w"), indent=1)
    print({k: rep_[k] for k in ("name", "parent_draws", "draws", "steps", "n_languages", "padded_tokens", "dropped_incomplete_step_draws")}, "floored:", len(floored))
    return rep_


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--stage", choices=["A", "B"], required=True); p.add_argument("--packed-dir"); p.add_argument("--packs-csv"); p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1234); p.add_argument("--steps", type=int, default=0); p.add_argument("--per-step", type=int, default=0); p.add_argument("--alpha", type=float, default=STAGE_A["alpha"])
    p.add_argument("--budget-tokens", type=int, default=STAGE_B["budget_tokens"]); p.add_argument("--budget-draws", type=int, default=0, help="stage B: total packs drawn (= steps x per-step)"); p.add_argument("--min-epochs", type=float, default=None); p.add_argument("--max-epochs", type=float, default=None); p.add_argument("--unweighted", action="store_true", help="stage B: no priority weights (pure UniMax water-filling between the floor and the cap)"); p.add_argument("--seen-from", default=None, help="stage B only: schedule dir of an earlier phase (A0); its packs are drawn last in each language's first pass")
    st = sub.add_parser("stitch"); st.add_argument("--old-schedule", required=True); st.add_argument("--prefix-steps", type=int, required=True); st.add_argument("--packed-dir", required=True); st.add_argument("--out", required=True)
    st.add_argument("--budget-draws", type=int, required=True); st.add_argument("--min-epochs", type=float, default=1.0); st.add_argument("--max-epochs", type=float, default=4.0); st.add_argument("--seed", type=int, default=1234)
    q = sub.add_parser("arms"); q.add_argument("--schedule", required=True); q.add_argument("--out", required=True)
    q.add_argument("--languages", default=None, help="comma list of ISO codes; default: the 20 specialist languages of common.SPECIALIST")
    r = sub.add_parser("select", help="G0: the A0 draws of a language list, same order"); r.add_argument("--schedule", required=True); r.add_argument("--out", required=True)
    r.add_argument("--languages", required=True); r.add_argument("--name", default="G0")
    h = sub.add_parser("head", help="the first N steps of a schedule"); h.add_argument("--schedule", required=True); h.add_argument("--out", required=True)
    h.add_argument("--steps", type=int, required=True); h.add_argument("--name", default="head")
    t = sub.add_parser("subset", help="A2: a seeded fraction of the A0 draws, stratified per language, same order")
    t.add_argument("--schedule", required=True); t.add_argument("--out", required=True); t.add_argument("--frac", type=float, required=True)
    t.add_argument("--stratify", choices=["language", "none"], default="language"); t.add_argument("--seed", type=int, default=1234)
    t.add_argument("--floor-draws", type=int, default=0, help="per-language floor: keep at least min(all draws of the language, this many)"); t.add_argument("--method", choices=["systematic", "random"], default="systematic")
    t.add_argument("--languages", default=None); t.add_argument("--name", default="A2")
    a = ap.parse_args()
    {"plan": plan, "stitch": stitch, "arms": arms, "select": select, "subset": subset, "head": head}[a.cmd](a)
