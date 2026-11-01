#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real-GPU check of the QE ensemble reward (D-55): the OLD reward (AfriCOMET-QE alone) and the NEW one (ensemble) on the SAME samples.

    python scripts/qe_ensemble_gpu_check.py --config configs/rl_science.yaml --v8 calibration/shaping_flores_dev_v8.json \
        --a5 /workspace/jobs/a5 --out /workspace/jobs/b1/gpu_check.json --steps 4

Samples = real policy samples (a5/policy_samples.jsonl: the final RL checkpoint at T=1, FLORES+ dev prompts, 8 samples per prompt), 84 prompts x 8 = 672 per step, mixed languages,
the real `reward:` block of the yaml (GlotLID off for the check). One worker process per scorer (transformers 4.x child, as in training). Reports per step: seconds of
QE (single-model worker vs the two-model worker), GPU memory of the worker, Pearson / Spearman of the two rewards, scale, share of chunks in the min branch, and a
hack probe (copies / partial copies from a5/hack_rows.jsonl) scored by both rewards.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gpu_mem_mb(pid: int) -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        return sum(int(l.split(",")[1]) for l in out.splitlines() if l.strip() and int(l.split(",")[0]) == pid)
    except Exception:
        return -1


class PeakPoller:
    """Samples the GPU memory of one process (nvidia-smi) in a thread; `.peak` in MiB."""

    def __init__(self, pid, every=0.5):
        import threading

        self.pid, self.every, self.peak, self._stop = pid, every, 0, threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, gpu_mem_mb(self.pid))
            self._stop.wait(self.every)

    def stop(self):
        self._stop.set()
        self.t.join(timeout=5)
        return self.peak


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--v8", required=True)
    ap.add_argument("--a5", default="/workspace/jobs/a5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--prompts-per-step", type=int, default=84)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=None, help="override scorer.batch_size (AfriCOMET)")
    ap.add_argument("--ensemble-batch-size", type=int, default=None, help="scorer.ensemble_batch_size (SSA-COMET-MTL; default = the AfriCOMET batch size)")
    ap.add_argument("--skip-hack", action="store_true")
    ap.add_argument("--long-probe", type=int, default=0, help="worst-case memory: score this many LONG chunks (src ~ --long-words words, mt of similar length) with both models and exit")
    ap.add_argument("--long-words", type=int, default=230)
    ap.add_argument("--lid", action="store_true", help="keep GlotLID on (default off for the check)")
    a = ap.parse_args(argv)

    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
    from gemma_mt.rl.comet import WorkerScorer
    from gemma_mt.rl.qe_ensemble import QEPairScorer
    from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
    from gemma_mt.rl.train_rl import RLConfig, build_scorer
    from gemma_mt.utils import dataclass_from_dict, load_yaml

    y = load_yaml(a.config)
    cfg = dataclass_from_dict(RLConfig, y)
    for d in cfg.extra_libs:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.append(d)
    base = dict(cfg.reward)
    base["calibration_path"] = a.v8
    if not a.lid:
        base["lid_enabled"] = False
    rc_old = dataclass_from_dict(RewardConfig, {**base, "qe_ensemble": False})
    rc_new = dataclass_from_dict(RewardConfig, {**base, "qe_ensemble": True})
    spec = dict(y["scorer"])
    if a.batch_size:
        spec["batch_size"] = a.batch_size
    if a.ensemble_batch_size:
        spec["ensemble_batch_size"] = a.ensemble_batch_size
    tok = FakeGemmaTokenizer()
    eot = SPECIALS["<end_of_turn>"]
    if a.long_probe:
        rows = [json.loads(l) for l in open(os.path.join(a.a5, "policy_rows.jsonl"), encoding="utf-8") if l.strip()]
        pairs, k = [], 0
        while len(pairs) < a.long_probe:
            src, mt = [], []
            while sum(len(x.split()) for x in src) < a.long_words:
                r = rows[k % len(rows)]
                src.append(r["src"]); mt.append(r["mt"]); k += 7
            pairs.append((" ".join(src), " ".join(mt)))
        pair = build_scorer({**spec, "ensemble_model_id": "McGill-NLP/ssa-comet-mtl"})
        poll = PeakPoller(pair.owner.proc.pid, every=0.2)
        try:
            lens = pair.primary.worker.last_stats
            t = time.time(); o1 = pair([p[0] for p in pairs], [p[1] for p in pairs]); t1 = time.time() - t
            t = time.time(); o2 = pair.score_secondary([p[0] for p in pairs], [p[1] for p in pairs]); t2 = time.time() - t
            st = pair.secondary.last_stats
        finally:
            peak = poll.stop()
            pair.close()
        res = {"n": len(pairs), "avg_src_words": sum(len(p[0].split()) for p in pairs) / len(pairs), "avg_mt_words": sum(len(p[1].split()) for p in pairs) / len(pairs),
               "batch_size": spec.get("batch_size"), "ensemble_batch_size": spec.get("ensemble_batch_size"), "seconds_africomet": t1, "seconds_ssa_mtl": t2, "ssa_stats": st,
               "peak_gpu_mb": peak, "mean_scores": [sum(o1) / len(o1), sum(o2) / len(o2)]}
        print(json.dumps(res))
        json.dump(res, open(a.out, "w"), indent=1)
        return 0

    # ---- samples: groups of `group` = one prompt
    pol = [json.loads(l) for l in open(os.path.join(a.a5, "policy_samples.jsonl"), encoding="utf-8") if l.strip()]
    groups = defaultdict(list)
    for r in pol:
        groups[(r["iso"], r["id"])].append(r)
    keys = sorted(k for k, v in groups.items() if len(v) == a.group)
    rng = random.Random(a.seed)
    rng.shuffle(keys)

    def mk(rows):
        return [Sample(tok.encode(r["raw"], add_special_tokens=False) + [eot], r["iso"], r["src"], domain="general") for r in rows]

    steps = [[r for k in keys[s * a.prompts_per_step:(s + 1) * a.prompts_per_step] for r in groups[k]] for s in range(a.steps)]

    class Timed:
        """Wraps a scorer: seconds spent inside the model calls."""

        def __init__(self, sc):
            self.sc, self.t = sc, 0.0
            for n in ("score_secondary", "pop_times", "token_lengths", "secondary", "primary"):
                if hasattr(sc, n):
                    setattr(self, n, getattr(sc, n))

        def __call__(self, s, m, r=None):
            t = time.time()
            o = self.sc(s, m)
            self.t += time.time() - t
            return o

    result = {"config": a.config, "v8": a.v8, "steps": [], "samples_per_step": a.prompts_per_step * a.group}
    # ---- OLD: single-model worker
    t0 = time.time()
    w1 = WorkerScorer(python=spec.get("python", sys.executable), model_id=spec.get("model_id"), batch_size=spec.get("batch_size", 32), pythonpath=spec.get("pythonpath"))
    result["old_worker_startup_s"] = time.time() - t0
    old_rewards, old_out = [], []
    poll_old = PeakPoller(w1.proc.pid)
    try:
        r_old = TranslationReward(rc_old, w1, tok, [eot])
        for si, rows in enumerate(steps):
            t = time.time()
            o = r_old(mk(rows), group_size=a.group)
            old_out.append((o, time.time() - t))
        result["old_gpu_mb_after_steps"] = gpu_mem_mb(w1.proc.pid)
        result["old_gpu_mb_peak"] = poll_old.stop()
        # hack probe, old
        hk = [json.loads(l) for l in open(os.path.join(a.a5, "hack_rows.jsonl"), encoding="utf-8") if l.strip()]
        probe_cls = ["human", "mismatch", "copy_source", "copy_noisy", "mix_half_src", "append_source", "swap25_src", "drop_half", "append_unrelated"]
        probe = defaultdict(list)
        for r in hk:
            if r["cls"] in probe_cls and r["j"] < 12:
                probe[r["cls"]].append(r)
        probe_old = {}
        for c, rows in ({} if a.skip_hack else probe).items():
            ss = [Sample(tok.encode(f"{r['iso']} {r['mt']}", add_special_tokens=False) + [eot], r["iso"], r["src"], domain="general") for r in rows]
            o = r_old(ss, group_size=None)
            probe_old[c] = o
    finally:
        w1.close()
    # ---- NEW: two models in one worker
    t0 = time.time()
    pair = build_scorer({**spec, "ensemble_model_id": rc_new.qe_ensemble_model_id})
    result["new_worker_startup_s"] = time.time() - t0
    poll_new = PeakPoller(pair.owner.proc.pid) if getattr(pair, "owner", None) is not None and hasattr(pair.owner, "proc") else None
    try:
        r_new = TranslationReward(rc_new, pair, tok, [eot])
        for si, rows in enumerate(steps):
            pair.pop_times()
            t = time.time()
            o = r_new(mk(rows), group_size=a.group)
            dt = time.time() - t
            tp, ts = o.stats.get("qe_ens/seconds_primary", 0.0), o.stats.get("qe_ens/seconds_secondary", 0.0)
            oo, dt_old = old_out[si]
            # compare on samples scored in both (same status and a QE-judged sample)
            idx = [i for i in range(len(rows)) if oo.status[i] == "ok" and o.status[i] == "ok" and oo.qe[i] is not None and o.qe2[i] is not None]
            x, z = [oo.rewards[i] for i in idx], [o.rewards[i] for i in idx]
            gcorr = []
            for g0 in range(0, len(rows), a.group):
                ii = [i for i in range(g0, g0 + a.group) if i in set(idx)]
                if len(ii) >= 4:
                    gcorr.append(spearman([oo.rewards[i] for i in ii], [o.rewards[i] for i in ii]))
            result["steps"].append({
                "n": len(rows), "n_compared": len(idx), "seconds_reward_old": dt_old, "seconds_reward_new": dt, "seconds_qe_primary": tp, "seconds_qe_secondary": ts,
                "pearson": pearson(x, z), "spearman": spearman(x, z), "within_group_spearman_mean": sum(gcorr) / max(len(gcorr), 1),
                "mean_old": sum(x) / len(x), "mean_new": sum(z) / len(z), "std_old": (sum((v - sum(x) / len(x)) ** 2 for v in x) / len(x)) ** 0.5,
                "std_new": (sum((v - sum(z) / len(z)) ** 2 for v in z) / len(z)) ** 0.5,
                "min_new": min(z), "max_new": max(z), "status_old": {k: oo.status.count(k) for k in set(oo.status)}, "status_new": {k: o.status.count(k) for k in set(o.status)},
                "weights_changed": sum(1 for i in range(len(rows)) if abs(oo.weights[i] - o.weights[i]) > 1e-9),
                "qe_ens": {k: v for k, v in o.stats.items() if k.startswith("qe_ens/") and k.count("/") == 1}})
        result["new_gpu_mb_after_steps"] = gpu_mem_mb(pair.owner.proc.pid)
        result["new_gpu_mb_peak"] = poll_new.stop() if poll_new else None
        probe_res = {}
        for c, rows in ({} if a.skip_hack else probe).items():
            ss = [Sample(tok.encode(f"{r['iso']} {r['mt']}", add_special_tokens=False) + [eot], r["iso"], r["src"], domain="general") for r in rows]
            o = r_new(ss, group_size=None)
            po = probe_old[c]
            ok = [i for i in range(len(rows)) if po.status[i] == "ok" and o.status[i] == "ok"]
            probe_res[c] = {"n": len(rows), "n_ok_both": len(ok), "mean_old": sum(po.rewards[i] for i in ok) / max(len(ok), 1), "mean_new": sum(o.rewards[i] for i in ok) / max(len(ok), 1),
                            "status_old": {k: po.status.count(k) for k in set(po.status)}, "status_new": {k: o.status.count(k) for k in set(o.status)},
                            "min_share": o.stats.get("qe_ens/min_share"), "min_share_ssa_high": o.stats.get("qe_ens/min_share_ssa_high")}
        result["hack_probe"] = probe_res
    finally:
        pair.close()
    json.dump(result, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "steps"}, indent=1))
    for s in result["steps"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k not in ("status_old", "status_new", "qe_ens")}, {k: round(v, 3) for k, v in s["qe_ens"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
