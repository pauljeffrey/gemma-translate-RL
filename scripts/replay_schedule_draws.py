# SPDX-License-Identifier: AGPL-3.0-or-later
"""Replay the (stateless, deterministic) RL prompt schedules of the finished RL phases -> prompts and sampled completions per language (D-65 evidence).

`PromptSchedule` depends only on (seed, step, pool structure, language weights), so the language of every prompt of every step of a finished run can be re-derived
without the run's logs. Used by `scripts/lang_weight_evidence.py` ("how many RL samples did language X see?").

Phases (configs/rl_rtxpro*.yaml of the general RL; all seed 1234, language_alpha 0 except phase 1):
    1    rl_rtxpro.yaml          300 steps, group 8, alpha 0.5 (draw ~ sqrt(pool)), weights {efi: 1e-4}         (APPROXIMATE: the language set of that time is unknown, the v2 pools are used)
    2    rl_rtxpro_phase2.yaml   497 steps, group 8, alpha 0, 17 languages x2.5, efi 1e-4
    2b   stage b64 of phase 2b   488 steps (b64 only; the diverged b256 stage after it was discarded), group 16, same weights
    2c   the run whose metrics.jsonl we have: steps 1-350 efi 1e-4, 351-450 efi absent (weight 1), 451-1433 efi 2.5 (recovered from the logged per-language keys of the run:
         `--check` compares the replay with the logged `reward/lid/fail_rate/<L>` key sets and finds the regime boundaries), group 16, curriculum b256 391 / b512 446 / b1024 149 / b2048 298 / mixed

    python scripts/replay_schedule_draws.py --prompts /workspace/sft_build/gemma/rl_prompts_v3 --out reports/data/lang_draws_by_phase.json
    python scripts/replay_schedule_draws.py --prompts ... --check /workspace/sft_build/runs/rl2c/metrics.jsonl     # regime check against a metrics file
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from typing import Dict, List

import numpy as np

W17 = {"tir": 2.5, "amh": 2.5, "kbp": 2.5, "nus": 2.5, "kam": 2.5, "bam": 2.5, "mos": 2.5, "umb": 2.5, "kmb": 2.5, "dyu": 2.5, "kea": 2.5, "luo": 2.5, "kik": 2.5, "orm": 2.5,
       "fon": 2.5, "yor": 2.5, "vmw": 2.5}
BINS_2C = {"b64": 64, "b256": 48, "b512": 42, "b1024": 42, "b2048": 21, "mixed": 42}
BINS_P2 = {"b64": 128, "b256": 96, "b512": 84, "b1024": 84, "mixed": 84}
BINS_P2B = {"b64": 64, "b256": 48, "b512": 42, "b1024": 42, "mixed": 42}
BINS_P1 = {"b64": 64, "b256": 48, "b512": 42, "b1024": 42, "mixed": 42}


def load_pool(path: str, max_tokens: int):
    import pyarrow.parquet as pq

    iso, lb = [], []
    for f in sorted(glob.glob(path.rstrip("/") + "/**/*.parquet", recursive=True)):
        t = pq.read_table(f, columns=["iso", "length_bin", "src_afr_tokens"]).to_pandas()
        t = t[t.src_afr_tokens <= max_tokens]
        iso += t.iso.tolist()
        lb += t.length_bin.tolist()
    return np.asarray(iso), np.asarray(lb)


def replay(iso, lb, *, seed, alpha, weights, curriculum, per_bin, per_step_default, steps, step_from=0):
    from gemma_mt.rl.data import PromptSchedule

    sch = PromptSchedule(iso, alpha, seed, length_bin=lb, curriculum=curriculum, language_weights=weights)
    out = []
    for s in range(step_from, steps):
        b = sch.bin_of_step(s)
        n = per_bin.get(b, per_step_default)
        out.append([str(iso[i]) for i in sch.indices(s, n)])
    return out


def regimes_2c():
    base = dict(W17)
    return [(0, 350, {**base, "efi": 0.0001}), (350, 450, dict(base)), (450, 1433, {**base, "efi": 2.5})]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="rl_prompts_v3 directory (parquet incl. zz_b2048/)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", default=None, help="metrics.jsonl of the 2c run: compare the replayed language sets with the logged ones")
    a = ap.parse_args(argv)
    iso, lb = load_pool(a.prompts, 2048)
    cur2c = {"b256": 391, "b512": 446, "b1024": 149, "b2048": 298}
    res: Dict[str, dict] = {}
    steps2c: List[List[str]] = []
    for (lo, hi, w) in regimes_2c():
        steps2c += replay(iso, lb, seed=1234, alpha=0.0, weights=w, curriculum=cur2c, per_bin=BINS_2C, per_step_default=42, steps=hi, step_from=lo)
    res["2c"] = {"group": 16, "steps": len(steps2c), "draws": steps2c}
    p2 = replay(iso, lb, seed=1234, alpha=0.0, weights={**W17, "efi": 0.0001}, curriculum={"b64": 98, "b256": 130, "b512": 75, "b1024": 45}, per_bin=BINS_P2, per_step_default=84, steps=497)
    res["2"] = {"group": 8, "steps": 497, "draws": p2}
    p2b = replay(iso, lb, seed=1234, alpha=0.0, weights={**W17, "efi": 0.0001}, curriculum={"b64": 488}, per_bin=BINS_P2B, per_step_default=64, steps=488)
    res["2b_b64"] = {"group": 16, "steps": 488, "draws": p2b}
    p1 = replay(iso, lb, seed=1234, alpha=0.5, weights={"efi": 0.0001}, curriculum={"b64": 40, "b256": 60, "b512": 60, "b1024": 60}, per_bin=BINS_P1, per_step_default=42, steps=300)
    res["1"] = {"group": 8, "steps": 300, "draws": p1}
    if a.check:
        seen = {}
        for line in open(a.check):
            r = json.loads(line)
            if "reward" in r:
                seen[r["step"]] = {m.group(1) for k in r for m in [re.match(r"reward/lid/fail_rate/(.+)$", k)] if m}
        ok = sum(1 for s, d in enumerate(steps2c, 1) if s in seen and (set(d) - {"pcm", "kon"}) == seen[s])
        print(f"2c replay: {ok}/{len(seen)} steps reproduce the logged language set exactly (pcm / kon have the GlotLID guard off and never appear in the key set)")
    summary = {}
    for ph, d in res.items():
        c = Counter(l for step in d["draws"] for l in step)
        summary[ph] = {"group": d["group"], "steps": d["steps"], "prompts": dict(sorted(c.items()))}
    if a.out:
        json.dump({"summary": summary, "steps_2c": steps2c}, open(a.out, "w"))
        print("wrote", a.out)
    for ph, s in summary.items():
        print(ph, "prompts", sum(s["prompts"].values()), "completions", sum(s["prompts"].values()) * s["group"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
