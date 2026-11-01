#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which prompts do the RL stages DRAW for the NLLB-signal languages? (clean rerun, agent T2, 2026-10-29)

The NLLB pseudo-reference signal (reward.nllb_signal) is a dictionary lookup: a prompt without a precomputed NLLB translation gets no signal. With the pool-wide precompute of R14 only 10-17 % of the drawn
prompts were covered (D-67), so the precompute has to be aimed at the prompts the stages actually draw. The schedule is deterministic in (seed, step): this script loads the real pool exactly as the trainer does
(same hold-out split, same `build_prompt_schedule`), replays every step of every stage yaml (optionally out to the +50 % extension horizon), keeps the draws of the NLLB languages in the allowed length bins, and
writes one parquet per language (columns english, source_dataset) that `scripts/precompute_nllb_pseudorefs.py --pools <out>/draws_pool/{iso}.parquet --max-sentences 0` translates in full.

    python scripts/clean_rerun_nllb_draws.py --root /workspace/clean_rerun --out /workspace/clean_rerun/nllb_draws          # A0 yamls, languages from configs/clean_rerun/nllb_languages.json
    python scripts/clean_rerun_nllb_draws.py ... --arm specialist --max-bin b512 --no-extension

Stages: those of the yamls that have `reward.nllb_signal: true`. `--max-bin` (default b256): windows of longer bins get no pseudo-reference (the sentence-level NLLB output of long windows is not validated), so in the mixed
stage and the final run the signal then acts on the b64 / b256 part of the draws only. Prints a table (unique prompts, sentences, minutes and USD at 25 / 45 / 60 sentences per second and USD 0.58 per hour).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Set

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

BIN_ORDER = ["b64", "b256", "b512", "b1024", "b2048"]
RATES = (25.0, 45.0, 60.0)          # sentences per second: R14 measured 25-60 for NLLB-3.3B int8 on a shared RTX PRO 6000
USD_PER_H = 0.58


def draws_of(prompts: Any, schedule: Any, cfg: Any, languages: Sequence[str], horizon: int, max_bin: Optional[str]) -> Dict[str, Set[int]]:
    """{iso: {prompt index}} drawn in steps 0 .. horizon-1 by `schedule` (indices into `prompts`) for `languages`, restricted to bins up to `max_bin`."""
    allowed = set(BIN_ORDER[: BIN_ORDER.index(max_bin) + 1]) if max_bin else None
    langs = set(languages)
    out: Dict[str, Set[int]] = {l: set() for l in languages}
    for step in range(horizon):
        b = schedule.bin_of_step(step) if cfg.length_bins else None
        n = cfg.prompts_per_step_by_bin.get(b, cfg.prompts_per_step) if b else cfg.prompts_per_step
        for i in schedule.indices(step, n):
            iso = prompts.iso[i]
            if iso in langs and (allowed is None or not prompts.length_bin or prompts.length_bin[i] in allowed):
                out[iso].add(int(i))
    return out


def stage_draws(yaml_path: str, languages: Sequence[str], extension: bool, max_bin: Optional[str], root_override: Optional[str] = None) -> Dict[str, Any]:
    from gemma_mt.rl.data import load_rl_prompts
    from gemma_mt.rl.train_rl import RLConfig, _subset, build_prompt_schedule, split_eval_prompts
    from gemma_mt.utils import dataclass_from_dict, load_yaml

    d = load_yaml(yaml_path)
    if root_override:
        d["prompts_path"] = d["prompts_path"].replace("/workspace/clean_rerun", root_override)
    cfg = dataclass_from_dict(RLConfig, d)
    prompts = load_rl_prompts(cfg.prompts_path, max_src_tokens=cfg.max_src_tokens)
    tr_idx, _ = split_eval_prompts(prompts, min(cfg.eval_prompts, len(prompts) // 10), cfg.seed, cfg.eval_min_per_bin)
    train = _subset(prompts, tr_idx)
    sch = build_prompt_schedule(cfg, train)
    horizon = int(cfg.steps * (1 + (cfg.extend_max_frac if (extension and cfg.extend_if_improving) else 0.0)) + 1e-9)
    drawn = draws_of(train, sch, cfg, languages, horizon, max_bin)
    return {"cfg": cfg, "train": train, "drawn": drawn, "horizon": horizon, "caps_horizon": cfg.steps}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/workspace/clean_rerun")
    ap.add_argument("--configs", default=None, help="yaml directory (default configs/clean_rerun or clean_rerun_specialist next to scripts/)")
    ap.add_argument("--arm", choices=["a0", "specialist"], default="a0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--languages-file", default=None, help="nllb_languages.json (default configs/clean_rerun/nllb_languages.json)")
    ap.add_argument("--langs", nargs="*", default=None, help="override the language list")
    ap.add_argument("--max-bin", default="b256", choices=BIN_ORDER + ["all"], help="longest length bin that gets a pseudo-reference (default b256)")
    ap.add_argument("--no-extension", action="store_true", help="replay the base caps only (default: out to the +50 %% extension horizon)")
    a = ap.parse_args(argv)

    sys.path.insert(0, HERE)
    import clean_rerun_make_configs as gen
    from gemma_mt.rl.segments import split_sentences
    from gemma_mt.utils import load_yaml

    cfg_dir = os.path.abspath(a.configs or os.path.join(HERE, "..", "configs", "clean_rerun" if a.arm == "a0" else "clean_rerun_specialist"))
    nl = gen.load_nllb_languages(a.arm == "specialist", a.languages_file)
    langs = list(a.langs) if a.langs else nl["languages"]
    max_bin = None if a.max_bin == "all" else a.max_bin
    if nl["provisional"] and not a.langs:
        print(f"WARNING: language list {langs} is PROVISIONAL", flush=True)
    union: Dict[str, Dict[str, Any]] = {l: {} for l in langs}
    rows_summary = []
    pools: Dict[str, Any] = {}
    for stage, S in gen.STAGES.items():
        path = os.path.join(cfg_dir, S["file"])
        y = load_yaml(path)
        if not (y.get("reward") or {}).get("nllb_signal") or stage == "preflight":
            continue
        r = stage_draws(path, langs, not a.no_extension, max_bin, a.root if a.root != "/workspace/clean_rerun" else None)
        train = r["train"]
        n_unique = 0
        n_sent = 0
        for l in langs:
            for i in sorted(r["drawn"][l]):
                en = train.english[i]
                if en not in union[l]:
                    union[l][en] = train.source_dataset[i]
                    n_unique += 1
                    n_sent += len(split_sentences(en))
        rows_summary.append({"stage": stage, "steps_replayed": r["horizon"], "cap": r["caps_horizon"], "languages": len(langs), "new_unique_prompts": n_unique, "new_sentences": n_sent})
    os.makedirs(os.path.join(a.out, "draws_pool"), exist_ok=True)
    total_prompts = sum(len(v) for v in union.values())
    total_sent = sum(len(split_sentences(en)) for v in union.values() for en in v)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for l, v in union.items():
            pq.write_table(pa.table({"english": list(v.keys()), "source_dataset": list(v.values())}), os.path.join(a.out, "draws_pool", f"{l}.parquet"), compression="zstd")
    except ImportError:
        print("pyarrow is not installed: only the summary was written (run this on the rental)", flush=True)
    est = {f"{int(rt)}_sent_per_s": {"minutes": round(total_sent / rt / 60, 1), "usd_at_0.58_per_h": round(total_sent / rt / 3600 * USD_PER_H, 3)} for rt in RATES}
    summ = {"languages": langs, "provisional_language_list": bool(nl["provisional"]), "max_bin": a.max_bin, "extension_horizon": not a.no_extension, "per_stage_new": rows_summary,
            "unique_prompts_total": total_prompts, "sentences_total": total_sent, "decode_time_and_cost": est,
            "commands": {"precompute": (f"python scripts/precompute_nllb_pseudorefs.py --langs {' '.join(langs)} --out {a.root}/assets/pseudorefs_reward_only_nc_v1 --ct2-dir <nllb-200-3.3B-ct2-int8> "
                                        f"--hf-dir <nllb tokenizer dir> --pools '{a.out}/draws_pool/{{iso}}.parquet' --max-sentences 0 --max-batch-tokens 16384")}}
    with open(os.path.join(a.out, "draws_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(f"{'stage':8s} {'steps replayed':>14s} {'new unique prompts':>19s} {'new sentences':>14s}")
    for r in rows_summary:
        print(f"{r['stage']:8s} {r['steps_replayed']:14d} {r['new_unique_prompts']:19d} {r['new_sentences']:14d}")
    print(f"total {total_prompts} unique prompts, {total_sent} sentences; NLLB-3.3B decode: " + "; ".join(f"{k}: {v['minutes']} min / USD {v['usd_at_0.58_per_h']}" for k, v in est.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
