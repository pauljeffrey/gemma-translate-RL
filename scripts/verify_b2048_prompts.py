#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-45 acceptance check of `rl_prompts_v3` against `rl_prompts_v2` with the REAL loader, hold-out and schedule of the trainer. CPU only (loads ~10 M prompts twice: ~25 GB RAM).

    CUDA_VISIBLE_DEVICES= python scripts/verify_b2048_prompts.py --v2 DIR --v3 DIR --old-config configs/rl_rtxpro_phase2c.yaml --new-config configs/rl_rtxpro_phase2c_b2048.yaml [--out report.json]

Checks (each prints PASS / FAIL, the exit code is 1 on any FAIL):
  1. counts per language and bin (loader view of v3: 77 languages x {b64,b256,b512,b1024} as in v2, + b2048); b2048 src_afr_tokens in 1025..2048; licence and source columns present
  2. the first len(v2) rows of v3, as the loader returns them, are exactly the rows of v2 (same order => every old prompt keeps its global index)
  3. hold-out: the plain hash selection of v3 equals that of v2 (no b2048 window enters it); with eval_min_per_bin the old hold-out is a subset; hold-out English is disjoint from the training English
  4. schedule: PromptSchedule.bin_of_step for steps 0..steps-1 of the old config vs the new one (equal up to the end of b1024, then b2048, then the mixture) and
     PromptSchedule.indices(step, n) of the post-hold-out tables identical for every step of the old curriculum (bytes of the English prompts compared)
  5. Gemma prompt lengths (--tokenizer) of ALL b2048 rows: max against vllm_max_model_len - max_new_tokens
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--v3", required=True)
    ap.add_argument("--old-config", required=True)
    ap.add_argument("--new-config", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    from gemma_mt import format as fmt
    from gemma_mt.rl.data import MIXED, PromptSchedule, load_rl_prompts
    from gemma_mt.rl.train_rl import RLConfig, _subset, split_eval_prompts
    from gemma_mt.utils import dataclass_from_dict, load_yaml

    ok = True
    res = {}

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg, flush=True)
        ok = ok and bool(cond)

    old = dataclass_from_dict(RLConfig, load_yaml(a.old_config))
    new = dataclass_from_dict(RLConfig, load_yaml(a.new_config))
    t0 = time.time()
    p2 = load_rl_prompts(a.v2, max_src_tokens=old.max_src_tokens)
    print(f"v2 loaded: {len(p2)} prompts, drops {p2.dropped} ({time.time() - t0:.0f}s)", flush=True)
    p3 = load_rl_prompts(a.v3, max_src_tokens=new.max_src_tokens)
    print(f"v3 loaded: {len(p3)} prompts, drops {p3.dropped}, name mismatches {p3.name_mismatches} ({time.time() - t0:.0f}s)", flush=True)
    n2 = len(p2)

    # 1. counts
    cnt = collections.Counter(zip(p3.iso, p3.length_bin))
    langs = sorted(set(p3.iso))
    check(len(langs) == 77, f"77 languages in v3 (got {len(langs)})")
    b2 = {l: cnt.get((l, "b2048"), 0) for l in langs}
    check(len(set(b2.values())) == 1 and 400 <= next(iter(b2.values())) <= 700, f"b2048 rows per language: {sorted(set(b2.values()))}")
    c2 = collections.Counter(zip(p2.iso, p2.length_bin))
    check(all(cnt.get(k, 0) == v for k, v in c2.items()) and set(k for k in cnt if k[1] != "b2048") == set(c2), "old (language, bin) counts identical in v2 and v3")
    res["rows_per_language_bin"] = {l: {b: cnt.get((l, b), 0) for b in ("b64", "b256", "b512", "b1024", "b2048")} for l in langs}
    toks = [t for t, b in zip(p3.src_afr_tokens, p3.length_bin) if b == "b2048"]
    check(toks and min(toks) >= 1025 and max(toks) <= 2048, f"b2048 src_afr_tokens in 1025..2048 (min {min(toks)}, max {max(toks)}, n {len(toks)})")
    import pyarrow.parquet as pq
    import glob
    import os

    t = pq.read_table(sorted(glob.glob(os.path.join(a.v3, "zz_b2048", "*.parquet")))[0]).to_pandas()
    check(set(t.licence) <= {"odc-by", "share_alike"} and t.licence.notna().all() and t.source_dataset.notna().all() and (t.length_bin == "b2048").all(),
          f"licence / source_dataset present in the b2048 files: {t.groupby(['source_dataset', 'licence']).size().to_dict()}")

    # 2. old rows identical and in the same positions
    check(p3.english[:n2] == p2.english and p3.iso[:n2] == p2.iso and p3.length_bin[:n2] == p2.length_bin and p3.src_afr_tokens[:n2] == p2.src_afr_tokens,
          f"the first {n2} rows of v3 are exactly the rows of v2 (every old prompt keeps its global index); the {len(p3) - n2} new rows are all b2048: {set(p3.length_bin[n2:]) == {'b2048'}}")

    # 3. hold-out
    n_eval = min(old.eval_prompts, len(p2) // 10)
    tr2, ev2 = split_eval_prompts(p2, n_eval, old.seed)
    tr3_plain, ev3_plain = split_eval_prompts(p3, min(new.eval_prompts, len(p3) // 10), new.seed)
    tr3, ev3 = split_eval_prompts(p3, min(new.eval_prompts, len(p3) // 10), new.seed, new.eval_min_per_bin)
    check(ev3_plain == ev2, f"plain hash hold-out of v3 == that of v2 ({len(ev2)} prompts; no b2048 window among the {n_eval} smallest hashes)")
    check(set(ev2) <= set(ev3), "hold-out of the running trainer is a subset of the new hold-out")
    added = sorted(set(ev3) - set(ev2))
    check(all(p3.length_bin[i] == "b2048" for i in added) and all(i >= n2 for i in added), f"hold-out additions are b2048 rows only: {len(added)} added")
    eb = collections.Counter(p3.length_bin[i] for i in ev3)
    check(all(eb[b] >= new.eval_min_per_bin for b in ("b64", "b256", "b512", "b1024", "b2048")), f"hold-out per bin: {dict(eb)}")
    ev_text = {p3.english[i] for i in ev3}
    clash = sum(1 for i in tr3 if p3.english[i] in ev_text)
    clash2 = sum(1 for i in tr2 if p2.english[i] in {p2.english[j] for j in ev2})
    print(f"  english of a hold-out prompt also in the training set: v3 {clash} rows, v2 (running trainer) {clash2} rows", flush=True)
    res["holdout"] = {"per_bin": dict(eb), "n": len(ev3), "english_clash_rows_v3": clash, "english_clash_rows_v2": clash2}
    check(not any(i >= n2 and p3.english[i] in ev_text for i in tr3), "no b2048 hold-out window is also a b2048 training prompt")

    # 4. schedule
    ptr2, ptr3 = _subset(p2, tr2), _subset(p3, tr3)
    so = PromptSchedule(ptr2.iso, old.language_alpha, old.seed, length_bin=ptr2.length_bin, curriculum=old.curriculum_steps, language_weights=old.language_weights)
    sn = PromptSchedule(ptr3.iso, new.language_alpha, new.seed, length_bin=ptr3.length_bin, curriculum=new.curriculum_steps, language_weights=new.language_weights)
    staged_old = sum(old.curriculum_steps.values())
    mism = [s for s in range(staged_old) if so.bin_of_step(s) != sn.bin_of_step(s)]
    check(not mism, f"bin_of_step identical for steps 0..{staged_old - 1} (old stages b256 / b512 / b1024)")
    seq_o = [so.bin_of_step(s) for s in range(old.steps)]
    seq_n = [sn.bin_of_step(s) for s in range(new.steps)]
    runs = lambda seq: [(k, len(list(g))) for k, g in __import__("itertools").groupby(seq)]
    print(f"  old stage map {runs(seq_o)}\n  new stage map {runs(seq_n)}", flush=True)
    check(runs(seq_n) == runs(seq_o)[:-1] + [("b2048", new.curriculum_steps["b2048"]), (MIXED, runs(seq_o)[-1][1])], "new stage map = old map with b2048 inserted before the (unchanged 149-step) mixed stage")
    bad = 0
    for s in range(staged_old):
        n_o = old.prompts_per_step_by_bin.get(so.bin_of_step(s), old.prompts_per_step)
        n_n = new.prompts_per_step_by_bin.get(sn.bin_of_step(s), new.prompts_per_step)
        io, inn = so.indices(s, n_o), sn.indices(s, n_n)
        if n_o != n_n or [ptr2.english[i] for i in io] != [ptr3.english[i] for i in inn] or [ptr2.iso[i] for i in io] != [ptr3.iso[i] for i in inn]:
            bad += 1
    check(bad == 0, f"PromptSchedule.indices: the prompts drawn at steps 0..{staged_old - 1} are identical (text and language) in the old and the new setup (bad steps: {bad})")
    b2_steps = range(staged_old, staged_old + new.curriculum_steps["b2048"])
    drawn = collections.Counter()
    for s in b2_steps:
        drawn.update(ptr3.iso[i] for i in sn.indices(s, new.prompts_per_step_by_bin["b2048"]))
        assert {ptr3.length_bin[i] for i in sn.indices(s, new.prompts_per_step_by_bin["b2048"])} == {"b2048"}
    wmax, wmin = max(drawn.values()), min(drawn.values())
    print(f"  b2048 stage: {sum(drawn.values())} prompts drawn over {len(drawn)} languages, per language {wmin}..{wmax} (weighted languages ~2.5x)", flush=True)
    res["b2048_stage_prompts_per_language"] = dict(drawn)
    uniq = len({ptr3.english[i] for s in b2_steps for i in sn.indices(s, new.prompts_per_step_by_bin["b2048"])})
    res["b2048_stage_unique_english"] = uniq
    print(f"  distinct b2048 sources drawn in the stage: {uniq} of {sum(drawn.values())}", flush=True)
    mixed_bins = collections.Counter()
    for s in range(staged_old + new.curriculum_steps["b2048"], new.steps):
        mixed_bins.update(ptr3.length_bin[i] for i in sn.indices(s, new.prompts_per_step_by_bin["mixed"]))
    tot = sum(mixed_bins.values())
    print(f"  mixture stage bins: {({k: round(v / tot, 3) for k, v in sorted(mixed_bins.items())})}", flush=True)
    check(set(mixed_bins) == {"b64", "b256", "b512", "b1024", "b2048"} and all(0.17 < v / tot < 0.23 for v in mixed_bins.values()), "the mixture draws uniformly over the 5 bins")
    res["mixed_bins"] = dict(mixed_bins)

    # 5. Gemma prompt lengths
    if a.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        idx = [i for i in range(n2, len(p3))]
        lens = []
        for k in range(0, len(idx), 512):
            chunk = idx[k:k + 512]
            enc = tok([fmt.prompt_text(tok, p3.english[i], p3.iso[i]) for i in chunk], add_special_tokens=False)["input_ids"]
            lens += [len(x) for x in enc]
        import numpy as np

        arr = np.array(lens)
        lim = new.vllm_max_model_len - new.max_new_tokens_by_bin["b2048"]
        print(f"  Gemma prompt tokens of all {len(arr)} b2048 rows: min {arr.min()} p50 {int(np.median(arr))} p99 {int(np.percentile(arr, 99))} max {arr.max()} (limit {lim})", flush=True)
        res["b2048_gemma_prompt_tokens"] = {"min": int(arr.min()), "p50": int(np.median(arr)), "p90": int(np.percentile(arr, 90)), "p99": int(np.percentile(arr, 99)), "max": int(arr.max()), "limit": int(lim)}
        check(arr.max() <= lim, "every b2048 prompt + max_new_tokens fits vllm_max_model_len")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(("ALL PASS" if ok else "SOME CHECK FAILED") + f" ({time.time() - t0:.0f}s)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
