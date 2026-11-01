#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Time the monitor / selection dev sets of an RL run exactly as the trainer evaluates them (D-56): greedy decoding through the vLLM sampler, macro chrF++, seconds per set.

    python scripts/time_monitor_sets.py --model runs/final_1433 --vllm-python /workspace/venv_vllm/bin/python --gpu-util 0.09 \
        --set weak_dev=/workspace/jobs/b2/bench_weak_flores_dev.jsonl:60:256 --set general_dev=.../bench_dev_clean.jsonl:40:256 --out times.json

--set NAME=PATH:PER_LANGUAGE:MAX_NEW_TOKENS[:BATCH]  (the entries of `benchmark_eval_path` / `benchmark_aux`). The trainer's eval runs on the policy's own engine (already
warm, weights synced); here the engine is started first and its start-up time is reported separately, so `seconds` is the per-evaluation cost to expect.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vllm-python", required=True)
    ap.add_argument("--gpu-util", type=float, default=0.09)
    ap.add_argument("--max-model-len", type=int, default=10240)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--extra-libs", default="/workspace/sft_build/pylibs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.extra_libs and a.extra_libs not in sys.path:
        sys.path.append(a.extra_libs)

    from gemma_mt.eval import runtime as RT
    from gemma_mt.languages import SUPPORTED_RL
    from gemma_mt.rl.train_rl import RLConfig, evaluate_benchmark, load_benchmark_dev

    tok, stop, pad = RT.load_tokenizer_and_stops(a.model)
    RT.wait_for_gpu(a.gpu_util * 96.0 * 0.9, 600)
    t0 = time.time()
    sampler = RT.make_vllm_sampler(a.model, python=a.vllm_python, gpu_util=a.gpu_util, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
                                   log_path=os.path.join(os.path.dirname(os.path.abspath(a.out)), "vllm_worker_monitor.log"))
    res = {"engine_start_seconds": time.time() - t0, "sets": {}}
    try:
        for spec in a.set:
            name, rest = spec.split("=", 1)
            parts = rest.split(":")
            path, per, mnt = parts[0], int(parts[1]), int(parts[2])
            bs = int(parts[3]) if len(parts) > 3 else 32
            rows = load_benchmark_dev(path, SUPPORTED_RL, per, 1234)
            cfg = RLConfig(benchmark_max_new_tokens=mnt, benchmark_batch_size=bs)
            t = time.time()
            r = evaluate_benchmark(None, tok, rows, cfg, stop.ids, pad, "cuda", None, sampler=sampler)
            sec = time.time() - t
            res["sets"][name] = {"rows": len(rows), "languages": len({x["iso"] for x in rows}), "seconds": sec, "chrf_macro": r["bench/chrf_macro"], "finished_rate": r["bench/finished_rate"],
                                 "chrf": {k.split("/")[-1]: v for k, v in r.items() if k.startswith("bench/chrf/")},
                                 "finished": {k.split("/")[-1]: v for k, v in r.items() if k.startswith("bench/finished/")}}
            print(f"[monitor] {name}: {len(rows)} rows, {res['sets'][name]['languages']} languages, {sec:.1f}s, chrF++ macro {r['bench/chrf_macro']:.2f}, finished {r['bench/finished_rate']:.3f}", flush=True)
    finally:
        sampler.close()
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
