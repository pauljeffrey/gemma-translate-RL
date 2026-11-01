# SPDX-License-Identifier: AGPL-3.0-or-later
"""The REAL TranslationReward (gemma_mt/rl/reward.py) over study items or generated groups, QE scores from the disk cache (R8).

  python reward_run.py --items items_main.jsonl --out rw_v1_items.jsonl [--set longdoc_v2=true ...] [--no-gpu]
  python reward_run.py --groups groups --items items_main.jsonl --out rw_v1_groups.jsonl [--set ...]

Reward config = the reward block of configs/rl_rtxpro_phase2c.yaml (the general final model's run) + qe_ensemble (D-55) + calibration v8, GlotLID off (the length effects
under study do not depend on it), keep_chunk_rewards on. `--set key=value` overrides RewardConfig fields (YAML scalars)."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import qe_cache  # noqa: E402

from gemma_mt.eval.longdoc_reward import build_reward, completion_ids, stop_ids_of  # noqa: E402
from gemma_mt.rl.reward import Sample  # noqa: E402

TREE = os.environ.get("R8_TREE", "/workspace/jobs/r8/gemma_mt_r8")


def parse_set(items):
    import yaml

    out = {}
    for kv in items or []:
        k, v = kv.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/workspace/jobs/r8/qe_cache")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--conds", nargs="*", default=None, help="restrict items to these conditions")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--qe-batch", type=int, default=32)
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--bins", nargs="*", type=int, default=None)
    ap.add_argument("--max-units", type=int, default=None)
    a = ap.parse_args()
    docs = C.core_docs()
    ov = {"qe_ensemble": True, "lid_enabled": False, "keep_chunk_rewards": True,
          "calibration_path": f"{TREE}/calibration/shaping_flores_dev_v8.json", "lid_model_path": None}
    ov.update(parse_set(a.set))
    if ov.get("longdoc_v2") and "longdoc_calibration_path" not in ov:
        ov["longdoc_calibration_path"] = f"{TREE}/calibration/longdoc_v2.json"
    scorer = qe_cache.build_scorer(a.cache, gpu=not a.no_gpu, batch_size=a.qe_batch)
    reward, tok, meta = build_reward(f"{TREE}/configs/rl_rtxpro_phase2c.yaml", overrides=ov, scorer=scorer, tokenizer_json="/workspace/sft_build/runs/final_1433/tokenizer.json")
    stop = stop_ids_of(tok)[0]
    rows = []
    if a.groups:
        for fn in sorted(os.listdir(a.groups)):
            if fn.endswith(".jsonl.gz"):
                with gzip.open(os.path.join(a.groups, fn), "rt") as f:
                    rows += [json.loads(l) for l in f]
        order = {}
        for r in rows:
            order.setdefault(r["rid"], []).append(r)
        units = [sorted(v, key=lambda r: (r["kind"] == "greedy", r["kind"])) for v in order.values()]       # sampled members first, the greedy one LAST (the trainer's layout)
        G = len(units[0])
    else:
        items = C.rd(a.items)
        if a.conds:
            items = [it for it in items if it["cond"] in a.conds]
        rows = items
        units = [[it] for it in items]
        G = None
    if a.langs:
        units = [u for u in units if u[0]["iso"] in a.langs]
    if a.bins:
        units = [u for u in units if u[0].get("bin") in a.bins]
    units.sort(key=lambda u: (u[0]["iso"], u[0].get("bin", 0)))
    if a.max_units:
        units = units[: a.max_units]
    out = []
    t0 = time.time()
    k = 0
    while k < len(units):
        batch = [u for u in units[k: k + a.batch] if len(u) == len(units[k])]
        if not batch:
            batch = [units[k]]
        k += max(len(batch), 1)
        samples, meta_rows = [], []
        for u in batch:
            for r in u:
                src = r["src_text"] if "src_text" in r else docs[r["doc_id"]]["text"]
                if a.groups:
                    ids = r["ids"]
                else:
                    ids = completion_ids(tok, r["iso"], r["out"], r.get("finished", True), stop)
                samples.append(Sample(ids, r["iso"], src))
                meta_rows.append(r)
        isos = {s.iso for s in samples}
        res = reward(samples, group_size=G if a.groups else None) if len(isos) >= 1 else None
        for r, rw, st, q, ch, ld, nc, fc in zip(meta_rows, res.rewards, res.status, res.qe, res.chunk_rewards, res.longdoc or [None] * len(meta_rows), res.n_chunks, res.failed_chunks):
            d = {k2: r[k2] for k2 in ("doc_id", "iso", "bin", "cond", "kind", "rid", "finish", "r6_loop", "r6_cov", "finished") if k2 in r}
            d.update(reward=float(rw), status=st, qe=q, chunks=ch, longdoc=ld, n_chunks=nc, failed_chunks=fc, n_tokens=len(r["ids"]) if a.groups else None)
            out.append(d)
        if (k // len(batch)) % 20 == 0:
            print(f"{k}/{len(units)} units, {time.time() - t0:.0f}s", flush=True)
    C.wr(a.out, out)
    scorer.close()
    print("done", len(out), "rows", f"{time.time() - t0:.0f}s", "cache", scorer.hits, scorer.misses, "qe seconds", [round(x, 1) for x in scorer.total_seconds], "qe pairs", scorer.total_pairs)


if __name__ == "__main__":
    main()
