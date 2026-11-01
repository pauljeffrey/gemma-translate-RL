#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mixture v2, component A: HIGH-CONFIDENCE rows of the RL best-translation archive as SFT windows.

    python scripts/mix2_archive.py --out /workspace/jobs/r11/work/arch [--archive rl2c=/workspace/sft_build/runs/rl2c/archive/archive.sqlite ...]

Selection (every step counts the rows it removes, per language, in OUT/archive_stats.json):
  1. archive_publish.row_problem (copy of the source, repetition, length ratio) ; --drop-after name=STEP for runs that diverged;
  2. repetition: word 3/4/5-gram repeat shares of the translation above max(human p97 of its length bucket, the source's own) -> rejected;
  3. GlotLID guard (calibration `lid` blocks: accepted labels / minimum length / on-off), diacritic deficit == 0, length ratio inside the human window (x1.35);
  4. the entry thresholds RAISED: mode qe: reward >= --min-reward-qe (entry 1.0); blend: reward >= --min-reward-blend (entry 1.0) and consensus chrF++ >= --min-chrf-blend
     (entry 50); consensus-only languages (no QE at all): consensus chrF++ >= --min-chrf-consensus (entry 58.8), by default VERY high (70) ;
  5. RE-JUDGED by the current QE ENSEMBLE (AfriCOMET-QE + SSA-COMET-MTL, D-55) on the aligned sentences (plan_chunks): source-length-weighted mean of the
     shaped sentence rewards >= --min-ens-mean and the 25th percentile of the sentence rewards >= --min-ens-p25 (qe / blend languages; the archive was filled
     under AfriCOMET alone, so a copy-friendly translation that only SSA-COMET likes - or the reverse - falls out here) ;
  6. hold-out (exact + word 8-gram, the same object as mix_v1: AfriScience dev/test, AfriLongDoc hard/core, bench_dev, FLORES+ dev, RL-vs-SFT pool) ;
  7. licence join (RL prompt pools, key = sha1(iso + NFC(source))[:16]), per-language cap in tokens (--cap-tokens): all b1024+ rows first, then b512, then the rest,
     highest ensemble mean first.
Output: OUT/archive_selected.jsonl (english, target, iso, key, mode, reward, qe, consensus_chrf, ens_mean, ens_p25, length_bin, step_best, source_dataset, licence, n_tokens).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mix2_common as C  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--archive", action="append", default=[], help="name=path (default rl2c=/workspace/sft_build/runs/rl2c/archive/archive.sqlite)")
    ap.add_argument("--drop-after", action="append", default=[])
    ap.add_argument("--prompts-dirs", default="/workspace/sft_build/gemma/rl_prompts_v3,/workspace/sft_build/gemma/rl_prompts_v3/zz_b2048,/workspace/sft_build/gemma/rl_prompts_v2")
    ap.add_argument("--min-reward-qe", type=float, default=1.4)
    ap.add_argument("--min-reward-blend", type=float, default=1.25)
    ap.add_argument("--min-chrf-blend", type=float, default=60.0)
    ap.add_argument("--min-chrf-consensus", type=float, default=70.0)
    ap.add_argument("--min-ens-mean", type=float, default=1.0)
    ap.add_argument("--min-ens-p25", type=float, default=0.3)
    ap.add_argument("--cap-tokens", type=int, default=60000, help="per-language cap on the SFT tokens (prompt + target) of this component")
    ap.add_argument("--model", default="/workspace/sft_build/runs/final_1433")
    ap.add_argument("--no-qe", action="store_true", help="skip the ensemble re-judgement (testing)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    arch = dict(x.split("=", 1) for x in a.archive) or {"rl2c": "/workspace/sft_build/runs/rl2c/archive/archive.sqlite"}
    drop_after = {x.split("=")[0]: int(x.split("=")[1]) for x in a.drop_after}

    from transformers import AutoTokenizer

    from gemma_mt import format as fmt
    from gemma_mt.rl.archive import make_key, read_rows
    from gemma_mt.rl.chunk_plan import plan_chunks

    AP = C.load_script("archive_publish")
    J = C.Judge(with_scorer=not a.no_qe)
    tok = AutoTokenizer.from_pretrained(a.model)
    ho = C.make_holdout()
    drops = collections.defaultdict(collections.Counter)      # reason -> iso -> n
    rows = []
    for name, path in arch.items():
        rs = read_rows(path)
        for r in rs:
            r["run"] = name
            if name in drop_after and r["step_best"] > drop_after[name]:
                drops["suspect_run_tail"][r["iso"]] += 1
                continue
            rows.append(r)
    log(f"{len(rows)} archive rows read from {list(arch)}")
    n_in = collections.Counter(r["iso"] for r in rows)
    bins_in = collections.Counter((r["length_bin"]) for r in rows)

    # ------------------------------------------------ 1-4 cheap filters
    keep = []
    for r in rows:
        iso = r["iso"]
        src, tr = unicodedata.normalize("NFC", r["source"]), unicodedata.normalize("NFC", r["translation"])
        r["source"], r["translation"] = src, tr
        mode = J.mode(iso)
        r["mode_now"] = mode
        p = AP.row_problem(r)
        if p:
            drops["row_problem:" + p][iso] += 1
            continue
        rep = J.repeat_bad(iso, src, tr)
        if rep:
            drops["repetition"][iso] += 1
            continue
        lid = J.lid_ok(iso, tr)
        if lid is False:
            drops["lid"][iso] += 1
            continue
        if (r["diacritic_deficit"] or 0) > 1e-9 or J.diacritic_deficit(iso, src, tr) > 1e-9:
            drops["diacritics"][iso] += 1
            continue
        if not J.len_ok(iso, src, tr):
            drops["length_window"][iso] += 1
            continue
        if r["mode"] == "qe":
            ok = r["reward"] >= a.min_reward_qe
        elif r["mode"] == "blend":
            ok = r["reward"] >= a.min_reward_blend and (r["consensus_chrf"] or 0) >= a.min_chrf_blend
        else:
            ok = (r["consensus_chrf"] or 0) >= a.min_chrf_consensus
        if not ok:
            drops["below_raised_threshold:" + r["mode"]][iso] += 1
            continue
        keep.append(r)
    log(f"after cheap filters and raised thresholds: {len(keep)}")

    # ------------------------------------------------ 5 ensemble re-judgement
    if not a.no_qe:
        segs, owner = [], []
        for i, r in enumerate(keep):
            if not J.has_qe(r["iso"]):
                continue
            plan = plan_chunks(r["source"], r["translation"], w_hi=(J.shaper.len_window.get(r["iso"]) or (None, None))[1])
            r["_wts"] = []
            for it in plan.items:
                if it[0] is not None:
                    s, m = plan.segments[it[0]]
                    segs.append((s, m, r["iso"]))
                    owner.append(i)
                    r["_wts"].append(float(max(len(s), 1)))
            r["_plan_failed"] = sum(1 for it in plan.items if it[0] is None)
        log(f"scoring {len(segs)} aligned sentence pairs of {len(set(owner))} rows with both judges")
        raw = J.qe_raw([s for s, _, _ in segs], [m for _, m, _ in segs], [l for _, _, l in segs])
        per = collections.defaultdict(list)
        for (s, m, l), (c1, c2), i in zip(segs, raw, owner):
            per[i].append(J.shaped(c1, c2, l)[2])
        for i, r in enumerate(keep):
            vals = per.get(i)
            if vals:
                w = r["_wts"]
                r["ens_mean"] = sum(v * x for v, x in zip(vals, w)) / sum(w)
                sv = sorted(vals)
                r["ens_p25"] = sv[int(0.25 * (len(sv) - 1))]
                r["n_sent"] = len(vals)
    final = []
    for r in keep:
        iso = r["iso"]
        if J.has_qe(iso) and not a.no_qe:
            if r.get("ens_mean") is None or r["ens_mean"] < a.min_ens_mean:
                drops["ensemble_mean"][iso] += 1
                continue
            if r["ens_p25"] < a.min_ens_p25:
                drops["ensemble_p25"][iso] += 1
                continue
            if r.get("_plan_failed"):
                drops["failed_chunk_in_plan"][iso] += 1
                continue
        # ------------------------------------------------ 6 hold-out
        h = ho.hit(r["source"])
        if h:
            drops["holdout_" + h][iso] += 1
            continue
        final.append(r)
    log(f"after ensemble and hold-out: {len(final)}")

    # ------------------------------------------------ 7 licence join, tokens, per-language cap
    keys = {r["key"] for r in final}
    lic = {}
    import pyarrow.parquet as pq

    for d in a.prompts_dirs.split(","):
        for iso in sorted({r["iso"] for r in final}):
            p = f"{d}/{iso}.parquet"
            if not os.path.exists(p):
                continue
            t = pq.read_table(p, columns=["english", "iso", "source_dataset", "licence"]).to_pydict()
            for e, i, sd, lc in zip(t["english"], t["iso"], t["source_dataset"], t["licence"]):
                k = make_key(i, e)
                if k in keys and k not in lic:
                    lic[k] = (sd or "", lc or "")
    for r in final:
        r["source_dataset"], r["licence"] = lic.get(r["key"], ("", ""))
        ex = fmt.build_example(r["source"], r["iso"], r["translation"], tok)
        r["n_tokens"], r["target_tokens"] = ex.n_tokens, ex.target_len
    by = collections.defaultdict(list)
    for r in final:
        by[r["iso"]].append(r)
    sel, capped = [], collections.Counter()
    order_bin = {"b2048": 0, "b1024": 1, "b512": 2, "b256": 3, "b64": 4}
    for iso, rs in by.items():
        rs.sort(key=lambda r: (order_bin.get(r["length_bin"], 9), -(r.get("ens_mean") if r.get("ens_mean") is not None else r["reward"])))
        used = 0
        for r in rs:
            if r["n_tokens"] > 8192:
                drops["longer_than_pack"][iso] += 1
                continue
            if used + r["n_tokens"] > a.cap_tokens:
                capped[iso] += 1
                continue
            used += r["n_tokens"]
            sel.append(r)
    log(f"selected {len(sel)} rows, {sum(r['n_tokens'] for r in sel):,} tokens, {len({r['iso'] for r in sel})} languages; capped away {sum(capped.values())}")

    with open(os.path.join(a.out, "archive_selected.jsonl"), "w", encoding="utf-8") as f:
        for r in sorted(sel, key=lambda r: (r["iso"], r["key"])):
            f.write(json.dumps({"english": r["source"], "target": r["translation"], "iso": r["iso"], "key": r["key"], "mode": r["mode"], "reward": r["reward"],
                                "qe": r["qe"], "consensus_chrf": r["consensus_chrf"], "ens_mean": r.get("ens_mean"), "ens_p25": r.get("ens_p25"),
                                "length_bin": r["length_bin"], "step_best": r["step_best"], "run": r["run"], "source_dataset": r["source_dataset"], "licence": r["licence"],
                                "n_tokens": r["n_tokens"], "target_tokens": r["target_tokens"]}, ensure_ascii=False) + "\n")
    langs = sorted(set(n_in) | {r["iso"] for r in sel})
    per_lang = {l: {"in": n_in.get(l, 0), "selected": sum(1 for r in sel if r["iso"] == l), "tokens": sum(r["n_tokens"] for r in sel if r["iso"] == l),
                    "by_bin": dict(collections.Counter(r["length_bin"] for r in sel if r["iso"] == l)), "capped": capped.get(l, 0),
                    "drops": {k: v[l] for k, v in drops.items() if v.get(l)}} for l in langs}
    stats = {"in": len(rows), "bins_in": dict(bins_in), "selected": len(sel), "tokens": sum(r["n_tokens"] for r in sel),
             "bins_selected": dict(collections.Counter(r["length_bin"] for r in sel)), "mode_selected": dict(collections.Counter(r["mode"] for r in sel)),
             "drops": {k: sum(v.values()) for k, v in drops.items()}, "config": vars(a), "per_language": per_lang}
    json.dump(stats, open(os.path.join(a.out, "archive_stats.json"), "w"), indent=1)
    J.close()
    log("done")


if __name__ == "__main__":
    main()
