#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mixture v2, component B step 4: choose the documents of the long-document distillation from the scored candidates (OUT/ld_scored.jsonl).

    python scripts/mix2_longdoc_select.py --work /workspace/jobs/r11/work/ld [--min-mean-r 1.0 --min-p25-r 0.4 --min-mean-conf 58.8 --min-p25-conf 45 --budget-tokens 52000]

A document is ELIGIBLE when: every sentence has a valid translation (coverage 100 %), no repetition loop, no word-n-gram repetition above the human p97, GlotLID accepts the whole
translation (where the guard is on), diacritic deficit 0, length ratio inside the human window (x1.25), AND
  QE languages        mean shaped ensemble reward >= --min-mean-r and the 25th percentile of the sentence rewards >= --min-p25-r  (reward 1.0 = the language's median human reference);
  consensus languages mean chrF++ of the picked sentences against the other candidates >= --min-mean-conf and p25 >= --min-p25-conf (no QE exists for these 23 languages).
Per language a token budget (--budget-tokens, prompt + target, measured with the model's tokenizer) is filled from the eligible documents bin by bin with the shares --bin-share
(b512 / b1024 / b2048 of the budget), highest score first, science and general documents alternating inside a bin. A document whose pair is longer than the pack is skipped.
Output OUT/ld_selected.jsonl (english, target, iso, bin, domain, source_dataset, licence, scores, n_tokens) and OUT/ld_select_stats.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", default="/workspace/sft_build/runs/final_1433")
    ap.add_argument("--rel-q", type=float, default=0.70, help="per-language threshold = this quantile of the language's own (hard-filter-passing) documents ...")
    ap.add_argument("--mean-r-clip", default="0.3,1.0", help="... clipped to [lo, hi]: QE languages, mean shaped reward (1.0 = median human reference, 0 = the language's 'mismatched' anchor)")
    ap.add_argument("--p25-r-clip", default="0.0,0.4")
    ap.add_argument("--mean-conf-clip", default="50,58.8", help="consensus-only languages: mean chrF++ of the pick against the other candidates (58.8 = the RL archive's consensus entry)")
    ap.add_argument("--p25-conf-clip", default="38,45")
    ap.add_argument("--budget-tokens", type=int, default=52000)
    ap.add_argument("--bin-share", default="b512=0.25,b1024=0.40,b2048=0.35")
    ap.add_argument("--max-pack", type=int, default=8192)
    ap.add_argument("--out-name", default="ld_selected", help="OUT/<name>.jsonl and <name>_stats.json")
    ap.add_argument("--total-cap", type=int, default=4_000_000)
    ap.add_argument("--lang-budget-mult", default="", help="iso=mult,... multiplier of the per-language budget (priority languages)")
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from gemma_mt import format as fmt

    tok = AutoTokenizer.from_pretrained(a.model)
    share = {k: float(v) for k, v in (x.split("=") for x in a.bin_share.split(","))}
    rows = [json.loads(l) for l in open(f"{a.work}/ld_scored.jsonl", encoding="utf-8")]
    drops = collections.defaultdict(collections.Counter)
    elig = collections.defaultdict(list)
    clip = lambda s_: tuple(float(x) for x in s_.split(","))

    def hard_ok(r):
        return bool(r.get("target")) and not r["loop"] and not r["repeat"] and r["lid_doc"] is not False and r["diacritic_deficit"] <= 1e-9 and r["len_ok"]

    def quant(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(q * len(v)))] if v else None

    thr = {}
    for iso in sorted({r["iso"] for r in rows}):
        pool = [r for r in rows if r["iso"] == iso and hard_ok(r)]
        if not pool:
            continue
        if pool[0]["mode"] == "qe":
            lo, hi = clip(a.mean_r_clip); lo2, hi2 = clip(a.p25_r_clip)
            thr[iso] = (min(max(quant([r["mean_r"] for r in pool], a.rel_q), lo), hi), min(max(quant([r["p25_r"] for r in pool], a.rel_q), lo2), hi2))
        else:
            lo, hi = clip(a.mean_conf_clip); lo2, hi2 = clip(a.p25_conf_clip)
            thr[iso] = (min(max(quant([r["mean_conf"] for r in pool], a.rel_q), lo), hi), min(max(quant([r["p25_conf"] for r in pool], a.rel_q), lo2), hi2))
    for r in rows:
        iso = r["iso"]
        why = None
        if not r.get("target"):
            why = "uncovered_sentence"
        elif r["loop"]:
            why = "loop"
        elif r["repeat"]:
            why = "repetition"
        elif r["lid_doc"] is False:
            why = "lid"
        elif r["diacritic_deficit"] > 1e-9:
            why = "diacritics"
        elif not r["len_ok"]:
            why = "length_window"
        elif r["mode"] == "qe":
            if r["mean_r"] < thr[iso][0]:
                why = "mean_r"
            elif r["p25_r"] < thr[iso][1]:
                why = "p25_r"
        else:
            if r["mean_conf"] < thr[iso][0]:
                why = "mean_conf"
            elif r["p25_conf"] < thr[iso][1]:
                why = "p25_conf"
        if why:
            drops[why][iso] += 1
            continue
        ex = fmt.build_example(r["english"], iso, r["target"], tok)
        r["n_tokens"], r["target_tokens"] = ex.n_tokens, ex.target_len
        if ex.n_tokens > a.max_pack:
            drops["longer_than_pack"][iso] += 1
            continue
        r["score"] = r["mean_r"] if r["mode"] == "qe" else r["mean_conf"]
        elig[iso].append(r)
    sel = []
    per_lang = {}
    for iso in sorted({r["iso"] for r in rows}):
        got = {"tokens": 0, "by_bin": collections.Counter(), "by_domain": collections.Counter()}
        pool = elig.get(iso, [])
        mult = {k: float(v) for k, v in (x.split("=") for x in a.lang_budget_mult.split(",") if x)}.get(iso, 1.0)
        lang_budget = a.budget_tokens * mult
        chosen = []
        for b in sorted(share, key=lambda x: -int(x[1:])):          # long bins first: they are the scarce ones
            bud = lang_budget * share[b]
            cand = [r for r in pool if r["bin"] == b]
            sci = sorted([r for r in cand if r["domain"] == "science"], key=lambda r: -r["score"])
            gen = sorted([r for r in cand if r["domain"] != "science"], key=lambda r: -r["score"])
            order = []
            while sci or gen:
                if sci:
                    order.append(sci.pop(0))
                if gen:
                    order.append(gen.pop(0))
            used = 0
            for r in order:
                if used + r["n_tokens"] > bud + 0.25 * r["n_tokens"]:      # a document may overshoot the bin budget a little
                    continue
                used += r["n_tokens"]
                chosen.append(r)
        # fill what is left of the language budget (a bin that had too few eligible documents) with the best remaining ones, shortest bins last
        left = lang_budget - sum(r["n_tokens"] for r in chosen)
        taken = {id(r) for r in chosen}
        for r in sorted([r for r in pool if id(r) not in taken], key=lambda r: -r["score"]):
            if r["n_tokens"] <= left:
                chosen.append(r)
                left -= r["n_tokens"]
        for r in chosen:
            got["tokens"] += r["n_tokens"]
            got["by_bin"][r["bin"]] += 1
            got["by_domain"][r["domain"]] += 1
        sel += chosen
        n_in = sum(1 for r in rows if r["iso"] == iso)
        per_lang[iso] = {"candidates": n_in, "eligible": len(pool), "selected": len(chosen), "tokens": got["tokens"], "by_bin": dict(got["by_bin"]), "by_domain": dict(got["by_domain"]),
                         "mode": next((r["mode"] for r in rows if r["iso"] == iso), None), "pass_rate": len(pool) / max(n_in, 1)}
    # global cap (should not bind at the default budget)
    tot = sum(r["n_tokens"] for r in sel)
    if tot > a.total_cap:
        sel.sort(key=lambda r: -r["score"])
        out, t = [], 0
        for r in sel:
            if t + r["n_tokens"] <= a.total_cap:
                out.append(r)
                t += r["n_tokens"]
        sel = out
    with open(f"{a.work}/{a.out_name}.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(sel, key=lambda r: (r["iso"], r["bin"], r["doc_key"])):
            keep = ("english", "target", "iso", "bin", "domain", "doc_key", "source_dataset", "licence", "n_tokens", "target_tokens", "mode", "mean_r", "p25_r", "min_r",
                    "mean_r_greedy", "p25_r_greedy", "mean_conf", "p25_conf", "n_sents", "frac_greedy", "field")
            f.write(json.dumps({k: r[k] for k in keep if k in r}, ensure_ascii=False) + "\n")
    stats = {"config": vars(a), "documents_scored": len(rows), "eligible": sum(len(v) for v in elig.values()), "selected": len(sel), "tokens": sum(r["n_tokens"] for r in sel),
             "target_tokens": sum(r["target_tokens"] for r in sel), "by_bin": dict(collections.Counter(r["bin"] for r in sel)), "by_domain": dict(collections.Counter(r["domain"] for r in sel)),
             "by_mode": dict(collections.Counter(r["mode"] for r in sel)), "drops": {k: sum(v.values()) for k, v in drops.items()}, "languages_with_data": sum(1 for v in per_lang.values() if v["selected"]),
             "per_language": per_lang, "thresholds": {k: list(v) for k, v in thr.items()}, "drops_by_language": {k: dict(v) for k, v in drops.items()}}
    json.dump(stats, open(f"{a.work}/{a.out_name}_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("per_language", "config", "drops_by_language")}, indent=1))


if __name__ == "__main__":
    main()
