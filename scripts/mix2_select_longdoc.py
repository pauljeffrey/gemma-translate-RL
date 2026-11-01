#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mixture v2, component B step 1: pick the SOURCE documents for the long-document distillation and write the sentence-level generation requests.

    python scripts/mix2_select_longdoc.py --out /workspace/jobs/r11/work/ld [--per-lang b512=22,b1024=16,b2048=10]

Sources: the RL prompt pools (`rl_prompts_v3` b512 / b1024, `rl_prompts_v3/zz_b2048`) and the science pool of A4 (`rl_prompts_sci_v1`, all sources
except `dsfsi/afriscience_mt`, whose human references are already in the science part). Per language and bin a seeded random sample, general vs science
shares per bin (`--sci-share`), filtered by: hold-out (exact + word 8-gram, the SAME hold-out object as mix_v1), stable sentence split, the Gemma-token
length of the English prompt (b2048 <= --max-prompt-b2048 so the pair fits the 8,192 pack), no duplicate English text within a language.

Writes OUT/docs.jsonl (one row per (language, document)), OUT/req_sent.jsonl (one request per unique (language, sentence piece): prompt ids + token cap)
and OUT/select_stats.json.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import importlib.util
import json
import os
import random
import sys
import time

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_builder():
    spec = importlib.util.spec_from_file_location("build_sft_science_mix", os.path.join(HERE, "build_sft_science_mix.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parse_kv(s):
    return {k: int(v) for k, v in (x.split("=") for x in s.split(","))}


def make_holdout(B, a=None):
    import pandas as pd
    from types import SimpleNamespace
    ns = SimpleNamespace(longdoc_dir="/workspace/jobs/a4/holdout/afrilongdoc", bench_dev="/workspace/sft_build/bench/bench_dev_clean.jsonl",
                         flores_dev_eng="/workspace/sft_build/hf/hub/datasets--openlanguagedata--flores_plus/snapshots/*/dev/eng_Latn.jsonl",
                         eval_pool="/workspace/rlcheck/res/pool.jsonl")
    afs = {sp: pd.read_parquet(f"/workspace/jobs/a4/afs_raw/corpus/{sp}-00000.parquet") for sp in ("dev", "test")}
    return B.build_holdout(ns, afs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/workspace/sft_build/runs/final_1433")
    ap.add_argument("--pool-v3", default="/workspace/sft_build/gemma/rl_prompts_v3")
    ap.add_argument("--pool-sci", default="/workspace/jobs/a4/sci/rl_prompts_sci_v1")
    ap.add_argument("--per-lang", default="b512=22,b1024=16,b2048=10", help="candidate documents per language and bin")
    ap.add_argument("--sci-share", default="b512=0.25,b1024=0.5,b2048=0.5")
    ap.add_argument("--max-prompt-b2048", type=int, default=1500, help="Gemma tokens of the prompt (incl. template) a b2048 document may have")
    ap.add_argument("--max-prompt-b1024", type=int, default=1100)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--languages", default="")
    ap.add_argument("--exclude", default="", help="comma separated docs.jsonl files whose documents (same language + text) must not be selected again")
    ap.add_argument("--kind", default="longdoc", help="tag written to every row (longdoc | anchor)")
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer
    from gemma_mt.eval.longdoc import split_long_sentence, split_sentences, sentences_are_stable
    from gemma_mt.format import build_prompt
    from gemma_mt.languages import ALL_ISOS

    B = load_builder()
    ho = make_holdout(B)
    tok = AutoTokenizer.from_pretrained(a.model)
    ntok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
    os.makedirs(a.out, exist_ok=True)
    per = parse_kv(a.per_lang)
    sci_share = {k: float(v) for k, v in (x.split("=") for x in a.sci_share.split(","))}
    langs = [l for l in ALL_ISOS if os.path.exists(f"{a.pool_v3}/{l}.parquet")]
    if a.languages:
        langs = [l for l in langs if l in a.languages.split(",")]
    sci_share = {b: sci_share.get(b, 0.25) for b in per}
    log(f"{len(langs)} languages")
    excl = collections.defaultdict(set)
    for fpath in [x for x in a.exclude.split(',') if x]:
        for l in open(fpath, encoding='utf-8'):
            d0 = json.loads(l)
            excl[d0['iso']].add(hashlib.md5(d0['english'].encode()).hexdigest())
    docs, stats = [], collections.defaultdict(collections.Counter)
    seen_global = collections.Counter()
    for iso in langs:
        rng = random.Random(f"{a.seed}|{iso}")
        # ---------------- pools of this language
        pools = {}                           # (domain, bin) -> list of row dicts
        t3 = pq.read_table(f"{a.pool_v3}/{iso}.parquet", columns=["english", "length_bin", "src_afr_tokens", "n_sentences", "source_dataset", "licence"])
        lb = t3.column("length_bin").to_pylist()
        for b in [x for x in per if x != "b2048"]:
            idx = [i for i, x in enumerate(lb) if x == b]
            rng.shuffle(idx)
            pools[("general", b)] = [(t3, i) for i in idx[:per[b] * 40]]
        if "b2048" in per:
            p = f"{a.pool_v3}/zz_b2048/{iso}.parquet"
            t2 = pq.read_table(p, columns=["english", "length_bin", "src_afr_tokens", "n_sentences", "source_dataset", "licence", "doc_id"])
            idx = list(range(t2.num_rows))
            rng.shuffle(idx)
            pools[("general", "b2048")] = [(t2, i) for i in idx[:per["b2048"] * 40]]
        ts = pq.read_table(f"{a.pool_sci}/{iso}.parquet", columns=["english", "length_bin", "src_afr_tokens", "n_sentences", "source_dataset", "licence", "domain", "field"])
        sb, ss, sd = ts.column("length_bin").to_pylist(), ts.column("source_dataset").to_pylist(), ts.column("domain").to_pylist()
        for b in per:
            idx = [i for i in range(ts.num_rows) if sb[i] == b and not ss[i].startswith("dsfsi/afriscience_mt") and sd[i] == "science"]
            rng.shuffle(idx)
            pools[("science", b)] = [(ts, i) for i in idx[:per[b] * 40]]
        # ---------------- take candidates
        seen_text = set(excl.get(iso, ()))
        for b in per:
            n_sci = int(round(per[b] * sci_share[b]))
            want = {"science": n_sci, "general": per[b] - n_sci}
            for dom in ("science", "general"):
                got = 0
                for (t, i) in pools[(dom, b)]:
                    if got >= want[dom]:
                        break
                    en = t.column("english")[i].as_py()
                    key = hashlib.md5(en.encode()).hexdigest()
                    if key in seen_text:
                        stats[iso]["dup"] += 1
                        continue
                    h = ho.hit(en)
                    if h:
                        stats[iso]["holdout_" + h] += 1
                        continue
                    sents = split_sentences(en)
                    if not sents or not sentences_are_stable(sents):
                        stats[iso]["unstable_split"] += 1
                        continue
                    pt = len(build_prompt(en, iso, tok))
                    if (b == "b2048" and pt > a.max_prompt_b2048) or (b == "b1024" and pt > a.max_prompt_b1024):
                        stats[iso]["too_long_prompt"] += 1
                        continue
                    seen_text.add(key)
                    got += 1
                    row = {"iso": iso, "bin": b, "domain": dom, "english": en, "sents": sents, "prompt_tokens": pt,
                           "source_dataset": t.column("source_dataset")[i].as_py(), "licence": t.column("licence")[i].as_py(),
                           "doc_key": key[:16], "kind": a.kind}
                    if dom == "science":
                        row["field"] = t.column("field")[i].as_py()
                    docs.append(row)
                    stats[iso][f"{b}/{dom}"] += 1
                if got < want[dom]:
                    stats[iso][f"short_{b}/{dom}"] = want[dom] - got
        log(f"{iso}: {sum(v for k, v in stats[iso].items() if '/' in k and not k.startswith('short'))} docs")
    with open(os.path.join(a.out, "docs.jsonl"), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    # ---------------- sentence requests
    uniq = {}
    for d in docs:
        for s in d["sents"]:
            for p in split_long_sentence(s, 700):
                uniq.setdefault((d["iso"], p), None)
    reqs = []
    for k, (iso, p) in enumerate(sorted(uniq)):
        ids = build_prompt(p, iso, tok)
        st = ntok(p)
        reqs.append({"rid": f"s{k}", "iso": iso, "text": p, "ids": ids, "src_tokens": st, "max_new": min(int(3.5 * st) + 16, 1024), "capped": False})
    with open(os.path.join(a.out, "req_sent.jsonl"), "w", encoding="utf-8") as f:
        for r in reqs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {"docs": len(docs), "requests": len(reqs), "by_bin": collections.Counter(d["bin"] for d in docs), "by_domain": collections.Counter(d["domain"] for d in docs),
            "sentences": sum(len(d["sents"]) for d in docs), "per_language": {k: dict(v) for k, v in stats.items()}}
    json.dump(summ, open(os.path.join(a.out, "select_stats.json"), "w"), indent=1)
    log(f"docs {len(docs)}, unique sentence requests {len(reqs)}, sentences {summ['sentences']}, bins {dict(summ['by_bin'])}")


if __name__ == "__main__":
    main()
