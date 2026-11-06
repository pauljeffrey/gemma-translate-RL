#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Editor-RL prompt pool (owner rules 2026-11-03): English source + DRAFT (+ human reference for about 55% of the prompts), 81 RL languages.

    python scripts/post_editor_rl_pool.py prompts --ref-pairs SEL/pairs.jsonl --news-dir NEWS --out DIR [--news-per-language 55]
        -> DIR/prompts.jsonl: pair_id, iso, english, reference ("" = reference-free news prompt), split "train", ref_source, ref_licence, domain, source_dataset, sft_input
    (drafts are then generated for DIR/prompts.jsonl with post_editor_make_drafts.py / post_editor_mt_drafts.py: RL stage A greedy / T0.7 / T1.0 / beam-4, SFT-final, A0, MADLAD-3B; NO NLLB, NO TranslatePsy)
    python scripts/post_editor_rl_pool.py choose --prompts DIR/prompts.jsonl --drafts DRAFTS_DIR --out DIR [--low 0.5 --clean 0.3]
        -> DIR/pool.parquet (the RL prompt file: english, iso, language, src_afr_tokens, source_dataset, length_bin, domain, reference, ref_source, ref_licence, draft, draft_system, draft_qe, draft_category)
           DIR/pool_report.json / pool_by_language.csv

Reference prompts: the human-reference pairs of the editor selection with the rules `exclude every pair E1 was trained on (selection v2 ids)`, no NLLB-mined / synthetic / machine / non-commercial rows, protected
index + benchmark text removed (post_editor_select_data.py). Reference-free prompts: the b64 bin of the news pool (BBC / C4-BBC / Wikinews ..., `rl_prompts_news`), English only.
Draft choice per prompt (deterministic hash): 50% `low` (a candidate whose AfriCOMET-QE is below the language's median QE of the RL greedy drafts), 30% `clean` (the best own-system candidate: do-no-harm),
20% `other` (MADLAD-3B / A0 / SFT-final: another system's error pattern). A draft that equals the reference (chrF++ >= 98) is never chosen; empty / unfinished candidates only as `low` when nothing else exists.
"""
import argparse, csv, glob, json, os, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
OWN = ("rl_greedy", "rl_t07", "rl_t10", "rl_beam", "rl_mbr")
OTHER = ("madlad3b_greedy", "a0_greedy", "sft_greedy", "sft_t07")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def cmd_prompts(a):
    import pandas as pd
    from gemma_mt.post_edit import stable_unit
    rows = []
    for r in read_jsonl(a.ref_pairs):
        if a.isos and r["iso"] not in a.isos: continue
        rows.append({"pair_id": r["pair_id"], "iso": r["iso"], "english": r["english"], "reference": r["reference"], "split": "train", "ref_source": r["source_family"], "ref_licence": "non_commercial" if r.get("non_commercial") else r["licence"],
                     "domain": "science" if r.get("domain") == "science" else "", "source_dataset": r["source_dataset"], "sft_input": r.get("sft_input", True), "cls": r["cls"]})
    have = {(r["iso"], r["english"].strip().casefold()) for r in rows}
    n_news = 0
    for f in sorted(glob.glob(os.path.join(a.news_dir, "*.parquet"))):
        iso = os.path.basename(f)[:-8]
        if a.isos and iso not in a.isos: continue
        d = pd.read_parquet(f)
        if "length_bin" in d.columns: d = d[d["length_bin"] == "b64"]
        d = d[(d["english"].str.split().str.len() >= 6) & (d["english"].str.split().str.len() <= 45)]
        recs = [r for r in d.to_dict("records") if (iso, r["english"].strip().casefold()) not in have]
        recs = sorted(recs, key=lambda r: stable_unit("rlnews", iso, r["english"]))[: a.news_per_language]
        for k, r in enumerate(recs):
            rows.append({"pair_id": f"{iso}-news-{k:04d}-{int(stable_unit('id', iso, r['english']) * 1e9):09d}", "iso": iso, "english": r["english"].strip(), "reference": "", "split": "train", "ref_source": "", "ref_licence": str(r.get("licence", "")),
                         "domain": "news", "source_dataset": str(r.get("source_dataset", "news")), "sft_input": False, "cls": "news"}); n_news += 1
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "prompts.jsonl"), "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ref = sum(bool(r["reference"]) for r in rows)
    print(json.dumps({"prompts": len(rows), "with_reference": n_ref, "reference_free_news": n_news, "ref_share": round(n_ref / len(rows), 3), "languages": len({r['iso'] for r in rows})}))


def cmd_choose(a):
    import numpy as np
    from gemma_mt.languages import LANGUAGES
    import random
    from gemma_mt.post_edit import pick_candidate, sample_desired_chrf, sentence_chrf, stable_hash, stable_unit
    prompts = {r["pair_id"]: r for r in read_jsonl(a.prompts)}
    cands = defaultdict(list)
    for fn in sorted(glob.glob(os.path.join(a.drafts, "drafts_*.jsonl"))):
        for d in read_jsonl(fn):
            if d["pair_id"] in prompts and "nllb" not in d["system"] and "translatepsy" not in d["system"].lower():
                cands[d["pair_id"]].append(d)
    # QE of every candidate draft (AfriCOMET-QE-STL-1.1, raw), cached
    from gemma_mt.rl.comet import QE_MODEL_ID, CometScorer
    import hashlib
    key = lambda e, h: hashlib.sha1(f"{e}\x00{h}".encode()).hexdigest()
    cache_p = os.path.join(a.out, "draft_qe_cache.jsonl"); cache = {}
    if os.path.exists(cache_p):
        for l in open(cache_p): k, v = json.loads(l); cache[k] = v
    todo = {}
    for pid, cs in cands.items():
        for d in cs:
            if d["draft"].strip():
                k = key(prompts[pid]["english"], d["draft"])
                if k not in cache: todo[k] = (prompts[pid]["english"], d["draft"])
    print("QE to score:", len(todo), flush=True)
    if todo:
        sc = CometScorer(QE_MODEL_ID, batch_size=a.batch_size, half=True)
        items = list(todo.items())
        with open(cache_p, "a") as f:
            for i in range(0, len(items), 4000):
                part = items[i:i + 4000]
                out = sc([v[0] for _, v in part], [v[1] for _, v in part], None)
                for (k, _), x in zip(part, out): cache[k] = float(x); f.write(json.dumps([k, float(x)]) + "\n")
                f.flush(); print(f"  {min(i + 4000, len(items))}/{len(items)}", flush=True)
    med = {}
    by_lang = defaultdict(list)
    for pid, cs in cands.items():
        for d in cs:
            if d["system"] == "rl_greedy" and d["draft"].strip(): by_lang[prompts[pid]["iso"]].append(cache[key(prompts[pid]["english"], d["draft"])])
    for iso, v in by_lang.items(): med[iso] = float(np.median(v))
    glob_med = float(np.median([x for v in by_lang.values() for x in v])) if by_lang else 0.0
    chosen, stats = [], defaultdict(int)
    for pid, p in sorted(prompts.items()):
        cs = [d for d in cands.get(pid, []) if d["draft"].strip() and d.get("finished", True)]
        cs = [dict(d, qe=cache[key(p["english"], d["draft"])]) for d in cs]
        if p["reference"]:
            cs = [dict(d, chrf=sentence_chrf(d["draft"], p["reference"])) for d in cs]
            cs = [d for d in cs if d["chrf"] < 98.0]
        if not cs:
            stats["no_candidate"] += 1; continue
        m = med.get(p["iso"], glob_med); u = stable_unit("cat", pid, a.seed)
        own = [d for d in cs if d["system"] in OWN]; oth = [d for d in cs if d["system"] in OTHER]
        # quantile matching (--match-chrf): on prompts with a reference the draft of the category's pool whose chrF++ is nearest to a target drawn from the BENCHMARK drafts' distribution
        # (the same sampler as the editor SFT selection: mean --match-chrf, sd 18); the draft is chosen with the reference, the TARGET of nothing (RL has no target)
        rng = random.Random(stable_hash("match", pid, a.seed)) if (a.match_chrf and p["reference"]) else None
        def choose(pool):
            if rng is not None:
                return pool[pick_candidate([d["chrf"] for d in pool], sample_desired_chrf(None, rng, mean=a.match_chrf, sd=18.0), rng)]
            return pool[int(stable_unit("pick", pid) * len(pool)) % len(pool)]
        if u < a.low:
            cat = "low"; pool = [d for d in cs if d["qe"] < m] or sorted(cs, key=lambda d: d["qe"])[:1]
            pick = choose(pool)
        elif u < a.low + a.clean:
            cat = "clean"
            if rng is not None:
                pool = [d for d in (own or cs) if d["qe"] >= m] or [max(own or cs, key=lambda d: d["qe"])]; pick = choose(pool)
            else:
                pick = max(own or cs, key=lambda d: d["qe"])
        else:
            cat = "other"; pool = oth or own or cs
            pick = choose(pool)
        stats[cat] += 1
        chosen.append({"english": p["english"], "iso": p["iso"], "language": LANGUAGES[p["iso"]].name, "source_dataset": p["source_dataset"], "domain": p["domain"], "reference": p["reference"], "ref_source": p["ref_source"],
                       "ref_licence": p["ref_licence"], "draft": pick["draft"], "draft_system": pick["system"], "draft_qe": round(pick["qe"], 4), "draft_category": cat, "draft_chrf": pick.get("chrf", 0.0) if p["reference"] else None,
                       "lang_median_qe": round(m, 4), "pair_id": pid})
    # AfriCOMET-encoder token count of the source (the loader's src_afr_tokens)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-large-76L")
    for i in range(0, len(chosen), 2000):
        part = chosen[i:i + 2000]
        ids = tok([c["english"] for c in part], add_special_tokens=True, truncation=False)["input_ids"]
        for c, x in zip(part, ids): c["src_afr_tokens"] = len(x)
    chosen = [c for c in chosen if c["src_afr_tokens"] <= 64]
    for c in chosen: c["length_bin"] = "b64"
    import pandas as pd
    pd.DataFrame(chosen).to_parquet(os.path.join(a.out, "pool.parquet"))
    ref = [c for c in chosen if c["reference"]]
    rep = {"prompts": len(chosen), "with_reference": len(ref), "ref_share": round(len(ref) / max(1, len(chosen)), 3), "categories": {k: v for k, v in stats.items()},
           "draft_systems": {s: sum(c["draft_system"] == s for c in chosen) for s in sorted({c["draft_system"] for c in chosen})},
           "mean_draft_chrf_ref_prompts": round(float(np.mean([c["draft_chrf"] for c in ref])), 2) if ref else None, "mean_draft_qe": round(float(np.mean([c["draft_qe"] for c in chosen])), 4),
           "languages": len({c["iso"] for c in chosen}), "religious_share_ref_prompts": None}
    json.dump(rep, open(os.path.join(a.out, "pool_report.json"), "w"), indent=1); print(json.dumps(rep, indent=1))
    with open(os.path.join(a.out, "pool_by_language.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "prompts", "with_reference", "mean_draft_chrf_ref", "mean_draft_qe", "low", "clean", "other"])
        for iso in sorted({c["iso"] for c in chosen}):
            x = [c for c in chosen if c["iso"] == iso]; r_ = [c for c in x if c["reference"]]
            w.writerow([iso, len(x), len(r_), round(float(np.mean([c["draft_chrf"] for c in r_])), 1) if r_ else "", round(float(np.mean([c["draft_qe"] for c in x])), 3)] + [sum(c["draft_category"] == k for c in x) for k in ("low", "clean", "other")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts"); p.add_argument("--ref-pairs", required=True); p.add_argument("--news-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--news-per-language", type=int, default=55); p.add_argument("--isos", nargs="*", default=None)
    c = sub.add_parser("choose"); c.add_argument("--prompts", required=True); c.add_argument("--drafts", required=True); c.add_argument("--out", required=True); c.add_argument("--low", type=float, default=0.5); c.add_argument("--clean", type=float, default=0.3)
    c.add_argument("--seed", type=int, default=0); c.add_argument("--match-chrf", type=float, default=0.0, help="reference prompts: target mean chrF++ of the chosen drafts (benchmark drafts: 30.6); 0 = off"); c.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    {"prompts": cmd_prompts, "choose": cmd_choose}[a.cmd](a)


if __name__ == "__main__":
    main()
