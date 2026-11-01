#!/usr/bin/env python3
"""(a) AfriScience-MT science windows from S3's FILTERED train pairs (agent C2, 2026-10-28).

    python s4_afs_windows.py --afs-train afriscience_clean_train.parquet --out OUT [--passes 2] [--multi-parallel pair|any] [--gap-policy join|split]
                             [--tokenizer google/gemma-3-270m-it] [--dev-from-hub | --dev-parquet FILE]

Recipe = scripts/build_sft_science_mix.py of the A2W stage (reports/sft_science_stage_2026-10-21.md): every paper is cut once per PASS into windows of one granularity
(S single sentences, A 2-4, B 5-12, W the whole paper; weights .35/.25/.20/.20; a paper never gets the same granularity twice), the same cut for all six languages, windows never cross
a paper and are joined by single spaces. Differences forced by the clean data (S3: the pairs were filtered against the protected index, so the six languages no longer have identical
sentence sets):

  --multi-parallel pair   (DEFAULT) every language keeps the sentences that survived in ITS OWN pair set; the cut is made on the union of the sentence ids of the paper, so the
                          six languages still share their window boundaries.
  --multi-parallel any    an English sentence that was removed in ANY language is removed in ALL six (the six languages stay strictly parallel; costs 486 mixed + 30 all-removed sentences
                          of 5,792, S3 report sec. 4).
  --gap-policy join       (DEFAULT, as the old builder) sentences removed by the filter are simply left out of the window text.
  --gap-policy split      a window is split wherever the kept sentence ids of a language are not consecutive (no stitched-over gap; shorter windows).
  --index/--manifest      the strict protected-index check at WINDOW level (3 of 28,936 windows of the full run hit a protected 8-gram that spans a sentence join and are dropped).
  --keep-len-outliers     by default the pairs flagged `flag_len_ratio_outlier` (target/source length ratio outside 0.3-3 x the language median, as in the old builder) are dropped.

Output: OUT/science_windows.jsonl (train rows with provenance + licence columns), OUT/val_sci.jsonl (AfriScience-MT DEV, sentence level + whole papers: the progress MONITOR of the stage;
it is protected text by design and is never part of the training mixture), OUT/science_windows_stats.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s4_common as K  # noqa: E402

MODES = ("S", "A", "B", "W")
MODE_WEIGHTS = {"S": 0.35, "A": 0.25, "B": 0.20, "W": 0.20}
REPO = "dsfsi/afriscience_mt"


def paper_modes(pid, passes: int, seed: int) -> List[str]:
    """One window granularity per pass for a paper, never the same twice while there are other granularities left (the old builder's function)."""
    rng = random.Random(f"{seed}|{pid}|modes")
    left, out = list(MODES), []
    for _ in range(passes):
        if not left:
            left = list(MODES)
        m = rng.choices(left, weights=[MODE_WEIGHTS[x] for x in left])[0]
        out.append(m)
        left.remove(m)
    return out


def segment(n: int, mode: str, rng: random.Random) -> List[Tuple[int, int]]:
    """Index ranges [a, b) that cut a paper of n sentences into consecutive windows (every sentence in exactly one window)."""
    if n <= 0:
        return []
    if mode == "W":
        return [(0, n)]
    lo, hi = {"S": (1, 1), "A": (2, 4), "B": (5, 12)}[mode]
    out, i = [], 0
    while i < n:
        j = min(n, i + rng.randint(lo, hi))
        out.append((i, j))
        i = j
    return out


def build_windows(rows: List[dict], passes: int = 2, seed: int = 20261021, multi_parallel: str = "pair", gap_policy: str = "join",
                  drop_len_outliers: bool = True) -> Tuple[List[dict], dict]:
    """rows: dicts with paper_id, sentence_id, domain, target_lang, source, target [, flag_len_ratio_outlier, licence, prov_*]. -> (windows, stats)."""
    assert multi_parallel in ("pair", "any") and gap_policy in ("join", "split")
    stats = {"dropped_len_outlier": collections.Counter(), "dropped_any_language": 0, "windows": collections.Counter(), "sentences_in_windows": collections.Counter(),
             "gap_splits": 0, "mixed_outcome_sentences": 0}
    langs = sorted({r["target_lang"] for r in rows})
    by = collections.defaultdict(dict)                      # (paper, lang) -> {sentence_id: row}
    for r in rows:
        if drop_len_outliers and r.get("flag_len_ratio_outlier"):
            stats["dropped_len_outlier"][r["target_lang"]] += 1
            continue
        by[(r["paper_id"], r["target_lang"])][int(r["sentence_id"])] = r
    papers = sorted({k[0] for k in by})
    # per paper: ids present per language
    out: List[dict] = []
    for pid in papers:
        ids_by_lang = {l: set(by.get((pid, l), {})) for l in langs}
        union = set().union(*ids_by_lang.values())
        common = set.intersection(*[s for s in ids_by_lang.values()]) if all(ids_by_lang.values()) else set()
        stats["mixed_outcome_sentences"] += len(union - common)
        if multi_parallel == "any":
            stats["dropped_any_language"] += len(union - common)
            axis = sorted(common)
        else:
            axis = sorted(union)
        modes = paper_modes(pid, passes, seed)
        for p in range(passes):
            rng = random.Random(f"{seed}|{pid}|{p}|cut")           # the same cut for the six languages of a paper
            for a, b in segment(len(axis), modes[p], rng):
                span = axis[a:b]
                for l in langs:
                    kept = [sid for sid in span if sid in by.get((pid, l), {})]
                    if not kept:
                        continue
                    runs = [kept]
                    if gap_policy == "split":
                        runs, cur = [], [kept[0]]
                        for s in kept[1:]:
                            if s == cur[-1] + 1:
                                cur.append(s)
                            else:
                                runs.append(cur)
                                cur = [s]
                                stats["gap_splits"] += 1
                        runs.append(cur)
                    for run in runs:
                        rr = [by[(pid, l)][s] for s in run]
                        r0 = rr[0]
                        out.append({"english": " ".join(K.nfc(x["source"]).strip() for x in rr), "target": " ".join(K.nfc(x["target"]).strip() for x in rr), "iso": l,
                                    "paper": int(pid), "domain": r0.get("domain", ""), "pass": p, "mode": modes[p], "n_sent": len(rr), "sentence_ids": [run[0], run[-1]],
                                    **K.provenance("science", source_dataset="dsfsi/afriscience_mt:train", licence=r0.get("licence", "apache-2.0"),
                                                   prov_repo=r0.get("prov_repo", ""), prov_ref=f"paper={pid};sentences={run[0]}-{run[-1]};pass={p};mode={modes[p]}",
                                                   prov_index=r0.get("prov_index", ""), generator="human reference (AfriScience-MT)")})
                        stats["windows"][f"{l}/{modes[p]}"] += 1
                        stats["sentences_in_windows"][l] += len(rr)
    stats = {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in stats.items()}
    stats["languages"] = langs
    return out, stats


def dev_rows(dev: List[dict]) -> List[dict]:
    """AfriScience-MT DEV in the two evaluation settings: every sentence on its own + every whole paper (sentences joined by single spaces). MONITOR only."""
    by = collections.defaultdict(list)
    for r in sorted(dev, key=lambda r: (r["target_lang"], r["paper_id"], r["sentence_id"])):
        by[(r["target_lang"], r["paper_id"])].append(r)
    out = []
    for (l, pid), rr in by.items():
        for r in rr:
            out.append({"english": K.nfc(r["source"]).strip(), "target": K.nfc(r["target"]).strip(), "iso": l, "paper": int(pid), "mode": "S", "n_sent": 1})
        out.append({"english": " ".join(K.nfc(r["source"]).strip() for r in rr), "target": " ".join(K.nfc(r["target"]).strip() for r in rr), "iso": l, "paper": int(pid), "mode": "W",
                    "n_sent": len(rr)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--afs-train", required=True, help="afriscience_clean_train.parquet of the prompts repo (S3 output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--multi-parallel", choices=("pair", "any"), default="pair")
    ap.add_argument("--gap-policy", choices=("join", "split"), default="join")
    ap.add_argument("--keep-len-outliers", action="store_true")
    ap.add_argument("--tokenizer", default="google/gemma-3-270m-it", help="only used for the token statistics and the pack-length check")
    ap.add_argument("--seq-len", type=int, default=K.SEQ_LEN)
    ap.add_argument("--dev-parquet", default="")
    ap.add_argument("--dev-from-hub", action="store_true", help="download corpus/dev-00000.parquet of dsfsi/afriscience_mt (HF_TOKEN) for the val_sci monitor")
    ap.add_argument("--limit-papers", type=int, default=0, help="testing: only the first N papers")
    ap.add_argument("--index", default="", help="protected index (.npz): windows are checked once more at WINDOW level (a window joins sentences, so an 8-gram can span the join) and a window with any hit is dropped")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--nproc", type=int, default=2)
    a = ap.parse_args(argv)
    import pyarrow.parquet as pq

    t = pq.read_table(a.afs_train).to_pylist()
    if a.limit_papers:
        keep = sorted({r["paper_id"] for r in t})[:a.limit_papers]
        t = [r for r in t if r["paper_id"] in keep]
    K.log(f"{len(t)} clean train pairs, {len({r['paper_id'] for r in t})} papers")
    wins, stats = build_windows(t, a.passes, a.seed, a.multi_parallel, a.gap_policy, not a.keep_len_outliers)
    window_hits = collections.Counter()
    if a.index:
        P = K.Protected(a.index, a.manifest, a.nproc)
        hh = P.names([w["english"] for w in wins] + [w["target"] for w in wins])
        keep_w = []
        for k, w in enumerate(wins):
            h = hh[k] + hh[len(wins) + k]
            if h:
                window_hits["windows"] += 1
                for nm in set(h):
                    window_hits[nm] += 1
                continue
            keep_w.append(w)
        wins = keep_w
        P.close()
        K.log(f"window-level protected-index check ({P.version}): {window_hits['windows']} windows dropped")
    tok = K.tokenizer(a.tokenizer)
    toks, long_, rows = collections.Counter(), 0, []
    tgt_toks = collections.Counter()
    for w in wins:
        ex = K.example(tok, w["english"], w["iso"], w["target"])
        if ex.n_tokens > a.seq_len:
            long_ += 1
            continue
        w["n_tokens"], w["target_tokens"] = ex.n_tokens, ex.target_len
        toks[w["iso"]] += ex.n_tokens
        tgt_toks[w["iso"]] += ex.target_len
        rows.append(w)
    K.write_jsonl(os.path.join(a.out, "science_windows.jsonl"), rows)
    stats.update({"options": {"passes": a.passes, "seed": a.seed, "multi_parallel": a.multi_parallel, "gap_policy": a.gap_policy, "drop_len_outliers": not a.keep_len_outliers},
                  "windows_total": len(rows), "windows_longer_than_pack": long_, "tokens_total": sum(toks.values()), "tokens_by_lang": dict(toks), "target_tokens_by_lang": dict(tgt_toks),
                  "a2w_reference_tokens": K.A2W_TOKENS["science"], "a2w_reference_windows": 29451, "licence_note": K.AFRISCIENCE_NOTE, "window_level_protected_hits": dict(window_hits), "index_version": (P.version if a.index else None)})
    # monitor split
    dev = []
    if a.dev_parquet:
        dev = pq.read_table(a.dev_parquet).to_pylist()
    elif a.dev_from_hub:
        from huggingface_hub import HfApi, hf_hub_download

        tok_ = os.environ["HF_TOKEN"]
        info = HfApi(token=tok_).dataset_info(REPO)
        p = hf_hub_download(REPO, "corpus/dev-00000.parquet", repo_type="dataset", token=tok_, local_dir=os.path.join(a.out, "_dev"), revision=info.sha)
        dev = pq.read_table(p).to_pylist()
        stats["dev_revision"] = info.sha
    if dev:
        dr = dev_rows(dev)
        keep = []
        for r in dr:
            ex = K.example(tok, r["english"], r["iso"], r["target"])
            if ex.n_tokens <= a.seq_len:
                keep.append(r)
        K.write_jsonl(os.path.join(a.out, "val_sci.jsonl"), keep)
        stats["val_sci_rows"] = len(keep)
    json.dump(stats, open(os.path.join(a.out, "science_windows_stats.json"), "w"), indent=1)
    K.log(f"science windows: {len(rows)} ({long_} longer than {a.seq_len} tokens dropped), {sum(toks.values()):,} tokens over {a.passes} passes "
          f"(A2W reference {K.A2W_TOKENS['science']:,} tokens / 29,451 windows), mode {a.multi_parallel}/{a.gap_policy}")


if __name__ == "__main__":
    main()
