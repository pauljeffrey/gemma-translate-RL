# SPDX-License-Identifier: AGPL-3.0-or-later
"""E1 pairwise comparison, long-input check, step 0 (CPU, box): turn the R6 (A2) sentence-by-sentence document translations of the two models into the
prediction-file format of the benchmark pipeline, so that pairwise_surface.py / pairwise_neural.py / pairwise_stats.py run on them unchanged.

    python pairwise_longdoc_prep.py --a2 /workspace/jobs/a2 --a sft --b rl2c --hyp greedy_sent --out /workspace/jobs/e1/longdoc

Reads (read-only): <a2>/work/items.jsonl (source documents + sentence split), <a2>/work/refs_nllb.jsonl (core: NLLB-200 3.3B pseudo-reference, sentence by sentence),
<a2>/work/refs_human.jsonl (SmolDoc human references), <a2>/hyp/<system>/<hyp>.jsonl (the R6 hypotheses; `hyp_sents` is aligned 1:1 with the source sentences).
Writes <out>/seg_<system>.jsonl and <out>/doc_<system>.jsonl in the predictions.jsonl format (fields benchmark, iso, id, english, reference, hyp, finished,
prefix_status, contaminated, pair_seen): seg = one row per aligned sentence (benchmarks core_b64 .. core_b1024, human_sm, human_st512, human_st1024), doc = one
row per document (all bins incl. core_b2048, core_b4096, human_st2048; `finished` = every sentence unit ended with the stop token).
Only documents both systems translated (the R6 subset B8 = hau yor swh amh zul wol kea sag for the SFT model) and only the CLEAN human documents are used.
"""
from __future__ import annotations

import argparse
import json

SEG_BINS_CORE = {64, 256, 512, 1024}
SEG_BINS_HUMAN = {"sm", "st512", "st1024"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2", default="/workspace/jobs/a2")
    ap.add_argument("--a", default="sft")
    ap.add_argument("--b", default="rl2c")
    ap.add_argument("--hyp", default="greedy_sent")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import os

    os.makedirs(a.out, exist_ok=True)
    items = {}
    for l in open(f"{a.a2}/work/items.jsonl", encoding="utf-8"):
        r = json.loads(l)
        items[r["iid"]] = r
    refs = {"core": {}, "human": {}}
    for fn, k in (("refs_nllb.jsonl", "core"), ("refs_human.jsonl", "human")):
        for l in open(f"{a.a2}/work/{fn}", encoding="utf-8"):
            r = json.loads(l)
            refs[k][f"{r['set']}|{r['doc_id']}|{r['iso']}"] = r
    H = {}
    for s in (a.a, a.b):
        H[s] = {}
        for l in open(f"{a.a2}/hyp/{s}/{a.hyp}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            H[s][r["iid"]] = r
    common = sorted(set(H[a.a]) & set(H[a.b]))
    names = {a.a: a.a, a.b: a.b}
    fs = {s: (open(f"{a.out}/seg_{s}.jsonl", "w", encoding="utf-8"), open(f"{a.out}/doc_{s}.jsonl", "w", encoding="utf-8")) for s in names}
    n_doc = n_seg = 0
    for iid in common:
        it = items[iid]
        if it["contam"]:
            continue
        rr = refs[it["set"]].get(iid)
        if rr is None or not (len(rr["ref_sents"]) == len(it["sents"]) == len(H[a.a][iid]["hyp_sents"]) == len(H[a.b][iid]["hyp_sents"])):
            continue
        hb = it["hbin"]
        bname = f"{it['set']}_b{hb}" if it["set"] == "core" else f"human_{hb}"
        seg_ok = (it["set"] == "core" and hb in SEG_BINS_CORE) or (it["set"] == "human" and hb in SEG_BINS_HUMAN)
        n_doc += 1
        for s in names:
            h = H[s][iid]
            base = {"iso": it["iso"], "contaminated": False, "pair_seen": False, "prefix_status": "ok"}
            fs[s][1].write(json.dumps({**base, "benchmark": bname, "id": iid, "english": it["text"], "reference": " ".join(rr["ref_sents"]),
                                       "hyp": " ".join(h["hyp_sents"]), "finished": bool(h["finished"])}, ensure_ascii=False) + "\n")
            if seg_ok:
                for k, (src, ref, hy) in enumerate(zip(it["sents"], rr["ref_sents"], h["hyp_sents"])):
                    fs[s][0].write(json.dumps({**base, "benchmark": bname, "id": f"{iid}#{k}", "english": src, "reference": ref, "hyp": hy, "finished": True},
                                              ensure_ascii=False) + "\n")
                    n_seg += 1
    for s in fs.values():
        for f in s:
            f.close()
    print("documents", n_doc, "segments (per system)", n_seg // 2)


if __name__ == "__main__":
    main()
