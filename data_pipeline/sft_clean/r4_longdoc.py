#!/usr/bin/env python3
"""r4: repetition limits of the long-document reward v2 (calibration/longdoc_v2.json -> clean_2026-10-28/longdoc_clean_v1.json), human text only, no benchmark outside H_cal.

Old fit (scripts/longdoc_v2/calibrate_repetition.py): SmolDoc references stitched (45 languages) + NLLB-3.3B references of the AfriLongDoc core set. SmolDoc and AfriLongDoc are evaluation sets: out.
New human documents (the NLLB part was only a reported test kind, never a source of the limits, and is dropped):
  (a) H_cal pairs (v2 split, calibration label asserted) grouped by FLORES+ article, stitched in random article order to the median source words of each bin (26 / 110 / 405 / 786 / 1579 / 3141),
      per language `--per-cell` documents per bin, the 45 languages with H_cal references (a bin can reuse articles after one pass: H_cal has ~6k words per language);
  (b) clean AfriScience-MT TRAIN papers (agent S3, protected-index filtered), contiguous sentence windows of the target length, 6 languages: the only long (>1,500 words) human text.
Limit per bucket = (quantile q of the EXCESS over the source's own repetition) * margin, floored by rl.longdoc.LEVEL_FLOOR, exactly as before. Buckets 4-5 (b2048/b4096) are not used by the clean run
(bins stop at 1,024 source tokens) but are filled from (b) so the file keeps its shape.
    python r4_longdoc.py --cal refs/cal.jsonl --split heldout_split_cal_sel_v2.json --afs afriscience_clean_train.parquet --code <repo gemma_mt/> --out longdoc_clean_v1.json --report longdoc_clean_report.json
`tail_gap` (needs QE-ensemble chunk rewards of the human documents) is NOT computed here: it is copied as null and listed as GPU work.
"""
import argparse, bisect, collections, json, os, random, sys, unicodedata

BIN_WORDS = {64: 26, 256: 110, 512: 405, 1024: 786, 2048: 1579, 4096: 3141}


def q(vals, p):
    v = sorted(vals)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def stitched(pool, tw, n, seed):
    rng = random.Random(seed)
    order = list(range(len(pool))); rng.shuffle(order); pos = 0; docs = []
    for _ in range(n):
        s, r, w = [], [], 0
        while w < tw:
            if pos >= len(order):
                rng.shuffle(order); pos = 0
            _, ss, rr = pool[order[pos]]; pos += 1
            s += ss; r += rr; w += sum(len(x.split()) for x in ss)
        docs.append((s, r))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", required=True); ap.add_argument("--split", required=True); ap.add_argument("--afs", default=None, help="clean AfriScience train parquet; OMIT for FLORES-only limits (owner decision 2026-10-29)"); ap.add_argument("--code", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--report", required=True)
    ap.add_argument("--per-cell", type=int, default=6); ap.add_argument("--quantile", type=float, default=0.995); ap.add_argument("--margin", type=float, default=1.15)
    ap.add_argument("--min-articles", type=int, default=100); ap.add_argument("--afs-per-paper", type=int, default=2)
    a = ap.parse_args()
    sys.path.insert(0, a.code); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from r4_qe_score import load_rows
    from gemma_mt.rl import longdoc as LD
    edges = LD.DEFAULT_BUCKET_EDGES
    sp = json.load(open(a.split)); art_of = {int(k): v for k, v in sp["article_index_of_id"]["map"].items()}
    rows = load_rows(a.cal, "calibration", a.split)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["iso"]][art_of[r["id"]]].append((r["id"], " ".join(unicodedata.normalize("NFC", r["english"]).split()), " ".join(unicodedata.normalize("NFC", r["reference"]).split())))
    data = []        # (bucket, kind, iso, prof, n_words, src_words, bin, src_prof)

    def add(iso, kind, s, r, b):
        text = " ".join(r); sw = sum(len(x.split()) for x in s)
        prof = LD.repetition_profile(text)
        data.append((bisect.bisect_left(list(edges), sw), kind, iso, prof, prof.n_words, sw, b, LD.repetition_profile(" ".join(s))))

    n_h = 0
    for iso, arts in sorted(by.items()):
        if len(arts) < a.min_articles:
            continue
        pool = [(k, [x[1] for x in sorted(v)], [x[2] for x in sorted(v)]) for k, v in sorted(arts.items())]
        for b, tw in BIN_WORDS.items():
            if tw > 1000:
                continue
            for s, r in stitched(pool, tw, a.per_cell, f"7-{iso}-{tw}"):
                add(iso, "human_flores_cal", s, r, b); n_h += 1
    if a.afs:
        import pyarrow.parquet as pq
    t = pq.read_table(a.afs, columns=["paper_id", "sentence_id", "target_lang", "source", "target", "contiguous_with_previous_kept"]).to_pydict() if a.afs else None
    papers = collections.defaultdict(list)
    for i in range(len(t["source"]) if t else 0):
        papers[(t["target_lang"][i], t["paper_id"][i])].append((int(t["sentence_id"][i]), " ".join(t["source"][i].split()), " ".join(t["target"][i].split())))
    n_a = 0
    for (iso, pid), v in sorted(papers.items()):
        v.sort()
        for b, tw in BIN_WORDS.items():
            used = 0
            for _ in range(a.afs_per_paper):
                s, r, w = [], [], 0
                while used < len(v) and w < tw:
                    s.append(v[used][1]); r.append(v[used][2]); w += len(v[used][1].split()); used += 1
                if w >= tw * 0.95:
                    add(iso, "human_afriscience_train", s, r, b); n_a += 1
    print("human docs: flores_cal", n_h, "afriscience_train", n_a, flush=True)
    levels = list(LD.LEVELS); rep = {"n_docs": {"flores_cal": n_h, "afriscience_train": n_a}, "buckets": []}
    exc = lambda r, lv: max(0.0, r[3].levels[lv] - r[7].levels[lv])
    t0 = []
    for bk in range(len(edges) + 1):
        sub = [r for r in data if r[0] == bk]
        nearest = False
        if not sub:
            have = {x[0] for x in data}; sub = [r for r in data if r[0] == min(have, key=lambda b: abs(b - bk))]; nearest = True
        lim = {lv: round(max(q([exc(r, lv) for r in sub], a.quantile) * a.margin, LD.LEVEL_FLOOR[lv]), 4) for lv in levels}
        t0.append(lim)
        rep["buckets"].append({"bucket": bk, "n": len(sub), "nearest_bucket_used": nearest, "kinds": dict(collections.Counter(r[1] for r in sub)),
                               "p995_excess": {lv: q([exc(r, lv) for r in sub], .995) for lv in levels}, "limit": lim,
                               "limit_by_kind": {k: {lv: round(max(q([exc(r, lv) for r in sub if r[1] == k], a.quantile) * a.margin, LD.LEVEL_FLOOR[lv]), 4) for lv in levels}
                                                 for k in sorted({r[1] for r in sub})}})
    cal = {"edges": list(edges), "t0": t0, "t1": LD.DEFAULT_T1, "tail_gap": None,
           "meta": {"quantile": a.quantile, "margin": a.margin, "n_docs": len(data), "split": "calibration (H_cal v2) + AfriScience-MT train (clean)",
                    "source": "H_cal FLORES+ pairs stitched by article (45 languages) + clean AfriScience-MT train paper windows (6 languages); no NLLB references, no SmolDoc, no AfriLongDoc",
                    "tail_gap": "NOT FITTED: needs QE-ensemble chunk rewards of the human documents (GPU), see the report"}}
    c = LD.LongdocCalibration(tuple(edges), t0, dict(LD.DEFAULT_T1))
    fp = collections.defaultdict(lambda: [0, 0, 0])
    for bk, kind, iso, prof, nw, sw, b, sp_ in data:
        s_, _ = c.severity(prof, sw, sp_)
        for key in ((bk, kind), (bk, "all")):
            fp[key][0] += 1; fp[key][1] += s_ > 0; fp[key][2] += s_ >= 0.5
    rep["fp_in_sample"] = {f"{k[0]}:{k[1]}": {"n": v[0], "any": v[1] / v[0], "ge_half": v[2] / v[0]} for k, v in sorted(fp.items())}
    json.dump(cal, open(a.out, "w"), indent=1); json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps(t0)); print(json.dumps(rep["fp_in_sample"]))


if __name__ == "__main__":
    main()
