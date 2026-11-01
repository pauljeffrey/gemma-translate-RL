#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""H1 step 4: long-document indicators of the final run's reward (long-document reward v2, anchors v3 at document level) on STORED outputs.

  L1  the long-context benchmark (12 languages, 290 documents, box_bulk tarball lc1/): SFT-final, RL-final, A2W, the final run at step 250 ("live") and final2, whole-document mode
      and the sentence-by-sentence wrapper with repair ("sentfix"). Per document: finished, repetition severity of reward v2 (calibration/longdoc_v2.json, excess over the English source),
      share of the source (characters) with no output after the anchored alignment (coverage, as in the reward), length ratio, padding / stop-word deficits (anchors v3, segmental
      type-token ratio), substring copy, number loss, GlotLID of the document and of its sentences.
  L2  the 77-language final2 long-document set (Hub dataset, `sentfix`, 414 documents per language): the same indicators, absolute (no RL-final outputs exist for this set).
  L3  AfriScience-MT whole papers (dev and test; RL-final, A2W, final2): the same indicators.
Outputs in $HE_OUT: longdoc_items.parquet is written to $HE_WORK (per document indicators, no text); CSV tables in $HE_OUT.
"""
import argparse
import gzip
import json
import os
import re
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hardening_evidence_common as C  # noqa: E402
import hardening_evidence_detectors as D  # noqa: E402
from gemma_mt.rl import anchors as AN  # noqa: E402
from gemma_mt.rl import longdoc as LD  # noqa: E402
from gemma_mt.rl.anchors_v2 import pad_ttr  # noqa: E402
from gemma_mt.rl.chunk_plan import plan_chunks  # noqa: E402
from gemma_mt.rl.reward_shaping import ContentShaper  # noqa: E402
from gemma_mt.rl.segments import split_sentences  # noqa: E402

warnings.filterwarnings("ignore")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_STATE = {}


def init():
    A = D.init()
    _STATE["A"] = A
    _STATE["cal"] = LD.LongdocCalibration.from_file(os.path.join(C.CAL, "longdoc_v2.json"))
    _STATE["shaper"] = ContentShaper.from_file(os.path.join(C.CAL, "shaping_flores_dev_v8.json"), reliability_from="same_source", mode="calibrated", preset="repo", neg_scale=2.0,
                                               default_anchors=(0.46, 0.6, 0.75), fail_floor=0.0)


def doc_row(args):
    """One (source document, output document) pair -> indicator dict. Mirrors the final run's reward: v2 repetition severity (excess over the source), anchored alignment with the
    language-aware length model and shortfall check (omission_short_frac 0.7, min 0.15), coverage = failed share of the source weight."""
    iso, src, hyp, hyp_sents = args
    A, cal, sh = _STATE["A"], _STATE["cal"], _STATE["shaper"]
    o = {}
    h = hyp or ""
    o["empty"] = float(not h.strip())
    src_words = len(src.split())
    o["src_words"] = src_words
    o["len_ratio"] = len(h.strip()) / max(len(src.strip()), 1)
    prof = LD.repetition_profile(h)
    sprof = LD.repetition_profile(src)
    sev, lvl = cal.severity(prof, src_words, sprof)
    o["rep_sev"], o["rep_level"] = sev, lvl
    for k, v in prof.levels.items():
        o["rep_" + k] = v
    o["rep_flag"] = float(sev > 0)
    o["rep_severe"] = float(sev >= 0.5)
    o["rep_degenerate"] = float(sev >= 1.0)
    # coverage (reward v2): anchored alignment, language-aware length model
    w_hi = (sh.len_window.get(iso) or (None, None))[1]
    lm = sh.length_model(iso, None, None)
    om = {}
    if lm is not None:
        om = {"length_model": lm, "short_frac": 0.7, "short_min": 0.15}
    try:
        cp = plan_chunks(src, h, chunking="sentences", chunk_align="gale_church", chunk_agg="source_length", extra_output_penalty=True, extra_ratio_limit=1.8,
                         length_slack=0.15, copy_jaccard=0.9, w_hi=w_hi, beads_fn=LD.beads_fn(lm, None), **{k: v for k, v in om.items() if k != "beads_fn"})
        o["cover_failed"] = LD.coverage_severity(cp.items, 1.0)
        o["n_failed_sent"] = cp.n_failed
        o["n_short"] = cp.n_short
        o["align_how"] = cp.how
        o["unmatched_out"] = float("unmatched_output" in cp.extra)
        o["appended"] = float("appended_text" in cp.extra)
        o["omit_or_copy_chunks"] = float(sum(1 for e in cp.extra if e == "omitted_or_copied"))
    except Exception as e:                       # never let one document stop the run
        o["cover_failed"] = float("nan")
        o["align_how"] = "error:" + type(e).__name__
    # anchors v3 on the document (type-token ratio is length independent by construction: ttr_mode seg100)
    ow, sw = AN.words(h), AN.words(src)
    f = A.func_deficit(iso, sw, ow)
    o["func_def"] = float(f) if f is not None else float("nan")
    p = A.pad_deficit(iso, ow)
    o["pad_def"] = float(p) if p is not None else float("nan")
    o["pad_flag"] = float(p > 0) if p is not None else float("nan")
    o["ttr_seg"] = float(pad_ttr(ow, "seg100")) if len(ow) >= 8 else float("nan")
    raw, fl, j = D.copy_raw(A, iso, src, h) if h.strip() else (float("nan"), float("nan"), 0.0)
    o["copy_frac"] = raw
    o["copy_flag"] = float(fl > 0) if j else float("nan")
    ns = [re.sub(r"[.,]", "", m.group()) for m in _NUM.finditer(src)]
    if ns:
        tn = {re.sub(r"[.,]", "", m.group()) for m in _NUM.finditer(h)}
        o["num_loss"] = 1.0 - sum(1 for n in ns if n in tn) / len(ns)
    else:
        o["num_loss"] = float("nan")
    o["n_sent_out"] = len(hyp_sents) if hyp_sents is not None else len(split_sentences(h))
    return o


def run_pool(items, procs):
    out = []
    t0 = time.time()
    with Pool(procs, initializer=init) as pool:
        for k, r in enumerate(pool.imap(doc_row, items, chunksize=8)):
            out.append(r)
            if k % 2000 == 0:
                print(f"  {k}/{len(items)} {round(time.time() - t0)}s", flush=True)
    return pd.DataFrame(out)


def add_lid(df, texts, isos, sents_list):
    lid = D.Lid()
    top = lid.top1([t[:1000] for t in texts])
    res = [lid.verdict(i, t, tp) for i, t, tp in zip(isos, texts, top)]
    df["lid_doc_judged"] = [r[0] for r in res]
    df["lid_doc_wrong"] = [r[1] for r in res]
    # sentence level
    uniq = list(dict.fromkeys(s for ss in sents_list for s in ss))
    tops = dict(zip(uniq, lid.top1(uniq)))
    share = []
    for iso, ss in zip(isos, sents_list):
        v = [lid.verdict(iso, s, tops[s]) for s in ss]
        j = [x for x in v if x[0] > 0]
        share.append(float(np.mean([x[1] for x in j])) if j else float("nan"))
    df["lid_sent_wrong_share"] = share


def load_lc1():
    tar = C.ensure_tar_members(["lc1/hyp/", "lc1/work/"])
    items = pd.DataFrame(C.read_jsonl(os.path.join(tar, "lc1", "work", "items.jsonl")))
    docs = {d["doc_id"]: d for d in C.read_jsonl(os.path.join(tar, "lc1", "work", "docs.jsonl"))}
    rows = []
    for sysn in ("sft", "rl", "a2w", "live", "final2"):
        for mode in ("whole", "sentfix"):
            p = os.path.join(tar, "lc1", "hyp", f"{sysn}__{mode}.jsonl")
            if not os.path.exists(p):
                continue
            for r in C.read_jsonl(p):
                rows.append(dict(system=sysn, mode=mode, iid=r["iid"], hyp=r.get("hyp") or "", finished=str(r.get("finished")).lower() == "true", new_tokens=r.get("new_tokens"),
                                 status=r.get("status"), hyp_sents=r.get("hyp_sents")))
    h = pd.DataFrame(rows).merge(items, on="iid", how="left")
    h["src"] = h["doc_id"].map(lambda d: docs[d]["text"])
    h["hyp_sents"] = h["hyp_sents"].map(lambda s: json.loads(s) if isinstance(s, str) and s.startswith("[") else (s if isinstance(s, list) else None))
    return h


def length_group(b):
    if b in ("b256", "b512", "sm", "st512"):
        return "short (<= ~512 tokens)"
    if b in ("b1024", "st1024"):
        return "~1,024 tokens"
    if b in ("b2048", "st2048", "b4096"):
        return ">= ~2,048 tokens"
    if str(b).startswith("sci"):
        return "AfriScience papers"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--parts", default="123")
    a = ap.parse_args()
    if "1" in a.parts:
        h = load_lc1()
        h = h[h.iso.isin(D.rl77())].reset_index(drop=True)
        print("lc1 items", len(h), h.groupby(["system", "mode"]).size().to_dict(), flush=True)
        items = list(zip(h.iso, h.src, h.hyp, h.hyp_sents))
        r = run_pool(items, a.procs)
        sents = [s if s is not None else split_sentences(t) for s, t in zip(h.hyp_sents, h.hyp)]
        add_lid(r, h.hyp.tolist(), h.iso.tolist(), sents)
        for c in ("system", "mode", "iid", "doc_id", "iso", "set", "bin", "finished", "new_tokens", "status"):
            r[c] = h[c].values
        r["unfinished"] = (~r["finished"].astype(bool)).astype(float)
        r["length_group"] = r["bin"].map(length_group)
        r.to_parquet(os.path.join(C.WORK, "longdoc_lc1.parquet"))
        print("lc1 done", len(r), flush=True)
    if "2" in a.parts:
        hub = C.ensure_hub(["long_docs/data/*"])
        parts = []
        for f in sorted(os.listdir(os.path.join(hub, "long_docs", "data"))):
            if f.endswith(".parquet"):
                parts.append(pd.read_parquet(os.path.join(hub, "long_docs", "data", f)))
        hd = pd.concat(parts, ignore_index=True)
        hd = hd[hd["source_en_included"].astype(bool)].reset_index(drop=True)
        print("hub long docs with source text", len(hd), hd.language.nunique(), flush=True)
        items = list(zip(hd.language, hd.source_en, hd.translation, [list(x) for x in hd.translation_sentences]))
        r = run_pool(items, a.procs)
        add_lid(r, hd.translation.tolist(), hd.language.tolist(), [list(x) for x in hd.translation_sentences])
        for c in ("language", "doc_id", "length_bin", "domain", "finished", "n_units", "n_unfinished_units", "n_repaired_units", "n_sentences", "n_looping_sentences", "n_copied_sentences",
                  "n_empty_sentences", "repetition_flag", "lid_ok", "length_ratio", "afri_qe", "ssa_qe"):
            r["hub_" + c] = hd[c].values
        r["iso"] = hd.language.values
        r.to_parquet(os.path.join(C.WORK, "longdoc_hub77.parquet"))
        print("hub77 done", len(r), flush=True)
    if "3" in a.parts:
        hub = C.ensure_hub(["eval_benchmarks/*"])
        rows = []
        srcs = {"dev": {"rl": os.path.join(C.JOBS, "a4", "eval_final1433", "predictions_dev.json.gz"), "a2w": os.path.join(C.JOBS, "r7", "eval", "A2W", "alpha_0.70", "afs_predictions_dev.json.gz"),
                        "final2": os.path.join(C.JOBS, "a4", "eval_final2_step401", "predictions_dev.json.gz")},
                "test": {"rl": os.path.join(C.JOBS, "a4", "eval_final1433", "predictions_test.json.gz"), "a2w": os.path.join(C.JOBS, "r7", "eval", "A2W_a0.7_test", "afs_predictions_test.json.gz"),
                         "final2": os.path.join(C.JOBS, "a4", "eval_final2_step401", "predictions_test.json.gz")}}
        for split, m in srcs.items():
            t = pd.read_parquet(os.path.join(hub, "eval_benchmarks", f"afriscience_mt_{split}_text.parquet"))
            t = t[t["level"] == "document"]
            for sysname, path in m.items():
                d = json.load(gzip.open(path, "rt", encoding="utf-8"))["document"]
                for iso, lst in d.items():
                    tt = t[t["language"] == iso].reset_index(drop=True)
                    assert len(tt) == len(lst), (split, sysname, iso, len(tt), len(lst))
                    for (_, row), e in zip(tt.iterrows(), lst):
                        rows.append(dict(system=sysname, split=split, iso=iso, paper=e["paper"], src=row["source_en"], hyp=e["hyp"], finished=bool(e["finished"])))
        h = pd.DataFrame(rows)
        h = h[h.iso.isin(D.rl77())].reset_index(drop=True)
        r = run_pool(list(zip(h.iso, h.src, h.hyp, [None] * len(h))), a.procs)
        add_lid(r, h.hyp.tolist(), h.iso.tolist(), [split_sentences(t) for t in h.hyp])
        for c in ("system", "split", "iso", "paper", "finished"):
            r[c] = h[c].values
        r["unfinished"] = (~r["finished"].astype(bool)).astype(float)
        r.to_parquet(os.path.join(C.WORK, "longdoc_afs_papers.parquet"))
        print("afs papers done", len(r), flush=True)


if __name__ == "__main__":
    main()
