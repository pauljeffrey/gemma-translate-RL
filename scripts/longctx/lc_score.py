"""Per-item metric rows for hypothesis files.  usage: lc_score.py NAME__MODE [NAME__MODE ...]    (hyp/NAME__MODE.jsonl -> sc/NAME__MODE.pkl)
A hyp row: {iid, hyp, finished, new_tokens, ...}.  Metrics per item: chrF++ / spBLEU(flores200) sufficient statistics against every reference system available
('human' for human + sci items, 'nllb' = NLLB-3.3B sentence-by-sentence pseudo-reference for all items), loop, length ratio, source-sentence coverage (plan_document),
GlotLID right-language share, finished, tokens."""
import sys, os, re, json, pickle, collections, functools
from lc_common import *
from gemma_mt.eval.longdoc import join_sentences, norm_text, repetition_loop, plan_document, make_length_model, split_sentences

docs = load_docs()
items = {i["iid"]: i for i in load_items()}
REFS = {"human": load_refs("human"), "nllb": {}, "gemini": load_refs("gemini")}      # gemini: optional (owner-supplied) pseudo-reference
for fn in ("refs_nllb_core.jsonl", "refs_nllb_sent.jsonl"):
    if os.path.exists(f"{WORK}/{fn}"):
        for r in rd(f"{WORK}/{fn}"):
            REFS["nllb"][r["iid"]] = r


@functools.lru_cache(maxsize=1)
def _m():
    from sacrebleu.metrics import BLEU, CHRF
    return CHRF(word_order=2), BLEU(tokenize="flores200")


def stat_vec(h, r):
    import numpy as np
    chrf, bleu = _m()
    h, r = norm_text(h), norm_text(r)
    return (np.array(chrf._extract_corpus_statistics([h], [[r]])[0], dtype=np.float64), np.array(bleu._extract_corpus_statistics([h], [[r]])[0], dtype=np.float64))


_LID = None
def lid():
    global _LID
    if _LID is None:
        from gemma_mt.rl.lid import GlotLIDClassifier
        _LID = GlotLIDClassifier(path="/workspace/sft_build/hf/hub/models--cis-lmu--glotlid/snapshots/85cd6716494360367b75f642b5bc78667605d0b4/model_v3.bin")
    return _LID


def accepted(iso):
    from gemma_mt.languages import LANGUAGES
    return set(LANGUAGES[iso].glotlid)


def lid_share(text, sents, iso, min_chars=25):
    ok = accepted(iso)
    ss = [s for s in (sents or split_sentences(text)) if len(s) >= min_chars]
    L = lid()
    labs = L(ss) if ss else []
    sent_share = (sum(1 for x in labs if x in ok) / len(labs)) if labs else None
    doc_lab = L([text])[0] if text.strip() else "und"
    return sent_share, (doc_lab in ok), doc_lab


LMS = {}
def length_model(iso, setname):
    k = (iso, setname)
    if k not in LMS:
        a, b = [], []
        for iid, it in items.items():
            if it["iso"] == iso and it["set"] == setname:
                rs = (REFS["human"].get(iid) if setname != "core" else None) or REFS["nllb"].get(iid)
                if rs:
                    a.extend(docs[it["doc_id"]]["sents"]); b.extend(rs["ref_sents"])
        lm = make_length_model(a, b)
        rr = sorted(len(r) / max(len(s), 1) for s, r in zip(a, b) if len(s) >= 10)
        LMS[k] = (lm, min(rr[int(0.95 * (len(rr) - 1))], 2.5) if rr else None)
    return LMS[k]


def one(row):
    it = items[row["iid"]]; d = docs[it["doc_id"]]
    hyp = row.get("hyp") or ""
    out = {"iid": row["iid"], "set": it["set"], "bin": it["bin"], "iso": it["iso"], "doc_id": it["doc_id"], "src_tokens": d["src_tokens"], "finished": bool(row.get("finished")),
           "new_tokens": row.get("new_tokens"), "status": row.get("status"), "hyp_chars": len(norm_text(hyp)), "n_sents": len(d["sents"]), "stats": {}, "extra": row.get("extra", {})}
    prim = None
    for sysname in ("human", "nllb", "gemini"):
        rs = REFS[sysname].get(row["iid"])
        if rs is None:
            continue
        rt = join_sentences(rs["ref_sents"], d["para_starts"])
        out["stats"][sysname] = stat_vec(hyp, rt)
        out.setdefault("ref_chars", {})[sysname] = len(norm_text(rt))
        if prim is None:
            prim = (sysname, rs, rt)
    if prim is not None:
        sysname, rs, rt = prim
        out["primary"] = sysname
        hl, rl = repetition_loop(hyp), repetition_loop(rt)
        out["loop"] = bool(hl["loop"]) and not rl["loop"]
        out["loop_tail"] = out["loop"] and bool(hl["at_tail"])
        out["length_ratio"] = len(norm_text(hyp)) / max(len(norm_text(rt)), 1)
        lm, w_hi = length_model(it["iso"], it["set"])
        pl = plan_document(d["sents"], rs["ref_sents"], hyp, lm, w_hi=w_hi)
        out.update({"cov_chars": pl.coverage_chars, "cov_sents": pl.coverage_sents, "omitted": pl.omitted_sents, "copied": pl.copied_sents, "short_beads": pl.short_beads,
                    "extra_share": pl.extra_share, "n_hyp_sents": pl.n_hyp})
    out["lid_sent"], out["lid_doc_ok"], out["lid_doc_label"] = lid_share(hyp, row.get("hyp_sents"), it["iso"])
    return out


def main():
    import multiprocessing as mp
    for name in sys.argv[1:]:
        rows = [r for r in rd(f"{HYP}/{name}.jsonl") if r["iid"] in items]
        for r in rows:
            r.setdefault("hyp_sents", None)
        ctx = mp.get_context("fork")
        with ctx.Pool(6) as pool:
            res = pool.map(one, rows, chunksize=8)
        os.makedirs(SC, exist_ok=True)
        pickle.dump(res, open(f"{SC}/{name}.pkl", "wb"))
        print("scored", name, len(res), "refs: human", len(REFS["human"]), "nllb", len(REFS["nllb"]), flush=True)


if __name__ == "__main__":
    main()
