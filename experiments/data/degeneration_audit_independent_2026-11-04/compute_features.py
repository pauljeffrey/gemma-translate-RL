"""E-08 step 1: per-row features for the independent degeneration audit.

For every prediction row (benchmark flores_plus / ntrex / mafand; one file per model) compute
  reward-family flags   : guards.is_repetitive, guards.is_self_repetitive, is_copy_of_source, length_ratio_bad, has_role_tail
                          (gemma_mt/gemma_mt/rl/guards.py = the reward's guards; identical in the box tree)
  independent detectors : lzma compression ratio and its residual against a reference-calibrated expectation of the same
                          benchmark/language cell, repeated char-8-gram share, repeated word-3-gram share, tail-repeat share,
                          final-punctuation / mid-word tail, length ratios (chars, words), copy measures (exact, word overlap
                          with the English source, chrF vs source when overlapping), NLLB-LID (facebook/fasttext-language-identification,
                          218 labels; NOT the reward's GlotLID v3) top-1 label and probability
and the same detector features on the human reference of the row (control).
Output: features_<model>.parquet next to this script; refs_features.parquet (once).
Usage: python compute_features.py [model ...]   (default all)
Paths: repo root is three levels above this file; the NLLB LID model is passed with env LID218 (a local copy of model.bin).
"""
import json, os, re, sys, lzma, zlib, unicodedata, importlib.util
from multiprocessing import Pool
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ARCH = os.path.join(ROOT, "research_archive_2026-11-03")
RLP = os.path.join(ARCH, "eval_385m_rl_predictions_2026-11-04")
FILES = {
    "sft": [os.path.join(ARCH, "eval_385m/out_m385/predictions.jsonl")],
    "a0": [os.path.join(ARCH, "eval_385m/out_a0/predictions.jsonl")],
    "stageA": [os.path.join(RLP, "out_stageA/predictions.jsonl"), os.path.join(RLP, "out_stageA_x/predictions.jsonl")],
    "b1best": [os.path.join(RLP, "out_b1best/predictions.jsonl"), os.path.join(RLP, "out_b1best_x/predictions.jsonl")],
    "mixedbest": [os.path.join(RLP, "out_wk_mixedbest/predictions.jsonl")],
    "weakfinal": [os.path.join(RLP, "out_wk_weakfinal/predictions.jsonl")],
}
BENCH = ("flores_plus", "ntrex", "mafand")

_spec = importlib.util.spec_from_file_location("guards", os.path.join(ROOT, "gemma_mt/gemma_mt/rl/guards.py"))
guards = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(guards)

_LZ = [{"id": lzma.FILTER_LZMA2, "preset": 9}]
_WORD = re.compile(r"\w+", re.UNICODE)
_FINAL = set(".!?…。！？।؟։۔\"”’»)]'")


def nfc(s):
    return unicodedata.normalize("NFC", s)


def rep_share(items):
    n = len(items)
    return 0.0 if n == 0 else 1.0 - len(set(items)) / n


def text_feats(t):
    """independent detector features of one text"""
    t = nfc(t or "")
    b = t.encode("utf-8")
    nb = len(b)
    f = {"chars": len(t), "bytes": nb, "words": len(t.split())}
    f["lz_bytes"] = len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=_LZ)) if nb else 0
    f["lz_ratio"] = f["lz_bytes"] / nb if nb else np.nan
    s = " ".join(t.lower().split())
    g8 = [s[i:i + 8] for i in range(len(s) - 7)]
    f["c8_rep"] = rep_share(g8)
    w = s.split()
    g3 = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    f["w3_rep"] = rep_share(g3)
    # tail repeat: share of the char-8-grams of the last third that already occur in the first two thirds
    if len(s) >= 48:
        k = len(s) * 2 // 3
        head = set(s[i:i + 8] for i in range(0, k - 7))
        tail = [s[i:i + 8] for i in range(k, len(s) - 7)]
        f["tail_rep"] = sum(1 for x in tail if x in head) / max(1, len(tail))
    else:
        f["tail_rep"] = 0.0
    st = t.rstrip()
    f["ends_final"] = bool(st) and (st[-1] in _FINAL)
    f["ends_alpha"] = bool(st) and st[-1].isalpha()
    return f


def row_feats(args):
    hyp, ref, eng, finished, prefix = args
    h = text_feats(hyp)
    out = {("h_" + k): v for k, v in h.items()}
    out["finished"] = bool(finished)
    out["prefix_status"] = prefix
    # reward-family flags (the reward's own guard functions)
    out["rw_rep"] = bool(guards.is_repetitive(hyp))
    out["rw_selfrep"] = bool(guards.is_self_repetitive(hyp))
    out["rw_loop"] = out["rw_rep"] or out["rw_selfrep"]
    out["rw_copy"] = bool(guards.is_copy_of_source(eng, hyp))
    out["rw_lenbad"] = bool(guards.length_ratio_bad(eng, hyp))
    out["rw_roletail"] = bool(guards.has_role_tail(hyp, eng))
    # copy measures (independent of the jaccard rule)
    ew = set(x for x in (m.lower() for m in _WORD.findall(eng)) if len(x) >= 3)
    hw = [x for x in (m.lower() for m in _WORD.findall(hyp)) if len(x) >= 3]
    out["h_en_overlap"] = (sum(1 for x in hw if x in ew) / len(hw)) if hw else 0.0
    out["h_n_long_words"] = len(hw)
    out["h_eq_src_exact"] = nfc(hyp).strip().lower() == nfc(eng).strip().lower()
    return out


def ref_feats(args):
    ref, eng = args
    r = text_feats(ref)
    out = {("r_" + k): v for k, v in r.items()}
    out["e_chars"] = len(eng); out["e_words"] = len(eng.split())
    st = eng.rstrip(); out["e_ends_final"] = bool(st) and st[-1] in _FINAL
    out["r_rw_loop"] = bool(guards.is_repetitive(ref) or guards.is_self_repetitive(ref))
    ew = set(x for x in (m.lower() for m in _WORD.findall(eng)) if len(x) >= 3)
    rw = [x for x in (m.lower() for m in _WORD.findall(ref)) if len(x) >= 3]
    out["r_en_overlap"] = (sum(1 for x in rw if x in ew) / len(rw)) if rw else 0.0
    out["r_n_long_words"] = len(rw)
    return out


def load(paths):
    rows, seen = [], set()
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r["benchmark"] not in BENCH:
                continue
            k = (r["benchmark"], r["iso"], str(r["id"]))
            if k in seen:
                continue
            seen.add(k); rows.append(r)
    return rows


def clean_for_fasttext(text, max_chars=1000):
    t = " ".join((text or "").replace("\x00", " ").split())
    if len(t) > max_chars:
        cut = t.rfind(" ", 0, max_chars)
        t = t[: cut if cut > max_chars // 2 else max_chars]
    return t


def lid_many(model, texts):
    labs, probs, l2, p2 = [], [], [], []
    cache = {}
    for t in texts:
        line = clean_for_fasttext(t)
        if line in cache:
            a = cache[line]
        else:
            if not line:
                a = (("und", 0.0), ("und", 0.0))
            else:
                pairs = model.f.predict(line + "\n", 2, 0.0, "strict")
                a = tuple((lab.replace("__label__", ""), float(p)) for p, lab in pairs)
                if len(a) < 2:
                    a = a + (("und", 0.0),)
            cache[line] = a
        labs.append(a[0][0]); probs.append(a[0][1]); l2.append(a[1][0]); p2.append(a[1][1])
    return labs, probs, l2, p2


def main():
    models = sys.argv[1:] or list(FILES)
    lid_path = os.environ.get("LID218")
    import fasttext
    fasttext.FastText.eprint = lambda *a, **k: None
    lid = fasttext.load_model(lid_path)
    pool = Pool(6)
    base = None
    for m in models:
        rows = load(FILES[m])
        print(m, len(rows), flush=True)
        df = pd.DataFrame({"benchmark": [r["benchmark"] for r in rows], "iso": [r["iso"] for r in rows], "id": [str(r["id"]) for r in rows]})
        feats = pool.map(row_feats, [(r["hyp"], r["reference"], r["english"], r["finished"], r["prefix_status"]) for r in rows], chunksize=2000)
        df = pd.concat([df, pd.DataFrame(feats)], axis=1)
        labs, probs, l2, p2 = lid_many(lid, [nfc(r["hyp"]) for r in rows])
        df["lid_label"], df["lid_prob"], df["lid_label2"], df["lid_prob2"] = labs, probs, l2, p2
        df["sent_chrf_src"] = np.nan
        df.to_parquet(os.path.join(HERE, f"features_{m}.parquet"))
        if m == "sft" or not os.path.exists(os.path.join(HERE, "refs_features.parquet")):
            rf = pool.map(ref_feats, [(r["reference"], r["english"]) for r in rows], chunksize=2000)
            rdf = pd.DataFrame({"benchmark": df["benchmark"], "iso": df["iso"], "id": df["id"]})
            rdf = pd.concat([rdf, pd.DataFrame(rf)], axis=1)
            labs, probs, l2, p2 = lid_many(lid, [nfc(r["reference"]) for r in rows])
            rdf["lid_label"], rdf["lid_prob"], rdf["lid_label2"], rdf["lid_prob2"] = labs, probs, l2, p2
            # english-side LID control (is the source recognised as English?) is not needed: skip
            rdf.to_parquet(os.path.join(HERE, "refs_features.parquet"))
            print("refs", len(rdf), flush=True)
    print("done")


if __name__ == "__main__":
    main()
