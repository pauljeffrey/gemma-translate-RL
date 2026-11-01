# SPDX-License-Identifier: AGPL-3.0-or-later
"""E1 pairwise benchmark comparison, step 1 (CPU, box): per-sentence surface metrics of one system from the eval pipeline's predictions.jsonl.

    python pairwise_surface.py <predictions.jsonl> <out.jsonl.gz> [--max-new-tokens 768] [--benchmarks a,b]

One output row per (benchmark, iso, id): the benchmark row (english, reference, contamination flags `contaminated` = English sentence in the SFT
text, `pair_seen` = reference pair in the SFT text) + the system's hypothesis and per-sentence metrics:
  chrf   sentence chrF++ (sacrebleu CHRF(word_order=2), NFC hyp/ref; 0..100)
  fin    the completion ended with the stop token inside the budget
  loop   not finished, or rl.guards.is_repetitive / is_self_repetitive, or empty   (the decoding study's definition)
  rep3   >= 15 words and fewer than 50% distinct word 3-grams (the pure word-3-gram repetition flag)
  lid    GlotLID v3 top-1 label is one of the language's own labels (lid_sib: sibling labels accepted too); null for languages without a label
  lratio hypothesis chars / reference chars
Run with the box's venv_gemma python, PYTHONPATH=/workspace/sft_build/gemma_mt_2c:/workspace/sft_build/pylibs.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
import unicodedata
from multiprocessing import Pool

sys.path.insert(0, "/workspace/sft_build/pylibs")
from sacrebleu.metrics import CHRF

from gemma_mt.languages import LANGUAGES
from gemma_mt.rl import guards as G
from gemma_mt.rl import lid as L

_CH = CHRF(word_order=2)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def rep3(text: str, min_words: int = 15, max_distinct: float = 0.5) -> bool:
    w = text.split()
    if len(w) < min_words:
        return False
    grams = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return len(set(grams)) / len(grams) < max_distinct


def _one(args):
    hyp, ref = args
    h, r = nfc(hyp), nfc(ref)
    c = _CH.sentence_score(h, [r]).score if h.strip() else 0.0
    return round(c, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("out")
    ap.add_argument("--benchmarks", default="")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    want = set(x for x in a.benchmarks.split(",") if x)
    rows = []
    with open(a.pred, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if want and r["benchmark"] not in want:
                continue
            if r["benchmark"] in ("gatitos", "smol_doc"):
                continue
            rows.append(r)
    print(len(rows), "rows", flush=True)
    with Pool(a.procs) as p:
        chrf = p.map(_one, [(r["hyp"], r["reference"]) for r in rows], chunksize=2000)
    lidbin = glob.glob("/workspace/sft_build/hf/hub/models--cis-lmu--glotlid/snapshots/*/model_v3.bin")[0]
    clf = L.GlotLIDClassifier(path=lidbin)
    L.selftest(clf)
    known = set(clf.labels())
    acc = {}
    for iso, lg in LANGUAGES.items():
        own = tuple(x for x in lg.glotlid if x in known)
        acc[iso] = (set(own), set(own) | {x for x in L.siblings_of(own) if x in known}) if own else None
    top = clf([r["hyp"] for r in rows])
    n = 0
    with gzip.open(a.out, "wt", encoding="utf-8") as g:
        for r, c, t in zip(rows, chrf, top):
            hyp, ref = r["hyp"], r["reference"]
            ac = acc.get(r["iso"])
            if ac is None:
                lid_ok = lid_sib = None
            elif not hyp.strip():
                lid_ok = lid_sib = False
            else:
                lid_ok, lid_sib = t in ac[0], t in ac[1]
            fin = bool(r.get("finished", True))
            loop = (not fin) or (not hyp.strip()) or G.is_repetitive(hyp) or G.is_self_repetitive(hyp)
            g.write(json.dumps({
                "b": r["benchmark"], "iso": r["iso"], "id": str(r["id"]),
                "contaminated": r.get("contaminated"), "pair_seen": r.get("pair_seen"),
                "en": r["english"], "ref": ref, "hyp": hyp,
                "chrf": c, "fin": fin, "loop": bool(loop), "rep3": bool(rep3(hyp)), "empty": not hyp.strip(),
                "lid": lid_ok, "lid_sib": lid_sib, "lratio": round(len(hyp) / max(1, len(ref)), 4),
                "prefix_ok": r.get("prefix_status") == "ok",
            }, ensure_ascii=False) + "\n")
            n += 1
    print("wrote", a.out, n)


if __name__ == "__main__":
    main()
