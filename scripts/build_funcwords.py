#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-language word statistics of the TARGET-language training text, for the stop-word profile of the anchors (D-61, rl/anchors_v2.py).

Text sources (both are training / pivot text, never the human reference sets that the thresholds are calibrated on):
  * the packed SFT / replay mixture of agent R4 (`--mix DIR`, e.g. /workspace/jobs/r4/mix_v1: tokens.u32.bin + sample_offset / prompt_len / iso_id npy, decoded with the model's tokenizer)
  * the English-pivot corpus of agent R3 (`--corpus DIR`, <iso>.parquet with a column `x`, not the held-out rows)
The text is cut into segments at . ! ? ; : and newlines; for every language the output keeps, for the `--top` words of highest DOCUMENT frequency (segments that contain the word), df and the
total count, plus the number of segments / tokens, plus the tf of every English function word (so the calibration can drop the ones that are also target words).

    python scripts/build_funcwords.py --mix /workspace/jobs/r4/mix_v1 --tokenizer /workspace/sft_build/runs/final_1433/tokenizer.json \
        --corpus /workspace/jobs/b3/corpus --out calibration/funcwords_v1.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import zlib
from typing import Dict, Iterable, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from gemma_mt.rl.textstats import EN_FUNC, out_tokens  # noqa: E402

_SEG = re.compile(r"[.!?;:\n]+|\s[-–—]\s")


def segments(text: str) -> Iterable[List[str]]:
    for s in _SEG.split(text):
        t = out_tokens(s)
        if t:
            yield t


def count_into(stats: Dict[str, dict], iso: str, text: str, cap_segments: int, bigrams: bool = False) -> None:
    st = stats.setdefault(iso, {"df": collections.Counter(), "tf": collections.Counter(), "bg": collections.Counter(), "n_seg": 0, "n_tok": 0})
    for toks in segments(text):
        if st["n_seg"] >= cap_segments:
            return
        st["n_seg"] += 1
        st["n_tok"] += len(toks)
        st["tf"].update(toks)
        st["df"].update(set(toks))
        if bigrams:
            st["bg"].update(zlib.crc32((a + "\x00" + b).encode("utf-8")) for a, b in zip(toks, toks[1:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=150)
    ap.add_argument("--bigrams-out", default=None, help="npz: <iso> -> sorted uint32 crc32 hashes of the word bigrams seen >= --bigram-min times (at most --bigram-cap per language)")
    ap.add_argument("--bigram-min", type=int, default=3)
    ap.add_argument("--bigram-cap", type=int, default=150_000)
    ap.add_argument("--cap-segments", type=int, default=250_000)
    ap.add_argument("--cap-corpus", type=int, default=100_000)
    a = ap.parse_args()
    stats: Dict[str, dict] = {}
    srcs: Dict[str, List[str]] = collections.defaultdict(list)
    if a.mix:
        import numpy as np
        from tokenizers import Tokenizer

        tk = Tokenizer.from_file(a.tokenizer)
        d = a.mix.rstrip("/") + "/train/"
        tok = np.memmap(d + "tokens.u32.bin", dtype=np.uint32, mode="r")
        off, pl, sl, iso_id = (np.load(d + f + ".npy") for f in ("sample_offset", "prompt_len", "sample_len", "iso_id"))
        direction = np.load(d + "direction.npy")
        tab = json.load(open(a.mix.rstrip("/") + "/meta.json"))["iso_table"]
        for i in range(len(off)):
            if direction[i] != 0:
                continue
            s = tok[off[i] + pl[i]: off[i] + sl[i]].tolist()
            text = tk.decode(s)
            iso = tab[iso_id[i]]
            if text.startswith(iso + " "):
                text = text[len(iso) + 1:]
            count_into(stats, iso, text, a.cap_segments, bool(a.bigrams_out))
        for iso in stats:
            srcs[iso].append("mix")
    if a.corpus:
        import pyarrow.parquet as pq

        for fn in sorted(os.listdir(a.corpus)):
            if not fn.endswith(".parquet"):
                continue
            iso = fn[:-8]
            t = pq.read_table(os.path.join(a.corpus, fn), columns=["x", "held"]).to_pydict()
            n0 = stats.get(iso, {}).get("n_seg", 0)
            c = 0
            for x, h in zip(t["x"], t["held"]):
                if h:
                    continue
                count_into(stats, iso, x, n0 + a.cap_corpus, bool(a.bigrams_out))
                c += 1
                if c >= a.cap_corpus:
                    break
            srcs[iso].append("corpus")
    out = {}
    for iso, st in sorted(stats.items()):
        top = [[w, c, st["tf"][w]] for w, c in st["df"].most_common(a.top)]
        out[iso] = {"n_seg": st["n_seg"], "n_tok": st["n_tok"], "sources": srcs[iso], "top": top,
                    "en_tf": {w: st["tf"][w] for w in sorted(EN_FUNC) if st["tf"][w]}}
        print(iso, st["n_seg"], st["n_tok"], srcs[iso], [t[0] for t in top[:12]], flush=True)
    if a.bigrams_out:
        import numpy as np

        arrs = {}
        for iso, st in stats.items():
            keep = [(c, h) for h, c in st["bg"].items() if c >= a.bigram_min]
            keep.sort(reverse=True)
            arrs[iso] = np.sort(np.array([h for _, h in keep[: a.bigram_cap]], dtype=np.uint32))
        np.savez_compressed(a.bigrams_out, **arrs)
        print("bigram tables:", {k: len(v) for k, v in sorted(arrs.items())})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
