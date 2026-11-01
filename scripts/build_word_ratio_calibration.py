# SPDX-License-Identifier: AGPL-3.0-or-later
"""Add a per-language WORD-ratio block to the shaping calibration (owner 2026-10-21): shaping_flores_dev_v6.json -> shaping_flores_dev_v7.json.

    python scripts/build_word_ratio_calibration.py --in calibration/shaping_flores_dev_v6.json --out calibration/shaping_flores_dev_v7.json \
        [--bench-dev bench/bench_dev.jsonl] [--no-hub]

For every RL language with a FLORES+ code the DEV sentences (never devtest) are fetched from the Hub (`openlanguagedata/flores_plus`, dev/<code>.jsonl and
dev/eng_Latn.jsonl, read token in HF_TOKEN), plus the `*_dev` rows of bench_dev.jsonl; the per-sentence ratio len(reference.split()) / len(english.split())
(English sentences with >= 4 words) gives {n, p05, p50, p95, p99}. A language with < 80 sentences gets no block and falls back to the pooled prior
`_word_ratio_pooled` (median over the languages of p95 and p99). The reward then penalises a segment only ABOVE p99 x (1 + word_ratio_slack)
(ContentShaper.word_excess): the word count per English word differs strongly between orthographies (Zulu 0.75 at the median, Setswana 1.29, Fon 1.37), so
no global cap. p99 is floored at 1.25 and capped at 3.0 (a few misaligned sentences make the raw p99 of Hausa 4.7 and of Shona 4.6).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

MIN_SRC_WORDS = 4
MIN_N = 80
P99_FLOOR, P99_CAP = 1.25, 3.0


def quantile(sorted_vals: Sequence[float], q: float) -> float:
    return float(sorted_vals[min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1)))])


def word_ratio_block(pairs: Sequence[Tuple[str, str]]) -> Dict[str, float]:
    """pairs: (english, reference). None when there are fewer than MIN_N usable sentences."""
    r = sorted(len(t.split()) / len(s.split()) for s, t in pairs if len(s.split()) >= MIN_SRC_WORDS and t.strip())
    if len(r) < MIN_N:
        return {}
    p99 = min(max(quantile(r, 0.99), P99_FLOOR), P99_CAP)
    return {"n": len(r), "p05": quantile(r, 0.05), "p50": quantile(r, 0.5), "p95": quantile(r, 0.95), "p99": p99}


def pooled(blocks: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    p95 = [b["p95"] for b in blocks.values()]
    p99 = [b["p99"] for b in blocks.values()]
    return {"p95": statistics.median(p95), "p99": statistics.median(p99), "n_languages": len(blocks)}


def read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench-dev", default=None)
    ap.add_argument("--no-hub", action="store_true")
    a = ap.parse_args()
    from gemma_mt.languages import LANGUAGES, SUPPORTED_RL

    cal = json.load(open(a.inp, encoding="utf-8"))
    pairs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    if a.bench_dev and os.path.exists(a.bench_dev):
        for r in read_jsonl(a.bench_dev):
            if str(r.get("benchmark", "")).endswith("_dev"):
                pairs[r["iso"]].append((r["english"], r["reference"]))
    if not a.no_hub:
        from huggingface_hub import hf_hub_download

        def hub(code: str) -> List[dict]:
            return read_jsonl(hf_hub_download("openlanguagedata/flores_plus", f"dev/{code}.jsonl", repo_type="dataset", token=os.environ.get("HF_TOKEN")))

        en = {int(r["id"]): r["text"] for r in hub("eng_Latn")}
        for iso in SUPPORTED_RL:
            code = LANGUAGES[iso].flores
            if not code:
                continue
            codes = ["twi_Latn_akua1239", "twi_Latn_asan1239"] if code == "twi_Latn" else [code]
            try:
                for c in codes:
                    for r in hub(c):
                        if int(r["id"]) in en and r["text"].strip():
                            pairs[iso].append((en[int(r["id"])], r["text"]))
            except Exception as e:  # a missing / gated file: that language uses the pooled prior
                print(f"[word_ratio] {iso} ({code}): {str(e)[:80]}", file=sys.stderr)
    blocks: Dict[str, Dict[str, float]] = {}
    for iso in SUPPORTED_RL:
        b = word_ratio_block(pairs.get(iso, []))
        if b:
            blocks[iso] = b
            if iso in cal and isinstance(cal[iso], dict):
                cal[iso]["word_ratio"] = b
            else:
                cal[iso] = {"word_ratio": b}
    cal["_word_ratio_pooled"] = pooled(blocks)
    meta = cal.setdefault("_meta", {})
    meta["word_ratio"] = {"built": "2026-10-21", "min_n": MIN_N, "p99_floor": P99_FLOOR, "p99_cap": P99_CAP, "languages": len(blocks), "source": "FLORES+ dev (+ bench_dev *_dev rows)"}
    json.dump(cal, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"word_ratio blocks for {len(blocks)} languages; pooled {cal['_word_ratio_pooled']}; languages without: {[i for i in SUPPORTED_RL if i not in blocks]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
