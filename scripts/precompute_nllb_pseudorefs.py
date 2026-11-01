#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Precompute NLLB translations of the RL prompts for the reward pseudo-reference signal (R14). REWARD ONLY: output is CC-BY-NC, rows carry licence_class 'reward_only_nc'
and the files must never be mixed into SFT / training data.

    python scripts/precompute_nllb_pseudorefs.py --langs kea dyu sag --out /workspace/jobs/r14/pseudorefs_v1 \
        --ct2-dir .../nllb3p3b-ct2-int8 --hf-dir .../nllb600m-hf --max-sentences 30000

One parquet per language: key (= rl/archive.make_key(iso, source) = sha1(iso + "\\n" + NFC(source))[:16]), iso, source, sentences, translations (aligned; '' = rejected sentence),
seq_scores, n_sentences, source_dataset, model, decode, licence_class. The English windows are those of the language's RL prompt pools
(rl_prompts_v3 + rl_prompts_sci_v1), in a seeded random order; documents are split with rl/segments.split_sentences (the reward's own splitter, so the chunk alignment of the
reward finds its sentences), translated sentence by sentence (beam 4 + no-repeat-4-gram, sentence level: NLLB fails on long inputs). Resumable: sentence translations are
cached in <out>/cache_<iso>.jsonl, and any `--reuse-cache` file (e.g. the stage-2 teacher translations, same model) is read first.
Cost: 3.3B int8 on a shared RTX PRO 6000: 25-60 sentences/s; a prompt window has ~10-20 sentences on average (b64 .. b2048 bins).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import unicodedata

from gemma_mt.rl import anchors as A
from gemma_mt.rl.nllb_signal import NLLB_CODE, REWARD_ONLY_NC, NllbCT2, key_of
from gemma_mt.rl.segments import split_sentences

POOLS = ["/workspace/sft_build/gemma/rl_prompts_v3/{iso}.parquet", "/workspace/jobs/r7/assets/rl_prompts_sci_v1/{iso}.parquet"]


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def reject(src: str, hyp: str) -> bool:
    """empty, verbatim copy of the English, or a translation loop."""
    if not hyp.strip():
        return True
    sw, hw = A.words(src), A.words(hyp)
    if not hw:
        return True
    if len(sw) >= 5 and len(set(sw) & set(hw)) / max(1, len(set(sw) | set(hw))) > 0.6:
        return True
    if len(hw) >= 5 and A.repeat_shares(hw, (3,))[3] > 0.25 and (len(sw) < 5 or A.repeat_shares(sw, (3,))[3] < 0.1):
        return True
    return False


def windows_of(iso: str, seed: int, pools=POOLS):
    import pyarrow.parquet as pq

    rows = []
    for p in pools:
        p = p.format(iso=iso)
        if os.path.exists(p):
            rows += pq.read_table(p, columns=["english", "source_dataset"]).to_pylist()
    order = list(range(len(rows)))
    random.Random(f"{seed}-{iso}").shuffle(order)
    for i in order:
        yield nfc(rows[i]["english"]), rows[i]["source_dataset"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ct2-dir", required=True)
    ap.add_argument("--hf-dir", required=True, help="directory with the NLLB tokenizer files")
    ap.add_argument("--model-name", default="nllb-200-3.3B")
    ap.add_argument("--compute-type", default="int8_float16")
    ap.add_argument("--max-sentences", type=int, default=30000, help="per language (0 = every window of the pools)")
    ap.add_argument("--max-sentence-chars", type=int, default=700)
    ap.add_argument("--max-batch-tokens", type=int, default=4096, help="CTranslate2 batch size in tokens (2026-10-29: raise to 16384+ on an idle 96 GB card for throughput; default unchanged)")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--reuse-cache", nargs="*", default=[], help="jsonl files {english, hyp} (one per language: {iso} is substituted) read before translating")
    ap.add_argument("--pools", nargs="*", default=POOLS)
    a = ap.parse_args(argv)

    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(a.out, exist_ok=True)
    m = NllbCT2(a.ct2_dir, a.hf_dir, compute_type=a.compute_type)
    for iso in a.langs:
        if iso not in NLLB_CODE:
            print(f"{iso}: not covered by NLLB (exact), skipped")
            continue
        t0 = time.time()
        cache_path = os.path.join(a.out, f"cache_{iso}.jsonl")
        cache = {}
        for p in [cache_path] + [x.format(iso=iso) for x in a.reuse_cache]:
            if os.path.exists(p):
                for l in open(p, encoding="utf-8"):
                    d = json.loads(l)
                    cache[d["english"]] = (d["hyp"], d.get("score"))
        wins, n_sent = [], 0
        for text, ds in windows_of(iso, a.seed, a.pools):
            ss = split_sentences(text)
            if not ss:
                continue
            wins.append((text, ds, ss))
            n_sent += len(ss)
            if a.max_sentences and n_sent >= a.max_sentences:
                break
        todo = sorted({s for _, _, ss in wins for s in ss if s not in cache and len(s) <= a.max_sentence_chars})
        with open(cache_path, "a", encoding="utf-8") as f:
            for j in range(0, len(todo), 2000):
                part = todo[j: j + 2000]
                hyps, sc = m.translate(part, iso, max_batch_tokens=a.max_batch_tokens, return_scores=True)
                for e, h, s in zip(part, hyps, sc):
                    cache[e] = (h, s)
                    f.write(json.dumps({"english": e, "hyp": h, "score": s}, ensure_ascii=False) + "\n")
                f.flush()
        cols = {k: [] for k in ("key", "iso", "source", "sentences", "translations", "seq_scores", "n_sentences", "source_dataset", "model", "decode", "licence_class")}
        rej = 0
        for text, ds, ss in wins:
            tr = []
            for s in ss:
                h, _ = cache.get(s, ("", None))
                if not h or reject(s, h):
                    h = ""
                    rej += 1
                tr.append(h)
            cols["key"].append(key_of(iso, text)); cols["iso"].append(iso); cols["source"].append(text); cols["sentences"].append(ss); cols["translations"].append(tr)
            cols["seq_scores"].append([float(cache.get(s, ("", None))[1] or 0.0) for s in ss]); cols["n_sentences"].append(len(ss)); cols["source_dataset"].append(ds)
            cols["model"].append(a.model_name); cols["decode"].append("beam4_no_repeat_ngram4"); cols["licence_class"].append(REWARD_ONLY_NC)
        pq.write_table(pa.table(cols), os.path.join(a.out, f"{iso}.parquet"), compression="zstd")
        print(f"{iso}: {len(wins)} windows, {n_sent} sentences ({len(todo)} translated now, {rej} rejected) in {time.time() - t0:.0f}s", flush=True)
    json.dump({"licence_class": REWARD_ONLY_NC, "model": a.model_name, "note": "NLLB output is CC-BY-NC-4.0: reward scoring only, never SFT/training data"},
              open(os.path.join(a.out, "LICENCE_REWARD_ONLY_NC.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
