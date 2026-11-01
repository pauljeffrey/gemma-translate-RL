#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the English->X probe sets of the new-language study (D-47): up to N sentences per (language, reference set), seeded, English 5..40 words.

    python scripts/newlang_probe_sets.py --out /workspace/newlang_probe/data/sets [--n 60] [--keys fuv gaa ...]

Reads the Hub (HF_TOKEN from the environment; FLORES+ is gated-auto, SMOL is open) and the local NTREX / SALT / IbomMT copies of the old box. Writes one
`<key>__<kind>_<code>.jsonl` per reference set ({english, reference, id}) and `sets_index.json` (full size of every set, probe size, mean words).
Nothing is pushed anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newlang_registry as R  # noqa: E402

LOCAL_ROOT = "/workspace/sft_build/bench/test_sets"


def _hf(repo, fn):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, fn, repo_type="dataset")


def _pq(repo, fn):
    import pyarrow.parquet as pq

    return pq.read_table(_hf(repo, fn)).to_pylist()


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_set(spec):
    """-> list of dict(english, reference, id, orig) for 'kind:code'."""
    kind, code = spec.split(":", 1)
    if kind == "flores":
        en = {r["id"]: r["text"] for r in _jsonl(_hf("openlanguagedata/flores_plus", "dev/eng_Latn.jsonl"))}
        rows = _jsonl(_hf("openlanguagedata/flores_plus", f"dev/{code}.jsonl"))
        return [dict(english=en[r["id"]], reference=r["text"], id=f"f{r['id']}", orig=True) for r in rows if r["id"] in en and r["text"].strip()]
    if kind == "floresorig":                     # the original FLORES-200 (facebook/flores): has tzm_Tfng (Central Atlas Tamazight), which FLORES+ does not
        en = {r["id"]: r["sentence"] for r in _pq("facebook/flores", "data/language/eng_Latn/dev-00000-of-00001.parquet")}
        rows = _pq("facebook/flores", f"data/language/{code}/dev-00000-of-00001.parquet")
        return [dict(english=en[r["id"]], reference=r["sentence"], id=f"o{r['id']}", orig=True) for r in rows if r["id"] in en and r["sentence"].strip()]
    if kind == "smol":
        rows = _jsonl(_hf("google/smol", f"smolsent/en_{code}.jsonl"))
        return [dict(english=r["src"], reference=r["trg"], id=f"s{r['id']}", orig=bool(r.get("is_src_orig"))) for r in rows if r.get("src") and r.get("trg")]
    if kind == "ntrex":
        base = f"{LOCAL_ROOT}/ntrex/NTREX-128"
        src = open(f"{base}/newstest2019-src.eng.txt", encoding="utf-8").read().split("\n")
        ref = open(f"{base}/newstest2019-ref.{code}.txt", encoding="utf-8").read().split("\n")
        return [dict(english=s.strip(), reference=t.strip(), id=f"n{i}", orig=True) for i, (s, t) in enumerate(zip(src, ref)) if s.strip() and t.strip()]
    if kind == "local":
        rows = _jsonl(f"{LOCAL_ROOT}/{code}")
        return [dict(english=r["src"], reference=r["trg"], id=f"l{r.get('id', i)}", orig=True) for i, r in enumerate(rows) if r.get("src") and r.get("trg")]
    raise ValueError(spec)


def pick(rows, n, seed, lo=5, hi=40):
    ok = [r for r in rows if lo <= len(r["english"].split()) <= hi]
    pref = [r for r in ok if r["orig"]]
    pool = pref if len(pref) >= n else ok
    pool = sorted(pool, key=lambda r: r["id"])
    random.Random(seed).shuffle(pool)
    return pool[:n]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--keys", nargs="*", default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    reg = {**{k: v["sets"] for k, v in R.NEW.items()}, **{k: v["sets"] for k, v in R.YARDSTICK.items()}}
    index = {}
    for key, sets in reg.items():
        if a.keys and key not in a.keys:
            continue
        for spec in sets:
            try:
                rows = load_set(spec)
            except Exception as e:                                          # noqa: BLE001
                index[f"{key}__{spec}"] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
                print(key, spec, "ERROR", str(e)[:120], flush=True)
                continue
            chosen = pick(rows, a.n, a.seed)
            name = f"{key}__{spec.replace(':', '_').replace('/', '_')}"
            with open(os.path.join(a.out, name + ".jsonl"), "w", encoding="utf-8") as f:
                for r in chosen:
                    f.write(json.dumps({k: r[k] for k in ("english", "reference", "id")}, ensure_ascii=False) + "\n")
            index[name] = {"full_size": len(rows), "probe_n": len(chosen), "orig_share": round(sum(r["orig"] for r in rows) / max(1, len(rows)), 3),
                           "mean_words_en": round(sum(len(r["english"].split()) for r in chosen) / max(1, len(chosen)), 1),
                           "mean_chars_ref": round(sum(len(r["reference"]) for r in chosen) / max(1, len(chosen)), 1)}
            print(name, index[name], flush=True)
    ip = os.path.join(a.out, "sets_index.json")
    old = json.load(open(ip)) if os.path.exists(ip) else {}
    old.update(index)
    json.dump(old, open(ip, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
