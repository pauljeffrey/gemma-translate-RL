#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validation of the long-document contamination scanner with positive / negative / partial / duplicate controls.

    build     writes the control documents as pool files into <work>/pool/ (never into the real pool dir) and <work>/controls.json
    evaluate  reads <work>/out/*.flags.jsonl (written by scripts/longdoc_contam_scan.py --pool-dir <work>/pool --out-dir <work>/out) and writes
              <work>/validation.json + a printed table

Controls (all seeded):
    pos_rl        300 random English windows of the RL prompts (gemma/rl_prompts_v2), whole window = document
    pos_sft       300 random English windows of the SFT stream (train 270 + val 30), decoded from the packed tokens, whole window = document
    pos_bench     100 passages of consecutive rows of FLORES+ devtest / NTREX (8 rows) and SMOL documents
    neg_wikinews  90 Wikinews articles of 2025-08 .. 2026-05 (fetched 2026-10-20 through the MediaWiki API; the training data predates them) plus 25 concatenations of 3
    neg_gutenberg 80 x 1000-word slices of 20 Project Gutenberg novels (public domain; NOT guaranteed absent from the SFT data - reported separately)
    partial_<src>_<f>  40 documents each: a 100-200 word slice of a training window (src = rl | sft) + fresh Wikinews words, f = 0.3 / 0.6 of the words copied
    dup           40 Wikinews originals + a full copy, a 60% prefix and a 30% prefix of each
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from gemma_mt.eval import longdoc_contam as L   # noqa: E402
from gemma_mt.eval.contamination import normalise   # noqa: E402


def write_pool(path: str, docs: List[Dict[str, Any]]) -> None:
    L.atomic_write_jsonl(path, docs)


def gutenberg_docs(raw_dir: str, rng: random.Random, n_per_book: int = 4, words: int = 1000) -> List[Dict[str, Any]]:
    out = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "pg*.txt"))):
        t = open(f, encoding="utf-8", errors="replace").read()
        a = t.find("*** START")
        if a >= 0:
            t = t[t.find("\n", a) + 1:]
        if "*** END" in t:
            t = t[: t.rfind("*** END")]
        w = " ".join(t.split()).split(" ")
        for q in range(n_per_book):
            s = len(w) // (n_per_book + 1) * (q + 1)
            out.append({"id": f"gut_{os.path.basename(f)[2:-4]}_{q}", "text": " ".join(w[s: s + words])})
    return out


def build(a: argparse.Namespace) -> None:
    rng = random.Random(a.seed)
    pool = os.path.join(a.work, "pool")
    os.makedirs(pool, exist_ok=True)
    meta: Dict[str, Any] = {"seed": a.seed}
    # fresh text: Wikinews 2025-08 .. 2026-05
    wn = [json.loads(x) for x in open(os.path.join(a.raw_dir, "wikinews_2026.jsonl"), encoding="utf-8")]
    wn = [w for w in wn if len(w["text"].split()) >= 120]
    neg = [{"id": f"wn_{i}", "text": w["text"]} for i, w in enumerate(wn)]
    trip = [{"id": f"wn3_{i}", "text": "\n\n".join(wn[j]["text"] for j in range(3 * i, 3 * i + 3))} for i in range(len(wn) // 3)][:25]
    write_pool(os.path.join(pool, "neg_wikinews.jsonl"), neg + trip)
    write_pool(os.path.join(pool, "neg_gutenberg.jsonl"), gutenberg_docs(a.raw_dir, rng))

    # RL windows
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(a.rl_dir, "*.parquet")))
    rl_rows, seen = [], set()
    for f in rng.sample(files, 40):
        t = pq.read_table(f, columns=["english", "length_bin"]).to_pydict()
        for i in rng.sample(range(len(t["english"])), 20):
            e = t["english"][i]
            if e not in seen:
                seen.add(e)
                rl_rows.append({"text": e, "bin": t["length_bin"][i], "file": os.path.basename(f)})
    rng.shuffle(rl_rows)
    rl_rows = rl_rows[:600]
    write_pool(os.path.join(pool, "pos_rl.jsonl"), [{"id": f"rl_{i}", "text": r["text"]} for i, r in enumerate(rl_rows[:300])])
    meta["pos_rl_bins"] = {f"rl_{i}": r["bin"] for i, r in enumerate(rl_rows[:300])}

    # SFT windows (decoded)
    tk = L.load_tokenizers_json(a.tokenizer)
    sft = []
    for split, k in (("train", 1500), ("val", 80)):
        sp = L.PackedSplit(os.path.join(a.sft_dir, split))
        ids = sorted(rng.sample(range(sp.n_samples), k))
        for i in ids:
            o, l, p = int(sp.sample_offset[i]), int(sp.sample_len[i]), int(sp.prompt_len[i])
            import numpy as np
            slab = np.asarray(sp.tokens[o: o + l])
            s, e, v = L.english_bounds(slab, np.array([0]), np.array([p]))
            if v[0]:
                sft.append({"text": tk.decode(slab[s[0]: e[0]].tolist(), skip_special_tokens=False), "split": split, "idx": int(i), "n_tokens": int(e[0] - s[0])})
    rng.shuffle(sft)
    tr, va = [x for x in sft if x["split"] == "train"], [x for x in sft if x["split"] == "val"]
    pos_sft = tr[:270] + va[:30]
    sft_rest = tr[270:] + va[30:]
    write_pool(os.path.join(pool, "pos_sft.jsonl"), [{"id": f"sft_{i}", "text": r["text"]} for i, r in enumerate(pos_sft)])
    meta["pos_sft_tokens"] = {f"sft_{i}": r["n_tokens"] for i, r in enumerate(pos_sft)}

    # benchmark passages
    bench = L.load_benchmark_english(a.bench_root)
    bp = []
    for name, k in (("flores_plus", 50), ("ntrex", 25)):
        rows = bench.get(name, [])
        for _ in range(k):
            s = rng.randrange(0, max(1, len(rows) - 8))
            bp.append({"id": f"bench_{name}_{s}", "text": " ".join(rows[s: s + 8]), "bench": name})
    sd = bench.get("smol_doc", [])
    for i in rng.sample(range(len(sd)), min(25, len(sd))):
        bp.append({"id": f"bench_smol_doc_{i}", "text": sd[i], "bench": "smol_doc"})
    write_pool(os.path.join(pool, "pos_bench.jsonl"), [{"id": b["id"], "text": b["text"]} for b in bp])
    meta["pos_bench"] = {b["id"]: b["bench"] for b in bp}

    # partial copies: slices of training windows + disjoint fresh Wikinews words
    fresh_words = []
    order = list(range(len(wn)))
    rng.shuffle(order)
    for i in order:
        fresh_words += wn[i]["text"].split()
    pos = 0
    meta["partial"] = {}
    for src, rows in (("rl", [r["text"] for r in rl_rows[300:]]), ("sft", [r["text"] for r in sft_rest])):
        rows = [r for r in rows if len(r.split()) >= 100]
        rng.shuffle(rows)
        for frac in (0.3, 0.6):
            docs = []
            for j in range(40):
                w = rows[len(docs) + (0 if frac == 0.3 else 40)].split()
                c = rng.randint(100, min(200, len(w)))
                a0 = rng.randint(0, len(w) - c)
                copied = w[a0: a0 + c]
                nf = int(round(c * (1 - frac) / frac))
                if pos + nf > len(fresh_words):
                    pos = 0                                  # (would reuse fresh words; does not happen with the default sizes)
                fresh = fresh_words[pos: pos + nf]
                pos += nf
                cut = rng.randint(0, nf)
                words = fresh[:cut] + copied + fresh[cut:]
                did = f"part_{src}_{int(frac * 100)}_{j}"
                docs.append({"id": did, "text": " ".join(words)})
                meta["partial"][did] = {"src": src, "frac": frac, "copied_words": len(normalise(" ".join(copied)).split()), "copied_text": " ".join(copied),
                                        "doc_words": len(normalise(" ".join(words)).split())}
            write_pool(os.path.join(pool, f"partial_{src}_{int(frac * 100)}.jsonl"), docs)

    # duplicates
    base = wn[:40]
    dd = []
    for i, w in enumerate(base):
        ws = w["text"].split()
        dd += [{"id": f"dup_orig_{i}", "text": w["text"]},
               {"id": f"dup_full_{i}", "text": w["text"]},
               {"id": f"dup_p60_{i}", "text": " ".join(ws[: int(len(ws) * 0.6)])},
               {"id": f"dup_p30_{i}", "text": " ".join(ws[: int(len(ws) * 0.3)])}]
    os.makedirs(os.path.join(a.work, "pool_dup"), exist_ok=True)
    write_pool(os.path.join(a.work, "pool_dup", "dup.jsonl"), dd)          # its own pool dir / scan: the originals also sit in neg_wikinews
    with open(os.path.join(a.work, "controls.json"), "w") as f:
        json.dump(meta, f)
    print("controls written to", pool)


def evaluate(a: argparse.Namespace) -> None:
    import numpy as np
    meta = json.load(open(os.path.join(a.work, "controls.json")))
    rows: Dict[str, Dict[str, Any]] = {}
    byfile: Dict[str, List[str]] = {}
    for p in sorted(glob.glob(os.path.join(a.work, "out", "*.flags.jsonl")) + glob.glob(os.path.join(a.work, "out_dup", "*.flags.jsonl"))):
        stem = os.path.basename(p)[:-len(".flags.jsonl")]
        for r in L.iter_jsonl(p):
            rows[r["id"]] = r
            byfile.setdefault(stem, []).append(r["id"])
    out: Dict[str, Any] = {}
    th = L.Thresholds()
    for r in rows.values():           # the controls share text with each other on purpose: judge them WITHOUT the duplicate rule (the dup control is judged on its own)
        r["flag_nodup"] = L.flag_row({**r, "dup_share": 0.0}, th)[0]

    def rate(ids, pred=lambda r: r["flag_nodup"] == "contaminated"):
        k = sum(1 for i in ids if pred(rows[i]))
        return {"n": len(ids), "hits": k, "rate": round(k / max(len(ids), 1), 4)}

    for stem in ("pos_rl", "pos_sft", "pos_bench"):
        ids = byfile.get(stem, [])
        if not ids:
            continue
        d = rate(ids)
        d["flag_counts_nodup"] = {k: sum(1 for i in ids if rows[i]["flag_nodup"] == k) for k in ("clean", "weak", "contaminated")}
        d["share_quantiles"] = {k: L.quantiles([rows[i].get(k, 0) for i in ids], (0.0, 0.05, 0.5))
                                for k in ("sft_share", "sft_word_share", "rl_share", "bench_share") if k in rows[ids[0]]}
        miss = [rows[i] for i in ids if rows[i]["flag_nodup"] != "contaminated"]
        reasons = {}
        for r in miss:
            nw, nt = r["n_ngrams_word"], r.get("n_ngrams_tok", 1)
            if nw == 0 or nt == 0:
                why = "too short (no 8-word / 12-token window)"
            else:
                why = f"shares sft={r.get('sft_share')}/sftw={r.get('sft_word_share')}/rl={r.get('rl_share')}/bench={r.get('bench_share')}"
            reasons.setdefault(why, []).append(r["id"])
        d["miss_reasons"] = {k: {"n": len(v), "ids": v[:6]} for k, v in reasons.items()}
        # detection by the path that is expected to catch it
        key = {"pos_rl": "rl_share", "pos_sft": "sft_share", "pos_bench": "bench_share"}[stem]
        d["detected_by_expected_source_share>=0.5"] = rate(ids, lambda r: (r.get(key) or 0) >= 0.5)
        if stem == "pos_sft":
            d["detected_by_sft_word_share>=0.5"] = rate(ids, lambda r: (r.get("sft_word_share") or 0) >= 0.5)
            for lo, hi in ((0, 12), (12, 40), (40, 10 ** 9)):
                d[f"windows_{lo}_to_{hi if hi < 10 ** 9 else 'inf'}_tokens"] = rate([i for i in ids if lo <= meta["pos_sft_tokens"][i] < hi])
        if stem == "pos_rl":
            bins = {}
            for i in ids:
                bins.setdefault(meta["pos_rl_bins"][i], []).append(i)
            d["by_length_bin"] = {b: rate(v) for b, v in sorted(bins.items())}
        if stem == "pos_bench":
            ok = sum(1 for i in ids if meta["pos_bench"][i] in rows[i]["bench_hit_benchmarks"])
            d["named_the_right_benchmark"] = {"n": len(ids), "hits": ok}
        out[stem] = d
    for stem in ("neg_wikinews", "neg_gutenberg"):
        ids = byfile.get(stem, [])
        if ids:
            d = rate(ids, lambda r: r["flag_nodup"] == "clean")
            d["flag_counts_nodup"] = {k: sum(1 for i in ids if rows[i]["flag_nodup"] == k) for k in ("clean", "weak", "contaminated")}
            d["weak_or_contaminated_ids"] = [i for i in ids if rows[i]["flag_nodup"] != "clean"][:12]
            d["share_quantiles"] = {k: L.quantiles([rows[i].get(k, 0) for i in ids], (0.5, 0.9, 0.95, 0.99, 1.0))
                                    for k in ("sft_share", "sft_word_share", "rl_share", "bench_share")}
            out[stem] = d
    part = {}
    tk = L.load_tokenizers_json(a.tokenizer)
    for src in ("rl", "sft"):
        for f in (30, 60):
            ids = byfile.get(f"partial_{src}_{f}", [])
            if not ids:
                continue
            key = "rl_share" if src == "rl" else "sft_share"
            n = L.WORD_NGRAM if src == "rl" else L.TOKEN_NGRAM
            nominal, expected = [], []
            for i in ids:
                m = meta["partial"][i]
                if src == "rl":                      # unit = normalised words
                    c, d = m["copied_words"], rows[i]["n_words"]
                else:                                # unit = Gemma tokens (the copied slice tokenised alone)
                    c, d = len(tk.encode(" ".join(m["copied_text"].split()), add_special_tokens=False).ids), rows[i]["n_tokens"]
                nominal.append(c / d)
                expected.append(max(c - n + 1, 0) / max(d - n + 1, 1))
            meas = [rows[i][key] for i in ids]
            diff = [m_ - e for m_, e in zip(meas, expected)]
            part[f"{src}_{f}"] = {"n": len(ids), "unit": "words" if src == "rl" else "tokens", "copied_fraction_mean": round(float(np.mean(nominal)), 4),
                                  "expected_share_mean": round(float(np.mean(expected)), 4), "measured_mean": round(float(np.mean(meas)), 4),
                                  "measured_min": round(min(meas), 4), "measured_max": round(max(meas), 4),
                                  "mean_abs_error_vs_expected": round(float(np.mean(np.abs(diff))), 4), "max_abs_error_vs_expected": round(float(np.max(np.abs(diff))), 4),
                                  "flags_nodup": {k: sum(1 for i in ids if rows[i]["flag_nodup"] == k) for k in ("clean", "weak", "contaminated")}}
            if src == "sft":
                part[f"{src}_{f}"]["sft_word_share_mean"] = round(float(np.mean([rows[i].get("sft_word_share", 0) for i in ids])), 4)
    out["partial"] = part
    ids = byfile.get("dup", [])
    if ids:
        d = {}
        for kind in ("orig", "full", "p60", "p30"):
            kid = [i for i in ids if i.startswith(f"dup_{kind}_")]
            d[kind] = {"n": len(kid), "flags": L.summarise_rows([rows[i] for i in kid])["flags"],
                       "dup_share_mean": round(float(np.mean([rows[i]["dup_share"] for i in kid])), 3),
                       "dup_of_set": sum(1 for i in kid if rows[i]["dup_of"]),
                       "dup_of_is_the_original": sum(1 for i in kid if rows[i]["dup_of"] == "dup_orig_" + i.split("_")[-1])}
        out["dup"] = d
    with open(os.path.join(a.work, "validation.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "evaluate"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--raw-dir", default="/workspace/sft_build/longdoc_bench/raw/contam_neg")
    ap.add_argument("--rl-dir", default="/workspace/sft_build/gemma/rl_prompts_v2")
    ap.add_argument("--sft-dir", default="/workspace/sft_build/gemma/packed_fwd")
    ap.add_argument("--bench-root", default="/workspace/sft_build/bench/test_sets")
    ap.add_argument("--tokenizer", default="/workspace/sft_build/hf/hub/models--google--gemma-3-270m-it/snapshots/*/tokenizer.json")
    ap.add_argument("--seed", type=int, default=20261020)
    a = ap.parse_args()
    (build if a.cmd == "build" else evaluate)(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
