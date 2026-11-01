#!/usr/bin/env python3
"""(b) Replay component from D2's CLEAN packed SFT data (agent C2, 2026-10-28).

    python s4_replay.py --pack-dir PACKED --out OUT [--windows-dir WINDOWS_SFT] [--science-tokens N | --replay-tokens N] [--weak-langs a,b,..] [--weak-scale 0.5]
                        --index protected_index.npz --manifest protected_manifest.json

`--pack-dir` is the output of `chain/pack.py prepare` (a packed dataset with `train/`, `val/`, `meta.json`; the GATED packs once the QE gate has run, the ungated ones for a dry run).
It is read exactly like the old builder read the original tarball: the per-sample arrays (iso_id, sample_len, sample_offset, prompt_len, target_len) and the token file; the English
and the target text are decoded from the tokens (the format is lossless; a re-encode check on a random sample is written to the stats).

Recipe of the A2W replay (scripts/build_sft_science_mix.py + mix2_build.py --weak-replay-scale 0.5):
  1. UniMax water filling over the languages: every language gets the same token budget c (a language with less data uses all of it, at most 1 epoch), the six science languages
     get --sci-weight (2) times the budget of an ordinary language;
  2. samples are drawn at random inside a language without replacement (the natural mixture of window lengths), oversampled by 1.3 to replace rejected ones;
  3. strict protected-index check (decon.py, no exemption) on the English AND the target side of every candidate window: a window with ANY hit is dropped, never repaired;
  4. the languages in --weak-langs (the 45 weak-signal languages of A2W) keep only --weak-scale (0.5) of their windows (seeded, by tokens): the replay of the old SFT is JW-heavy
     and pulled the RL policy back, hence the halving (D-63).
The total is sized so that the replay is 43.0 / 31.7 of the science tokens (A2W shares) AFTER step 4; `--replay-tokens` overrides.
`--windows-dir` (windows parquet of chain/windows.py) is an optional side table: a pack sample carries no source/licence, so source_dataset, licence, QE and length bin are joined from
the window with the same (language, English text); the join rate is in the stats. Rows without a match are labelled `unknown(pack-only)` in the licence columns.
Output: OUT/replay_train.jsonl, OUT/replay_val.jsonl (<= --val-per-language windows per language from the pack's val split: the 77-language forgetting monitor), OUT/replay_stats.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s4_common as K  # noqa: E402


def load_split(d):
    g = lambda n: np.load(os.path.join(d, n + ".npy"), mmap_mode="r")
    toks = np.memmap(os.path.join(d, "tokens.u32.bin"), dtype=np.uint32, mode="r")
    return {"toks": toks, "iso": np.asarray(g("iso_id")), "len": np.asarray(g("sample_len")), "off": np.asarray(g("sample_offset")), "pl": np.asarray(g("prompt_len")),
            "tl": np.asarray(g("target_len"))}


def side_table(windows_dir):
    """(iso, xxh64(nfc english)) -> provenance dict from the windows parquet of chain/windows.py."""
    import glob

    import pyarrow.parquet as pq
    import xxhash

    tab = {}
    cols = ["english", "iso", "source_dataset", "licence", "qe", "length_bin", "n_pairs"]
    for f in sorted(glob.glob(os.path.join(windows_dir, "*.parquet"))):
        t = pq.read_table(f, columns=cols).to_pydict()
        for i in range(len(t["iso"])):
            k = (t["iso"][i], xxhash.xxh64_intdigest(K.nfc(t["english"][i]).encode("utf-8", "surrogatepass")))
            tab.setdefault(k, (t["source_dataset"][i], t["licence"][i], t["qe"][i], t["length_bin"][i], t["n_pairs"][i], os.path.basename(f)))
    return tab


def row_prov(table, iso, english):
    import xxhash

    k = (iso, xxhash.xxh64_intdigest(K.nfc(english).encode("utf-8", "surrogatepass")))
    hit = table.get(k) if table is not None else None
    if hit:
        sd, lic, qe, lb, npairs, f = hit
        return K.provenance("replay", source_dataset=sd or "", licence=lic or "", prov_ref=f"windows:{f}", generator="human reference (clean SFT pairs)"), {"qe": qe, "length_bin": lb, "n_pairs": npairs}, True
    return K.provenance("replay", source_dataset="unknown(pack-only)", licence="", licence_class="unknown", prov_ref="pack-only (no windows side table match)",
                        generator="human reference (clean SFT pairs)"), {}, False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows-dir", default="")
    ap.add_argument("--science-tokens", type=float, default=float(K.A2W_TOKENS["science"]))
    ap.add_argument("--replay-tokens", type=float, default=0.0, help="override the total replay tokens (after the weak-language halving)")
    ap.add_argument("--sci-weight", type=float, default=2.0)
    ap.add_argument("--weak-langs", default=",".join(K.WEAK45))
    ap.add_argument("--weak-scale", type=float, default=0.5)
    ap.add_argument("--max-epochs", type=float, default=1.0)
    ap.add_argument("--min-examples", type=int, default=600)
    ap.add_argument("--oversample", type=float, default=1.3)
    ap.add_argument("--val-per-language", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--index", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--reencode-check", type=int, default=300)
    a = ap.parse_args(argv)
    K.gemma_pkg()
    from gemma_mt.data.prepare_sft import tokenizer_fingerprint
    from gemma_mt.languages import ALL_ISOS, iso_to_id

    meta = json.load(open(os.path.join(a.pack_dir, "meta.json")))
    tok = K.tokenizer(a.tokenizer or meta["tokenizer"]["requested"])
    if tokenizer_fingerprint(tok) != meta["tokenizer"]["fingerprint"]:
        raise SystemExit("the tokenizer fingerprint differs from the pack's")
    if meta["iso_table"] != list(ALL_ISOS):
        raise SystemExit("the iso table of the pack differs from gemma_mt.languages.ALL_ISOS")
    tr = load_split(os.path.join(a.pack_dir, "train"))
    n_langs = len(ALL_ISOS)
    avail = np.bincount(tr["iso"], weights=tr["len"].astype(np.float64), minlength=n_langs)
    langs = [i for i in range(n_langs) if avail[i] > 0]
    weak = {x for x in a.weak_langs.split(",") if x}
    ws = a.weak_scale
    wts = np.array([a.sci_weight if ALL_ISOS[i] in K.SCI_LANGS else 1.0 for i in langs])
    target_total = a.replay_tokens or a.science_tokens * K.A2W_SHARES["replay"] / K.A2W_SHARES["science"]
    pre = target_total
    for _ in range(6):                                           # pre-total such that the total AFTER the weak halving is the target
        b = np.zeros(n_langs)
        b[langs] = K.unimax_budgets(avail[langs], pre, a.max_epochs, wts)
        wk = sum(b[i] for i in langs if ALL_ISOS[i] in weak)
        f = wk / max(b.sum(), 1)
        pre = target_total / max(1 - (1 - ws) * f, 1e-6)
    bud = np.zeros(n_langs)
    bud[langs] = K.unimax_budgets(avail[langs], pre, a.max_epochs, wts)
    K.log(f"replay target {target_total:,.0f} tokens after halving; pre-halving budget {pre:,.0f}; {len(langs)} languages in the pack; per-language level "
          f"min {bud[langs].min():,.0f} max {bud[langs].max():,.0f}")
    # ---- plan: seeded permutation per language, cut after budget x oversample tokens
    order = np.argsort(tr["iso"], kind="stable")
    bounds = np.searchsorted(tr["iso"][order], np.arange(n_langs + 1))
    plan = {}
    for l in langs:
        idx = order[bounds[l]:bounds[l + 1]]
        perm = idx[np.random.default_rng([a.seed, l]).permutation(len(idx))]
        cum = np.cumsum(tr["len"][perm].astype(np.int64))
        k = max(int(np.searchsorted(cum, bud[l] * a.oversample)) + 1, int(math.ceil(a.min_examples * a.oversample)))
        plan[l] = perm[:min(k, len(perm))]
    table = side_table(a.windows_dir) if a.windows_dir else None
    P = K.Protected(a.index, a.manifest, a.nproc)
    K.log(f"protected index {P.version} loaded (strict, no exemption)")
    stats = {"hit": collections.Counter(), "hit_windows": 0, "hit_windows_only_dev_sets": 0, "undecodable": 0, "forbidden_source": 0, "joined": 0, "no_join": 0, "short_of_budget": [], "weak_dropped_tokens": collections.Counter()}
    rows, rng = [], np.random.default_rng([a.seed, 7777])
    sample_for_check = []
    CH = 2000
    for l in langs:
        iso = ALL_ISOS[l]
        cand = []
        for i in plan[l]:
            o, n, pl, tl = int(tr["off"][i]), int(tr["len"][i]), int(tr["pl"][i]), int(tr["tl"][i])
            ids = np.asarray(tr["toks"][o:o + n])
            en = K.english_of_prompt(tok, ids, pl)
            tg = K.target_of_sample(tok, ids, pl, tl, iso)
            if en is None or tg is None:
                stats["undecodable"] += 1
                continue
            cand.append((i, ids, en, tg, n, tl))
        hits = P.names([c[2] for c in cand] + [c[3] for c in cand])
        h_en, h_tg = hits[:len(cand)], hits[len(cand):]
        got, kept = 0, []
        for c, he, ht in zip(cand, h_en, h_tg):
            if he or ht:
                stats["hit_windows"] += 1
                if all(x.endswith(":dev") for x in set(he) | set(ht)):           # only dev halves (FLORES+ dev train half etc.), no test set
                    stats["hit_windows_only_dev_sets"] += 1
                for nm in set(he) | set(ht):
                    stats["hit"][nm] += 1
                continue
            if got >= bud[l] and len(kept) >= a.min_examples:
                break
            prov, extra, ok = row_prov(table, iso, c[2])
            stats["joined" if ok else "no_join"] += 1
            if K.is_forbidden_source(prov["source_dataset"]):
                stats["forbidden_source"] += 1
                continue
            kept.append({"english": c[2], "target": c[3], "iso": iso, "n_tokens": c[4], "target_tokens": c[5], **extra, **prov})
            got += c[4]
            if len(sample_for_check) < a.reencode_check and rng.random() < 0.05:
                sample_for_check.append((c[1], c[2], iso, c[3]))
        if got < 0.98 * bud[l] and got < avail[l] * a.max_epochs * 0.98:
            stats["short_of_budget"].append(iso)
        if iso in weak and ws < 1.0:                              # weak-language halving, seeded, window level
            keep2 = []
            for r in kept:
                if rng.random() <= ws:
                    keep2.append(r)
                else:
                    stats["weak_dropped_tokens"][iso] += r["n_tokens"]
            kept = keep2
        rows += kept
    # ---- integrity: decode -> re-encode must reproduce the pack's tokens
    bad = 0
    for ids, en, iso, tg in sample_for_check:
        ex = K.example(tok, en, iso, tg)
        if list(ex.input_ids) != [int(x) for x in ids]:
            bad += 1
    # ---- validation split (77-language forgetting monitor): the pack's val/ split, <= val-per-language per language, same strict check
    val_rows = []
    vd = os.path.join(a.pack_dir, "val")
    if os.path.isdir(vd):
        va = load_split(vd)
        for l in langs:
            idx = np.flatnonzero(va["iso"] == l)
            idx = idx[np.random.default_rng([a.seed, 99, l]).permutation(len(idx))][:int(a.val_per_language * 1.5)]
            cand = []
            for i in idx:
                o, n, pl, tl = int(va["off"][i]), int(va["len"][i]), int(va["pl"][i]), int(va["tl"][i])
                ids = np.asarray(va["toks"][o:o + n])
                en, tg = K.english_of_prompt(tok, ids, pl), K.target_of_sample(tok, ids, pl, tl, ALL_ISOS[l])
                if en is not None and tg is not None:
                    cand.append((en, tg))
            hh = P.names([c[0] for c in cand] + [c[1] for c in cand])
            for k, c in enumerate(cand):
                if hh[k] or hh[len(cand) + k]:
                    continue
                if sum(1 for r in val_rows if r["iso"] == ALL_ISOS[l]) < a.val_per_language:
                    val_rows.append({"english": c[0], "target": c[1], "iso": ALL_ISOS[l]})
    P.close()
    K.write_jsonl(os.path.join(a.out, "replay_train.jsonl"), rows)
    K.write_jsonl(os.path.join(a.out, "replay_val.jsonl"), val_rows)
    tok_by = collections.Counter()
    for r in rows:
        tok_by[r["iso"]] += r["n_tokens"]
    out = {"index_version": P.version, "pack_dir": a.pack_dir, "tokenizer_fingerprint": meta["tokenizer"]["fingerprint"], "target_total_tokens": target_total, "pre_halving_budget": pre, "rows": len(rows), "tokens": sum(tok_by.values()),
           "tokens_by_lang": dict(tok_by), "a2w_reference_tokens": K.A2W_TOKENS["replay"], "a2w_reference_windows": 47141, "protected_hit_windows_removed": stats["hit_windows"], "protected_hit_windows_dev_sets_only": stats["hit_windows_only_dev_sets"], "protected_hits_removed_by_set": dict(stats["hit"]),
           "undecodable": stats["undecodable"], "forbidden_source": stats["forbidden_source"], "provenance_join": {"joined": stats["joined"], "no_match": stats["no_join"]},
           "short_of_budget_languages": stats["short_of_budget"], "weak_languages": sorted(weak), "weak_scale": ws, "weak_dropped_tokens": sum(stats["weak_dropped_tokens"].values()),
           "reencode_check": {"n": len(sample_for_check), "mismatches": bad}, "val_rows": len(val_rows), "options": {k: v for k, v in vars(a).items() if k not in ("weak_langs",)}}
    json.dump(out, open(os.path.join(a.out, "replay_stats.json"), "w"), indent=1)
    K.log(f"replay: {len(rows)} windows, {out['tokens']:,} tokens (A2W reference {K.A2W_TOKENS['replay']:,} / 47,141 windows); protected hits removed {sum(stats['hit'].values())}; "
          f"join {stats['joined']}/{stats['joined'] + stats['no_join']}; re-encode mismatches {bad}/{len(sample_for_check)}; val {len(val_rows)}")


if __name__ == "__main__":
    main()
