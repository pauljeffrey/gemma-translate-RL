#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mixture v2 = mix_v1 (science + replay, UNCHANGED samples) + archive (A) + long-document distillation (B) + weak-language extras (C), re-packed with R4's tooling.

    python scripts/mix2_build.py --out /workspace/jobs/r11/mix_v2 --archive work/arch/archive_selected.jsonl --longdoc work/ld/ld_selected.jsonl [--weak FILE.jsonl]

* the samples of mix_v1/train are read back from its packs (token ids, prompt / target lengths, language) and classified science vs replay by identity with the cached replay
  pool (`replay_pool_*.npz`); the counts must reproduce mix_v1's tables (29,556 science + 67,383 replay samples) or the build stops;
* new components: rows {english, target, iso, ...} -> `format.build_example` (the SFT format of the original data: prompt `Translate the text to X:  <english>`, target `iso <translation><end_of_turn>`),
  hold-out re-checked (the SAME hold-out object as mix_v1, exact + word 8-gram) and exact-duplicate-checked against the whole mixture; samples longer than the pack length are dropped;
* val/ and val_sci/ are mix_v1's (copied: the validation sets are unchanged, so the numbers are comparable with R4's step-0 baselines);
* writes train/ (packed by `prepare_sft._write_split`, one extra file `source.npy`: 0 science, 1 replay, 2 archive, 3 longdoc, 4 weak-extra, per packed sample), meta.json (v1's meta with
  the new train split + a `mixture_v2` block), mixture_tables_v2.{json,md}.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mix2_common as C  # noqa: E402

SOURCES = ("science", "replay", "archive", "longdoc", "anchor", "r10_extra", "r14_teacher", "r10_share_alike", "r10_code_switch", "r10_nc")
ORIGIN_CODE = {"r14": 6, "r10/share_alike": 7, "r10/owner_code_switch": 8, "r10/NC_noncommercial": 9}


LIC_CLASSES = ("commercial_ok", "share_alike", "nc", "unknown", "nc_teacher", "owner")


def licence_class(r) -> int:
    """Row -> class code. R10's files carry `licence_class`; own components carry the licence string of the English SOURCE (RL prompt pools)."""
    c = (r.get("licence_class") or "").strip().lower()
    if c in LIC_CLASSES:
        return LIC_CLASSES.index(c)
    l = (r.get("licence") or "").strip().lower()
    if not l:
        return 3
    if "nc" in l.replace("-", "_").split("_") or "non" in l:
        return 2
    if "share" in l or "-sa" in l or "gfdl" in l:
        return 1
    if l.startswith(("cc-by", "cc0", "odc-by", "apache", "mit", "bsd", "public", "cc_by", "pddl", "unlicense")):
        return 0
    return 3


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def read_split(d):
    tok = np.memmap(os.path.join(d, "tokens.u32.bin"), dtype=np.uint32, mode="r")
    off = np.load(os.path.join(d, "sample_offset.npy"))
    ln = np.load(os.path.join(d, "sample_len.npy"))
    pl = np.load(os.path.join(d, "prompt_len.npy"))
    tl = np.load(os.path.join(d, "target_len.npy"))
    iso = np.load(os.path.join(d, "iso_id.npy"))
    return tok, off, ln, pl, tl, iso


def h(a: np.ndarray) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(a, dtype=np.uint32).tobytes(), digest_size=12).digest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v1", default="/workspace/jobs/r4/mix_v1")
    ap.add_argument("--archive", default="")
    ap.add_argument("--longdoc", default="")
    ap.add_argument("--anchor", default="", help="jsonl of self-distilled short/medium windows of the weak languages (mix2_longdoc_select.py on the anchor documents)")
    ap.add_argument("--weak", default="", help="extras jsonl of mix2_extras.py (R10 windows + R14 teacher windows; column origin r10/... or r14)")
    ap.add_argument("--weak-langs", default="/workspace/jobs/r11/work/weak_langs.txt")
    ap.add_argument("--weak-replay-scale", type=float, default=1.0, help="<1: keep only this share (by tokens, seeded) of the HUMAN-REFERENCE replay windows of the weak languages (variant mix_v2b)")
    ap.add_argument("--model", default="/workspace/sft_build/runs/final_1433")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--exclude-classes", default="nc,unknown,nc_teacher", help="licence classes dropped from the NEW components (owner 2026-10-22: SFT data must be commercial-clean; the replay tarball has no per-row licence and is assumed clean)")
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from gemma_mt import format as fmt
    from gemma_mt.data.prepare_sft import tokenizer_fingerprint
    from gemma_mt.languages import ALL_ISOS, LANGUAGES, iso_to_id

    B = C.load_script("build_sft_science_mix")
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    fmt.assert_template_matches_reference(tok)
    fp = tokenizer_fingerprint(tok)
    meta1 = json.load(open(os.path.join(a.v1, "meta.json")))
    if meta1["tokenizer"]["fingerprint"] != fp:
        raise SystemExit("tokenizer fingerprint of mix_v1 differs from the model's")
    ho = C.make_holdout()

    # ------------------------------------------------ mix_v1 train samples, classified
    t_ids, t_off, t_len, t_pl, t_tl, t_iso = read_split(os.path.join(a.v1, "train"))
    pool = glob.glob(os.path.join(a.v1, "replay_pool_*.npz"))
    if len(pool) != 1:
        raise SystemExit(f"expected one replay pool in {a.v1}")
    z = np.load(pool[0], allow_pickle=True)
    rep_hash = {h(t) for t in z["toks"]}
    items = []                                   # (ids, prompt_len, target_len, iso_id, source_code)
    cnt = collections.Counter()
    for i in range(len(t_len)):
        ids = np.asarray(t_ids[int(t_off[i]): int(t_off[i]) + int(t_len[i])])
        src = 1 if h(ids) in rep_hash else 0
        cnt[src] += 1
        items.append((ids, int(t_pl[i]), int(t_tl[i]), int(t_iso[i]), src, 0 if src == 0 else 3))   # science: Apache-2.0; replay: the tarball has NO per-row licence -> unknown
    log(f"mix_v1 train: {len(items)} samples; science {cnt[0]}, replay {cnt[1]}")
    t1 = meta1["splits"]["train"]
    if len(items) != t1["n_samples"]:
        raise SystemExit("sample count of mix_v1 train differs from its meta")
    exp_rep = sum(1 for _ in z["toks"])
    log(f"replay pool has {exp_rep} candidate samples; classified replay {cnt[1]} (mix_v1 tables: 67,383)")
    amb = cnt[1] - 67383
    if amb < 0 or cnt[0] + cnt[1] != 96939 or cnt[0] != 29556 - amb:
        raise SystemExit("classification does not reproduce the mix_v1 tables (29,556 science / 67,383 replay)")
    log(f"{amb} science samples have token content IDENTICAL to a replay-pool window (same pair in both sources) and count as replay here; science {cnt[0]} / replay {cnt[1]}")
    if a.weak_replay_scale < 1.0:
        weak = set(open(a.weak_langs).read().split(","))
        weak_ids = {iso_to_id(x.strip()) for x in weak if x.strip()}
        rng = np.random.default_rng(a.seed)
        keep_items, dropped = [], collections.Counter()
        for it in items:
            if it[4] == 1 and it[3] in weak_ids and rng.random() > a.weak_replay_scale:
                dropped[ALL_ISOS[it[3]]] += it[0].shape[0]
                continue
            keep_items.append(it)
        log(f"weak replay scale {a.weak_replay_scale}: dropped {sum(dropped.values()):,} replay tokens of {len(dropped)} weak languages")
        items = keep_items
    seen = {h(it[0]) for it in items}

    # ------------------------------------------------ new components
    excl_classes = set(x for x in a.exclude_classes.split(",") if x)
    comp_info = {}
    for code, path in ((2, a.archive), (3, a.longdoc), (4, a.anchor), (5, a.weak)):
        if not path or not os.path.exists(path):
            continue
        name = SOURCES[code]
        st = collections.Counter()
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("target") or not r.get("english"):
                st["empty"] += 1
                continue
            lc = licence_class(r)
            if LIC_CLASSES[lc] in excl_classes:
                st["licence_excluded:" + LIC_CLASSES[lc]] += 1
                st["licence_excluded_tokens"] += int(r.get("n_tokens") or 0)
                continue
            hit = ho.hit(r["english"])
            if hit:
                st["holdout_" + hit] += 1
                continue
            ex = fmt.build_example(r["english"], r["iso"], r["target"], tok)
            if ex.n_tokens > a.seq_len:
                st["longer_than_pack"] += 1
                continue
            hh = h(np.asarray(ex.input_ids))
            if hh in seen:
                st["duplicate"] += 1
                continue
            seen.add(hh)
            c2 = ORIGIN_CODE.get(r.get("origin"), code) if code == 5 else code
            items.append((np.asarray(ex.input_ids, dtype=np.uint32), ex.prompt_len, ex.target_len, iso_to_id(r["iso"]), c2, licence_class(r)))
            st["kept"] += 1
        comp_info[name] = dict(st)
        log(f"{name}: {dict(st)}")

    # ------------------------------------------------ write train (R4's writer)
    train_items = [(SimpleNamespace(input_ids=it[0], prompt_len=it[1], target_len=it[2]), it[3]) for it in items]
    code_of = {h(it[0]): (it[4], it[5]) for it in items}
    meta_train = B.write_split(train_items, a.out, a.seq_len, a.seed, "train")
    tok2, off2, len2, pl2, tl2, iso2 = read_split(os.path.join(a.out, "train"))
    src = np.zeros(len(len2), dtype=np.uint8)
    lic = np.zeros(len(len2), dtype=np.uint8)
    for i in range(len(len2)):
        src[i], lic[i] = code_of[h(np.asarray(tok2[int(off2[i]): int(off2[i]) + int(len2[i])]))]
    np.save(os.path.join(a.out, "train", "source.npy"), src)
    np.save(os.path.join(a.out, "train", "licence_class.npy"), lic)

    # ------------------------------------------------ tables
    tab = {s: collections.defaultdict(lambda: {"examples": 0, "tokens": 0, "target_tokens": 0}) for s in SOURCES}
    for i in range(len(len2)):
        d = tab[SOURCES[int(src[i])]][ALL_ISOS[int(iso2[i])]]
        d["examples"] += 1
        d["tokens"] += int(len2[i])
        d["target_tokens"] += int(tl2[i])
    isos = [i for i in ALL_ISOS if i in meta1["splits"]["train"]["per_language"] or any(i in tab[s] for s in SOURCES)]
    tot = {s: sum(v["tokens"] for v in tab[s].values()) for s in SOURCES}
    allt = sum(tot.values())
    json.dump({s: {k: v for k, v in tab[s].items()} for s in SOURCES}, open(os.path.join(a.out, "mixture_tables_v2.json"), "w"), indent=1)
    L = ["| tag | language | " + " | ".join(f"{s} ex | {s} tok" for s in SOURCES) + " | total tok | non-original share |", "|---|---|" + "---:|" * (2 * len(SOURCES) + 2)]
    for iso in isos:
        row = [tab[s].get(iso, {"examples": 0, "tokens": 0}) for s in SOURCES]
        t = sum(r["tokens"] for r in row)
        new = sum(r["tokens"] for r in row[2:])
        L.append(f"| {iso} | {LANGUAGES[iso].name} | " + " | ".join(f"{r['examples']:,} | {r['tokens']:,}" for r in row) + f" | {t:,} | {100 * new / max(t, 1):.1f}% |")
    L.append("| **all** | | " + " | ".join(f"{sum(v['examples'] for v in tab[s].values()):,} | {tot[s]:,}" for s in SOURCES) + f" | {allt:,} | {100 * sum(tot[s] for s in SOURCES[2:]) / allt:.1f}% |")
    ctab = {c: collections.defaultdict(int) for c in LIC_CLASSES}
    for i in range(len(len2)):
        ctab[LIC_CLASSES[int(lic[i])]][ALL_ISOS[int(iso2[i])]] += int(len2[i])
    ctot = {c: sum(v.values()) for c, v in ctab.items()}
    L += ["", "Per-language tokens by LICENCE CLASS of the training rows (replay = unknown: the original SFT tarball has no per-row licence; science = Apache-2.0; archive / longdoc / anchor = licence of the English source text in the RL prompt pools; R10 rows = their own licence_class)", "",
          "| tag | language | " + " | ".join(LIC_CLASSES) + " | commercial_ok share excl. replay |", "|---|---|" + "---:|" * (len(LIC_CLASSES) + 1)]
    for iso in isos:
        row = [ctab[c].get(iso, 0) for c in LIC_CLASSES]
        rep_t = tab["replay"].get(iso, {"tokens": 0})["tokens"]
        non_rep = sum(row) - rep_t
        L.append(f"| {iso} | {LANGUAGES[iso].name} | " + " | ".join(f"{x:,}" for x in row) + f" | {100 * row[0] / max(non_rep, 1):.0f}% |")
    L.append("| **all** | | " + " | ".join(f"{ctot[c]:,}" for c in LIC_CLASSES) + " | |")
    json.dump({c: dict(v) for c, v in ctab.items()}, open(os.path.join(a.out, "licence_class_tables.json"), "w"), indent=1)
    open(os.path.join(a.out, "mixture_tables_v2.md"), "w").write("\n".join(L) + "\n")

    # ------------------------------------------------ validation splits (unchanged) + meta
    for sp in ("val", "val_sci"):
        shutil.rmtree(os.path.join(a.out, sp), ignore_errors=True)
        shutil.copytree(os.path.join(a.v1, sp), os.path.join(a.out, sp))
    meta = dict(meta1)
    meta["splits"] = dict(meta1["splits"])
    meta["splits"]["train"] = meta_train
    meta["mixture_v2"] = {"built_from": a.v1, "components": comp_info, "tokens_by_source": tot, "share_by_source": {s: tot[s] / allt for s in SOURCES}, "seq_len": a.seq_len,
                          "seed": a.seed, "inputs": {"archive": a.archive, "longdoc": a.longdoc, "anchor": a.anchor, "weak": a.weak}, "weak_replay_scale": a.weak_replay_scale, "holdout": dict(ho.counts), "source_codes": list(SOURCES), "licence_classes": list(LIC_CLASSES), "tokens_by_licence_class": ctot}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1, ensure_ascii=False)
    shutil.copy(os.path.join(a.v1, "replay_fractions.json"), os.path.join(a.out, "replay_fractions.json")) if os.path.exists(os.path.join(a.v1, "replay_fractions.json")) else None
    tr = meta_train
    log(f"TRAIN v2: {tr['n_samples']:,} samples, {tr['n_packs']:,} packs of {a.seq_len}, {tr['n_tokens']:,} tokens ({tr['target_tokens']:,} target), efficiency {tr['packing_efficiency']:.4f}; "
        f"shares {', '.join(f'{s} {tot[s] / allt:.3f}' for s in SOURCES)}")


if __name__ == "__main__":
    main()
