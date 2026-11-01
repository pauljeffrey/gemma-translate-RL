#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build calibration/anchors_v2.json (D-61): the v1 anchors calibration + per-language blocks `entity2`, `copy`, `func`, `pad` fitted on HUMAN references.

    python scripts/build_anchors_v2_calibration.py --v1 calibration/anchors_v1.json --human human_units.jsonl --funcwords calibration/funcwords_v1.json \
        --bigrams /path/bigrams_v1.npz --out calibration/anchors_v2.json

Inputs: `human_units.jsonl` (the units of scripts/build_anchors_calibration.py --dump-human: iso, src, ref, set, id; sets smol_doc / *_doc.jsonl are whole documents and only feed the pseudo-documents),
`funcwords_v1.json` (scripts/build_funcwords.py: word statistics of the TARGET-language training text) and optionally the bigram table (npz) of the same text. The v1 file is copied unchanged
(blocks of v1 are never edited); every language gets the new blocks next to them. Measurements per language (FP, detection, weights) are stored in the blocks and printed as a table.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemma_mt.rl import anchors_attacks as AT  # noqa: E402
from gemma_mt.rl import anchors_calibrate_v2 as C  # noqa: E402
from gemma_mt.rl import textstats as T  # noqa: E402

DOC_SETS = ("smol_doc",)


def fold_of(u: dict) -> int:
    import zlib

    return zlib.crc32((u["iso"] + "|" + " ".join(u["src"].split()).casefold()).encode("utf-8")) & 1


def load_humans(path: str, fold: str = "all"):
    sents: Dict[str, List[dict]] = collections.defaultdict(list)
    seen = set()
    n_docs = 0
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if r["set"] in DOC_SETS or r["set"].endswith("_doc.jsonl"):
            n_docs += 1
            continue
        k = (r["iso"], " ".join(r["src"].split()).casefold(), " ".join(r["ref"].split()).casefold())
        if k in seen or (fold != "all" and fold_of(r) != (0 if fold == "A" else 1)):
            continue
        seen.add(k)
        sents[r["iso"]].append(r)
    return sents, n_docs


def load_pivot(corpus: str, isos, fold: str):
    """Languages without human references (13 of the 45 weak ones): the HELD-OUT rows of agent R3's English-pivot corpus (`held` True: 300-600 whole sentences per language never used to train a lexicon;
    mostly religious text) are the only human-written pairs there are. Provisional calibration: thresholds come from that genre."""
    import pyarrow.parquet as pq

    out = {}
    for iso in isos:
        fn = os.path.join(corpus, iso + ".parquet")
        if not os.path.exists(fn):
            continue
        t = pq.read_table(fn, columns=["en", "x", "held"]).to_pydict()
        us = []
        for en, x, h in zip(t["en"], t["x"], t["held"]):
            if h and en and x and len(en.split()) >= 3:
                u = {"iso": iso, "src": en.strip(), "ref": x.strip(), "set": "pivot_held", "id": str(len(us))}
                if fold == "all" or fold_of(u) == (0 if fold == "A" else 1):
                    us.append(u)
        if len(us) >= 60:
            out[iso] = us
    return out


def pseudo_docs(units: List[dict], ks=(4, 10, 25)) -> List[dict]:
    import anchors_common as AC

    return AC.pseudo_documents(units, ks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--human", required=True)
    ap.add_argument("--funcwords", required=True)
    ap.add_argument("--bigrams", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--fw-k", type=int, default=40)
    ap.add_argument("--min-bigrams", type=int, default=20000)
    ap.add_argument("--lex-cal", default=None, help="anchors_v1_lex.json: its `lexicon` blocks (R3 / D-59) are merged into the output so one file serves the whole reward")
    ap.add_argument("--pivot-corpus", default=None, help="R3 pivot corpus dir: held-out rows calibrate the languages that have no human references (provisional)")
    ap.add_argument("--only", default=None, help="comma-separated iso codes (debug)")
    ap.add_argument("--fold", default="all", choices=["all", "A", "B"], help="fit on half of the human sentences (split by the source sentence): the evaluation then uses the other half")
    a = ap.parse_args()
    t0 = time.time()
    cal = json.load(open(a.v1, encoding="utf-8"))
    fwd = json.load(open(a.funcwords, encoding="utf-8"))
    bg_file = np.load(a.bigrams) if a.bigrams else None
    sents, n_docs = load_humans(a.human, a.fold)
    provisional = set()
    if a.pivot_corpus:
        need = [i for i in fwd if len(sents.get(i, [])) < 100]
        piv = load_pivot(a.pivot_corpus, need, a.fold)
        for i, us in piv.items():
            sents[i] = sents.get(i, []) + us if len(sents.get(i, [])) else us
            provisional.add(i)
        print(f"[v2] provisional calibration from the pivot corpus held-out rows: {sorted(provisional)}", flush=True)
    isos = sorted(sents) if not a.only else a.only.split(",")
    print(f"[v2] {len(isos)} languages with references, {sum(len(v) for v in sents.values())} sentence units, {n_docs} document rows (pseudo-documents only)", flush=True)
    langs = cal.setdefault("languages", {})
    pooled = cal.setdefault("pooled", {})

    # ---- entity2 (two passes: the shrinkage target needs every language's drop rate)
    raw = {iso: C.fit_entity2(sents[iso]) for iso in isos}
    pool = C.pool_d({k: v for k, v in raw.items() if v})
    per_lang_e = {}
    for iso in isos:
        b = C.fit_entity2(sents[iso], pool)
        if b:
            per_lang_e[iso] = b
            langs.setdefault(iso, {})["entity2"] = b
    pooled["entity2"] = C.pooled_entity2(per_lang_e)
    print(f"[v2] entity2 done {time.time() - t0:.0f}s", flush=True)

    # ---- copy
    per_lang_c = {}
    for iso in isos:
        b = C.fit_copy(sents[iso], a.k, a.m)
        if b:
            per_lang_c[iso] = b
            langs.setdefault(iso, {})["copy"] = b
        print(f"[v2] copy {iso} {'-' if not b else 't_w=%.3f t_c=%.3f fp_cv=%.4f wl=%d enabled=%s detect=%s' % (b['t_w'], b['t_c'], b['fp_cv'], len(b['whitelist']), b['enabled'], b['detect'])}", flush=True)
    pooled["copy"] = C.pooled_copy(per_lang_c)
    print(f"[v2] copy done {time.time() - t0:.0f}s", flush=True)

    # ---- func + pad
    all_refs = {iso: [u["ref"] for u in sents[iso]] for iso in isos}
    for iso in isos:
        ent = fwd.get(iso)
        if not ent:
            continue
        units = sents[iso]
        docs = pseudo_docs(units)
        hdf: collections.Counter = collections.Counter()
        for u in units:
            hdf.update(set(T.out_tokens(u["ref"])))
        fw = C.select_fw(ent["top"], ent["n_seg"], hdf, len(units), a.fw_k)
        ex = C.english_excl(ent["en_tf"], ent["n_tok"], fw)
        fws, exs = frozenset(fw), frozenset(ex)
        rows = np.concatenate([C.func_rows(units, fws, exs), C.func_rows(docs, fws, exs)])
        geez = C.is_geez([u["ref"] for u in units[:300]])
        fpm = C.GEEZ_FP_MAX if geez else C.FUNC_FP_MAX
        fb = C.fit_func(rows, fp_max=fpm)
        if fb:
            fb.update({"geez": geez, "fw": fw, "en_excl": ex, "k": a.fw_k, "dict_source": ent.get("sources"), "n_train_seg": ent["n_seg"]})
        bg = bg_file[iso] if bg_file is not None and iso in bg_file.files else None
        use_bg = bg is not None and bg.size >= a.min_bigrams
        prow = np.concatenate([C.pad_rows(units, bg if use_bg else None), C.pad_rows(docs, bg if use_bg else None)])
        pb = C.fit_pad(prow, fp_max=fpm)
        if pb:
            pb["n_bigrams"] = int(bg.size) if bg is not None else 0
            pb["geez"] = geez
        if fb or pb:
            # a quick internal attack check (AUC of the language-level deficits on held-in humans: an upper-bound sanity figure; the report's AUCs come from scripts/eval_anchors_v2.py)
            others = [r for j in isos if j != iso for r in random.Random(1).sample(all_refs[j], min(20, len(all_refs[j])))]
            atk = AT.make_attacks(units, ent["top"], fw, others, random.Random(2), fw_share=float(rows[:, 1].mean()) if fb else 0.2)
            if fb:
                dh = C._func_deficits(rows, fb)
                auc = {}
                for name in ("en_copy", "other_lang", "stuff", "fw_only"):
                    ra = C.func_rows([{"src": u["src"], "ref": t} for u, t in zip(units, atk[name])], fws, exs)
                    da = C._func_deficits(ra, fb) if len(ra) else np.zeros(1)
                    auc[name] = float((da > 0).mean())
                fb["detect"] = auc
                fb["enabled"] = bool(fb["fp_cv"] <= fpm and auc["stuff"] >= 0.5)
                langs.setdefault(iso, {})["func"] = fb
            if pb:
                pb["enabled"] = bool(pb["fp_cv"] <= fpm)
                langs.setdefault(iso, {})["pad"] = pb
        print(f"[v2] func/pad {iso} fw={fw[:8]} func={'-' if not fb else 'fp_cv=%.3f detect=%s en=%s' % (fb['fp_cv'], fb.get('detect'), fb.get('enabled'))} pad={'-' if not pb else 'fp_cv=%.3f bg=%s' % (pb['fp_cv'], pb['has_bigrams'])}", flush=True)
    if a.lex_cal:
        for iso, blk in json.load(open(a.lex_cal, encoding="utf-8"))["languages"].items():
            if "lexicon" in blk:
                langs.setdefault(iso, {})["lexicon"] = blk["lexicon"]
        print(f"[v2] merged the lexicon blocks of {a.lex_cal}", flush=True)
    for iso in provisional:
        for blk in ("entity2", "copy", "func", "pad"):
            b = (langs.get(iso) or {}).get(blk)
            if isinstance(b, dict):
                b["basis"] = "pivot_held (religious-heavy, provisional)"
                for c in ("num", "name"):
                    if isinstance(b.get(c), dict):
                        b[c]["basis"] = "pivot_held"
    cal["v2_meta"] = {"built": time.strftime("%Y-%m-%d"), "k": a.k, "m": a.m, "fw_k": a.fw_k, "min_bigrams": a.min_bigrams, "humans": "human_units.jsonl (FLORES+ dev, smol_sent, ntrex, ibom_mt, salt, flores_moz, glot_story)",
                      "provisional_languages": sorted(provisional), "copy_alpha": C.COPY_ALPHA, "copy_floor": C.COPY_FLOOR, "func_alpha": C.FUNC_ALPHA, "entity2_max_drop": 0.6, "name_scale": 0.5}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(cal, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(f"[v2] wrote {a.out} ({os.path.getsize(a.out) / 1e6:.2f} MB) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
