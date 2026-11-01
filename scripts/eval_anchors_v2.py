#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure the D-61 detectors (rl/anchors_v2.py) out of sample: false positives on held-out HUMAN references, AUC and detection against synthetic attacks and the corruption classes of agent A5.

Use a calibration built on fold A (`build_anchors_v2_calibration.py --fold A`) and the human sentences of fold B (split by the source sentence, crc32 parity): nothing measured here was
used to fit a threshold (the function-word dictionary comes from the training text and, for the geometric-mean selection, from the fold-A references).

    python scripts/eval_anchors_v2.py --cal calibration_foldA.json --human human_units.jsonl --funcwords funcwords_v1.json --bigrams bigrams_v1.npz \
        --hack-rows hack_rows.jsonl --lexicon-dir lexicon --shaping calibration/shaping_flores_dev_v8.json --lex-cal calibration/anchors_v1_lex.json --out eval.json

Output (JSON): per language `fp` (share of held-out human references with a deficit > 0, mean deficit) per detector, `auc` / `det` per attack x detector (AUC of the uncapped raw score human vs attack;
`det` = share of attacks with a deficit > 0 and the mean deficit), `hack` per corruption class (copy detector, stop-word profile), plus the pooled tables the report prints.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.stats import rankdata  # noqa: E402

from gemma_mt.rl import anchors as A1  # noqa: E402
from gemma_mt.rl import anchors_attacks as AT  # noqa: E402
from gemma_mt.rl import anchors_v2 as V  # noqa: E402
from gemma_mt.rl import guards  # noqa: E402
from gemma_mt.rl import textstats as T  # noqa: E402


def auc(h, a) -> float:
    """P(score_attack > score_human) (ties 1/2)."""
    h, a = np.asarray(h, float), np.asarray(a, float)
    if len(h) == 0 or len(a) == 0:
        return float("nan")
    r = rankdata(np.concatenate([h, a]))
    return float((r[len(h):].sum() - len(a) * (len(a) + 1) / 2) / (len(h) * len(a)))


def fold_of(iso: str, src: str) -> int:
    return zlib.crc32((iso + "|" + " ".join(src.split()).casefold()).encode("utf-8")) & 1


def raw_copy(A: V.Anchors2, lang: str, src: str, mt: str):
    """Uncapped copy score: max(frac_w / t_w, frac_c / t_c); also the thresholded fraction the reward would use."""
    b = A._blk2(lang, "copy")
    if not b:
        return None, None
    wl = frozenset(b.get("whitelist") or ())
    r = T.copy_fractions(T.source_tokens(src, wl), T.out_tokens(mt), k=int(b["k"]), m=int(b["m"]), min_unprot=int(b.get("min_unprot", 2)))
    if r.frac_w is None and r.frac_c is None:
        return None, None
    sc = max([x for x in (None if r.frac_w is None else r.frac_w / max(b["t_w"], 1e-6), None if r.frac_c is None else r.frac_c / max(b["t_c"], 1e-6)) if x is not None])
    return sc, A.copy_fraction(lang, src, mt)


def detect_all(A: V.Anchors2, shaper, lang: str, src: str, mt: str, lex_cal, lexdir, cfgw):
    """All detector values for one (source, output) pair (None = not judged / not available)."""
    sw, ow = T.out_tokens(src), T.out_tokens(mt)
    out = {}
    rs, cf = raw_copy(A, lang, src, mt)
    out["copy_raw"], out["copy"] = rs, cf
    fb = A._blk2(lang, "func")
    f = A.func_features(lang, sw, ow)
    out["func_raw"] = V.func_deficit_from(fb, int(f["n"]), f["fw"], f["en"], f["ratio"], clip=False) if f else None
    out["func"] = A.func_deficit(lang, sw, ow)
    pb = A._blk2(lang, "pad")
    pf = A.pad_features(lang, ow)
    out["pad_raw"] = V.pad_deficit_from(pb, int(pf["n"]), pf.get("gain"), pf["ttr"], clip=False) if pf else None
    out["pad"] = A.pad_deficit(lang, ow)
    # existing guards (the ones of configs/rl_science_final.yaml)
    out["gate_copy"] = float(guards.is_copy_of_source(src, mt, 0.9))
    out["gate_loop"] = float(guards.is_repetitive(mt) or guards.is_self_repetitive(mt))
    out["gate_len"] = float(guards.length_ratio_bad(src, mt, 0.4, 5.0))
    if shaper is not None:
        out["len"] = shaper.length_excess(lang, len(src), len(mt), 0.15) * cfgw["length"]
        out["wordratio"] = shaper.word_excess(lang, src, mt, 0.15) * cfgw["wordratio"]
        out["lexical"] = shaper.lexical_deficit(lang, src, mt) * cfgw["lexical"]
    p = A.prepare(lang, src, mt)
    out["repeat"] = A.repeat_deficit(lang, p)
    if lex_cal is not None:
        b = (lex_cal.get(lang) or {}).get("lexicon")
        if b and b.get("enabled"):
            ov = A.lexicon_overlap(lang, src, mt)
            out["lexicon_ov"] = ov
            out["lexicon"] = None if ov is None else max(0.0, (b["t"] - ov) / b["t"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", required=True)
    ap.add_argument("--human", required=True)
    ap.add_argument("--funcwords", required=True)
    ap.add_argument("--bigrams", default=None)
    ap.add_argument("--hack-rows", default=None)
    ap.add_argument("--lexicon-dir", default=None)
    ap.add_argument("--shaping", default=None)
    ap.add_argument("--lex-cal", default=None, help="anchors_v1_lex.json (for the `lexicon` blocks)")
    ap.add_argument("--fold", default="B", choices=["A", "B", "all"])
    ap.add_argument("--max-units", type=int, default=600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--docs", type=int, default=0, help="evaluate pseudo-documents of this many consecutive held-out sentences instead of sentences (the long-text case)")
    a = ap.parse_args()
    t0 = time.time()
    cal = json.load(open(a.cal, encoding="utf-8"))
    fwd = json.load(open(a.funcwords, encoding="utf-8"))
    lex_cal = json.load(open(a.lex_cal, encoding="utf-8"))["languages"] if a.lex_cal else None
    if lex_cal:
        for iso, blk in lex_cal.items():
            if "lexicon" in blk:
                cal["languages"].setdefault(iso, {})["lexicon"] = blk["lexicon"]
    A = V.Anchors2(cal, detectors=("entity", "repeat", "boiler", "lexicon"), lexicon_dir=a.lexicon_dir, bigram_path=a.bigrams)
    shaper = None
    if a.shaping:
        from gemma_mt.rl.reward_shaping import ContentShaper

        shaper = ContentShaper.from_file(a.shaping, reliability_from="same_source", mode="calibrated", preset="repo", neg_scale=2.0)
    cfgw = {"length": 1.0, "wordratio": 0.5, "lexical": 1.0}
    want = int(a.fold == "B") if a.fold != "all" else None
    units = collections.defaultdict(list)
    seen = set()
    for l in open(a.human, encoding="utf-8"):
        r = json.loads(l)
        if r["set"] in ("smol_doc",) or r["set"].endswith("_doc.jsonl"):
            continue
        k = (r["iso"], " ".join(r["src"].split()).casefold(), " ".join(r["ref"].split()).casefold())
        if k in seen or (want is not None and fold_of(r["iso"], r["src"]) != want):
            continue
        seen.add(k)
        units[r["iso"]].append(r)
    isos = [i for i in sorted(units) if i in cal["languages"] and (not a.only or i in a.only.split(","))]
    all_refs = {i: [u["ref"] for u in units[i]] for i in units}
    res = {}
    rng = random.Random(0)
    for iso in isos:
        us = units[iso][:]
        if a.docs:
            import anchors_common as ACM

            us = ACM.pseudo_documents(us, (a.docs,))
        rng.shuffle(us)
        us = us[: a.max_units]
        blk = cal["languages"][iso]
        fb = blk.get("func")
        ent = fwd.get(iso)
        lex = None
        if a.lexicon_dir and lex_cal and (lex_cal.get(iso) or {}).get("lexicon"):
            from gemma_mt.rl import lexicon as L

            lex = L.load_lexicon(iso, a.lexicon_dir)
        fw_list = (fb or {}).get("fw") or ([] if not ent else [t[0] for t in ent["top"][:80]])
        fw_share = 0.2
        if fb and ent:
            ex = frozenset(fb.get("en_excl") or ())
            fws = frozenset(fw_list)
            vals = [T.func_features(T.out_tokens(u["ref"]), fws, ex)["fw"] for u in us[:200]]
            fw_share = float(np.mean(vals)) if vals else 0.2
        others = [r for j in isos if j != iso for r in random.Random(1).sample(all_refs[j], min(20, len(all_refs[j])))]
        atk = AT.make_attacks(us, ent["top"], fw_list, others, random.Random(2), lex=lex, fw_share=fw_share) if ent else {}
        H = [detect_all(A, shaper, iso, u["src"], u["ref"], lex_cal, a.lexicon_dir, cfgw) for u in us]
        R = {"n": len(us), "fp": {}, "auc": {}, "det": {}}
        for det in ("copy", "func", "pad", "repeat", "len", "wordratio", "lexical", "lexicon", "gate_copy", "gate_loop", "gate_len"):
            v = [h[det] for h in H if h.get(det) is not None]
            if v:
                R["fp"][det] = {"share": float(np.mean(np.asarray(v) > 0)), "mean": float(np.mean(v)), "n": len(v)}
        hraw = {d: [h[d + "_raw"] for h in H if h.get(d + "_raw") is not None] for d in ("copy", "func", "pad")}
        for name, texts in atk.items():
            D = [detect_all(A, shaper, iso, u["src"], t, lex_cal, a.lexicon_dir, cfgw) for u, t in zip(us, texts)]
            R["auc"][name], R["det"][name] = {}, {}
            for det in ("copy", "func", "pad"):
                araw = [d[det + "_raw"] for d in D if d.get(det + "_raw") is not None]
                if len(araw) >= 10 and len(hraw[det]) >= 10:
                    R["auc"][name][det] = auc(hraw[det], araw)
            for det in ("copy", "func", "pad", "repeat", "len", "wordratio", "lexical", "lexicon", "gate_copy", "gate_loop", "gate_len"):
                v = [d[det] for d in D if d.get(det) is not None]
                if v:
                    R["det"][name][det] = {"share": float(np.mean(np.asarray(v) > 0)), "mean": float(np.mean(v)), "n": len(v)}
            # per-sample table for the combination analysis (kept compact)
            R.setdefault("samples", {})[name] = [[d.get("copy"), d.get("func"), d.get("pad"), d.get("repeat"), d.get("len"), d.get("wordratio"), d.get("lexical"), d.get("lexicon"),
                                                  d.get("gate_copy"), d.get("gate_loop"), d.get("gate_len")] for d in D]
        R.setdefault("samples", {})["human"] = [[h.get("copy"), h.get("func"), h.get("pad"), h.get("repeat"), h.get("len"), h.get("wordratio"), h.get("lexical"), h.get("lexicon"),
                                                 h.get("gate_copy"), h.get("gate_loop"), h.get("gate_len")] for h in H]
        res[iso] = R
        print(f"[eval] {iso} n={len(us)} fp_func={R['fp'].get('func', {}).get('share', float('nan')):.3f} fp_pad={R['fp'].get('pad', {}).get('share', float('nan')):.3f} fp_copy={R['fp'].get('copy', {}).get('share', float('nan')):.3f}"
              f" auc_stuff={R['auc'].get('stuff', {})} ({time.time() - t0:.0f}s)", flush=True)
    # ---- hack classes of agent A5 (FLORES+ dev, 80 sentences x 45 languages x 12 classes)
    hack = collections.defaultdict(lambda: collections.defaultdict(list))
    if a.hack_rows:
        for l in open(a.hack_rows, encoding="utf-8"):
            r = json.loads(l)
            iso = r["iso"]
            if iso not in res or (want is not None and fold_of(iso, r["src"]) != want):
                continue
            d = detect_all(A, shaper, iso, r["src"], r["mt"], lex_cal, a.lexicon_dir, cfgw)
            hack[r["cls"]][iso].append([d.get("copy"), d.get("func"), d.get("pad"), d.get("repeat"), d.get("gate_copy"), d.get("gate_loop"), d.get("gate_len")])
    out = {"fold": a.fold, "cal": os.path.basename(a.cal), "languages": res, "hack": {c: {i: v for i, v in per.items()} for c, per in hack.items()}, "seconds": time.time() - t0}
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print("[eval] wrote", a.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
