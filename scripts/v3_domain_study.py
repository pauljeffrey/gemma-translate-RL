#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-68: how large is the domain shift of a religious-text calibration, and does the domain-robust recipe (rl/anchors_calibrate_v3.py) remove it?

For every language that has BOTH religious-style text (R3 pivot held-out rows, R15 JW windows) and out-of-domain human text (SmolSent, GlotStoryBook, Wikimedia, IbomMT, SALT, FLORES+,
FineTranslations web text ...): fit the detectors on half of the religious text only -- (a) the v2 'pivot_held' method (pooled thresholds, geometric-mean dictionary, bigram-order feature),
(b) the v3 recipe for a grid of margins -- and measure the share of OUT-OF-DOMAIN human sentences / 10-sentence documents with a deficit > 0 and with a deficit >= 0.9 (the gate level),
next to the share of held-out religious text. Output: JSON + markdown on stdout.
    python scripts/v3_domain_study.py --funcwords funcwords_v1.json --bigrams anchors_v2_bigrams.npz --out v3_study.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3_data as D  # noqa: E402
from gemma_mt.rl import anchors_calibrate_v2 as C  # noqa: E402
from gemma_mt.rl import anchors_calibrate_v3 as C3  # noqa: E402
from gemma_mt.rl import textstats as T  # noqa: E402

LAMS = (0.0, 1.0, 2.0, 3.0, 4.0)
COPY_CFG = ((0.15, 1.25), (0.15, 1.5), (0.2, 1.5))
KFW = 60


def naive_blocks(rel_train, funcs, bg):
    """The v2 pivot_held method: pooled units + pseudo-documents, geometric-mean dictionary, bigram table, CV-selected margin."""
    units = rel_train + C3.pseudo(rel_train, (4, 10, 25))
    hdf = collections.Counter()
    for u in rel_train:
        hdf.update(set(T.out_tokens(u["ref"])))
    fw = C.select_fw(funcs["top"], funcs["n_seg"], hdf, len(rel_train), 80)
    ex = frozenset(C.english_excl(funcs["en_tf"], funcs["n_tok"], fw))
    fws = frozenset(fw)
    rows = C.func_rows(units, fws, ex)
    fb = C.fit_func(rows)
    if fb:
        fb.update({"fw": fw, "en_excl": sorted(ex)})
    use_bg = bg is not None and bg.size >= 20000
    pr = C.pad_rows(units, bg if use_bg else None)
    pb = C.fit_pad(pr)
    cb = C.fit_copy(rel_train, 3, 20)
    return {"copy": cb, "func": fb, "pad": pb}


def v3_blocks(rel_train, funcs, lam, cfg):
    by = collections.defaultdict(list)
    for u in rel_train:
        by[u["set"]].append(u)
    dom = dict(by)
    for s, us in list(by.items()):
        for k, nm in ((4, "w4"), (10, "w10")):
            p = [x for x in C3.pseudo(us, (k,))]
            if p:
                dom[f"{s}/{nm}"] = p
    fw = C3.robust_fw(funcs["top"], funcs["n_seg"], {n: [u["ref"] for u in us] for n, us in by.items()}, KFW)
    ex = frozenset(C.english_excl(funcs["en_tf"], funcs["n_tok"], fw))
    cb = C3.fit_copy_v3(rel_train, floor=cfg[0], scale=cfg[1])
    return {"copy": cb, "func": C3.fit_func_v3(dom, fw, ex, lam), "pad": C3.fit_pad_v3(dom, lam)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--funcwords", required=True)
    ap.add_argument("--bigrams", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="kri,nbl,ach,nyn,ndc,dag,tsc,her,ibb,kea,sid,bas,kon")
    a = ap.parse_args()
    fwd = json.load(open(a.funcwords, encoding="utf-8"))
    bgf = np.load(a.bigrams) if a.bigrams else None
    res = {}
    for iso in a.langs.split(","):
        sets = D.load_sets(iso)
        rel, ood = D.split_dom(sets)
        if len(rel) < 100 or len(ood) < 100 or iso not in fwd:
            print(iso, "skipped", {k: len(v) for k, v in sets.items()}, flush=True)
            continue
        ex0 = frozenset(C.english_excl(fwd[iso]["en_tf"], fwd[iso]["n_tok"], []))
        n_ood0 = len(ood)
        ood = [u for u in ood if T.func_features(T.out_tokens(u["ref"]), frozenset(), ex0)["en"] < 0.12]         # drop rows whose "reference" is English text (untranslated web / UI strings): not a translation
        if len(ood) < 100:
            print(iso, "skipped after the English filter", n_ood0, len(ood), flush=True)
            continue
        tr = [u for u in rel if D._fold(iso, u["src"]) == 0]
        te = [u for u in rel if D._fold(iso, u["src"]) == 1]
        ood_eval = ood + C3.pseudo(ood, (10,))
        te_eval = te + C3.pseudo(te, (10,))
        bg = bgf[iso] if bgf is not None and iso in bgf.files else None
        R = {"ood_dropped_english": n_ood0 - len(ood), "n_rel_train": len(tr), "n_rel_test": len(te), "n_ood": len(ood), "ood_sets": {k: len(v) for k, v in sets.items() if v and v[0]["dom"] == "ood"}, "variants": {}}
        nb = naive_blocks(tr, fwd[iso], bg)
        R["variants"]["naive_v2"] = {"ood": C3.fp_report(nb, ood_eval, bg), "rel_heldout": C3.fp_report(nb, te_eval, bg)}
        for lam in LAMS:
            vb = v3_blocks(tr, fwd[iso], lam, COPY_CFG[0])
            vb["copy"] = None
            R["variants"][f"v3_lam{lam}"] = {"ood": C3.fp_report(vb, ood_eval), "rel_heldout": C3.fp_report(vb, te_eval)}
        for cfg in COPY_CFG:
            cb = C3.fit_copy_v3(tr, floor=cfg[0], scale=cfg[1])
            R["variants"][f"v3_copy{cfg[0]}x{cfg[1]}"] = {"ood": C3.fp_report({"copy": cb}, ood_eval), "rel_heldout": C3.fp_report({"copy": cb}, te_eval), "t_w": cb and cb["t_w"], "wl": cb and len(cb["whitelist"])}
        res[iso] = R
        v = R["variants"]
        f = lambda n, d, k: "-" if k not in v[n][d] else "%.3f/%.3f" % (v[n][d][k]["fp"], v[n][d][k]["sat"])
        print(f"{iso}: ood n={len(ood)} {R['ood_sets']} | naive ood copy {f('naive_v2', 'ood', 'copy')} func {f('naive_v2', 'ood', 'func')} pad {f('naive_v2', 'ood', 'pad')} | "
              f"v3(lam2) func {f('v3_lam2.0', 'ood', 'func')} pad {f('v3_lam2.0', 'ood', 'pad')} copy(0.15x1.25) {f('v3_copy0.15x1.25', 'ood', 'copy')}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
