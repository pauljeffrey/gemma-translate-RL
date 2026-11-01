#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build calibration/anchors_v1.json: the thresholds of the constraint penalties of rl/anchors.py (D-56), fitted on HUMAN references.

    python scripts/build_anchors_calibration.py --shaping calibration/shaping_flores_dev_v7.json --out calibration/anchors_v1.json \
        --test-sets /workspace/sft_build/bench/test_sets [--cache-dir DIR] [--doc-file bench_afrisci_dev_doc.jsonl] [--lid-model model_v3.bin] [--lexicon-dir lexicon]

Data (thresholds only, never training data): FLORES+ dev from the Hub (HF_TOKEN; loader as in build_word_ratio_calibration.py) + the dev-style public sets under --test-sets
(smol_sent, ibom_mt, glot_story, salt, flores_moz, ntrex) + smol_doc / --doc-file documents for the long length buckets of the repetition detector. Every RL language with human
data gets its own blocks; the POOLED prior (used by the languages without references) is fitted on the blend / consensus languages only (shaping `content.mode`).
See gemma_mt/rl/anchors_calibrate.py for the rules (alpha, cross-validated false-positive rate, usefulness criterion) and rl/anchors.py for what each detector measures.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import anchors_common as C  # noqa: E402
from gemma_mt.languages import LANGUAGES, SUPPORTED_RL  # noqa: E402
from gemma_mt.rl import anchors_calibrate as K  # noqa: E402

MAX_UNITS = 1800
PROXIES = {"swc": "swh", "wes": "pcm"}      # languages without references borrow the blocks of a closely related / same-orthography language


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shaping", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-sets", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--doc-file", action="append", default=[])
    ap.add_argument("--lid-model", default=None)
    ap.add_argument("--lexicon-dir", default=None)
    ap.add_argument("--no-hub", action="store_true")
    ap.add_argument("--alpha", type=float, default=K.ALPHA)
    ap.add_argument("--languages", nargs="*", default=None, help="restrict to these languages (default: all RL languages with human data)")
    ap.add_argument("--dump-human", default=None, help="write the human units used (jsonl) for scripts/eval_anchors.py")
    a = ap.parse_args(argv)

    shaping = json.load(open(a.shaping, encoding="utf-8"))
    mode = {k: (v.get("content") or {}).get("mode", "qe") for k, v in shaping.items() if isinstance(v, dict) and not k.startswith("_")}
    weak = [i for i in SUPPORTED_RL if mode.get(i) in ("blend", "consensus")]
    isos = a.languages or list(SUPPORTED_RL)
    t0 = time.time()
    human: Dict[str, List[Dict[str, str]]] = {i: [] for i in isos}
    if not a.no_hub:
        for i, us in C.flores_dev(isos, a.cache_dir).items():
            human[i] += us
    if a.test_sets:
        for i, us in C.local_sets(isos, a.test_sets).items():
            human[i] += us
    docs = C.documents(isos, a.test_sets, a.doc_file) if a.test_sets or a.doc_file else {}
    rng = random.Random(7)
    for i in isos:
        human[i] = C.dedupe(human[i])
        if len(human[i]) > MAX_UNITS:
            human[i] = rng.sample(human[i], MAX_UNITS)
    print(f"[anchors] human data in {time.time() - t0:.0f}s: " + ", ".join(f"{i}={len(v)}" for i, v in human.items() if v), flush=True)
    if a.dump_human:
        with open(a.dump_human, "w", encoding="utf-8") as f:
            for i, us in human.items():
                for u in us:
                    f.write(json.dumps({"iso": i, **u}, ensure_ascii=False) + "\n")
            for i, us in docs.items():
                for u in us:
                    f.write(json.dumps({"iso": i, "doc": True, **u}, ensure_ascii=False) + "\n")

    langs: Dict[str, Dict[str, Any]] = {}
    ent_tabs: Dict[str, Dict[str, np.ndarray]] = {}
    rep_tabs: Dict[str, np.ndarray] = {}
    boil_shares: Dict[str, np.ndarray] = {}
    for i in isos:
        us = human[i]
        if not us:
            continue
        blk: Dict[str, Any] = {"n_units": len(us), "sets": {s: sum(1 for u in us if u["set"] == s) for s in sorted({u["set"] for u in us})}, "mode": mode.get(i, "qe")}
        ent_tabs[i] = K.entity_table(us)
        e = {}
        for ch, nmin in K.CHANNELS:
            b = K.fit_entity_channel(ent_tabs[i]["own"], ent_tabs[i]["other"], ch, nmin, a.alpha)
            if b:
                e[ch] = b
        if e:
            blk["entity"] = e
        texts = us + C.pseudo_documents(us) + docs.get(i, [])
        rep_tabs[i] = K.repeat_table(texts)["t"]
        boil_shares[i] = K.boiler_shares(us, i)
        langs[i] = blk
        print(f"[anchors] {i}: {len(us)} units, {len(texts)} texts, entity {({k: (round(v['t'], 2), v['enabled']) for k, v in e.items()})}", flush=True)

    pool_isos = [i for i in weak if i in ent_tabs]
    pooled: Dict[str, Any] = {}
    per_ent = {i: langs[i]["entity"] for i in pool_isos if "entity" in langs[i]}
    pooled["entity"] = K.pooled_entity(per_ent, {i: ent_tabs[i] for i in pool_isos}, q=0.10)
    floor = K.doc_floor([K.repeat_table(docs[i])["t"] for i in docs if docs[i]], a.alpha)
    pooled_rep = K.pooled_repeat({i: rep_tabs[i] for i in pool_isos}, a.alpha, floor)
    pooled_rep["doc_floor"] = floor
    pooled["repeat"] = pooled_rep
    for i in langs:
        r = K.fit_repeat(rep_tabs[i], pooled_rep["thr"], a.alpha, floor=floor)
        if r:
            langs[i]["repeat"] = r
    bl = {}
    for i in langs:
        b = K.fit_boiler(boil_shares[i], a.alpha)
        if b:
            langs[i]["boiler"] = b
            if i in pool_isos:
                bl[i] = {**b, "_sh": boil_shares[i].tolist()}
    pooled["boiler"] = K.pooled_boiler(bl)
    if a.lid_model:
        from gemma_mt.rl.lid import GlotLIDClassifier

        lid = GlotLIDClassifier(path=a.lid_model)
        for i in langs:
            e = (shaping.get(i) or {}).get("lid") or {}
            labels = e.get("labels") or list(LANGUAGES[i].glotlid)
            if not labels:
                continue
            b = K.fit_lid_prob(lid.top([u["ref"] for u in human[i]], 1), labels)
            if b:
                b["guard_enabled_in_shaping"] = bool(e.get("enabled"))
                langs[i]["lid_prob"] = b
        ps = [b["lid_prob"]["pmin"] for b in langs.values() if "lid_prob" in b and b["lid_prob"].get("guard_enabled_in_shaping")]
        if ps:
            pooled["lid_prob"] = {"pmin": float(np.quantile(ps, 0.1)), "enabled": True, "n_languages": len(ps)}
    if a.lexicon_dir:                      # (d): thresholds for the languages whose lexicon (rl/lexicon.py, lexicon/<iso>.json) validated
        from gemma_mt.rl import anchors as A

        for i in langs:
            fn = (lambda s_, r_, _i=i: A.lexicon_overlap(_i, s_, r_, a.lexicon_dir))
            b = K.fit_lexicon(fn, human[i], a.alpha)
            if b:
                langs[i]["lexicon"] = b
    proxies = {k: v for k, v in PROXIES.items() if k not in langs and v in langs}

    out = {"_meta": {"built": time.strftime("%Y-%m-%d"), "alpha": a.alpha, "shaping": os.path.basename(a.shaping), "weak_languages": weak, "pool_languages": pool_isos,
                     "fp_cv_max": K.FP_CV_MAX, "detect_min": K.DETECT_MIN, "repeat_cap": K.REPEAT_CAP, "boiler_cap": K.BOILER_CAP, "length_edges": list(K.A.LENGTH_EDGES),
                     "sources": "FLORES+ dev (Hub) + dev-style public sets + smol_doc / whole-paper references; thresholds only",
                     "languages_without_human_data": [i for i in weak if i not in langs]},
           "proxies": proxies, "pooled": pooled, "languages": langs}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[anchors] wrote {a.out} ({len(langs)} languages, pooled entity {({k: (round(v['t'], 2), v['enabled']) for k, v in pooled['entity'].items()})}, "
          f"repeat enabled {pooled['repeat']['enabled']}, boiler thr {pooled['boiler']['thr']:.3f} enabled {pooled['boiler']['enabled']}) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
