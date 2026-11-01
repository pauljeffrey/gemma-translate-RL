# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared helpers of the mixture-v2 builders (archive component A, long-document component B): the judges of the RL reward, outside the trainer.

* `Judge`   shaped sentence reward of the QE ENSEMBLE (AfriCOMET-QE + SSA-COMET-MTL, rule of D-55: mean if |r1 - r2| <= delta else min), the content mode of a
            language (qe / blend / consensus), the GlotLID guard (labels / min chars / on-off from the calibration `lid` blocks), the diacritic deficit,
            the human length window, the repetition thresholds of anchors_v1 (human p97 per length bucket).
* `holdout` the hold-out object of mix_v1 (R4's builder), the SAME sets: AfriScience dev+test, AfriLongDoc hard+core, bench_dev_clean, FLORES+ dev English, RL-vs-SFT pool.
Run in /workspace/venv_gemma with PYTHONPATH=/workspace/jobs/r11/code:/workspace/sft_build/pylibs.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import unicodedata
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

CAL = "/workspace/jobs/r11/code/calibration/shaping_flores_dev_v8.json"
ANCHORS = "/workspace/jobs/r11/code/calibration/anchors_v1.json"
GLOTLID = "/workspace/sft_build/hf/hub/models--cis-lmu--glotlid/snapshots/85cd6716494360367b75f642b5bc78667605d0b4/model_v3.bin"
SCORER_SPEC = {"type": "worker", "python": "/workspace/venv_gemma/bin/python", "pythonpath": "/workspace/sft_build/pylibs_tf4:/workspace/sft_build/pylibs",
               "model_id": "masakhane/africomet-qe-stl-1.1", "ensemble_model_id": "McGill-NLP/ssa-comet-mtl", "batch_size": 96, "max_batch_chars": 48000}
SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def make_holdout():
    import pandas as pd

    B = load_script("build_sft_science_mix")
    ns = SimpleNamespace(longdoc_dir="/workspace/jobs/a4/holdout/afrilongdoc", bench_dev="/workspace/sft_build/bench/bench_dev_clean.jsonl",
                         flores_dev_eng="/workspace/sft_build/hf/hub/datasets--openlanguagedata--flores_plus/snapshots/*/dev/eng_Latn.jsonl",
                         eval_pool="/workspace/rlcheck/res/pool.jsonl")
    afs = {sp: pd.read_parquet(f"/workspace/jobs/a4/afs_raw/corpus/{sp}-00000.parquet") for sp in ("dev", "test")}
    return B.build_holdout(ns, afs)


def words(s: str) -> List[str]:
    from gemma_mt.rl.anchors import words as _w

    return _w(s)


class Judge:
    def __init__(self, with_scorer: bool = True, with_lid: bool = True):
        from gemma_mt.rl import qe_ensemble as Q
        from gemma_mt.rl.reward_shaping import ContentShaper

        self.shaper = ContentShaper.from_file(CAL, reliability_from="same_source", preset="repo", neg_scale=2.0, fail_floor=0.0)
        self.ens = Q.QEEnsemble.from_file(CAL, self.shaper, delta=1.0, prior_langs=True, reliability="ensemble", reliability_from="same_source", preset="repo",
                                          neg_scale=2.0, fail_floor=0.0)
        self.anch = json.load(open(ANCHORS, encoding="utf-8"))
        self.scorer = Q.build_pair_scorer(SCORER_SPEC) if with_scorer else None
        self.lid = None
        if with_lid:
            from gemma_mt.rl.lid import GlotLIDClassifier

            self.lid = GlotLIDClassifier(GLOTLID)
        cal = json.load(open(CAL, encoding="utf-8"))
        self.cal = {k.split(">")[-1].strip(): v for k, v in cal.items() if not k.startswith("_")}

    # ---------------------------------------------------------------- content mode
    def mode(self, lang: str) -> str:
        return (self.shaper.content.get(lang) or {}).get("mode", "qe")

    def has_qe(self, lang: str) -> bool:
        return self.mode(lang) != "consensus" and (self.ens.has1(lang) or self.ens.has2(lang))

    # ---------------------------------------------------------------- sentence QE (ensemble)
    def qe_raw(self, srcs: Sequence[str], mts: Sequence[str], langs: Sequence[str], chunk: int = 6000) -> List[Tuple[Optional[float], Optional[float]]]:
        """(AfriCOMET-QE, SSA-COMET-MTL) per pair; the first only where the ensemble plan uses it, the second only where it is used (None otherwise)."""
        n = len(srcs)
        out: List[List[Optional[float]]] = [[None, None] for _ in range(n)]
        i1 = [k for k in range(n) if self.ens.plan(langs[k]) in ("both", "only1") and self.has_qe(langs[k])]
        i2 = [k for k in range(n) if self.ens.plan(langs[k]) in ("both", "only2") and self.has_qe(langs[k])]
        for idx, which in ((i1, 0), (i2, 1)):
            for s in range(0, len(idx), chunk):
                part = idx[s:s + chunk]
                f = self.scorer if which == 0 else self.scorer.score_secondary
                vals = f([srcs[k] for k in part], [mts[k] for k in part])
                for k, v in zip(part, vals):
                    out[k][which] = float(v)
        return [(a, b) for a, b in out]

    def shaped(self, c1: Optional[float], c2: Optional[float], lang: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """(r1, r2, combined reward) of a sentence from its two raw scores (the rule of D-55)."""
        from gemma_mt.rl.qe_ensemble import combine_pair

        p = self.ens.plan(lang)
        r1 = float(self.shaper(c1, lang)) if (c1 is not None and p != "only2") else None
        r2 = float(self.ens.shaper2(c2, lang)) if (c2 is not None and p != "only1") else None
        if r1 is not None and r2 is not None:
            return r1, r2, combine_pair(r1, r2, self.ens.delta)[0]
        return r1, r2, (r2 if r2 is not None else r1)

    # ---------------------------------------------------------------- guards
    def lid_spec(self, lang: str) -> Tuple[Tuple[str, ...], int, bool]:
        e = self.shaper.lid.get(lang)
        if not e or not e.get("enabled", False):
            return tuple((e or {}).get("labels", ())), 10 ** 9, False
        return tuple(e.get("labels", ())), max(15, int(e.get("min_chars", 15))), True

    def lid_ok(self, lang: str, text: str) -> Optional[bool]:
        """True/False = the GlotLID top-1 label of the text is / is not accepted for the language; None = not checked (guard off or text shorter than min_chars)."""
        labels, mn, active = self.lid_spec(lang)
        if not active or len(text) < mn:
            return None
        return self.lid([text])[0] in labels

    def lid_ok_many(self, lang: str, texts: Sequence[str]) -> List[Optional[bool]]:
        labels, mn, active = self.lid_spec(lang)
        out: List[Optional[bool]] = [None] * len(texts)
        if not active:
            return out
        idx = [i for i, t in enumerate(texts) if len(t) >= mn]
        if idx:
            got = self.lid([texts[i] for i in idx])
            for i, g in zip(idx, got):
                out[i] = g in labels
        return out

    def diacritic_deficit(self, lang: str, src: str, mt: str) -> float:
        from gemma_mt.rl.diacritics import diacritic_report

        if not (self.shaper.diacritics.get(lang) or {}).get("applies"):
            return 0.0
        return float(diacritic_report(mt, lang, stats=self.shaper.diacritics, src=src)["deficit"])

    def len_ok(self, lang: str, src: str, mt: str, slack: float = 1.35) -> bool:
        w = self.shaper.len_window.get(lang) or self.shaper.len_window.get("*")
        if not w:
            return 0.3 <= len(mt) / max(len(src), 1) <= 4.0
        r = len(mt) / max(len(src), 1)
        return w[0] / slack <= r <= w[1] * slack

    # ---------------------------------------------------------------- repetition (word 3/4/5-grams against the human p97 per length bucket, rl/anchors.py)
    def repeat_thr(self, lang: str) -> Dict[str, List[float]]:
        L = self.anch["languages"].get(lang) or self.anch["languages"].get(self.anch.get("proxies", {}).get(lang, ""), None)
        blk = (L or {}).get("repeat") or {}
        if blk.get("thr"):
            return blk["thr"]
        return self.anch["pooled"]["repeat"]["thr"]

    def repeat_bad(self, lang: str, src: str, mt: str, extra: float = 0.0) -> Optional[str]:
        """'rep<n>' when the share of repeated word n-grams (n = 3, 4, 5) of the translation is above max(human p97 of its length bucket, the source's own share) + extra;
        None otherwise (and for outputs below 12 words)."""
        from gemma_mt.rl.anchors import REPEAT_MIN_WORDS, length_bucket, repeat_shares, words as aw

        w = aw(mt)
        if len(w) < REPEAT_MIN_WORDS:
            return None
        sw = aw(src)
        k = length_bucket(len(w))
        sh = repeat_shares(w)
        ssh = repeat_shares(sw) if len(sw) >= REPEAT_MIN_WORDS else {3: 0.0, 4: 0.0, 5: 0.0}
        thr = self.repeat_thr(lang)
        for n in (3, 4, 5):
            t = float(thr[str(n)][k]) if thr.get(str(n)) else 0.2
            if sh[n] > max(t, ssh[n]) + extra:
                return f"rep{n}"
        return None

    def close(self) -> None:
        if self.scorer is not None:
            self.scorer.close()
