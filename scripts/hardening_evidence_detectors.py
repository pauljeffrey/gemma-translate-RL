# SPDX-License-Identifier: AGPL-3.0-or-later
"""H1: the detectors of the final run's reward applied to STORED outputs (no model, no GPU). Everything is the project's own code with the calibration the final run used:

    gemma_mt.rl.anchors_v2.Anchors2   (copy, graded entity2, stop-word profile, padding / type-token ratio, repetition) with calibration/anchors_v3.json
    gemma_mt.rl.guards                (Jaccard copy, repetition, length ratio: the hard failure classes both rewards had)
    gemma_mt.rl.lid.GlotLIDClassifier (GlotLID v3, accepted labels from calibration/shaping_flores_dev_v8.json `lid`)
    gemma_mt.rl.longdoc               (long-document reward v2: repetition severity, anchored alignment, coverage) with calibration/longdoc_v2.json

The anchors object is built with the settings of runs/final2/config.json (anchors_modes [consensus, blend] for entity / repeat / boiler / lexicon, v2 modes [qe, blend, consensus]
for copy / stop-word / padding, copy scale 2.0 mode cap, func weight 0.3, pad weight 0.5, entity weight 0.5, repeat weight 1.0, boiler weight 1.0, stuff gate OFF).
Not available offline: the bigram table of the padding detector (anchors_v2_bigrams.npz is on the box only), so the padding deficit of the 30 languages that have a bigram
component is the type / token-ratio part alone (a lower bound of the final run's deficit); the lexicon detector (10 languages, weight <= 0.25) is not computed.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import hardening_evidence_common as C  # noqa: F401  (puts gemma_mt on sys.path)
from gemma_mt.rl import anchors as AN
from gemma_mt.rl import guards
from gemma_mt.rl import textstats as T
from gemma_mt.rl.anchors_v2 import Anchors2, pad_ttr

W_ENTITY, W_REPEAT, W_FUNC, W_PAD, COPY_SCALE = 0.5, 1.0, 0.3, 0.5, 2.0       # final2 config.json
MODE1 = ("consensus", "blend")                                                  # anchors_modes: entity / repeat / boiler / lexicon apply only to these content modes

_A: Optional[Anchors2] = None
_MODES: Dict[str, str] = {}
_CAL: Dict = {}


def init(cal_path: Optional[str] = None) -> Anchors2:
    global _A, _MODES, _CAL
    if _A is None:
        cal_path = cal_path or os.path.join(C.CAL, "anchors_v3.json")
        _CAL = json.load(open(cal_path, encoding="utf-8"))
        _A = Anchors2(_CAL, modes=list(MODE1), off_languages=[], detectors=["entity", "repeat", "boiler"], v2_detectors=["entity2", "copy", "func", "pad"],
                      v2_modes=["qe", "blend", "consensus"], copy_scale=COPY_SCALE, copy_mode="cap", bigram_path=None)
        _MODES = C.content_modes()
    return _A


def mode_of(iso: str) -> str:
    init()
    return _MODES.get(iso, "none")


def rl77() -> List[str]:
    init()
    return sorted(_CAL["languages"])


def _nan() -> float:
    return float("nan")


def copy_raw(A: Anchors2, iso: str, src: str, mt: str) -> Tuple[float, float, float]:
    """(raw copied fraction = max(word level, char level), flagged fraction (0 below the language's human threshold), judged 1/0). NaN when the detector is off for the language
    or the source has too few unprotected words. Same arithmetic as Anchors2.copy_fraction, which is asserted equal in the tests of this module's caller."""
    b = A._blk2(iso, "copy")
    if not b:
        return _nan(), _nan(), 0.0
    wl = A._whitelist(iso, b)
    st = A._src_cache.get_or((src, id(wl)), lambda: T.source_tokens(src, wl))
    r = T.copy_fractions(st, T.out_tokens(mt), k=int(b.get("k", 3)), m=int(b.get("m", 0)), min_unprot=int(b.get("min_unprot", 2)))
    if r.frac_w is None and r.frac_c is None:
        return _nan(), _nan(), 0.0
    raw = max(x for x in (r.frac_w, r.frac_c) if x is not None)
    tw, tc = float(b.get("t_w", 1.0)), float(b.get("t_c", 1.0))
    fl = 0.0
    if r.frac_w is not None and r.frac_w > tw:
        fl = max(fl, r.frac_w)
    if r.frac_c is not None and r.frac_c > tc:
        fl = max(fl, r.frac_c)
    return float(raw), float(fl), 1.0


def sentence_row(iso: str, src: str, hyp: str) -> Dict[str, float]:
    """All sentence-level indicators of one (source, output) pair. NaN = not judged (detector off for the language, text too short); flags are 'deficit > 0'."""
    A = init()
    o: Dict[str, float] = {}
    h = hyp or ""
    ow, sw = AN.words(h), AN.words(src)
    o["n_words"] = len(ow)
    o["empty"] = float(not h.strip())
    o["len_ratio"] = (len(h.strip()) / len(src.strip())) if src.strip() else _nan()
    o["len_bad"] = float(guards.length_ratio_bad(src, h, 0.2, 5.0)) if h.strip() else 0.0
    o["jacc_copy"] = float(guards.is_copy_of_source(src, h, 0.9)) if h.strip() else 0.0
    o["degenerate"] = float(guards.is_repetitive(h)) if h.strip() else 0.0
    o["selfrep"] = float(guards.is_self_repetitive(h)) if h.strip() else 0.0
    # --- substring copy (anchors v2/v3)
    raw, fl, j = copy_raw(A, iso, src, h) if h.strip() else (_nan(), _nan(), 0.0)
    o["copy_frac"], o["copy_flagged_frac"], o["copy_judged"] = raw, fl, j
    o["copy_flag"] = float(fl > 0) if j else _nan()
    # --- entities: raw recall of the source's numbers and names, and the graded v3 penalty
    ec = AN.entity_counts(src, h)
    o["num_n"], o["num_hit"], o["name_n"], o["name_hit"] = ec.n_num, ec.hit_num, ec.n_name, ec.hit_name
    pu, nj = A.entity2_penalty(iso, src, h) if h.strip() else (0.0, 0)
    o["ent2_pen"] = float(pu) if nj else _nan()
    o["ent2_judged"] = float(nj > 0)
    o["ent2_flag"] = float(pu > 0) if nj else _nan()
    # --- stop-word profile, padding
    f = A.func_features(iso, sw, ow)
    if f is not None:
        o["func_fw"], o["func_en"] = float(f["fw"]), float(f["en"])
        d = A.func_deficit(iso, sw, ow)
    else:
        o["func_fw"] = o["func_en"] = _nan()
        d = None
    o["func_def"] = float(d) if d is not None else _nan()
    o["func_flag"] = float(d > 0) if d is not None else _nan()
    o["func_sat"] = float(d >= 0.9) if d is not None else _nan()
    pdef = A.pad_deficit(iso, ow)
    o["pad_def"] = float(pdef) if pdef is not None else _nan()
    o["pad_flag"] = float(pdef > 0) if pdef is not None else _nan()
    o["pad_sat"] = float(pdef >= 0.9) if pdef is not None else _nan()
    o["ttr_seg"] = float(pad_ttr(ow, "seg100")) if len(ow) >= 8 else _nan()
    # --- repetition
    sh = AN.repeat_shares(ow, (3, 4, 5))
    o["rep3"], o["rep4"], o["rep5"] = sh[3], sh[4], sh[5]
    p = AN.prepare(iso, src, h, with_grams=False)
    rd = A.repeat_deficit(iso, p)
    o["rep_def"] = float(rd) if rd is not None else _nan()
    o["rep_flag"] = float(rd > 0) if rd is not None else _nan()
    # --- the size of what the final run's reward would subtract for this output (reward units; copy counted at its cap 2 x fraction; only the detectors that applied to the language's mode)
    m = mode_of(iso)
    ap1 = m in MODE1
    pen = 0.0
    if ap1 and o["ent2_judged"]:
        pen += W_ENTITY * (o["ent2_pen"] if o["ent2_pen"] == o["ent2_pen"] else 0.0)
    if ap1 and rd is not None:
        pen += W_REPEAT * rd
    if d is not None:
        pen += W_FUNC * d
    if pdef is not None:
        pen += W_PAD * pdef
    if j and fl > 0:
        pen += COPY_SCALE * fl
    o["pen_total"] = pen
    return o


# ------------------------------------------------------------------------------------------------ language identification
class Lid:
    def __init__(self):
        from gemma_mt.rl.lid import GlotLIDClassifier

        self.clf = GlotLIDClassifier(path=C.glotlid_path(), cache_size=2_000_000)
        self.clf._load()
        self.blocks = C.lid_blocks()

    def top1(self, texts: Sequence[str]) -> List[Tuple[str, float]]:
        out = []
        for r in self.clf.top(list(texts), 1):
            out.append(r[0] if r else ("und", 0.0))
        return out

    def verdict(self, iso: str, text: str, top: Tuple[str, float]) -> Tuple[float, float]:
        """(judged, wrong): judged = the language has a measured GlotLID block with the guard ENABLED and the text is long enough; wrong = top-1 label not accepted."""
        b = self.blocks.get(iso)
        if not b or not b.get("enabled"):
            return 0.0, _nan()
        if len((text or "").strip()) < max(15, int(b.get("min_chars") or 0)):
            return 0.0, _nan()
        return 1.0, float(top[0] not in set(b.get("labels") or ()))
