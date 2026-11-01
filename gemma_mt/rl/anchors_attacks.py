# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic attacks on the weak-signal reward (D-61): outputs that a policy could drift towards when the content signal is consensus / a recall-only dictionary overlap.

All generators are deterministic given the rng and need only the language's word statistics (the `top` list of calibration/funcwords_v1.json) and, for the glosses, a validated R3 lexicon.
A "unit" is {src, ref}; every attack returns one text per unit, so detectors are compared on the same sources.

    en_copy        the English source itself
    other_lang     a human reference of ANOTHER language of similar length
    stuff          distinct frequent CONTENT words of the language (dictionary ranks beyond the function words), a bag of words with the length of the reference
    stuff30        the same, 30 words whatever the reference length (the shape of R3's stuffing attack)
    pad            R3's padded attack: 30 stuffed content words + up to 30 filler words (the words of another human reference of the language, in order, without the stuffed words)
    pad_fit        stuffed content words + the same filler, cut to the length of the reference
    fw_only        function words only, sampled by frequency, length of the reference
    stuff_fw       stuffed content words and function words in the human share of the language, shuffled (an attacker who knows the stop-word profile)
    rep_stuff      30 stuffed words repeated to the length of the reference
    gloss          (lexicon languages) the top lexicon candidate of every content word of the source, in source order - a word-by-word dictionary translation without function words
    gloss_pad      gloss + filler words, length of the reference
    gloss_fw       gloss with function words of the language inserted at the human share (the strongest cheap attack)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

from gemma_mt.rl import textstats as T

ATTACKS = ("en_copy", "other_lang", "stuff", "stuff30", "pad", "pad_fit", "fw_only", "stuff_fw", "rep_stuff")
ATTACKS_GLOSS = ("gloss", "gloss_pad", "gloss_fw")


def content_pool(top: Sequence[Sequence[Any]], fw: Sequence[str], n: int = 500) -> List[str]:
    f = set(fw)
    return [t[0] for t in top if t[0] not in f and len(t[0]) >= 3 and not any(ch.isdigit() for ch in t[0])][:n]


def _sample_distinct(pool: List[str], n: int, rng: random.Random) -> List[str]:
    if n <= 0 or not pool:
        return []
    if n <= len(pool):
        return rng.sample(pool, n)
    out = list(pool)
    rng.shuffle(out)
    while len(out) < n:
        out.append(rng.choice(pool))
    return out


def make_attacks(units: Sequence[Dict[str, str]], top: Sequence[Sequence[Any]], fw: Sequence[str], other_refs: Sequence[str], rng: random.Random, lex: Any = None,
                 fw_share: float = 0.2, n_stuff: int = 30) -> Dict[str, List[str]]:
    pool = content_pool(top, fw)
    fwl = list(fw)
    fw_w = {t[0]: t[1] for t in top}
    weights = [fw_w.get(w, 1) for w in fwl]
    out: Dict[str, List[str]] = {a: [] for a in ATTACKS}
    if lex is not None:
        out.update({a: [] for a in ATTACKS_GLOSS})
    refs = [u["ref"] for u in units]
    for i, u in enumerate(units):
        rw = u["ref"].split()
        L = max(len(rw), 3)
        out["en_copy"].append(u["src"])
        cand = [r for r in (other_refs[rng.randrange(len(other_refs))] for _ in range(6))] if other_refs else [u["src"]]
        out["other_lang"].append(min(cand, key=lambda r: abs(len(r.split()) - L)))
        out["stuff"].append(" ".join(_sample_distinct(pool, L, rng)))
        st30 = _sample_distinct(pool, n_stuff, rng)
        out["stuff30"].append(" ".join(st30))
        filler_src = T.out_tokens(refs[(i + 17) % len(refs)])
        fset = set(st30)
        filler = [w for w in filler_src if w not in fset]
        out["pad"].append(" ".join(st30 + filler[:30]))
        out["pad_fit"].append(" ".join((st30 + filler)[:L]))
        out["fw_only"].append(" ".join(rng.choices(fwl, weights=weights, k=L)) if fwl else "")
        nf = int(round(L * fw_share))
        mixed = _sample_distinct(pool, L - nf, rng) + (rng.choices(fwl, weights=weights, k=nf) if fwl else [])
        rng.shuffle(mixed)
        out["stuff_fw"].append(" ".join(mixed))
        out["rep_stuff"].append(" ".join((st30 * (L // len(st30) + 1))[:L]) if st30 else "")
        if lex is not None:
            gl = []
            for stem in __import__("gemma_mt.rl.lexicon", fromlist=["x"]).english_content_stems(u["src"]):
                e = lex.entries.get(stem)
                if e and e[2]:
                    gl.append(e[2][0][0])
            out["gloss"].append(" ".join(gl))
            gset = set(gl)
            gfill = [w for w in filler_src if w not in gset]
            out["gloss_pad"].append(" ".join((gl + gfill)[:max(L, len(gl))]))
            nfg = int(round(max(len(gl), 1) * fw_share / max(1e-6, 1 - fw_share)))
            gmix = list(gl)
            for _ in range(nfg):
                gmix.insert(rng.randrange(len(gmix) + 1), rng.choices(fwl, weights=weights, k=1)[0] if fwl else "")
            out["gloss_fw"].append(" ".join(gmix))
    return out
