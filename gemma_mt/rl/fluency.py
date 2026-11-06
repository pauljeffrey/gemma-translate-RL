# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference-free FLUENCY signal for the RL reward (D-62, 2026-11-01): compact per-language character n-gram + word-order models, scored with numpy only (no Python dicts, no torch).

What it is. For each language, a character 5-gram model (interpolated Witten-Bell) and a word-order score (mean over adjacent word pairs of log P(w | previous) - log P(w)), trained on
monolingual text of that language and stored as sorted hashed arrays (scripts/build_fluency_models.py -> artifacts/fluency_models/<iso>.npz + manifest.json). A sentence gets
    char_z = (mean log-prob per character - mu_c) / sd_c           (mu, sd measured on 300 held-out sentences of the training text)
    word_z = (word-order score - mu_w) / sd_w                       (None for sentences with fewer than 3 words)
    comb_z = (char_z + word_z) / sqrt(2)  (char_z alone when word_z is None)                      unit scale if the two are independent; the AUCs of the validation are invariant to this factor.
The reward uses comb_z ONLY as a PENALTY below a floor (reward.fluency, default OFF); never as a bonus (a bonus would reward repetitive, over-frequent text).

What it can and cannot see (validated on FLORES+ dev human text, 45 languages, reports/data/fluency_eval_flores_2026-11-01/):
  * garbled characters, text of another language, wrong word order (AUC about 0.99 / 0.99 / 0.87 on average): YES;
  * a CLOSE SIBLING language (zul/xho/nbl/nde, kin/run, swh/swc, bam/dyu ...): partly (AUC about 0.93); `sibling_ratio` adds the likelihood ratio against the sibling model;
  * DIACRITICS: NO. Stripping diacritics is detected at chance level (AUC about 0.55): the web and Wikipedia text the models are trained on writes them inconsistently (Yoruba tone marks,
    Hausa hooked letters). The calibrated diacritic penalty of the reward (D-31, human references) stays in charge of that.
  * adequacy (does it MEAN the source): NO. This is fluency only; AfriCOMET-QE and the consensus signal are the adequacy signals.
Limits: models are trained on web / Wikipedia / translation-target text (noisy, partly machine translated); a language is `usable` only when its shuffle, garble and foreign AUCs are >= 0.8
on human text (FLORES) or on held-out web text (languages without FLORES); unusable languages get no penalty.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import unicodedata
from typing import Dict, List, Optional, Sequence

import numpy as np

ORDER = 5
BOS, EOS = "\x02", "\x03"
_M = np.array([7339824421600949753, 6262745397512020723, 6843411114658059665, 9161425613072040709, 5238603880273327495], dtype=np.uint64)     # fixed odd constants, one per offset
_BIMUL = 4616499379670828959
_MASK = (1 << 64) - 1
PUNCT = ".,;:!?\"'()[]{}«»“”‘’„‟<>–—-…/\\|*"                       # same set as scripts/fluency_lm.py


def words(text: str) -> List[str]:
    return [w for w in (x.strip(PUNCT) for x in text.lower().split()) if w]


def word_hash(w: str) -> int:
    return int.from_bytes(hashlib.blake2b(w.encode("utf-8"), digest_size=8).digest(), "little")


def bigram_hash(ha: int, hb: int) -> int:
    return (ha * _BIMUL + hb) & _MASK


def codes_of(text: str, order: int = ORDER) -> np.ndarray:
    s = BOS * (order - 1) + text + EOS
    return np.fromiter((ord(c) + 1 for c in s), dtype=np.uint64, count=len(s))


def gram_hashes(codes: np.ndarray, k: int, order: int = ORDER) -> np.ndarray:
    """hash of the k-gram ending at every position i >= order - 1 (uint64 wraparound): sum_j codes[i-j] * M[j]."""
    L = len(codes); lo = order - 1
    h = np.zeros(L - lo, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(k):
            h = h + codes[lo - j: L - j] * _M[j]
    return h


def ctx_hashes(codes: np.ndarray, k: int, order: int = ORDER) -> np.ndarray:
    """hash of the (k-1) characters BEFORE every position i >= order - 1 (the context of the k-gram ending at i); 0 for k = 1."""
    L = len(codes); lo = order - 1
    h = np.zeros(L - lo, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(k - 1):
            h = h + codes[lo - 1 - j: L - 1 - j] * _M[j]
    return h


def _lookup(keys: np.ndarray, vals: np.ndarray, h: np.ndarray) -> np.ndarray:
    if len(keys) == 0:
        return np.zeros(len(h), dtype=np.float64)
    idx = np.minimum(np.searchsorted(keys, h), len(keys) - 1)
    return np.where(keys[idx] == h, vals[idx], 0).astype(np.float64)


class CompactLanguage:
    """One language: arrays of the character model and the word-order model + the z-score statistics."""

    def __init__(self, z: Dict[str, np.ndarray], meta: Dict):
        self.z = z
        self.order = int(meta.get("order", ORDER))
        self.V = int(z["V"][0])
        self.N = int(z["N"][0]) if "N" in z else 0                    # char-only (spelling) models have no word arrays
        self.mu_c, self.sd_c = meta["char"]["mu"], meta["char"]["sd"]
        w = meta.get("word")
        self.mu_w, self.sd_w = (w["mu"], w["sd"]) if w else (0.0, 1.0)
        self.has_word = "uk" in z
        self.meta = meta

    # -- raw scores ---------------------------------------------------------------------------------------------
    def char_score(self, text: str) -> float:
        codes = codes_of(text, self.order)
        m = len(codes) - (self.order - 1)
        p = np.full(m, 1.0 / self.V)
        for k in range(1, self.order + 1):
            tot = _lookup(self.z[f"ck{k}"], self.z[f"ct{k}"], ctx_hashes(codes, k, self.order))
            typ = _lookup(self.z[f"ck{k}"], self.z[f"cy{k}"], ctx_hashes(codes, k, self.order))
            cnt = _lookup(self.z[f"gk{k}"], self.z[f"gc{k}"], gram_hashes(codes, k, self.order))
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = tot / (tot + typ)
                newp = lam * cnt / tot + (1.0 - lam) * p
            p = np.where(tot > 0, newp, p)
        return float(np.log(p).sum() / m)

    def word_score(self, text: str) -> Optional[float]:
        w = words(text)
        if len(w) < 3 or not self.has_word:
            return None
        hs = [word_hash(x) for x in w]
        ha = np.array(hs[:-1], dtype=np.uint64); hb = np.array(hs[1:], dtype=np.uint64)
        cv = _lookup(self.z["uk"], self.z["uc"], ha); cw = _lookup(self.z["uk"], self.z["uc"], hb)
        ty = _lookup(self.z["tk"], self.z["tv"], ha)
        bk = np.array([bigram_hash(int(a), int(b)) for a, b in zip(ha, hb)], dtype=np.uint64)
        cb = _lookup(self.z["bk"], self.z["bc"], bk)
        out = np.zeros(len(ha))
        ok = (cv > 0) & (cw > 0)
        if ok.any():
            lam = cv[ok] / (cv[ok] + ty[ok] + 1.0); pu = cw[ok] / self.N
            pb = lam * cb[ok] / cv[ok] + (1.0 - lam) * pu
            out[ok] = np.log(pb / pu)
        return float(out.mean())

    def score(self, text: str) -> Dict[str, Optional[float]]:
        c = self.char_score(text); w = self.word_score(text)
        cz = (c - self.mu_c) / self.sd_c
        wz = None if w is None else (w - self.mu_w) / self.sd_w
        comb = cz if wz is None else (cz + wz) / math.sqrt(2.0)
        return {"char": c, "word": w, "char_z": cz, "word_z": wz, "comb_z": comb}


class FluencyScorer:
    """Loads artifacts/fluency_models/ (manifest.json + one npz per language) and scores sentences. `usable(iso)`: the language passed the validity test."""

    def __init__(self, directory: str):
        self.dir = directory
        self.manifest = json.load(open(os.path.join(directory, "manifest.json"), encoding="utf-8"))
        self.langs: Dict[str, CompactLanguage] = {}
        for iso, meta in self.manifest["languages"].items():
            z = np.load(os.path.join(directory, meta["file"]))
            self.langs[iso] = CompactLanguage({k: z[k] for k in z.files}, dict(meta, order=self.manifest.get("order", ORDER)))
        self.spell: Dict[str, CompactLanguage] = {}            # char-only SPELLING models trained on Bible text (a few languages: where they police diacritics, AUC >= 0.8 on FLORES)
        for iso, meta in self.manifest["languages"].items():
            sp = meta.get("spelling")
            if sp and sp.get("enabled"):
                z = np.load(os.path.join(directory, sp["file"]))
                self.spell[iso] = CompactLanguage({k: z[k] for k in z.files}, {"order": self.manifest.get("order", ORDER), "char": {"mu": sp["mu"], "sd": sp["sd"]}})
        self.siblings: Dict[str, List[str]] = {i: [s for s in m.get("siblings", []) if s in self.langs] for i, m in self.manifest["languages"].items()}

    @classmethod
    def load(cls, directory: str) -> "FluencyScorer":
        return cls(directory)

    def has(self, iso: str) -> bool:
        return iso in self.langs

    def usable(self, iso: str) -> bool:
        return iso in self.langs and bool(self.manifest["languages"][iso].get("usable"))

    def score(self, iso: str, text: str) -> Dict[str, object]:
        """dict(char_z, word_z, comb_z, usable) of `text` under the model of `iso` (NFC-normalised first); unknown language: usable False and None scores."""
        if iso not in self.langs:
            return {"char_z": None, "word_z": None, "comb_z": None, "usable": False}
        r = self.langs[iso].score(unicodedata.normalize("NFC", text))
        return {"char_z": r["char_z"], "word_z": r["word_z"], "comb_z": r["comb_z"], "usable": self.usable(iso)}

    def spelling_z(self, iso: str, text: str) -> Optional[float]:
        """z-score of the character log-prob under the BIBLE-trained model of `iso` (consistent diacritics / orthography), None when the language has no validated spelling model.
        The floor below which the reward penalises is `spelling_floor(iso)` (from held-out Bible verses)."""
        m = self.spell.get(iso)
        if m is None:
            return None
        return (m.char_score(unicodedata.normalize("NFC", text)) - m.mu_c) / m.sd_c

    def spelling_floor(self, iso: str) -> Optional[float]:
        sp = (self.manifest["languages"].get(iso) or {}).get("spelling")
        return None if not (sp and sp.get("enabled")) else float(sp["floor"])

    def sibling_ratio(self, iso: str, text: str) -> Optional[float]:
        """mean log-prob per character under `iso` minus the best (largest) one under any of its sibling models: > 0 = the text is closer to `iso` than to every sibling.
        None when the language has no sibling model."""
        sibs = self.siblings.get(iso) or []
        if iso not in self.langs or not sibs:
            return None
        t = unicodedata.normalize("NFC", text)
        return self.langs[iso].char_score(t) - max(self.langs[s].char_score(t) for s in sibs)

    def ram_bytes(self) -> int:
        return sum(a.nbytes for l in self.langs.values() for a in l.z.values())
