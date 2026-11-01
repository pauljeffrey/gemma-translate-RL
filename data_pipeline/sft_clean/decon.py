#!/usr/bin/env python3
"""Hash-based decontamination index with the SAME decisions as gemma_mt/scripts/contam_audit_12_decontaminate.py (rules R1-R4).

Why a second implementation: the reference keeps Python sets/dicts of full strings (fine for 1M protected sentences, too heavy for 3M protected sentences x 2 variants and
tens of millions of pairs). This one stores 64-bit xxhash values in sorted numpy arrays and looks batches up with np.searchsorted. Semantics are identical, and
`test_decon.py` checks that on planted cases and on random text against the reference class.

Rules (applied to each side of a pair separately; a pair is dropped when ANY side has ANY hit):
  R1_exact       normalised text, or one of its sentence parts, equals a protected normalised sentence (protected sentences with < 3 tokens are never indexed)
  R2_contiguous  a protected sentence of >= 8 tokens occurs inside the text as a whole-token run
  R3_8gram       the text shares at least one word 8-gram with a protected sentence (strict rule chosen by the lead for this build)
  R3_near_ge50   >= 50% of the text's 8-grams occur in the protected index (implied by R3_8gram, reported separately like the reference)
  R4 variants    every test is run on two normal forms: NFKC+casefold+quote fold+non-alphanumerics -> space, and the same after NFD + dropping combining marks
Normalisation = contam_audit_12.norm (and therefore gemma_mt/eval/contamination.normalise plus the quote fold and the mark-stripped variant).

Index = sorted uint64 arrays: sentence hash -> set bitmask, 8-gram hash -> set bitmask, (first 8-gram, token count, sentence hash, set bitmask) rows for R2.
Set bitmask: bit i = protected set number i (<= 64 sets).
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import xxhash

_NON = re.compile(r"[\W_]+", re.UNICODE)
_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "″": '"', "ʼ": "'", "ʹ": "'"})
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
N = 8
U64 = np.uint64
_M = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5,
               0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53, 0xD6E8FEB86659FD93, 0xA0761D6478BD642F], dtype=np.uint64)

RULES = ("R1_exact", "R2_contiguous", "R3_8gram", "R3_near_ge50")


def norm(t: str, strip_marks: bool = False) -> str:
    t = unicodedata.normalize("NFKC", t or "").translate(_QUOTES).casefold()
    if strip_marks:
        t = "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c))
    return _NON.sub(" ", t).strip()


_tok_cache: Dict[str, int] = {}


def tok_hash(tok: str) -> int:
    h = _tok_cache.get(tok)
    if h is None:
        if len(_tok_cache) > 3_000_000:
            _tok_cache.clear()
        h = _tok_cache[tok] = xxhash.xxh64_intdigest(tok.encode("utf-8", "surrogatepass"))
    return h


def sent_hash(n: str) -> int:
    return xxhash.xxh64_intdigest(n.encode("utf-8", "surrogatepass"))


def gram_hashes(tokh: np.ndarray) -> np.ndarray:
    """8-gram hashes (uint64, position dependent mix) of one token-hash vector; empty when < 8 tokens."""
    L = len(tokh)
    if L < N:
        return np.empty(0, U64)
    m = L - N + 1
    acc = np.zeros(m, U64)
    with np.errstate(over="ignore"):
        for k in range(N):
            acc += tokh[k:k + m] * _M[k]
        acc ^= acc >> U64(29)
        acc *= U64(0xBF58476D1CE4E5B9)
        acc ^= acc >> U64(32)
    return acc


def variants(text: str) -> List[Tuple[bool, str]]:
    """The normal forms to test. The mark-stripped variant is skipped for ASCII text (identical result by construction: NFKC/NFD are no-ops on ASCII)."""
    out = [(False, norm(text, False))]
    if not text.isascii():
        out.append((True, norm(text, True)))
    return out


def parts_of(text: str, sm: bool) -> List[str]:
    ps = _SPLIT.split(text)
    if len(ps) <= 1:
        return []
    return [norm(p, sm) for p in ps if p.strip()]


class Builder:
    """Accumulates protected text; `finish()` returns an Index."""

    def __init__(self):
        self.set_names: List[str] = []
        self._sid: Dict[str, int] = {}
        self.count: Dict[str, int] = {}          # protected sentences added per set (before dedupe)
        self.s_keys: List[int] = []
        self.s_bits: List[int] = []
        self.g_keys: List[np.ndarray] = []
        self.g_bits: List[int] = []
        self._g_main = (np.empty(0, U64), np.empty(0, U64))
        self._s_main = (np.empty(0, U64), np.empty(0, U64))
        self._f_main = tuple(np.empty(0, U64) for _ in range(4))
        self.dump = None        # optional open text file: 'set<TAB>normalised sentence' per indexed variant (kept LOCAL only, for the independent exposure check; never pushed)
        self.f_rows: List[Tuple[int, int, int, int]] = []     # first gram, token count, sentence hash, bit

    @classmethod
    def extend(cls, ix: "Index") -> "Builder":
        """A builder that starts from an existing index (same set bit numbers, new sets are appended): used to add a protected set after the main build."""
        b = cls()
        b.set_names = list(ix.set_names)
        b._sid = {n: i for i, n in enumerate(ix.set_names)}
        b.count = dict(ix.count)
        b._s_main = (ix.s_keys, ix.s_bits)
        b._g_main = (ix.g_keys, ix.g_bits)
        b._f_main = (ix.f_gram, ix.f_len, ix.f_sent, ix.f_bits)
        return b

    def bit(self, name: str) -> int:
        if name not in self._sid:
            if len(self.set_names) >= 64:
                raise ValueError("more than 64 protected sets")
            self._sid[name] = len(self.set_names)
            self.set_names.append(name)
            self.count[name] = 0
        return 1 << self._sid[name]

    def add(self, name: str, text: str) -> None:
        b = self.bit(name)
        for sm, n in variants(text):
            toks = n.split()
            if len(toks) < 3:
                continue
            sh = sent_hash(n)
            if self.dump is not None:
                self.dump.write(f"{name}\t{int(sm)}\t{n}\n")
            self.s_keys.append(sh)
            self.s_bits.append(b)
            if len(toks) >= N:
                th = np.fromiter((tok_hash(t) for t in toks), U64, len(toks))
                gs = gram_hashes(th)
                self.g_keys.append(np.unique(gs))
                self.g_bits.append(b)
                if len(self.g_keys) >= 100_000:
                    self._compact()
                self.f_rows.append((int(gs[0]), len(toks), sh, b))
        self.count[name] += 1

    def _compact(self) -> None:
        if not self.g_keys:
            return
        gk = np.concatenate(self.g_keys + [self._g_main[0]])
        gb = np.concatenate([np.full(len(a), b, U64) for a, b in zip(self.g_keys, self.g_bits)] + [self._g_main[1]])
        self._g_main = _merge(gk, gb)
        self.g_keys, self.g_bits = [], []

    def add_document(self, name: str, text: str) -> None:
        """A long text (document set): the whole text and each of its sentences are protected."""
        self.add(name, text)
        for p in _SPLIT.split(text):
            if p.strip() and p != text:
                self.add(name, p)

    def finish(self) -> "Index":
        ix = Index()
        ix.set_names = list(self.set_names)
        ix.count = dict(self.count)
        sk = np.concatenate([np.array(self.s_keys, U64), self._s_main[0]])
        sb = np.concatenate([np.array(self.s_bits, U64), self._s_main[1]])
        ix.s_keys, ix.s_bits = _merge(sk, sb)
        self._compact()
        ix.g_keys, ix.g_bits = self._g_main
        if self.f_rows or len(self._f_main[0]):
            f = np.concatenate([np.array(self.f_rows, dtype=U64).reshape(-1, 4), np.stack(self._f_main, axis=1)]) if len(self._f_main[0]) else np.array(self.f_rows, dtype=U64)
            order = np.argsort(f[:, 0], kind="stable")
            f = f[order]
            ix.f_gram, ix.f_len, ix.f_sent, ix.f_bits = (np.ascontiguousarray(f[:, i]) for i in range(4))
        return ix


def _merge(keys: np.ndarray, bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(keys) == 0:
        return keys, bits
    o = np.argsort(keys, kind="stable")
    keys, bits = keys[o], bits[o]
    first = np.concatenate(([True], keys[1:] != keys[:-1]))
    idx = np.nonzero(first)[0]
    return keys[idx], np.bitwise_or.reduceat(bits, idx)


class Index:
    def __init__(self):
        self.set_names: List[str] = []
        self.count: Dict[str, int] = {}
        e = np.empty(0, U64)
        self.s_keys = self.s_bits = self.g_keys = self.g_bits = e
        self.f_gram = self.f_len = self.f_sent = self.f_bits = e
        self.exempt: "Index | None" = None      # protected-2026-10-28.2: sentences / 8-grams that are NOT protected (FLORES+ dev train half, see devhalf.py)

    # ----------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        np.savez(path, s_keys=self.s_keys, s_bits=self.s_bits, g_keys=self.g_keys, g_bits=self.g_bits, f_gram=self.f_gram, f_len=self.f_len,
                 f_sent=self.f_sent, f_bits=self.f_bits, meta=np.array(json.dumps({"set_names": self.set_names, "count": self.count})))

    @classmethod
    def load(cls, path: str) -> "Index":
        z = np.load(path, allow_pickle=False)
        ix = cls()
        for k in ("s_keys", "s_bits", "g_keys", "g_bits", "f_gram", "f_len", "f_sent", "f_bits"):
            setattr(ix, k, z[k])
        m = json.loads(str(z["meta"]))
        ix.set_names, ix.count = m["set_names"], m["count"]
        return ix

    def digest(self) -> Dict[str, str]:
        h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        return {"sentence_hashes_sha256": h(self.s_keys), "sentence_set_bits_sha256": h(self.s_bits), "gram_hashes_sha256": h(self.g_keys),
                "n_sentence_hashes": int(len(self.s_keys)), "n_gram_hashes": int(len(self.g_keys)), "n_first_gram_rows": int(len(self.f_gram))}

    def names(self, mask: int) -> List[str]:
        return [n for i, n in enumerate(self.set_names) if mask >> i & 1]

    # ----------------------------------------------------------------- lookups
    @staticmethod
    def _lookup(keys: np.ndarray, bits: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(keys) == 0 or len(q) == 0:
            return np.zeros(len(q), bool), np.zeros(len(q), U64)
        pos = np.searchsorted(keys, q)
        pos[pos >= len(keys)] = len(keys) - 1
        hit = keys[pos] == q
        return hit, np.where(hit, bits[pos], U64(0))

    def check_texts(self, texts: Sequence[str]) -> List[Tuple[int, int, int, bool]]:
        """-> per text (r1_mask, r2_mask, r3_mask, near_ge50). Masks are protected-set bitmasks (0 = no hit)."""
        n_t = len(texts)
        r1 = np.zeros(n_t, U64)
        r2 = np.zeros(n_t, U64)
        r3 = np.zeros(n_t, U64)
        near = np.zeros(n_t, bool)
        s_q: List[int] = []
        s_owner: List[int] = []
        tok_lists: List[List[str]] = []            # (text idx, variant tokens) for the R2 verification
        flat_tok: List[np.ndarray] = []
        g_owner: List[np.ndarray] = []
        g_tokref: List[Tuple[int, int]] = []       # (index into tok_lists, gram count)
        per_text_grams: Dict[int, int] = {}
        for i, text in enumerate(texts):
            if not text:
                continue
            for sm, n in variants(text):
                if not n:
                    continue
                s_q.append(sent_hash(n)); s_owner.append(i)
                for p in parts_of(text, sm):
                    if p:
                        s_q.append(sent_hash(p)); s_owner.append(i)
                toks = n.split()
                if len(toks) >= N:
                    th = np.fromiter((tok_hash(t) for t in toks), U64, len(toks))
                    gs = gram_hashes(th)
                    flat_tok.append(gs)
                    g_owner.append(np.full(len(gs), i, np.int64))
                    tok_lists.append(toks)
                    g_tokref.append((i, len(gs)))
        if s_q:
            q = np.array(s_q, U64)
            hit, bits = self._lookup(self.s_keys, self.s_bits, q)
            if self.exempt is not None and hit.any():
                eh, _ = self._lookup(self.exempt.s_keys, self.exempt.s_bits, q)
                hit = hit & ~eh
                bits = np.where(hit, bits, U64(0))
            if hit.any():
                own = np.array(s_owner, np.int64)
                np.bitwise_or.at(r1, own[hit], bits[hit])
        if flat_tok:
            gq = np.concatenate(flat_tok)
            go = np.concatenate(g_owner)
            hit, bits = self._lookup(self.g_keys, self.g_bits, gq)
            if self.exempt is not None and hit.any():
                eg, _ = self._lookup(self.exempt.g_keys, self.exempt.g_bits, gq)
                hit = hit & ~eg
                bits = np.where(hit, bits, U64(0))
            if hit.any():
                np.bitwise_or.at(r3, go[hit], bits[hit])
                # near: share of hit grams, per (text, variant) segment
                off = 0
                for (i, m), toks in zip(g_tokref, tok_lists):
                    seg = hit[off:off + m]
                    if seg.sum() * 2 >= m:
                        near[i] = True
                    # R2: verify full protected sentences whose first gram is hit at this position
                    if len(self.f_gram):
                        hp = np.nonzero(seg)[0]
                        if len(hp):
                            fg = gq[off:off + m][hp]
                            lo = np.searchsorted(self.f_gram, fg, "left")
                            hi = np.searchsorted(self.f_gram, fg, "right")
                            for p_, a, b in zip(hp, lo, hi):
                                for r in range(a, b):
                                    L = int(self.f_len[r])
                                    if p_ + L <= len(toks) and sent_hash(" ".join(toks[p_:p_ + L])) == int(self.f_sent[r]):
                                        if self.exempt is not None and self._lookup(self.exempt.s_keys, self.exempt.s_bits, np.array([self.f_sent[r]], U64))[0][0]:
                                            continue
                                        r2[i] |= self.f_bits[r]
                    off += m
        return [(int(r1[i]), int(r2[i]), int(r3[i]), bool(near[i])) for i in range(n_t)]
