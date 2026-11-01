# SPDX-License-Identifier: AGPL-3.0-or-later
"""RL prompts: schema, loading, and the deterministic prompt schedule. No reference translation is needed.

Schema (parquet or jsonl), one row per prompt:
    english         str   English source text
    iso             str   target language tag (canonical tags of gemma_mt.languages)
    language        str   display name of the language (checked against the canonical name, mismatches reported)
    src_afr_tokens  int   length of `english` in Davlan/afro-xlmr-large-76L tokens (the AfriCOMET encoder), <= 256
    source_dataset  str   where the sentence came from (kept for diagnostics / per-source reporting)
    domain          str   optional (D-52, science run): "science" / "general"; read by `PromptSchedule(domain=..., domain_weights=...)`
Extra columns are ignored. Rows with an unknown language, a language outside SUPPORTED_RL, an empty source or
`src_afr_tokens` above the limit are dropped and counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gemma_mt.data.io import iter_rows
from gemma_mt.languages import LANGUAGES, SUPPORTED_RL, canonical_iso

RL_COLUMNS = ("english", "iso", "language", "src_afr_tokens", "source_dataset", "length_bin", "domain")
MAX_SRC_AFR_TOKENS = 256
# Input-length bins (AfriCOMET-encoder tokens of the source), named by their UPPER bound (owner 2026-10-12: 64, 256, 512, 1024
# for Gemma, 2048 added for SabiYarn). Prompt sets are balanced uniformly across them per language; training is staged
# b64 -> b256 -> b512 -> b1024 (-> b2048) and ends in a random mixture of all bins.
LENGTH_BINS = ("b64", "b256", "b512", "b1024", "b2048")
BIN_EDGES = {"b64": (0, 65), "b256": (65, 257), "b512": (257, 513), "b1024": (513, 1025), "b2048": (1025, 2049)}
MIXED = "mixed"                                  # the random-mixture stage (no single bin)


def bin_of_tokens(n: Optional[int]) -> str:
    n = 0 if n is None else int(n)
    for name in LENGTH_BINS:
        lo, hi = BIN_EDGES[name]
        if lo <= n < hi:
            return name
    return LENGTH_BINS[-1]


@dataclass
class RLPrompts:
    english: List[str]
    iso: List[str]
    source_dataset: List[str]
    src_afr_tokens: List[Optional[int]]
    dropped: Dict[str, int]
    name_mismatches: int = 0
    length_bin: List[str] = field(default_factory=list)       # one of LENGTH_BINS per prompt (derived from src_afr_tokens)
    domain: List[str] = field(default_factory=list)           # D-52: "" / "general" / "science" ... per prompt (empty list = the file has no domain column)

    def __len__(self) -> int:
        return len(self.english)

    def item(self, i: int) -> Dict[str, Any]:
        return {"index": i, "english": self.english[i], "iso": self.iso[i], "source_dataset": self.source_dataset[i]}

    def domain_of(self, i: int) -> str:
        return self.domain[i] if self.domain else ""

    def languages(self) -> List[str]:
        return sorted(set(self.iso))


def load_rl_prompts(path: str, *, max_src_tokens: int = MAX_SRC_AFR_TOKENS, languages: Optional[Sequence[str]] = None,
                    token_counter: Optional[Callable[[Sequence[str]], Sequence[int]]] = None,
                    limit: int = 0) -> RLPrompts:
    """`token_counter` fills `src_afr_tokens` for rows that lack it (AfriCOMET encoder tokens)."""
    keep_langs = {canonical_iso(l) for l in languages} if languages else None
    eng, iso, src, ntok, lbin, dom = [], [], [], [], [], []
    dropped: Dict[str, int] = {}
    mism = 0

    def drop(why: str):
        dropped[why] = dropped.get(why, 0) + 1

    for row in iter_rows(path, columns=RL_COLUMNS):
        e, tag = row.get("english"), canonical_iso(row.get("iso"))
        if not isinstance(e, str) or not e.strip():
            drop("empty_english")
            continue
        if tag is None:
            drop("unknown_iso")
            continue
        if tag not in SUPPORTED_RL:
            drop("not_rl_supported")
            continue
        if keep_langs is not None and tag not in keep_langs:
            drop("language_filtered")
            continue
        n = row.get("src_afr_tokens")
        if n is not None and int(n) > max_src_tokens:
            drop("source_too_long")
            continue
        name = row.get("language")
        if name and str(name) != LANGUAGES[tag].name:
            mism += 1
        eng.append(e.strip())
        iso.append(tag)
        src.append(str(row.get("source_dataset") or ""))
        ntok.append(None if n is None else int(n))
        lb = row.get("length_bin")
        lbin.append(str(lb) if lb in LENGTH_BINS else "")
        dm = row.get("domain")
        dom.append(str(dm) if isinstance(dm, str) else "")
        if limit and len(eng) >= limit:
            break
    if token_counter is not None:
        missing = [i for i, n in enumerate(ntok) if n is None]
        if missing:
            counts = token_counter([eng[i] for i in missing])
            for i, c in zip(missing, counts):
                ntok[i] = int(c)
        keep = [i for i, n in enumerate(ntok) if n is None or n <= max_src_tokens]
        if len(keep) != len(eng):
            drop_n = len(eng) - len(keep)
            dropped["source_too_long"] = dropped.get("source_too_long", 0) + drop_n
            eng, iso, src, ntok, lbin, dom = ([x[i] for i in keep] for x in (eng, iso, src, ntok, lbin, dom))
    lbin = [b or bin_of_tokens(n) for b, n in zip(lbin, ntok)]
    return RLPrompts(eng, iso, src, ntok, dropped, mism, lbin, dom if any(dom) else [])


class PromptSchedule:
    """Deterministic, stateless prompt schedule (resume needs only the step counter).

    Language l is drawn with probability proportional to n_l ** alpha (alpha=1: natural frequency, alpha=0: uniform
    over languages), then a prompt of that language uniformly (with replacement). Ranks take disjoint strided
    slices of the same global draw."""

    def __init__(self, iso: Sequence[str], alpha: float = 1.0, seed: int = 0, length_bin: Optional[Sequence[str]] = None,
                 curriculum: Optional[Dict[str, int]] = None, language_weights: Optional[Dict[str, float]] = None,
                 domain: Optional[Sequence[str]] = None, domain_weights: Optional[Dict[str, float]] = None,
                 domain_bin_weights: Optional[Dict[str, Dict[str, float]]] = None, floor_steps: int = 0, floor_prompts: int = 1):
        """`length_bin` (one entry per prompt) switches on the length stages. With `curriculum` ({bin: steps}, in order) the
        first stage draws ONLY bin-1 prompts for its steps, then bin 2, ...; after the last stage every step is "mixed":
        each prompt's bin is drawn uniformly over the bins present. Without `curriculum` the bins are cycled step by step.
        Bins without prompts are skipped. The language is always drawn by n_l ** alpha inside the bin.

        DOMAIN MIXTURE (D-52, science run; default off = the stream above is untouched): with `domain` (one label per prompt) and `domain_weights` ({label: share}) every prompt
        of a step first draws its DOMAIN (probability proportional to the weight, over the domains that have prompts in the current stage), then its bin (a single-bin stage: that
        bin; the mixed stage: `domain_bin_weights[domain]` = {bin: weight}, default uniform over the bins the domain has), then the language (n_l ** alpha x language_weights inside
        that domain+bin pool), then the prompt. Needs `length_bin`. Stateless like the rest: the draw depends on (seed, step) only.

        LANGUAGE FLOOR (D-59, default off; needs the domain mixture): with `floor_steps` = P > 0 every language of the table gets AT LEAST `floor_prompts` = m prompts in every aligned block of P
        steps (steps k*P .. k*P+P-1), whatever the random draw does. The block's m x |languages| forced slots are shuffled with (seed, block) and spread evenly over its P steps (so a step
        holds floor(m*|L|/P) or one more forced slots); a forced slot still draws its domain and bin as usual, only the LANGUAGE is fixed (the language of a forced slot is replaced by a random
        one only if the drawn (domain, bin) pool lacks it). The remaining slots are drawn exactly as without the floor. Still stateless in (seed, step)."""
        arr = np.asarray(list(iso))
        self.floor_steps, self.floor_prompts = int(floor_steps or 0), int(floor_prompts)
        if self.floor_steps < 0 or self.floor_prompts < 1:
            raise ValueError("floor_steps must be >= 0 and floor_prompts >= 1")
        if self.floor_steps and not domain_weights:
            raise ValueError("the language floor (floor_steps > 0) needs the domain mixture (domain_weights)")
        self.floor_langs = sorted(set(arr.tolist()))
        self.alpha, self.seed = alpha, seed
        # multiplies the sampling probability of a language (owner 2026-10-13: low-resource languages are drawn more often)
        self.language_weights = {canonical_iso(k): float(v) for k, v in (language_weights or {}).items()}
        present = [b for b in LENGTH_BINS if length_bin is not None and any(x == b for x in length_bin)]
        self.cycle = present
        self.stages = [(b, int(n)) for b, n in (curriculum or {}).items() if b in present and int(n) > 0]
        self.curriculum = bool(curriculum)
        self.domain_weights = {str(k): float(v) for k, v in (domain_weights or {}).items()}
        self.domain_bin_weights = {str(d): {str(b): float(w) for b, w in bw.items()} for d, bw in (domain_bin_weights or {}).items()}
        self.dom_data: Dict[Tuple[str, str], tuple] = {}
        if self.domain_weights:
            if length_bin is None or domain is None or len(domain) != len(arr):
                raise ValueError("domain_weights needs `length_bin` and one `domain` label per prompt")
            if any(v < 0 for v in self.domain_weights.values()) or sum(self.domain_weights.values()) <= 0:
                raise ValueError("domain_weights must be non-negative with a positive sum")
            for d, bw in self.domain_bin_weights.items():
                bad = set(bw) - set(LENGTH_BINS)
                if bad or any(w < 0 for w in bw.values()):
                    raise ValueError(f"domain_bin_weights[{d!r}]: unknown bin(s) {sorted(bad)} or a negative weight")
        if length_bin is None:
            self.langs, self.by_lang, self.p = self._build(arr, np.arange(len(arr)))
            self.bins_data = {}
        else:
            lb = np.asarray(list(length_bin))
            self.bins_data = {b: self._build(arr, np.nonzero(lb == b)[0]) for b in present}
            self.langs, self.by_lang, self.p = (None, None, None)
            if self.domain_weights:
                dm = np.asarray(list(domain))
                for d in sorted(set(self.domain_weights)):
                    for b in present:
                        rows = np.nonzero((dm == d) & (lb == b))[0]
                        if len(rows):
                            self.dom_data[(d, b)] = self._build(arr, rows)
                missing = [d for d, w in self.domain_weights.items() if w > 0 and not any(k[0] == d for k in self.dom_data)]
                if missing:
                    raise ValueError(f"domain_weights lists {missing} but no prompt carries that domain label")

    def _p(self, langs, by_lang):
        w = np.asarray([len(ix) ** self.alpha * self.language_weights.get(l, 1.0) for l, ix in zip(langs, by_lang)], dtype=np.float64)
        if not np.all(np.isfinite(w)) or w.sum() <= 0:
            raise ValueError("a prompt pool has no language with a positive draw weight")
        return w / w.sum()

    def _build(self, arr, rows):
        langs = sorted(set(arr[rows].tolist()))
        by_lang = [rows[arr[rows] == l] for l in langs]
        return langs, by_lang, self._p(langs, by_lang)

    def set_language_weights(self, language_weights: Dict[str, float]) -> None:
        """D-65: replace the draw weights from now on (run control LANG_WEIGHTS.json / adaptive updater, rl/lang_weights.py). The draw stays a pure function of (seed, step, weights in force):
        only the probability vectors of the pools are recomputed, the pools themselves are untouched. Raises ValueError (and changes nothing) when a pool would be left without a positive weight."""
        new = {canonical_iso(k): float(v) for k, v in language_weights.items()}
        if None in new or any(v < 0 or not np.isfinite(v) for v in new.values()):
            raise ValueError("language_weights: unknown language or a negative / non-finite weight")
        old = self.language_weights
        self.language_weights = new
        try:
            if self.p is not None:
                p = self._p(self.langs, self.by_lang)
            bins = {b: (l, bl, self._p(l, bl)) for b, (l, bl, _) in self.bins_data.items()}
            doms = {k: (l, bl, self._p(l, bl)) for k, (l, bl, _) in self.dom_data.items()}
        except ValueError:
            self.language_weights = old
            raise
        if self.p is not None:
            self.p = p
        self.bins_data, self.dom_data = bins, doms

    def bin_of_step(self, step: int) -> Optional[str]:
        """Bin trained at `step`; MIXED after the staged part of a curriculum; None when bins are not in use."""
        if not self.cycle:
            return None
        if self.curriculum:
            s = int(step)
            for b, n in self.stages:
                if s < n:
                    return b
                s -= n
            return MIXED
        return self.cycle[step % len(self.cycle)]

    def stage_boundaries(self) -> List[Tuple[str, int]]:
        """[(bin, first step of the stage), ..., (MIXED, first mixed step)] for logging."""
        out, s = [], 0
        for b, n in self.stages:
            out.append((b, s))
            s += n
        return out + [(MIXED, s)] if self.curriculum else out

    def floor_slots(self, step: int) -> List[str]:
        """Languages whose slots this step is forced to hold (D-59 floor; [] when the floor is off)."""
        if not self.floor_steps:
            return []
        P, m, L = self.floor_steps, self.floor_prompts, self.floor_langs
        block, j = divmod(int(step), P)
        order = np.random.default_rng([self.seed, 7919, block]).permutation(np.repeat(np.arange(len(L)), m))
        F = len(order)
        return [L[int(k)] for k in order[(j * F) // P:((j + 1) * F) // P]]

    def _domain_indices(self, rng, cur: Optional[str], total: int, forced: Sequence[str] = ()) -> List[int]:
        """D-52: domain -> bin -> language -> prompt, per prompt (see __init__). `forced` (D-59): languages of the first len(forced) prompts."""
        if len(forced) > total:
            raise ValueError(f"the language floor needs {len(forced)} forced prompts in this step but the step has only {total}: raise floor_steps or prompts_per_step")
        picks: List[int] = []
        for k in range(total):
            cand = [d for d, w in self.domain_weights.items() if w > 0 and any(k[0] == d and (cur == MIXED or k[1] == cur) for k in self.dom_data)]
            if not cand:
                raise ValueError(f"no domain has prompts for the stage {cur!r}")
            w = np.asarray([self.domain_weights[d] for d in cand], dtype=np.float64)
            d = cand[int(rng.choice(len(cand), p=w / w.sum()))]
            if cur == MIXED:
                bins = [b for b in self.cycle if (d, b) in self.dom_data]
                bw = self.domain_bin_weights.get(d)
                wb = np.asarray([(bw.get(b, 0.0) if bw else 1.0) for b in bins], dtype=np.float64)
                if wb.sum() <= 0:
                    raise ValueError(f"domain_bin_weights[{d!r}] gives weight 0 to every bin the domain has ({bins})")
                b = bins[int(rng.choice(len(bins), p=wb / wb.sum()))]
            else:
                b = cur
            langs_, by_lang, p = self.dom_data[(d, b)]
            l = int(rng.choice(len(langs_), p=p))
            if k < len(forced) and forced[k] in langs_:
                l = langs_.index(forced[k])
            picks.append(int(by_lang[l][int(rng.integers(len(by_lang[l])))]))
        return picks

    def indices(self, step: int, n_prompts: int, rank: int = 0, world: int = 1) -> List[int]:
        rng = np.random.default_rng([self.seed, step])
        cur = self.bin_of_step(step)
        total = n_prompts * world
        if self.domain_weights and self.cycle:
            return self._domain_indices(rng, cur, total, self.floor_slots(step))[rank::world]
        if cur == MIXED:                               # every prompt: a uniformly drawn bin, then language, then prompt
            picks = []
            bins = self.cycle
            for _ in range(total):
                langs_, by_lang, p = self.bins_data[bins[int(rng.integers(len(bins)))]]
                l = int(rng.choice(len(langs_), p=p))
                picks.append(int(by_lang[l][int(rng.integers(len(by_lang[l])))]))
            return picks[rank::world]
        langs_, by_lang, p = (self.bins_data[cur] if self.cycle else (self.langs, self.by_lang, self.p))
        langs = rng.choice(len(langs_), size=total, p=p)
        u = rng.random(total)
        picks = [int(by_lang[l][int(u[k] * len(by_lang[l]))]) for k, l in enumerate(langs)]
        return picks[rank::world]
