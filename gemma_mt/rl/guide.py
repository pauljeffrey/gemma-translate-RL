# SPDX-License-Identifier: AGPL-3.0-or-later
"""Guided group member (owner 2026-11-03): the beam-4 translation, or for weak languages an MBR pick over a LARGER pool, replaces the last sampled member of a group.

Per step, for `fraction` of the prompts (a fixed-size random subset, a function of (seed, key)):
  mode "beam"  the beam-`beams` output of the current policy (HF `generate` on the trainer's policy).
  mode "mbr"   (languages in `mode_by_lang`) minimum Bayes risk over the candidates  group members (all G rows of the prompt: the samples and the greedy member) + the beam output + `extra` additional samples
               (temperature 0.7, top-p 0.95, drawn from the sampler for these prompts only), utility = sentence chrF++ (rl/consensus: the consensus-reward utility; decode study: MBR-16 38.29 vs beam-4 37.53
               vs greedy 35.00). The candidate with the highest mean chrF++ against all the others wins. A winner that is already a member of the group would add nothing (the guide would be a copy of a
               member), so then the beam output is kept as the guide ("mbr_member->beam"). The winner replaces the last sampled member, exactly like the beam guide (ratio 1, hard-failure drop and the
               advantage share cap are applied by the trainer).
Only candidates that pass the cheap hard checks (finished, nothing after the end token, language tag present, non-empty, not a copy of the source, not repetitive) take part. With fewer than 3 valid
candidates the beam output is kept ("mbr_few->beam"); when the running overhead of the guides exceeds `max_overhead_frac` the extra samples are skipped for the step ("mbr_skipped->beam").
`Sample.guide_src` records which case occurred.

The pairwise chrF++ of ~33 candidates is computed with numpy (`chrf_matrix`): one dense n-gram count matrix per order and a broadcast minimum, equal to `consensus.chrf_sentence` pair by pair (tested)."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gemma_mt.languages import canonical_iso
from gemma_mt.rl import consensus as cons
from gemma_mt.rl import guards
from gemma_mt.rl.text import decode_completion, translation_from

MODES = ("beam", "mbr")


@dataclass
class GuideSpec:
    fraction: float
    beams: int = 4
    mode_by_lang: Dict[str, str] = field(default_factory=dict)       # iso -> "mbr" | "beam"; languages not listed use "beam"
    extra: int = 16                                                  # additional samples per MBR prompt (0 = the MBR languages use the beam guide)
    temperature: float = 0.7
    top_p: float = 0.95
    skip_extra: bool = False                                         # the overhead budget is used up: no extra samples this step

    def mode(self, iso: str) -> str:
        return "mbr" if self.extra > 0 and self.mode_by_lang.get(iso) == "mbr" else "beam"


@dataclass
class GuideInfo:
    src: str            # beam | mbr_beam | mbr_extra | mbr_member->beam | mbr_few->beam | mbr_skipped->beam
    mode: str
    candidates: int = 0


# ------------------------------------------------------------------------------------------------ MBR utility
def chrf_matrix(texts: Sequence[str]) -> np.ndarray:
    """F[i, j] = sentence chrF++ of texts[i] as hypothesis against texts[j] as reference (0..100), equal to `consensus.chrf_sentence(texts[i], texts[j])`."""
    n = len(texts)
    profs = [cons.Profile(t) for t in texts]
    tot = np.asarray([p.totals for p in profs], dtype=np.float64)                    # [n, ORDER]
    M = np.zeros((n, n, cons.ORDER), dtype=np.float64)
    for o in range(cons.ORDER):
        vocab: Dict[str, int] = {}
        for p in profs:
            for g in p.grams[o]:
                if g not in vocab:
                    vocab[g] = len(vocab)
        if not vocab:
            continue
        X = np.zeros((n, len(vocab)), dtype=np.int16)
        for i, p in enumerate(profs):
            for g, c in p.grams[o].items():
                X[i, vocab[g]] = c
        for i in range(n):                                                            # row by row: keeps the temporary at [n, V] instead of [n, n, V]
            M[i, :, o] = np.minimum(X[i][None, :], X).sum(axis=1)
    h = tot[:, None, :]                                                               # hypothesis totals  [n, 1, O]
    r = tot[None, :, :]                                                               # reference totals   [1, n, O]
    eff = (h > 0) & (r > 0)                                                           # sacrebleu: an order counts only when both sides have n-grams of it
    cnt = eff.sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where(eff, M / np.where(h > 0, h, 1.0), 0.0).sum(axis=2) / np.maximum(cnt, 1)
        rc = np.where(eff, M / np.where(r > 0, r, 1.0), 0.0).sum(axis=2) / np.maximum(cnt, 1)
        f = np.where((cnt > 0) & (pr + rc > 0), 100.0 * (1 + cons.BETA ** 2) * pr * rc / np.where(pr + rc > 0, cons.BETA ** 2 * pr + rc, 1.0), 0.0)
    return f


def mbr_pick(texts: Sequence[str]) -> Tuple[int, List[float]]:
    """(index of the candidate with the highest mean chrF++ against all the OTHER candidates, that mean per candidate)."""
    F = chrf_matrix(texts)
    n = len(texts)
    np.fill_diagonal(F, 0.0)
    util = (F.sum(axis=1) / max(n - 1, 1)).tolist()
    return int(np.argmax(util)), util


class OverheadTracker:
    """Running share of the step time spent in the guides over the last `window` steps."""

    def __init__(self, window: int = 10):
        self.h: Deque[Tuple[float, float]] = deque(maxlen=window)

    def add(self, guide_s: float, step_s: float) -> None:
        self.h.append((float(guide_s), float(step_s)))

    def running(self) -> float:
        g, s = sum(x[0] for x in self.h), sum(x[1] for x in self.h)
        return g / s if s > 0 else 0.0


# ------------------------------------------------------------------------------------------------ candidate text
def candidate_text(reward: Any, token_ids: Sequence[int], iso: str, source: str, post_stop: int = 0) -> Optional[str]:
    """The translation text of a candidate as the reward would see it, or None when it fails a cheap hard check (such a candidate cannot be a guide that survives the hard-failure drop)."""
    cfg = reward.cfg
    dec = decode_completion(reward.tok, list(token_ids), reward.stop_ids, pad_ids=reward.pad_ids, normalise=cfg.nfc)
    if not dec.finished or dec.post_stop_tokens + int(post_stop) > 0 or dec.dangling:
        return None
    strip = translation_from(dec, iso)
    text = strip.text if strip.has_prefix else None
    if not text or not text.strip():
        return None
    if guards.is_copy_of_source(source, text, cfg.copy_jaccard) or (cfg.repetition_guard and guards.is_repetitive(text)):
        return None
    return text


# ------------------------------------------------------------------------------------------------ the guides of one rollout
def apply_guides(spec: GuideSpec, *, roll: Any, G: int, indices: Sequence[int], prompt_ids: Sequence[Sequence[int]], isos: Sequence[str], sources: Sequence[str], reward: Any, policy: Any,
                 generate_beam: Any, generate_extra: Any, seed: int, key: int, timings: Optional[Dict[str, float]] = None) -> Dict[int, GuideInfo]:
    """Replace the row `g * G + G - 2` of the picked groups by their guide (in place in `roll`); returns {row: GuideInfo}.
    `generate_beam(prompt_ids) -> Rollout` (one beam output per prompt), `generate_extra(prompt_ids, n) -> Rollout` (n samples per prompt, rows grouped per prompt)."""
    timings = timings if timings is not None else {}
    n_g = min(len(indices), max(1, int(round(spec.fraction * len(indices)))))
    pick = sorted(int(x) for x in np.random.default_rng([seed, int(key), 777]).choice(len(indices), size=n_g, replace=False))
    t0 = time.time()
    beam = generate_beam([prompt_ids[g] for g in pick])
    timings["guide_s"] = timings.get("guide_s", 0.0) + time.time() - t0
    modes = {g: spec.mode(isos[g]) for g in pick}
    mbr_groups = [g for g in pick if modes[g] == "mbr"]
    extra = None
    if mbr_groups and not spec.skip_extra:
        t0 = time.time()
        extra = generate_extra([prompt_ids[g] for g in mbr_groups], spec.extra)
        timings["guide_extra_s"] = timings.get("guide_extra_s", 0.0) + time.time() - t0
    info: Dict[int, GuideInfo] = {}
    t_mbr = 0.0
    for j, g in enumerate(pick):
        row = g * G + G - 2
        comp, fin, post, src, ncand = beam.completions[j], beam.finished[j], beam.post_stop_tokens[j], "beam", 0
        if modes[g] == "mbr":
            if extra is None:
                src = "mbr_skipped->beam"
            else:
                t1 = time.time()
                k = mbr_groups.index(g)
                cands: List[Tuple[str, Any, Any, Any]] = []          # (origin, token ids, finished, post-stop tokens)
                for r in range(g * G, g * G + G):
                    cands.append(("member", roll.completions[r], roll.finished[r], roll.post_stop_tokens[r] if roll.post_stop_tokens else 0))
                cands.append(("beam", beam.completions[j], beam.finished[j], beam.post_stop_tokens[j]))
                for e in range(spec.extra):
                    q = k * spec.extra + e
                    cands.append(("extra", extra.completions[q], extra.finished[q], extra.post_stop_tokens[q] if extra.post_stop_tokens else 0))
                texts, keep = [], []
                for ci, (origin, ids_, fin_, post_) in enumerate(cands):
                    t = candidate_text(reward, ids_, isos[g], sources[g], post_)
                    if t is not None:
                        texts.append(t)
                        keep.append(ci)
                ncand = len(texts)
                if ncand < 3:
                    src = "mbr_few->beam"
                else:
                    w, _ = mbr_pick(texts)
                    origin, ids_, fin_, post_ = cands[keep[w]]
                    if origin == "member":
                        src = "mbr_member->beam"
                    elif origin == "beam":
                        src = "mbr_beam"
                    else:
                        comp, fin, post, src = ids_, fin_, post_, "mbr_extra"
                t_mbr += time.time() - t1
        roll.completions[row], roll.finished[row] = comp, fin
        if roll.post_stop_tokens:
            roll.post_stop_tokens[row] = post
        if roll.sampler_logp is not None:
            roll.sampler_logp[row] = [0.0] * len(comp)                  # placeholder: the sampler log-prob check and the confidence rules skip guide rows
        info[row] = GuideInfo(src, modes[g], ncand)
    timings["guide_mbr_s"] = timings.get("guide_mbr_s", 0.0) + t_mbr
    return info
