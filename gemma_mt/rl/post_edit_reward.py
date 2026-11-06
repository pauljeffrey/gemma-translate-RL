# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reward of the post-editor RL (2026-11-03): did the edit make the draft better, and was it the right edit?

One sample = one (English source, draft translation) prompt, one editor output. Everything here is pure Python on text (no GPU); the two model-based numbers
(quality of the draft and of the edit, `q_d` / `q_e`) come from the existing translation reward (`TranslationReward`: QE ensemble, shaped, failures at the floor) and
are passed in, so the editor is judged by exactly the scorer the translator is.

Components (per sample; `Weights` holds the defaults, all of them are to be checked against the human-reference chrF++ gain before a run, see docs/POST_EDITOR_RL.md):

    gain      quality gain q_e - q_d with a dead zone (|delta| <= tau is noise), regressions weighted `kappa` times, gains on DEFECTIVE drafts weighted up (1 + focus x severity)
    ref       (only prompts that have a human reference) tanh of the chrF++ gain against the reference
    ledger    what the edit did, op by op (word-level alignment of draft -> edit):
              with a reference: fixed = share of the reference words the draft lacked that the edit now has; broke = share of the reference words the draft had that the edit lost
              without: source anchors (numbers, names) the edit recovered / lost, numbers the edit invented
    repair    repetition (4-gram) removed / created
    diacritics (optional, passed in) reduction of the diacritic deficit
    ungrounded insertions of >= 3 new words that are neither in the source, the draft nor the reference, charged only when the QE gain is not above the dead zone (hallucination)
    churn     edits that change words without any measured gain cost per changed word; copying a DEFECTIVE draft costs `copy` x severity; copying a clean draft is 0
              (the unchanged draft is also a member of every sampled group, so GRPO measures every edit against doing nothing)

Why a plain "reward(edit) - reward(draft)" is not enough: GRPO standardises inside a group and every member of a group shares one draft, so q_d cancels. The difference only
steers training through what is NOT constant: the identity member, the dead zone, the regression asymmetry and the defect focus. They are all here.
"""
from __future__ import annotations

import difflib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from gemma_mt.rl import guards
from gemma_mt.rl import lexicon as lex

_NUM = re.compile(r"\d")
_CAP = re.compile(r"[A-Za-zÀ-ɏ]")


@dataclass
class Weights:
    gain: float = 1.0
    ref: float = 0.6
    ref_tanh_scale: float = 15.0        # chrF++ points at which tanh reaches 0.76
    fix: float = 0.5
    brk: float = 1.0                    # breaking a reference word costs twice what fixing one earns
    anchor_rec: float = 0.3
    anchor_lost: float = 0.6
    invented_num: float = 0.5
    repair: float = 0.4
    diacritics: float = 0.2
    ungrounded: float = 0.3
    churn: float = 0.15                 # x edit_rate when the edit has no measured gain
    copy: float = 0.5                   # x defect severity when a defective draft is returned unchanged
    tau: float = 0.02                   # QE noise margin (to be re-set from group_stats.jsonl percentiles)
    kappa: float = 2.0                  # regressions count kappa times
    q_lo: float = -1.0                  # scored range of the translation reward (read from shaper.reward_range at run time)
    q_hi: float = 1.0
    focus: float = 1.0                  # a gain on a draft of severity 1 counts (1 + focus) times: the editor is paid most where the draft is missing something
    severity_q_scale: float = 0.25      # QE points below the language's low anchor that make a draft fully defective
    clip: float = 2.0
    ungrounded_min_words: int = 3
    refcomet: float = 1.0               # reference-based AfriCOMET gain (prompts with a human reference, ~21 languages): weight of tanh(dead-zoned delta / scale)
    refcomet_tau: float = 0.01          # dead zone of the reference-COMET delta (its noise level; to be re-set from the measured noise)
    refcomet_scale: float = 0.05        # delta at which the tanh reaches 0.76


@dataclass
class Ops:
    inserted: int = 0
    deleted: int = 0
    substituted: int = 0                # words of the edit that replace draft words (max of the two spans)
    diacritic_only: int = 0             # substitutions that are the same word up to tone / accent marks
    edit_rate: float = 0.0
    identical: bool = False


@dataclass
class Components:
    gain: float = 0.0
    ref: float = 0.0
    refcomet: float = 0.0
    ledger: float = 0.0
    repair: float = 0.0
    diacritics: float = 0.0
    ungrounded: float = 0.0
    churn: float = 0.0
    copy: float = 0.0
    total: float = 0.0
    detail: Dict[str, float] = field(default_factory=dict)


def tokens(text: str) -> List[str]:
    """`lexicon.x_tokens` that also keeps numbers (an edit that only changes a number must not look like a copy)."""
    out: List[str] = []
    for tok in unicodedata.normalize("NFC", text).lower().translate(lex._APOS).split():
        i, j = 0, len(tok)
        while i < j and lex._is_edge_junk(tok[i]):
            i += 1
        while j > i and lex._is_edge_junk(tok[j - 1]):
            j -= 1
        tok = tok[i:j]
        if tok and any(c.isalnum() for c in tok):
            out.append(tok)
    return out


def align_ops(draft: str, edit: str) -> Tuple[Ops, List[Tuple[str, Sequence[str], Sequence[str]]]]:
    """Word-level alignment draft -> edit. Returns the counts and the opcodes (tag, draft words, edit words) that are not 'equal'."""
    a, b = tokens(draft), tokens(edit)
    if a == b:
        return Ops(identical=True), []
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ops = Ops()
    changed: List[Tuple[str, Sequence[str], Sequence[str]]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        da, eb = a[i1:i2], b[j1:j2]
        changed.append((tag, da, eb))
        if tag == "insert":
            ops.inserted += len(eb)
        elif tag == "delete":
            ops.deleted += len(da)
        else:
            ops.substituted += max(len(da), len(eb))
            if len(da) == len(eb):
                ops.diacritic_only += sum(1 for x, y in zip(da, eb) if x != y and lex.fold(x) == lex.fold(y))
    ops.edit_rate = (ops.inserted + ops.deleted + ops.substituted) / max(len(a), len(b), 1)
    return ops, changed


def source_anchors(source: str) -> Tuple[set, set]:
    """(numbers, names) of the English source, folded: numbers keep their digits, names are capitalised words that do not start a sentence."""
    nums, names = set(), set()
    for sent in re.split(r"(?<=[.!?])\s+", source):
        for k, w in enumerate(re.findall(r"[\w'’-]+", sent)):
            if _NUM.search(w):
                nums.add(lex.fold(w))
            elif k > 0 and w[:1].isupper() and len(w) > 1 and not w.isupper():
                names.add(lex.fold(w))
    return nums, names


def anchor_tokens(text: str) -> List[str]:
    """Folded word tokens that KEEP numbers (`tokens` drops tokens without a letter)."""
    return [lex.fold(t) for t in re.findall(r"[\w'\u2019-]+", text)]


def _present(anchor_set: set, text: str) -> set:
    folded = set(anchor_tokens(text))
    return {a for a in anchor_set if a in folded}


def invented_numbers(source: str, draft: str, edit: str) -> int:
    """Numbers in the edit that are neither in the source nor in the draft."""
    known = {lex.fold(t) for t in re.findall(r"\d[\d.,]*", source + " " + draft)}
    return sum(1 for t in re.findall(r"\d[\d.,]*", edit) if lex.fold(t) not in known)


def defect_severity(source: str, draft: str, q_d: float, w: Weights, *, finished: bool = True, wrong_language: bool = False, q_low: Optional[float] = None) -> float:
    """0 = nothing wrong with the draft that we can see, 1 = clearly broken (empty / unfinished / looping / wrong language / lost anchors / QE below the language's low anchor)."""
    if not draft.strip() or not finished or wrong_language:
        return 1.0
    sev = 0.0
    rep = guards.repeated_ngram_fraction(draft)
    if guards.is_repetitive(draft) or guards.is_self_repetitive(draft):
        sev = max(sev, 1.0)
    else:
        sev = max(sev, min(1.0, rep / 0.35))
    nums, names = source_anchors(source)
    anchors = nums | names
    if anchors:
        lost = len(anchors - _present(anchors, draft)) / len(anchors)
        sev = max(sev, min(1.0, lost * 1.5))
    if q_low is not None and q_d < q_low:
        sev = max(sev, min(1.0, (q_low - q_d) / w.severity_q_scale))
    return sev


def score(source: str, draft: str, edit: str, q_d: float, q_e: float, *, reference: str = "", chrf_d: Optional[float] = None, chrf_e: Optional[float] = None,
          dia_d: Optional[float] = None, dia_e: Optional[float] = None, severity: float = 0.0, w: Weights = Weights(), failed: bool = False, rc_d: Optional[float] = None, rc_e: Optional[float] = None,
          failure_value: float = -2.0) -> Components:
    """Reward of one editor output. `failed` (format failure / wrong language / loop created / unfinished) returns the failure value untouched."""
    c = Components()
    if failed:
        c.total = failure_value
        return c
    ops, _ = align_ops(draft, edit)
    if ops.identical:
        c.copy = -w.copy * severity
        c.total = c.copy
        c.detail = {"identical": 1.0, "severity": severity}
        return c

    delta = q_e - q_d
    if delta > w.tau:
        eff = delta - w.tau
    elif delta < -w.tau:
        eff = w.kappa * (delta + w.tau)
    else:
        eff = 0.0
    c.gain = w.gain * max(-w.clip, min(w.clip, eff * (1.0 + w.focus * severity if eff > 0 else 1.0)))

    if reference.strip() and chrf_d is not None and chrf_e is not None:
        c.ref = w.ref * math.tanh((chrf_e - chrf_d) / w.ref_tanh_scale)

    if reference.strip() and rc_d is not None and rc_e is not None:          # reference-based COMET gain with a dead zone and the same regression asymmetry
        dr = rc_e - rc_d
        er = dr - w.refcomet_tau if dr > w.refcomet_tau else (w.kappa * (dr + w.refcomet_tau) if dr < -w.refcomet_tau else 0.0)
        c.refcomet = w.refcomet * math.tanh(er / w.refcomet_scale)
        c.detail["refcomet_delta"] = dr

    d_toks, e_toks = tokens(draft), tokens(edit)
    nums, names = source_anchors(source)
    anchors = nums | names
    if reference.strip():
        r_set, d_set, e_set = set(tokens(reference)), set(d_toks), set(e_toks)
        missing = r_set - d_set
        had = r_set & d_set
        fixed = len(missing & e_set) / len(missing) if missing else 0.0
        broke = len(had - e_set) / len(had) if had else 0.0
        c.ledger = w.fix * fixed - w.brk * broke
        c.detail.update(fixed=fixed, broke=broke)
    if anchors:
        pd, pe_ = _present(anchors, draft), _present(anchors, edit)
        rec = len(pe_ - pd) / len(anchors)
        lost = len(pd - pe_) / len(anchors)
        c.ledger += w.anchor_rec * rec - w.anchor_lost * lost
        c.detail.update(anchors_recovered=rec, anchors_lost=lost)
    inv = invented_numbers(source, draft, edit)
    if inv:
        c.ledger -= w.invented_num * min(inv, 3) / 3
        c.detail["invented_numbers"] = float(inv)

    rep_d, rep_e = guards.repeated_ngram_fraction(draft), guards.repeated_ngram_fraction(edit)
    c.repair = w.repair * max(-1.0, min(1.0, (rep_d - rep_e) / 0.35))

    if dia_d is not None and dia_e is not None:
        c.diacritics = w.diacritics * max(-1.0, min(1.0, dia_d - dia_e))

    ins_words = _inserted_words(draft, edit)
    if ins_words >= w.ungrounded_min_words and eff <= 0.0:       # new words the QE ensemble did not reward (a real repair also inserts new words)
        known = set(d_toks) | {lex.fold(t) for t in re.findall(r"\w+", source)} | set(tokens(reference))
        unk = [t for t in _inserted_tokens(draft, edit) if t not in known and lex.fold(t) not in known and len(t) >= 4]
        c.ungrounded = -w.ungrounded * min(1.0, len(unk) / max(ins_words, 1))
        c.detail["ungrounded_share"] = len(unk) / max(ins_words, 1)

    if c.gain == 0.0 and c.ref == 0.0 and c.refcomet == 0.0:
        c.churn = -w.churn * ops.edit_rate
    c.detail.update(edit_rate=ops.edit_rate, inserted=float(ops.inserted), deleted=float(ops.deleted), substituted=float(ops.substituted), diacritic_only=float(ops.diacritic_only),
                    delta_q=delta, severity=severity)
    c.total = c.gain + c.ref + c.refcomet + c.ledger + c.repair + c.diacritics + c.ungrounded + c.churn
    return c


def _inserted_tokens(draft: str, edit: str) -> List[str]:
    a, b = tokens(draft), tokens(edit)
    out: List[str] = []
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ("insert", "replace"):
            out.extend(b[j1:j2])
    return out


def _inserted_words(draft: str, edit: str) -> int:
    return len(_inserted_tokens(draft, edit))
