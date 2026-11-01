# SPDX-License-Identifier: AGPL-3.0-or-later
"""Language DRAW WEIGHTS as a run control, and an optional automatic updater (D-65, agent R13, 2026-10-22). Pure python, no torch.

The language of every RL prompt is drawn with probability ~ n_l ** language_alpha x language_weights[l] (PromptSchedule). Until now the weights were a fixed yaml dict. Two layers on top:

(1) RUN CONTROL  <out_dir>/LANG_WEIGHTS.json  (like WINDDOWN / LR_SCALE of rl/run_controls.py): read again at every step (a stat() call; parsed only when the file changed), validated,
    nothing to restart. Format (all keys optional):

        {"weights":     {"yor": 3.0, "zul": 1.0},     # ABSOLUTE draw weights, they override the yaml / the adaptive weight of those languages
         "multipliers": {"hau": 0.8},                  # relative to the weight that would apply without this file (yaml or adaptive)
         "mode":        "merge",                       # merge (default): languages not listed keep their weight | replace: languages not listed get 1.0 (the yaml dict is ignored)
         "adaptive":    "on",                          # on (default) | freeze (the updater keeps its last weights and stops updating) | off (updater ignored, yaml weights)
         "note":        "free text, logged"}

    A file that cannot be parsed or fails validation (unknown language, weight not finite / < 0 / > 20, a pool left without any positive weight) is IGNORED as a whole and the last valid
    weights stay in force (logged once). Deleting the file removes the override. Precedence: yaml `language_weights` -> adaptive weights -> LANG_WEIGHTS.json.

(2) AUTOMATIC UPDATER  (`adaptive_lang_weights: true`, default OFF; `adaptive_lang_weights_dry_run: true` only logs what it would do). Every `every` steps (default 50 = the dev-benchmark cadence)
    it turns the per-language training statistics of the window into a bounded multiplier f_i of the language's base weight:

        f_i = clip( u_i * d_i * m_i , 1/ratio, ratio ) with the terms
          u_i  boost for UNDER-PERFORMANCE: max(need_qe, need_fail) * reliability * learnability, only for languages whose signals are not flagged
                  learnability = min(1, share of the language's groups that carry a gradient / flat_ref): a group whose samples all got the same reward teaches nothing, so drawing more of
                  them is not "headroom"
                  need_qe   = clip((parity - R_i) / (parity - R_floor), 0, 1), R_i = shrunk, bin- and domain-adjusted mean reward of the language; parity = 1.0 = the reward of the MEDIAN
                              HUMAN reference (calibrated t_mid), qe-mode languages only (a blend / consensus reward has no human parity: its level is a sampling-agreement statistic)
                  need_fail = clip((fail_i - max(median fail, fail_ref)) / fail_span, 0, 1), failures = status != ok (wrong language, loops, no end, ...): the one signal every mode shares
          d_i  SATURATION shrink: qe-mode, reward >= sat_reward, no failures, not flagged. Shrinks to down_min_ref only when a human-reference dev trend CONFIRMS the plateau (flat or falling),
                  to the milder down_min_noref when there is no reference (QE alone saturates before chrF++ does: Yoruba's greedy reward was 1.85 of 2.0 while its dev chrF++ was 33), never when
                  the dev trend is still rising
          m_i  MARGINAL-GAIN term from the dev chrF++ trend of the language (general dev 19, weak dev 13, AfriScience dev 6 languages): t >= up_t -> x1.15 (productive), t <= -up_t -> x0.85 AND
                  the language is flagged (reward rising while the reference falls = divergence; no boost until it recovers)
        FLAGS (a flagged language is never boosted: f_i <= 1): QE vs consensus disagree inside groups (blend languages, correlation < flag_rho), constraint-penalty mean above flag_anchor,
                  loops (degenerate / no end / post-eos / dangling) above flag_loop, advantage weight (reliability) below flag_reliability, dev trend falling, mean KL above flag_kl x the median
                  (only when the trainer supplies a per-sample KL: it does not today, see LangWindow.add(kl=...)).
        target_i = clip(base_i * f_i, [min_ratio, max_ratio] * base_i, [w_min, w_max]); new_i = prev_i + eta * (target_i - prev_i) (at most +-max_step of prev_i per update);
        the weights of the treated languages are rescaled so that their SUM stays equal to the sum of their base weights (a bounded total draw budget); the D-59 language floor of the schedule is
        untouched (every language still gets >= floor prompts per block, whatever its weight).
    Languages without a usable signal (fewer than `min_groups` groups in the window, no dev data) keep f = 1: no signal, no move.

State and resume: <out_dir>/LANG_WEIGHTS.state.json (history of the weights in force, the window accumulators and the EMAs at every checkpoint `save_state(step)`); on resume the history is cut to
the resumed step and, when the sidecar was written at exactly that step, the window continues; otherwise the next update is skipped for lack of a complete window. PromptSchedule stays stateless in
(seed, step, weights in force): `weights_at(step)` of the history reproduces any past draw.

Tests: tests/test_lang_weights.py. Offline replay of a metrics.jsonl: scripts/replay_lang_weights.py.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

W_MAX_FILE = 20.0
LOOP_STATUSES = ("degenerate", "no_end_of_turn", "post_eos_rubbish", "dangling_marker")
FILE_NAME = "LANG_WEIGHTS.json"
STATE_NAME = "LANG_WEIGHTS.state.json"
R_FLOOR = -2.0


# ------------------------------------------------------------------------------------------------------------------------------ the control file
@dataclass
class FileSpec:
    weights: Dict[str, float] = field(default_factory=dict)
    multipliers: Dict[str, float] = field(default_factory=dict)
    mode: str = "merge"
    adaptive: str = "on"
    note: str = ""

    def empty(self) -> bool:
        return not self.weights and not self.multipliers and self.mode == "merge" and self.adaptive == "on"


def parse_weights_text(text: str, known: Iterable[str], canon: Callable[[str], Optional[str]] = lambda s: s) -> Tuple[Optional[FileSpec], List[str]]:
    """-> (spec, warnings). spec None = the whole file is invalid (errors in warnings). Unknown language tags are errors (a typo must not silently do nothing)."""
    known = set(known)
    errs: List[str] = []
    try:
        raw = json.loads(text)
    except ValueError as e:
        return None, [f"not valid JSON: {e}"]
    if not isinstance(raw, dict):
        return None, ["top level must be a JSON object"]
    bad_keys = set(raw) - {"weights", "multipliers", "mode", "adaptive", "note"}
    if bad_keys:
        errs.append(f"unknown key(s) {sorted(bad_keys)}")
    spec = FileSpec(note=str(raw.get("note", ""))[:200])
    mode = raw.get("mode", "merge")
    if mode not in ("merge", "replace"):
        errs.append(f"mode must be merge|replace, got {mode!r}")
    spec.mode = mode if mode in ("merge", "replace") else "merge"
    ad = raw.get("adaptive", "on")
    if ad not in ("on", "freeze", "off"):
        errs.append(f"adaptive must be on|freeze|off, got {ad!r}")
    spec.adaptive = ad if ad in ("on", "freeze", "off") else "on"
    for key, lo, hi in (("weights", 0.0, W_MAX_FILE), ("multipliers", 0.0, 10.0)):
        d = raw.get(key, {})
        if not isinstance(d, dict):
            errs.append(f"{key} must be an object {{iso: number}}")
            continue
        out = {}
        for k, v in d.items():
            iso = canon(k)
            if iso is None or iso not in known:
                errs.append(f"{key}: unknown language {k!r}")
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not (lo <= float(v) <= hi):
                errs.append(f"{key}[{k}]: {v!r} is not a number in [{lo:g}, {hi:g}]")
                continue
            out[iso] = float(v)
        setattr(spec, key, out)
    return (None, errs) if errs else (spec, [])


def compose_weights(languages: Sequence[str], base: Dict[str, float], auto: Optional[Dict[str, float]], spec: Optional[FileSpec]) -> Dict[str, float]:
    """The weights in force: yaml base -> adaptive -> file. Every language of `languages` gets an entry (1.0 default)."""
    spec = spec or FileSpec()
    if spec.mode == "replace":                              # the yaml dict and the adaptive weights are both ignored
        w = {l: 1.0 for l in languages}
    else:
        w = {l: float(base.get(l, 1.0)) for l in languages}
        if auto and spec.adaptive != "off":                 # "freeze" keeps the adaptive weights in force, "off" drops them
            for l, v in auto.items():
                if l in w:
                    w[l] = float(v)
    for l, v in spec.weights.items():
        if l in w:
            w[l] = v
    for l, m in spec.multipliers.items():
        if l in w:
            w[l] = w[l] * m
    return w


# ------------------------------------------------------------------------------------------------------------------------------ per-language window statistics
class _Acc:
    __slots__ = ("n", "groups", "sum_r", "sum_gstd", "flat", "fail", "loop", "qn", "qsum", "cn", "csum", "pen", "advw", "sdq", "sdc", "sqq", "scc", "ncorr", "rn", "gn", "kl", "kn")

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, 0.0)


class LangWindow:
    """Accumulates the per-sample outcomes of the steps since the last update."""

    def __init__(self):
        self.acc: Dict[str, _Acc] = defaultdict(_Acc)
        self.keyed: Dict[str, Dict[str, List[float]]] = defaultdict(dict)      # iso -> bin|domain key -> [n, sum reward]
        self.key_tot: Dict[str, List[float]] = {}                              # key -> [n, sum reward] over all languages
        self.steps = 0

    def add(self, isos: Sequence[str], keys: Sequence[str], rewards: Sequence[float], status: Sequence[str], G: int, qe=None, adv_w=None, cons=None, pen=None, kl=None) -> None:
        """One step. isos / keys are per PROMPT (group), the rest per sample (group g = rows [g*G, (g+1)*G)). `kl` (optional, per sample): the sequence's mean KL to the reference; the trainer
        does not compute it per sequence today (it would need a few lines in the update loop), the field is the hook for it."""
        self.steps += 1
        for g, (iso, key) in enumerate(zip(isos, keys)):
            lo, hi = g * G, (g + 1) * G
            r = rewards[lo:hi]
            a = self.acc[iso]
            mean = sum(r) / len(r)
            std = math.sqrt(sum((x - mean) ** 2 for x in r) / len(r))
            a.n += len(r)
            a.rn += len(r)
            a.groups += 1
            a.gn += 1
            a.sum_r += sum(r)
            a.sum_gstd += std
            a.flat += 1.0 if std < 1e-6 else 0.0
            for j in range(lo, hi):
                st = status[j]
                a.fail += 0.0 if st == "ok" else 1.0
                a.loop += 1.0 if st in LOOP_STATUSES else 0.0
                if adv_w is not None:
                    a.advw += float(adv_w[j])
                if pen is not None:
                    a.pen += float(pen[j] or 0.0)
                if kl is not None and kl[j] is not None:
                    a.kl += float(kl[j])
                    a.kn += 1
            if qe is not None:
                for j in range(lo, hi):
                    if qe[j] is not None:
                        a.qn += 1
                        a.qsum += float(qe[j])
            if cons is not None:
                for j in range(lo, hi):
                    if cons[j] is not None:
                        a.cn += 1
                        a.csum += float(cons[j])
            if qe is not None and cons is not None:                                  # within-group agreement of the two content signals (blend languages)
                pq = [(float(qe[j]), float(cons[j])) for j in range(lo, hi) if qe[j] is not None and cons[j] is not None]
                if len(pq) >= 3:
                    mq = sum(x for x, _ in pq) / len(pq)
                    mc = sum(y for _, y in pq) / len(pq)
                    for x, y in pq:
                        a.sdq += (x - mq) * (y - mc)
                        a.sqq += (x - mq) ** 2
                        a.scc += (y - mc) ** 2
                    a.ncorr += len(pq)
            k = self.keyed[iso].setdefault(key, [0.0, 0.0])
            k[0] += len(r)
            k[1] += sum(r)
            t = self.key_tot.setdefault(key, [0.0, 0.0])
            t[0] += len(r)
            t[1] += sum(r)

    def add_aggregate(self, iso: str, key: str, n: float, groups: float, reward: Optional[float] = None, fail: float = 0.0, loop: float = 0.0, pen: float = 0.0, std: Optional[float] = None) -> None:
        """A step's worth of ONE language given as aggregates (offline replay of metrics files that only carry rates; `lang/*` keys). `reward` None = no reward information (the
        language then contributes only failure / loop / penalty rates)."""
        a = self.acc[iso]
        a.n += n
        a.groups += groups
        a.fail += fail * n
        a.loop += loop * n
        a.pen += pen * n
        if reward is not None:
            a.rn += n
            a.sum_r += reward * n
            k = self.keyed[iso].setdefault(key, [0.0, 0.0])
            k[0] += n
            k[1] += reward * n
            t = self.key_tot.setdefault(key, [0.0, 0.0])
            t[0] += n
            t[1] += reward * n
        if std is not None:
            a.gn += groups
            a.sum_gstd += std * groups

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Per language: n, groups, reward (raw), reward_adj (bin/domain-adjusted, re-centred on the window mean), std (mean within-group reward std), flat, fail, loop, qe, cons, pen, advw, qc_corr."""
        tot_n = sum(v[0] for v in self.key_tot.values())
        gmean = sum(v[1] for v in self.key_tot.values()) / tot_n if tot_n else 0.0
        out = {}
        for iso, a in self.acc.items():
            if a.n <= 0:
                continue
            adj = 0.0
            for key, (n, s) in self.keyed[iso].items():
                tn, ts = self.key_tot[key]
                adj += s - n * (ts / tn if tn else 0.0)
            out[iso] = {"n": a.n, "rn": a.rn, "groups": a.groups, "reward": (a.sum_r / a.rn) if a.rn else None, "reward_adj": (gmean + adj / a.rn) if a.rn else None,
                        "std": (a.sum_gstd / a.gn) if a.gn else None, "flat": (a.flat / a.gn) if a.gn else None,
                        "fail": a.fail / a.n, "loop": a.loop / a.n, "qe": (a.qsum / a.qn) if a.qn else None, "cons": (a.csum / a.cn) if a.cn else None,
                        "pen": a.pen / a.n, "advw": a.advw / a.n if a.advw else None, "kl": (a.kl / a.kn) if a.kn else None,
                        "qc_corr": (a.sdq / math.sqrt(a.sqq * a.scc)) if a.ncorr >= 24 and a.sqq > 0 and a.scc > 0 else None}
        return out

    def to_json(self) -> dict:
        return {"steps": self.steps, "acc": {k: [getattr(v, s) for s in _Acc.__slots__] for k, v in self.acc.items()}, "keyed": self.keyed, "key_tot": self.key_tot}

    @classmethod
    def from_json(cls, d: dict) -> "LangWindow":
        w = cls()
        w.steps = int(d.get("steps", 0))
        for k, vals in d.get("acc", {}).items():
            a = w.acc[k]
            for s, v in zip(_Acc.__slots__, vals):
                setattr(a, s, v)
        for k, m in d.get("keyed", {}).items():
            w.keyed[k] = {kk: list(vv) for kk, vv in m.items()}
        w.key_tot = {k: list(v) for k, v in d.get("key_tot", {}).items()}
        return w


# ------------------------------------------------------------------------------------------------------------------------------ dev (human-reference) trend
def dev_trend(points: Sequence[Tuple[int, float, int]], min_evals: int, sd_per_sentence: float = 14.7, last: int = 8) -> Optional[Dict[str, float]]:
    """points = [(step, chrF++, n_sentences)] of ONE dev set of ONE language. Least-squares slope (chrF++ per 100 steps) over the last `last` evaluations and its t statistic. The noise
    of a corpus chrF++ DIFFERENCE on n sentences is ~ sd_per_sentence / sqrt(n) (median over the 136 language x benchmark cells of reports/final_model_per_language_wide.csv)."""
    pts = list(points)[-last:]
    if len(pts) < min_evals:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    n = max(min(p[2] for p in pts), 1)
    se = (sd_per_sentence / math.sqrt(n)) / math.sqrt(sxx)
    return {"slope100": slope * 100.0, "se100": se * 100.0, "t": slope / se if se > 0 else 0.0, "n_evals": len(pts), "last": ys[-1]}


# ------------------------------------------------------------------------------------------------------------------------------ the rule
@dataclass
class AdaptiveParams:
    first_step: int = 100            # no update before this step (two complete windows of data)
    min_groups: int = 6              # a language needs this many groups in the window, else no signal
    ema: float = 0.5                 # weight of the newest window in the EMAs of reward / fail / std
    shrink_k: float = 48.0           # samples of prior (the mean of the language's content-mode group) in the shrunk reward
    parity: float = 1.0              # reward of the median human reference (t_mid)
    sat_reward: float = 1.4          # reward above which a qe-mode language is "QE-saturated"
    fail_ref: float = 0.05           # failure rate that counts as normal
    fail_span: float = 0.30
    up_max: float = 1.5              # largest boost factor u
    down_min_ref: float = 0.6        # saturation shrink when a human-reference dev trend confirms the plateau
    down_min_noref: float = 0.85     # ... without a reference
    dev_min_evals: int = 4
    dev_up_t: float = 1.5            # t of the dev slope for "rising" / "falling"
    dev_flat_se: float = 0.5         # chrF++ per 100 steps: standard error below which "no trend" counts as a confirmed plateau
    dev_flat_slope: float = 0.3      # ... and the slope itself is within +-0.3 per 100 steps
    dev_productive: float = 1.15
    dev_decline: float = 0.85
    min_ratio: float = 0.5           # target within [min_ratio, max_ratio] x base weight
    max_ratio: float = 1.6
    w_min: float = 0.25
    w_max: float = 4.5
    eta: float = 0.5                 # move this share of the way to the target at each update
    max_step: float = 0.25           # ... but at most +-25 % of the previous weight
    flag_rho: float = 0.05           # blend languages: within-group correlation of QE and consensus below this = the two signals disagree
    flag_anchor: float = 0.25        # mean constraint penalty per sample above this
    flag_loop: float = 0.25
    flag_reliability: float = 0.3    # advantage weight (calibrated reliability) below this
    flat_ref: float = 0.5            # share of the language's groups that must carry a gradient (non-flat) for the full boost; fewer = proportionally smaller boost (extra draws teach nothing)
    flag_kl: float = 3.0             # mean KL above this multiple of the median over languages = flagged (only when the trainer supplies per-sample KL)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "AdaptiveParams":
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        bad = set(d) - names
        if bad:
            raise ValueError(f"adaptive_lang_weights_params: unknown key(s) {sorted(bad)}; known: {sorted(names)}")
        return cls(**{k: type(getattr(cls(), k))(v) for k, v in d.items()})


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return 0.0 if n == 0 else (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def factors(sig: Dict[str, Dict[str, Any]], modes: Dict[str, str], dev: Dict[str, Dict[str, float]], p: AdaptiveParams) -> Dict[str, Dict[str, Any]]:
    """sig[iso] = EMA statistics {n, groups, reward_adj, std, flat, fail, loop, pen, advw, qc_corr}; dev[iso] = dev_trend dict (best-informed set). -> per language {f, u, d, m, flags, why}."""
    usable = {l: s for l, s in sig.items() if s.get("groups", 0) >= p.min_groups}
    by_mode: Dict[str, List[float]] = defaultdict(list)
    for l, s in usable.items():
        if s.get("reward_adj") is not None:
            by_mode[modes.get(l, "qe")].append(s["reward_adj"])
    mode_mean = {m: sum(v) / len(v) for m, v in by_mode.items() if v}
    med_fail = _median([s["fail"] for s in usable.values()]) if usable else 0.0
    med_kl = _median([s["kl"] for s in usable.values() if s.get("kl") is not None])
    out: Dict[str, Dict[str, Any]] = {}
    for l, s in sig.items():
        if l not in usable:
            out[l] = {"f": 1.0, "u": 1.0, "d": 1.0, "m": 1.0, "flags": [], "why": "no_signal"}
            continue
        mode = modes.get(l, "qe")
        n = s.get("rn", s["n"])                                                      # samples that carry a reward (a replayed file may have failure rates for more)
        radj = s.get("reward_adj")
        r_shr = None if radj is None else (n * radj + p.shrink_k * mode_mean.get(mode, radj)) / (n + p.shrink_k)       # shrunk toward the mean of the language's content-mode group
        flags: List[str] = []
        if s.get("qc_corr") is not None and mode == "blend" and s["qc_corr"] < p.flag_rho:
            flags.append("qe_consensus_disagree")
        if s.get("pen", 0.0) > p.flag_anchor:
            flags.append("anchor_penalty")
        if s.get("loop", 0.0) > p.flag_loop:
            flags.append("loops")
        if s.get("advw") is not None and s["advw"] < p.flag_reliability:
            flags.append("low_reliability")
        if s.get("kl") is not None and med_kl > 0 and s["kl"] > p.flag_kl * med_kl:
            flags.append("kl_high")
        dv = dev.get(l)
        dstate = "unknown"
        if dv is not None:
            if dv["t"] >= p.dev_up_t:
                dstate = "rising"
            elif dv["t"] <= -p.dev_up_t:
                dstate = "falling"
            elif dv.get("se100", 0.0) <= p.dev_flat_se and abs(dv.get("slope100", 0.0)) <= p.dev_flat_slope:
                dstate = "flat"                                                       # a plateau is CONFIRMED only when the trend is both small and precisely measured
            else:
                dstate = "unknown"                                                    # an imprecise "no trend" says nothing (40-60 sentence cells cannot see +-1 chrF++ per 100 steps)
        if dstate == "falling":
            flags.append("dev_falling")
        rel = 1.0 if s.get("advw") is None else max(0.0, min(1.0, s["advw"]))
        need_qe = max(0.0, min(1.0, (p.parity - r_shr) / (p.parity - R_FLOOR))) if (mode == "qe" and r_shr is not None) else 0.0
        need_fail = max(0.0, min(1.0, (s["fail"] - max(med_fail, p.fail_ref)) / p.fail_span))
        learn = 1.0 if s.get("flat") is None else min(1.0, (1.0 - s["flat"]) / max(p.flat_ref, 1e-6))        # groups without a gradient (all samples equal) teach nothing: no point drawing more of them
        u = 1.0 + (p.up_max - 1.0) * max(need_qe, need_fail) * rel * learn
        d = 1.0
        if mode == "qe" and r_shr is not None and r_shr >= p.sat_reward and s["fail"] <= p.fail_ref and not flags:
            sat = min(1.0, (r_shr - p.sat_reward) / max(2.0 - p.sat_reward, 1e-6))
            floor_ = p.down_min_ref if dstate in ("flat", "falling") else (p.down_min_noref if dstate == "unknown" else 1.0)
            d = 1.0 - (1.0 - floor_) * (0.5 + 0.5 * sat)
        m = p.dev_productive if dstate == "rising" else (p.dev_decline if dstate == "falling" else 1.0)
        f = u * d * m
        if flags:
            f = min(f, 1.0)
        f = min(max(f, p.min_ratio), p.max_ratio)
        out[l] = {"f": f, "u": u, "d": d, "m": m, "flags": flags, "dev": dstate, "R": r_shr, "fail": s["fail"], "need_qe": need_qe, "need_fail": need_fail, "learn": learn, "why": "ok"}
    return out


def targets_from_factors(base: Dict[str, float], prev: Dict[str, float], fac: Dict[str, Dict[str, Any]], treated: Sequence[str], p: AdaptiveParams) -> Dict[str, float]:
    """Bounded, smoothed, budget-preserving new weights of the treated languages."""
    new: Dict[str, float] = {}
    for l in treated:
        b = float(base.get(l, 1.0))
        f = fac.get(l, {}).get("f", 1.0)
        tgt = min(max(b * f, p.min_ratio * b), p.max_ratio * b)
        tgt = min(max(tgt, p.w_min), max(p.w_max, b))
        cur = float(prev.get(l, b))
        nxt = cur + p.eta * (tgt - cur)
        nxt = min(max(nxt, cur * (1 - p.max_step)), cur * (1 + p.max_step))
        new[l] = nxt
    budget = sum(float(base.get(l, 1.0)) for l in treated)
    for _ in range(4):                                      # rescale to the base budget, re-clip, repeat (the clips are the only non-linearity)
        s = sum(new.values())
        if s <= 0:
            break
        k = budget / s
        for l in treated:
            b = float(base.get(l, 1.0))
            new[l] = min(max(new[l] * k, p.min_ratio * b, p.w_min), max(p.w_max, p.max_ratio * b))
    return new


# ------------------------------------------------------------------------------------------------------------------------------ the controller
def _guarded(default):
    """A bug or a bad value inside the controller must never stop a multi-day run: the exception is logged (with its traceback, once per kind) and `default` is returned; after 5 errors
    the adaptive updater switches itself off (the weights in force stay, the file control keeps working)."""
    def deco(fn):
        def wrapper(self, *a, **kw):
            try:
                return fn(self, *a, **kw)
            except Exception as e:                                           # noqa: BLE001
                import traceback

                self.errors += 1
                key = f"{fn.__name__}:{type(e).__name__}"
                if key not in self._err_seen:
                    self._err_seen.add(key)
                    self.log(f"[gemma_mt] lang weights: ERROR in {fn.__name__} ({e!r}); the run goes on with the weights in force\n" + traceback.format_exc())
                if self.errors >= 5 and self.adaptive:
                    self.adaptive = False
                    self.log("[gemma_mt] lang weights: 5 errors, the adaptive updater is switched OFF (file control stays)")
                return default() if callable(default) else default
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return deco


class LangWeightController:
    """Owns the weights in force. Trainer hooks: update(step) at the top of every step, observe(...) after the rewards, observe_eval(step, ev), save_state(step), summary()."""

    def __init__(self, out_dir: str, schedule, base: Dict[str, float], languages: Sequence[str], modes: Dict[str, str], *, file_control: bool = True, adaptive: bool = False,
                 dry_run: bool = False, every: int = 50, params: Optional[AdaptiveParams] = None, treated: Optional[Sequence[str]] = None, canon: Callable[[str], Optional[str]] = lambda s: s,
                 start_step: int = 0, log: Callable[[str], None] = print, log_stats: bool = False, log_weights_every: int = 0):
        self.dir, self.schedule, self.log = out_dir, schedule, log
        self.base = {canon(k): float(v) for k, v in base.items() if canon(k)}
        self.languages = list(languages)
        self.modes = dict(modes)
        self.file_control, self.adaptive, self.dry_run = bool(file_control), bool(adaptive), bool(dry_run)
        self.every = int(every)
        self.p = params or AdaptiveParams()
        self.treated = [l for l in (treated or self.languages) if l in set(self.languages)]
        self.canon = canon
        self.log_stats, self.log_weights_every = bool(log_stats), int(log_weights_every)
        self.p_file, self.p_state = os.path.join(out_dir, FILE_NAME), os.path.join(out_dir, STATE_NAME)
        self.spec: Optional[FileSpec] = None
        self._file_sig: Optional[Tuple[int, int]] = None
        self._bad_sig: Optional[Tuple[int, int]] = None
        self.window = LangWindow()
        self.ema: Dict[str, Dict[str, float]] = {}
        self.auto: Dict[str, float] = {}
        self.dry: Dict[str, float] = {}
        self.dev: Dict[str, Dict[str, List[Tuple[int, float, int]]]] = defaultdict(lambda: defaultdict(list))     # iso -> dev set -> [(step, chrF, n)]
        self.history: List[Dict[str, Any]] = []
        self.in_force: Dict[str, float] = {}
        self.last_update = -1
        self.partial = False
        self.n_updates = 0
        self.dev_n: Dict[Tuple[str, Optional[str]], int] = {}
        self.errors = 0
        self._err_seen: set = set()
        self._last_diag: Dict[str, Dict[str, Any]] = {}
        self._resume(start_step)

    # -------------------------------------------------------------------------- persistence
    def _resume(self, start_step: int) -> None:
        try:
            st = json.load(open(self.p_state)) if os.path.exists(self.p_state) else None
        except (OSError, ValueError):
            st = None
        if not st or not start_step:
            return
        hist = [h for h in st.get("history", []) if h.get("step", 0) <= start_step]
        self.history = hist
        upd = [h for h in hist if h.get("kind") == "update"]
        if upd:
            last = upd[-1]
            self.auto = {k: float(v) for k, v in last.get("auto", {}).items()}
            self.ema = {k: dict(v) for k, v in last.get("ema", {}).items()}
            self.last_update = int(last["step"])
            self.n_updates = len(upd)
        if int(st.get("saved_step", -1)) == start_step:
            self.window = LangWindow.from_json(st.get("window", {}))
            self.dev = defaultdict(lambda: defaultdict(list), {k: defaultdict(list, {kk: [tuple(x) for x in vv] for kk, vv in v.items()}) for k, v in st.get("dev", {}).items()})
            self.last_update = max(self.last_update, int(st.get("last_update", -1)))
            if st.get("dry"):
                self.dry = {k: float(v) for k, v in st["dry"].items()}
            self.log(f"[gemma_mt] lang weights: resumed at step {start_step} with the sidecar written at that step ({len(upd)} adaptive updates so far)")
        else:
            self.partial = True
            self.log(f"[gemma_mt] lang weights: resumed at step {start_step}, sidecar step {st.get('saved_step')}: the statistics window is incomplete, the next adaptive update is skipped (weights restored from the history)")

    @_guarded(None)
    def save_state(self, step: int) -> None:
        d = {"version": 1, "saved_step": int(step), "last_update": self.last_update, "history": self.history, "window": self.window.to_json(),
             "dev": {k: {kk: [list(x) for x in vv] for kk, vv in v.items()} for k, v in self.dev.items()}, "dry": self.dry}
        tmp = self.p_state + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(d, f)
            os.replace(tmp, self.p_state)
        except OSError as e:
            self.log(f"[gemma_mt] lang weights: could not write {self.p_state}: {e}")

    def weights_at(self, step: int) -> Dict[str, float]:
        """The weights that were in force when schedule step `step` was drawn (history; base before the first entry)."""
        w = {l: float(self.base.get(l, 1.0)) for l in self.languages}
        for h in self.history:
            if h["step"] <= step and "weights" in h:
                w = dict(h["weights"])
        return w

    # -------------------------------------------------------------------------- the file
    def _read_file(self, step: int) -> None:
        if not self.file_control:
            return
        try:
            stt = os.stat(self.p_file)
            sig = (int(stt.st_mtime_ns), int(stt.st_size))
        except OSError:
            if self.spec is not None:
                self.log(f"[gemma_mt] LANG_WEIGHTS.json removed: override cleared (step {step})")
            self.spec, self._file_sig, self._bad_sig = None, None, None
            return
        if sig == self._file_sig or sig == self._bad_sig:
            return
        try:
            text = open(self.p_file).read()
        except OSError:
            return
        spec, errs = parse_weights_text(text, self.languages, self.canon)
        if spec is None:
            self._bad_sig = sig
            self.log(f"[gemma_mt] LANG_WEIGHTS.json ignored (step {step}): " + "; ".join(errs[:4]) + (" ..." if len(errs) > 4 else ""))
            return
        self._file_sig, self._bad_sig = sig, None
        self.spec = spec
        self.log(f"[gemma_mt] LANG_WEIGHTS.json read (step {step}): mode={spec.mode} adaptive={spec.adaptive} {len(spec.weights)} absolute, {len(spec.multipliers)} multiplier(s)"
                 + (f" note={spec.note!r}" if spec.note else ""))

    # -------------------------------------------------------------------------- observations
    @_guarded(None)
    def observe(self, step: int, isos: Sequence[str], keys: Sequence[str], rewards, status, G: int, qe=None, adv_w=None, cons=None, pen=None, kl=None) -> None:
        self.window.add(isos, keys, rewards, status, G, qe, adv_w, cons, pen, kl)

    @_guarded(dict)
    def step_record(self, isos, keys, rewards, status, G: int) -> Dict[str, float]:
        """Per-language statistics of THIS step for the metrics file (only the languages drawn): lang/n (samples), lang/reward (mean), lang/radj (mean minus the step mean of the same
        length-bin x domain key: the bin-adjusted deviation the updater works with), lang/fail (share of samples whose status is not ok)."""
        rec: Dict[str, float] = {}
        kt: Dict[str, List[float]] = {}
        per: Dict[str, List[int]] = defaultdict(list)
        for g, (iso, key) in enumerate(zip(isos, keys)):
            per[iso].append(g)
            t = kt.setdefault(key, [0.0, 0.0])
            for j in range(g * G, (g + 1) * G):
                t[0] += 1
                t[1] += rewards[j]
        for iso, gs in per.items():
            rr = [rewards[j] for g in gs for j in range(g * G, (g + 1) * G)]
            ss = [status[j] for g in gs for j in range(g * G, (g + 1) * G)]
            dev = [rewards[j] - kt[keys[g]][1] / kt[keys[g]][0] for g in gs for j in range(g * G, (g + 1) * G)]
            rec[f"lang/n/{iso}"] = float(len(rr))
            rec[f"lang/reward/{iso}"] = sum(rr) / len(rr)
            rec[f"lang/radj/{iso}"] = sum(dev) / len(dev)
            rec[f"lang/fail/{iso}"] = sum(1.0 for s in ss if s != "ok") / len(ss)
        return rec

    @_guarded(None)
    def observe_eval(self, step: int, ev: Dict[str, float]) -> None:
        """Dev chrF++ per language from an evaluation record (`bench/chrf/<L>`, `bench_aux/<name>/chrf/<L>`); n_sentences unknown here -> `dev_n` per set via set_dev_sizes."""
        for k, v in ev.items():
            parts = k.split("/")
            if parts[0] == "bench" and len(parts) == 3 and parts[1] == "chrf":
                name, iso = "bench", parts[2]
            elif parts[0] == "bench_aux" and len(parts) == 4 and parts[2] == "chrf":
                name, iso = parts[1], parts[3]
            else:
                continue
            if iso in self.modes or iso in self.languages:
                n = self.dev_n.get((name, iso), self.dev_n.get((name, None), 40))
                self.dev[iso][name].append((int(step), float(v), int(n)))

    def set_dev_sizes(self, sizes: Dict[Tuple[str, Optional[str]], int]) -> None:
        self.dev_n = dict(sizes)

    # -------------------------------------------------------------------------- the update
    def _best_dev(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for iso, sets in self.dev.items():
            best = None
            for name, pts in sets.items():
                t = dev_trend(pts, self.p.dev_min_evals)
                if t is not None and (best is None or abs(t["t"]) > abs(best["t"])):
                    best = dict(t, set=name)
            if best is not None:
                out[iso] = best
        return out

    def _adaptive_update(self, step: int) -> Optional[str]:
        if self.partial:
            self.partial = False
            self.window = LangWindow()
            return "window incomplete after a resume: update skipped"
        summ = self.window.summary()
        self.window = LangWindow()
        for l, s in summ.items():
            e = self.ema.setdefault(l, {})
            for k in ("reward_adj", "std", "flat", "fail", "loop", "pen", "kl"):
                new, old = s[k], e.get(k)
                if new is None:
                    continue                                                        # no information in this window: the EMA keeps its value
                e[k] = new if old is None else self.p.ema * new + (1 - self.p.ema) * old
            e.update({"n": s["n"], "rn": s["rn"], "groups": s["groups"], "advw": s["advw"], "qc_corr": s["qc_corr"]})
        sig = {l: dict(v) for l, v in self.ema.items() if l in summ}
        fac = factors(sig, self.modes, self._best_dev(), self.p)
        cur_map = self.dry if self.dry_run else self.auto
        prev = {l: cur_map.get(l, float(self.base.get(l, 1.0))) for l in self.treated}
        new = targets_from_factors(self.base, prev, fac, self.treated, self.p)
        self._last_diag = fac
        if self.dry_run:
            self.dry = new
            self.history.append({"step": int(step), "kind": "dry", "auto": new, "factors": {l: round(v["f"], 4) for l, v in fac.items()}})
            self.log(f"[gemma_mt] lang weights (dry run) step {step}: would move " + self._moves(prev, new))
        else:
            self.auto = new
            self.n_updates += 1
            self.history.append({"step": int(step), "kind": "update", "auto": {l: round(v, 6) for l, v in new.items()}, "ema": {l: {k: v for k, v in e.items() if v is not None} for l, e in self.ema.items()},
                                 "factors": {l: round(v["f"], 4) for l, v in fac.items()}, "flags": {l: v["flags"] for l, v in fac.items() if v["flags"]}})
            self.log(f"[gemma_mt] lang weights adaptive update @ {step}: " + self._moves(prev, new))
        self.last_update = step
        return None

    @staticmethod
    def _moves(prev: Dict[str, float], new: Dict[str, float]) -> str:
        mv = sorted(((new[l] / prev[l] - 1.0, l) for l in new if prev.get(l)), key=lambda t: -abs(t[0]))[:8]
        return ", ".join(f"{l} {prev[l]:.2f}->{new[l]:.2f}" for _, l in mv) or "nothing"

    @_guarded(dict)
    def update(self, step: int) -> Dict[str, float]:
        """Top of the step `step` (= completed steps). Applies the weights in force to the schedule; returns the metrics to log ({} when nothing changed)."""
        rec: Dict[str, float] = {}
        self._read_file(step)
        if self.adaptive and step >= self.p.first_step and step % self.every == 0 and step != self.last_update and step > 0:
            frozen = self.spec is not None and self.spec.adaptive in ("freeze", "off")
            if not frozen:
                why = self._adaptive_update(step)
                if why:
                    self.log(f"[gemma_mt] lang weights @ {step}: {why}")
                else:
                    self.save_state(step)                  # the sidecar then holds the update AND the emptied window: a resume from this step's checkpoint does not repeat it
            else:
                self.window = LangWindow()
        use_auto = self.adaptive and not self.dry_run and not (self.spec is not None and self.spec.adaptive == "off")
        w = compose_weights(self.languages, self.base, self.auto if use_auto else None, self.spec)
        changed = w != self.in_force
        if changed:
            try:
                self.schedule.set_language_weights(w)
            except ValueError as e:
                self.log(f"[gemma_mt] lang weights: the weights in force were REJECTED by the schedule ({e}); keeping the previous ones")
                w, changed = self.in_force or w, False
                if self.spec is not None:
                    self.spec, self._file_sig = None, None
            else:
                self.in_force = dict(w)
                if not self.history or self.history[-1].get("weights") != w:
                    self.history.append({"step": int(step), "kind": "weights", "weights": {l: round(v, 6) for l, v in w.items()}})
        if changed or (self.log_weights_every and step % self.log_weights_every == 0) or step == 0:
            for l in self.languages:
                rec[f"reward/lang_weights/{l}"] = float(self.in_force.get(l, 1.0))
            rec["reward/lang_weights/adaptive_updates"] = float(self.n_updates)
            rec["reward/lang_weights/file_active"] = 1.0 if self.spec is not None else 0.0
        if self.dry_run and self.dry and (changed or step == self.last_update):
            for l, v in self.dry.items():
                rec[f"reward/lang_weights_dry/{l}"] = float(v)
        return rec

    def summary(self) -> Dict[str, Any]:
        return {"adaptive": self.adaptive, "dry_run": self.dry_run, "updates": self.n_updates, "last_update": self.last_update, "weights_in_force": {l: round(v, 4) for l, v in self.in_force.items()},
                "file_active": self.spec is not None}


def build_controller(cfg, schedule, prompts_languages: Sequence[str], mode_of: Dict[str, str], *, start_step: int, canon, log=print) -> Optional[LangWeightController]:
    """Called by train_rl after the schedule exists. None when nothing is switched on."""
    file_control = bool(getattr(cfg, "lang_weights_file", False)) and bool(getattr(cfg, "run_controls", True))
    adaptive = bool(getattr(cfg, "adaptive_lang_weights", False))
    stats = bool(getattr(cfg, "lang_stats_log", False))
    if not (file_control or adaptive or stats):
        return None
    params = AdaptiveParams.from_dict(getattr(cfg, "adaptive_lang_weights_params", None))
    params.first_step = int(getattr(cfg, "adaptive_lang_weights_first_step", params.first_step))
    treated = [canon(x) for x in (getattr(cfg, "adaptive_lang_weights_languages", None) or [])]
    ctl = LangWeightController(cfg.out_dir, schedule, dict(cfg.language_weights), prompts_languages, mode_of, file_control=file_control, adaptive=adaptive,
                               dry_run=bool(getattr(cfg, "adaptive_lang_weights_dry_run", False)), every=int(getattr(cfg, "adaptive_lang_weights_every", 50)), params=params,
                               treated=[t for t in treated if t] or None, canon=canon, start_step=start_step, log=log, log_stats=stats or adaptive,
                               log_weights_every=int(getattr(cfg, "lang_weights_log_every", 50)))
    se_ = int(getattr(cfg, "save_every", 0) or 0)
    if adaptive and se_ and ctl.every % se_ != 0:
        log(f"[gemma_mt] lang weights: WARNING adaptive_lang_weights_every={ctl.every} is not a multiple of save_every={se_}: a resume from a checkpoint will find an incomplete statistics window and skip one update")
    log(f"[gemma_mt] lang weights: file control {'on' if file_control else 'off'}, adaptive {'DRY RUN' if (adaptive and ctl.dry_run) else ('on' if adaptive else 'off')}"
        f"{' (every %d steps from step %d, %d treated languages)' % (ctl.every, params.first_step, len(ctl.treated)) if adaptive else ''}, per-language stats {'logged' if ctl.log_stats else 'not logged'}")
    return ctl
