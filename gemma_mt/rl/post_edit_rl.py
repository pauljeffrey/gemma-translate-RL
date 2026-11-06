# SPDX-License-Identifier: AGPL-3.0-or-later
"""Editor RL (2026-11-03): glue between `train_rl` and `post_edit_reward`. Everything the editor mode adds to the RL loop lives here; `train_rl.rollout_and_reward` calls it when `cfg.editor_mode`.

One prompt = (English source, draft); the policy is the post-editor E1 (prompt `gemma_mt.post_edit.build_pe_prompt`, answer `{tag} {text}<end_of_turn>`, i.e. a translation-shaped completion).
Group of 16 = 13 sampled edits + the UNCHANGED DRAFT (identity member) + the beam-4 edit (guide) + the greedy edit.
Reward of an edit = `post_edit_reward.score(...)`:
    q_d, q_e   the translation reward (`TranslationReward`: shaped AfriCOMET-QE (+ SSA-COMET) with hard failures at the floor) of the draft (scored once per prompt, cached) and of the edit; the edit's failures
               (format, wrong language, loop, unfinished: status != ok) keep the failure value
    prompts WITH a human reference: reference chrF++ gain (weight `ref_weights.ref`, default 1.0), reference-AfriCOMET gain with a dead zone for the languages that model covers (`refcomet`), the word ledger
               against the reference, and the QE gain at the reduced weight (`ref_weights.gain`, default 0.35)
    prompts WITHOUT a reference: QE gain (languages whose content mode is QE), the rules (anchors, repair, copy / churn, ungrounded insertions)
The group consensus signal of the translation reward is OFF in this mode (it rewards agreement between members, i.e. copying the draft); rewards are clipped to the reward range.
Not in the reward and therefore usable as success metrics: reference chrF++, spBLEU, SSA-COMET of the benchmark outputs (the QE ensemble of the reward may contain SSA-COMET: see the run report).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gemma_mt import format as fmt
from gemma_mt import post_edit as pe
from gemma_mt.languages import QE_VALIDATED, canonical_iso
from gemma_mt.rl import consensus as cons
from gemma_mt.rl import post_edit_reward as per


@dataclass
class EditorOptions:
    max_draft_tokens: int = pe.MAX_DRAFT_TOKENS
    identity_member: bool = True
    weights: Dict[str, float] = field(default_factory=dict)             # overrides of post_edit_reward.Weights for prompts without a reference
    ref_weights: Dict[str, float] = field(default_factory=lambda: {"gain": 0.35, "ref": 1.0})   # overrides on prompts that carry a human reference
    ref_weights_refcomet_langs: Dict[str, float] = field(default_factory=lambda: {"ref": 0.6})  # languages covered by the reference-COMET model: chrF++ weight lower, the COMET gain is added
    off_terms: List[str] = field(default_factory=list)                  # terms switched off by the pre-launch correlation check (names of Components fields, e.g. "churn", "diacritics")
    qe_gain_languages: str = "qe_mode"                                  # "qe_mode": the QE gain only for languages whose content mode is QE (others: consensus-only, no reliable QE); "all"
    ref_comet: Dict[str, Any] = field(default_factory=dict)             # {"python": ..., "pythonpath": ..., "batch_size": 32, "fake": False}; empty = no reference-COMET term
    ref_comet_languages: List[str] = field(default_factory=lambda: list(QE_VALIDATED))
    weight_floor_unscored: float = 0.5                                  # advantage weight of a group of a language the QE shaper cannot score (weight 0) when the prompt has a reference or the rules apply


def editor_options(d: Dict[str, Any]) -> EditorOptions:
    known = {f.name for f in dataclasses.fields(EditorOptions)}
    bad = set(d) - known
    if bad:
        raise ValueError(f"unknown editor options {sorted(bad)}; known: {sorted(known)}")
    return EditorOptions(**d)


def editor_prompt_ids(tokenizer, prompts, indices: Sequence[int], max_draft_tokens: int = pe.MAX_DRAFT_TOKENS) -> List[List[int]]:
    return [list(pe.build_pe_prompt(prompts.english[i], prompts.draft_of(i), prompts.iso[i], tokenizer, max_draft_tokens)) for i in indices]


def identity_completion(tokenizer, english: str, draft: str, iso: str, max_draft_tokens: int = pe.MAX_DRAFT_TOKENS) -> List[int]:
    """Token ids of the completion `{tag} {draft}<end_of_turn>` (the unchanged draft as an editor output), built by the same code as the SFT target."""
    d = pe.truncate_draft(draft, tokenizer, max_draft_tokens)
    ex = pe.build_pe_example(english, d, iso, d, tokenizer, max_draft_tokens=max_draft_tokens)
    return [t for t, l in zip(ex.input_ids, ex.labels) if l != fmt.IGNORE_INDEX]


def insert_identity(roll, G: int, ids_per_prompt: Sequence[List[int]]) -> List[int]:
    """Replace the row `g * G + G - 3` of every group (a sampled member, just before the beam-guide row and the greedy row) by the identity completion; returns the rows. In place."""
    rows = []
    for g, ids in enumerate(ids_per_prompt):
        row = g * G + G - 3
        roll.completions[row] = list(ids)
        roll.finished[row] = True
        if roll.post_stop_tokens:
            roll.post_stop_tokens[row] = 0
        if roll.sampler_logp is not None:
            roll.sampler_logp[row] = [0.0] * len(ids)
        rows.append(row)
    return rows


class EditorReward:
    """Turns the translation reward's output of a batch of edits into the editor reward."""

    def __init__(self, reward, opts: EditorOptions, ref_scorer: Any = None):
        self.reward, self.o, self.ref_scorer = reward, opts, ref_scorer
        self.lo, self.hi = reward.reward_range
        self._draft_cache: Dict[Tuple[str, str, str], Tuple[float, str, str, bool]] = {}
        self._rc_cache: Dict[Tuple[str, str, str, str], float] = {}
        self.covered = set(opts.ref_comet_languages) if ref_scorer is not None else set()
        self.stats_hist: List[Dict[str, float]] = []

    # ------------------------------------------------------------------------------------------------
    def weights_for(self, iso: str, has_ref: bool) -> per.Weights:
        w = dataclasses.replace(per.Weights(), q_lo=self.lo, q_hi=self.hi, **self.o.weights)
        if has_ref:
            w = dataclasses.replace(w, **self.o.ref_weights)
            if iso in self.covered:
                w = dataclasses.replace(w, **self.o.ref_weights_refcomet_langs)
        for t in self.o.off_terms:
            if t in ("gain", "ref", "refcomet", "repair", "churn", "copy", "ungrounded", "diacritics"):
                w = dataclasses.replace(w, **{t: 0.0})
        return w

    def draft_scores(self, sources: Sequence[str], drafts: Sequence[str], isos: Sequence[str], draft_ids: Sequence[List[int]]) -> List[Tuple[float, str, str, bool]]:
        """(q_d, draft text as the scorer saw it, status, ok) per prompt; scored once, cached."""
        from gemma_mt.rl.reward import Sample

        miss = [i for i in range(len(sources)) if (isos[i], sources[i], drafts[i]) not in self._draft_cache]
        if miss:
            smp = [Sample(list(draft_ids[i]), isos[i], sources[i]) for i in miss]
            out = self.reward(smp, group_size=1)
            for j, i in enumerate(miss):
                self._draft_cache[(isos[i], sources[i], drafts[i])] = (float(out.rewards[j]), out.texts[j] or drafts[i], out.status[j], out.status[j] == "ok")
        return [self._draft_cache[(isos[i], sources[i], drafts[i])] for i in range(len(sources))]

    def ref_comet(self, items: Sequence[Tuple[str, str, str, str]]) -> List[Optional[float]]:
        """items = (iso, source, mt, reference); reference-based AfriCOMET for the covered languages (cached by text); None where not applicable."""
        out: List[Optional[float]] = [None] * len(items)
        todo, pos = [], []
        for k, (iso, src, mt, ref) in enumerate(items):
            if self.ref_scorer is None or iso not in self.covered or not ref or not mt.strip():
                continue
            key = (iso, src, mt, ref)
            if key in self._rc_cache:
                out[k] = self._rc_cache[key]
            elif key not in todo:
                todo.append(key)
        if todo:
            sc = self.ref_scorer([t[1] for t in todo], [t[2] for t in todo], [t[3] for t in todo])
            for t, v in zip(todo, sc):
                self._rc_cache[t] = float(v)
            for k, (iso, src, mt, ref) in enumerate(items):
                if out[k] is None and (iso, src, mt, ref) in self._rc_cache:
                    out[k] = self._rc_cache[(iso, src, mt, ref)]
        return out

    # ------------------------------------------------------------------------------------------------
    def __call__(self, samples, out, prompts_info: Sequence[Dict[str, str]], G: int, draft_ids: Sequence[List[int]], identity_rows: Sequence[int] = ()) -> Tuple[List[float], Dict[str, float]]:
        """`out` = the TranslationReward output of the edits (consensus off). `prompts_info[g]` = {english, draft, iso, reference}. Returns (rewards, metrics) and replaces nothing in `out`."""
        n_g = len(prompts_info)
        srcs = [p["english"] for p in prompts_info]; drafts = [p["draft"] for p in prompts_info]; isos = [p["iso"] for p in prompts_info]; refs = [p.get("reference", "") for p in prompts_info]
        dsc = self.draft_scores(srcs, drafts, isos, draft_ids)
        N = len(samples)
        edit_text = [out.texts[i] if out.status[i] == "ok" else "" for i in range(N)]
        rc_items, rc_pos = [], []
        for g in range(n_g):
            if refs[g] and isos[g] in self.covered:
                rc_items.append((isos[g], srcs[g], dsc[g][1], refs[g])); rc_pos.append(("d", g))
        for i in range(N):
            g = i // G
            if refs[g] and isos[g] in self.covered and edit_text[i].strip() and edit_text[i] != dsc[g][1]:
                rc_items.append((isos[g], srcs[g], edit_text[i], refs[g])); rc_pos.append(("e", i))
        rc_vals = self.ref_comet(rc_items)
        rc_d = {k: v for (t, k), v in zip(rc_pos, rc_vals) if t == "d"}
        rc_e = {k: v for (t, k), v in zip(rc_pos, rc_vals) if t == "e"}
        rewards: List[float] = []
        self.last_components = []                      # Components of every sample of the last call (read by the pre-launch correlation check)
        comps = {k: [] for k in ("gain", "ref", "refcomet", "ledger", "repair", "ungrounded", "churn", "copy")}
        n_ident = n_changed = 0
        for i in range(N):
            g = i // G
            q_d, text_d, st_d, ok_d = dsc[g]
            has_ref = bool(refs[g])
            w = self.weights_for(isos[g], has_ref)
            if self.o.qe_gain_languages == "qe_mode" and self.reward.content_spec(canonical_iso(isos[g]) or isos[g])[0] != "qe":
                w = dataclasses.replace(w, gain=0.0)
            if out.status[i] != "ok":
                c = per.score(srcs[g], text_d, "", q_d, self.reward.penalty, failed=True, failure_value=self.reward.penalty)
            else:
                text_e = edit_text[i]
                chrf_d = cons.chrf_sentence(text_d, refs[g]) if has_ref else None
                chrf_e = cons.chrf_sentence(text_e, refs[g]) if has_ref else None
                sev = per.defect_severity(srcs[g], text_d, q_d, w, finished=ok_d, wrong_language=(st_d == "wrong_language"))
                c = per.score(srcs[g], text_d, text_e, q_d, float(out.rewards[i]), reference=refs[g], chrf_d=chrf_d, chrf_e=chrf_e, severity=sev, w=w, rc_d=rc_d.get(g), rc_e=rc_e.get(i, rc_d.get(g) if text_e == text_d else None))
                n_ident += int(c.detail.get("identical", 0.0) == 1.0)
                n_changed += int(c.detail.get("identical", 0.0) != 1.0)
            rewards.append(float(min(self.hi, max(self.lo, c.total))))
            self.last_components.append(c)
            for k in comps:
                comps[k].append(getattr(c, k))
        m = {f"pe/{k}": float(np.mean(v)) for k, v in comps.items()}
        m["pe/identical_share"] = n_ident / max(n_ident + n_changed, 1)
        m["pe/draft_q_mean"] = float(np.mean([d[0] for d in dsc]))
        m["pe/draft_failed_share"] = float(np.mean([not d[3] for d in dsc]))
        m["pe/edit_failed_share"] = float(np.mean([s != "ok" for s in out.status]))
        if identity_rows:
            m["pe/identity_reward"] = float(np.mean([rewards[r] for r in identity_rows]))
        m["pe/reward_mean"] = float(np.mean(rewards))
        m["pe/refcomet_coverage"] = float(len(rc_e) / max(N, 1))
        try:                                            # give the allocator's cached blocks back before the update (the QE / reference-COMET workers share the card)
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return rewards, m
