# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference-anchored reward (owner 2026-11-03; RewardConfig.reference_weight): r = (1 - w) r_existing + w consensus_value(chrF++(translation, reference)) for samples that passed
every check and sit above the failure penalty; nothing else changes. Loader / schedule support for the `ref` domain."""
import json
import os

import pytest

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import consensus as cons
from gemma_mt.rl.data import PromptSchedule, load_rl_prompts
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward

TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]
STOP = [EOT, SPECIALS["<eos>"]]
SRC = "The market opens at eight o'clock and the farmers bring fresh vegetables."
TEXT = "Oja n si ni agogo mejo, awon agbe si mu ewebe tutu wa."


class ConstScorer:
    def __init__(self, value=0.7):
        self.value = value

    def __call__(self, s, m, r=None):
        return [self.value] * len(s)


def ids(text, end=True):
    out = TOK.encode(text, add_special_tokens=False)
    return out + [EOT] if end else out


def reward(w, **kw):
    return TranslationReward(RewardConfig(reference_weight=w, **kw), ConstScorer(), TOK, STOP)


def sample(text=TEXT, ref="", end=True, iso="yor"):
    return Sample(ids(f"{iso} {text}", end), iso, SRC, reference=ref)


def test_off_by_default_and_without_reference_nothing_changes():
    base = reward(0.0)([sample(ref=TEXT)]).rewards
    assert reward(0.35)([sample(ref="")]).rewards == base                                 # no reference on the prompt
    out = reward(0.0)([sample(ref=TEXT)])
    assert out.rewards == base and out.ref_chrf == []                                    # weight 0: the reference is ignored


def test_combined_reward_is_the_convex_mix_with_the_fixed_chrf_map():
    r0 = reward(0.0)([sample()]).rewards[0]
    w = 0.35
    out = reward(w)([sample(ref=TEXT)])                                                  # identical to the reference: chrF++ = 100 -> the top of the map
    lo, hi = reward(w).shaper.reward_range
    assert out.ref_chrf == [pytest.approx(100.0)]
    assert out.rewards[0] == pytest.approx((1 - w) * r0 + w * hi)
    other = "Ni oja, awon eniyan ra nkan ti won nilo ni ojoojumo lai ronu."
    out2 = reward(w)([sample(ref=other)])
    c = cons.chrf_sentence(TEXT, other)
    assert 0 < c < 100 and out2.ref_chrf == [pytest.approx(c)]
    assert out2.rewards[0] == pytest.approx((1 - w) * r0 + w * cons.consensus_value(c, lo, hi))
    assert out2.rewards[0] < out.rewards[0]                                              # a better match to the human text scores higher
    assert out.stats["ref/n"] == 1.0 and out.stats["ref/chrf_mean"] == pytest.approx(100.0) and out.stats["ref/reward_shift_mean"] == pytest.approx(out.rewards[0] - r0)


def test_hard_failures_keep_their_penalty_and_guards_are_unchanged():
    r = reward(0.35)
    out = r([sample(ref=TEXT, end=False), sample(ref=TEXT), Sample(ids("yor " + SRC), "yor", SRC, reference=TEXT)])      # truncated, fine, copy of the source
    r0 = reward(0.0)([sample(end=False), sample(), Sample(ids("yor " + SRC), "yor", SRC)])
    assert out.status == r0.status and out.status[0] == "no_end_of_turn" and out.status[2] == "copy_source"
    assert out.rewards[0] == r0.rewards[0] == r.class_penalty["no_end_of_turn"] and out.rewards[2] == r0.rewards[2]
    assert out.rewards[1] != r0.rewards[1] and out.ref_chrf[0] is None and out.ref_chrf[2] is None and out.ref_chrf[1] == pytest.approx(100.0)


def test_a_sample_at_the_penalty_is_never_lifted_and_the_range_holds():
    r = reward(0.9)
    lo, hi = r.reward_range
    for text, ref in ((TEXT, TEXT), (TEXT, "zzzz qqqq"), ("x", TEXT), (TEXT, "x")):
        out = r([sample(text=text, ref=ref)])
        assert lo - 1e-9 <= out.rewards[0] <= hi + 1e-9
    worst = r([sample(text=TEXT, ref="zzzz qqqq")]).rewards[0]
    best = r([sample(text=TEXT, ref=TEXT)]).rewards[0]
    assert worst < best


def test_group_mixture_only_samples_with_reference_move():
    r = reward(0.35)
    a, b = reward(0.0)([sample(), sample(text=TEXT + " Ni ojo na.")]).rewards, r([sample(ref=TEXT), sample(text=TEXT + " Ni ojo na.", ref=TEXT)]).rewards
    assert b[0] != a[0] and b[1] != a[1] and (b[0] - a[0]) != (b[1] - a[1])


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_loader_reads_reference_columns_and_the_ref_domain_is_drawn_by_its_share(tmp_path):
    root = str(tmp_path / "pool")
    for iso in ("yor", "hau"):
        _write(f"{root}/{iso}.jsonl", [{"english": f"General {i} {iso}.", "iso": iso, "src_afr_tokens": 12, "length_bin": "b64"} for i in range(100)])
        _write(f"{root}/ref/{iso}.jsonl", [{"english": f"Ref {i} {iso}.", "iso": iso, "src_afr_tokens": 12, "length_bin": "b64", "domain": "ref", "reference": f"{iso} human {i}", "ref_source": "afrisci",
                                            "ref_licence": "apache"} for i in range(100)])
    p = load_rl_prompts(root)
    assert len(p.reference) == len(p) == 400 and p.reference_of(0) == "" and any(p.reference) and all((d == "ref") == bool(rf) for d, rf in zip(p.domain, p.reference))
    assert {p.ref_source[i] for i in range(len(p)) if p.reference[i]} == {"afrisci"} and p.ref_source[0] == ""
    sch = PromptSchedule(p.iso, 0.0, 1, length_bin=p.length_bin, curriculum={"b64": 0}, domain=p.domain, domain_weights={"general": 0.85, "ref": 0.15})
    n_ref = n = 0
    for step in range(100):
        for i in sch.indices(step, 64):
            n += 1
            n_ref += bool(p.reference_of(i))
    assert abs(n_ref / n - 0.15) < 0.02
    plain = str(tmp_path / "plain")
    _write(f"{plain}/yor.jsonl", [{"english": "A.", "iso": "yor", "src_afr_tokens": 12}])
    q = load_rl_prompts(plain)
    assert q.reference == [] and q.reference_of(0) == ""


def test_subset_and_rollout_carry_the_reference(tmp_path):
    pytest.importorskip("torch")
    from gemma_mt.rl.train_rl import _subset

    root = str(tmp_path / "pool")
    _write(f"{root}/ref/yor.jsonl", [{"english": f"Ref {i}.", "iso": "yor", "src_afr_tokens": 12, "length_bin": "b64", "domain": "ref", "reference": f"human {i}", "ref_source": "s"} for i in range(5)])
    p = load_rl_prompts(root)
    sub = _subset(p, [3, 1])
    assert sub.reference == ["human 3", "human 1"] and sub.ref_source == ["s", "s"] and sub.reference_of(1) == "human 1"
