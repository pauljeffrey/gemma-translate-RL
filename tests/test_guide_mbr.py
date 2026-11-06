# SPDX-License-Identifier: AGPL-3.0-or-later
"""MBR guide for weak languages (owner 2026-11-03; rl/guide.py): the vectorised chrF++ matrix equals the reference implementation, the pick, the winner-source cases, the budget, the cost, and (with torch) the loop."""
import json
import random
import time

import numpy as np
import pytest

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import consensus as cons
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.guide import GuideSpec, OverheadTracker, apply_guides, chrf_matrix, mbr_pick
from gemma_mt.rl.reward import RewardConfig, TranslationReward
from gemma_mt.rl.sampling import Rollout

TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]
STOP = [EOT, SPECIALS["<eos>"]]
SRC = "The market opens at eight o'clock and the farmers bring fresh vegetables."
WORDS = "oja n si ni agogo mejo awon agbe si mu ewebe tutu wa lati ibi ti won ti ra nkan ati bi won ti ta".split()


def rand_text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n)) + rng.choice([".", "", "!", ","])


def test_chrf_matrix_equals_the_reference_implementation_pair_by_pair():
    rng = random.Random(3)
    texts = [rand_text(rng, rng.randint(1, 18)) for _ in range(14)] + ["a", "ɔ̀bɛ́ ɛ̀ɛ̀ ŋ̀gú tété", "Pidgin go dey fine fine.", "x y z"]
    F = chrf_matrix(texts)
    for i in range(len(texts)):
        for j in range(len(texts)):
            assert F[i, j] == pytest.approx(cons.chrf_sentence(texts[i], texts[j]), abs=1e-9), (i, j)


def test_mbr_picks_the_central_candidate():
    base = "awon agbe si mu ewebe tutu wa lati ibi ti won ti ra nkan"
    near = [base, base + " ati", base.replace("tutu", "titun"), base.replace("agbe", "agbe"), base + "."]
    odd = ["ni oja agogo", "zzzz qqqq xxxx"]
    w, util = mbr_pick(odd + near)
    assert w >= 2 and util[0] < util[w] and util[1] < util[w]


def test_cost_of_33_candidates_for_40_prompts_per_step():
    rng = random.Random(5)
    t0 = time.time()
    for _ in range(40):
        texts = [rand_text(rng, rng.randint(8, 16)) for _ in range(33)]
        mbr_pick(texts)
    dt = time.time() - t0
    print(f"MBR over 33 candidates x 40 prompts: {dt:.2f} s")
    assert dt < 6.0, f"too slow: {dt:.2f} s"


def ids(text):
    return TOK.encode(text, add_special_tokens=False) + [EOT]


def make_reward():
    return TranslationReward(RewardConfig(), FakeScorer(), TOK, STOP)


class Fake:
    """Scripted beam / extra generators and a group of G members for each prompt."""

    def __init__(self, beam_texts, extra_texts, G, member_texts, calls):
        self.beam_texts, self.extra_texts, self.G, self.member_texts, self.calls = beam_texts, extra_texts, G, member_texts, calls

    def roll(self, n_groups):
        comps = [ids(t) for g in range(n_groups) for t in self.member_texts[g]]
        return Rollout([[1]] * len(comps), comps, [True] * len(comps), [[0.0] * len(c) for c in comps], [0] * len(comps))

    def beam(self, pids):
        self.calls.append("beam")
        c = [ids(self.beam_texts[i]) for i in range(len(pids))]
        return Rollout([list(p) for p in pids], c, [True] * len(c), None, [0] * len(c))

    def extra(self, pids, n):
        self.calls.append(("extra", n))
        c = [ids(self.extra_texts[i][e]) for i in range(len(pids)) for e in range(n)]
        return Rollout([list(p) for p in pids for _ in range(n)], c, [True] * len(c), None, [0] * len(c))


GOOD = "yor awon agbe si mu ewebe tutu wa lati ibi ti won ti ra nkan"
ODD = "yor ni oja agogo kekere ti o wa nibe lana"


def run_case(modes, members, beam_text, extra_text, extra_n=4, skip=False, fraction=1.0, n_groups=1, G=4):
    calls = []
    fk = Fake([beam_text] * n_groups, [[extra_text] * extra_n] * n_groups, G, [members] * n_groups, calls)
    roll = fk.roll(n_groups)
    spec = GuideSpec(fraction, 4, modes, extra_n, 0.7, 0.95, skip_extra=skip)
    info = apply_guides(spec, roll=roll, G=G, indices=list(range(n_groups)), prompt_ids=[[1]] * n_groups, isos=["yor"] * n_groups, sources=[SRC] * n_groups, reward=make_reward(), policy=None,
                        generate_beam=fk.beam, generate_extra=fk.extra, seed=1, key=3, timings={})
    return roll, info, calls


def test_mbr_winner_from_the_extra_samples_replaces_the_row_and_the_group_member_case_keeps_the_beam():
    # members are scattered, the extra samples all agree with each other (a large central cluster): the winner is an extra sample
    members = [ODD, ODD + " lana", "yor eja nla ni odo", GOOD + " ati"]
    roll, info, calls = run_case({"yor": "mbr"}, members, "yor beam output kan ti o yato patapata", GOOD, extra_n=6)
    assert info[2].src == "mbr_extra" and info[2].mode == "mbr" and info[2].candidates >= 8 and ("extra", 6) in calls
    assert roll.completions[2] == ids(GOOD)                                                     # row G - 2 = the guide
    # the winner is one of the group's own members (all members identical, the extras and the beam differ): the beam output stays as the guide
    roll2, info2, _ = run_case({"yor": "mbr"}, [GOOD] * 4, "yor beam output kan ti o yato patapata", "yor eja nla ni odo", extra_n=2)
    assert info2[2].src == "mbr_member->beam" and roll2.completions[2] == ids("yor beam output kan ti o yato patapata")
    # the beam output itself is the central candidate
    roll3, info3, _ = run_case({"yor": "mbr"}, [ODD, "yor eja nla ni odo", "yor omi tutu", "yor ile nla"], GOOD, "yor ni oja agogo", extra_n=1)
    assert info3[2].src in ("mbr_beam", "mbr_member->beam", "mbr_extra")                          # (which one depends on the chrF++ geometry; the cases above pin the other two)


def test_beam_mode_languages_never_ask_for_extra_samples_and_the_extra_zero_switch():
    roll, info, calls = run_case({}, [GOOD] * 4, "yor beam", GOOD)
    assert info[2].src == "beam" and info[2].mode == "beam" and calls == ["beam"]
    roll, info, calls = run_case({"yor": "mbr"}, [GOOD] * 4, "yor beam", GOOD, extra_n=0)       # extra = 0: the MBR languages use the beam guide, no extra generation
    assert info[2].src == "beam" and calls == ["beam"]


def test_budget_skip_and_too_few_valid_candidates():
    roll, info, calls = run_case({"yor": "mbr"}, [GOOD] * 4, "yor beam", GOOD, skip=True)
    assert info[2].src == "mbr_skipped->beam" and calls == ["beam"] and roll.completions[2] == ids("yor beam")
    # nothing valid: no language tag in any candidate (bad_prefix) -> fewer than 3 valid candidates -> the beam output stays
    roll, info, calls = run_case({"yor": "mbr"}, ["no tag here"] * 4, "no tag beam", "no tag extra", extra_n=2)
    assert info[2].src == "mbr_few->beam" and info[2].candidates < 3


def test_only_the_picked_fraction_of_the_groups_gets_a_guide_and_the_pick_is_deterministic():
    G, n = 4, 8
    roll, info, _ = run_case({}, [GOOD] * G, "yor beam", GOOD, fraction=0.25, n_groups=n)
    assert len(info) == 2 and all(r % G == G - 2 for r in info)
    roll2, info2, _ = run_case({}, [GOOD] * G, "yor beam", GOOD, fraction=0.25, n_groups=n)
    assert sorted(info) == sorted(info2)


def test_overhead_tracker():
    t = OverheadTracker(3)
    assert t.running() == 0.0
    for g, s in ((1, 10), (2, 10), (3, 10), (30, 10)):
        t.add(g, s)
    assert t.running() == pytest.approx((2 + 3 + 30) / 30)                                     # only the last 3 steps


def test_loop_with_mbr_languages_and_the_budget(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from gemma_mt.rl.train_rl import run_smoke

    s = run_smoke(sft_steps=60, overrides={"group_size": 4, "guide_fraction": 1.0, "guide_beams": 2, "guide_mode_by_language": {"yor": "mbr", "hau": "mbr"}, "guide_mbr_extra_samples": 4,
                                           "guide_mbr_max_overhead_frac": 10.0, "guide_adv_share_cap": 0.25})
    rows = [json.loads(l) for l in open(s["workdir"] + "/rl/metrics.jsonl")]
    train = [r for r in rows if "reward" in r]
    assert len(train) == 4 and all(abs(r["ratio_mean"] - 1.0) < 1e-5 and r["guide/extra_seconds"] > 0 and r["guide/mbr_skipped"] == 0.0 for r in train)
    table = json.load(open(s["workdir"] + "/rl/guide_by_language.json"))
    assert table
    for iso, v in table.items():
        if iso in ("yor", "hau"):
            assert v["mode"] == "mbr" and all(k.startswith("mbr_") for k in v["src"]) and v["candidates_mean"] >= 0
        else:
            assert v["mode"] == "beam" and set(v["src"]) == {"beam"}
    # a zero budget: the first step spends something, the later steps skip the extra samples
    s2 = run_smoke(sft_steps=60, overrides={"group_size": 4, "guide_fraction": 1.0, "guide_beams": 2, "guide_mode_by_language": {"yor": "mbr", "hau": "mbr", "ibo": "mbr"}, "guide_mbr_extra_samples": 4,
                                            "guide_mbr_max_overhead_frac": 0.0})
    tr2 = [json.loads(l) for l in open(s2["workdir"] + "/rl/metrics.jsonl")]
    tr2 = [r for r in tr2 if "reward" in r]
    assert tr2[0]["guide/mbr_skipped"] == 0.0 and any(r["guide/mbr_skipped"] == 1.0 for r in tr2[1:])
