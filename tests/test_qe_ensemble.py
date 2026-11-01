# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-55: the QE ensemble (rl/qe_ensemble.py), the joint-length budget of UnifiedMetric scorers (rl/joint_budget.py), the two-model worker and the reward integration.
Everything with stub scorers: no model, no GPU."""
import json
import math

import pytest

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import joint_budget as jb
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.qe_ensemble import MEAN, MIN, QEEnsemble, QEPairScorer, combine_pair
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
from gemma_mt.rl.reward_shaping import ContentShaper
from gemma_mt.utils import dataclass_from_dict

TOK = FakeGemmaTokenizer()
EOT, EOS = SPECIALS["<end_of_turn>"], SPECIALS["<eos>"]
STOP = [EOT, EOS]
SRC = "The market opens at eight o'clock. The farmers bring fresh vegetables."
GOOD = "yor Oja n si ni agogo mejo. Awon agbe mu ewebe tutu wa."


def ids(text):
    return TOK.encode(text, add_special_tokens=False) + [EOT]


# ------------------------------------------------------------------------------------------------ the rule
def test_rule_mean_inside_delta_min_outside():
    assert combine_pair(1.0, 1.4, 0.5) == (pytest.approx(1.2), MEAN)
    assert combine_pair(1.4, 1.0, 0.5) == (pytest.approx(1.2), MEAN)             # symmetric
    assert combine_pair(2.0, 0.5, 0.5) == (0.5, MIN)
    assert combine_pair(0.5, 2.0, 0.5) == (0.5, MIN)
    assert combine_pair(-1.0, 1.0, 0.5) == (-1.0, MIN)
    r, br = combine_pair(0.3, 0.8, 0.5)                                        # exactly delta apart: still the mean
    assert br == MEAN and r == pytest.approx(0.55)
    assert combine_pair(1.0, 1.0, 0.0) == (1.0, MEAN)
    assert combine_pair(1.0, 1.0001, 0.0)[1] == MIN                            # delta 0 = always the pessimist unless equal


def test_rule_never_exceeds_the_mean_and_equals_one_model_when_they_agree():
    for r1, r2 in [(0.0, 0.0), (1.0, 2.0), (-2.0, 2.25), (0.4, -0.3)]:
        r, _ = combine_pair(r1, r2, 0.6)
        assert min(r1, r2) - 1e-12 <= r <= (r1 + r2) / 2 + 1e-12


# ------------------------------------------------------------------------------------------------ joint budget
WORD = lambda t: len(t.split())


def test_pair_inside_the_budget_is_untouched():
    assert jb.split_pair("a b c", "x y", WORD, budget=10) == [("a b c", "x y")]
    assert jb.JOINT_CONTENT_BUDGET == 508


def test_long_pair_is_cut_into_pieces_that_fit_and_no_word_is_lost():
    src = " ".join(f"s{i}" for i in range(300))
    mt = " ".join(f"t{i}" for i in range(500))
    pieces = jb.split_pair(src, mt, WORD, budget=100)
    assert len(pieces) > 1
    assert all(WORD(s) + WORD(m) <= 100 for s, m in pieces)
    assert " ".join(s for s, _ in pieces).split() == src.split()
    assert " ".join(m for _, m in pieces).split() == mt.split()


def test_cut_prefers_sentence_boundaries_and_handles_unspaced_and_empty_texts():
    t = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    l, r = jb.cut_text(t, 0.5)
    assert l.endswith(".") and r.startswith(("Seven", "Four", "Ten")) and (l + " " + r).split() == t.split()
    l, r = jb.cut_text("x" * 100, 0.5)                                         # no whitespace: cut inside the string
    assert (len(l), len(r)) == (50, 50)
    assert jb.cut_text("", 0.5) == ("", "")
    pieces = jb.split_pair("w " * 400, "", WORD, budget=100)                   # empty translation: pieces with an empty mt, all within budget
    assert len(pieces) >= 4 and all(WORD(s) <= 100 for s, _ in pieces)


def test_weights_are_source_characters_and_the_combination_is_a_weighted_mean():
    pieces = [("aaaa", "x"), ("aa", "y")]
    w = jb.piece_weights(pieces)
    assert w == [4.0, 2.0] and jb.combine([1.0, 4.0], w) == pytest.approx(2.0)
    assert jb.piece_weights([("", "x"), ("", "y")]) == [1.0, 1.0]


def test_plan_pieces_only_splits_what_is_over_budget():
    srcs = ["a b c", " ".join(["w"] * 60)]
    mts = ["x y", " ".join(["v"] * 60)]
    plans, n = jb.plan_pieces(srcs, mts, lambda ts: [WORD(t) for t in ts], budget=100)
    assert n == 1 and plans[0] == [(srcs[0], mts[0])] and len(plans[1]) > 1


# ------------------------------------------------------------------------------------------------ ensemble object
def shaper1(**kw):
    sh = ContentShaper(calibration={"yor": (0.5, 0.6, 0.7), "hau": (0.5, 0.6, 0.7), "dyu": (0.5, 0.6, 0.7)}, fail_floor=0.0, **kw)
    sh.rel = {"yor": 0.6, "hau": 0.9, "dyu": 0.4}
    return sh


QE2 = {"yor": {"anchors": [0.30, 0.55, 0.70], "usable": True, "reliability": 0.9, "reliability_same_source": 1.0, "anchors_source": "calibrated"},
       "hau": {"anchors": [0.30, 0.55, 0.70], "usable": False, "reliability": 0.0, "anchors_source": "calibrated"},
       "dyu": {"anchors": [0.35, 0.55, 0.68], "usable": True, "anchors_source": "pooled_prior"}}
ENS = {"yor": {"reliability": 0.8, "reliability_same_source": 1.0}, "dyu": {"reliability": 0.5, "reliability_same_source": 0.6}}


def ens(delta=0.5, **kw):
    return QEEnsemble(shaper1(), QE2, delta, ens=ENS, **kw)


def test_shape_is_the_rule_applied_to_the_two_language_curves():
    e = ens()
    r1 = shaper1()(0.65, "yor")                                                 # between t_mid and t_great of model 1
    r2 = e.shaper2(0.65, "yor")                                                 # model 2 has its OWN anchors: different value, same scale
    assert r1 != r2 and -2.0 <= r1 <= 2.25 and -2.0 <= r2 <= 2.25
    r = e.shape(0.65, 0.65, "yor")
    assert r == pytest.approx(combine_pair(r1, r2, 0.5)[0])
    big = e.shape(0.2, 0.7, "yor")                                              # model 1 fails the chunk (-2 side), model 2 loves it -> the pessimist wins
    assert big == pytest.approx(shaper1()(0.2, "yor"))
    assert e.shape(0.2, 0.7, "yor") < 0.5 * (shaper1()(0.2, "yor") + e.shaper2(0.7, "yor"))


def test_fallbacks_one_model_only():
    e = ens()
    assert e.plan("yor") == "both" and e.plan("hau") == "only1"                # hau: model 2 unusable -> AfriCOMET alone
    assert e.shape(0.65, None, "hau") == pytest.approx(shaper1()(0.65, "hau"))
    sh = shaper1()
    sh.unusable.add("yor")                                                      # model 1 unusable for yor, model 2 fine -> model 2 alone
    e2 = QEEnsemble(sh, QE2, 0.5, ens=ENS)
    assert e2.plan("yor") == "only2"
    assert e2.shape(None, 0.65, "yor") == pytest.approx(e2.shaper2(0.65, "yor"))
    assert ens().plan("zzz") == "only1"                                         # no qe2 block at all
    assert not ens().uses_second("hau") and ens().uses_second("yor")


def test_prior_languages_can_be_left_on_the_primary():
    assert ens(prior_langs=True).plan("dyu") == "both"
    assert ens(prior_langs=False).plan("dyu") == "only1"
    assert ens(prior_langs=False).plan("yor") == "both"


def test_weights_follow_the_config():
    e = ens()
    assert e.weight("yor", 0.6) == 0.8 and e.weight("hau", 0.9) == 0.9 and e.weight("zzz", 0.3) == 0.3
    assert QEEnsemble(shaper1(), QE2, 0.5, ens=ENS, reliability_from="same_source").weight("dyu", 0.4) == 0.6
    assert QEEnsemble(shaper1(), QE2, 0.5, ens=ENS, reliability="keep").weight("yor", 0.6) == 0.6


def test_validation():
    with pytest.raises(ValueError):
        QEEnsemble(shaper1(), QE2, -0.1)
    with pytest.raises(ValueError):
        QEEnsemble(shaper1(), QE2, 0.5, reliability="x")
    with pytest.raises(ValueError):
        QEEnsemble(ContentShaper(mode="adaptive"), QE2, 0.5)


def test_pop_stats_reports_branches_differences_and_resets():
    e = ens(delta=0.5)
    e.shape(0.65, 0.65, "yor")
    e.shape(0.2, 0.7, "yor")
    e.shape(0.65, None, "hau")
    st = e.pop_stats()
    assert st["qe_ens/chunks"] == 3 and st["qe_ens/both_share"] == pytest.approx(2 / 3) and st["qe_ens/only1_share"] == pytest.approx(1 / 3)
    assert st["qe_ens/min_share"] == pytest.approx(0.5) and st["qe_ens/min_share_ssa_high"] == pytest.approx(0.5) and st["qe_ens/min_share_afri_high"] == 0.0
    assert st["qe_ens/abs_diff_mean"] > 0 and st["qe_ens/min_share/yor"] == pytest.approx(0.5)
    assert st["qe_ens/raw_mean/africomet"] == pytest.approx((0.65 + 0.2 + 0.65) / 3) and st["qe_ens/raw_mean/ssa_mtl"] == pytest.approx((0.65 + 0.7) / 2)
    assert e.pop_stats()["qe_ens/chunks"] == 0


# ------------------------------------------------------------------------------------------------ file + config
def write_v8(tmp_path, delta=0.6, model="McGill-NLP/ssa-comet-mtl"):
    d = {"_meta_v8": {"delta": delta, "secondary_model": model},
         "yor": {"anchors": [0.5, 0.6, 0.7], "usable": True, "reliability": 0.6, "qe2": QE2["yor"], "ens": ENS["yor"]},
         "hau": {"anchors": [0.5, 0.6, 0.7], "usable": True, "reliability": 0.9}}
    p = tmp_path / "v8.json"
    p.write_text(json.dumps(d))
    return str(p)


def test_from_config_reads_delta_and_blocks_from_the_calibration(tmp_path):
    path = write_v8(tmp_path)
    cfg = RewardConfig(qe_ensemble=True, calibration_path=path, fail_floor=0.0)
    e = QEEnsemble.from_config(cfg, ContentShaper.from_file(path, fail_floor=0.0))
    assert e.delta == 0.6 and e.plan("yor") == "both" and e.plan("hau") == "only1"
    cfg2 = RewardConfig(qe_ensemble=True, qe_ensemble_delta=0.2, calibration_path=path, fail_floor=0.0)
    assert QEEnsemble.from_config(cfg2, ContentShaper.from_file(path)).delta == 0.2
    with pytest.raises(ValueError, match="differs"):
        QEEnsemble.from_config(RewardConfig(qe_ensemble=True, calibration_path=write_v8(tmp_path, model="other/model")), ContentShaper.from_file(path))
    v7 = tmp_path / "v7.json"
    v7.write_text(json.dumps({"yor": {"anchors": [0.5, 0.6, 0.7]}}))
    with pytest.raises(ValueError, match="qe2"):
        QEEnsemble.from_config(RewardConfig(qe_ensemble=True, calibration_path=str(v7), qe_ensemble_delta=0.5), ContentShaper.from_file(str(v7)))


def test_config_flags_parse_from_a_yaml_style_dict_and_default_off():
    assert RewardConfig().qe_ensemble is False
    rc = dataclass_from_dict(RewardConfig, {"qe_ensemble": True, "qe_ensemble_delta": 0.55, "qe_ensemble_prior_langs": False, "qe_ensemble_reliability": "keep",
                                            "qe_ensemble_calibration_path": "/x/v8.json", "qe_ensemble_model_id": "McGill-NLP/ssa-comet-mtl"})
    assert rc.qe_ensemble and rc.qe_ensemble_delta == 0.55 and not rc.qe_ensemble_prior_langs and rc.qe_ensemble_reliability == "keep"


# ------------------------------------------------------------------------------------------------ scorers
class Const:
    def __init__(self, v):
        self.v, self.calls = v, []

    def __call__(self, s, m, r=None):
        self.calls.append((list(s), list(m)))
        return [self.v] * len(s)


def test_pair_scorer_primary_is_the_call_and_token_lengths_forward():
    a, b = FakeScorer(), Const(0.5)
    p = QEPairScorer(a, b)
    assert p(["x y"], ["z"]) == a(["x y"], ["z"]) and hasattr(p, "token_lengths") and p.token_lengths(["a b c"]) == [5]
    assert p.score_secondary(["x"], ["y"]) == [0.5]
    t1, t2 = p.pop_times()
    assert t1 >= 0 and t2 >= 0 and p.pop_times() == (0.0, 0.0)
    assert not hasattr(QEPairScorer(Const(0.1), Const(0.2)), "token_lengths")      # a primary without token_lengths: the attribute does not exist (reward.py hasattr test)


def test_worker_holds_two_models_and_addresses_them_by_index():
    from gemma_mt.rl.comet import WorkerScorer

    w = WorkerScorer(fake=True, extra_model_ids=["McGill-NLP/ssa-comet-mtl"], startup_timeout=120, request_timeout=60)
    try:
        v0, v1 = w.view(0), w.view(1)
        s0 = v0(["The market opens."], ["Oja n si."])
        s1 = v1(["The market opens."], ["Oja n si."])
        assert len(s0) == len(s1) == 1 and s0 == s1                                # FakeScorer in both slots
        assert w(["a b"], ["c"]) == FakeScorer()(["a b"], ["c"])                   # the single-model call is unchanged
        with pytest.raises(RuntimeError, match="error"):
            w(["a"], ["b"], model=5)
    finally:
        w.close()


def test_build_pair_scorer_fake():
    from gemma_mt.rl.qe_ensemble import build_pair_scorer

    p = build_pair_scorer({"type": "fake", "ensemble_model_id": "McGill-NLP/ssa-comet-mtl"})
    assert isinstance(p, QEPairScorer) and p.score_secondary(["The market."], ["Oja."]) == p(["The market."], ["Oja."])


# ------------------------------------------------------------------------------------------------ reward integration
def reward(tmp_path, s1=0.65, s2=0.65, delta=0.5, **cfgkw):
    path = write_v8(tmp_path, delta=delta)
    cfg = RewardConfig(qe_ensemble=True, calibration_path=path, fail_floor=0.0, eos_discipline=False, lid_enabled=False, **cfgkw)
    sc = QEPairScorer(Const(s1), Const(s2))
    return TranslationReward(cfg, sc, TOK, STOP), sc


def test_reward_with_ensemble_uses_the_rule_and_reports_both_scores(tmp_path):
    r, sc = reward(tmp_path, 0.65, 0.65)
    out = r([Sample(ids(GOOD), "yor", SRC)])
    sh1, sh2 = r.shaper, r.qe_ens.shaper2
    assert out.status == ["ok"] and out.rewards[0] == pytest.approx(combine_pair(float(sh1(0.65, "yor")), float(sh2(0.65, "yor")), 0.6)[0])
    assert out.qe == [pytest.approx(0.65)] and out.qe2 == [pytest.approx(0.65)]
    assert out.weights == [0.8]                                                  # the ensemble's baked weight for yor (v7: 0.6)
    assert out.stats["qe_ens/both_share"] == 1.0 and "qe_ens/abs_diff_mean" in out.stats and out.stats["reward_mean"] == out.rewards[0]
    assert len(sc.primary.calls) == 1 and len(sc.secondary.calls) == 1 and sc.secondary.calls[0][1] == sc.primary.calls[0][1]      # same segments


def test_disagreement_takes_the_lower_judge_and_the_hack_signature_is_logged(tmp_path):
    r, _ = reward(tmp_path, 0.30, 0.70)                                          # AfriCOMET says failed, SSA-COMET-MTL says great
    out = r([Sample(ids(GOOD), "yor", SRC)])
    assert out.rewards[0] == pytest.approx(float(r.shaper(0.30, "yor")))
    assert out.stats["qe_ens/min_share"] == 1.0 and out.stats["qe_ens/min_share_ssa_high"] == 1.0


def test_language_without_second_model_is_scored_by_the_first_only(tmp_path):
    r, sc = reward(tmp_path, 0.65, 0.1)
    out = r([Sample(ids(GOOD.replace("yor", "hau", 1)), "hau", SRC)])
    assert out.rewards[0] == pytest.approx(float(r.shaper(0.65, "hau"))) and out.qe2 == [None] and out.weights == [0.9]
    assert sc.secondary.calls == []                                              # the second model was not even called


def test_chunked_reward_scores_every_chunk_with_both_models(tmp_path):
    r, sc = reward(tmp_path, 0.65, 0.65, chunking="sentences", omission_detector=False, extra_output_penalty=False)
    out = r([Sample(ids(GOOD), "yor", SRC)])
    assert len(sc.primary.calls[0][0]) == 2 == len(sc.secondary.calls[0][0]) and out.stats["qe_ens/chunks"] == 2.0


def test_ensemble_off_is_the_old_reward_bit_for_bit(tmp_path):
    path = write_v8(tmp_path)
    base = dict(calibration_path=path, fail_floor=0.0, eos_discipline=False, lid_enabled=False)
    a = TranslationReward(RewardConfig(**base), Const(0.65), TOK, STOP)
    b_sc = QEPairScorer(Const(0.65), Const(0.1))
    b = TranslationReward(RewardConfig(**base), b_sc, TOK, STOP)                  # the pair scorer is there but the flag is off: only the primary runs
    s = [Sample(ids(GOOD), "yor", SRC)]
    oa, ob = a(s), b(s)
    assert oa.rewards == ob.rewards and oa.weights == ob.weights == [0.6] and ob.qe2 == [] and not any(k.startswith("qe_ens/") for k in ob.stats)
    assert b_sc.secondary.calls == []


def test_ensemble_needs_a_two_model_scorer(tmp_path):
    with pytest.raises(ValueError, match="ensemble_model_id"):
        TranslationReward(RewardConfig(qe_ensemble=True, calibration_path=write_v8(tmp_path), fail_floor=0.0), Const(0.5), TOK, STOP)


def test_blend_keeps_the_consensus_term_around_the_ensemble_value(tmp_path):
    path = write_v8(tmp_path)
    d = json.load(open(path))
    d["yor"]["content"] = {"mode": "blend", "beta": 0.5}
    json.dump(d, open(path, "w"))
    cfg = RewardConfig(qe_ensemble=True, calibration_path=path, fail_floor=0.0, eos_discipline=False, lid_enabled=False, consensus=True)
    r = TranslationReward(cfg, QEPairScorer(Const(0.65), Const(0.65)), TOK, STOP)
    out = r([Sample(ids(GOOD), "yor", SRC)])                                       # one sample alone: no consensus -> blend falls back to the ensemble value only
    assert out.content_mode == ["qe"] and out.rewards[0] == pytest.approx(combine_pair(float(r.shaper(0.65, "yor")), float(r.qe_ens.shaper2(0.65, "yor")), 0.6)[0])


def test_smoke_cpu_with_the_ensemble_on(tmp_path):
    """The whole RL loop on CPU with stub scorers: calibration with qe2 blocks, a QEPairScorer, the reward/qe_ens/* statistics in the log."""
    pytest.importorskip("torch")
    from gemma_mt.rl.train_rl import run_smoke

    langs = ["yor", "hau", "ibo"]
    cal = {"_meta_v8": {"delta": 0.5, "secondary_model": "McGill-NLP/ssa-comet-mtl"}}
    for l in langs:
        cal[l] = {"anchors": [0.2, 0.4, 0.6], "usable": True, "reliability": 1.0, "qe2": {"anchors": [0.1, 0.3, 0.5], "usable": True, "reliability": 0.9, "anchors_source": "calibrated"}}
    p = tmp_path / "v8.json"
    p.write_text(json.dumps(cal))
    out = run_smoke(str(tmp_path / "w"), sft_steps=40, overrides={"reward": {"max_comet_tokens": 510, "qe_ensemble": True, "calibration_path": str(p), "fail_floor": 0.0},
                                                                    "scorer": {"type": "fake", "ensemble_model_id": "McGill-NLP/ssa-comet-mtl"}})
    assert out["steps"] >= 1
    rows = [json.loads(l) for l in open(tmp_path / "w" / "rl" / "metrics.jsonl")] if (tmp_path / "w" / "rl" / "metrics.jsonl").exists() else []
    if rows:
        assert any(any(k.startswith("reward/qe_ens/") for k in r) for r in rows)


# ------------------------------------------------------------------------------------------------ calibration script (compose v8 / choose delta) on synthetic scores
def _script():
    import importlib.util
    import os

    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "calibrate_qe_ensemble.py")
    spec = importlib.util.spec_from_file_location("calibrate_qe_ensemble", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fake_scores(rng, n, human, mism, copy, half, append):
    g = lambda mu: [round(min(1.0, max(0.0, rng.gauss(mu, 0.05))), 4) for _ in range(n)]
    return {"human": g(human), "mismatch": g(mism), "copy": g(copy), "half": g(half), "append": g(append)}


def _write_work(tmp_path, langs=("yor", "dyu", "kea"), n=60):
    import random

    rng = random.Random(3)
    for name, prof in (("africomet", (0.70, 0.40, 0.25, 0.45, 0.50)), ("ssa_mtl", (0.65, 0.30, 0.72, 0.40, 0.45))):      # model 2 LIKES the copy of the source (the known weakness)
        d = tmp_path / "work" / name
        d.mkdir(parents=True, exist_ok=True)
        for l in langs:
            (d / f"{l}.json").write_text(json.dumps({"iso": l, "model": name, "n": n, "source": "synthetic", "scores": _fake_scores(rng, n, *prof)}))
    v7 = {"_meta": {"x": 1},
          "yor": {"anchors": [0.5, 0.6, 0.7], "usable": True, "reliability": 0.6, "reliability_same_source": 0.7, "content": {"mode": "qe"}, "n": 997},
          "dyu": {"anchors": [0.5, 0.55, 0.6], "usable": True, "reliability": 0.4, "reliability_same_source": 0.4, "weight_reason": "owner 2026-10-19: 0.4", "content": {"mode": "blend", "beta": 0.5}},
          "kea": {"usable": True, "reliability": 0.5, "reliability_same_source": 0.5, "weight_reason": "owner 2026-10-19: 0.5", "content": {"mode": "consensus", "beta": 1.0}},
          "loz": {"anchors": [0.5, 0.6, 0.7], "usable": True, "reliability": 0.3, "reliability_same_source": 0.3, "anchors_source": "pooled prior: median", "weight_reason": "owner 2026-10-19: low",
                  "content": {"mode": "blend", "beta": 0.5}},
          "swc": {"anchors": [0.5, 0.6, 0.7], "usable": True, "reliability": 0.3, "reliability_same_source": 0.3, "anchors_source": "borrowed from swh", "weight_reason": "owner: low", "content": {"mode": "blend"}}}
    (tmp_path / "v7.json").write_text(json.dumps(v7))
    return v7


def test_compose_v8_keeps_v7_adds_blocks_and_fixes_the_copy_weakness(tmp_path):
    v7 = _write_work(tmp_path)
    s = _script()
    assert s.main(["compose", "--work", str(tmp_path / "work"), "--v7", str(tmp_path / "v7.json"), "--out", str(tmp_path / "v8.json"), "--delta", "0.9", "--table", str(tmp_path / "t.md")]) == 0
    v8 = json.load(open(tmp_path / "v8.json"))
    for k, v in v7.items():                                                      # v8 = v7 + new blocks: every v7 key untouched
        assert all(v8[k][kk] == vv for kk, vv in v.items()), k
    assert v8["_meta_v8"]["delta"] == 0.9 and v8["_meta_v8"]["secondary_model"] == "McGill-NLP/ssa-comet-mtl"
    y = v8["yor"]
    assert y["qe2"]["usable"] and len(y["qe2"]["anchors"]) == 3 and y["qe2"]["auc_vs_mismatched"] > 0.95
    assert y["qe2"]["same_source_accuracy"]["copy"] < 0.3                         # the joint encoder accepts the copy ...
    assert y["ens"]["same_source_accuracy"]["copy"] > 0.9                         # ... the ensemble (min branch) does not
    assert y["ens"]["min_branch_share"]["copy"] > 0.9 and y["ens"]["min_branch_share"]["human"] < 0.5
    assert y["ens"]["reliability"] >= v7["yor"]["reliability"] and y["ens"]["reliability"] <= 1.0
    assert v8["dyu"]["ens"]["reliability"] == 0.4 and "owner" in v8["dyu"]["ens"]["reliability_rule"]       # owner-set weights are never raised
    assert "qe2" not in v8["kea"] and v8["kea"]["qe2_diagnostic"]["n"] == 60       # consensus-only: measured for the report, never called
    assert v8["loz"]["qe2"]["anchors_source"] == "pooled_prior" and v8["loz"]["qe2"]["reliability"] == 0.3 and "ens" not in v8["loz"]
    assert v8["swc"]["qe2"]["anchors_source"] == "pooled_prior" or v8["swc"]["qe2"]["anchors_source"] == "borrowed"
    assert (tmp_path / "t.md").read_text().count("\n") > 5
    e = QEEnsemble.from_file(str(tmp_path / "v8.json"), ContentShaper.from_file(str(tmp_path / "v8.json"), fail_floor=0.0), fail_floor=0.0)
    assert e.delta == 0.9 and e.plan("yor") == "both" and e.plan("loz") == "both" and e.weight("loz", 0.3) == 0.3 and e.weight("yor", 0.6) == y["ens"]["reliability"]
    assert QEEnsemble.from_file(str(tmp_path / "v8.json"), ContentShaper.from_file(str(tmp_path / "v8.json"), fail_floor=0.0), prior_langs=False, fail_floor=0.0).plan("loz") == "only1"


def test_compose_marks_a_language_unusable_for_the_second_model_and_falls_back(tmp_path):
    import random

    _write_work(tmp_path, langs=("yor",))
    rng = random.Random(5)
    bad = _fake_scores(rng, 60, 0.5, 0.5, 0.5, 0.5, 0.5)                          # SSA-COMET-MTL cannot tell a reference from a mismatched one for this language
    (tmp_path / "work" / "ssa_mtl" / "yor.json").write_text(json.dumps({"iso": "yor", "model": "m", "n": 60, "source": "synthetic", "scores": bad}))
    s = _script()
    s.main(["compose", "--work", str(tmp_path / "work"), "--v7", str(tmp_path / "v7.json"), "--out", str(tmp_path / "v8.json"), "--delta", "0.9"])
    v8 = json.load(open(tmp_path / "v8.json"))
    assert v8["yor"]["qe2"]["usable"] is False and "ens" not in v8["yor"] and "yor" in v8["_meta_v8"]["languages_unusable_for_secondary"]
    e = QEEnsemble.from_file(str(tmp_path / "v8.json"), ContentShaper.from_file(str(tmp_path / "v8.json"), fail_floor=0.0), fail_floor=0.0)
    assert e.plan("yor") == "only1" and e.weight("yor", 0.6) == 0.6


def test_delta_search_picks_the_smallest_delta_inside_the_min_branch_budget(tmp_path):
    import random

    _write_work(tmp_path, langs=("yor", "dyu"))
    s = _script()
    s.main(["compose", "--work", str(tmp_path / "work"), "--v7", str(tmp_path / "v7.json"), "--out", str(tmp_path / "v8.json"), "--delta", "1.0"])
    a5 = tmp_path / "a5"
    (a5 / "policy_scores").mkdir(parents=True)
    (a5 / "hack_scores").mkdir()
    rng = random.Random(1)
    pol, prow, p1, p2 = [], [], [], []
    for i in range(400):
        iso = "yor" if i % 2 else "dyu"
        pol.append({"iso": iso, "prefix_status": "ok"})
        prow.append({"src": "s", "mt": "t", "ref": "r"})
        p1.append(round(rng.uniform(0.3, 0.8), 3))
        p2.append(round(rng.uniform(0.2, 0.8), 3))
    hk, h1, h2 = [], [], []
    for iso in ("yor", "dyu"):
        for cls, (m1, m2) in {"human": (0.7, 0.65), "mismatch": (0.4, 0.3), "copy_source": (0.25, 0.62)}.items():
            for j in range(30):
                hk.append({"iso": iso, "cls": cls, "j": j, "src": "s", "mt": "t", "ref": "r"})
                h1.append(m1 + 0.01 * (j % 3))
                h2.append(m2 + 0.01 * (j % 3))
    for name, rows in (("policy_samples", pol), ("policy_rows", prow), ("hack_rows", hk)):
        (a5 / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    json.dump(p1, open(a5 / "policy_scores" / "africomet_qe_stl11.json", "w")); json.dump(p2, open(a5 / "policy_scores" / "ssa_mtl_qe.json", "w"))
    json.dump(h1, open(a5 / "hack_scores" / "africomet_qe_stl11.json", "w")); json.dump(h2, open(a5 / "hack_scores" / "ssa_mtl_qe.json", "w"))
    assert s.main(["delta", "--v8", str(tmp_path / "v8.json"), "--a5", str(a5), "--out", str(tmp_path / "delta.json")]) == 0
    rep = json.load(open(tmp_path / "delta.json"))
    grid = rep["grid"]
    chosen = next(g for g in grid if g["delta"] == rep["delta"])
    assert chosen["policy_min_share"] <= 0.15 and all(g["policy_min_share"] > 0.15 for g in grid if g["delta"] < rep["delta"])      # the SMALLEST delta inside the budget
    assert all(a["policy_min_share"] >= b["policy_min_share"] for a, b in zip(grid, grid[1:]))                                         # monotone: a larger delta never adds min-branch chunks
    assert (tmp_path / "delta.md").exists() and set(rep["per_language"]) == {"yor", "dyu"}


def test_the_shipped_v8_calibration_is_a_superset_of_v7():
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p8, p7 = os.path.join(root, "calibration", "shaping_flores_dev_v8.json"), os.path.join(root, "calibration", "shaping_flores_dev_v7.json")
    if not (os.path.exists(p8) and os.path.exists(p7)):
        pytest.skip("shaping_flores_dev_v8.json not built yet")
    v7, v8 = json.load(open(p7)), json.load(open(p8))
    for k, v in v7.items():
        assert k in v8 and (k.startswith("_") and v8[k] == v or all(v8[k][kk] == vv for kk, vv in v.items())), k
    meta = v8["_meta_v8"]
    assert 0.0 < meta["delta"] < 5.0
    for k, v in v8.items():
        if k.startswith("_"):
            continue
        mode = (v.get("content") or {}).get("mode", "qe")
        if mode == "consensus":
            assert "qe2" not in v
            continue
        assert v["qe2"]["anchors"][0] < v["qe2"]["anchors"][1] < v["qe2"]["anchors"][2] and 0.0 <= v["qe2"].get("reliability", 1.0) <= 1.0, k
        if "ens" in v:
            assert 0.0 <= v["ens"]["reliability"] <= 1.0 and v["ens"]["reliability"] >= v["reliability"] - 1e-9 and v["ens"]["delta"] == meta["delta"], k
    e = QEEnsemble.from_file(p8, ContentShaper.from_file(p8, fail_floor=0.0), fail_floor=0.0)
    assert e.delta == meta["delta"] and e.plan("yor") == "both"


# ------------------------------------------------------------------------------------------------ batches of long chunks
def test_cost_capped_batches_keep_order_and_never_exceed_the_cap():
    from gemma_mt.rl.comet import batched_in_length_order

    items = [5, 900, 7, 3, 1200, 40, 41, 800, 6, 2000]
    seen = []

    def run(b):
        seen.append(list(b))
        return [float(x) for x in b]

    out = batched_in_length_order(items, lambda x: x, run, batch_size=4, max_cost=2500)
    assert out == [float(x) for x in items]                                     # original order
    assert all(len(b) * max(b) <= 2500 or len(b) == 1 for b in seen) and all(len(b) <= 4 for b in seen) and sum(len(b) for b in seen) == len(items)
    seen.clear()
    assert batched_in_length_order(items, lambda x: x, run, batch_size=4) == [float(x) for x in items]
    assert [len(b) for b in seen] == [4, 4, 2]                                  # no cap: the old fixed-size batches
