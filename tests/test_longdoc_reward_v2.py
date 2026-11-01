# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reward v2 for long documents (D-60, rl/longdoc.py): detectors, severity, tail-sensitive aggregation, anchored alignment, and the switch in TranslationReward.
CPU only; the QE model is a fake scorer."""
import json
import random

import pytest

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import longdoc as LD
from gemma_mt.rl.chunk_plan import aggregate, plan_chunks
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
from gemma_mt.rl.segments import LengthModel, split_sentences

TOK = FakeGemmaTokenizer()
EOT, EOS = SPECIALS["<end_of_turn>"], SPECIALS["<eos>"]
STOP = [EOT, EOS]


_SYL = ["ka", "to", "mi", "ra", "su", "ne", "lo", "pa", "di", "ve", "zu", "ha", "ki", "mo", "ta", "ro", "ni", "sa", "bu", "fe"]


def pw(rng, lead):
    """A pseudo-word without digits (3 syllables, ~8,000 possible words) starting with a language-specific letter so English and 'translated' words never collide."""
    return lead + "".join(rng.choice(_SYL) for _ in range(3))


def words(rng, n, vocab=None):
    ws = [pw(rng, "w") for _ in range(n)]
    ws[0] = ws[0].capitalize()
    return " ".join(ws)


def make_doc(n_sents, seed=0, numbers=True, dense=False):
    """(English source, translation) with n_sents sentences; every 7th sentence carries a number and every 5th a name, translation keeps them (`dense`: every sentence has its own name)."""
    rng = random.Random(seed)
    src, out = [], []
    for i in range(n_sents):
        s = f"Sentence {i} " + " ".join(pw(rng, "e") for _ in range(rng.randrange(8, 16)))
        t = f"Fahrasa {i} " + " ".join(pw(rng, "t") for _ in range(rng.randrange(8, 16)))
        if numbers and i % 7 == 3:
            s += f" in {1900 + i} total"
            t += f" ni {1900 + i}"
        if i % 5 == 1 or dense:
            s += f" with Tanzania{i}"
            t += f" na Tanzania{i}"
        src.append(s + ".")
        out.append(t + ".")
    return src, out


# ------------------------------------------------------------------------------------------------ repetition profile
def test_natural_text_has_low_levels_and_loops_have_high_ones():
    rng = random.Random(1)
    nat = ". ".join(words(rng, 12) for _ in range(60)) + "."
    p = LD.repetition_profile(nat)
    assert p.rep4 < 0.05 and p.cover8 < 0.03 and p.dup_sent == 0 and p.cycle == 0
    loop = nat + " " + " ".join(["muna muna ina"] * 40)
    pl = LD.repetition_profile(loop)
    assert pl.cycle > 0.12 and pl.cycle_at_tail and pl.cycle_period == 3
    sents = split_sentences(nat)
    dup = " ".join(sents + sents[-6:] * 3)
    pd = LD.repetition_profile(dup)
    assert pd.dup_sent > 0.2 and pd.dup_block > 0.2 and pd.cover8 > 0.2


def test_short_repeated_sentences_are_not_duplicates():
    t = "Yes. No. Yes. No. Yes. No. " + words(random.Random(2), 30) + "."
    p = LD.repetition_profile(t)
    assert p.dup_sent == 0.0


def test_drop_redundant_keeps_first_copies_and_order():
    s = ["a b c d e f.", "g h i j k l.", "a b c d e f.", "m n o p q r.", "g h i j k l.", "m n o p q r."]
    kept, removed = LD.drop_redundant(s)
    assert kept == s[:2] + [s[3]] and removed == [2, 4, 5]
    assert LD.drop_redundant(s[:2])[1] == []


def test_word_cycles_period_and_tail():
    w = [f"u{i}" for i in range(30)] + ["a", "b", "c"] * 10
    sh, per, tail = LD.word_cycles(w)
    assert per == 3 and tail and 0.3 < sh < 0.5
    assert LD.word_cycles([f"u{i}" for i in range(50)]) == (0.0, 0, False)
    assert LD.word_cycles(["x"] * 5 + [f"u{i}" for i in range(40)])[0] == 0.0       # period 1 needs 6 repeats


# ------------------------------------------------------------------------------------------------ severity
def test_severity_is_zero_below_the_limit_and_one_above_t1_and_monotone():
    cal = LD.LongdocCalibration()
    lo, hi = LD.RepProfile(n_words=500), LD.RepProfile(n_words=500, cover8=0.9)
    assert cal.severity(lo, 400)[0] == 0.0
    assert cal.severity(hi, 400) == (1.0, "cover8")
    prev = 0.0
    for x in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
        s = cal.severity(LD.RepProfile(n_words=500, cover8=x), 400)[0]
        assert s >= prev - 1e-12
        prev = s


def test_excess_over_the_source_removes_repetitive_sources():
    cal = LD.LongdocCalibration()
    out = LD.RepProfile(n_words=500, cover8=0.4)
    assert cal.severity(out, 400)[0] > 0.5
    assert cal.severity(out, 400, LD.RepProfile(n_words=500, cover8=0.4))[0] == 0.0


def test_apply_severity_endpoints_and_floor():
    assert LD.apply_severity(1.5, 0.0, -2.0) == 1.5
    assert LD.apply_severity(1.5, 1.0, -2.0) == -2.0
    assert LD.apply_severity(1.5, 0.5, -2.0) == pytest.approx(-0.25)
    assert LD.apply_severity(-2.0, 1.0, -2.0) == -2.0 and LD.apply_severity(-3.0, 0.7, -2.0) == -3.0     # never raises a reward
    assert LD.combine_severity(0.0, 0.0) == 0.0 and LD.combine_severity(1.0, 0.3) == 1.0
    assert LD.combine_severity(0.5, 0.5) == pytest.approx(0.75)


def test_calibration_file_roundtrip_and_validation(tmp_path):
    t0 = [dict(LD.DEFAULT_T0) for _ in range(len(LD.DEFAULT_BUCKET_EDGES) + 1)]
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"edges": list(LD.DEFAULT_BUCKET_EDGES), "t0": t0, "t1": LD.DEFAULT_T1, "meta": {"x": 1}}))
    c = LD.LongdocCalibration.from_file(str(p))
    assert c.bucket(10) == 0 and c.bucket(100) == 1 and c.bucket(5000) == len(LD.DEFAULT_BUCKET_EDGES)
    bad = tmp_path / "b.json"
    bad.write_text(json.dumps({"edges": [1, 2], "t0": t0}))
    with pytest.raises(ValueError):
        LD.LongdocCalibration.from_file(str(bad))


# ------------------------------------------------------------------------------------------------ aggregation
def test_mean_mode_equals_v1_aggregate():
    items = [(0, 10.0), (None, 5.0), (1, 20.0)]
    rw = {0: 1.0, 1: 0.2}
    assert LD.aggregate_tail(items, rw, -2.0, mode="mean") == pytest.approx(aggregate(items, rw, -2.0))
    short = [(0, 1.0), (1, 1.0)]                                   # fewer than min_chunks: the plain mean whatever the mode
    assert LD.aggregate_tail(short, rw, -2.0, mode="cvar") == pytest.approx(aggregate(short, rw, -2.0))


def test_cvar_is_at_most_the_mean_and_sees_a_localised_failure():
    n = 40
    items = [(k, 100.0) for k in range(n)]
    good = {k: 1.0 for k in range(n)}
    bad = dict(good)
    for k in range(10):
        bad[k] = -1.0                                              # a quarter of the document is bad
    m_good, m_bad = LD.aggregate_tail(items, good, -2.0, mode="mean"), LD.aggregate_tail(items, bad, -2.0, mode="mean")
    c_good, c_bad = LD.aggregate_tail(items, good, -2.0, mode="cvar", alpha=0.5), LD.aggregate_tail(items, bad, -2.0, mode="cvar", alpha=0.5)
    assert c_good == pytest.approx(m_good) and c_bad < m_bad
    assert (c_good - c_bad) > (m_good - m_bad)
    assert LD.aggregate_tail(items, bad, -2.0, mode="min", alpha=1.0) == -1.0
    assert LD.aggregate_tail(items, bad, -2.0, mode="p25", alpha=1.0) == -1.0
    with pytest.raises(ValueError):
        LD.aggregate_tail(items, bad, -2.0, mode="median")


def test_weighted_tail_mean_splits_the_boundary_value():
    assert LD.weighted_tail_mean([0.0, 1.0], [1.0, 3.0], 0.5) == pytest.approx((0.0 * 1 + 1.0 * 1) / 2)
    assert LD.weighted_tail_mean([3.0, 1.0, 2.0], [1, 1, 1], 1.0) == pytest.approx(2.0)


def test_coverage_severity():
    items = [(0, 70.0), (None, 30.0)]
    assert LD.coverage_severity(items) == pytest.approx(0.3) and LD.coverage_severity(items, 0.5) == pytest.approx(0.15)
    assert LD.coverage_severity([(0, 1.0)]) == 0.0 and LD.coverage_severity([]) == 0.0


# ------------------------------------------------------------------------------------------------ anchored alignment
LM = LengthModel(ratio=1.0, s2=3.0)


def true_pairs(beads):
    return {(i, j) for si, oi in beads for i in si for j in oi}


def test_numbers_and_names_become_anchors():
    src, out = make_doc(30)
    pairs = LD.anchor_scores(src, out)
    assert pairs and all(i == j for i, j in pairs)
    assert LD.anchor_chain(pairs) == sorted(pairs)


def test_equal_counts_keep_the_identity():
    src, out = make_doc(25)
    b = LD.anchored_beads(src, out, LM)
    assert b == [((i,), (i,)) for i in range(25)]


def test_dropped_sentences_are_found_exactly_with_anchors():
    src, out = make_doc(60, seed=3, dense=True)
    drop = {5, 6, 20, 41}
    out2 = [o for i, o in enumerate(out) if i not in drop]
    beads = LD.anchored_beads(src, out2, LM)
    omitted = sorted(i for si, oi in beads if not oi for i in si)
    assert omitted == sorted(drop)
    keep = [i for i in range(60) if i not in drop]
    tp = true_pairs(beads) & {(i, k) for k, i in enumerate(keep)}
    assert len(tp) == len(keep)
    sparse_src, sparse_out = make_doc(60, seed=3)                     # anchors only on some sentences: the number of omissions is right, which ones inside a gap is the length model's guess
    sb = LD.anchored_beads(sparse_src, [o for i, o in enumerate(sparse_out) if i not in drop], LM)
    assert sum(len(si) for si, oi in sb if not oi) == len(drop)


def test_duplicate_run_and_loop_tail_do_not_shift_the_alignment():
    src, out = make_doc(50, seed=4)
    out_dup = out[:20] + out[18:21] + out[20:]
    b = LD.anchored_beads(src, out_dup, LM)
    extras = sorted(j for si, oi in b if not si for j in oi)
    assert extras == [20, 21, 22] or extras == [18, 19, 20] or len(extras) == 3
    pairs = true_pairs(b)
    assert (49, 52) in pairs                                                    # the last source sentence is still matched to its translation
    tail = out + out[-2:] * 10
    b2 = LD.anchored_beads(src, tail, LM)
    assert (49, 49) in true_pairs(b2)
    assert len([j for si, oi in b2 if not si for j in oi]) == 20


def test_truncated_output_marks_the_tail_as_omitted():
    src, out = make_doc(40, seed=5)
    beads = LD.anchored_beads(src, out[:25], LM)
    omitted = sorted(i for si, oi in beads if not oi for i in si)
    assert omitted == list(range(25, 40))


def test_degenerate_inputs():
    assert LD.anchored_beads([], [], LM) == []
    assert LD.anchored_beads(["a b."], ["c d.", "e f."], LM) == [((0,), (0, 1))]
    assert LD.anchored_beads(["a b.", "c d."], [], LM) == [((0, 1), ())]


def test_beads_fn_plugs_into_plan_chunks_and_matches_v1_on_equal_counts():
    src, out = make_doc(20, seed=6)
    s, o = " ".join(src), " ".join(out)
    v1 = plan_chunks(s, o, length_model=LM, short_frac=0.7)
    v2 = plan_chunks(s, o, length_model=LM, short_frac=0.7, beads_fn=LD.beads_fn(LM))
    assert v1.segments == v2.segments and v1.items == v2.items and v2.how == "aligned"
    o2 = " ".join(out[:12])
    p = plan_chunks(s, o2, length_model=LM, short_frac=0.7, beads_fn=LD.beads_fn(LM))
    assert p.n_failed == 8 and sum(1 for seg, _ in p.items if seg is None) == 8


# ------------------------------------------------------------------------------------------------ the reward
class SentenceScorer:
    """0.8 for every pair (a constant QE) so differences between outputs come from the structure alone."""

    def __call__(self, s, m, r=None):
        return [0.8] * len(s)


def ids(text, end=True):
    out = TOK.encode(text, add_special_tokens=False)
    return out + [EOT] if end else out


def reward(**kw):
    cfg = RewardConfig(chunking="sentences", omission_detector=False, eos_discipline=False, keep_chunk_rewards=True, **kw)
    return TranslationReward(cfg, SentenceScorer(), TOK, STOP)


def long_sample(out_sents, src_sents, end=True):
    return Sample(ids("yor " + " ".join(out_sents), end), "yor", " ".join(src_sents))


def test_flags_default_off_and_v1_path_unchanged():
    cfg = RewardConfig()
    assert cfg.longdoc_v2 is False and cfg.keep_chunk_rewards is False
    src, out = make_doc(30, seed=7)
    a = reward()
    b = reward(longdoc_v2=False, longdoc_agg="p25", longdoc_coverage_kappa=1.0)
    s = [long_sample(out, src), long_sample(out[:20], src), long_sample(out + out[-3:] * 4, src)]
    ra, rb = a(s), b(s)
    assert ra.rewards == rb.rewards and ra.status == rb.status
    assert all(x is None for x in ra.longdoc) or ra.longdoc == []


def test_v2_reward_ordering_complete_gt_truncated_gt_loop_and_floor():
    src, out = make_doc(60, seed=8)
    r = reward(longdoc_v2=True, longdoc_min_src_words=100)
    samples = [long_sample(out, src),                                   # complete
               long_sample(out[:40], src),                              # last third missing
               long_sample(out[:20], src),                              # only a third left
               long_sample(out + out[-4:] * 8, src),                    # 4-sentence loop at the tail
               long_sample(out, src, end=False)]                        # truncated: no end token
    res = r(samples)
    a, b, c, d, e = res.rewards
    assert res.status[:4] == ["ok"] * 4 and res.status[4] == "no_end_of_turn"
    assert a > b > c
    assert d < a and d >= r.penalty and e == r.penalty
    assert all(x >= r.penalty - 1e-9 for x in res.rewards)
    assert res.longdoc[0]["rep_severity"] == 0.0 and res.longdoc[3]["rep_severity"] > 0.0
    assert res.stats["longdoc/long_rate"] == 1.0 and "longdoc/severity" in res.stats


def test_v2_does_not_flag_a_long_natural_text_that_v1_flags():
    rng = random.Random(9)
    vocab = [f"k{i}" for i in range(150)]                       # a 1,500-word text over 150 distinct words: unique ratio 0.1 < 0.3 trips v1
    src = [" ".join(pw(rng, "e") for _ in range(15)) + "." for _ in range(100)]
    out = [" ".join(rng.choice(vocab) for _ in range(15)) + "." for _ in range(100)]
    v1 = TranslationReward(RewardConfig(chunking="sentences", eos_discipline=False), SentenceScorer(), TOK, STOP)([long_sample(out, src)])
    assert v1.status == ["degenerate"]
    v2 = reward(longdoc_v2=True, longdoc_min_src_words=100)([long_sample(out, src)])
    assert v2.status == ["ok"] and v2.longdoc[0]["rep_severity"] < 0.5


def test_full_loop_gets_the_failure_penalty_and_stays_out_of_the_consensus_pool():
    src, out = make_doc(40, seed=10)
    looping = out[:6] + [out[5]] * 200
    r = reward(longdoc_v2=True, longdoc_min_src_words=100)
    res = r([long_sample(looping, src)])
    assert res.rewards[0] == r.penalty


def test_config_validation():
    with pytest.raises(ValueError):
        reward(longdoc_v2=True, longdoc_agg="median")
    with pytest.raises(ValueError):
        reward(longdoc_v2=True, longdoc_align="nope")
    with pytest.raises(ValueError):
        reward(longdoc_v2=True, longdoc_coverage_kappa=1.5)


def test_a_stuttering_sentence_is_a_failed_chunk_not_a_failed_document():
    src, out = make_doc(60, seed=11)
    bad = list(out)
    bad[30] = bad[30] + " " + " ".join(["shaidar"] * 9)
    r = reward(longdoc_v2=True, longdoc_min_src_words=100)
    res = r([long_sample(out, src), long_sample(bad, src)])
    assert res.status == ["ok", "ok"]
    assert res.rewards[1] < res.rewards[0] and res.rewards[1] > r.penalty + 0.5
    # the same text with longdoc_v2 off: the v1 run rule fails the whole document
    v1 = TranslationReward(RewardConfig(chunking="sentences", eos_discipline=False), SentenceScorer(), TOK, STOP)
    assert v1([long_sample(bad, src)]).status == ["degenerate"]
    assert LD.has_stutter("a b a b a b a b a b a b c") and LD.has_stutter("x " * 6) and not LD.has_stutter("one two three four five six")


def test_cvar_excess_leaves_human_like_spread_alone_and_charges_the_excess():
    n = 40
    items = [(k, 100.0) for k in range(n)]
    base = {k: 1.0 - 0.02 * (k % 7) for k in range(n)}                       # a normal spread: tail deficit well below gap0
    mean = LD.aggregate_tail(items, base, -2.0, mode="mean")
    assert LD.aggregate_tail(items, base, -2.0, mode="cvar_excess", alpha=1.0, gap0=0.9) == pytest.approx(mean)
    bad = dict(base)
    for k in range(10):
        bad[k] = -1.5                                                         # a quarter of the document is bad
    m = LD.aggregate_tail(items, bad, -2.0, mode="mean")
    e = LD.aggregate_tail(items, bad, -2.0, mode="cvar_excess", alpha=1.0, gap0=0.9)
    assert e < m and e == pytest.approx(m - max(0.0, (m - LD.weighted_tail_mean([bad[k] for k in range(n)], [100.0] * n, 0.25)) - 0.9))
    cal = LD.LongdocCalibration()
    assert cal.gap0(10) == cal.tail_gap[0] and cal.gap0(5000) == cal.tail_gap[-1]


def test_duplicates_that_the_source_has_too_are_not_removed():
    src, out = make_doc(30, seed=12, dense=True)
    src2 = src[:10] + [src[3]] + src[10:]                     # the English repeats sentence 3 (eg a repeated table line)
    out2 = out[:10] + [out[3]] + out[10:]                     # ... and so does the translation
    b = LD.anchored_beads(src2, out2, LM)
    assert b == [((i,), (i,)) for i in range(31)]
    looped = out2 + [out2[-1]] * 6
    b2 = LD.anchored_beads(src2, looped, LM)
    assert sorted(j for si, oi in b2 if not si for j in oi) == list(range(31, 37)) or len([1 for si, oi in b2 if not si]) >= 5
    assert LD.drop_redundant(out2, keep=1)[1] == [] and LD.drop_redundant(out2, keep=0)[1] == [10]
