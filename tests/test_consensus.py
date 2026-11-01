# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-38: the QE-free consensus signal: chrF++ equal to sacrebleu, group pools, failure-mode guards, the three content modes in the reward."""
import random
import time

import pytest

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import consensus as cons
from gemma_mt.rl import guards
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
from gemma_mt.rl.reward_shaping import ContentShaper

TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]
STOP = [EOT]
SRC = "The market opens at eight o'clock in the morning and the farmers from the village bring fresh vegetables to sell."
WORDS = "ala ni ogun kan wa si ile ti oja agbe omi igi ewe ina oru ojo osu ọdún, ń! (náà) tí 12.5 \"ọjà\" 'fẹ́' ... ?".split()


def _rt(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def ids(text):
    return TOK.encode(text, add_special_tokens=False) + [EOT]


class CountingScorer(FakeScorer):
    def __init__(self):
        super().__init__()
        self.n_scored = 0

    def __call__(self, s, m, r=None):
        self.n_scored += len(s)
        return super().__call__(s, m, r)


def shaper_with(content, lid=None, rel=None):
    sh = ContentShaper.from_file(None)
    sh.content = dict(content)
    sh.lid = dict(lid or {})
    sh.rel = dict(rel or {})
    return sh


def reward(content=None, lid=None, scorer=None, lid_fn=None, rel=None, **kw):
    cfg = RewardConfig(chunking="sentences", consensus=kw.pop("consensus", True), **kw)
    return TranslationReward(cfg, scorer or CountingScorer(), TOK, STOP, lid=lid_fn, shaper=shaper_with(content or {}, lid, rel))


def group(lang, texts, source=SRC):
    return [Sample(ids(f"{lang} {t}"), lang, source) for t in texts]


GOOD = ["Oja n si ni agogo mejo owuro, awon agbe lati abule si mu ewebe tutu wa lati ta.",
        "Oja n si ni agogo mejo owuro, awon agbe lati abule mu ewebe tutu wa lati ta.",
        "Oja n si ni agogo mejo ni owuro, awon agbe lati abule si mu ewebe tutu wa lati ta.",
        "Oja ni won n si ni agogo mejo owuro, awon agbe lati abule mu ewebe tutu wa lati ta.",
        "Oja n si ni agogo mejo owuro, awon agbe lati abule si mu ewebe tuntun wa lati ta.",
        "Oja n si ni agogo mejo owuro, awon agbe lati abule si mu ewebe tutu wa fun tita.",
        "Oja n si ni agogo mejo owuro, awon agbe lati abule si mu ewebe tutu wa lati ta.",
        "Ni agogo mejo owuro oja n si, awon agbe lati abule si mu ewebe tutu wa lati ta."]


# ------------------------------------------------------------------------------------------------ chrF++
def test_chrf_equals_sacrebleu_sentence_chrf_pp():
    sacrebleu = pytest.importorskip("sacrebleu")
    from sacrebleu.metrics import CHRF

    c = CHRF(word_order=2)
    rng = random.Random(0)
    worst = 0.0
    for _ in range(300):
        a, b = _rt(rng, rng.randint(0, 25)), _rt(rng, rng.randint(0, 25))
        worst = max(worst, abs(c.sentence_score(a, [b]).score - cons.chrf_sentence(a, b)))
    assert worst < 1e-9
    assert cons.chrf_sentence("", "") == 0.0 and cons.chrf_sentence("abc", "") == 0.0 and abs(cons.chrf_sentence("hello world", "hello world") - 100.0) < 1e-9


def test_profile_cache_and_pair_symmetry():
    rng = random.Random(1)
    a, b = _rt(rng, 20), _rt(rng, 22)
    pa, pb = cons.Profile(a), cons.Profile(b)
    assert cons.matches(pa, pb) == cons.matches(pb, pa)
    cache = cons.ProfileCache(size=2)
    assert cache.get(a) is cache.get(a) and cache.get(b) is not cache.get(a)


def test_group_consensus_pool_rules():
    rng = random.Random(2)
    base = _rt(rng, 25)
    ps = [cons.Profile(base), cons.Profile(base), cons.Profile(base + " extra"), cons.Profile(_rt(rng, 25))]
    v = cons.group_consensus(ps)
    assert all(x is not None for x in v)
    assert v[0] == v[1] > v[3]                                   # the outlier agrees least with the others
    # invalid members neither receive nor give; < 3 valid -> the whole group is skipped
    v2 = cons.group_consensus([ps[0], None, ps[2], None])
    assert v2 == [None] * 4
    v3 = cons.group_consensus([ps[0], None, ps[2], ps[3]])
    assert v3[1] is None and v3[0] is not None
    # removing the invalid sample from the pool gives exactly the value of the three-member group
    assert v3 == [x for x in cons.group_consensus([ps[0], ps[2], ps[3]])][:1] + [None] + cons.group_consensus([ps[0], ps[2], ps[3]])[1:]
    # identical samples: perfect agreement (a collapsed group is flat)
    same = [cons.Profile(base)] * 8
    assert all(abs(x - 100.0) < 1e-9 for x in cons.group_consensus(same))


def test_consensus_value_is_the_fixed_linear_map():
    assert cons.consensus_value(0.0) == -2.0 and cons.consensus_value(100.0) == 2.25 and abs(cons.consensus_value(50.0) - 0.125) < 1e-12
    assert cons.consensus_value(150.0) == 2.25 and cons.consensus_value(-5.0) == -2.0
    assert abs(cons.consensus_value(40.0, -1, 1) - (-0.2)) < 1e-12


def test_split_groups_handles_ragged_batches_and_prompt_changes():
    keys = ["a"] * 8 + ["b"] * 8 + ["c"] * 3
    assert cons.split_groups(keys, 8) == [list(range(8)), list(range(8, 16)), [16, 17, 18]]
    # a block that changes prompt in the middle (e.g. a batch assembled from smaller pieces)
    keys2 = ["a"] * 5 + ["b"] * 3 + ["c"] * 8
    assert cons.split_groups(keys2, 8) == [[0, 1, 2, 3, 4], [5, 6, 7], list(range(8, 16))]
    assert cons.split_groups(["a"] * 4, None) == [[0], [1], [2], [3]] and cons.split_groups(["a"] * 4, 1) == [[0], [1], [2], [3]]
    assert cons.split_groups([], 8) == []


def test_cost_of_a_step_is_small():
    rng = random.Random(3)
    texts = [" ".join(rng.choice(WORDS) + "".join(rng.choice("abcdefgh") for _ in range(rng.randint(1, 5))) for _ in range(150)) for _ in range(8)]
    t = time.time()
    for _ in range(10):                                           # 10 groups of 8 long (~1,100 character) completions
        cons.group_consensus([cons.Profile(x) for x in texts])
    assert (time.time() - t) / 10 < 0.5                           # measured ~10 ms per group; a generous bound for slow CI machines


# ------------------------------------------------------------------------------------------------ guards for the failure modes
def _para(seed, n=40):
    rng = random.Random(seed)
    vocab = ["".join(rng.choice("abdefghiklmnoprstuwy") for _ in range(rng.randint(3, 8))) for _ in range(300)]
    return " ".join(rng.choice(vocab) for _ in range(n))


def test_self_repetition_guard_flags_near_loops_not_human_text():
    human = " ".join(_para(i, 12) for i in range(8))                          # varied text
    assert not guards.is_self_repetitive(human)
    para = _para(1)
    loop = " ".join([para, para, para[:120]])                                 # a paragraph repeated inside the answer: enough distinct words to pass the plain guard
    assert guards.is_self_repetitive(loop) and not guards.is_repetitive(loop)
    assert not guards.is_self_repetitive("short text repeated repeated repeated")        # too short to judge
    assert guards.repeated_ngram_fraction("a b c") == 0.0
    # human FLORES+ dev documents of 3-40 sentences have at most 0.21 repeated 4-grams (measured, 6,000 documents of 40 languages); a doubled answer has > 0.5
    assert guards.repeated_ngram_fraction(" ".join([para, para])) > 0.45


# ------------------------------------------------------------------------------------------------ the reward in the three modes
def test_qe_mode_is_bit_identical_with_and_without_the_consensus_machinery():
    texts = GOOD
    a = reward(content={}, consensus=False)(group("yor", texts), group_size=8)
    b = reward(content={}, consensus=True)(group("yor", texts), group_size=8)
    assert a.rewards == b.rewards and a.status == b.status and a.qe == b.qe
    assert all(m == "qe" for m in b.content_mode) and all(c is None for c in b.consensus_chrf)


def test_blend_mode_mixes_shaped_qe_and_consensus_value():
    sc = CountingScorer()
    r = reward(content={"kam": {"mode": "blend", "beta": 0.5}}, scorer=sc)
    out = r(group("kam", GOOD), group_size=8)
    plain = reward(content={}, consensus=False)(group("kam", GOOD), group_size=8)
    assert all(m == "blend" for m in out.content_mode) and sc.n_scored >= 8
    assert all(c is not None for c in out.consensus_chrf)
    for i in range(8):
        lo, hi = r.shaper.reward_range
        cv = cons.consensus_value(out.consensus_chrf[i], lo, hi)
        assert abs(out.rewards[i] - (0.5 * plain.rewards[i] + 0.5 * cv)) < 1e-9, i
    # the sample that agrees least with its siblings has the lowest consensus
    order = sorted(range(8), key=lambda i: out.consensus_chrf[i])
    assert out.consensus_chrf[order[0]] < out.consensus_chrf[order[-1]]


def test_consensus_mode_never_calls_the_scorer_and_uses_the_map():
    sc = CountingScorer()
    r = reward(content={"ach": {"mode": "consensus", "beta": 1.0}}, scorer=sc)
    out = r(group("ach", GOOD), group_size=8)
    assert sc.n_scored == 0 and all(q is None for q in out.qe) and all(m == "consensus" for m in out.content_mode)
    for i in range(8):
        assert abs(out.rewards[i] - cons.consensus_value(out.consensus_chrf[i], -2.0, 2.25)) < 1e-9
    assert out.stats["consensus/applied_rate"] == 1.0 and out.stats["mode/consensus_rate"] == 1.0 and "consensus/chrf/ach" in out.stats


def test_failures_neither_receive_nor_give_consensus():
    texts = list(GOOD)
    bad = group("ach", texts)
    bad[0] = Sample(ids("ach " + GOOD[0])[:-1], "ach", SRC)                          # truncated: no end of turn
    bad[1] = Sample(ids("ach " + " ".join(["ewe"] * 12)), "ach", SRC)                # word loop (degenerate)
    bad[2] = Sample(ids("ach " + SRC), "ach", SRC)                                   # copy of the English source
    r = reward(content={"ach": {"mode": "consensus"}})
    out = r(bad, group_size=8)
    assert out.status[:3] == ["no_end_of_turn", "degenerate", "copy_source"]
    assert out.rewards[:3] == [-2.0, -2.0, -2.0] and out.consensus_chrf[:3] == [None, None, None]
    # the five valid samples agree with each other only: identical to a group made of those five alone
    five = reward(content={"ach": {"mode": "consensus"}})(group("ach", texts[3:]), group_size=5)
    assert [round(x, 9) for x in out.consensus_chrf[3:]] == [round(x, 9) for x in five.consensus_chrf]


def test_fewer_than_three_valid_samples_skip_the_signal():
    texts = list(GOOD)
    smp = group("ach", texts)
    for k in range(2, 8):
        smp[k] = Sample(ids("ach " + GOOD[k])[:-1], "ach", SRC)                       # six truncated -> two valid
    out = reward(content={"ach": {"mode": "consensus"}}, consensus_skip_value=0.25)(smp, group_size=8)
    assert out.consensus_chrf[0] is None and out.consensus_chrf[1] is None and out.stats["consensus/groups_skipped"] == 1.0
    assert out.rewards[0] == 0.25 and out.rewards[1] == 0.25                           # rubric-only value
    blend = reward(content={"kam": {"mode": "blend", "beta": 0.5}})(
        [Sample(s.token_ids, "kam", SRC) for s in smp], group_size=8)
    plain = reward(content={}, consensus=False)([Sample(s.token_ids, "kam", SRC) for s in smp], group_size=8)
    assert blend.rewards == plain.rewards and blend.content_mode[0] == "qe"           # blend falls back to QE only


def test_a_collapsed_group_is_flat_and_a_majority_of_loops_is_excluded():
    same = group("ach", [GOOD[0]] * 8)
    out = reward(content={"ach": {"mode": "consensus"}})(same, group_size=8)
    assert len(set(out.rewards)) == 1 and abs(out.consensus_chrf[0] - 100.0) < 1e-9          # flat -> zero advantage downstream
    # five near-loops (an answer repeated inside itself) agree with each other, but never enter the pool
    para = _para(5)
    loops = [" ".join([para, para, para[: 100 + k]]) for k in range(5)]
    mixed = group("ach", loops + GOOD[:3])
    out2 = reward(content={"ach": {"mode": "consensus"}})(mixed, group_size=8)
    assert out2.status[:5] == ["degenerate"] * 5 and out2.rewards[:5] == [-2.0] * 5
    assert all(out2.status[5:][k] == "ok" for k in range(3)) and all(c is not None for c in out2.consensus_chrf[5:])


def test_very_short_outputs_hit_the_pooled_length_bound_in_uncalibrated_languages():
    short = ["Oja n si.", "Oja n si.", "Oja n si.", "Oja n si.", "Oja n si.", "Oja ni.", "Oja.", "Oja n."]
    long_src = SRC * 3
    out = reward(content={"ach": {"mode": "consensus"}})(group("ach", short, long_src), group_size=8)
    assert set(out.status) == {"length_ratio"}                       # < 0.4 x the source: failure, not consensus
    # a calibrated language keeps the ordinary 0.2 bound (and its own window penalty)
    sh = shaper_with({"dyu": {"mode": "blend", "beta": 0.5}})
    sh.len_window = {"dyu": (0.7, 1.4)}
    r = TranslationReward(RewardConfig(chunking="sentences", consensus=True), CountingScorer(), TOK, STOP, shaper=sh)
    ok = r(group("dyu", ["Oja n si ni agogo mejo owuro awon agbe lati abule mu ewebe wa"] * 8, SRC), group_size=8)
    assert set(ok.status) == {"ok"}


def test_penalties_and_failed_chunks_apply_in_consensus_mode_where_a_calibration_exists():
    sh = shaper_with({"dyu": {"mode": "consensus", "beta": 1.0}})
    sh.len_window = {"dyu": (0.7, 1.1)}
    r = TranslationReward(RewardConfig(chunking="sentences", consensus=True, length_penalty=1.0), CountingScorer(), TOK, STOP, shaper=sh)
    r0 = TranslationReward(RewardConfig(chunking="sentences", consensus=True, length_penalty=0.0), CountingScorer(), TOK, STOP, shaper=sh)
    inside = r(group("dyu", GOOD), group_size=8)
    assert inside.rewards == r0(group("dyu", GOOD), group_size=8).rewards                 # inside the human window: no penalty
    # a translation that is far too long for the human window is penalised on top of the consensus value
    longer = [t + " " + _para(k, 9) for k, t in enumerate(GOOD)]
    b, b0 = r(group("dyu", longer), group_size=8), r0(group("dyu", longer), group_size=8)
    assert all(x <= y + 1e-12 for x, y in zip(b.rewards, b0.rewards)) and sum(b.rewards) < sum(b0.rewards) - 0.02
    # two source sentences, the translation of the second is missing: a failed chunk (-2) shares the weight
    src2 = "The market opens at eight o'clock in the morning. The farmers bring fresh vegetables to sell in the village."
    drop = [t.split(",")[0] + "." for t in GOOD]
    r2 = TranslationReward(RewardConfig(chunking="sentences", consensus=True), CountingScorer(), TOK, STOP, shaper=sh)
    c = r2(group("dyu", drop, src2), group_size=8)
    assert c.stats["chunk/segments_per_sample"] >= 1.0
    assert max(c.rewards) < max(cons.consensus_value(x, -2, 2.25) for x in c.consensus_chrf) + 1e-9


def test_consensus_off_gives_the_rubric_only_value_and_greedy_evaluation_has_no_groups():
    out = reward(content={"ach": {"mode": "consensus"}}, consensus=False, consensus_skip_value=0.0)(group("ach", GOOD), group_size=8)
    assert all(abs(x) < 1e-9 for x in out.rewards) and all(c is None for c in out.consensus_chrf)
    ev = reward(content={"ach": {"mode": "consensus"}, "kam": {"mode": "blend", "beta": 0.5}})
    one = ev([Sample(ids("ach " + GOOD[0]), "ach", SRC), Sample(ids("kam " + GOOD[0]), "kam", SRC)], group_size=1)
    assert one.rewards[0] == 0.0 and one.content_mode == ["consensus", "qe"] and one.consensus_chrf == [None, None]


def test_ragged_batch_and_mixed_languages_in_one_call():
    smp = group("ach", GOOD) + group("yor", GOOD) + group("kam", GOOD[:5])
    out = reward(content={"ach": {"mode": "consensus"}, "kam": {"mode": "blend", "beta": 0.5}})(smp, group_size=8)
    assert out.content_mode[:8] == ["consensus"] * 8 and out.content_mode[8:16] == ["qe"] * 8 and out.content_mode[16:] == ["blend"] * 5
    assert all(c is None for c in out.consensus_chrf[8:16]) and all(c is not None for c in out.consensus_chrf[16:])
    assert len(out.rewards) == 21 and abs(out.stats["mode/qe_rate"] - 8 / 21) < 1e-9


def test_content_mode_overrides_and_validation():
    r = reward(content={"ach": {"mode": "consensus"}}, content_modes={"ach": "qe", "kam": "blend"}, consensus_beta={"kam": 0.25})
    assert r.content_spec("ach") == ("qe", 0.0) and r.content_spec("kam") == ("blend", 0.25) and r.content_spec("yor") == ("qe", 0.0)
    bad = reward(content={"ach": {"mode": "magic"}})
    with pytest.raises(ValueError):
        bad.content_spec("ach")


def test_the_training_loop_runs_with_blend_and_consensus_languages(tmp_path):
    """End to end on the CPU smoke model: the trainer passes the group size, the rewards stay inside the asserted range, the metrics carry the consensus statistics."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import json

    import gemma_mt.rl.train_rl as T

    T.run_smoke(str(tmp_path), sft_steps=40, overrides=dict(
        steps=3, eval_every=0, reward={"max_comet_tokens": 510, "consensus": True, "content_modes": {"yor": "blend", "hau": "consensus"},
                                       "consensus_skip_value": 0.0, "chunking": "sentences"}))
    rows = [json.loads(l) for l in open(tmp_path / "rl" / "metrics.jsonl") if l.strip()]
    steps = [r for r in rows if "reward/mode/consensus_rate" in r]
    assert len(steps) == 3 and all("reward/consensus/groups" in r and "reward/consensus/seconds" in r for r in steps)
    assert any(r["reward/mode/consensus_rate"] > 0 and r["reward/mode/blend_rate"] >= 0 for r in steps)
