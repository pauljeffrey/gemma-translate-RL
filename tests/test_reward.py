# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import random

import pytest

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import guards
from gemma_mt.rl.comet import FakeScorer, batched_in_length_order
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
from gemma_mt.rl.text import cut_at_stop, decode_completion

TOK = FakeGemmaTokenizer()
EOT, EOS = SPECIALS["<end_of_turn>"], SPECIALS["<eos>"]
STOP = [EOT, EOS]
SRC = "The market opens at eight o'clock and the farmers bring fresh vegetables."


def ids(text, end=True, extra_after=""):
    out = TOK.encode(text, add_special_tokens=False)
    if end:
        out = out + [EOT]
    if extra_after:
        out = out + TOK.encode(extra_after, add_special_tokens=False)
    return out


class ConstScorer:
    def __init__(self, value=0.7):
        self.value, self.calls = value, []

    def __call__(self, s, m, r=None):
        self.calls.append((list(s), list(m)))
        return [self.value] * len(s)


def make(cfg=None, scorer=None, **kw):
    return TranslationReward(cfg or RewardConfig(**kw), scorer or FakeScorer(), TOK, STOP)


GOOD = "yor Oja n si ni agogo mejo, awon agbe si mu ewebe tutu wa."


def test_cut_at_stop_and_text_never_contains_end_of_turn():
    row = ids("yor hello") + ids("junk after the first stop")
    kept, fin = cut_at_stop(row, STOP)
    assert fin and kept[-1] == EOT and kept == ids("yor hello")
    d = decode_completion(TOK, row, STOP)
    assert d.text == "yor hello" and "<end_of_turn>" not in d.text and "junk" not in d.text
    d2 = decode_completion(TOK, ids("yor no stop", end=False), STOP)
    assert not d2.finished


def test_special_tokens_are_stripped_not_scored():
    row = TOK.encode("yor Oja ", add_special_tokens=False) + [SPECIALS["<bos>"], SPECIALS["<start_of_turn>"]] + \
        TOK.encode("n si ni<end_of_turn>agogo", add_special_tokens=False)
    row = row[:-len(TOK.encode("agogo", add_special_tokens=False))]       # ends with the textual end-of-turn marker
    d = decode_completion(TOK, row + [EOT], STOP)
    assert "<" not in d.text and d.special_ids_stripped


def test_good_sample_gets_shaped_qe_and_reward_sees_only_the_translation():
    sc = ConstScorer(0.80)
    r = make(RewardConfig(eos_discipline=False), scorer=sc)       # junk after the end token is a failure under D-26 (see below)
    out = r([Sample(ids(GOOD) + ids("ignored"), "yor", SRC)])
    assert out.status == ["ok"]
    assert sc.calls[0][1] == [GOOD[len("yor "):]]            # no tag, no <end_of_turn>, nothing after it
    assert sc.calls[0][0] == [SRC]
    # 0.8 is above the default t_great (0.75): repo preset -> 2 + slope towards 2.25
    assert 2.0 < out.rewards[0] <= 2.25


def test_all_failure_classes_get_the_floor_and_are_not_scored():
    sc = ConstScorer(0.9)
    r = make(scorer=sc)
    cases = {
        "no_end_of_turn": ids(GOOD, end=False),
        "bad_prefix": ids("hau Oja n si ni agogo mejo, awon agbe si mu ewebe tutu wa."),
        "bad_prefix2": ids("Oja n si ni agogo mejo, awon agbe si mu ewebe tutu wa."),
        "empty": ids("yor "),
        "copy_source": ids("yor " + SRC),
        "degenerate": ids("yor " + "oja " * 30),
        "length_ratio": ids("yor " + " ".join(f"w{i}" for i in range(200))),
    }
    names = list(cases)
    out = r([Sample(cases[n], "yor", SRC) for n in names])
    for n, st, rew in zip(names, out.status, out.rewards):
        assert st != "ok", n
        assert rew == -2.0, n
    assert out.status[0] == "no_end_of_turn" and out.status[3] == "empty" and out.status[4] == "copy_source"
    assert out.status[5] == "degenerate" and out.status[6] == "length_ratio"
    assert sc.calls == [], "failures must never reach the scorer"
    assert out.stats["gate_rate"] == 0.0 and out.stats["fail/no_end_of_turn"] == pytest.approx(1 / 7)


def test_prefix_policies():
    sc = ConstScorer(0.7)
    plain = ids("Oja n si ni agogo mejo, awon agbe si mu ewebe tutu wa.")
    opt = make(RewardConfig(prefix_policy="optional"), sc)
    assert opt([Sample(plain, "yor", SRC)]).status == ["ok"]
    forbid = make(RewardConfig(prefix_policy="forbid"), sc)
    assert forbid([Sample(ids(GOOD), "yor", SRC)]).status == ["bad_prefix"]
    assert forbid([Sample(plain, "yor", SRC)]).status == ["ok"]


def test_penalty_may_not_exceed_the_floor():
    with pytest.raises(ValueError):
        make(RewardConfig(penalty=-1.0))                    # -1 is above the floor -2: a failure could outscore QE
    r = make(RewardConfig(penalty=-3.0))
    assert r.reward_range[0] == -3.0


def test_owner_note_preset_caps_at_plus_one():
    r = make(RewardConfig(preset="owner_note"), ConstScorer(0.99))
    out = r([Sample(ids(GOOD), "yor", SRC)])
    assert out.rewards == [1.0] and r.reward_range == (-2.0, 1.0)


def test_per_language_calibration_changes_reward_and_weights(tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"yor": {"anchors": [0.60, 0.75, 0.85], "usable": True, "reliability": 1.0},
                               "hau": {"anchors": [0.46, 0.55, 0.65], "usable": True, "reliability": 0.4},
                               "wol": {"anchors": [0.46, 0.55, 0.65], "usable": False}}))
    r = make(RewardConfig(calibration_path=str(cal)), ConstScorer(0.62))
    src = SRC
    out = r([Sample(ids(GOOD), "yor", src), Sample(ids(GOOD.replace("yor", "hau", 1)), "hau", src),
             Sample(ids(GOOD.replace("yor", "wol", 1)), "wol", src)])
    assert out.rewards[0] < out.rewards[1]
    assert out.weights == [1.0, 0.4, 0.0]


def test_plumbing_scores_stay_aligned_after_shuffling_and_length_sorting():
    rng = random.Random(0)
    rows = []
    for i in range(40):
        src = f"Sentence {i} " + "w" * rng.randint(1, 60)
        tr = "x" * rng.randint(5, 80) + f" #{i}"
        rows.append((src, tr))
    rng.shuffle(rows)

    class Tagged:
        """Score = number encoded in the translation, so a misaligned pair is detectable."""
        batch_size = 7

        def __call__(self, srcs, mts, refs=None):
            def run(batch):
                return [float(m.split("#")[-1]) / 100.0 + 0.0 * len(s) for s, m in batch]
            return batched_in_length_order(list(zip(srcs, mts)), lambda p: len(p[0]) + len(p[1]), run, self.batch_size)

    r = make(RewardConfig(min_len_ratio=0.0, max_len_ratio=1e9, repetition_guard=False, calibration_path=None),
             Tagged())
    samples = [Sample(ids("yor " + tr), "yor", src) for src, tr in rows]
    out = r(samples)
    for (src, tr), q in zip(rows, out.qe):
        assert q == pytest.approx(float(tr.split("#")[-1]) / 100.0), "score landed on the wrong completion"


def test_batched_in_length_order_restores_input_order():
    items = ["ccc", "a", "bb", "dddd", "ee"]
    out = batched_in_length_order(items, len, lambda b: [len(x) * 10 + ord(x[0]) % 10 for x in b], 2)
    assert out == [len(x) * 10 + ord(x[0]) % 10 for x in items]
    with pytest.raises(RuntimeError):
        batched_in_length_order(items, len, lambda b: [0.0], 2)


def test_guards():
    assert guards.is_copy_of_source("Hello, world!", "hello world")
    assert not guards.is_copy_of_source(SRC, GOOD)
    assert guards.is_repetitive("a b c d e f " * 10)
    assert guards.is_repetitive("go go go go go go go")
    assert not guards.is_repetitive("This is a perfectly ordinary sentence with different words in it.")
    assert guards.length_ratio_bad("a" * 100, "b" * 5) and guards.length_ratio_bad("a" * 10, "b" * 200)
    assert not guards.length_ratio_bad("a" * 100, "b" * 90)


def test_lid_guard_only_for_languages_with_labels_and_long_enough_text():
    calls = []

    def lid(texts):
        calls.append(list(texts))
        return ["eng_Latn"] * len(texts)

    cfg = RewardConfig(lid_enabled=True, lid_min_chars=10)
    r = TranslationReward(cfg, ConstScorer(0.7), TOK, STOP, lid=lid)
    out = r([Sample(ids(GOOD), "yor", SRC),
             Sample(ids("amh " + GOOD[4:]), "amh", SRC) if False else Sample(ids(GOOD.replace("yor", "pcm", 1)), "pcm", SRC)])
    assert out.status == ["wrong_language", "wrong_language"]
    assert out.rewards == [-2.0, -2.0]
    off = TranslationReward(RewardConfig(lid_enabled=False), ConstScorer(0.7), TOK, STOP, lid=lid)
    assert off([Sample(ids(GOOD), "yor", SRC)]).status == ["ok"]


# ------------------------------------------------------------------ chunked (sentence-level) scoring, D-23
SRC2 = "The market opens at eight o'clock. The farmers bring fresh vegetables. Everyone is happy."
OUT3 = "yor Oja n si ni agogo mejo. Awon agbe mu ewebe tutu wa. Inu gbogbo eniyan dun."


class ByTextScorer:
    """Scores a (source, mt) segment by a lookup on the mt text; records every call."""
    def __init__(self, table, default=0.5):
        self.table, self.default, self.calls = table, default, []

    def __call__(self, s, m, r=None):
        self.calls.append((list(s), list(m)))
        return [next((v for k, v in self.table.items() if k in x), self.default) for x in m]


def test_chunked_reward_is_mean_of_shaped_sentence_scores_without_any_newline():
    sc = ByTextScorer({"Oja": 0.9, "Awon": 0.5, "Inu": 0.7})
    r = make(RewardConfig(chunking="sentences", chunk_agg="mean"), scorer=sc)
    out = r([Sample(ids(OUT3), "yor", SRC2)])
    assert len(sc.calls) == 1 and len(sc.calls[0][0]) == 3                   # one scorer call, three segments
    assert sc.calls[0][0][0].startswith("The market") and sc.calls[0][1][0].startswith("Oja")
    ref = make(RewardConfig())
    single = [ref.shaper(c, "yor") for c in (0.9, 0.5, 0.7)]
    assert out.rewards[0] == pytest.approx(sum(single) / 3)
    assert out.qe[0] == pytest.approx((0.9 + 0.5 + 0.7) / 3)
    assert out.stats["chunk/aligned_rate"] == 1.0 and out.stats["chunk/segments_per_sample"] == 3.0


def test_chunked_reward_falls_back_to_whole_pair_when_sentence_counts_differ():
    sc = ConstScorer(0.7)
    r = make(RewardConfig(chunking="sentences"), scorer=sc)
    merged = "yor Oja n si ni agogo mejo, awon agbe mu ewebe tutu wa, inu gbogbo eniyan dun."   # 1 sentence for 3
    out = r([Sample(ids(merged), "yor", SRC2)])
    assert len(sc.calls[0][0]) == 1 and sc.calls[0][0][0] == SRC2                  # whole source, never a shifted pairing
    assert out.stats["chunk/count_mismatch_rate"] == 1.0 and out.stats["chunk/aligned_rate"] == 0.0
    assert out.rewards[0] == pytest.approx(make(scorer=ConstScorer(0.7))([Sample(ids(merged), "yor", SRC2)]).rewards[0])


def test_chunked_reward_copied_sentence_is_a_failed_chunk_and_single_sentence_input_is_whole_text():
    sc = ByTextScorer({"Oja": 0.9, "Inu": 0.9}, default=0.9)
    r = make(RewardConfig(chunking="sentences", chunk_agg="mean"), scorer=sc)
    copied = "yor Oja n si ni agogo mejo. The farmers bring fresh vegetables. Inu gbogbo eniyan dun."
    out = r([Sample(ids(copied), "yor", SRC2)])
    assert len(sc.calls[0][0]) == 2                                                # the copied sentence is not scored
    good = make(RewardConfig(chunking="sentences"), scorer=ByTextScorer({}, 0.9))([Sample(ids(OUT3), "yor", SRC2)]).rewards[0]
    assert out.rewards[0] < good
    lo = r.penalty
    assert out.rewards[0] == pytest.approx((2 * float(r.shaper(0.9, "yor")) + lo) / 3)
    one = make(RewardConfig(chunking="sentences"), scorer=sc)
    n_before = len(sc.calls)
    one([Sample(ids(GOOD), "yor", SRC)])                                           # a single sentence: whole text
    assert len(sc.calls) == n_before + 1 and len(sc.calls[-1][0]) == 1


def test_chunk_agg_source_length_weights_by_source_characters():
    table = {"Oja": 0.9, "Awon": 0.2}
    src = "Short one. The farmers bring fresh vegetables to the market every single morning."
    out_txt = "yor Oja. Awon agbe mu ewebe tutu wa si oja ni gbogbo owuro."
    base = make(RewardConfig(chunking="sentences", chunk_agg="mean"), scorer=ByTextScorer(table))([Sample(ids(out_txt), "yor", src)])
    wtd = make(RewardConfig(chunking="sentences", chunk_agg="source_length"), scorer=ByTextScorer(table))([Sample(ids(out_txt), "yor", src)])
    assert wtd.rewards[0] < base.rewards[0]                                        # the long (bad) sentence counts more


def test_omitted_source_sentence_is_a_failed_chunk():
    long_src = ("The market opens at eight o'clock in the morning. The farmers bring fresh vegetables from the village. "
                "Children sell oranges near the gate. Everyone is happy to see the new bridge.")
    good = ("yor Oja n si ni agogo mejo owuro. Awon agbe mu ewebe tutu wa lati abule. Awon omode n ta osan leba geeti. "
            "Inu gbogbo eniyan dun lati ri afara tuntun.")
    short = "yor Oja n si ni agogo mejo owuro. Awon agbe mu ewebe tutu wa lati abule. Inu gbogbo eniyan dun lati ri afara tuntun."
    r = make(RewardConfig(chunking="sentences"), scorer=ConstScorer(0.9))
    a = r([Sample(ids(good), "yor", long_src)])
    b = r([Sample(ids(short), "yor", long_src)])
    assert a.rewards[0] > b.rewards[0], "dropping a sentence must lower the reward"


def test_hallucinated_extra_sentence_is_punished_in_chunked_mode():
    src = "The farmers bring fresh vegetables to the market every single morning."
    ok = "yor Awon agbe mu ewebe tutu wa si oja ni gbogbo owuro."
    extra = ok + " Gbogbo eniyan ni won pe lati wa si ipade nla ni ojo Satide to n bo ni ilu naa ati si ibi ayeye."
    a = make(RewardConfig(chunking="sentences"), scorer=ConstScorer(0.9))([Sample(ids(ok), "yor", src)]).rewards[0]
    b = make(RewardConfig(chunking="sentences"), scorer=ConstScorer(0.9))([Sample(ids(extra), "yor", src)]).rewards[0]
    off = make(RewardConfig(chunking="sentences", extra_output_penalty=False), scorer=ConstScorer(0.9))([Sample(ids(extra), "yor", src)]).rewards[0]
    assert b < a - 0.3, "appended unrelated text must lower the reward"
    assert off == pytest.approx(a), "with the penalty off the extra text is invisible (the measured blind spot)"
    # a normal, proportional translation of a two-sentence source is untouched
    two_src = "The market opens at eight o'clock in the morning. The farmers bring fresh vegetables from the village."
    two_ok = "yor Oja n si ni agogo mejo owuro. Awon agbe mu ewebe tutu wa lati abule."
    x = make(RewardConfig(chunking="sentences"), scorer=ConstScorer(0.9))([Sample(ids(two_ok), "yor", two_src)]).rewards[0]
    y = make(RewardConfig(chunking="sentences", extra_output_penalty=False), scorer=ConstScorer(0.9))([Sample(ids(two_ok), "yor", two_src)]).rewards[0]
    assert x == pytest.approx(y)


def test_length_window_penalty_uses_the_calibrated_human_window(tmp_path):
    import json as _json
    cal = tmp_path / "cal.json"
    cal.write_text(_json.dumps({"yor": {"anchors": [0.46, 0.55, 0.65], "usable": True, "reliability": 0.9,
                                        "len_ratio": {"p05": 0.8, "p50": 1.0, "p95": 1.3}}}))
    src = "The farmers bring fresh vegetables to the market every single morning."
    normal = "yor Awon agbe mu ewebe tutu wa si oja ni gbogbo owuro."
    long = normal + " Ati pe awon omode ti won n ta osan leba geeti n gba owo lowo awon onibara ti won wa si oja."
    r0 = make(RewardConfig(calibration_path=str(cal), length_penalty=0.0), scorer=ConstScorer(0.9))
    r1 = make(RewardConfig(calibration_path=str(cal), length_penalty=2.0), scorer=ConstScorer(0.9))
    a0, a1 = (r.__call__([Sample(ids(normal), "yor", src)]).rewards[0] for r in (r0, r1))
    b0, b1 = (r.__call__([Sample(ids(long), "yor", src)]).rewards[0] for r in (r0, r1))
    assert a0 == pytest.approx(a1), "inside the human window nothing changes"
    assert b1 < b0 - 0.5, "far outside the window the reward drops"
    assert b1 >= r1.penalty - 1e-9, "never below the failure floor"


# ------------------------------------------------------------------ end-of-turn discipline, D-26
PAD = SPECIALS["<pad>"]


def test_eos_truncated_completion_is_the_old_no_end_of_turn_failure():
    sc = ConstScorer(0.9)
    out = make(scorer=sc)([Sample(ids(GOOD, end=False), "yor", SRC)])
    assert out.status == ["no_end_of_turn"] and out.rewards == [-2.0] and sc.calls == []


def test_eos_text_after_the_first_end_token_is_a_failure_even_though_the_text_scored_would_be_clean():
    sc = ConstScorer(0.9)
    r = make(scorer=sc)
    junk = ids(GOOD) + ids("more words after the end token")                       # a second answer, no padding
    turn = ids(GOOD) + [SPECIALS["<start_of_turn>"]] + TOK.encode("user\nagain", add_special_tokens=False)
    padded = ids(GOOD) + [PAD] * 5                                                    # padding is fine
    both_stops = ids(GOOD) + [EOS, PAD]                                               # a second stop id is not rubbish
    out = r([Sample(junk, "yor", SRC), Sample(turn, "yor", SRC), Sample(padded, "yor", SRC), Sample(both_stops, "yor", SRC)])
    assert out.status == ["post_eos_rubbish", "post_eos_rubbish", "ok", "ok"]
    assert out.rewards[:2] == [-2.0, -2.0] and out.rewards[2] > 0
    assert len(sc.calls) == 1 and len(sc.calls[0][0]) == 2, "failures never reach the QE model"
    assert out.stats["fail/post_eos_rubbish"] == 0.5


def test_eos_tokens_cut_by_the_sampler_are_reported_through_the_sample():
    sc = ConstScorer(0.9)
    r = make(scorer=sc)
    out = r([Sample(ids(GOOD), "yor", SRC, post_stop_tokens=3), Sample(ids(GOOD), "yor", SRC, post_stop_tokens=0)])
    assert out.status == ["post_eos_rubbish", "ok"]


def test_eos_scored_text_is_only_the_part_before_the_first_end_token():
    sc = ConstScorer(0.9)
    make(RewardConfig(eos_discipline=False), scorer=sc)([Sample(ids(GOOD) + ids("tail"), "yor", SRC)])
    assert sc.calls[0][1] == [GOOD[len("yor "):]]


def test_eos_dangling_turn_marker_or_role_line_before_the_end_token_is_a_failure():
    sc = ConstScorer(0.9)
    r = make(scorer=sc)
    marker = TOK.encode("yor Oja n si ni agogo mejo.", add_special_tokens=False) + [SPECIALS["<start_of_turn>"]] + \
        TOK.encode("model\nOja", add_special_tokens=False) + [EOT]
    role = ids("yor Oja n si ni agogo mejo, awon agbe si mu ewebe tutu wa.\nmodel")
    role_u = ids("yor Oja n si ni agogo mejo, awon agbe si mu ewebe tutu wa.\nuser:")
    inside = ids("yor Awon agbe ni model ati user ninu ile itaja, won si n ta won.")   # role words inside a sentence are fine
    out = r([Sample(marker, "yor", SRC), Sample(role, "yor", SRC), Sample(role_u, "yor", SRC), Sample(inside, "yor", SRC)])
    assert out.status == ["dangling_marker", "dangling_marker", "dangling_marker", "ok"]
    assert out.rewards[:3] == [-2.0] * 3 and len(sc.calls[0][0]) == 1
    # a source that itself ends with a role line is exempt from the bare-role-line check
    src = "The meeting notes.\nuser"
    ok = make(scorer=ConstScorer(0.8))([Sample(ids("yor Awon akiyesi ipade.\nuser"), "yor", src)])
    assert ok.status == ["ok"]


def test_eos_discipline_switch_and_penalty_values():
    junk = ids(GOOD) + ids("tail")
    off = make(RewardConfig(eos_discipline=False), scorer=ConstScorer(0.9))([Sample(junk, "yor", SRC)])
    assert off.status == ["ok"]
    r = make(RewardConfig(eos_post_end_penalty=-3.0, eos_dangling_penalty=-2.5, eos_truncated_penalty=-4.0), scorer=ConstScorer(0.9))
    out = r([Sample(junk, "yor", SRC), Sample(ids(GOOD, end=False), "yor", SRC),
             Sample(TOK.encode("yor x", add_special_tokens=False) + [SPECIALS["<start_of_turn>"], EOT], "yor", SRC)])
    assert out.status == ["post_eos_rubbish", "no_end_of_turn", "dangling_marker"]
    assert out.rewards == [-3.0, -4.0, -2.5] and r.reward_range[0] == -4.0
    for kw in ("eos_post_end_penalty", "eos_dangling_penalty", "eos_truncated_penalty"):
        with pytest.raises(ValueError):
            make(RewardConfig(**{kw: -1.0}))                                         # above the floor: a failure could outscore QE


def test_eos_failure_classes_are_listed_in_the_diagnostics():
    from gemma_mt.rl.reward import EOS_FAILURES, FAILURES
    assert set(EOS_FAILURES) <= set(FAILURES)
    out = make(scorer=ConstScorer(0.9))([Sample(ids(GOOD), "yor", SRC)])
    for f in EOS_FAILURES:
        assert out.stats[f"fail/{f}"] == 0.0


def test_pad_id_is_taken_from_the_tokenizer_or_given_explicitly():
    r = TranslationReward(RewardConfig(), ConstScorer(0.9), TOK, STOP)
    assert r.pad_ids == [PAD]
    r2 = TranslationReward(RewardConfig(), ConstScorer(0.9), TOK, STOP, pad_id=7)
    out = r2([Sample(ids(GOOD) + [7, 7], "yor", SRC), Sample(ids(GOOD) + [PAD], "yor", SRC)])
    assert out.status == ["ok", "post_eos_rubbish"]
