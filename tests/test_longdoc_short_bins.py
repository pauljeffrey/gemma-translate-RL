# SPDX-License-Identifier: AGPL-3.0-or-later
"""385M plan (owner 2026-11-02): the long-document reward (longdoc_v2) is applied to SHORT sources too (longdoc_min_src_words = 1).
What must hold: a normal short translation is rewarded exactly as on the plain path; loops / duplicated sentences / a looping chunk are caught; nothing divides by zero for 0-3 chunks;
the `longdoc_compare_v1` diagnostic logs the difference. CPU only, fake scorer."""
import math

import pytest

from gemma_mt.rl import longdoc as LD
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward

from test_longdoc_reward_v2 import SentenceScorer, STOP, TOK, ids  # noqa: E402  (the harness of the long-document tests)

SRC = "The farmers sell vegetables at the market every morning. They buy clothes for their children at the end of the month. The rain came late this year."
OUT = "Awon agbe n ta ewebe ni oja ni gbogbo owuro. Won n ra aso fun awon omo won ni opin osu. Ojo de pelu ni odun yii."


def make(**kw):
    cfg = RewardConfig(chunking="sentences", omission_detector=False, eos_discipline=False, keep_chunk_rewards=True, **kw)
    return TranslationReward(cfg, SentenceScorer(), TOK, STOP)


def sample(out, src=SRC, end=True):
    return Sample(ids("yor " + out, end), "yor", src)


def test_a_normal_short_translation_is_rewarded_as_on_the_plain_path():
    plain = make()
    short = make(longdoc_v2=True, longdoc_min_src_words=1, longdoc_compare_v1=True)
    for src, out in ((SRC, OUT), (SRC.split(". ")[0] + ".", OUT.split(". ")[0] + "."), ("Thank you very much.", "O ku ise.")):
        a, b = plain([sample(out, src)]), short([sample(out, src)])
        assert a.status == b.status == ["ok"], (a.status, b.status)
        assert abs(a.rewards[0] - b.rewards[0]) < 1e-9, (src, a.rewards, b.rewards)
        assert b.longdoc[0]["severity"] == 0.0 and b.longdoc[0]["n_items"] >= 1


def test_loops_and_duplicates_are_still_failures_on_short_outputs():
    short = make(longdoc_v2=True, longdoc_min_src_words=1)
    plain = make()
    bad = {"word loop": " ".join(["muna"] * 15), "cycle": " ".join(["ina jin dadi"] * 8), "duplicated sentences": " ".join([OUT.split(". ")[0] + "."] * 4)}
    for name, out in bad.items():
        r = short([sample(out)])
        assert r.rewards[0] <= plain([sample(OUT)]).rewards[0] - 1.0, (name, r.rewards, r.status)
        assert r.rewards[0] >= short.penalty - 1e-9


def test_zero_to_three_chunks_do_not_degenerate():
    pen = -2.0
    cal = LD.LongdocCalibration.from_file("calibration/longdoc_v2.json")
    for items, seg in (([], {}), ([(None, 1.0)], {}), ([(0, 1.0)], {0: 1.2}), ([(0, 1.0), (None, 1.0)], {0: 1.5}), ([(0, 3.0), (1, 1.0), (2, 2.0)], {0: 1.0, 1: -0.5, 2: 1.5})):
        mean = LD.aggregate_tail(items, seg, pen, mode="mean")
        tail = LD.aggregate_tail(items, seg, pen, mode="cvar_excess", alpha=1.0, q=0.25, gap0=cal.gap0(30))
        assert math.isfinite(tail) and abs(tail - mean) < 1e-12                    # fewer than 8 chunks: the plain mean
        cov = LD.coverage_severity(items, 0.5)
        out = LD.apply_severity(tail, LD.combine_severity(0.0, cov), pen)
        assert math.isfinite(out) and pen - 1e-9 <= out <= max(tail, pen) + 1e-9
    assert cal.bucket(30) == 0 and cal.bucket(100) == 1


def test_a_partially_failed_short_output_pays_the_coverage_term():
    plain, short = make(), make(longdoc_v2=True, longdoc_min_src_words=1)
    out = OUT.split(". ")[0] + "."                                  # the output covers one of the three source sentences
    a, b = plain([sample(out)]), short([sample(out)])
    assert b.rewards[0] <= a.rewards[0] + 1e-9


def test_the_diagnostic_logs_the_difference_next_to_the_reward():
    short = make(longdoc_v2=True, longdoc_min_src_words=1, longdoc_compare_v1=True)
    res = short([sample(OUT), sample(" ".join(["muna"] * 15))])
    st = res.stats
    for k in ("longdoc/delta_reward_short", "longdoc/delta_nonzero_rate_short", "longdoc/v1_reject_rate", "longdoc/v1_only_reject_rate", "longdoc/ld_only_reject_rate"):
        assert k in st, k
    assert st["longdoc/long_rate"] == 1.0 and 0.0 <= st["longdoc/v1_reject_rate"] <= 1.0
    off = make(longdoc_v2=True, longdoc_min_src_words=1)([sample(OUT)])
    assert not any(k.startswith("longdoc/delta") or k.startswith("longdoc/v1") for k in off.stats)


def test_the_default_threshold_keeps_short_sources_on_the_plain_path():
    r = make(longdoc_v2=True)([sample(OUT)])
    assert r.stats["longdoc/long_rate"] == 0.0 and r.longdoc[0] is None
