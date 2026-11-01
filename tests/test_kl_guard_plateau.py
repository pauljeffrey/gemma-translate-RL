# SPDX-License-Identifier: AGPL-3.0-or-later
"""2026-10-28 (clean rerun, agent T2): KL-guard persistence + optional recovery, the plateau / saturation stop, the greedy-member monitoring in every stage.

Pure-python parts (KLGuard, plateau_slope_test, PlateauStop, RunControls additions) run without torch; the trainer tests use the CPU smoke."""
import json
import math
import os
import random

import pytest

from gemma_mt.rl.run_controls import KLGuard, PlateauStop, RunControls, plateau_slope_test


# ------------------------------------------------------------------------------------------------------------------ KL guard
def _legacy(kls, thr=0.15, factor=0.5, min_scale=0.125, window=5, cooldown=10):
    """The guard exactly as it was inlined in train_rl.py before 2026-10-28 (lines 878-887)."""
    hist, lr_scale, last_cut, trace = [], 1.0, -10 ** 9, []
    for i, kl in enumerate(kls):
        step = i + 1
        hist = (hist + [kl])[-max(window, 1):]
        med = sorted(hist)[len(hist) // 2]
        if len(hist) >= window and med > thr and step - last_cut >= cooldown and lr_scale > min_scale + 1e-12:
            lr_scale = max(lr_scale * factor, min_scale)
            last_cut = step
        trace.append(lr_scale)
    return trace


def _run(g, kls, start=1):
    trace, events = [], []
    for i, kl in enumerate(kls):
        ev = g.update(start + i, kl)
        if ev:
            events.append(ev)
        trace.append(g.lr_scale)
    return trace, events


def test_default_guard_is_the_legacy_guard_on_random_series():
    rng = random.Random(3)
    for _ in range(30):
        kls = [abs(rng.gauss(0.1, 0.08)) + (0.5 if rng.random() < 0.03 else 0.0) for _ in range(300)]
        g = KLGuard(0.15)
        assert _run(g, kls)[0] == _legacy(kls)


def test_state_round_trip_equals_an_uninterrupted_run_and_a_reset_does_not():
    rng = random.Random(5)
    kls = [abs(rng.gauss(0.14, 0.06)) for _ in range(400)]
    full, _ = _run(KLGuard(0.15), kls)
    assert min(full) < 1.0
    for cut_at in (50, 123, 240):
        a = KLGuard(0.15)
        first, _ = _run(a, kls[:cut_at])
        state = json.loads(json.dumps(a.state_dict()))                 # goes through JSON like a checkpoint dict could
        b = KLGuard(0.15)
        info = b.load_state_dict(state)
        assert info["lr_scale"] == a.lr_scale
        second, _ = _run(b, kls[cut_at:], start=cut_at + 1)
        assert first + second == full
    # the old behaviour (state lost at a restart) differs: scale back to 1.0
    a = KLGuard(0.15)
    _run(a, kls[:150])
    assert a.lr_scale < 1.0
    assert KLGuard(0.15).lr_scale == 1.0


def test_checkpoint_without_guard_state_changes_nothing_and_config_changes_are_clamped():
    g = KLGuard(0.15)
    assert g.load_state_dict(None) is None and g.load_state_dict({}) is None and g.lr_scale == 1.0
    g.load_state_dict({"lr_scale": 0.01, "kl_hist": [0.1] * 50})
    assert g.lr_scale == g.min_scale and len(g.kl_hist) == 5
    g2 = KLGuard(0.15, min_scale=0.25, max_scale=1.0)
    g2.load_state_dict({"lr_scale": 7.0})
    assert g2.lr_scale == 1.0


def test_disabled_guard_does_nothing():
    g = KLGuard(0.0)
    assert not g.enabled and g.update(1, 99.0) is None and g.lr_scale == 1.0


def test_recovery_after_a_cut_needs_a_quiet_window_never_exceeds_one_and_is_rate_limited():
    g = KLGuard(0.15, recover=True, recover_factor=1.25, recover_below=0.5, recover_window=20, cooldown=10)
    _, ev = _run(g, [0.30] * 6)                                         # median of 5 above 0.15 at step 5 -> cut
    assert [e["event"] for e in ev] == ["cut"] and g.lr_scale == 0.5 and ev[0]["step"] == 5
    # KL drops to 0.05 (< 0.075): no recovery before 20 steps after the cut have been seen
    steps = []
    for s in range(7, 140):
        e = g.update(s, 0.05)
        if e:
            steps.append((s, e["event"], round(g.lr_scale, 4)))
    assert steps[0][0] == 25 and steps[0][1] == "recover" and steps[0][2] == 0.625      # 5 + 20 = 25: the window is entirely after the cut
    gaps = [b[0] - a[0] for a, b in zip(steps, steps[1:])]
    assert all(x >= 20 for x in gaps)                                   # at most once per window / cooldown
    assert [x[2] for x in steps] == [0.625, 0.7812, 0.9766, 1.0]       # capped at 1.0 (the planned lr is the ceiling)
    assert g.lr_scale == 1.0 and g.n_recoveries == 4
    # nothing more once at 1.0
    assert all(g.update(s, 0.0) is None for s in range(140, 200))


def test_recovery_is_blocked_by_a_median_above_the_fraction_and_off_by_default():
    g = KLGuard(0.15, recover=True)
    _run(g, [0.30] * 6)
    ev = [g.update(s, 0.09) for s in range(7, 120)]                    # 0.09 > 0.5 * 0.15: not quiet enough
    assert not any(ev) and g.lr_scale == 0.5
    h = KLGuard(0.15)                                                   # recover=False: lr stays cut forever
    _run(h, [0.30] * 6)
    assert not any(h.update(s, 0.0) for s in range(7, 200)) and h.lr_scale == 0.5


def test_a_cut_after_a_recovery_is_possible_and_logged_as_a_cut():
    g = KLGuard(0.15, recover=True, cooldown=10, recover_window=20)
    _, ev = _run(g, [0.3] * 6 + [0.01] * 30)
    assert [e["event"] for e in ev] == ["cut", "recover"]
    _, ev2 = _run(g, [0.4] * 6, start=37)
    assert ev2 and ev2[0]["event"] == "cut" and ev2[0]["scale_new"] < ev2[0]["scale_old"]


# ------------------------------------------------------------------------------------------------------------------ plateau test
def _series(slope_per_step, n, sd, seed, base=0.0):
    rng = random.Random(seed)
    return [base + slope_per_step * i + rng.gauss(0, sd) for i in range(n)]


def test_slope_test_synthetic_series():
    flat = plateau_slope_test(_series(0.0, 100, 0.1, 1))
    assert flat["contains_zero"] and flat["n_blocks"] == 5 and flat["ci_low"] <= 0 <= flat["ci_high"]
    rising = plateau_slope_test(_series(0.01, 100, 0.05, 2))            # +1.0 over the window against a block-mean noise of 0.011
    assert not rising["contains_zero"] and rising["ci_low"] > 0 and abs(rising["slope"] - 0.01) < 0.004
    falling = plateau_slope_test(_series(-0.01, 100, 0.05, 3))
    assert not falling["contains_zero"] and falling["ci_high"] < 0
    const = plateau_slope_test([0.3] * 100)
    assert const["contains_zero"] and const["slope"] == 0.0
    assert plateau_slope_test([0.1] * 30)["insufficient"] and not plateau_slope_test([0.1] * 30)["contains_zero"]       # < 4 blocks: never a stop
    # aligned to the END: 110 values use the last 100
    a = plateau_slope_test(_series(0.0, 100, 0.1, 4))
    b = plateau_slope_test([9.0] * 10 + _series(0.0, 100, 0.1, 4))
    assert a["block_means"] == b["block_means"]
    # deterministic
    assert plateau_slope_test(_series(0.0, 100, 0.1, 1), seed=7) == plateau_slope_test(_series(0.0, 100, 0.1, 1), seed=7)


def test_false_stop_rate_on_flat_and_missed_stop_rate_on_rising_series_are_small():
    stop_flat = sum(plateau_slope_test(_series(0.0, 100, 0.1, s), n_boot=1000, seed=s)["contains_zero"] for s in range(60))
    stop_rise = sum(plateau_slope_test(_series(0.012, 100, 0.1, 100 + s), n_boot=1000, seed=s)["contains_zero"] for s in range(60))
    assert stop_flat >= 55                                              # a flat series is called flat (this is the point of the CI test; the CI is wide at 5 blocks)
    assert stop_rise <= 12                                              # +1.2 over the window against a per-step sd of 0.1: mostly seen as rising


def _feed(p, n, fn, start=1, fin=1.0, lr=1.0, stage="b256"):
    for s in range(start, start + n):
        p.observe(s, fn(s), fin, lr, stage)


def test_plateau_stop_fires_on_a_flat_series_and_not_on_a_rising_one():
    rng = random.Random(0)
    flat = PlateauStop(window=100, block=20, min_finished=0.99)
    _feed(flat, 300, lambda s: 0.5 + rng.gauss(0, 0.1))
    d = flat.check(300, "b256", 0, 150, True, 1.0)
    assert d["fire"] and d["blocked"] is None and "slope_per_100_steps" in d and d["ci_low_per_100_steps"] <= 0 <= d["ci_high_per_100_steps"]
    rise = PlateauStop(window=100, block=20, min_finished=0.99)
    _feed(rise, 300, lambda s: 0.5 + 0.004 * s + rng.gauss(0, 0.05))
    d = rise.check(300, "b256", 0, 150, True, 1.0)
    assert not d["fire"] and d["blocked"] == "rising" and d["ci_low_per_100_steps"] > 0


def test_plateau_stop_cannot_fire_in_the_first_min_steps_or_on_a_confounded_window():
    def flat_stop(**kw):
        p = PlateauStop(window=100, block=20, min_finished=0.99)
        _feed(p, 300, lambda s: 0.5 + 0.0 * s, **kw)
        return p

    p = flat_stop()
    assert p.check(149, "b256", 0, 150, True, 1.0)["blocked"] == "min_steps"          # even though 100 flat steps are there
    assert p.check(300, "b256", 200, 150, True, 1.0)["blocked"] == "min_steps"       # stage began at 200: only 100 steps in it
    assert p.check(300, "b256", 0, 150, False, 1.0)["blocked"] == "not_last_stage"
    assert p.check(300, "b256", 0, 150, True, 0.95)["blocked"] == "finished"          # the greedy evaluation does not finish yet
    # a guard cut / recovery inside the window (or right before it) blocks; one well before it does not
    for ev_step in (201, 250, 299, 300, 200):
        p = flat_stop()
        p.note_guard_event(ev_step)
        assert p.check(300, "b256", 0, 150, True, 1.0)["blocked"] == "guard_event", ev_step
    p = flat_stop()
    p.note_guard_event(150)
    assert p.check(300, "b256", 0, 150, True, 1.0)["fire"]
    # learning-rate multiplier changing inside the window (operator file, wind-down) or low
    p = PlateauStop(window=100, block=20)
    _feed(p, 250, lambda s: 0.5, lr=1.0)
    _feed(p, 50, lambda s: 0.5, start=251, lr=0.75)
    assert p.check(300, "b256", 0, 150, True, 1.0)["blocked"] == "lr_changed"
    p = PlateauStop(window=100, block=20, min_lr_scale=0.5)
    _feed(p, 300, lambda s: 0.5, lr=0.25)
    assert p.check(300, "b256", 0, 150, True, 1.0)["blocked"] == "lr_low"
    p = PlateauStop(window=100, block=20, min_lr_scale=0.5)
    _feed(p, 300, lambda s: 0.5, lr=0.5)
    assert p.check(300, "b256", 0, 150, True, 1.0)["fire"]
    # training finished rate below the limit, a hole in the history, history of another stage
    p = PlateauStop(window=100, block=20, min_finished=0.99)
    _feed(p, 300, lambda s: 0.5, fin=0.97)
    assert p.check(300, "b256", 0, 150, True, 1.0)["blocked"] == "finished"
    p = PlateauStop(window=100, block=20)
    _feed(p, 150, lambda s: 0.5)
    _feed(p, 50, lambda s: 0.5, start=251)
    assert p.check(300, "b256", 0, 150, True, 1.0)["blocked"] == "history"
    p = PlateauStop(window=100, block=20)
    _feed(p, 250, lambda s: 0.5, stage="b64")
    _feed(p, 50, lambda s: 0.5, start=251, stage="b256")
    assert p.check(300, "b256", 0, 150, True, 1.0)["blocked"] == "history"
    p = PlateauStop(window=100, block=20)
    assert p.check(300, "b256", 0, 150, True, 1.0)["blocked"] == "history"


def test_ci_high_max_blocks_a_window_that_is_too_noisy_to_exclude_a_gain():
    rng = random.Random(11)
    series = [0.5 + rng.gauss(0, 0.8) for _ in range(300)]                  # noisy flat series: CI wide
    loose = PlateauStop(window=100, block=20)
    strict = PlateauStop(window=100, block=20, ci_high_max=0.05)
    for p in (loose, strict):
        for i, v in enumerate(series):
            p.observe(i + 1, v, 1.0, 1.0, "b256")
    d = loose.check(300, "b256", 0, 150, True, 1.0)
    assert d["fire"] and d["ci_high_per_100_steps"] > 0.05
    assert strict.check(300, "b256", 0, 150, True, 1.0)["blocked"] == "ci_too_wide"
    quiet = PlateauStop(window=100, block=20, ci_high_max=0.05)
    for i in range(300):
        quiet.observe(i + 1, 0.5 + rng.gauss(0, 0.02), 1.0, 1.0, "b256")
    assert quiet.check(300, "b256", 0, 150, True, 1.0)["fire"]


def test_plateau_state_round_trip_keeps_the_window_across_a_restart():
    rng = random.Random(1)
    p = PlateauStop(window=100, block=20)
    _feed(p, 250, lambda s: 0.5 + rng.gauss(0, 0.1))
    p.note_guard_event(10)
    q = PlateauStop(window=100, block=20)
    q.load_state_dict(json.loads(json.dumps(p.state_dict())))
    assert q.check(250, "b256", 0, 150, True, 1.0) == p.check(250, "b256", 0, 150, True, 1.0)
    _feed(p, 50, lambda s: 0.5 + rng.gauss(0, 0.1), start=251)
    assert p.check(300, "b256", 0, 150, True, 1.0)["fire"]
    p.triggered = {"x": 1}
    assert p.check(300, "b256", 0, 150, True, 1.0)["blocked"] == "already_triggered"


# ------------------------------------------------------------------------------------------------------------------ run controls
def test_request_winddown_acts_like_the_operator_file_and_the_run_ends_at_start_plus_n(tmp_path):
    msgs = []
    c = RunControls(str(tmp_path), 100, log=msgs.append)
    assert c.update(40).end_step == 100
    assert c.request_winddown(10, "plateau_stop", {"slope": 0.0})
    assert not c.request_winddown(10, "again")                          # already requested
    s = c.update(41)
    assert s.winddown and s.end_step == 51 and s.lr_mult == 1.0
    assert abs(c.update(46).lr_mult - 0.5) < 1e-9
    assert json.load(open(tmp_path / "WINDDOWN.reason.json"))["reason"] == "plateau_stop"
    rec = c.finish(51)
    assert rec["completed"] and rec["started_at_step"] == 41 and (tmp_path / "WINDDOWN.done").exists()
    # near the hard cap: max steps stays the cap
    d = tmp_path / "b"
    d.mkdir()
    c2 = RunControls(str(d), 100, log=lambda m: None)
    c2.request_winddown(50, "plateau_stop")
    assert c2.update(95).end_step == 100


def test_restore_lr_scale_writes_only_a_missing_file(tmp_path):
    c = RunControls(str(tmp_path), 100, log=lambda m: None)
    assert not c.restore_lr_scale(1.0) and not (tmp_path / "LR_SCALE").exists()
    assert c.restore_lr_scale(0.5) and c.update(0).lr_mult == 0.5
    (tmp_path / "LR_SCALE").write_text("0.75")
    assert not c.restore_lr_scale(0.5) and c.update(1).lr_mult == 0.75            # an existing (operator) file wins


# ------------------------------------------------------------------------------------------------------------------ config validation
def test_new_config_fields_are_validated_and_defaults_are_off():
    from gemma_mt.rl.train_rl import RLConfig, validate_config

    c = RLConfig()
    assert not c.kl_guard_recover and not c.plateau_stop and c.greedy_share_warn == 0.0 and c.kl_guard_restore and c.kl_guard_threshold == 0.0
    validate_config(c)
    validate_config(RLConfig(kl_guard_threshold=0.15, kl_guard_recover=True))
    for bad in (dict(kl_guard_recover=True), dict(kl_guard_threshold=0.15, kl_guard_recover=True, kl_guard_recover_factor=1.0),
                dict(kl_guard_threshold=0.15, kl_guard_recover=True, kl_guard_recover_below=1.5), dict(kl_guard_threshold=0.15, kl_guard_recover=True, kl_guard_recover_window=0),
                dict(kl_guard_threshold=0.15, kl_guard_factor=1.5), dict(kl_guard_threshold=-1.0)):
        with pytest.raises(ValueError):
            validate_config(RLConfig(**bad))
    ok = dict(plateau_stop=True, plateau_stop_min_steps=150)
    validate_config(RLConfig(**ok))
    validate_config(RLConfig(**ok, plateau_stop_min_steps_by_bin={"b64": 100, "b2048": 200}))
    for bad in (dict(run_controls=False), dict(eval_every=0), dict(plateau_stop_min_steps=50), dict(plateau_stop_window=60, plateau_stop_min_steps=150),
                dict(plateau_stop_min_steps_by_bin={"b99": 150}), dict(plateau_stop_min_steps_by_bin={"b64": 50}), dict(plateau_stop_min_finished=0.0),
                dict(plateau_stop_winddown=0), dict(plateau_stop_ci_high_max=-0.1), dict(plateau_stop_min_lr_scale=1.5), dict(plateau_stop_bootstrap=10)):
        with pytest.raises(ValueError):
            validate_config(RLConfig(**{**ok, **bad}))
    validate_config(RLConfig(group_size=8, greedy_in_group=True, greedy_share_warn=0.35))
    for bad in (dict(group_size=8, greedy_in_group=False, greedy_share_warn=0.35), dict(group_size=8, greedy_in_group=True, greedy_share_warn=1.0)):
        with pytest.raises(ValueError):
            validate_config(RLConfig(**bad))


# ------------------------------------------------------------------------------------------------------------------ loop flags
def test_loop_flags_see_a_loop_that_ran_into_the_length_cap():
    pytest.importorskip("torch")
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.rl.comet import FakeScorer
    from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
    from gemma_mt.rl.train_rl import loop_flags

    tok = FakeGemmaTokenizer()
    r = TranslationReward(RewardConfig(max_comet_tokens=510), FakeScorer(), tok, [106, 1])
    samples = [Sample([5], "yor", "The river is wide and long."), Sample([5], "yor", "The river is wide and long."), Sample([5], "yor", "The river is wide and long."), Sample([5], "yor", "The river is wide and long.")]
    looped = "odo odo odo odo odo odo odo odo odo odo odo odo"
    status = ["no_end_of_turn", "no_end_of_turn", "degenerate", "ok"]
    decoded = [looped, "odo nla gan ni lati wa si ibi ti a nlo", "x", "fine text here"]
    assert loop_flags(r, samples, status, decoded, [False, False, True, True]) == [True, False, True, False]


# ------------------------------------------------------------------------------------------------------------------ trainer (CPU smoke)
def _rows(tmp_path, langs=("yor", "hau", "ibo")):
    from gemma_mt.smoke import ENGLISH, write_jsonl

    rows = []
    for b, tok in (("b64", 12), ("b256", 100), ("b512", 300)):
        for i in range(24):
            rows.append({"english": f"{ENGLISH[i % len(ENGLISH)]} {b} {i}", "iso": langs[i % 3], "language": "", "src_afr_tokens": tok, "source_dataset": "smoke", "length_bin": b})
    return write_jsonl(str(tmp_path / "p.jsonl"), rows)


def _recs(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def test_greedy_member_scoring_runs_in_every_stage_with_the_final_run_form_and_is_monitored(tmp_path):
    """greedy_in_group + greedy_stats sampled + greedy_adv_scale 0.3 + greedy_adv_cap 3 through b64 -> b256 -> b512 -> mixed, evaluation of every bin, HF sampler."""
    pytest.importorskip("torch")
    from gemma_mt.rl.train_rl import run_smoke

    summ = run_smoke(workdir=str(tmp_path / "w"), sft_steps=150, overrides=dict(
        prompts_path=_rows(tmp_path), max_src_tokens=2048, length_bins=True, curriculum_steps={"b64": 1, "b256": 1, "b512": 1}, steps=5, eval_prompts=12, eval_min_per_bin=2, eval_every=2, save_every=0,
        prompts_per_step_by_bin={"b64": 3, "b256": 2, "b512": 2, "mixed": 3}, max_new_tokens_by_bin={"b64": 24, "b256": 24, "b512": 32, "mixed": 32},
        group_size=4, micro_batch_seqs=4, greedy_in_group=True, greedy_stats="sampled", greedy_adv_scale=0.3, greedy_adv_cap=3.0, greedy_share_warn=0.01,
        reward={"max_comet_tokens": 510}))
    assert summ["steps"] == 5 and summ["stop_reason"] == "max_steps"
    recs = _recs(tmp_path / "w" / "rl" / "metrics.jsonl")
    tr = [r for r in recs if "reward" in r]
    assert len(tr) == 5
    for r in tr:
        for k in ("greedy_finished_rate", "greedy_loop_rate", "greedy_reward_mean", "sampled_reward_mean", "greedy_adv_token_share", "adv_abs_mean_greedy", "adv_abs_mean_sampled", "lr_mult_effective"):
            assert k in r and math.isfinite(r[k]), k
        assert 0.0 <= r["greedy_adv_token_share"] <= 1.0 and 0.0 <= r["greedy_loop_rate"] <= 1.0
    ev = [r for r in recs if "eval/reward" in r]
    assert ev
    for r in ev:
        assert "eval/greedy_loop_rate" in r and "eval/greedy_finished_rate" in r
        bins = [k.split("/")[-1] for k in r if k.startswith("eval/greedy_loop_rate_bin/")]
        assert set(bins) == {"b64", "b256", "b512"}
        for b in bins:
            assert 0.0 <= r[f"eval/greedy_loop_rate_bin/{b}"] <= 1.0 and 0.0 <= r[f"eval/greedy_finished_rate_bin/{b}"] <= 1.0 and r[f"eval/greedy_n_bin/{b}"] >= 1
    # the greedy member's advantage is scaled: the logged mean |A| of the greedy member is below the sampled members' for this setting only in expectation, but the share is bounded by the cap
    assert any(r["greedy_adv_token_share"] > 0 for r in tr)


def test_greedy_member_scoring_through_the_vllm_code_path(tmp_path):
    pytest.importorskip("torch")
    from gemma_mt.rl.train_rl import run_smoke

    summ = run_smoke(workdir=str(tmp_path / "w"), sft_steps=20, vllm_fake=True, overrides=dict(
        prompts_path=_rows(tmp_path), max_src_tokens=2048, length_bins=True, curriculum_steps={"b64": 1, "b256": 1}, steps=3, eval_prompts=9, eval_min_per_bin=2, eval_every=3, save_every=0,
        prompts_per_step_by_bin={"b64": 3, "b256": 2, "mixed": 3}, max_new_tokens_by_bin={"b64": 24, "b256": 24, "mixed": 32},
        group_size=4, micro_batch_seqs=4, greedy_in_group=True, greedy_stats="sampled", greedy_adv_scale=0.3, greedy_adv_cap=3.0, reward={"max_comet_tokens": 510}))
    assert summ["steps"] == 3 and summ["engine_generate_calls"] > 0
    recs = _recs(tmp_path / "w" / "rl" / "metrics.jsonl")
    assert all("greedy_loop_rate" in r for r in recs if "reward" in r) and any("eval/greedy_loop_rate_bin/b64" in r for r in recs)


def test_kl_guard_state_survives_a_restart_and_every_event_is_logged(tmp_path):
    pytest.importorskip("torch")
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.rl.comet import FakeScorer
    from gemma_mt.rl.train_rl import RLConfig, run_smoke, train_rl
    from gemma_mt.utils import dataclass_from_dict

    # threshold 1e-9: the (non-zero) KL of any step after the first update is above it -> the guard cuts every `cooldown` steps
    summ = run_smoke(workdir=str(tmp_path / "w"), sft_steps=150, overrides=dict(steps=6, eval_every=0, save_every=2, kl_guard_threshold=1e-9, kl_guard_window=1, kl_guard_cooldown=2,
                                                                            kl_guard_factor=0.5, kl_guard_min_scale=0.125, learning_rate=1e-4, warmup_steps=1))
    out = tmp_path / "w" / "rl"
    ev1 = _recs(out / "kl_guard_events.jsonl")
    cuts1 = [e for e in ev1 if e["event"] == "cut"]
    assert len(cuts1) >= 2 and all(e["scale_new"] < e["scale_old"] for e in cuts1)
    m1 = [r for r in _recs(out / "metrics.jsonl") if "reward" in r]
    scale_end = m1[-1]["kl_guard_lr_scale"]
    assert scale_end < 1.0 and "kl_guard_median" in m1[-1] and "kl_guard_cut" in m1[-1] and "lr_mult_effective" in m1[-1]
    assert summ["kl_guard"]["cuts_total"] == len(cuts1) and summ["kl_guard"]["final_scale"] == scale_end
    # restart from last/: the guard state is restored, the learning rate of the first resumed step is the CUT one (before: back to the planned lr)
    cfgd = json.load(open(out / "config.json"))["config"]
    cfgd.update(steps=8, resume=True)
    train_rl(dataclass_from_dict(RLConfig, cfgd), tokenizer=FakeGemmaTokenizer(), scorer=FakeScorer())
    ev2 = _recs(out / "kl_guard_events.jsonl")
    restore = [e for e in ev2 if e["event"] == "restore"]
    assert len(restore) == 1 and restore[0]["step"] == 6 and abs(restore[0]["scale_new"] - scale_end) < 1e-12
    m2 = [r for r in _recs(out / "metrics.jsonl") if "reward" in r]
    first_resumed = [r for r in m2 if r["step"] == 7][-1]
    assert abs(first_resumed["lr_mult_effective"] - scale_end) < 1e-12 and first_resumed["lr"] < 1e-4 * 0.99
    # opting out reproduces the old behaviour (reset to the planned lr)
    cfgd.update(steps=10, kl_guard_restore=False)
    train_rl(dataclass_from_dict(RLConfig, cfgd), tokenizer=FakeGemmaTokenizer(), scorer=FakeScorer())
    m3 = [r for r in _recs(out / "metrics.jsonl") if "reward" in r]
    assert [r for r in m3 if r["step"] == 9][-1]["lr_mult_effective"] == 1.0


def test_plateau_stop_ends_the_run_through_the_winddown_with_a_summary_record(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from gemma_mt.rl import run_controls
    from gemma_mt.rl.train_rl import run_smoke

    # the statistics are tested above on synthetic series; here the wiring: force "CI contains 0" so that the first admissible boundary fires
    real = run_controls.plateau_slope_test
    monkeypatch.setattr(run_controls, "plateau_slope_test", lambda *a, **k: dict(real([0.0] * 8, 2, 200), contains_zero=True))
    summ = run_smoke(workdir=str(tmp_path / "w"), sft_steps=150, overrides=dict(
        steps=40, eval_every=2, eval_prompts=4, save_every=0, plateau_stop=True, plateau_stop_window=8, plateau_stop_block=2, plateau_stop_min_steps=8, plateau_stop_winddown=3,
        plateau_stop_min_finished=0.001, plateau_stop_bootstrap=200))
    out = tmp_path / "w" / "rl"
    chk = _recs(out / "plateau_checks.jsonl")
    assert chk[0]["blocked"] == "min_steps" and chk[0]["step"] < 8
    fired = [c for c in chk if c["fire"]]
    assert len(fired) == 1 and fired[0]["step"] >= 8 and fired[0]["steps_in_stage"] >= 8
    assert summ["stop_reason"] == "plateau_stop" and summ["steps"] == fired[0]["step"] + 3 < 40
    assert summ["plateau_stop"]["triggered"] and summ["plateau_stop"]["evidence"]["step"] == fired[0]["step"]
    assert summ["winddown"]["completed"] and (out / "WINDDOWN.done").exists() and json.load(open(out / "WINDDOWN.reason.json"))["reason"] == "plateau_stop"
    assert (out / "last" / "trainer_state.pt").exists() and (out / "summary.json").exists()
    lrs = [r["lr"] for r in _recs(out / "metrics.jsonl") if "reward" in r and r["step"] > fired[0]["step"]]
    assert lrs and lrs[-1] < lrs[0] and all("winddown_factor" in r for r in _recs(out / "metrics.jsonl") if "reward" in r and r["step"] > fired[0]["step"] + 1)


def test_plateau_stop_off_by_default_runs_to_max_steps(tmp_path):
    pytest.importorskip("torch")
    from gemma_mt.rl.train_rl import run_smoke

    summ = run_smoke(workdir=str(tmp_path / "w"), sft_steps=20, overrides=dict(steps=4, eval_every=2, save_every=0))
    assert summ["steps"] == 4 and summ["stop_reason"] == "max_steps" and "plateau_stop" not in summ and "kl_guard" not in summ
    assert not os.path.exists(tmp_path / "w" / "rl" / "plateau_checks.jsonl") and not os.path.exists(tmp_path / "w" / "rl" / "kl_guard_events.jsonl")
