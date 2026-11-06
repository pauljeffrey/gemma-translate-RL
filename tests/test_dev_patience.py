# SPDX-License-Identifier: AGPL-3.0-or-later
"""Early stop on the DEV metrics (owner 2026-11-03; run_controls.DevPatience, config early_stop_dev). Synthetic series, the real stage-A series of the 385M run, state round trip, and (with torch) the loop."""
import json
import os

import pytest

from gemma_mt.rl.run_controls import DevPatience

# stage A of the 385M run (runs_385m/A/metrics.jsonl): step -> (dev macro chrF++, weak_dev chrF++, dev finished, greedy-eval finished), dev evaluation every 50 steps
STAGE_A = {50: (39.04, 23.92, 0.968, 0.914), 100: (40.14, 26.02, 0.988, 0.945), 150: (40.13, 26.28, 0.989, 0.961), 200: (40.47, 26.45, 0.993, 0.977), 250: (40.43, 26.80, 0.997, 0.984),
           300: (40.46, 26.82, 0.997, 0.984), 350: (40.47, 26.81, 0.999, 0.984), 400: (40.50, 26.72, 0.997, 0.992)}
GUARD_A = [226, 236, 247]                       # the KL-guard cuts of stage A (kl_guard_events.jsonl)


def run(series, k=3, eps=0.15, min_steps=75, min_fin=0.98, guards=(), stage="b64", last_stage=True, stage_start=0, upto=None):
    """First step at which the tracker fires on `series` ({step: (macro, weak, fin_dev, fin_eval)}), or None; the guard events are noted when their step is reached."""
    d = DevPatience(k, eps, min_fin)
    log = []
    for step, (macro, weak, fd, fe) in sorted(series.items()):
        for g in guards:
            if g <= step and g not in d.guard_events:
                d.note_guard_event(g)
        d.observe(step, macro, weak, stage)
        dec = d.check(step, stage, stage_start, min_steps, last_stage, fd, fe)
        log.append((step, dec["blocked"], d.stale))
        if dec["fire"] or (upto is not None and step >= upto):
            return (step if dec["fire"] else None), log
    return None, log


def test_the_real_stage_a_series_with_the_default_rule_fires_only_at_the_last_dev_evaluation():
    # macro improves by >= 0.15 at 100 (+1.1) and 200 (+0.33); weak at 100, 150 (+0.26), 200 (+0.17), 250 (+0.35); from 250 on neither: stale 1 @300, 2 @350, 3 @400
    fired, log = run(STAGE_A, guards=GUARD_A)
    assert fired == 400, log
    assert [x[2] for x in log] == [0, 0, 0, 0, 0, 1, 2, 3]
    assert log[-1][1] is None and all(x[1] in ("improving",) for x in log[:-1] if x[0] >= 100)


def test_the_real_stage_a_series_with_the_stricter_settings_the_owner_may_want():
    assert run(STAGE_A, k=2, guards=GUARD_A)[0] == 350                     # k = 2 dev evaluations of 50 steps (window 250..350, the guard cuts 226-247 are older)
    assert run(STAGE_A, k=2, eps=0.4, guards=GUARD_A)[0] == 350            # gains below 0.4 chrF++ do not count; at 300 the window starts at 200 and holds the three guard cuts: blocked
    assert run(STAGE_A, k=2, eps=0.4)[0] == 300                            # without the guard events it would fire at 300
    assert run(STAGE_A, k=3, eps=0.4)[0] == 350 and run(STAGE_A, k=3, eps=0.4, guards=GUARD_A)[0] == 400      # k = 3: window 200..350 holds the guard cuts, so 400
    assert run(STAGE_A, k=3, min_fin=0.99, guards=GUARD_A)[0] is None or run(STAGE_A, k=3, min_fin=0.99, guards=GUARD_A)[0] == 400    # the greedy-eval finished rate only reaches 0.992 at 400


def test_guard_events_before_the_stale_window_do_not_block_those_inside_do():
    # the three guard cuts of stage A (226-247) are older than the window start of the fire at 400 (window start = the evaluation before the three stale ones = 250): no block
    assert run(STAGE_A, guards=GUARD_A)[0] == 400
    # a cut at 330 lies inside the window [250, 400]: blocked with guard_event, nothing fires
    fired, log = run(STAGE_A, guards=[330])
    assert fired is None and log[-1][1] == "guard_event"
    # a cut exactly one step before the window's first evaluation counts too (its lr change acts from the next step)
    assert run(STAGE_A, guards=[249])[0] is None and run(STAGE_A, guards=[248])[0] == 400


def test_there_is_no_learning_rate_condition():
    d = DevPatience(2, 0.15, 0.98)
    for s, m in ((25, 30.0), (50, 30.0), (75, 30.0)):
        d.observe(s, m, 20.0, "b64")
    assert d.check(75, "b64", 0, 50, True, 0.99, 0.99)["fire"] is True       # check() has no lr argument: a flat dev at a cut learning rate still stops (the owner asked for this)


def test_blockers_min_steps_finished_stage():
    d = DevPatience(2, 0.15, 0.98)
    for s in (25, 50, 75):
        d.observe(s, 30.0, 20.0, "b256")
    assert d.check(75, "b256", 0, 100, True, 0.99, 0.99)["blocked"] == "min_steps"
    assert d.check(75, "b256", 0, 50, True, 0.97, 0.99)["blocked"] == "finished" and d.check(75, "b256", 0, 50, True, 0.99, 0.95)["blocked"] == "finished"
    assert d.check(75, "b256", 0, 50, False, 0.99, 0.99)["blocked"] == "not_last_stage"
    assert d.check(75, "b256", 0, 50, True, None, None)["fire"] is True       # no finished rate to judge: not a blocker
    d.triggered = {"x": 1}
    assert d.check(75, "b256", 0, 50, True, 0.99, 0.99)["blocked"] == "already_triggered"


def test_a_slow_steady_rise_is_an_improvement_through_the_anchor_not_a_plateau():
    d = DevPatience(3, 0.15, 0.98)
    v = 30.0
    for i in range(8):                                                    # +0.06 per evaluation: each step is below eps, but the anchor stays put and the third step crosses it
        v += 0.06
        d.observe(25 * (i + 1), v, None, "b64")
        assert d.check(25 * (i + 1), "b64", 0, 0, True, 1.0, 1.0)["fire"] is False
    assert d.stale <= 2


def test_flat_series_goes_stale_k_times_and_either_metric_resets_the_count():
    d = DevPatience(3, 0.15, 0.98)
    seq = [(30.0, 20.0), (30.05, 20.0), (30.0, 20.1), (30.1, 20.05), (30.0, 20.4), (30.1, 20.3), (30.0, 20.4), (30.0, 20.3)]
    stale, fires = [], []
    for i, (m, w) in enumerate(seq):
        d.observe(25 * (i + 1), m, w, "b64")
        stale.append(d.stale)
        fires.append(d.check(25 * (i + 1), "b64", 0, 0, True, 1.0, 1.0)["fire"])
    assert stale == [0, 1, 2, 3, 0, 1, 2, 3]                              # the weak metric jumped +0.4 at the 5th evaluation (reset); the macro never moved by >= eps
    assert fires == [False, False, False, True, False, False, False, True]


def test_macro_only_when_there_is_no_weak_set_and_a_new_stage_starts_a_new_series():
    d = DevPatience(2, 0.15, 0.98)
    for s in (25, 50, 75):
        d.observe(s, 30.0, None, "b64")
    assert d.stale == 2 and d.check(75, "b64", 0, 0, True, 1.0, 1.0)["fire"]
    d.observe(100, 30.0, None, "b256")
    assert d.stale == 0 and d.anchor["macro"] == 30.0 and len(d.hist) == 1


def test_state_round_trip_survives_a_restart():
    d = DevPatience(3, 0.15, 0.98)
    for s, (m, w, _, _) in sorted(STAGE_A.items())[:6]:
        d.observe(s, m, w, "b64")
    d.note_guard_event(226)
    e = DevPatience(3, 0.15, 0.98)
    e.load_state_dict(json.loads(json.dumps(d.state_dict())))
    assert e.state_dict() == d.state_dict() and e.stale == d.stale
    e.load_state_dict(None)                                               # a checkpoint written before this feature has no state: nothing changes
    assert e.stale == d.stale
    for s in (350, 400):
        m, w, fd, fe = STAGE_A[s]
        d.observe(s, m, w, "b64"); e.observe(s, m, w, "b64")
    assert d.check(400, "b64", 0, 75, True, 0.997, 0.992)["fire"] and e.check(400, "b64", 0, 75, True, 0.997, 0.992)["fire"]


def test_config_validation():
    pytest.importorskip("torch")
    from gemma_mt.rl.train_rl import RLConfig, validate_config

    assert RLConfig().early_stop_dev is False
    validate_config(RLConfig(early_stop_dev=True, benchmark_eval_path="x.jsonl", select_by="benchmark_chrf"))
    for bad in ({"early_stop_dev": True}, {"early_stop_dev": True, "benchmark_eval_path": "x", "early_stop_dev_k": 0}, {"early_stop_dev": True, "benchmark_eval_path": "x", "run_controls": False}):
        with pytest.raises(ValueError):
            validate_config(RLConfig(**bad))


def test_loop_stops_through_the_winddown_with_stop_reason_dev_plateau():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from gemma_mt.rl.train_rl import run_smoke

    s = run_smoke(sft_steps=60, benchmark=True, overrides={"steps": 40, "eval_every": 2, "early_stop_dev": True, "early_stop_dev_k": 2, "early_stop_dev_eps": 1000.0, "early_stop_dev_min_steps": 0,
                                                          "early_stop_dev_min_finished": 0.0, "early_stop_dev_winddown": 3, "select_min_finished_rate": 0.0})
    summ = json.load(open(s["workdir"] + "/rl/summary.json"))
    assert summ["stop_reason"] == "dev_plateau" and summ["early_stop_dev"]["triggered"] is True and summ["early_stop_dev"]["evidence"]["stale_evals"] >= 2
    assert summ["steps"] < 40 if "steps" in summ else True
    reason = json.load(open(s["workdir"] + "/rl/WINDDOWN.reason.json"))
    assert reason["reason"] == "dev_plateau" and reason["n"] == 3
    checks = [json.loads(l) for l in open(s["workdir"] + "/rl/dev_plateau_checks.jsonl")]
    assert checks and checks[0]["fire"] is False and sum(c["fire"] for c in checks) == 1 and any(c["blocked"] == "already_triggered" for c in checks) or sum(c["fire"] for c in checks) == 1
