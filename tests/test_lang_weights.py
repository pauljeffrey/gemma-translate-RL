# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-65: language draw weights as a run control (LANG_WEIGHTS.json) and the optional adaptive updater (rl/lang_weights.py). CPU, no torch except the smoke test."""
import json
import math
import os
import time

import numpy as np
import pytest

from gemma_mt.languages import canonical_iso
from gemma_mt.rl import lang_weights as lw
from gemma_mt.rl.data import MIXED, PromptSchedule

LANGS = ["afr", "yor", "hau", "zul", "kbp", "fon", "bam", "swh"]


def _schedule(weights=None, seed=1, alpha=0.0, floor=(0, 1)):
    iso, lbin, dom = [], [], []
    for l in LANGS:
        for b in ("b64", "b256"):
            for k in range(30):
                iso.append(l)
                lbin.append(b)
                dom.append("science" if k % 3 else "general")
    return PromptSchedule(iso, alpha, seed, length_bin=lbin, curriculum={"b64": 0}, language_weights=weights, domain=dom, domain_weights={"science": 0.7, "general": 0.3},
                          floor_steps=floor[0], floor_prompts=floor[1])


def _counts(sch, steps=300, n=16):
    c = {l: 0 for l in LANGS}
    iso = np.asarray([None])
    for s in range(steps):
        for i in sch.indices(s, n):
            c[_lang_of(sch, i)] += 1
    return c


def _lang_of(sch, i):
    for (d, b), (langs, by_lang, p) in sch.dom_data.items():
        for l, ix in zip(langs, by_lang):
            if i in set(ix.tolist()):
                return l
    raise KeyError(i)


# ------------------------------------------------------------------------------------------------------------------------------ the file
def test_parse_valid_and_invalid():
    ok, errs = lw.parse_weights_text('{"weights": {"yor": 3, "zul": 0.5}, "multipliers": {"hau": 0.8}, "mode": "merge", "adaptive": "freeze", "note": "x"}', LANGS, canonical_iso)
    assert errs == [] and ok.weights == {"yor": 3.0, "zul": 0.5} and ok.multipliers == {"hau": 0.8} and ok.adaptive == "freeze"
    for bad, frag in (('{"weights": {"xxx": 1}}', "unknown language"), ('{"weights": {"yor": -1}}', "not a number"), ('{"weights": {"yor": 21}}', "not a number"),
                      ('{"weights": {"yor": "2"}}', "not a number"), ('{"weights": {"yor": NaN}}', "not a number"), ('{"mode": "x"}', "mode"), ('{"adaptive": "maybe"}', "adaptive"),
                      ('{"wights": {}}', "unknown key"), ("[1]", "object"), ("{not json", "JSON"), ('{"weights": {"nde": 1}}', "unknown language")):
        spec, errs = lw.parse_weights_text(bad, LANGS, canonical_iso)
        assert spec is None and any(frag in e for e in errs), (bad, errs)
    spec, errs = lw.parse_weights_text('{"weights": {"YOR": 2}}', LANGS, canonical_iso)          # canonical spelling accepted
    assert spec.weights == {"yor": 2.0}


def test_compose_precedence():
    base = {"yor": 2.5, "kbp": 2.5}
    auto = {"yor": 3.0, "zul": 0.8}
    spec = lw.FileSpec(weights={"kbp": 1.0}, multipliers={"yor": 0.5})
    w = lw.compose_weights(LANGS, base, auto, spec)
    assert w["kbp"] == 1.0 and w["yor"] == 1.5 and w["zul"] == 0.8 and w["afr"] == 1.0                  # file > adaptive > yaml; multiplier on top of the adaptive weight
    assert lw.compose_weights(LANGS, base, auto, lw.FileSpec(adaptive="off"))["yor"] == 2.5             # adaptive ignored
    assert lw.compose_weights(LANGS, base, auto, lw.FileSpec(adaptive="freeze"))["yor"] == 3.0          # frozen adaptive weights stay in force
    w = lw.compose_weights(LANGS, base, auto, lw.FileSpec(mode="replace", weights={"hau": 4.0}))
    assert w["hau"] == 4.0 and w["yor"] == 1.0 and w["kbp"] == 1.0                                         # replace: yaml and adaptive dropped


# ------------------------------------------------------------------------------------------------------------------------------ the schedule
def test_set_language_weights_equals_a_fresh_schedule_and_changes_the_mixture():
    a = _schedule({"yor": 2.0})
    b = _schedule({"yor": 1.0, "kbp": 5.0, "zul": 0.0})
    a.set_language_weights({"yor": 1.0, "kbp": 5.0, "zul": 0.0})
    for s in (0, 1, 7, 60):
        assert a.indices(s, 16) == b.indices(s, 16)                                                       # stateless draw: (seed, step, weights in force) only
    c = _counts(b, steps=120)
    assert c["zul"] == 0 and c["kbp"] > 2 * c["afr"]


def test_set_language_weights_rejects_a_dead_pool_and_changes_nothing():
    a = _schedule({"yor": 2.0})
    before = a.indices(3, 16)
    with pytest.raises(ValueError):
        a.set_language_weights({l: 0.0 for l in LANGS})
    with pytest.raises(ValueError):
        a.set_language_weights({"yor": -1.0})
    assert a.indices(3, 16) == before and a.language_weights == {"yor": 2.0}


def test_floor_survives_zero_weight():
    a = _schedule(None, floor=(20, 2))
    a.set_language_weights({"zul": 0.0})
    seen = [0] * 20
    for s in range(20):
        for i in a.indices(s, 16):
            seen[s] += _lang_of(a, i) == "zul"
    assert sum(seen) >= 2                                                                                  # the D-59 floor still forces >= 2 prompts per 20 steps


# ------------------------------------------------------------------------------------------------------------------------------ the controller: file
def _ctl(tmp_path, sched=None, **kw):
    msgs = []
    sched = sched or _schedule({"yor": 2.5})
    c = lw.LangWeightController(str(tmp_path), sched, {"yor": 2.5}, LANGS, {l: "qe" for l in LANGS}, canon=canonical_iso, log=msgs.append, **kw)
    return c, sched, msgs


def test_file_control_applies_clears_and_ignores_garbage(tmp_path):
    c, sch, msgs = _ctl(tmp_path, file_control=True)
    r = c.update(0)
    assert r["reward/lang_weights/yor"] == 2.5 and r["reward/lang_weights/afr"] == 1.0 and "reward/lang_weights/hau" in r          # step 0: everything logged
    assert c.update(1) == {}                                                                              # nothing changed: nothing logged
    (tmp_path / "LANG_WEIGHTS.json").write_text('{"weights": {"yor": 1.0, "kbp": 4.0}}')
    r = c.update(2)
    assert r["reward/lang_weights/kbp"] == 4.0 and r["reward/lang_weights/yor"] == 1.0 and sch.language_weights["kbp"] == 4.0 and r["reward/lang_weights/file_active"] == 1.0
    time.sleep(0.01)
    (tmp_path / "LANG_WEIGHTS.json").write_text("garbage")
    assert c.update(3) == {} and c.in_force["kbp"] == 4.0                                                 # invalid file: the last valid weights stay
    assert sum("ignored" in m for m in msgs) == 1
    c.update(4)
    assert sum("ignored" in m for m in msgs) == 1                                                         # logged once per file version
    (tmp_path / "LANG_WEIGHTS.json").write_text('{"weights": {"nde": 2}}')
    c.update(5)
    assert c.in_force["kbp"] == 4.0
    os.remove(tmp_path / "LANG_WEIGHTS.json")
    r = c.update(6)
    assert r["reward/lang_weights/kbp"] == 1.0 and r["reward/lang_weights/yor"] == 2.5 and any("removed" in m for m in msgs)
    assert c.weights_at(0)["yor"] == 2.5 and c.weights_at(2)["kbp"] == 4.0 and c.weights_at(6)["kbp"] == 1.0


def test_file_that_kills_a_pool_is_rejected_by_the_schedule(tmp_path):
    c, sch, msgs = _ctl(tmp_path, file_control=True)
    c.update(0)
    (tmp_path / "LANG_WEIGHTS.json").write_text(json.dumps({"mode": "replace", "weights": {l: 0.0 for l in LANGS}}))
    c.update(1)
    assert any("REJECTED" in m for m in msgs) and c.in_force["yor"] == 2.5 and sch.language_weights["yor"] == 2.5


# ------------------------------------------------------------------------------------------------------------------------------ statistics
def _feed(win, lang, rewards_by_group, status=None, key="b64|science", qe=None, cons=None, adv_w=None, pen=None, G=4):
    isos = [lang] * len(rewards_by_group)
    rs = [x for g in rewards_by_group for x in g]
    st = status or ["ok"] * len(rs)
    win.add(isos, [key] * len(isos), rs, st, G, qe=qe, adv_w=adv_w, cons=cons, pen=pen)


def test_window_bin_adjustment_and_group_std():
    w = lw.LangWindow()
    # language A only saw the easy bin (reward 1.5), language B only the hard bin (0.5): raw means differ by 1, adjusted means are equal (each bin has one language)
    _feed(w, "afr", [[1.4, 1.6, 1.5, 1.5]] * 3, key="b64|science")
    _feed(w, "yor", [[0.4, 0.6, 0.5, 0.5]] * 3, key="b2048|science")
    _feed(w, "yor", [[1.4, 1.6, 1.5, 1.5]] * 3, key="b64|science")
    s = w.summary()
    assert s["afr"]["reward"] == pytest.approx(1.5) and s["yor"]["reward"] == pytest.approx(1.0)
    assert s["yor"]["reward_adj"] == pytest.approx(s["afr"]["reward_adj"])                                    # the bin effect is removed: both sit at the window mean
    assert s["afr"]["std"] == pytest.approx(math.sqrt(0.5 * 0.01), rel=1e-6) and s["afr"]["flat"] == 0.0
    w2 = lw.LangWindow()
    _feed(w2, "hau", [[1.0] * 4] * 5, status=["ok"] * 16 + ["degenerate"] * 4)
    s2 = w2.summary()["hau"]
    assert s2["flat"] == 1.0 and s2["fail"] == pytest.approx(0.2) and s2["loop"] == pytest.approx(0.2)


def test_window_roundtrip_and_qe_consensus_correlation():
    w = lw.LangWindow()
    rng = np.random.default_rng(0)
    qe = [float(x) for x in rng.random(48)]
    cons = [float(q * 40 + rng.normal(0, 1)) for q in qe]                       # strongly agreeing signals
    _feed(w, "fon", [[0.0] * 4] * 12, qe=qe, cons=cons)
    r = w.summary()["fon"]["qc_corr"]
    assert r is not None and r > 0.5
    w3 = lw.LangWindow.from_json(json.loads(json.dumps(w.to_json())))
    assert w3.summary()["fon"] == w.summary()["fon"]


def test_dev_trend_detects_a_slope_and_stays_quiet_on_noise():
    pts = [(50 * k, 30.0 + 0.02 * 50 * k, 200) for k in range(1, 9)]            # +1 chrF per 50 steps = 2 per 100
    t = lw.dev_trend(pts, 4)
    assert t["slope100"] == pytest.approx(2.0) and t["t"] > 5
    flat = [(50 * k, 30.0 + (0.3 if k % 2 else -0.3), 60) for k in range(1, 9)]
    assert abs(lw.dev_trend(flat, 4)["t"]) < 1.5
    assert lw.dev_trend(pts[:3], 4) is None


# ------------------------------------------------------------------------------------------------------------------------------ the rule
def _sig(reward, fail=0.0, groups=20, **kw):
    d = {"n": groups * 16, "rn": groups * 16, "groups": groups, "reward_adj": reward, "std": 0.4, "flat": 0.0, "fail": fail, "loop": 0.0, "pen": 0.0, "advw": 1.0, "qc_corr": None}
    d.update(kw)
    return d


P = lw.AdaptiveParams()


def test_underperforming_qe_language_is_boosted_and_flags_cap_it_at_one():
    sig = {"yor": _sig(-0.5), "hau": _sig(0.9), "afr": _sig(1.0), "zul": _sig(0.9)}
    modes = {l: "qe" for l in sig}
    f = lw.factors(sig, modes, {}, P)
    assert f["yor"]["f"] > 1.2 and f["afr"]["f"] == pytest.approx(1.0, abs=0.03)
    for kw, flag in ((dict(loop=0.4), "loops"), (dict(pen=0.5), "anchor_penalty"), (dict(advw=0.1), "low_reliability")):
        g = lw.factors({**sig, "yor": _sig(-0.5, **kw)}, modes, {}, P)
        assert g["yor"]["f"] <= 1.0 and flag in g["yor"]["flags"], (kw, g["yor"])
    # a blend language whose QE and consensus disagree inside groups is never boosted by its low QE
    g = lw.factors({**sig, "kbp": _sig(-0.5, qc_corr=-0.1)}, {**modes, "kbp": "blend"}, {}, P)
    assert g["kbp"]["f"] <= 1.0 and "qe_consensus_disagree" in g["kbp"]["flags"]
    # consensus / blend languages have no human parity: low reward level alone does nothing
    g = lw.factors({**sig, "kbp": _sig(-1.0)}, {**modes, "kbp": "consensus"}, {}, P)
    assert g["kbp"]["f"] == pytest.approx(1.0)


def test_groups_without_a_gradient_and_high_kl_limit_the_boost():
    sig = {"yor": _sig(-0.5, flat=0.95), "hau": _sig(0.9), "afr": _sig(1.0), "zul": _sig(0.9)}
    modes = {l: "qe" for l in sig}
    f = lw.factors(sig, modes, {}, P)
    assert f["yor"]["f"] < lw.factors({**sig, "yor": _sig(-0.5, flat=0.0)}, modes, {}, P)["yor"]["f"] and f["yor"]["learn"] < 0.2               # almost no group carries a gradient
    sig2 = {"yor": _sig(-0.5, kl=0.2), "hau": _sig(0.9, kl=0.02), "afr": _sig(1.0, kl=0.02), "zul": _sig(0.9, kl=0.02)}
    g = lw.factors(sig2, modes, {}, P)
    assert "kl_high" in g["yor"]["flags"] and g["yor"]["f"] <= 1.0


def test_failures_drive_a_boost_in_every_mode():
    sig = {"a": _sig(0.0, fail=0.05), "b": _sig(0.0, fail=0.05), "c": _sig(0.0, fail=0.05), "kbp": _sig(0.0, fail=0.6)}
    f = lw.factors(sig, {"a": "qe", "b": "qe", "c": "qe", "kbp": "consensus"}, {}, P)
    assert f["kbp"]["f"] > 1.3 and f["a"]["f"] == pytest.approx(1.0, abs=0.35)


def test_qe_saturation_shrinks_only_mildly_without_a_reference_and_not_while_the_reference_rises():
    sig = {"yor": _sig(1.85), "afr": _sig(1.0), "zul": _sig(0.9)}
    modes = {l: "qe" for l in sig}
    no_ref = lw.factors(sig, modes, {}, P)["yor"]
    assert 0.85 - 1e-9 <= no_ref["d"] < 1.0 and no_ref["f"] >= 0.85 - 1e-9                                 # Yoruba's QE is saturated while its chrF++ is low: never shrunk below 0.85 on QE alone
    flat = lw.factors(sig, modes, {"yor": {"t": 0.2, "slope100": 0.0, "se100": 0.3}}, P)["yor"]
    assert flat["d"] < no_ref["d"] and flat["f"] >= P.min_ratio
    imprecise = lw.factors(sig, modes, {"yor": {"t": 0.2, "slope100": 0.0, "se100": 2.0}}, P)["yor"]           # "no trend" measured with a 2 chrF++ error bar confirms nothing
    assert imprecise["d"] == no_ref["d"]
    rising = lw.factors(sig, modes, {"yor": {"t": 3.0, "slope100": 1.0, "se100": 0.3}}, P)["yor"]
    assert rising["d"] == 1.0 and rising["f"] > 1.0
    falling = lw.factors(sig, modes, {"yor": {"t": -3.0, "slope100": -1.0, "se100": 0.3}}, P)["yor"]
    assert "dev_falling" in falling["flags"] and falling["f"] < 1.0


def test_no_signal_means_no_move():
    f = lw.factors({"yor": _sig(-1.5, groups=2)}, {"yor": "qe"}, {}, P)
    assert f["yor"]["f"] == 1.0 and f["yor"]["why"] == "no_signal"


def test_targets_budget_bounds_smoothing_and_floor():
    base = {l: 1.0 for l in LANGS}
    base.update({"yor": 3.0, "kbp": 2.5})
    fac = {l: {"f": 1.0} for l in LANGS}
    fac["afr"]["f"], fac["yor"]["f"] = 0.5, 1.6
    new = lw.targets_from_factors(base, dict(base), fac, LANGS, P)
    assert sum(new.values()) == pytest.approx(sum(base.values()), rel=1e-6)                                # bounded total draw budget
    assert new["afr"] >= max(P.min_ratio * 1.0, P.w_min) - 1e-9 and new["yor"] <= P.max_ratio * 3.0 + 1e-9
    for l in LANGS:                                                                                         # smoothing: one update moves at most +-25 % (before the budget rescale)
        assert 0.6 * base[l] <= new[l] <= 1.5 * base[l]
    iso_fac = {l: {"f": 0.0} for l in LANGS}                                                                # even f -> 0 never goes below the ratio floor
    nw = lw.targets_from_factors(base, dict(base), iso_fac, LANGS, P)
    assert min(nw[l] / base[l] for l in LANGS) >= 0.74


# ------------------------------------------------------------------------------------------------------------------------------ the adaptive controller
def _run_window(c, bad="yor", good="afr", steps=50, step0=0):
    G = 4
    rng = np.random.default_rng(3)
    for s in range(step0, step0 + steps):
        isos, keys, rewards, status = [], [], [], []
        for l in LANGS:
            isos.append(l)
            keys.append("b64|science")
            base = {bad: -0.8, good: 1.0}.get(l, 0.9)
            r = list(base + 0.3 * rng.standard_normal(G))
            rewards += r
            status += ["ok"] * G
        c.observe(s, isos, keys, rewards, status, G)


def test_adaptive_boosts_the_laggard_keeps_the_budget_and_logs(tmp_path):
    sch = _schedule({"yor": 2.5})
    c, _, msgs = _ctl(tmp_path, sched=sch, file_control=True, adaptive=True, every=50, params=lw.AdaptiveParams(first_step=50, min_groups=4))
    c.update(0)
    _run_window(c)
    r = c.update(50)
    assert c.n_updates == 1 and r["reward/lang_weights/adaptive_updates"] == 1.0
    assert 2.5 < c.in_force["yor"] <= 2.5 * 1.25 + 1e-9 and r["reward/lang_weights/yor"] == c.in_force["yor"]          # boosted, within +25 % per update
    assert all(c.in_force[l] < 1.0 for l in LANGS if l != "yor")                                            # the others give the budget back
    assert sum(c.in_force.values()) == pytest.approx(sum(c.weights_at(0).values()), rel=1e-6)
    assert sch.language_weights == {k: pytest.approx(v) for k, v in c.in_force.items()}
    assert any("adaptive update @ 50" in m for m in msgs)
    assert c.update(51) == {}                                                                              # no second update on the same step


def test_dry_run_changes_nothing_but_logs_the_proposal(tmp_path):
    sch = _schedule({"yor": 2.5})
    c, _, msgs = _ctl(tmp_path, sched=sch, file_control=True, adaptive=True, dry_run=True, every=50, params=lw.AdaptiveParams(first_step=50, min_groups=4))
    c.update(0)
    _run_window(c)
    r = c.update(50)
    assert c.in_force["yor"] == 2.5 and sch.language_weights["yor"] == 2.5 and all(c.in_force[l] == 1.0 for l in LANGS if l != "yor")
    assert r["reward/lang_weights_dry/yor"] > 2.5 and c.n_updates == 0 and any("dry run" in m for m in msgs)
    assert "reward/lang_weights/yor" not in r or r["reward/lang_weights/yor"] == 2.5                       # the weights in force are untouched


def test_treated_subset_and_freeze_and_off(tmp_path):
    c, _, _ = _ctl(tmp_path, file_control=True, adaptive=True, every=50, params=lw.AdaptiveParams(first_step=50, min_groups=4), treated=["yor", "afr", "hau"])
    c.update(0)
    _run_window(c)
    c.update(50)
    for l in ("zul", "kbp", "fon", "bam", "swh"):
        assert c.in_force[l] == (2.5 if l == "yor" else 1.0)                                                # untreated languages keep their base weight (the control group)
    assert sum(c.in_force[l] for l in ("yor", "afr", "hau")) == pytest.approx(2.5 + 1.0 + 1.0, rel=1e-6)
    frozen = dict(c.in_force)
    (tmp_path / "LANG_WEIGHTS.json").write_text('{"adaptive": "freeze"}')
    _run_window(c, step0=50)
    c.update(100)
    assert c.in_force == frozen and c.n_updates == 1                                                       # frozen: no new update, last adaptive weights stay
    (tmp_path / "LANG_WEIGHTS.json").write_text('{"adaptive": "off"}')
    c.update(101)
    assert c.in_force["yor"] == 2.5 and c.in_force["afr"] == 1.0                                           # off: back to the yaml weights


def test_resume_is_exact_and_a_missing_window_skips_the_update(tmp_path):
    kw = dict(file_control=True, adaptive=True, every=50, params=lw.AdaptiveParams(first_step=50, min_groups=4))
    sch = _schedule({"yor": 2.5})
    c, _, _ = _ctl(tmp_path, sched=sch, **kw)
    c.update(0)
    _run_window(c)
    c.save_state(50)                                                                                         # the trainer's checkpoint hook (before the update of the next loop top)
    c.update(50)                                                                                             # writes the sidecar again, after the update
    w50 = dict(c.in_force)
    # case 1: crash after the update, resume from the checkpoint of step 50 -> same weights, update not repeated
    sch2 = _schedule({"yor": 2.5})
    c2 = lw.LangWeightController(str(tmp_path), sch2, {"yor": 2.5}, LANGS, {l: "qe" for l in LANGS}, canon=canonical_iso, log=lambda m: None, start_step=50, **kw)
    c2.update(50)
    assert c2.in_force == pytest.approx(w50) and c2.n_updates == 1 and sch2.language_weights == pytest.approx(w50)
    # case 2: the sidecar is older than the checkpoint -> weights from the history, next update skipped (window incomplete)
    c.save_state(100)
    (tmp_path / "LANG_WEIGHTS.state.json").write_text(json.dumps({**json.load(open(tmp_path / "LANG_WEIGHTS.state.json")), "saved_step": 50}))
    msgs = []
    c3 = lw.LangWeightController(str(tmp_path), _schedule({"yor": 2.5}), {"yor": 2.5}, LANGS, {l: "qe" for l in LANGS}, canon=canonical_iso, log=msgs.append, start_step=100, **kw)
    assert c3.partial
    _run_window(c3, step0=100)
    c3.update(100)
    assert any("skipped" in m for m in msgs) and c3.in_force == pytest.approx(w50)
    # case 3: a fresh run with the same directory (start_step 0) ignores the old sidecar
    c4 = lw.LangWeightController(str(tmp_path), _schedule({"yor": 2.5}), {"yor": 2.5}, LANGS, {l: "qe" for l in LANGS}, canon=canonical_iso, log=lambda m: None, start_step=0, **kw)
    assert c4.n_updates == 0 and c4.history == []


def test_params_validation():
    with pytest.raises(ValueError):
        lw.AdaptiveParams.from_dict({"nonsense": 1})
    p = lw.AdaptiveParams.from_dict({"eta": 0.3, "first_step": 200, "min_groups": 3})
    assert p.eta == 0.3 and p.first_step == 200 and p.min_groups == 3


def test_a_controller_with_nothing_active_draws_exactly_like_no_controller(tmp_path):
    base = {"yor": 2.5, "kbp": 3.0}
    ref = _schedule(base)
    sch = _schedule(base)
    c = lw.LangWeightController(str(tmp_path), sch, base, LANGS, {l: "qe" for l in LANGS}, file_control=False, adaptive=False, canon=canonical_iso, log=lambda m: None, log_stats=True)
    c.update(0)
    for s in (0, 1, 2, 17, 99):
        assert sch.indices(s, 16) == ref.indices(s, 16)


def test_a_failing_controller_never_raises_and_the_updater_switches_itself_off(tmp_path):
    msgs = []
    c, _, _ = _ctl(tmp_path, file_control=True, adaptive=True, every=50, params=lw.AdaptiveParams(first_step=50, min_groups=1))
    c.log = msgs.append
    c.window = None                                                           # corrupt the state on purpose
    for step in (50, 100, 150, 200, 250):
        c.update(step)                                                         # must not raise
    c.observe(1, ["yor"], ["k"], [0.0] * 4, ["ok"] * 4, 4)
    assert c.errors >= 5 and not c.adaptive and any("ERROR" in m for m in msgs) and any("switched OFF" in m for m in msgs)
    assert sum("ERROR in update" in m for m in msgs) == 1                      # the traceback is printed once per kind


def test_everything_off_is_no_controller():
    class Cfg:
        lang_weights_file = False
        adaptive_lang_weights = False
        lang_stats_log = False
        run_controls = True
        language_weights = {}
        out_dir = "/nonexistent"

    assert lw.build_controller(Cfg(), _schedule(), LANGS, {}, start_step=0, canon=canonical_iso) is None


# ------------------------------------------------------------------------------------------------------------------------------ end to end (CPU smoke)
def test_cpu_smoke_with_file_control_adaptive_and_resume(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from gemma_mt.rl.train_rl import run_smoke

    work = tmp_path / "w"
    (work / "rl").mkdir(parents=True)
    (work / "rl" / "LANG_WEIGHTS.json").write_text('{"weights": {"ibo": 4.0}, "note": "smoke"}')
    summ = run_smoke(workdir=str(work), sft_steps=20, overrides=dict(
        steps=6, eval_every=0, save_every=2, warmup_steps=1, learning_rate=1e-4, lang_weights_file=True, adaptive_lang_weights=True, adaptive_lang_weights_every=2,
        adaptive_lang_weights_first_step=2, lang_weights_log_every=2, language_weights={"yor": 2.0}, adaptive_lang_weights_params={"min_groups": 1, "shrink_k": 4.0}))
    assert summ["steps"] == 6 and summ["lang_weights"]["adaptive"] and summ["lang_weights"]["updates"] >= 1
    rows = [json.loads(l) for l in open(work / "rl" / "metrics.jsonl") if l.strip()]
    steps = [r for r in rows if "lr" in r]
    assert steps[0]["reward/lang_weights/ibo"] == 4.0 and steps[0]["reward/lang_weights/yor"] == 2.0           # the file wins over nothing, the yaml dict is the base
    assert all(any(k.startswith("lang/reward/") for k in r) for r in steps)
    assert any(r.get("reward/lang_weights/adaptive_updates", 0) >= 1 for r in steps)
    assert (work / "rl" / "LANG_WEIGHTS.state.json").exists()
    st = json.load(open(work / "rl" / "LANG_WEIGHTS.state.json"))
    assert any(h["kind"] == "update" for h in st["history"])


# ------------------------------------------------------------------------------------------------------------------------------ the offline replay tool
def _load_replay():
    import importlib.util

    spec = importlib.util.spec_from_file_location("replay_lang_weights", os.path.join(os.path.dirname(__file__), "..", "scripts", "replay_lang_weights.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _calib(tmp_path, langs=("yor", "afr", "hau", "kbp")):
    cal = {l: {"content": {"mode": "blend" if l == "kbp" else "qe"}} for l in langs}
    cal["_meta"] = {}
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(cal))
    return str(p)


def test_replay_of_lang_stats_file_boosts_the_laggard(tmp_path):
    m = _load_replay()
    langs = ["yor", "afr", "hau", "kbp"]
    rows = []
    rng = np.random.default_rng(0)
    for s in range(1, 151):
        row = {"step": s, "reward": 0.5}
        for l in langs:
            row[f"lang/n/{l}"] = 16.0
            row[f"lang/reward/{l}"] = {"yor": -0.8, "afr": 1.0, "hau": 0.9, "kbp": 0.9}[l] + 0.05 * rng.standard_normal()
            row[f"lang/radj/{l}"] = row[f"lang/reward/{l}"] - 0.5
            row[f"lang/fail/{l}"] = 0.0
        rows.append(row)
    mp = tmp_path / "m.jsonl"
    mp.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "traj.csv"
    assert m.main(["--metrics", str(mp), "--calib", _calib(tmp_path), "--base-json", '{"yor": 2.0}', "--every", "50", "--first-step", "50", "--params", '{"min_groups": 3}',
                   "--out-csv", str(out), "--quiet"]) == 0
    import csv

    rows_csv = list(csv.DictReader(open(out)))
    last = max(int(r["step"]) for r in rows_csv)
    final = {r["iso"]: float(r["weight"]) for r in rows_csv if int(r["step"]) == last}
    assert last == 150 and final["yor"] > 2.0 * 1.1 and final["afr"] < 1.0                                  # updates at 50, 100 and 150; the laggard gained, the budget came from the others
    assert sum(final.values()) == pytest.approx(5.0, rel=1e-6)


def test_replay_of_an_old_file_uses_failure_rates_and_dev_gates_only(tmp_path):
    m = _load_replay()
    rows = []
    for s in range(1, 121):
        row = {"step": s, "reward": 0.0, "reward/lid/fail_rate/yor": 0.0, "reward/lid/fail_rate/afr": 0.0, "reward/lid/fail_rate/hau": 0.0, "reward/lid/fail_rate/kbp": 0.0,
               "reward/fail/degenerate/hau": 0.7, "reward/fail/no_end_of_turn/hau": 0.0}
        rows.append(row)
        if s % 50 == 0:
            rows.append({"step": s, "eval/reward": 0.1, "bench/chrf/yor": 30.0 + 0.01 * s, "bench/chrf_macro": 30.0})
    mp = tmp_path / "m.jsonl"
    mp.write_text("\n".join(json.dumps(r) for r in rows))
    assert m.main(["--metrics", str(mp), "--calib", _calib(tmp_path), "--every", "50", "--first-step", "100", "--params", '{"min_groups": 3}', "--quiet"]) == 0
