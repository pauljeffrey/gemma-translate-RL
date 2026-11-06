# SPDX-License-Identifier: AGPL-3.0-or-later
"""Patch 3 (owner 2026-11-03): held-out AfriCOMET monitor (rl/dev_comet.py), reward-hacking alarm, combined dev stop, per-language stall logging, dev prediction saving."""
import json
import os

import numpy as np
import pytest

from gemma_mt.rl.dev_comet import HeldoutComet, macro_with_sd
from gemma_mt.rl.run_controls import DevPatience, HackingAlarm, LangStall


# ------------------------------------------------------------------------------------------------ hacking alarm
def feed(h, rows, stage="b256"):
    out = None
    for step, chrf, comet, qe in rows:
        h.observe(step, chrf, comet, qe, stage)
        out = h.check(step)
    return out


def test_alarm_fires_when_qe_rises_and_both_dev_measures_fall():
    h = HackingAlarm(3, 0.3, 0.005)
    d = feed(h, [(25, 40.5, 0.700, -0.60), (50, 40.4, 0.698, -0.58), (75, 40.2, 0.694, -0.56), (100, 40.1, 0.690, -0.55)])
    assert d["fire"] is True and d["qe_slope_per_100_steps"] > 0 and d["chrf_change"] == pytest.approx(-0.4) and d["comet_change"] == pytest.approx(-0.010) and d["window_steps"] == [25, 100]


def test_alarm_is_blocked_unless_all_three_conditions_hold():
    base = [(25, 40.5, 0.700, -0.60), (50, 40.4, 0.698, -0.58), (75, 40.2, 0.694, -0.56), (100, 40.1, 0.690, -0.55)]
    assert feed(HackingAlarm(3, 0.3, 0.005), base[:3])["blocked"] == "history"                              # k + 1 evaluations needed
    assert feed(HackingAlarm(3, 0.3, 0.005), [(s, c, m, -q) for s, c, m, q in base])["blocked"] == "qe_not_rising"      # the QE falls too: not hacking
    assert feed(HackingAlarm(3, 0.3, 0.005), [(s, 40.5 - 0.02 * i, m, q) for i, (s, c, m, q) in enumerate(base)])["blocked"] == "chrf_not_fallen"      # -0.06 chrF++ is noise
    assert feed(HackingAlarm(3, 0.3, 0.005), [(s, c, 0.70 - 0.001 * i, q) for i, (s, c, m, q) in enumerate(base)])["blocked"] == "comet_not_fallen"    # -0.003 AfriCOMET is below eps
    assert feed(HackingAlarm(3, 0.3, 0.005), [(s, c, None, q) for s, c, m, q in base])["blocked"] == "no_comet"        # monitor failed / off: never fires
    assert feed(HackingAlarm(3, 0.3, 0.005), [(s, c, m, None) for s, c, m, q in base])["blocked"] == "no_qe"


def test_alarm_stage_change_resets_and_state_round_trip():
    h = HackingAlarm(2, 0.3, 0.005)
    for step, chrf, comet, qe in [(25, 41.0, 0.71, -0.6), (50, 40.5, 0.70, -0.5)]:
        h.observe(step, chrf, comet, qe, "b256")
    h.observe(75, 40.0, 0.69, -0.4, "mixed")                       # a new stage: the series starts again
    assert len(h.hist) == 1 and h.check(75)["blocked"] == "history"
    g = HackingAlarm(2, 0.3, 0.005)
    g.load_state_dict(json.loads(json.dumps(h.state_dict())))
    assert g.hist == h.hist
    g.load_state_dict(None)
    assert g.hist == h.hist


def test_alarm_noise_of_the_real_stage_a_series_does_not_fire():
    """The dev macro of stage A moved by <= 0.04 between consecutive evaluations after step 250 (sd of one evaluation ~0.1): with eps 0.3 over k = 3 it needs a 3-sigma fall."""
    macro = [40.43, 40.46, 40.47, 40.50]
    h = HackingAlarm(3, 0.3, 0.005)
    d = feed(h, [(250 + 50 * i, m, 0.70, -0.5 + 0.01 * i) for i, m in enumerate(macro)])
    assert d["fire"] is False and d["blocked"] == "chrf_not_fallen"


# ------------------------------------------------------------------------------------------------ combined stop
def test_combined_stop_needs_the_heldout_comet_to_be_flat_too():
    chrf = [(25, 30.0), (50, 30.0), (75, 30.0), (100, 30.0)]                    # flat chrF++ for three evaluations after the first
    comet_rising = [0.700, 0.705, 0.710, 0.715]                                 # +0.005 per evaluation: keeps improving by >= 0.004
    comet_flat = [0.700, 0.701, 0.700, 0.701]
    for require, comet, expect in ((False, comet_rising, True), (True, comet_rising, False), (True, comet_flat, True), (False, comet_flat, True)):
        d = DevPatience(3, 0.15, 0.98, eps_comet=0.004, require_comet=require)
        fired = False
        for (s, c), m in zip(chrf, comet):
            d.observe(s, c, 20.0, "mixed", m)
            fired = d.check(s, "mixed", 0, 50, True, 1.0, 1.0)["fire"]
        assert fired is expect, (require, comet)
    d = DevPatience(3, 0.15, 0.98, eps_comet=0.004, require_comet=True)          # the comet value is missing at some evaluations (monitor failed): the chrF++ rule decides then
    for s, c in chrf:
        d.observe(s, c, 20.0, "mixed", None)
    assert d.check(100, "mixed", 0, 50, True, 1.0, 1.0)["fire"] is True
    e = DevPatience(3, 0.15, 0.98, 0.004, True)
    e.load_state_dict({"anchor": {"macro": 30.0, "weak": 20.0}, "stale": 1, "hist": [], "guard_events": [], "triggered": None})       # a patch-2 checkpoint has no comet anchor
    assert e.anchor["comet"] is None and e.stale == 1


# ------------------------------------------------------------------------------------------------ per-language stall flags
def test_lang_stall_flags_after_k_evaluations_without_a_gain():
    ls = LangStall(2, 0.5, 0.01)
    assert ls.observe({"yor": 20.0, "hau": 30.0}, {"yor": 0.60, "hau": 0.70}) == {"yor": False, "hau": False}
    assert ls.observe({"yor": 20.1, "hau": 31.0}, {"yor": 0.601, "hau": 0.70}) == {"yor": False, "hau": False}          # hau gained 1.0 chrF++ (resets), yor stale 1
    assert ls.observe({"yor": 20.2, "hau": 31.1}, {"yor": 0.60, "hau": 0.72}) == {"yor": True, "hau": False}            # yor stale 2 = stalled; hau gained 0.02 AfriCOMET
    assert ls.observe({"yor": 21.0}, None)["yor"] is False                                                                 # a gain resets
    g = LangStall(2, 0.5, 0.01)
    g.load_state_dict(json.loads(json.dumps(ls.state_dict())))
    assert g.stale == ls.stale and g.anchor == ls.anchor


# ------------------------------------------------------------------------------------------------ the monitor
class ScriptedScorer:
    def __init__(self):
        self.calls, self.closed = [], 0

    def __call__(self, srcs, mts, refs=None):
        self.calls.append((list(srcs), list(mts), list(refs)))
        return [0.5 + 0.1 * (len(m) % 3) for m in mts]

    def close(self):
        self.closed += 1


def preds():
    rows = []
    for iso in ("yor", "hau", "xxx"):
        for i in range(6):
            rows.append({"set": "bench", "iso": iso, "id": str(i), "src": f"source {i}", "hyp": "x" * (i + 1), "ref": f"reference {i}", "finished": True})
    for iso in ("yor",):
        for i in range(4):
            rows.append({"set": "weak_dev", "iso": iso, "id": str(i), "src": f"s {i}", "hyp": "y" * (i + 2), "ref": f"r {i}", "finished": True})
    return rows


def test_monitor_scores_supported_languages_only_and_reports_macro_sd_and_counts():
    sc = ScriptedScorer()
    hc = HeldoutComet(scorer=sc, bootstrap=100, languages={"yor", "hau"})
    out = hc.score(preds())
    assert len(sc.calls) == 1 and len(sc.calls[0][0]) == 16                              # "xxx" is not an AfriCOMET language; sources, translations and references go together
    assert sc.calls[0][2][0] == "reference 0"
    assert out["bench/comet_n/yor"] == 6 and out["bench/comet_n/hau"] == 6 and "bench/comet/xxx" not in out
    assert out["bench/comet_macro"] == pytest.approx((out["bench/comet/yor"] + out["bench/comet/hau"]) / 2)
    assert 0.0 < out["bench/comet_macro_sd"] < 0.2 and out["bench_aux/weak_dev/comet_macro"] == pytest.approx(out["bench_aux/weak_dev/comet/yor"])
    assert out["heldout_comet/languages"] == 2 and out["heldout_comet/n"] == 16 and out["heldout_comet/seconds"] >= 0
    assert sc.closed == 0 and hc.score([p for p in preds() if p["iso"] == "xxx"]) == {}  # an injected scorer is not owned: not closed; nothing supported -> {}


def test_monitor_failure_is_swallowed_and_counted():
    class Boom:
        def __call__(self, *a, **k):
            raise RuntimeError("out of memory")
    hc = HeldoutComet(scorer=Boom(), languages={"yor"})
    assert hc.score(preds()) == {} and hc.failures == 1


def test_bootstrap_sd_matches_the_standard_error_of_the_macro():
    rng = np.random.default_rng(1)
    per = {l: rng.normal(0.6, 0.15, 45).tolist() for l in "abcdefghijklmnopqrs"}          # 19 languages x 45 sentences, sentence sd 0.15
    m = macro_with_sd(per, 400, 3)
    assert m["macro"] == pytest.approx(np.mean([np.mean(v) for v in per.values()]))
    assert m["sd"] == pytest.approx(0.15 / np.sqrt(45 * 19), rel=0.25)                       # ~0.005: the noise of the macro by sentence sampling
    assert macro_with_sd(per, 400, 3) == m                                                 # deterministic


def test_hub_id_and_defaults():
    from gemma_mt.rl.comet import REF_MODEL_ID
    from gemma_mt.rl.dev_comet import REF_MODEL_ID as MID

    assert MID == REF_MODEL_ID == "masakhane/africomet-stl-1.1"


# ------------------------------------------------------------------------------------------------ the loop
def _rows(path):
    return [json.loads(l) for l in open(path)]


def test_config_validation():
    pytest.importorskip("torch")
    from gemma_mt.rl.train_rl import RLConfig, validate_config

    c = RLConfig()
    assert not (c.heldout_comet or c.hacking_alarm or c.lang_stall_log or c.save_dev_predictions or c.early_stop_dev_require_comet)
    validate_config(RLConfig(heldout_comet=True, hacking_alarm=True, benchmark_eval_path="x", early_stop_dev_require_comet=True, lang_stall_log=True))
    for bad in ({"heldout_comet": True}, {"hacking_alarm": True, "benchmark_eval_path": "x"}, {"early_stop_dev_require_comet": True}, {"heldout_comet": True, "benchmark_eval_path": "x", "hacking_alarm": True, "hacking_alarm_k": 0}):
        with pytest.raises(ValueError):
            validate_config(RLConfig(**bad))


def test_loop_with_monitor_predictions_alarm_and_language_history():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from gemma_mt.rl.train_rl import run_smoke

    s = run_smoke(sft_steps=60, benchmark=True, overrides={"steps": 8, "eval_every": 2, "benchmark_eval_every": 2, "select_min_finished_rate": 0.0, "heldout_comet": True, "save_dev_predictions": True,
                                                          "hacking_alarm": True, "hacking_alarm_k": 2, "hacking_alarm_eps_chrf": -1000.0, "hacking_alarm_eps_comet": -1000.0, "lang_stall_log": True,
                                                          "lang_stall_k": 1, "early_stop_dev": True, "early_stop_dev_require_comet": True, "early_stop_dev_k": 99, "early_stop_dev_min_steps": 0})
    rows = _rows(s["workdir"] + "/rl/metrics.jsonl")
    dev = [r for r in rows if "bench/comet_macro" in r]
    assert dev and all(0.0 <= r["bench/comet_macro"] <= 1.0 and r["bench/comet_macro_sd"] >= 0 and r["heldout_comet/seconds"] > 0 and r["heldout_comet/languages"] == 3 for r in dev)
    files = sorted(os.listdir(s["workdir"] + "/rl/dev_predictions"))
    assert files and files[0].startswith("step_")
    preds_ = _rows(s["workdir"] + "/rl/dev_predictions/" + files[-1])
    assert preds_ and {"set", "iso", "id", "src", "hyp", "ref", "finished", "step"} <= set(preds_[0]) and len(preds_) == dev[-1]["heldout_comet/n"]
    byl = _rows(s["workdir"] + "/rl/dev_by_language.jsonl")
    assert byl and set(byl[0]["bench"]["chrf"]) == {"yor", "hau", "ibo"} and byl[0]["bench"]["comet"] and "stalled" in byl[0]
    assert any("dev_stalled_count" in r for r in rows)
    chk = _rows(s["workdir"] + "/rl/hacking_alarm_checks.jsonl")
    assert chk and all(c["blocked"] in (None, "history", "qe_not_rising", "no_qe", "already_triggered") for c in chk)       # eps -1000: the fall conditions are trivially true, only the QE slope can block it
    summ = json.load(open(s["workdir"] + "/rl/summary.json"))
    assert summ["heldout_comet"]["failures"] == 0 and "lang_stall" in summ and summ["stop_reason"] in ("hacking_alarm", "max_steps")
    if summ["stop_reason"] == "hacking_alarm":
        assert json.load(open(s["workdir"] + "/rl/WINDDOWN.reason.json"))["reason"] == "hacking_alarm" and summ["hacking_alarm"]["triggered"]
