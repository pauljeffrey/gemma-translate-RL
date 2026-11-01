# SPDX-License-Identifier: AGPL-3.0-or-later
"""2026-10-29 (Amendment 4, agent T2): patience-based early stop, surgical cap extension, Muon in the RL trainer, vLLM-only evaluation, private hub repos."""
import os
import json
import math
import random

import pytest

from gemma_mt.rl.run_controls import PatienceStop, RunControls, extension_decision


# ------------------------------------------------------------------------------------------------------------------ PatienceStop
def _feed(p, series, start=1, fin=1.0, lr=1.0, stage="b64", loop=None, gfin=None):
    for i, v in enumerate(series):
        p.observe(start + i, v, fin, lr, stage, None if loop is None else loop[i], gfin)


def _saturating(n, rate=0.03, top=1.0, sd=0.0, seed=0):
    rng = random.Random(seed)
    return [top * (1 - math.exp(-rate * (i + 1))) + rng.gauss(0, sd) for i in range(n)]


def test_patience_fires_on_a_plateau_after_patience_and_min_steps_and_not_on_a_rise():
    p = PatienceStop(patience=30, block=10, eps_reward=0.02)
    _feed(p, _saturating(200, 0.05, 1.0, 0.02, 1))
    d = p.check(200, "b64", 0, 30, True)
    assert d["fire"] and d["steps_since_improve"] >= 30 and d["last_improve_by"] in ("reward", "finished")
    # the clock starts at the last improvement: the first step at which it fires is exactly last_improve + patience
    q = PatienceStop(patience=30, block=10, eps_reward=0.02)
    fired = None
    for i, v in enumerate(_saturating(200, 0.05, 1.0, 0.02, 1)):
        q.observe(i + 1, v, 1.0, 1.0, "b64")
        if fired is None and q.check(i + 1, "b64", 0, 30, True)["fire"]:
            fired = i + 1
    assert fired == q.last_improve_step + 30 or fired < 200 and fired >= 30 + 10
    r = PatienceStop(patience=30, block=10, eps_reward=0.02)
    _feed(r, [0.01 * i for i in range(200)])                         # rises 0.1 per block for ever
    d = r.check(200, "b64", 0, 30, True)
    assert not d["fire"] and d["blocked"] == "improving" and d["steps_since_improve"] < 30


def test_slow_creep_below_epsilon_does_not_reset_the_clock_but_a_new_best_does():
    p = PatienceStop(patience=40, block=10, eps_reward=0.05)
    _feed(p, [0.5 + 0.00005 * i for i in range(300)])                # +0.015 over the whole run: never beats the first block mean by 0.05
    assert p.last_improve_step == 10 and p.check(300, "b64", 0, 40, True)["fire"]
    q = PatienceStop(patience=40, block=10, eps_reward=0.05)
    _feed(q, [0.5] * 150 + [0.7] * 20)                                 # a real step up at 150
    assert q.last_improve_step is not None and 150 <= q.last_improve_step <= 160
    assert not q.check(170, "b64", 0, 40, True)["fire"]


def test_greedy_loop_rate_and_finished_rate_count_as_improvements():
    p = PatienceStop(patience=30, block=10, eps_loop=0.005)
    loop = [0.2] * 60 + [0.1] * 60 + [0.1] * 60                      # reward flat, the greedy loop rate falls at step ~60-70
    _feed(p, [0.5] * 180, loop=loop)
    assert p.last_improve_by == "loop" and 60 <= p.last_improve_step <= 75
    d = p.check(180, "b64", 0, 30, True)
    assert d["fire"] and d["steps_since_improve"] >= 100
    # finished rate rising from 0.97 to 0.999 is an improvement too
    q = PatienceStop(patience=30, block=10, eps_finished=0.002, min_finished=0.0)
    for i in range(100):
        q.observe(i + 1, 0.5, 0.97 if i < 50 else 0.999, 1.0, "b64")
    assert q.last_improve_by == "finished" and q.last_improve_step >= 50


def test_patience_never_fires_in_a_confounded_window_or_below_the_floors():
    def flat(n=200, **kw):
        p = PatienceStop(patience=30, block=10, **kw)
        _feed(p, [0.5] * n)
        return p

    p = flat()
    assert p.check(200, "b64", 0, 30, True)["fire"]
    assert p.check(25, "b64", 0, 30, True)["blocked"] == "min_steps"
    assert p.check(200, "b64", 190, 30, True)["blocked"] == "min_steps"            # stage began at 190
    assert p.check(200, "b64", 0, 30, False)["blocked"] == "not_last_stage"
    for ev in (171, 185, 200, 170):
        p = flat()
        p.note_guard_event(ev)
        assert p.check(200, "b64", 0, 30, True)["blocked"] == "guard_event", ev
    p = flat()
    p.note_guard_event(150)
    assert p.check(200, "b64", 0, 30, True)["fire"]
    p = PatienceStop(patience=30, block=10)
    _feed(p, [0.5] * 180, lr=1.0)
    _feed(p, [0.5] * 20, start=181, lr=0.75)
    assert p.check(200, "b64", 0, 30, True)["blocked"] == "lr_changed"
    p = PatienceStop(patience=30, block=10, min_lr_scale=0.5)
    _feed(p, [0.5] * 200, lr=0.25)
    assert p.check(200, "b64", 0, 30, True)["blocked"] == "lr_low"
    p = PatienceStop(patience=30, block=10)
    _feed(p, [0.5] * 200, fin=0.98)
    assert p.check(200, "b64", 0, 30, True)["blocked"] == "finished"
    p = PatienceStop(patience=30, block=10)
    _feed(p, [0.5] * 200, gfin=0.9)                                                 # the greedy member does not finish
    assert p.check(200, "b64", 0, 30, True)["blocked"] == "finished"
    p = PatienceStop(patience=30, block=10)
    _feed(p, [0.5] * 170)
    _feed(p, [0.5] * 30, start=181)                                                 # hole (steps 171-180) inside the patience window
    assert p.check(200, "b64", 0, 30, True)["blocked"] == "history"


def test_patience_state_round_trip_new_stage_resets_and_trigger_is_final():
    p = PatienceStop(patience=30, block=10)
    _feed(p, _saturating(120, 0.05, 1.0, 0.02, 3))
    p.note_guard_event(5)
    q = PatienceStop(patience=30, block=10)
    q.load_state_dict(json.loads(json.dumps(p.state_dict())))
    assert q.check(120, "b64", 0, 30, True) == p.check(120, "b64", 0, 30, True)
    _feed(p, [1.0] * 10, start=121)
    _feed(q, [1.0] * 10, start=121)
    assert p.state_dict() == q.state_dict()
    p.observe(131, 0.0, 1.0, 1.0, "b256")                                           # another stage inside the same run: clocks restart
    assert p.last_improve_step is None and p.check(131, "b256", 130, 1, True)["blocked"] == "no_signal_yet"
    q.triggered = {"x": 1}
    assert q.check(130, "b64", 0, 30, True)["blocked"] == "already_triggered"


# ------------------------------------------------------------------------------------------------------------------ extension
def test_extension_decision_blocks_of_a_quarter_up_to_fifty_percent_only_while_improving():
    d = extension_decision(400, 400, 400, 100, 0.5, 10, 25)
    assert d["extend"] and d["new_cap"] == 500 and d["reason"] == "improving" and d["extension_room"] == 200
    d2 = extension_decision(500, 400, 500, 100, 0.5, 5, 25)
    assert d2["extend"] and d2["new_cap"] == 600
    d3 = extension_decision(600, 400, 600, 100, 0.5, 5, 25)
    assert not d3["extend"] and d3["reason"] == "max_extension_reached" and d3["new_cap"] == 600
    assert not extension_decision(400, 400, 400, 100, 0.5, 25, 25)["extend"] and extension_decision(400, 400, 400, 100, 0.5, 25, 25)["reason"] == "not_improving"
    assert extension_decision(400, 400, 400, 100, 0.5, None, 25)["reason"] == "no_signal"
    d4 = extension_decision(150, 150, 150, 40, 0.5, 3, 25)               # 150 -> 190 -> 225 (the last block is shortened to the +50 % limit)
    d5 = extension_decision(190, 150, 190, 40, 0.5, 3, 25)
    assert d4["new_cap"] == 190 and d5["new_cap"] == 225 and d5["extend_steps"] == 35
    assert not extension_decision(225, 150, 225, 40, 0.5, 3, 25)["extend"]


# ------------------------------------------------------------------------------------------------------------------ config validation
def test_config_validation_of_the_new_fields():
    from gemma_mt.rl.train_rl import RLConfig, validate_config

    c = RLConfig()
    assert not c.early_stop and not c.extend_if_improving and c.optimizer == "adamw" and not c.eval_vllm_only and c.planned_winddown_last == 0
    validate_config(c)
    ok = dict(early_stop=True, early_stop_patience=30, early_stop_min_steps=30)
    validate_config(RLConfig(**ok))
    validate_config(RLConfig(**ok, extend_if_improving=True, extend_steps=25))
    validate_config(RLConfig(**ok, extend_if_improving=True, extend_steps=40, planned_winddown_last=30, steps=150))
    for bad in (dict(run_controls=False), dict(early_stop_block=1), dict(early_stop_patience=5), dict(early_stop_min_steps=5), dict(early_stop_eps_reward=-1.0),
                dict(early_stop_min_finished=0.0), dict(early_stop_winddown=0), dict(extend_if_improving=True),                       # extension without steps
                dict(extend_if_improving=True, extend_steps=10, extend_max_frac=0.6), dict(extend_if_improving=True, extend_steps=10, extend_recent=0),
                dict(extend_if_improving=True, extend_steps=20, planned_winddown_last=30, steps=150), dict(planned_winddown_last=200, steps=150), dict(planned_winddown_last=-1)):
        with pytest.raises(ValueError):
            validate_config(RLConfig(**{**ok, **bad}))
    with pytest.raises(ValueError):
        validate_config(RLConfig(extend_if_improving=True, extend_steps=10))                  # needs early_stop
    validate_config(RLConfig(optimizer="muon"))
    for bad in (dict(optimizer="sgd"), dict(optimizer="muon", muon_momentum=1.0), dict(optimizer="muon", muon_ns_steps=0), dict(eval_vllm_only=True),
                dict(hub_repo_id=(os.environ['HF_ORG'] + "/x"), hub_private=False)):
        with pytest.raises(ValueError):
            validate_config(RLConfig(**bad))
    validate_config(RLConfig(sampler="vllm", eval_vllm_only=True, vllm_max_model_len=8192))
    validate_config(RLConfig(hub_repo_id=(os.environ['HF_ORG'] + "/x"), hub_private=False, hub_allow_public=True))


def test_hf_evaluation_fallback_is_refused_when_vllm_only(tmp_path):
    pytest.importorskip("torch")
    from gemma_mt.eval.eval_translation import generate_translations
    from gemma_mt.rl.train_rl import RLConfig, evaluate_benchmark, evaluate_policy

    with pytest.raises(RuntimeError):
        generate_translations(None, None, [], stop_ids=[1], pad_id=0, device="cpu", require_vllm=True)
    cfg = RLConfig(sampler="vllm", eval_vllm_only=True)
    with pytest.raises(RuntimeError):
        evaluate_benchmark(None, None, [], cfg, [1], 0, "cpu", None, sampler=None)
    with pytest.raises(RuntimeError):
        evaluate_policy(None, None, None, None, cfg, [1], 0, "cpu", None, sampler=None)


def test_pusher_confirms_private(tmp_path):
    from gemma_mt.hub.push_checkpoint import CheckpointPusher

    calls = []

    class Api:
        def create_repo(self, *a, **k):
            calls.append(("create_repo", k.get("private")))

        def update_repo_settings(self, *a, **k):
            calls.append(("update_repo_settings", k.get("private")))

        def upload_folder(self, *a, **k):
            calls.append(("upload_folder",))

        def super_squash_history(self, *a, **k):
            pass

        def list_repo_refs(self, *a, **k):
            return type("R", (), {"tags": []})()

        def create_tag(self, *a, **k):
            pass

        def create_branch(self, *a, **k):
            pass

    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text("{}")
    pu = CheckpointPusher((os.environ['HF_ORG'] + "/x"), private=True, stage="rl")
    pu._client = lambda: Api()
    pu._upload(str(d), 1, False)
    assert ("create_repo", True) in calls and ("update_repo_settings", True) in calls
    assert calls.index(("update_repo_settings", True)) < calls.index(("upload_folder",))


# ------------------------------------------------------------------------------------------------------------------ Muon
def _tiny_policy():
    from tests.test_rl_loop import build_tiny_model
    return build_tiny_model(seed=3, attn_implementation="sdpa")


def test_muon_partition_matches_the_sft_convention_and_state_round_trips():
    import torch

    from gemma_mt.rl.optim import CombinedOptimizer, build_rl_optimizer, optimizer_kind_of_state
    from gemma_mt.rl.train_rl import RLConfig

    policy = _tiny_policy()
    dev = torch.device("cpu")
    ad = build_rl_optimizer(policy, RLConfig(optimizer="adamw", learning_rate=1e-3), dev)
    assert isinstance(ad, torch.optim.AdamW) and optimizer_kind_of_state(ad.state_dict()) == "adamw"
    cfg = RLConfig(optimizer="muon", learning_rate=1e-3)
    mu = build_rl_optimizer(policy, cfg, dev)
    assert isinstance(mu, CombinedOptimizer) and optimizer_kind_of_state(mu.state_dict()) == "muon"
    muon_ids = {id(p) for g in mu.optimizers[0].param_groups for p in g["params"]}
    adam_ids = {id(p) for g in mu.optimizers[1].param_groups for p in g["params"]}
    names = {id(p): n for n, p in policy.named_parameters()}
    assert muon_ids and all(names[i].count(".layers.") and dict(policy.named_parameters())[names[i]].ndim == 2 for i in muon_ids)
    assert not (muon_ids & adam_ids) and muon_ids | adam_ids == {id(p) for p in policy.parameters()}
    assert any("embed_tokens" in names[i] for i in adam_ids) and any(dict(policy.named_parameters())[names[i]].ndim == 1 for i in adam_ids)
    # one step changes every parameter group, set lr reaches both halves, state round trip reproduces the next step
    for g in mu.param_groups:
        g["lr"] = 5e-4
    assert {g["lr"] for g in mu.optimizers[0].param_groups} == {5e-4} and {g["lr"] for g in mu.optimizers[1].param_groups} == {5e-4}
    x = torch.randint(5, 50, (2, 12))

    def loss():
        return policy(input_ids=x).logits.float().pow(2).mean()

    loss().backward()
    mu.step()
    mu.zero_grad()
    import copy

    sd = copy.deepcopy(mu.state_dict())                                  # optimizer.state_dict() references the live state tensors

    snap = copy.deepcopy(policy.state_dict())
    loss().backward()
    mu.step()
    after_a = copy.deepcopy(policy.state_dict())
    policy.load_state_dict(snap)
    mu2 = build_rl_optimizer(policy, cfg, dev)
    for g in mu2.param_groups:
        g["lr"] = 5e-4
    mu2.load_state_dict(copy.deepcopy(sd))
    policy.zero_grad()
    loss().backward()
    mu2.step()
    for k in after_a:
        assert torch.allclose(after_a[k].float(), policy.state_dict()[k].float(), atol=1e-6), k
    with pytest.raises(ValueError):
        mu2.load_state_dict(ad.state_dict())                              # an AdamW state is refused (fresh optimizer per stage)


def test_muon_rl_smoke_with_the_vllm_stand_in_pushes_weights_and_resumes_with_the_optimizer_state(tmp_path):
    pytest.importorskip("torch")
    import torch

    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.rl.comet import FakeScorer
    from gemma_mt.rl.train_rl import RLConfig, run_smoke, train_rl
    from gemma_mt.utils import dataclass_from_dict

    summ = run_smoke(workdir=str(tmp_path / "w"), sft_steps=150, vllm_fake=True, overrides=dict(steps=4, optimizer="muon", learning_rate=1e-4, warmup_steps=1, eval_every=2, save_every=2, eval_vllm_only=True))
    assert summ["steps"] == 4 and summ["engine_pushes"] >= 4                              # the engine got the new weights after every optimizer step
    ck = torch.load(tmp_path / "w" / "rl" / "last" / "trainer_state.pt", map_location="cpu", weights_only=False)
    assert "combined" in ck["optimizer"] and ck["state"]["optimizer_kind"] == "muon"
    muon_state = ck["optimizer"]["combined"][0]["state"]
    assert muon_state and all("momentum_buffer" in v for v in muon_state.values())
    # resume: the Muon state is restored and training continues
    cfgd = json.load(open(tmp_path / "w" / "rl" / "config.json"))["config"]
    cfgd.update(steps=6, resume=True)
    vs = None
    from gemma_mt.rl.vllm_sampler import HFBackedEngine, VLLMSampler

    vs = VLLMSampler(HFBackedEngine(cfgd["model_name"]), seed=cfgd["seed"])
    s2 = train_rl(dataclass_from_dict(RLConfig, cfgd), tokenizer=FakeGemmaTokenizer(), scorer=FakeScorer(), vllm_sampler=vs)
    assert s2["steps"] == 6
    # an AdamW run in the same out_dir is refused loudly
    cfgd.update(steps=8, optimizer="adamw")
    with pytest.raises(ValueError):
        train_rl(dataclass_from_dict(RLConfig, cfgd), tokenizer=FakeGemmaTokenizer(), scorer=FakeScorer(), vllm_sampler=VLLMSampler(HFBackedEngine(cfgd["model_name"]), seed=cfgd["seed"]))


# ------------------------------------------------------------------------------------------------------------------ trainer wiring
def _rows(tmp_path):
    from tests.test_kl_guard_plateau import _rows as rows
    return rows(tmp_path)


def _recs(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def test_early_stop_ends_the_run_through_the_winddown_and_records_the_evidence(tmp_path):
    pytest.importorskip("torch")
    from gemma_mt.rl.train_rl import run_smoke

    summ = run_smoke(workdir=str(tmp_path / "w"), sft_steps=150, overrides=dict(
        steps=60, eval_every=10, eval_prompts=4, save_every=0, early_stop=True, early_stop_patience=8, early_stop_block=4, early_stop_min_steps=8, early_stop_min_finished=0.001,
        early_stop_eps_reward=10.0, early_stop_eps_finished=10.0, early_stop_eps_loop=10.0, early_stop_winddown=3))                    # eps 10: only the first smoothed value ever improves
    out = tmp_path / "w" / "rl"
    assert summ["stop_reason"] == "early_stop" and summ["steps"] < 60 and summ["early_stop"]["triggered"]
    ev = summ["early_stop"]["evidence"]
    assert ev["step"] >= 8 and ev["steps_since_improve"] >= 8 and ev["patience"] == 8
    assert summ["winddown"]["completed"] and summ["steps"] == ev["step"] + 3
    assert json.load(open(out / "WINDDOWN.reason.json"))["reason"] == "early_stop"
    rec = [r for r in _recs(out / "metrics.jsonl") if "reward" in r]
    assert all("early_stop/steps_since_improve" in r for r in rec)


def test_extension_in_the_loop_keeps_the_bin_logs_every_decision_and_stops_at_plus_fifty_percent(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from gemma_mt.rl import data as D
    from gemma_mt.rl import train_rl as T
    from gemma_mt.rl.run_controls import PatienceStop
    from gemma_mt.rl.train_rl import run_smoke

    monkeypatch.setattr(PatienceStop, "steps_since_improve", lambda self, step: 0)         # "always improving"
    seen = []

    class Sched(D.PromptSchedule):
        def bin_of_step(self, step):
            b = super().bin_of_step(step)
            seen.append((step, b))
            return b

    monkeypatch.setattr(T, "PromptSchedule", Sched)
    summ = run_smoke(workdir=str(tmp_path / "w"), sft_steps=20, overrides=dict(
        prompts_path=_rows(tmp_path), max_src_tokens=2048, length_bins=True, curriculum_steps={"b64": 6}, steps=6, eval_prompts=6, eval_min_per_bin=1, eval_every=3, save_every=0,
        prompts_per_step_by_bin={"b64": 3, "mixed": 3}, max_new_tokens_by_bin={"b64": 16, "mixed": 16}, group_size=4, micro_batch_seqs=4,
        early_stop=True, early_stop_patience=1000, early_stop_block=2, early_stop_min_steps=1000, extend_if_improving=True, extend_steps=2, extend_max_frac=0.5, extend_recent=5, reward={"max_comet_tokens": 510}))
    assert summ["steps"] == 9 and summ["stop_reason"] == "max_steps"                         # 6 + 2 + 1 (the last block is shortened to the +50 % limit)
    dec = [json.loads(l) for l in open(tmp_path / "w" / "rl" / "extensions.jsonl")]
    assert [(d["step"], d["extend"], d["new_cap"]) for d in dec] == [(6, True, 8), (8, True, 9), (9, False, 9)]
    assert dec[-1]["reason"] == "max_extension_reached" and summ["cap"]["final"] == 9 and summ["cap"]["base"] == 6
    assert {b for s, b in seen if s < 9} == {"b64"}                                          # the extension never leaks into the mixed stage


def test_extension_not_granted_when_not_improving_and_restart_restores_the_cap(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.rl.comet import FakeScorer
    from gemma_mt.rl.run_controls import PatienceStop
    from gemma_mt.rl.train_rl import RLConfig, run_smoke, train_rl
    from gemma_mt.utils import dataclass_from_dict

    monkeypatch.setattr(PatienceStop, "steps_since_improve", lambda self, step: 99)
    kw = dict(early_stop=True, early_stop_patience=1000, early_stop_block=2, early_stop_min_steps=1000, extend_if_improving=True, extend_steps=2, extend_recent=5, save_every=1, eval_every=0)
    summ = run_smoke(workdir=str(tmp_path / "w"), sft_steps=20, overrides=dict(steps=4, **kw))
    assert summ["steps"] == 4 and summ["cap"]["final"] == 4
    d = [json.loads(l) for l in open(tmp_path / "w" / "rl" / "extensions.jsonl")]
    assert len(d) == 1 and not d[0]["extend"] and d[0]["reason"] == "not_improving"
    monkeypatch.setattr(PatienceStop, "steps_since_improve", lambda self, step: 0)
    cfgd = json.load(open(tmp_path / "w" / "rl" / "config.json"))["config"]
    cfgd.update(steps=4, resume=True)
    s2 = train_rl(dataclass_from_dict(RLConfig, cfgd), tokenizer=FakeGemmaTokenizer(), scorer=FakeScorer())
    assert s2["steps"] == 4                                                                    # resumed at the end of the (already complete) base cap: no decision is taken twice


def test_planned_winddown_moves_with_an_extension(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from gemma_mt.rl.run_controls import PatienceStop
    from gemma_mt.rl.train_rl import run_smoke

    monkeypatch.setattr(PatienceStop, "steps_since_improve", lambda self, step: 0)
    summ = run_smoke(workdir=str(tmp_path / "w"), sft_steps=20, overrides=dict(
        steps=10, save_every=0, eval_every=0, early_stop=True, early_stop_patience=1000, early_stop_block=2, early_stop_min_steps=1000, extend_if_improving=True, extend_steps=4, extend_max_frac=0.5,
        extend_recent=5, planned_winddown_last=3))
    dec = [json.loads(l) for l in open(tmp_path / "w" / "rl" / "extensions.jsonl")]
    assert [(d["step"], d["extend"], d["new_cap"]) for d in dec] == [(7, True, 14), (11, True, 15), (12, False, 15)]
    assert summ["steps"] == 15 and summ["stop_reason"] == "planned_winddown" and summ["winddown"]["started_at_step"] == 12
    lr = [r["lr"] for r in _recs(tmp_path / "w" / "rl" / "metrics.jsonl") if "reward" in r]
    assert lr[11] > lr[12] > lr[13] > 0 or lr[11] >= lr[12] > lr[13]                          # the final 3 steps decay


def test_planned_winddown_without_extension_matches_the_old_external_watcher(tmp_path):
    pytest.importorskip("torch")
    from gemma_mt.rl.train_rl import run_smoke

    summ = run_smoke(workdir=str(tmp_path / "w"), sft_steps=20, overrides=dict(steps=8, save_every=0, eval_every=0, planned_winddown_last=3))
    assert summ["steps"] == 8 and summ["winddown"]["started_at_step"] == 5 and summ["winddown"]["completed"]
