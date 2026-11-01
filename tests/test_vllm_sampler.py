# SPDX-License-Identifier: AGPL-3.0-or-later
"""The vLLM rollout sampler (D-28) without vLLM: scripted / HF-backed stand-in engines, the worker protocol over a real socket, config."""
import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gemma_mt import format as fmt
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
from gemma_mt.modeling import build_tiny_model, save_model
from gemma_mt.rl.sampling import generate_completions
from gemma_mt.rl.vllm_engine import EngineConfig, RawCompletion, SamplerError, WorkerEngine
from gemma_mt.rl.vllm_sampler import HFBackedEngine, HFSampler, VLLMSampler, derive_seeds, policy_bf16_weights

TOK = FakeGemmaTokenizer()
STOP = [106, 1]


class ScriptedEngine:
    """Returns the rows it was given; records the specs it was asked for."""

    def __init__(self, rows=None, fail=None):
        self.rows, self.fail, self.specs, self.pushed = rows, fail, [], []

    def generate(self, prompts, spec):
        self.specs.append((list(map(list, prompts)), spec))
        if self.fail:
            raise self.fail
        return self.rows(prompts, spec) if callable(self.rows) else self.rows

    def push_weights(self, named):
        self.pushed.append({n: t.dtype for n, t in named})
        return {"seconds": 0.0}

    def close(self):
        pass


def _sampler(rows=None, fail=None, synced=True):
    eng = ScriptedEngine(rows, fail)
    s = VLLMSampler(eng, seed=5)
    if synced:
        s._version = 0
    return s, eng


def _gen(s, prompts, G=2, mnt=8, **kw):
    return s.generate(prompts, group_size=G, max_new_tokens=mnt, temperature=1.0, top_p=1.0, top_k=0, stop_ids=STOP, pad_id=0, **kw)


def test_derive_seeds_are_deterministic_distinct_and_leave_room_for_the_group():
    a = derive_seeds(1234, 7, 20)
    assert a == derive_seeds(1234, 7, 20)
    assert a != derive_seeds(1234, 8, 20) and a != derive_seeds(1235, 7, 20)
    assert len(set(a)) == 20 and all(0 <= x < 2 ** 31 for x in a)
    assert min(b - a_ for a_, b in zip(a, a[1:])) >= 64            # the G children of a prompt use seed .. seed+G-1
    assert derive_seeds(1, 0, 0) == []


def test_rollout_contract_rows_cut_at_first_stop_and_truncation_reported():
    prompts = [[2, 10, 11], [2, 12]]
    rows = [[RawCompletion([300, 301, 106], "stop"), RawCompletion([300] * 8, "length")],
            [RawCompletion([400, 1], "stop", 1), RawCompletion([401, 402, 106], "stop")]]
    s, eng = _sampler(rows)
    roll = _gen(s, prompts, G=2, mnt=8)
    assert roll.prompts == [prompts[0], prompts[0], prompts[1], prompts[1]]            # rows grouped, prompt i owns i*G..i*G+G-1
    assert roll.completions == [[300, 301, 106], [300] * 8, [400, 1], [401, 402, 106]]    # verbatim, stop id included
    assert roll.finished == [True, False, True, True]
    assert roll.post_stop_tokens == [0, 0, 0, 0] and roll.sampler_logp is None
    spec = eng.specs[0][1]
    assert (spec.n, spec.temperature, spec.top_p, spec.top_k, spec.max_tokens, spec.stop_token_ids) == (2, 1.0, 1.0, 0, 8, (106, 1))
    assert spec.seeds == derive_seeds(5, 0, 2) and s.stats["finish_length"] == 1 and s.stats["finish_stop"] == 3


def test_greedy_asks_for_temperature_zero_and_one_completion_per_prompt():
    s, eng = _sampler([[RawCompletion([5, 106], "stop")], [RawCompletion([6, 106], "stop")]])
    roll = _gen(s, [[2], [2, 3]], G=8, greedy=True)
    assert len(roll.completions) == 2 and roll.prompts == [[2], [2, 3]]
    spec = eng.specs[0][1]
    assert spec.n == 1 and spec.temperature == 0.0 and spec.seeds is None


def test_stop_id_missing_from_the_ids_is_restored_and_inconsistent_answers_are_rejected():
    s, _ = _sampler([[RawCompletion([300, 301], "stop", 106)]])
    assert _gen(s, [[2]], G=1).completions == [[300, 301, 106]]                          # vLLM reported it in stop_reason only
    s, _ = _sampler([[RawCompletion([300, 301], "stop", None)]])
    with pytest.raises(SamplerError):
        _gen(s, [[2]], G=1)
    s, _ = _sampler([[RawCompletion([300, 106, 301, 106], "length")]])                     # a stop id inside, but reported as truncated
    with pytest.raises(SamplerError):
        _gen(s, [[2]], G=1)
    s, _ = _sampler([[RawCompletion([300] * 7 + [106], "length")]])                       # the stop id is the last allowed token: finished, like HF
    roll = _gen(s, [[2]], G=1, mnt=8)
    assert roll.finished == [True] and roll.completions == [[300] * 7 + [106]]
    s, _ = _sampler([[RawCompletion([300] * 20, "length")]])
    with pytest.raises(SamplerError):
        _gen(s, [[2]], G=1, mnt=8)
    s, _ = _sampler([[RawCompletion([], "stop")]])
    with pytest.raises(SamplerError):
        _gen(s, [[2]], G=1)
    s, _ = _sampler([[RawCompletion([300], "abort")]])
    with pytest.raises(SamplerError):
        _gen(s, [[2]], G=1)
    s, _ = _sampler([[RawCompletion([300, 106], "stop")]])                               # wrong number of completions for n=2
    with pytest.raises(SamplerError):
        _gen(s, [[2]], G=2)


def test_post_stop_tokens_counts_what_the_engine_kept_after_the_first_stop():
    s, _ = _sampler([[RawCompletion([300, 106, 400, 500, 106, 0], "stop")]])
    roll = _gen(s, [[2]], G=1, mnt=8)
    assert roll.completions == [[300, 106]] and roll.post_stop_tokens == [2]               # 400 and 500; the second stop id and padding are benign


def test_engine_failure_is_loud_there_is_no_fallback():
    s, _ = _sampler(fail=SamplerError("CUDA out of memory"))
    with pytest.raises(SamplerError):
        _gen(s, [[2]], G=1)


def test_generate_needs_a_sync_and_sync_is_versioned():
    s, eng = _sampler([[RawCompletion([5, 106], "stop")]], synced=False)
    with pytest.raises(SamplerError):
        _gen(s, [[2]], G=1)
    policy = build_tiny_model(seed=1)
    assert s.sync(policy, 0) >= 0.0 and len(eng.pushed) == 1
    assert s.sync(policy, 0) == 0.0 and len(eng.pushed) == 1                                 # same version: nothing to do
    s.sync(policy, 1)
    assert len(eng.pushed) == 2
    s.invalidate()
    with pytest.raises(SamplerError):
        _gen(s, [[2]], G=1)


def test_weights_are_bf16_named_like_the_policy_and_tied_weights_go_once():
    policy = build_tiny_model(seed=2)
    named = list(policy_bf16_weights(policy))
    assert [n for n, _ in named] == [n for n, _ in policy.named_parameters()]
    assert all(t.dtype == torch.bfloat16 for _, t in named)
    assert not any(n.startswith("lm_head") for n, _ in named), "lm_head is tied to the embedding: one copy only"
    assert any(n == "model.embed_tokens.weight" for n, _ in named)
    assert all(t.dtype == torch.float32 for _, t in policy_bf16_weights(policy, torch.float32))


def _perturb(model, seed, rel=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * rel * p.std())


def test_engine_samples_the_pushed_policy_not_a_stale_one(tmp_path):
    """The plumbing the trainer relies on: after sync() the engine's greedy output equals the policy's; without it, it does not."""
    policy = build_tiny_model(seed=3).eval()
    save_model(policy, None, str(tmp_path / "ckpt"))
    eng = HFBackedEngine(str(tmp_path / "ckpt"))
    s = VLLMSampler(eng, seed=1, weight_dtype=torch.float32)
    prompts = [fmt.build_prompt("Good morning", "yor", TOK), fmt.build_prompt("Water is life.", "hau", TOK)]
    _perturb(policy, 11)
    s.sync(policy, 1)
    ref = lambda: generate_completions(policy, prompts, group_size=1, max_new_tokens=12, temperature=1.0, top_p=1.0, top_k=0,
                                       stop_ids=STOP, pad_id=0, device="cpu", greedy=True)
    got = _gen(s, prompts, G=1, mnt=12, greedy=True)
    assert got.completions == ref().completions and got.finished == ref().finished
    _perturb(policy, 12, rel=0.6)                                                             # the trainer took an optimizer step
    stale = _gen(s, prompts, G=1, mnt=12, greedy=True)
    assert stale.completions != ref().completions, "the perturbation must change the output for this test to mean something"
    s.sync(policy, 2)
    assert _gen(s, prompts, G=1, mnt=12, greedy=True).completions == ref().completions
    assert eng.pushes == 2


def test_hf_backed_engine_refuses_partial_weight_updates(tmp_path):
    policy = build_tiny_model(seed=4)
    save_model(policy, None, str(tmp_path / "ckpt"))
    eng = HFBackedEngine(str(tmp_path / "ckpt"))
    with pytest.raises(SamplerError):
        eng.push_weights(list(policy_bf16_weights(policy))[:-1])


def test_seeded_sampling_is_reproducible_per_key(tmp_path):
    policy = build_tiny_model(seed=5).eval()
    save_model(policy, None, str(tmp_path / "ckpt"))
    s = VLLMSampler(HFBackedEngine(str(tmp_path / "ckpt")), seed=9, weight_dtype=torch.float32)
    s.sync(policy, 0)
    p = [fmt.build_prompt("Thank you very much.", "ibo", TOK)]
    a, b, c = _gen(s, p, G=4, mnt=10, key=3), _gen(s, p, G=4, mnt=10, key=3), _gen(s, p, G=4, mnt=10, key=4)
    assert a.completions == b.completions and a.completions != c.completions


def test_hf_sampler_is_the_original_path():
    from contextlib import nullcontext

    model = build_tiny_model(seed=6).eval()
    p = [fmt.build_prompt("The river is very wide.", "swh", TOK)]
    hf = HFSampler(model, "cpu", nullcontext)
    a = hf.generate(p, group_size=3, max_new_tokens=8, temperature=1.0, top_p=1.0, top_k=0, stop_ids=STOP, pad_id=0, greedy=True)
    b = generate_completions(model, p, group_size=3, max_new_tokens=8, temperature=1.0, top_p=1.0, top_k=0, stop_ids=STOP, pad_id=0,
                             device="cpu", greedy=True)
    assert a.completions == b.completions and hf.sync(model, 5) == 0.0


# ---------------------------------------------------------------------------------------------------- worker protocol
def test_worker_process_protocol_weights_through_shared_memory_and_failures_are_loud(tmp_path):
    policy = build_tiny_model(seed=7).eval()
    save_model(policy, None, str(tmp_path / "ckpt"))
    ec = EngineConfig(model=str(tmp_path / "ckpt"), extra={"_engine": "hf_backed"})
    eng = WorkerEngine(ec, python=sys.executable, sync_dir=str(tmp_path), log_path=str(tmp_path / "w.log"), startup_timeout=180,
                       request_timeout=120)
    try:
        s = VLLMSampler(eng, seed=3, weight_dtype=torch.float32)
        s.sync(policy, 0)
        nbytes = sum(p.numel() * 4 for p in policy.parameters())
        assert os.path.getsize(os.path.join(eng._dir, "weights.bin")) >= nbytes, "one shared-memory file holds all weights, rewritten in place"
        assert s.last_sync_info["n_tensors"] == len(list(policy.named_parameters()))
        prompts = [fmt.build_prompt("Good morning", "yor", TOK)]
        ref = generate_completions(policy, prompts, group_size=1, max_new_tokens=10, temperature=1.0, top_p=1.0, top_k=0, stop_ids=STOP,
                                   pad_id=0, device="cpu", greedy=True)
        assert _gen(s, prompts, G=1, mnt=10, greedy=True).completions == ref.completions
        _perturb(policy, 21, rel=0.6)
        s.sync(policy, 1)
        ref2 = generate_completions(policy, prompts, group_size=1, max_new_tokens=10, temperature=1.0, top_p=1.0, top_k=0, stop_ids=STOP,
                                    pad_id=0, device="cpu", greedy=True)
        assert ref2.completions != ref.completions
        assert _gen(s, prompts, G=1, mnt=10, greedy=True).completions == ref2.completions
        roll = _gen(s, prompts, G=3, mnt=6, key=1, return_logprobs=True)
        assert len(roll.completions) == 3 and all(len(l) == len(c) for l, c in zip(roll.sampler_logp, roll.completions))
        os.kill(eng.proc.pid, 9)
        with pytest.raises(SamplerError):
            _gen(s, prompts, G=1, mnt=4, greedy=True)
        with pytest.raises(SamplerError):                                                    # and it stays dead
            s.sync(policy, 2)
    finally:
        eng.close()
    assert not os.path.exists(eng._dir)


def test_worker_start_up_failure_raises_with_the_reason(tmp_path):
    ec = EngineConfig(model=str(tmp_path / "no_such_checkpoint"), extra={"_engine": "hf_backed"})
    with pytest.raises(SamplerError):
        WorkerEngine(ec, python=sys.executable, sync_dir=str(tmp_path), log_path=str(tmp_path / "w.log"), startup_timeout=120)


# ---------------------------------------------------------------------------------------------------- config / loop
def test_config_validation_for_the_sampler_switch():
    from gemma_mt.rl.train_rl import RLConfig, validate_config

    validate_config(RLConfig())
    validate_config(RLConfig(sampler="vllm"))
    with pytest.raises(ValueError):
        validate_config(RLConfig(sampler="sglang"))
    with pytest.raises(ValueError):
        validate_config(RLConfig(sampler="vllm", vllm_mode="remote"))
    with pytest.raises(ValueError):
        validate_config(RLConfig(sampler="vllm", vllm_max_model_len=1200, max_new_tokens=1024))   # prompt + completion cannot fit
    with pytest.raises(ValueError):
        validate_config(RLConfig(sampler="vllm", vllm_gpu_memory_utilization=1.5))


def test_the_shipped_rl_yaml_selects_a_valid_sampler():
    from gemma_mt.rl.train_rl import RLConfig, validate_config
    from gemma_mt.utils import dataclass_from_dict, load_yaml

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = load_yaml(os.path.join(here, "configs", "rl.yaml"))
    assert d["sampler"] in ("hf", "vllm")
    validate_config(dataclass_from_dict(RLConfig, d))
    validate_config(dataclass_from_dict(RLConfig, {**d, "sampler": "vllm"}))
    validate_config(dataclass_from_dict(RLConfig, {**d, "sampler": "hf"}))


def test_cpu_rl_loop_through_the_vllm_code_path(tmp_path):
    """Whole train_rl loop (sampling, reward, update, eval, dev benchmark) with the stand-in engine: it must be synced after every
    optimizer step, and its log-probs must match the trainer's recomputed ones (the on-policy check)."""
    from gemma_mt.rl.train_rl import run_smoke

    summary = run_smoke(workdir=str(tmp_path), sft_steps=60, benchmark=True, vllm_fake=True)
    rows = [json.loads(l) for l in open(tmp_path / "rl" / "metrics.jsonl")]
    steps = [r for r in rows if "t_sync" in r]
    assert len(steps) == 4
    assert all(r["sampler_logp_gap_mean"] < 0.05 for r in steps), [r["sampler_logp_gap_mean"] for r in steps]
    assert all(abs(r["ratio_mean"] - 1.0) < 1e-5 for r in steps)
    # baseline eval (version 0) + 4 steps + eval/benchmark passes after steps 2 and 4 (new versions only): engine saw each version once
    assert summary["engine_pushes"] == 5, summary["engine_pushes"]
    assert summary["engine_generate_calls"] >= 4 + 2
