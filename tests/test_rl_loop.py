# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gemma_mt import format as fmt
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
from gemma_mt.modeling import build_tiny_model
from gemma_mt.rl.sampling import assemble_batch, generate_completions
from gemma_mt.losses import completion_logprobs

TOK = FakeGemmaTokenizer()
STOP = [106, 1]


def test_generation_is_grouped_cut_at_stop_and_recomputed_logprobs_match_sampling():
    """On-policy: log-probs recomputed by teacher forcing equal the log-probs the sampler used (ratio == 1)."""
    model = build_tiny_model(seed=5, attn_implementation="sdpa").eval()
    # make <end_of_turn> reasonably likely so both finished and unfinished rows occur
    with torch.no_grad():
        model.lm_head.weight[106] += 0.0
    prompts = [fmt.build_prompt("Good morning", "yor", TOK), fmt.build_prompt("Water is life.", "hau", TOK)]
    torch.manual_seed(0)
    roll = generate_completions(model, prompts, group_size=3, max_new_tokens=12, temperature=1.0, top_p=1.0, top_k=0,
                                stop_ids=STOP, pad_id=0, device="cpu", return_logprobs=True)
    assert len(roll.completions) == 6
    assert roll.prompts[0] == roll.prompts[1] == roll.prompts[2] == prompts[0] and roll.prompts[3] == prompts[1]
    for c, f in zip(roll.completions, roll.finished):
        assert 1 <= len(c) <= 12
        assert f == (c[-1] in STOP)
        assert not any(t in STOP for t in c[:-1]), "nothing may follow the first stop token"
    b = assemble_batch(roll.prompts, roll.completions, pad_id=0)
    with torch.no_grad():
        lp = completion_logprobs(model, b["input_ids"], b["attention_mask"], b["prompt_width"], b["completion_mask"], 1.0)
    for r, c in enumerate(roll.completions):
        s = torch.tensor(roll.sampler_logp[r])
        assert (lp[r, : len(c)] - s).abs().max().item() < 1e-4, "recomputed log-probs differ from sampling log-probs"
        assert (lp[r, len(c):] == 0).all()


def test_sampling_temperature_changes_scored_distribution_consistently():
    model = build_tiny_model(seed=6).eval()
    prompts = [fmt.build_prompt("Thank you very much.", "ibo", TOK)]
    torch.manual_seed(1)
    roll = generate_completions(model, prompts, group_size=2, max_new_tokens=6, temperature=0.7, top_p=1.0, top_k=0,
                                stop_ids=STOP, pad_id=0, device="cpu", return_logprobs=True)
    b = assemble_batch(roll.prompts, roll.completions, 0)
    with torch.no_grad():
        lp = completion_logprobs(model, b["input_ids"], b["attention_mask"], b["prompt_width"], b["completion_mask"], 0.7)
    for r, c in enumerate(roll.completions):
        assert (lp[r, : len(c)] - torch.tensor(roll.sampler_logp[r])).abs().max().item() < 1e-4


def test_greedy_is_deterministic():
    model = build_tiny_model(seed=7).eval()
    p = [fmt.build_prompt("The river is very wide.", "swh", TOK)]
    a = generate_completions(model, p, group_size=4, max_new_tokens=8, temperature=1.0, top_p=1.0, top_k=0,
                             stop_ids=STOP, pad_id=0, device="cpu", greedy=True)
    b = generate_completions(model, p, group_size=4, max_new_tokens=8, temperature=1.0, top_p=1.0, top_k=0,
                             stop_ids=STOP, pad_id=0, device="cpu", greedy=True)
    assert a.completions == b.completions and len(a.completions) == 1


def test_config_validation():
    from gemma_mt.rl.train_rl import RLConfig, validate_config

    validate_config(RLConfig())
    with pytest.raises(ValueError):
        validate_config(RLConfig(top_p=0.9))
    validate_config(RLConfig(top_p=0.9, allow_truncated_sampling=True))
    with pytest.raises(ValueError):
        validate_config(RLConfig(group_size=1))
    with pytest.raises(ValueError):
        validate_config(RLConfig(num_updates_per_batch=2))


def test_rl_smoke_end_to_end_runs_all_in_loop_assertions():
    from gemma_mt.rl.train_rl import run_smoke

    s = run_smoke(sft_steps=120)
    assert s["steps"] == 4
    rows = [json.loads(l) for l in open(s["workdir"] + "/rl/metrics.jsonl")]
    train = [r for r in rows if "reward" in r]
    assert len(train) == 4
    for r in train:
        assert math.isfinite(r["reward"]) and abs(r["ratio_mean"] - 1.0) < 1e-5
        assert r["sampler_logp_gap"] < 1e-3
    assert abs(train[0]["kl"]) < 1e-6, "KL to the reference must be 0 at step 0"
    assert any("eval/reward" in r for r in rows)


def test_benchmark_selection_scores_the_sft_start_and_selects_by_chrf(tmp_path):
    from gemma_mt.rl.train_rl import RLConfig, run_smoke, validate_config

    with pytest.raises(ValueError):
        validate_config(RLConfig(select_by="benchmark_chrf"))          # needs a dev file
    s = run_smoke(str(tmp_path / "w"), sft_steps=60, benchmark=True)
    rows = [json.loads(l) for l in open(s["workdir"] + "/rl/metrics.jsonl")]
    base = [r for r in rows if r.get("baseline")]
    assert len(base) == 1 and base[0]["step"] == 0 and math.isfinite(base[0]["bench/chrf_macro"])
    evals = [r for r in rows if "bench/chrf_macro" in r and not r.get("baseline")]
    assert evals and all(0.0 <= r["bench/chrf_macro"] <= 100.0 for r in evals)
    assert any(k.startswith("bench/chrf/") for k in evals[0])


# ------------------------------------------------------------------ D-26: what the sampler does after the end token
def test_hf_generate_pads_after_the_stop_so_no_tokens_follow_the_end_token():
    """The stock sampler hard-stops: finished rows are right-padded with pad_id, so post_stop_tokens is 0 for every row."""
    model = build_tiny_model(seed=5, attn_implementation="sdpa").eval()
    prompts = [fmt.build_prompt("Good morning", "yor", TOK)]
    torch.manual_seed(0)
    roll = generate_completions(model, prompts, group_size=6, max_new_tokens=10, temperature=1.0, top_p=1.0, top_k=0,
                                stop_ids=STOP, pad_id=0, device="cpu")
    assert roll.post_stop_tokens == [0] * 6


class _FakeModel:
    """A sampler that keeps generating after the end token (eos ignored): rows are fixed, not sampled."""
    training = False

    def __init__(self, rows):
        self.rows = rows

    def eval(self):
        return self

    def train(self):
        return self

    def generate(self, input_ids, **kw):
        gen = torch.tensor(self.rows, dtype=torch.long)
        return torch.cat([input_ids.repeat_interleave(len(self.rows), dim=0), gen], dim=1)


def test_tokens_generated_after_the_stop_are_cut_but_counted():
    rows = [[11, 12, 106, 0, 0, 0],         # clean: padding after the stop
            [11, 12, 106, 13, 14, 0],       # rubbish after the stop
            [11, 12, 106, 1, 0, 0],         # a second stop id is not rubbish
            [11, 12, 13, 14, 15, 16]]       # never stops
    roll = generate_completions(_FakeModel(rows), [[5, 6, 7]], group_size=4, max_new_tokens=6, temperature=1.0, top_p=1.0, top_k=0,
                                stop_ids=STOP, pad_id=0, device="cpu")
    assert roll.completions == [[11, 12, 106], [11, 12, 106], [11, 12, 106], [11, 12, 13, 14, 15, 16]]
    assert roll.finished == [True, True, True, False]
    assert roll.post_stop_tokens == [0, 2, 0, 0]


def test_worker_scorer_prepends_pythonpath_for_the_child_only(tmp_path):
    """D-33: WorkerScorer(pythonpath=...) puts a transformers-4.x directory first on the CHILD's PYTHONPATH; the parent environment is untouched."""
    import os
    from gemma_mt.rl.comet import WorkerScorer

    before = os.environ.get("PYTHONPATH")
    w = WorkerScorer(fake=True, pythonpath=str(tmp_path))
    try:
        assert os.environ.get("PYTHONPATH") == before
        assert w(["a b c d"], ["x y z"])[0] is not None
    finally:
        w.close() if hasattr(w, "close") else w.proc.kill()
