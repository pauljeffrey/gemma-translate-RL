# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-38 audit (owner 2026-10-19): RL sampling and the normalisation before the reward are standard, and identical to what the evaluation sees.

(a) the sampling parameters that reach the engine are temperature 1.0, top_p 1.0, top_k 0 (= disabled in vLLM 0.31 and HF), no penalties, stop ids =
    <end_of_turn> + <eos>, n = G independent samples with distinct seeds, ids kept exactly as generated;
(b) ids -> hypothesis string is ONE chain for the reward and for the evaluation / benchmark metrics (cut at the first stop id, decode, scrub special strings,
    NFC, strip, language tag): the same ids give the identical string in both paths;
(c) the group-advantage / loss conventions (see the report): pinned numerically on a small example.
"""
import os
import unicodedata

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from gemma_mt import format as fmt
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import grpo
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
from gemma_mt.rl.sampling import Rollout
from gemma_mt.rl.train_rl import RLConfig, rollout_and_reward
from gemma_mt.rl.vllm_engine import RawCompletion
from gemma_mt.rl.vllm_sampler import VLLMSampler, derive_seeds
from gemma_mt.utils import dataclass_from_dict, load_yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = FakeGemmaTokenizer()
EOT, EOS = SPECIALS["<end_of_turn>"], SPECIALS["<eos>"]
STOP = [EOT, EOS]


class CapturingEngine:
    def __init__(self, ids_for):
        self.ids_for, self.specs = ids_for, []

    def generate(self, prompts, spec):
        self.specs.append(spec)
        return [[RawCompletion(list(self.ids_for(p, j)), "stop", None) for j in range(spec.n)] for p in prompts]

    def push_weights(self, named):
        return {}

    def close(self):
        pass


class Rows:
    def __init__(self, english, iso, n):
        self.english, self.iso = english, iso
        self.n = n

    def __len__(self):
        return self.n


def _prompts(n_prompts=3):
    from gemma_mt.rl.data import RLPrompts

    eng = [f"The market opens at eight o'clock number {i}." for i in range(n_prompts)]
    return RLPrompts(eng, ["yor"] * n_prompts, [""] * n_prompts, [10] * n_prompts, {}, 0, ["b64"] * n_prompts)


@pytest.mark.parametrize("name", ["rl.yaml", "rl_rtxpro.yaml"])
def test_the_configs_use_the_standard_sampling_and_loss_settings(name):
    cfg = dataclass_from_dict(RLConfig, load_yaml(os.path.join(ROOT, "configs", name)))
    assert (cfg.temperature, cfg.top_p, cfg.top_k, cfg.allow_truncated_sampling) == (1.0, 1.0, 0, False)
    assert cfg.group_size == 8 and cfg.num_updates_per_batch == 1 and cfg.clip_eps is None          # exactly on-policy: ratio == 1
    assert cfg.advantage_norm == "mean_std" and cfg.kl_estimator == "k3" and cfg.loss_agg == "token" and cfg.kl_coef == 0.05
    assert cfg.sampler == "vllm"
    for k, v in cfg.max_new_tokens_by_bin.items():
        assert cfg.vllm_max_model_len >= v + 512


def test_what_reaches_the_engine_is_the_plain_sampling_distribution():
    cfg = dataclass_from_dict(RLConfig, load_yaml(os.path.join(ROOT, "configs", "rl.yaml")))
    eng = CapturingEngine(lambda p, j: TOK.encode("yor Oja n si", add_special_tokens=False) + [EOT])
    s = VLLMSampler(eng, seed=cfg.seed)
    s._version = 0
    reward = TranslationReward(RewardConfig(), FakeScorer(), TOK, STOP)
    prompts = _prompts(3)
    roll, samples, out = rollout_and_reward(None, TOK, reward, prompts, [0, 1, 2], cfg, STOP, TOK.pad_token_id, "cpu", None, max_new_tokens=384, sampler=s, key=7)
    sp = eng.specs[0]
    assert (sp.n, sp.temperature, sp.top_p, sp.top_k, sp.max_tokens) == (8, 1.0, 1.0, 0, 384)
    assert sp.stop_token_ids == (EOT, EOS) and sp.logprobs is False                            # log-probs are not requested (the loss recomputes them)
    assert sp.seeds == derive_seeds(cfg.seed, 7, 3) and len(set(sp.seeds)) == 3                  # one seed per prompt; the n children use seed .. seed+7 (vLLM)
    assert max(sp.seeds) - min(sp.seeds) >= 2 * 1024 and all(b - a >= 8 for a, b in zip(sp.seeds, sp.seeds[1:]))
    # ids are kept exactly as generated: nothing is decoded or re-encoded between the engine and the reward
    assert roll.completions[0] == TOK.encode("yor Oja n si", add_special_tokens=False) + [EOT] and len(roll.completions) == 24
    assert [x.token_ids for x in samples] == roll.completions


def test_engine_parameters_in_the_source_are_the_audited_ones():
    """The vLLM 0.31 defaults for everything not listed are: min_p 0, repetition / presence / frequency penalties 1.0 / 0 / 0, min_tokens 0, no logit bias,
    no bad words (read on the rental with `SamplingParams(...)`); top_k = 0 means 'disabled' ('top_k must be 0 (disable), or at least 1'); the n children of a request
    with a seed get seed + index (v1/engine/parallel_sampling.py). Here: the keyword arguments of OUR call."""
    src = open(os.path.join(ROOT, "gemma_mt", "rl", "vllm_engine.py"), encoding="utf-8").read()
    i = src.index("SamplingParams(")
    call = src[i: i + 700]
    for needle in ("n=1 if greedy else spec.n", "temperature=0.0 if greedy else spec.temperature", "top_p=spec.top_p", "top_k=spec.top_k", "max_tokens=spec.max_tokens",
                   "stop_token_ids=list(spec.stop_token_ids)", "detokenize=False", "ignore_eos=False", "skip_special_tokens=False"):
        assert needle in call, needle
    for forbidden in ("repetition_penalty", "presence_penalty", "frequency_penalty", "min_tokens", "min_p", "logit_bias", "bad_words", "allowed_token_ids"):
        assert forbidden not in call, forbidden
    assert 'generation_config="vllm"' in src and "skip_tokenizer_init" in src          # the checkpoint's top_k 64 / top_p 0.95 defaults are ignored; ids in, ids out
    hf = open(os.path.join(ROOT, "gemma_mt", "rl", "sampling.py"), encoding="utf-8").read()
    assert "do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k" in hf and "repetition_penalty" not in hf and "min_new_tokens=" not in hf


def test_the_stop_ids_are_end_of_turn_and_eos():
    stop = fmt.resolve_stop_tokens(TOK)
    assert stop.ids == [EOT, EOS]


# ------------------------------------------------------------------------------------------------ (b) one normalisation chain
NFD_YOR = unicodedata.normalize("NFD", "Àwọn àgbẹ̀ láti Ibadan mú ẹ̀fọ́ tútù wá sí ọjà.")


def _ids(text, end=True):
    return TOK.encode(text, add_special_tokens=False) + ([EOT] if end else [])


CASES = {
    "plain": _ids("yor Oja n si ni agogo mejo."),
    "nfd": _ids("yor " + NFD_YOR),
    "space_before_tag_and_after": _ids("  yor   Oja n si ni agogo mejo.   "),
    "special_strings_typed_out": _ids("yor Oja <unused5>n si <pad>ni agogo."),
    "special_ids_inside": TOK.encode("yor Oja ", add_special_tokens=False) + [SPECIALS["<bos>"]] + TOK.encode("n si ni agogo.", add_special_tokens=False) + [EOT],
    "after_the_first_stop": _ids("yor Oja n si.") + _ids("junk after the stop"),
    "second_stop_id": TOK.encode("yor Oja n si ni agogo.", add_special_tokens=False) + [EOS],
    "interior_whitespace_kept": _ids("yor Oja   n si\tni agogo."),
}


class IdsSampler:
    """A 'sampler' that returns scripted ids: the evaluation function runs unchanged on it."""

    def __init__(self, rows):
        self.rows = rows

    def generate(self, prompts, **kw):
        return Rollout([list(p) for p in prompts], [list(r) for r in self.rows[: len(prompts)]], [True] * len(prompts))


def test_reward_and_evaluation_derive_the_identical_hypothesis_from_the_same_ids():
    from gemma_mt.eval.eval_translation import generate_translations

    names = list(CASES)
    reward = TranslationReward(RewardConfig(), FakeScorer(), TOK, STOP)
    src = "The farmers from Ibadan bring fresh vegetables to the big market every morning."
    out = reward([Sample(CASES[n], "yor", src) for n in names])
    rows = [{"english": src, "reference": "x", "iso": "yor", "benchmark": "t", "id": str(i)} for i in range(len(names))]
    ev = generate_translations(None, TOK, rows, stop_ids=STOP, pad_id=TOK.pad_token_id, device="cpu", sampler=IdsSampler([CASES[n] for n in names]))
    compared = 0
    for i, n in enumerate(names):
        assert out.status[i] in ("ok", "bad_prefix", "dangling_marker", "post_eos_rubbish"), (n, out.status[i])
        if out.status[i] == "ok":
            assert out.texts[i] == ev[i]["hyp"], n                         # THE test: reward text == evaluation hypothesis
            assert ev[i]["prefix_status"] == "ok" and ev[i]["finished"]
            compared += 1
    assert compared >= 6
    nfd = names.index("nfd")
    assert out.texts[nfd] == unicodedata.normalize("NFC", out.texts[nfd]) == ev[nfd]["hyp"]            # NFC in both paths
    sp = names.index("special_strings_typed_out")
    assert "<" not in out.texts[sp] and out.texts[sp] == ev[sp]["hyp"]
    ws = names.index("interior_whitespace_kept")
    assert "   " in out.texts[ws] and out.texts[ws] == ev[ws]["hyp"]                                    # the interior of the translation is never touched
    # the benchmark metric adds nothing but NFC (idempotent on this text)
    from gemma_mt.eval import metrics as M

    assert M.nfc([ev[nfd]["hyp"]]) == [ev[nfd]["hyp"]]


def test_a_missing_or_wrong_tag_is_a_reward_failure_but_the_same_text_is_scored_in_the_evaluation():
    """The only intended difference: the evaluation scores whatever follows (or the whole text without a tag) and reports the tag status; the reward fails the row."""
    from gemma_mt.eval.eval_translation import generate_translations

    reward = TranslationReward(RewardConfig(), FakeScorer(), TOK, STOP)
    ids = _ids("Oja n si ni agogo mejo.")
    out = reward([Sample(ids, "yor", "The market opens at eight.")])
    ev = generate_translations(None, TOK, [{"english": "x", "reference": "y", "iso": "yor"}], stop_ids=STOP, pad_id=0, device="cpu", sampler=IdsSampler([ids]))
    assert out.status == ["bad_prefix"] and ev[0]["prefix_status"] == "missing" and ev[0]["hyp"] == "Oja n si ni agogo mejo."


# ------------------------------------------------------------------------------------------------ (c) conventions
def test_group_advantage_and_loss_conventions_are_the_grpo_ones():
    r = torch.tensor([2.0, 1.0, 0.0, -1.0, 5.0, 5.0, 5.0, 5.0])
    adv, flat = grpo.group_advantages(r, 4, "mean_std", 1e-4)
    g = r[:4]
    assert torch.allclose(adv[:4], (g - g.mean()) / (g.std(unbiased=False) + 1e-4), atol=1e-6)      # population std per group, eps 1e-4
    assert torch.all(adv[4:] == 0) and flat == 0.5                                                      # a flat group has zero advantage (DAPO would drop it)
    dr, _ = grpo.group_advantages(r, 4, "mean")
    assert torch.allclose(dr[:4], g - g.mean())                                                         # Dr.GRPO: no std division
    # token-level aggregation over the whole step, KL k3 with the reference, ratio 1 and no clipping when on-policy
    lp = torch.tensor([[-1.0, -2.0, 0.0], [-0.5, 0.0, 0.0]], requires_grad=True)
    m = torch.tensor([[1, 1, 1], [1, 0, 0]])
    ref = lp.detach() + 0.1
    res = grpo.grpo_token_loss(lp, torch.tensor([1.0, -1.0]), m, ref_logp=ref, kl_coef=0.05, kl_estimator="k3", loss_agg="token", normaliser=4.0)
    k3 = float(torch.exp(torch.tensor(0.1))) - 0.1 - 1.0
    expect = (-(1.0) * 3 + 1.0 * 1 + 0.05 * k3 * 4) / 4.0
    assert abs(float(res["loss"]) - float(expect)) < 1e-5 and float(res["ratio_sum"]) == 4.0 and float(res["clip_sum"]) == 0.0
