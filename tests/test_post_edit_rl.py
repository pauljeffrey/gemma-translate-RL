# SPDX-License-Identifier: AGPL-3.0-or-later
"""Editor RL glue: identity member, editor reward arithmetic on a stub translation reward, and the whole RL loop in editor mode on a tiny random model (CPU)."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

from gemma_mt import format as fmt
from gemma_mt import post_edit as pe
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
from gemma_mt.rl import post_edit_rl as pr

TOK = FakeGemmaTokenizer()


def test_identity_completion_is_the_sft_target_of_the_draft():
    with fmt.tag_mode("single_token"):
        ids = pr.identity_completion(TOK, "Good morning to you all", "E kaaro gbogbo yin", "yor")
        ex = pe.build_pe_example("Good morning to you all", "E kaaro gbogbo yin", "yor", "E kaaro gbogbo yin", TOK)
    assert ids == [t for t, l in zip(ex.input_ids, ex.labels) if l != fmt.IGNORE_INDEX]
    assert ids[-1] == TOK.convert_tokens_to_ids(fmt.END_OF_TURN)


def test_insert_identity_replaces_row_G_minus_3():
    roll = SimpleNamespace(completions=[[i] for i in range(12)], finished=[False] * 12, post_stop_tokens=[1] * 12, sampler_logp=[[0.5]] * 12)
    rows = pr.insert_identity(roll, 6, [[7, 8, 9], [4, 5]])
    assert rows == [3, 9] and roll.completions[3] == [7, 8, 9] and roll.completions[9] == [4, 5]
    assert roll.finished[3] and roll.post_stop_tokens[3] == 0 and roll.sampler_logp[3] == [0.0, 0.0, 0.0]
    assert roll.completions[0] == [0]                                    # nothing else touched


class StubReward:
    """The parts of TranslationReward the editor reward uses: the call (q per sample), reward_range, penalty, content_spec."""
    reward_range = (-2.0, 2.25)
    penalty = -2.0

    def __init__(self, q_of):
        self.q_of = q_of

    def content_spec(self, lang):
        return ("qe", 1.0) if lang != "cns" else ("consensus", 1.0)

    def __call__(self, samples, group_size=None):
        texts = [TOK.decode(s.token_ids[1:-1]).strip() for s in samples]
        q = [self.q_of(t) for t in texts]
        return SimpleNamespace(rewards=q, texts=texts, status=["ok" if t else "empty" for t in texts], weights=[1.0] * len(samples), stats={})


def _edit_samples(texts, iso="yor"):
    from gemma_mt.rl.reward import Sample
    with fmt.tag_mode("single_token"):
        return [Sample(pr.identity_completion(TOK, "Good morning to you all", t, iso), iso, "Good morning to you all") for t in texts]


def test_editor_reward_arithmetic_on_a_stub():
    q = {"bad draft here now": -0.5, "bad draft here now fixed": 0.5, "perfect": 1.0}
    stub = StubReward(lambda t: q.get(t, 0.0))
    er = pr.EditorReward(stub, pr.EditorOptions())
    draft = "bad draft here now"
    with fmt.tag_mode("single_token"):
        d_ids = [pr.identity_completion(TOK, "Good morning to you all", draft, "yor")]
        samples = _edit_samples([draft, "bad draft here now fixed"])
        out = stub(samples)
        rewards, m = er(samples, out, [{"english": "Good morning to you all", "draft": draft, "iso": "yor", "reference": ""}], 2, d_ids, identity_rows=[0])
    assert rewards[0] == pytest.approx(0.0, abs=1e-6)                      # the unchanged draft of a (nearly) clean draft earns nothing
    assert rewards[1] > 0.5                                                # a real gain is paid
    assert m["pe/identical_share"] == 0.5 and m["pe/identity_reward"] == pytest.approx(0.0, abs=1e-6)


def test_editor_reward_failure_is_the_penalty_and_regression_costs_more():
    q = {"good draft text now": 0.8, "worse text": 0.2}
    stub = StubReward(lambda t: q.get(t, 0.0))
    er = pr.EditorReward(stub, pr.EditorOptions())
    with fmt.tag_mode("single_token"):
        d_ids = [pr.identity_completion(TOK, "Good morning to you all", "good draft text now", "yor")]
        smp = _edit_samples(["worse text", "good draft text now"])
        out = stub(smp); out.status[0] = "no_end_of_turn"                    # a hard failure of the edit
        rew, _ = er(smp, out, [{"english": "Good morning to you all", "draft": "good draft text now", "iso": "yor", "reference": ""}], 2, d_ids)
    assert rew[0] == -2.0
    smp2 = _edit_samples(["worse text"])
    with fmt.tag_mode("single_token"):
        rew2, _ = er(smp2, stub(smp2), [{"english": "Good morning to you all", "draft": "good draft text now", "iso": "yor", "reference": ""}], 1, d_ids)
    assert rew2[0] < -0.5                                                  # kappa x the drop


def test_reference_prompt_uses_reference_terms_and_reduced_qe_weight():
    q = {"a b c d e f": 0.0, "a b c x y z": 0.0}
    stub = StubReward(lambda t: q.get(t, 0.0))
    er = pr.EditorReward(stub, pr.EditorOptions())
    with fmt.tag_mode("single_token"):
        d_ids = [pr.identity_completion(TOK, "Good morning to you all", "a b c d e f", "yor")]
        smp = _edit_samples(["a b c x y z"])
        rew, m = er(smp, stub(smp), [{"english": "Good morning to you all", "draft": "a b c d e f", "iso": "yor", "reference": "a b c x y z"}], 1, d_ids)
    assert rew[0] > 0.3                                                    # the QE gain is 0, the reference chrF++ and the word ledger pay for the edit that matches the reference
    assert er.weights_for("yor", True).gain == 0.35 and er.weights_for("yor", True).ref == 1.0 and er.weights_for("yor", False).gain == 1.0


def test_off_terms_and_qe_languages():
    er = pr.EditorReward(StubReward(lambda t: 0.0), pr.EditorOptions(off_terms=["churn"]))
    assert er.weights_for("yor", False).churn == 0.0
    er = pr.EditorReward(StubReward(lambda t: 0.0), pr.EditorOptions(), ref_scorer=lambda s, m, r: [0.5] * len(s))
    assert er.weights_for("yor", True).ref == 0.6 and er.covered                # COMET-covered language: chrF++ weight lower, the COMET gain added


def test_rl_loop_in_editor_mode_runs_on_a_tiny_model(tmp_path):
    pytest.importorskip("torch"); pytest.importorskip("transformers")
    from gemma_mt.rl.train_rl import run_smoke
    from gemma_mt.smoke import ENGLISH, fake_translation
    langs = ["yor", "hau", "ibo"]
    rows = []
    for i in range(40):
        en = ENGLISH[i % len(ENGLISH)]; iso = langs[i % 3]
        rows.append({"english": en, "iso": iso, "language": "", "src_afr_tokens": 12, "source_dataset": "smoke", "draft": fake_translation(en, iso)[:-3] or "x", "draft_system": "tiny",
                     "reference": fake_translation(en, iso) if i % 2 == 0 else ""})
    pp = str(tmp_path / "pool.jsonl")
    with open(pp, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    with fmt.tag_mode("single_token"):
        s = run_smoke(str(tmp_path / "w"), sft_steps=20, overrides=dict(prompts_path=pp, editor_mode=True, group_size=6, prompts_per_step=3, guide_fraction=1.0, guide_beams=2, steps=3, eval_every=3,
                                                                       editor={"ref_comet": {"fake": True, "python": sys.executable}, "ref_comet_languages": ["yor", "hau"]},
                                                                       hard_fail_mask="off", greedy_in_group=True, check_sampler_logprobs=False))
    recs = [json.loads(l) for l in open(os.path.join(s["workdir"], "rl", "metrics.jsonl"))]
    keys = [k for r in recs for k in r if k.endswith("pe/identical_share")]
    assert keys, list(recs[0])[:40]                                       # the editor statistics are logged (prefixed by the reward logger)
    assert all(abs(r.get(k2, 0)) <= 2.25 for r in recs for k2 in r if k2.endswith("pe/reward_mean"))
