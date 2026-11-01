# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-44 addendum: scoring document translations with the REAL reward (rl/reward.py) - completion reconstruction, failure classes, group (consensus) mode, summary, CLI.
Offline: FakeGemmaTokenizer + FakeScorer, no model, no network."""
import json

import pytest

from gemma_mt.eval import longdoc as L
from gemma_mt.eval import longdoc_reward as R
from gemma_mt.fake_tokenizer import SPECIALS, FakeGemmaTokenizer
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.reward import RewardConfig, TranslationReward

TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]
SRC = ("The market opens at eight o'clock. The farmers bring fresh vegetables from the villages. Many families come early to buy rice and beans. "
       "By noon the stalls are almost empty. The traders count their money and talk about the weather.")
GOOD = ("Oja n si ni agogo mejo. Awon agbe mu ewebe tutu wa lati awon abule. Opolopo idile wa ni kutukutu lati ra iresi ati ewa. "
        "Ni osan awon ibudo ti fere ofo. Awon onisowo ka owo won won si soro nipa oju ojo.")


def reward(**kw):
    cfg = RewardConfig(chunking="sentences", **kw)
    return TranslationReward(cfg, FakeScorer(), TOK, [EOT, SPECIALS["<eos>"]])


def test_completion_ids_reconstruct_the_policy_output():
    ids = R.completion_ids(TOK, "yor", GOOD, True, EOT)
    assert ids[-1] == EOT and TOK.decode(ids[:-1]).startswith("yor Oja n si ni")
    assert R.completion_ids(TOK, "yor", GOOD, False, EOT)[-1] != EOT                      # unfinished: no stop id
    assert R.completion_ids(TOK, "yor", GOOD, None, EOT, assume_finished=True)[-1] == EOT
    assert R.completion_ids(TOK, "yor", GOOD, None, EOT, assume_finished=False)[-1] != EOT
    assert R.stop_ids_of(TOK) == [EOT, SPECIALS["<eos>"]]


def test_score_rows_complete_vs_truncated_vs_copy_vs_unfinished():
    rw = reward()
    half = " ".join(L.split_sentences(GOOD)[:2])
    rows = [{"doc_id": "d1", "iso": "yor", "system": "s", "hyp": GOOD, "finished": True},
            {"doc_id": "d1", "iso": "yor", "system": "t", "hyp": half, "finished": True},
            {"doc_id": "d1", "iso": "yor", "system": "u", "hyp": SRC, "finished": True},
            {"doc_id": "d1", "iso": "yor", "system": "v", "hyp": GOOD, "finished": False},
            {"doc_id": "d1", "iso": "yor", "system": "w", "hyp": "ha " * 80, "finished": True}]
    sc = {r["system"]: r for r in R.score_rows(rw, TOK, rows, {"d1": SRC}, batch_size=2)}
    lo, hi = rw.reward_range
    assert sc["s"]["status"] == "ok" and lo < sc["s"]["reward"] <= hi and sc["s"]["n_chunks"] >= 3 and sc["s"]["failed_chunks"] == 0
    assert sc["t"]["reward"] < sc["s"]["reward"]                                           # a document with missing sentences is worth less
    assert sc["u"]["status"] == "copy_source" and sc["u"]["reward"] == lo
    assert sc["v"]["status"] == "no_end_of_turn" and sc["v"]["reward"] == lo
    assert sc["w"]["status"] in ("degenerate", "length_ratio") and sc["w"]["reward"] == lo
    assert sc["s"]["content_mode"] == "qe" and sc["s"]["qe"] is not None and sc["u"]["qe"] is None


def test_score_rows_keeps_input_order_and_batches_per_language():
    rw = reward()
    rows = [{"doc_id": f"d{i}", "iso": "yor" if i % 2 else "hau", "system": "s", "hyp": GOOD, "finished": True} for i in range(5)]
    out = R.score_rows(rw, TOK, rows, {f"d{i}": SRC for i in range(5)}, batch_size=2)
    assert [o["doc_id"] for o in out] == [f"d{i}" for i in range(5)] and [o["iso"] for o in out] == [r["iso"] for r in rows]
    assert len({round(o["reward"], 9) for o in out}) == 1                                  # identical text, identical reward in any batch


def test_group_systems_gives_consensus_languages_a_real_signal():
    rw = reward(consensus=True, content_modes={"yor": "consensus"}, consensus_min_valid=2)
    d = L.split_sentences(GOOD)
    systems = {"a": GOOD, "b": GOOD, "c": " ".join(d[:3]) + " Awon onisowo ka owo won."}
    rows = [{"doc_id": "d1", "iso": "yor", "system": k, "hyp": v, "finished": True} for k, v in systems.items()]
    alone = {r["system"]: r for r in R.score_rows(rw, TOK, rows, {"d1": SRC})}
    grouped = {r["system"]: r for r in R.score_rows(rw, TOK, rows, {"d1": SRC}, group_systems=True)}
    assert all(alone[k]["consensus_chrf"] is None and alone[k]["content_mode"] == "consensus" for k in alone)
    assert all(grouped[k]["consensus_chrf"] is not None for k in grouped)
    assert grouped["a"]["consensus_chrf"] > grouped["c"]["consensus_chrf"] and grouped["a"]["reward"] > grouped["c"]["reward"]
    with pytest.raises(ValueError):
        R.score_rows(rw, TOK, rows[:2] + [dict(rows[2], doc_id="d2")], {"d1": SRC, "d2": SRC}, group_systems=True)         # ragged groups are refused


def test_summarise_and_meta(tmp_path):
    rw = reward()
    rows = [{"doc_id": "d1", "iso": "yor", "system": "s", "hyp": GOOD, "finished": True}, {"doc_id": "d1", "iso": "yor", "system": "t", "hyp": GOOD, "finished": False}]
    sc = R.score_rows(rw, TOK, rows, {"d1": SRC})
    s = R.summarise(sc, rw.reward_range)
    assert s["systems"]["s"]["languages"]["yor"]["ok_rate"] == 1.0 and s["systems"]["t"]["languages"]["yor"]["status"] == {"no_end_of_turn": 1}
    assert s["systems"]["t"]["macro_reward_mean"] == rw.reward_range[0] and s["reward_range"] == list(rw.reward_range)
    p = tmp_path / "cal.json"
    p.write_text("{}")
    assert len(R.file_sha256(str(p))) == 64 and R.file_sha256(str(tmp_path / "none")) is None


def test_build_reward_from_a_yaml_and_the_cli(tmp_path):
    yaml = pytest.importorskip("yaml")
    cfg = tmp_path / "rl.yaml"
    cfg.write_text(yaml.safe_dump({"reward": {"chunking": "sentences", "omission_detector": True, "lid_enabled": False, "nfc": True}, "scorer": {"type": "fake"}}))
    rw, tok, meta = R.build_reward(str(cfg), scorer_kind="fake", tokenizer=TOK, overrides={"length_penalty": 0.0})
    assert meta["chunking"] == "sentences" and meta["omission_detector"] is True and meta["scorer"] == "FakeScorer" and meta["reward_range"][0] < 0
    with pytest.raises(ValueError):
        R.build_reward(str(cfg), scorer_kind="fake", tokenizer=TOK, overrides={"no_such_field": 1})          # strict: a typo in the config is an error
    L.write_jsonl(str(tmp_path / "docs.jsonl"), [{"doc_id": "d1", "text": SRC}])
    L.write_jsonl(str(tmp_path / "preds.jsonl"), [{"doc_id": "d1", "iso": "yor", "system": "s", "hyp": GOOD, "finished": True},
                                                  {"doc_id": "d1", "iso": "hau", "system": "s", "text": GOOD}, {"doc_id": "zz", "iso": "yor", "system": "s", "hyp": GOOD}])
    rc = R.main(["--preds", str(tmp_path / "preds.jsonl"), "--docs", str(tmp_path / "docs.jsonl"), "--config", str(cfg), "--scorer", "fake", "--tokenizer", "fake",
                 "--out", str(tmp_path / "o" / "scores.jsonl")])
    assert rc == 0
    rows = L.read_jsonl(str(tmp_path / "o" / "scores.jsonl"))
    summ = json.load(open(tmp_path / "o" / "scores_summary.json"))
    assert len(rows) == 2 and summ["meta"]["scorer"] == "FakeScorer" and set(summ["systems"]["s"]["languages"]) == {"yor", "hau"}
