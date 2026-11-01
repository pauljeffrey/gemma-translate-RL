# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-44 addendum: the public per-(document, language, system) record: schema, builder (human / pseudo references, tiers, private split, non-permissive translators
never published), validation, reward fields from the real reward code. Offline."""
import json

import pytest

pytest.importorskip("sacrebleu")

from gemma_mt.eval import longdoc as L
from gemma_mt.eval import longdoc_release as RL
from gemma_mt.eval import longdoc_reward as R
from gemma_mt.fake_tokenizer import SPECIALS, FakeGemmaTokenizer
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.reward import RewardConfig, TranslationReward

TOK = FakeGemmaTokenizer()
SRC = ("The market opens at eight o'clock. The farmers bring fresh vegetables from the villages. Many families come early to buy rice and beans. "
       "By noon the stalls are almost empty. The traders count their money and talk about the weather.")
HUM = ("Oja n si ni agogo mejo. Awon agbe mu ewebe tutu wa lati awon abule. Opolopo idile wa ni kutukutu lati ra iresi ati ewa. "
       "Ni osan awon ibudo ti fere ofo. Awon onisowo ka owo won won si soro nipa oju ojo.")
PSE = HUM.replace("Opolopo", "Opo").replace("tutu", "titun")


def docrow(i, split="public", subset="core"):
    sents = L.canonical_sentences(SRC)
    return {"doc_id": f"wikinews:{i}", "text": SRC, "sents": sents, "domain": "news", "bin": 512, "source": "wikinews", "licence": "CC-BY-2.5", "licence_class": "permissive",
            "url": f"https://en.wikinews.org/{i}", "attribution": "Wikinews contributors", "split": split, "subset": subset, "n_tokens": 400}


def setup(n_private=0):
    docs = {}
    refdocs, hyps, eval_rows, reward_rows = [], {}, {}, {}
    sents = L.canonical_sentences(SRC)
    rw = TranslationReward(RewardConfig(chunking="sentences"), FakeScorer(), TOK, [SPECIALS["<end_of_turn>"], SPECIALS["<eos>"]])
    for i in range(3):
        d = docrow(i, split="private" if i < n_private else "public")
        docs[d["doc_id"]] = d
        refdocs.append(L.RefDoc(d["doc_id"], "yor", SRC, sents, L.canonical_sentences(PSE), "pseudo", "madlad400-7b-mt-bt", "news", 512))
        refdocs.append(L.RefDoc(d["doc_id"], "yor", SRC, sents, L.canonical_sentences(PSE), "pseudo", "nllb-200-3.3b", "news", 512))      # non-commercial weights: published with a usage note
        hyps[("yor", "sysA", d["doc_id"])] = L.Hyp(d["doc_id"], "yor", HUM if i != 1 else " ".join(L.split_sentences(HUM)[:3]), True, "sysA", 300)
    pseudo_docs = [r for r in refdocs if r.ref_system == "madlad400-7b-mt-bt"]
    run = L.evaluate_run(pseudo_docs, {(k[0], k[2]): v for k, v in hyps.items()}, tiers={"yor": {"tier": "B", "system": "madlad400-7b-mt-bt"}}, scorer_ref=FakeScorer(),
                         scorer_qe=FakeScorer(), n_boot=5, keep_rows=True)
    for row in run["pseudo"]["yor"]["rows"]:
        eval_rows[("yor", "sysA", row["doc_id"])] = row
    rows = [{"doc_id": k[2], "iso": "yor", "system": "sysA", "hyp": v.text, "finished": True} for k, v in hyps.items()]
    for s in R.score_rows(rw, TOK, rows, {d: docs[d]["text"] for d in docs}):
        reward_rows[("yor", "sysA", s["doc_id"])] = s
    return docs, refdocs, hyps, eval_rows, reward_rows, rw


def test_columns_are_unique_documented_and_rendered():
    names = [c[0] for c in RL.RECORD_COLUMNS]
    assert len(names) == len(set(names)) == len(RL.COLUMN_NAMES) and all(c[3] for c in RL.RECORD_COLUMNS)
    md = RL.columns_markdown()
    assert md.count("\n") == len(names) + 2 and "`reward_total`" in md and "`translation`" in md and "`references`" in md
    for needed in ("source_text", "system_id", "translation", "references", "bleu", "chrf", "africomet_ref", "africomet_qe", "coverage_chars", "loop", "finished", "reward_total"):
        assert needed in names


def test_build_records_fields_references_and_validation():
    docs, refdocs, hyps, eval_rows, reward_rows, rw = setup()
    meta = {"reward_range": list(rw.reward_range), "calibration_sha256": "ab" * 32}
    pub, priv = RL.build_records(docs=docs, refdocs=refdocs, hyps=hyps, eval_rows=eval_rows, reward_rows=reward_rows, tiers={"yor": {"tier": "B"}},
                                 systems={"sysA": {"checkpoint": "org/model@abc123", "decoding": "greedy"}}, reward_meta=meta,
                                 translator_licence={"madlad400-7b-mt-bt": "permissive", "nllb-200-3.3b": "noncommercial_weights"})
    assert len(pub) == 3 and priv == []
    r = pub[0]
    assert RL.validate_record(r) == [], RL.validate_record(r)
    assert r["record_id"] == "wikinews:0|yor|sysA" and r["language"] == "Yoruba" and r["reference_kind"] == "pseudo" and r["reference_tier"] == "B"
    assert r["reference_system"] == "madlad400-7b-mt-bt" and sorted(x["system_id"] for x in r["references"]) == ["madlad400-7b-mt-bt", "nllb-200-3.3b"]       # the NLLB reference is published, with its usage note
    assert [x for x in r["references"] if x["system_id"] == "nllb-200-3.3b"][0]["usage_note"].startswith("the model weights")
    assert r["africomet_status"] == "validated" and r["africomet_ref"] is not None and r["africomet_qe"] is not None
    assert 0 < r["chrf"] < 100 and 0 <= r["bleu"] <= 100 and r["coverage_chars"] == 1.0 and r["finished"] is True and r["new_tokens"] == 300
    assert r["reward_total"] is not None and r["reward_status"] == "ok" and r["reward_range"] == meta["reward_range"] and r["reward_calibration_sha256"] == "ab" * 32
    assert r["canary"] == RL.CANARY and "canary GUID" in RL.CANARY
    assert r["system_checkpoint"] == "org/model@abc123" and r["source_licence"] == "CC-BY-2.5" and r["source_attribution"] == "Wikinews contributors"
    short = [x for x in pub if x["doc_id"] == "wikinews:1"][0]
    assert short["coverage_chars"] < 1.0 and short["reward_total"] < r["reward_total"]
    json.dumps(pub)                                                                         # JSON-serialisable


def test_private_split_never_in_the_public_file_and_validation_catches_errors():
    docs, refdocs, hyps, eval_rows, reward_rows, rw = setup(n_private=1)
    pub, priv = RL.build_records(docs=docs, refdocs=refdocs, hyps=hyps, eval_rows=eval_rows, reward_rows=reward_rows, translator_licence={"madlad400-7b-mt-bt": "permissive"})
    assert len(pub) == 2 and len(priv) == 1 and priv[0]["split"] == "private" and all(p["split"] == "public" for p in pub)
    assert RL.validate_record(priv[0]) == ["a non-public record in the public file"]
    bad = dict(pub[0], chrf=140.0, reference_tier="Z", subset="x", extra_col=1)
    del bad["translation"]
    probs = " ".join(RL.validate_record(bad))
    assert "missing translation" in probs and "chrf out of range" in probs and "bad reference_tier" in probs and "bad subset" in probs and "unknown columns" in probs
    nl = dict(pub[0], references=[dict(pub[0]["references"][0], licence_class="unknown")])
    assert "pseudo reference from a translator of unknown licence" in RL.validate_record(nl)
    assert any("outside reward_range" in p for p in RL.validate_record(dict(pub[0], reward_total=99.0, reward_range=[-2.0, 2.25])))


def test_language_without_africomet_gets_null_metrics_and_human_reference_tier_h():
    docs, refdocs, hyps, eval_rows, reward_rows, rw = setup()
    sents = L.canonical_sentences(SRC)
    hd = L.RefDoc("smoldoc:x", "ibb", SRC, sents, L.canonical_sentences(HUM), "human", "smoldoc", "smoldoc_topic", 0)
    docs["smoldoc:x"] = dict(docrow(9, subset="human_smoldoc"), doc_id="smoldoc:x", source="smoldoc", licence="CC-BY-4.0", bin=0)
    h = L.Hyp("smoldoc:x", "ibb", HUM, True, "sysA", 200)
    run = L.evaluate_run([hd], {("ibb", "smoldoc:x"): h}, scorer_ref=FakeScorer(), scorer_qe=FakeScorer(), n_boot=3, keep_rows=True)
    er = {("ibb", "sysA", "smoldoc:x"): run["human"]["ibb"]["rows"][0]}
    pub, _ = RL.build_records(docs=docs, refdocs=[hd], hyps={("ibb", "sysA", "smoldoc:x"): h}, eval_rows=er)
    r = pub[0]
    assert r["africomet_status"] == "not_covered" and r["africomet_ref"] is None and r["africomet_qe"] is None
    assert r["reference_kind"] == "human" and r["reference_tier"] == "H" and r["subset"] == "human_smoldoc" and r["reward_total"] is None
    assert RL.validate_record(r) == []
    assert RL.africomet_status("yor") == "validated" and RL.africomet_status("ven") == "listed"


def test_cli_columns_and_build(tmp_path, capsys=None):
    docs, refdocs, hyps, eval_rows, reward_rows, rw = setup()
    L.write_jsonl(str(tmp_path / "docs.jsonl"), docs.values())
    L.write_jsonl(str(tmp_path / "refs.jsonl"), [{"doc_id": r.doc_id, "iso": "yor", "ref_sents": r.ref_sents, "ref_kind": "pseudo", "ref_system": r.ref_system} for r in refdocs])
    L.write_jsonl(str(tmp_path / "preds.jsonl"), [{"doc_id": k[2], "iso": "yor", "system": "sysA", "hyp": v.text, "finished": True, "new_tokens": 300} for k, v in hyps.items()])
    pseudo = [r for r in refdocs if r.ref_system == "madlad400-7b-mt-bt"]
    run = L.evaluate_run(pseudo, {(k[0], k[2]): v for k, v in hyps.items()}, tiers={"yor": {"tier": "A", "system": "madlad400-7b-mt-bt"}}, scorer_ref=FakeScorer(), scorer_qe=FakeScorer(),
                         n_boot=3, keep_rows=True)
    (tmp_path / "res.json").write_text(json.dumps(run))
    L.write_jsonl(str(tmp_path / "rew.jsonl"), reward_rows.values())
    (tmp_path / "lic.json").write_text(json.dumps({"madlad400-7b-mt-bt": "permissive", "nllb-200-3.3b": "noncommercial_weights"}))
    (tmp_path / "tiers.json").write_text(json.dumps({"tiers": {"yor": {"tier": "A"}}}))
    rc = RL.main(["build", "--docs", str(tmp_path / "docs.jsonl"), "--refs", str(tmp_path / "refs.jsonl"), "--preds", str(tmp_path / "preds.jsonl"),
                  "--eval-json", f"sysA={tmp_path / 'res.json'}", "--reward-scores", str(tmp_path / "rew.jsonl"), "--tiers", str(tmp_path / "tiers.json"),
                  "--translator-licences", str(tmp_path / "lic.json"), "--out", str(tmp_path / "rel")])
    assert rc == 0
    recs = L.read_jsonl(str(tmp_path / "rel" / "records_public.jsonl"))
    assert len(recs) == 3 and recs[0]["reference_tier"] == "A" and all(RL.validate_record(x) == [] for x in recs)
    assert RL.main(["columns-md"]) == 0
