# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/eval_afriscience.py (D-52): corpus loading, decoding spec, metrics aggregation, and the whole run through a stand-in sampler."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("sacrebleu")
import eval_afriscience as E  # noqa: E402


def _corpus(tmp_path, split="test"):
    rows = []
    for iso in ("hau", "zul"):
        for pid, sents in ((1, ["Water boils at 100 degrees.", "It was measured twice."]), (2, ["Zinc is an essential trace element in humans."])):
            for sid, s in enumerate(sents):
                rows.append({"paper_id": pid, "sentence_id": sid, "domain": "health" if pid == 1 else "biology", "lang_pair": f"eng-{iso}", "source_lang": "eng", "target_lang": iso,
                             "source": s, "target": f"[{iso}] {s}"})
    d = tmp_path / "corpus"
    d.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / f"{split}-00000.parquet")
    return str(d)


def test_load_corpus_orders_sentences_and_builds_documents(tmp_path):
    c = E.load_corpus(_corpus(tmp_path), "test", ["hau", "zul"])
    assert [r["src"] for r in c["hau"]["sentences"]] == ["Water boils at 100 degrees.", "It was measured twice.", "Zinc is an essential trace element in humans."]
    docs = c["zul"]["documents"]
    assert [d["paper"] for d in docs] == ["1", "2"] and docs[0]["src"] == "Water boils at 100 degrees. It was measured twice." and docs[0]["n_sentences"] == 2
    assert docs[0]["ref"] == "[zul] Water boils at 100 degrees. [zul] It was measured twice."
    with pytest.raises(FileNotFoundError):
        E.load_corpus(str(tmp_path / "nowhere"), "test")
    with pytest.raises(ValueError):
        E.load_corpus(_corpus(tmp_path), "test", ["yor"])


def test_decoding_spec():
    assert E.parse_decoding("greedy")["greedy"] is True
    d = E.parse_decoding("sample:temperature=0.3,top_p=0.9,seed=5")
    assert d["greedy"] is False and d["temperature"] == 0.3 and d["top_p"] == 0.9 and d["seed"] == 5
    with pytest.raises(ValueError):
        E.parse_decoding("beam:4")
    with pytest.raises(ValueError):
        E.parse_decoding("sample:foo=1")


def test_metrics_perfect_and_empty():
    h = ["Ruwa yana tafasa a digiri 100.", "An auna shi sau biyu."]
    m = E.text_metrics(h, h, sentence_mean=True)
    assert m["chrf"] == pytest.approx(100.0) and m["chrf_pp"] == pytest.approx(100.0) and m["bleu"] == pytest.approx(100.0) and m["chrf_sentence_mean"] == pytest.approx(100.0)
    assert m["length_ratio"] == pytest.approx(1.0)
    z = E.text_metrics(["", ""], h)
    assert z["chrf"] == 0.0 and z["length_ratio"] == 0.0


class _Roll:
    def __init__(self, completions):
        self.completions = completions


class _Sampler:
    """Copies the English source back (as ids of a fake tokenizer would): used to check the plumbing, not the model."""

    def __init__(self, tok):
        self.tok, self.calls = tok, []

    def generate(self, prompts, **kw):
        self.calls.append(kw)
        from gemma_mt.fake_tokenizer import SPECIALS

        outs = []
        for p in prompts:
            text = self.tok.decode(p, skip_special_tokens=False)
            outs.append(self.tok.encode("hau translated", add_special_tokens=False) + [SPECIALS["<end_of_turn>"]])
        return _Roll(outs)


def test_end_to_end_with_a_stand_in_sampler(tmp_path):
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS

    tok = FakeGemmaTokenizer()
    c = E.load_corpus(_corpus(tmp_path), "test", ["hau", "zul"])
    s = _Sampler(tok)
    args = type("A", (), {"max_new_tokens_sentence": 32, "max_new_tokens_document": 64})()
    res = E.run_split(c, ["sentence", "document"], lambda items, mnt: E.generate(s, tok, [SPECIALS["<end_of_turn>"], SPECIALS["<eos>"]], 0, items, E.parse_decoding("greedy"), mnt, log=lambda *_: None), args)
    sc = res["scores"]
    assert set(sc) == {"sentence", "document"} and set(sc["sentence"]["per_language"]) == {"hau", "zul"}
    assert sc["sentence"]["per_language"]["hau"]["n"] == 3 and sc["document"]["per_language"]["hau"]["n"] == 2
    assert sc["sentence"]["per_language"]["hau"]["finished_rate"] == 1.0
    assert [k["max_tokens" if "max_tokens" in k else "max_new_tokens"] for k in s.calls] == [32, 64]
    assert all(k["greedy"] is True for k in s.calls)
    assert set(sc["document"]["per_domain_chrf"]["hau"]) == {"health", "biology"}
    assert len(res["predictions"]["document"]["zul"]) == 2


def test_published_table_and_markdown(tmp_path):
    p = tmp_path / "pub.csv"
    p.write_text("system,iso,chrf,bleu,ssa_comet\nNLLB,hau,67.7,46.9,0.63\ndoc: GPT,hau,66.8,32.9,0.45\n")
    pub = E.published_table(str(p))
    assert pub["NLLB"]["hau"] == 67.7 and E.published_table(None) == {}
    blk = {"per_language": {"hau": {"chrf": 50.0, "chrf_pp": 49.0, "bleu": 20.0, "spbleu": 21.0, "finished_rate": 1.0, "length_ratio": 1.0}}, "macro": {"chrf": 50.0, "chrf_pp": 49.0, "bleu": 20.0, "spbleu": 21.0, "finished_rate": 1.0, "length_ratio": 1.0}}
    md = E.markdown("m", "test", {"scores": {"sentence": blk, "document": blk}}, pub)
    assert "published chrF: NLLB" in md and md.count("published chrF: doc: GPT") == 1
