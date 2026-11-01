# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-43: QE-free calibration of a language from human references (rl/calibrate_refs.py) and its use by the reward as a consensus-only language."""
import json
import os
import random

import pytest

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import calibrate_refs as C
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
from gemma_mt.rl.reward_shaping import ContentShaper

TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]


def _write(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _sets(tmp_path):
    t = str(tmp_path / "ts")
    _write(f"{t}/smol_sent/en_xx.jsonl", [{"id": i, "src": f"Sentence number {i} about the market.", "trg": f"Ke\u0301 {i} ọja n\u0304 kpọ."} for i in range(40)])
    _write(f"{t}/ibom_mt/en_xx.jsonl", [{"id": str(i), "src": f"Sentence number {i} about the market.", "trg": f"dup {i}"} for i in range(5)]       # same English as smol_sent 0..4
    + [{"id": "9" + str(i), "src": f"Another thing {i}.", "trg": f"ọkpọ {i}."} for i in range(10)])
    _write(f"{t}/smol_doc/en_xx.jsonl", [{"id": "d1", "srcs": ["First one.", "Second one here.", "Third."], "trgs": ["Kiet.", "Iba.", "Ita."]}])
    c = str(tmp_path / "ts" / "_contamination")
    _write_json(f"{c}/contamination_smol_sent.json", {"exact": {"xx:0": True}, "sentence": {"xx:0": True, "xx:1": False}, "near": {}, "pair": {"xx:2": True}})
    _write_json(f"{c}/contamination_smol_doc.json", {"exact": {"xx:d1": False}, "sentence": {}, "near": {"xx:d1": True}, "pair": {}})
    return t, c


def _write_json(p, d):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(d, open(p, "w"))


def test_load_pairs_dedupes_by_english_nfc_and_attaches_contamination(tmp_path):
    t, c = _sets(tmp_path)
    rows = C.load_pairs(t, "xx", {"smol_sent": "xx", "ibom_mt": "xx", "smol_doc": "xx"}, c, sources=("smol_sent", "ibom_mt", "smol_doc"))
    assert len(rows) == 40 + 10 + 3                                           # 5 ibom rows duplicate smol_sent English: dropped
    by = {r["english"]: r for r in rows}
    assert by["Sentence number 0 about the market."]["source"] == "smol_sent" and by["Sentence number 0 about the market."]["english_seen"] is True
    assert by["Sentence number 1 about the market."]["english_seen"] is False and by["Sentence number 2 about the market."]["pair_seen"] is True
    assert by["First one."]["english_seen"] is True and by["First one."]["doc"] == "d1"        # document: exact OR near
    assert all(r["reference"] == C.nfc(r["reference"]) for r in rows)
    assert by["Sentence number 0 about the market."]["reference"].startswith("K\u00e9")          # NFD input came out NFC


def test_split_is_deterministic_disjoint_and_about_half(tmp_path):
    rows = [{"english": f"This is sentence {i}."} for i in range(2000)]
    a1, h1 = C.split_rows(rows)
    a2, h2 = C.split_rows(list(reversed(rows)))
    assert {r["english"] for r in a1} == {r["english"] for r in a2} and not ({r["english"] for r in a1} & {r["english"] for r in h1})
    assert len(a1) + len(h1) == 2000 and 0.45 < len(a1) / 2000 < 0.55
    assert C.split_bucket("x") == C.split_bucket("x\u0301".replace("\u0301", "")) and 0 <= C.split_bucket("y") < 1
    a3, _ = C.split_rows(rows, 0.2)
    assert {r["english"] for r in a3} <= {r["english"] for r in a1}                          # a smaller fraction is a subset


def test_spearman_and_decision_rule():
    assert C.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert C.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert C.spearman([1, 1, 1], [1, 2, 3]) is None and C.spearman([1, 2], [1, 2]) is None
    assert C.spearman([1, 2, 2, 3], [1, 3, 2, 4]) == pytest.approx(0.9487, abs=1e-3)           # ties: average ranks
    assert [C.weight_from_within_prompt_spearman(x) for x in (0.4, 0.25, 0.2, 0.1, 0.09, -0.3, None)] == [0.5, 0.5, 0.3, 0.3, 0.0, 0.0, 0.0]


def _rows(n=120, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        src = " ".join(rng.choice("market farmers village water morning bring fresh sell eight open".split()) for _ in range(rng.randint(6, 14))) + f" {i}."
        trg = " ".join(rng.choice(["ọkpọ", "ñwed", "mbọk", "ekpe", "usụn\u0304", "idem", "ufọk"]) for _ in range(rng.randint(5, 14))) + f" {i}."
        rows.append({"english": src, "reference": trg, "iso": "xx", "source": "ibom_mt" if i % 2 else "smol_sent", "doc": None, "idx": str(i), "english_seen": i % 3 == 0, "pair_seen": False})
    return rows


def test_reference_stats_and_entry_are_a_consensus_only_entry(tmp_path):
    st = C.reference_stats(_rows())
    assert st["n"] == 120 and set(st["len_ratio"]) == {"p05", "p50", "p95", "p01", "p02", "corpus", "s2"}
    assert st["diacritics"]["applies"] and st["omission"]["cost"] in C.__dict__.get("_GRID", (4.0, 5.0, 6.0, 8.0, 12.0))
    lid = {"enabled": True, "labels": ["xx_Latn"], "min_chars": 30, "recall": 0.99, "recall_strict": 0.99, "extra_labels": [], "source": "t", "n": 10, "short_recall": {}, "cross_accept": 0.0}
    e = C.consensus_entry(st, lid, 0.5, {"sets": ["ibom_mt"]})
    assert e["usable"] and e["reliability"] == e["reliability_same_source"] == 0.5 and e["content"]["mode"] == "consensus" and "anchors" not in e and e["lid"] == lid
    off = C.consensus_entry(st, lid, 0.0, {})
    assert off["usable"] is False and off["reliability"] == 0.0 and off["content"]["mode"] == "consensus"

    cal = {"_meta": {}, "xx": e}                                                              # the shaper reads it like any consensus-only language
    p = str(tmp_path / "cal.json")
    json.dump(cal, open(p, "w"))
    sh = ContentShaper.from_file(p, reliability_from="same_source", fail_floor=0.0)
    assert sh.is_usable("xx") and sh.reliability("xx") == 0.5 and sh.content["xx"]["mode"] == "consensus" and "xx" not in sh.cal
    assert sh.len_window["xx"][0] > 0 and sh.len_stats["xx"]["corpus"] == pytest.approx(st["len_ratio"]["corpus"]) and sh.len_stats["xx"]["omit_cost"] == st["omission"]["cost"]
    assert sh.diacritics["xx"]["applies"] and sh.lid["xx"]["labels"] == ["xx_Latn"]
    sh0 = ContentShaper.from_file(str(_dump(tmp_path, {"xx": off})), reliability_from="same_source")
    assert not sh0.is_usable("xx") and sh0.reliability("xx") == 0.0 and sh0.content["xx"]["mode"] == "consensus"


def _dump(tmp_path, d):
    p = tmp_path / "cal0.json"
    json.dump(d, open(p, "w"))
    return p


class NoQE:
    def __call__(self, *a, **k):
        raise AssertionError("QE called for a consensus-mode language")


def test_reward_never_calls_qe_for_the_entry_and_gives_a_consensus(tmp_path):
    e = C.consensus_entry(C.reference_stats(_rows(), omission=False), None, 0.5, {})
    p = tmp_path / "cal.json"
    json.dump({"efi": e}, open(p, "w"))                                                       # the real iso: efi is in QE_VALIDATED, the entry's mode decides
    cfg = RewardConfig(chunking="sentences", consensus=True, lid_enabled=False, calibration_path=str(p), reliability_from="same_source")
    r = TranslationReward(cfg, NoQE(), TOK, [EOT])
    assert r.content_spec("efi") == ("consensus", 1.0) and r.shaper.reliability("efi") == 0.5
    src = "The market opens at eight in the morning and the farmers bring fresh vegetables."
    texts = ["Ufọk ekpe idem mbọk ke usụn̄ ọkpọ ñwed.", "Ufọk ekpe idem mbọk ke usụn̄ ọkpọ ñwed ufọk.", "Ufọk ekpe idem mbọk ke ọkpọ ñwed.", "Idem ufọk ekpe mbọk ke usụn̄ ọkpọ."]
    out = r([Sample(TOK.encode(f"efi {t}", add_special_tokens=False) + [EOT], "efi", src) for t in texts], group_size=4)
    assert out.content_mode == ["consensus"] * 4 and all(c is not None and 0 < c <= 100 for c in out.consensus_chrf)
    assert out.weights == [0.5] * 4
