"""Unit tests of the pure-Python parts of the science-stage rebuild (agent C2). No model, no index, no network.

    python -m pytest s4_tests.py -q
"""
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s4_afs_windows as AW  # noqa: E402
import s4_common as K  # noqa: E402
import s4_mix as MX  # noqa: E402


def rows_for(papers=3, n=12, langs=("amh", "hau"), drop=None):
    out = []
    for p in range(papers):
        for s in range(n):
            for l in langs:
                if drop and (p, s, l) in drop:
                    continue
                out.append({"paper_id": p, "sentence_id": s, "domain": "health", "target_lang": l, "source": f"English sentence {p}-{s}.", "target": f"{l} sentence {p}-{s}.",
                            "licence": "apache-2.0", "prov_repo": "x", "prov_index": "i"})
    return out


def test_every_sentence_once_per_pass():
    w, st = AW.build_windows(rows_for(), passes=2, seed=1)
    for l in ("amh", "hau"):
        n = sum(x["n_sent"] for x in w if x["iso"] == l)
        assert n == 3 * 12 * 2                                   # 3 papers x 12 sentences x 2 passes


def test_same_cut_for_all_languages_and_two_granularities():
    w, _ = AW.build_windows(rows_for(), passes=2, seed=3)
    for p in range(3):
        assert len({x["mode"] for x in w if x["paper"] == p and x["iso"] == "amh"}) == 2     # a paper never gets the same granularity twice
        cuts = {l: [tuple(x["sentence_ids"]) for x in w if x["paper"] == p and x["iso"] == l] for l in ("amh", "hau")}
        assert cuts["amh"] == cuts["hau"]


def test_multi_parallel_pair_vs_any():
    drop = {(0, 3, "hau"), (1, 5, "amh")}
    r = rows_for(drop=drop)
    wp, sp = AW.build_windows(r, passes=1, seed=2, multi_parallel="pair")
    wa, sa = AW.build_windows(r, passes=1, seed=2, multi_parallel="any")
    assert sum(x["n_sent"] for x in wp if x["iso"] == "amh") == 36 - 1       # amh lost only its own sentence
    assert sum(x["n_sent"] for x in wa if x["iso"] == "amh") == 36 - 2       # amh also loses the sentence removed in hau
    assert sa["dropped_any_language"] == 2


def test_gap_policy_split():
    r = rows_for(papers=1, n=10, langs=("amh",), drop={(0, 4, "amh")})
    w, st = AW.build_windows(r, passes=1, seed=5, multi_parallel="pair", gap_policy="split")
    for x in w:
        a, b = x["sentence_ids"]
        assert not (a < 4 < b)                                           # no window stitches over the removed sentence
    wj, _ = AW.build_windows(r, passes=1, seed=5, gap_policy="join")
    assert sum(x["n_sent"] for x in wj) == sum(x["n_sent"] for x in w) == 9


def test_len_outlier_flag_drops():
    r = rows_for(papers=1, n=4, langs=("amh",))
    r[1]["flag_len_ratio_outlier"] = True
    w, st = AW.build_windows(r, passes=1, seed=1)
    assert sum(x["n_sent"] for x in w) == 3 and st["dropped_len_outlier"] == {"amh": 1}


def test_unimax_total_and_caps():
    b = K.unimax_budgets([100, 1000, 1000, 10], 900, 1.0, [1, 1, 2, 1])
    assert abs(b.sum() - 900) < 1e-6 and b[3] == 10 and b[2] > b[1]


def test_licence_tags():
    assert K.licence_class("cc-by-nc-sa-4.0") == "nc" and K.licence_class("odc-by") == "commercial_ok" and K.licence_class("share_alike") == "share_alike" and K.licence_class("") == "unknown"
    p = K.provenance("longdoc", source_dataset="common-pile/libretexts", licence="cc-by")
    assert p["licence_unresolved"] and "CC BY-NC-SA" in p["licence_note"] and p["licence_class"] == "commercial_ok" and not p["uses_nllb_or_translatepsy"]
    assert not K.provenance("replay", source_dataset="wikimedia/wikipedia:20231101.en", licence="share_alike")["licence_unresolved"]


def test_forbidden_sources():
    assert K.is_forbidden_source("nllb200-teacher:x") and K.is_forbidden_source("", "TranslatePsy-AfriSLM") and not K.is_forbidden_source("HuggingFaceFW/fineweb-edu:sample-10BT")


def test_trim_prefers_high_score_and_is_proportional():
    rows = [{"iso": "a" if i % 2 else "b", "n_tokens": 10, "ens_mean": float(i)} for i in range(40)]
    kept, dropped = MX.trim(rows, 200, "archive", 1)
    assert sum(r["n_tokens"] for r in kept) == 200 and min(r["ens_mean"] for r in kept) >= 8       # lowest scores of each language fall out
    kept2, _ = MX.trim(rows, 10 ** 9, "archive", 1)
    assert len(kept2) == 40


def test_anchor_languages():
    assert len(K.ANCHOR_LANGS) == 47 and "amh" in K.ANCHOR_LANGS and len(K.WEAK45) == 45 and abs(sum(K.A2W_SHARES.values()) - 0.999) < 1e-9
    assert sum(K.A2W_TOKENS.values()) == 20_880_766
