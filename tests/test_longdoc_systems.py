# SPDX-License-Identifier: AGPL-3.0-or-later
"""Long-document benchmark: the candidate-translator registry (gemma_mt/eval/longdoc_systems.py). Offline, stdlib only.

The expected numbers below were verified by hand against the model cards / papers on 2026-10-20 (see
reports/longdoc_benchmark_plan_2026-10-20_parts/system_cards.json), not generated from the registry.
"""
import re
import string

import pytest

from gemma_mt.eval import longdoc_systems as L
from gemma_mt.eval.longdoc_systems import SYSTEMS, SystemSpec
from gemma_mt.languages import SUPPORTED_RL

ALLOWED_PLACEHOLDERS = {"src_lang", "tgt_lang", "tgt_code", "text"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _placeholders(template):
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


def test_iso77_matches_supported_rl():
    assert tuple(SUPPORTED_RL[:77]) == L.ISO77                  # ISO77 = the original RL languages; 8 consensus languages were appended 2026-11-01
    assert len(L.ISO77) == 77 == len(set(L.ISO77))


def test_registry_is_not_empty_and_keys_match_names():
    assert len(SYSTEMS) >= 25
    for key, s in SYSTEMS.items():
        assert isinstance(s, SystemSpec)
        assert key == s.name


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_codes_are_clean(name):
    s = SYSTEMS[name]
    assert set(s.codes) <= set(SUPPORTED_RL), set(s.codes) - set(SUPPORTED_RL)
    assert set(s.proxy_codes) <= set(SUPPORTED_RL)
    assert not (set(s.codes) & set(s.proxy_codes)), "an iso is both supported and proxy"
    for iso, c in list(s.codes.items()) + list(s.proxy_codes.items()):
        assert isinstance(c, str) and c.strip() == c and c, (name, iso, c)
    assert isinstance(s.src_code, str) and s.src_code


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_fields_are_valid(name):
    s = SYSTEMS[name]
    assert s.family in L.FAMILIES
    assert s.prompt_style in L.PROMPT_STYLES
    assert s.licence_class in L.LICENCE_CLASSES
    assert s.backends and set(s.backends) <= set(L.BACKENDS)
    assert s.checked == "2026-10-20"
    assert s.revision == "" or HEX40.match(s.revision), s.revision
    assert s.params_b > 0 and s.max_input_tokens >= 128
    assert 0 < s.vram_gb_int8 < s.vram_gb_fp16
    assert s.weights_licence and s.licence_url.startswith("https://")
    assert s.card_url.startswith("https://")
    assert len(s.output_note) > 40 and len(s.notes) > 80
    assert "transformers 5.19" in s.notes or s.name.startswith("opus-mt-en-x") or "hf:" in s.notes
    # vram is weights only: fp16 ~ 2 bytes/param (the per-pair Marian entry quotes one small model)
    if s.name != "opus-mt-en-x":
        assert abs(s.vram_gb_fp16 - 2 * s.params_b) <= 0.15
        assert abs(s.vram_gb_int8 - s.params_b) <= 0.15


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_prompt_style_needs_are_filled(name):
    s = SYSTEMS[name]
    st = s.prompt_style
    if st in ("chat", "t5prompt"):
        assert "{text}" in s.template
        assert _placeholders(s.template) <= ALLOWED_PLACEHOLDERS, _placeholders(s.template)
        assert s.hf_id and s.hf_id_template == ""
    elif st == "nllb":
        assert s.template == "" and s.hf_id
        assert s.src_code == "eng_Latn"
        assert all(re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", c) for c in list(s.codes.values()) + list(s.proxy_codes.values()))
    elif st == "madlad":
        assert s.template == "" and s.hf_id and s.src_code == "en"
    elif st == "seamless":
        assert s.template == "" and s.hf_id and s.src_code == "eng"
    elif st == "marian":
        per_pair = s.hf_id_template != ""
        if per_pair:
            assert "{code}" in s.hf_id_template and s.hf_id == "" and s.revision == "" and s.template == ""
        else:
            assert s.hf_id and ">>{tgt_code}<<" in s.template and "{text}" in s.template
            assert _placeholders(s.template) <= ALLOWED_PLACEHOLDERS
    else:  # pragma: no cover
        raise AssertionError(st)
    if st not in ("marian",):
        assert s.hf_id_template == ""


def test_licence_classes_follow_the_licence_strings():
    for s in SYSTEMS.values():
        w = s.weights_licence.lower()
        if w.startswith("apache") or w.startswith("cc-by-4"):
            assert s.licence_class == "permissive", s.name
        if "nc" in w.split("-") or "cc-by-nc" in w:
            assert s.licence_class == "noncommercial", s.name
        if w == "gemma":
            assert s.licence_class == "gemma_terms"
        if s.licence_class == "llama":
            assert "llama" in w


def test_hand_verified_counts():
    n = L.coverage_counts()
    # NLLB: 44 of our languages are FLORES-200 languages in the tokenizer's 202-code list (same for all sizes)
    assert {n[k] for k in SYSTEMS if k.startswith("nllb")} == {44}
    # MADLAD: language in BOTH the card's 419-code list and the sentencepiece '<2xx>' vocab (same for all four checkpoints)
    assert {n[k] for k in SYSTEMS if k.startswith("madlad")} == {44}
    assert n["seamless-m4t-v2-large"] == 12
    assert n["aya-101"] == 15
    assert n["mt0-xxl-mt"] == 13
    assert n["tiny-aya-earth"] == n["tiny-aya-global"] == 10
    assert n["opus-mt-en-x"] == 35
    assert n["opus-mt-tc-bible-big-eng-mul"] == 59
    assert n["opus-mt-tc-bible-big-eng-bnt"] == 31
    assert n["opus-mt-tc-bible-big-eng-afa"] == 8
    assert n["sunflower-qwen3.8-27b"] == n["sunflower-qwen3.5-9b"] == 31
    assert n["translategemma-27b-it"] == 47
    assert n["gemma-3-27b-it"] == 2 and n["gemma-4-31b-it"] == 0
    for k, v in n.items():
        assert v == len(L.languages_covered(k)) == len(SYSTEMS[k].codes)


def test_look_alike_codes_are_handled():
    nl = SYSTEMS["nllb-200-3.3b"]
    assert nl.codes["orm"] == "gaz_Latn" and nl.codes["mlg"] == "plt_Latn"          # West Central Oromo / Plateau Malagasy
    assert nl.codes["twi"] == "twi_Latn" and nl.codes["kon"] == "kon_Latn" and nl.codes["lua"] == "lua_Latn"
    assert nl.proxy_codes["swc"] == "swh_Latn" and "swc" not in nl.codes             # Congo Swahili only via Swahili
    assert nl.proxy_codes["shi"] == "tzm_Tfng" and "shi" not in nl.codes
    assert "ven" not in nl.codes and "nbl" not in nl.codes and nl.codes["tso"] == "tso_Latn"
    md = SYSTEMS["madlad400-7b-mt"]
    assert md.codes["tso"] == "ts" and md.codes["tsc"] == "tsc" and md.codes["tsn"] == "tn"
    assert md.codes["kon"] == "kg" and md.codes["ven"] == "ve" and md.codes["nbl"] == "nr" and md.codes["nso"] == "nso"
    assert md.codes["twi"] == "ak"
    assert md.proxy_codes["lua"] == "lu" and "lua" not in md.codes                   # Luba-Katanga is not Luba-Kasai
    assert "ach" not in md.codes and "pcm" not in md.codes and "bem" not in md.codes # no <2xx> token / no data
    assert md.proxy_codes["shi"] == "ber"
    sf = SYSTEMS["sunflower-qwen3.8-27b"]
    assert sf.codes["swh"] == "swa" and "twi" not in sf.codes and sf.proxy_codes["twi"] == "aka"
    assert sf.codes["nyn"] == "nyn" and sf.codes["ach"] == "ach" and sf.codes["pcm"] == "pcm"
    tg = SYSTEMS["translategemma-12b-it"]
    assert tg.codes["zul"] == "zu" and tg.codes["nbl"] == "nr" and tg.codes["twi"] == "ak" and tg.codes["lua"] == "lua"
    assert "kab" not in tg.codes and tg.proxy_codes["kab"] == "ber-Latn" and tg.proxy_codes["nyn"] == "cgg"
    sm = SYSTEMS["seamless-m4t-v2-large"]
    assert sm.codes["orm"] == "gaz" and "hau" not in sm.codes and "lin" not in sm.codes
    op = SYSTEMS["opus-mt-en-x"]
    assert op.codes["swc"] == "swc" and op.codes["kqn"] == "kqn" and op.codes["lua"] == "lua" and "zul" not in op.codes
    assert op.hf_id_template.format(code=op.codes["kin"]) == "Helsinki-NLP/opus-mt-en-rw"


def test_flores200_codes():
    f = L.FLORES200_CODE
    assert set(f) <= set(L.ISO77)
    assert all(re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", c) for c in f.values())
    assert len(f) == 44
    assert f["orm"] == "gaz_Latn" and f["mlg"] == "plt_Latn" and f["twi"] == "twi_Latn" and f["amh"] == "amh_Ethi"
    assert "vmw" not in f and "pcm" not in f and "efi" not in f       # FLORES+ only / not in FLORES-200
    # the NLLB column is exactly FLORES-200 (NLLB's code set == FLORES-200 code set)
    assert SYSTEMS["nllb-200-distilled-600m"].codes == f
    # differs from the FLORES+ column in languages.py for twi (variant-specific FLORES+ id) and kon (column empty);
    # vmw is the reverse case: a FLORES+ code exists but FLORES-200 has none
    from gemma_mt.languages import LANGUAGES
    diff = {i for i in f if LANGUAGES[i].flores != f[i]}
    assert diff == {"twi", "kon"}, diff
    assert LANGUAGES["vmw"].flores == "vmw_Latn" and "vmw" not in f


def test_uncovered_and_licence_helpers():
    assert set(L.uncovered_languages()) == {"urh", "bin", "oke", "bbj", "yao", "sid", "ish", "dua", "wes"}
    assert L.best_licence_systems_for("urh") == []
    assert set(L.best_licence_systems_for("kqn")) == {"opus-mt-en-x"}           # sole open option, Apache-2.0
    assert set(L.best_licence_systems_for("ndc")) == {"translategemma-4b-it", "translategemma-12b-it", "translategemma-27b-it"}
    yo = L.best_licence_systems_for("yor")
    assert "madlad400-7b-mt" in yo and "sunflower-qwen3.8-27b" in yo and not any(SYSTEMS[x].licence_class != "permissive" for x in yo)
    assert L.only_nonpermissive_languages() == ["ndc"]
    for iso in L.ISO77:
        cov = L.covering_systems(iso)
        assert set(L.best_licence_systems_for(iso)) <= set(cov)


def test_matrix_renders():
    md = L.coverage_matrix_markdown()
    lines = md.strip().split("\n")
    assert len(lines) == 2 + 77
    ncols = 2 + len(SYSTEMS)
    for ln in lines:
        assert ln.count("|") == ncols + 1, ln[:80]
    assert "(p)" in md and "| gaz_Latn" in md or "gaz_Latn" in md
    row = [ln for ln in lines if ln.startswith("| urh |")][0]
    assert set(c.strip() for c in row.split("|")[3:-1]) == {"-"}
    sub = L.coverage_matrix_markdown(["madlad400-7b-mt", "sunflower-qwen3.8-27b"])
    assert sub.count("\n") == 79 and "swc" in sub
    row = [ln for ln in sub.split("\n") if ln.startswith("| swc |")][0]
    assert "sw(p)" in row and "swa(p)" in row
