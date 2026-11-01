# SPDX-License-Identifier: AGPL-3.0-or-later
import pytest

from gemma_mt import languages as L


def test_no_duplicate_tags_names_or_codes():
    isos = [l.iso for l in L._LANGUAGES]
    names = [l.name.lower() for l in L._LANGUAGES]
    codes = [l.africomet_code for l in L._LANGUAGES if l.africomet_code]
    assert len(isos) == len(set(isos))
    assert len(names) == len(set(names)), "two languages share a display name"
    assert len(codes) == len(set(codes)), "two languages share an AfriCOMET code"
    for kind in ("flores", "ntrex", "mafand"):
        vals = [getattr(l, kind) for l in L._LANGUAGES if getattr(l, kind)]
        assert len(vals) == len(set(vals)), f"duplicate {kind} codes"


def test_tags_are_three_lowercase_letters_and_aliases_resolve():
    for iso in L.ALL_ISOS:
        assert len(iso) == 3 and iso.isalpha() and iso.islower()
    for alias, target in L.ALIASES.items():
        assert target in L.LANGUAGES, alias
        assert alias not in L.LANGUAGES, f"alias {alias} shadows a canonical tag"
    assert L.canonical_iso("SWA") == "swh"
    assert L.canonical_iso("ti") == "tir"
    assert L.canonical_iso("yor_Latn") == "yor"
    assert L.canonical_iso("xxx") is None


def test_owner_display_names():
    expect = {"pcm": "Pidgin", "efi": "Efik", "fon": "Fon", "yor": "Yoruba", "hau": "Hausa", "ibo": "Igbo",
              "swh": "Swahili", "zul": "Zulu", "xho": "Xhosa", "afr": "Afrikaans", "twi": "Twi", "ewe": "Ewe",
              "wol": "Wolof", "kin": "Kinyarwanda", "nya": "Chichewa", "sna": "Shona", "sot": "Sesotho",
              "tsn": "Setswana", "som": "Somali", "orm": "Oromo"}
    for iso, name in expect.items():
        assert L.display_name(iso) == name


def test_supported_rl_is_the_77_sft_targets_in_three_groups():
    """D-37 / D-38: 77 RL languages = 42 validated for QE + 12 named on the AfriCOMET card (blended) + 23 outside AfriCOMET (consensus only)."""
    assert set(L.SUPPORTED_RL) <= set(L.ALL_ISOS) and len(L.SUPPORTED_RL) == 85 and len(L.SUPPORTED_RL[:77]) == 77
    # the repo's AFRICOMET_SUPPORTED (without eng) = the first 20 languages; D-29 (2026-10-17) added 22 more whose QE validity was measured
    # (reports/ssa_comet_vs_africomet_2026-10-15.md): 17 with human-vs-mismatched AUC >= 0.80, 5 marginal (0.70-0.80)
    repo = {"pcm", "yor", "hau", "ibo", "fon", "ewe", "twi", "efi", "afr", "zul", "sna", "nya", "xho", "sot", "kin",
            "tsn", "orm", "wol", "som", "swh"}
    added = {"amh", "tir", "lin", "lug", "mlg", "run", "bem", "tso", "nso", "tum", "mos", "kik", "luo", "bam", "kmb", "ssw", "lua",
             "umb", "kab", "kam", "kbp", "nus"}
    assert len(added) == 22 and not (added & repo)
    assert set(L.QE_VALIDATED) == repo | added and len(L.QE_VALIDATED) == 42
    # named on the AfriCOMET-QE-1.1 card (fetched 2026-10-19), SFT targets, not validated: QE blended with the consensus signal
    listed = {"loz", "ven", "iso", "tiv", "kqn", "wal", "lun", "kua", "kon", "dyu", "sag", "swc"}
    assert set(L.QE_LANGUAGES) == repo | added | listed and len(L.QE_LANGUAGES) == 54
    for iso in listed:
        assert L.LANGUAGES[iso].africomet_status == L.LISTED and L.LANGUAGES[iso].qe_covered and L.LANGUAGES[iso].rl_eligible
    # not on the card at all: consensus only (no QE call)
    consensus = {"ach", "bas", "bbj", "bin", "dag", "dua", "her", "ibb", "ish", "kea", "kri", "nbl", "ndc", "nyn", "oke", "seh", "shi", "sid", "tsc", "urh", "vmw", "wes", "yao"}
    new_consensus = {"guw", "lue", "tll", "zne", "lub", "crs", "mfe", "arz"}                    # 2026-11-01: new languages with >= 2,000 stage-B packs and a GlotLID guard
    assert set(L.CONSENSUS_ONLY) == consensus | new_consensus and len(consensus) == 23 and len(new_consensus) == 8
    consensus = consensus | new_consensus
    for iso in consensus:
        assert not L.LANGUAGES[iso].qe_covered and L.LANGUAGES[iso].rl_eligible and iso in L.SUPPORTED_RL
    assert L.LANGUAGES["ibb"].africomet_status == L.UNSUPPORTED and "ibb" in L.SUPPORTED_RL
    assert set(L.SUPPORTED_RL) == set(L.QE_LANGUAGES) | consensus
    # table rows without SFT training data (the SFT run has 77 target languages) are not RL languages
    assert not ({"nde", "fat", "knc", "gaa", "men", "luy", "tzm", "ful"} & set(L.SUPPORTED_RL))
    assert len(L.ALL_ISOS) - len(L.SUPPORTED_RL) == 40
    # still not usable at all: N'Ko / Fulfulde are no table rows (taq cjk dik were added 2026-11-01 as SFT-only targets)
    assert not ({"nqo", "fuv"} & set(L.ALL_ISOS))


def test_new_rl_languages_have_flores_codes_and_glotlid_labels():
    flores = {"amh": "amh_Ethi", "tir": "tir_Ethi", "lin": "lin_Latn", "lug": "lug_Latn", "mlg": "plt_Latn", "run": "run_Latn", "bem": "bem_Latn",
              "tso": "tso_Latn", "nso": "nso_Latn", "tum": "tum_Latn", "mos": "mos_Latn", "kik": "kik_Latn", "luo": "luo_Latn",
              "bam": "bam_Latn", "kmb": "kmb_Latn", "ssw": "ssw_Latn", "lua": "lua_Latn", "umb": "umb_Latn", "kab": "kab_Latn",
              "kam": "kam_Latn", "kbp": "kbp_Latn", "nus": "nus_Latn"}
    for iso, code in flores.items():
        l = L.LANGUAGES[iso]
        assert l.flores == code and l.africomet_code == iso and l.africomet_status == L.VERIFIED and l.rl_eligible
        # GlotLID v3 label: script-specific (Ethiopic for amh / tir; Malagasy is plt_Latn there, there is no mlg_Latn label)
        assert l.glotlid == ((code,) if iso != "mlg" else ("plt_Latn",)), iso
    # language ids (positions in the table) did not move when languages were promoted: they are stored in packed-data indexes
    assert L.ALL_ISOS.index("yor") == 0 and L.ALL_ISOS.index("ibb") == 25 and len(L.ALL_ISOS) == 125 and L.ALL_ISOS.index("guw") == 85 and L.ALL_ISOS.index("adh") == 124


def test_iso_ids_roundtrip():
    for iso in L.ALL_ISOS:
        assert L.id_to_iso(L.iso_to_id(iso)) == iso
