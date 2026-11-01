# SPDX-License-Identifier: AGPL-3.0-or-later
"""Registry of the languages of the new-language viability study (D-47): where each language's human reference sets are, what the instruction calls it,
which GlotLID labels count as "this language", and which trained relative is used as the control tag.

Kinds of reference set (all English -> X; `build_sets` in newlang_probe_sets.py reads them):
    floresorig:<code>  facebook/flores (original FLORES-200 dev), used for tzm_Tfng
    flores:<code>   openlanguagedata/flores_plus dev split (997 sentences, CC-BY-SA-4.0), English = dev/eng_Latn
    smol:<code>     google/smol smolsent/en_<code>.jsonl (CC-BY-4.0); rows with is_src_orig first
    ntrex:<code>    NTREX-128 newstest2019-ref.<code>.txt aligned to newstest2019-src.eng.txt (CC-BY-SA-4.0), local copy on the old box
    local:<path>    jsonl with src / trg in /workspace/sft_build/bench/test_sets

`trained` languages are the yardstick: they keep their own tag. New languages are prompted with `name` and (condition "prefill") the plain ISO code.
"""
from __future__ import annotations

FULA_LABELS = ("fuv_Latn", "fub_Latn", "fuh_Latn", "fuf_Latn", "fuq_Latn", "ffm_Latn", "fue_Latn", "fud_Latn", "fuv_Arab")

# key: dict(group, trained, name, tag, glot (accepted GlotLID labels), sets (first = primary), controls (trained relatives' tags), speakers (rough, millions; the order of magnitude only))
NEW = {
    # --- group A: named on the AfriCOMET card, no en->X SFT data
    "fuv": dict(group="A", name="Fula", tag="fuv", glot=FULA_LABELS, sets=["flores:fuv_Latn", "smol:ff", "smol:fuc"], controls=["wol", "hau"], speakers=40),
    "gaa": dict(group="A", name="Ga", tag="gaa", glot=("gaa_Latn",), sets=["smol:gaa"], controls=["twi", "ewe"], speakers=1.0),
    "knc": dict(group="A", name="Kanuri", tag="knc", glot=("knc_Latn", "kby_Latn"), sets=["flores:knc_Latn", "smol:kr"], controls=["hau"], speakers=10),
    "tzm": dict(group="A", name="Tamazight", tag="tzm", glot=("tzm_Tfng", "zgh_Tfng"), sets=["flores:zgh_Tfng", "smol:ber", "floresorig:tzm_Tfng"], controls=["shi", "kab"], speakers=8),
    "dik": dict(group="A", name="Dinka", tag="dik", glot=("dik_Latn", "dip_Latn"), sets=["flores:dik_Latn", "smol:din"], controls=["nus"], speakers=4.5),
    "cjk": dict(group="A", name="Chokwe", tag="cjk", glot=("cjk_Latn",), sets=["flores:cjk_Latn"], controls=["lun", "umb"], speakers=1.5),
    "taq": dict(group="A", name="Tamasheq", tag="taq", glot=("taq_Latn", "taq_Tfng"), sets=["flores:taq_Latn", "flores:taq_Tfng"], controls=["kab", "shi"], speakers=1.0),
    "nqo": dict(group="A", name="N'Ko", tag="nqo", glot=("nqo_Nkoo",), sets=["flores:nqo_Nkoo", "smol:nqo", "smol:bm-Nkoo"], controls=["bam"], speakers=5),
    # --- group A': in our language table, no SFT data
    "nde": dict(group="A'", name="North Ndebele", tag="nde", glot=("nde_Latn",), sets=["ntrex:nde", "smol:nd"], controls=["zul", "nbl"], speakers=2.5),
    # --- group B: SMOL has a human set, nothing else
    "mfe": dict(group="B", name="Mauritian Creole", tag="mfe", glot=("mfe_Latn",), sets=["flores:mfe_Latn", "smol:mfe"], controls=["kea", "pcm"], speakers=1.3),
    "dje": dict(group="B", name="Zarma", tag="dje", glot=("dje_Latn",), sets=["smol:dje"], controls=["hau", "wol"], speakers=5),
    "sus": dict(group="B", name="Susu", tag="sus", glot=("sus_Latn", "sus_Arab"), sets=["smol:sus"], controls=["bam"], speakers=1.5),
    "bci": dict(group="B", name="Baoule", tag="bci", glot=("bci_Latn",), sets=["smol:bci"], controls=["twi", "ewe"], speakers=4),
    "tig": dict(group="B", name="Tigre", tag="tig", glot=("tig_Ethi",), sets=["smol:tig"], controls=["tir", "amh"], speakers=1.1),
    "alz": dict(group="B", name="Alur", tag="alz", glot=("alz_Latn",), sets=["smol:alz"], controls=["ach", "luo"], speakers=1.5),
    "dov": dict(group="B", name="Dombe", tag="dov", glot=("dov_Latn",), sets=["smol:dov"], controls=["nya", "sna"], speakers=0.1),
}

# trained yardstick languages (own tag, own name from gemma_mt.languages)
YARDSTICK = {
    "yor": dict(sets=["flores:yor_Latn"]), "hau": dict(sets=["flores:hau_Latn"]), "swh": dict(sets=["flores:swh_Latn"]),
    "wol": dict(sets=["flores:wol_Latn"]), "kbp": dict(sets=["flores:kbp_Latn"]), "bem": dict(sets=["flores:bem_Latn"]),
    "nus": dict(sets=["flores:nus_Latn"]), "kea": dict(sets=["flores:kea_Latn"]), "dyu": dict(sets=["flores:dyu_Latn"]),
    "sag": dict(sets=["flores:sag_Latn"]), "umb": dict(sets=["flores:umb_Latn"]),
    "ach": dict(sets=["smol:ach"]), "efi": dict(sets=["smol:efi"]), "shi": dict(sets=["ntrex:shi"]),
    "nyn": dict(sets=["local:salt/en_nyn.jsonl"]), "ibb": dict(sets=["local:ibom_mt/en_ibb.jsonl"]),
}

# GlotLID labels accepted per yardstick language = the production calibration `lid` blocks; the box copy of the v6 file is read at run time, this is the fallback.
YARD_GLOT_FALLBACK = {"yor": ("yor_Latn",), "hau": ("hau_Latn",), "swh": ("swh_Latn", "swc_Latn", "swa_Latn"), "wol": ("wol_Latn",), "kbp": ("kbp_Latn",), "bem": ("bem_Latn",),
                      "nus": ("nus_Latn",), "kea": ("kea_Latn",), "dyu": ("dyu_Latn", "bam_Latn"), "sag": ("sag_Latn",), "umb": ("umb_Latn",), "ach": ("ach_Latn",),
                      "efi": ("efi_Latn", "ibb_Latn"), "shi": ("shi_Latn",), "nyn": ("nyn_Latn", "lug_Latn"), "ibb": ("ibb_Latn", "efi_Latn")}

ALL_KEYS = tuple(NEW) + tuple(YARDSTICK)
