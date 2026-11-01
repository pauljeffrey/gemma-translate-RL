# SPDX-License-Identifier: AGPL-3.0-or-later
"""Our 85 language tags -> NLLB-200 / FLORES-200 codes (R14, 2026-10-22).

kind: 'exact'  = the same language (macro-language caveats in the note); 'proxy' = NLLB covers only a related variety; 'none' = not covered.
Coverage was checked against the 202 language tokens in the tokenizer of facebook/nllb-200-distilled-600M (out/nllb_lang_codes.json).
The table is data only (no imports) so that it can be used by scripts and by gemma_mt/rl/nllb_signal.py.
"""
NLLB_MAP = {
    # iso: (nllb_code or None, kind, note)
    "yor": ("yor_Latn", "exact", ""), "hau": ("hau_Latn", "exact", ""), "ibo": ("ibo_Latn", "exact", ""),
    "pcm": (None, "none", "Nigerian Pidgin is not in NLLB-200 (nearest: eng_Latn / tpi_Latn Tok Pisin, not a substitute)"),
    "efi": (None, "none", "Efik not in NLLB-200"),
    "twi": ("twi_Latn", "exact", "NLLB has both twi_Latn and aka_Latn; FLORES+ twi = Akuapem Twi"),
    "ewe": ("ewe_Latn", "exact", ""), "fon": ("fon_Latn", "exact", ""), "wol": ("wol_Latn", "exact", ""),
    "swh": ("swh_Latn", "exact", ""), "kin": ("kin_Latn", "exact", ""), "som": ("som_Latn", "exact", ""),
    "orm": ("gaz_Latn", "exact", "West Central Oromo (gaz) = the FLORES code of 'Oromo'"),
    "amh": ("amh_Ethi", "exact", "Ge'ez script"), "tir": ("tir_Ethi", "exact", "Ge'ez script"),
    "lug": ("lug_Latn", "exact", ""), "lin": ("lin_Latn", "exact", ""), "zul": ("zul_Latn", "exact", ""), "xho": ("xho_Latn", "exact", ""),
    "afr": ("afr_Latn", "exact", ""), "sna": ("sna_Latn", "exact", ""), "nya": ("nya_Latn", "exact", ""), "sot": ("sot_Latn", "exact", ""),
    "tsn": ("tsn_Latn", "exact", ""), "mlg": ("plt_Latn", "exact", "Plateau Malagasy"),
    "ibb": (None, "none", "Ibibio not in NLLB-200 (Efik/Ibibio are not covered)"), "urh": (None, "none", "Urhobo not in NLLB-200"),
    "bem": ("bem_Latn", "exact", ""), "tum": ("tum_Latn", "exact", ""), "run": ("run_Latn", "exact", ""),
    "ven": (None, "none", "Venda not in NLLB-200 (nearest covered Bantu: tso/nso/sot, not a substitute)"),
    "tso": ("tso_Latn", "exact", ""), "nso": ("nso_Latn", "exact", ""), "ssw": ("ssw_Latn", "exact", ""),
    "nbl": (None, "none", "South Ndebele not in NLLB-200 (zul_Latn is the closest, not used)"),
    "nde": (None, "none", "North Ndebele not in NLLB-200"),
    "sag": ("sag_Latn", "exact", ""), "tiv": (None, "none", "Tiv not in NLLB-200"), "umb": ("umb_Latn", "exact", ""), "kmb": ("kmb_Latn", "exact", ""),
    "mos": ("mos_Latn", "exact", ""), "kik": ("kik_Latn", "exact", ""), "luo": ("luo_Latn", "exact", ""), "kam": ("kam_Latn", "exact", ""),
    "bin": (None, "none", "Edo not in NLLB-200"), "dag": (None, "none", "Dagbani not in NLLB-200"), "oke": (None, "none", "Okpe not in NLLB-200"),
    "shi": (None, "none", "Tachelhit (Latin/Tifinagh) not in NLLB-200; tzm_Tfng (Central Atlas Tamazight) is a related Tamazight variety, not Tachelhit"),
    "dyu": ("dyu_Latn", "exact", ""), "bam": ("bam_Latn", "exact", ""), "kbp": ("kbp_Latn", "exact", ""),
    "bbj": (None, "none", "Ghomala not in NLLB-200"), "kon": ("kon_Latn", "exact", "code exists in NLLB-200 but FLORES-200 has no kon evaluation set"),
    "kea": ("kea_Latn", "exact", ""), "fat": ("twi_Latn", "proxy", "Fante is an Akan variety; NLLB has aka_Latn/twi_Latn (Akuapem Twi)"),
    "knc": ("knc_Latn", "exact", "Central Kanuri; knc_Arab also exists (we have no SFT data for knc)"),
    "gaa": (None, "none", "Ga not in NLLB-200"), "kab": ("kab_Latn", "exact", ""), "nus": ("nus_Latn", "exact", ""),
    "swc": ("swh_Latn", "proxy", "Congo Swahili; NLLB swh_Latn is (East African) Swahili, same macro-language"),
    "lua": ("lua_Latn", "exact", "Luba-Kasai"), "kqn": (None, "none", "Kaonde not in NLLB-200"), "lun": (None, "none", "Lunda not in NLLB-200"),
    "iso": (None, "none", "Isoko not in NLLB-200"), "loz": (None, "none", "Lozi not in NLLB-200"), "wal": (None, "none", "Wolaytta not in NLLB-200"),
    "ach": (None, "none", "Acholi not in NLLB-200 (luo_Latn is Dholuo, a different language)"), "tsc": (None, "none", "Tswa not in NLLB-200"),
    "vmw": (None, "none", "Makhuwa not in NLLB-200"), "kua": (None, "none", "Kuanyama not in NLLB-200"), "seh": (None, "none", "Sena not in NLLB-200"),
    "nyn": (None, "none", "Nyankore not in NLLB-200 (lug_Latn is Luganda)"), "her": (None, "none", "Herero not in NLLB-200"), "yao": (None, "none", "Yao not in NLLB-200"),
    "sid": (None, "none", "Sidamo not in NLLB-200"), "ndc": (None, "none", "Ndau not in NLLB-200 (sna_Latn is Shona)"), "men": (None, "none", "Mende not in NLLB-200"),
    "kri": (None, "none", "Krio not in NLLB-200"), "luy": (None, "none", "Luyia not in NLLB-200"), "ish": (None, "none", "Esan not in NLLB-200"),
    "dua": (None, "none", "Duala not in NLLB-200"), "bas": (None, "none", "Basaa not in NLLB-200"), "wes": (None, "none", "Cameroon Pidgin not in NLLB-200"),
    "tzm": ("tzm_Tfng", "exact", "Central Atlas Tamazight, Tifinagh script (our tzm has no SFT data)"),
    "ful": ("fuv_Latn", "proxy", "Nigerian Fulfulde (fuv) only; 'ful' is a macro-language (Pular/Pulaar/Fulfulde varieties differ)"),
}
EXACT = {k: v[0] for k, v in NLLB_MAP.items() if v[1] == "exact"}
PROXY = {k: v[0] for k, v in NLLB_MAP.items() if v[1] == "proxy"}
