#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Assemble the per-language data table of the new-language study (D-47) from the raw audit files (scripts/newlang_audit_hub.py, newlang_lid_recall.py, the scans of final_v2) and the
HF crawl catalogue (data-catalog/african_hf, 2026-10-11).

    python scripts/newlang_audit_table.py --raw reports/newlang_2026-10-20/audit_raw --catalog ../data-catalog/african_hf --out reports/newlang_2026-10-20

Writes language_table.csv / language_table.md (one row per language: groups, codes, GlotLID, AfriCOMET card, human references, parallel data, monolingual data, speakers) and
other_african_languages.csv (the catalogue's other African languages with the same columns that can be filled mechanically).

Hand-written columns (`speakers_m`, codes, notes) are recollection, marked as such in the report; every number that comes from a file is measured by the audit scripts.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

csv.field_size_limit(10 ** 9)

# key -> hand-written metadata. sft_labels: label(s) of final_v2 (source_lang / targ_lang / language columns) that are this language
META = {
    "fuv": dict(name="Fula / Fulfulde", groups="A,B", iso="fuv (Nigerian Fulfulde), ful (macro), ff (639-1), fuc (Pulaar, SMOL), fub fuh fuq fuf fue ffm fui (varieties)", sft_labels=["ful"], seed=["fuv_Latn"],
                flores=["fuv_Latn"], smol=["ff", "fuc"], card="fuv", speakers_m="40-65 (all varieties, L1+L2)", wiki=["ff"], bible=dict(ffm=7936, fuf=7943, fuh=7949, fue=4669),
                catalog=["fuv", "ful", "fub", "fuh", "fuq", "fui"], fw2=["fuv_Latn", "fuf_Latn", "ffm_Latn", "fuh_Latn", "fub_Latn", "fue_Latn", "fuq_Latn"], madlad=["ff", "ffm"]),
    "gaa": dict(name="Ga", groups="A,B", iso="gaa", sft_labels=["gaa"], seed=[], flores=[], smol=["gaa"], card="gaa", speakers_m="~1", wiki=[], bible={}, catalog=["gaa"], fw2=["gaa_Latn"], madlad=[]),
    "knc": dict(name="Kanuri", groups="A,B", iso="knc (Central), kr (639-1), kby (Manga)", sft_labels=["knc"], seed=["knc_Latn", "knc_Arab"], flores=["knc_Latn", "knc_Arab"], smol=["kr"], card="knc",
                speakers_m="~5-10", wiki=["kr (closed)"], bible={}, catalog=["knc", "kby"], fw2=["knc_Latn", "kby_Latn"], madlad=["kr (canary only)"]),
    "tzm": dict(name="Tamazight (Central Atlas) / Standard Moroccan Tamazight", groups="A", iso="tzm (Central Atlas, Tifinagh), zgh (Standard Moroccan), shi (Tachelhit: trained as `shi`), ber (SMOL: Tifinagh)",
                sft_labels=[], seed=["zgh_Tfng"], flores=["zgh_Tfng (FLORES+)", "tzm_Tfng (FLORES-200 original)"], smol=["ber"], card="tzm", speakers_m="~5-8 (tzm); zgh is a standard register",
                wiki=["zgh", "shi (Tachelhit)"], bible={}, catalog=["tzm", "zgh", "ber"], fw2=["tzm_Tfng", "zgh_Tfng"], madlad=["tzm (canary only)", "ber_Latn"]),
    "dik": dict(name="Dinka", groups="A,B", iso="dik (Southwestern), din (macro, SMOL), dip (NE), dks (Southeastern)", sft_labels=["din"], seed=["dik_Latn"], flores=["dik_Latn"], smol=["din"], card="dik", speakers_m="~4.5",
                wiki=["din"], bible=dict(dik=7905), catalog=["din"], fw2=["dik_Latn", "dks_Latn"], madlad=["din (canary only)"]),
    "cjk": dict(name="Chokwe", groups="A", iso="cjk", sft_labels=[], seed=[], flores=["cjk_Latn"], smol=[], card="cjk", speakers_m="~1.5", wiki=[], bible={}, catalog=[], fw2=["cjk_Latn"], madlad=[]),
    "taq": dict(name="Tamasheq", groups="A", iso="taq (Tamasheq; Latin + Tifinagh), tmh (Tamashek macro), ttq (Tawallammat)", sft_labels=["tmh"], seed=["taq_Latn", "taq_Tfng"], flores=["taq_Latn", "taq_Tfng"], smol=[], card="taq",
                speakers_m="~1 (Tuareg ~2-3)", wiki=[], bible={}, catalog=[], fw2=["taq_Latn", "taq_Tfng", "ttq_Latn"], madlad=["taq (canary only)"]),
    "nqo": dict(name="N'Ko (script of the Manding languages)", groups="A,B", iso="nqo (N'Ko script; the language is Manding, near Bambara), bm-Nkoo (SMOL: identical text)", sft_labels=[], seed=["nqo_Nkoo"],
                flores=["nqo_Nkoo"], smol=["nqo", "bm-Nkoo"], card="nqo", speakers_m="users: ~0.3-1; Manding L1 ~40", wiki=["nqo"], bible={}, catalog=[], fw2=["nqo_Nkoo"], madlad=[]),
    "nde": dict(name="North Ndebele", groups="A' (table), B", iso="nde (639-3), nd (639-1, SMOL)", sft_labels=["nde"], seed=[], flores=[], smol=["nd"], card=None, speakers_m="~2.5-3", wiki=[], bible={}, catalog=["nde"], fw2=["nde_Latn"], madlad=[], ntrex="nde (1,997)"),
    "fat": dict(name="Fante", groups="A'", iso="fat (Fante; Akan dialect, SMOL ak = Akan is trained as twi)", sft_labels=["fat"], seed=[], flores=[], smol=[], card=None, speakers_m="~3 (Akan ~22)", wiki=[], bible={}, catalog=["fat"], fw2=["fat_Latn"], madlad=[]),
    "men": dict(name="Mende", groups="A'", iso="men", sft_labels=["men"], seed=[], flores=[], smol=[], card=None, speakers_m="~1.5", wiki=[], bible={}, catalog=["men"], fw2=["men_Latn"], madlad=[]),
    "luy": dict(name="Luyia (Luhya cluster)", groups="A'", iso="luy (macro); varieties lri lko lsm lwg bxk nyd ...", sft_labels=[], seed=[], flores=[], smol=[], card=None, speakers_m="~6-7 (cluster)", wiki=[], bible={}, catalog=["luy"], fw2=["lsm_Latn", "lwg_Latn"], madlad=[]),
    "mfe": dict(name="Mauritian Creole", groups="B", iso="mfe", sft_labels=[], seed=[], flores=["mfe_Latn"], smol=["mfe"], card=None, speakers_m="~1.3", wiki=[], bible={}, catalog=[], fw2=["mfe_Latn"], madlad=["mfe"]),
    "dje": dict(name="Zarma", groups="B", iso="dje", sft_labels=[], seed=[], flores=[], smol=["dje"], card=None, speakers_m="~5", wiki=[], bible={}, catalog=[], fw2=["dje_Latn"], madlad=["dje"]),
    "sus": dict(name="Susu", groups="B", iso="sus", sft_labels=[], seed=[], flores=[], smol=["sus"], card=None, speakers_m="~1.5", wiki=[], bible=dict(sus=62306), catalog=["sus"], fw2=["sus_Latn", "sus_Arab"], madlad=[]),
    "bci": dict(name="Baoule", groups="B", iso="bci", sft_labels=[], seed=[], flores=[], smol=["bci"], card="bci", speakers_m="~4", wiki=[], bible={}, catalog=["bci"], fw2=["bci_Latn"], madlad=["bci"]),
    "tig": dict(name="Tigre", groups="B", iso="tig", sft_labels=[], seed=[], flores=[], smol=["tig"], card=None, speakers_m="~1", wiki=[], bible={}, catalog=["tig"], fw2=["tig_Ethi"], madlad=[]),
    "alz": dict(name="Alur", groups="B", iso="alz", sft_labels=[], seed=[], flores=[], smol=["alz"], card=None, speakers_m="~1.5", wiki=[], bible={}, catalog=[], fw2=["alz_Latn"], madlad=["alz"]),
    "dov": dict(name="Dombe", groups="B", iso="dov", sft_labels=[], seed=[], flores=[], smol=["dov"], card=None, speakers_m="<0.5 (a Tonga-cluster variety)", wiki=[], bible=dict(dov=7952), catalog=[], fw2=[], madlad=["dov"]),
    "mnk": dict(name="Mandinka", groups="C", iso="mnk (Mandinka); man (macro), msc (Maninka, Sankaran)", sft_labels=[], seed=[], flores=[], smol=[], card=None, speakers_m="~1.5-2 (Gambia, Senegal)", wiki=["bm (Bambara, other language)"], bible=dict(msc=2930),
                catalog=[], fw2=["mnk_Latn", "msc_Latn"], madlad=[]),
    "snk": dict(name="Soninke", groups="C", iso="snk", sft_labels=[], seed=[], flores=[], smol=[], card=None, speakers_m="~2-3", wiki=[], bible={}, catalog=[], fw2=[], madlad=[]),
    "srr": dict(name="Serer", groups="C", iso="srr (Serer-Sine)", sft_labels=[], seed=[], flores=[], smol=[], card=None, speakers_m="~1.5", wiki=[], bible={}, catalog=[], fw2=["srr_Latn"], madlad=[]),
    "mas": dict(name="Maasai", groups="C", iso="mas", sft_labels=[], seed=[], flores=[], smol=[], card=None, speakers_m="~1.5", wiki=[], bible={}, catalog=["mas"], fw2=["mas_Latn", "kpe_Latn"], madlad=["mas"]),
    "kln": dict(name="Kalenjin", groups="C", iso="kln (macro); sgc Kipsigis, niq Nandi, enb Markweeta", sft_labels=["kln"], seed=[], flores=[], smol=[], card=None, speakers_m="~5-6 (cluster)", wiki=[], bible={}, catalog=["kln"], fw2=["sgc_Latn", "enb_Latn"], madlad=["niq"]),
    "mer": dict(name="Meru (Kimeru)", groups="C", iso="mer", sft_labels=[], seed=[], flores=[], smol=[], card=None, speakers_m="~2", wiki=[], bible={}, catalog=["mer"], fw2=["mer_Latn"], madlad=[]),
}

GLOT_MAP = {"fuv": "fuv_Latn fub_Latn fuh_Latn fuf_Latn fuq_Latn ffm_Latn fue_Latn fud_Latn fuv_Arab", "gaa": "gaa_Latn", "knc": "knc_Latn knc_Arab kby_Latn", "tzm": "tzm_Tfng zgh_Tfng shi_Latn",
            "dik": "dik_Latn dip_Latn dks_Latn", "cjk": "cjk_Latn", "taq": "taq_Latn taq_Tfng ttq_Latn ttq_Tfng thv_Latn", "nqo": "nqo_Nkoo", "nde": "nde_Latn", "fat": "fat_Latn", "men": "men_Latn",
            "luy": "(no luy label; lsm_Latn lwg_Latn xog_Latn exist)", "mfe": "mfe_Latn", "dje": "dje_Latn", "sus": "sus_Latn sus_Arab", "bci": "bci_Latn", "tig": "tig_Ethi", "alz": "alz_Latn",
            "dov": "(no dov label; GlotLID reads Dombe as toi_Latn)", "mnk": "mnk_Latn msc_Latn", "snk": "(none)", "srr": "srr_Latn", "mas": "mas_Latn kpe_Latn(Kpelle, other)", "kln": "(no kln label; sgc_Latn niq_Latn enb_Latn)",
            "mer": "mer_Latn"}

# key -> best English-pivot / parallel repos [(repo, rows, licence, note)]
PARALLEL = {
    "fuv": [("michsethowusu/english-fulah_sentence-pairs", 1511069, "none declared", "NLLB-mined, noisy"), ("michsethowusu/french-fulah_sentence-pairs", 701161, "none declared", "NLLB-mined fr-ff"),
            ("ayymen/Pontoon-Translations en-ff", 15359, "MPL-2.0", "software strings"), ("Tamazight-NLP/Weblate-Translations en-ff", 2705, "other", "software strings"),
            ("openlanguagedata/oldi_seed fuv_Latn", 6193, "CC-BY-SA-4.0", "human translation of NLLB-Seed English"), ("michsethowusu/Code-170k-fulani", 176999, "apache-2.0", "machine-translated code instructions"),
            ("nordpolemil/biblenlp-corpus ffm/fuf/fuh/fue", 28497, "per-row CC (bible)", "NT, 4.7-7.9k verses per variety")],
    "gaa": [("AfriSpeech/african-corpus-jw Ga_gaa_v1.csv (55 MB, verse-aligned to English)", 350000, "other (jw.org copyright)", "rows estimated from bytes"), ("ghanaopenai/ghana-farmer-qa-ga", 151427, "CC-BY-NC-4.0", "farmer Q&A in Ga with English side (NC)"), ("MaLA-LM/PolyWrite (in our corpus)", 66, "?", "")],
    "knc": [("CLEAR-Global/Synthetic-Text (in our corpus)", 583, "none declared", "synthetic"), ("ciatech-africa/kanuri-english-lexicon", 26, "none declared", "lexicon"), ("openlanguagedata/oldi_seed knc_Latn/knc_Arab", 6193, "CC-BY-SA-4.0", "human"),
            ("michsethowusu/Code-170k-kanuri", 176999, "apache-2.0", "machine-translated code instructions")],
    "tzm": [("michsethowusu/english-central_atlas_tamazight_sentence-pairs", 212771, "none declared", "NLLB-mined"), ("michsethowusu/english-tachelhit_sentence-pairs_mt560", 15288, "CC-BY-4.0", "JW300-derived (Tachelhit)"),
            ("Tamazight-NLP/Weblate-Translations en-zgh/en-tzm/en-ber", 12000, "other", "software strings"), ("ayymen/Pontoon-Translations en-zgh/en-tzm", 1660, "MPL-2.0", ""),
            ("openlanguagedata/oldi_seed zgh_Tfng", 6193, "CC-BY-SA-4.0", "human")],
    "dik": [("michsethowusu/dinka-english_sentence-pairs", 159138, "none declared", "NLLB-mined"), ("michsethowusu/english-southwestern-dinka_sentence-pairs_mt560", 13319, "CC-BY-4.0", "JW300-derived"),
            ("nordpolemil/biblenlp-corpus dik", 7905, "per-row CC (bible)", "NT"), ("openlanguagedata/oldi_seed dik_Latn", 6193, "CC-BY-SA-4.0", "human"), ("michsethowusu/dinka-english-emotions-corpus", 38798, "none declared", "emotion texts"),
            ("(our corpus: din label)", 42050, "?", "sentiment/emotion text, no translation task")],
    "cjk": [("michsethowusu/english-chokwe_sentence-pairs_mt560", 35767, "CC-BY-4.0", "JW300-derived")],
    "taq": [("michsethowusu/english-tamasheq_sentence-pairs", 450227, "none declared", "NLLB-mined"), ("michsethowusu/english-tamashek_sentence-pairs_mt560", 5363, "CC-BY-4.0", "JW300-derived"),
            ("openlanguagedata/oldi_seed taq_Latn/taq_Tfng", 6193, "CC-BY-SA-4.0", "human")],
    "nqo": [("openlanguagedata/oldi_seed nqo_Nkoo", 6193, "CC-BY-SA-4.0", "human"), ("Tamazight-NLP/Weblate-Translations en-nqo", 744, "other", "software strings")],
    "nde": [("dsfsi/umsuka-english (in our corpus)", 140, "CC-BY-4.0", "")],
    "fat": [("AfriSpeech/african-corpus-jw Fante_fat_v1.csv (11.8 MB)", 75000, "other (jw.org copyright)", "rows estimated from bytes"), ("Ghana-NLP/FANTE_ENGLISH_PARALLEL_TEXT (in our corpus)", 1863, "none declared", "")],
    "men": [("michsethowusu/english-mende_sentence-pairs_mt560", 1668, "CC-BY-4.0", "JW300-derived (tiny)")],
    "luy": [],
    "mfe": [("AfriSpeech/african-corpus-jw Mauritian_Creole_mfe_v1.csv (32.6 MB)", 200000, "other (jw.org copyright)", "rows estimated from bytes"), ("michsethowusu/english-morisyen_sentence-pairs_mt560", 181560, "CC-BY-4.0", "JW300-derived")],
    "dje": [("michsethowusu/english-zarma_sentence-pairs_mt560", 60515, "CC-BY-4.0", "JW300-derived")],
    "sus": [("2ADT-Consulting/susu-parallel", 267369, "other (YouVersion ToS derived)", "Bible"), ("nordpolemil/biblenlp-corpus sus", 62306, "per-row CC (bible)", "")],
    "bci": [("AfriSpeech/african-corpus-jw Baoule_bci_v1.csv (24.5 MB)", 150000, "other (jw.org copyright)", "rows estimated from bytes")],
    "tig": [("ayymen/Pontoon-Translations en-tig + Weblate en_GB-tig", 4188, "MPL-2.0 / other", "software strings")],
    "alz": [("michsethowusu/english-alur_sentence-pairs_mt560", 7567, "CC-BY-4.0", "JW300-derived")],
    "dov": [("nordpolemil/biblenlp-corpus dov", 7952, "per-row CC (bible)", "NT"), ("michsethowusu/english-tonga_sentence-pairs_mt560", 3898, "CC-BY-4.0", "Tonga (toi), NOT Dombe")],
    "mnk": [("nordpolemil/biblenlp-corpus msc (Maninka)", 2930, "per-row CC (bible)", "NT portion")], "snk": [], "srr": [], "mas": [], "kln": [], "mer": [],
}


def load(p):
    return json.load(open(p, encoding="utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    raw = lambda n: load(os.path.join(a.raw, n))                  # noqa: E731
    lid = raw("lid_recall.json")
    card = raw("card.json")["masakhane/africomet-qe-stl-1.1"]["readme"]
    card_codes = set(re.findall(r"^- .+? \(([a-z]{3})\)", card, re.M))
    sz2 = raw("dataset_sizes2.json")
    fw = raw("fineweb2_rows.json")
    scan = raw("scan_final_v2_newlangs.json")
    wiki = raw("wiki.json")
    corp = raw("corpora.json")

    def rows_sft(labels):
        tr = tot = 0
        for r in scan:
            if r["language"] in labels or r["source_lang"] in labels or r["targ_lang"] in labels:
                tot += r["rows"]
                if r["task"] == "translation" and ("eng" in (r["source_lang"], r["targ_lang"])):
                    tr += r["rows"]
        return tr, tot

    def cat_counts(codes):
        out = {}
        for b in ("rl", "pretrain", "sft"):
            n = 0
            for c in codes:
                for f in glob.glob(os.path.join(a.catalog, b, f"{c}_*.csv")):
                    n += sum(1 for _ in csv.DictReader(open(f, encoding="utf-8")))
            out[b] = n
        return out

    def cat_top(codes, bucket="rl", k=3):
        rows = []
        for c in codes:
            for f in glob.glob(os.path.join(a.catalog, bucket, f"{c}_*.csv")):
                for r in csv.DictReader(open(f, encoding="utf-8")):
                    if r["scope"] in ("african_only", "african_plus_pivot"):
                        try:
                            n = int(float(r["num_rows"] or 0))
                        except ValueError:
                            n = 0
                        rows.append((n, r["dataset_id"], r["license"], r["commercial_use"]))
        rows.sort(reverse=True)
        return "; ".join(f"{d} ({n:,}, {l}, {cu})" for n, d, l, cu in rows[:k])

    table = []
    for key, m in META.items():
        best_recall, best_set = None, None
        for k2, v in lid.items():
            if k2.startswith(key + "__") and isinstance(v, dict) and "recall" in v:
                if best_recall is None or v["recall"] > best_recall:
                    best_recall, best_set = v["recall"], k2.split("__", 1)[1]
        tr, tot = rows_sft(m["sft_labels"])
        fw_docs = sum(fw["HuggingFaceFW/fineweb-2"].get(c, 0) for c in m["fw2"])
        ft_docs = sum(fw["HuggingFaceFW/finetranslations"].get(c, 0) for c in m["fw2"])
        wiki_txt = ", ".join(f"{c.split(' ')[0]}: {wiki.get(c.split(' ')[0], {}).get('articles', 'closed/none')}" for c in m["wiki"]) if m["wiki"] else "none"
        par = PARALLEL.get(key, [])
        ccount = cat_counts(m["catalog"]) if m["catalog"] else {"rl": 0, "pretrain": 0, "sft": 0}
        row = dict(key=key, language=m["name"], groups=m["groups"], iso_codes=m["iso"], glotlid_labels=GLOT_MAP.get(key, ""), glotlid_recall_on_human_text=best_recall, glotlid_recall_set=best_set,
                   africomet_card=("yes (" + m["card"] + ")") if m["card"] in card_codes else "no", flores_plus=", ".join(m["flores"]) or "no", smolsent=", ".join(m["smol"]) or "no",
                   ntrex=m.get("ntrex", "no"), nllb_seed=", ".join(m["seed"]) + (" (6,193 sentences each)" if m["seed"] else "") if m["seed"] else "no",
                   our_sft_translation_rows_with_english=tr, our_corpus_rows_any_task=tot,
                   parallel_hoard="; ".join(f"{r}: {n:,} [{l}]{(' - ' + nt) if nt else ''}" for r, n, l, nt in par) or "none found",
                   parallel_rows_max=max([n for _, n, _, nt in par], default=0),
                   parallel_rows_max_natural=max([n for _, n, l, nt in par if not any(w in (nt + l) for w in ("synthetic", "machine-translated", "NC", "emotion", "lexicon", "software", "Q&A"))], default=0),
                   fineweb2_train_parquet_kb=round(sum(sz for c in m["fw2"] for f, sz in corp["HuggingFaceFW/fineweb-2"]["hits"].get(c.split("_")[0], []) if f"/{c}/train/" in f) / 1024), fineweb2_docs=fw_docs, finetranslations_docs=ft_docs, wikipedia_articles=wiki_txt,
                   madlad=", ".join(m["madlad"]) or "no", catalog_rl_datasets=ccount["rl"], catalog_pretrain_datasets=ccount["pretrain"], catalog_top_parallel=cat_top(m["catalog"]) if m["catalog"] else "",
                   speakers_m=m["speakers_m"])
        table.append(row)

    os.makedirs(a.out, exist_ok=True)
    cols = list(table[0].keys())
    with open(os.path.join(a.out, "language_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)

    # ---- appendix: the catalogue's other African languages that we do not train
    ours = set(json.load(open(os.path.join(a.raw, "scan_final_v2_langs.json")))["language"])
    glot = json.load(open(os.path.join(a.raw, "glotlid_labels.json"))) if os.path.exists(os.path.join(a.raw, "glotlid_labels.json")) else []
    man = list(csv.DictReader(open(os.path.join(a.catalog, "MANIFEST.csv"), encoding="utf-8")))
    seen = {}
    for r in man:
        if r["bucket"] != "pretrain":
            continue
        seen[r["language_iso3"]] = (r["language"], int(r["datasets"]))
    skip = set(META) | {"ful", "tzm", "zgh", "din", "tmh", "ber"}
    appendix = []
    flo = corp["openlanguagedata/flores_plus"]["hits"] if "hits" in corp.get("openlanguagedata/flores_plus", {}) else {}
    for iso, (name, nds) in sorted(seen.items()):
        if iso in skip:
            continue
        gl = sorted({l for l in glot if l.split("_")[0] == iso})
        fwd = sum(v for k, v in fw["HuggingFaceFW/fineweb-2"].items() if k.split("_")[0] == iso)
        appendix.append(dict(iso=iso, language=name, catalog_pretrain_datasets=nds, glotlid_labels=" ".join(gl), fineweb2_docs=fwd, finetranslations_docs=sum(v for k, v in fw["HuggingFaceFW/finetranslations"].items() if k.split("_")[0] == iso)))
    with open(os.path.join(a.out, "other_african_languages.csv"), "w", newline="", encoding="utf-8") as f:
        if appendix:
            w = csv.DictWriter(f, fieldnames=list(appendix[0].keys()))
            w.writeheader()
            w.writerows(appendix)
    print("rows", len(table), "appendix", len(appendix))


if __name__ == "__main__":
    main()
