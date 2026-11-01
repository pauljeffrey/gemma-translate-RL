"""Second-round translation sources: every pair has DECLARED languages, audited with GlotLID (ext_common.pairs); both directions are written.

AfriCOMET-QE (computed offline by twi50m_filter.py, looked up by row position) gates the NLLB-derived michsethowusu sets. Sources whose licence is
non-commercial are not converted at all (license_scan.json decides, see EXCLUDED in ext_policy.py).
"""

from __future__ import annotations

import json
import re

from sft_new import hubio
from sft_new import langdetect as L
from sft_new.compose import Rec, Skip
from sft_new.converters import register
from sft_new.ext_common import (batched, code_of, files_of, iter_any, iter_files, pairs, qe_scores, require_qe, resolve, translation_recs)
from sft_new.langaudit import audit
from sft_new.textutil import blank, clean, degenerate

QE_KEEP = 0.46            # AfriCOMET-QE failure line (below it a translation is gibberish or an omission)


def _bom(r: dict, name: str):
    return r.get(name) if name in r else r.get("﻿" + name)


@register("kweku_eng_twi", "KwekuAinoo/eng_twi (ENGLISH, TWI) -> eng<->twi translation, GlotLID-audited")
def kweku_eng_twi(rng, limit=None):
    ds = "KwekuAinoo/eng_twi"
    yield from pairs(hubio.iter_repo(ds, limit=None), lambda r: _bom(r, "ENGLISH"), lambda r: _bom(r, "TWI"), "eng", "twi", ds, rng, limit=limit)


@register("navigation_corpus", "fiifinketia/navigation-corpus-{twi,ewe,dagbani} (english, translated); Dagbani needs the <dag> tag (owner 2026-10-10)")
def navigation_corpus(rng, limit=None):
    for suffix, code in (("twi", "twi"), ("ewe", "ewe"), ("dagbani", "dag")):
        ds = f"fiifinketia/navigation-corpus-{suffix}"
        yield from pairs(hubio.iter_repo(ds), "english", "translated", "eng", code, ds, rng, limit=limit, max_chars=6000)


@register("fon_pairs", "beethogedeon/Helsinki-en-fon-opus-20k, beethogedeon/fr_fon, FrancophonIA/Fon_French_Daily_Dialogues -> eng<->fon, fra<->fon (train splits)")
def fon_pairs(rng, limit=None):
    ds = "beethogedeon/Helsinki-en-fon-opus-20k"
    rows = (r for r in hubio.iter_repo(ds, match="train"))
    yield from pairs(rows, "source_text", "translated_text", "eng", "fon", ds, rng, limit=limit)
    ds = "beethogedeon/fr_fon"
    yield from pairs(hubio.iter_repo(ds, match="train"), "french", "fon", "fra", "fon", ds, rng, limit=limit)
    ds = "FrancophonIA/Fon_French_Daily_Dialogues_Parallel_Data"
    yield from pairs(hubio.iter_repo(ds), "French_Translation", "Fon_Sentence", "fra", "fon", ds, rng, limit=limit, min_chars=3)


@register("wolof_pairs", "galsenai/{centralized_wolof_french_translation_data, english-wolof-smol-translation}, dofbi/jolof, sudoping01/english-wolof-translation, "
                         "cibfaye/wolof-english-bible -> wol<->fra, wol<->eng")
def wolof_pairs(rng, limit=None):
    ds = "galsenai/centralized_wolof_french_translation_data"
    yield from pairs(hubio.iter_repo(ds), "wo", "fr", "wol", "fra", ds, rng, limit=limit, min_chars=3)
    ds = "galsenai/english-wolof-smol-translation"
    yield from pairs(hubio.iter_repo(ds), "en", "wo", "eng", "wol", ds, rng, limit=limit, min_chars=3)
    ds = "dofbi/jolof"
    yield from pairs(hubio.iter_repo(ds, match=".parquet"), "wolof", "french", "wol", "fra", ds, rng, limit=limit, min_chars=3)
    ds = "sudoping01/english-wolof-translation"
    yield from pairs(hubio.iter_repo(ds), "en", "wo", "eng", "wol", ds, rng, limit=limit, min_chars=3)
    ds = "cibfaye/wolof-english-bible"
    yield from pairs(hubio.iter_repo(ds, match="train"), lambda r: (r.get("translation") or {}).get("en"), lambda r: (r.get("translation") or {}).get("wo"),
                     "eng", "wol", ds, rng, limit=limit)


@register("wori_wolof", "m-a-d-i/wori-wolof-instructions -> wol<->fra translation (text and instruction) + Wolof/French instruction following")
def wori(rng, limit=None):
    from sft_new import templates as T
    ds = "m-a-d-i/wori-wolof-instructions"
    rows = list(hubio.iter_repo(ds, limit=limit))
    yield from pairs(iter(rows), "text_wo", "text_fr", "wol", "fra", ds, rng, max_chars=6000)
    yield from pairs(iter(rows), "instruction_wo", "instruction_fr", "wol", "fra", ds, rng, meta={"unit": "instruction"}, min_chars=6)
    for lang, ik, tk in (("wol", "instruction_wo", "text_wo"), ("fra", "instruction_fr", "text_fr")):
        for r in rows:
            q, a = clean(r.get(ik)), clean(r.get(tk))
            if not q or not a or degenerate(a) or len(a) < 20:
                yield Skip("empty_or_short")
                continue
            yield Rec("instruction_following", ds, lang, lang, a, input="", prompt_override=q, plan=["<|chat|>"], meta={"unit": "instruction_following"})


@register("vukuzenzele", "dsfsi/vukuzenzele-sentence-aligned (train splits of every pair whose two languages we carry: afr eng zul xho sot tsn)")
def vukuzenzele(rng, limit=None):
    ds = "dsfsi/vukuzenzele-sentence-aligned"
    ours = {"afr", "eng", "zul", "xho", "sot", "tsn"}
    for p, _ in files_of(ds):
        m = re.match(r"([a-z]{3})-([a-z]{3})/train-", p)
        if not m or m.group(1) not in ours or m.group(2) not in ours:
            continue
        a, b = m.groups()
        yield from pairs(iter_files(ds, [p]), a, b, a, b, ds, rng, limit=limit, row_ok=lambda r: (r.get("score") is None or float(r["score"]) >= 0.6))


SSA_LANG = {"Amharic": "amh", "Ewe": "ewe", "Hausa": "hau", "Igbo": "ibo", "Kinyarwanda": "kin", "Nyanja": "nya", "Sesotho": "sot", "Swahili": "swh",
            "Twi": "twi", "Wolof": "wol", "Yoruba": "yor", "Zulu": "zul"}


@register("ssa_mte", "McGill-NLP/SSA-MTE train split: English source + the HUMAN reference translation (system outputs are not used) -> eng<->X")
def ssa_mte(rng, limit=None):
    ds = "McGill-NLP/SSA-MTE"
    import io
    for name, code in SSA_LANG.items():
        path = f"train/{name}.json"
        seen, rows = set(), []
        with hubio.fs().open(f"datasets/{ds}/{path}", "rb") as fh:
            for line in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                key = (o.get("src_sent"), o.get("ref"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"src": o.get("src_sent"), "ref": (o.get("ref") or "").strip(), "id": o.get("input_id")})
        # the "English" source column of some files is French (the benchmark was built from French originals): detect per row, never assume
        res = resolve([r["src"][:600] for r in rows], "eng")
        for src_code in ("eng", "fra"):
            sub = [r for r, (c, _) in zip(rows, res) if c == src_code]
            if sub:
                yield from pairs(iter(sub), "src", "ref", src_code, code, ds, rng, limit=limit, meta={"language_file": name}, min_chars=3)


COMET_FILES = {"Amharic": "amh", "Ewe": "ewe", "French": "fra", "Hausa": "hau", "Igbo": "ibo", "Kinyarwanda": "kin", "Swahili": "swh", "Yoruba": "yor",
               "Zulu": "zul", "Xhosa": "xho", "Shona": "sna", "Somali": "som", "Twi": "twi", "Wolof": "wol", "Oromo": "orm"}


@register("comet_score", "Tadesse/COMET_score (english_<Language>.csv, machine translations scored by COMET; kept at score >= 0.60 and not degenerate)")
def comet_score(rng, limit=None):
    ds = "Tadesse/COMET_score"
    for p, _ in files_of(ds):
        m = re.match(r"english_([A-Za-z]+)(?:_scores)?\.csv$", p)
        code = COMET_FILES.get(m.group(1)) if m else None
        if not code:
            yield Skip(f"language_not_supported:{p}")
            continue
        col = m.group(1)
        yield from pairs(hubio.iter_file(ds, p), "english", col, "eng", code, ds, rng, limit=limit,
                         row_ok=lambda r: float(r.get("score") or 0) >= 0.60, meta={"comet": "tadesse"})


PONTOON = {"af": "afr", "am": "amh", "ee": "ewe", "ha": "hau", "ig": "ibo", "rw": "kin", "ny": "nya", "om": "orm", "sn": "sna", "st": "sot",
           "sw": "swh", "tn": "tsn", "tw": "twi", "wo": "wol", "xh": "xho", "yo": "yor", "zu": "zul", "so": "som"}


@register("pontoon", "ayymen/Pontoon-Translations (Mozilla localisation strings, en-<locale>.tsv for the locales we carry)")
def pontoon(rng, limit=None):
    from sft_new.ext_common import read_tsv
    ds = "ayymen/Pontoon-Translations"
    for loc, code in PONTOON.items():
        try:
            rows = read_tsv(ds, f"en-{loc}.tsv")
            first = next(rows, None)
        except Exception:
            continue
        if first is None:
            continue
        import itertools
        yield from pairs(itertools.chain([first], rows), "source_string", "target_string", "eng", code, ds, rng, limit=limit, min_chars=3, max_chars=1200,
                         meta={"domain": "software_localisation"})


@register("afrolingu_mt", "UBC-NLP/AfroLingu-MT train (langcode src-tgt; only pairs whose two languages we carry)")
def afrolingu(rng, limit=None):
    ds = "UBC-NLP/AfroLingu-MT"
    path = next(p for p, _ in files_of(ds) if p.endswith("train.jsonl"))
    by: dict = {}
    n = 0
    for r in hubio.iter_file(ds, path):
        lc = str(r.get("langcode") or "")
        a, _, b = lc.partition("-")
        ca, cb = code_of(a), code_of(b)
        if not ca or not cb or ca == cb:
            yield Skip("pair_not_supported")
            continue
        by.setdefault((ca, cb), []).append({"i": r.get("input"), "o": r.get("output")})
        n += 1
    for (ca, cb), rows in by.items():
        yield from pairs(iter(rows), "i", "o", ca, cb, ds, rng, limit=limit, both=True, meta={"langcode": f"{ca}-{cb}"})


@register("mt560_qe", "michsethowusu/english-{fon,oromo,setswana}_sentence-pairs_mt560 (NLLB-derived): AfriCOMET-QE >= 0.46 AND GlotLID-audited on both sides")
def mt560_qe(rng, limit=None):
    for slug, code in (("english-fon_sentence-pairs_mt560", "fon"), ("english-oromo_sentence-pairs_mt560", "orm"), ("english-setswana_sentence-pairs_mt560", "tsn")):
        ds = f"michsethowusu/{slug}"
        scores = require_qe(slug)
        yield from pairs(hubio.iter_repo(ds), "eng", code, "eng", code, ds, rng, limit=limit, scores=scores, min_score=QE_KEEP)


@register("ghananlp_pristine", "ghananlpcommunity/pristine-twi-english-bidirectional-multilen (source=twi, target=english) -- NON-COMMERCIAL licence, OWNER OVERRIDE required")
def pristine(rng, limit=None):
    ds = "ghananlpcommunity/pristine-twi-english-bidirectional-multilen"
    yield from pairs(hubio.iter_repo(ds, match="train"), "source", "target", "twi", "eng", ds, rng, limit=limit, both=False,
                     row_ok=lambda r: str(r.get("direction") or "twi_to_english").lower().startswith(("twi", "tw", "ak")) or True)
