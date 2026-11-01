"""Third round (owner 2026-10-11): the datasets of queue_2026-10-11.json.

Same rules as ext_*.py: streamed, declared languages audited with GlotLID, both translation directions, instruction pools, tool calls in the v2
format, no coding data, non-commercial licences skipped before conversion (ext_policy / the licence scan). A language with no tag in the tokenizer
is not invented: its rows are dropped with the reason `assembly:input_lang '<code>' is not a supported language tag`, which is how the build report
counts the languages that would need a tag swap (first come, first served).
"""

from __future__ import annotations

import itertools
import json
import re

from sft_new import hubio
from sft_new import langdetect as L
from sft_new import templates as T
from sft_new.compose import A, Rec, Skip
from sft_new.converters import register, search_tool_def, SEARCH_TOOLS, tools_with_distractors, WIKI_SYSTEM
from sft_new.ext_common import (batched, code_of, files_of, iter_any, iter_files, label_stream, pairs, read_tsv, resolve, translation_recs, unwrap_markup)
from sft_new import ext_instruct as EI          # during a circular import this module object is only partly filled: names are looked up at call time
from sft_new.rag_tools import rag_convo
from sft_new.taskguess import guess
from sft_new.textutil import blank, clean, degenerate, rep_ratio

from sft_new.ext_common import lang   # noqa: E402  (shared with ext_crawl / ext_mich; kept importable from here)


def _none(*xs):
    return any(x is None for x in xs)


# ======================================================================= translation families

def _pairs_or_skip(rows, ka, kb, la, lb, ds, rng, limit=None, **kw):
    if _none(la, lb):
        yield Skip(f"language_unknown:{la}>{lb}")
        return
    yield from pairs(rows, ka, kb, la, lb, ds, rng, limit=limit, **kw)


@register("pidgin_pairs", "coderGit/Eng-PidginBioData, coderGit/Eng_PidginEdu (en, pcm columns) -> eng<->pcm translation")
def pidgin_pairs(rng, limit=None):
    for repo in ("coderGit/Eng-PidginBioData", "coderGit/Eng_PidginEdu"):
        for path, _ in files_of(repo, patterns=(".csv",)):
            first = next(iter(hubio.iter_file(repo, path)), None)
            if not first:
                continue
            keys = {k.lower().strip(): k for k in first}
            ka = next((keys[k] for k in keys if k in ("en", "eng", "english") or k.startswith("english")), None)
            kb = next((keys[k] for k in keys if k in ("pcm", "pidgin") or k.startswith("pidgin") or "pcm" in k), None)
            if not ka or not kb:
                yield Skip(f"no_en_pcm_columns:{path[:40]}")
                continue
            yield from pairs(hubio.iter_file(repo, path), ka, kb, "eng", "pcm", repo, rng, limit=limit, min_chars=3, meta={"file": path[:60]})


@register("hypa_fleurs", "hypaai/Hypa_Fleurs (english_text, naija_text; one file per language, audio columns never read) -> eng<->hau/ibo/yor")
def hypa_fleurs(rng, limit=None):
    ds = "hypaai/Hypa_Fleurs"
    for path, _ in files_of(ds, patterns=(".parquet",)):
        code = {"hausa": "hau", "igbo": "ibo", "yoruba": "yor"}.get(path.split("/")[-1].split("-")[0])
        if code:
            yield from pairs(hubio.iter_file(ds, path, columns=["english_text", "naija_text"]), "english_text", "naija_text", "eng", code, ds, rng, limit=limit,
                             meta={"source_corpus": "FLORES-derived (FLEURS text)"})


@register("afriscience_mt", "dsfsi/afriscience_mt corpus/train + glossary terms (dev and test are evaluation splits) -> translation")
def afriscience(rng, limit=None):
    ds = "dsfsi/afriscience_mt"
    for path, _ in files_of(ds, patterns=(".parquet",)):
        if not (path.startswith("corpus/train") or path.startswith("glossary/")):
            continue
        by: dict = {}
        for r in hubio.iter_file(ds, path):
            la, lb = lang(r.get("source_lang")), lang(r.get("target_lang"))
            by.setdefault((la, lb), []).append(r)
        for (la, lb), rows in by.items():
            yield from _pairs_or_skip(iter(rows), "source", "target", la, lb, ds, rng, limit=limit, min_chars=2, meta={"domain": "science", "part": path.split("/")[0]})


SA_LOC = {"af": "afr", "nr": "nbl", "nso": "nso", "ss": "ssw", "st": "sot", "tn": "tsn", "ts": "tso", "ve": "ven", "xh": "xho", "zu": "zul", "en": "eng"}


@register("mehgoss_sa", "Mehgoss/sa-languages-translation train (<xx>-en folders: text, en_text) -> <xx><->eng translation")
def mehgoss(rng, limit=None):
    ds = "Mehgoss/sa-languages-translation"
    for path, _ in files_of(ds, patterns=(".parquet",)):
        m = re.match(r"([a-z]+)-en/train-", path)
        if m:
            code = SA_LOC.get(m.group(1))
            yield from _pairs_or_skip(hubio.iter_file(ds, path), "text", "en_text", code, "eng", ds, rng, limit=limit)


@register("sa_parallel", "sello-ralethe/SA-Parallel-Corpora train (source_en, target; per eng-xxx folder) -> eng<->xxx")
def sa_parallel(rng, limit=None):
    ds = "sello-ralethe/SA-Parallel-Corpora"
    for path, _ in files_of(ds, patterns=(".csv",)):
        m = re.match(r"data/eng-([a-z]+)/train\.csv", path)
        if m:
            code = lang(m.group(1)) or SA_LOC.get(m.group(1))
            yield from _pairs_or_skip(hubio.iter_file(ds, path), "source_en", "target", "eng", code, ds, rng, limit=limit,
                                      row_ok=lambda r: str(r.get("split") or "train").lower() in ("train", ""))


@register("multi_hall_sa", "sello-ralethe/Multi-Hall-SA (factual statements and the hallucinated ones, English <-> isiXhosa / isiZulu / Sepedi / Sesotho) -> translation")
def multi_hall(rng, limit=None):
    ds = "sello-ralethe/Multi-Hall-SA"
    for path, _ in files_of(ds, patterns=(".csv",)):
        m = re.match(r"data/([a-z]+)\.csv", path)
        code = lang(m.group(1)) if m else None
        if not m:
            continue
        rows = list(hubio.iter_file(ds, path))
        yield from _pairs_or_skip(iter(rows), "english_factual_statement", "target_factual_statement", "eng", code, ds, rng, limit=limit, meta={"kind": "factual"}, min_chars=8)
        yield from _pairs_or_skip(iter(rows), "english_hallucinated_statement", "target_hallucinated_statement", "eng", code, ds, rng, limit=limit,
                                  meta={"kind": "hallucinated_statement_translated"}, min_chars=8)


@register("sa_knowledge", "sello-ralethe/SA-Knowledge + SA-QA-Eval: question <-> translated_question as translation; choices/answerKey as multiple choice ONLY for SA-Knowledge")
def sa_knowledge(rng, limit=None):
    for ds, mcq in (("sello-ralethe/SA-Knowledge", True), ("sello-ralethe/SA-QA-Eval", False)):
        for path, _ in files_of(ds, patterns=(".csv", ".jsonl")):
            stem = path.rsplit("/", 1)[-1]
            m = re.match(r"([a-z]{3}|isi[a-z]+|sepedi|sesotho|setswana|xitsonga|tshivenda)[_ .]", stem.lower())
            code = lang(m.group(1)) if m else None
            if code is None:
                yield Skip(f"language_from_file_name_unknown:{stem[:30]}")
                continue
            rows = list(iter_any(ds, path))
            yield from _pairs_or_skip(iter(rows), "question", "translated_question", "eng", code, ds, rng, limit=limit, meta={"file": stem[:50]})


@register("clear_global_synthetic", "CLEAR-Global/Synthetic-Text (English sentence + target-language sentence; the language is the file name) -> translation")
def clear_global(rng, limit=None):
    ds = "CLEAR-Global/Synthetic-Text"
    for path, _ in files_of(ds, patterns=(".csv",)):
        code = lang(path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
        rows = list(hubio.iter_file(ds, path))
        if not rows:
            continue
        tgt_key = next((k for k in rows[0] if k.startswith("Sentence (Translation")), None)
        if not tgt_key:
            continue
        yield from _pairs_or_skip(iter(rows), "Sentence (English)", tgt_key, "eng", code, ds, rng, limit=limit, meta={"synthetic": True})


@register("okwu_parallel", "Okwu/african-language-parallel-corpus (source_text, english_translation; pcm, swa, yor) -> translation")
def okwu(rng, limit=None):
    ds = "Okwu/african-language-parallel-corpus"
    for path, _ in files_of(ds, patterns=(".csv",)):
        m = re.match(r"data/([a-z]{3})-eng/", path)
        code = {"pcm": "pcm", "swa": "swh", "yor": "yor"}.get(m.group(1)) if m else None
        if code:
            yield from pairs(hubio.iter_file(ds, path), "source_text", "english_translation", code, "eng", ds, rng, limit=limit)


@register("kentrans", "Kencorpus/KenTrans (swa <-> dho/lbk/lch/llg) -> translation; lbk/lch/llg have no tag yet and are counted as such")
def kentrans(rng, limit=None):
    ds = "Kencorpus/KenTrans"
    for path, _ in files_of(ds, patterns=(".parquet",)):
        m = re.match(r"([a-z]{3})/train\.parquet", path)
        if not m:
            continue
        rows = list(hubio.iter_file(ds, path))
        la = lang(rows[0].get("src_lang")) if rows else None
        lb = lang(rows[0].get("tgt_lang")) if rows else None
        la = {"dho": "luo"}.get(la, la)
        lb = {"dho": "luo"}.get(lb, lb)
        yield from _pairs_or_skip(iter(rows), "source", "target", la, lb, ds, rng, limit=limit, meta={"dialect": m.group(1)})


@register("za_mafoko", "dsfsi/za-mafoko-dsac terminology lists (one row = one term in up to 11 languages) -> term translation between the languages that have a tag")
def za_mafoko(rng, limit=None):
    ds = "dsfsi/za-mafoko-dsac"
    cols = {"eng": "eng", "afr": "afr", "zul": "zul", "xho": "xho", "ssw": "ssw", "nbl": "nbl", "tsn": "tsn", "nso": "nso", "sot": "sot", "ven": "ven", "tso": "tso"}
    for path, _ in files_of(ds, patterns=(".jsonl",)):
        rows = list(hubio.iter_file(ds, path))
        langs = [c for c in cols if rows and any(blank(r.get(c)) for r in rows[:50])]
        for r in rows:
            have = [c for c in langs if blank(r.get(c)) and len(blank(r.get(c))) <= 200]
            if len(have) < 2:
                yield Skip("fewer_than_two_languages")
                continue
            a, b = rng.sample(have, 2)
            if a not in A.VALID_LANGS or b not in A.VALID_LANGS:
                yield Skip(f"language_has_no_tag:{a if a not in A.VALID_LANGS else b}")
                continue
            if blank(r.get(a)).casefold() == blank(r.get(b)).casefold():
                yield Skip("identical")
                continue
            for rec in translation_recs(clean(r.get(a)), clean(r.get(b)), a, b, ds, rng, unsure_flag(a), unsure_flag(b), both=True, meta={"unit": "term", "list": path[:60]}):
                yield rec


def unsure_flag(code):
    return code in ("nbl",)


@register("govza_vukuzenzele_all", "dsfsi/govza-sa-cabinet-statements-sentence-aligned + vukuzenzele (every <a>-<b>/train* folder: interlanguage translation; the score column gates)")
def govza_all(rng, limit=None):
    for ds in ("dsfsi/govza-sa-cabinet-statements-sentence-aligned", "dsfsi/vukuzenzele-sentence-aligned"):
        for path, _ in files_of(ds, patterns=(".parquet",)):
            m = re.match(r"([a-z]{3})-([a-z]{3})/train", path)
            if not m:
                continue
            a, b = m.groups()
            yield from pairs(hubio.iter_file(ds, path), a, b, a, b, ds, rng, limit=limit, row_ok=lambda r: r.get("score") is None or float(r["score"]) >= 0.6)


@register("okapi_polywrite", "MaLA-LM/PolyWrite (prompt_en <-> prompt_translated per language file; chrF++ >= 55 against the back-translation) -> translation of writing prompts")
def polywrite(rng, limit=None):
    ds = "MaLA-LM/PolyWrite"
    for path, _ in files_of(ds, patterns=(".jsonl",)):
        code = lang(path.split("_")[0])
        if code is None or code == "eng":
            continue
        yield from pairs(hubio.iter_file(ds, path), "prompt_en", "prompt_translated", "eng", code, ds, rng, limit=limit,
                         row_ok=lambda r: float(r.get("chrf++") or 0) >= 55, meta={"kind": "writing_prompt"}, max_chars=2500)


# ----------------------------------------------------------------------- almanach (LLM translation work-flows)

ALM_LANGS = {"Xhosa": "xho", "Hausa": "hau", "Igbo": "ibo", "Kinyarwanda": "kin", "Somali": "som", "Swahili": "swh", "Zulu": "zul", "Shona": "sna", "Yoruba": "yor",
             "Afrikaans": "afr", "Amharic": "amh", "Oromo": "orm", "Wolof": "wol", "Chichewa": "nya", "Sesotho": "sot", "Setswana": "tsn", "Tswana": "tsn", "Akan": "twi",
             "Twi": "twi", "Ewe": "ewe", "Fon": "fon", "Tigrinya": "ti", "Lingala": "lin", "Luganda": "lug", "Kikuyu": "kik", "Bemba": "bem", "Luo": "luo", "English": "eng",
             "French": "fra"}


def _alm_rows(ds):
    for path, _ in files_of(ds, patterns=(".parquet",)):
        for r in hubio.iter_file(ds, path):
            r["__file"] = path
            yield r


def _alm_langs(r):
    return ALM_LANGS.get(str(r.get("source_language"))), ALM_LANGS.get(str(r.get("target_language")))


def _think_split(t: str):
    from sft_new.ext_common import split_think
    think, answer = split_think(t or "")
    return (think, answer) if think else (None, None)


@register("almanach_translation", "almanach/{Xhosa, topxgen-gemma-3-27b-it-BoA}: source -> target (languages we carry). Llama-4 sets are left out: the Llama 4 licence forbids using its outputs")
def almanach_plain(rng, limit=None):
    for ds in ("almanach/Xhosa", "almanach/topxgen-gemma-3-27b-it-BoA"):
        by: dict = {}
        for r in _alm_rows(ds):
            la, lb = _alm_langs(r)
            if la and lb and la != lb and lb in ALM_LANGS.values():
                by.setdefault((la, lb), []).append(r)
        for (la, lb), rows in by.items():
            yield from pairs(iter(rows), "source", "target", la, lb, ds, rng, limit=limit, both=False, meta={"llm_translated": True, "license_note": "LLM output (Llama 4 / Gemma terms)"})


def _alm_workflow(ds, rng, limit, kind):
    """CoT / self-refinement workflows: the prompt carries the source and (for refinement) the draft; the think field is the reasoning; the answer is the final translation."""
    n = 0
    for r in _alm_rows(ds):
        la, lb = _alm_langs(r)
        src, trans = EI.clean_keep_lines(blank(r.get("source"))), EI.clean_keep_lines(blank(r.get("better-translation") or r.get("translation")))
        if kind == "cot":                                                       # the answer sits in `target` (<think>...</think> translation), not in a translation column
            trans = "x"
        if not la or not lb or lb not in ALM_LANGS.values() or la == lb or not src or not trans or len(src) > 1500 or len(trans) > 1500:
            yield Skip("language_or_shape")
            continue
        ins = T.make("translate" if False else "refine_translation" if kind in ("refine", "tear", "sbys") else "assess_translation", rng, target=lb, source=la,
                     prefer=la if la in T.languages("refine_translation") else None)
        if kind == "cot":
            think, answer = _think_split(blank(r.get("target")))
            if not think or not answer or "<" in think:
                yield Skip("no_think_pair")
                continue
            from africomet_translation.instructions import make_instruction
            mi = make_instruction("eng" if rng.random() < 0.6 else rng.choice(["eng", "yor", "hau", "ibo"]), la, lb, rng)
            yield Rec("translation", ds, la, lb, answer.strip(), input=src, instruction=mi.text, position=mi.position, plan=["<translate>"], think=think[:6000],
                      meta={"cot": True, "instruction_language": mi.language})
            n += 1
        elif kind == "sbys":
            draft, refined = EI.clean_keep_lines(blank(r.get("drafting"))), EI.clean_keep_lines(blank(r.get("proofreading") or r.get("refinement")))
            if not draft or not refined:
                yield Skip("no_draft")
                continue
            prompt = f"{ins.text}\n\nSource ({r.get('source_language')}):\n{src}\n\nDraft ({r.get('target_language')}):\n{draft}"
            yield Rec("translation_refinement", ds, la, lb, refined, input="", prompt_override=prompt, plan=["<translate>", "<correct>"], input_langs=[la, lb][:2],
                      meta={"workflow": "SBYS"})
            n += 1
        elif kind == "tear":
            draft, better = EI.clean_keep_lines(blank(r.get("drafting"))), EI.clean_keep_lines(blank(r.get("better-translation")))
            est = EI.clean_keep_lines(blank(r.get("estimation")))
            if not draft or not better:
                yield Skip("no_draft")
                continue
            prompt = f"{ins.text}\n\nSource ({r.get('source_language')}):\n{src}\n\nDraft ({r.get('target_language')}):\n{draft}"
            yield Rec("translation_refinement", ds, la, lb, better, input="", prompt_override=prompt, plan=["<translate>", "<correct>"], input_langs=[la, lb][:2],
                      think=est[:3000] or None, meta={"workflow": "TEaR"})
            n += 1
        elif kind == "comptra":
            phrases, ptr = r.get("phrases") or [], r.get("phrases translations") or []
            if not phrases or len(phrases) != len(ptr):
                yield Skip("no_phrases")
                continue
            think = "\n".join(f"{i + 1}. {p} -> {t}" for i, (p, t) in enumerate(zip(phrases, ptr)))
            from africomet_translation.instructions import make_instruction
            mi = make_instruction("eng", la, lb, rng)
            yield Rec("translation", ds, la, lb, trans, input=src, instruction=mi.text, position=mi.position, plan=["<translate>"], think=think[:3000],
                      meta={"workflow": "CompTra"})
            n += 1
        if limit and n >= limit:
            return


@register("almanach_cot", "almanach/wmt19-gemma-3-27b-it-CoT: <think> reasoning + translation -> translation with <think> (Llama-4 CoT set left out)")
def almanach_cot(rng, limit=None):
    for ds in ("almanach/wmt19-gemma-3-27b-it-CoT",):
        yield from _alm_workflow(ds, rng, limit, "cot")


@register("almanach_refine", "almanach/{topxgen-gemma-3-4b-it-SBYS, wmt19-gemma-3-27b-it-TEaR, topxgen-gemma-3-27b-it-CompTra}: draft -> refined translation (and phrase decomposition)")
def almanach_refine(rng, limit=None):
    yield from _alm_workflow("almanach/topxgen-gemma-3-4b-it-SBYS", rng, limit, "sbys")
    yield from _alm_workflow("almanach/wmt19-gemma-3-27b-it-TEaR", rng, limit, "tear")
    yield from _alm_workflow("almanach/topxgen-gemma-3-27b-it-CompTra", rng, limit, "comptra")


@register("wmt_human_mt", "zouhar/wmt-human-all (src, mt, score, dataset): MT outputs scored by humans; keep pairs of our languages with a high human score -> translation")
def wmt_human(rng, limit=None):
    ds = "zouhar/wmt-human-all"
    by: dict = {}
    mx = 0.0
    for r in hubio.iter_repo(ds):
        m = re.search(r"([a-z]{2,3})[-_]([a-z]{2,3})", str(r.get("dataset") or ""))
        if not m:
            continue
        la, lb = lang(m.group(1)), lang(m.group(2))
        if la and lb and (la in L.SUPPORTED_SET or la == "fra") and (lb in L.SUPPORTED_SET or lb == "fra"):
            try:
                s = float(r.get("score"))
            except (TypeError, ValueError):
                continue
            by.setdefault((la, lb), []).append((s, r))
    for (la, lb), items in by.items():
        top = max(s for s, _ in items)
        thr = 85.0 if top > 5 else 0.85                                        # DA 0-100 or z/0-1 scales
        yield from pairs(iter([r for s, r in items if s >= thr]), "src", "mt", la, lb, ds, rng, limit=limit, both=False,
                         meta={"kind": "mt_output_human_scored", "score_threshold": thr})


# ======================================================================= classification / labelled text

def _lang_of(r, keys=("language", "lang", "dialect", "language_code")):
    for k in keys:
        v = r.get(k)
        if v:
            c = lang(v)
            if c:
                return c
    return None


@register("scam_naija", "Nihilitybot666/naijascam (text, label, scam_type, red_flags, language) -> scam / not-scam with an explanation built from scam_type and red_flags")
def scam_naija(rng, limit=None):
    ds = "Nihilitybot666/naijascam"
    for path, _ in files_of(ds, patterns=(".jsonl",)):
        if not any(s in path for s in ("train", "validation")):
            continue
        for chunk in batched(hubio.iter_file(ds, path), 128):
            rows = []
            for r in chunk:
                t = clean(r.get("text"))
                lab = str(r.get("label") if r.get("label") is not None else "").lower()
                if len(t) < 15:
                    yield Skip("empty")
                    continue
                rows.append((t, lab, r))
            if not rows:
                continue
            for (t, lab, r), (code, unsure) in zip(rows, resolve([x[0][:800] for x in rows], None)):
                if not code or code not in L.SUPPORTED_SET:
                    yield Skip("language_unsupported_or_undetermined")
                    continue
                flags = r.get("red_flags")
                flags = [flags] if isinstance(flags, str) else list(flags or [])
                flags = [clean(f) for f in flags if clean(f)]
                stype = clean(r.get("scam_type")).replace("_", " ")
                is_scam = lab in ("1", "true", "scam", "fraud", "phishing", "spam") or (stype and stype.lower() not in ("none", "legit", "legitimate", "not scam", "ham", ""))
                if is_scam:
                    why = f"Scam" + (f" ({stype})" if stype else "") + "."
                    if flags:
                        why += " Warning signs: " + "; ".join(flags[:6]) + "."
                else:
                    why = "Not a scam. The message does not show the usual warning signs of fraud."
                ins = T.make("scam", rng, target="eng", source=code, prefer=code if code in T.languages("scam") else None)
                yield Rec("scam_detection", ds, code, "eng", why, input=t, instruction=ins.text, position=ins.position, plan=["<classify>", "<|explain|>"], unsure_src=unsure,
                          meta={"instruction_language": ins.language, "scam_type": stype, "label": lab})


def _text_cat(rows_iter, text_key, label_fn, ds, rng, declared, task, instr_task, label_token="<topic>", limit=None, meta=None, plan=None):
    yield from label_stream(rows_iter, text_key, label_fn, ds, rng, task=task, instr_task=instr_task, plan=plan or ["<classify>"], label_token=label_token, declared=declared,
                            limit=limit, min_chars=6, meta=meta)


@register("kppwdfgu1_topics", "Kppwdfgu1 speech sets (kalenjin, somali, kikuyu, ss400, kin-s-5): transcription -> topic from `domain` / `category`; audio columns never read")
def kppwdfgu1_topics(rng, limit=None):
    specs = [("Kppwdfgu1/anv-data-ke-somali-full", "transcription", "domain", "language", "som"),
             ("Kppwdfgu1/anv_data_ke_kikuyu_scripted", "transcription", "domain", "language", "kik"),
             ("Kppwdfgu1/kalenjin-speech-full", "transcription", "domain", "language", "kln"),
             ("Kppwdfgu1/ss400", "transcription", "category", None, None),
             ("Kppwdfgu1/kin-s-5", "transcription", "category", None, "kin")]
    for ds, tk, lk, langk, default in specs:
        cols = [tk, lk] + ([langk] if langk else [])
        paths = [p for p, _ in files_of(ds, patterns=(".parquet",)) if "/train" in p]
        n = 0
        for p in paths:
            def label(r, lk=lk):
                v = clean(r.get(lk)).replace("_", " ")
                return v.lower() if v else None
            def decl(r, langk=langk, default=default):
                return (lang(r.get(langk)) if langk else None) or default
            for rec in label_stream(hubio.iter_file(ds, p, columns=cols), tk, label, ds, rng, task="topic", instr_task="topic", plan=["<classify>"], label_token="<topic>",
                                    declared=decl, min_chars=6, meta={"label_column": lk}):
                yield rec
                n += isinstance(rec, Rec)
            if limit and n >= limit:
                break


@register("vocare_intent", "Kppwdfgu1/vocare-hasa (transcript_raw, intent_label, dialect) -> intent classification, Hausa")
def vocare(rng, limit=None):
    ds = "Kppwdfgu1/vocare-hasa"
    n = 0
    for p, _ in files_of(ds, patterns=(".parquet",)):
        if not (p.startswith("asr/train") or "/train" in p):
            continue
        for rec in label_stream(hubio.iter_file(ds, p, columns=["transcript_raw", "intent_label", "dialect"]), "transcript_raw",
                                lambda r: clean(r.get("intent_label")).replace("_", " ").lower() or None, ds, rng, task="intent", instr_task="intent", plan=["<classify>"],
                                label_token="<intent>", declared="hau", min_chars=4, meta={"corpus": "vocare"}):
            yield rec
            n += isinstance(rec, Rec)
        if limit and n >= limit:
            break


@register("afrisenti_hf", "AfriSenti (HF repo is a loading script; the TSVs are read from the afrisenti-semeval GitHub, train split only) -> sentiment. Licence conflicting upstream: flagged")
def afrisenti(rng, limit=None):
    import csv
    import io
    import requests
    LANGS = {"amh": "amh", "hau": "hau", "ibo": "ibo", "yor": "yor", "twi": "twi", "tso": "tso", "tir": "ti", "orm": "orm", "pcm": "pcm", "kin": "kin", "swa": "swh"}
    base = "https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/data"
    for src, code in LANGS.items():
        try:
            r = requests.get(f"{base}/{src}/train.tsv", timeout=60)
            r.raise_for_status()
        except Exception as e:
            yield Skip(f"download_failed:{src}:{type(e).__name__}")
            continue
        rows = list(csv.DictReader(io.StringIO(r.text), delimiter="\t"))
        tk = next((k for k in (rows[0] if rows else {}) if k.lower() in ("tweet", "text")), "tweet")
        yield from label_stream(iter(rows), tk, lambda x: clean(x.get("label")).capitalize() or None, "shmuhammad/AfriSenti-twitter-sentiment", rng, task="sentiment",
                                instr_task="sentiment", plan=["<classify>"], label_token="<sentiment>", declared=code, limit=limit,
                                meta={"split": "train", "license_flag": "AfriSenti: CC-BY-4.0 on the card of some languages, CC-BY-NC-SA upstream for others"})


@register("swati_sentiment", "michsethowusu/swati-sentiments-corpus (Swati text, model-labelled binary sentiment) -> sentiment (labels come from a classifier on the English meaning)")
def swati(rng, limit=None):
    ds = "michsethowusu/swati-sentiments-corpus"
    yield from label_stream(hubio.iter_repo(ds), "Swati", lambda r: clean(r.get("sentiment")).capitalize() or None, ds, rng, task="sentiment", instr_task="sentiment",
                            plan=["<classify>"], label_token="<sentiment>", declared="ssw", limit=limit, meta={"labels": "model-generated"})


@register("setswana_sentiment", "dsfsi/setswana-sentiment (Text; kept where both annotators agree) -> sentiment")
def setswana_sentiment(rng, limit=None):
    ds = "dsfsi/setswana-sentiment"
    names = {"pos": "Positive", "positive": "Positive", "neg": "Negative", "negative": "Negative", "neu": "Neutral", "neutral": "Neutral"}
    def lab(r):
        a, b = str(r.get("ann1_label") or "").strip().lower(), str(r.get("ann2_label") or "").strip().lower()
        return names.get(a) if a and a == b else None
    for p in ("train.csv", "dev.csv"):
        yield from label_stream(hubio.iter_file(ds, p), "text", lab, ds, rng, task="sentiment", instr_task="sentiment", plan=["<classify>"], label_token="<sentiment>",
                                declared="tsn", limit=limit, meta={"agreement": "both annotators", "file": p})


@register("amharic_topic_instruct", "KoseiUemura/AfriInstruct-Data-amh train (instruction already carries the label set; output = label) -> classification")
def afriinstruct_amh(rng, limit=None):
    ds = "KoseiUemura/AfriInstruct-Data-amh"
    path = next(p for p, _ in files_of(ds, patterns=(".parquet",)) if "/train" in p)
    n = 0
    for chunk in batched(hubio.iter_file(ds, path), 256):
        items = [(blank(r.get("instruction")), blank(r.get("output")), r) for r in chunk]
        items = [x for x in items if x[0] and x[1] and len(x[0]) < 4000]
        if not items:
            continue
        for (ins, out, r), (code, unsure) in zip(items, resolve([i[:700] for i, _, _ in items], "amh")):
            if code not in ("amh", "eng", "ti"):
                yield Skip("language")
                continue
            task, plan = guess(ins)
            yield Rec("classification", ds, "amh", "amh" if len(out) > 25 else "eng", re.sub(r"^\s*Label:\s*", "", out), input="", prompt_override=ins, plan=["<classify>"],
                      label="<topic>", input_langs=["amh", "eng"], unsure_src=unsure, meta={"task_column": r.get("task")})
            n += 1
        if limit and n >= limit:
            return


@register("pidgin_lexicon", "prince4332/unfiltered-pidgin-english-lexicon (term, meaning) -> what a Pidgin term means / how to say it")
def pidgin_lexicon(rng, limit=None):
    ds = "prince4332/unfiltered-pidgin-english-lexicon"
    for r in hubio.iter_repo(ds):
        term, meaning = clean(r.get("term")), clean(r.get("meaning"))
        if not term or len(meaning) < 5 or len(term) > 80 or degenerate(meaning):
            yield Skip("empty_or_short")
            continue
        ins = T.make("define_term", rng, target="eng", source="pcm", prefer="pcm" if rng.random() < 0.4 else None, title=term)
        yield Rec("dictionary", ds, "pcm", "eng", meaning, input="", prompt_override=ins.text, plan=["<|explain|>"], input_langs=list(dict.fromkeys([ins.language, "pcm"]))[:3],
                  meta={"instruction_language": ins.language})


# ======================================================================= QA with context / RAG

@register("kenswquad", "Kencorpus/KenSwQuAD (Swahili context, question, answer) -> reading comprehension and RAG tool call")
def kenswquad(rng, limit=None):
    ds = "Kencorpus/KenSwQuAD"
    n = 0
    for chunk in batched(hubio.iter_repo(ds), 128):
        rows = [(EI.clean_keep_lines(blank(r.get("context"))), clean(r.get("question")), clean(r.get("answer"))) for r in chunk]
        rows = [x for x in rows if len(x[0]) > 60 and x[1] and x[2] and len(x[0]) < 6000]
        if not rows:
            continue
        for (ctx, q, a), (c, unsure) in zip(rows, resolve([q[:300] for _, q, _ in rows], "swh")):
            if rng.random() < 0.25:                                      # 75% as function-call RAG (owner 2026-10-11), the rest in-prompt context
                ins = T.make("qa_context", rng, target="swh", source="swh", prefer="swh" if rng.random() < 0.3 else None)
                body = f"{ctx}\n\n{q}" if ins.position == "prepend" else f"{q}\n\n{ctx}"
                prompt = f"{ins.text}\n\n{body}" if ins.position == "prepend" else f"{body}\n\n{ins.text}"
                yield Rec("reading_comprehension", ds, "swh", "swh", a, input="", prompt_override=prompt, plan=["<qa>"], label="<answer>", input_langs=list(dict.fromkeys(["swh", ins.language]))[:3],
                          unsure_src=unsure, meta={"instruction_language": ins.language})
            else:
                yield rag_convo(q, ctx, a, "swh", ds, rng, unsure=unsure)
            n += 1
        if limit and n >= limit:
            return


@register("squad_xh", "simpra/squad_xh (Xhosa SQuAD: context, question, answer, answer_en) -> reading comprehension; some rows ask for the answer in English")
def squad_xh(rng, limit=None):
    ds = "simpra/squad_xh"
    path = next(p for p, _ in files_of(ds, patterns=(".parquet",)) if "/train" in p)
    n = 0
    for r in hubio.iter_file(ds, path):
        ctx, q, a, ae = EI.clean_keep_lines(blank(r.get("context"))), clean(r.get("question")), clean(r.get("answer")), clean(r.get("answer_en"))
        if len(ctx) < 60 or not q or not a or len(ctx) > 6000:
            yield Skip("empty")
            continue
        ins = T.make("qa_context", rng, target="xho", source="xho", prefer=None)
        body = f"{ctx}\n\n{q}"
        eng_answer = ae and rng.random() < 0.3
        if not eng_answer and rng.random() < 0.7:
            yield rag_convo(q, ctx, a, "xho", ds, rng)
            continue
        if eng_answer:
            suf = T.make("answer_in", rng, target="eng", source="xho", prefer=None, position="append")
            prompt = f"{ins.text}\n\n{body}\n\n{suf.text}"
            yield Rec("reading_comprehension", ds, "xho", "eng", ae, input="", prompt_override=prompt, plan=["<qa>"], label="<answer>",
                      input_langs=list(dict.fromkeys(["xho", ins.language, suf.language]))[:3], meta={"answer_language": "eng"})
        else:
            prompt = f"{ins.text}\n\n{body}"
            yield Rec("reading_comprehension", ds, "xho", "xho", a, input="", prompt_override=prompt, plan=["<qa>"], label="<answer>",
                      input_langs=list(dict.fromkeys(["xho", ins.language]))[:3], meta={"instruction_language": ins.language})
        n += 1
        if limit and n >= limit:
            return


@register("webfaq_v2", "PaDaS-Lab/webfaq-v2 (question, answer per language; PaDaS-Lab only, `webfaq-test` is a test set and is excluded) -> question answering")
def webfaq(rng, limit=None):
    ds = "PaDaS-Lab/webfaq-v2"
    CAP = 60_000
    for path, _ in files_of(ds, patterns=(".parquet",)):
        code = lang(path.rsplit("/", 1)[-1].split(".")[0])
        if not code or (code not in L.SUPPORTED_SET and code not in ("fra", "amh", "ti", "ssw", "nso", "tso", "ven", "kik", "lug", "lin", "luo", "bem")):
            continue
        n = 0
        for chunk in batched(hubio.iter_file(ds, path, columns=["question", "answer"]), 256):
            rows = [(clean(r.get("question")), EI.clean_keep_lines(blank(r.get("answer")))) for r in chunk]
            rows = [x for x in rows if len(x[0]) > 8 and 20 < len(x[1]) < 3000 and not degenerate(x[1])]
            if not rows:
                continue
            rq = resolve([q[:400] for q, _ in rows], code)
            for (q, a), (c, unsure) in zip(rows, rq):
                if c != code:
                    yield Skip(f"lid_mismatch:{code}>{c}")
                    continue
                yield Rec("question_answering", ds, code, code, a, input="", prompt_override=q, plan=["<qa>"], unsure_src=unsure, unsure_tgt=unsure, meta={"faq": True})
                n += 1
            if n >= (limit or CAP):
                break


# ======================================================================= instruction / chat

@register("inkuba_instruct", "lelapa/Inkuba-instruct train files (instruction, inputs, targets, task; one language per split) -> instruction following by task")
def inkuba(rng, limit=None):
    ds = "lelapa/Inkuba-instruct"
    LN = {"hausa": "hau", "isizulu": "zul", "swahili": "swh", "xhosa": "xho", "yoruba": "yor", "english": "eng"}
    TASK = {"mmt": ("translation", ["<translate>"]), "ner": ("ner", ["<|data_extract|>"]), "pos": ("pos", ["<classify>"]), "sentiment": ("sentiment", ["<classify>"]),
            "topic": ("topic", ["<classify>"]), "qa": ("question_answering", ["<qa>"]), "summ": ("summarization", ["<summarize>"])}
    cap = limit or 80_000
    for path, _ in files_of(ds, patterns=(".parquet",)):
        m = re.match(r"data/([a-z]+)_train-", path)
        if not m or m.group(1) == "english":
            continue
        code = LN.get(m.group(1))
        n = 0
        for chunk in batched(hubio.iter_file(ds, path), 256):
            rows = []
            for r in chunk:
                ins, inp, tgt = clean(r.get("instruction")), EI.clean_keep_lines(blank(r.get("inputs"))), EI.clean_keep_lines(blank(r.get("targets")))
                if not ins or not tgt or len(inp) > 3000 or len(tgt) > 3000 or rng.random() > 0.15:
                    continue
                rows.append((ins, inp, tgt, str(r.get("task") or "").lower(), r))
            if not rows:
                continue
            ro = resolve([t[:500] for _, _, t, _, _ in rows], None)
            for (ins, inp, tgt, task, r), (co, uo) in zip(rows, ro):
                if not co:
                    yield Skip("language_undetermined")
                    continue
                tname, plan = TASK.get(task) or guess(ins)
                yield Rec(tname, ds, code, co, tgt, input="", prompt_override=EI.join_prompt(ins, inp), plan=plan, lang_hints=["eng", code], unsure_tgt=uo,
                          meta={"inkuba_task": task, "data_source": r.get("data_source")})
                n += 1
            if n >= cap:
                break


@register("pidgin_chat", "Ephraimmm/pidgin_clean (conversations: list of role/content) -> Pidgin chat")
def pidgin_chat(rng, limit=None):
    ds = "Ephraimmm/pidgin_clean"
    n = 0
    for r in hubio.iter_repo(ds):
        rec = EI.chat_convo(r.get("conversations"), ds, rng, declared=None, meta={"language_hint": "pcm"})
        yield rec
        n += isinstance(rec, Rec)
        if limit and n >= limit:
            return


@register("smoltalk_xhosa", "simpra/smoltalk (Xhosa everyday conversations, messages role/content) -> chat")
def smoltalk(rng, limit=None):
    ds = "simpra/smoltalk"
    for path, _ in files_of(ds, patterns=(".parquet",)):
        if "/train" not in path:
            continue
        for r in hubio.iter_file(ds, path):
            yield EI.chat_convo(r.get("messages"), ds, rng, declared="xho", meta={"topic": r.get("full_topic")})


FRONTIER_DROP = {"coding", "debugging", "tutorial", "pattern recognition", "data analysis", "math", "specific format writing", "information extraction", "text manipulation",
                 "spatial reasoning", "logical reasoning", "question generation", "text comparison", "text classification", "structured data"}


@register("frontier_chat", "Rafi757/Frontier (content = message list, teacher_response, category): general-knowledge categories only; no coding / debugging / extraction -> chat")
def frontier(rng, limit=None):
    ds = "Rafi757/Frontier"
    n, cap = 0, limit or 120_000
    path = next(p for p, _ in files_of(ds, patterns=(".parquet",)) if "/train" in p)
    for r in hubio.iter_file(ds, path):
        cat = str(r.get("category") or "").lower()
        if cat in FRONTIER_DROP:
            yield Skip("category:" + cat)
            continue
        msgs = list(r.get("content") or [])
        resp = blank(r.get("teacher_response"))
        if not msgs or not resp or not EI.general_knowledge(msgs + [{"content": resp}]):
            yield Skip("technical_or_empty")
            continue
        if rng.random() > 0.7:
            yield Skip("not_sampled")
            continue
        msgs = [m for m in msgs if m.get("role") in ("system", "user", "assistant")]
        msgs.append({"role": "assistant", "content": resp})
        rec = EI.chat_convo(msgs, ds, rng, declared="eng", meta={"category": cat})
        yield rec
        n += isinstance(rec, Rec)
        if n >= cap:
            return


_NOESIS_ROUTER = re.compile(r"should be classified as|\*\*SELF_SOLVE\*\*|\bROUTE_TO\b|\brouting decision\b|classified as \*\*", re.I)
_NOESIS_SPLIT = re.compile(r"(?:^|\n\n)(User|Assistant):[ \t]*")


@register("noesis_chat", "AMAImedia/NOESIS-1M (text = flattened `User:` / `Assistant:`; domains general, math, reasoning; complete samples only, no code, no router outputs) -> chat with <think>")
def noesis(rng, limit=None):
    ds = "AMAImedia/NOESIS-1M-reasoning-router-code-math-psych-opus47-deepseek4-qwen36-gemini31-r1-gpt54"
    path = next(p for p, _ in files_of(ds, patterns=(".jsonl",)))
    n, cap = 0, limit or 150_000
    for r in hubio.iter_file(ds, path):
        dom = str(r.get("domain") or "")
        if dom not in ("general", "math", "reasoning"):
            continue
        text = blank(r.get("text"))
        parts = _NOESIS_SPLIT.split(text)
        # split -> ['', 'User', 'q', 'Assistant', 'a', ...]
        msgs, i = [], 1
        while i + 1 < len(parts):
            msgs.append({"role": "user" if parts[i] == "User" else "assistant", "content": parts[i + 1].strip()})
            i += 2
        if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
            yield Skip("shape")
            continue
        last = msgs[-1]["content"]
        if _NOESIS_ROUTER.search(last) or last.count("<think>") != last.count("</think>") or not re.search(r"[.!?)\]$]\s*$", last):
            yield Skip("router_incomplete_or_truncated")
            continue
        if dom != "math" and not EI.general_knowledge(msgs):
            yield Skip("technical")
            continue
        if rng.random() > 0.6:
            yield Skip("not_sampled")
            continue
        rec = EI.chat_convo(msgs, ds, rng, declared=None, meta={"noesis_domain": dom, "noesis_src": r.get("src")})
        yield rec
        n += isinstance(rec, Rec)
        if n >= cap:
            return


_CHATML = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.S)


def chatml_messages(text: str) -> list:
    return [{"role": r, "content": c.strip()} for r, c in _CHATML.findall(text or "")]


@register("chatml_dolly_webglm", "Felladrin/ChatML-databricks-dolly-15k and ChatML-WebGLM-QA (<|im_start|>user ... <|im_end|> text) -> QA with context / chat")
def chatml(rng, limit=None):
    for ds, cap in (("Felladrin/ChatML-databricks-dolly-15k", 20_000), ("Felladrin/ChatML-WebGLM-QA", limit or 40_000)):
        path = next(p for p, _ in files_of(ds, patterns=(".parquet",)))
        n = 0
        for r in hubio.iter_file(ds, path):
            msgs = chatml_messages(r.get("text"))
            if len(msgs) < 2:
                yield Skip("parse")
                continue
            if "WebGLM" in ds:
                u = msgs[0]["content"]
                m = re.match(r"^Context:\s*(.*?)\n\nQuestion:\s*(.*)$", u, re.S) or re.match(r"^Question:\s*(.*?)\n\nContext:\s*(.*)$", u, re.S)
                if not m:
                    yield Skip("no_context_question")
                    continue
                refs, q = (m.group(1), m.group(2)) if u.startswith("Context:") else (m.group(2), m.group(1))
                ins = T.make("qa_context", rng, target="eng", source="eng", prefer=None)
                prompt = f"{ins.text}\n\nContext:\n{refs.strip()[:6000]}\n\nQuestion:\n{q.strip()}" if rng.random() < 0.5 else f"{ins.text}\n\nQuestion:\n{q.strip()}\n\nContext:\n{refs.strip()[:6000]}"
                msgs = [{"role": "user", "content": prompt}, msgs[1]]
            rec = EI.chat_convo(msgs, ds, rng, declared="eng", meta={"chatml": True})
            yield rec
            n += isinstance(rec, Rec)
            if n >= cap:
                break


@register("hypa_text", "hypaai/Hypa-Text-10k (messages with thinking / tool_calls / tool_responses + tools_json: translation, dictionary, language id, JSON, correction) -> our scaffold")
def hypa_text(rng, limit=None):
    ds = "hypaai/Hypa-Text-10k"
    for r in hubio.iter_repo(ds):
        msgs = r.get("messages") or []
        tools = None
        try:
            tl = json.loads(r.get("tools_json") or "[]")
            tools = tl if tl else None
        except ValueError:
            tools = None
        out, ok = [], True
        system = None
        think_mode = False
        for m in msgs:
            role, content = m.get("role"), blank(m.get("content"))
            if role == "system":
                system = re.sub(r"\s*THINKING\s*=\s*\w+\s*$", "", content.replace("<|think>", "").strip(), flags=re.I).strip()
                think_mode = "<|think>" in (m.get("content") or "")
                continue
            if role == "user":
                out.append({"role": "user", "content": content})
            elif role == "assistant":
                calls = m.get("tool_calls") or []
                think = blank(m.get("thinking"))
                if calls:
                    for k, c in enumerate(calls):
                        fn = (c.get("function") or c)
                        try:
                            args = json.loads(fn.get("arguments")) if isinstance(fn.get("arguments"), str) else (fn.get("arguments") or {})
                        except ValueError:
                            ok = False
                            break
                        out.append(A.assistant_message({"input_lang": ["eng"], "task_plan": ["<|plan|>"], "tool_call": {"name": fn.get("name"), "arguments": args}}))
                        resp = (m.get("tool_responses") or [])
                        if k < len(resp):
                            out.append({"role": "tool", "content": str(resp[k].get("response") if isinstance(resp[k], dict) else resp[k])})
                    if not ok:
                        break
                    if content:
                        out.append({"_final": True, "content": content, "think": think})
                else:
                    out.append({"_final": True, "content": content, "think": think})
        if not ok or not out or not any(m.get("_final") for m in out):
            yield Skip("unconvertible")
            continue
        # the answers are scaffolded with the detected languages
        users = [m["content"] for m in out if m.get("role") == "user"]
        finals = [m for m in out if m.get("_final")]
        rl = resolve([u[:500] for u in users] + [f["content"][:500] for f in finals], None)
        ru, rf = rl[: len(users)], rl[len(users):]
        convo, ui, fi, ok2 = ([{"role": "system", "content": system}] if system else []), 0, 0, True
        for m in out:
            if m.get("role") == "user":
                convo.append(m)
            elif m.get("_final"):
                cu, uu = ru[min(ui - 0, len(ru) - 1)] if ru else (None, False)
                ca, ua = rf[fi]
                fi += 1
                cu = cu or "eng"
                ca = ca or cu
                try:
                    c = A.assistant_content(input_lang=[cu], target_lang=ca, task_plan=guess(users[-1] if users else "")[1], response=m["content"], think=(m["think"] or None),
                                            unsure_input=uu, unsure_target=ua)
                except A.AssemblyError as e:
                    ok2 = False
                    break
                convo.append({"role": "assistant", "content": c})
            else:
                convo.append(m)
            if m.get("role") == "user":
                ui += 1
        if not ok2 or not convo or convo[-1]["role"] != "assistant":
            yield Skip("assembly")
            continue
        yield Rec("chat" if not tools else "function_calling", ds, "eng", "eng", finals[-1]["content"], input=users[0] if users else "", convo=convo, tools=tools,
                  meta={"hypa": True, "thinking_mode": think_mode})


@register("gimmy256_qa", "gimmy256/adaption-* and uganda-edu-gemma (SFT columns only; reasoning traces and checklists are RL material) -> QA / explanation, language detected")
def gimmy(rng, limit=None):
    specs = [("gimmy256/adaption-african-culture-qa", "prompt", "completion", None, "cultural"), ("gimmy256/adaption-african-smallholder-agri-qa", "instruction", "output", "input", "agriculture"),
             ("gimmy256/uganda-edu-gemma", "prompt", "response", None, "education (Uganda, primary/secondary)"),
             ("gimmy256/adaption-africatravel-qa-bench", "question", "answer", None, "travel"),
             ("gimmy256/adaption-african-culinary-traditions", "enhanced_prompt", "enhanced_completion", None, "culinary")]
    for ds, ik, ok, ink, domain in specs:
        path = next(p for p, _ in files_of(ds, patterns=(".jsonl",)))
        def tag(r, ds=ds):
            c = lang(r.get("target_language_code")) or lang(r.get("target_language")) if "culinary" in ds else None
            return c
        yield from EI.instruct_rows(hubio.iter_file(ds, path), ik, ink, ok, ds, rng, declared=None, limit=limit,
                                 meta={"domain": domain, "source_dataset_family": "gimmy256"}, row_ok=None)


@register("ugandan_languages", "gimmy256/ugandan_languages (English + Acoli, Luganda, Lumasaaba, Runyakore, Kiswahili) -> translation between the languages that have a tag")
def ugandan(rng, limit=None):
    ds = "gimmy256/ugandan_languages"
    path = next(p for p, _ in files_of(ds, patterns=(".parquet",)))
    rows = list(hubio.iter_file(ds, path))
    cols = {"English Sentence": "eng", "Acoli Translation": "ach", "Luganda Translation": "lug", "Lumasaaba Translation": "myx", "Runyakore Translation": "nyn", "Kiswahili Translation": "swh"}
    for ka, la in cols.items():
        for kb, lb in cols.items():
            if ka < kb and (la == "eng" or lb == "eng" or True):
                if la not in A.VALID_LANGS or lb not in A.VALID_LANGS:
                    yield Skip(f"language_has_no_tag:{la if la not in A.VALID_LANGS else lb}")
                    continue
                yield from pairs(iter(rows), ka, kb, la, lb, ds, rng, limit=limit, min_chars=4)


@register("aya_templated_only", "janavivekariya/aya_collection templated_* and aya_dataset sub-collections of our languages (the translated_* sub-collections are NLLB output and are left out)")
def aya_templated(rng, limit=None):
    ds = "janavivekariya/aya_collection"
    AYA_TASK = {"question-answering": ("qa", ["<qa>"]), "summarization": ("summarization", ["<summarize>"]), "translation": ("translation", ["<translate>"]),
                "paraphrase-identification": ("classification", ["<classify>"]), "dialogue": ("chat", ["<|chat|>"]), "generation": ("generation", ["<|generate|>"])}
    paths = [p for p, _ in files_of(ds, patterns=(".parquet",)) if (p.startswith("templated_") or p.startswith("aya_dataset")) and "/train" in p]
    cap_lang: dict = {}
    for p in paths:
        for chunk in batched(hubio.iter_file(ds, p), 256):
            sel = []
            for r in chunk:
                code = lang(r.get("language"))
                if not code or code == "eng" or (code not in L.SUPPORTED_SET and code not in ("amh", "ti", "fra")):
                    continue
                if cap_lang.get(code, 0) >= (limit or 40_000):
                    continue
                sel.append((code, r))
            if not sel:
                continue
            for code, r in sel:
                ins, out = EI.clean_keep_lines(blank(r.get("inputs"))), EI.clean_keep_lines(blank(r.get("targets")))
                if not ins or not out or len(ins) > 6000 or len(out) > 4000:
                    yield Skip("empty_or_long")
                    continue
                task, plan = AYA_TASK.get(str(r.get("task_type") or "").lower()) or guess(ins)
                yield Rec(task, ds, code, code, out, input="", prompt_override=ins, plan=plan, lang_hints=[code, "eng"], meta={"aya_sub": r.get("sub_dataset_name"), "templated": True})
                cap_lang[code] = cap_lang.get(code, 0) + 1


@register("alpaca_xhosa_taco_new", "saillab/alpaca_xhosa_taco (TaCo pivot) -> Xhosa instruction following with the English pivot in <think>")
def alpaca_xhosa(rng, limit=None):
    ds = "saillab/alpaca_xhosa_taco"
    n = 0
    path = next(p for p, _ in files_of(ds, patterns=(".parquet",)) if "/train" in p)
    for chunk in batched(hubio.iter_file(ds, path), 128):
        items = []
        for r in chunk:
            ins, inp = blank(r.get("instruction")), blank(r.get("input"))
            m = EI.TACO.search(r.get("output") or r.get("text") or "")
            if not ins or not m:
                yield Skip("no_taco_structure")
                continue
            items.append((ins, inp, EI.clean_keep_lines(m.group("ie")), EI.clean_keep_lines(m.group("re")), EI.clean_keep_lines(m.group("ans"))))
        if not items:
            continue
        for (ins, inp, ie, re_, ans), (ci, ui), (ca, ua) in zip(items, resolve([i[0][:500] for i in items], "xho"), resolve([i[4][:500] for i in items], "xho")):
            if ci != "xho" or ca != "xho":
                yield Skip(f"lid_mismatch:xho>{ci if ci != 'xho' else ca}")
                continue
            task, plan = guess(ie, inp)
            yield Rec(task, ds, "xho", "xho", ans, input="", prompt_override=EI.join_prompt(ins, inp), plan=plan, think=f"Instruction in English: {ie}\nResponse in English: {re_}",
                      input_langs=["xho"], unsure_src=ui, unsure_tgt=ua, meta={"pivot": "english"})
            n += 1
        if limit and n >= limit:
            return


@register("massivesumm_short", "MaLA-LM/MassiveSumm_short (<lang>.all.jsonl: title, text, summary) -> summarization + headline for the languages we carry")
def massivesumm(rng, limit=None):
    ds = "MaLA-LM/MassiveSumm_short"
    cap = limit or 25_000
    for path, _ in files_of(ds, patterns=(".jsonl",)):
        code = lang(path.split(".")[0])
        if not code or (code not in L.SUPPORTED_SET and code not in ("amh", "ti", "fra", "kik", "lug", "lin", "luo", "bem", "ssw", "nso", "tso", "ven")):
            continue
        n = 0
        for chunk in batched(hubio.iter_file(ds, path), 128):
            rows = [(EI.clean_keep_lines(blank(r.get("text"))), clean(r.get("summary")), clean(r.get("title"))) for r in chunk]
            rows = [x for x in rows if len(x[0]) > 300 and len(x[1]) > 20 and len(x[0]) < 12000]
            if not rows:
                continue
            for (text, summ, title), (c, unsure) in zip(rows, resolve([t[:900] for t, _, _ in rows], code)):
                if c != code:
                    yield Skip(f"lid_mismatch:{code}>{c}")
                    continue
                ins = T.make("summarize", rng, target=code, source=code, prefer=code if code in T.languages("summarize") else None)
                yield Rec("summarization", ds, code, code, summ, input=text[:10000], instruction=ins.text, position=ins.position, plan=["<summarize>"], unsure_src=unsure,
                          unsure_tgt=unsure, meta={"instruction_language": ins.language})
                if title and 8 < len(title) < 200:
                    ih = T.make("headline", rng, target=code, source=code, prefer=code if code in T.languages("headline") else None)
                    yield Rec("headline", ds, code, code, title, input=text[:6000], instruction=ih.text, position=ih.position, plan=["<summarize>"], label="<headline>",
                              unsure_src=unsure, unsure_tgt=unsure, meta={"instruction_language": ih.language})
                n += 1
            if n >= cap:
                break


@register("emulated_knowledge_pool", "Emulated-Inc/multilingual-knowledge-training-pool: only rows whose source split is train/validation (the pool also re-uses benchmark TEST splits, which are left out) and general-education subjects -> multiple choice")
def emulated(rng, limit=None):
    ds = "Emulated-Inc/multilingual-knowledge-training-pool"
    EDU = {"history", "geography", "biology", "economics", "literature", "language", "sociology", "health", "chemistry", "physics", "mathematics", "agriculture", "religious studies",
           "environmental studies and forestry", "philosophy", "visual arts", "general knowledge", "science", "social studies", "civics", "environmental science", "music"}
    n, cap = 0, limit or 120_000
    stats = {}
    for r in hubio.iter_file(ds, "pool.jsonl"):
        split = str(r.get("source_split") or "").lower()
        subj = str(r.get("subject") or "").lower()
        if split in ("test", "dev", "devtest") or str(r.get("provenance_class") or "") not in ("human", ""):
            yield Skip(f"source_split_{split or 'unknown'}")
            continue
        if subj not in EDU:
            yield Skip("subject_out_of_scope")
            continue
        opts = r.get("options") or r.get("choices") or []
        ans = str(r.get("answer") or "").strip().upper()
        q, ctx = clean(r.get("question")), clean(r.get("context"))
        if len(opts) < 2 or ans not in "ABCDEFGHIJ"[:len(opts)] or not q:
            yield Skip("bad_row")
            continue
        code = lang(r.get("language"))
        if code is None or (code not in L.SUPPORTED_SET and code not in ("amh", "fra", "ti")):
            yield Skip("language_not_carried")
            continue
        body = (ctx + "\n" if ctx else "") + q + "\n" + "\n".join(f"{chr(65 + i)}. {clean(str(o))}" for i, o in enumerate(opts))
        ins = T.make("mcq", rng, target=code, source=code, prefer=code if code in T.languages("mcq") else None)
        yield Rec("multiple_choice", ds, code, code, f"{ans}. {clean(str(opts['ABCDEFGHIJ'.index(ans)]))}", input=body, instruction=ins.text, position=ins.position, plan=["<qa>"],
                  label="<answer>", meta={"subject": subj, "source": r.get("source"), "instruction_language": ins.language})
        n += 1
        if n >= cap:
            return
