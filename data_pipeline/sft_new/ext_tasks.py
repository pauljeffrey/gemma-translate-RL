"""Second-round task datasets: classification, emotion, language identification, NER, diacritics, NLI, math.

Every declared language goes through ext_common.resolve / langaudit (GlotLID, confident-other => dropped or relabelled to a language we carry).
Label words are English in every instruction language (the existing sentiment / topic / intent convention).
"""

from __future__ import annotations
import os

import ast
import itertools
import json
import re

from sft_new import hubio
from sft_new import langdetect as L
from sft_new import templates as T
from sft_new.compose import Rec, Skip
from sft_new.converters import SIB200, register
from sft_new.ext_common import (batched, classify_rec, code_of, files_of, iter_any, iter_files, label_stream, read_tsv, resolve, unwrap_markup)
from sft_new.langaudit import audit
from sft_new.textutil import blank, clean, degenerate

SIB_PLAN = dict(task="topic", instr_task="topic", plan=["<classify>"], label_token="<topic>")


# ------------------------------------------------------------------ topic classification

@register("sib200_ibom", "howard-nlp/sib200-ibom (Efik, Ibibio, English; every split: owner says all of it matters) -> <classify><topic>")
def sib200_ibom(rng, limit=None):
    ds = "howard-nlp/sib200-ibom"
    for p, _ in files_of(ds):
        m = re.match(r"data/([a-z]{3})_Latn/(train|dev|test)\.tsv$", p)
        code = code_of(m.group(1)) if m else None
        if not m:
            continue
        if not code:
            yield Skip(f"language_not_supported:{m.group(1)}")                 # anw (Anaang) and orx (Oro) have no tag in our set
            continue
        yield from label_stream(read_tsv(ds, p), "text", lambda r: clean(r.get("category")), ds, rng, declared=code, limit=limit,
                                meta={"split": m.group(2)}, **SIB_PLAN)


@register("sib200_lexcgen", "BatsResearch/sib200-LexC-Gen (synthetic, lexicon-translated; Twi `ak` and Ewe `ee`, the 100k files) -> <classify><topic>; class NAME, not index")
def sib200_lexcgen(rng, limit=None):
    ds = "BatsResearch/sib200-LexC-Gen"
    for p, _ in files_of(ds):
        m = re.match(r"(ak|ee)_100k/train-", p)
        if not m:
            continue
        code = {"ak": "twi", "ee": "ewe"}[m.group(1)]
        yield from label_stream(hubio.iter_file(ds, p), "text", lambda r: SIB200.get(str(r.get("label"))), ds, rng, declared=code, limit=limit,
                                meta={"synthetic": "LexC-Gen", "note": "English text with words substituted from a bilingual lexicon"}, **SIB_PLAN)


SA_LABELS = ["afr", "eng", "nbl", "nso", "sot", "ssw", "tsn", "tso", "ven", "xho", "zul"]       # alphabetical, checked: label 9 = Xhosa text
SA_NAMES = {"afr": "Afrikaans", "eng": "English", "sot": "Sesotho", "tsn": "Setswana", "xho": "isiXhosa", "zul": "isiZulu"}


@register("south_african_lid", "mteb/SouthAfricanLangClassification train -> language identification (<identify><lang_ID>); only languages we carry")
def south_african_lid(rng, limit=None):
    ds = "mteb/SouthAfricanLangClassification"
    def lab(r):
        try:
            c = SA_LABELS[int(r["label"])]
        except (ValueError, TypeError, IndexError):
            return None
        return c if c in SA_NAMES else None
    rows = hubio.iter_repo(ds, match="train")
    for chunk in batched(rows, 256):
        items = [(clean(r.get("text")).strip('" '), lab(r)) for r in chunk]
        keep = [(t, c) for t, c in items if t and c and len(t) >= 12]
        for _ in range(len(items) - len(keep)):
            yield Skip("language_not_carried_or_empty")
        if not keep:
            continue
        verdict = {}
        for code in {c for _, c in keep}:
            texts = [t for t, c in keep if c == code]
            for t, v in zip(texts, audit(texts, code)):
                verdict[(t, code)] = v
        for t, c in keep:
            v = verdict[(t, c)]
            if not v.ok:
                yield Skip(f"lid_mismatch:{c}>{(v.detected or '?')[:3]}")
                continue
            ins = T.make("language_id", rng, target="eng", source=c, prefer=c if c in T.languages("language_id") else None)
            yield Rec("language_id", ds, c, "eng", SA_NAMES[c], input=t, instruction=ins.text, position=ins.position, plan=["<identify>"], label="<lang_ID>",
                      unsure_src=v.unsure, meta={"instruction_language": ins.language})


# ------------------------------------------------------------------ emotion

EMO_SLOT = {"labels": "anger, disgust, fear, joy, sadness, surprise, neutral"}


@register("tswana_emotions", "michsethowusu/tswana-emotions-corpus -> <classify> emotion; Setswana text GlotLID-audited (NLLB-derived set)")
def tswana_emotions(rng, limit=None):
    ds = "michsethowusu/tswana-emotions-corpus"
    yield from label_stream(hubio.iter_repo(ds), "Tswana", lambda r: clean(r.get("emotion")).lower() or None, ds, rng, declared="tsn", limit=limit,
                            task="emotion", instr_task="emotion", plan=["<classify>"], label_token="<answer>", slots=EMO_SLOT)


EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
ETHIO = {"amh": "amh", "orm": "orm", "som": "som", "tir": "ti"}


@register("ethioemo", "Tadesse/EthioEmo-intensities train (amh orm som tir): the emotion(s) at the highest intensity, or neutral")
def ethioemo(rng, limit=None):
    ds = "Tadesse/EthioEmo-intensities"
    for name, code in ETHIO.items():
        def lab(r):
            try:
                vals = {e: int(float(r.get(e) or 0)) for e in EMOTIONS}
            except ValueError:
                return None
            top = max(vals.values())
            return "neutral" if top == 0 else ", ".join(e for e in EMOTIONS if vals[e] == top)
        yield from label_stream(hubio.iter_file(ds, f"train/{name}.csv"), "text", lab, ds, rng, declared=code, limit=limit, task="emotion", instr_task="emotion",
                                plan=["<classify>"], label_token="<answer>", slots=EMO_SLOT)


# ------------------------------------------------------------------ headline (Amharic news)

@register("news_topic_amh", "Tadesse/news-topic (Amharic news body -> its headline) -> <summarize><headline>")
def news_topic_amh(rng, limit=None):
    ds = "Tadesse/news-topic"
    n = 0
    for chunk in batched(read_tsv(ds, "train.tsv"), 128):
        rows = [(clean(r.get("news")), clean(r.get("topic"))) for r in chunk]
        keep = [(b, h) for b, h in rows if len(b) >= 200 and 8 <= len(h) <= 300]
        for _ in range(len(rows) - len(keep)):
            yield Skip("short_or_empty")
        if not keep:
            continue
        res = resolve([b[:1200] for b, _ in keep], "amh")
        for (b, h), (code, unsure) in zip(keep, res):
            if code != "amh":
                yield Skip("language_not_amharic")
                continue
            ins = T.make("headline", rng, target="amh", source="amh", prefer=None)
            yield Rec("headline", ds, "amh", "amh", h, input=b[:6000], instruction=ins.text, position=ins.position, plan=["<summarize>"], label="<headline>",
                      unsure_src=unsure, unsure_tgt=unsure, meta={"instruction_language": ins.language})
            n += 1
        if limit and n >= limit:
            return


# ------------------------------------------------------------------ diacritics and noise

@register("afriadr_diacritics", "masakhane/AfriADR train+validation (yor ibo fon wol; text -> text with diacritics) -> <diacritize>")
def afriadr(rng, limit=None):
    ds = "masakhane/AfriADR"
    for split in ("train", "validation"):
        for name, code in (("yor", "yor"), ("ibo", "ibo"), ("fon", "fon"), ("wol", "wol")):
            kept = 0
            for chunk in batched(read_tsv(ds, f"diacritics/{split}/{name}.tsv"), 256):
                rows = [(clean(r.get("text")), clean(r.get("target"))) for r in chunk]
                keep = []
                for a, b in rows:
                    if not a or not b or a == b or len(b) < 8 or len(b) > 2500 or degenerate(b):
                        yield Skip("empty_identical_or_degenerate")
                    else:
                        keep.append((a, b))
                if not keep:
                    continue
                for (a, b), v in zip(keep, audit([b for _, b in keep], code)):
                    if not v.ok:
                        yield Skip(f"lid_mismatch:{code}>{(v.detected or '?')[:3]}")
                        continue
                    ins = T.make("diacritize", rng, target=code, source=code, prefer=code if code in T.languages("diacritize") else None)
                    yield Rec("diacritization", ds, code, code, b, input=a, instruction=ins.text, position=ins.position, plan=["<diacritize>"],
                              unsure_src=v.unsure, unsure_tgt=v.unsure, meta={"instruction_language": ins.language, "split": split})
                    kept += 1
                if limit and kept >= limit:
                    break


def _detected_pairs(rows, in_key, out_key, task_pool, task, plan, dataset, rng, limit, meta=None):
    n = 0
    for chunk in batched(rows, 256):
        keep = []
        for r in chunk:
            a, b = clean(r.get(in_key)), clean(r.get(out_key))
            if not a or not b or a == b or len(b) < 10 or len(b) > 2500 or degenerate(b):
                yield Skip("empty_identical_or_degenerate")
            else:
                keep.append((a, b))
        if not keep:
            continue
        for (a, b), (code, unsure) in zip(keep, resolve([b for _, b in keep], None)):
            if not code or code not in L.SUPPORTED_SET:
                yield Skip("language_unsupported_or_undetermined")
                continue
            ins = T.make(task_pool, rng, target=code, source=code, prefer=code if code in T.languages(task_pool) else None)
            yield Rec(task, dataset, code, code, b, input=a, instruction=ins.text, position=ins.position, plan=list(plan), unsure_src=unsure, unsure_tgt=unsure,
                      meta={**(meta or {}), "instruction_language": ins.language})
            n += 1
        if limit and n >= limit:
            return


@register("aletheia_diacritics", (os.environ['HF_ORG'] + "/diacritics train (no_diacritics -> text; the `end_tag` column is not trusted, language is detected) -> <diacritize>"))
def aletheia_diacritics(rng, limit=None):
    ds = (os.environ['HF_ORG'] + "/diacritics")
    yield from _detected_pairs(hubio.iter_repo(ds, match="train"), "no_diacritics", "text", "diacritize", "diacritization", ["<diacritize>"], ds, rng, limit)


@register("aletheia_noisy", (os.environ['HF_ORG'] + "/noisy_dataset train (noisy_text -> text; language detected) -> <clean><correct>"))
def aletheia_noisy(rng, limit=None):
    ds = (os.environ['HF_ORG'] + "/noisy_dataset")
    yield from _detected_pairs(hubio.iter_repo(ds, match="train"), "noisy_text", "text", "denoise", "denoising", ["<clean>", "<correct>"], ds, rng, limit)


# ------------------------------------------------------------------ NER

_NER_NAMES: dict = {}


def ner_names() -> dict:
    """Label id -> name of bltlab/open-ner-standardized (one standardized scheme for every config), read from its dataset card."""
    if not _NER_NAMES:
        import os
        import yaml
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("bltlab/open-ner-standardized", "README.md", repo_type="dataset", token=hubio.token(),
                            cache_dir=os.path.join(os.environ.get("HF_HOME", "/tmp"), "tmp_readme"))
        text = open(p, encoding="utf-8").read()
        front = text.split("---", 2)[1]
        info = yaml.safe_load(front)["dataset_info"][0]
        for f in info["features"]:
            if f["name"] == "ner_tags":
                names = f["list"]["class_label"]["names"]
                _NER_NAMES.update({int(k): v for k, v in names.items()})
    return _NER_NAMES


def ner_rec(words: list, tags: list, lang: str, unsure: bool, dataset: str, rng, meta=None, instr_pool="ner") -> Rec | None:
    if not words or len(words) != len(tags) or len(words) > 400:
        return None
    sent = " ".join(words)
    resp = " ".join(f"{w} <tag> {t}" for w, t in zip(words, tags))
    ins = T.make(instr_pool, rng, target=lang, source=lang, prefer=lang if lang in T.languages(instr_pool) else None)
    return Rec("ner", dataset, lang, lang, resp, input=sent, instruction=ins.text, position=ins.position, plan=["<NER>"], sequence_labels=True,
               unsure_src=unsure, unsure_tgt=unsure, meta={**(meta or {}), "instruction_language": ins.language})


@register("open_ner", "bltlab/open-ner-standardized, train split of the African subsets (MasakhaNER, MasakhaNER_2) -> word <tag> BIO-label sequences")
def open_ner(rng, limit=None):
    ds = "bltlab/open-ner-standardized"
    names = ner_names()
    for p, _ in hubio.list_files(ds):
        m = re.match(r"(MasakhaNER|MasakhaNER_2)/([a-z]{3})/train-", p)
        if not m:
            continue
        code = code_of(m.group(2))
        if not code:
            yield Skip(f"language_not_supported:{m.group(2)}")
            continue
        n = 0
        for chunk in batched(hubio.iter_file(ds, p), 256):
            sents = [(r.get("tokens") or [], [names.get(int(t), "O") for t in (r.get("ner_tags") or [])]) for r in chunk]
            sents = [(w, t) for w, t in sents if w and len(w) == len(t)]
            if not sents:
                continue
            res = resolve([" ".join(w)[:1000] for w, _ in sents], code)
            for (w, t), (c2, unsure) in zip(sents, res):
                if c2 != code:
                    yield Skip(f"lid_mismatch:{code}>{c2}")
                    continue
                rec = ner_rec(w, t, code, unsure, ds, rng, meta={"scheme": m.group(1)})
                if rec is None:
                    yield Skip("bad_row")
                else:
                    yield rec
                    n += 1
            if limit and n >= limit:
                break


def _span_tags(text: str, spans: list) -> tuple[list, list]:
    """Whitespace words of `text` and a BIO tag per word from byte-offset spans ({'start_byte', 'limit_byte', 'label'})."""
    words, ranges, pos = [], [], 0
    b = text.encode("utf-8")
    for w in text.split():
        wb = w.encode("utf-8")
        i = b.find(wb, pos)
        if i < 0:
            return [], []
        words.append(w)
        ranges.append((i, i + len(wb)))
        pos = i + len(wb)
    tags = ["O"] * len(words)
    for s in spans:
        try:
            d = s if isinstance(s, dict) else ast.literal_eval(s)
            a, z, lab = int(d["start_byte"]), int(d["limit_byte"]), str(d["label"])
        except Exception:
            return [], []
        first = True
        for k, (ws, we) in enumerate(ranges):
            if we > a and ws < z:
                tags[k] = ("B-" if first else "I-") + lab
                first = False
    return words, tags


@register("masakhaner_x", "anrilombard/masakhaner-x-parquet train (tn xh zu; byte-offset spans -> BIO per word) -> word <tag> label sequences")
def masakhaner_x(rng, limit=None):
    ds = "anrilombard/masakhaner-x-parquet"
    for loc, code in (("tn", "tsn"), ("xh", "xho"), ("zu", "zul")):
        n = 0
        for chunk in batched(hubio.iter_file(ds, f"data/{loc}/train.parquet"), 256):
            items = []
            for r in chunk:
                words, tags = _span_tags(clean(r.get("text")), r.get("spans") or [])
                if words:
                    items.append((words, tags))
                else:
                    yield Skip("span_alignment_failed")
            if not items:
                continue
            res = resolve([" ".join(w)[:1000] for w, _ in items], code)
            for (w, t), (c2, unsure) in zip(items, res):
                if c2 != code:
                    yield Skip(f"lid_mismatch:{code}>{c2}")
                    continue
                rec = ner_rec(w, t, code, unsure, ds, rng, meta={"scheme": "masakhaner-x"})
                if rec:
                    yield rec
                    n += 1
            if limit and n >= limit:
                break


# ------------------------------------------------------------------ NLI (cross-lingual, from xsim++)

XSIM = {"hau_Latn": "hau", "ibo_Latn": "ibo", "yor_Latn": "yor", "swh_Latn": "swh", "zul_Latn": "zul", "xho_Latn": "xho", "sna_Latn": "sna", "kin_Latn": "kin",
        "som_Latn": "som", "afr_Latn": "afr", "nya_Latn": "nya", "wol_Latn": "wol", "tsn_Latn": "tsn", "sot_Latn": "sot", "twi_Latn": "twi", "ewe_Latn": "ewe",
        "fon_Latn": "fon", "gaz_Latn": "orm", "amh_Ethi": "amh", "tir_Ethi": "ti"}


@register("xsim_nli", "jaygala24/xsimplusplus (query = sentence in language X, positive = its English meaning, negative = a minimally perturbed English sentence) -> <nli>")
def xsim_nli(rng, limit=None):
    """Cross-lingual NLI built from a translation-similarity benchmark: (X premise, English hypothesis) is `entailment` for the positive and
    `contradiction` for the hard negative (changed number / entity / negation). dev + devtest are all this repo has; they are used on the owner's
    instruction, which is recorded in the row metadata."""
    ds = "jaygala24/xsimplusplus"
    for p, _ in files_of(ds):
        m = re.match(r"data/eng_Latn-([a-z]{3}_[A-Za-z]+)/(dev|devtest)\.parquet$", p)
        code = XSIM.get(m.group(1)) if m else None
        if not code:
            continue
        n = 0
        for chunk in batched(hubio.iter_file(ds, p), 256):
            qs = [clean(r.get("query")) for r in chunk]
            res = resolve([q[:600] for q in qs], code)
            for r, q, (c2, unsure) in zip(chunk, qs, res):
                if c2 != code:
                    yield Skip(f"lid_mismatch:{code}>{c2}")
                    continue
                pos = r.get("positive") or []
                neg = r.get("negative") or []
                pos = [pos] if isinstance(pos, str) else list(pos)
                neg = [neg] if isinstance(neg, str) else list(neg)
                variants = []
                if pos and clean(pos[0]):
                    variants.append((clean(pos[0]), "entailment"))
                if neg and clean(neg[0]):
                    variants.append((clean(neg[0]), "contradiction"))
                for hyp, lab in variants:
                    ins = T.make("nli", rng, target="eng", source=code, prefer=code if code in T.languages("nli") else None)
                    body = f"Premise: {q}\nHypothesis: {hyp}" if ins.language == "eng" else f"{q}\n{hyp}"
                    yield Rec("nli", ds, code, "eng", lab, input=body, instruction=ins.text, position=ins.position, plan=["<nli>"], label="<answer>",
                              input_langs=list(dict.fromkeys([ins.language, code, "eng"]))[:3], unsure_src=unsure,
                              meta={"instruction_language": ins.language, "split": m.group(2), "built_from": "translation-similarity benchmark"})
                    n += 1
            if limit and n >= limit:
                break


# ------------------------------------------------------------------ maths in African languages

GSM_LANG = {"afr": "afr", "amh": "amh", "gaz": "orm", "hau": "hau", "ibo": "ibo", "kin": "kin", "nya": "nya", "pcm": "pcm", "sna": "sna", "som": "som",
            "swh": "swh", "tsn": "tsn", "wol": "wol", "xho": "xho", "yor": "yor", "zul": "zul"}


@register("gsm_symbolic_african", "danish-foundation-models/multilingual-gsm-symbolic, African languages only (question -> worked answer); used on the owner's instruction although its splits are named test_*")
def gsm_symbolic(rng, limit=None):
    from sft_new.converters import _gsm_answer
    ds = "danish-foundation-models/multilingual-gsm-symbolic"
    for p, _ in files_of(ds):
        m = re.match(r"data/([a-z]{3})/(test_original|test_synthetic)", p)
        code = GSM_LANG.get(m.group(1)) if m else None
        if not code:
            continue
        for chunk in batched(hubio.iter_file(ds, p), 128):
            qs = [(clean(r.get("question")), _gsm_answer(r.get("answer") or "")) for r in chunk]
            keep = [(q, a) for q, a in qs if len(q) > 15 and len(a) > 10]
            for _ in range(len(qs) - len(keep)):
                yield Skip("empty_or_broken_answer")
            if not keep:
                continue
            res = resolve([q for q, _ in keep], code)
            for (q, a), (c2, unsure) in zip(keep, res):
                if c2 != code:
                    yield Skip(f"lid_mismatch:{code}>{c2}")
                    continue
                ins = T.make("math", rng, target=code, source=code, prefer=code if code in T.languages("math") else None)
                yield Rec("math", ds, code, code, a, input=q, instruction=ins.text, position=ins.position, plan=["<|is_technical|>", "<|math|>"], unsure_src=unsure, unsure_tgt=unsure,
                          meta={"instruction_language": ins.language, "split": m.group(2), "note": "test_* splits used on owner instruction"})


BIGMATH_LANG = {"HAUSA": "hau", "IGBO": "ibo", "SWAHILI": "swh", "YORUBA": "yor", "NIGERIAN PIDGIN": "pcm", "AFRIKAANS": "afr", "ZULU": "zul", "SOMALI": "som"}


@register("big_math_african", "taresco/big_math_translated_african_languages (user/assistant math in 8 of our languages; Arabic has no tag) -> math")
def big_math_african(rng, limit=None):
    ds = "taresco/big_math_translated_african_languages"
    for chunk in batched(hubio.iter_repo(ds), 128):
        keep = []
        for r in chunk:
            code = BIGMATH_LANG.get(str(r.get("language") or "").upper())
            msgs = r.get("messages") or []
            u = unwrap_markup(next((clean(m.get("content")) for m in msgs if m.get("role") == "user"), ""))
            a = unwrap_markup(next((clean(m.get("content")) for m in msgs if m.get("role") == "assistant"), ""))
            if not code:
                yield Skip("language_not_carried")
            elif len(u) < 15 or len(a) < 20 or degenerate(a):
                yield Skip("empty_or_degenerate")
            else:
                keep.append((code, u, a))
        if not keep:
            continue
        us = resolve([u[:800] for _, u, _ in keep], None)
        as_ = resolve([a[:800] for _, _, a in keep], None)
        for (code, u, a), (cu, uu), (ca, ua) in zip(keep, us, as_):
            if cu != code and not (code == "pcm" and cu == "eng"):
                yield Skip(f"lid_mismatch:{code}>{cu}")
                continue
            yield Rec("math", ds, code, code, a, input="", prompt_override=u, plan=["<|is_technical|>", "<|math|>"],
                      unsure_src=uu, unsure_tgt=ua or ca not in (None, code, "eng"), meta={"declared_language": code})
