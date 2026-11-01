"""Large multi-language collections: the Aya collection language split, xP3 (mt, all) and the xP3x African subset, the tagged translation corpora, and
a few small pair / chat sets. Everything is capped (per file / per sub-dataset / per language) so that no single collection dominates the SFT mix; the caps
are constants below and are reported in the build statistics.

Evaluation hygiene: xP3 and xP3x bundle the TEST / DEV splits of many benchmarks (FLORES dev/devtest, XL-Sum test, ...). Only files whose name carries the
`train` split are read; `merged_*.jsonl` (all splits concatenated) is never read.
"""

from __future__ import annotations
import os

import re

from sft_new import hubio
from sft_new import langdetect as L
from sft_new import templates as T
from sft_new.compose import A, Rec, Skip
from sft_new.converters import register, tagged_stream
from sft_new.ext_common import batched, code_of, files_of, iter_files, pairs, resolve, translation_recs
from sft_new.taskguess import guess
from sft_new.textutil import blank, clean, degenerate

AYA_CAP_LANG = 200_000
AYA_CAP_SUB = 25_000
XP3_CAP_FILE = 2_500
XP3_CAP_LANG = 60_000

AYA_TASK = {"question-answering": ("qa", ["<qa>"]), "summarization": ("summarization", ["<summarize>"]), "translation": ("translation", ["<translate>"]),
            "text-simplification": ("editing", ["<|edit|>"]), "paraphrase-identification": ("classification", ["<classify>"]),
            "dialogue": ("chat", ["<|chat|>"]), "generation": ("generation", ["<|generate|>"]), "long-form generation": ("generation", ["<|generate|>"])}
AYA_LIST_MARKERS = ("Na farko.", "Na biyu.", "Ta yaya za mu iya yin hakan?", "Abin da ya faru.", "A'a ba haka ba")


@register("aya_language_split", "CohereLabs/aya_collection_language_split, train files of the languages we carry (task_type mapped; caps per sub-dataset and language)")
def aya_language_split(rng, limit=None):
    ds = "CohereLabs/aya_collection_language_split"
    folders: dict = {}
    for p, _ in files_of(ds):
        top = p.split("/")[0]
        code = code_of(top.replace("_", " "))
        if code and code != "eng" and "train" in p.rsplit("/", 1)[-1]:
            folders.setdefault((top, code), []).append(p)
    if not folders:
        raise RuntimeError("no Aya language folders matched")
    for (top, code), paths in sorted(folders.items()):
        sub_n: dict = {}
        total = [0]

        def ok_row(r, sub_n=sub_n, total=total):
            if total[0] >= (limit or AYA_CAP_LANG):
                return False
            k = str(r.get("sub_dataset_name") or r.get("dataset_name"))
            if sub_n.get(k, 0) >= AYA_CAP_SUB:
                return False
            if code == "hau" and any(m in (r.get("targets") or "") for m in AYA_LIST_MARKERS):
                return False                                         # machine-translated numbered-list artefact (quality_check.artefacts)
            sub_n[k] = sub_n.get(k, 0) + 1
            return True

        def task_of(r):
            return AYA_TASK.get(str(r.get("task_type") or "").lower())

        from sft_new.ext_instruct import instruct_rows
        for rec in instruct_rows(iter_files(ds, paths), "inputs", None, "targets", ds, rng, declared=code, row_ok=ok_row, task_of=task_of,
                                 meta={"aya_language_folder": top}):
            if isinstance(rec, Rec):
                total[0] += 1
            yield rec


def _xp3_task(path: str):
    n = path.lower()
    for key, task, plan in (("xlsum", "summarization", ["<summarize>"]), ("wiki_lingua", "summarization", ["<summarize>"]),
                            ("flores", "translation", ["<translate>"]), ("tatoeba", "translation", ["<translate>"]), ("opus", "translation", ["<translate>"]),
                            ("ted", "translation", ["<translate>"]), ("wmt", "translation", ["<translate>"]), ("afriqa", "qa", ["<qa>"]),
                            ("tydiqa", "qa", ["<qa>"]), ("xquad", "qa", ["<qa>"]), ("mlqa", "qa", ["<qa>"]), ("xnli", "nli", ["<nli>"]),
                            ("masakhanews", "classification", ["<classify>"]), ("afrisenti", "sentiment", ["<classify>"])):
        if key in n:
            return task, plan
    return None


def _xp3_run(repo: str, ds_label: str, rng, limit):
    paths_by_lang: dict = {}
    for p, _ in files_of(repo, patterns=(".jsonl",)):
        top = p.split("/")[0]
        code = code_of(top)
        name = p.rsplit("/", 1)[-1]
        if not code or code == "eng" or not name.startswith("xp3_") or "_train_" not in name:
            continue
        paths_by_lang.setdefault((top, code), []).append(p)
    for (top, code), paths in sorted(paths_by_lang.items()):
        n_lang = 0
        for p in sorted(paths):
            if n_lang >= (limit or XP3_CAP_LANG):
                break
            task_plan = _xp3_task(p)
            n = 0
            for chunk in batched(hubio.iter_file(repo, p), 128):
                items = []
                for r in chunk:
                    i, t = clean(r.get("inputs")), clean(r.get("targets"))
                    if len(i) < 8 or not t or len(i) > 8000 or len(t) > 6000 or degenerate(t):
                        yield Skip("empty_long_or_degenerate")
                    else:
                        items.append((i, t))
                if not items:
                    continue
                for (i, t), (ct, ut) in zip(items, resolve([t[:700] for _, t in items], None)):
                    tgt = ct or (code if re.search(r"[A-Za-z]", t) else "eng")
                    if tgt not in L.SUPPORTED_SET and tgt not in ("fra", "amh", "ti", "dag"):
                        yield Skip("target_language_not_carried")
                        continue
                    task, plan = task_plan or guess(i)
                    yield Rec(task, ds_label, code, tgt, t, input="", prompt_override=i, plan=plan, lang_hints=[code, "eng"], unsure_tgt=ut,
                              meta={"xp3_file": p.rsplit("/", 1)[-1][:90], "folder": top})
                    n += 1
                    n_lang += 1
                if n >= XP3_CAP_FILE:
                    break


@register("xp3mt_african", "bigscience/xP3mt: train files of our African languages (ig ny rw sn st sw tn tw wo xh yo zu fon ak ...); capped per file and language")
def xp3mt(rng, limit=None):
    yield from _xp3_run("bigscience/xP3mt", "bigscience/xP3mt", rng, limit)


@register("xp3all_african", "bigscience/xP3all: same as xp3mt_african (largely overlapping; the content hash removes the exact duplicates)")
def xp3all(rng, limit=None):
    yield from _xp3_run("bigscience/xP3all", "bigscience/xP3all", rng, limit)


@register("xp3x_african", "taresco/xP3x_african_subset: split == train rows whose `language` is one of ours; capped per language")
def xp3x(rng, limit=None):
    ds = "taresco/xP3x_african_subset"
    by: dict = {}
    for p, _ in files_of(ds, patterns=(".jsonl",)):
        m = re.match(r"data/([a-z]{3}_[A-Za-z]+)/", p)
        code = code_of(m.group(1).split("_")[0]) if m else None
        if code is None and m and m.group(1) == "tir_Ethi":
            code = "ti"
        if not code or code == "eng" or "_train_" not in p:
            continue
        by.setdefault(code, []).append(p)
    for code, paths in sorted(by.items()):
        n = 0
        for p in sorted(paths):
            if n >= (limit or XP3_CAP_LANG):
                break
            kept_file = 0
            for chunk in batched(hubio.iter_file(ds, p), 128):
                items = [(clean(r.get("inputs")), clean(r.get("targets")), r) for r in chunk if str(r.get("split")) == "train"]
                items = [x for x in items if len(x[0]) >= 8 and x[1] and len(x[0]) <= 8000 and len(x[1]) <= 6000 and not degenerate(x[1])]
                if not items:
                    continue
                for (i, t, r), (ct, ut) in zip(items, resolve([x[1][:700] for x in items], None)):
                    tgt = ct
                    if not tgt:
                        yield Skip("target_language_undetermined")
                        continue
                    task, plan = _xp3_task(p) or guess(i)
                    yield Rec(task, ds, code, tgt, t, input="", prompt_override=i, plan=plan, lang_hints=[code, "eng"], unsure_tgt=ut,
                              meta={"xp3x_dataset": r.get("dataset"), "config": r.get("config")})
                    n += 1
                    kept_file += 1
                if kept_file >= XP3_CAP_FILE:
                    break


# ------------------------------------------------------------------ tagged "<translate> A <lang> B" corpora

TAGGED_CAP = 1_500_000


def _tagged(repo: str, rng, limit, quality=None):
    yield from tagged_stream(repo, rng, limit or TAGGED_CAP, min_quality=quality)


@register("tds_long", (os.environ['HF_ORG'] + "/tds_long (tagged translations; africomet_score >= 0.55; capped)"))
def tds_long(rng, limit=None):
    yield from _tagged((os.environ['HF_ORG'] + "/tds_long"), rng, limit, {"africomet_score": 0.55})


@register("processed_data_tagged", (os.environ['HF_ORG'] + "/processed_data (tagged translations; africomet_score >= 0.55; capped)"))
def processed_data(rng, limit=None):
    yield from _tagged((os.environ['HF_ORG'] + "/processed_data"), rng, limit, {"africomet_score": 0.55})


@register("clean_data_tagged", (os.environ['HF_ORG'] + "/clean_data (tagged translations, no QE column; languages re-detected; capped)"))
def clean_data(rng, limit=None):
    yield from _tagged((os.environ['HF_ORG'] + "/clean_data"), rng, limit)


@register("clean_data2_tagged", (os.environ['HF_ORG'] + "/clean_data2 (tagged translations; capped)"))
def clean_data2(rng, limit=None):
    yield from _tagged((os.environ['HF_ORG'] + "/clean_data2"), rng, limit)


@register("translation_tagged", (os.environ['HF_ORG'] + "/translation (tagged translations; capped)"))
def translation_tagged(rng, limit=None):
    yield from _tagged((os.environ['HF_ORG'] + "/translation"), rng, limit)


@register("clean_translate_tagged", (os.environ['HF_ORG_LEGACY'] + "/Clean_translate (tagged translations; capped)"))
def clean_translate(rng, limit=None):
    yield from _tagged((os.environ['HF_ORG_LEGACY'] + "/Clean_translate"), rng, limit)


# ------------------------------------------------------------------ language-name tagged pairs: "<chichewa> text <tumbuka> text"

NAME_PAIR = re.compile(r"^\s*<([a-z_ ]+)>\s*(.*?)\s*<([a-z_ ]+)>\s*(.*?)\s*$", re.S)


@register("african_languages_translation", (os.environ['HF_ORG'] + "/african_languages_translation (`<lang> text <lang> text` rows; only file pairs whose two languages we carry)"))
def african_languages_translation(rng, limit=None):
    ds = (os.environ['HF_ORG'] + "/african_languages_translation")
    by: dict = {}
    for p, _ in files_of(ds):
        m = re.match(r"([a-z]+)-([a-z]+)_sentence-pairs", p)
        if not m:
            continue
        a, b = code_of(m.group(1)), code_of(m.group(2))
        if a and b and a != b:
            by.setdefault((a, b), []).append(p)
    for (a, b), paths in sorted(by.items()):
        rows = []
        for r in iter_files(ds, paths):
            m = NAME_PAIR.match(r.get("text") or "")
            if not m:
                yield Skip("no_name_tags")
                continue
            x, y = code_of(m.group(1).replace("_", " ")), code_of(m.group(3).replace("_", " "))
            if {x, y} != {a, b}:
                yield Skip("tags_disagree_with_file")
                continue
            rows.append({x: m.group(2), y: m.group(4)})
        yield from pairs(iter(rows), a, b, a, b, ds, rng, limit=limit, meta={"pair": f"{a}-{b}"})


# ------------------------------------------------------------------ small pair sets and English chat

@register("bible_pairs", (os.environ['HF_ORG'] + "/biblenlp-corpus-mmteb (eng, yor), -swahili (eng, swh), bitext_nollysenti_miners (en-yo reviews) -> translation"))
def bible_pairs(rng, limit=None):
    ds = (os.environ['HF_ORG'] + "/biblenlp-corpus-mmteb")
    yield from pairs(hubio.iter_repo(ds, match="train"), "eng", "yor", "eng", "yor", ds, rng, limit=limit)
    ds = (os.environ['HF_ORG'] + "/biblenlp-corpus-mmteb-swahili")
    yield from pairs(hubio.iter_repo(ds, match="train"), "eng", "swh", "eng", "swh", ds, rng, limit=limit)
    ds = (os.environ['HF_ORG'] + "/bitext_nollysenti_miners")
    yield from pairs(hubio.iter_repo(ds), "sentence1", "sentence2", "eng", "yor", ds, rng, limit=limit, row_ok=lambda r: "yo" in str(r.get("lang")))


@register("ultrachat_filtered", (os.environ['HF_ORG_LEGACY'] + "/ultrachat-filtered (English multi-turn chat, `data` = alternating user/assistant) -> chat; capped"))
def ultrachat(rng, limit=None):
    from sft_new.ext_instruct import chat_convo
    ds = (os.environ['HF_ORG_LEGACY'] + "/ultrachat-filtered")
    n, cap = 0, limit or 150_000
    for r in hubio.iter_repo(ds):
        d = r.get("data")
        if not isinstance(d, list) or len(d) < 2:
            yield Skip("shape")
            continue
        if rng.random() > 0.35:
            yield Skip("not_sampled")
            continue
        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(d)]
        rec = chat_convo(msgs, ds, rng, declared="eng", meta={"source": "ultrachat"})
        yield rec
        n += isinstance(rec, Rec)
        if n >= cap:
            return
