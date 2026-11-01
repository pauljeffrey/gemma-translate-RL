"""Registers the multilingual instruction-template banks (owner 2026-10-11: "for each task a pool of multilingual instruction-following prompt templates,
as many instruction languages as you can cover, bigger pools per language").

Data lives in `pools_data_*.py`; each exports

    BANKS = {instruction_language: {task: {"pre": [...], "post": [...], "any": [...]}}}   # task "translate" feeds africomet_translation.TEMPLATES
    NAMES = {instruction_language: {language code: [name as written in that language, ...]}}
    TIER  = {instruction_language: "high" | "medium" | "low"}

Rules the validator enforces (tests/test_pools_multi.py):
  * only the placeholders {target} {source} {labels} {title} {seed} {n} (and the Nguni prefix forms {target.nge} / .yi / .lwe / .ku / .kwi in zul / xho);
  * no empty / duplicate templates, no stray whitespace, every template renders with a sample name;
  * a template that asks for a label set names it the way the datasets do (English label words stay English in every instruction language);
  * `pre` templates must not end by pointing back at text ("above") and `post` templates must not point forward ("below" / "following").

Instruction languages and their confidence tier are recorded in `template_confidence` of every record (CONFIDENCE), so any tier can be dropped later.
"""
from __future__ import annotations

import importlib
import re
import string

from africomet_translation import instructions as AI
from sft_new import ext_pools  # noqa: F401  (the older pools must be registered first: REPLACE edits them, and a later _reg() would overwrite ours)
from sft_new import templates as T

DATA_MODULES = ("pools_data_core", "pools_data_a", "pools_data_b", "pools_data_c", "pools_data_d", "pools_data_restore")
ALLOWED = {"target", "source", "labels", "title", "seed", "n"}
# words that point at text ABOVE / BELOW the instruction, per instruction language (a `pre` template must not say "above", a `post` template not "below")
ABOVE = {"eng": r"above|preceding", "swh": r"hapo juu", "fra": r"ci-dessus|précède", "afr": r"hierbo|bostaande", "zul": r"ngenhla", "xho": r"ngentla",
         "kin": r"hejuru", "sna": r"pamusoro", "som": r"kor ku (xusan|qoran)", "orm": r"olii", "tsn": r"fa godimo", "sot": r"ka holimo", "nya": r"pamwambapa", "amh": r"ከላይ",
         "pcm": r"wey dey up", "yor": r"lókè", "hau": r"da ke sama|a sama", "ibo": r"n'elu"}
BELOW = {"eng": r"below|following", "swh": r"hapa chini", "fra": r"ci-dessous", "afr": r"hieronder", "zul": r"ezansi", "xho": r"ezantsi", "kin": r"hasi",
         "sna": r"pazasi", "som": r"hoose", "orm": r"gadii", "tsn": r"fa tlase", "sot": r"ka tlase", "nya": r"m'munsi", "amh": r"ከታች", "yor": r"nísàlẹ̀",
         "hau": r"da ke (ƙasa|kasa)", "ibo": r"n'okpuru"}
NGUNI_ATTRS = {"nge", "yi", "lwe", "ku", "kwi"}
_loaded = False


def _extend(dst: list, src: list) -> None:
    for t in src:
        if t not in dst:
            dst.append(t)


def register_all() -> dict:
    """Idempotent. Returns {instruction_language: {task: n_templates}} of everything registered."""
    global _loaded
    summary: dict = {}
    for mod_name in DATA_MODULES:
        try:
            mod = importlib.import_module(f"sft_new.{mod_name}")
        except ModuleNotFoundError as e:
            if mod_name in str(e):
                continue
            raise
        for lang, tasks in mod.BANKS.items():
            for task, bank in tasks.items():
                pre, post, any_ = list(bank.get("pre", [])), list(bank.get("post", [])), list(bank.get("any", []))
                if task == "translate":
                    cur = AI.TEMPLATES.setdefault(lang, {"pre": [], "post": [], "any": []})
                    for k, v in (("pre", pre), ("post", post), ("any", any_)):
                        _extend(cur.setdefault(k, []), v)
                    n = sum(len(cur[k]) for k in ("pre", "post", "any"))
                else:
                    cur = T._POOLS.setdefault(task, {}).setdefault(lang, {"pre": [], "post": [], "any": []})
                    for k, v in (("pre", pre), ("post", post), ("any", any_)):
                        _extend(cur[k], v)
                    n = sum(len(cur[k]) for k in ("pre", "post", "any"))
                summary.setdefault(lang, {})[task] = n
            if lang not in AI.LANGS:
                AI.LANGS = tuple(AI.LANGS) + (lang,)
        for lang, tasks in getattr(mod, "REPLACE", {}).items():
            for task, mapping in tasks.items():
                bank = AI.TEMPLATES.get(lang) if task == "translate" else T._POOLS.get(task, {}).get(lang)
                if bank is None:
                    continue
                for kind in ("pre", "post", "any"):
                    bank[kind] = [mapping.get(t, t) for t in bank[kind]]
        for lang, tier in getattr(mod, "TIER", {}).items():
            AI.CONFIDENCE.setdefault(lang, tier)
        for lang, names in getattr(mod, "NAMES", {}).items():
            cur = AI.LOCAL_NAMES.setdefault(lang, {})
            for code, lst in names.items():
                cur.setdefault(code, [])
                _extend(cur[code], lst)
                AI.DISPLAY_NAMES.setdefault(code, [lst[0]])
    _loaded = True
    return summary


def validate() -> list[str]:
    """Return a list of problems (empty = clean)."""
    import random
    problems: list = []
    rng = random.Random(0)
    fmt = string.Formatter()
    seen: dict = {}
    for mod_name in DATA_MODULES:
        try:
            mod = importlib.import_module(f"sft_new.{mod_name}")
        except ModuleNotFoundError as e:
            if mod_name in str(e):
                continue
            raise
        for lang, tasks in mod.BANKS.items():
            for task, bank in tasks.items():
                for kind in ("pre", "post", "any"):
                    lst = bank.get(kind, [])
                    if len(set(lst)) != len(lst):
                        problems.append(f"{lang}/{task}/{kind}: duplicate templates inside the bank")
                    for t in lst:
                        where = f"{lang}/{task}/{kind}: {t[:70]!r}"
                        if not t.strip() or t != t.strip() or "  " in t or "\n" in t:
                            problems.append(f"{where} whitespace")
                            continue
                        try:
                            parsed = [f for _, f, _, _ in fmt.parse(t) if f]
                        except ValueError as e:
                            problems.append(f"{where} bad format string: {e}")
                            continue
                        for field in parsed:
                            base = field.split(".")[0]
                            if base not in ALLOWED:
                                problems.append(f"{where} unknown placeholder {{{field}}}")
                            if "." in field and (lang not in ("zul", "xho") or field.split(".", 1)[1] not in NGUNI_ATTRS):
                                problems.append(f"{where} attribute placeholder {{{field}}} only valid in zul / xho")
                        if task in ("emotion",) and "{labels}" not in t:
                            problems.append(f"{where} emotion template without {{labels}}")
                        if task in ("define_term", "wiki_ask") and "{title}" not in t:
                            problems.append(f"{where} {task} template without {{title}}")
                        if kind == "pre" and lang in ABOVE and re.search(ABOVE[lang], t, re.I):
                            problems.append(f"{where} pre-template points at text above")
                        if kind == "post" and lang in BELOW and re.search(BELOW[lang], t, re.I):
                            problems.append(f"{where} post-template points at text below")
                        try:
                            slots = {"target": AI._name("eng", lang, rng), "source": AI._name("hau", lang, rng), "labels": "a, b", "title": "T", "seed": "S", "n": "3"}
                            t.format(**slots)
                        except Exception as e:                                # noqa: BLE001
                            problems.append(f"{where} does not render: {type(e).__name__}: {e}")
                        seen.setdefault(t, set()).add((lang, task))
    for t, keys in seen.items():
        if len({k[1] for k in keys}) > 1 and len(t) > 25:
            problems.append(f"same template reused across tasks {sorted(keys)}: {t[:50]!r}")
    return problems


register_all()
