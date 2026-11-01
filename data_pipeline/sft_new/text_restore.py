"""Text restoration tasks built from real pretraining text (owner 2026-10-15): diacritization, cleaning, and both together.

  diacritize        input = text with its diacritics removed (60% of the rows: all of them; 40%: a random share), output = the original
                    plan <diacritize>;     only for languages whose text really carries diacritics / special letters
  clean             input = text with characters deleted / substituted / exchanged / added (noise budget <= 50% of the characters, mostly
                    3-35%), output = the original                                    plan <clean>;        every language
  diacritize+clean  input = degraded diacritics AND character noise, output = the original (the diacritic languages only)
                    plan <clean><diacritize>

The instruction is drawn from the task's multilingual pool (sft_new.templates; English share 35-55% by `choose_language`); the combined task joins a
clean and a diacritize instruction of the SAME instruction language. Output rows have the `artificial-*` schema of sft_new/artificial.py.
Pure-python core (no pyarrow / network) so it can be unit tested; `main` does the I/O.
"""
from __future__ import annotations
import os

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# letters that are "diacritics" for restoration purposes: a plain-keyboard writer types the base letter instead
SPECIAL = {"ɛ": "e", "ɔ": "o", "ɖ": "d", "ƒ": "f", "ɣ": "g", "ŋ": "n", "ʋ": "v", "ɓ": "b", "ɗ": "d", "ƙ": "k", "ɠ": "g", "ɲ": "n",
           "ə": "e", "ɩ": "i", "ʊ": "u", "ɨ": "i", "ʉ": "u", "ɱ": "m", "ʃ": "s", "ʒ": "z", "ɦ": "h",
           "Ɛ": "E", "Ɔ": "O", "Ɖ": "D", "Ƒ": "F", "Ɣ": "G", "Ŋ": "N", "Ʋ": "V", "Ɓ": "B", "Ɗ": "D", "Ƙ": "K", "Ɠ": "G", "Ɲ": "N",
           "Ə": "E", "Ɩ": "I", "Ʊ": "U", "’": "'", "ʼ": "'"}
_ADJ = {  # QWERTY neighbours for plausible substitutions
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
    "a": "qsz", "s": "awdx", "d": "sefc", "f": "drgv", "g": "fthb", "h": "gyjn", "j": "hukm", "k": "jil", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk"}


def is_mark(c: str) -> bool:
    return unicodedata.category(c) == "Mn"


def diacritic_units(text: str) -> int:
    """How many characters a full strip would remove or fold (combining marks + special letters)."""
    nfd = unicodedata.normalize("NFD", text)
    return sum(1 for c in nfd if is_mark(c) or c in SPECIAL)


def diacritic_density(text: str) -> float:
    return diacritic_units(text) / max(len(unicodedata.normalize("NFD", text)), 1)


def degrade_diacritics(text: str, rng: random.Random) -> Tuple[str, str, float]:
    """-> (degraded text, mode, share removed). mode: full (60%) | marks | letters | both. Partial modes remove each mark / special letter
    independently with a strength drawn from U(0.15, 0.85)."""
    nfd = unicodedata.normalize("NFD", text)
    full = rng.random() < 0.60
    if full:
        strength, mode = 1.0, "full"
    else:
        strength = rng.uniform(0.15, 0.85)
        has_marks = any(is_mark(c) for c in nfd)
        has_letters = any(c in SPECIAL for c in nfd)
        modes, weights = ["marks", "letters", "both"], [0.5, 0.2, 0.3]
        if not has_letters:            # a mode that cannot change anything is not drawn (Yoruba has marks only, Hausa letters only, ...)
            modes, weights = ["marks"], [1.0]
        elif not has_marks:
            modes, weights = ["letters"], [1.0]
        mode = rng.choices(modes, weights=weights)[0]
    out: List[str] = []
    for c in nfd:
        if is_mark(c):
            if (full or mode in ("marks", "both")) and (full or rng.random() < strength):
                continue
            out.append(c)
        elif c in SPECIAL:
            if (full or mode in ("letters", "both")) and (full or rng.random() < strength):
                out.append(SPECIAL[c])
            else:
                out.append(c)
        else:
            out.append(c)
    return unicodedata.normalize("NFC", "".join(out)), mode, strength


def add_noise(text: str, rng: random.Random, frac: Optional[float] = None) -> Tuple[str, float]:
    """Character-level noise: deletion (30%), substitution (30%, 70% keyboard neighbour / 30% any letter of the text), exchange of two
    adjacent characters (20%), addition (20%, a neighbour or duplicate). `frac` = share of the characters hit, default drawn from
    U(0.02, 0.35) (10% of the samples up to 0.50). Returns (noisy text, share actually edited)."""
    chars = list(text)
    n = len(chars)
    if n < 4:
        return text, 0.0
    if frac is None:
        frac = rng.uniform(0.02, 0.50) if rng.random() < 0.10 else rng.uniform(0.02, 0.35)
    k = max(1, min(int(round(frac * n)), n // 2))
    letters = [c for c in chars if c.isalpha()] or list("etaoin")
    positions = rng.sample(range(n), k)
    hit = {}
    for i in positions:
        c = chars[i]
        op = rng.choices(["del", "sub", "swap", "add"], weights=[0.3, 0.3, 0.2, 0.2])[0]
        hit[i] = op
    out: List[str] = []
    skip_next = False
    for i, c in enumerate(chars):
        if skip_next:
            skip_next = False
            continue
        op = hit.get(i)
        if op is None or c.isspace() and op != "add":
            out.append(c)
            continue
        if op == "del":
            continue
        if op == "sub":
            low = c.lower()
            pool = _ADJ.get(low, "") if rng.random() < 0.7 else ""
            r = rng.choice(pool) if pool else rng.choice(letters)
            out.append(r.upper() if c.isupper() else r)
        elif op == "swap" and i + 1 < n and not chars[i + 1].isspace() and not c.isspace():
            out.extend([chars[i + 1], c])
            skip_next = True
        else:  # add
            low = c.lower()
            pool = _ADJ.get(low, "")
            out.append(c)
            out.append(rng.choice(pool) if pool and rng.random() < 0.6 else c)
    noisy = "".join(out)
    return noisy, k / n


TASKS = ("diacritize", "clean", "both")
PLAN = {"diacritize": ["<diacritize>"], "clean": ["<clean>"], "both": ["<clean>", "<diacritize>"]}
TARGETS = {"diacritize": (100_000, 150_000), "clean": (50_000, 100_000), "both": (40_000, 60_000)}   # rows per language (owner)
MIN_DENSITY = 0.008          # a language "uses diacritics" when a full strip changes >= 0.8% of its characters


def pick_density_languages(samples: Dict[str, List[str]]) -> Dict[str, float]:
    out = {}
    for lang, texts in samples.items():
        if not texts:
            continue
        d = sum(diacritic_units(t) for t in texts) / max(sum(len(unicodedata.normalize("NFD", t)) for t in texts), 1)
        if d >= MIN_DENSITY:
            out[lang] = d
    return out


ENGLISH_SHARE = 0.35        # owner: 30-40% of the instructions in English; the rest in the text's own language (30%) or any other bank language


def pick_instruction_language(rng: random.Random, text_lang: str, have: List[str]) -> str:
    r = rng.random()
    if r < ENGLISH_SHARE or len(have) == 1:
        return "eng" if "eng" in have else have[0]
    if r < ENGLISH_SHARE + 0.30 and text_lang in have:
        return text_lang
    return rng.choice([l for l in have if l != "eng"] or have)


_LABEL_TAIL = re.compile(r"\s+[^\s:：]+\s?[:：]\s*[^.:]+\.$")


def _drop_shared_label(a: str, b: str) -> str:
    """the label-style templates end with `Language: {name}.`; in a joined instruction it appears once (at the end)"""
    ma, mb = _LABEL_TAIL.search(a), _LABEL_TAIL.search(b)
    if ma and mb and ma.group(0).strip() == mb.group(0).strip():
        return a[:ma.start()].rstrip()
    return a


def build_instruction(task: str, rng: random.Random, lang: str, templates) -> Tuple[str, str, str]:
    """-> (instruction text, position 'prepend'|'append', instruction language) from the multilingual pools; the combined task joins a
    clean and a diacritize instruction of the same instruction language."""
    if task == "both":
        have = [l for l in templates.languages("diacritize") if l in templates.languages("denoise")]
        il = pick_instruction_language(rng, lang, have)
        a = templates.make("denoise", rng, target=lang, source=lang, lang=il)
        b = templates.make("diacritize", rng, target=lang, source=lang, lang=il, position=a.position)
        return f"{_drop_shared_label(a.text, b.text)} {b.text}", a.position, il
    pool = "denoise" if task == "clean" else "diacritize"
    il = pick_instruction_language(rng, lang, templates.languages(pool))
    ins = templates.make(pool, rng, target=lang, source=lang, lang=il)
    return ins.text, ins.position, il


def make_example(task: str, text: str, lang: str, rng: random.Random, templates) -> Optional[dict]:
    """One training example (dict with instruction / input / response / position / meta) or None when the corruption changed nothing."""
    text = unicodedata.normalize("NFC", text)
    meta = {}
    if task == "diacritize":
        bad, mode, s = degrade_diacritics(text, rng)
        if bad == text or diacritic_units(text) - diacritic_units(bad) < 1:
            return None
        meta = {"mode": mode, "strength": round(s, 2)}
    elif task == "clean":
        bad, frac = add_noise(text, rng)
        if bad == text:
            return None
        meta = {"noise": round(frac, 3)}
    else:
        deg, mode, s = degrade_diacritics(text, rng)
        bad, frac = add_noise(deg, rng)
        if bad == text or deg == text:
            return None
        meta = {"mode": mode, "strength": round(s, 2), "noise": round(frac, 3)}
    ins, pos, il = build_instruction(task, rng, lang, templates)
    return {"instruction": ins, "position": pos, "instruction_lang": il, "input": bad, "response": text, "meta": meta}


def user_content(ex: dict) -> str:
    return f"{ex['instruction']}\n\n{ex['input']}" if ex["position"] == "prepend" else f"{ex['input']}\n\n{ex['instruction']}"


# ------------------------------------------------------------------------------------------------- text sources (catalog pretrain/ list)
# GlotLID label prefixes (ISO 639-3) accepted for a language that is outside langdetect's 23 supported ones: those get no `code` from
# sft_new.langdetect, so the verification falls back to GlotLID's own top-1 label
GLOT_ALIASES = {"ful": {"ful", "fuv", "fub", "ffm", "fuf", "fuh", "fue", "fvr", "fuc", "fuq"}, "mlg": {"mlg", "plt"}, "orm": {"orm", "gaz", "hae", "gax"},
                "swh": {"swh", "swa"}, "twi": {"twi", "aka", "fat"}, "kin": {"kin"}, "luo": {"luo"}, "lug": {"lug"}, "lin": {"lin"}}


def lid_ok(det, lang: str) -> bool:
    """a snippet is kept when sft_new.langdetect says `lang` (certain / likely), or - for the languages it does not know - GlotLID's label does"""
    if getattr(det, "code", None) == lang and getattr(det, "level", "") in ("certain", "likely"):
        return True
    from sft_new.langdetect import SUPPORTED
    if lang in SUPPORTED:                      # a language langdetect knows needs its own two-signal agreement
        return False
    label = (getattr(det, "label", None) or "").split("_")[0]
    return bool(label) and label in GLOT_ALIASES.get(lang, {lang}) and getattr(det, "code", None) is None


EVAL_PATH = re.compile(r"(^|[-_ /.])(eval|test|dev|devtest|benchmark)([-_ ./]|$)|flores|ntrex", re.I)


# display names for languages the translation-instruction module does not know (its DISPLAY_NAMES raised KeyError: 'bam' / 'loz' and killed
# two build groups on 2026-10-15); registered into africomet_translation.instructions before the first template is rendered
EXTRA_NAMES = {"bm": "Bambara", "ti": "Tigrinya", "bam": "Bambara", "ful": "Fula", "mlg": "Malagasy", "tir": "Tigrinya", "run": "Kirundi", "loz": "Lozi", "lua": "Luba-Kasai",
               "luo": "Luo", "bem": "Bemba", "ven": "Venda", "tso": "Tsonga", "ssw": "Swati", "nbl": "South Ndebele", "nso": "Sepedi",
               "kik": "Kikuyu", "lin": "Lingala", "lug": "Luganda", "amh": "Amharic", "ibb": "Ibibio", "urh": "Urhobo", "kin": "Kinyarwanda",
               "som": "Somali", "orm": "Oromo", "wol": "Wolof", "efi": "Efik", "fon": "Fon", "ewe": "Ewe", "twi": "Twi", "pcm": "Nigerian Pidgin"}


# catalog / GlotLID language code -> the tag text that exists in the tokenizer, where they differ (`<bm>`, `<ti>`; loz and lua have no tag:
# their rows would be rejected by the assembler, so they need a token swap before they can be built)
TAG_OF = {"bam": "bm", "tir": "ti"}
NO_TAG = {"loz", "lua"}


def register_names() -> None:
    from africomet_translation import instructions as AI
    for code, name in EXTRA_NAMES.items():
        AI.DISPLAY_NAMES.setdefault(code, [name])


def load_sources(path: str) -> Dict[str, List[dict]]:
    """catalog_pretrain_sources.json (africomet_translation.catalog_rl_sources --folder pretrain) -> {lang: [source, ...]}"""
    by: Dict[str, List[dict]] = defaultdict(list)
    for e in json.loads(Path(path).read_text()):
        by[e["lang"]].append(e)
    return by


def source_snippets(lang: str, sources: List[dict], rng: random.Random, n: int, lo: int = 4, hi: int = 70, per_source_rows: int = 60_000,
                    max_files: int = 12, min_units: int = 0):
    """Sentences / short passages of `lang` read from the catalog sources (language-specific, commercial-OK or unspecified licence, no NLLB, no
    evaluation sets). Every snippet is GlotLID-verified (sft_new.langdetect: code == lang and level certain / likely) before it is used."""
    from sft_new import ext_common as E
    from sft_new import langdetect as L
    from africomet_translation.text_utils import good_sentence, normalize, split_sentences
    order = list(sources)
    rng.shuffle(order)
    got = 0
    buf: List[str] = []

    def verified(batch: List[str]):
        if not batch:
            return
        for sn, d in zip(batch, L.detect_many(batch)):
            if lid_ok(d, lang):
                yield sn

    for src in order:
        try:
            files = [f for f, size in E.files_of(src["repo"]) if not EVAL_PATH.search(f)][:max_files]
            rows = 0
            for r in E.iter_files(src["repo"], files, columns=src["cols"]):
                rows += 1
                for c in src["cols"]:
                    t = r.get(c) if isinstance(r, dict) else None
                    if not isinstance(t, str) or len(t) < 20:
                        continue
                    sents = [normalize(x) for x in split_sentences(t)]
                    i = 0
                    while i < len(sents):
                        k = rng.choice((1, 1, 2, 3))
                        sn = " ".join(sents[i:i + k])
                        i += k + rng.choice((0, 1, 2))
                        nw = len(sn.split())
                        if lo <= nw <= hi and good_sentence(sn, min_words=3, max_words=hi + 20, require_terminal=False, lenient=True):
                            sn = unicodedata.normalize("NFC", sn)
                            if min_units and diacritic_units(sn) < min_units:       # keep only snippets that actually carry marks / special letters
                                continue
                            buf.append(sn)
                if len(buf) >= 1500:
                    for sn in verified(buf):
                        yield sn
                        got += 1
                        if got >= n:
                            return
                    buf = []
                if rows >= per_source_rows:
                    break
        except Exception as e:                      # one unreadable repo must not stop the language
            print(f"  source {src.get('repo')}: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
            continue
    for sn in verified(buf):
        yield sn
        got += 1
        if got >= n:
            return


# ------------------------------------------------------------------------------------------------- I/O
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the per-language targets (0.01 for a smoke test)")
    ap.add_argument("--sources", default="/workspace/sft_build/catalog_pretrain_sources.json")
    ap.add_argument("--tasks", nargs="*", choices=TASKS, help="build only these tasks (default: all)")
    ap.add_argument("--min-units", type=int, default=0,
                    help="keep only snippets with at least this many diacritic marks / special letters (a language whose text mostly lacks them, e.g. Hausa)")
    ap.add_argument("--force-diacritic", action="store_true", help="treat the language as a diacritic language even below MIN_DENSITY (use with --min-units)")
    a = ap.parse_args(argv)
    HERE = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(HERE))
    from africomet_translation.build import hf_token
    from sft_new import structured as S
    from sft_new import templates as T
    from sft_new.artificial import SCHEMA
    from sft_new import pools_multi
    pools_multi.register_all()                       # the 18-language banks incl. pools_data_restore (diacritize / denoise)
    register_names()
    hf_token()
    by_source = load_sources(a.sources)
    names = a.langs or sorted(by_source)
    out_dirs = {t: S.Out(Path(a.out), f"artificial-restore_{t}", schema=SCHEMA) for t in TASKS}
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for lang in names:
        if lang in NO_TAG:
            print(lang, "no tag in the tokenizer: skipped (needs a token swap)", file=sys.stderr, flush=True)
            continue
        rng = random.Random(f"restore-{lang}")
        run_tasks = [t for t in TASKS if not a.tasks or t in a.tasks]
        need = {t: int(rng.randint(*TARGETS[t]) * a.scale) for t in run_tasks}
        pool = []
        total_need = sum(need.values())
        for sn in source_snippets(lang, by_source.get(lang, []), rng, int(total_need * 1.3) + 100, min_units=a.min_units):
            pool.append(sn)
        if not pool:
            print(lang, "no text", file=sys.stderr)
            continue
        src_lang, lang = lang, TAG_OF.get(lang, lang)           # text was found under the catalog code; the rows carry the tokenizer tag
        dens = sum(diacritic_units(t) for t in pool[:3000]) / max(sum(len(t) for t in pool[:3000]), 1)
        di_lang = dens >= MIN_DENSITY or a.force_diacritic
        tasks = [t for t in run_tasks if t == "clean" or di_lang]
        rng.shuffle(pool)
        share = {t: need[t] / sum(need[x] for x in tasks) for t in tasks}
        cursor = 0
        for t in tasks:
            take = int(len(pool) * share[t])
            part = pool[cursor:cursor + take]
            cursor += take
            n = 0
            for sn in part:
                ex = make_example(t, sn, lang, rng, T)
                if ex is None:
                    continue
                try:
                    reply = S.turn(lang, lang, PLAN[t], ex["response"])
                except Exception:
                    stats[lang]["assemble_rejected"] += 1
                    continue
                out_dirs[t].add({"instruction": ex["instruction"], "input": ex["input"], "response": ex["response"], "language": lang,
                                 "task": {"diacritize": "diacritization", "clean": "denoising", "both": "diacritization_denoising"}[t],
                                 "messages": json.dumps([{"role": "user", "content": user_content(ex)},
                                                         {"role": "assistant", "content": reply}], ensure_ascii=False),
                                 "source_dataset": (os.environ['HF_ORG_LEGACY'] + "/pretrain-new (artificially corrupted, text_restore)")})
                n += 1
                if n >= need[t]:
                    break
            stats[lang][t] = n
        stats[lang]["diacritic_density"] = round(dens, 4)
        print(lang, dict(stats[lang]), file=sys.stderr, flush=True)
    for o in out_dirs.values():
        o.flush()
    stat_dir = Path(a.out) / "stats"
    stat_dir.mkdir(parents=True, exist_ok=True)
    for t in TASKS:
        (stat_dir / f"artificial-restore_{t}.json").write_text(json.dumps({l: s.get(t, 0) for l, s in stats.items()}, indent=1))


if __name__ == "__main__":
    main()
