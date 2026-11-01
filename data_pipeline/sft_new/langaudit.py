"""Audit of a DECLARED language against GlotLID, for sources whose language is stated by a column or a repo name.

Why: michsethowusu/oromo-emotions-corpus has an "Oromo" column that holds romanised Urdu. A declared language is a claim, not a fact (the
NLLB-derived michsethowusu sets are the worst offenders, but any scraped source can do it), and a mislabelled row teaches the model a wrong
language tag. Every declared language is therefore checked with GlotLID, and a row is dropped when -- and only when -- GlotLID is CONFIDENT
(top-1 probability >= TAU) that the text is a DIFFERENT language. Text GlotLID has no model for, or is unsure about, is kept (the language it
claims may simply be one GlotLID reads poorly) and marked `unsure`, which the assistant turn then writes as "unsure, but most likely <tag>".

Verdicts
    confirmed    GlotLID's top label is the declared language (or a sibling/variant of it) with reasonable confidence, or the stopword evidence is certain
    unconfirmed  GlotLID is not confident about anything else: kept, flagged unsure
    mismatch     GlotLID is confident (>= TAU) about a language that is not the declared one nor a sibling of it: DROP

Besides the 23 languages of langdetect.SUPPORTED this module registers French (`fra`), Amharic (`amh`) and Tigrinya (`ti`): all three have a tag in
the tokenizer (<fra> 134, <amh> 124, <ti> 119); Dagbani (`dag`) is added on the owner's instruction, its tag arrives by replacing a redundant special token. They are never *detected* (no stopword lists) -- only declared and audited.
"""

from __future__ import annotations

from dataclasses import dataclass

from sft_new import langdetect as L

TAU = 0.55
CONFIRM_P = 0.30                       # a top label equal to the declared language needs at least this probability to count as confirmed

EXTRA_LANGS = {"fra": "French", "amh": "Amharic", "ti": "Tigrinya", "dag": "Dagbani", "bin": "Edo", "ssw": "Swati", "nso": "Northern Sotho", "tso": "Tsonga",
               "ven": "Venda", "kik": "Kikuyu", "lug": "Luganda", "lin": "Lingala", "luo": "Luo", "bem": "Bemba",
               "nbl": "Southern Ndebele", "kln": "Kalenjin", "llg": "Lulogooli", "lch": "Lumarachi"}   # every tag is in the tokenizer (dag, bin, nbl, kln, llg, lch by swaps)
_EXTRA_NAMES = {"french": "fra", "francais": "fra", "français": "fra", "fr": "fra", "fra": "fra", "fre": "fra",
                "amharic": "amh", "am": "amh", "amh": "amh",
                "tigrinya": "ti", "tigrigna": "ti", "ti": "ti", "tir": "ti", "dagbani": "dag", "dagbanli": "dag", "dag": "dag", "edo": "bin", "bini": "bin", "bin": "bin", "swati": "ssw", "siswati": "ssw", "ssw": "ssw", "ss": "ssw",
                "northern sotho": "nso", "sepedi": "nso", "nso": "nso", "tsonga": "tso", "xitsonga": "tso", "tso": "tso", "ts": "tso", "venda": "ven", "tshivenda": "ven",
                "ven": "ven", "ve": "ven", "kikuyu": "kik", "gikuyu": "kik", "kik": "kik", "ki": "kik", "luganda": "lug", "lug": "lug", "lg": "lug", "lingala": "lin",
                "lin": "lin", "ln": "lin", "luo": "luo", "dholuo": "luo", "bemba": "bem", "bem": "bem",
                "southern ndebele": "nbl", "south ndebele": "nbl", "isindebele": "nbl", "ndebele (south africa)": "nbl", "nbl": "nbl", "nr": "nbl",
                "kalenjin": "kln", "kln": "kln", "lulogooli": "llg", "logooli": "llg", "maragoli": "llg", "llg": "llg", "lumarachi": "lch", "marachi": "lch", "lch": "lch"}

# GlotLID labels (ISO 639-3 prefix) that are the same language as our code.
SAME = {
    "eng": {"eng"}, "pcm": {"pcm", "wes", "eng"}, "hau": {"hau"}, "ibo": {"ibo"}, "yor": {"yor"}, "efi": {"efi"}, "ibb": {"ibb"}, "urh": {"urh"},
    "twi": {"twi", "aka", "fat"}, "ewe": {"ewe"}, "fon": {"fon"}, "kin": {"kin"}, "sna": {"sna"}, "zul": {"zul"}, "swh": {"swh", "swa", "swc"},
    "orm": {"orm", "gaz", "hae", "gax"}, "wol": {"wol"}, "xho": {"xho"}, "tsn": {"tsn"}, "nya": {"nya"}, "afr": {"afr"}, "som": {"som"},
    "sot": {"sot"}, "fra": {"fra"}, "amh": {"amh"}, "ti": {"tir"}, "dag": {"dag"}, "bin": {"bin"}, "ssw": {"ssw"}, "nso": {"nso"},
    "tso": {"tso"}, "ven": {"ven"}, "kik": {"kik"}, "lug": {"lug"}, "lin": {"lin"}, "luo": {"luo", "luo"}, "bem": {"bem"}, "nbl": {"nbl"}, "kln": {"kln", "enb", "pko", "niq", "tuy"}, "llg": {"llg", "lks", "rag"}, "lch": {"lch", "lri"},
}
SAME["eng"] = {"eng", "pcm", "wes"}                  # the detector reads Pidgin as English, so English-vs-Pidgin is never a data error
# Close relatives GlotLID confuses with each other: a confident verdict for a sibling is NOT evidence that the declaration is wrong.
SIBLING_GROUPS = [{"zul", "xho", "nbl", "ssw", "nde"}, {"tsn", "sot", "nso"}, {"efi", "ibb", "anw"}, {"twi", "aka", "fat"}, {"orm", "gaz", "hae", "gax"},
                  {"swh", "swa", "swc"}, {"kin", "run"},
                  # GlotLID's labels for English and French VARIANTS: a confident "Scots" / "Middle English" / "Jamaican" verdict on a sentence of ordinary English
                  # (or "Norman" / "Picard" on French) is detector noise, not a mislabelled column (kweku_eng_twi and fon_pairs smoke test, 2026-10-10)
                  {"eng", "sco", "enm", "jam", "acf", "gcr", "bah", "bzj", "svc", "vic", "aig", "srm", "tpi", "pis", "pih", "ang", "pcm", "wes"},
                  {"fra", "nrm", "pcd", "wln", "frp", "rcf", "fro", "frm"},
                  # the Gbe cluster: Fon is confused with its closest neighbours (Aja, Gen, Maxi, Xwela); Ewe is deliberately NOT included
                  {"fon", "ajg", "gej", "mxl", "xwe", "gbe"}]


# Languages whose tag already exists in the tokenizer (specials 94-159 and a few older ones) but that had no converter support: declaring them costs no swap.
# code = the tag text between < >; names are what datasets call them; `iso3` is the code GlotLID uses when it differs from the tag.
EXISTING_TAG_LANGS = {
    "anw": ["Anaang"], "orx": ["Oro"], "igl": ["Igala"], "snq": ["Sangu"], "nup": ["Nupe"], "oke": ["Okpe"], "knc": ["Kanuri"], "umb": ["Umbundu"], "kam": ["Kamba"],
    "son": ["Songhay"], "bm": ["Bambara", "Bamanankan"], "dyu": ["Dyula", "Jula"], "bbj": ["Ghomala"], "mzw": ["Deg"], "din": ["Dinka"], "kon": ["Kongo", "Kikongo"],
    "gaa": ["Ga"], "kea": ["Kabuverdianu", "Cape Verdean"], "kbp": ["Kabiye", "Kabiyè"], "kab": ["Kabyle"], "xsm": ["Kasem"], "kmb": ["Kimbundu"], "ngl": ["Lomwe"],
    "lus": ["Mizo", "Lushai"], "mlg": ["Malagasy"], "mos": ["Mossi", "Moore", "Mooré"], "nus": ["Nuer"], "run": ["Rundi", "Kirundi"], "sag": ["Sango"], "tmh": ["Tamashek", "Tuareg"],
    "tum": ["Tumbuka"], "vai": ["Vai"], "vag": ["Vagla"], "mey": ["Hassaniya"], "nde": ["North Ndebele", "Northern Ndebele"], "shi": ["Tashelhit", "Tachelhit"], "fat": ["Fante", "Fanti"],
    "tzm": ["Central Atlas Tamazight", "Tamazight"], "tiv": ["Tiv"], "ijo": ["Ijo", "Ijaw"], "bin": ["Edo", "Bini"],
}
_ISO3_TO_TAG = {"bam": "bm", "mlg": "mlg", "plt": "mlg"}                     # GlotLID / ISO 639-3 code -> the tag text, where they differ


def register_existing_tag_languages() -> None:
    for code, names in EXISTING_TAG_LANGS.items():
        EXTRA_LANGS.setdefault(code, names[0])
        SAME.setdefault(code, {code})
        for n in names:
            _EXTRA_NAMES.setdefault(n.lower(), code)
        _EXTRA_NAMES.setdefault(code, code)
    _EXTRA_NAMES.setdefault("bam", "bm")
    SAME["bm"] = {"bm", "bam"}
    SAME["mlg"] = {"mlg", "plt"}


def accepted(code: str) -> set:
    out = set(SAME.get(code, {code}))
    for g in SIBLING_GROUPS:
        if out & g:
            out |= g
    return out


def code_of(name: str | None) -> str | None:
    """Like langdetect.code_of, plus French / Amharic / Tigrinya."""
    c = L.code_of(name)
    if c:
        return c
    if not name:
        return None
    return _EXTRA_NAMES.get(str(name).strip().strip("<>|").strip().lower())


def register_extra_languages() -> None:
    """Make fra / amh / ti ... legal in assistant turns and nameable in instructions (idempotent)."""
    from africomet_translation import instructions as I
    from sft_new.compose import A
    register_existing_tag_languages()
    for code, names in EXISTING_TAG_LANGS.items():
        I.DISPLAY_NAMES.setdefault(code, names)
    A.VALID_LANGS |= set(EXTRA_LANGS)
    A.VALID_VERBS |= {"<nli>"}                                   # token 93
    I.DISPLAY_NAMES.setdefault("fra", ["French", "Français"])
    I.DISPLAY_NAMES.setdefault("amh", ["Amharic", "አማርኛ"])
    I.DISPLAY_NAMES.setdefault("ti", ["Tigrinya", "ትግርኛ"])
    I.DISPLAY_NAMES.setdefault("dag", ["Dagbani", "Dagbanli"])
    for c, names in (("bin", ["Edo", "Bini"]), ("ssw", ["Swati", "siSwati"]), ("nso", ["Northern Sotho", "Sepedi"]), ("tso", ["Tsonga", "Xitsonga"]),
                     ("ven", ["Venda", "Tshivenda"]), ("kik", ["Kikuyu", "Gikuyu"]), ("lug", ["Luganda"]), ("lin", ["Lingala"]), ("luo", ["Dholuo", "Luo"]),
                     ("bem", ["Bemba", "Chibemba"]), ("nbl", ["Southern Ndebele", "isiNdebele"]), ("kln", ["Kalenjin"]), ("llg", ["Lulogooli", "Logooli"]), ("lch", ["Lumarachi", "Marachi"])):
        I.DISPLAY_NAMES.setdefault(c, names)


@dataclass
class Verdict:
    status: str                         # confirmed | unconfirmed | mismatch
    detected: str | None = None         # GlotLID's raw top label
    prob: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status != "mismatch"

    @property
    def unsure(self) -> bool:
        return self.status != "confirmed"


def _iso(label: str | None) -> str | None:
    return label.split("_")[0] if label else None


def _letter_ratio(text: str | None) -> float:
    if not text:
        return 1.0
    return sum(ch.isalpha() for ch in text) / max(1, len(text.replace(" ", "")))


def judge(d: "L.Detection", declared: str, tau: float = TAU, text: str | None = None) -> Verdict:
    """One detection against a declared language. `text` (optional) lets the judge treat formula-, number- or symbol-heavy strings (under 60% letters)
    as unjudgeable: a confident verdict on "x^2 + 3 = 12" says nothing about the language of the sentence it came from."""
    top = _iso(d.label)
    acc = accepted(declared)
    if top in acc:
        sure = d.prob >= CONFIRM_P or (d.code == declared and d.level == "certain")
        return Verdict("confirmed" if sure else "unconfirmed", d.label, d.prob)
    if d.code == declared and d.level == "certain":              # the stopword evidence is decisive and our own detector agrees
        return Verdict("confirmed", d.label, d.prob)
    if top and d.prob >= tau and _letter_ratio(text) >= 0.60:
        return Verdict("mismatch", d.label, d.prob)
    return Verdict("unconfirmed", d.label, d.prob)


def audit(texts: list, declared: str, tau: float = TAU) -> list:
    """Verdicts for texts that are all declared to be in `declared`. Empty / non-letter texts are unconfirmed (nothing to judge)."""
    idx = [i for i, t in enumerate(texts) if t and any(ch.isalpha() for ch in t)]
    out = [Verdict("unconfirmed")] * len(texts)
    if idx:
        for i, d in zip(idx, L.detect_many([texts[i][:1500] for i in idx])):
            out[i] = judge(d, declared, tau, texts[i])
    return out


def audit_pairs(a: list, b: list, la: str, lb: str, tau: float = TAU) -> list:
    return list(zip(audit(a, la, tau), audit(b, lb, tau)))
