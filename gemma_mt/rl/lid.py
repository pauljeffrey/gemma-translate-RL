# SPDX-License-Identifier: AGPL-3.0-or-later
"""Language identification for the RL reward (GlotLID, D-37): robust loading, batched prediction, a cache, a fake for tests.

`GlotLIDClassifier` wraps the fastText model `cis-lmu/glotlid` (`model_v3.bin`, 2,102 labels such as `yor_Latn`, `amh_Ethi`, `plt_Latn`).

numpy 2 vs fastText: `fasttext.FastText._FastText.predict` (fasttext-wheel 0.9.2) calls `np.array(probs, copy=False)`, which numpy >= 2 rejects
("Unable to avoid copy while creating an array as requested"). The classifier therefore never calls `predict`; it calls the C++ binding
`model.f.predict(line + "\n", k, threshold, on_unicode_error)` directly, which returns plain `(probability, label)` pairs, and builds the labels
itself. Texts must not contain a line break (fastText reads one line per call): newlines become spaces. Long texts are truncated to
`max_chars` (the model is a bag of n-gram embeddings, so the first ~1,000 characters decide the same as the rest).

Interface used by the reward: `lid(texts) -> list of top-1 labels` (e.g. "yor_Latn"; "und" for nothing). `lid.top(texts, k)` gives
[(label, prob), ...] per text for measurement. Results are cached by text (RL samples repeat across the 8 members of a group).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_REPO = "cis-lmu/glotlid"
DEFAULT_FILE = "model_v3.bin"
UNDETERMINED = "und"


def clean_for_fasttext(text: str, max_chars: int = 1000) -> str:
    """One line, whitespace collapsed, at most `max_chars` characters (cut at a space when possible)."""
    t = " ".join(text.replace("\x00", " ").split())
    if len(t) > max_chars:
        cut = t.rfind(" ", 0, max_chars)
        t = t[: cut if cut > max_chars // 2 else max_chars]
    return t


class GlotLIDClassifier:
    """fastText GlotLID. `path`: a local `model_v3.bin` (no network); otherwise it is fetched with `huggingface_hub` (token from HF_TOKEN)."""

    def __init__(self, path: Optional[str] = None, repo: str = DEFAULT_REPO, filename: str = DEFAULT_FILE, max_chars: int = 1000,
                 cache_size: int = 20000):
        self.path, self.repo, self.filename, self.max_chars = path, repo, filename, int(max_chars)
        self._model = None
        self._cache: "OrderedDict[str, Tuple[Tuple[str, float], ...]]" = OrderedDict()
        self._cache_size = int(cache_size)
        self.calls = 0                      # texts that reached the model (cache misses)

    # ------------------------------------------------------------------------------------------------ model
    def _load(self):
        if self._model is None:
            import fasttext

            path = self.path
            if not path or not os.path.exists(path):
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(self.repo, self.filename, token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
            try:
                fasttext.FastText.eprint = lambda *a, **k: None          # "Warning : `load_model` does not return WordVectorModel" noise
            except Exception:
                pass
            self._model = fasttext.load_model(path)
        return self._model

    def labels(self) -> List[str]:
        """All labels of the model without the `__label__` prefix."""
        return [l.replace("__label__", "") for l in self._load().get_labels()]

    # ------------------------------------------------------------------------------------------------ prediction
    def _predict_many(self, texts: Sequence[str], k: int) -> List[Tuple[Tuple[str, float], ...]]:
        """[(label, prob), ...] per text, from the C++ binding `model.f.predict(line, k, threshold, on_unicode_error)` that returns [(prob, "__label__xxx_Latn"), ...]
        (numpy is not involved). `multilinePredict` is NOT used: in fasttext-wheel 0.9.2 it returns the top probability for every rank."""
        model = self._load()
        out: List[Tuple[Tuple[str, float], ...]] = []
        for t in texts:
            line = clean_for_fasttext(t, self.max_chars)
            if not line:
                out.append(((UNDETERMINED, 0.0),))
                continue
            pairs = model.f.predict(line + "\n", k, 0.0, "strict")        # fastText reads one line; the binding needs the terminating newline
            self.calls += 1
            got = tuple((lab.replace("__label__", ""), float(p)) for p, lab in pairs)
            out.append(got or ((UNDETERMINED, 0.0),))
        return out

    def top(self, texts: Sequence[str], k: int = 1) -> List[List[Tuple[str, float]]]:
        keys = [f"{k}|{t}" for t in texts]
        miss = []
        seen = set()
        for key, t in zip(keys, texts):
            if key not in self._cache and key not in seen:
                miss.append((key, t))
                seen.add(key)
        if miss:
            for (key, _), hit in zip(miss, self._predict_many([t for _, t in miss], k)):
                self._cache[key] = hit
        res = [list(self._cache[key]) for key in keys]
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return res

    def __call__(self, texts: Sequence[str]) -> List[str]:
        return [r[0][0] for r in self.top(texts, 1)]


class FakeLID:
    """Test double: `rules` maps a substring of the text to the label returned (first match wins), `default` otherwise; counts calls."""

    def __init__(self, rules: Optional[Dict[str, str]] = None, default: str = "und", fn: Optional[Callable[[str], str]] = None):
        self.rules, self.default, self.fn = dict(rules or {}), default, fn
        self.calls: List[List[str]] = []

    def __call__(self, texts: Sequence[str]) -> List[str]:
        self.calls.append(list(texts))
        out = []
        for t in texts:
            lab = self.fn(t) if self.fn else None
            if lab is None:
                lab = next((v for k, v in self.rules.items() if k in t), self.default)
            out.append(lab)
        return out


# Groups of GlotLID labels of genuinely close languages / varieties. A language's accepted label set may contain, besides its own labels, the labels of its group
# that were MEASURED to absorb >= 3% of its human sentences (scripts/measure_lid.py; D-37). Nothing outside a group is ever added: English and other far
# languages never become acceptable, whatever the confusion rate (Pidgin text that GlotLID calls English therefore keeps the guard OFF for pcm).
SIBLING_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("zul_Latn", "xho_Latn", "ssw_Latn", "nbl_Latn", "nde_Latn"),            # Nguni
    ("sot_Latn", "tsn_Latn", "nso_Latn"),                                    # Sotho-Tswana
    ("kin_Latn", "run_Latn"),                                                # Kinyarwanda / Kirundi
    ("swh_Latn", "swc_Latn", "swa_Latn"),                                    # Swahili / Congo Swahili
    ("lug_Latn", "nyn_Latn", "nyo_Latn", "xog_Latn"),                        # Luganda / Nyankore and relatives
    ("efi_Latn", "ibb_Latn"),                                                # Efik / Ibibio
    ("bam_Latn", "dyu_Latn", "mnk_Latn", "emk_Latn", "mku_Latn"),            # Manding
    ("kon_Latn", "kng_Latn", "kwy_Latn", "ktu_Latn", "kikongo_Latn"),        # Kikongo varieties
    ("twi_Latn", "aka_Latn", "fat_Latn"),                                    # Akan
    ("pcm_Latn", "wes_Latn"),                                                # West African / Cameroon Pidgin
    ("gaz_Latn", "orm_Latn", "hae_Latn", "gax_Latn"),                        # Oromo
    ("plt_Latn", "mlg_Latn"),                                                # Malagasy
    ("tso_Latn", "tsc_Latn"),                                                # Tsonga / Tswa
    ("lun_Latn", "lue_Latn", "cjk_Latn"),                                    # Lunda / Luvale / Chokwe
)


def siblings_of(labels: Sequence[str]) -> Tuple[str, ...]:
    """All labels that share a SIBLING_GROUPS group with one of `labels` (excluding the labels themselves)."""
    own = set(labels)
    out = []
    for g in SIBLING_GROUPS:
        if own & set(g):
            out += [x for x in g if x not in own]
    return tuple(dict.fromkeys(out))


def accepted_labels(lang: str, entry: Optional[dict], fallback: Sequence[str]) -> Tuple[Tuple[str, ...], int, bool]:
    """(accepted labels, minimum text length, active) of a language from its calibration `lid` block; `fallback` (languages.py glotlid) when the
    calibration has no block for it."""
    if entry is None:
        return tuple(fallback), 0, bool(fallback)
    labels = tuple(entry.get("labels") or ())
    return labels, int(entry.get("min_chars") or 0), bool(entry.get("enabled")) and bool(labels)


SELFTEST = (("Hello, how are you today? I hope that you and your family are well.", ("eng_Latn",)),
            ("Mo fẹ́ lọ sí ọjà lónìí láti ra ẹ̀fọ́ àti ẹran fún ìdílé mi.", ("yor_Latn",)),
            ("Ina son zuwa kasuwa yau domin in sayi kayan lambu da nama ga iyalina.", ("hau_Latn",)))


def selftest(lid: Callable[[Sequence[str]], Sequence[str]]) -> None:
    """Three canned sentences must come back with the right label: catches a broken fastText / numpy combination or a wrong model file at construction."""
    got = list(lid([t for t, _ in SELFTEST]))
    bad = [(t[:30], g, want) for (t, want), g in zip(SELFTEST, got) if g not in want]
    if bad:
        raise RuntimeError(f"the language-identification model fails its self-test: {bad}")


def make_lid(path: Optional[str] = None):
    """The GlotLID classifier of the reward, loaded and self-tested NOW (a run must not discover a broken guard at step 1 or train without it)."""
    lid = GlotLIDClassifier(path=path)
    lid._load()
    selftest(lid)
    return lid
