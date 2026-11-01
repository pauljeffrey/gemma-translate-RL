# SPDX-License-Identifier: AGPL-3.0-or-later
"""Source-repository -> provenance kind and quality tier (a NAME-PATTERN HEURISTIC, not a per-dataset audit).

The en->X build (`final_v2`) keeps `source_dataset` per row, e.g. `michsethowusu/english-zulu_sentence-pairs_mt560` or
`<HF_ORG>/tds:train-00003-of-00010`. The inter-African builder needs a coarse "how was this text made" label to (a) cap the share of machine
generated pairs, (b) rank candidates for the same (English sentence, language) and (c) tag the output so users can filter. The rules below are
checked in order, the first match wins; anything unmatched is `mixed` (tier 1).

kinds / tiers (higher tier = more trusted):
  3 human_curated   curated or professionally translated corpora (government text, community-curated sentence pairs, news, dictionaries of sentences)
  2 religious_jw    Bible / JW / OPUS-MT560 style corpora: human translations, but one domain and often archaic or literal
  1 mixed           aggregate repositories whose parts we cannot tell apart, web-mined pairs (CCMatrix, Paracrawl, Tatoeba mining)
  0 synthetic_mt    machine translated or LLM generated text (tds, *synthetic*, topxgen, back-translations, GEC synthetic)
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

KINDS = {3: "human_curated", 2: "religious_jw", 1: "mixed", 0: "synthetic_mt"}
KIND_TIER = {v: k for k, v in KINDS.items()}

# (regex on the lower-cased repo id, kind). First match wins.
_RULES: Tuple[Tuple[str, str], ...] = (
    (r"(^|/)tds($|:)|synthetic|topxgen|backtransl|back-transl|gec-|_gec|aya_collection|aya_interlanguage|nllb|machine|mt-gen|generated|translatepsy", "synthetic_mt"),
    (r"bible|jw300|jw_|-jw|african-corpus-jw|_mt560|mt560|quran|alxuraan|biblenlp|opus_xhosanavy|wolof-english-bible", "religious_jw"),
    (r"ccmatrix|paracrawl|opus-100|opus_|helsinki-nlp|tatoeba|wikimatrix|cc-?aligned|laser|mined|bitext_", "mixed"),
    (r"^(dsfsi|masakhane|mbazanlp|digitalumuganda|kencorpus|nwu-ctext|sello-ralethe|ghana-nlp|tamazight-nlp|maliba-ai|galsenai|francophonia|clear-global|sunbird|"
     r"taqbaylit|ubc-nlp|robotsmali|uvci|capitainvigs|oumar199|maroneai|lahad|cibfaye|sudoping01|tufaax|rasyosef|pkyoyetera|kambale|"
     r"fiifinketia|naathnlp|dayomtechnologies|rogendo|ashuchufamo|zarnite|takum23|joshordin|okwu|african-languages-lab|tysby101|dofbi|"
     r"abdelhaqueidali|agbalu|ayymen)/", "human_curated"),
)
_COMPILED = tuple((re.compile(p), k) for p, k in _RULES)


def repo_of(source_dataset: str) -> str:
    """`owner/name:config` -> `owner/name`."""
    return (source_dataset or "").split(":", 1)[0]


def classify_repo(source_dataset: str) -> str:
    r = repo_of(source_dataset).lower()
    for rx, kind in _COMPILED:
        if rx.search(r):
            return kind
    return "mixed"


def tier_of(source_dataset: str) -> int:
    return KIND_TIER[classify_repo(source_dataset)]


def pair_tier(tier_a: int, tier_b: int) -> int:
    """A pivot-derived pair is as trustworthy as its weaker side."""
    return min(int(tier_a), int(tier_b))


# --- licence classes written by gemma_sft_pairs / gemma_merge_pairs.py (non_commercial pairs were already dropped there)
LICENCE_ORDER = ("cc0", "mit", "apache", "cc-by", "odc-by", "mpl", "other", "unspecified", "share_alike", "non_commercial")


def combine_licences(a: str, b: str) -> str:
    """The more restrictive of two licence classes (order in LICENCE_ORDER); unknown names count as `other`."""
    ia = LICENCE_ORDER.index(a) if a in LICENCE_ORDER else LICENCE_ORDER.index("other")
    ib = LICENCE_ORDER.index(b) if b in LICENCE_ORDER else LICENCE_ORDER.index("other")
    return LICENCE_ORDER[max(ia, ib)]


def licence_class(raw) -> str:
    """Licence class of a declared licence string (same rules as `gemma_merge_pairs.lic_class` of the en->X build)."""
    s = re.sub(r"[ _]+", "-", (raw or "").lower().strip())
    if not s or s in {"none", "unknown", "unspecified"}:
        return "unspecified"
    if re.search(r"(^|-)nc(-|$)|non-?commercial", s):
        return "non_commercial"
    if re.search(r"-sa|share-?alike|gfdl|odbl|gpl", s):
        return "share_alike"
    for pat, name in ((r"cc0|public.?domain", "cc0"), (r"apache", "apache"), (r"\bmit\b", "mit"), (r"cc-by", "cc-by"), (r"odc-by", "odc-by"), (r"mpl", "mpl")):
        if re.search(pat, s):
            return name
    return "other"
