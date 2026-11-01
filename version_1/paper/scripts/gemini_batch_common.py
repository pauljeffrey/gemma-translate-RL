"""Shared definitions for the Gemini Batch pseudo-reference pipeline (LC1 long-context benchmark).

Standard library only.  Used by gemini_count_tokens.py (box), gemini_batch_cost.py, gemini_batch_requests.py
and gemini_batch_ingest.py, so that the instruction text, language names, chunking and the reduced-set
sampling rule are defined exactly once.
"""
import random

LANG_NAMES = {
    "hau": "Hausa", "yor": "Yoruba", "swh": "Swahili", "amh": "Amharic", "ibo": "Igbo", "mlg": "Malagasy",
    "zul": "isiZulu", "xho": "isiXhosa", "kin": "Kinyarwanda", "som": "Somali", "wol": "Wolof",
    "twi": "Twi (Akuapem)", "lug": "Luganda", "nso": "Northern Sotho (Sepedi)",
}
HUMAN_LANGS = "hau yor swh amh ibo mlg zul xho kin som wol twi".split()
SCI_LANGS = "amh hau lug nso yor zul".split()

# Reduced set: 8 languages x 5 documents per bin x 3 bins.  The 8 languages are the benchmark languages for which
# Gemini-2.5-Flash FLORES chrF++ beats NLLB-3.3B by >= 3 points on both NLLB references (screening rule, plan section 3).
# amh (no Gemini number), yor, wol (below NLLB) and twi (+2) are NOT in the reduced set.  Change here only.
REDUCED_LANGS = "hau swh ibo mlg zul xho kin som".split()
REDUCED_BINS = ["b512", "b1024", "b2048"]
REDUCED_DOCS_PER_BIN = 5
REDUCED_SEED = 20261023

# ---- the instruction (exactly this text is sent as system_instruction) ----------------------------------------
INSTR_WHOLE = (
    "You are a professional translator. Translate the user's English text into {lang}. "
    "Translate everything: do not omit, summarize, shorten or add anything. "
    "Keep the paragraph breaks (blank lines) and line structure exactly as in the input. "
    "Keep names, numbers, URLs and symbols unchanged. "
    "Output only the translation: no commentary, notes, preface or code fences."
)
INSTR_CHUNK = (
    "You are a professional translator. Translate the user's English text into {lang}. "
    "The input has {n} line(s), each one sentence. Translate every line; output exactly {n} line(s), one translation "
    "per input line, in the same order, with no numbering and no blank lines. "
    "Do not omit, summarize, merge or add anything. Keep names, numbers, URLs and symbols unchanged. "
    "Output only the translation: no commentary, notes, preface or code fences."
)
FRAMING_TOKENS = 10        # assumed extra tokens per request for roles / turn markers (not measurable offline)


def instruction(iso, mode="whole", n_lines=1):
    lang = LANG_NAMES[iso]
    if mode == "whole":
        return INSTR_WHOLE.format(lang=lang)
    return INSTR_CHUNK.format(lang=lang, n=n_lines)


def chunk_spans(n_sents, n):
    """[(lo, hi), ...] half-open sentence ranges of at most n sentences."""
    if n <= 0:
        raise ValueError("chunk size must be >= 1")
    return [(lo, min(lo + n, n_sents)) for lo in range(0, n_sents, n)]


def est_tokens_chars4(text):
    return (len(text) + 3) // 4


def reduced_doc_ids(rows, bins=REDUCED_BINS, per_bin=REDUCED_DOCS_PER_BIN, seed=REDUCED_SEED, set_name="core"):
    """Fixed-seed sample of documents.  rows: iterable of dicts with keys doc_id, set, bin.
    Returns {bin: [doc_id, ...]}.  Depends only on the sorted doc_id list per bin, so the cost script (items CSV)
    and the request generator (docs.jsonl) select the same documents."""
    by_bin = {}
    for r in rows:
        if r["set"] == set_name and r["bin"] in bins:
            by_bin.setdefault(r["bin"], set()).add(r["doc_id"])
    out = {}
    for b in bins:
        ids = sorted(by_bin.get(b, []))
        rng = random.Random("%s|%s|%s" % (seed, set_name, b))
        out[b] = sorted(rng.sample(ids, min(per_bin, len(ids))))
    return out


def select_items(items, selection, seed=REDUCED_SEED):
    """items: dicts with iid, doc_id, iso, set, bin.  selection in {full, reduced_core, reduced_sci_dev, reduced}.
    reduced = reduced_core + reduced_sci_dev."""
    items = list(items)
    if selection == "full":
        return items
    out = []
    if selection in ("reduced_core", "reduced"):
        keep = reduced_doc_ids(items, seed=seed)
        ok = {d for ds in keep.values() for d in ds}
        out += [i for i in items if i["set"] == "core" and i["doc_id"] in ok and i["iso"] in REDUCED_LANGS]
    if selection in ("reduced_sci_dev", "reduced"):
        out += [i for i in items if i["set"] == "sci" and i["bin"] == "sci_dev"]
    if selection not in ("reduced_core", "reduced_sci_dev", "reduced"):
        raise ValueError("unknown selection %r" % selection)
    return out


KEY_SEP = "@@c"


def make_key(iid, chunk=None):
    return iid if chunk is None else "%s%s%04d" % (iid, KEY_SEP, chunk)


def split_key(key):
    """key -> (iid, chunk index or None)."""
    if KEY_SEP in key:
        iid, c = key.rsplit(KEY_SEP, 1)
        if c.isdigit():
            return iid, int(c)
    return key, None
