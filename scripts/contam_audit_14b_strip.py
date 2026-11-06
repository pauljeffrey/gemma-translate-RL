#!/usr/bin/env python3
"""Variant pass of the 385M contamination scan: the R4 rule of contam_audit_12 (diacritics dropped after NFD) on top of the normalisation of gemma_mt.eval.contamination.

Imports the scanner `contam_audit_14_385m_scan` unchanged (read only, no bytecode written next to it), replaces `gemma_mt.eval.contamination.normalise` by the mark-stripping
normalisation and runs its CLI (scan / finalize) with the same arguments. Reason: Yoruba, Fon, Ewe, Igbo ... are written with and without under-dots / tone marks and in NFC / NFD in different
corpora; a benchmark sentence that differs from a training sentence only in marks is the same sentence. Quote folding of R4 is a no-op here: every quote / apostrophe is a non-alphanumeric
and becomes a space in both normalisations.
"""
import os, sys
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"; os.environ.setdefault("TOKENIZERS_PARALLELISM", "false"); os.environ.setdefault("RAYON_NUM_THREADS", "1")
    os.execv(sys.executable, [sys.executable] + sys.argv)
sys.dont_write_bytecode = True
import re, unicodedata
SCANNER_DIR = os.environ.get("SCANNER_DIR", "/workspace/eval_385m/contam_audit"); TREE = os.environ.get("TREE", "/workspace/eval_385m/gemma_mt_tree")
sys.path.insert(0, TREE); sys.path.insert(0, SCANNER_DIR)
from gemma_mt.eval import contamination as C

# combining marks of the scripts of the registry (Latin diacritics incl. U+0323 / U+0329, Cyrillic, Hebrew, Arabic points, Ge'ez combining, combining symbols, half marks)
_COMB = re.compile("[̀-ͯ҃-҉֑-ׇֽֿׁׂׅׄؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ"
                   "፝-፟᪰-᫿᷀-᷿⃐-⃿︠-︯]")


def normalise_strip(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").casefold()
    if not t.isascii():
        t = _COMB.sub("", unicodedata.normalize("NFD", t))
    return C._NON_ALNUM.sub(" ", t).strip()


assert normalise_strip("Ẹ káàbọ̀ sí Ìbàdàn, ìlú") == "e kaabo si ibadan ilu", normalise_strip("Ẹ káàbọ̀ sí Ìbàdàn, ìlú")
assert normalise_strip("it’s  Fine") == "it s fine"
C.normalise = normalise_strip
import contam_audit_14_385m_scan as M
if __name__ == "__main__":
    M.main()
