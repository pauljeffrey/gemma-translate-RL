#!/usr/bin/env python3
"""Decontamination filter for training / reward-reference text against protected evaluation sets (protocol of section 5 of the 2026-10-26 audit).

Rules (all applied to BOTH sides of a pair, English and target, in any language, because the benchmarks are multi-way):
  R1 exact      normalised sentence equality (NFKC, case-fold, every run of non-alphanumerics -> one space; <3 tokens never match on their own)
  R2 contiguous the normalised protected sentence occurs inside the text as a whole-token run (catches windows and concatenations)
  R3 8-gram     >= 1 shared word 8-gram (strict) / share >= 0.5 of the text's 8-grams (near); texts < 8 tokens are exact-only
  R4 variants   the same test after dropping diacritics (NFD, strip combining marks) and after folding Latin look-alike quotes: Yoruba, Fon, Ewe etc. appear in NFC / NFD /
                with U+0323 vs U+0329 under-dots in different corpora (28% of the Yoruba FLORES+ references are not NFC), so a single normal form under-detects
Protected set files: one sentence per line (.txt) or jsonl with the fields given by --fields. Usage:
    python contam_audit_12_decontaminate.py --protected flores_dev.txt flores_devtest.txt ntrex.txt ... --input train.jsonl --fields english target --out clean.jsonl --report report.json
    python contam_audit_12_decontaminate.py --selftest
The report lists, per protected set, the rows removed by each rule. Nothing is written except --out and --report.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict

_NON = re.compile(r"[\W_]+", re.UNICODE)
_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "″": '"', "ʼ": "'", "ʹ": "'"})
N = 8


def norm(t: str, strip_marks: bool = False) -> str:
    t = unicodedata.normalize("NFKC", t or "").translate(_QUOTES).casefold()
    if strip_marks:
        t = "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c))
    return _NON.sub(" ", t).strip()


def grams(tok):
    return [hash(tuple(tok[i:i + N])) for i in range(len(tok) - N + 1)]


class Protected:
    def __init__(self):
        self.sent = defaultdict(set)         # normalised sentence -> {set names}
        self.gram = defaultdict(set)         # 8-gram hash -> {set names}
        self.first = defaultdict(list)       # first 8-gram -> [normalised sentences]
        self.sets = Counter()

    def add(self, name: str, text: str):
        for sm in (False, True):
            n = norm(text, sm)
            if len(n.split()) < 3:
                continue
            self.sent[n].add(name)
            tok = n.split()
            gs = grams(tok)
            for g in set(gs):
                self.gram[g].add(name)
            if gs:
                self.first[gs[0]].append(n)
        self.sets[name] += 1

    def check(self, text: str):
        """-> {rule: set(names)} for a text (a sentence, a window or a document)."""
        hit = defaultdict(set)
        for sm in (False, True):
            n = norm(text, sm)
            if not n:
                continue
            parts = [norm(p, sm) for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
            for p in [n] + parts:
                if p in self.sent:
                    hit["R1_exact"] |= self.sent[p]
            tok = n.split()
            if len(tok) >= N:
                gs = grams(tok)
                joined = None
                for g in gs:
                    c = self.gram.get(g)
                    if c:
                        hit["R3_8gram"] |= c
                        for s in self.first.get(g, ()):
                            joined = joined or (" " + n + " ")
                            if (" " + s + " ") in joined:
                                hit["R2_contiguous"] |= self.sent[s]
                share = sum(1 for g in gs if g in self.gram) / len(gs)
                if share >= 0.5:
                    hit["R3_near_ge50pct"] |= {"*"}
        return hit


def load_protected(files, fields):
    P = Protected()
    for f in files:
        name = f.rsplit("/", 1)[-1]
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith("{"):
                    r = json.loads(line)
                    for k in fields:
                        if isinstance(r.get(k), str):
                            P.add(name, r[k])
                else:
                    P.add(name, line)
    return P


def selftest():
    P = Protected()
    P.add("flores_devtest", "The quick brown fox jumped over the lazy dog near the old river bank.")
    P.add("flores_devtest", "Ẹ káàbọ̀ sí Ìbàdàn, ìlú tí ó tóbi jùlọ ní ilẹ̀ Yorùbá lónìí.")
    assert "R1_exact" in P.check("the quick brown fox jumped over the lazy dog near the old river bank")
    assert "R2_contiguous" in P.check("Intro. The quick brown fox jumped over the lazy dog near the old river bank! More text here.")
    assert "R3_8gram" in P.check("They said the quick brown fox jumped over the lazy dog yesterday afternoon")
    assert "R1_exact" in P.check(unicodedata.normalize("NFD", "Ẹ káàbọ̀ sí Ìbàdàn, ìlú tí ó tóbi jùlọ ní ilẹ̀ Yorùbá lónìí."))
    assert "R1_exact" in P.check("Ẹ kaabo si Ibadan, ilu ti o tobi julo ni ile Yoruba lonii")      # diacritics-free variant
    assert not P.check("Completely unrelated text about something else entirely, nothing in common at all, really.")
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protected", nargs="*", default=[])
    ap.add_argument("--fields", nargs="*", default=["english", "target", "text", "src", "trg", "eng", "reference"])
    ap.add_argument("--input")
    ap.add_argument("--out")
    ap.add_argument("--report")
    ap.add_argument("--strict", action="store_true", help="drop on any shared 8-gram (default: exact, contiguous and at least 50 percent of the 8-grams)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    P = load_protected(a.protected, a.fields)
    rep = Counter(); kept = removed = 0
    with open(a.input, encoding="utf-8") as fi, open(a.out, "w", encoding="utf-8") as fo:
        for line in fi:
            r = json.loads(line)
            hit = defaultdict(set)
            for k in a.fields:
                if isinstance(r.get(k), str):
                    for rule, names in P.check(r[k]).items():
                        hit[rule] |= names
            drop = bool(hit.get("R1_exact") or hit.get("R2_contiguous") or hit.get("R3_near_ge50pct") or (a.strict and hit.get("R3_8gram")))
            for rule, names in hit.items():
                for nme in names:
                    rep[f"{rule}:{nme}"] += 1
            if drop:
                removed += 1
            else:
                kept += 1
                fo.write(line)
    out = {"kept": kept, "removed": removed, "by_rule_and_set": dict(rep), "protected_sentences": dict(P.sets)}
    if a.report:
        json.dump(out, open(a.report, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
