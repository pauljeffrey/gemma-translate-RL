"""Q2a: census of the 262,144 Gemma-3 tokens by Unicode script, and how many could be freed without touching the 77 target languages.
Output: reports/data/ge_ez_step0_2026-10-25/vocab_census.csv, vocab_free_scenarios.csv, vocab_token_class.npy (class label per id), free_ids_*.npy, vocab_census_examples.json

Token string = the tokenizer.json vocab key (SentencePiece style; U+2581 is the word-start marker and is ignored). Script of a character = first matching script word in its Unicode name
(Latin, Cyrillic, Ethiopic ...); digits, punctuation, symbols, spaces and emoji = Common; combining marks = Inherited (they travel with the neighbouring letter).
A token is assigned to the set of scripts of its letters: no letters -> 'common', one script -> that script (Latin is split into ASCII-only and non-ASCII Latin), several -> 'mixed'.
Classes outside the scripts: special (ids < 107 incl. <pad>/<bos>/<eos>/<unk>/turn tokens/<unusedN> low block), byte (<0xNN>), unused_tail (<unusedN> ids 256001-262143), html (the 45 markup tokens ids 168-220 minus unused).
"""
import os, sys, json, glob, re, unicodedata, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from ge_ez_step0_common import *

p = glob.glob(f"{CACHE}/models--google--gemma-3-1b-it/snapshots/*/tokenizer.json")[0]
j = json.load(open(p, encoding="utf-8"))
V = 262144
tok = [None] * V
for t, i in j["model"]["vocab"].items():
    tok[i] = t
assert all(t is not None for t in tok)
added = {t["id"]: t["content"] for t in j["added_tokens"]}

SCRIPTS = ["LATIN", "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "ETHIOPIC", "DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM", "SINHALA",
           "THAI", "LAO", "TIBETAN", "MYANMAR", "KHMER", "GEORGIAN", "ARMENIAN", "HANGUL", "HIRAGANA", "KATAKANA", "CJK", "BOPOMOFO", "COPTIC", "TIFINAGH", "NKO", "SYRIAC", "THAANA", "MONGOLIAN", "CHEROKEE", "RUNIC", "OGHAM"]
_c = {}
def script_of(ch):
    if ch in _c:
        return _c[ch]
    cat = unicodedata.category(ch)
    if ch == "▁" or cat[0] in "ZC" or cat[0] in "NPS":
        r = "common"
    elif cat == "Mn" or cat == "Me":
        r = "inherited"
    else:
        try:
            nm = unicodedata.name(ch)
        except ValueError:
            nm = ""
        r = "other"
        if "KATAKANA" in nm: r = "katakana"      # e.g. U+30FC prolonged sound mark (category Lm)
        for s in ([] if r != "other" else SCRIPTS):
            if s in nm.split() or (s == "CJK" and ("IDEOGRAPH" in nm or nm.startswith("CJK"))) or (s == "LATIN" and "LATIN" in nm):
                r = s.lower(); break
        if r == "other" and cat[0] == "L":
            r = "common"      # modifier letters (U+02BC glottal apostrophe etc.), letterlike symbols: needed by African orthographies, treated like punctuation
        elif r == "other":
            r = "common"
    _c[ch] = r
    return r

cls = np.empty(V, object)
for i, t in enumerate(tok):
    if t is None:
        cls[i] = "?"; continue
    if i < 107 and not (t.startswith("<unused") is False and i >= 6 and False):
        cls[i] = "special"; continue          # ids 0-106: pad, eos, bos, unk, mask, [multimodal], unused0-98, start/end_of_turn
    if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", t):
        cls[i] = "byte"; continue
    if t.startswith("<unused") and i >= 256001:
        cls[i] = "unused_tail"; continue
    if i in added and added[i].startswith("<") and added[i].endswith(">") and re.fullmatch(r"</?[A-Za-z0-9_]+>", added[i]):
        cls[i] = "html_markup"; continue
    letters = set(script_of(c) for c in t if script_of(c) not in ("common", "inherited"))
    if not letters:
        cls[i] = "common_ascii" if all(ord(c) < 128 or c == "▁" for c in t) else "common_nonascii"
    elif len(letters) == 1:
        s = next(iter(letters))
        if s == "latin":
            cls[i] = "latin_ascii" if all(ord(c) < 128 or c == "▁" for c in t) else "latin_nonascii"
        else:
            cls[i] = s
    else:
        cls[i] = "mixed:" + "+".join(sorted(letters))
np.save(os.path.join(OUT, "vocab_token_class.npy"), cls.astype(str))
cnt = collections.Counter(cls)
mixed = {k: v for k, v in cnt.items() if k.startswith("mixed:")}
rows = []
for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
    if k.startswith("mixed:"):
        continue
    ex = [tok[i] for i in np.where(cls == k)[0][:6]]
    rows.append(dict(cls=k, tokens=v, share=v / V, examples=" ".join(e.replace("\n", "\\n") for e in ex)))
rows.append(dict(cls="mixed (all combinations)", tokens=sum(mixed.values()), share=sum(mixed.values()) / V, examples=""))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "vocab_census.csv"), index=False, float_format="%.5f")
pd.Series(mixed).sort_values(ascending=False).to_csv(os.path.join(OUT, "vocab_census_mixed.csv"), header=["tokens"])
print(df.to_string())
print("mixed:", sorted(mixed.items(), key=lambda x: -x[1])[:15])

# --- freeable scenarios ---------------------------------------------------------------
def mixed_scripts(c):
    return set(c[6:].split("+")) if c.startswith("mixed:") else None
keepers_base = {"special", "byte", "common_ascii", "common_nonascii", "html_markup", "latin_ascii", "latin_nonascii", "ethiopic"}
def free_mask(keep_scripts, keep_latin_nonascii=True):
    """a token is KEPT if it is special/byte/common/markup/Latin/Ethiopic or in a kept script; mixed tokens are kept only if ALL their scripts are kept."""
    out = np.zeros(V, bool)
    ks = {s for s in keep_scripts} | {"latin", "ethiopic"}
    for i, c in enumerate(cls):
        if c in keepers_base or c in keep_scripts:
            continue
        ms = mixed_scripts(c)
        if ms is not None:
            if ms <= ks:
                continue
        out[i] = True   # unused_tail, other scripts, mixed with a non-kept script
    return out
sc = {
  "S1 strict: keep Latin + Ethiopic + common + byte + markup + specials (English/French/Ge'ez only)": free_mask(set()),
  "S2 + Arabic (Arabic-script source text, assumption)": free_mask({"arabic"}),
  "S3 + Arabic + Greek (math/science symbols as letters)": free_mask({"arabic", "greek"}),
  "S4 + Arabic + Greek + Cyrillic + Hebrew (names / loanwords in running text)": free_mask({"arabic", "greek", "cyrillic", "hebrew"}),
}
out = []
for k, m in sc.items():
    sub = collections.Counter(cls[m])
    out.append(dict(scenario=k, freeable=int(m.sum()), share_of_vocab=m.sum() / V, of_which_unused_tail=sub.get("unused_tail", 0),
                    of_which_other_scripts=int(m.sum() - sub.get("unused_tail", 0)), kept=int(V - m.sum())))
    np.save(os.path.join(OUT, "free_ids_" + k.split()[0] + ".npy"), np.where(m)[0].astype(np.int32))
fs = pd.DataFrame(out)
fs.to_csv(os.path.join(OUT, "vocab_free_scenarios.csv"), index=False, float_format="%.4f")
print(fs.to_string())

# compare with the VA keep list (159,629 tokens)
kl = glob.glob(os.path.join(ROOT, "translate_pipeline/vocab_ablation/results/keep/keep_lite.npy"))
if kl:
    keep = np.zeros(V, bool); keep[np.load(kl[0])] = True
    s1 = sc[list(sc)[0]]
    print("VA keep_lite size", keep.sum(), " my S1 kept", (~s1).sum(), " in VA keep but freed by S1:", int((keep & s1).sum()), " kept by S1 not in VA keep:", int((~s1 & ~keep).sum()))
    sub = collections.Counter(cls[~s1 & ~keep]); print("S1-kept but VA-dropped by class:", sub.most_common(8))
    sub = collections.Counter(cls[keep & s1]); print("VA-kept but S1-freed by class:", sub.most_common(8))
    json.dump(dict(va_keep=int(keep.sum()), s1_kept=int((~s1).sum()), va_keep_in_s1_free=int((keep & s1).sum()), s1_kept_not_va=int((~s1 & ~keep).sum()),
                   s1_kept_not_va_by_class=dict(collections.Counter(cls[~s1 & ~keep])), va_keep_s1_free_by_class=dict(collections.Counter(cls[keep & s1]))),
              open(os.path.join(OUT, "vocab_census_vs_va_keep.json"), "w"), indent=1)

# Ethiopic coverage: which Ethiopic characters of the Ge'ez Unicode block have a single-char token?
eth_chars = [chr(c) for c in range(0x1200, 0x1380) if unicodedata.category(chr(c))[0] == "L"]
single = {t.lstrip("▁") for t in tok if t and len(t.lstrip("▁")) == 1}
miss = [c for c in eth_chars if c not in single]
print("Ethiopic letters in U+1200-137F:", len(eth_chars), "with a single-char token:", len(eth_chars) - len(miss), "without:", len(miss))
json.dump(dict(ethiopic_letters=len(eth_chars), with_single_token=len(eth_chars) - len(miss), without=len(miss), missing_sample="".join(miss[:80])),
          open(os.path.join(OUT, "ethiopic_single_char_coverage.json"), "w"), ensure_ascii=False, indent=1)
