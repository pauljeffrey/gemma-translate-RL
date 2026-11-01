#!/usr/bin/env python3
"""Unit test: the hashed index in decon.py takes the same decisions as the reference class in gemma_mt/scripts/contam_audit_12_decontaminate.py.

Planted cases (exact, contiguous inside a window, 8-gram, NFD/under-dot variant, diacritic-free variant, quote variants, sentence part of a paragraph, short text, near duplicate,
negative controls) plus random text built from a shared vocabulary so that partial 8-gram overlaps happen. Run:  python test_decon.py   (or pytest)
"""
import importlib.util
import os
import random
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(__file__))
import decon as D  # noqa: E402

REF = os.path.join(os.path.dirname(__file__), "..", "..", "gemma_mt", "scripts", "contam_audit_12_decontaminate.py")
if not os.path.exists(REF):
    REF = os.environ.get("CONTAM_REF", "contam_audit_12_decontaminate.py")
spec = importlib.util.spec_from_file_location("ref12", REF)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

PROT = [
    ("flores_devtest", "The quick brown fox jumped over the lazy dog near the old river bank."),
    ("flores_devtest", "Ẹ káàbọ̀ sí Ìbàdàn, ìlú tí ó tóbi jùlọ ní ilẹ̀ Yorùbá lónìí."),
    ("ntrex", "The committee announced on Tuesday that the new regional health programme will start in March next year."),
    ("ntrex", "Short one here."),
    ("smol", "“Don’t worry,” she said, “we will find the missing goat before the rain begins tonight.”"),
    ("mafand_test", "Ọmọ náà ń ṣeré ní àgbàlá ilé baba rẹ̀ nígbà tí òjò bẹ̀rẹ̀ sí í rọ̀ lójijì."),
    ("sib200", "Ndị ọrụ ahụ na-arụ ọrụ n'ime ụlọ ọrụ ahụ ụbọchị niile n'afọ a."),
    ("flores_dev", "A very long protected sentence that contains many distinct words alpha beta gamma delta epsilon zeta eta theta iota kappa lambda."),
]
VOC = "the of and to in a is that for it as was with be by on not he this are or his from at which but have an had they you were their one all we can her has there been if more when will would who so no".split()
PROT_RANDOM = [" ".join(random.Random(i).choices(VOC, k=random.Random(i).randint(3, 22))) for i in range(300)]


def build():
    B = D.Builder()
    P = R.Protected()
    for n, t in PROT + [("random_vocab", t) for t in PROT_RANDOM]:
        B.add(n, t)
        P.add(n, t)
    return B.finish(), P


def ref_masks(P, ix, text):
    h = P.check(text)
    bit = {n: 1 << i for i, n in enumerate(ix.set_names)}
    m = lambda rule: sum(bit[n] for n in h.get(rule, ()) if n in bit)
    return m("R1_exact"), m("R2_contiguous"), m("R3_8gram"), "R3_near_ge50pct" in h


def planted():
    nfd = unicodedata.normalize("NFD", "Ẹ káàbọ̀ sí Ìbàdàn, ìlú tí ó tóbi jùlọ ní ilẹ̀ Yorùbá lónìí.")
    return [
        "the quick brown fox jumped over the lazy dog near the old river bank",                                  # R1
        "Intro. The quick brown fox jumped over the lazy dog near the old river bank! More text here.",           # part + R2
        "They said the quick brown fox jumped over the lazy dog yesterday afternoon",                              # R3
        nfd,                                                                                                       # NFD
        "Ẹ kaabo si Ibadan, ilu ti o tobi julo ni ile Yoruba lonii",                                              # marks stripped
        "Don't worry, she said, we will find the missing goat before the rain begins tonight",                      # quote folding
        "“Don’t worry,” she said, “we will find the missing goat before the rain begins tonight.”",
        "Short one here.", "short one here", "x short one here x",                                                  # short texts: exact only
        "Completely unrelated text about something else entirely, nothing in common at all, really.",
        "",
        "The committee announced on Tuesday that the new regional health programme will start in March next year. Then something else happened later today in the market.",
        "Nobody saw that the committee announced on Tuesday that the new regional health programme will start in",   # partial 8-gram
        "Ọmọ náà ń ṣeré ní àgbàlá ilé baba rẹ̀ nígbà tí òjò bẹ̀rẹ̀ sí í rọ̀ lójijì",
        unicodedata.normalize("NFD", "Ọmọ náà ń ṣeré ní àgbàlá ilé baba rẹ̀ nígbà tí òjò bẹ̀rẹ̀ sí í rọ̀ lójijì."),
        "Ndị ọrụ ahụ na-arụ ọrụ n'ime ụlọ ọrụ ahụ ụbọchị niile n'afọ a.",
        "Ndi oru ahu na-aru oru n'ime ulo oru ahu ubochi niile n'afo a",
        "A very long protected sentence that contains many distinct words alpha beta gamma delta epsilon zeta eta theta iota kappa lambda",
        "prefix words here A very long protected sentence that contains many distinct words alpha beta gamma delta epsilon zeta eta theta iota kappa lambda suffix words",
        "A very long protected sentence that contains many distinct words alpha beta gamma delta",                    # shares 8-grams, not R2
        "line one\nThe quick brown fox jumped over the lazy dog near the old river bank.\nline three",
        "ＴＨＥ ＱＵＩＣＫ brown fox jumped over the lazy dog near the old river bank",                                # NFKC fullwidth
    ]


def randoms(n=600, seed=7):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        k = rnd.randint(2, 40)
        w = rnd.choices(VOC, k=k)
        if rnd.random() < 0.3 and PROT_RANDOM:                       # splice a protected fragment in
            frag = rnd.choice(PROT_RANDOM).split()
            a = rnd.randint(0, len(frag) // 2)
            w[rnd.randint(0, len(w)):] = frag[a:] + w[:2]
        s = " ".join(w)
        if rnd.random() < 0.3:
            s = s.capitalize() + rnd.choice([".", "!", "?", ".\n", ". "]) + " " + " ".join(rnd.choices(VOC, k=5))
        out.append(s)
    return out


def test_same_decisions():
    ix, P = build()
    texts = planted() + randoms()
    got = ix.check_texts(texts)
    bad = []
    for t, g in zip(texts, got):
        r = ref_masks(P, ix, t)
        drop_r = bool(r[0] or r[1] or r[2] or r[3])
        drop_g = bool(g[0] or g[1] or g[2] or g[3])
        if g != r or drop_r != drop_g:
            bad.append((t[:80], g, r))
    assert not bad, bad[:5]
    n_hit = sum(1 for g in got if g[0] or g[1] or g[2])
    assert n_hit > 100, n_hit            # the test really contains hits
    assert any(g[0] for g in got) and any(g[1] for g in got) and any(g[2] for g in got) and any(g[3] for g in got)
    print(f"ok: {len(texts)} texts, {n_hit} with a hit, identical rule masks and identical drop decisions")


def test_planted_hits():
    ix, _ = build()
    out = dict(zip(planted(), ix.check_texts(planted())))
    t = planted()
    assert out[t[0]][0]
    assert out[t[1]][1] and out[t[1]][0]
    assert out[t[2]][2] and not out[t[2]][0]
    assert out[t[3]][0] and out[t[4]][0]
    assert out[t[5]][0] and out[t[6]][0]
    assert out[t[8]][0] and out[t[9]] == (0, 0, 0, False)         # short exact only, embedded short text does not match
    assert out[t[10]] == (0, 0, 0, False) and out[t[11]] == (0, 0, 0, False)
    assert out[t[12]][1] and out[t[12]][2]                       # protected sentence embedded in a longer text
    assert out[t[13]][2] and not out[t[13]][1]                   # shared 8-gram only
    assert out[t[15]][0] and out[t[17]][0]                       # NFD / mark-stripped copies
    assert out[t[18]][0] and out[t[19]][1] and not out[t[19]][0]
    assert out[t[20]][2] and not out[t[20]][1]
    assert out[t[21]][0] and out[t[22]][2]


def test_save_load(tmp_path=None):
    import tempfile
    ix, _ = build()
    p = os.path.join(tempfile.mkdtemp(), "ix.npz")
    ix.save(p)
    ix2 = D.Index.load(p)
    texts = planted()
    assert ix.check_texts(texts) == ix2.check_texts(texts)
    assert ix.digest() == ix2.digest()


def test_exemption():
    a = "The quick brown fox jumped over the lazy dog near the old river bank today in spring."
    b = "A completely different protected sentence about harbours, ships and the long autumn storms of the north."
    B = D.Builder(); B.add("s", a); B.add("s", b); ix = B.finish()
    E = D.Builder(); E.add("exempt", a); ix.exempt = E.finish()
    r = ix.check_texts([a, b, a + " " + b, "Intro. " + a, "nothing in common here at all, just some other words entirely"])
    assert r[0] == (0, 0, 0, False) and r[3] == (0, 0, 0, False)
    assert r[1][0] and r[1][2] and r[2][2] and not r[4][2]
    ix.exempt = None
    assert ix.check_texts([a])[0][0]


if __name__ == "__main__":
    test_exemption()
    test_planted_hits()
    test_same_decisions()
    test_save_load()
    print("all tests passed")
