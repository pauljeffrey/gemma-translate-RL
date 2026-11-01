# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline tests of the HARD / DOMAIN split library (gemma_mt/eval/longdoc_hard.py): formulas, cleaning, number / unit / date preservation, protected tokens, formula
preservation, term consistency, the modality audit list and `evaluate_hard`.  No model, no network."""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from gemma_mt.eval import longdoc as L  # noqa: E402
from gemma_mt.eval import longdoc_hard as H  # noqa: E402


def test_find_and_mask_formulas():
    t = "Let $x^2+1$ be positive. Then $$\\int_0^1 f(x)\\,dx = 3$$ and \\(a_n\\) holds. It costs $5 and $10 each."
    fs = [s for _, _, s in H.find_formulas(t)]
    assert fs == ["$x^2+1$", "$$\\int_0^1 f(x)\\,dx = 3$$", "\\(a_n\\)"]            # prices are not formulas
    assert "x^2" not in H.mask_formulas(t) and "costs $5" in H.mask_formulas(t)
    assert H.formula_core("$ x^2 $") == H.formula_core("\\(x^2\\)") == H.formula_core("$$x^2$$") == "x^2"


def test_clean_removes_urls_emails_phones_keeps_formulas_numbers_citations():
    p = "See https://example.org/a?b=1 or mail me@x.org or call 555-123-4567: dose 5 mg [12], $x_1$ (Smith et al., 2019) in Section 3.2."
    c = H.clean_hard_paragraph(p)
    assert "http" not in c and "@" not in c and "555" not in c
    assert "5 mg" in c and "[12]" in c and "$x_1$" in c and "(Smith et al., 2019)" in c and "Section 3.2" in c


def test_paragraph_guards():
    good = "The patients received 5 mg of amoxicillin twice daily for 7 days, and the dose was reduced when $\\mathrm{eGFR} < 30$ in the second week of treatment."
    assert H.paragraph_ok_hard(good)[0]
    assert not H.paragraph_ok_hard("Smith J, Jones K. Title of a paper. J Med 2019;12:33-40. doi:10.1000/xyz ISBN 123 pp. 5-9.")[0]
    assert not H.paragraph_ok_hard("THIS IS A SHOUTING HEADING THAT GOES ON FOR MANY WORDS IN CAPITALS ONLY.")[0]
    assert not H.paragraph_ok_hard("a | b | c | d | e | f | g | h | i | j | k.")[0]
    assert not H.paragraph_ok_hard("The study was done \\cite{abc} with the usual methods and results in the third year.")[0]
    ok, why = H.paragraph_ok_hard("This sentence has no ending punctuation and goes on for a while but stops")
    assert not ok and why == "unterminated"


def test_digit_scripts_and_separators():
    assert H.to_ascii_digits("١٢٣ १२") == "123 12"
    s = H.extract_numbers("There were 1,234.5 cases and 12 deaths; the dose is 3.5 mg.")
    assert [x[0] for x in s] == ["1234.5", "12", "3.5"]
    h = H.extract_numbers("Kulikuwa na visa 1.234,5 na vifo ١٢; kipimo ni 3,5 mg.", source=False)
    m, un_s, un_t = H._multiset_match([x[0] for x in s], [x[1] for x in h])
    assert m == 3 and not un_s and not un_t
    # thin-space thousands
    assert H.number_variants("1 234", False)[0] == "1234"


def test_number_metrics_perfect_missing_extra():
    src = "In 2019 the trial enrolled 1,200 patients; 45.5% improved, 3 died, the dose was 5 mg."
    ok = H.number_metrics(src, "Mnamo 2019 jaribio lilisajili wagonjwa 1.200; 45,5% waliboreka, 3 walikufa, dozi ilikuwa 5 mg.")
    assert ok["num_f1"] == 1.0 and ok["any_error_strict"] is False and ok["year_recall"] == 1.0
    assert ok["pair_recall"] == 1.0                                            # 45.5% and 5 mg
    bad = H.number_metrics(src, "In 2018 the trial enrolled 1,200 patients; 45.5% improved, three died, the dose was 50 mg.")
    assert bad["any_error_strict"] and bad["num_recall"] < 1.0 and bad["year_recall"] == 0.0
    assert bad["pair_recall"] < 1.0
    # a spelled-out small integer is an error in the strict variant only
    w = H.number_metrics("Three of 12 patients had 2 events in 2020.", "Watatu kati ya wagonjwa 12 walikuwa na matukio mawili mnamo 2020.")
    assert w["any_error_strict"] and w["any_error_lenient"] is False and w["missing_small_int_share"] == 1.0


def test_protected_tokens_and_recall():
    src = "The WHO recommends imatinib for BRCA1 carriers; see Section 5(1)(a) of Regulation 2016/635 [12]. Plasmodium falciparum causes malaria (Smith et al., 2019)."
    toks = {t for _, t in H.protected_tokens(src)}
    assert {"WHO", "BRCA1", "5(1)(a)", "2016/635", "[12]", "imatinib", "Plasmodium falciparum", "(Smith et al., 2019)"} <= toks
    full = H.protected_metrics(src, "WHO inapendekeza imatinib kwa BRCA1; tazama Sehemu 5(1)(a) ya Kanuni 2016/635 [12]. Plasmodium falciparum husababisha malaria (Smith et al., 2019).")
    assert full["protected_recall"] == 1.0
    lost = H.protected_metrics(src, "Shirika linapendekeza dawa kwa wabebaji; tazama Sehemu 5 ya kanuni.")
    assert lost["protected_recall"] < 0.4 and lost["protected_lost"]
    # non-Latin script: names are not scored, numbers / brackets still are
    am = H.protected_metrics(src, "ዓለም ጠናዊ ዕፉት በእንግሊዝኛ [12] 5(1)(a) 2016/635 ዓለም ጠናዊ ዕፉት")
    assert am["protected_latin_target"] is False and "drug" not in am["protected_by_class"]


def test_formula_metrics():
    src = "Let $a+b=c$ and $$x^2$$ and \\(y_1\\)."
    same = H.formula_metrics(src, "Acha $a + b = c$ na $$x^2$$ na $y_1$.")                   # spacing and delimiters may change
    assert same["formula_exact_share"] == 1.0 and same["formula_ordered_share"] == 1.0 and same["formula_edit_distance"] == 0
    swapped = H.formula_metrics(src, "$$x^2$$ $a+b=c$ $y_1$")
    assert swapped["formula_exact_share"] == 1.0 and swapped["formula_ordered_share"] < 1.0 and swapped["formula_edit_distance"] > 0
    broken = H.formula_metrics(src, "$a+b=d$ $x^2$")
    assert broken["formula_exact_share"] < 1.0 and broken["formula_extra"] >= 1
    assert H.formula_metrics("no formula here", "x")["formula_exact_share"] is None
    masked, spans = H.mask_formulas_with_ids(src)
    assert "[[F1]]" in masked and H.unmask_formulas(masked, spans) == src


def _doc_sents():
    return ["The kidney filters blood and removes waste.", "Dialysis replaces kidney function when the kidney fails.", "Patients with kidney disease need regular checks.",
            "The heart pumps blood through the body.", "A healthy diet protects the heart.", "Doctors must not stop the drug without advice.", "Kidney stones cause pain.",
            "Water helps the kidney.", "Exercise is good.", "The clinic opens on Monday."]


def test_term_consistency_detects_drift():
    src = _doc_sents()
    kid = [i for i, x in enumerate(src) if "kidney" in x.lower()]
    assert len(kid) >= 4
    consistent = [(f"figo m{i} q{i}" if i in kid else f"other{i} n{i} z{i}") for i in range(len(src))]
    drift = [((f"figo m{i} q{i}" if j % 2 == 0 else f"renal m{i} q{i}") if i in kid else f"other{i} n{i} z{i}") for j, i in enumerate(range(len(src)))]
    ok = H.term_consistency(src, consistent)
    dr = H.term_consistency(src, drift)
    assert ok["n_terms_determined"] >= 1 and ok["term_consistency"] == 1.0
    assert dr["term_consistency"] is not None and dr["term_consistency"] < 0.9
    assert any(r["term"].startswith("kidney") and r["rendering"] == "figo" for r in ok["terms"])
    assert H.term_consistency(src[:3], src[:3])["term_consistency"] is None                 # too short


def test_modality_flags_learned_lexicon_and_short_fallback():
    cues = ["He must not smoke.", "She could not go.", "They did not stay.", "It does not matter.", "You may leave now.", "We never agreed.", "No one came."]
    plain = ["The dog ran.", "The sun rose.", "Rain fell today.", "The road is long.", "Birds sing at dawn.", "Tea is hot.", "Maria likes music.", "The train was late."]
    src = [c[:-1] + f" item{i}." for i in range(8) for c in cues] + [c[:-1] + f" item{i}." for i in range(8) for c in plain] * 2
    # a toy language in which 'not' is rendered by the word 'hapana' (reference), everything else by filler words
    ref = []
    for s in src:
        ref.append(("hapana " if "not" in s or s.startswith("No ") or "never" in s else "") + " ".join(f"w{sum(map(ord, w)) % 97}" for w in s.split()))
    lex = H.learn_cue_lexicon(src, ref)
    assert "not" in lex and lex["not"][0] == "hapan"
    bead_s = ["He must not smoke.", "The dog ran."]
    bead_r = ["hapana a b c", "x y z"]
    bead_t_bad = ["a b c d e f g", "x y z"]
    fl = H.modality_flags(bead_s, bead_t_bad, bead_r, lex)
    assert len(fl) == 1 and fl[0]["reason"] == "cue_missing" and fl[0]["bead"] == 0
    assert H.modality_flags(bead_s, ["hapana a b c", "x y z"], bead_r, lex) == []
    short = H.modality_flags(["You may leave."], ["ab"], ["a long reference translation here"], {})
    assert short and short[0]["reason"] == "short_vs_ref"
    assert H.modality_flags(["You may leave."], ["You may leave."], ["zzz"], {})[0]["reason"] == "copied"


def test_evaluate_hard_end_to_end_and_config(tmp_path):
    src = ["In 2019, 1,200 patients received 5 mg of imatinib [3].", "The effect size was $d=0.8$ in the WHO trial.", "The dose must not be doubled.",
           "Mortality fell by 12% after 6 months.", "See Section 4(2) of Regulation 2016/635.", "No adverse events were reported.",
           "The kidney filters blood.", "Dialysis helps the kidney.", "The kidney is vital."]
    ref = [s.replace("patients", "wagonjwa") for s in src]
    d = L.RefDoc("hard:1", "swh", " ".join(src), src, ref, ref_kind="pseudo", ref_system="x", domain="medicine", bin=512)
    d2 = L.RefDoc("hard:2", "swh", " ".join(src), src, ref, ref_kind="pseudo", ref_system="x", domain="law", bin=512)
    good = L.Hyp("hard:1", "swh", " ".join(ref))
    bad = L.Hyp("hard:2", "swh", " ".join(ref).replace("1,200", "1,000").replace("$d=0.8$", "$d=0.9$"))
    res = H.evaluate_hard([d, d2], {"hard:1": good, "hard:2": bad}, iso="swh", with_core=False, n_boot=50)
    assert res["n_docs"] == 2 and len(res["per_doc"]) == 2
    r1, r2 = res["per_doc"]
    assert r1["num_f1"] == 1.0 and r1["formula_exact_share"] == 1.0 and r1["protected_recall"] == 1.0
    assert r2["num_f1"] < 1.0 and r2["formula_exact_share"] == 0.0
    assert res["rates"]["any_error_strict"]["value"] == 0.5
    assert 0.0 < res["metrics"]["num_f1"]["value"] < 1.0 and set(res["by_domain"]) == {"medicine", "law"}
    p = tmp_path / "metric_config.json"
    H.save_config(str(p))
    cfg = json.loads(p.read_text(encoding="utf-8"))
    assert "acronym" in cfg["protected"]["classes"] and cfg["term_consistency"]["min_occurrences"] == 3
    # with the core metrics of longdoc.evaluate_language
    full = H.evaluate_hard([d, d2], {"hard:1": good, "hard:2": bad}, iso="swh", with_core=True, n_boot=20)
    assert "chrf" in full["core"] and full["core"]["n_docs"] == 2


def test_candidate_helpers_and_selection_shapes():
    import importlib.util
    script = os.path.join(os.path.dirname(HERE), "scripts", "longdoc_build_hard.py")
    spec = importlib.util.spec_from_file_location("longdoc_build_hard", script)
    B = importlib.util.module_from_spec(spec)
    sys.modules["longdoc_build_hard"] = B
    spec.loader.exec_module(B)
    quotas = B.hard_quotas(150, H.DOMAINS)
    assert sum(quotas.values()) == 150 and all(v > 0 for v in quotas.values())
    assert {b for (_, b) in quotas} == {512, 1024, 2048, 4096}
    assert all(sum(v for (d, _), v in quotas.items() if d == dom) == 50 for dom in H.DOMAINS)
    assert B.domain_of_pool_row({"source": "pmc", "domain": "health"}) == ("medicine", "pmc")
    assert B.domain_of_pool_row({"source": "federalregister", "domain": "law_government"})[0] == "law"
    assert B.domain_of_pool_row({"source": "cp_books", "domain": "health", "_pool_file": "cp_pressbooks__permissive"}) == ("medicine", "cp_pressbooks")
    assert B.domain_of_pool_row({"source": "cp_books", "domain": "science_explainer", "_pool_file": "cp_libretexts__permissive"}) == ("math_science", "cp_libretexts")
    assert B.domain_of_pool_row({"source": "nasa", "domain": "science_explainer"})[0] is None            # NASA belongs to the core


def _load_script(name):
    import importlib.util
    path = os.path.join(os.path.dirname(HERE), "scripts", name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
    spec.loader.exec_module(mod)
    return mod


def test_fetcher_parsers_wikipedia_arxiv_display_merge():
    FH = _load_script("longdoc_fetch_hard")
    ex = ("Intro text here.\n\nDefinition\nThe transform of a function \n  \n    \n      \n        f\n        (\n        x\n        )\n      \n    \n    {\\displaystyle f(x)}\n  \n"
          " on the real line, is ⁠\n  \n    \n        g\n    \n    {\\displaystyle {\\widehat {f}}(\\xi )}\n  \n⁠, defined by the integral.\n\n\n    {\\displaystyle \\int f}\n\nNext paragraph.")
    ps = FH.wp_extract_to_paras(ex)
    assert ps[2].startswith("The transform of a function $f(x)$ on the real line, is ${\\widehat {f}}(\\xi )$") and ps[2].endswith("defined by the integral.")
    assert ps[3] == "$$\\int f$$" and "spill" not in " ".join(ps) and "\n" not in ps[2]
    merged = FH.merge_display(["We prove that", "$$a^2+b^2=c^2$$", "where $a$ is a leg of the triangle."])
    assert merged == ["We prove that $$a^2+b^2=c^2$$ where $a$ is a leg of the triangle."]
    md = ("Title of the paper\n\nAuthor One, Author Two\n\n###### Abstract\n\nWe study $x^2$ and show a result in this paper.\n\n# 1 Introduction\n\n"
          "| a | b |\n|---|---|\n| 1 | 2 |\n\nWe now proceed to the main text of the paper in this section.\n\n# References\n\n[1] A. Author. Title.")
    bl = FH.arxiv_blocks(md)
    assert bl == ["We study $x^2$ and show a result in this paper.", "We now proceed to the main text of the paper in this section."]
