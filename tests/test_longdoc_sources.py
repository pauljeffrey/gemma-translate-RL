# SPDX-License-Identifier: AGPL-3.0-or-later
"""Long-document benchmark source pool (2026-10-20): the parsers / cleaners of `scripts/longdoc_fetch_sources.py` on tiny inline fixtures.
CPU only, no network (the fetchers themselves are exercised on the rental, not here)."""
import bz2
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "scripts", "longdoc_fetch_sources.py")
_spec = importlib.util.spec_from_file_location("longdoc_fetch_sources", SCRIPT)
L = importlib.util.module_from_spec(_spec)
sys.modules["longdoc_fetch_sources"] = L
_spec.loader.exec_module(L)

lxml = pytest.importorskip("lxml")

P1 = ("The committee met on Tuesday to discuss the new rules for water supply in rural districts, and the members agreed that the plan "
      "should be reviewed again before the end of the year.")
P2 = ("Farmers in the region said that the changes would help them to plant earlier, because the rains have been arriving later than they "
      "used to in the past, and they hope that more support will follow.")
P3 = ("In the second half of the meeting the delegates looked at the budget, which had been prepared by the ministry together with several "
      "local organisations that work with schools and clinics.")
PARAS = [P1, P2, P3]


def _big():
    """six distinct paragraphs, ~450 estimated tokens (a document of the same paragraph twice would now be de-duplicated)."""
    return [P1 * 2, P2 * 2, P3 * 2, P1 + " " + P2, P2 + " " + P3, P3 + " " + P1]


# ------------------------------------------------------------------------------------------------------------------ text cleaning
def test_clean_paragraph_normalises_unicode_space_and_citations():
    s = "Café  of the​ town  [1] was   open [citation needed] today ."
    out = L.clean_paragraph(s)
    assert out == "Café of the town was open today."
    assert L.clean_paragraph("a\nb\n\nc") == "a b c"


def test_strip_urls_drops_parenthesised_urls_and_rejects_running_urls():
    assert L.strip_urls("The report (https://example.org/report) was published in May.") == "The report was published in May."
    assert L.strip_urls("See www.example.org for the full text of the report today.") is None
    assert L.strip_urls("Write to someone@example.org to get a copy of the document.") is None
    assert L.strip_urls("No link here at all, only plain words in a sentence.") == "No link here at all, only plain words in a sentence."


def test_keep_paragraph_rules():
    assert L.keep_paragraph(P1)
    assert not L.keep_paragraph("Read more about this story on our website today")       # boilerplate
    assert not L.keep_paragraph("CHAPTER ONE")                                           # heading-like, too short
    assert not L.keep_paragraph("Introduction to the history of the town")               # no sentence end, capitalised, short
    assert not L.keep_paragraph("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15")                    # digits
    assert not L.keep_paragraph("x = y + z where the values of the terms are given in the table below. [1] [2] {3} |4| ^5 ~6")
    assert not L.keep_paragraph("Smith J, Jones K, et al. Journal of Tropical Medicine, 2014, with the results of the trial.")  # reference
    assert L.keep_paragraph("Wash your hands with soap and clean water before you eat and after you use the toilet.")


def test_english_ok():
    assert L.english_ok(" ".join(PARAS) * 2)
    fr = ("Le comité s'est réuni mardi pour discuter des nouvelles règles sur l'approvisionnement en eau dans les districts ruraux et les membres "
          "ont convenu que le plan devrait être examiné de nouveau avant la fin de l'année. ") * 3
    assert not L.english_ok(fr)


def test_fingerprint_ignores_case_and_punctuation():
    assert L.fingerprint("Hello, World!") == L.fingerprint("hello world")
    assert L.fingerprint("a b") != L.fingerprint("a c")


# ---------------------------------------------------------------------------------------------------------------------------- html
def test_html_paragraphs_skips_headings_tables_figures_and_footnote_refs():
    html = ("<div><h2>Heading</h2><p>%s<sup class='reference'><a href='#x'>[1]</a></sup></p><table><tr><td>cell</td></tr></table>"
            "<figure><figcaption>Photo: someone</figcaption></figure><ul><li>%s</li></ul><div class='caption'>A caption text</div></div>" % (P1, P2))
    ps = L.html_paragraphs(L.parse_html(html), drop_classes=("caption",))
    assert [L.clean_paragraph(p) for p in ps] == [P1, P2]


def test_html_paragraphs_inline_markup_stays_in_one_paragraph():
    ps = L.html_paragraphs(L.parse_html("<p>The <em>quick</em> brown <a href='x'>fox</a> jumps<br>over the lazy dog.</p>"))
    assert [L.clean_paragraph(p) for p in ps] == ["The quick brown fox jumps over the lazy dog."]


def test_cut_to_tokens_cuts_at_paragraph_boundary():
    tc = L.TokenCounter("/nonexistent/tokenizer.json")
    paras = [("word " * 400).strip() + "." for _ in range(6)]
    kept, truncated = L.cut_to_tokens(paras, tc, max_tokens=1000)
    assert truncated and 0 < len(kept) < 6
    kept, truncated = L.cut_to_tokens(paras[:1], tc, max_tokens=10 ** 6)
    assert not truncated and kept == paras[:1]


# ---------------------------------------------------------------------------------------------------------------------------- pool
def _doc(i, paras, **kw):
    d = dict(id=f"t:{i}", domain="news", licence="CC-BY-4.0", licence_class="permissive", licence_evidence="https://x.example/licence",
             url=f"https://x.example/{i}", title=f"T{i}", attribution="A", date="2026-01-01", paragraphs=list(paras))
    d.update(kw)
    return d


def test_pool_filters_dedups_resumes_and_renames_atomically(tmp_path):
    tc = L.TokenCounter("/nonexistent/tokenizer.json")
    out = str(tmp_path / "p" / "t__permissive.jsonl")
    big = _big()                           # ~ 450 estimated tokens
    pool = L.Pool(out, "t", tc, max_docs=10)
    assert pool.add(_doc(1, big))
    assert not pool.add(_doc(1, big))                             # same id
    assert not pool.add(_doc(2, big))                             # same text, other id -> duplicate
    assert not pool.add(_doc(3, PARAS[:2]))                       # fewer than 3 paragraphs
    assert not pool.add(_doc(4, [P1, P2, P3]))                    # < 400 tokens
    assert pool.n == 1 and pool.stats["duplicate"] == 1 and pool.stats["too_short"] == 1
    assert os.path.exists(out + ".part") and not os.path.exists(out)
    pool.close(complete=False)
    pool2 = L.Pool(out, "t", tc, max_docs=10)                      # resume: ids and hashes are read back from the .part file
    assert pool2.seen("t:1") and pool2.n == 1
    assert not pool2.add(_doc(5, big))                             # duplicate of the first document
    pool2.close(complete=True)
    assert os.path.exists(out) and not os.path.exists(out + ".part")
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(rows) == 1
    r = rows[0]
    assert set(["id", "source", "domain", "licence", "licence_class", "licence_evidence", "url", "title", "attribution", "date", "text", "n_chars",
                "n_gemma_tokens"]) <= set(r)
    assert r["n_chars"] == len(r["text"]) and "\n\n" in r["text"]


def test_pool_snapshot_is_complete_json_lines(tmp_path):
    tc = L.TokenCounter("/nonexistent/tokenizer.json")
    out = str(tmp_path / "s.jsonl")
    pool = L.Pool(out, "t", tc, max_docs=5)
    pool.publish_every = 1
    big = _big()
    assert pool.add(_doc(1, big))
    assert [json.loads(l)["id"] for l in open(out, encoding="utf-8")] == ["t:1"]      # visible before close()
    pool.close(complete=False)


def test_pool_stats_counts_bins_and_classes(tmp_path):
    d = tmp_path / "pool"
    d.mkdir()
    rows = []
    for i, t in enumerate([500, 900, 2000, 5000, 5100]):
        rows.append({"id": f"a:{i}", "source": "a", "domain": "news", "licence": "L", "licence_class": "pd", "n_gemma_tokens": t, "text": "x" * 300})
    (d / "a__pd.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    st = L.pool_stats(str(d), str(tmp_path / "samples"))
    f = st["files"]["a__pd.jsonl"]
    assert f["docs"] == 5 and f["docs_per_bin"] == {"400-800": 1, "800-1600": 1, "1600-3200": 1, ">=3200": 2}
    assert st["domain_bin_class"]["news|pd"][">=3200"] == 2
    assert os.path.exists(tmp_path / "samples" / "a.jsonl")


def test_registry_is_consistent():
    assert {"gutenberg", "govuk", "wikinews", "wikivoyage", "wikipedia", "openstax", "globalvoices", "owid", "nasa", "fedhealth", "legislation",
            "federalregister", "pmc"} <= set(L.SOURCES)
    for name, m in L.SOURCES.items():
        assert m["cls"] in L.LICENCE_CLASSES, name
        assert m["evidence"] and m["access"], name
        assert callable(m["fn"])
    ap = L.build_parser()
    a = ap.parse_args(["govuk", "--out", "x.jsonl", "--max-docs", "5"])
    assert a.max_docs == 5 and a.source == "govuk"


# --------------------------------------------------------------------------------------------------------------------- Gutenberg
PG_TEXT = """The Project Gutenberg eBook of Some Tale

Title: Some Tale

*** START OF THE PROJECT GUTENBERG EBOOK SOME TALE ***

CHAPTER I

It was a bright cold day in April, and the clocks were striking thirteen, and the old
man walked slowly down the road towards the village where he had lived as a boy.
[Illustration: the road]

He had not seen the place for forty years, and he found that the little church had been
replaced by a school, which stood where the churchyard used to be.



CHAPTER II

The next morning he went to see the schoolmaster, who was a young man with a loud voice
and a habit of walking up and down the room while he talked about the children.

A line of verse
that is very short
and goes on and on
like a song

*** END OF THE PROJECT GUTENBERG EBOOK SOME TALE ***
This is the license text and should never be seen.
"""


def test_gutenberg_paragraphs_strip_header_footer_and_mark_headings():
    ps = L.gutenberg_paragraphs(PG_TEXT)
    texts = [t for t, h in ps]
    assert not any("license text" in t or "Project Gutenberg" in t for t in texts)
    assert ("CHAPTER I", True) in ps and ("CHAPTER II", True) in ps
    body = [t for t, h in ps if not h]
    assert body[0].startswith("It was a bright cold day") and "\n" not in body[0] and "Illustration" not in body[0]
    assert not any("line of verse" in t for t in body)          # short-line verse is dropped


def test_gutenberg_segments_split_at_headings_and_merge_short_ones():
    tc = L.TokenCounter("/nonexistent/tokenizer.json")
    paras = [("CHAPTER I", True)] + [(P1, False), (P2, False), (P3, False)] * 6 + [("CHAPTER II", True), (P1, False), (P2, False), (P3, False)]
    segs = L.gutenberg_segments(paras, tc, min_tokens=400)
    assert len(segs) == 1 and segs[0][0] == "CHAPTER I" and len(segs[0][1]) == 18      # the short second chapter has no successor: dropped


def test_gutenberg_catalog_helpers():
    assert L.pg_death_year("Doyle, Arthur Conan, 1859-1930") == 1930
    assert L.pg_death_year("Doe, John, 1850-") is None
    assert L.pg_death_year("A, B, 1800-1850; C, D, 1801-1960") == 1960
    assert L.pg_domain("Fairy tales; Children's stories", "PZ", "") == "stories"
    assert L.pg_domain("Africa -- Description and travel", "DT", "") == "travel"
    assert L.pg_domain("Hygiene", "RA", "") == "health"
    assert L.pg_domain("Poetry", "PR", "") is None
    csv_text = ("Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves\n"
                '1,Text,2000-01-01,Old Tale,en,"Old, Author, 1800-1870",Fairy tales,PZ,\n'
                '2,Text,2000-01-01,Young Tale,en,"New, Author, 1900-1999",Fairy tales,PZ,\n'
                '3,Text,2000-01-01,Fr Tale,fr,"Old, Author, 1800-1870",Fairy tales,PZ,\n')
    c = L.pg_candidates(csv_text)
    assert [b["id"] for b in c["stories"]] == ["1"]


# ----------------------------------------------------------------------------------------------------------------------- GOV.UK
def test_govuk_content_paragraphs_use_body_and_parts():
    c = {"details": {"body": "<div class='govspeak'><h2>H</h2><p>%s</p><div class='call-to-action'><p>%s</p></div></div>" % (P1, P2),
                     "parts": [{"title": "One", "body": "<p>%s</p>" % P3}]}}
    ps = [L.clean_paragraph(p) for p in L.govuk_content_paragraphs(c)]
    assert ps == [P1, P3]


# ------------------------------------------------------------------------------------------------------------------- MediaWiki
WT = """{{date|October 3, 2012}}{{publish}}{{Japan}}
'''TOKYO''' -- The [[Japan|Japanese]] government said on {{w|Tuesday}} that the plan, first announced by the [[Cabinet (government)|cabinet]]<ref>Reuters</ref>
last year, will go ahead.<!-- hidden comment -->

== Background ==
The {{w|Han class submarine|Han-class}} vessel was seen near the coast by [http://example.org local fishermen] earlier this week, officials said.
[[File:Boat.jpg|thumb|A boat]]
{| class="wikitable"
| cell
|}
* First point of the list is a complete sentence about the matter at hand.

== Sources ==
* Reuters, this must be cut.
"""


def test_wikitext_to_paragraphs_expands_inline_templates_and_cuts_sources():
    ps = [L.clean_paragraph(p) for p in L.wikitext_to_paragraphs(WT)]
    joined = " ".join(ps)
    assert "Japanese government said on Tuesday that the plan" in joined
    assert "Han-class vessel" in joined and "local fishermen" in joined
    assert "Reuters" not in joined and "hidden comment" not in joined and "thumb" not in joined and "cell" not in joined
    assert "First point of the list is a complete sentence about the matter at hand." in ps


def test_wikinews_date_and_licence_follow_the_copyright_policy():
    assert L.wikinews_date(WT) == "2012-10-03"
    assert L.wikinews_date("{{date|3 May 2007}}") == "2007-05-03"
    assert L.wikinews_licence("2005-09-24")[1] == "pd"
    assert L.wikinews_licence("2005-09-25") == ("CC-BY-2.5", "permissive")
    assert L.wikinews_licence("2024-12-15") == ("CC-BY-2.5", "permissive")
    assert L.wikinews_licence("2024-12-16") == ("CC-BY-4.0", "permissive")
    assert L.wikinews_licence("")[1] == "unclear"


def test_strip_dateline():
    assert L.strip_dateline("LONDON, England -- The market fell on Monday.") == "The market fell on Monday."
    assert L.strip_dateline("Monday, May 3, 2010 -- The market fell.") == "The market fell."
    assert L.strip_dateline("- The market fell.") == "The market fell."
    assert L.strip_dateline("The market fell.") == "The market fell."


def test_iter_dump_pages_reads_bz2_and_skips_redirects(tmp_path):
    xml = ('<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/"><page><title>A</title><ns>0</ns><id>7</id><revision><timestamp>2020-01-02T00:00:00Z</timestamp>'
           '<text>Hello world.</text></revision></page><page><title>B</title><ns>0</ns><id>8</id><redirect title="A"/><revision><text>#REDIRECT [[A]]</text></revision></page>'
           '<page><title>Talk:C</title><ns>1</ns><id>9</id><revision><text>talk</text></revision></page></mediawiki>')
    path = tmp_path / "d.xml.bz2"
    path.write_bytes(bz2.compress(xml.encode("utf-8")))
    rows = list(L.iter_dump_pages(str(path)))
    assert rows == [("A", "Hello world.", "2020-01-02T00:00:00Z", "7")]


def test_sampled_is_deterministic_and_proportional():
    keys = [str(i) for i in range(4000)]
    a = [k for k in keys if L.sampled(k, 1, 0.25)]
    assert a == [k for k in keys if L.sampled(k, 1, 0.25)]
    assert 800 < len(a) < 1200 and all(L.sampled(k, 1, 1.0) for k in keys[:10])


# ------------------------------------------------------------------------------------------------------------------------ OpenStax
CNXML = """<document xmlns="http://cnx.rice.edu/cnxml" xmlns:m="http://www.w3.org/1998/Math/MathML"><title>Cells</title><content>
<section><title>Intro</title>
<para id="p1">The cell is the basic unit of life, and every living thing is made of one or more cells that work together to keep the organism alive.</para>
<para id="p2">The equation <m:math><m:mi>x</m:mi></m:math> is dropped because the sentence would be broken without the formula inside it.</para>
<list><item>Cells can be very small, and most of them cannot be seen without the help of a microscope or a similar tool.</item></list>
<note><para>This note box is skipped entirely and is never part of the text.</para></note>
<figure><caption>Figure one</caption></figure>
<para>Smith, John. Cell Biology of Plants. Oxford University Press, 2009, 1998, 2001.</para>
</section></content></document>"""

COLXML = """<col:collection xmlns:col="http://cnx.rice.edu/collxml" xmlns:md="http://cnx.rice.edu/mdml"><col:metadata><md:title>Biology</md:title></col:metadata>
<col:content><col:subcollection><md:title>Chapter 1</md:title><col:content><col:module document="m100"><md:title>The Cell</md:title></col:module>
<col:module document="m101"><md:title>Tissue</md:title></col:module></col:content></col:subcollection></col:content></col:collection>"""


def test_cnxml_paragraphs_drop_math_notes_figures_and_references():
    title, ps = L.cnxml_paragraphs(CNXML.encode("utf-8"))
    ps = [L.clean_paragraph(p) for p in ps]
    assert title == "Cells"
    assert len(ps) == 2 and ps[0].startswith("The cell is the basic unit of life") and ps[1].startswith("Cells can be very small")


def test_openstax_collection_order_and_titles():
    book, mods = L.openstax_collection(COLXML.encode("utf-8"))
    assert book == "Biology"
    assert mods == [("Chapter 1", "m100", "The Cell"), ("Chapter 1", "m101", "Tissue")]


# ---------------------------------------------------------------------------------------------------------------- web page parsers
def test_gv_paragraphs_drop_captions_embeds_and_credit_lines():
    html = ("<div class='wp-caption'><p class='wp-caption-text'>Featured image: someone</p></div><p>This post is part of a series on elections.</p><p>%s</p>"
            "<blockquote class='twitter-tweet'><p>A tweet that should not be here at all, ok?</p></blockquote><p>Umair Khan contributed to the post.</p><p>%s</p>" % (P1, P2))
    ps = [L.clean_paragraph(p) for p in L.gv_paragraphs(html)]
    assert ps == [P1, P2]


def test_owid_paragraphs_cut_at_license_boilerplate():
    html = ("<html><head><meta property='og:title' content='Malaria - Our World in Data'><meta property='article:published_time' content='2024-05-06T10:00:00Z'>"
            "<title>Malaria | Our World in Data</title></head><body><article><h1>Malaria</h1><p>%s</p><p>%s</p>"
            "<p>Our articles and data visualizations rely on work from many different people and organizations.</p><p>%s</p></article></body></html>" % (P1, P2, P3))
    title, date, ps = L.owid_paragraphs(html)
    assert title == "Malaria" and date == "2024-05-06"
    assert [L.clean_paragraph(p) for p in ps] == [P1, P2]


def test_nasa_paragraphs_stop_at_media_contacts():
    html = "<p>%s</p><p>Credit: NASA/JPL-Caltech some words here for the credit line.</p><p>%s</p><p>Media Contact: Jane Doe</p><p>%s</p>" % (P1, P2, P3)
    assert [L.clean_paragraph(p) for p in L.nasa_paragraphs(html)] == [P1, P2]


def test_fed_paragraphs_skip_menu_and_stop_at_last_reviewed():
    html = ("<html><head><title>Diabetes - NIDDK</title></head><body><nav><p>Skip to main content and go on</p></nav><main><h1>Diabetes</h1><p>%s</p><p>%s</p>"
            "<p>Last Reviewed May 2024 by the editors</p><p>%s</p></main></body></html>" % (P1, P2, P3))
    title, date, ps = L.fed_paragraphs(html)
    assert title == "Diabetes" and [L.clean_paragraph(p) for p in ps] == [P1, P2]


def test_legislation_notes_strip_paragraph_numbers():
    html = ("<html><body><div id='viewLegContents'><p>12.%s</p><p>Section 3: Duty of care to the public</p><p>13. %s</p></div></body></html>" % (P1, P2))
    ps = L.legislation_note_paragraphs(html)
    assert ps == [P1, P2]


def test_fr_paragraphs_drop_headings_box_and_boilerplate():
    html = ("<div class='preamble'><div class='document-headings'><p>Document headings vary by document type but may contain things</p></div>"
            "<p>AGENCY: Nuclear Regulatory Commission is a thing.</p><p>%s</p><p>Authority: 5 U.S.C. 552 and other things</p><p>%s</p></div>" % (P1, P2))
    ps = [L.clean_paragraph(p) for p in L.fr_paragraphs(html)]
    assert ps == [P1, P2]


JATS = """<article><front><article-meta><article-id pub-id-type="doi">10.1/x</article-id><title-group><article-title>Clean water in schools</article-title></title-group>
<permissions><license><ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/">https://creativecommons.org/licenses/by/4.0/</ali:license_ref></license></permissions>
<abstract><p>%s</p></abstract></article-meta></front><body>
<sec><title>Background</title><p>%s<xref ref-type="bibr" rid="b1">1</xref> and then more words follow in the same sentence.</p></sec>
<sec><title>Methods</title><p>%s</p></sec>
<sec><title>Discussion</title><p>%s</p></sec></body></article>""" % (P1, P2, P3, P3)


def test_jats_paragraphs_keep_abstract_background_discussion_only():
    title, lic, ps, doi = L.jats_paragraphs(JATS)
    assert title == "Clean water in schools" and lic.endswith("/by/4.0/") and doi == "10.1/x"
    assert len(ps) == 3 and ps[0] == P1 and ps[2] == P3 and "1 and then" not in ps[1]


def test_pmc_licence_mapping():
    assert L.pmc_licence("https://creativecommons.org/licenses/by/4.0/") == ("CC-BY-4.0", "permissive")
    assert L.pmc_licence("https://creativecommons.org/licenses/by/3.0/") == ("CC-BY-3.0", "permissive")
    assert L.pmc_licence("https://creativecommons.org/publicdomain/zero/1.0/")[1] == "pd"
    assert L.pmc_licence("https://creativecommons.org/licenses/by-nc/4.0/")[1] == "noncommercial"
    assert L.pmc_licence("https://creativecommons.org/licenses/by-sa/4.0/")[1] == "share_alike"
    assert L.pmc_licence("")[1] == "permissive"


def test_window_paragraphs_stop_at_token_target():
    tc = L.TokenCounter("/nonexistent/tokenizer.json")
    paras = [("w " * 200).strip() + "." for _ in range(10)]
    w = L.window_paragraphs(paras, tc, 2, 250)
    assert 1 <= len(w) <= 3


# ------------------------------------------------------------------------------------------- later additions (coordinator review 2026-10-20)
def test_has_gap_flags_removed_markup_but_not_normal_prose():
    assert L.has_gap("In a recent interview with the ’s Representative to the Americas, Tashi Wangdi said so.")
    assert L.has_gap("The vote was held in the, country's parliament, said the speaker.")
    assert L.has_gap("According to ' contributions the plan goes ahead and ' is fine.")
    assert L.has_gap("They met with leaders of and other diseases were discussed.")
    assert not L.has_gap("The meeting began at 9 a.m. and ended at noon, with a short break to talk about the plan.")
    assert not L.has_gap("There was nothing to be afraid of in the plan, and nobody who had been subjected to it complained.")
    assert not L.keep_paragraph("In a recent interview with the ’s Representative to the Americas, Tashi Wangdi said that talks would go on.")


def test_abbreviation_dense_measure_tables_are_dropped():
    p = "Oz. Lbs. Grains Troy Avoir. Cwt. In board measure boards are counted by the foot, and in. ft. yd. rods are used for long lines."
    assert L.abbreviation_density(p) > 0.10 and not L.keep_paragraph(p)


def test_wikitext_interwiki_links_keep_their_label():
    ps = L.wikitext_to_paragraphs("The [[w:Tashi Wangdi|Dalai Lama's representative]] and [[Wikipedia:Tibet|Tibet]] met on [[n:Talk|Tuesday]].\n[[fr:Article]]\n[[Category:Tibet]]")
    assert [L.clean_paragraph(p) for p in ps][0] == "The Dalai Lama's representative and Tibet met on Tuesday."
    assert len(ps) == 1


def test_cnxml_drops_learning_objectives_and_run_on_list_items():
    xml = ("<document xmlns='http://cnx.rice.edu/cnxml'><title>T</title><content><section><para>By the end of this section, you will be able to:</para>"
           "<list><item>Identify the main parts of a cell</item><item>Explain how cells divide.</item></list>"
           "<para>The cell is the basic unit of life, and every living thing is made of one or more cells that work together.</para>"
           "<list><item>Cells can be very small, and most of them cannot be seen without a microscope or a similar tool.</item><item>cells and tissues</item></list>"
           "<para>Items such as <list><item>one</item></list> are inside this paragraph.</para></section></content></document>")
    _, ps = L.cnxml_paragraphs(xml.encode("utf-8"))
    ps = [L.clean_paragraph(p) for p in ps]
    assert ps == ["The cell is the basic unit of life, and every living thing is made of one or more cells that work together.",
                  "Cells can be very small, and most of them cannot be seen without a microscope or a similar tool."]


def test_voa_paragraphs_use_article_body_and_drop_editor_notes():
    html = ("<html><head><meta property='og:title' content='Story title'><meta property='article:published_time' content='2026-10-01T10:00:00Z'></head><body>"
            "<div class='wsw'><p>Editor's note: this story was translated from the original VOA Dari article.</p><p>%s</p><p>%s</p></div></body></html>" % (P1, P2))
    title, date, ps = L.voa_paragraphs(html)
    assert title == "Story title" and date == "2026-10-01" and [L.clean_paragraph(p) for p in ps] == [P1, P2]


def test_eb_articles_split_at_headwords():
    raw = ("*** START OF THE PROJECT GUTENBERG EBOOK EB ***\n\nABBEY (Lat. abbatia), the name given to a monastery governed by an abbot, which grew up in the early middle ages.\n\n"
           "The earliest abbeys were small houses of prayer in the east, and the rule of St Benedict later spread them over the whole of western Europe.\n\n"
           "ABBOT, GEORGE (1562-1633), archbishop of Canterbury, was born at Guildford, the son of a cloth-worker who sent him to Oxford.\n\n"
           "*** END OF THE PROJECT GUTENBERG EBOOK EB ***")
    arts = L.eb_articles(raw)
    assert [h for h, _ in arts] == ["Abbey", "Abbot, George"] and len(arts[0][1]) == 2


# ------------------------------------------------------------------------------------------------------------- v2 pool (Common Pile, GovReport, ...)
def test_cp_licence_accepts_cc_by_and_pd_only():
    assert L.cp_licence("Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/") == ("CC-BY-4.0", "permissive", "https://creativecommons.org/licenses/by/4.0")
    assert L.cp_licence("Public Domain")[1] == "pd"
    assert L.cp_licence("https://creativecommons.org/publicdomain/zero/1.0/")[:2] == ("CC0-1.0", "pd")
    assert L.cp_licence("Creative Commons - Attribution-ShareAlike - https://creativecommons.org/licenses/by-sa/4.0/") is None
    assert L.cp_licence("https://creativecommons.org/licenses/by-nc/4.0/") is None
    assert L.cp_licence("https://creativecommons.org/licenses/by-nd/3.0/") is None
    assert L.cp_licence("") is None and L.cp_licence("All rights reserved") is None


def test_markdown_paragraphs_drop_headings_tables_questions_and_keep_prose():
    md = ("# Chapter 3\n\n| a | b |\n|---|---|\n\n**Rivers** carry [water](https://x.org/a) from the hills to the sea, and many cities grew along their banks over the centuries.\n\n"
          "- Rivers also move silt and sand that build new land at the delta, which farmers have used for thousands of years.\n- short item\n\n"
          "What are three benefits of globalization in terms of tourism?\n\nLearn more about how Pressbooks supports open publishing practices in higher education.")
    ps = L.markdown_paragraphs(md)
    assert ps[0].startswith("Rivers carry water from the hills") and "**" not in ps[0] and "x.org" not in ps[0]
    assert any(p.startswith("Rivers also move silt") for p in ps)
    assert not any("short item" in p or "Pressbooks" in p or p.startswith("What are") for p in ps)


def test_plain_blocks_reflow_and_noise():
    raw = "[[Page 12]]\nThe committee met on Tuesday and\n  agreed that the plan should\n  be reviewed.\n\n----------\n\n``Quoted words,'' said the chairman.\n\n<GRAPHIC>"
    bl = L.plain_blocks(raw)
    assert bl[0] == "The committee met on Tuesday and agreed that the plan should be reviewed."
    assert bl[1] == "“Quoted words,” said the chairman." and len(bl) == 2


def test_keyword_domain_and_skip():
    assert L.keyword_domain("Medicare patients and hospital drug costs in the Medicaid program") == "health"
    assert L.keyword_domain("A history of the world: culture and religion in the ancient world") == "encyclopedia"
    assert L.keyword_domain("travel tourism and hotel destination marketing", skip=("travel",)) is None
    assert L.keyword_domain("nothing relevant here", default="x") == "x"


def test_govreport_url_and_paragraphs():
    assert L.govreport_url("crs_R46266") == ("CRS", "https://crsreports.congress.gov/product/pdf/R/R46266")
    assert L.govreport_url("gao_GAO-18-123") == ("GAO", "https://www.gao.gov/products/GAO-18-123")
    inp = "\tIntroduction\n\nThe 21 st Century Cures Act ( P.L. 114-255 ) was signed in 2016 , after a long debate.\nSecond paragraph of the report.\n\tHeading two\nThird."
    ps = L.govreport_paragraphs(inp)
    assert ps[0] == "The 21st Century Cures Act (P.L. 114-255) was signed in 2016, after a long debate." and len(ps) == 3 and not any(p.startswith("\t") for p in ps)


def test_billsum_paragraphs_reflow_and_drop_headers():
    txt = ("SECTION 1. SHORT TITLE.\n\n    This Act may be cited as the ``Water Act of \n2016''.\n\nSEC. 2. DEFINITIONS.\n\n    In this Act:\n"
           "            (1) Secretary.--The term ``Secretary'' means the \n        Secretary of the Interior and any person who acts for the Secretary.\n")
    ps = L.billsum_paragraphs(txt)
    assert any(p.startswith("This Act may be cited as the “Water Act of 2016”.") for p in ps)
    assert any(p.startswith("The term “Secretary” means the Secretary of the Interior") for p in ps)
    assert not any(p.startswith("SEC") for p in ps)


def test_cp_news_doc_fields_and_licence_filter():
    rec = {"id": 7, "text": "A title - 360info\nJane Doe\nPublished on May 1, 2022\n" + P1 + "\n" + P2, "created": "Published on May 1, 2022", "source": "news-360info",
           "metadata": {"license": "Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/", "url": "https://360info.org/x/", "author": "Jane Doe"}}
    d = L.cp_news_doc(rec, "360info")
    assert d["title"] == "A title" and d["licence"] == "CC-BY-4.0" and d["domain"] == "opinion_magazine" and "Jane Doe" in d["attribution"]
    rec["metadata"]["license"] = "Creative Commons - Attribution-NonCommercial - https://creativecommons.org/licenses/by-nc/4.0/"
    assert L.cp_news_doc(rec, "360info") is None
    rec["metadata"]["license"] = "Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/"
    assert L.cp_news_doc(rec, "altnews")["domain"] == "news"


def test_pick_long_and_random_keeps_longest_share():
    cands = [(i, {"i": i}) for i in range(100)]
    chosen = L.pick_long_and_random(cands, 20, "s", long_share=0.5)
    assert len(chosen) == 20 and {c["i"] for c in chosen} >= set(range(90, 100))
    assert chosen == L.pick_long_and_random(cands, 20, "s", long_share=0.5)               # deterministic


def test_pool_exclude_from_skips_known_ids_and_texts(tmp_path):
    tc = L.TokenCounter("/nonexistent/tokenizer.json")
    big = _big()
    a = str(tmp_path / "a.jsonl")
    pa = L.Pool(a, "t", tc, max_docs=5)
    assert pa.add(_doc(1, big))
    pa.close(complete=True)
    b = str(tmp_path / "b.jsonl")
    pb = L.Pool(b, "t", tc, max_docs=5, exclude_from=[a])
    assert pb.seen("t:1") and not pb.add(_doc(1, big)) and not pb.add(_doc(9, big)) and pb.n == 0       # same id / same text as an excluded document
    pb.close(complete=False)


def test_capacity_table_excludes_files_and_non_core_classes(tmp_path):
    rows = lambda dom, cls, toks: "".join(json.dumps({"id": f"{dom}{i}", "domain": dom, "licence_class": cls, "n_gemma_tokens": t}) + "\n" for i, t in enumerate(toks))
    (tmp_path / "a__pd.jsonl").write_text(rows("news", "pd", [500, 900, 2000, 4000]))
    (tmp_path / "w__share_alike.jsonl").write_text(rows("news", "share_alike", [500, 500]))
    (tmp_path / "openstax_ccby_snapshot__permissive.jsonl").write_text(rows("news", "permissive", [500]))
    res = L.capacity_table(str(tmp_path))["domains"]["news"]
    assert (res["r400"], res["r800"], res["r1600"], res["r3200"], res["usable1024"], res["usable4096"]) == (1, 1, 1, 1, 3, 1)


def test_pool_drops_repeated_paragraphs_inside_a_document(tmp_path):
    tc = L.TokenCounter("/nonexistent/tokenizer.json")
    pool = L.Pool(str(tmp_path / "r.jsonl"), "t", tc, max_docs=3)
    uniq = _big()
    big = [uniq[0], uniq[0], uniq[1], uniq[2], uniq[1], uniq[3], uniq[4], uniq[5], uniq[2]]
    assert pool.add(_doc(1, big))
    pool.close(complete=True)
    row = json.loads(open(str(tmp_path / "r.jsonl"), encoding="utf-8").readline())
    assert row["text"].split("\n\n") == [L.clean_paragraph(u) for u in uniq]


# ================================================================================================================ v3: curated benchmark text
SGM = """<srcset setid="newstest2019" srclang="any">
<doc sysid="ref" docid="bbc.1" genre="news" origlang="en">
<p>
<seg id="1">Welsh AMs worried about 'looking like muppets'</seg>
<seg id="2">There is consternation among some AMs &amp; others at a suggestion.</seg>
<seg id="3">It has arisen because of plans to change the name.</seg>
</p>
</doc>
<doc sysid="ref" docid="x.2" genre="news" origlang="de">
<p><seg id="1">Hallo.</seg></p>
</doc>
</srcset>"""

XML = """<?xml version='1.0' encoding='utf-8'?>
<dataset id="wmttest2022"><collection id="general">
<doc origlang="en" id="rt.com.5" domain="news"><src lang="en"><p><seg id="1">Headline of the story</seg></p><p><seg id="2">First paragraph sentence one.</seg><seg id="3">Sentence two.</seg></p></src>
<hyp system="A" lang="de"><p><seg id="1">Hallo</seg></p></hyp></doc>
<doc origlang="en" id="tsuite-x-seg1" testsuite="tsuite-x" domain="news"><src lang="en"><p><seg id="1">A challenge sentence.</seg></p></src></doc>
<doc origlang="en" id="chat_1" domain="conversation"><src lang="en"><p><seg id="1">Hi there, how are you today?</seg></p></src></doc>
<doc origlang="de" id="de.1" domain="news"><src lang="de"><p><seg id="1">Hallo.</seg></p></src></doc>
</collection></dataset>"""


def _tar_gz(tmp_path, members):
    import io
    import tarfile
    path = str(tmp_path / "t.tar.gz")
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            b = data.encode("utf-8")
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    return path


def test_wmt_sgm_parse_and_layout_headline_split():
    docs = list(L.wmt_parse_sgm(SGM))
    assert len(docs) == 2 and docs[0][0]["docid"] == "bbc.1" and docs[1][0]["origlang"] == "de"
    texts, ranges, layout = L.wmt_layout(docs[0][1])
    assert layout == "headline+single_block" and texts[0] == "Welsh AMs worried about 'looking like muppets'"
    assert texts[1] == "There is consternation among some AMs & others at a suggestion. It has arisen because of plans to change the name." and ranges == [[1, 1], [2, 3]]


def test_wmt_layout_keeps_source_paragraphs_and_full_sentence_first_segment():
    texts, ranges, layout = L.wmt_layout([[(1, "Headline of the story")], [(2, "One."), (3, "Two.")]])
    assert layout == "source_paragraphs(headline_first)" and texts == ["Headline of the story", "One. Two."]
    texts, ranges, layout = L.wmt_layout([[(1, "The minister said that the plan will work."), (2, "Others disagree.")]])
    assert layout == "single_block" and len(texts) == 1


def test_wmt_docs_from_tar_skip_testsuites_other_domains_and_foreign_origin(tmp_path):
    path = _tar_gz(tmp_path, {"x/xml/wmttest2022.en-de.all.xml": XML, "x/xml/wmttest2022.en-cs.all.xml": XML, "x/xml/wmttest2022.de-en.all.xml": XML})
    docs = list(L.wmt_docs(path, "wmttest2022"))
    assert [d["docid"] for d in docs] == ["rt.com.5"]                    # test suite, conversation domain, origlang de and the same document in a second pair are skipped
    row = L.wmt_row(docs[0], "wmttest2022")
    assert row["licence_class"] == "unclear" and row["id"] == "wmt:wmttest2022:rt.com.5" and row["extra"]["outlet"] == "rt.com"
    assert row["title"] == "Headline of the story" and row["extra"]["paragraph_segments"] == [[1, 1], [2, 3]] and "freely used for research purposes" in row["licence_note"]
    assert "statmt.org/wmt22" in row["licence_evidence"]


def test_wmt_terms_quote_wording_per_year():
    assert "we just ask that you cite the WMT19 shared task overview paper" in L.wmt_terms("newstest2019")[1]
    assert "we ask that you cite the WMT23 shared task overview paper" in L.wmt_terms("wmttest2023")[1]
    assert "overview findings paper" in L.wmt_terms("wmttest2024")[1] and "wmt25" in L.wmt_terms("wmttest2025")[0]


def test_wmt_mqm_docs_group_segments_by_document(tmp_path):
    tsv = tmp_path / "m.tsv"
    tsv.write_text("system\tdoc\tdoc_id\tseg_id\trater\tsource\ttarget\tcategory\tseverity\n"
                   "A\tbbc.1\t1\t2\tr\tSecond.\tZwei\tno-error\tno-error\nA\tbbc.1\t1\t1\tr\tFirst headline\tEins\tno-error\tno-error\n"
                   "B\tbbc.1\t1\t1\tr\tFirst headline\tEins\tno-error\tno-error\nA\tcnn.2\t2\t1\tr\tOther.\tAnder\tno-error\tno-error\n", encoding="utf-8")
    docs = list(L.wmt_mqm_docs(str(tsv)))
    assert [d["docid"] for d in docs] == ["bbc.1", "cnn.2"] and docs[0]["paras"] == [[(1, "First headline"), (2, "Second.")]]


def test_wmt25_docs_dedupe_languages_and_split_lines(tmp_path):
    rec = lambda tgt, dom, text: json.dumps({"doc_id": f"en-{tgt}_#_{dom}_#_doc1", "domain": dom, "src_lang": "en", "tgt_lang": tgt, "src_text": text})
    f = tmp_path / "w.jsonl"
    f.write_text("\n".join([rec("de_DE", "news", "Title line\n\nPara one.\nPara two."), rec("cs_CZ", "news", "Title line\n\nPara one.\nPara two."), rec("de_DE", "speech", "um yes"),
                            json.dumps({"doc_id": "cs-de_DE_#_news_#_c", "domain": "news", "src_lang": "cs", "src_text": "Ahoj"})]), encoding="utf-8")
    docs = list(L.wmt25_docs(str(f)))
    assert len(docs) == 1 and docs[0]["docid"] == "news:doc1" and [p[0][1] for p in docs[0]["paras"]] == ["Title line", "Para one.", "Para two."]


def test_pool_verbatim_keeps_short_documents_and_repeated_paragraphs(tmp_path):
    tc = L.TokenCounter("/nonexistent/tokenizer.json")
    pool = L.Pool(str(tmp_path / "v.jsonl"), "t", tc, max_docs=5, min_tokens=5, min_paragraphs=1, verbatim=True)
    assert pool.add(dict(id="t:1", domain="news", licence="x", licence_class="unclear", licence_evidence="e", paragraphs=["Takes a delicate touch!", "Takes a delicate touch!"]))
    assert not pool.add(dict(id="t:2", domain="news", licence="x", licence_class="unclear", licence_evidence="e", paragraphs=["Это совсем не английский текст, а русский."]))
    pool.close(complete=True)
    row = json.loads(open(str(tmp_path / "v.jsonl"), encoding="utf-8").readline())
    assert row["text"] == "Takes a delicate touch!\n\nTakes a delicate touch!"
    assert L.short_english_ok("Takes a delicate touch!") and not L.short_english_ok("Это не английский текст вообще")


def test_flores_passages_group_by_article_and_chunk_long_runs():
    rows = [{"id": i, "text": f"Sentence {i}.", "url": "https://en.wikinews.org/wiki/A_b", "domain": "wikinews", "topic": "t"} for i in range(0, 4)]
    rows += [{"id": i, "text": f"Sentence {i}.", "url": "https://en.wikivoyage.org/wiki/Paris", "domain": "wikivoyage", "topic": "u"} for i in range(4, 15)]
    ps = list(L.flores_passages(rows, "dev"))
    assert [p["ids"] for p in ps] == [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14]] or [len(p["ids"]) for p in ps] == [4, 6, 5]
    assert ps[0]["id"] == "flores_plus:dev:0-3" and ps[0]["title"] == "A b" and all(2 <= len(p["ids"]) <= 6 for p in ps)


def test_flores_ids_map_and_ntrex_lines_map(tmp_path):
    f = tmp_path / "f.jsonl"
    f.write_text(json.dumps({"id": "flores_plus:dev:0-3", "extra": {"flores_split": "dev", "flores_ids": [0, 1, 2, 3]}}) + "\n"
                 + json.dumps({"id": "ntrex128:d", "extra": {"ntrex_lines": [5, 9], "ntrex_file": "f.txt", "docid": "d"}}) + "\n", encoding="utf-8")
    assert L.flores_ids_map(str(f)) == {"flores_plus:dev:0-3": [["dev", [0, 1, 2, 3]]]}
    assert L.ntrex_lines_map(str(f)) == {"ntrex128:d": {"file": "f.txt", "lines": [5, 9], "docid": "d"}}


def test_glotstorybook_licence_classes_title_and_stories():
    assert L.gsb_licence("CC-BY") == ("CC-BY", "permissive") and L.gsb_licence("CC BY")[1] == "permissive" and L.gsb_licence("Public Domain")[1] == "pd"
    assert L.gsb_licence("CC-BY-NC")[1] == "noncommercial" and L.gsb_licence("CC BY-NC-SA")[1] == "noncommercial" and L.gsb_licence("weird")[1] == "unclear"
    assert L.gsb_title("0001_a-very-tall-man.md") == "A very tall man" and L.gsb_title("0003_new-in-town_en.md") == "New in town"
    rows = [{"ISO639-3": "eng", "Source": "global-asp/asp-source", "File Name": "0001_a.md", "Text Number": "1", "Text": "Second.", "License": "CC-BY", "Parallel ID": "asp_0001"},
            {"ISO639-3": "eng", "Source": "global-asp/asp-source", "File Name": "0001_a.md", "Text Number": "0", "Text": "First.", "License": "CC-BY", "Parallel ID": "asp_0001"},
            {"ISO639-3": "swa", "Source": "global-asp/asp-source", "File Name": "0001_a.md", "Text Number": "0", "Text": "Kwanza.", "License": "CC-BY", "Parallel ID": "asp_0001"}]
    st = list(L.gsb_stories(rows))
    assert len(st) == 1 and [r["Text"] for r in st[0]["rows"]] == ["First.", "Second."] and st[0]["cls"] == "permissive"


def test_un_story_paragraphs_main_text_only():
    html = ('<html><head><meta property="og:title" content="Hunger is bad | UN News"/><meta property="article:published_time" content="2026-10-19T16:00:36-04:00"/></head><body>'
            '<div class="clearfix text-formatted field field--name-field-text-column field__item"><p>Addressing the committee in Rome, the High Commissioner said that the right to food matters.&nbsp;</p>'
            '<p>Read more about it here</p><p>Global food supply now exceeds three thousand calories per person per day.</p></div>'
            '<div class="caption"><p>A photo caption that is not part of the main text of the story at all.</p></div></body></html>')
    title, date, paras = L.un_story_paragraphs(html)
    assert title == "Hunger is bad" and date == "2026-10-19" and len(paras) == 2 and paras[0].startswith("Addressing the committee")


def test_nalibali_stories_english_only_and_author_prefix():
    rows = [{"iso639-3": "eng", "parallel_id": "w", "story": "Line one.\n\nLine two.\n", "title": " Wolf ", "author": "By Griselda Jacobs", "link": "https://nalibali.org/x"},
            {"iso639-3": "zul", "parallel_id": "w", "story": "Okokuqala.", "title": "Impisi", "author": "", "link": ""}]
    st = list(L.nalibali_stories(rows))
    assert len(st) == 1 and st[0]["paras"] == ["Line one.", "Line two."] and st[0]["author"] == "Griselda Jacobs" and st[0]["title"] == "Wolf"


def test_wmt24pp_map_matches_document_ids(tmp_path):
    f = tmp_path / "sw.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in [
        {"document_id": "test-en-news_a.1", "segment_id": 1, "is_bad_source": False, "source": "x"}, {"document_id": "test-en-news_a.1", "segment_id": 2, "is_bad_source": False, "source": "y"},
        {"document_id": "canary", "segment_id": 0, "is_bad_source": True, "source": "CANARY"}]), encoding="utf-8")
    pool = tmp_path / "p.jsonl"
    pool.write_text(json.dumps({"id": "wmt:wmttest2024:test-en-news_a.1", "extra": {"testset": "wmttest2024", "docid": "test-en-news_a.1"}}) + "\n"
                    + json.dumps({"id": "wmt:newstest2020:b", "extra": {"testset": "newstest2020", "docid": "b"}}) + "\n", encoding="utf-8")
    m = L.wmt24pp_map([str(pool)], configs=("en-sw_KE",), hf_files={"en-sw_KE": str(f)})
    assert list(m) == ["wmt:wmttest2024:test-en-news_a.1"] and m["wmt:wmttest2024:test-en-news_a.1"]["segment_ids"] == {"en-sw_KE": [1, 2]}


def test_wmt24_segments_are_paragraphs():
    w = dict(docid="test-en-news_a.1", domain_label="news", pair_file="x", paras=[[(1, "Headline here"), (2, "A long first paragraph of the story. It has two sentences."), (3, "Second paragraph.")]])
    row = L.wmt_row(w, "wmttest2024")
    assert row["paragraphs"] == ["Headline here", "A long first paragraph of the story. It has two sentences.", "Second paragraph."]
    assert row["extra"]["layout"] == "paragraph_segments" and row["extra"]["paragraph_segments"] == [[1, 1], [2, 2], [3, 3]]
    row = L.wmt_row(dict(w, paras=[[(1, "Headline here"), (2, "One sentence."), (3, "Another.")]]), "wmttest2022")
    assert row["extra"]["layout"] == "headline+single_block" and len(row["paragraphs"]) == 2


def test_mafand_items_merge_same_english_text_across_pairs():
    jl = lambda *pairs: "\n".join(json.dumps({"translation": {"en": e, **{k: v for k, v in [("hau", h)] if h}, "yor": y}}) for e, h, y in pairs)
    files = {("hau", "test"): "\n".join(json.dumps({"translation": {"en": e, "hau": r}}) for e, r in [("The minister spoke on Monday.", "Ministan ya yi magana."), ("", "x")]),
             ("yor", "test"): json.dumps({"translation": {"en": "The minister spoke on Monday.", "yor": "Minisita sọrọ."}})}
    items = list(L.mafand_items(files))
    assert len(items) == 1 and items[0]["parallel"] == {"hau": "Ministan ya yi magana.", "yor": "Minisita sọrọ."} and items[0]["where"] == ["hau:test:0", "yor:test:0"]
