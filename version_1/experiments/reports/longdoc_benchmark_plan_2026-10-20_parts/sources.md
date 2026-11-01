# Source pool of English documents for the long-document benchmark (2026-10-20)

All numbers below are measured from the pool files in `/workspace/sft_build/longdoc_bench/pool/` on the rental (token counts = Gemma-3 `tokenizer.json`, no BOS; `POOL_STATS.json` in the same directory has the same figures per file). Pool = 17 JSONL files, 21,646 documents (16,593 public domain / permissive including the 1,889-document OpenStax snapshot, 5,053 share-alike; coordinator's recount of `POOL_STATS.json`, the first draft of this sentence said 31,000), 162 MB. Fetcher: `gemma_mt/scripts/longdoc_fetch_sources.py` (tests: `gemma_mt/tests/test_longdoc_sources.py`, 38 passed on the laptop and on the rental with `venv_gemma`).

Reading guide. (1) The coordinator cuts long documents at paragraph boundaries, so a document of N tokens can serve every bin whose lower bound is <= N: the "usable" columns of section (c) are the capacity that matters, the plain bin columns count each document once at its own length. (2) Domain labels are the taxonomy of the task (news, encyclopedia, health, law_government, education, stories, travel, science_explainer); no change proposed. (3) `licence_class` follows the task definition; the extra pool key `licence_note` carries the per-source caveat, `extra` carries source specific metadata (OpenStax book / commit).

## (a) Source / licence table

| pool file (source) | domain(s) | licence | class | evidence URL (licence statement) | attribution rule | access | docs | median tok | 400-799 / 800-1599 / 1600-3199 / >=3200 | caveats |
|---|---|---|---|---|---|---|---|---|---|---|
| `wikinews__permissive` (wikinews) | news | CC-BY-2.5 for articles published 2005-09-25..2024-12-15, CC-BY-4.0 from 2024-12-16 (licence chosen per article from its `{{date}}`) | permissive | https://en.wikinews.org/wiki/Wikinews:Copyright (read 2026-10-20) | credit "Wikinews" (policy: 'may be attributed to "Wikinews"'); pool has `Wikinews contributors, "title", en.wikinews.org` + licence | official XML dump `enwikinews-latest-pages-articles.xml.bz2` | 5,988 | 546 | 4838 / 956 / 151 / 43 | fair-use quotes of third-party text and press releases inside articles; images excluded; articles with dropped-template gaps removed; mostly short (a 512-bin source) |
| `wikinews__pd` (wikinews) | news | public domain: published before 2005-09-25 (same policy page) | pd | same | none required | same dump | 573 | 548 | 476 / 82 / 12 / 3 | 2004-2005 articles only; style of early Wikinews |
| `globalvoices__permissive` (globalvoices) | news | CC BY 3.0 "unless otherwise stated" (site footer: "licensed as Creative Commons Attribution 3.0") | permissive | https://globalvoices.org/about/global-voices-attribution-policy/ ; https://globalvoices.org/about/legal-notices/ | name of the author + link to the original story (policy); the pool has the URL, **the author name is NOT in the pool** (the REST `_embed=author` returned none): fill it from the page when the credit file is built | WordPress REST API (`/wp-json/wp/v2/posts`, 100 posts/page, robots Crawl-delay 10) | 1,299 | 970 | 442 / 733 / 117 / 7 | photos / video / audio of other creators are NOT under the licence (removed); embedded tweets removed; posts partly translated from other languages; fetched from the laptop (see (e)) because globalvoices.org was unreachable from the rental for ~1.5 h, tokens recounted on the rental |
| `govuk__permissive` (govuk) | travel, news, law_government, health, education, science_explainer | OGL v3.0 | permissive | https://www.gov.uk/help/terms-conditions ("all content is available under the Open Government Licence v3.0, except where otherwise stated") | "Contains public sector information licensed under the Open Government Licence v3.0" + Crown copyright + department (in `attribution`) | Search API + Content API (JSON) | 1,223 | 1,132 | 432 / 338 / 297 / 156 | third-party quotes, logos, images excluded; UK-specific subject matter; travel advice = 210 FCDO country pages |
| `federalregister__pd` (federalregister) | law_government | US federal government work | pd | https://www.govinfo.gov/about/policies (Public Domain & Copyright Notice) | credit agency + FR citation (in `attribution`) | federalregister.gov API v1 + `full_text/html` | 898 | 4,363 | 30 / 34 / 213 / 621 | rules / proposed rules / notices 2003-2026; bureaucratic register; may quote third-party comments |
| `nasa__pd` (nasa) | science_explainer | US government work, "generally not copyrighted" | pd | https://www.nasa.gov/nasa-brand-center/images-and-media/ | "NASA" + URL | WordPress REST API of science.nasa.gov and www.nasa.gov | 1,598 | 783 | 833 / 594 / 113 / 58 | material credited to ESA, CSA, JAXA, universities, contractors may be copyrighted; images excluded; blogs excluded |
| `fedhealth__pd` (fedhealth: NIDDK, NIMH, NIDA) | health (243), science_explainer (230) | US federal government text; NIDDK "majority of information ... copyright free" | pd | https://www.niddk.nih.gov/copyright ; NIMH/NIDA copyright pages (URL stored per doc) | credit the institute; NIDDK: logos removed from edited content, no implied endorsement | sitemap.xml + page HTML | 473 | 769 | 250 / 168 / 45 / 10 | co-sponsored documents and some graphics are third-party; short pages |
| `nps__pd` (fedhealth: nps-history / nps-articles / nps-nature) | encyclopedia | National Park Service: "generally considered in the public domain" unless otherwise indicated | pd | https://www.nps.gov/aboutus/disclaimer.htm | "National Park Service" + URL | sitemap.xml + page HTML | 240 | 1,051 | 87 / 88 / 43 / 22 | photos / quoted material may be third-party; US-centred |
| `voa__pd` (voa) | news | "All text ... produced exclusively by the Voice of America is in the public domain" | pd | https://www.voanews.com/p/5338.html | "Voice of America" + URL | sitemaps + article HTML | 274 | 670 | 182 / 87 / 5 / 0 | licensed AFP / AP / Reuters content is NOT public domain: every article mentioning a wire service was dropped; short |
| `gutenberg__pd` (gutenberg) | stories 289, travel 128, science_explainer 113, education 82, law_government 82, health 80, encyclopedia 26 | public domain (US); authors died <= 1954 so also life+70 | pd | https://www.gutenberg.org/policy/permission.html (+ catalog death year) | author + title + "Project Gutenberg #id"; the PG trademark text is stripped | catalog CSV + `pg<id>.txt` | 800 | 3,136 | 91 / 101 / 220 / 388 | archaic 18th-20th century English; books with a "Translator:" line excluded; chapter-level cuts; <= 3 chapters per book |
| `eb1911__pd` (eb1911) | encyclopedia | Encyclopaedia Britannica 11th ed. (1910-11), public domain (US) | pd | https://www.gutenberg.org/policy/permission.html | "Encyclopaedia Britannica, 11th ed., article X; via Project Gutenberg #id" | PG slices | 421 | 2,312 | 7 / 112 / 153 / 149 | contributors not individually dated (a few died after 1954): US-PD certain, life+70 not verified; Edwardian style, bibliography tails |
| `openstax_ccby_current__permissive` (openstax) | education, science_explainer, law_government | CC BY 4.0, LICENSE of the repository HEAD says "Attribution 4.0" (Statistics, Physics, Introduction to Intellectual Property, Introduction to Business, Life Liberty and Pursuit of Happiness) | permissive | `https://github.com/openstax/osbooks-<repo>/blob/<sha>/LICENSE` (stored per doc); CMS API https://openstax.org/apps/cms/api/v2/pages/ confirms by/4.0 | "OpenStax, <book>, Rice University, CC BY 4.0" | `git clone --filter=blob:none` sparse (modules/, collections/) | 258 | 1,094 | 81 / 104 / 62 / 11 | figures / third-party items may differ (images excluded) |
| `openstax_ccby_snapshot__permissive` (openstax) | education 1056, science_explainer 320, health 320, law_government 193 | CC BY 4.0 as of the newest commit whose LICENSE says "Attribution 4.0" | permissive **with owner decision** | per doc: LICENSE at the pinned commit (`extra.commit`) | as above | as above | 1,889 | 1,936 | 237 / 509 / 733 / 410 | OpenStax relicensed these books to CC BY-NC-SA 4.0 in 2025-26 (e.g. Microbiology 2026-03-19, commit `f1d8d4e` "updating license to CC BY NC-SA"). CC BY 4.0 is irrevocable for text already distributed under it, but the licensor's current offer is non-commercial: **exclude from a public release unless the owner accepts this**. Pool file kept separate for that reason |
| `owid__permissive` (owid) | science_explainer 372, health 112 | CC BY 4.0 (text and charts; third-party data keeps its own licence) | permissive | https://ourworldindata.org/faqs#can-i-use-our-content | "Our World in Data" + authors + URL | sitemap.xml + article HTML | 484 | 1,660 | 90 / 140 / 188 / 66 | data and charts not included; some pages are topic pages with summary text |
| `pmc__permissive` (pmc) | health | CC BY (version read from each article's `<license>`; CC0 would be `pd`) | permissive | licence URL inside each article's JATS XML (stored in `licence_evidence`) | `authors (year). title. journal. doi. licence` (in `attribution`) | Europe PMC REST: `search` (LICENSE:"cc by") + `fullTextXML` | 175 | 1,718 | 15 / 64 / 65 / 31 | only intro / discussion / conclusion prose (methods, results, back matter dropped); research register; figures and third-party material inside an article may differ; stopped early (175 of 1,100 planned) |
| `wikipedia__share_alike` (wikipedia) | encyclopedia | CC BY-SA 4.0 (text since 2023-06) / CC BY-SA 3.0 + GFDL | share_alike | https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use | "Wikipedia contributors, title, en.wikipedia.org" + licence | HF `wikimedia/wikipedia` 20231101.en shards 3, 14, 22, 31 (revision b04c8d1), 2% hash sample | 2,568 | 1,070 | 704 / 1188 / 476 / 200 | **share-alike**; Wikipedia is in the SFT/RL English source: expect heavy contamination flags |
| `wikivoyage__share_alike` (wikivoyage) | travel | CC BY-SA 4.0 (siteinfo `rightsinfo`, read 2026-10-20) | share_alike | https://en.wikivoyage.org/wiki/Wikivoyage:Terms_of_use | "Wikivoyage contributors, title" | official dump `enwikivoyage-latest-pages-articles.xml.bz2`, 30% hash sample | 2,485 | 1,156 | 792 / 837 / 542 / 314 | **share-alike**; listing entries (see / eat / sleep) were removed with their templates, a few residual link lists remain |

Not in the pool although implemented: `legislation` (UK Explanatory Notes, OGL v3): the first run crawled with a cleaner bug (every paragraph was dropped), the corrected run was stopped at 29 documents when the coordinator asked to wrap up; the file was deleted (an incomplete pool of 29 documents is useless). The fetcher is fixed and tested: `python scripts/longdoc_fetch_sources.py legislation --out ... --max-docs 900` (about 2 h because robots.txt asks for a 5 s Crawl-delay).

## (b) Rejected or not usable sources (do not look again)

| source | reason (evidence) |
|---|---|
| OpenStax live books (about 100 titles) | CC BY-NC-SA 4.0 since 2025-26 (CMS API https://openstax.org/apps/cms/api/v2/pages/?type=books.Book: only Statistics, Physics, Intro to IP are live CC BY; retired editions are CC BY). Noncommercial: not built into a pool (only the CC-BY snapshot above) |
| OpenStax archive API (`/apps/archive/...`) | disallowed by openstax.org robots.txt |
| World Bank blogs / site content | World Bank terms of use grant a non-commercial limited licence, no CC statement found on the site or the blog pages (checked 2026-10-20) |
| World Bank documents (WDS / OKR) | per-item licence and PDF-only text; not pursued |
| WHO publications | CC BY-NC-SA 3.0 IGO (https://www.who.int/about/policies/publishing/copyright): noncommercial |
| FAO | web content under FAO reuse terms; CC BY 4.0 applies to statistical databases only: no prose |
| The Conversation | CC BY-ND ("Attribution/No derivatives", https://theconversation.com/republishing-guidelines): ND forbids translation-derived release |
| UN (un.org terms of use page) | no CC statement; reproduction terms not CC |
| African Storybook | CC BY 4.0 (site footer) but stories are 100-400 words: below the 400-token floor; also probably in the training data |
| StoryWeaver | search API path 404 from the rental; stories short |
| SciDev.Net | footer says CC Attribution (no version) but the WordPress API returns 0 posts; not pursued |
| CIA World Factbook (GitHub mirror) | public domain but prose is one 'Background' paragraph (about 600 tokens) per country, mirror is old |
| MedlinePlus | page text mixes public-domain NLM text and A.D.A.M./HealthDay copyrighted content (https://medlineplus.gov/about/using/usingcontent/); health-topic summaries are also < 400 tokens |
| NCI PDQ, CDC pages, NHLBI, NIAID, FDA | not fetched (time; CDC / NCI statements: https://www.cdc.gov/other/agencymaterials.html, https://www.cancer.gov/policies/copyright-reuse say mostly public domain); good candidates if more health text is needed |
| Blocked from the rental IP (403 / WAF / no route) | congress.gov and CRS reports, state.gov, travel.state.gov, peacecorps.gov, loc.gov (Cloudflare challenge), eur-lex.europa.eu (AWS WAF 202), noaa.gov, usda.gov, opentextbc.ca, nia.nih.gov (405), api.nps.gov, smartraveller.gov.au; globalvoices.org was blocked for ~1.5 h (fetched from the laptop) |
| Share-alike encyclopedic fall-backs | Simple Wikipedia, Wikibooks, Wikiversity, Wikisource (SA or mixed): not fetched, would only duplicate the SA pools |
| HF corpora (FineWeb, pile-of-law, sedthh/gutenberg_english) | not used: FineWeb mixed copyright; pile-of-law dataset licence is NC-SA although the underlying US documents are PD (we fetched from the primary sources instead); Gutenberg directly from gutenberg.org (catalog gives the author death year) |

## (c) Pool sizes (measured from the pool files)

Documents per domain x length bin (Gemma tokens of the stored text, which is cut at 6,500 tokens: `truncated: true` on long ones), then the capacity when long documents are cut down ("usable for bin" = documents with at least the bin's lower bound: 512 -> >=400, 1024 -> >=800, 2048 -> >=1600, 4096 -> >=3200).

### pd + permissive (includes the OpenStax CC-BY *snapshot* file, 1,889 documents; subtract it if the owner rejects it)

| domain | 400-799 | 800-1599 | 1600-3199 | >=3200 | total | usable 512 | usable 1024 | usable 2048 | usable 4096 |
|---|---|---|---|---|---|---|---|---|---|
| education | 238 | 357 | 528 | 299 | 1422 | 1422 | 1184 | 827 | 299 |
| encyclopedia | 99 | 203 | 197 | 188 | 687 | 687 | 588 | 385 | 188 |
| health | 255 | 314 | 320 | 195 | 1084 | 1084 | 829 | 515 | 195 |
| law_government | 135 | 224 | 401 | 740 | 1500 | 1500 | 1365 | 1141 | 740 |
| news | 6146 | 1927 | 300 | 53 | 8426 | 8426 | 2280 | 353 | 53 |
| science_explainer | 1195 | 1025 | 416 | 211 | 2847 | 2847 | 1652 | 627 | 211 |
| stories | 8 | 31 | 98 | 152 | 289 | 289 | 281 | 250 | 152 |
| travel | 15 | 29 | 157 | 137 | 338 | 338 | 323 | 294 | 137 |
| all | 8091 | 4110 | 2417 | 1975 | 16593 | 16593 | 8502 | 4392 | 1975 |

Without the OpenStax snapshot file (measured): education 366 documents (254 usable for 1024, 155 for 2048, **47 for 4096**), health 764 (538 / 296 / 101), science_explainer 2,527 (1,388 / 488 / 177), law_government 1,307 (1,198 / 1,028 / 710); encyclopedia, news, stories, travel are unchanged. With the snapshot every domain meets the usable-capacity targets (>= 150 documents >= 800 tokens, >= 60 above 3,200) except **news >= 3,200 tokens (53 documents, target 60)**; without it, education falls short at the 4096 bin (47). Honest weak spots: news is almost entirely short pieces (Wikinews, VOA, Global Voices, GOV.UK news); encyclopedia without share-alike depends on EB1911 (Edwardian) and NPS; health has only 175 PMC articles, 473 NIH pages and OpenStax clinical books; stories is Gutenberg only (archaic English), no modern CC-BY fiction was found at >= 400 tokens.

### share_alike (separate files `wikipedia__share_alike`, `wikivoyage__share_alike`)

| domain | 400-799 | 800-1599 | 1600-3199 | >=3200 | total | usable 512 | usable 1024 | usable 2048 | usable 4096 |
|---|---|---|---|---|---|---|---|---|---|
| encyclopedia | 704 | 1188 | 476 | 200 | 2568 | 2568 | 1864 | 676 | 200 |
| travel | 792 | 837 | 542 | 314 | 2485 | 2485 | 1693 | 856 | 314 |

Per file: see the table in (a) and `/workspace/sft_build/longdoc_bench/pool/POOL_STATS.json` (also per-file token quantiles, domain and licence counts). Samples (3 short documents per source, <= 600 characters): `/workspace/sft_build/longdoc_bench/pool/samples/<source>.jsonl`.

## (d) Known licence risks of the whole pool and recommended attribution file

Risks.
1. **Third-party material inside permissive text.** Wikinews (fair-use quotes, press-release text), Global Voices (quoted posts, translations of other-language posts), GOV.UK (OGL "except where otherwise stated"; quoted letters), VOA (licensed wire text: only articles that mention no wire service were kept, residual risk remains), NASA / NPS / NIH (credited third-party photos and partner-agency text), OWID (third-party data). Images, charts, captions and embedded media are removed from every document; only running text is released.
2. **Per-document licence variation.** Wikinews changes licence by publication date (PD / CC-BY-2.5 / CC-BY-4.0): kept per document in `licence`; PMC articles are CC BY of varying version; Global Voices and GOV.UK say "unless otherwise stated". The `licence`, `licence_evidence` and `licence_note` fields are per document; the release must keep them.
3. **OpenStax snapshot** (1,889 documents): CC BY 4.0 at the pinned commit, relicensed to NC-SA by the licensor afterwards (see (a)). Recommendation: drop from a public release unless the owner takes the legal position that irrevocable CC BY applies; the current-CC-BY file (258 documents) is safe.
4. **Share-alike** (Wikipedia, Wikivoyage; also GFDL for Wikipedia): translations of these documents are derivatives and must be released under CC BY-SA; a mixed release needs a per-document licence column and a separate SA subset. The owner excluded SA from training; for the benchmark it is a decision.
5. **Public-domain claims are US-centred.** Gutenberg (authors died <= 1954, so life+70 is also met), EB1911 (contributors not individually checked), US federal works (17 USC 105; works by contractors or with copyrighted inserts are not covered), VOA / NASA / NPS / NIH statements are "generally" worded.
6. **Trademarks and names.** The Project Gutenberg licence text and trademark are stripped, NIDDK / NIH / NASA logos are not included; NIDDK asks that its name is not used to imply endorsement of the translated text. Do not present benchmark outputs as endorsed by these bodies.
7. **Contamination and register.** Wikipedia, Gutenberg, OpenStax, GOV.UK, PMC and Wikinews are certainly in web-scale pre-training data and probably in the SFT/RL English sources (the contamination scanner decides). Gutenberg / EB1911 text is archaic; Federal Register is bureaucratic; the pool is Anglo-American in subject matter.
8. **Personal data.** News and oral-history pages (NPS interviews, Wikinews) name public figures and sometimes private individuals; no scrub was done beyond removing e-mail addresses, URLs and phone-like numbers (digit-heavy paragraphs are dropped).

Recommended attribution file for a public release: one line per document in `ATTRIBUTION.jsonl` (plus a human-readable `ATTRIBUTION.md` grouped by source):
`{"id", "source", "title", "author_or_publisher", "url", "date", "licence", "licence_url", "licence_note", "changes"}` with `changes` = "extracted running text only; headings, tables, figures, captions, references and links removed; cut at paragraph boundary (n tokens); translated into <language> by <system>" (CC BY requires indicating changes) and a header block per source with the exact credit string required by the licensor: Wikinews -> 'Wikinews'; Global Voices -> author + link to the original story; GOV.UK / legislation -> "Contains public sector information licensed under the Open Government Licence v3.0"; OpenStax -> "OpenStax, <book>, Rice University, CC BY 4.0"; OWID -> "Our World in Data, CC BY 4.0"; PMC -> authors, journal, DOI, licence; Wikipedia / Wikivoyage -> "Wikipedia contributors", CC BY-SA 4.0 link, and the SA statement. All of these fields already exist in the pool records (`attribution`, `url`, `date`, `licence`, `licence_evidence`, `licence_note`), except the Global Voices author names.

## (e) Commands that reproduce the pool

Environment (rental): `cd /workspace/sft_build/longdoc_bench; export PYTHONPATH=/workspace/sft_build/longdoc_bench/code:/workspace/sft_build/pylibs; PY=/workspace/venv_gemma/bin/python`; CPU only: prefix `nice -n 10 env CUDA_VISIBLE_DEVICES=`. Script: `code/scripts/longdoc_fetch_sources.py` (laptop copy `gemma_mt/scripts/longdoc_fetch_sources.py`); `python scripts/longdoc_fetch_sources.py list` prints the registry.

```
S="$PY code/scripts/longdoc_fetch_sources.py"
$S wikinews        --out pool/wikinews__permissive.jsonl --max-docs 8000 --raw-dir raw/dumps --only permissive
$S wikinews        --out pool/wikinews__pd.jsonl         --max-docs 8000 --raw-dir raw/dumps --only pd
$S govuk           --out pool/govuk__permissive.jsonl    --max-docs 1500          # stopped at 1,224 (about 1 h at 1 req/s)
$S gutenberg       --out pool/gutenberg__pd.jsonl        --max-docs 800 --per-book 3
$S eb1911          --out pool/eb1911__pd.jsonl           --max-docs 420 --per-book 6
$S openstax        --out pool/openstax_ccby_current__permissive.jsonl  --max-docs 1200 --per-book 60 --only current  --raw-dir raw/
$S openstax        --out pool/openstax_ccby_snapshot__permissive.jsonl --max-docs 3000 --per-book 80 --only snapshot --raw-dir raw/
$S owid            --out pool/owid__permissive.jsonl     --max-docs 700
$S nasa            --out pool/nasa__pd.jsonl             --max-docs 1600
$S fedhealth       --out pool/fedhealth__pd.jsonl        --max-docs 1800           # NIDDK, NIMH, NIDA; stopped at 475
$S fedhealth --only nps-articles,nps-history,nps-nature --out pool/nps__pd.jsonl --max-docs 900   # stopped at 240
$S federalregister --out pool/federalregister__pd.jsonl  --max-docs 900
$S voa             --out pool/voa__pd.jsonl              --max-docs 700            # stopped at 274
$S pmc             --out pool/pmc__permissive.jsonl      --max-docs 1100           # stopped at 175
$S globalvoices    --out pool/globalvoices__permissive.jsonl --max-docs 2500        # run on the laptop (rental blocked), then upload + retokenize
$S wikivoyage      --out pool/wikivoyage__share_alike.jsonl --max-docs 2500 --raw-dir raw/dumps --sample 0.3
$S wikipedia       --out pool/wikipedia__share_alike.jsonl  --max-docs 3500 --raw-dir raw/ --sample 0.02
# quality pass over every file (re-applies the cleaner, recounts Gemma tokens, drops < 400 tokens and exact duplicates):
$S retokenize --in pool/X.jsonl --out tmp/qc/X.jsonl && mv tmp/qc/X.jsonl pool/X.jsonl
$S stats --pool-dir pool --out pool/POOL_STATS.json --samples-dir pool/samples
cd code && $PY tests/run_tests.py tests/test_longdoc_sources.py     # 38 passed
```

Notes for reproduction: runs are resumable (`<out>.part`) and deterministic for a fixed `--seed` (20261020), except that the live web sources (GOV.UK, Federal Register, Global Voices, PMC, VOA, NASA, OWID) change over time; dump sources are pinned only by "latest" (Wikinews / Wikivoyage 2026-10-20 dumps; Wikipedia revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, OpenStax commit ids are stored per document). Politeness: User-Agent `SabiYarn-longdoc-benchmark/0.1 (research; contact <email>)`, robots.txt honoured including Crawl-delay (Global Voices 10 s, legislation.gov.uk 5 s), at most 1 request/s per host, no authentication, nothing pushed anywhere. Raw dumps and clones were deleted afterwards (pool 162 MB, raw 31 MB).

Cleaning rules applied everywhere (all in `longdoc_fetch_sources.py`, unit-tested): prose paragraphs only (headings, tables, figures, captions, code, references, navigation dropped), URLs / e-mail addresses removed or the paragraph dropped, NFC, paragraphs joined by a blank line, paragraphs with >= 12% digits / many symbols / bibliographic patterns / dense abbreviations ("Oz. Lbs. Cwt.") / broken-markup gaps ("the ’s representative", "of,") dropped, English stop-word check, >= 400 Gemma tokens, exact duplicates removed, documents cut at a paragraph boundary at 6,500 tokens.
