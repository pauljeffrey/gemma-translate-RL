385M English-to-African translation paper, ACL format (review mode, double-blind). Draft of 2026-11-05.

Layout
  main.tex            ACL skeleton (\usepackage[review]{acl}), Times, A4, two columns
  acl.sty, acl_natbib.bst   official ACL style files (acl-org/acl-style-files)
  macros.tex          small macros (\ci intervals, \verify, \verifycite, \pending)
  sections/           abstract, s1..s6 (core), limitations, ethics, appendix (trimmed)
  refs.bib            only entries that are cited; copied unchanged from the project bib files
  figures/            PDF figures actually used (copied from paper_385m/figures)

Core = title ... Section 6 (limit: 8 pages). Limitations, Ethical considerations, references and the appendix
(one-column from the appendix on) are outside the core. main.aux label "coreend" gives the last core page,
"appstart" the first appendix page.

Rebuild
  Overleaf: upload the zip, compiler pdfLaTeX (default), LuaLaTeX or XeLaTeX also work. Main document main.tex.
  Local: tectonic main.tex   (checked with tectonic = XeTeX engine; pdfLaTeX not available on the build Mac)
  BibTeX runs automatically (tectonic) or: pdflatex, bibtex, pdflatex, pdflatex.

Camera-ready: replace [review] by [final] in main.tex and enable the commented \author line.

Placeholders (deliberately visible in the PDF)
  [OWNER: ...]               decision still open (release licence, which data parts are released)
The appendix is generated from paper.md by a converter and kept as hand-editable LaTeX; claim IDs and report keys are in % comments.

Trimmed version: the appendix is condensed to about 14 pages (appendix pages 13-26); the full appendix is in the sibling folder latex_acl. The core, Limitations, Ethical considerations and references are identical.

Two builds (same sources)
  Review build (anonymous, line numbers, no author-identifying links):  tectonic main.tex        -> main.pdf
  Preprint build (author block 'Jeffrey Otoibhi, Aletheia AI', live Hugging Face links):  tectonic main_preprint.tex -> main_preprint.pdf
  On Overleaf: set main.tex or main_preprint.tex as the main document. main_preprint.tex only defines \preprintbuild and inputs main.tex;
  the switch is \ifanon in main.tex (\anonreal{review text}{preprint text} in macros.tex). pdfLaTeX, XeLaTeX and LuaLaTeX are all fine.

Build check (run in this folder before every build or zip; prints the offending files and fails if any .tex under sections/ is empty or any \input target is missing)
  test -z "$(find sections -name '*.tex' -size 0)" && for f in $(grep -ho '\\input{[^}]*}' main.tex sections/*.tex | sed 's/\\input{//;s/}//'); do test -f "$f.tex" -o -f "$f" || { echo "missing input: $f"; exit 1; }; done && echo "sources ok" || { find sections -name '*.tex' -size 0; exit 1; }
