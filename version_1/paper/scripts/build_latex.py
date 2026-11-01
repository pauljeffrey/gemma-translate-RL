#!/usr/bin/env python3
"""Build an Overleaf-ready LaTeX project in paper/latex/ from paper/sections/*.md, paper/tables/*.md, paper/figures/*.pdf and paper/refs.bib.

    python paper/scripts/build_latex.py [--pandoc PATH] [--zip]

Needs pandoc >= 3 (pip install pypandoc_binary gives one: pass --pandoc <venv>/lib/python*/site-packages/pypandoc/files/pandoc). No TeX needed to build the sources;
to test-compile: `tectonic -X compile main.tex` or `latexmk -xelatex main.tex` (XeLaTeX or LuaLaTeX; on Overleaf set Menu > Compiler to XeLaTeX).

What it does (same conventions as build_paper.py): {{figure:NAME|caption|width%}} -> figure float, {{table:NAME|caption}} -> table float (or longtable above 28 rows),
{{fref:NAME}} / {{tref:NAME}} -> Figure~\\ref / Table~\\ref, [@key] -> \\citep, @key -> \\citet (natbib, paper/refs.bib), TODO(...) markers -> red bold text,
"## 6 Title" headings -> \\section{Title} with automatic numbering, "## Appendix A Title" -> appendix sections. The generic article layout in main.tex is meant
to be swapped for a venue template: the body lives in sections/*.tex and is class-independent.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(PAPER, "latex")
LONG_ROWS = 28


def pandoc(pd, text, extra=(), fancy=True):
    fmt = "markdown+tex_math_dollars+pipe_tables-smart" + ("" if fancy else "-fancy_lists-example_lists")
    r = subprocess.run([pd, "-f", fmt, "-t", "latex", "--natbib", "--wrap=none", *extra],
                       input=text, capture_output=True, text=True)
    if r.returncode:
        sys.exit("pandoc failed: " + r.stderr[:500])
    return r.stdout.strip()


def fix_code_escapes(s):
    """`no\\_pair` in the markdown sources: the backslash is literal inside a code span; drop it."""
    return re.sub(r"`([^`\n]*)`", lambda m: "`" + m.group(1).replace("\\_", "_") + "`", s)


def esc(s):
    return re.sub(r"([#$%&_{}])", r"\\\1", s.replace("\\", "\\textbackslash{}")).replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")


def tab_to_tabular(tex):
    """pandoc longtable -> plain tabular (for tables inside a table float)."""
    lines = tex.splitlines()
    out, in_head = [], False
    for l in lines:
        if l.startswith("\\begin{longtable}"):
            spec = re.sub(r"^\\begin\{longtable\}\[\]", "", l)
            out.append("\\begin{tabular}" + spec)
            continue
        if l.strip() in ("\\endhead", "\\endfirsthead", "\\endfoot", "\\endlastfoot"):
            continue
        if l.startswith("\\end{longtable}"):
            out.append("\\bottomrule")
            out.append("\\end{tabular}")
            continue
        out.append(l)
    # pandoc puts \bottomrule before the first body row (end of the repeated foot); drop that one
    res, seen_mid, skip_next_bottom = [], False, True
    for l in out:
        if l.startswith("\\bottomrule") and skip_next_bottom and seen_mid:
            skip_next_bottom = False
            continue
        if l.startswith("\\midrule"):
            seen_mid = True
        res.append(l)
    return "\n".join(res).replace("\\noalign{}", "")


def build(pd):
    sec_dir = os.path.join(PAPER, "sections")
    files = sorted(f for f in os.listdir(sec_dir) if f.endswith(".md"))
    src = fix_code_escapes("\n\n".join(open(os.path.join(sec_dir, f), encoding="utf-8").read().rstrip() for f in files) + "\n")

    # numbering by order of appearance (for the unresolved-reference check only; LaTeX numbers floats by itself)
    fig_names = []
    tab_names = []
    for m in re.finditer(r"\{\{figure:([^|}]+)", src):
        if m.group(1) not in fig_names:
            fig_names.append(m.group(1))
    for m in re.finditer(r"\{\{table:([^|}]+)", src):
        if m.group(1) not in tab_names:
            tab_names.append(m.group(1))

    # title / authors / abstract
    title = re.match(r"# (.+)", src).group(1).strip()
    src = re.sub(r"^# .+\n", "", src, count=1)
    src = re.sub(r"^\[AUTHORS\][ \t]*\n\[AFFILIATIONS\][ \t]*\n", "", src, flags=re.M)
    src = re.sub(r"^\*\*Jeffrey Otoibhi\*\*[ \t]*\nAletheia AI[ \t]*\n", "", src, count=1, flags=re.M)
    abstract = ""
    m = re.search(r'<div class="abstract" markdown="1">\s*(.*?)\s*</div>', src, flags=re.S)
    if m:
        abstract = m.group(1)
        src = src.replace(m.group(0), "XXABSTRACTXX")

    blocks = {}   # placeholder -> LaTeX
    def put(tex):
        k = f"XXBLK{len(blocks)}XX"
        blocks[k] = tex
        return k

    # TODO markers
    def todo(m):
        return put("{\\color{red}\\textbf{" + esc(m.group(0)) + "}}")
    def protect_todo(s):
        return re.sub(r"TODO\((?:[^()]|\([^()]*\))*\)", todo, s)
    src = protect_todo(src)

    # appendix switch
    first_app = re.search(r"^## Appendix [A-Z] ", src, flags=re.M)
    if first_app:
        src = src[:first_app.start()] + "XXAPPENDIXXX\n\n" + src[first_app.start():]
    src = re.sub(r"^(#{2,4}) (?:Appendix [A-Z] |\d+(?:\.\d+)* )", r"\1 ", src, flags=re.M)

    # fref / tref
    src = re.sub(r"\{\{fref:([^}]+)\}\}", lambda m: put("Figure~\\ref{fig:" + m.group(1) + "}") if m.group(1) in fig_names else "Figure ??", src)
    src = re.sub(r"\{\{tref:([^}]+)\}\}", lambda m: put("Table~\\ref{tab:" + m.group(1) + "}") if m.group(1) in tab_names else "Table ??", src)

    def cap_tex(cap):
        return pandoc(pd, re.sub(r"^\((\w)\)", r"\\(\1)", protect_todo(cap)), fancy=False) if cap.strip() else ""

    def fig(m):
        name, cap, *w = m.group(1).split("|")
        width = min(float(w[0]) if w else 100.0, 100.0) / 100.0
        tex = ("\\begin{figure}[t]\n\\centering\n\\includegraphics[width=%.2f\\linewidth]{figures/%s.pdf}\n\\caption{%s}\n\\label{fig:%s}\n\\end{figure}"
               % (width, name, cap_tex(cap), name))
        return "\n\n" + put(tex) + "\n\n"
    src = re.sub(r"\{\{figure:([^}]+)\}\}", fig, src)

    def tab(m):
        name, cap = m.group(1).split("|", 1)
        tpath = os.path.join(PAPER, "tables", name + ".md")
        body = open(tpath, encoding="utf-8").read() if os.path.exists(tpath) else f"| missing table {name} |\n|--|\n"
        body = fix_code_escapes("\n".join(l for l in body.splitlines() if not l.startswith("#")))
        rows = [l for l in body.splitlines() if l.strip().startswith("|")]
        notes = "\n".join(l for l in body.splitlines() if not l.strip().startswith("|")).strip()
        cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
        sep_i = next((i for i, r in enumerate(rows) if re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", r.strip())), 1)
        ncol = max(len(c) for c in cells)
        n_body = len(rows) - sep_i - 1
        body_len = [max([len(re.sub(r"[*`]", "", c[j])) for i, c in enumerate(cells) if i not in (0, sep_i) and j < len(c)] + [1]) for j in range(ncol)]
        hdr_len = [len(re.sub(r"[*`]", "", cells[0][j])) if j < len(cells[0]) else 1 for j in range(ncol)]
        est = sum(max(b, h) for b, h in zip(body_len, hdr_len)) + 2 * ncol
        long_ = n_body > LONG_ROWS
        wrapped = max(body_len) > 40 or (long_ and est > 105) or est > 170
        small = "\\scriptsize" if ncol >= 9 else "\\footnotesize"
        if wrapped:   # relative widths from the dashes, as pandoc does for pipe tables; headers may wrap
            aligns = [(c[0] == ":", c[-1] == ":") for c in cells[sep_i]] if sep_i < len(cells) else [(False, False)] * ncol
            seps = []
            for j in range(ncol):
                w = max(4, min(max(body_len[j], min(hdr_len[j], 12)), 40))
                a = aligns[j] if j < len(aligns) else (False, False)
                seps.append((":" if a[0] else "-") + "-" * w + (":" if a[1] else "-"))
            rows[sep_i] = "|" + "|".join(seps) + "|"
        md = "\n".join(rows)
        md = protect_todo(md)
        extra = ["--columns=20"] if wrapped else ["--columns=100000"]
        if long_:
            md += "\n\n: " + protect_todo(cap) + " {#tab:" + name + "}"
            tex = pandoc(pd, md, extra)
            tex = tex.replace("\\begin{longtable}", "{" + small + "\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}", 1)
            tex = tex.rstrip() + "\n}"
            if notes:
                tex += "\n\n{\\footnotesize " + pandoc(pd, protect_todo(notes)) + "}"
            return "\n\n" + put(tex) + "\n\n"
        tex = tab_to_tabular(pandoc(pd, md, extra))
        inner = ("\\begin{adjustbox}{max width=\\linewidth}\n" + tex + "\n\\end{adjustbox}") if not wrapped else tex
        out = ("\\begin{table}[t]\n\\centering\n" + small + "\n\\setlength{\\tabcolsep}{3pt}\n\\caption{%s}\n\\label{tab:%s}\n%s\n"
               % (cap_tex(cap), name, inner))
        if notes:
            out += "\\par\\smallskip{\\footnotesize " + pandoc(pd, protect_todo(notes)) + "}\n"
        out += "\\end{table}"
        return "\n\n" + put(out) + "\n\n"
    src = re.sub(r"\{\{table:([^}]+)\}\}", tab, src)

    body_tex = pandoc(pd, src, ["--top-level-division=section", "--shift-heading-level-by=-1"])
    abs_tex = pandoc(pd, protect_todo(abstract)) if abstract else ""
    abs_tex = re.sub(r"XXBLK(\d+)XX", lambda m: blocks["XXBLK%sXX" % m.group(1)], abs_tex)

    # resolve placeholders (nested ones too)
    for _ in range(3):
        body_tex = re.sub(r"XXBLK\d+XX", lambda m: blocks[m.group(0)], body_tex)
    body_tex = body_tex.replace("XXAPPENDIXXX", "\\clearpage\n\\appendix")
    body_tex = body_tex.replace("XXABSTRACTXX", "")
    # pandoc emits \hypertarget{..}{%  \section{..}\label{..}} wrappers and \tightlist: keep (hyperref/ \providecommand in main.tex)
    return title, abs_tex, body_tex, files


MAIN = r"""% Generated by paper/scripts/build_latex.py from paper/sections/*.md. Edit the markdown sources (or this file once you take the project over to Overleaf).
% Compiler: XeLaTeX or LuaLaTeX (Overleaf: Menu > Compiler). Generic article layout; replace the preamble by a venue template if needed.
\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{fontspec}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{placeins,etoolbox}
\pretocmd{\subsection}{\FloatBarrier}{}{}
\usepackage{booktabs,longtable,array,calc,adjustbox}
\usepackage[table]{xcolor}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\usepackage{enumitem}
\setlist{itemsep=2pt,topsep=3pt}
\usepackage[authoryear,round]{natbib}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\usepackage{url}
\usepackage[htt]{hyphenat}
\setlength{\parskip}{0.4em}
\setlength{\parindent}{0pt}
\newcounter{none} % pandoc longtables
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}
\sloppy
\emergencystretch=2em

\title{%(title)s}
\author{Jeffrey Otoibhi\\ Aletheia AI}
\date{Preprint draft of 2026-10-25}

\begin{document}
\maketitle
\begin{abstract}
\noindent
%(abstract)s
\end{abstract}

\input{sections/body}

\bibliographystyle{plainnat}
\bibliography{refs}
\end{document}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pandoc", default=shutil.which("pandoc") or "pandoc")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    title, abs_tex, body_tex, files = build(a.pandoc)
    os.makedirs(os.path.join(OUT, "sections"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    open(os.path.join(OUT, "sections", "body.tex"), "w", encoding="utf-8").write(body_tex + "\n")
    open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8").write(MAIN.replace("%(title)s", esc(title)).replace("%(abstract)s", abs_tex))
    shutil.copy(os.path.join(PAPER, "refs.bib"), os.path.join(OUT, "refs.bib"))
    for f in os.listdir(os.path.join(PAPER, "figures")):
        if f.endswith(".pdf"):
            shutil.copy(os.path.join(PAPER, "figures", f), os.path.join(OUT, "figures", f))
    open(os.path.join(OUT, "README.txt"), "w").write(
        "Overleaf project generated from paper/sections/*.md by paper/scripts/build_latex.py.\n"
        "Upload the zip (New Project > Upload Project), set Menu > Compiler to XeLaTeX (or LuaLaTeX), main document main.tex.\n"
        "Body: sections/body.tex (all sections, tables and figure floats); bibliography: refs.bib (natbib, plainnat); figures: figures/*.pdf.\n"
        "Red bold text marks open items (TODO(owner: ...), TODO(ablation)). To use a venue template, keep sections/body.tex and refs.bib and replace the preamble/front matter of main.tex.\n")
    print("wrote", OUT, "(", len(body_tex.splitlines()), "lines of body )")
    if a.zip:
        z = os.path.join(PAPER, "overleaf_paper.zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, fs in os.walk(OUT):
                for f in fs:
                    if f.endswith((".aux", ".log", ".out", ".blg", ".bbl", ".xdv", ".pdf")) and root == OUT:
                        continue
                    p = os.path.join(root, f)
                    zf.write(p, os.path.relpath(p, OUT))
        print("wrote", z)


if __name__ == "__main__":
    main()
