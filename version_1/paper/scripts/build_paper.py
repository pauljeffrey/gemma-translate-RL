#!/usr/bin/env python3
"""Build paper/paper.html and paper/paper.pdf from paper/paper.md (pandoc-style markdown) + paper/refs.bib + figures/ + tables/.

    python paper/scripts/build_paper.py            # needs: markdown (pip), Google Chrome (headless PDF); MathJax is loaded from cdnjs

Source conventions (all pandoc-compatible except the three {{...}} directives, which a pandoc filter would replace):
    [@key]  [@a; @b]  @key          citation (author-year), resolved against paper/refs.bib; the reference list is generated
    {{figure:NAME|caption|width%}}  figures/NAME.png with an auto-numbered caption;   {{fref:NAME}} -> "Figure N"
    {{table:NAME|caption}}          tables/NAME.md with an auto-numbered caption;     {{tref:NAME}} -> "Table N"
    $...$ / $$...$$                 LaTeX math (MathJax)
    TODO(final2)                    highlighted marker for text that depends on the unfinished final2 run
    TODO(owner: ...)                highlighted marker for an open question to the owner (also listed in paper/TODO.md)
    TODO(ablation)                  highlighted marker for the controlled ablation that has not been run (Section 8.4)
    TODO(vocab)                     highlighted marker for the untested vocabulary-restriction idea (Section 3.3)
    TODO(E1)                        highlighted marker for the benchmark-based pairwise comparison (Section 6.9) while its results are not yet in the paper
"""
import html
import os
import re
import subprocess
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, ".."))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# ------------------------------------------------------------------------------------------------ bibtex
def parse_bib(path):
    txt = open(path, encoding="utf-8").read()
    entries, i = {}, 0
    while True:
        m = re.search(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", txt[i:])
        if not m:
            break
        start = i + m.end()
        depth, j = 1, start
        while j < len(txt) and depth:
            depth += {"{": 1, "}": -1}.get(txt[j], 0)
            j += 1
        body = txt[start:j - 1]
        fields = {}
        k = 0
        while True:
            fm = re.search(r"(\w+)\s*=\s*", body[k:])
            if not fm:
                break
            p = k + fm.end()
            if p >= len(body):
                break
            if body[p] == "{":
                d, q = 1, p + 1
                while q < len(body) and d:
                    d += {"{": 1, "}": -1}.get(body[q], 0)
                    q += 1
                val = body[p + 1:q - 1]
            elif body[p] == '"':
                q = body.index('"', p + 1) + 1
                val = body[p + 1:q - 1]
            else:
                mm = re.match(r"[^,\n]+", body[p:])
                val = mm.group(0).strip(); q = p + len(mm.group(0))
            fields[fm.group(1).lower()] = re.sub(r"\s+", " ", val).strip()
            k = q
        entries[m.group(2)] = dict(type=m.group(1).lower(), **fields)
        i = j
    return entries


def clean_tex(s):
    s = re.sub(r"\\['`^\"~=.](\{?)([A-Za-z])\}?", lambda m: m.group(2), s)
    s = s.replace("\\&", "&").replace("{\\'e}", "e").replace("\\_", "_")
    s = re.sub(r"[{}]", "", s)
    return s


def authors_list(a):
    out = []
    for p in re.split(r"\s+and\s+", a):
        raw = p.strip()
        if raw.lower() in ("others", "and others"):
            continue
        if raw.startswith("{") and raw.endswith("}") and raw.count("{") == 1:
            out.append((clean_tex(raw).strip(), ""))
            continue
        p = clean_tex(p).strip()
        if "," in p:
            last, first = [x.strip() for x in p.split(",", 1)]
        else:
            bits = p.split()
            last, first = (bits[-1], " ".join(bits[:-1])) if bits else (p, "")
        out.append((last, first))
    return out


def cite_text(e):
    au = authors_list(e.get("author", e.get("editor", "Anon")))
    yr = e.get("year", "n.d.")
    if len(au) == 1:
        return f"{au[0][0]}, {yr}"
    if len(au) == 2:
        return f"{au[0][0]} and {au[1][0]}, {yr}"
    return f"{au[0][0]} et al., {yr}"


def ref_entry(key, e):
    au = authors_list(e.get("author", e.get("editor", "Anon")))
    names = ", ".join(f"{f} {l}".strip() if f else l for l, f in au[:6]) + (" et al." if len(au) > 6 else "")
    title = clean_tex(e.get("title", ""))
    venue = clean_tex(e.get("journal") or e.get("booktitle") or e.get("publisher") or e.get("howpublished") or "")
    yr = e.get("year", "")
    link = e.get("url") or (f"https://arxiv.org/abs/{e['eprint']}" if e.get("eprint") else "")
    s = f"{html.escape(names)} ({yr}). {html.escape(title)}."
    if venue:
        s += f" <i>{html.escape(venue)}</i>."
    if link:
        s += f' <a href="{html.escape(link)}">{html.escape(link)}</a>'
    return f'<li id="ref-{key}">{s}</li>'


# ------------------------------------------------------------------------------------------------ build
def main():
    bib = parse_bib(os.path.join(PAPER, "refs.bib"))
    sec_dir = os.path.join(PAPER, "sections")
    if os.path.isdir(sec_dir):
        parts = sorted(f for f in os.listdir(sec_dir) if f.endswith(".md"))
        gen = "\n\n".join(open(os.path.join(sec_dir, f), encoding="utf-8").read().rstrip() for f in parts) + "\n"
        open(os.path.join(PAPER, "paper.md"), "w", encoding="utf-8").write(gen)
    src = open(os.path.join(PAPER, "paper.md"), encoding="utf-8").read()

    # protect math
    math = []
    def stash(m):
        math.append(m.group(0)); return f"@@MATH{len(math)-1}@@"
    src = re.sub(r"\$\$.+?\$\$", stash, src, flags=re.S)
    src = re.sub(r"(?<![\\\w])\$(?!\s)[^$\n]+?(?<!\s)\$", stash, src)

    # figure / table numbering in order of appearance
    fig_order, tab_order = [], []
    for m in re.finditer(r"\{\{figure:([^|}]+)", src):
        if m.group(1) not in fig_order:
            fig_order.append(m.group(1))
    for m in re.finditer(r"\{\{table:([^|}]+)", src):
        if m.group(1) not in tab_order:
            tab_order.append(m.group(1))
    src = re.sub(r"\{\{fref:([^}]+)\}\}", lambda m: f"[Figure {fig_order.index(m.group(1))+1}](#fig-{m.group(1)})" if m.group(1) in fig_order else "Figure ?", src)
    src = re.sub(r"\{\{tref:([^}]+)\}\}", lambda m: f"[Table {tab_order.index(m.group(1))+1}](#tab-{m.group(1)})" if m.group(1) in tab_order else "Table ?", src)

    md_ext = markdown.Markdown(extensions=["tables", "footnotes", "attr_list", "md_in_html", "sane_lists"])
    def inline(txt):
        h = markdown.markdown(txt.replace("\\_", "\\\\_") if False else txt)
        return re.sub(r"^<p>|</p>$", "", h.strip())
    blocks = {}
    def fig(m):
        name, cap, *w = m.group(1).split("|")
        width = (w[0] if w else "100") + "%"
        n = fig_order.index(name) + 1
        key = f"@@BLOCK{len(blocks)}@@"
        blocks[key] = (f'<figure id="fig-{name}" style="width:{width}"><img src="figures/{name}.png" alt="{html.escape(name)}"/>'
                       f'<figcaption><b>Figure {n}.</b> {inline(cap)}</figcaption></figure>')
        return "\n\n" + key + "\n\n"
    src = re.sub(r"\{\{figure:([^}]+)\}\}", fig, src)
    def tab(m):
        name, cap = m.group(1).split("|", 1)
        n = tab_order.index(name) + 1
        tpath = os.path.join(PAPER, "tables", name + ".md")
        body = open(tpath, encoding="utf-8").read() if os.path.exists(tpath) else f"| missing table {name} |\n|--|\n"
        body = re.sub(r"@@MATH\d+@@", "", body) if False else body
        body = "\n".join(l for l in body.splitlines() if not l.startswith("#")) + "\n"   # table files from other agents may carry their own heading; the caption replaces it
        # math inside tables
        body = re.sub(r"(?<![\\\w])\$(?!\s)[^$\n]+?(?<!\s)\$", lambda mm: (math.append(mm.group(0)) or f"@@MATH{len(math)-1}@@"), body)
        t = markdown.Markdown(extensions=["tables"]).convert(body)
        key = f"@@BLOCK{len(blocks)}@@"
        blocks[key] = f'<div class="tablewrap" id="tab-{name}"><div class="tcap"><b>Table {n}.</b> {inline(cap)}</div>{t}</div>'
        return "\n\n" + key + "\n\n"
    src = re.sub(r"\{\{table:([^}]+)\}\}", tab, src)

    # citations
    cited = []
    def cite_group(m):
        keys = re.findall(r"@([\w:.-]+)", m.group(1))
        parts = []
        for k in keys:
            if k not in bib:
                parts.append(f'<span class="missing">?{k}</span>')
                continue
            if k not in cited:
                cited.append(k)
            parts.append(f'<a class="cite" href="#ref-{k}">{html.escape(cite_text(bib[k]))}</a>')
        return "(" + "; ".join(parts) + ")"
    src = re.sub(r"\[((?:@[\w:.-]+;?\s*)+)\]", cite_group, src)
    def cite_inline(m):
        k = m.group(1)
        if k not in bib:
            return f'<span class="missing">?{k}</span>'
        if k not in cited:
            cited.append(k)
        au = authors_list(bib[k].get("author", "Anon"))
        nm = au[0][0] + ("" if len(au) == 1 else (" and " + au[1][0] if len(au) == 2 else " et al."))
        return f'<a class="cite" href="#ref-{k}">{html.escape(nm)} ({bib[k].get("year","n.d.")})</a>'
    src = re.sub(r"(?<![\w@])@([a-z][\w:.-]*\d{4}[\w:.-]*)", cite_inline, src)

    # markers
    src = re.sub(r"TODO\(final2\)", '<span class="todo">TODO(final2)</span>', src)
    src = re.sub(r"TODO\(owner:([^)]*)\)", r'<span class="todoo">TODO(owner:\1)</span>', src)
    src = re.sub(r"TODO\(E1([^)]*)\)", r'<span class="todoa">TODO(E1\1)</span>', src)
    src = re.sub(r"TODO\(vocab([^)]*)\)", r'<span class="todoa">TODO(vocab\1)</span>', src)
    src = re.sub(r"TODO\(ablation([^)]*)\)", r'<span class="todoa">TODO(ablation\1)</span>', src)

    body = md_ext.convert(src)
    for k, v in blocks.items():
        body = body.replace(f"<p>{k}</p>", v).replace(k, v)
    for i, mth in enumerate(math):
        body = body.replace(f"@@MATH{i}@@", html.escape(mth, quote=False))
    # undo markdown mangling of escaped asterisks etc is not needed

    body = re.sub(r"<code>(.*?)</code>", lambda m: "<code>" + m.group(1).replace("\\_", "_") + "</code>", body, flags=re.S)
    # bibliography
    keys = sorted(cited, key=lambda k: (cite_text(bib[k]).lower()))
    refs = "<h2>References</h2><ol class='refs'>" + "".join(ref_entry(k, bib[k]) for k in keys) + "</ol>"
    body = body.replace("<!--REFERENCES-->", refs) if "<!--REFERENCES-->" in body else body + refs

    css = open(os.path.join(HERE, "paper.css"), encoding="utf-8").read()
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Small but Sabi: How Far Can Reinforcement Learning Take a 270M Translator into 77 African Languages?</title><style>{css}</style>
<script>window.MathJax={{tex:{{inlineMath:[['$','$']],displayMath:[['$$','$$']]}},svg:{{fontCache:'global'}}}};</script>
<script async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.js"></script></head><body><div class="paper">{body}</div></body></html>"""
    out_html = os.path.join(PAPER, "paper.html")
    open(out_html, "w", encoding="utf-8").write(doc)
    missing = [k for k in re.findall(r'class="missing">\?([^<]+)', doc)]
    print("html written;", len(cited), "references cited;", "MISSING citation keys:", sorted(set(missing)) or "none")
    if "--no-pdf" in sys.argv:
        return
    pdf = os.path.join(PAPER, "paper.pdf")
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=30000", "--allow-file-access-from-files",
           f"--print-to-pdf={pdf}", "file://" + out_html]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    print("pdf:", pdf, os.path.getsize(pdf) if os.path.exists(pdf) else "NOT BUILT", r.stderr[-300:] if r.returncode else "")


if __name__ == "__main__":
    main()
