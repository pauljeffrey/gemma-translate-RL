#!/usr/bin/env python3
"""Long-context benchmark tables and figures (agent LC1, 2026-10-23).

    python paper/scripts/make_longctx.py [tables|figs]        (default: both; needs numpy + matplotlib for figs)

Inputs: reports/data/longctx/{macro,macro_common,cells,pair_mode,pair_mode_common}.csv  (written on the GPU box by /workspace/jobs/lc1/lc_agg.py, copied here),
        reports/data/longctx/{bench_summary.json,models.json}.
Outputs: paper/tables/t_longctx_*.md, paper/figures/fig14_longctx_chrf.*, fig15_longctx_failure.*, fig16_longctx_pseudoref.*
`macro_common` = 6 core/human-reference languages (hau yor swh amh zul ibo; the NLLB whole-document runs were limited to these for compute) and 5 AfriScience languages (amh hau lug yor zul), all supported by TranslatePsy-AfriSLM, so that every system is scored on the same languages;
`macro` = every system on all languages it was run on (n_lang column).
"""
import csv
import json
import os
import sys
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RDATA, TAB, FIG, OI  # noqa: E402

LC = os.path.join(RDATA, "longctx")
MODELS = OrderedDict([  # key: (label, family, colour, nllb_lineage_flag)
    ("sft", ("SabiYarn-270M SFT-final", "ours", OI["sky"], "possible")),
    ("rl", ("SabiYarn-270M RL-final (2c)", "ours", OI["blue"], "possible")),
    ("a2w", ("SabiYarn-270M science start (A2W a0.7)", "ours", "#1B4F8A", "possible")),
    ("live", ("SabiYarn-270M live run (step 250)", "ours", OI["black"], "possible")),
    ("final2", ("SabiYarn-270M final run (final2, step 401)", "ours", "#B8860B", "possible")),
    ("nllb600m", ("NLLB-200-distilled-600M", "nllb", OI["orange"], "same family")),
    ("nllb3b3", ("NLLB-200-3.3B (CT2 int8)", "nllb", OI["verm"], "self")),
    ("tpsy08", ("TranslatePsy-AfriSLM-0.8B", "tpsy", OI["green"], "distilled from NLLB-3.3B")),
    ("tpsy2b", ("TranslatePsy-AfriSLM-2B", "tpsy", "#005F46", "distilled from NLLB-3.3B")),
    ("qwen08", ("Qwen3.5-0.8B (zero-shot)", "zs", OI["pink"], "no")),
    ("qwen2b", ("Qwen3.5-2B (zero-shot)", "zs", "#8E4A72", "no")),
    ("gemma1b", ("Gemma-3-1B-it (zero-shot)", "zs", OI["grey"], "no")),
])
CORE_BINS = ["b256", "b512", "b1024", "b2048", "b4096"]
HUM_BINS = ["sm", "st512", "st1024", "st2048"]
SCI_BINS = ["sci_dev", "sci_test"]
BIN_LABEL = {"b256": "b256 (control)", "b512": "b512", "b1024": "b1024", "b2048": "b2048", "b4096": "b4096", "sm": "SmolDoc short", "st512": "stitched ~512",
             "st1024": "stitched ~1,024", "st2048": "stitched ~2,048", "sci_dev": "AfriScience dev", "sci_test": "AfriScience test"}


def read(name):
    p = os.path.join(LC, name + ".csv")
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(open(p, encoding="utf-8")))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def index(rows, keys):
    return {tuple(r[k] for k in keys): r for r in rows}


def smode(idx, model, setn, binn, ref):
    """the sentence-by-sentence mode of a model: sentfix (ours, with repair) else sent"""
    for m in ("sentfix", "sent"):
        if (model, m, setn, binn, ref) in idx:
            return m
    return None


def cell(r, col="chrf", nd=1, ci=True):
    if r is None:
        return "n/a"
    v = f(r.get(col))
    if v != v:
        return "n/a"
    s = f"{v:.{nd}f}"
    if ci and (col + "_lo") in r and f(r.get(col + "_lo")) == f(r.get(col + "_lo")):
        s += f" [{f(r[col + '_lo']):.{nd}f}, {f(r[col + '_hi']):.{nd}f}]"
    return s


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join([":---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def write(name, text):
    with open(os.path.join(TAB, name), "w", encoding="utf-8") as fh:
        fh.write(text)
    print("table", name)


def models_present(idx):
    return [m for m in MODELS if any(k[0] == m for k in idx)]


def build_tables():
    macro = read("macro_common") or read("macro")
    macro_all = read("macro")
    cells = read("cells")
    pairs = read("pair_mode_common") or read("pair_mode")
    idx = index(macro, ["model", "mode", "set", "bin", "ref"])
    idx_all = index(macro_all, ["model", "mode", "set", "bin", "ref"])
    present = models_present(idx)
    MODE_NAME = {"whole": "whole document", "sentfix": "sentence-by-sentence + repair", "sent": "sentence-by-sentence"}

    def rowfor(m, mode, setn, bins, ref, col="chrf", nd=1, ci=True):
        return [cell(idx.get((m, mode, setn, b, ref)), col, nd, ci) for b in bins]

    # ---- main table 1: chrF++ against human references
    for col, nm, title in (("chrf", "chrf", "chrF++"), ("spbleu", "spbleu", "spBLEU (flores200)")):
        hdr = ["model", "mode"] + [BIN_LABEL[b] for b in HUM_BINS + SCI_BINS]
        rows = []
        for m in present:
            for mode in ("whole", smode(idx, m, "human", "sm", "human") or smode(idx, m, "sci", "sci_dev", "human")):
                if mode is None:
                    continue
                rows.append([MODELS[m][0], MODE_NAME[mode]] + rowfor(m, mode, "human", HUM_BINS, "human", col) + rowfor(m, mode, "sci", SCI_BINS, "human", col))
        write(f"t_longctx_human_{nm}.md",
              f"### Table LC-H. {title} against HUMAN references, macro over the common languages (6 SmolDoc languages hau yor swh amh zul ibo; 5 AfriScience languages amh hau lug yor zul), 95 % bootstrap CI over documents\n\n"
              + md(hdr, rows) + "\nSmolDoc bins: 60 / 38 / 33 / 16 English documents; AfriScience dev 25 and test 28 papers. Stitched = unrelated clean SmolDoc documents concatenated (tests length, not discourse). "
              "`n/a` = the model was not run on that bin. AfriScience test was scored for all five SabiYarn checkpoints including the live-run step 250 (an intermediate checkpoint, no selection used it) and the final checkpoint (step 401, the last step of the same run, no selection used it): if the final test read must stay pristine, treat the live row's test column as dev-like. A2W and the live run were trained on the AfriScience TRAIN split (in-domain advantage); the live run selects checkpoints on dev. Document-level pooled chrF++ (about 10 points above the sentence-level numbers of the earlier AfriScience reports).\n")

    # ---- main table 2: core documents against the NLLB-3.3B pseudo-reference
    hdr = ["model", "mode", "NLLB-lineage of the model"] + [BIN_LABEL[b] for b in CORE_BINS]
    rows = []
    for m in present:
        for mode in ("whole", smode(idx, m, "core", "b512", "nllb")):
            if mode is None or m == "nllb3b3":
                continue
            rows.append([MODELS[m][0], MODE_NAME[mode], MODELS[m][3]] + rowfor(m, mode, "core", CORE_BINS, "nllb"))
    write("t_longctx_core_nllbref.md",
          "### Table LC-N. chrF++ of AfriLongDoc core documents against the NLLB-3.3B sentence-by-sentence pseudo-reference (6 languages hau yor swh amh zul ibo), 95 % bootstrap CI over documents\n\n" + md(hdr, rows) +
          "\nBIASED reference: NLLB-3.3B is scored against itself nowhere (row omitted); NLLB-600M is the same model family, TranslatePsy-AfriSLM was distilled from NLLB-3.3B translations, and the "
          "SabiYarn models may share NLLB-derived training data (R6 found NLLB-referenced scores 1-16 chrF++ above human-referenced ones). Use this table for the ORDER of systems/modes, not for absolute quality. "
          "Documents are 20 per bin (10 at b4096), the same English documents for every language and model; NLLB whole-document runs have no b4096.\n")

    # ---- failure table: finished / loop / coverage / length ratio / LID
    for setn, ref, bins, nm in (("core", "nllb", ["b512", "b1024", "b2048"], "core"), ("human", "human", ["st1024", "st2048"], "human"), ("sci", "human", ["sci_dev", "sci_test"], "sci")):
        hdr = ["model", "mode"]
        for b in bins:
            hdr += [f"{BIN_LABEL[b]}: finished / loop / coverage / len. ratio / LID"]
        rows = []
        for m in present:
            for mode in ("whole", smode(idx, m, setn, bins[0], ref)):
                if mode is None:
                    continue
                r = [MODELS[m][0], MODE_NAME[mode]]
                for b in bins:
                    x = idx.get((m, mode, setn, b, ref)) or idx.get((m, mode, setn, b, "none"))
                    r.append("n/a" if x is None else f"{f(x['finished']):.2f} / {f(x['loop']):.2f} / {f(x['cov_sents']):.2f} / {f(x['length_ratio']):.2f} / {f(x['lid_sent']):.2f}")
                rows.append(r)
        write(f"t_longctx_failure_{nm}.md",
              f"### Table LC-F-{nm}. Failure metrics, macro over languages ({setn} set)\n\nfinished = stopped at its own end token / before the length limit; loop = word-cycle repetition absent from the reference; "
              "coverage = share of source sentences with an aligned, non-copied output sentence (alignment plan of the long-document reward v2); len. ratio = hypothesis / reference characters; "
              "LID = share of output sentences (>= 25 characters) that GlotLID labels as the target language.\n\n" + md(hdr, rows))

    # ---- sentence-by-sentence minus whole-document (paired bootstrap)
    pi = defaultdict(dict)
    for r in pairs:
        if r["metric"] == "chrf":
            pi[(r["model"], r["set"], r["bin"], r["ref"])] = r
    hdr = ["model", "set"] + ["bin", "ref", "delta chrF++ (sentence - whole) [95 % CI]"]
    rows = []
    for (m, setn, binn, ref), r in sorted(pi.items(), key=lambda kv: (list(MODELS).index(kv[0][0]) if kv[0][0] in MODELS else 99, kv[0][1], kv[0][2])):
        if ref == "none" or (setn == "core" and ref != "nllb") or (setn != "core" and ref != "human"):
            continue
        rows.append([MODELS.get(m, (m,))[0], setn, BIN_LABEL.get(binn, binn), ref, f"{f(r['delta']):+.1f} [{f(r['lo']):+.1f}, {f(r['hi']):+.1f}]"])
    write("t_longctx_sentence_minus_whole.md", "### Table LC-D. Sentence-by-sentence wrapper minus whole-document input, same model, same documents (paired document bootstrap)\n\n" + md(hdr, rows))

    # ---- pseudo-reference validation: NLLB-3.3B (sentence-by-sentence) against the human references, per language
    ci_ = index(cells, ["model", "mode", "set", "bin", "lang", "ref"])
    langs = sorted({k[4] for k in ci_ if k[0] == "nllb3b3"})
    if langs:
        hdr = ["language", "NLLB-3.3B vs human (mean over SmolDoc bins)", "NLLB-3.3B vs human (AfriScience dev+test)", "best SabiYarn (live / RL-final)", "TranslatePsy-0.8B", "NLLB-600M",
               "NLLB-3.3B minus best SabiYarn", "pseudo-reference verdict"]
        hb = [("human", b) for b in HUM_BINS]
        sb = [("sci", b) for b in SCI_BINS]

        def mean_over(m, mode_pref, bins, l):
            v = []
            for setn, b in bins:
                for md_ in mode_pref:
                    x = ci_.get((m, md_, setn, b, l, "human"))
                    if x:
                        v.append(f(x["chrf"])); break
            return sum(v) / len(v) if v else float("nan")

        rows = []
        for l in langs:
            nh, ns = mean_over("nllb3b3", ("sent",), hb, l), mean_over("nllb3b3", ("sent",), sb, l)
            ref = nh if nh == nh else ns
            ours = [mean_over(m, ("sentfix", "sent"), hb if nh == nh else sb, l) for m in ("live", "rl")]
            ours = [o for o in ours if o == o]
            tp = mean_over("tpsy08", ("sent",), hb if nh == nh else sb, l)
            n6 = mean_over("nllb600m", ("sent",), hb if nh == nh else sb, l)
            best = max(ours) if ours else float("nan")
            d = ref - best
            verdict = "usable (>= 2 above SabiYarn)" if d >= 2 else "marginal (0-2 above)" if d >= 0 else "NOT above SabiYarn: do not use as a quality yardstick"
            fm = lambda v: "-" if v != v else f"{v:.1f}"
            rows.append([l, fm(nh), fm(ns), fm(best), fm(tp), fm(n6), "-" if d != d else f"{d:+.1f}", verdict])
        write("t_longctx_pseudoref_validation.md",
              "### Table LC-V. Is the NLLB-3.3B pseudo-reference trustworthy? chrF++ of NLLB-3.3B (sentence-by-sentence, beam 4) against the HUMAN references, per language, next to the systems it is used to judge\n\n" + md(hdr, rows) +
              "\nAll systems in sentence-by-sentence mode; chrF++ against ONE human reference rarely exceeds 60-65, so 'usable' is a relative statement: a pseudo-reference can rank systems that are clearly below it. "
              "Rule used in the report: NLLB-3.3B counts as a better-than-ours yardstick for a language only when it beats the best SabiYarn model by >= 2 chrF++ on the human references.\n")

    # ---- agreement with NLLB: chrF++ vs NLLB minus chrF++ vs human, same documents
    hdr = ["model", "mode", "stitched ~1,024: vs human", "vs NLLB-3.3B", "gap", "AfriScience dev: vs human", "vs NLLB-3.3B", "gap"]
    rows = []
    for m in present:
        if m == "nllb3b3":
            continue
        mode = smode(idx, m, "human", "st1024", "human")
        if mode is None:
            continue
        r = [MODELS[m][0], MODE_NAME[mode]]
        for setn, b in (("human", "st1024"), ("sci", "sci_dev")):
            h = idx.get((m, mode, setn, b, "human")); n = idx.get((m, mode, setn, b, "nllb"))
            r += [cell(h, "chrf", 1, False), cell(n, "chrf", 1, False), "n/a" if not (h and n) else f"{f(n['chrf']) - f(h['chrf']):+.1f}"]
        rows.append(r)
    write("t_longctx_agreement_nllb.md", "### Table LC-A. Agreement with NLLB-3.3B versus quality against humans (same documents, sentence-by-sentence mode)\n\n" + md(hdr, rows) +
          "\ngap = chrF++ against the NLLB-3.3B outputs minus chrF++ against the human references. A large positive gap = the system resembles NLLB more than it resembles a human translator.\n")

    # ---- per-language chrF++ for the key bins
    keym = [m for m in ("rl", "a2w", "live", "final2", "nllb600m", "nllb3b3", "tpsy08") if m in present]
    for setn, binn, nm in (("human", "st2048", "st2048"), ("sci", "sci_test", "scitest")):
        langs = sorted({k[4] for k in ci_ if k[2] == setn and k[3] == binn and k[5] == "human"})
        hdr = ["language"] + [f"{MODELS[m][0]} {'whole' if md_ == 'whole' else 'sent.'}" for m in keym for md_ in ("whole", "sent")]
        rows = []
        for l in langs:
            r = [l]
            for m in keym:
                for md_ in ("whole", "sent"):
                    mode = "whole" if md_ == "whole" else ("sentfix" if (m, "sentfix", setn, binn, l, "human") in ci_ else "sent")
                    x = ci_.get((m, mode, setn, binn, l, "human"))
                    r.append("-" if x is None else f"{f(x['chrf']):.1f}")
            rows.append(r)
        write(f"t_longctx_perlang_{nm}.md", f"### Table LC-L-{nm}. chrF++ against human references per language ({BIN_LABEL[binn]})\n\n" + md(hdr, rows))

    # ---- benchmark definition and models
    bs = os.path.join(LC, "bench_summary.json")
    if os.path.exists(bs):
        b = json.load(open(bs))
        hdr = ["bin", "documents", "median Gemma tokens (min-max)", "languages", "reference"]
        gt = b["gemma_tokens_per_bin"]
        refname = {**{x: "NLLB-3.3B sentence-by-sentence (pseudo)" for x in CORE_BINS}, **{x: "SmolDoc professional translation (human)" for x in HUM_BINS},
                   **{x: "AfriScience-MT professional translation (human)" for x in SCI_BINS}}
        nl = {**{x: 11 + 1 for x in CORE_BINS}, **{x: 12 for x in HUM_BINS}, **{x: 6 for x in SCI_BINS}}
        rows = []
        for k in CORE_BINS + HUM_BINS + SCI_BINS:
            if k in gt:
                g = gt[k]
                rows.append([BIN_LABEL[k], g["n"], f"{g['median']:.0f} ({g['min']}-{g['max']})", nl[k], refname[k]])
        write("t_longctx_benchmark.md", "### Table LC-B. LongCtx-MT benchmark definition (English -> African languages)\n\n" + md(hdr, rows))
    mj = os.path.join(LC, "models.json")
    if os.path.exists(mj):
        mm = json.load(open(mj))
        hdr = ["model", "params", "documented max context", "documented limits for documents", "fits b512 / b1024 / b2048 / AfriScience", "languages run"]
        write("t_longctx_models.md", "### Table LC-M. Systems compared\n\n" + md(hdr, [[r["label"], r["params"], r["context"], r["limits"], r["fits"], r["langs"]] for r in mm]))


def build_figs():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import setup_matplotlib, save
    setup_matplotlib()
    macro = read("macro_common") or read("macro")
    idx = index(macro, ["model", "mode", "set", "bin", "ref"])
    present = models_present(idx)
    panels = [("core", "nllb", CORE_BINS, "A  core docs vs NLLB-3.3B (pseudo)"), ("human", "human", HUM_BINS, "B  SmolDoc vs human refs"), ("sci", "human", SCI_BINS, "C  AfriScience vs human refs")]

    def xs(setn, ref, bins, m, mode):
        pts = []
        for b in bins:
            r = idx.get((m, mode, setn, b, ref))
            if r:
                pts.append((f(r["src_tokens_mean"]), r))
        return sorted(pts, key=lambda t: t[0])

    def pick_mode(m, kind, setn, ref):
        if kind == "whole":
            return "whole"
        for md_ in ("sentfix", "sent"):
            if any(k[0] == m and k[1] == md_ for k in idx):
                return md_
        return None

    # ---- fig14: chrF++ vs document length; rows = whole / sentence, columns = benchmark sets
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), sharey="row")
    for i, kind in enumerate(("whole", "sent")):
        for j, (setn, ref, bins, title) in enumerate(panels):
            ax = axes[i][j]
            for m in present:
                if m == "nllb3b3" and ref == "nllb":
                    continue
                mode = pick_mode(m, kind, setn, ref)
                pts = xs(setn, ref, bins, m, mode) if mode else []
                if not pts:
                    continue
                x = [p[0] for p in pts]; y = [f(p[1]["chrf"]) for p in pts]
                lo = [f(p[1]["chrf_lo"]) for p in pts]; hi = [f(p[1]["chrf_hi"]) for p in pts]
                col = MODELS[m][2]
                ls = "-" if MODELS[m][1] == "ours" else "--" if MODELS[m][1] in ("nllb", "tpsy") else ":"
                if len(x) == 1:
                    ax.errorbar(x, y, yerr=[[y[0] - lo[0]], [hi[0] - y[0]]], fmt="o", color=col, ms=3.5, lw=1)
                else:
                    ax.plot(x, y, ls, color=col, marker="o", ms=2.8, lw=1.2)
                    ax.fill_between(x, lo, hi, color=col, alpha=0.10, lw=0)
            ax.set_xscale("log", base=2)
            ax.set_xticks([128, 256, 512, 1024, 2048, 4096]); ax.set_xticklabels(["128", "256", "512", "1k", "2k", "4k"])
            if setn == "sci":
                ax.set_xscale("linear"); ax.set_xlim(780, 1020); ax.set_xticks([800, 900, 1000]); ax.set_xticklabels(["800", "900", "1,000"])
            if i == 0:
                ax.set_title(title, fontsize=7.2, loc="left")
            if setn != "sci":
                ax.set_xlim(100, 5000)
            if j == 0:
                ax.set_ylabel("whole-document input\nchrF++" if i == 0 else "sentence-by-sentence\nchrF++")
            ax.set_ylim(0, 82)
    handles = [plt.Line2D([0], [0], color=MODELS[m][2], lw=1.5, ls="-" if MODELS[m][1] == "ours" else "--" if MODELS[m][1] in ("nllb", "tpsy") else ":", label=MODELS[m][0]) for m in present]
    fig.supxlabel("English source tokens (Gemma-3 tokenizer)", fontsize=8, y=0.02)
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=6.2, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig14_longctx_chrf")

    # ---- fig15: failure metrics of whole-document input vs length (core + human sets pooled by their source length)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True)
    mets = [("finished", "finished rate"), ("loop", "loop rate"), ("cov_sents", "source-sentence coverage")]
    for i, kind in enumerate(("whole", "sent")):
        for j, (k, lab) in enumerate(mets):
            ax = axes[i][j]
            for m in present:
                mode = pick_mode(m, kind, "core", "nllb")
                pts = []
                for setn, ref, bins, _ in panels[:2]:
                    for b in bins:
                        r = idx.get((m, mode, setn, b, ref)) or idx.get((m, mode, setn, b, "none"))
                        if r and setn == "core":
                            pts.append((f(r["src_tokens_mean"]), f(r[k])))
                pts.sort()
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-" if MODELS[m][1] == "ours" else "--" if MODELS[m][1] in ("nllb", "tpsy") else ":", color=MODELS[m][2], marker="o", ms=2.6, lw=1.2)
            ax.set_xscale("log", base=2); ax.set_xticks([128, 256, 512, 1024, 2048, 4096]); ax.set_xticklabels(["128", "256", "512", "1k", "2k", "4k"])
            ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title(lab, fontsize=7.5, loc="left")
            if j == 0:
                ax.set_ylabel("whole document" if i == 0 else "sentence-by-sentence")
    fig.supxlabel("English source tokens (core documents)", fontsize=8, y=0.02)
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=6.2, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig15_longctx_failure")

    # ---- fig16: pseudo-reference validation
    cells = read("cells")
    ci_ = index(cells, ["model", "mode", "set", "bin", "lang", "ref"])
    langs = sorted({k[4] for k in ci_ if k[0] == "nllb3b3" and k[2] == "human" and k[3] == "st1024"})
    if langs:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
        ax = axes[0]
        v = [f(ci_[("nllb3b3", "sent", "human", "st1024", l, "human")]["chrf"]) for l in langs]
        order = np.argsort(v)
        ax.barh([langs[o] for o in order], [v[o] for o in order], color=OI["verm"], height=0.6)
        for yy, o in enumerate(order):
            ax.text(v[o] + 0.5, yy, f"{v[o]:.0f}", va="center", fontsize=6)
        ax.set_xlabel("chrF++ of NLLB-3.3B vs human references\n(stitched ~1,024-token SmolDoc documents)"); ax.set_xlim(0, 80)
        ax.set_title("A  Pseudo-reference quality per language", fontsize=7.5, loc="left")
        ax = axes[1]
        for m, sym in (("rl", "o"), ("nllb600m", "s"), ("tpsy08", "^")):
            mode = "sentfix" if m in ("rl",) else "sent"
            xs_, ys_ = [], []
            for l in langs:
                h = ci_.get((m, mode, "human", "st1024", l, "human")); n = ci_.get((m, mode, "human", "st1024", l, "nllb"))
                if h and n:
                    xs_.append(f(h["chrf"])); ys_.append(f(n["chrf"]))
            if xs_:
                ax.scatter(xs_, ys_, marker=sym, s=14, color=MODELS[m][2], label=MODELS[m][0], alpha=0.85)
        ax.plot([20, 80], [20, 80], color="#999", lw=0.7)
        ax.set_xlabel("chrF++ vs human references"); ax.set_ylabel("chrF++ vs NLLB-3.3B outputs")
        ax.set_title("B  Agreement with NLLB exceeds quality vs humans", fontsize=7.5, loc="left")
        ax.legend(fontsize=6)
        fig.tight_layout()
        save(fig, "fig16_longctx_pseudoref")


if __name__ == "__main__":
    what = sys.argv[1:] or ["tables", "figs"]
    if "tables" in what:
        build_tables()
    if "figs" in what:
        build_figs()
