#!/usr/bin/env python3
"""Tables of reports/eval_afriscience_document_level_2026-11-04.md.   python build_doc_tables.py --models a0 a05 rl2c [sci_rl_best sci_rl_final] > tables.md
Inputs in this directory: documents_eval_<model>.json (scripts/eval_afriscience_documents.py), doc_neural.json (AfriCOMET-STL + SSA-COMET-MTL, reference-based, per language), ssa_qe.json
(SSA-COMET-QE, reference-free), paper_doc_rows.json (the paper's released document-level test metrics, English -> 6 languages, unweighted mean)."""
import argparse, json, os, statistics as st

H = os.path.dirname(os.path.abspath(__file__))
LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
LABEL = {"a0": "385M start (alpha 0, no science SFT)", "a05": "385M science SFT (alpha 0.5)", "rl2c": "old 270M RL-final (rl2c, first research)",
         "sci_rl_best": "385M science RL (best)", "sci_rl_final": "385M science RL (final)", "long_rl_best": "385M long-document RL (best, step 50)", "long_rl_final": "385M long-document RL (final, step 80)"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True); a = ap.parse_args()
    ev = {m: json.load(open(f"{H}/documents_eval_{m}.json")) for m in a.models}
    nj = json.load(open(f"{H}/doc_neural.json")); qe = json.load(open(f"{H}/ssa_qe.json")).get
    qd = json.load(open(f"{H}/ssa_qe.json"))
    pap = json.load(open(f"{H}/paper_doc_rows.json"))
    out = []; P = out.append
    mean = lambda xs: sum(xs) / len(xs)

    def neural(m, mode, key):
        v = [nj[f"{m}/{mode}/{l}"][key] for l in LANGS if f"{m}/{mode}/{l}" in nj and key in nj[f"{m}/{mode}/{l}"]]
        return mean(v) if len(v) == 6 else None

    def qev(m, mode, key):
        d = qd.get(f"doc/{m}", {})
        v = [d[f"{mode}/{l}"][key] for l in LANGS if f"{mode}/{l}" in d and key in d[f"{mode}/{l}"]]
        return mean(v) if len(v) == 6 else None

    f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"
    P("## Table 1. Document level, 168 test documents (28 papers x 6 languages), macro over the 6 languages\n")
    P("**Headline = SSA-COMET-QE on 4-sentence WINDOWS (the honest document score), then plain chrF.** The COMET encoders only read the first 512 tokens, so the *concatenated* document score (`concat`) sees only the first part of a ~1,000-token document and is NOT a full-document score; it is shown only because it is the literal 'metric on the concatenated document'. Windows need a sentence-aligned hypothesis, so they exist for the `chunked` mode only; for `whole` mode (unaligned output) only the truncated `concat` value exists and chrF / failure statistics are the real evidence.\n")
    P("| model | mode | **SSA-COMET-QE window** | chrF | AfriCOMET window | SSA-MTL(ref) window | SSA-QE concat (first 512 tokens) | AfriCOMET concat | SSA-MTL(ref) concat | chrF++ | BLEU | spBLEU | finished | loop | length ratio | tail coverage | concat COMET x100 - chrF |")
    P("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in a.models:
        for mode in ("whole", "chunked"):
            mac = ev[m]["scores"][mode]["macro"]; q = qev(m, mode, "ssa_qe_concat")
            gap = None if q is None else q * 100 - mac["chrf"]
            P(f"| {LABEL.get(m, m)} | {mode} | " + (f"**{f(qev(m, mode, 'ssa_qe_window'))}**" if mode == "chunked" else "-") + f" | {mac['chrf']:.1f} | "
              + (f"{f(neural(m, mode, 'africomet_window'))} | {f(neural(m, mode, 'ssa_comet_mtl_window'))}" if mode == "chunked" else "- | -")
              + f" | {f(q)} | {f(neural(m, mode, 'africomet_concat'))} | {f(neural(m, mode, 'ssa_comet_mtl_concat'))} | {mac['chrf_pp']:.1f} | {mac['bleu']:.1f} | {mac['spbleu']:.1f} | {mac['finished_rate']:.2f} | {mac['loop_rate']:.2f} | {mac['length_ratio']:.2f} | {mac['tail_coverage']:.2f} | {f(gap, 1)} |")
    P("")
    P("`whole` = the document in one generation (greedy, 4,096 new tokens); `chunked` = sentence by sentence (greedy) + best-of-4 repair of unfinished sentences, sentences joined by single spaces. finished = share of documents (whole) or of documents with every sentence finished (chunked); loop = repetition detector (rl/guards); tail coverage = chrF recall of the last third of the reference against the hypothesis (does the output reach the end of the paper); the last column is the paper's degeneration flag computed with the truncated concat value (a COMET far above chrF means fluent but off-target output).\n")
    P("## Table 2. Chunked mode: sentence coverage and repairs\n")
    P("| model | sentences | unfinished before repair (repaired) | unfinished after | sentence coverage |")
    P("|---|---:|---:|---:|---:|")
    for m in a.models:
        pl = ev[m]["scores"]["chunked"]["per_language"]
        P(f"| {LABEL.get(m, m)} | 5,820 | {sum(v['repaired_sentences'] for v in pl.values()):.0f} repaired | {sum(v['unfinished_after_repair'] for v in pl.values()):.0f} | {mean([v['sentence_coverage'] for v in pl.values()]):.3f} |")
    P("")
    P("## Table 3. Per language (chunked mode): chrF / SSA-COMET-QE window\n")
    P("| model | " + " | ".join(LANGS) + " |"); P("|---|" + "---:|" * 6)
    for m in a.models:
        pl = ev[m]["scores"]["chunked"]["per_language"]
        P(f"| {LABEL.get(m, m)} | " + " | ".join(f"{pl[l]['chrf']:.1f} / " + (f"{qd['doc/' + m]['chunked/' + l]['ssa_qe_window']:.3f}" if f"doc/{m}" in qd else "-") for l in LANGS) + " |")
    P("")
    P("## Table 4. The paper's released document-level results (English -> the same 6 languages, test, 28 documents per language, unweighted mean; dataset config `metrics`)\n")
    P("| system (model / experiment / prompt) | SSA-COMET | chrF | BLEU |"); P("|---|---:|---:|---:|")
    for r in pap[:6]:
        P(f"| {r['system'].replace('|None', '').replace('llm_doc|', '')} | {r['ssa_comet']:.3f} | {r['chrf']:.1f} | {r['bleu']:.1f} |")
    tg = [r for r in pap if r["system"].startswith("translategemma")][:1]
    for r in tg:
        P(f"| {r['system'].replace('|None', '').replace('llm_doc|', '')} | {r['ssa_comet']:.3f} | {r['chrf']:.1f} | {r['bleu']:.1f} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
