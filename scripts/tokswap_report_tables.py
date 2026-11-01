#!/usr/bin/env python3
"""tokswap_report_tables.py -- turn the raw test outputs of the vocabulary-swap build into the markdown tables of reports/token_swap_build_2026-10-25.md.

Input dir (default reports/data/tokswap_2026-10-25/tests): identity_<variant>.json, fertility.csv, sanity_<variant>.json
Output: <dir>/tables.md  (also printed)
"""
import argparse, csv, glob, json, os, collections

PROJECT_ISO = None


def load_project_isos(names_path):
    return set(json.load(open(names_path, encoding="utf-8")))


def identity_tables(d, project):
    out = []
    files = sorted(glob.glob(os.path.join(d, "identity_*.json")), key=lambda p: (("fon" in p) + 2 * ("both" in p), p))
    out.append("### Identity test (token-id sequences, original vs swapped tokenizer)\n")
    out.append("| variant | sentences tested (all sets) | plain: n | plain: differences | special-script: n | special: changed | other-freed-script: n | freed-script: changed | vocab strings tested | vocab-string diffs (plain) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    per_proj = []
    fon_rows = {}
    for p in files:
        v = os.path.basename(p)[len("identity_"):-5]
        j = json.load(open(p, encoding="utf-8"))
        t = j["totals"]
        n_all = sum(t.get(k, 0) for k in ("n_plain", "n_special", "n_freed_script"))
        out.append(f"| {v} | {n_all} | {t.get('n_plain',0)} | **{t.get('diff_plain',0)}** | {t.get('n_special',0)} | {t.get('diff_special',0)} | {t.get('n_freed_script',0)} | {t.get('diff_freed_script',0)} | "
                   f"{j['vocab_strings']['kept_token_strings_tested']} | {j['vocab_strings']['diff_plain']} |")
        # project-language subset
        npl = dpl = ns = 0
        langs = collections.defaultdict(lambda: [0, 0, 0, 0])
        for k, c in j["per_set"].items():
            if k.startswith("flores_"):
                iso = k.split("_")[2]
                if iso in project or iso in ("eng", "fra", "arb", "ara"):
                    npl += c.get("n_plain", 0); dpl += c.get("diff_plain", 0)
                langs[iso][0] += c.get("n_special", 0); langs[iso][1] += c.get("diff_special", 0)
                langs[iso][2] += c.get("n_plain", 0) + c.get("n_freed_script", 0); langs[iso][3] += c.get("diff_plain", 0)
        per_proj.append((v, npl, dpl))
        fon_rows[v] = langs
    out.append("\nProject languages only (the 85 language codes of cpu_translate/language_names.json that exist in FLORES+, plus eng, fra, arb), dev + devtest, sentences with NO special-script character:\n")
    out.append("| variant | sentences | differences |\n|---|---|---|")
    for v, n, dd in per_proj:
        out.append(f"| {v} | {n} | **{dd}** |")
    # which languages are touched by the Fon letters
    for v, langs in fon_rows.items():
        if v.startswith("fon") or v.startswith("both"):
            rows = [(iso, c[0], c[1]) for iso, c in langs.items() if c[0] > 0]
            rows.sort(key=lambda r: -r[1])
            out.append(f"\nSentences containing a Fon-specific letter (ɖ ɛ ɔ ŋ, upper case too) and how many changed, variant {v} (dev + devtest, per ISO code):\n")
            out.append("| ISO | sentences with the letters | changed |\n|---|---|---|")
            for iso, n, c in rows:
                out.append(f"| {iso}{' (project language)' if iso in project else ''} | {n} | {c} |")
            break
    return "\n".join(out)


def fertility_tables(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = []
    langs = []
    for r in rows:
        if r["lang"] not in langs:
            langs.append(r["lang"])
    variants = []
    for r in rows:
        if r["variant"] not in variants:
            variants.append(r["variant"])
    for split in ("dev", "devtest"):
        out.append(f"\n### Fertility, FLORES+ {split}{' (MEASUREMENT ONLY, never used to train or choose)' if split == 'devtest' else ' (used for choosing N)'}: mean tokens per sentence (tokens per word) [change vs original]\n")
        out.append("| language | original | " + " | ".join(v for v in variants if v != "original") + " |")
        out.append("|---|---|" + "---|" * (len(variants) - 1))
        for lg in langs:
            rr = {r["variant"]: r for r in rows if r["lang"] == lg and r["split"] == split}
            if not rr:
                continue
            o = rr["original"]
            cells = [f"{float(o['tok_per_sent_mean']):.1f} ({float(o['tok_per_word']):.2f})"]
            for v in variants:
                if v == "original":
                    continue
                r = rr[v]
                ch = float(r["tokens"]) / float(o["tokens"]) - 1
                cells.append(f"{float(r['tok_per_sent_mean']):.1f} ({float(r['tok_per_word']):.2f}) [{ch*100:+.1f}%]")
            out.append(f"| {lg} | " + " | ".join(cells) + " |")
    out.append("\nByte-fallback share of output tokens, devtest (original -> variants):\n")
    out.append("| language | " + " | ".join(variants) + " |\n|---|" + "---|" * len(variants))
    for lg in langs:
        rr = {r["variant"]: r for r in rows if r["lang"] == lg and r["split"] == "devtest"}
        if lg.split("_")[0] in ("amh", "tir", "fon"):
            out.append(f"| {lg} | " + " | ".join(f"{float(rr[v]['byte_fallback_share'])*100:.2f}%" for v in variants) + " |")
    return "\n".join(out)


def sanity_tables(d):
    out = ["\n### Model sanity (CPU, fp32 compute from bf16 weights; original SFT-final cast to bf16 vs swapped / control)\n"]
    out.append("Loss: teacher-forced NLL of the reference under the exact prompt; `tok` = target tokens per sentence (orig -> swapped); nll/token and nll/char (chars of '{iso} {text}' + 1).\n")
    out.append("| checkpoint | language | n | identical tokenisation | tok/sent orig -> swapped | nll/token orig -> swapped | nll/char orig -> swapped | max abs sentence-NLL diff, identically tokenised (full softmax) |")
    out.append("|---|---|---|---|---|---|---|---|")
    gen = ["\nGreedy translation of the first 50 FLORES+ dev English sources (prompt format of training), original vs swapped/control:\n",
           "| checkpoint | language | token-id identical | text identical | chrF++ orig / swapped vs reference (n=50, sanity only) |", "|---|---|---|---|---|"]
    for p in sorted(glob.glob(os.path.join(d, "sanity_*.json"))):
        if ".gen_" in p:
            continue
        v = os.path.basename(p)[len("sanity_"):-5]
        j = json.load(open(p, encoding="utf-8"))
        for lg, x in j["loss"].items():
            mx = x.get("identical_sentences_max_abs_diff_full_nll", float("nan"))
            out.append(f"| {v} | {lg} | {x['n']} | {x.get('n_identical_tokenisation','?')} | {x['orig_tokens']/x['n']:.1f} -> {x['swapped_tokens']/x['n']:.1f} | "
                       f"{x['orig_nll_per_token']:.4f} -> {x['swapped_nll_per_token']:.4f} | {x['orig_nll_per_char']:.4f} -> {x['swapped_nll_per_char']:.4f} | {mx:.3g} |")
        for lg, x in j["gen"].items():
            gen.append(f"| {v} | {lg} | {x['token_id_identical']}/{x['n']} | {x['text_identical']}/{x['n']} | {x['chrF++_original_vs_ref']:.2f} / {x['chrF++_swapped_vs_ref']:.2f} |")
    return "\n".join(out + gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--names", required=True)
    a = ap.parse_args()
    project = load_project_isos(a.names)
    parts = []
    if glob.glob(os.path.join(a.dir, "identity_*.json")):
        parts.append(identity_tables(a.dir, project))
    if os.path.exists(os.path.join(a.dir, "fertility.csv")):
        parts.append(fertility_tables(os.path.join(a.dir, "fertility.csv")))
    if glob.glob(os.path.join(a.dir, "sanity_*.json")):
        parts.append(sanity_tables(a.dir))
    txt = "\n\n".join(parts)
    open(os.path.join(a.dir, "tables.md"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
