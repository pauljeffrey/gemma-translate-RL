#!/usr/bin/env python3
"""tokswap_eval.py -- tokenizer-level tests of a swapped Gemma-3 tokenizer against the original (CPU, no model).

  identity   byte-for-byte identity of the token-id sequences on every FLORES+ language (dev + devtest), English, French, code
             lines, LaTeX / digit strings, and on the decoded string of every kept vocabulary token.
             A sentence is "special" if it contains an Ethiopic character (geez variants) or a Fon-specific character (fon variants);
             a sentence is "freed-script" if it contains a character of a FREE script (CJK, Hangul, Thai, ... = scripts whose ids were re-used).
             Claim tested: sentences that are neither special nor freed-script -> 0 differences.
  fertility  tokens per sentence / word / character before and after on FLORES+ dev and devtest (devtest = MEASUREMENT ONLY).

Both write JSON next to --out.  Usage examples are in reports/token_swap_build_2026-10-25.md.
"""
import argparse, collections, glob, json, os, random, re, sys, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokswap_swap_vocab import char_script, FREE_SCRIPTS, is_eth, FON_TRIGGERS, SP, BYTE_RE  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

CODE_SNIPPETS = [
    "def f(x, y=3):\n    return [i**2 for i in range(x) if i % 2 == 0]\n", "for (int i = 0; i < n; ++i) { sum += a[i] * b[i]; }",
    "SELECT name, COUNT(*) FROM users WHERE age >= 18 GROUP BY name ORDER BY 2 DESC;", "import numpy as np\nx = np.zeros((3, 4), dtype=np.float32)",
    "$$\\frac{\\partial L}{\\partial w} = \\sum_{i=1}^{n} (y_i - \\hat{y}_i) x_i$$", "\\begin{equation} E = mc^2 \\end{equation}", "{\"a\": [1, 2, 3], \"b\": {\"c\": null, \"d\": true}}",
    "https://example.com/path/to/page?id=123&lang=en#section", "john.doe@example.org +234 803 123 4567 2026-10-25 12:30:45 UTC",
    "0123456789 3.14159 2,718,281.828 1e-9 -42 0x7FFF 100% $19.99 (555) 010-2030", "α β γ δ ∑ ∫ ≤ ≥ ≠ ∞ → ← ∈ ∀ ∃ √ π", "😀 👍 ❤ ✓ ✗ • … — – “quoted” ‘single’ «guillemets» ¿qué? ¡hola!",
    "<html><body><p class=\"x\">Hello</p></body></html>", "git commit -m \"fix: handle empty input\" && git push origin main", "    \t  mixed   whitespace\n\n\n\nand newlines",
    "Écoute, l'été dernier à Genève, François a mangé du pain et de la crème brûlée. Où est-ce qu'on va ? C'est déjà l'heure !",
    "مرحبا بكم في هذا الاختبار. كيف حالك اليوم؟ ١٢٣٤٥٦٧٨٩٠", "The quick brown fox jumps over the lazy dog. 1234567890",
]


def has_special(s, mode):
    for c in s:
        if mode in ("geez", "both") and is_eth(c):
            return True
        if mode in ("fon", "both") and c in FON_TRIGGERS:
            return True
    return False


def has_freed(s):
    for c in s:
        sc = char_script(c)
        if sc and sc in FREE_SCRIPTS:
            return True
    return False


def load_flores(root):
    out = {}
    for split in ("dev", "devtest"):
        for p in sorted(glob.glob(os.path.join(root, split, "*.jsonl"))):
            lang = os.path.basename(p)[:-6]
            out[(split, lang)] = [json.loads(l)["text"] for l in open(p, encoding="utf-8")]
    return out


def enc_all(tk, texts):
    r = []
    for k in range(0, len(texts), 4000):
        r.extend(e.ids for e in tk.encode_batch(texts[k:k + 4000], add_special_tokens=False))
    return r


def cmd_identity(a):
    to = Tokenizer.from_file(a.orig); ts = Tokenizer.from_file(os.path.join(a.swapped, "tokenizer.json"))
    sets = load_flores(a.flores)
    rng = random.Random(0)
    extra = {}
    extra["code_snippets"] = CODE_SNIPPETS
    if a.code_dir:
        lines = []
        for p in glob.glob(os.path.join(a.code_dir, "**/*.py"), recursive=True)[:600]:
            try:
                lines += [l.rstrip("\n") for l in open(p, encoding="utf-8") if len(l.strip()) >= 8]
            except Exception:
                pass
        rng.shuffle(lines)
        extra["python_code_lines"] = lines[:30000]
    digits = [" ".join("".join(rng.choice("0123456789") for _ in range(rng.randint(1, 12))) for _ in range(rng.randint(1, 6))) for _ in range(5000)]
    extra["random_digit_strings"] = digits
    res = {"mode": a.mode, "per_set": {}, "totals": {}}
    tot = collections.Counter()
    diff_examples = []
    for name, texts in [(f"flores_{sp}_{lg}", t) for (sp, lg), t in sets.items()] + [(k, v) for k, v in extra.items()]:
        io = enc_all(to, texts); is_ = enc_all(ts, texts)
        c = collections.Counter()
        for t, x, y in zip(texts, io, is_):
            sp_, fr = has_special(t, a.mode), has_freed(t)
            cls = "special" if sp_ else ("freed_script" if fr else "plain")
            c["n_" + cls] += 1
            if x != y:
                c["diff_" + cls] += 1
                if cls == "plain" and len(diff_examples) < 20:
                    diff_examples.append({"set": name, "text": t[:200]})
        res["per_set"][name] = dict(c)
        for k, v in c.items():
            tot[k] += v
    # vocabulary strings: every kept token's own string must tokenise identically
    do = json.load(open(a.orig, encoding="utf-8")); ds = json.load(open(os.path.join(a.swapped, "tokenizer.json"), encoding="utf-8"))
    vo, vs = do["model"]["vocab"], ds["model"]["vocab"]
    added = {t["id"] for t in do["added_tokens"]}
    kept = [s for s, i in vo.items() if i not in added and vs.get(s) == i]
    kept_txt = [s.replace(SP, " ") for s in kept if not BYTE_RE.match(s)]
    io = enc_all(to, kept_txt); is_ = enc_all(ts, kept_txt)
    nd = sum(1 for t, x, y in zip(kept_txt, io, is_) if x != y and not has_special(t, a.mode) and not has_freed(t))
    nd_special = sum(1 for t, x, y in zip(kept_txt, io, is_) if x != y and has_special(t, a.mode))
    res["vocab_strings"] = {"kept_token_strings_tested": len(kept_txt), "diff_plain": nd, "diff_special_script": nd_special}
    res["totals"] = dict(tot)
    res["plain_diff_examples"] = diff_examples
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n_sets = len(res["per_set"])
    print(f"[{a.mode}] sets={n_sets} sentences: plain={tot['n_plain']} (diff {tot['diff_plain']}), special={tot['n_special']} (diff {tot['diff_special']}), "
          f"freed_script={tot['n_freed_script']} (diff {tot['diff_freed_script']}); vocab-string test: {len(kept_txt)} strings, plain diffs {nd}")


def cmd_fertility(a):
    import numpy as np
    to = Tokenizer.from_file(a.orig)
    ts = {name: Tokenizer.from_file(os.path.join(p, "tokenizer.json")) for name, p in (x.split("=", 1) for x in a.swapped)}
    sets = load_flores(a.flores)
    rows = []
    rng = np.random.default_rng(0)
    for (split, lang) in sorted(sets):
        if lang.split("_")[0] not in a.langs:
            continue
        texts = sets[(split, lang)]
        words = sum(len(t.split()) for t in texts); chars = sum(len(t) for t in texts)
        io = enc_all(to, texts)
        for name, tk in [("original", to)] + list(ts.items()):
            ids = io if name == "original" else enc_all(tk, texts)
            ntok = np.array([len(x) for x in ids])
            nbyte = sum(1 for x in ids for i in x if BYTE_RE.match(tk.id_to_token(i)))
            row = {"split": split, "lang": lang, "variant": name, "sentences": len(texts), "words": words, "chars": chars, "tokens": int(ntok.sum()),
                   "tok_per_word": ntok.sum() / words, "tok_per_char": ntok.sum() / chars, "tok_per_sent_mean": float(ntok.mean()),
                   "tok_per_sent_p95": float(np.percentile(ntok, 95)), "byte_fallback_share": nbyte / max(1, ntok.sum())}
            if name != "original":
                base = np.array([len(x) for x in io])
                ratio = ntok / base
                bs = [ratio[rng.integers(0, len(ratio), len(ratio))].mean() for _ in range(1000)]
                row.update({"tokens_ratio_vs_original": float(ntok.sum() / base.sum()), "mean_sent_ratio": float(ratio.mean()),
                            "ratio_ci_lo": float(np.percentile(bs, 2.5)), "ratio_ci_hi": float(np.percentile(bs, 97.5)),
                            "sentences_changed": int(sum(1 for x, y in zip(io, ids) if x != y))})
            rows.append(row)
    import csv
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("split", "lang", "variant"), k))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {a.out}: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("identity"); p.add_argument("--orig", required=True); p.add_argument("--swapped", required=True); p.add_argument("--flores", required=True)
    p.add_argument("--mode", choices=["geez", "fon", "both"], required=True); p.add_argument("--code-dir"); p.add_argument("--out", required=True); p.set_defaults(f=cmd_identity)
    p = sp.add_parser("fertility"); p.add_argument("--orig", required=True); p.add_argument("--swapped", nargs="+", required=True, help="name=dir ...")
    p.add_argument("--flores", required=True); p.add_argument("--langs", nargs="+", default=["eng", "amh", "tir", "fon", "hau", "yor", "swh", "kin", "ibo"])
    p.add_argument("--out", required=True); p.set_defaults(f=cmd_fertility)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
