#!/usr/bin/env python3
"""Greedy vs beam-4 vs MBR-16 from the json files of `decode_study_385m.py`: per-language and macro chrF++ / spBLEU, paired bootstrap of every system against greedy (same resample, macro = equal weight
per language), and surface statistics of the hypotheses. CPU only.

    python scripts/decode_analyze.py --json decode_m385.json [--json beam_part1.json ...] --tokenizer MODEL_DIR --out DIR [--tag NAME] [--boot 1000] [--max-new 256]

Several --json files are merged as parts of one run (their items/hyps are concatenated; a language must be in exactly one part).
`trunc_est` = share of hypotheses whose token count (tokenizer of the model, no special tokens) is >= max_new - 4: the json keeps text only, so 'finished' of beam/MBR is ESTIMATED this way.
"""
import argparse, csv, json, os, sys, unicodedata
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
nfc = lambda s: unicodedata.normalize("NFC", s)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", action="append", required=True); ap.add_argument("--tokenizer", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="run"); ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--max-new", type=int, default=256); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--systems", default="greedy,beam4,mbr16")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from sacrebleu.metrics import CHRF, BLEU
    from transformers import AutoTokenizer
    from gemma_mt.rl.guards import is_copy_of_source, is_repetitive, is_self_repetitive
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    items = []; H = {}; cfg = []
    for p in a.json:
        d = json.load(open(p, encoding="utf-8")); cfg.append({"file": os.path.basename(p), "n_per_language": d.get("n_per_language"), "sampling": d.get("sampling")})
        base = len(items); items += [tuple(x) for x in d["hyp"]["items"]]
        for k in ("greedy", "beam4", "mbr16"):
            if k in d["hyp"] and d["hyp"][k] and len(d["hyp"][k]) == len(d["hyp"]["items"]):
                H.setdefault(k, []).extend(d["hyp"][k])
            elif k in H:
                sys.exit(f"{p} lacks {k}")
    systems = [k for k in a.systems.split(",") if k in H and len(H[k]) == len(items)]
    chrf, bleu = CHRF(word_order=2), BLEU(tokenize="flores200")
    langs = sorted({it[0] for it in items}); idx = {l: [i for i, it in enumerate(items) if it[0] == l] for l in langs}
    refs = {l: [nfc(items[i][2]) for i in idx[l]] for l in langs}
    rng = np.random.default_rng(a.seed)
    stats = {}
    for s in systems:
        for l in langs:
            hy = [nfc(H[s][i]) for i in idx[l]]
            stats[(s, l)] = (np.array(chrf._extract_corpus_statistics(hy, [refs[l]]), dtype=float), np.array(bleu._extract_corpus_statistics(hy, [refs[l]]), dtype=float))
    sc_c = lambda v: chrf._compute_score_from_stats(list(v)).score
    sc_b = lambda v: bleu._compute_score_from_stats(list(v)).score
    rows = []; surf = {}
    for s in systems:
        for l in langs:
            hy = [H[s][i] for i in idx[l]]; ens = [items[i][1] for i in idx[l]]
            ntok = [len(tok(h, add_special_tokens=False)["input_ids"]) if h.strip() else 0 for h in hy]
            surf[(s, l)] = dict(n=len(hy), trunc_est=float(np.mean([t >= a.max_new - 4 for t in ntok])), empty=float(np.mean([not h.strip() for h in hy])),
                                repeat=float(np.mean([bool(h.strip()) and (is_repetitive(h) or is_self_repetitive(h)) for h in hy])),
                                copy=float(np.mean([bool(h.strip()) and len(e.split()) >= 4 and is_copy_of_source(e, h, 0.9) for h, e in zip(hy, ens)])),
                                len_ratio=float(sum(len(h) for h in hy) / max(1, sum(len(r) for r in refs[l]))))
    with open(os.path.join(a.out, f"{a.tag}_per_language.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["language", "system", "n", "chrf", "spbleu", "trunc_est", "empty", "repeat", "copy", "len_ratio_chars_vs_ref", "d_chrf_vs_greedy", "lo", "hi"])
        per_boot = {}
        for l in langs:
            n = len(idx[l]); cnt = rng.multinomial(n, np.full(n, 1.0 / n), size=a.boot).astype(float)
            for s in systems:
                st = stats[(s, l)]; c = sc_c(st[0].sum(0)); b = sc_b(st[1].sum(0))
                if s == "greedy": d = lo = hi = ""; per_boot[(s, l)] = None
                else:
                    g = stats[("greedy", l)]; rs, rg = cnt @ st[0], cnt @ g[0]
                    bs = np.array([sc_c(x) - sc_c(y) for x, y in zip(rs, rg)]); per_boot[(s, l)] = bs
                    d = c - sc_c(g[0].sum(0)); lo, hi = np.percentile(bs, [2.5, 97.5])
                u = surf[(s, l)]; w.writerow([l, s, n, c, b, u["trunc_est"], u["empty"], u["repeat"], u["copy"], u["len_ratio"], d, lo, hi])
    # macro
    with open(os.path.join(a.out, f"{a.tag}_macro.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["system", "n_languages", "n_sentences_per_language", "chrf_macro", "spbleu_macro", "d_chrf_vs_greedy", "lo", "hi", "d_spbleu_vs_greedy", "slo", "shi", "trunc_est", "empty", "repeat", "copy", "p_better_chrf", "config"])
        nper = int(np.median([len(idx[l]) for l in langs]))
        sb = {}
        for s in systems:
            c = np.mean([sc_c(stats[(s, l)][0].sum(0)) for l in langs]); b = np.mean([sc_b(stats[(s, l)][1].sum(0)) for l in langs])
            u = {k: float(np.mean([surf[(s, l)][k] for l in langs])) for k in ("trunc_est", "empty", "repeat", "copy")}
            if s == "greedy": d = lo = hi = ds = slo = shi = pb = ""
            else:
                rng2 = np.random.default_rng(a.seed + 1)
                bc = np.zeros(a.boot); bb = np.zeros(a.boot)
                for l in langs:
                    n = len(idx[l]); cnt = rng2.multinomial(n, np.full(n, 1.0 / n), size=a.boot).astype(float)
                    st, g = stats[(s, l)], stats[("greedy", l)]
                    bc += np.array([sc_c(x) - sc_c(y) for x, y in zip(cnt @ st[0], cnt @ g[0])]); bb += np.array([sc_b(x) - sc_b(y) for x, y in zip(cnt @ st[1], cnt @ g[1])])
                bc /= len(langs); bb /= len(langs); lo, hi = np.percentile(bc, [2.5, 97.5]); slo, shi = np.percentile(bb, [2.5, 97.5])
                d = c - np.mean([sc_c(stats[("greedy", l)][0].sum(0)) for l in langs]); ds = b - np.mean([sc_b(stats[("greedy", l)][1].sum(0)) for l in langs]); pb = float((bc > 0).mean())
            w.writerow([s, len(langs), nper, c, b, d, lo, hi, ds, slo, shi, u["trunc_est"], u["empty"], u["repeat"], u["copy"], pb, json.dumps(cfg)])
    print("languages", len(langs), "systems", systems, "-> ", a.out)


if __name__ == "__main__":
    main()
