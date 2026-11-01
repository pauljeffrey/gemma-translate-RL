#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""P5 (box, CPU): AfriScience-MT dev / test, sentence level and paper level: final2 (step 401) vs RL-final (1433) vs A2W (alpha 0.7).
Predictions: a4/eval_final2_step401 and a4/eval_final1433 (afriscience_eval.py, same run settings), r7/eval/{A2W/alpha_0.70 (dev), A2W_a0.7_test, final_1433_test, final_1433?}
(the stage evaluation that produced the numbers of paper Section 6.6).  Corpus chrF / chrF++ (sacrebleu, word_order 0 / 2, NFC) per language, macro over the 6 languages,
bootstrap over sentences (sentence level) or papers (paper level), one index matrix per (split, level, language), shared by the systems.
Usage: python p5_afs_compare.py OUTDIR
"""
import csv, json, os, sys, unicodedata, zlib
import numpy as np
sys.path.append("/workspace/sft_build/pylibs")
from sacrebleu.metrics import CHRF
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
B = 1000
J = "/workspace/jobs/"
SRC = {
    ("final2", "dev"): J + "a4/eval_final2_step401/predictions_dev.json", ("final2", "test"): J + "a4/eval_final2_step401/predictions_test.json",
    ("rl_a4", "dev"): J + "a4/eval_final1433/predictions_dev.json", ("rl_a4", "test"): J + "a4/eval_final1433/predictions_test.json",
    ("a2w", "dev"): J + "r7/eval/A2W/alpha_0.70/afs_predictions_dev.json", ("a2w", "test"): J + "r7/eval/A2W_a0.7_test/afs_predictions_test.json",
    ("rl_r7", "dev"): J + "r7/eval/final_1433/afs_predictions_dev.json", ("rl_r7", "test"): J + "r7/eval/final_1433_test/afs_predictions_test.json",
}
LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
nfc = lambda s: unicodedata.normalize("NFC", s or "")
C0, C2 = CHRF(word_order=0), CHRF(word_order=2)
score = lambda m, st: m._compute_score_from_stats(list(st)).score
D = {}
for (sysn, split), p in SRC.items():
    if not os.path.exists(p):
        print("MISSING", p); continue
    d = json.load(open(p))
    if "sentence" not in d and "afs" in d:
        d = d["afs"]
    D[(sysn, split)] = d
rows, pair = [], []
for split in ("dev", "test"):
    for level in ("sentence", "document"):
        systems = [s for s in ("rl_a4", "rl_r7", "a2w", "final2") if (s, split) in D and level in D[(s, split)]]
        reps, pts, fin = {}, {}, {}
        for lang in LANGS:
            ref_n = None
            ST = {}
            for s in systems:
                L = D[(s, split)][level].get(lang)
                if L is None:
                    continue
                if ref_n is None:
                    ref_n = [nfc(x["ref"]) for x in L]; srcrefs = ref_n
                elif [nfc(x["ref"]) for x in L] != ref_n:
                    print("REF MISMATCH", s, split, level, lang)
                h = [nfc(x["hyp"]) for x in L]
                ST[s] = (np.array(C0._extract_corpus_statistics(h, [ref_n]), float), np.array(C2._extract_corpus_statistics(h, [ref_n]), float),
                         np.array([bool(x.get("finished", True)) for x in L]))
            if not ST:
                continue
            n = len(ref_n)
            rng = np.random.default_rng(zlib.crc32(f"{split}|{level}|{lang}".encode()))
            ix = rng.integers(0, n, size=(B, n))
            for s, (s0, s2, f) in ST.items():
                for key, m, st in (("chrf", C0, s0), ("chrfpp", C2, s2)):
                    pts[(s, lang, key)] = score(m, st.sum(0))
                    reps[(s, lang, key)] = np.array([score(m, r) for r in st[ix].sum(1)])
                pts[(s, lang, "fin")] = float(f.mean())
                reps[(s, lang, "fin")] = f[ix].mean(1)
        for s in systems:
            for key in ("chrf", "chrfpp", "fin"):
                v = np.mean([pts[(s, l, key)] for l in LANGS]); rp = np.mean([reps[(s, l, key)] for l in LANGS], axis=0)
                rows.append(dict(split=split, level=level, system=s, metric=key, value=round(float(v), 4), lo=round(float(np.percentile(rp, 2.5)), 4), hi=round(float(np.percentile(rp, 97.5)), 4),
                                 **{l: round(pts[(s, l, key)], 4) for l in LANGS}))
        for a in ("rl_a4", "rl_r7", "a2w"):
            if a not in systems or "final2" not in systems:
                continue
            for key in ("chrf", "chrfpp", "fin"):
                d = np.mean([pts[("final2", l, key)] - pts[(a, l, key)] for l in LANGS])
                dr = np.mean([reps[("final2", l, key)] - reps[(a, l, key)] for l in LANGS], axis=0)
                lang_d = {l: pts[("final2", l, key)] - pts[(a, l, key)] for l in LANGS}
                lang_ci = {l: np.percentile(reps[("final2", l, key)] - reps[(a, l, key)], [2.5, 97.5]) for l in LANGS}
                pair.append(dict(split=split, level=level, a=a, b="final2", metric=key, delta=round(float(d), 4), lo=round(float(np.percentile(dr, 2.5)), 4), hi=round(float(np.percentile(dr, 97.5)), 4),
                                 langs_clearly_up=sum(lang_ci[l][0] > 0 for l in LANGS), langs_clearly_down=sum(lang_ci[l][1] < 0 for l in LANGS),
                                 **{l: round(lang_d[l], 3) for l in LANGS}))
for fn, r in (("afs_macro.csv", rows), ("afs_paired.csv", pair)):
    with open(f"{OUT}/{fn}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r[0])); w.writeheader(); w.writerows(r)
print("done", len(rows), len(pair))
