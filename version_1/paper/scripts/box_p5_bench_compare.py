#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""P5 (box, CPU only): final2 (step 401) vs RL-final vs SFT-final on FLORES+ / NTREX-128 / MAFAND-MT / SmolSent, from the hypotheses of the E1 files
(sys_rl, sys_sft: /workspace/jobs/e1) and of the final2 generations (/workspace/jobs/e2/out_final2 [flores_plus], out_final2_other [ntrex, mafand, smol_sent]).
Per (benchmark, language, subset) corpus chrF++ (sacrebleu CHRF word_order 2, NFC) and spBLEU (flores200); macro over languages; paired sentence bootstrap
(B=1000, ONE index matrix per (benchmark, language, subset) shared by the three systems; the macro CI resamples the sentences of every language, not the languages);
sentence-level chrF++ win / tie / loss of final2 over RL-final (threshold 1.0) on the pooled no_pair rows.
Usage: python p5_bench_compare.py OUTDIR
"""
import csv, gzip, json, os, sys, unicodedata, zlib
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.append("/workspace/sft_build/pylibs")
from sacrebleu.metrics import CHRF, BLEU

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
B = 1000
BENCH = ["flores_plus", "ntrex", "mafand", "smol_sent"]
nfc = lambda s: unicodedata.normalize("NFC", s or "")
CH, BL = CHRF(word_order=2), BLEU(tokenize="flores200")


def load_sys(path):
    d = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r["b"] in BENCH:
                d[(r["b"], r["iso"], str(r["id"]))] = dict(en=r["en"], ref=r["ref"], hyp=r["hyp"], fin=bool(r["fin"]), cont=r["contaminated"], pair=r["pair_seen"])
    return d


def load_pred(paths):
    d = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for l in f:
                r = json.loads(l)
                if r["benchmark"] not in BENCH:
                    continue
                d[(r["benchmark"], r["iso"], str(r["id"]))] = dict(en=r["english"], ref=r["reference"], hyp=r["hyp"], fin=bool(r.get("finished", True)),
                                                                   cont=r.get("contaminated"), pair=r.get("pair_seen"))
    return d


def stats_job(a):
    key, hyps, refs = a
    h = [nfc(x) for x in hyps]; r = [nfc(x) for x in refs]
    return key, np.array(CH._extract_corpus_statistics(h, [r]), dtype=float), np.array(BL._extract_corpus_statistics(h, [r]), dtype=float)


def score(m, st):
    return m._compute_score_from_stats(list(st)).score


def main():
    S = {"sft": load_sys("/workspace/jobs/e1/sys_sft.jsonl.gz"), "rl": load_sys("/workspace/jobs/e1/sys_rl.jsonl.gz"),
         "final2": load_pred(["/workspace/jobs/e2/out_final2/predictions.jsonl", "/workspace/jobs/e2/out_final2_other/predictions.jsonl"])}
    for k, v in S.items():
        print(k, len(v), flush=True)
    keys = sorted(set(S["rl"]) & set(S["sft"]) & set(S["final2"]))
    print("common rows", len(keys), "missing from final2:", len(set(S["rl"]) - set(S["final2"])), flush=True)
    # consistency of the rows
    bad = 0
    for k in keys:
        a, b, c = S["rl"][k], S["sft"][k], S["final2"][k]
        if not (a["en"] == b["en"] == c["en"] and a["ref"] == b["ref"] == c["ref"] and a["pair"] == c["pair"] and a["cont"] == c["cont"]):
            bad += 1
    print("rows whose english / reference / flags differ between the three systems:", bad, flush=True)
    groups = defaultdict(list)
    for k in keys:
        groups[(k[0], k[1])].append(k)
    jobs = []
    for (b, iso), ks in groups.items():
        for s in S:
            jobs.append(((b, iso, s), [S[s][k]["hyp"] for k in ks], [S[s][k]["ref"] for k in ks]))
    with Pool(10) as p:
        res = p.map(stats_job, jobs, chunksize=4)
    ST = {k: (c, bl) for k, c, bl in res}
    per = []           # per (b, iso, subset)
    reps = {}          # (b, iso, subset, sys, metric) -> (pt, reps)
    sent = defaultdict(lambda: defaultdict(list))   # (b, subset) -> lists for sentence-level wtl
    fin = {}
    for (b, iso), ks in sorted(groups.items()):
        pair = np.array([S["rl"][k]["pair"] is False for k in ks]); cl = np.array([S["rl"][k]["cont"] is False for k in ks])
        masks = {"all": np.ones(len(ks), bool), "no_pair": pair, "clean": cl}
        for sub, mk in masks.items():
            n = int(mk.sum())
            if n == 0:
                continue
            rng = np.random.default_rng(zlib.crc32(f"{b}|{iso}|{sub}".encode()))
            ix = rng.integers(0, n, size=(B, n))
            row = dict(benchmark=b, iso=iso, subset=sub, n=n)
            for s in S:
                cst, bst = ST[(b, iso, s)]
                for key, m, st in (("chrf", CH, cst[mk]), ("spbleu", BL, bst[mk])):
                    pt = score(m, st.sum(0))
                    rp = np.array([score(m, r) for r in st[ix].sum(1)])
                    reps[(b, iso, sub, s, key)] = (pt, rp)
                    row[f"{s}_{key}"] = pt
                fin[(b, iso, sub, s)] = float(np.mean([S[s][k]["fin"] for k, m_ in zip(ks, mk) if m_]))
                row[f"{s}_fin"] = fin[(b, iso, sub, s)]
            for a_, b_ in (("rl", "final2"), ("sft", "final2")):
                for key in ("chrf", "spbleu"):
                    d = reps[(b, iso, sub, b_, key)][0] - reps[(b, iso, sub, a_, key)][0]
                    dr = reps[(b, iso, sub, b_, key)][1] - reps[(b, iso, sub, a_, key)][1]
                    row[f"d_final2_{a_}_{key}"] = d
                    row[f"d_final2_{a_}_{key}_lo"], row[f"d_final2_{a_}_{key}_hi"] = np.percentile(dr, [2.5, 97.5])
            per.append(row)
            # sentence-level chrF++ (final2 vs rl) for the pooled win/tie/loss
            if sub in ("no_pair", "clean", "all"):
                cr, cf = ST[(b, iso, "rl")][0][mk], ST[(b, iso, "final2")][0][mk]
                sr = np.array([score(CH, x) if x.sum() else 0.0 for x in cr]); sf = np.array([score(CH, x) if x.sum() else 0.0 for x in cf])
                sent[(b, sub)]["rl"].append(sr); sent[(b, sub)]["final2"].append(sf); sent[(b, sub)]["iso"].append(iso)
    with open(f"{OUT}/per_language_bench.csv", "w", newline="") as f:
        cols = list(per[0]); w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in per:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    # macro
    macro, paired = [], []
    for b in BENCH:
        for sub in ("all", "no_pair", "clean"):
            isos = sorted({r["iso"] for r in per if r["benchmark"] == b and r["subset"] == sub})
            if not isos:
                continue
            for key in ("chrf", "spbleu"):
                mp = {}
                for s in S:
                    pts = np.mean([reps[(b, i, sub, s, key)][0] for i in isos]); rp = np.mean([reps[(b, i, sub, s, key)][1] for i in isos], axis=0)
                    mp[s] = (pts, rp)
                    macro.append(dict(benchmark=b, subset=sub, system=s, metric=key, k=len(isos), value=round(float(pts), 4), lo=round(float(np.percentile(rp, 2.5)), 4),
                                      hi=round(float(np.percentile(rp, 97.5)), 4), finished=round(100 * float(np.mean([fin[(b, i, sub, s)] for i in isos])), 2)))
                for a_ in ("rl", "sft"):
                    d = mp["final2"][0] - mp[a_][0]; dr = mp["final2"][1] - mp[a_][1]
                    pl = [r for r in per if r["benchmark"] == b and r["subset"] == sub]
                    better = sum(r[f"d_final2_{a_}_{key}_lo"] > 0 for r in pl); worse = sum(r[f"d_final2_{a_}_{key}_hi"] < 0 for r in pl)
                    up = sum(r[f"d_final2_{a_}_{key}"] > 0 for r in pl)
                    paired.append(dict(benchmark=b, subset=sub, a=a_, b="final2", metric=key, k=len(isos), delta=round(float(d), 4), lo=round(float(np.percentile(dr, 2.5)), 4),
                                       hi=round(float(np.percentile(dr, 97.5)), 4), langs_up=up, langs_clearly_better=better, langs_clearly_worse=worse))
    for fn, rows in (("macro_bench.csv", macro), ("paired_bench.csv", paired)):
        with open(f"{OUT}/{fn}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # sentence-level wins (final2 over rl), pooled over languages, with a sentence bootstrap
    wtl = []
    rng = np.random.default_rng(0)
    for (b, sub), d in sorted(sent.items()):
        sr = np.concatenate(d["rl"]); sf = np.concatenate(d["final2"]); dd = sf - sr
        n = len(dd)
        win = (dd > 1.0); loss = (dd < -1.0)
        pref = win.sum() / max(1, win.sum() + loss.sum())
        ix = rng.integers(0, n, size=(B, n))
        wr = win[ix].sum(1); lr = loss[ix].sum(1)
        prefr = wr / np.maximum(1, wr + lr)
        dm = dd[ix].mean(1)
        wtl.append(dict(benchmark=b, subset=sub, n=n, mean_rl=round(float(sr.mean()), 3), mean_final2=round(float(sf.mean()), 3), delta=round(float(dd.mean()), 3),
                        delta_lo=round(float(np.percentile(dm, 2.5)), 3), delta_hi=round(float(np.percentile(dm, 97.5)), 3), win=round(float(win.mean()), 4),
                        tie=round(float(1 - win.mean() - loss.mean()), 4), loss=round(float(loss.mean()), 4), pref=round(float(pref), 4),
                        pref_lo=round(float(np.percentile(prefr, 2.5)), 4), pref_hi=round(float(np.percentile(prefr, 97.5)), 4)))
    with open(f"{OUT}/sentence_wtl_final2_vs_rl.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wtl[0])); w.writeheader(); w.writerows(wtl)
    print("done", len(per), len(macro), len(paired), len(wtl))


if __name__ == "__main__":
    main()
