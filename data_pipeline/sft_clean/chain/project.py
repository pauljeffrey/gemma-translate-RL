#!/usr/bin/env python3
"""PROJECTED breakdown without the GPU: per language and per bin (b64 b256 b512 b1024) windows, English tokens, Gemma target tokens, packs; the gated projection; the stage A / B shares and
effective epochs (old vs new, same columns as reports/gemma_sft_training_plan_2026-10-14.md) for A0 and S1.

    python project.py bins   --windows windows_stats --out bins            # needs windows.py --stats-only --gemma-tokenizer ... output
    python project.py gate-estimate --bins bins --keep-csv keep.csv         # keep.csv: iso,keep_rate [,source]; missing languages get the flat 15% drop (ESTIMATE)
    python project.py stages --bins bins --out bins                        # A0 / S1 / B0 projected tables

FINAL FIGURES NEED THE REAL QE SCORES: the gated tables here are labelled ESTIMATE. Packs per language = ceil(sum of sample tokens / (4,096 x efficiency)); sample tokens = prompt template +
English Gemma tokens + `{iso} ` tag + target Gemma tokens + end-of-turn tail, the template overhead measured with gemma_mt.format.build_example on real windows (see `overhead`).
Packs per bin are the language's packs apportioned by the bin's share of sample tokens (a pack mixes bins).
"""
import argparse, collections, csv, glob, json, math, os, re, sys
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CYCLE, SEQ_LEN, SPECIALIST, STAGE_A, STAGE_B

OLD_B_RATIO = 732_416 / 538_480          # stage B draws / packs of the old run (sft_unimax.yaml comment)
EFF = 0.9994          # packing efficiency of the old build (plan section 2); verified again on the 20k-pair test (test_chain.py / README)


def read_old_plan(repo):
    txt = open(os.path.join(repo, "reports", "gemma_sft_training_plan_2026-10-14.md"), encoding="utf-8").read()
    old = {}
    for line in txt.splitlines():
        m = re.match(r"\| ([a-z]{3}) \| ([^|]+) \| ([\d,]+) / ([\d,]+) / ([\d,]+) / ([\d,]+) \| ([\d,]+) \| ([\d,]+) \| ([\d,]+) \| ([\d.]+)% \| ([\d.]+) \| ([\d,]+) \|", line)
        if m:
            g = [x.replace(",", "") for x in m.groups()]
            old[g[0]] = dict(name=g[1].strip(), b64=int(g[2]), b256=int(g[3]), b512=int(g[4]), b1024=int(g[5]), train=int(g[6]), tokens=int(g[7]), packs=int(g[8]), share=float(g[9]), epochs=float(g[10]), val=int(g[11]))
    return old


def overhead(tok_name, windows_dirs, isos):
    """mean (prompt template + tag + tail) tokens per sample for each language, measured with the real format on up to 12 windows per language"""
    sys.path.insert(0, os.environ.get("GEMMA_MT_PKG", "/workspace/work/gemma_mt_pkg"))
    from transformers import AutoTokenizer
    from gemma_mt import format as fmt
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    res = {}
    for iso in isos:
        res[iso] = None
    for f in sorted(glob.glob(os.path.join(windows_dirs, "*.parquet"))):
        t = pq.read_table(f, columns=[c for c in ("english", "target", "iso") if c in pq.ParquetFile(f).schema_arrow.names]).to_pydict()
        if "english" not in t:
            return None
        for en, tg, iso in zip(t["english"], t["target"], t["iso"]):
            if res.get(iso) is None:
                res[iso] = []
            if isinstance(res[iso], list) and len(res[iso]) < 12:
                ex = fmt.build_example(en, iso, tg, tok)
                res[iso].append(len(ex.input_ids) - len(tok(en, add_special_tokens=False)["input_ids"]) - len(tok(tg, add_special_tokens=False)["input_ids"]))
    return {k: (float(np.mean(v)) if v else None) for k, v in res.items()}


def bins(a):
    os.makedirs(a.out, exist_ok=True)
    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros(5)))      # iso -> bin -> [windows, afro_en, gemma_en, gemma_tgt, pairs]
    gr = json.load(open(os.path.join(a.windows, "gemma_ratio.json"))) if os.path.exists(os.path.join(a.windows, "gemma_ratio.json")) else {}
    grm = {k: (np.mean([v[k] for v in gr.values()]) if gr else 1.0) for k in ("en", "tg")}
    for f in sorted(glob.glob(os.path.join(a.windows, "windows-*.parquet"))):
        t = pq.read_table(f, columns=["iso", "length_bin", "src_afr_tokens", "src_gemma_tokens", "tgt_gemma_tokens", "n_pairs", "tgt_afr_tokens"]).to_pydict()
        for iso, b, sa, sg, tg, n, ta in zip(t["iso"], t["length_bin"], t["src_afr_tokens"], t["src_gemma_tokens"], t["tgt_gemma_tokens"], t["n_pairs"], t["tgt_afr_tokens"]):
            if sg == 0 and sa:                       # Gemma tokens estimated from the per-language sampled ratio
                rr = gr.get(iso)
                sg = sa * (rr["en"] if rr else grm["en"]); tg = ta * (rr["tg"] if rr else grm["tg"])
            v = agg[iso][b]; v[0] += 1; v[1] += sa; v[2] += sg; v[3] += tg; v[4] += n
    oh = json.load(open(a.overhead)) if a.overhead and os.path.exists(a.overhead) else {}
    default_oh = float(np.mean([v for v in oh.values() if v])) if oh else 28.0
    rows, lang = [], {}
    for iso, d in agg.items():
        o = oh.get(iso) or default_oh
        tot_tok = sum(v[2] + v[3] + o * v[0] for v in d.values())
        packs = math.ceil(tot_tok / (SEQ_LEN * EFF))
        lang[iso] = {"windows": int(sum(v[0] for v in d.values())), "pairs": int(sum(v[4] for v in d.values())), "target_tokens": int(sum(v[3] for v in d.values())), "sample_tokens": tot_tok, "packs": packs, "overhead": o}
        for b in CYCLE:
            v = d.get(b)
            if v is None:
                v = np.zeros(5)
            st = v[2] + v[3] + o * v[0]
            rows.append([iso, b, int(v[0]), int(v[4]), int(v[1]), int(v[2]), int(v[3]), int(st), round(packs * st / max(tot_tok, 1), 1)])
    with open(os.path.join(a.out, "windows_per_language_bin_ungated.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iso", "bin", "windows", "pairs", "english_tokens_afroxlmr", "english_tokens_gemma", "target_tokens_gemma", "sample_tokens_gemma", "packs_apportioned"])
        w.writerows(sorted(rows, key=lambda r: (-lang[r[0]]["target_tokens"], r[0], CYCLE.index(r[1]))))
    with open(os.path.join(a.out, "per_language_ungated.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "windows", "pairs", "target_tokens_gemma", "sample_tokens_gemma", "packs", "template_overhead_tokens"])
        for iso, l in sorted(lang.items(), key=lambda kv: -kv[1]["target_tokens"]):
            w.writerow([iso, l["windows"], l["pairs"], int(l["target_tokens"]), int(l["sample_tokens"]), l["packs"], round(l["overhead"], 1)])
    tb = collections.defaultdict(lambda: np.zeros(4))
    for r in rows:
        tb[r[1]] += np.array([r[2], r[4], r[6], r[8]])
    with open(os.path.join(a.out, "per_bin_totals_ungated.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["bin", "windows", "english_tokens_afroxlmr", "target_tokens_gemma", "packs_apportioned"])
        for b in CYCLE: w.writerow([b, int(tb[b][0]), int(tb[b][1]), int(tb[b][2]), round(tb[b][3], 1)])
    print({b: [int(x) for x in tb[b]] for b in CYCLE}, "languages", len(lang), "packs", sum(l["packs"] for l in lang.values()))


def gate_estimate(a):
    keep = {}
    if a.keep_csv and os.path.exists(a.keep_csv):
        for r in csv.DictReader(open(a.keep_csv)):
            keep[r["iso"]] = (float(r["keep_rate"]), r.get("source", "QE sample"))
    rows = list(csv.DictReader(open(os.path.join(a.bins, "windows_per_language_bin_ungated.csv"))))
    out, per = [], collections.defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        kr, src = keep.get(r["iso"], (1 - a.flat_drop, f"ESTIMATE flat {a.flat_drop:.0%} drop"))
        out.append([r["iso"], r["bin"], int(float(r["windows"]) * kr), int(float(r["english_tokens_afroxlmr"]) * kr), int(float(r["target_tokens_gemma"]) * kr), round(float(r["packs_apportioned"]) * kr, 1), round(kr, 4), src])
        p = per[r["iso"]]; p[0] += int(float(r["windows"]) * kr); p[1] += int(float(r["target_tokens_gemma"]) * kr); p[2] += float(r["packs_apportioned"]) * kr
    with open(os.path.join(a.bins, "windows_per_language_bin_gated_ESTIMATE.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "bin", "windows", "english_tokens_afroxlmr", "target_tokens_gemma", "packs_apportioned", "assumed_keep_rate", "keep_rate_source"]); w.writerows(out)
    with open(os.path.join(a.bins, "per_language_gated_ESTIMATE.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "windows", "target_tokens_gemma", "packs", "assumed_keep_rate", "keep_rate_source"])
        for iso, p in sorted(per.items(), key=lambda kv: -kv[1][1]):
            kr, src = keep.get(iso, (1 - a.flat_drop, f"ESTIMATE flat {a.flat_drop:.0%} drop")); w.writerow([iso, p[0], p[1], math.ceil(p[2]), round(kr, 4), src])
    print("gated estimate written;", len(keep), "languages with a QE-sample keep rate")


def stages(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sampler as S
    PD, ALL_ISOS = S._pd()
    repo = a.repo
    old = read_old_plan(repo)
    for tag in ("ungated", "gated_ESTIMATE"):
        src = os.path.join(a.bins, f"per_language_{tag}.csv")
        if not os.path.exists(src):
            continue
        L = list(csv.DictReader(open(src)))
        packs = {r["iso"]: int(r["packs"]) for r in L if r["iso"] in ALL_ISOS}
        pcsv = os.path.join(a.bins, f"packs_{tag}.csv")
        with open(pcsv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["iso", "packs"]); [w.writerow([k, v]) for k, v in sorted(packs.items())]
        base = os.path.join(a.bins, f"sched_{tag}")
        S.plan(argparse.Namespace(stage="A", packed_dir=None, packs_csv=pcsv, out=base + "/A0", seed=1234, steps=0, per_step=0, alpha=STAGE_A["alpha"], budget_tokens=0))
        S.arms(argparse.Namespace(schedule=base + "/A0", out=base))
        n_packs_tot = sum(packs.values())
        b_steps = int(round(OLD_B_RATIO * n_packs_tot / STAGE_B["per_step"]))          # the old stage B drew 1.360 x the number of packs (732,416 draws for 538,480 packs)
        lo = sum(math.ceil(STAGE_B["min_epochs"] * v) for v in packs.values()); hi = sum(int(STAGE_B["max_epochs"] * v) for v in packs.values())
        b_steps = int(min(max(b_steps, math.ceil(lo / STAGE_B["per_step"])), hi // STAGE_B["per_step"]))
        S.plan(argparse.Namespace(stage="B", packed_dir=None, packs_csv=pcsv, out=base + "/B0", seed=1234, steps=b_steps, per_step=0, alpha=0.5, budget_tokens=0))
        draws = {r["iso"]: r for r in csv.DictReader(open(base + "/A0/draws_per_language.csv"))}
        s1 = np.load(base + "/S1/schedule.npy"); pl = np.load(base + "/S1/pack_lang.npy")
        cnt = collections.Counter(ALL_ISOS[i] for i in pl[s1.reshape(-1)])
        tot1 = sum(cnt.values())
        bdr = {r["iso"]: r for r in csv.DictReader(open(base + "/B0/draws_per_language.csv"))}
        per = {r["iso"]: r for r in L}
        with open(os.path.join(a.bins, f"stages_{tag}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iso", "old_packs", "new_windows", "new_target_tokens_gemma", "new_packs", "A0_share_of_draws_pct_new", "A0_share_old_pct", "A0_effective_epochs_new", "A0_effective_epochs_old",
                        "S1_share_of_S1_draws_pct", "S1_effective_epochs", "B_share_pct_new", "B_effective_epochs_new", "val_windows_est"])
            for iso, r in sorted(per.items(), key=lambda kv: -int(kv[1]["target_tokens_gemma"])):
                d = draws.get(iso); o = old.get(iso, {}); b = bdr.get(iso)
                w.writerow([iso, o.get("packs", ""), r["windows"], r["target_tokens_gemma"], r["packs"], round(100 * float(d["share"]), 3) if d else "", o.get("share", ""), d["effective_epochs"] if d else "", o.get("epochs", ""),
                            round(100 * cnt[iso] / tot1, 3) if cnt.get(iso) else "", d["effective_epochs"] if (d and iso in SPECIALIST) else "", round(100 * float(b["share"]), 3) if b else "", b["effective_epochs"] if b else "",
                            max(60, int(0.002 * int(float(r["windows"]))))])
        print(tag, "stages written; S1 draws", tot1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bins"); b.add_argument("--windows", required=True); b.add_argument("--out", required=True); b.add_argument("--overhead", default=None)
    g = sub.add_parser("gate-estimate"); g.add_argument("--bins", required=True); g.add_argument("--keep-csv", default=None); g.add_argument("--flat-drop", type=float, default=0.15)
    s = sub.add_parser("stages"); s.add_argument("--bins", required=True); s.add_argument("--repo", default="/path/to/repo")
    s.add_argument("--out", default=None)
    q = sub.add_parser("qe-sample"); q.add_argument("--pairs", required=True); q.add_argument("--scores", required=True); q.add_argument("--out", required=True)
    o = sub.add_parser("overhead"); o.add_argument("--windows", required=True); o.add_argument("--out", required=True); o.add_argument("--tokenizer", default="google/gemma-3-270m-it")
    a = ap.parse_args()
    if a.cmd == "qe-sample":
        import pyarrow.dataset as pads
        from common import key_of
        pr = pq.read_table(os.path.join(a.pairs, "pairs-0000.parquet"), columns=["english", "target", "iso"]).to_pydict()
        sc = pads.dataset(a.scores, format="parquet").to_table(columns=["key", "qe"]).to_pydict(); m = dict(zip(sc["key"], sc["qe"]))
        by = collections.defaultdict(list)
        for e, g, i in zip(pr["english"], pr["target"], pr["iso"]):
            k = key_of(e, i, g)
            if k in m: by[i].append(m[k])
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["iso", "keep_rate", "source", "n", "median_qe", "q15", "share_below_0.20"])
            for i, v in sorted(by.items()):
                v = np.array(v); q15 = float(np.quantile(v, 0.15)); lo = float((v < 0.20).mean())
                w.writerow([i, round(min(0.85, 1 - lo), 4), f"QE sample n={len(v)} (CPU, AfriCOMET-QE-STL-1.1), rule OLD: min(0.85, share >= 0.20)", len(v), round(float(np.median(v)), 3), round(q15, 3), round(lo, 3)])
        print("qe sample keep rates written for", len(by), "languages"); sys.exit(0)
    if a.cmd == "overhead":
        isos = sorted({r for f in glob.glob(os.path.join(a.windows, "*.parquet")) for r in pq.read_table(f, columns=["iso"])["iso"].to_pylist()})
        res = overhead(a.tokenizer, a.windows, isos)
        json.dump(res, open(a.out, "w"), indent=1); print({k: round(v, 1) for k, v in list(res.items())[:6]}, "mean", round(float(np.mean([v for v in res.values() if v])), 1)); sys.exit(0)
    {"bins": bins, "gate-estimate": gate_estimate, "stages": stages}[a.cmd](a)
