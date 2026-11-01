#!/usr/bin/env python3
"""STEP (b): the QE gate. BOTH rules are computed side by side and the keep rate per language is reported BEFORE anything is dropped; `apply` then writes the keep list of one rule.

    python gate.py fit   --src data data_devhalf specialist --scores scores --refs-scores scores_refs --out gate [--allow-stub]
    python gate.py apply --rule old|cal --src ... --scores scores --out gate        # writes gate/keep_<rule>.parquet (keys to keep) and gate/gated_<rule>_summary.csv

Rule OLD (gemma_gate_pairs.py, unchanged): threshold = max(min_qe 0.20, 15th percentile of the language's scored pairs); keep qe >= threshold; a language with fewer than 2,000
kept pairs (or not in the language table) is dropped.
Rule CAL (new, pre-registered intent: language-calibrated on human text): threshold = max(0.20, 10th percentile of the QE of that language's HUMAN references (FLORES+ dev H_cal,
scored with the same model)); languages without FLORES dev use the POOLED MEDIAN OFFSET d = median over languages with references of (P10(ref QE) - P50(training QE)):
threshold = max(0.20, P50(training QE of that language) + d). Cap: no language loses more than 40% (threshold <= P40 of the language's scored pairs). Floor: at least 5% are
dropped (threshold >= P05). Min 2,000 rows as in OLD. The 0.20 minimum applies before cap and floor, so the cap can lower it for a very weak language (reported in `binding`).
Stub scores (scorer column STUB-NOT-REAL) are refused unless --allow-stub, and then every output file is prefixed STUB_.
"""
import argparse, collections, csv, json, os, sys
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import pyarrow.dataset as pads
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import key_of, shard_files

MIN_QE, MIN_ROWS = 0.20, 2000          # --min-rows overrides (tests on small samples)


def load_scores(dirs, allow_stub):
    """one or several score directories (CPU and GPU shards are interchangeable; the FIRST score per key wins, directories in the given order)"""
    dirs = [dirs] if isinstance(dirs, str) else list(dirs)
    tabs = []
    for d in dirs:
        if os.path.isdir(d) and any(f.endswith(".parquet") for f in os.listdir(d)):
            tabs.append(pads.dataset(d, format="parquet").to_table(columns=["key", "qe", "scorer"]))
    t = pa.concat_tables(tabs)
    scorer = set(t["scorer"].unique().to_pylist())
    stub = any("STUB" in s for s in scorer)
    if stub and not allow_stub:
        raise SystemExit(f"{dirs}: STUB scores (not real QE). Re-run with --allow-stub only for pipeline tests; outputs are then prefixed STUB_.")
    k = t["key"].to_numpy(); q = t["qe"].to_numpy()
    o = np.argsort(k, kind="stable"); return k[o], q[o], stub


def collect(src, sk, sq, exclude_nc=False, allow=()):
    """(keys, iso_names, iso_idx, qe) of the unique scored pairs of the source dirs"""
    names, keys, isos, qes = [], [], [], []
    all_k, all_i = [], []                                    # every scoreable pair (scored or not): the "before cap" counts
    seen_names = {}
    for f in shard_files(src):
        pf = pq.ParquetFile(f)
        cols = [c for c in ("key", "english", "target", "iso", "licence", "source_dataset") if c in pf.schema_arrow.names]
        t = pf.read(columns=cols).to_pydict()
        n = len(t["iso"])
        k = np.array(t["key"], np.uint64) if "key" in t else np.array([key_of(t["english"][i], t["iso"][i], t["target"][i]) for i in range(n)], np.uint64)
        ii = np.array([seen_names.setdefault(x, len(seen_names)) for x in t["iso"]], np.int32)
        pos = np.searchsorted(sk, k); pos[pos >= len(sk)] = len(sk) - 1
        hit = sk[pos] == k
        scoreable = np.ones(n, bool)
        if exclude_nc and "licence" in t:
            sd = t.get("source_dataset") or [""] * n
            scoreable = np.array([(x != "non_commercial") or (s_.split(":")[0] in allow) for x, s_ in zip(t["licence"], sd)])
            hit &= scoreable
        all_k.append(k[scoreable]); all_i.append(ii[scoreable])
        keys.append(k[hit]); isos.append(ii[hit]); qes.append(sq[pos[hit]])
    keys, isos, qes = np.concatenate(keys), np.concatenate(isos), np.concatenate(qes)
    _, first = np.unique(keys, return_index=True)
    ak, ai = np.concatenate(all_k), np.concatenate(all_i)
    _, f2 = np.unique(ak, return_index=True)
    bc = np.bincount(ai[f2], minlength=len(seen_names))
    collect.before_cap = {nm: int(bc[j]) for nm, j in seen_names.items()}
    return keys[first], list(seen_names), isos[first], qes[first]


def fit(a):
    sk, sq, stub = load_scores(a.scores, a.allow_stub)
    a.min_scored = getattr(a, "min_scored", 0)
    keys, names, iso_i, qe = collect(a.src, sk, sq, getattr(a, "exclude_nc", False), set(x.partition("=")[0] for x in (getattr(a, "nc_allow", None) or [])))
    pre = "STUB_" if stub else ""
    refs = {}
    rj = getattr(a, "refs_json_dir", None)
    if rj:                                           # QE scores of the H_cal human references computed by the calibration agent (heldout_split_cal_sel_v2): <dir>/<iso>.human.json = {iso, n, ids, scores}
        for fn in sorted(os.listdir(rj)):
            if fn.endswith(".human.json"):
                d = json.load(open(os.path.join(rj, fn)))
                refs[d["iso"]] = [float(x) for x in d["scores"]]
    if a.refs_scores:
        rk, rq, rstub = load_scores(a.refs_scores, a.allow_stub)
        rt = pq.read_table(os.path.join(a.refs, "pairs-0000.parquet"), columns=["key", "iso"]).to_pydict()
        pos = np.searchsorted(rk, np.array(rt["key"], np.uint64)); pos[pos >= len(rk)] = len(rk) - 1
        for k, iso, p in zip(rt["key"], rt["iso"], pos):
            if rk[p] == k:
                refs.setdefault(iso, []).append(float(rq[p]))
    res, pooled = {}, []
    totals = {}
    if getattr(a, "totals", None):
        totals = {r["iso"]: int(r["pairs_after"]) for r in csv.DictReader(open(a.totals))}       # per_language.csv of the build: extrapolate the sample keep rate to all pairs
    for ii, name in enumerate(names):
        v = qe[iso_i == ii]
        if len(v) == 0 or len(v) < a.min_scored:
            continue
        q05, q15, q40, q50 = (float(np.quantile(v, x)) for x in (0.05, 0.15, 0.40, 0.50))
        r = {"n_scored": int(len(v)), "n_before_cap": collect.before_cap.get(name), "q05": q05, "q15": q15, "q40": q40, "median": q50}
        r["thr_old"] = max(MIN_QE, q15)
        r["keep_old"] = int((v >= r["thr_old"]).sum())
        if name in refs and len(refs[name]) >= a.min_refs:
            p10 = float(np.quantile(refs[name], 0.10)); r["n_refs"] = len(refs[name]); r["ref_p10"] = p10; r["ref_median"] = float(np.median(refs[name]))
            r["cal_source"] = "own FLORES+ dev H_cal references"
            pooled.append(p10 - q50)
        res[name] = r
    d_off = float(np.median(pooled)) if pooled else 0.0
    for name, r in res.items():
        v = qe[iso_i == names.index(name)]
        if r.get("cal_source"):
            raw = max(MIN_QE, r["ref_p10"])
        else:
            raw = max(MIN_QE, r["median"] + d_off); r["cal_source"] = f"pooled median offset {d_off:+.4f} (no H_cal references)"; r["n_refs"] = 0
        thr, binding = raw, "none"
        if thr > r["q40"]:
            thr, binding = r["q40"], "cap40pct"
        if thr < r["q05"]:
            thr, binding = r["q05"], "floor5pct"
        r["thr_cal_raw"], r["thr_cal"], r["cal_binding"] = raw, thr, binding
        r["keep_cal"] = int((v >= thr).sum())
        for rule in ("old", "cal"):
            r[f"keep_rate_{rule}"] = r[f"keep_{rule}"] / r["n_scored"]
            nt = totals.get(name)
            r["n_total_pairs"] = nt
            r[f"est_keep_total_{rule}"] = int(r[f"keep_rate_{rule}"] * nt) if nt else None
            r[f"min_rows_ok_{rule}"] = (r[f"est_keep_total_{rule}"] if nt else r[f"keep_{rule}"]) >= a.min_rows
    os.makedirs(a.out, exist_ok=True)
    json.dump({"pooled_offset": d_off, "n_languages_with_refs": len(pooled), "languages": res, "stub": stub, "min_qe": MIN_QE, "min_rows": a.min_rows}, open(f"{a.out}/{pre}thresholds.json", "w"), indent=1)
    cols = ["iso", "n_pairs_before_cap", "n_scored", "median_qe", "q05", "q15", "q40", "thr_old", "keep_rate_old", "keep_old", "n_refs", "ref_p10", "cal_source", "thr_cal_raw", "thr_cal", "cal_binding", "keep_rate_cal", "keep_cal", "n_total_pairs", "est_keep_total_old", "est_keep_total_cal", "min_rows_ok_old", "min_rows_ok_cal"]
    with open(f"{a.out}/{pre}keep_rate_per_language.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for name, r in sorted(res.items(), key=lambda kv: -kv[1]["n_scored"]):
            w.writerow([name, r["n_before_cap"], r["n_scored"], round(r["median"], 4), round(r["q05"], 4), round(r["q15"], 4), round(r["q40"], 4), round(r["thr_old"], 4), round(r["keep_rate_old"], 4), r["keep_old"],
                        r["n_refs"], round(r.get("ref_p10", float("nan")), 4), r["cal_source"], round(r["thr_cal_raw"], 4), round(r["thr_cal"], 4), r["cal_binding"], round(r["keep_rate_cal"], 4),
                        r["keep_cal"], r["n_total_pairs"], r["est_keep_total_old"], r["est_keep_total_cal"], int(r["min_rows_ok_old"]), int(r["min_rows_ok_cal"])])
    tot = sum(r["n_scored"] for r in res.values())
    print(f"{len(res)} languages, {tot:,} scored pairs; keep OLD {sum(r['keep_old'] for r in res.values()) / tot:.3f}, keep CAL {sum(r['keep_cal'] for r in res.values()) / tot:.3f}; "
          f"pooled offset {d_off:+.4f} from {len(pooled)} languages with references; NOTHING DROPPED (fit only)" + ("  [STUB scores]" if stub else ""))


def apply_(a):
    sk, sq, stub = load_scores(a.scores, a.allow_stub)
    pre = "STUB_" if stub else ""
    th = json.load(open(f"{a.out}/{pre}thresholds.json"))["languages"]
    keys, names, iso_i, qe = collect(a.src, sk, sq, getattr(a, "exclude_nc", False), set(x.partition("=")[0] for x in (getattr(a, "nc_allow", None) or [])))
    keep = np.zeros(len(keys), bool)
    bc = collect.before_cap
    summ = collections.Counter(); per = collections.defaultdict(lambda: [0, 0])
    for ii, name in enumerate(names):
        r = th.get(name)
        m = iso_i == ii
        if r is None or not r[f"min_rows_ok_{a.rule}"]:
            per[name][0] += int(m.sum()); continue                       # language dropped (min rows)
        k = m & (qe >= r["thr_" + a.rule])
        keep |= k; per[name][0] += int(m.sum()); per[name][1] += int(k.sum())
    pq.write_table(pa.table({"key": pa.array(keys[keep], pa.uint64())}), f"{a.out}/{pre}keep_{a.rule}.parquet", compression="zstd")
    with open(f"{a.out}/{pre}gated_{a.rule}_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "pairs_before_cap", "scored_after_cap", "kept_after_gate", "kept_rate_of_scored", "scored_share_of_pairs"])
        for k, (n, kk) in sorted(per.items(), key=lambda kv: -kv[1][0]): w.writerow([k, bc.get(k, n), n, kk, round(kk / max(1, n), 4), round(n / max(1, bc.get(k, n)), 4)])
    print(f"rule {a.rule}: kept {int(keep.sum()):,} of {len(keys):,} unique scored pairs ({keep.mean():.3f}); pairs before the cap {sum(bc.values()):,}; NOT scored = NOT kept" + ("  [STUB]" if stub else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("fit", "apply"):
        s = sub.add_parser(n); s.add_argument("--src", nargs="+", required=True); s.add_argument("--scores", nargs="+", required=True); s.add_argument("--out", required=True); s.add_argument("--allow-stub", action="store_true")
        s.add_argument("--refs-scores", nargs="*", default=None); s.add_argument("--refs-json-dir", default=None, help="dir with <iso>.human.json (QE of the H_cal human references, calibration agent, split v2); overrides --refs-scores for those languages"); s.add_argument("--refs", default="refs_hcal"); s.add_argument("--min-refs", type=int, default=30); s.add_argument("--nc-allow", nargs="*", default=[], help="owner exceptions that stay in the thresholds and the keep list (repo[=tag])"); s.add_argument("--exclude-nc", action="store_true", help="leave rows of licence class non_commercial out of the thresholds and the keep list"); s.add_argument("--min-rows", type=int, default=MIN_ROWS); s.add_argument("--min-scored", type=int, default=0, help="skip languages with fewer scored pairs"); s.add_argument("--totals", default=None, help="per_language.csv (iso, pairs_after): extrapolate sample keep rates to all pairs")
        if n == "apply": s.add_argument("--rule", choices=["old", "cal"], required=True)
    a = ap.parse_args()
    (fit if a.cmd == "fit" else apply_)(a)
