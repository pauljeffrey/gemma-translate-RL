#!/usr/bin/env python3
"""Projections WITHOUT the non_commercial rows (Amendment 4) and WITH the gate (rule CAL, ESTIMATE from the partial QE fit): packs, steps and arm sizes for A0, S1/S2, G0 and the 25% stratified A2 subset (T1's sampler.py select / subset).
Laptop/CPU only: scales the measured per-language/bin window statistics (bins/) by the removed token share; the real numbers come from windows.py + pack.py on the new instance.

    python project_nc.py --root <repo> --out reports/data/sft_clean_2026-10-28/bins_commercial
"""
import argparse, collections, csv, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sampler as S
from common import SPECIALIST, STAGE_A


def rd(p):
    return list(csv.DictReader(open(p)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/path/to/repo"); ap.add_argument("--out", required=True); ap.add_argument("--a2-frac", type=float, default=0.25); ap.add_argument("--cap", type=int, default=0, help="QE cap per language (qe_score --cap-per-lang): pairs outside the cap are not scored and therefore not kept; 0 = no cap")
    ap.add_argument("--extra-bins", default=None, help="bins directory of the Amendment 6 languages (project.py bins on extra_sources.py output): added to every variant")
    ap.add_argument("--nc-allow", nargs="*", default=["howard-nlp/ibom-mt"], help="owner exceptions: NC sources that stay (IbomMT, 2026-10-29 evening)"); a = ap.parse_args()
    D = os.path.join(a.root, "reports/data/sft_clean_2026-10-28"); B = D + "/bins"; os.makedirs(a.out, exist_ok=True)
    lang = {r["iso"]: r for r in rd(B + "/per_language_ungated.csv")}
    pairs_union = json.load(open(D + "/qe_cpu/queue_meta.json"))["pairs_by_language"]
    extra_rows = []
    if a.extra_bins:                                     # nde ful knc tzm (Amendment 6): per-language and per-bin window statistics of the extra sources (R1-estimated filter)
        for r in rd(a.extra_bins + "/per_language_ungated.csv"):
            lang[r["iso"]] = r; pairs_union[r["iso"]] = pairs_union.get(r["iso"], 0) + int(r["pairs"])
        extra_rows = rd(a.extra_bins + "/windows_per_language_bin_ungated.csv")
    # ---- NC rows per language: measured from the build statistics
    nc = {}
    for r in rd(D + "/licence_by_language.csv"):                       # main output: pairs, ESTIMATED target tokens (chars / chars-per-token)
        if r["licence_class"] == "non_commercial":
            nc[r["iso"]] = {"pairs": int(r["pairs_after"]), "tokens": int(r["est_target_tokens"]), "source": "howard-nlp/ibom-mt (main output)"}
    for r in rd(D + "/specialist/per_language_source.csv"):           # specialist set: pairs and exact Gemma target tokens, by source
        if r["licence_class"] == "non_commercial":
            x = nc.setdefault(r["iso"], {"pairs": 0, "tokens": 0, "source": ""})
            x["pairs"] += int(r["pairs"]); x["tokens"] += int(r["gemma_target_tokens"]); x["source"] += ("; " if x["source"] else "") + f"{r['source']} ({int(r['pairs']):,})"
    # ---- owner exceptions (--nc-allow): their rows stay; subtract them (pairs from per_language_source.csv, tokens in proportion of pairs)
    allowed_back = {}
    for r in rd(D + "/per_language_source.csv"):
        if r["source_repo"] in a.nc_allow and r["iso"] in nc:
            x = nc[r["iso"]]; n_ = int(r["pairs_after"]); share = min(1.0, n_ / max(x["pairs"], 1))
            allowed_back[r["iso"]] = (n_, int(x["tokens"] * share), r["source_repo"])
            x["pairs"] -= n_; x["tokens"] -= int(x["tokens"] * share)
            if x["pairs"] <= 0: del nc[r["iso"]]
    json.dump({k: {"pairs_kept": v[0], "tokens_kept": v[1], "source": v[2]} for k, v in allowed_back.items()}, open(a.out + "/nc_owner_exception_kept.json", "w"), indent=1)
    with open(a.out + "/nc_removed_per_language.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "pairs_removed", "target_tokens_removed", "pairs_before", "share_of_pairs_removed_pct", "sources"])
        for iso, x in sorted(nc.items(), key=lambda kv: -kv[1]["pairs"]):
            pb = pairs_union.get(iso, 0); w.writerow([iso, x["pairs"], x["tokens"], pb, round(100 * x["pairs"] / max(pb, 1), 2), x["source"]])
    # ---- scale factors (token share; ibb loses everything)
    f_tok = {}
    for iso, l in lang.items():
        tot = float(l["target_tokens_gemma"]); rm = nc.get(iso, {}).get("tokens", 0)
        f_tok[iso] = 0.0 if (iso in nc and pairs_union.get(iso, 0) <= nc[iso]["pairs"]) else max(0.0, 1 - rm / tot)
    # ---- keep rates of the gate: rule CAL (partial fit, ~155 pairs per language) and OLD (85%)
    fit = {r["iso"]: r for r in rd(D + "/qe_cpu/keep_rate_per_language.csv")}
    keep_cal = {i: float(r["keep_rate_cal"]) for i, r in fit.items()}; keep_old = {i: float(r["keep_rate_old"]) for i, r in fit.items()}
    capf = (lambda i: min(1.0, a.cap / max(1, pairs_union.get(i, 0)))) if a.cap > 0 else (lambda i: 1.0)       # share of a language's pairs inside the cap (the hash sample is uniform, so tokens and windows scale alike)
    variants = {"ungated_with_NC(old projection)": (lambda i: 1.0, False), "commercial_ungated": (lambda i: 1.0, True), "commercial_CAL_ESTIMATE": (lambda i: keep_cal.get(i, 0.85), True),
                "commercial_OLD_ESTIMATE": (lambda i: keep_old.get(i, 0.85), True),
                "commercial_CAL_ESTIMATE_ful50": (lambda i: 0.5 if i == "ful" else keep_cal.get(i, 0.85), True)}      # sensitivity: the NLLB-mined Fulah set (92% of ful) keeps only half at the QE gate
    summ = []
    for name, (kr, drop_nc) in variants.items():
        rows = []
        for iso, l in lang.items():
            f = (f_tok.get(iso, 1.0) if drop_nc else 1.0) * kr(iso) * capf(iso)
            rows.append([iso, int(float(l["windows"]) * f), int(float(l["target_tokens_gemma"]) * f), int(float(l["sample_tokens_gemma"]) * f), max(0, math.ceil(int(l["packs"]) * f)) if f > 0 else 0, round(f, 4)])
        p = a.out + "/" + name.split("(")[0]
        with open(p + "_per_language.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["iso", "windows", "target_tokens_gemma", "sample_tokens_gemma", "packs", "combined_factor"]); w.writerows(sorted(rows, key=lambda r: -r[2]))
        packs = {r[0]: r[4] for r in rows if r[4] > 0}
        with open(p + "_packs.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["iso", "packs"]); [w.writerow([k, v]) for k, v in sorted(packs.items())]
        base = a.out + "/sched_" + name.split("(")[0]
        S.plan(argparse.Namespace(stage="A", packed_dir=None, packs_csv=p + "_packs.csv", out=base + "/A0", seed=1234, steps=0, per_step=0, alpha=STAGE_A["alpha"], budget_tokens=0))
        S.arms(argparse.Namespace(schedule=base + "/A0", out=base))
        S.select(argparse.Namespace(schedule=base + "/A0", out=base + "/G0", languages="amh,tir", name="G0"))
        S.subset(argparse.Namespace(schedule=base + "/A0", out=base + "/A2", frac=a.a2_frac, stratify="language", seed=1234, languages=None, name="A2", floor_draws=0, method="systematic"))
        npk = {k: int(np.load(base + "/" + k + "/schedule.npy").size) for k in ("A0", "S1", "G0", "A2")}
        spec_packs = sum(v for k, v in packs.items() if k in SPECIALIST)
        summ.append([name, sum(r[1] for r in rows), sum(r[2] for r in rows), sum(packs.values()), npk["A0"] // 32, npk["S1"] // 32, npk["G0"] // 32, npk["A2"] // 32, npk["A0"], npk["S1"], npk["G0"], npk["A2"], spec_packs, len(packs)])
        binrows = collections.defaultdict(lambda: np.zeros(4))
        for r in rd(B + "/windows_per_language_bin_ungated.csv") + extra_rows:
            fct = (f_tok.get(r["iso"], 1.0) if drop_nc else 1.0) * kr(r["iso"]) * capf(r["iso"])
            binrows[r["bin"]] += np.array([float(r["windows"]), float(r["english_tokens_afroxlmr"]), float(r["target_tokens_gemma"]), float(r["packs_apportioned"])]) * fct
        with open(p + "_per_bin_totals.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["bin", "windows", "english_tokens_afroxlmr", "target_tokens_gemma", "packs_apportioned"])
            for b_ in ("b64", "b256", "b512", "b1024"): w.writerow([b_, *[int(x) for x in binrows[b_]]])
        with open(p + "_arm_draws_per_language.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["iso", "packs", "A0_draws", "A0_epochs", "S1_draws", "G0_draws", "A2_draws"])
            ps = np.load(base + "/A0/pack_lang.npy"); PD, ALL = S._pd()
            cnt = {k: np.bincount(ps[np.load(base + "/" + k + "/schedule.npy").reshape(-1)], minlength=len(ALL)) for k in ("A0", "S1", "G0", "A2")}
            hv = np.bincount(ps, minlength=len(ALL))
            for i, iso in enumerate(ALL):
                if hv[i]: w.writerow([iso, int(hv[i]), int(cnt["A0"][i]), round(cnt["A0"][i] / hv[i], 3), int(cnt["S1"][i]), int(cnt["G0"][i]), int(cnt["A2"][i])])
    with open(a.out + "/arms_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["variant", "windows", "target_tokens_gemma", "packs", "A0_steps", "S1_steps", "G0_steps", "A2_steps", "A0_draws", "S1_draws", "G0_draws", "A2_draws", "specialist_packs", "languages_with_data"]); w.writerows(summ)
    for r in summ: print(r)
    # ---- languages under 2,000 pairs after NC removal and the gate (CAL ESTIMATE)
    with open(a.out + "/languages_under_2000_pairs_ESTIMATE.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["iso", "pairs_before", "nc_pairs_removed", "pairs_after_NC", "keep_rate_cal_est", "pairs_after_NC_and_CAL_est", "pairs_after_NC_and_OLD_est", "status"])
        PD_, ALL_ = S._pd()
        allisos = sorted(set(pairs_union) | set(fit) | set(lang) | set(ALL_))
        for iso in sorted(allisos, key=lambda i: pairs_union.get(i, 0)):
            pb = pairs_union.get(iso, 0); nr = nc.get(iso, {}).get("pairs", 0); pn = pb - nr
            kc = keep_cal.get(iso, 0.85); ko = keep_old.get(iso, 0.85)
            if pn * kc < 2000 or pn * ko < 2000:
                w.writerow([iso, pb, nr, pn, kc, int(pn * kc), int(pn * ko), "no data at all" if pb == 0 else ("all rows were NC" if pn == 0 else "below 2,000")])
    print(open(a.out + "/languages_under_2000_pairs_ESTIMATE.csv").read())
    with open(a.out + "/cap_language_table_CAL_ESTIMATE.csv", "w", newline="") as fh:         # pairs before cap / after cap / after gate (CAL keep rates of the partial fit; 0.85 where none)
        w = csv.writer(fh); w.writerow(["iso", "pairs_before_cap", "cap", "pairs_after_cap", "keep_rate_cal_est", "pairs_after_gate_est"])
        for iso in sorted(pairs_union, key=lambda i: -pairs_union[i]):
            pb = pairs_union[iso] - nc.get(iso, {}).get("pairs", 0)
            if pb <= 0: continue
            pc = int(pb * capf(iso)); w.writerow([iso, pb, a.cap, pc, round(keep_cal.get(iso, 0.85), 4), int(pc * keep_cal.get(iso, 0.85))])


if __name__ == "__main__":
    main()
