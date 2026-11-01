#!/usr/bin/env python3
"""r4: the calibrated objects of the shaping file that need NO model: length ratio window, Gale-Church variance, omission cost, diacritic block, lexical block, word-ratio block.

    python r4_noqe.py --cal refs/cal.jsonl --sel refs/sel.jsonl --split heldout_split_cal_sel_v2.json --code <repo gemma_mt/> --sft-targets sft_targets.jsonl \
        --old shaping_flores_dev_v8.json --out clean/noqe_blocks.json --report clean/noqe_report.json

Fitted on H_cal (v2) human references only (split label asserted). The SFT-target sample (clean T1 text) only WIDENS the diacritic inventory, and only with clusters / marks / letters
that occur >= --ext-min-count times in the sample (the human references are 261 sentences instead of 997, so the inventory of rare clusters would otherwise be too narrow and would
penalise valid text). H_sel (role S) is used for ONE thing: the false-penalty rate of the diacritic guard and of the length window on held-out human text, old (v8) vs new, reported
only (no value is tuned on it). Writes numbers only (the report contains no sentence).
"""
import argparse, collections, json, os, random, sys, unicodedata


def nfc(s):
    return unicodedata.normalize("NFC", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", required=True); ap.add_argument("--sel", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--code", required=True); ap.add_argument("--sft-targets", default=None); ap.add_argument("--old", default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--report", required=True)
    ap.add_argument("--ext-min-count", type=int, default=5)
    ap.add_argument("--omission-target", type=float, default=0.02)
    ap.add_argument("--omission-fixed", type=float, default=None, help="owner 2026-10-29: use this omission cost for every language (4.0); the false-alarm rate at it is kept")
    ap.add_argument("--tilings", type=int, default=6)
    a = ap.parse_args()
    sys.path.insert(0, a.code); sys.path.insert(0, os.path.join(a.code, "scripts")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from r4_qe_score import load_rows
    from gemma_mt.rl import diacritics, lexical
    from gemma_mt.rl.omission_eval import length_stats, choose_omission_cost
    from build_word_ratio_calibration import word_ratio_block, pooled

    cal = load_rows(a.cal, "calibration", a.split)
    sel = load_rows(a.sel, "selection", a.split)
    by, bysel = collections.defaultdict(list), collections.defaultdict(list)
    for r in cal:
        by[r["iso"]].append(dict(r, english=nfc(r["english"]), reference=nfc(r["reference"])))
    for r in sel:
        bysel[r["iso"]].append(dict(r, english=nfc(r["english"]), reference=nfc(r["reference"])))
    ext = collections.defaultdict(list)
    if a.sft_targets and os.path.exists(a.sft_targets):
        for l in open(a.sft_targets, encoding="utf-8"):
            r = json.loads(l)
            ext[r["iso"]].append(r["text"])
    old = json.load(open(a.old)) if a.old else {}
    out, rep = {}, {}
    wr_blocks = {}
    for iso in sorted(by):
        rows = sorted(by[iso], key=lambda r: r["id"])
        src = [r["english"] for r in rows]; ref = [r["reference"] for r in rows]
        ratios = sorted(len(r) / max(len(s), 1) for s, r in zip(src, ref))
        ls = length_stats(src, ref)
        e = {"n": len(rows), "len_ratio": {"p05": ratios[int(0.05 * (len(ratios) - 1))], "p50": ratios[len(ratios) // 2], "p95": ratios[int(0.95 * (len(ratios) - 1))],
                                           "p01": ls["p01"], "p02": ls["p02"], "corpus": ls["corpus"], "s2": ls["s2"]}}
        # diacritics: H_cal references define density / threshold; the inventory is widened by frequency-filtered clean training text
        d0 = diacritics.summarise(ref)
        if d0:
            if ext.get(iso):
                dx = diacritics.summarise(ref, extra_refs=ext[iso])
                cnt_ext = collections.defaultdict(collections.Counter)
                for t in ext[iso]:
                    for k, c in diacritics.inventory(nfc(t)).items():
                        cnt_ext[k].update(c)
                allowed = {k: dict(v) for k, v in d0["allowed"].items()}
                added = 0
                for k, c in cnt_ext.items():
                    for h, n in c.items():
                        if n >= a.ext_min_count and h not in allowed[k]:
                            allowed[k][h] = n; added += 1
                d0 = dict(d0, allowed=allowed, inventory_widened_by_clean_sft_targets={"n_texts": len(ext[iso]), "min_count": a.ext_min_count, "entries_added": added})
            e["diacritics"] = d0
        lex = lexical.summarise(zip(src, ref))
        if lex:
            e["lexical"] = lex
        wb = word_ratio_block(list(zip(src, ref)))
        if wb:
            e["word_ratio"] = wb; wr_blocks[iso] = wb
        # omission cost (D-30 rule: smallest cost with <= 2% false alarms on untouched human documents); windows of 3-12 consecutive H_cal pairs (H_cal ids are not contiguous in
        # the article, the documents are parallel but not coherent: stated in the report)
        try:
            e["omission"] = choose_omission_cost([dict(r) for r in rows], a.omission_target, tilings=a.tilings)
            if a.omission_fixed is not None and "false_alarm_by_cost" in e["omission"]:
                fa = e["omission"]["false_alarm_by_cost"][str(float(a.omission_fixed))]
                e["omission"].update({"cost": float(a.omission_fixed), "false_alarm": fa, "met_target": bool(fa <= a.omission_target), "note": "cost fixed by the owner (2026-10-29)"})
        except Exception as ex:                                    # noqa
            e["omission"] = {"error": str(ex)[:120]}
        out[iso] = e
        # report: old (v8) vs new, numbers only; false-penalty rate on H_sel
        o = old.get(iso, {})
        r_ = {"n_cal": len(rows), "n_sel": len(bysel.get(iso, []))}
        for k in ("p05", "p50", "p95", "p02", "corpus", "s2"):
            if o.get("len_ratio"):
                r_["len_" + k] = [round(o["len_ratio"][k], 3), round(e["len_ratio"][k], 3)]
        if o.get("omission") and "cost" in e["omission"]:
            r_["omission_cost"] = [o["omission"]["cost"], e["omission"]["cost"]]; r_["omission_false_alarm"] = [round(o["omission"].get("false_alarm", -1), 3), round(e["omission"]["false_alarm"], 3)]
        if e.get("diacritics") and o.get("diacritics"):
            r_["dia_thr"] = [round(o["diacritics"]["threshold"], 4), round(e["diacritics"]["threshold"], 4)]
            r_["dia_inventory_clusters"] = [len(o["diacritics"]["allowed"]["clusters"]), len(e["diacritics"]["allowed"]["clusters"])]
            # held-out false-penalty rate: share of H_sel human references the diacritic guard would punish
            for tag, ent in (("old", o["diacritics"]), ("new", e["diacritics"])):
                if ent.get("applies"):
                    r_["dia_false_penalty_sel_" + tag] = round(sum(1 for x in bysel.get(iso, []) if diacritics.deficit(x["reference"], ent) > 0) / max(len(bysel.get(iso, [])), 1), 4)
        if bysel.get(iso) and o.get("len_ratio"):
            for tag, lr in (("old", o["len_ratio"]), ("new", e["len_ratio"])):
                r_["len_window_outside_sel_" + tag] = round(sum(1 for x in bysel[iso] if not lr["p02"] * 0.9 <= len(x["reference"]) / max(len(x["english"]), 1) <= lr["p95"] * 1.1) / len(bysel[iso]), 4)
        rep[iso] = r_
        print(iso, json.dumps(r_), flush=True)
    out["_word_ratio_pooled"] = pooled(wr_blocks)
    out["_meta"] = {"split_file": os.path.basename(a.split), "split_label": "calibration", "n_languages": len(by), "ext_min_count": a.ext_min_count, "omission_target": a.omission_target}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    json.dump(rep, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    main()
