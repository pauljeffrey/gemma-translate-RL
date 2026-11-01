#!/usr/bin/env python3
"""r4: apply the owner decisions of 2026-10-29 to the LOCAL numbers of gemma_mt/calibration/clean_2026-10-28/ (no benchmark text is read: only the json files of that folder).

    python r4_apply_owner_decisions.py --dir gemma_mt/calibration/clean_2026-10-28

Decisions applied (the files written are NEW, the v1 files stay as they were):
  1. Commercial use: nothing derived from IbomMT (CC-BY-NC-4.0) may stay. efi_ibb_blocks.json is moved to _quarantine_nc/; the efi / ibb shaping entries become plain consensus entries
     with no human-reference block (like the other consensus-only languages), LID off (the proxy text of the v1 measurement contained IbomMT rows); the efi / ibb entries of lid_measure,
     funcwords and the anchors v1/v2 are dropped.
  2. Pooled prior (no per-language entry) for languages without clean human references (the 16 anchors-v3 languages and the 17 whose test-set data was removed): anchors_clean_v3.json = anchors v2
     without efi / ibb; the pooled blocks still contain efi / ibb units -> flagged `needs_refit` (the script r4_run_all.py refits them without IbomMT).
  3. pcm: pooled prior, LID off (already so in shaping_clean_v1; re-asserted).
  4. Omission cost fixed at 4.0 for every language (false-alarm rate at 4 kept as information).
  5. Long-document repetition limits from the FLORES-only (H_cal) human documents; buckets 4-5 copy bucket 3 as before (nearest bucket).
  6. Diacritic inventory widening: min count 5 (unchanged).
"""
import argparse, copy, datetime, json, os, shutil


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True)
    a = ap.parse_args(); D = a.dir
    ld = lambda f: json.load(open(os.path.join(D, f)))
    sv = lambda f, o: json.dump(o, open(os.path.join(D, f), "w"), indent=1, ensure_ascii=False)
    log = []
    # ---- 1. quarantine the NC-derived file
    q = os.path.join(D, "_quarantine_nc"); os.makedirs(q, exist_ok=True)
    if os.path.exists(os.path.join(D, "efi_ibb_blocks.json")):
        shutil.move(os.path.join(D, "efi_ibb_blocks.json"), os.path.join(q, "efi_ibb_blocks.json"))
        open(os.path.join(q, "README.txt"), "w").write("Derived from howard-nlp/ibom-mt (CC-BY-NC-4.0). Not usable commercially (owner decision 2026-10-29). Do not ship, do not push to a shared repo.\n")
        log.append("efi_ibb_blocks.json moved to _quarantine_nc/")
    # ---- 4. noqe: omission cost 4.0
    nq = ld("noqe_blocks.json"); n_changed = 0
    for iso, e in nq.items():
        if iso.startswith("_") or "omission" not in e or "false_alarm_by_cost" not in e["omission"]:
            continue
        o = e["omission"]; fa = o["false_alarm_by_cost"]["4.0"]
        if o.get("cost") != 4.0:
            n_changed += 1
        o.update({"cost": 4.0, "false_alarm": fa, "met_target": bool(fa <= o.get("target", 0.02)), "note": "owner decision 2026-10-29: cost fixed at 4.0 (the 2% rule is noisy at about 140 documents); rate at cost 4 kept"})
    nq["_meta"]["omission_policy"] = "fixed 4.0 (owner 2026-10-29)"
    sv("noqe_blocks_v2.json", nq); log.append(f"noqe_blocks_v2.json: omission cost fixed at 4.0 ({n_changed} languages changed)")
    # ---- lid / funcwords without efi, ibb
    lid = ld("lid_measure_clean.json"); [lid.pop(i, None) for i in ("efi", "ibb")]; sv("lid_measure_clean_v2.json", lid)
    fw = ld("funcwords_clean_v1.json"); [fw.pop(i, None) for i in ("efi", "ibb")]
    json.dump(fw, open(os.path.join(D, "funcwords_clean_v2.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    log.append("lid_measure_clean_v2.json, funcwords_clean_v2.json: efi / ibb removed")
    # ---- 5. longdoc FLORES-only
    ldc, rep = ld("longdoc_clean_v1.json"), ld("longdoc_clean_report.json")
    t0 = [b["limit_by_kind"]["human_flores_cal"] for b in rep["buckets"]]
    new = copy.deepcopy(ldc); new["t0"] = t0
    new["meta"].update({"source": "H_cal v2 FLORES+ pairs stitched by article (45 languages) ONLY (owner decision 2026-10-29: no AfriScience windows, no NLLB, no SmolDoc, no AfriLongDoc)",
                        "n_docs": sum(b["kinds"].get("human_flores_cal", 0) for b in rep["buckets"][:4]), "limits_replaced": "t0 from limit_by_kind.human_flores_cal; buckets 4-5 = bucket 3 (nearest, as before)"})
    sv("longdoc_clean_v2.json", new); log.append("longdoc_clean_v2.json: FLORES-only limits " + json.dumps([{k: v for k, v in x.items() if k in ("rep4", "cover8")} for x in t0[:4]]))
    # ---- anchors v3 (= v2 without efi/ibb; pooled prior for everything without an entry)
    a2 = ld("anchors_clean_v2.json"); a3 = copy.deepcopy(a2)
    for i in ("efi", "ibb"):
        a3["languages"].pop(i, None)
    a3["_meta"]["pool_languages"] = [i for i in a3["_meta"].get("pool_languages", []) if i not in ("efi", "ibb")]
    a3["_meta"]["needs_refit"] = ["pooled entity / repeat / boiler blocks were fitted WITH the efi / ibb units (IbomMT train): refit without them (r4_run_all.py does it)"]
    a3["_meta"]["languages_with_entry"] = sorted(a3["languages"])
    a3["_meta"]["pooled_prior_for"] = "every language without an entry (the 16 anchors-v3 languages urh sid wal bas dua yao wes oke bbj bin ish iso swc kon kri pcm, plus ach dag her kqn kua loz lun nbl ndc nyn shi tiv tsc ven efi ibb): owner decision 2026-10-29"
    a3["_meta"]["date"] = datetime.date.today().isoformat()
    if "v2_meta" in a3:                                    # the builder prints a stale label of the OLD inputs; the real ones of this file:
        a3["v2_meta"]["humans"] = "H_cal v2 FLORES+ pairs + SSA-MTE train references (12 languages, CC-BY-SA) + AfriScience-MT train papers as documents (clean)"
    a3.get("proxies", {}).pop("wes", None)
    sv("anchors_clean_v3.json", a3); log.append("anchors_clean_v3.json: efi / ibb removed, pooled prior elsewhere, pooled blocks flagged needs_refit")
    # ---- shaping v2
    s = ld("shaping_clean_v1.json"); sh = copy.deepcopy(s)
    for iso, e in sh.items():
        if iso.startswith("_") or not isinstance(e, dict):
            continue
        if iso in nq and isinstance(nq[iso], dict) and nq[iso].get("omission") and iso in sh and "omission" in e:
            e["omission"] = nq[iso]["omission"]
    for iso in ("efi", "ibb"):
        sh[iso] = {"usable": True, "n": 0, "reliability": 0.5, "reliability_same_source": 0.5, "weight_provisional": True,
                   "content": {"mode": "consensus", "beta": 1.0, "why": "QE-free consensus; NO human-reference block: IbomMT (the only reference set) is CC-BY-NC and was removed (owner 2026-10-29), pooled-prior behaviour as the other consensus-only languages"},
                   "weight_reason": "consensus-only default 0.5 (no model-dependent validation possible without IbomMT hold-out)",
                   "lid": {"enabled": False, "labels": [f"{iso}_Latn"], "reason": "no commercially usable human or proxy text measured yet; r4_run_all.py re-measures efi on the cc-by mt560 Efik rows if enough (>= 250) survive"},
                   "calibration_source": "none"}
    sh["pcm"]["lid"] = dict(sh["pcm"].get("lid", {}), enabled=False, reason="owner 2026-10-29: pcm LID off, pooled prior")
    sh["pcm"]["content"] = {"mode": "blend", "beta": 0.5, "why": "no clean human reference set for Pidgin: pooled prior, LID off (owner 2026-10-29)"}
    sh["_meta"].update({"version": "v2 (owner decisions 2026-10-29 applied)", "omission_policy": "fixed 4.0", "efi_ibb": "no IbomMT-derived value", "ibom_removed": True})
    sh["_meta"]["incomplete"] = ["AfriCOMET anchors / AUC / reliability for 39 of 42 QE languages (r4_run_all.py), qe2 / ens (SSA-COMET-MTL), pooled-prior anchors recomputed from all 42",
                                 "LID of efi (and ibb) if a commercially usable proxy is found", "lexicons (44 weak languages)", "model-dependent items: delta check, consensus weights, NLLB rule, tail_gap"]
    sv("shaping_clean_v2.json", sh); log.append("shaping_clean_v2.json written (3 QE languages scored; efi / ibb consensus without IbomMT; pcm pooled + LID off; omission 4)")
    # ---- status.json
    st = ld("status.json")
    st["_log_2026-10-29"] = {"owner_decisions": ["commercial use only: IbomMT-derived values removed (efi, ibb)", "pooled prior for languages without human references", "pcm LID off + pooled prior",
                                                  "omission cost 4", "FLORES-only long-document limits", "diacritic ext-min-count 5"], "actions": log,
                             "files_new": ["noqe_blocks_v2.json", "lid_measure_clean_v2.json", "funcwords_clean_v2.json", "longdoc_clean_v2.json", "anchors_clean_v3.json", "shaping_clean_v2.json"],
                             "files_superseded_but_kept": ["noqe_blocks.json", "lid_measure_clean.json", "funcwords_clean_v1.json", "longdoc_clean_v1.json", "anchors_clean_v1.json", "anchors_clean_v2.json", "shaping_clean_v1.json"],
                             "files_quarantined": ["_quarantine_nc/efi_ibb_blocks.json"],
                             "box_2": "LOST: refs, qe_work, sft_targets and the CPU QE jobs are gone; the Hub folder calibration/ holds the pre-decision numbers (incl. efi_ibb_blocks.json: DELETE it there)"}
    sv("status.json", st)
    print("\n".join(log))


if __name__ == "__main__":
    main()
