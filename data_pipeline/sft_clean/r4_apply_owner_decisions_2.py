#!/usr/bin/env python3
"""r4: owner update of 2026-10-29 evening, applied to the LOCAL numbers of gemma_mt/calibration/clean_2026-10-28/ (json only, no benchmark text).

    python r4_apply_owner_decisions_2.py --dir gemma_mt/calibration/clean_2026-10-28

1. IbomMT (efi, ibb) is an explicit owner EXCEPTION: the efi / ibb consensus entries (D-43 rule on the IbomMT TRAIN calibration halves), their LID entries, function-word entries and anchors units are
   restored from the 2026-10-28 files and tagged  nc_exception: "ibom-mt"  (so they can be disclosed or dropped later). The rest of the NC exclusion stays.
2. Function-word tables and bigram statistics are statistics only: no licence filter (all clean SFT text); metadata says so (key `_statistics_only`).
Outputs (new files, nothing overwritten): shaping_clean_v3.json, lid_measure_clean_v3.json, funcwords_clean_v3.json, anchors_clean_v4.json, efi_ibb_blocks.json (moved back from _quarantine_nc/).
"""
import argparse, copy, datetime, json, os, shutil

NC = "ibom-mt"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True)
    a = ap.parse_args(); D = a.dir
    ld = lambda f: json.load(open(os.path.join(D, f)))
    sv = lambda f, o: json.dump(o, open(os.path.join(D, f), "w"), indent=1, ensure_ascii=False)
    log = []
    q = os.path.join(D, "_quarantine_nc", "efi_ibb_blocks.json")
    if os.path.exists(q):
        shutil.move(q, os.path.join(D, "efi_ibb_blocks.json"))
        shutil.rmtree(os.path.join(D, "_quarantine_nc"), ignore_errors=True)
        log.append("efi_ibb_blocks.json restored from _quarantine_nc/ (quarantine folder removed)")
    eb = ld("efi_ibb_blocks.json"); eb["_nc_exception"] = {"source": NC, "licence": "CC-BY-NC-4.0", "owner_exception": "2026-10-29", "tag": f"nc_exception: {NC}"}
    sv("efi_ibb_blocks.json", eb)
    # ---- shaping v3 = v2 + efi / ibb entries of v1 (IbomMT calibration half), omission cost 4, tagged
    v1, v2 = ld("shaping_clean_v1.json"), ld("shaping_clean_v2.json"); v3 = copy.deepcopy(v2)
    for iso in ("efi", "ibb"):
        e = copy.deepcopy(v1[iso]); om = e.get("omission", {})
        if "false_alarm_by_cost" in om:
            fa = om["false_alarm_by_cost"]["4.0"]
            om.update({"cost": 4.0, "false_alarm": fa, "met_target": bool(fa <= om.get("target", 0.02)), "note": "owner decision 2026-10-29: cost fixed at 4.0"})
        e["nc_exception"] = NC
        if isinstance(e.get("calibration_source"), dict):
            e["calibration_source"]["nc_exception"] = NC
        if isinstance(e.get("lid"), dict):
            e["lid"]["nc_exception"] = NC
        v3[iso] = e
    for iso in ("bbj", "dag"):                              # no row in the clean SFT data -> no clean LID measurement; the v8 value (old SFT proxy) is replaced by OFF, as r4_assemble.py does
        if iso in v3 and isinstance(v3[iso].get("lid"), dict) and v3[iso]["lid"].get("enabled"):
            v3[iso]["lid"] = {"enabled": False, "labels": v3[iso]["lid"].get("labels", [f"{iso}_Latn"]), "reason": "no clean measurement (no rows in the clean SFT data); v8 proxy value dropped"}
    v3["_meta"].update({"version": "v3 (owner decisions 2026-10-29 evening: IbomMT exception restored)", "ibom_removed": False, "nc_exception_languages": ["efi", "ibb"],
                        "nc_exception_note": "efi / ibb entries are derived from howard-nlp/ibom-mt TRAIN (CC-BY-NC-4.0), owner exception; drop them (consensus without a reference block, see shaping_clean_v2.json) to remove the exception"})
    v3["_meta"].pop("efi_ibb", None)
    sv("shaping_clean_v3.json", v3); log.append("shaping_clean_v3.json: efi / ibb restored from shaping_clean_v1.json (omission cost 4), tagged nc_exception")
    # ---- lid v3
    l1, l2 = ld("lid_measure_clean.json"), ld("lid_measure_clean_v2.json"); l3 = copy.deepcopy(l2)
    for iso in ("efi", "ibb"):
        if iso in l1:
            l3[iso] = dict(copy.deepcopy(l1[iso]), nc_exception=NC)
    sv("lid_measure_clean_v3.json", l3); log.append("lid_measure_clean_v3.json: efi / ibb restored, tagged")
    # ---- funcwords v3 (statistics only)
    f1, f2 = ld("funcwords_clean_v1.json"), ld("funcwords_clean_v2.json"); f3 = copy.deepcopy(f2)
    for iso in ("efi", "ibb"):
        if iso in f1:
            f3[iso] = dict(copy.deepcopy(f1[iso]), nc_exception=NC)
    f3["_statistics_only"] = {"note": "word counts and document frequencies of ALL clean SFT target text per language; licence filtering does not apply to these statistics (owner 2026-10-29). "
                                    "No text is stored, only counts of the 150 most frequent words and the frequency of English function words.",
                              "licence_filter": "none", "nc_exception_languages": {"efi": NC, "ibb": NC}}
    json.dump(f3, open(os.path.join(D, "funcwords_clean_v3.json"), "w"), ensure_ascii=False, separators=(",", ":")); log.append("funcwords_clean_v3.json: efi / ibb restored, metadata statistics-only")
    # ---- anchors v4 = anchors v2 (with efi / ibb units) + pooled prior for everything without an entry
    a2 = ld("anchors_clean_v2.json"); a4 = copy.deepcopy(a2)
    for iso in ("efi", "ibb"):
        if iso in a4["languages"]:
            a4["languages"][iso]["nc_exception"] = NC
    a4["_meta"]["nc_exception_languages"] = {"efi": NC, "ibb": NC}
    a4["_meta"]["languages_with_entry"] = sorted(a4["languages"])
    a4["_meta"]["pooled_prior_for"] = "every language without an entry (the 16 anchors-v3 languages urh sid wal bas dua yao wes oke bbj bin ish iso swc kon kri pcm, plus ach dag her kqn kua loz lun nbl ndc nyn shi tiv tsc ven): owner decision 2026-10-29"
    a4["_meta"]["pooled_blocks_note"] = "the pooled blocks were fitted WITH the efi / ibb units (owner exception); pcm / wes proxy removed"
    a4["_meta"]["date"] = datetime.date.today().isoformat()
    a4.get("proxies", {}).pop("wes", None)
    if "v2_meta" in a4:
        a4["v2_meta"]["humans"] = "H_cal v2 FLORES+ pairs + SSA-MTE train references (12 languages, CC-BY-SA) + IbomMT train calibration half (efi, ibb; nc_exception) + AfriScience-MT train papers as documents (clean)"
    sv("anchors_clean_v4.json", a4); log.append("anchors_clean_v4.json: anchors v2 with efi / ibb units (tagged), pooled prior for the languages without an entry")
    # ---- status
    st = ld("status.json")
    st["_log_2026-10-29_evening"] = {"owner_updates": ["IbomMT (efi, ibb) explicit exception: entries restored and tagged nc_exception: ibom-mt", "function-word tables and bigram statistics: no licence filter, statistics only",
                                                      "AfriScience-MT train, AfriCOMET-QE and SSA-COMET accepted", "NLLB pseudo-references wanted during RL only (very poorly resourced languages): NLLB anchors / language rule stay in the plan as RL-only",
                                                      "GPU parts of r4_run_all.py run concurrently with large batches"],
                                     "actions": log, "files_new": ["shaping_clean_v3.json", "lid_measure_clean_v3.json", "funcwords_clean_v3.json", "anchors_clean_v4.json", "efi_ibb_blocks.json (restored)"],
                                     "superseded_by_this": ["shaping_clean_v2.json", "lid_measure_clean_v2.json", "funcwords_clean_v2.json", "anchors_clean_v3.json"],
                                     "still_unchanged": ["noqe_blocks_v2.json", "longdoc_clean_v2.json", "science_anchor_clean_v1.json", "monitors_and_weights_clean_v1.json"],
                                     "note": "shaping_clean_v3.json is still PARTIAL (AfriCOMET anchors for 3 of 42 QE languages): the rest comes from r4_run_all.py"}
    sv("status.json", st)
    print("\n".join(log))


if __name__ == "__main__":
    main()
