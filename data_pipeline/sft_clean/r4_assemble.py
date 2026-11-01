#!/usr/bin/env python3
"""r4: assemble calibration/clean_2026-10-28/shaping_clean_v1.json (same schema as shaping_flores_dev_v8.json) from the clean refits. Safe to re-run when more QE scores exist.

    python r4_assemble.py --code <repo gemma_mt/> --old shaping_flores_dev_v8.json --work /workspace/calib_r4 --out clean/shaping_clean_v1.json --status clean/status.json

Inputs (all under --work): clean/noqe_blocks.json (r4_noqe.py), clean/lid_measure_clean.json (measure_lid.py), qe_work/<model>/<iso>.<set>.json + COPY.json (r4_qe_score.py),
The OLD v8 file is used only as the TEMPLATE of keys that are not refitted (weight_reason, content.why, listed /
consensus-only languages, owner-set weights); every fitted number of a language with clean references is replaced, and fitted numbers of languages whose clean references do not
exist (pcm) are REMOVED and flagged. Rules are the unchanged rules of calibrate_shaping.py / build_calibration_v5.py / calibrate_qe_ensemble.py (min AUC 0.7, reliability linear between
AUC 0.65 and 0.90, reliability_same_source = ((mismatch + half-omitted accuracy)/2 - 0.5)/0.4, blend below AUC 0.75 or on the owner's weak list, t_fail = P90 of the language's mismatched QE with fail_floor 0.0).
A language whose half/append scores are missing keeps `reliability_same_source_provisional: true` (value from the mismatch accuracy alone). qe2 / ens blocks are written only when the SSA-COMET-MTL scores exist.
Numbers only; no text.
"""
import argparse, copy, datetime, json, os, statistics, sys

OWNER_WEAK = ("kbp", "nus", "kam", "umb", "kmb", "mos", "bam")
LISTED_MEASURED = ("dyu", "sag")
SETS = ("human", "mismatch", "half", "append")


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def load_scores(work, model, iso):
    out = {}
    for s in SETS:
        d = jl(os.path.join(work, "qe_work", model, f"{iso}.{s}.json"))
        if d:
            out[s] = dict(zip(d["ids"], d["scores"]))
            out["_ids"] = d["ids"]
    cp = jl(os.path.join(work, "qe_work", model, "COPY.json"))
    if cp and "_ids" in out:
        m = dict(zip(cp["ids"], cp["scores"]))
        out["copy"] = {i: m[i] for i in out["_ids"] if i in m}
    return out


def lang_qe_stats(sc, RS, fail_floor=0.0, min_auc=0.7):
    """AfriCOMET-style statistics from the available sets (phase A: human + mismatch + copy; phase B adds half + append)."""
    if "human" not in sc or "mismatch" not in sc:
        return None
    ids = sc["_ids"]
    human = [sc["human"][i] for i in ids]
    mism = [sc["mismatch"][i] for i in ids]
    acc = lambda bad: float(sum(h > b for h, b in zip(human, bad)) / len(human))
    e = {"n": len(human), "anchors": list(RS.pair_anchors(human, mism, fail_floor)), "fail_floor": fail_floor,
         "human_quantiles_p10_p50_p90": list(RS.compute_anchors(human)), "mismatched_p90": float(sorted(mism)[int(0.9 * (len(mism) - 1))]),
         "auc_vs_mismatched": RS.auc(human, mism), "mean_human": sum(human) / len(human), "mean_mismatched": sum(mism) / len(mism),
         "share_human_below_0.46": sum(h < 0.46 for h in human) / len(human)}
    ss = {"mismatch": acc(mism)}
    if "copy" in sc and len(sc["copy"]) == len(ids):
        cp = [sc["copy"][i] for i in ids]
        ss["copy"] = acc(cp); e["auc_vs_copy"] = RS.auc(human, cp); e["mean_copy"] = sum(cp) / len(cp)
    if "half" in sc:
        hf = [sc["half"][i] for i in ids]; ss["half_omitted"] = acc(hf); e["auc_vs_half_omitted"] = RS.auc(human, hf); e["mean_half_omitted"] = sum(hf) / len(hf)
    if "append" in sc:
        ap_ = [sc["append"][i] for i in ids]; ss["append_unrelated"] = acc(ap_); e["auc_vs_append"] = RS.auc(human, ap_); e["mean_append"] = sum(ap_) / len(ap_)
    e["same_source_accuracy"] = ss
    e["usable"] = bool(e["auc_vs_mismatched"] >= min_auc)
    e["reliability"] = RS.reliability_from_auc(e["auc_vs_mismatched"]) if e["usable"] else 0.0
    if "half_omitted" in ss:
        hard = (ss["mismatch"] + ss["half_omitted"]) / 2
        e["reliability_same_source"] = float(min(1.0, max(0.0, (hard - 0.5) / 0.4))) if e["usable"] else 0.0
    else:
        e["reliability_same_source"] = float(min(1.0, max(0.0, (ss["mismatch"] - 0.5) / 0.4))) if e["usable"] else 0.0
        e["reliability_same_source_provisional"] = True
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True); ap.add_argument("--old", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--status", required=True); ap.add_argument("--split", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.code); sys.path.insert(0, os.path.join(a.code, "scripts"))
    from gemma_mt.rl import reward_shaping as RS
    from gemma_mt.languages import SUPPORTED_RL, QE_VALIDATED, CONSENSUS_ONLY
    from build_calibration_v5 import lid_block
    old = json.load(open(a.old)); noqe = jl(os.path.join(a.work, "clean/noqe_blocks.json")); lid = jl(os.path.join(a.work, "clean/lid_measure_clean.json"))
    out = {k: copy.deepcopy(v) for k, v in old.items() if k.startswith("_") is False}
    status = {}
    cal_langs = [i for i in QE_VALIDATED if i in noqe and i not in ("efi", "pcm")]
    # ---- QE languages with clean references
    fitted = {}
    for iso in cal_langs + list(LISTED_MEASURED):
        if iso in fitted or iso not in noqe:
            continue
        sc = load_scores(a.work, "africomet", iso)
        st = lang_qe_stats(sc, RS)
        fitted[iso] = st
    pooled_src = [i for i, st in fitted.items() if st and st["usable"] and i in QE_VALIDATED]
    pooled = [float(statistics.median(fitted[i]["anchors"][k] for i in pooled_src)) for k in range(3)] if pooled_src else None
    for iso in SUPPORTED_RL:
        e = out.get(iso)
        if e is None:
            continue
        s = {"qe_phase": None, "noqe": False, "lid": False}
        nq = noqe.get(iso)
        # strip every fitted number of the old calibration that no clean reference supports (they were fitted on the old FLORES+ dev 997 / MAFAND dev / IbomMT test)
        has_clean_refs = nq is not None and iso not in ("efi",)
        old_mode = (e.get("content") or {}).get("mode", "qe")
        if iso in cal_langs or iso in LISTED_MEASURED:
            st = fitted.get(iso)
            for k in ("qe2", "ens", "qe2_diagnostic", "same_source_accuracy", "lexical_same_source_accuracy", "model", "anchors_at_floor_0.46", "auc_vs_copy", "auc_vs_half_omitted", "mean_copy",
                      "mean_half_omitted", "auc_vs_append", "mean_append"):
                e.pop(k, None)
            for k, v in (nq or {}).items():
                if k in ("len_ratio", "diacritics", "lexical", "omission", "word_ratio"):
                    e[k] = v
                elif k == "n":
                    e["n_noqe"] = v
            s["noqe"] = True
            if st:
                e.update(st)
                s["qe_phase"] = "A+B" if "half_omitted" in st["same_source_accuracy"] else "A"
                if iso in OWNER_WEAK or st["auc_vs_mismatched"] < 0.75:
                    e["content"] = {"mode": "blend", "beta": 0.5, "why": "weak QE: " + ("owner's list" if iso in OWNER_WEAK else f"AUC {st['auc_vs_mismatched']:.3f} < 0.75") + " (D-38), refit on H_cal v2"}
                elif iso not in LISTED_MEASURED:
                    e["content"] = {"mode": "qe"}
                if iso in LISTED_MEASURED:                       # D-38: forced usable, owner weight 0.4 capped by the measured value
                    m = st.get("reliability_same_source") or 0.0
                    w = 0.4 if not st["usable"] else min(0.4, m)
                    e.update({"usable_by_auc": st["usable"], "usable": True, "measured_reliability_same_source": m, "reliability": w, "reliability_same_source": w,
                              "content": {"mode": "blend", "beta": 0.5, "why": f"named on the card; AUC {st['auc_vs_mismatched']:.3f} on H_cal v2: blended with consensus, weight {w}"}})
                e["calibration_source"] = "H_cal v2 (FLORES+ dev, 261 sentences), AfriCOMET-QE-STL-1.1 on CPU fp32"
            else:
                for k in ("anchors", "human_quantiles_p10_p50_p90", "mismatched_p90", "auc_vs_mismatched", "mean_human", "mean_mismatched", "share_human_below_0.46"):
                    e.pop(k, None)
                e["calibration_pending"] = "QE scores of H_cal not computed yet (python r4_qe_score.py ...); the v8 numbers were removed (old FLORES+ dev 997)"
                e["usable"] = False
                e["reliability"] = e["reliability_same_source"] = 0.0
        elif iso == "pcm":
            for k in ("anchors", "anchors_at_floor_0.46", "human_quantiles_p10_p50_p90", "mismatched_p90", "auc_vs_mismatched", "auc_vs_copy", "auc_vs_half_omitted", "mean_human", "mean_mismatched",
                      "mean_copy", "mean_half_omitted", "share_human_below_0.46", "same_source_accuracy", "lexical", "lexical_same_source_accuracy", "len_ratio", "diacritics", "omission", "qe2", "ens",
                      "word_ratio"):
                e.pop(k, None)
            e["needs_decision"] = "pcm had no FLORES+ dev file: v8 was fitted on MAFAND dev (now protected). Pooled prior until the owner decides on a clean human source."
            if pooled:
                e["anchors"] = pooled; e["anchors_source"] = "pooled prior of the clean calibration"
            e["usable"] = True; e["reliability"] = e["reliability_same_source"] = 0.3
            e["content"] = {"mode": "blend", "beta": 0.5, "why": "no clean human reference set for Pidgin: pooled-prior anchors, low weight, pending owner decision"}
        elif iso in CONSENSUS_ONLY and nq is not None:       # kea seh vmw: only word_ratio is kept by the D-38 entry shape, plus lid below
            if nq.get("word_ratio"):
                e["word_ratio"] = nq["word_ratio"]
            s["noqe"] = True
        elif iso not in CONSENSUS_ONLY and iso not in ("efi",) and pooled and e.get("anchors") and "pooled prior" in str(e.get("anchors_source", "")):
            e["anchors"] = pooled; e["anchors_source"] = f"pooled prior: median of the {len(pooled_src)} clean-calibrated usable languages (t_fail, t_mid, t_great)"
            e.pop("qe2", None); e.pop("ens", None)
        elif iso == "swc" and fitted.get("swh"):
            e["anchors"] = list(fitted["swh"]["anchors"]); e.pop("qe2", None); e.pop("ens", None)
        # lid
        if lid and iso not in lid and iso not in ("efi", "ibb"):
            e["lid"] = {"enabled": False, "labels": list(e.get("lid", {}).get("labels", [])), "reason": "no clean measurement (old v8 value came from the old SFT proxy / dev sets): OFF"}
        if lid and iso in lid:
            e["lid"] = lid_block(lid[iso]); s["lid"] = True
            if iso == "pcm":                                  # the clean proxy is machine-translated SFT text (recall 0.92 incl. 4.5% English); v8 measured human MAFAND text (0.809, OFF): keep OFF until the owner decides
                e["lid"]["enabled"] = False; e["lid"]["reason"] = "kept OFF: clean proxy text is not human Pidgin (see report); v8 human measurement 0.809"
        out[iso] = e
        status[iso] = s
    # ---- efi / ibb: IbomMT TRAIN calibration half (CC-BY-NC-4.0): EXPLICIT OWNER EXCEPTION (2026-10-29), every piece is tagged nc_exception: ibom-mt so it can be disclosed or dropped.
    #      Consensus-shape entry (D-43 rule), no QE call, weight provisional until validate_consensus.py on the new policy (hold-out half).
    efi = jl(os.path.join(a.work, "clean/efi_ibb_blocks.json"))
    for iso in ("efi", "ibb"):
        e = out.get(iso, {})
        for k in list(e):
            if k in ("anchors", "anchors_at_floor_0.46", "human_quantiles_p10_p50_p90", "mismatched_p90", "auc_vs_mismatched", "auc_vs_copy", "auc_vs_half_omitted", "mean_human", "mean_mismatched",
                     "mean_copy", "mean_half_omitted", "share_human_below_0.46", "same_source_accuracy", "lexical_same_source_accuracy", "qe2", "ens", "calibration_source", "len_ratio", "diacritics",
                     "lexical", "omission", "word_ratio"):
                e.pop(k)
        e.update({"usable": True, "n": 0, "reliability": 0.5, "reliability_same_source": 0.5, "weight_provisional": True, "nc_exception": "ibom-mt",
                  "content": {"mode": "consensus", "beta": 1.0, "why": "QE-free consensus (D-43 on clean data): human-reference statistics from the calibration half of IbomMT TRAIN (nc_exception: ibom-mt)"},
                  "weight_reason": "D-43 rule needs within-prompt Spearman on the new policy (WAITS-FOR-MODEL); provisional 0.5", "calibration_source": "none"})
        if efi and iso in efi:
            st = efi[iso]["stats"]
            for k in ("len_ratio", "diacritics", "lexical", "omission"):
                if st.get(k):
                    e[k] = st[k]
            om = e.get("omission", {})
            if "false_alarm_by_cost" in om:                                    # owner 2026-10-29: omission cost 4.0 everywhere
                fa = om["false_alarm_by_cost"]["4.0"]
                om.update({"cost": 4.0, "false_alarm": fa, "met_target": bool(fa <= om.get("target", 0.02)), "note": "cost fixed at 4.0 (owner)"})
            e["n"] = st["n"]
            e["calibration_source"] = {"set": "howard-nlp/ibom-mt TRAIN (CC-BY-NC-4.0), owner exception 2026-10-29", "nc_exception": "ibom-mt", "split": "calibration half, sha1(calib-split-v1+English)<0.5",
                                       "n": st["n"], "holdout_n": efi[iso]["n_holdout"], "calibration_half_sha256": efi[iso]["calibration_half_sha256"]}
        if lid and iso in lid:
            e["lid"] = dict(lid_block(lid[iso]), nc_exception="ibom-mt")
        else:
            e["lid"] = {"enabled": False, "labels": [f"{iso}_Latn"], "reason": "no measurement"}
        out[iso] = e
        status[iso] = {"efi_ibb": "ibom_mt_exception"}
    # ---- qe2 (SSA-COMET-MTL) and ens (ensemble) blocks, only where the SSA scores exist (calibrate_qe_ensemble.compose semantics, delta fixed at D-55's 1.0; re-check on the new policy)
    import calibrate_qe_ensemble as QE
    from gemma_mt.rl.reward_shaping import ContentShaper
    DELTA = 1.0
    ssa = {}
    for iso in cal_langs + list(LISTED_MEASURED):
        st2 = lang_qe_stats(load_scores(a.work, "ssa_mtl", iso), RS)
        if st2:
            ssa[iso] = st2
    ssa_pool_src = [i for i in ssa if ssa[i]["usable"] and i in QE_VALIDATED]
    pooled2 = [float(statistics.median(ssa[i]["anchors"][k] for i in ssa_pool_src)) for k in range(3)] if ssa_pool_src else None
    ens_done = []
    if ssa:
        tmp = a.out + ".tmp_noens.json"
        json.dump(out, open(tmp, "w"), ensure_ascii=False)
        sh1 = ContentShaper.from_file(tmp, fail_floor=0.0)
        sh2 = ContentShaper(calibration={l: tuple(ssa[l]["anchors"]) for l in ssa}, default_anchors=tuple(pooled2), fail_floor=0.0)
        for iso, st2 in ssa.items():
            e = out[iso]
            qe2 = dict(st2); qe2.update({"model": "McGill-NLP/ssa-comet-mtl", "anchors_source": "own H_cal v2 calibration"})
            e["qe2"] = qe2
            if st2["usable"] and e.get("usable_by_auc", e.get("usable")) and e.get("anchors") and "half_omitted" in e.get("same_source_accuracy", {}) and "half_omitted" in st2["same_source_accuracy"]:
                s1, s2 = load_scores(a.work, "africomet", iso), load_scores(a.work, "ssa_mtl", iso)
                ids = s1["_ids"]
                c1 = {k: [s1[k][i] for i in ids] for k in ("human", "mismatch", "copy", "half", "append")}
                c2 = {k: [s2[k][i] for i in ids] for k in ("human", "mismatch", "copy", "half", "append")}
                comb, share = QE.combined_sets(c1, c2, sh1, sh2, iso, DELTA)
                ev = QE.lang_stats(comb, fail_floor=-9.0)
                hard = (ev["same_source_accuracy"]["mismatch"] + ev["same_source_accuracy"]["half_omitted"]) / 2
                m_rel = RS.reliability_from_auc(ev["auc_vs_mismatched"]); m_rss = float(min(1.0, max(0.0, (hard - 0.5) / 0.4)))
                r7, rss7 = float(e.get("reliability", 1.0)), float(e.get("reliability_same_source", 1.0))
                owner = bool(e.get("weight_reason")) and "owner" in str(e.get("weight_reason"))
                b_rel, b_rss = (r7, rss7) if owner else (max(r7, m_rel), max(rss7, m_rss))
                e["ens"] = {"delta": DELTA, "auc_vs_mismatched": ev["auc_vs_mismatched"], "same_source_accuracy": ev["same_source_accuracy"], "reliability_measured": m_rel,
                            "reliability_same_source_measured": m_rss, "reliability": b_rel, "reliability_same_source": b_rss,
                            "reliability_rule": "owner-set weight kept" if owner else "max(AfriCOMET, measured ensemble)", "min_branch_share": share, "mean_human": sum(comb["human"]) / len(comb["human"])}
                ens_done.append(iso)
        for iso in SUPPORTED_RL:                              # no clean reference: pooled / borrowed qe2 prior as D-55 (swc borrows swh)
            e = out.get(iso)
            if e is None or iso in ssa or (e.get("content") or {}).get("mode") == "consensus" or not e.get("anchors") or iso in ("efi", "ibb"):
                continue
            if iso == "swc" and "swh" in ssa:
                e["qe2"] = {"model": "McGill-NLP/ssa-comet-mtl", "anchors": list(ssa["swh"]["anchors"]), "anchors_source": "borrowed from swh", "usable": True,
                            "reliability": e.get("reliability", 0.3), "reliability_same_source": e.get("reliability_same_source", 0.3), "n": 0}
            elif pooled2:
                e["qe2"] = {"model": "McGill-NLP/ssa-comet-mtl", "anchors": pooled2, "anchors_source": "pooled_prior", "usable": True,
                            "reliability": e.get("reliability", 0.3), "reliability_same_source": e.get("reliability_same_source", 0.3), "n": 0}
        os.remove(tmp)
    # ---- meta
    out["_word_ratio_pooled"] = noqe["_word_ratio_pooled"]
    sp = json.load(open(a.split))
    out["_meta"] = {"fail_floor": 0.0, "min_auc": 0.7, "qe_model": "masakhane/africomet-qe-stl-1.1", "split": "calibration (H_cal v2); never dev/devtest/test outside H_cal/H_sel", "licence": "commercial use, except the explicit owner exception for IbomMT efi / ibb (tag nc_exception: ibom-mt); strict-licence SFT proxies for LID / diacritic widening",
                    "split_file": "heldout_split_cal_sel_v2.json", "h_cal_sha256": sp["h_cal_sha256"], "n_cal_ids": sp["n_cal"], "created": datetime.date.today().isoformat(),
                    "base_template": os.path.basename(a.old) + " (keys only; all fitted numbers of languages with clean references replaced)",
                    "pooled_prior_anchors": pooled, "pooled_prior_from": f"median of {len(pooled_src)} clean-calibrated usable QE-validated languages",
                    "qe_phases_present": {iso: s.get("qe_phase") for iso, s in status.items() if s.get("qe_phase")},
                    "qe2_languages": sorted(ssa), "ens_languages": sorted(ens_done), "ens_delta": DELTA, "pooled_prior_qe2_anchors": pooled2,
                    "incomplete": ["qe2/ens blocks (SSA-COMET-MTL scores) unless listed under qe_phases_present_ssa", "reliability_same_source_provisional languages need phase B (half, append)",
                                   "efi/ibb: IbomMT consensus entries (nc_exception), weight provisional 0.5", "pcm: pooled prior, LID off (owner 2026-10-29)"]}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    json.dump(status, open(a.status, "w"), indent=1)
    print("wrote", a.out, "QE phases:", collections_count([s.get("qe_phase") for s in status.values()]), "pooled", pooled)


def collections_count(xs):
    import collections
    return dict(collections.Counter(xs))


if __name__ == "__main__":
    main()
