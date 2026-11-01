#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build calibration/shaping_flores_dev_v5.json = v4 + the added languages + the `lid` blocks + the `content` blocks (D-37, D-38). v4 is never modified.

    python scripts/build_calibration_v5.py --v4 calibration/shaping_flores_dev_v4.json --lid lid_measure.json \\
        --calibrated shaping_dyu.json shaping_sag.json --out calibration/shaping_flores_dev_v5.json

What is written
  * the 42 validated languages: entries unchanged, plus `lid` (measured, scripts/measure_lid.py) and `content` {mode, beta}.
      mode "blend" (beta 0.5) for the weak-QE languages: the owner's list kbp nus kam umb kmb mos bam, plus every language whose human-vs-mismatched AUC is < 0.75
      (kab fon, by the same rule); "qe" for all others.
  * 12 languages named on the AfriCOMET card but not validated (QE-BLENDED, low advantage weight, beta 0.5):
      dyu sag     own FLORES+ DEV calibration (997 sentences, `--calibrated`): AUC 0.683 (dyu, below the 0.70 bar -> `usable_by_auc: false`, forced usable) and 0.783
                  (sag: the 0.689 of the 60-sentence study was a small-sample estimate); the owner's weight 0.4 is kept for both (never above the measured same-source value)
      swc         Congo Swahili = Swahili (card: swa): the anchors of swh, no own calibration, weight 0.3
      loz ven iso tiv kqn wal lun kua kon   no reference set (ven: NTREX test only, never used): pooled-prior anchors = the MEDIAN of the 41 calibrated languages'
                  (t_fail, t_mid, t_great), no length / lexical / diacritic penalty, weight 0.3
  * 23 languages outside AfriCOMET: mode "consensus" (no QE, no anchors), advantage weight 0.5.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import statistics
import sys

from gemma_mt.languages import CONSENSUS_ONLY, LANGUAGES, QE_VALIDATED, SUPPORTED_RL

OWNER_WEAK = ("kbp", "nus", "kam", "umb", "kmb", "mos", "bam")
AUC_BLEND = 0.75
BETA = 0.5
W_UNTESTED, W_WEAK_MEASURED, W_CONSENSUS = 0.3, 0.4, 0.5
LISTED_UNTESTED = ("loz", "ven", "iso", "tiv", "kqn", "wal", "lun", "kua", "kon")
LISTED_MEASURED = ("dyu", "sag")


def lid_block(m: dict) -> dict:
    return {"enabled": bool(m["enabled"]), "labels": m["labels"], "min_chars": m["min_chars"], "recall": round(m["recall"], 4), "recall_strict": round(m["recall_strict"], 4),
            "extra_labels": m["extra_labels"], "source": m["source"], "n": m["n"], "short_recall": {k: round(v, 3) for k, v in m["short_recall"].items()},
            "cross_accept": round(m["cross_accept"], 4),
            **({"recall_sft_targets": round(m["recall_sft_targets"], 4)} if "recall_sft_targets" in m else {}),
            **({} if m["enabled"] else {"reason": "measured human recall below 0.90 (or too few sentences): the guard is OFF for this language"})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v4", required=True)
    ap.add_argument("--lid", required=True)
    ap.add_argument("--calibrated", nargs="*", default=[], help="calibrate_shaping outputs for dyu / sag")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    v4 = json.load(open(a.v4, encoding="utf-8"))
    lid = json.load(open(a.lid, encoding="utf-8"))
    out = copy.deepcopy(v4)
    usable41 = [k for k, v in v4.items() if not k.startswith("_") and isinstance(v, dict) and v.get("usable") and v.get("anchors")]
    pooled = [round(statistics.median(v4[k]["anchors"][i] for k in usable41), 4) for i in range(3)]
    measured = {}
    for p in a.calibrated:
        for k, v in json.load(open(p, encoding="utf-8")).items():
            if not k.startswith("_"):
                measured[k] = v

    # ---- the 42 validated languages
    blend42 = []
    for k in QE_VALIDATED:
        e = out[k]
        auc = e.get("auc_vs_mismatched")
        mode = "blend" if (k in OWNER_WEAK or (auc is not None and auc < AUC_BLEND)) else "qe"
        if k == "efi":
            mode = "qe"
        e["content"] = {"mode": mode}
        if mode == "blend":
            e["content"].update({"beta": BETA, "why": "weak QE: " + ("owner's list" if k in OWNER_WEAK else f"AUC {auc:.3f} < {AUC_BLEND}") + " (D-38)"})
            blend42.append(k)

    # ---- 12 listed languages
    def listed(k, anchors, source, weight, extra=None):
        e = {"usable": True, "n": 0, "anchors": anchors, "fail_floor": 0.0, "anchors_source": source, "reliability": weight, "reliability_same_source": weight,
             "content": {"mode": "blend", "beta": BETA, "why": "named on the AfriCOMET card, not validated (D-38)"},
             "weight_reason": "owner 2026-10-19: low advantage weight for QE-blended languages without a validated QE"}
        e.update(extra or {})
        return e

    for k in LISTED_MEASURED:
        m = copy.deepcopy(measured[k])
        auc = m["auc_vs_mismatched"]
        measured_w = m.get("reliability_same_source") or 0.0
        w = W_WEAK_MEASURED if not m.get("usable") else min(W_WEAK_MEASURED, measured_w)
        m.update({"usable_by_auc": bool(m.get("usable")), "usable": True, "reliability": w, "reliability_same_source": w, "measured_reliability_same_source": measured_w,
                  "content": {"mode": "blend", "beta": BETA, "why": f"named on the card; AUC {auc:.3f} on FLORES+ dev ({'above' if m.get('usable') else 'BELOW'} the 0.70 usability bar): blended with consensus, weight {w}"},
                  "weight_reason": "owner 2026-10-19: 0.4 for the weak dyu / sag (0.3 for the untested); capped by the measured value", "anchors_source": "own FLORES+ dev calibration (997 sentences)"})
        out[k] = m
    sw = v4["swh"]
    out["swc"] = listed("swc", list(sw["anchors"]), "borrowed from swh (Congo Swahili is Swahili; the card names swa, validated AUC %.3f)" % sw["auc_vs_mismatched"], W_UNTESTED)
    for k in LISTED_UNTESTED:
        out[k] = listed(k, pooled, "pooled prior: median of the 41 calibrated languages (t_fail, t_mid, t_great)", W_UNTESTED)

    # ---- 23 consensus-only languages
    for k in CONSENSUS_ONLY:
        out[k] = {"usable": True, "n": 0, "reliability": W_CONSENSUS, "reliability_same_source": W_CONSENSUS,
                  "content": {"mode": "consensus", "beta": 1.0, "why": "not named on the AfriCOMET card: no QE call, consensus (+ rubric) only (D-38)"},
                  "weight_reason": "owner 2026-10-19: 0.5 for the rubric + consensus only languages"}

    # ---- lid blocks (every language that was measured)
    missing = []
    for k in SUPPORTED_RL:
        if k in lid and k in out:
            out[k]["lid"] = lid_block(lid[k])
        else:
            missing.append(k)
    if missing:
        print("no lid measurement for:", missing, file=sys.stderr)

    en = sorted(k for k in SUPPORTED_RL if out.get(k, {}).get("lid", {}).get("enabled"))
    off = sorted(k for k in SUPPORTED_RL if k not in en)
    out["_meta_v5"] = {
        "created": datetime.date.today().isoformat(), "base": "shaping_flores_dev_v4.json (unchanged)", "decisions": "D-37 (GlotLID), D-38 (consensus, added languages)",
        "pooled_prior_anchors": pooled, "pooled_prior_from": f"median of {len(usable41)} usable v4 languages",
        "content_modes": {m: sorted(k for k in SUPPORTED_RL if out[k].get("content", {}).get("mode") == m) for m in ("qe", "blend", "consensus")},
        "advantage_weights_added": {"untested_listed": W_UNTESTED, "measured_weak_listed": W_WEAK_MEASURED, "consensus_only": W_CONSENSUS},
        "lid": {"model": lid["_meta"]["model"], "target_recall": lid["_meta"]["target_recall"], "enabled": en, "disabled": off,
                "note": "recall measured on FLORES+ dev references (dev only) where a dev set exists, else on a random sample of the SFT training targets (proxy)"},
        "n_languages": len(SUPPORTED_RL)}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {a.out}: {len(SUPPORTED_RL)} languages; blend among the 42: {blend42}; LID on {len(en)}, off {off}; pooled anchors {pooled}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
