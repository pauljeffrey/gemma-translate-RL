#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""calibration/clean_2026-10-28/shaping_clean_v2.json = shaping_clean_v1.json + the 8 new consensus-only languages (owner 2026-11-01; 385M RL).

    python scripts/build_calibration_clean_v2.py [--v1 ...] [--v6 ...] [--out ...]

Why not use shaping_flores_dev_v6.json directly: v6 is built on v5, whose anchors were fitted on references that the clean rerun forbids (NLLB-derived). The clean RL plan uses
shaping_clean_v1.json (refit on T2 only). So this script copies ONLY the blocks of the 8 new languages (guw lue tll zne lub crs mfe arz) from v6. Those blocks carry NO reference-derived
numbers: content mode `consensus` (beta 1.0), reliability 0.5, and a `lid` block measured on 800 training targets per language (GlotLID v3, reports/data/lid_precision_recall_fixed_2026-11-01.csv).
Every v1 entry is copied unchanged (checked: json.dumps equal before and after, the file is written with the same formatting as v1, indent=1, no trailing newline); the new keys are
inserted after the last language of v1 and before the `_...` keys; `_meta` gets one extra key `v2_additions`.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
NEW8 = ["guw", "lue", "tll", "zne", "lub", "crs", "mfe", "arz"]
FORBIDDEN = {"t_fail", "t_mid", "t_great", "anchors", "auc", "qe2", "refs", "omission_cost", "word_ratio", "length"}      # keys that would carry reference-derived numbers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default=os.path.join(ROOT, "calibration", "clean_2026-10-28", "shaping_clean_v1.json"))
    ap.add_argument("--v6", default=os.path.join(ROOT, "calibration", "shaping_flores_dev_v6.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "calibration", "clean_2026-10-28", "shaping_clean_v2.json"))
    a = ap.parse_args()
    raw1 = open(a.v1, encoding="utf-8").read(); v1 = json.loads(raw1); v6 = json.load(open(a.v6, encoding="utf-8"))
    assert json.dumps(v1, indent=1, ensure_ascii=False) == raw1, "v1 formatting is not reproducible: refuse to rewrite it"
    out = {}
    langs = [k for k in v1 if not k.startswith("_")]; tail = [k for k in v1 if k.startswith("_")]
    for k in langs: out[k] = v1[k]
    for iso in NEW8:
        assert iso not in v1, f"{iso} already in v1"
        b = v6[iso]
        assert b["content"]["mode"] == "consensus" and b["content"]["beta"] == 1.0 and b["reliability"] == 0.5, f"{iso}: not a consensus-only block"
        bad = FORBIDDEN & set(b); assert not bad, f"{iso}: reference-derived keys {bad}"
        assert b["lid"]["labels"] and b["lid"]["recall"] >= 0.9, f"{iso}: language-ID block not usable"
        out[iso] = b
    for k in tail:
        out[k] = dict(v1[k], v2_additions={"languages": NEW8, "source": "shaping_flores_dev_v6.json blocks of these languages only (consensus mode, no reference-derived numbers)", "created": "2026-11-01"}) if k == "_meta" else v1[k]
    for k in v1: assert json.dumps(out[k], sort_keys=True) == json.dumps(v1[k], sort_keys=True) or k == "_meta", f"v1 entry {k} changed"
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=1, ensure_ascii=False))
    chk = json.load(open(a.out, encoding="utf-8"))
    assert all(json.dumps(chk[k]) == json.dumps(v1[k]) for k in langs + [t for t in tail if t != "_meta"])
    print(f"wrote {a.out}: {len(langs)} v1 languages unchanged + {len(NEW8)} new = {len([k for k in chk if not k.startswith('_')])} languages")


if __name__ == "__main__":
    main()
