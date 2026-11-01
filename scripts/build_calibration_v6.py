#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build calibration/shaping_flores_dev_v6.json = v5 + the new consensus-only languages (owner 2026-11-01). v5 is never modified.

    python scripts/build_calibration_v6.py --v5 calibration/shaping_flores_dev_v5.json --lid calibration/lid_blocks_new8_2026-11-01.json --out calibration/shaping_flores_dev_v6.json

Each added language (the keys of --lid, which are the languages of languages.CONSENSUS_ONLY that v5 does not know) gets exactly what the 23 consensus-only languages of v5 get (D-38): mode "consensus"
(beta 1.0, no QE call, no anchors, so the global `min_len_ratio_uncalibrated` bound applies), advantage weight 0.5, plus a `lid` block (GlotLID labels, minimum length and recall measured on
the shard-6 training targets, scripts via data_pipeline/sft_clean/fresh/lid_measure.py; `weak: true` when the label set cannot separate the language from a close one: arz, toi).
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import sys

from gemma_mt.languages import CONSENSUS_ONLY, SUPPORTED_RL

W_CONSENSUS = 0.5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v5", required=True); ap.add_argument("--lid", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    v5 = json.load(open(a.v5, encoding="utf-8")); lid = json.load(open(a.lid, encoding="utf-8"))
    out = copy.deepcopy(v5); added = []
    for k, m in lid.items():
        assert k in CONSENSUS_ONLY and k in SUPPORTED_RL, f"{k} is not a consensus-only RL language of languages.py"
        assert k not in v5, f"{k} is already in v5"
        weak = bool(m.get("guard_note")) and k in ("arz", "toi")
        block = {"enabled": bool(m["enabled"]), "labels": m["labels"], "min_chars": m["min_chars"], "recall": m["recall"], "recall_strict": m["recall_strict"], "extra_labels": m["extra_labels"],
                 "source": m["source"], "n": m["n"], "short_recall": m["short_recall"], "cross_accept": m["cross_accept"]}
        if weak:
            block["weak"] = True; block["weak_note"] = m["guard_note"]
        if not m["enabled"]:
            block["reason"] = "measured recall below 0.90: the guard is OFF for this language"
        out[k] = {"usable": True, "n": 0, "reliability": W_CONSENSUS, "reliability_same_source": W_CONSENSUS,
                  "content": {"mode": "consensus", "beta": 1.0, "why": "new language (2026-11-01): not on the AfriCOMET card: no QE call, consensus (+ rubric) only (D-38)"},
                  "weight_reason": "owner 2026-10-19: 0.5 for the rubric + consensus only languages (applied to the 2026-11-01 additions)", "lid": block}
        added.append(k)
    out["_meta_v6"] = {"created": datetime.date.today().isoformat(), "base": "shaping_flores_dev_v5.json (unchanged)", "added": sorted(added),
                       "n_languages": len([k for k in out if not k.startswith("_")]), "lid_source": "GlotLID v3 on 800 training targets per language of stage-B shard 6 (a proxy; not human references)"}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {a.out}: {out['_meta_v6']['n_languages']} languages, added {sorted(added)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
