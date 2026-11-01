#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the WEAK-LANGUAGE monitor set of the RL run (D-56): FLORES+ dev for every language whose reward has no (consensus) or a weak (blend) AfriCOMET signal.

    python scripts/build_weak_dev.py --shaping calibration/shaping_flores_dev_v7.json --out /workspace/jobs/b2/bench_weak_flores_dev.jsonl [--cache-dir DIR]

Rows (the schema of every dev file of the trainer): english, reference, iso, benchmark = flores_plus_dev, id. Dev only (never devtest). `benchmark_aux` entry:

    - {name: weak_dev, path: .../bench_weak_flores_dev.jsonl, per_language: 60, max_new_tokens: 256, batch_size: 64}

`per_language: 60` takes the 60 sentences with the smallest hash of the English text (train_rl.load_benchmark_dev), the SAME sentences for every language, so the macro chrF++ over
the languages compares like with like. Logged as bench_aux/weak_dev/chrf_macro | bleu_macro | finished_rate | chrf/<iso> | finished/<iso> | seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchors_common as C  # noqa: E402
from gemma_mt.languages import SUPPORTED_RL  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shaping", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--modes", nargs="*", default=["blend", "consensus"])
    a = ap.parse_args(argv)
    shaping = json.load(open(a.shaping, encoding="utf-8"))
    mode = {k: (v.get("content") or {}).get("mode", "qe") for k, v in shaping.items() if isinstance(v, dict) and not k.startswith("_")}
    weak = [i for i in SUPPORTED_RL if mode.get(i) in a.modes]
    fl = C.flores_dev(weak, a.cache_dir)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for iso in weak:
            for u in fl.get(iso, []):
                f.write(json.dumps({"english": u["src"], "reference": u["ref"], "iso": iso, "benchmark": "flores_plus_dev", "id": u["id"]}, ensure_ascii=False) + "\n")
                n += 1
    have = [i for i in weak if fl.get(i)]
    print(f"[weak-dev] {len(weak)} weak languages, {len(have)} with FLORES+ dev ({n} rows) -> {a.out}; without: {[i for i in weak if i not in have]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
