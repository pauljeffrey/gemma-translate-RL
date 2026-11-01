"""AfriCOMET-QE sample audit (200 short pairs) of every en-<African language> folder of parallel-sentences-ccmatrix -> out/audit_ccm.json."""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from sft_new.ext_common import iter_files  # noqa: E402
from sft_new.ext_mich import CCM_REPO, ccm_folders  # noqa: E402
from sft_new.quality_check import scorer, summarize  # noqa: E402


def main():
    from rl.bin_data import AFRICOMET_CODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--comet-python", required=True)
    ap.add_argument("--pool", type=int, default=30_000)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    done = json.loads(a.out.read_text()) if a.out.exists() else {}
    rng = random.Random(0)
    score, proc = scorer(a.comet_python)
    try:
        for folder, code, paths in ccm_folders():
            if folder in done:
                continue
            if code not in AFRICOMET_CODE:
                done[folder] = {"code": code, "qe": "unavailable"}
                continue
            rows = []
            for r in itertools.islice(iter_files(CCM_REPO, paths[:1]), a.pool):
                e, t = (r.get("english") or "").strip(), (r.get("non_english") or "").strip()
                if 20 <= len(e) <= 400 and 20 <= len(t) <= 400:
                    rows.append((e, t))
            rng.shuffle(rows)
            sample = rows[: a.n]
            if len(sample) < 30:
                done[folder] = {"code": code, "qe": "too_few_pairs"}
                continue
            done[folder] = {"code": code, **summarize(score(sample))}
            a.out.write_text(json.dumps(done, indent=1))
            print(folder, code, done[folder], flush=True)
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)
        a.out.write_text(json.dumps(done, indent=1))


if __name__ == "__main__":
    main()
