"""AfriCOMET-QE SAMPLE audit (about 200 pairs per repo, owner 2026-10-11: QE is expensive) of the michsethowusu OPUS-MT560 sets of the languages we carry.

    python -m sft_new.audit_mich --accounts accounts.json --out audit_mich.json --comet-python .../comet_python.sh

Output per repo: mean / median / P10 / share below the 0.46 failure line of a random sample of short pairs (English, language). `mich_mt560` keeps a set
only when the sample says it is worth it (mean >= 0.50 and share below 0.46 <= 0.30); a language AfriCOMET cannot judge (not in rl.bin_data.AFRICOMET_CODE) is
kept on the language-ID gate alone and marked `qe: unavailable`.
"""
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
from sft_new import hubio  # noqa: E402
from sft_new.ext_mich import mt560_repos  # noqa: E402
from sft_new.quality_check import scorer, summarize  # noqa: E402


def main():
    from rl.bin_data import AFRICOMET_CODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--comet-python", required=True)
    ap.add_argument("--pool", type=int, default=40_000)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    done = json.loads(a.out.read_text()) if a.out.exists() else {}
    rng = random.Random(0)
    score, proc = scorer(a.comet_python)
    try:
        for repo, code, ncol in mt560_repos(a.accounts):
            if repo in done:
                continue
            if code not in AFRICOMET_CODE:
                done[repo] = {"code": code, "qe": "unavailable"}
                continue
            rows = []
            try:
                for r in itertools.islice(hubio.iter_repo(repo), a.pool):
                    ncol = ncol or next((k for k in r if k not in ("eng", "__file") and isinstance(r[k], str)), None)
                    e, t = (r.get("eng") or "").strip(), (r.get(ncol) or "").strip()
                    if 20 <= len(e) <= 400 and 20 <= len(t) <= 400:
                        rows.append((e, t))
            except Exception as ex:
                done[repo] = {"code": code, "error": type(ex).__name__}
                continue
            rng.shuffle(rows)
            sample = rows[: a.n]
            if len(sample) < 30:
                done[repo] = {"code": code, "qe": "too_few_pairs"}
                continue
            sc = score(sample)
            done[repo] = {"code": code, **summarize(sc)}
            a.out.write_text(json.dumps(done, indent=1))
            print(repo, code, done[repo], flush=True)
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)
        a.out.write_text(json.dumps(done, indent=1))


if __name__ == "__main__":
    main()
