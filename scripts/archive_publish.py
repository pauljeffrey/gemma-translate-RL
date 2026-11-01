# SPDX-License-Identifier: AGPL-3.0-or-later
"""Merge the RL best-translation archives, apply a quality gate and (only if it passes) push them to a PRIVATE Hub dataset.

Owner 2026-10-20: "push the archives to <HF_ORG> as a private dataset only if they are good. do this at the end of all training."

Steps
  1. read every archive given with --archive name=path (rl2, rl2b, rl2c ...), all columns, plus a `run` column;
  2. drop rows that are suspect by construction: --drop-after name=STEP removes that run's rows whose best step is later than STEP
     (phase 2b diverged from step ~489: its later rows are not trusted);
  3. row filters (deterministic, no model): translation is not a copy of the source, not a repetition loop, plausible length ratio;
  4. the same (language, source) in several runs keeps the highest reward;
  5. licence column: joined from the RL prompt files by the archive key (sha1(iso + NFC(source))[:16]);
  6. dataset gate: enough rows and languages, most rows survive the row filters, mean reward high enough. A failed gate pushes nothing and writes the reasons.

Outputs (always): <out>/archive_merged.parquet + .jsonl, <out>/PUBLISH_REPORT.md, <out>/PUBLISH_STATUS (pushed | gate_failed | no_token | dry_run).
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from gemma_mt.rl.archive import make_key, read_rows


def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def row_problem(r: Dict[str, Any]) -> Optional[str]:
    src, tr = unicodedata.normalize("NFC", r["source"]), unicodedata.normalize("NFC", r["translation"])
    if not tr.strip():
        return "empty"
    s, t = src[:400], tr[:400]
    if t.strip().lower() == s.strip().lower() or difflib.SequenceMatcher(None, s.lower(), t.lower(), autojunk=False).ratio() > 0.85:
        return "copy_of_source"
    toks = _tokens(tr)
    if len(toks) >= 20 and len(set(toks)) / len(toks) < 0.3:
        return "repetition"
    grams = Counter(zip(toks, toks[1:], toks[2:]))
    if grams and grams.most_common(1)[0][1] >= 5:
        return "repetition"
    ratio = len(tr) / max(len(src), 1)
    if ratio < 0.3 or ratio > 4.0:
        return "length_ratio"
    return None


def licences(keys: set, isos: set, prompts_dir: str) -> Dict[str, Tuple[str, str]]:
    import pyarrow.parquet as pq

    out: Dict[str, Tuple[str, str]] = {}
    for iso in sorted(isos):
        p = os.path.join(prompts_dir, f"{iso}.parquet")
        if not os.path.exists(p):
            continue
        t = pq.read_table(p, columns=["english", "iso", "source_dataset", "licence"]).to_pydict()
        for e, i, sd, lic in zip(t["english"], t["iso"], t["source_dataset"], t["licence"]):
            k = make_key(i, e)
            if k in keys and k not in out:
                out[k] = (sd or "", lic or "")
    return out


def card(n_rows: int, n_langs: int, runs: Dict[str, int], lic: Counter, thresholds: Dict[str, Any], repo: str) -> str:
    lic_lines = "\n".join(f"- `{k or 'unknown'}`: {v:,}" for k, v in lic.most_common())
    run_lines = "\n".join(f"- `{k}`: {v:,} rows" for k, v in runs.items())
    return f"""---
pretty_name: Gemma MT RL best translations
license: other
language: [en]
tags: [machine-translation, african-languages, reinforcement-learning, africomet]
---
# {repo}

**Private.** The best English -> African-language translation found for each source sentence during the GRPO reinforcement learning of
`gemma-3-270m-translate` ({n_rows:,} rows, {n_langs} target languages). Every translation passed a reward gate and the filters below; each row carries the score(s) it was
selected with.

## Columns
`key` (sha1 of iso + NFC(source), first 16 hex digits), `iso`, `source` (English), `translation`, `mode` (how it was scored: `qe` = AfriCOMET-QE per chunk,
`blend` = QE + group consensus, `consensus` = group chrF++ only), `reward` (total RL reward), `qe` (AfriCOMET-QE), `consensus_chrf` (chrF++ against the other samples of its group),
`diacritic_deficit`, `lid_ok`, `lid_label` (GlotLID result), `n_chunks`, `step_first`, `step_best`, `length_bin`, `stage`, `times_seen`, `n_improved`, `run` (rl2 / rl2b / rl2c),
`source_dataset`, `licence` (of the English source sentence).

## Selection
Entry thresholds: {json.dumps(thresholds, ensure_ascii=False)}. Rows of a run after the point where it diverged are dropped; copies of the source, repetition loops and implausible length ratios are removed;
the highest reward is kept when several runs found a translation for the same source.

## Rows by run
{run_lines}

## Source licences
{lic_lines}

The translations are model outputs scored by learned metrics, not human references: spot-check before using them as training targets. Respect the licence of each English source
sentence (several are non-commercial or share-alike), which is why the dataset is private.
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", action="append", required=True, help="name=path/to/archive.sqlite (repeatable)")
    ap.add_argument("--drop-after", action="append", default=[], help="name=STEP: drop that run's rows whose step_best is greater than STEP")
    ap.add_argument("--prompts-dir", default="/workspace/sft_build/gemma/rl_prompts_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=(os.environ['HF_ORG'] + "/gemma-mt-rl-best-translations"))
    ap.add_argument("--min-rows", type=int, default=3000)
    ap.add_argument("--min-languages", type=int, default=40)
    ap.add_argument("--min-keep-rate", type=float, default=0.90, help="share of (non-suspect) rows that must survive the row filters")
    ap.add_argument("--min-mean-reward", type=float, default=1.0)
    ap.add_argument("--max-language-share", type=float, default=0.25, help="no language may exceed this share of the rows")
    ap.add_argument("--token-file", default="/workspace/sft_build/.hf_write_token")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    drop_after = {x.split("=")[0]: int(x.split("=")[1]) for x in a.drop_after}

    rows: List[Dict[str, Any]] = []
    per_run_in: Dict[str, int] = {}
    suspect = 0
    for spec in a.archive:
        name, path = spec.split("=", 1)
        rs = read_rows(path)
        per_run_in[name] = len(rs)
        for r in rs:
            r["run"] = name
            if name in drop_after and r["step_best"] > drop_after[name]:
                suspect += 1
                continue
            rows.append(r)
    problems: Counter = Counter()
    good: List[Dict[str, Any]] = []
    for r in rows:
        p = row_problem(r)
        if p:
            problems[p] += 1
        else:
            good.append(r)
    keep_rate = len(good) / max(len(rows), 1)
    best: Dict[str, Dict[str, Any]] = {}
    for r in good:
        if r["key"] not in best or r["reward"] > best[r["key"]]["reward"]:
            best[r["key"]] = r
    final = sorted(best.values(), key=lambda r: (r["iso"], r["key"]))
    lic = licences({r["key"] for r in final}, {r["iso"] for r in final}, a.prompts_dir) if final else {}
    for r in final:
        r["source_dataset"], r["licence"] = lic.get(r["key"], ("", ""))

    langs = Counter(r["iso"] for r in final)
    mean_reward = sum(r["reward"] for r in final) / max(len(final), 1)
    top_share = (langs.most_common(1)[0][1] / len(final)) if final else 1.0
    fails = []
    if len(final) < a.min_rows:
        fails.append(f"rows {len(final)} < {a.min_rows}")
    if len(langs) < a.min_languages:
        fails.append(f"languages {len(langs)} < {a.min_languages}")
    if keep_rate < a.min_keep_rate:
        fails.append(f"row-filter keep rate {keep_rate:.3f} < {a.min_keep_rate}")
    if mean_reward < a.min_mean_reward:
        fails.append(f"mean reward {mean_reward:.3f} < {a.min_mean_reward}")
    if top_share > a.max_language_share:
        fails.append(f"largest language share {top_share:.3f} > {a.max_language_share}")

    cols = list(final[0].keys()) if final else []
    if final:
        with open(os.path.join(a.out, "archive_merged.jsonl"), "w", encoding="utf-8") as f:
            for r in final:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.table({c: [r[c] for r in final] for c in cols}), os.path.join(a.out, "archive_merged.parquet"))
        except ImportError:
            print("pyarrow missing: jsonl only", file=sys.stderr)
    runs_final = Counter(r["run"] for r in final)
    lic_counts = Counter(r["licence"] for r in final)
    thresholds = {"qe": 1.0, "blend": 1.0, "consensus": 0.5, "blend_min_consensus_chrf": 50.0, "diacritic_deficit": 0}
    report = [f"# Archive publish report", "", f"rows read {per_run_in}, suspect rows dropped by --drop-after: {suspect}, after row filters {len(good)} ({keep_rate:.1%} of non-suspect),",
              f"after merging runs {len(final)} rows, {len(langs)} languages, mean reward {mean_reward:.3f}, largest language share {top_share:.1%}.", "",
              f"row filter rejects: {dict(problems)}", f"rows by run: {dict(runs_final)}", f"licences: {dict(lic_counts)}", "",
              "gate: " + ("PASSED" if not fails else "FAILED: " + "; ".join(fails))]
    status = "dry_run" if a.dry_run else "gate_failed"
    if not fails and not a.dry_run:
        tok = open(a.token_file).read().strip() if os.path.exists(a.token_file) else ""
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=tok)
            api.whoami()
        except Exception as e:  # invalid / missing token: nothing is pushed
            status = "no_token"
            report.append(f"\nHub token unusable ({str(e)[:120]}): NOT pushed. Re-run with a valid token: python scripts/archive_publish.py (same arguments).")
        else:
            api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
            with open(os.path.join(a.out, "README.md"), "w", encoding="utf-8") as f:
                f.write(card(len(final), len(langs), dict(runs_final), lic_counts, thresholds, a.repo))
            api.upload_file(path_or_fileobj=os.path.join(a.out, "README.md"), path_in_repo="README.md", repo_id=a.repo, repo_type="dataset")
            api.upload_file(path_or_fileobj=os.path.join(a.out, "archive_merged.parquet"), path_in_repo="data/train.parquet", repo_id=a.repo, repo_type="dataset")
            status = "pushed"
            report.append(f"\nPUSHED (private) to {a.repo}")
    with open(os.path.join(a.out, "PUBLISH_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    with open(os.path.join(a.out, "PUBLISH_STATUS"), "w") as f:
        f.write(status + "\n")
    print("\n".join(report))
    print("status:", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
