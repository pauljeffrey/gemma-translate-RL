"""Run 1 (QE-prompt validation): rebuild the 300 pilot segments with the Figure 9 (no reference) prompt. No API calls.

    python judge_j2_run1_build.py --pilot-items <scratch j1/work/items.jsonl> --scratch <dir> --out <repo reports/data/judge_j2_2026-10-25>
Same segments, same demonstrations (pilot demo_ids), same order as the pilot; only the prompt variant differs.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from judge_j2_common import build_qe_prompt
from judge_pilot_common import build_mte_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-items", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scratch, out = Path(a.scratch), Path(a.out)
    scratch.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in open(a.pilot_items, encoding="utf-8")]
    assert len(items) == 300
    rows = []
    for it in items:
        assert build_mte_prompt(it["demos"], it["src"], it["mt"], it["ref"]) == it["prompt"]  # pilot prompt reproduced
        rows.append(dict(id=it["id"], set=it["set"], lp=it["lp"], z=it["z"], da=it["da"], batch="qe", prompt=build_qe_prompt(it["demos"], it["src"], it["mt"]),
                         ssa_mtl_ref=it["ssa_mtl_ref"], africomet_stl11_ref=it["africomet_stl11_ref"], africomet_mtl_ref=it["africomet_mtl_ref"],
                         ssa_mtl_qe=it["ssa_mtl_qe"], ssa_qe=it["ssa_qe"]))
    with open(scratch / "items_run1.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with gzip.open(out / "prompts_run1_qe.jsonl.gz", "wt", encoding="utf-8") as f:  # auditable record of every request
        for r in rows:
            f.write(json.dumps(dict(id=r["id"], prompt=r["prompt"]), ensure_ascii=False) + "\n")
    next_yor = next(r for r in rows if r["lp"] == "eng-yor")
    next_swh = next(r for r in rows if r["lp"] == "eng-swh")
    (out / "prompt_example_qe_run1_eng-yor.txt").write_text(next_yor["prompt"], encoding="utf-8")
    (out / "prompt_example_qe_run1_eng-swh.txt").write_text(next_swh["prompt"], encoding="utf-8")
    print("wrote", len(rows), "QE prompts; mean chars", sum(len(r["prompt"]) for r in rows) / len(rows))


if __name__ == "__main__":
    main()
