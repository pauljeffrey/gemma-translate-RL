"""Second AfriCOMET-QE audit pass, kept CHEAP on purpose (owner 2026-10-10: QE is expensive; full QE only where explicitly asked, otherwise look at about
200 random samples of a dataset and decide whether more is worth it).

    python -m sft_new.audit_extra BUILD_DIR [--per-dataset 200] [--comet-python .../comet_python.sh]

audit_translation.py only sees the reservoir samples of SHORT translation rows (<= 500 characters on both sides) and was stopped after 573 groups. This pass
gives every translation dataset that has fewer than --enough scored pairs ONE sample of --per-dataset random rows (across its language pairs):
  * both sides cut at a sentence boundary to <= 450 characters (QE scores fall with length, so these numbers are for comparison between datasets, not with
    the short-row numbers);
  * `alpaca_taco` rows: the English response kept in the <think> pivot against the answer in the target language, and the English instruction against the
    target-language instruction (the only translated instruction sets that keep their English side).
Output: BUILD_DIR/audit_extra.json (same fields as audit.json) and BUILD_DIR/drop_list_extra.json (same rule: >= 20 scored, share below 0.46 of 0.8 or more,
mean below 0.38). finalize.py --drop-list applies both lists.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from sft_new.quality_check import LOW, scorer, summarize  # noqa: E402

UNRELIABLE = {("tsn", "sot"), ("sot", "tsn"), ("fon", "ewe"), ("ewe", "twi")}


def cut(s: str, n: int = 450) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    m = list(re.finditer(r"[.!?።]\s", s[:n]))
    return s[: m[-1].end()].strip() if m and m[-1].end() > 80 else s[:n]


def main():
    from rl.bin_data import AFRICOMET_CODE
    ap = argparse.ArgumentParser()
    ap.add_argument("build", type=Path)
    ap.add_argument("--per-dataset", type=int, default=200)
    ap.add_argument("--enough", type=int, default=150, help="datasets with at least this many pairs already scored are skipped")
    ap.add_argument("--shards-per-step", type=int, default=2)
    ap.add_argument("--comet-python", default=str(HERE.parent / ".venv-comet" / "bin" / "python"))
    a = ap.parse_args()
    done = json.loads((a.build / "audit.json").read_text()) if (a.build / "audit.json").exists() else {}
    scored = collections.Counter()
    for k, v in done.items():
        scored[k.split("|")[0]] += v["n"]
    rng = random.Random(0)
    groups: dict = collections.defaultdict(list)
    for f in sorted((a.build / "stats").glob("*.json")):
        try:
            j = json.loads(f.read_text())
        except ValueError:
            continue
        tag, conv = f.stem, j["converter"]
        for shard in sorted((a.build / "data" / tag).glob("part-*.parquet"))[: a.shards_per_step]:
            t = pq.read_table(shard, columns=["task", "source_dataset", "source_lang", "targ_lang", "input", "response", "messages"]).to_pydict()
            for i, task in enumerate(t["task"]):
                sl, tl = t["source_lang"][i], t["targ_lang"][i]
                if conv == "alpaca_taco" and sl == tl:
                    m = re.search(r"<think>Instruction in English: (.*?)\nResponse in English: (.*?)</think>", t["messages"][i], re.S)
                    if m and tl in AFRICOMET_CODE:
                        msgs = json.loads(t["messages"][i])
                        ans = msgs[-1]["content"].split("<response>", 1)[-1]
                        groups[f"{t['source_dataset'][i]} (pivot response)"].append((cut(m.group(2)), cut(ans)))
                    continue
                if task != "translation" or sl not in AFRICOMET_CODE or tl not in AFRICOMET_CODE or (sl, tl) in UNRELIABLE:
                    continue
                key = t["source_dataset"][i]
                if scored.get(key, 0) >= a.enough:
                    continue
                x, y = cut(t["input"][i]), cut(t["response"][i])
                if len(x) >= 20 and len(y) >= 20:
                    groups[key].append((x, y))
    prev = json.loads((a.build / "audit_extra.json").read_text()) if (a.build / "audit_extra.json").exists() else {}
    score, proc = scorer(a.comet_python)
    res, t0, n = dict(prev), time.time(), 0
    try:
        for key, rows in sorted(groups.items()):
            if key in prev or len(rows) < 20:
                continue
            rng.shuffle(rows)
            sc = score(rows[: a.per_dataset])
            n += len(sc)
            res[key] = {**summarize(sc), "population_seen": len(rows)}
            (a.build / "audit_extra.json").write_text(json.dumps(res, indent=1))
            print(key[:100], res[key], f"{n / (time.time() - t0):.2f} pairs/s", flush=True)
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)
    drop = [{"group": k, **m} for k, m in res.items() if m["n"] >= 20 and m[f"share<{LOW}"] >= 0.8 and m["mean"] < 0.38]
    (a.build / "drop_list_extra.json").write_text(json.dumps(drop, indent=1))
    print(f"{len(res)} extra groups scored, {len(drop)} on the drop list")


if __name__ == "__main__":
    main()
