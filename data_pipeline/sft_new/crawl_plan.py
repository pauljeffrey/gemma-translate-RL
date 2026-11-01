"""Decide, for every repo of the owner's HF accounts that is not in the explicit queue, what happens to it. Pure metadata: nothing is downloaded but the cards.

    python -m sft_new.crawl_plan --probe probe_new.json --accounts accounts.json --queued newrepos.txt --out crawl_plan.json [--cards]

Verdicts (first match wins), written with the reason so the owner can audit them:
  skip:nc_licence        licence field or licence_name is non-commercial (size logged)
  skip:nllb              the card names NLLB / "No Language Left Behind" (owner 2026-10-11: no NLLB-derived data for now)
  skip:llama4            the card names Llama-4 as the generator (the Llama 4 licence forbids using its outputs)
  skip:rl_material       preference / reward / judge / diarization-style data (owner 2026-10-11: no RL-type data for now)
  skip:coding            programming / code data (owner: none)
  skip:evaluation        name or splits mark it as an evaluation / test set
  skip:audio_or_no_text  audio-only, or no text column a task can be built from
  skip:no_data_files     script-only or empty repo
  skip:licence_unclear   licence `other` whose name does not say commercial use is fine
  convert:<shape>        chat | qa | instruction | translation
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

NC = re.compile(r"(^|-)nc(-|$)|non-?commercial|research-only|cc-by-nd|noncommercial", re.I)
RL = re.compile(r"\b(dpo|rlhf|reward|preference|chosen|rejected|judge|mr3|rubric|diariz|demucs|snr|pesq|figure3|sbpn|gdrive|chunks?-)\b|(^|[-_])(dpo|rm|prm)([-_]|$)", re.I)
CODE = re.compile(r"(code|coding|program|leetcode|python|sql|compiler|github|swe[-_]|apps|humaneval|codeforces|debug)", re.I)
EVAL = re.compile(r"(^|[-_ ])(eval|evals|test|benchmark|bench|leaderboard|devset|testset)([-_ ]|$)", re.I)
AUDIO = re.compile(r"(audio|speech|asr|tts|voice|wav)", re.I)


def shape_of(cols: list) -> str | None:
    c = [str(x).lower() for x in cols]
    s = set(c)
    if s & {"messages", "conversations", "conversation"}:
        return "chat"
    if ({"instruction", "output"} <= s or {"prompt", "response"} <= s or {"prompt", "completion"} <= s or {"inputs", "targets"} <= s or {"instruction", "response"} <= s
            or {"instruction", "answer"} <= s):
        return "instruction"
    if {"question", "answer"} <= s or {"question", "context"} <= s or {"question_text", "explanation"} <= s or {"question", "answers"} <= s:
        return "qa"
    if {"source", "target"} <= s or {"src", "tgt"} <= s or "translation" in s or {"english", "pidgin"} <= s:
        return "translation"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--accounts", type=Path, required=True)
    ap.add_argument("--queued", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cards", action="store_true", help="read each candidate's card (README) to look for NLLB / Llama-4")
    a = ap.parse_args()
    P = json.loads(a.probe.read_text())
    ACC = json.loads(a.accounts.read_text())
    queued = {l.strip() for l in a.queued.read_text().splitlines() if l.strip()}
    plan, skipped = [], []
    for acct, lst in ACC.items():
        for d in lst:
            rid = d["id"]
            if rid in queued:
                continue
            if acct == "michsethowusu":                           # handled last by its own converters, after the owner's decisions on NLLB
                skipped.append({"account": acct, "repo": rid, "verdict": "deferred:michsethowusu"})
                continue
            v = P.get(rid) or {}
            name = rid.split("/", 1)[1]
            cfg = v.get("configs") or {}
            cols, rows = [], 0
            for c, x in cfg.items():
                cols = cols or x["columns"]
                rows += x["splits"].get("train") or 0
            if not cfg and v.get("parquet"):
                cols, rows = v["parquet"].get("columns") or [], v["parquet"].get("rows") or 0
            gb = None
            lic, lname = str(v.get("license") or ""), str(v.get("license_name") or "")
            rec = {"account": acct, "repo": rid, "licence": lic or None, "rows": rows, "columns": cols[:12], "configs": len(cfg)}
            def skip(why):
                skipped.append({**rec, "verdict": why})
            if "error" in v or not v:
                skip("skip:no_data_files")
            elif NC.search(lic) or NC.search(lname):
                skip("skip:nc_licence")
            elif RL.search(name) or any(re.search(r"chosen|rejected|reward", str(c), re.I) for c in cols) and not shape_of(cols):
                skip("skip:rl_material")
            elif CODE.search(name):
                skip("skip:coding")
            elif EVAL.search(name):
                skip("skip:evaluation")
            elif lic == "other" and not re.search(r"cc-by|apache|mit|odc|permissive|cdla", lname, re.I):
                skip("skip:licence_unclear")
            elif not cols:
                skip("skip:no_data_files")
            elif any(AUDIO.search(str(c)) for c in cols) and not shape_of(cols):
                skip("skip:audio_or_no_text")
            else:
                sh = shape_of(cols)
                if sh is None:
                    skip("skip:audio_or_no_text")
                else:
                    plan.append({**rec, "shape": sh, "verdict": f"convert:{sh}"})
    if a.cards:
        from huggingface_hub import hf_hub_download
        tok = os.environ.get("HF_TOKEN")
        keep = []
        for p in plan:
            try:
                f = hf_hub_download(p["repo"], "README.md", repo_type="dataset", token=tok, cache_dir="/tmp/crawl_cards")
                t = Path(f).read_text(encoding="utf-8", errors="replace")
            except Exception:
                t = ""
            if re.search(r"nllb|no language left behind", t, re.I):
                skipped.append({**p, "verdict": "skip:nllb"})
            elif re.search(r"llama[- ]?4", t, re.I) and re.search(r"generat|distill|synthetic|translat", t, re.I):
                skipped.append({**p, "verdict": "skip:llama4"})
            else:
                keep.append(p)
        plan = keep
    a.out.write_text(json.dumps({"convert": plan, "skipped": skipped}, indent=1, ensure_ascii=False))
    import collections
    print("convert:", len(plan), dict(collections.Counter(p["shape"] for p in plan)))
    print("skipped:", dict(collections.Counter(s["verdict"] for s in skipped)))


if __name__ == "__main__":
    main()
