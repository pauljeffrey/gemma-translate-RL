"""Gemini BATCH-API runner for the J2 judge runs (gemini-3.8-flash only), with a hard USD budget ledger.

    judge_j2_batch.py list                                                     (free: show existing batch jobs, display names only)
    judge_j2_batch.py count  --items items.jsonl --cache counts.json           (free count_tokens on every prompt)
    judge_j2_batch.py submit --items items.jsonl --group G --cap N --counts counts.json --ledger ledger.json [--only-ids f.txt]
    judge_j2_batch.py fetch  --ledger ledger.json --group G --out dir          (poll once; write raw results + real usage)

Same protocol as judge_pilot_batch.py (temperature 0, thinking at the model default, max_output_tokens = cap, Batch API).
Differences: BUDGET 8.00 USD for all J2 jobs together; the display name is "j2-<group>" and a submit is REFUSED if a batch job with that display
name already exists on the account (job creation is not idempotent and is billed), or if the group is already in the ledger.
Budget rule: worst_case = exact prompt tokens x batch input price + n x cap x batch output price (thinking is inside the cap), +5%; REFUSED if
sum(done jobs' real cost) + sum(unfinished jobs' worst case) + this worst case > BUDGET. The API key is read from .env and never printed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import dotenv_values
from google import genai
from google.genai import types

ENV = Path(__file__).resolve().parents[2] / ".env"
MODEL = "gemini-3.8-flash"
BUDGET = 8.00
MARGIN = 1.05
PRICE = (0.75, 3.75)  # USD per 1M tokens, standard list price given by the owner (until 2026-12-31); batch = half
BATCH_FACTOR = 0.5


def client():
    return genai.Client(api_key=dotenv_values(ENV)["GEMINI_API_KEY"])


def batch_cost(in_tok, out_tok):
    return BATCH_FACTOR * (in_tok * PRICE[0] + out_tok * PRICE[1]) / 1e6


def load_items(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def ledger_total(led):
    return sum((e["actual_usd"] if e["status"] == "done" else e["worst_usd"]) for e in led)


def existing_names(c):
    return [(j.display_name, j.name, str(j.state)) for j in c.batches.list()]


def cmd_list(a):
    for n in existing_names(client()):
        print(n)


def cmd_count(a):
    c = client()
    items = load_items(a.items)
    cache = json.load(open(a.cache)) if Path(a.cache).exists() else {}
    todo = [i for i in items if i["id"] not in cache]

    def one(i):
        err = ""
        for t in range(5):
            try:
                return i["id"], c.models.count_tokens(model=MODEL, contents=i["prompt"]).total_tokens
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:150]}"
                time.sleep(2 + 3 * t)
        raise RuntimeError(err)

    with ThreadPoolExecutor(6) as ex:
        for k, v in ex.map(one, todo):
            cache[k] = v
    json.dump(cache, open(a.cache, "w"))
    vals = [cache[i["id"]] for i in items]
    print(f"{MODEL}: {len(vals)} prompts, total {sum(vals)} tokens, mean {sum(vals)/len(vals):.0f}, max {max(vals)}")


def cmd_submit(a):
    led = json.load(open(a.ledger)) if Path(a.ledger).exists() else []
    if any(e["group"] == a.group for e in led):
        sys.exit(f"REFUSED: group {a.group} is already in the ledger (one run per request, no repeats)")
    c = client()
    display = f"j2-{a.group}"
    if any(n[0] == display for n in existing_names(c)):
        sys.exit(f"REFUSED: a batch job with display name {display} already exists on the account")
    items = load_items(a.items)
    if a.only_ids:
        keep = {l.strip() for l in open(a.only_ids) if l.strip()}
        items = [i for i in items if i["id"] in keep]
        assert len(items) == len(keep), (len(items), len(keep))
    counts = json.load(open(a.counts))
    in_tok = sum(counts[i["id"]] for i in items)
    worst = batch_cost(in_tok, len(items) * a.cap) * MARGIN
    running = ledger_total(led)
    print(f"{display}: n={len(items)} input_tokens={in_tok} cap={a.cap} worst_case=${worst:.3f} running_before=${running:.3f}")
    if running + worst > BUDGET:
        sys.exit(f"REFUSED: running ${running:.3f} + worst ${worst:.3f} = ${running+worst:.3f} > budget ${BUDGET:.2f}")
    entry = dict(model=MODEL, group=a.group, n=len(items), cap=a.cap, input_tokens=in_tok, worst_usd=worst, status="creating", job=None,
                 submitted=time.time(), config=dict(temperature=0.0, max_output_tokens=a.cap, thinking="model default"))
    led.append(entry)
    json.dump(led, open(a.ledger, "w"), indent=1)  # written BEFORE the create call so that a crash cannot lose the record
    cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=a.cap)
    reqs = [types.InlinedRequest(contents=i["prompt"], metadata={"id": i["id"]}, config=cfg) for i in items]
    job = c.batches.create(model=MODEL, src=reqs, config=types.CreateBatchJobConfig(display_name=display))
    entry.update(status="submitted", job=job.name)
    json.dump(led, open(a.ledger, "w"), indent=1)
    print("submitted", job.name, job.state)


def cmd_fetch(a):
    led = json.load(open(a.ledger))
    e = next(e for e in led if e["group"] == a.group)
    c = client()
    job = c.batches.get(name=e["job"])
    st = str(job.state)
    print(a.group, st, getattr(job, "completion_stats", None))
    if "SUCCEEDED" not in st:
        return
    rows, in_tot, out_tot = [], 0, 0
    for r in job.dest.inlined_responses:
        rid = (r.metadata or {}).get("id")
        text, um, fin = None, None, None
        if r.response is not None:
            text = r.response.text if r.response.candidates else None
            um = r.response.usage_metadata
            fin = str(r.response.candidates[0].finish_reason) if r.response.candidates else "NO_CANDIDATE"
        pt = (um.prompt_token_count or 0) if um else 0
        ct = (um.candidates_token_count or 0) if um else 0
        tt = (um.thoughts_token_count or 0) if um else 0
        in_tot += pt
        out_tot += ct + tt
        rows.append(dict(id=rid, text=text, finish=fin, prompt_tokens=pt, candidate_tokens=ct, thought_tokens=tt, error=str(r.error) if r.error else None))
    cost = batch_cost(in_tot, out_tot)
    e.update(status="done", actual_usd=cost, actual_input_tokens=in_tot, actual_output_tokens=out_tot, n_returned=len(rows))
    json.dump(led, open(a.ledger, "w"), indent=1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"raw_{a.group}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"done: n={len(rows)} in={in_tot} out(incl. thinking)={out_tot} cost=${cost:.4f}; running total ${ledger_total(led):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, f in (("list", cmd_list), ("count", cmd_count), ("submit", cmd_submit), ("fetch", cmd_fetch)):
        p = sub.add_parser(n)
        p.set_defaults(fn=f)
        for k in ("--items", "--cache", "--counts", "--group", "--ledger", "--out", "--only-ids"):
            p.add_argument(k)
        p.add_argument("--cap", type=int)
    a = ap.parse_args()
    a.fn(a)
