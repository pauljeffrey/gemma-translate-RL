"""Gemini BATCH-API runner for the J1 judge pilot, with a hard USD budget ledger.

    judge_pilot_batch.py count  --items items.jsonl --model M --cache counts_M.json     (free count_tokens on every prompt)
    judge_pilot_batch.py submit --items items.jsonl --model M --group probe|main --cap N --counts counts_M.json --ledger ledger.json --out dir
    judge_pilot_batch.py fetch  --ledger ledger.json --model M --group G --out dir        (poll once; write results + real usage)

Budget rule: before a submit, worst_case = exact prompt tokens x batch input price + n x max_output_tokens x batch output price
(thinking tokens are billed as output and are inside max_output_tokens), +5% margin. The submit is REFUSED if
(sum of finished batches' real cost + sum of unfinished batches' worst case + this worst case) > BUDGET.
The API key is read from .env (GEMINI_API_KEY) and is never printed, logged or written.
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
BUDGET = 6.00
MARGIN = 1.05
# USD per 1M tokens, STANDARD list prices given by the owner; batch = half.
PRICE = {"gemini-2.5-pro": (1.25, 10.0), "gemini-3.1-pro-preview": (2.0, 12.0), "gemini-3.8-flash": (0.75, 3.75)}
# gemini-2.5-pro: 404 "no longer available to new users" on this key (count_tokens and batch); 3.1-pro-preview (Google's named
# successor, list price 2.00/12.00, batch half, checked on ai.google.dev/gemini-api/docs/pricing 2026-10-25) is the substitute.
BATCH_FACTOR = 0.5


def client():
    key = dotenv_values(ENV)["GEMINI_API_KEY"]
    return genai.Client(api_key=key)


def batch_cost(model, in_tok, out_tok):
    pi, po = PRICE[model]
    return BATCH_FACTOR * (in_tok * pi + out_tok * po) / 1e6


def load_items(p, group=None):
    its = [json.loads(l) for l in open(p, encoding="utf-8")]
    return [i for i in its if group in (None, i["batch"])]


def cmd_count(a):
    c = client()
    items = load_items(a.items)
    cache = json.load(open(a.cache)) if Path(a.cache).exists() else {}
    todo = [i for i in items if i["id"] not in cache]

    def one(i):
        for t in range(5):
            try:
                r = c.models.count_tokens(model=a.model, contents=i["prompt"])
                return i["id"], r.total_tokens
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:150]}"
                time.sleep(2 + 3 * t)
        raise RuntimeError(err)

    with ThreadPoolExecutor(4) as ex:
        for k, v in ex.map(one, todo):
            cache[k] = v
    json.dump(cache, open(a.cache, "w"))
    vals = [cache[i["id"]] for i in items]
    print(f"{a.model}: {len(vals)} prompts, total {sum(vals)} tokens, mean {sum(vals)/len(vals):.0f}, max {max(vals)}")


def ledger_total(led):
    return sum((e["actual_usd"] if e["status"] == "done" else e["worst_usd"]) for e in led)


def cmd_submit(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    led = json.load(open(a.ledger)) if Path(a.ledger).exists() else []
    if any(e["model"] == a.model and e["group"] == a.group for e in led):
        sys.exit(f"REFUSED: {a.model}/{a.group} already submitted (one run per judge, no repeats)")
    items = load_items(a.items, a.group)
    counts = json.load(open(a.counts))
    in_tok = sum(counts[i["id"]] for i in items)
    worst = batch_cost(a.model, in_tok, len(items) * a.cap) * MARGIN
    running = ledger_total(led)
    print(f"{a.model}/{a.group}: n={len(items)} input_tokens={in_tok} cap={a.cap} worst_case=${worst:.3f} running_before=${running:.3f}")
    if running + worst > BUDGET:
        sys.exit(f"REFUSED: running ${running:.3f} + worst ${worst:.3f} = ${running+worst:.3f} > budget ${BUDGET:.2f}")
    cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=a.cap)  # thinking left at the model default
    reqs = [types.InlinedRequest(contents=i["prompt"], metadata={"id": i["id"]}, config=cfg) for i in items]
    c = client()
    job = c.batches.create(model=a.model, src=reqs, config=types.CreateBatchJobConfig(display_name=f"j1-{a.model}-{a.group}"))
    led.append(dict(model=a.model, group=a.group, n=len(items), cap=a.cap, input_tokens=in_tok, worst_usd=worst, status="submitted", job=job.name, submitted=time.time(),
                    config=dict(temperature=0.0, max_output_tokens=a.cap, thinking="model default")))
    json.dump(led, open(a.ledger, "w"), indent=1)
    print("submitted", job.name, job.state)


def cmd_fetch(a):
    led = json.load(open(a.ledger))
    e = next(e for e in led if e["model"] == a.model and e["group"] == a.group)
    c = client()
    job = c.batches.get(name=e["job"])
    st = str(job.state)
    print(a.model, a.group, st, getattr(job, "completion_stats", None))
    if "SUCCEEDED" not in st:
        return
    resp = job.dest.inlined_responses
    rows, in_tot, out_tot = [], 0, 0
    for r in resp:
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
    cost = batch_cost(a.model, in_tot, out_tot)
    e.update(status="done", actual_usd=cost, actual_input_tokens=in_tot, actual_output_tokens=out_tot, n_returned=len(rows))
    json.dump(led, open(a.ledger, "w"), indent=1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"raw_{a.model}_{a.group}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"done: n={len(rows)} in={in_tot} out(incl. thinking)={out_tot} cost=${cost:.4f}; running total ${ledger_total(led):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, f in (("count", cmd_count), ("submit", cmd_submit), ("fetch", cmd_fetch)):
        p = sub.add_parser(n)
        p.set_defaults(fn=f)
        p.add_argument("--model", required=True)
        p.add_argument("--items")
        p.add_argument("--cache")
        p.add_argument("--counts")
        p.add_argument("--group")
        p.add_argument("--cap", type=int)
        p.add_argument("--ledger")
        p.add_argument("--out")
    a = ap.parse_args()
    a.fn(a)
