#!/usr/bin/env python
"""Write Gemini Batch API input JSONL for the long-document pseudo-reference benchmark.  NO API call, NO key.

Schema (https://ai.google.dev/gemini-api/docs/batch-api, read 2026-10-23, file-based batch): one JSON object per line
    {"key": "<iid>" | "<iid>@@c0003", "request": {"contents": [{"parts": [{"text": ...}]}],
      "system_instruction": {"parts": [{"text": <instruction>}]},
      "generation_config": {"temperature": 0, "max_output_tokens": N, "thinking_config": {...}}}}
The batch page shows `generation_config` (snake_case) in its JSONL examples and `systemInstruction` in one JS example; the
REST proto-JSON parser accepts both spellings.  Default here is the documented snake_case; `--casing camel` switches.

Inputs (box or laptop): docs.jsonl + items.jsonl of the LC1 benchmark (see /workspace/jobs/lc1/work/).
Examples:
  python gemini_batch_requests.py --docs docs.jsonl --items items.jsonl --out-dir gem_req --model gemini-2.5-flash-lite \
      --mode whole --selection reduced --thinking-budget 0
  python gemini_batch_requests.py ... --mode chunk --chunk-size 8 --langs hau,swh --bins b512,b1024
  python gemini_batch_requests.py ... --pilot 50 --langs hau,swh,yor            # sentence pilot from SmolDoc `sm` docs
Outputs in --out-dir: requests_NN.jsonl (split by size / request / est.-input-token caps), requests_index.jsonl (key -> iid, chunk,
sentence range, paragraph breaks), manifest.json and SHA256SUMS.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemini_batch_common as C  # noqa: E402

MAX_OUT_CAP = 65536          # Gemini 2.5/3 Flash maximum output tokens (incl. thinking); UNVERIFIED for every model, override with --max-output-tokens
DOC_BYTES_CAP = 1_900_000_000  # documented file limit is 2 GB (batch page / rate-limits page)


def rd(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def est_tokens(text, stored=None):
    return max(C.est_tokens_chars4(text), stored or 0)


def sentence_text(s):
    return " ".join(str(s).split())


def gen_config(a, max_out):
    g = {"temperature": 0, "max_output_tokens": int(max_out)}
    tc = {}
    if a.thinking_budget is not None:
        tc["thinking_budget"] = a.thinking_budget          # 2.5-style (0 = off where the model allows; -1 = dynamic) -- semantics UNVERIFIED on today's docs
    if a.thinking_level:
        tc["thinking_level"] = a.thinking_level            # 3.x-style, documented on the thinking page
    if tc:
        g["thinking_config"] = tc
    return g


def camelize(o):
    if isinstance(o, dict):
        return {"".join(w if i == 0 else w.capitalize() for i, w in enumerate(k.split("_"))): camelize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [camelize(x) for x in o]
    return o


def build_request(system, text, gcfg, casing):
    req = {"contents": [{"parts": [{"text": text}]}], "system_instruction": {"parts": [{"text": system}]}, "generation_config": gcfg}
    return camelize(req) if casing == "camel" else req


def think_headroom(a):
    """Tokens reserved inside max_output_tokens for hidden thinking (thinking counts against the output cap)."""
    if a.thinking_budget is not None:
        return max(0, a.thinking_budget) if a.thinking_budget >= 0 else 8192
    if a.thinking_level:
        return {"minimal": 512, "low": 1024, "medium": 2048, "high": 4096}.get(a.thinking_level, 4096)
    return 4096        # model default (dynamic thinking on 2.5 Flash/Pro and 3.x); harmless for models that think less


def max_out_for(a, src_tok_est, n_lines=None):
    if a.max_output_tokens:
        return a.max_output_tokens
    # 4.5 x source tokens covers the 3.5 multiplier plus margin
    return int(min(MAX_OUT_CAP, max(1024, math.ceil(4.5 * src_tok_est) + 256 + think_headroom(a))))


def make_requests(a, docs, items):
    sel = C.select_items(items, a.selection, seed=a.seed)
    if a.langs:
        keep = set(a.langs.split(","))
        sel = [i for i in sel if i["iso"] in keep]
    if a.bins:
        kb = set(a.bins.split(","))
        sel = [i for i in sel if i["bin"] in kb]
    if a.sets:
        ks = set(a.sets.split(","))
        sel = [i for i in sel if i["set"] in ks]
    sel.sort(key=lambda i: i["iid"])
    reqs = []   # (key, request dict, index row, est_in_tokens)
    for it in sel:
        d = docs[it["doc_id"]]
        sents = d["sents"] if isinstance(d["sents"], list) else []
        if a.mode == "whole":
            sysm = C.instruction(it["iso"], "whole")
            g = gen_config(a, max_out_for(a, est_tokens(d["text"], d.get("src_tokens"))))
            req = build_request(sysm, d["text"], g, a.casing)
            reqs.append((it["iid"], req, dict(key=it["iid"], iid=it["iid"], chunk=None, n_chunks=1, lo=0, hi=len(sents), max_output_tokens=g["max_output_tokens"]),
                         est_tokens(sysm) + est_tokens(d["text"], d.get("src_tokens"))))
        else:
            if not sents:
                raise SystemExit("document %s has no sentence list; chunk mode impossible" % it["doc_id"])
            pstarts = set(d.get("para_starts", [0]))
            spans = C.chunk_spans(len(sents), a.chunk_size)
            for ci, (lo, hi) in enumerate(spans):
                lines = [sentence_text(s) for s in sents[lo:hi]]
                text = "\n".join(lines)
                sysm = C.instruction(it["iso"], "chunk", hi - lo)
                g = gen_config(a, max_out_for(a, est_tokens(text)))
                key = C.make_key(it["iid"], ci)
                req = build_request(sysm, text, g, a.casing)
                reqs.append((key, req, dict(key=key, iid=it["iid"], chunk=ci, n_chunks=len(spans), lo=lo, hi=hi,
                                            para_break_before=[(k in pstarts and k != 0) for k in range(lo, hi)], max_output_tokens=g["max_output_tokens"]),
                             est_tokens(sysm) + est_tokens(text)))
    return reqs


def make_pilot(a, docs, items):
    """a.pilot sentences per language, sampled with a fixed seed from SmolDoc `sm` documents (which have human references)."""
    langs = a.langs.split(",") if a.langs else C.HUMAN_LANGS
    reqs = []
    if a.pilot_file:   # {"id":..., "text":...} per line, e.g. FLORES+ devtest sentences prepared by the owner
        sents = rd(a.pilot_file)
        for iso in langs:
            rng = random.Random("%s|pilotfile|%s" % (a.seed, iso))
            pick = rng.sample(sents, min(a.pilot, len(sents)))
            for s in sorted(pick, key=lambda x: str(x["id"])):
                key = "pilot|%s|%s" % (s["id"], iso)
                sysm = C.instruction(iso, "chunk", 1)
                g = gen_config(a, max_out_for(a, est_tokens(s["text"])))
                reqs.append((key, build_request(sysm, sentence_text(s["text"]), g, a.casing),
                             dict(key=key, iid=key, chunk=None, n_chunks=1, lo=0, hi=1, max_output_tokens=g["max_output_tokens"]), est_tokens(sysm) + est_tokens(s["text"])))
        return reqs
    for iso in langs:
        pool = []
        for it in sorted((i for i in items if i["set"] == "human" and i["bin"] == "sm" and i["iso"] == iso), key=lambda i: i["iid"]):
            sents = docs[it["doc_id"]]["sents"]
            pool += [(it["doc_id"], k, sents[k]) for k in range(len(sents))]
        rng = random.Random("%s|pilot|%s" % (a.seed, iso))
        for doc_id, k, s in sorted(rng.sample(pool, min(a.pilot, len(pool)))):
            key = "pilot|%s|%d|%s" % (doc_id, k, iso)
            sysm = C.instruction(iso, "chunk", 1)
            g = gen_config(a, max_out_for(a, est_tokens(s)))
            reqs.append((key, build_request(sysm, sentence_text(s), g, a.casing),
                         dict(key=key, iid=key, chunk=None, n_chunks=1, lo=0, hi=1, max_output_tokens=g["max_output_tokens"]), est_tokens(sysm) + est_tokens(s)))
    return reqs


def write_split(reqs, out_dir, a):
    os.makedirs(out_dir, exist_ok=True)
    files, cur, cur_bytes, cur_tok = [], [], 0, 0

    def flush():
        nonlocal cur, cur_bytes, cur_tok
        if not cur:
            return
        name = "requests_%02d.jsonl" % (len(files) + 1)
        path = os.path.join(out_dir, name)
        h = hashlib.sha256()
        with open(path, "wb") as f:
            for line, _ in cur:
                f.write(line)
                h.update(line)
        files.append(dict(name=name, requests=len(cur), bytes=cur_bytes, est_input_tokens=cur_tok, sha256=h.hexdigest()))
        for _, key in cur:
            index_file[key] = name
        cur, cur_bytes, cur_tok = [], 0, 0

    index_file = {}
    for key, req, _row, et in reqs:
        line = (json.dumps({"key": key, "request": req}, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        if cur and (cur_bytes + len(line) > a.max_file_bytes or len(cur) >= a.max_requests_per_file or
                    (a.max_est_input_tokens_per_file and cur_tok + et > a.max_est_input_tokens_per_file)):
            flush()
        cur.append((line, key))
        cur_bytes += len(line)
        cur_tok += et
    flush()
    return files, index_file


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True, help="e.g. gemini-2.5-flash-lite; used only for the manifest and the batch create call")
    ap.add_argument("--mode", choices=["whole", "chunk"], default="whole")
    ap.add_argument("--chunk-size", type=int, default=8, help="sentences per request in chunk mode")
    ap.add_argument("--selection", choices=["full", "reduced_core", "reduced_sci_dev", "reduced"], default="full")
    ap.add_argument("--seed", type=int, default=C.REDUCED_SEED)
    ap.add_argument("--langs", default="", help="comma list of iso codes (filter)")
    ap.add_argument("--bins", default="", help="comma list of bins (filter)")
    ap.add_argument("--sets", default="", help="comma list of sets core,human,sci (filter)")
    ap.add_argument("--thinking-budget", type=int, default=None, help="thinking_config.thinking_budget (2.5 models; 0 = off where allowed)")
    ap.add_argument("--thinking-level", default=None, help="thinking_config.thinking_level (3.x/4 models: minimal|low|medium|high)")
    ap.add_argument("--max-output-tokens", type=int, default=0, help="fixed cap for every request; default = per-request 4.5 x source tokens + margin")
    ap.add_argument("--casing", choices=["snake", "camel"], default="snake")
    ap.add_argument("--pilot", type=int, default=0, help="instead of the benchmark: N sentences per language (1 per request)")
    ap.add_argument("--pilot-file", default="", help="JSONL {id,text} of pilot sentences (e.g. FLORES+ devtest) instead of SmolDoc sm sentences")
    ap.add_argument("--max-file-bytes", type=int, default=DOC_BYTES_CAP)
    ap.add_argument("--max-requests-per-file", type=int, default=10 ** 9)
    ap.add_argument("--max-est-input-tokens-per-file", type=int, default=0, help="keep each file under your tier's enqueued-token cap (rate-limits page)")
    a = ap.parse_args(argv)
    if a.thinking_budget is not None and a.thinking_level:
        ap.error("use --thinking-budget OR --thinking-level")
    docs = {d["doc_id"]: d for d in rd(a.docs)}
    items = rd(a.items)
    reqs = make_pilot(a, docs, items) if a.pilot else make_requests(a, docs, items)
    if not reqs:
        raise SystemExit("no requests selected")
    keys = [r[0] for r in reqs]
    assert len(keys) == len(set(keys)), "duplicate keys"
    files, index_file = write_split(reqs, a.out_dir, a)
    with open(os.path.join(a.out_dir, "requests_index.jsonl"), "w", encoding="utf-8") as f:
        for key, _req, row, et in reqs:
            row = dict(row, file=index_file[key], est_input_tokens=et)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = dict(model=a.model, mode="pilot" if a.pilot else a.mode, chunk_size=a.chunk_size if a.mode == "chunk" else None, selection=a.selection,
                    seed=a.seed, langs=a.langs, bins=a.bins, sets=a.sets, thinking_budget=a.thinking_budget, thinking_level=a.thinking_level,
                    casing=a.casing, temperature=0, requests=len(reqs), est_input_tokens=sum(r[3] for r in reqs), files=files,
                    instruction_whole=C.INSTR_WHOLE, instruction_chunk=C.INSTR_CHUNK)
    with open(os.path.join(a.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    with open(os.path.join(a.out_dir, "SHA256SUMS"), "w", encoding="utf-8") as f:
        for fi in files:
            f.write("%s  %s\n" % (fi["sha256"], fi["name"]))
    print("wrote %d requests in %d file(s) to %s; est. input tokens %d" % (len(reqs), len(files), a.out_dir, manifest["est_input_tokens"]))
    return manifest


if __name__ == "__main__":
    main()
