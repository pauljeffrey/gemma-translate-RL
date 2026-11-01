#!/usr/bin/env python3
"""(d) long-document distillation and (e) self-distilled anchors over the CLEAN pools (agent C2, 2026-10-28). One script, four steps; the same steps serve both components (--kind).

  select  pick source documents per language and bin from clean_v3 (general) and clean_sci_v2 (science) and write docs.jsonl + req_sent.jsonl (one request per unique sentence piece)
  gen     best-of-4 candidates per sentence unit with the NEW RL model: greedy + 3 samples at T 0.7 (engine vllm on the GPU box | hf on CPU for the smoke)
  score   decode, validate and judge every candidate (QE ensemble for the QE languages, chrF++ medoid for the consensus-only ones), assemble and score the documents
          (= the old scripts/mix2_longdoc_score.py, run with the REFIT calibration)
  pick    per-language relative thresholds, per-language token budgets -> <kind>_selected.jsonl with provenance + licence columns
          (= the old scripts/mix2_longdoc_select.py)

  python s4_longdoc.py select --kind longdoc --pool-v3 clean_v3 --pool-sci clean_sci_v2 --out W --model NEW_RL_MODEL --index I.npz --manifest M.json
  python s4_longdoc.py gen    --work W --model NEW_RL_MODEL --engine vllm
  python s4_longdoc.py score  --work W --model NEW_RL_MODEL --calibration CAL --anchors ANCH --glotlid LID --scorer scorer.json
  python s4_longdoc.py pick   --work W --model NEW_RL_MODEL --kind longdoc

Defaults reproduce A2W: long documents b512 / b1024 (b2048 was dropped with the new bins; it was 40 % of the A2W long-document tokens, so the candidates per language are RAISED from 22 / 16 /
10 to 70 / 60 at b512 / b1024 = 130 per language, as many as the 114 of the A2W run, to reach the same 13.5 % share; the mixer reports the realised share) (general / science share 0.75 / 0.25 at b512 and 0.5 /
0.5 at b1024), per-language relative thresholds (70th percentile of the language's own documents, clipped), 60,000 tokens per language, amh and tir x1.5; anchors: 47 weak / Ge'ez languages,
bins b64 / b256 / b512 (30 / 40 / 24 candidates = 94 per language, as A2W), UNFILTERED by quality (own outputs, only the hard failure filters), 100,000 tokens per language. LibreTexts documents are INCLUDED and
carry source_dataset, the pool licence label and the unresolved-conflict note in every row. Nothing here reads NLLB outputs, TranslatePsy or any benchmark text.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s4_common as K  # noqa: E402

KIND = {
    "longdoc": dict(per_lang="b512=70,b1024=60", sci_share="b512=0.25,b1024=0.5", langs=None, rel_q=0.70, budget=60000, bin_share="b512=0.4,b1024=0.6", mult="amh=1.5,tir=1.5",
                    conf_clips=("0.3,1.0", "0.0,0.4", "50,58.8", "38,45")),
    "anchor": dict(per_lang="b64=30,b256=40,b512=24", sci_share="b64=0.25,b256=0.25,b512=0.25", langs=K.ANCHOR_LANGS, rel_q=0.0, budget=100000, bin_share="b64=0.2,b256=0.5,b512=0.3",
                   mult="amh=1.5,tir=1.5", conf_clips=("-9,-9", "-9,-9", "-9,-9", "-9,-9")),
}


def parse_kv(s):
    return {k: float(v) for k, v in (x.split("=") for x in s.split(",") if x)}


# ---------------------------------------------------------------------------------------------------------------------------- select
def cmd_select(a):
    import pyarrow.parquet as pq

    K.gemma_pkg()
    from gemma_mt.eval.longdoc import sentences_are_stable, split_long_sentence, split_sentences
    from gemma_mt.format import build_prompt
    from gemma_mt.languages import ALL_ISOS

    d = KIND[a.kind]
    tok = K.tokenizer(a.model)
    ntok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
    per = {k: int(v) for k, v in parse_kv(a.per_lang or d["per_lang"]).items()}
    sci_share = parse_kv(a.sci_share or d["sci_share"])
    langs = [l for l in ALL_ISOS if os.path.exists(os.path.join(a.pool_v3, f"{l}.parquet"))]
    want = [x for x in a.languages.split(",") if x] or (list(d["langs"]) if d["langs"] else [])
    if want:
        langs = [l for l in langs if l in want]
    excl_src = [x.lower() for x in a.exclude_source.split(",") if x]
    excl = collections.defaultdict(set)
    for f in [x for x in a.exclude.split(",") if x]:
        for r in K.read_jsonl(f):
            excl[r["iso"]].add(hashlib.md5(r["english"].encode()).hexdigest())
    P = K.Protected(a.index, a.manifest, a.nproc)
    os.makedirs(a.out, exist_ok=True)
    K.log(f"{a.kind}: {len(langs)} languages, per-language candidates {per}, protected index {P.version}")
    docs, stats = [], collections.defaultdict(collections.Counter)
    for iso in langs:
        rng = random.Random(f"{a.seed}|{iso}")
        pools = {}
        cols = ["english", "length_bin", "source_dataset", "licence", "prov_pool", "prov_file", "prov_row", "prov_index"]
        t3 = pq.read_table(os.path.join(a.pool_v3, f"{iso}.parquet"), columns=cols)
        lb = t3.column("length_bin").to_pylist()
        for b in per:
            idx = [i for i, x in enumerate(lb) if x == b]
            rng.shuffle(idx)
            pools[("general", b)] = [(t3, i) for i in idx[:per[b] * 40]]
        sp = os.path.join(a.pool_sci, f"{iso}.parquet")
        if os.path.exists(sp):
            ts = pq.read_table(sp, columns=cols + ["domain", "field"])
            sb, ss, sd = ts.column("length_bin").to_pylist(), ts.column("source_dataset").to_pylist(), ts.column("domain").to_pylist()
            for b in per:
                idx = [i for i in range(ts.num_rows) if sb[i] == b and sd[i] == "science" and not ss[i].startswith("dsfsi/afriscience_mt")]
                rng.shuffle(idx)
                pools[("science", b)] = [(ts, i) for i in idx[:per[b] * 40]]
        seen = set(excl.get(iso, ()))
        for b in per:
            n_sci = int(round(per[b] * sci_share.get(b, 0.25)))
            wantn = {"science": n_sci, "general": per[b] - n_sci}
            for dom in ("science", "general"):
                got = 0
                cands = pools.get((dom, b), [])
                chunk = max(24, 3 * wantn[dom])
                for c0 in range(0, len(cands), chunk):
                    if got >= wantn[dom]:
                        break
                    part = cands[c0:c0 + chunk]
                    texts = [t.column("english")[i].as_py() for t, i in part]
                    hits = P.names(texts)                          # the protected-index check on the English side, one chunk at a time (a hit is dropped, never repaired)
                    for (t, i), en, h in zip(part, texts, hits):
                        if got >= wantn[dom]:
                            break
                        key = hashlib.md5(en.encode()).hexdigest()
                        sdn, lic = t.column("source_dataset")[i].as_py(), t.column("licence")[i].as_py()
                        if K.is_forbidden_source(sdn):
                            stats[iso]["forbidden_source"] += 1
                            continue
                        if any(x in (sdn or "").lower() for x in excl_src):
                            stats[iso]["excluded_source"] += 1
                            continue
                        if key in seen:
                            stats[iso]["dup"] += 1
                            continue
                        if h:
                            stats[iso]["protected_hit"] += 1
                            continue
                        sents = split_sentences(en)
                        if not sents or not sentences_are_stable(sents):
                            stats[iso]["unstable_split"] += 1
                            continue
                        pt = len(build_prompt(en, iso, tok))
                        if pt > a.max_prompt.get(b, 10 ** 9):
                            stats[iso]["too_long_prompt"] += 1
                            continue
                        seen.add(key)
                        got += 1
                        row = {"iso": iso, "bin": b, "domain": dom, "english": en, "sents": sents, "prompt_tokens": pt, "source_dataset": sdn, "licence": lic, "doc_key": key[:16],
                               "kind": a.kind, "prov_pool": t.column("prov_pool")[i].as_py(), "prov_file": t.column("prov_file")[i].as_py(), "prov_row": t.column("prov_row")[i].as_py(),
                               "prov_index": t.column("prov_index")[i].as_py()}
                        if dom == "science":
                            row["field"] = t.column("field")[i].as_py()
                        docs.append(row)
                        stats[iso][f"{b}/{dom}"] += 1
                if got < wantn[dom]:
                    stats[iso][f"short_{b}/{dom}"] = wantn[dom] - got
        K.log(f"{iso}: {sum(v for k, v in stats[iso].items() if '/' in k and not k.startswith('short'))} docs")
    P.close()
    K.write_jsonl(os.path.join(a.out, "docs.jsonl"), docs)
    uniq = {}
    for dd in docs:
        for s in dd["sents"]:
            for p in split_long_sentence(s, 700):
                uniq.setdefault((dd["iso"], p), None)
    reqs = []
    for k, (iso, p) in enumerate(sorted(uniq)):
        st = len(tok(p, add_special_tokens=False)["input_ids"])
        reqs.append({"rid": f"s{k}", "iso": iso, "text": p, "ids": build_prompt(p, iso, tok), "src_tokens": st, "max_new": min(int(3.5 * st) + 16, 1024), "capped": False})
    K.write_jsonl(os.path.join(a.out, "req_sent.jsonl"), reqs)
    summ = {"kind": a.kind, "docs": len(docs), "requests": len(reqs), "by_bin": dict(collections.Counter(x["bin"] for x in docs)), "by_domain": dict(collections.Counter(x["domain"] for x in docs)),
            "by_source": dict(collections.Counter(x["source_dataset"] for x in docs)), "libretexts_docs": sum(1 for x in docs if "libretexts" in x["source_dataset"].lower()),
            "sentences": sum(len(x["sents"]) for x in docs), "src_tokens_requests": sum(r["src_tokens"] for r in reqs), "index_version": P.version,
            "per_language": {k: dict(v) for k, v in stats.items()}}
    json.dump(summ, open(os.path.join(a.out, "select_stats.json"), "w"), indent=1)
    K.log(f"{a.kind}: docs {len(docs)}, unique sentence requests {len(reqs)}, sentences {summ['sentences']}, bins {summ['by_bin']}, libretexts docs {summ['libretexts_docs']}")


# ---------------------------------------------------------------------------------------------------------------------------- gen
def hf_generate(a):
    """CPU engine for the smoke test: greedy and 3 samples at T 0.7 per request, same output format as the vLLM script."""
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(2)
    m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    reqs = K.read_jsonl(os.path.join(a.work, "req_sent.jsonl"))
    if a.limit_requests:
        reqs = reqs[:a.limit_requests]
    os.makedirs(os.path.join(a.work, "gen"), exist_ok=True)
    for job in a.jobs.split(","):
        out_path = os.path.join(a.work, "gen", f"{job}.jsonl.gz")
        if os.path.exists(out_path):
            continue
        rows, t0 = [], time.time()
        for r in reqs:
            ids = torch.tensor([r["ids"]])
            mx = min(r["max_new"], a.hf_max_new)
            outs = []
            n = 1 if job == "greedy" else 3
            for j in range(n):
                if job != "greedy":
                    torch.manual_seed(int(hashlib.md5(f"{r['rid']}|{j}".encode()).hexdigest()[:8], 16) % 2 ** 31)
                with torch.no_grad():
                    g = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=mx, do_sample=(job != "greedy"), temperature=0.7 if job != "greedy" else None,
                                   top_p=1.0 if job != "greedy" else None, top_k=0 if job != "greedy" else None, eos_token_id=[106, 1], pad_token_id=0)
                new = g[0, ids.shape[1]:].tolist()
                fin = "stop" if (new and new[-1] in (106, 1)) else "length"
                outs.append({"ids": new, "finish": fin})
            rows.append({"rid": r["rid"], "outs": outs})
        with gzip.open(out_path + ".tmp", "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(out_path + ".tmp", out_path)
        K.log(f"[hf] {job}: {len(rows)} requests in {time.time() - t0:.0f}s (SMOKE engine, max_new {a.hf_max_new})")


def fake_generate(a):
    """SMOKE ONLY: stand-in generations so that score / pick / mix can be exercised when the CPU model cannot translate. For each request the candidates are real target-language
    sentences taken from --fake-bank (a jsonl with iso + target, e.g. replay_train.jsonl) whose length is closest to 1.15 x the source: not translations of the source."""
    K.gemma_pkg()
    from gemma_mt.eval.longdoc import split_sentences

    bank = collections.defaultdict(list)
    for r in K.read_jsonl(a.fake_bank):
        for s_ in split_sentences(r["target"]):
            if 15 <= len(s_) <= 700:
                bank[r["iso"]].append(s_)
    tok = K.tokenizer(a.model)
    reqs = K.read_jsonl(os.path.join(a.work, "req_sent.jsonl"))
    os.makedirs(os.path.join(a.work, "gen"), exist_ok=True)
    open(os.path.join(a.work, "gen", "FAKE_ENGINE_SMOKE"), "w").write("these generations are NOT translations (s4_longdoc.py gen --engine fake); score refuses them without --allow-fake\n")
    for job in a.jobs.split(","):
        out_path = os.path.join(a.work, "gen", f"{job}.jsonl.gz")
        if os.path.exists(out_path):
            continue
        n = 1 if job == "greedy" else 3
        with gzip.open(out_path + ".tmp", "wt", encoding="utf-8") as f:
            for r in reqs:
                pool = bank.get(r["iso"]) or [r["text"]]
                near = sorted(pool, key=lambda x: abs(len(x) - 1.15 * len(r["text"])))[:8]
                h = int(hashlib.md5(r["rid"].encode()).hexdigest()[:8], 16)
                outs = []
                for j in range(n):
                    t = near[(h + (0 if job == "greedy" else j + 1)) % len(near)]
                    outs.append({"ids": tok(f"{r['iso']} {t}", add_special_tokens=False)["input_ids"] + [106], "finish": "stop"})
                f.write(json.dumps({"rid": r["rid"], "outs": outs}) + "\n")
        os.replace(out_path + ".tmp", out_path)
        K.log(f"[fake] {job}: {len(reqs)} requests (SMOKE-FAKE-GEN: not translations)")


def cmd_gen(a):
    if a.engine == "hf":
        return hf_generate(a)
    if a.engine == "fake":
        return fake_generate(a)
    old = K.old_scripts()
    cmd = [a.python, os.path.join(old, "mix2_gen_vllm.py"), "--req", os.path.join(a.work, "req_sent.jsonl"), "--out", os.path.join(a.work, "gen"), "--model", a.model, "--jobs", a.jobs,
           "--gpu-util", str(a.gpu_util), "--max-seqs", str(a.max_seqs), "--chunk", str(a.chunk), "--max-model-len", str(a.max_model_len)]
    K.log(" ".join(cmd))
    for attempt in range(4):                                      # the old driver retried vLLM start failures 4 times
        if subprocess.call(cmd) == 0:
            return
        time.sleep(20)
    raise SystemExit("vLLM generation failed 4 times")


# ---------------------------------------------------------------------------------------------------------------------------- score
def cmd_score(a):
    if os.path.exists(os.path.join(a.work, "gen", "FAKE_ENGINE_SMOKE")) and not a.allow_fake:
        raise SystemExit("the generations in this work dir come from the FAKE engine (smoke only); pass --allow-fake to score them anyway")
    C = K.setup_judge(a.calibration or None, a.anchors or None, a.glotlid or None, a.scorer or None)
    import mix2_longdoc_score as S

    argv = ["--work", a.work, "--model", a.model] + (["--limit-docs", str(a.limit_docs)] if a.limit_docs else [])
    S.main(argv)


# ---------------------------------------------------------------------------------------------------------------------------- pick
def cmd_pick(a):
    d = KIND[a.kind]
    K.gemma_pkg()
    K.old_scripts()
    import mix2_longdoc_select as SEL

    c = d["conf_clips"]
    argv = ["--work", a.work, "--model", a.model, "--rel-q", str(a.rel_q if a.rel_q is not None else d["rel_q"]), f"--mean-r-clip={c[0]}", f"--p25-r-clip={c[1]}", f"--mean-conf-clip={c[2]}",
            f"--p25-conf-clip={c[3]}", "--budget-tokens", str(a.budget_tokens or d["budget"]), "--bin-share", a.bin_share or d["bin_share"], "--max-pack", str(K.SEQ_LEN),
            "--out-name", f"{a.kind}_picked", "--lang-budget-mult", a.lang_budget_mult if a.lang_budget_mult is not None else d["mult"]]
    SEL.main(argv)
    docs = {(r["iso"], r["doc_key"]): r for r in K.read_jsonl(os.path.join(a.work, "docs.jsonl"))}
    out = []
    for r in K.read_jsonl(os.path.join(a.work, f"{a.kind}_picked.jsonl")):
        dd = docs.get((r["iso"], r["doc_key"]), {})
        prov = K.provenance(a.kind, source_dataset=r.get("source_dataset", ""), licence=r.get("licence", ""), prov_repo=dd.get("prov_file", ""),
                            prov_ref=f"pool={dd.get('prov_pool', '')};row={dd.get('prov_row', '')};doc_key={r['doc_key']}", prov_index=dd.get("prov_index", ""),
                            generator=f"{a.model_label or a.model} best-of-4 (greedy + 3 x T0.7), sentence-wise pick by the QE ensemble / chrF++ medoid")
        out.append({**r, **prov})
    n = K.write_jsonl(os.path.join(a.work, f"{a.kind}_selected.jsonl"), out)
    tok = sum(r["n_tokens"] for r in out)
    ref = K.A2W_TOKENS["longdoc" if a.kind == "longdoc" else "anchor"]
    K.log(f"{a.kind}: {n} documents, {tok:,} tokens (A2W reference {ref:,}), libretexts rows {sum(1 for r in out if r['licence_unresolved'] and 'libretexts' in r['source_dataset'].lower())}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--kind", choices=tuple(KIND), required=True)
    s.add_argument("--pool-v3", required=True)
    s.add_argument("--pool-sci", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--model", required=True, help="the NEW RL model (its tokenizer builds the prompts)")
    s.add_argument("--index", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--per-lang", default="")
    s.add_argument("--sci-share", default="")
    s.add_argument("--languages", default="")
    s.add_argument("--exclude", default="", help="docs.jsonl files whose English texts must not be selected again (anchors vs long documents)")
    s.add_argument("--exclude-source", default="", help="e.g. libretexts (default: INCLUDED)")
    s.add_argument("--seed", type=int, default=20261028)
    s.add_argument("--nproc", type=int, default=2)
    s.add_argument("--max-prompt-b1024", type=int, default=1100)
    s.set_defaults(fn=cmd_select)
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--engine", choices=("vllm", "hf", "fake"), default="vllm")
    g.add_argument("--fake-bank", default="", help="fake engine (smoke): jsonl with iso + target")
    g.add_argument("--jobs", default="greedy,t07x3")
    g.add_argument("--python", default=sys.executable, help="python of the vLLM environment")
    g.add_argument("--gpu-util", type=float, default=0.85)
    g.add_argument("--max-seqs", type=int, default=512)
    g.add_argument("--chunk", type=int, default=20000)
    g.add_argument("--max-model-len", type=int, default=4096)
    g.add_argument("--limit-requests", type=int, default=0, help="hf engine: first N requests only (smoke)")
    g.add_argument("--hf-max-new", type=int, default=48, help="hf engine: token cap (smoke)")
    g.set_defaults(fn=cmd_gen)
    c = sub.add_parser("score")
    c.add_argument("--work", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--calibration", default="")
    c.add_argument("--anchors", default="")
    c.add_argument("--glotlid", default="")
    c.add_argument("--scorer", default="")
    c.add_argument("--limit-docs", type=int, default=0)
    c.add_argument("--allow-fake", action="store_true", help="smoke only: score generations of the fake engine")
    c.set_defaults(fn=cmd_score)
    p = sub.add_parser("pick")
    p.add_argument("--work", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-label", default="")
    p.add_argument("--kind", choices=tuple(KIND), required=True)
    p.add_argument("--rel-q", type=float, default=None)
    p.add_argument("--budget-tokens", type=int, default=0)
    p.add_argument("--bin-share", default="")
    p.add_argument("--lang-budget-mult", default=None)
    p.set_defaults(fn=cmd_pick)
    a = ap.parse_args(argv)
    if a.cmd == "select":
        a.max_prompt = {"b1024": a.max_prompt_b1024}
    a.fn(a)


if __name__ == "__main__":
    main()
