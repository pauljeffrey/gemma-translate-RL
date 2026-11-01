#!/usr/bin/env python3
"""AfriLongDoc check of the OLD intermediate RL checkpoints (agent C2, 2026-10-28; lean-schedule E.4, owner: "we will carry out the $3 stage").

Question: did the b512 / b1024 / b2048 stages of the old RL run (phase 2c) change whole-document translation? No intermediate checkpoint was ever evaluated on AfriLongDoc
(paper 6.4). The old 2c run pushed stage branches to the private Hub repos below (VERIFIED by READ-ONLY Hub API calls on 2026-10-28, each with a 1.07 GB model.safetensors):

  name         repo @ revision                                        training step        role
  sft_final    <HF_ORG>/gemma3-270m-translate @ step-27638        27,638 (SFT)         baseline
  rl2b_b64     <HF_ORG>/gemma-3-270m-translate-rl2b @ stage-b64   488 (phase 2b)       START of phase 2c (end of the b64 stage)
  rl2c_b256    <HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c @ stage-b256   391 (2c)          end of the b256 stage
  rl2c_b512    ... @ stage-b512                                        837 (2c)            end of the b512 stage
  rl2c_b1024   ... @ stage-b1024                                       986 (2c)            end of the b1024 stage
  rl2c_b2048   ... @ stage-b2048                                       1,284 (2c)          end of the b2048 stage
  rl2c_mixed   ... @ stage-mixed (= latest)                            1,433 (2c)          RL-final (the headline checkpoint of the paper)
  (<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c `main` holds NO weights, only README + an empty `latest` file; use the branches above.)

Evaluation = the paper's section 6.4 protocol for ONE decoding setting: AfriLongDoc CORE documents of the bins b256 / b512 / b1024 / b2048 (90 / 100 / 89 / 63 English documents), 8
languages (hau yor swh amh zul wol kea sag = the B8 set of the A2 evaluation), WHOLE-DOCUMENT GREEDY decoding (prompt = gemma_mt.format.build_prompt, max_new = min(3.5 x source
tokens, 10,240 - prompt), stop ids 106 / 1, vLLM bf16), metrics per language and bin, macro over the 8 languages:
  finished rate  the generation ended with its own end token inside the budget
  loop rate      the output contains a repetition loop that the reference does not (gemma_mt.eval.longdoc.repetition_loop)
  chrF++         document-level corpus chrF++ (every document one segment, sufficient statistics summed per language) against the NLLB-200 3.3B sentence-by-sentence PSEUDO-references
                 of <HF_ORG>/AfriLongDoc (EVALUATION use of NLLB output; nothing here is training data); BLEU and the median length ratio are reported next to it.
95% CIs by bootstrap over DOCUMENTS (the same resample for every checkpoint, so the stage deltas are paired).

  python s4_afrilongdoc_stagecheck.py fetch  --out W [--ckpts all]          # AfriLongDoc core docs + NLLB refs of the 8 languages + the checkpoints (token from HF_TOKEN; never printed)
  python s4_afrilongdoc_stagecheck.py prep   --out W
  python s4_afrilongdoc_stagecheck.py gen    --out W --engine vllm --python /path/to/vllm/python        # one subprocess (and one vLLM engine) per checkpoint
  python s4_afrilongdoc_stagecheck.py score  --out W
  python s4_afrilongdoc_stagecheck.py report --out W                                                    # W/stagecheck_report.md + W/stagecheck_results.json

Cost (ESTIMATED; counts MEASURED by `prep` on the real data: 2,560 items = 342 documents x 8 languages (invalid NLLB references skipped), 2.09 M source tokens, per bin 687 / 719 / 676 / 478
items; ~4.2 M generated tokens per checkpoint at 2.0 x the source tokens, 5.7 M at 2.7 x (loops), hard cap 7.3 M): per checkpoint 3.5-4.7 min at 20 k tok/s (7-9.5 min at 10 k) + ~1 min
vLLM start-up; 7 checkpoints = 0.5 h central [0.4-1.2]; checkpoint downloads 7 x 1.07 GB (15 s each measured on box 2); CPU scoring ~10 min; setup/margin 0.2 h. On an RTX PRO 6000 at
USD 1.19/h: ~0.9 h [0.7-2.0] = USD 1.1 [0.8-2.4] (the lean-schedule guess in E.4 was 2-2.5 h / USD 3). Only `gen --engine vllm` needs a GPU; the CPU smoke uses `--engine hf`.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CKPTS = collections.OrderedDict([
    ("sft_final", dict(repo=(os.environ['HF_ORG'] + "/gemma3-270m-translate"), rev="step-27638", step=27638, phase="SFT", role="baseline (SFT-final)")),
    ("rl2b_b64", dict(repo=(os.environ['HF_ORG'] + "/gemma-3-270m-translate-rl2b"), rev="stage-b64", step=488, phase="2b", role="start of phase 2c = end of the b64 stage")),
    ("rl2c_b256", dict(repo=(os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c"), rev="stage-b256", step=391, phase="2c", role="end of the b256 stage")),
    ("rl2c_b512", dict(repo=(os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c"), rev="stage-b512", step=837, phase="2c", role="end of the b512 stage")),
    ("rl2c_b1024", dict(repo=(os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c"), rev="stage-b1024", step=986, phase="2c", role="end of the b1024 stage")),
    ("rl2c_b2048", dict(repo=(os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c"), rev="stage-b2048", step=1284, phase="2c", role="end of the b2048 stage")),
    ("rl2c_mixed", dict(repo=(os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c"), rev="stage-mixed", step=1433, phase="2c", role="end of the mixed stage = RL-final")),
])
STAGE_BIN = {"rl2c_b256": 256, "rl2c_b512": 512, "rl2c_b1024": 1024, "rl2c_b2048": 2048}      # the bin each stage trained on (mixed trains on all)
B8 = "hau yor swh amh zul wol kea sag".split()
BINS = [256, 512, 1024, 2048]
CTX = 10240
STOP = [106, 1]
LONGDOC_REPO = (os.environ['HF_ORG'] + "/AfriLongDoc")
MODEL_PATTERNS = ["model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
                  "added_tokens.json", "chat_template.jinja", "training_state.json"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rd(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def wr(p, rows):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(p + ".tmp", p)


def pkg():
    for p in (os.environ.get("GEMMA_MT_PKG"), "/workspace/work/gemma_mt_pkg", os.path.abspath(os.path.join(HERE, "..", "..", "gemma_mt"))):
        if p and os.path.isdir(os.path.join(p, "gemma_mt")):
            if p not in sys.path:
                sys.path.insert(0, p)
            return p
    raise SystemExit("gemma_mt package not found (set GEMMA_MT_PKG)")


def ckpt_dir(W, name):
    """models/<name> (a downloaded checkpoint) or, for the CPU smoke, a path given with --model-dir name=path."""
    return os.path.join(W, "models", name)


def selected(a):
    names = list(CKPTS) if a.ckpts in ("all", "") else a.ckpts.split(",")
    for n in names:
        if n not in CKPTS and not (a.model_dir and any(x.startswith(n + "=") for x in a.model_dir)):
            raise SystemExit(f"unknown checkpoint {n}; known: {list(CKPTS)}")
    return names


def model_path(a, name):
    for x in a.model_dir or []:
        k, v = x.split("=", 1)
        if k == name:
            return v
    return ckpt_dir(a.out, name)


# ---------------------------------------------------------------------------------------------------------------------------------- fetch
def cmd_fetch(a):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    tok = os.environ.get("HF_TOKEN")
    if not tok:
        sys.exit("HF_TOKEN is not set")
    W = a.out
    os.makedirs(W, exist_ok=True)
    if not a.no_data:
        for f in ("core/docs.jsonl", "pseudo_refs/pseudo_refs_nllb.tar.zst"):
            p = hf_hub_download(LONGDOC_REPO, f, repo_type="dataset", token=tok, local_dir=os.path.join(W, "ald"))
            log(f"got {f} {os.path.getsize(p)}")
        nl = os.path.join(W, "ald", "nllb")
        os.makedirs(nl, exist_ok=True)
        members = " ".join(f'"./nllb-200-3.3b/{i}.refs.jsonl"' for i in B8)
        subprocess.check_call(f'zstd -dc "{os.path.join(W, "ald", "pseudo_refs", "pseudo_refs_nllb.tar.zst")}" | tar -xf - -C "{nl}" --wildcards {members}', shell=True)
        log(f"NLLB reference files for {B8}: {sorted(os.listdir(os.path.join(nl, 'nllb-200-3.3b')))}")
    if not a.no_models:
        api = HfApi(token=tok)
        for n in selected(a):
            c = CKPTS[n]
            d = ckpt_dir(W, n)
            if os.path.exists(os.path.join(d, "model.safetensors")):
                log(f"{n}: present")
                continue
            info = api.model_info(c["repo"], revision=c["rev"])
            snapshot_download(c["repo"], revision=c["rev"], token=tok, local_dir=d, allow_patterns=MODEL_PATTERNS)
            json.dump({"name": n, **c, "commit": info.sha}, open(os.path.join(d, "S4_CHECKPOINT.json"), "w"))
            log(f"{n}: {c['repo']}@{c['rev']} commit {info.sha[:10]} downloaded")


# ---------------------------------------------------------------------------------------------------------------------------------- prep
def cmd_prep(a):
    pkg()
    from transformers import AutoTokenizer

    from gemma_mt.data.prepare_sft import tokenizer_fingerprint
    from gemma_mt.eval.longdoc import join_sentences
    from gemma_mt.format import build_prompt

    names = selected(a)
    W = a.out
    toks = {n: AutoTokenizer.from_pretrained(model_path(a, n)) for n in names}
    fps = {n: tokenizer_fingerprint(t) for n, t in toks.items()}
    if len(set(fps.values())) != 1:
        raise SystemExit(f"the checkpoints do not share one tokenizer: {fps}")
    tok = toks[names[0]]
    tok.save_pretrained(os.path.join(W, "tok"))                 # scoring needs only the tokenizer (the checkpoints may be deleted after their generation)
    bins = [int(x) for x in a.bins.split(",")]
    langs = a.langs.split(",")
    docs = [d for d in rd(os.path.join(W, "ald", "core", "docs.jsonl")) if d["bin"] in bins]
    if a.limit_docs:
        by = collections.defaultdict(list)
        for d in docs:
            by[d["bin"]].append(d)
        docs = [d for b in bins for d in sorted(by[b], key=lambda x: x["doc_id"])[:a.limit_docs]]
    items, refs = [], []
    for iso in langs:
        rp = os.path.join(W, "ald", "nllb", "nllb-200-3.3b", f"{iso}.refs.jsonl")
        rmap = {r["doc_id"]: r for r in rd(rp)}
        for d in docs:
            r = rmap.get(d["doc_id"])
            if r is None or not r.get("valid", True):
                continue
            iid = f"core|{d['doc_id']}|{iso}"
            n_src = len(tok(d["text"], add_special_tokens=False)["input_ids"])
            ids = build_prompt(d["text"], iso, tok)
            items.append({"iid": iid, "iso": iso, "bin": d["bin"], "doc_id": d["doc_id"], "src_tokens": n_src, "ids": ids, "max_new": max(1, min(int(3.5 * n_src), CTX - len(ids)))})
            refs.append({"iid": iid, "ref": join_sentences(r["ref_sents"], d["para_starts"]), "flags": r.get("flags", [])})
    wr(os.path.join(W, "work", "items.jsonl"), items)
    wr(os.path.join(W, "work", "refs_nllb.jsonl"), refs)
    log(f"{len(items)} items ({len(docs)} documents x {len(langs)} languages, invalid NLLB references skipped), bins {dict(collections.Counter(i['bin'] for i in items))}, "
        f"tokenizer fingerprint {next(iter(fps.values()))[:12]}")


# ---------------------------------------------------------------------------------------------------------------------------------- gen
def hf_gen_one(a, name):
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(2)
    m = AutoModelForCausalLM.from_pretrained(model_path(a, name), torch_dtype=torch.float32).eval()
    rows = []
    for r in rd(os.path.join(a.out, "work", "items.jsonl")):
        ids = torch.tensor([r["ids"]])
        with torch.no_grad():
            g = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=min(r["max_new"], a.hf_max_new), do_sample=False, eos_token_id=STOP, pad_token_id=0)
        new = g[0, ids.shape[1]:].tolist()
        rows.append({"iid": r["iid"], "ids": new, "finish": "stop" if (new and new[-1] in STOP) else "length"})
    return rows


def vllm_gen_one(a, name):
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    from vllm import LLM, SamplingParams

    items = rd(os.path.join(a.out, "work", "items.jsonl"))
    items.sort(key=lambda r: -(len(r["ids"]) + r["max_new"]))
    t0 = time.time()
    llm = LLM(model=model_path(a, name), dtype="bfloat16", max_model_len=CTX, max_num_seqs=a.max_seqs, gpu_memory_utilization=a.gpu_util, enable_prefix_caching=True, seed=0,
              skip_tokenizer_init=True, generation_config="vllm", disable_log_stats=True)
    log(f"{name}: engine up in {time.time() - t0:.0f}s")
    rows, ntok, t1 = [], 0, time.time()
    for s in range(0, len(items), a.chunk):
        part = items[s:s + a.chunk]
        params = [SamplingParams(max_tokens=r["max_new"], temperature=0.0, stop_token_ids=STOP, detokenize=False, skip_special_tokens=False) for r in part]
        outs = llm.generate([{"prompt_token_ids": r["ids"]} for r in part], params, use_tqdm=False)
        for r, o in zip(part, outs):
            c = o.outputs[0]
            rows.append({"iid": r["iid"], "ids": [int(t) for t in c.token_ids], "finish": str(c.finish_reason)})
            ntok += len(c.token_ids)
        log(f"{name}: {min(s + a.chunk, len(items))}/{len(items)} documents, {ntok} tokens, {time.time() - t1:.0f}s")
    json.dump({"ckpt": name, "documents": len(items), "generated_tokens": ntok, "wall_s": time.time() - t1, "tok_per_s": ntok / max(time.time() - t1, 1e-9)},
              open(os.path.join(a.out, "gen", f"{name}.stats.json"), "w"))
    return rows


def cmd_gen_one(a):
    name = a.ckpt
    out = os.path.join(a.out, "gen", f"{name}.jsonl.gz")
    if os.path.exists(out):
        log(f"{name}: done")
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = hf_gen_one(a, name) if a.engine == "hf" else vllm_gen_one(a, name)
    with gzip.open(out + ".tmp", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(out + ".tmp", out)
    log(f"{name}: wrote {len(rows)} generations")


def cmd_gen(a):
    for n in selected(a):
        if os.path.exists(os.path.join(a.out, "gen", f"{n}.jsonl.gz")):
            log(f"{n}: done")
            continue
        cmd = [a.python or sys.executable, os.path.abspath(__file__), "gen-one", "--out", a.out, "--ckpt", n, "--engine", a.engine, "--gpu-util", str(a.gpu_util),
               "--max-seqs", str(a.max_seqs), "--chunk", str(a.chunk), "--hf-max-new", str(a.hf_max_new)] + [x for md in (a.model_dir or []) for x in ("--model-dir", md)]
        log(" ".join(cmd))
        for attempt in range(2):
            if subprocess.call(cmd) == 0:
                break
        else:
            raise SystemExit(f"generation failed for {n}")
        if a.delete_after and not (a.model_dir or []):
            import shutil

            shutil.rmtree(ckpt_dir(a.out, n), ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------------------------- score
def cmd_score(a):
    pkg()
    from transformers import AutoTokenizer

    from gemma_mt.eval.longdoc import doc_stat_vectors, norm_text, repetition_loop
    from gemma_mt.rl.text import decode_completion, translation_from

    W = a.out
    items = {r["iid"]: r for r in rd(os.path.join(W, "work", "items.jsonl"))}
    refs = {r["iid"]: r for r in rd(os.path.join(W, "work", "refs_nllb.jsonl"))}
    ref_loop = {k: bool(repetition_loop(v["ref"])["loop"]) for k, v in refs.items()}
    out = []
    for name in selected(a):
        gp = os.path.join(W, "gen", f"{name}.jsonl.gz")
        if not os.path.exists(gp):
            log(f"{name}: no generations, skipped")
            continue
        tok = AutoTokenizer.from_pretrained(os.path.join(W, "tok"))
        hyps, meta = [], []
        with gzip.open(gp, "rt", encoding="utf-8") as f:
            for l in f:
                g = json.loads(l)
                it = items.get(g["iid"])
                if it is None:
                    continue
                d = decode_completion(tok, g["ids"], STOP)
                t = translation_from(d, it["iso"])
                hyps.append(t.text)
                meta.append({"iid": g["iid"], "iso": it["iso"], "bin": it["bin"], "doc_id": it["doc_id"], "finished": bool(d.finished), "status": t.status, "new_tokens": len(g["ids"]),
                             "loop": bool(repetition_loop(t.text)["loop"]) and not ref_loop[g["iid"]]})
        cs, bs = doc_stat_vectors(hyps, [refs[m["iid"]]["ref"] for m in meta])
        for m, c, b, h in zip(meta, cs, bs, hyps):
            m["chrf_stats"], m["bleu_stats"] = c.tolist(), b.tolist()
            m["len_ratio"] = len(norm_text(h)) / max(len(norm_text(refs[m["iid"]]["ref"])), 1)
        wr(os.path.join(W, "scores", f"{name}.jsonl"), meta)
        log(f"{name}: scored {len(meta)} documents")


# ---------------------------------------------------------------------------------------------------------------------------------- report
def chrf_vec(S):
    """sacrebleu chrF++ (beta 2, char 6, word 2) from summed statistics (..., 24)."""
    import numpy as np

    S = np.asarray(S, dtype=np.float64)
    nh, nr, nm = S[..., 0::3], S[..., 1::3], S[..., 2::3]
    ok = (nh > 0) & (nr > 0)
    prec = np.where(nh > 0, nm / np.maximum(nh, 1), 0.0)
    rec = np.where(nr > 0, nm / np.maximum(nr, 1), 0.0)
    eff = ok.sum(-1)
    ap = (prec * ok).sum(-1) / np.maximum(eff, 1)
    ar = (rec * ok).sum(-1) / np.maximum(eff, 1)
    den = 4 * ap + ar
    return np.where(den > 0, 100 * 5 * ap * ar / np.where(den > 0, den, 1), 0.0)


def cmd_report(a):
    import warnings

    import numpy as np

    warnings.simplefilter("ignore", RuntimeWarning)

    W = a.out
    names = [n for n in selected(a) if os.path.exists(os.path.join(W, "scores", f"{n}.jsonl"))]
    data = {n: rd(os.path.join(W, "scores", f"{n}.jsonl")) for n in names}
    langs = sorted({m["iso"] for n in names for m in data[n]})
    res = {"checkpoints": {n: CKPTS.get(n, {}) for n in names}, "bins": {}, "settings": {"decoding": "whole-document greedy", "languages": langs, "reference": "NLLB-200 3.3B pseudo-reference",
                                                                                           "bootstrap_resamples": a.boot, "seed": a.seed}}
    rng = np.random.default_rng(a.seed)
    for b in BINS:
        docs = sorted({m["doc_id"] for n in names for m in data[n] if m["bin"] == b})
        if not docs:
            continue
        di = {d: i for i, d in enumerate(docs)}
        arr = {}
        for n in names:
            C = np.zeros((len(langs), len(docs), 24))
            B = np.zeros((len(langs), len(docs), 10))
            F = np.full((len(langs), len(docs)), np.nan)
            L = np.full((len(langs), len(docs)), np.nan)
            R = np.full((len(langs), len(docs)), np.nan)
            M = np.zeros((len(langs), len(docs)), bool)
            for m in data[n]:
                if m["bin"] != b:
                    continue
                i, j = langs.index(m["iso"]), di[m["doc_id"]]
                C[i, j], B[i, j] = m["chrf_stats"], m["bleu_stats"]
                F[i, j], L[i, j], R[i, j], M[i, j] = float(m["finished"]), float(m["loop"]), m["len_ratio"], True
            arr[n] = (C, B, F, L, R, M)
        M0 = np.logical_and.reduce([arr[n][5] for n in names])           # documents scored for every checkpoint and language
        draws = [rng.integers(0, len(docs), len(docs)) for _ in range(a.boot)]

        def macro_chrf(C, M, idx):
            Cm = C * M[..., None]
            s = Cm[:, idx, :].sum(1)
            return float(chrf_vec(s).mean())

        def macro_rate(X, M, idx):
            Xm = np.where(M, X, np.nan)[:, idx]
            cnt, sm = (~np.isnan(Xm)).sum(1), np.nansum(Xm, 1)
            ok = cnt > 0                                           # a language without a scored document in this bin does not enter the macro
            return float((sm[ok] / cnt[ok]).mean()) if ok.any() else float("nan")

        res["bins"][str(b)] = {"documents": len(docs), "cells": int(M0.sum()), "per_checkpoint": {}, "stage_delta": {}}
        point, boots = {}, {}
        for n in names:
            C, B_, F, L, R, _ = arr[n]
            allidx = np.arange(len(docs))
            point[n] = {"chrf": macro_chrf(C, M0, allidx), "finished": macro_rate(F, M0, allidx), "loop": macro_rate(L, M0, allidx),
                        "len_ratio_median": float(np.nanmedian(np.where(M0, R, np.nan))), "per_language_chrf": {l: float(chrf_vec((C[i] * M0[i][:, None]).sum(0))) for i, l in enumerate(langs)},
                        "per_language_finished": {l: float(np.nanmean(np.where(M0[i], F[i], np.nan))) for i, l in enumerate(langs)},
                        "per_language_loop": {l: float(np.nanmean(np.where(M0[i], L[i], np.nan))) for i, l in enumerate(langs)}}
            boots[n] = {"chrf": np.array([macro_chrf(C, M0, ix) for ix in draws]), "finished": np.array([macro_rate(F, M0, ix) for ix in draws]),
                        "loop": np.array([macro_rate(L, M0, ix) for ix in draws])}
            ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            point[n].update({"chrf_ci": ci(boots[n]["chrf"]), "finished_ci": ci(boots[n]["finished"]), "loop_ci": ci(boots[n]["loop"])})
        res["bins"][str(b)]["per_checkpoint"] = point
        prev = None
        for n in names:
            if prev is not None:
                d = {k: (boots[n][k] - boots[prev][k]) for k in ("chrf", "finished", "loop")}
                res["bins"][str(b)]["stage_delta"][f"{prev}->{n}"] = {k: {"delta": point[n][k] - point[prev][k], "ci": [float(np.percentile(d[k], 2.5)), float(np.percentile(d[k], 97.5))]} for k in d}
            prev = n
    # ---- verdicts (rule fixed here, before any result exists): a stage MATTERED for its own bin when the paired chrF++ delta CI excludes 0 and is >= min_chrf, or the finished-rate delta CI
    #      excludes 0 and is >= min_fin; the lean schedule then raises that bin to its high cap (b512 200 steps); if NO long-bin stage mattered, b1024 drops to 40 steps.
    verdict = {}
    for st, b in STAGE_BIN.items():
        if st not in names:
            continue
        i = names.index(st)
        if i == 0:
            continue
        dd = res["bins"].get(str(b), {}).get("stage_delta", {}).get(f"{names[i - 1]}->{st}")
        if dd:
            c, f = dd["chrf"], dd["finished"]
            verdict[st] = {"bin": b, "chrf_delta": c["delta"], "chrf_ci": c["ci"], "finished_delta": f["delta"], "finished_ci": f["ci"],
                           "mattered": bool((c["ci"][0] > 0 and c["delta"] >= a.min_chrf) or (f["ci"][0] > 0 and f["delta"] >= a.min_fin))}
    res["verdict_rule"] = f"stage mattered for its own bin if paired chrF++ delta >= {a.min_chrf} with CI > 0, or finished-rate delta >= {a.min_fin} with CI > 0"
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(W, "stagecheck_results.json"), "w"), indent=1)
    L = ["# AfriLongDoc stage check of the old RL checkpoints" + (" (SMOKE: base model, tiny counts, not a result)" if a.smoke else ""), "",
         f"Whole-document greedy, NLLB-200 3.3B pseudo-references, macro over {len(langs)} languages ({', '.join(langs)}), bootstrap over documents ({a.boot} resamples). ",
         "", "## chrF++ / finished / loop per checkpoint and bin", "", "| bin | checkpoint | step | chrF++ [95% CI] | finished % | loop % | median length ratio |", "|---|---|---:|---|---:|---:|---:|"]
    for b in BINS:
        pb = res["bins"].get(str(b))
        if not pb:
            continue
        for n in names:
            p = pb["per_checkpoint"][n]
            L.append(f"| b{b} | {n} | {CKPTS.get(n, {}).get('step', '')} | {p['chrf']:.1f} [{p['chrf_ci'][0]:.1f}, {p['chrf_ci'][1]:.1f}] | {100 * p['finished']:.1f} | {100 * p['loop']:.1f} | {p['len_ratio_median']:.2f} |")
    L += ["", "## Stage deltas (paired bootstrap)", "", "| bin | step | d chrF++ [CI] | d finished (pp) [CI] | d loop (pp) [CI] |", "|---|---|---|---|---|"]
    for b in BINS:
        pb = res["bins"].get(str(b))
        if not pb:
            continue
        for k, v in pb["stage_delta"].items():
            L.append(f"| b{b} | {k} | {v['chrf']['delta']:+.2f} [{v['chrf']['ci'][0]:+.2f}, {v['chrf']['ci'][1]:+.2f}] | {100 * v['finished']['delta']:+.1f} [{100 * v['finished']['ci'][0]:+.1f}, "
                     f"{100 * v['finished']['ci'][1]:+.1f}] | {100 * v['loop']['delta']:+.1f} [{100 * v['loop']['ci'][0]:+.1f}, {100 * v['loop']['ci'][1]:+.1f}] |")
    L += ["", f"## Verdict rule (fixed before the run): {res['verdict_rule']}", "", "| stage | own bin | d chrF++ | d finished | mattered |", "|---|---|---:|---:|---|"]
    for st, v in verdict.items():
        L.append(f"| {st} | b{v['bin']} | {v['chrf_delta']:+.2f} {v['chrf_ci'][0]:+.2f}..{v['chrf_ci'][1]:+.2f} | {100 * v['finished_delta']:+.1f} pp | {'YES' if v['mattered'] else 'no'} |")
    open(os.path.join(W, "stagecheck_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", required=True)
        p.add_argument("--ckpts", default="all", help="comma separated names (default: all of " + ",".join(CKPTS) + ")")
        p.add_argument("--model-dir", action="append", default=[], help="name=path: use a local model directory for a checkpoint name (CPU smoke)")

    for n, fn in (("fetch", cmd_fetch), ("prep", cmd_prep), ("gen", cmd_gen), ("gen-one", cmd_gen_one), ("score", cmd_score), ("report", cmd_report)):
        p = sub.add_parser(n)
        common(p)
        p.set_defaults(fn=fn)
        if n == "fetch":
            p.add_argument("--no-data", action="store_true")
            p.add_argument("--no-models", action="store_true")
        if n == "prep":
            p.add_argument("--bins", default=",".join(map(str, BINS)))
            p.add_argument("--langs", default=",".join(B8))
            p.add_argument("--limit-docs", type=int, default=0, help="smoke: first N documents per bin")
        if n in ("gen", "gen-one"):
            p.add_argument("--engine", choices=("vllm", "hf"), default="vllm")
            p.add_argument("--python", default="", help="python of the vLLM environment")
            p.add_argument("--gpu-util", type=float, default=0.85)
            p.add_argument("--max-seqs", type=int, default=512)
            p.add_argument("--chunk", type=int, default=2500)
            p.add_argument("--hf-max-new", type=int, default=40)
            p.add_argument("--delete-after", action="store_true", help="delete the downloaded checkpoint after its generation (disk)")
            if n == "gen-one":
                p.add_argument("--ckpt", required=True)
        if n == "report":
            p.add_argument("--boot", type=int, default=400)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--min-chrf", type=float, default=1.0)
            p.add_argument("--min-fin", type=float, default=0.05)
            p.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
