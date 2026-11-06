#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-editor, step 2: machine-translation DRAFTS of the English side of the chosen pairs, from several systems of DIFFERENT quality (the editor must not learn to copy a good draft).

One invocation = ONE draft system (one model/engine, or one rule-based generator); it writes DIR/drafts_<name>.jsonl. `--summarize` merges them and writes the quality tables.

    # model systems (GPU box; venv_vllm; GEMMA_MT_TAG_MODE does not matter: the prompt has no tag and the decoder accepts text tags and <unusedK> tags)
    python scripts/post_editor_make_drafts.py --pairs OUT/pairs.jsonl --out DIR --system sft_best   --model /workspace/clean_rerun/models/g15_385m_sft_final --decoding best   # beam-4, MBR-16 for --mbr-languages
    python scripts/post_editor_make_drafts.py ... --system sft_greedy --model ... --decoding greedy
    python scripts/post_editor_make_drafts.py ... --system sft_t07    --model ... --decoding sample --temperature 0.7 --top-p 0.95
    python scripts/post_editor_make_drafts.py ... --system sft_t10    --model ... --decoding sample --temperature 1.0 --top-p 1.0
    python scripts/post_editor_make_drafts.py ... --system a0_greedy  --model /workspace/eval_385m/model_a0 --decoding greedy                       # weaker 270M checkpoint
    python scripts/post_editor_make_drafts.py ... --system early_greedy --model <g1b-s5 dir> --decoding greedy --exposure-map OUT/exposure_map.csv --unseen-after-shard 5
    # model-free systems
    python scripts/post_editor_make_drafts.py ... --system corrupt --type corrupt      # eight corruptions of the HUMAN reference (drop / insert / swap / truncate / repeat / number-entity / English clause / neighbour sentence)
    python scripts/post_editor_make_drafts.py ... --system identity --type identity    # draft == reference (the leave-alone control)
    python scripts/post_editor_make_drafts.py ... --system near --type near            # draft = reference with 1-2 tiny edits
    python scripts/post_editor_make_drafts.py --summarize --pairs OUT/pairs.jsonl --out DIR [--val-system sft_best]

Decodings: greedy | beam (width 4, length penalty 1.0, no fallback) | sample (n=1) | mbr (16 samples T 0.7 top-p 0.95 + greedy as 17th candidate, chrF++ utility; the decode study's recipe) |
best (= beam, but mbr for the languages in --mbr-languages: default orm ewe lug tir, where the decode study found MBR clearly ahead of beam).
`--exposure-map CSV(pair_id,first_shard)` + `--unseen-after-shard K`: only pairs whose first appearance in the SFT stream is AFTER shard K are drafted (the checkpoint saved after shard K has never seen them);
rows without an entry are skipped. `--backend hf --model tiny-random --fake-tokenizer` runs everything on CPU with a tiny random model (smoke tests).
"""
import argparse, csv, json, os, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
MBR_LANGS = ["orm", "ewe", "lug", "tir"]


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_pairs(path, splits=None, limit=0, isos=None):
    rows = [r for r in read_jsonl(path) if (not splits or r["split"] in splits) and (not isos or r["iso"] in isos)]
    return rows[:limit] if limit else rows


def make_tokenizer(a):
    if a.fake_tokenizer:
        from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
        return FakeGemmaTokenizer()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(a.tokenizer)


def make_backend(a, tok):
    from gemma_mt.post_edit import make_backend as mb
    return mb(a.backend, a.model, tok, a.dtype, a.gpu_util, a.max_model_len, a.batch_tokens, a.seed)


def model_drafts(a, rows, tok, be, spec):
    """spec = {decoding, temperature, top_p}; tok / be are shared by all systems of the plan (one engine load per model)."""
    from gemma_mt import format as fmt
    from gemma_mt.post_edit import decode_gen, mbr_select, sentence_chrf
    a = argparse.Namespace(**{**vars(a), **spec})
    prompts = [list(fmt.build_prompt(r["english"], r["iso"], tok)) for r in rows]
    dec = lambda g, r: decode_gen(tok, g, r["iso"])
    out = [None] * len(rows)

    def run(idx, mode):
        P = [prompts[i] for i in idx]
        if not P: return
        if mode == "greedy":
            gs = be.greedy(P, a.max_new)
        elif mode == "beam":
            gs = be.beam(P, a.beam_width, a.max_new)
        elif mode == "sample":
            gs = [x[0] for x in be.sample(P, 1, a.temperature, a.top_p, a.max_new, a.seed)]
        elif mode == "mbr":
            greedy = be.greedy(P, a.max_new); samp = be.sample(P, a.k, a.temperature if a.temperature else 0.7, a.top_p if a.top_p < 1 else 0.95, a.max_new, a.seed)
            allc = [[g0] + list(ss) for g0, ss in zip(greedy, samp)]
            texts = [[dec(c, rows[i])[0] for c in cs] for i, cs in zip(idx, allc)]
            if a.mbr_workers > 1:
                from multiprocessing import Pool
                with Pool(a.mbr_workers) as pool: pick = pool.map(mbr_select, texts, chunksize=16)
            else:
                pick = [mbr_select(t) for t in texts]
            gs = [cs[j] for cs, j in zip(allc, pick)]
        for i, g in zip(idx, gs):
            text, fin, st = dec(g, rows[i]); out[i] = (text, fin, st, mode)

    if a.decoding == "best":
        mbr_idx = [i for i, r in enumerate(rows) if r["iso"] in set(a.mbr_languages)]
        rest = [i for i in range(len(rows)) if i not in set(mbr_idx)]
        run(mbr_idx, "mbr"); run(rest, "beam")
    else:
        run(list(range(len(rows))), a.decoding)
    return out


def rule_drafts(a, rows):
    from gemma_mt.post_edit import CORRUPTIONS, corrupt_reference, perturb_light, stable_hash
    by_iso = defaultdict(list)
    for i, r in enumerate(rows): by_iso[r["iso"]].append(i)
    out = [None] * len(rows)
    for iso, idx in by_iso.items():
        idx_sorted = sorted(idx, key=lambda i: rows[i]["pair_id"])
        for pos, i in enumerate(idx_sorted):
            r = rows[i]
            if a.type == "identity":
                out[i] = (r["reference"], True, "ok", "identity")
            elif a.type == "near":
                out[i] = (perturb_light(r["reference"], [r["pair_id"]]), True, "ok", "near")
            else:
                kind = CORRUPTIONS[stable_hash("kind", r["pair_id"], a.seed) % len(CORRUPTIONS)]
                nb = rows[idx_sorted[(pos + 1) % len(idx_sorted)]]["reference"] if len(idx_sorted) > 1 else None
                out[i] = (corrupt_reference(r["reference"], r["english"], kind, [r["pair_id"], a.seed], nb), True, "ok", "corrupt:" + kind)
    return out


def write_drafts(a, rows, out):
    from gemma_mt.post_edit import sentence_chrf
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"drafts_{a.system}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r, o in zip(rows, out):
            if o is None: continue
            text, fin, st, how = o
            f.write(json.dumps({"pair_id": r["pair_id"], "system": a.system, "how": how, "draft": text, "finished": bool(fin), "tag_status": st, "chrf": round(sentence_chrf(text, r["reference"]), 3),
                                "decoding": how.split(":")[0], "licence": "own" if a.type == "model" else "none", "model": getattr(a, "model", None) if a.type == "model" else a.type}, ensure_ascii=False) + "\n")
    return path


def summarize(a):
    import numpy as np
    from sacrebleu.metrics import CHRF
    from gemma_mt.post_edit import nfc
    pairs = {r["pair_id"]: r for r in read_jsonl(a.pairs)}
    files = sorted(f for f in os.listdir(a.out) if f.startswith("drafts_") and f.endswith(".jsonl"))
    chrf = CHRF(word_order=2); rows_out = []; allv = defaultdict(list)
    from gemma_mt.rl.guards import is_copy_of_source, is_repetitive, is_self_repetitive
    for fn in files:
        by = defaultdict(list)
        for d in read_jsonl(os.path.join(a.out, fn)):
            by[pairs[d["pair_id"]]["iso"]].append(d)
        for iso, ds in sorted(by.items()):
            hyp = [nfc(d["draft"]) for d in ds]; ref = [nfc(pairs[d["pair_id"]]["reference"]) for d in ds]; en = [pairs[d["pair_id"]]["english"] for d in ds]
            s = np.array([d["chrf"] for d in ds])
            rows_out.append({"system": ds[0]["system"], "iso": iso, "n": len(ds), "corpus_chrf": round(chrf.corpus_score(hyp, [ref]).score, 2), "mean_sent_chrf": round(float(s.mean()), 2),
                             "p10": round(float(np.percentile(s, 10)), 1), "p50": round(float(np.percentile(s, 50)), 1), "p90": round(float(np.percentile(s, 90)), 1),
                             "share_equal_ref": round(float(np.mean([h == r for h, r in zip(hyp, ref)])), 4), "share_empty": round(float(np.mean([not h.strip() for h in hyp])), 4),
                             "share_unfinished": round(1 - float(np.mean([d["finished"] for d in ds])), 4),
                             "share_loop": round(float(np.mean([bool(h.strip()) and (is_repetitive(h) or is_self_repetitive(h)) for h in hyp])), 4),
                             "share_copy_of_source": round(float(np.mean([bool(h.strip()) and len(e.split()) >= 4 and is_copy_of_source(e, h, 0.9) for h, e in zip(hyp, en)])), 4)})
            allv[ds[0]["system"]].extend(s.tolist())
    with open(os.path.join(a.out, "draft_quality_by_language_system.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0])) if rows_out else None
        if w: w.writeheader(); w.writerows(rows_out)
    glob_rows = [{"system": k, "n": len(v), "mean_sent_chrf": round(float(np.mean(v)), 2), "p10": round(float(np.percentile(v, 10)), 1), "p50": round(float(np.percentile(v, 50)), 1), "p90": round(float(np.percentile(v, 90)), 1)} for k, v in sorted(allv.items())]
    with open(os.path.join(a.out, "draft_quality_by_system.csv"), "w", newline="") as f:
        if glob_rows:
            w = csv.DictWriter(f, fieldnames=list(glob_rows[0])); w.writeheader(); w.writerows(glob_rows)
    print(json.dumps(glob_rows, indent=1))
    if a.val_system:                                   # validation drafts in the evaluation's predictions format: the editor can be run on them with post_editor_eval.py
        p = os.path.join(a.out, f"drafts_{a.val_system}.jsonl"); n = 0
        with open(os.path.join(a.out, "val_predictions.jsonl"), "w", encoding="utf-8") as f:
            for d in read_jsonl(p):
                r = pairs[d["pair_id"]]
                if r["split"] != "val": continue
                f.write(json.dumps({"system": a.val_system, "benchmark": "pe_val_" + r["cls"], "iso": r["iso"], "id": r["pair_id"], "english": r["english"], "reference": r["reference"],
                                    "hyp": d["draft"], "finished": d["finished"], "contaminated": None, "pair_seen": None}, ensure_ascii=False) + "\n"); n += 1
        print("val_predictions.jsonl:", n, "rows")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--summarize", action="store_true"); ap.add_argument("--val-system", default=None)
    ap.add_argument("--system", default=None, help="name of this draft system (file drafts_<name>.jsonl)"); ap.add_argument("--type", default="model", choices=["model", "corrupt", "identity", "near"])
    ap.add_argument("--plan", default=None, help="several systems on ONE engine: \"rl_greedy=greedy;rl_t07=sample,0.7,0.95;rl_best=best\""); ap.add_argument("--model", default=None); ap.add_argument("--tokenizer", default="google/gemma-3-270m-it", help="the ORIGINAL training tokenizer (prompts are token ids)")
    ap.add_argument("--decoding", default="greedy", choices=["greedy", "beam", "sample", "mbr", "best"]); ap.add_argument("--beam-width", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--top-p", type=float, default=0.95); ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--mbr-languages", nargs="*", default=MBR_LANGS); ap.add_argument("--mbr-workers", type=int, default=4); ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--per-language", type=int, default=0); ap.add_argument("--splits", nargs="*", default=None); ap.add_argument("--isos", nargs="*", default=None); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exposure-map", default=None); ap.add_argument("--unseen-after-shard", type=int, default=None)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf", "hfgpu"]); ap.add_argument("--batch-tokens", type=int, default=100000); ap.add_argument("--fake-tokenizer", action="store_true")
    ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--gpu-util", type=float, default=0.25); ap.add_argument("--max-model-len", type=int, default=2048); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.summarize:
        return summarize(a)
    if not a.system and not a.plan: ap.error("--system (or --plan) is required")
    rows = load_pairs(a.pairs, a.splits, a.limit, a.isos)
    if a.per_language:                                           # deterministic subset: N train pairs per language (hash order; the validation pairs all stay)
        from gemma_mt.post_edit import stable_unit
        by = defaultdict(list)
        for r in rows:
            if r["split"] == "train": by[r["iso"]].append(r)
        keep = {r["pair_id"] for v in by.values() for r in sorted(v, key=lambda r: stable_unit("pl", r["pair_id"]))[: a.per_language]}
        rows = [r for r in rows if r["split"] != "train" or r["pair_id"] in keep]
    if a.exposure_map and a.unseen_after_shard is not None:
        fs = {r["pair_id"]: r["first_shard"] for r in csv.DictReader(open(a.exposure_map))}
        rows = [r for r in rows if fs.get(r["pair_id"], "") not in ("", None) and int(fs[r["pair_id"]]) > a.unseen_after_shard]
        print(f"exposure filter: {len(rows)} pairs unseen by the checkpoint after shard {a.unseen_after_shard}")
    t0 = time.time()
    if a.type != "model":
        out = rule_drafts(a, rows); print(write_drafts(a, rows, out), time.time() - t0)
        return
    plan = parse_plan(a)
    tok = make_tokenizer(a); be = make_backend(a, tok)
    for name, spec in plan:
        a.system = name; t1 = time.time()
        out = model_drafts(a, rows, tok, be, spec)
        path = write_drafts(a, rows, out)
        print(f"{path}: {sum(o is not None for o in out)} drafts in {time.time()-t1:.0f}s (engine + earlier systems: {time.time()-t0:.0f}s)", flush=True)


def parse_plan(a):
    """--plan "name=greedy;name2=sample,0.7,0.95;name3=best;name4=beam;name5=mbr" or the single --system / --decoding."""
    if not a.plan:
        return [(a.system, {"decoding": a.decoding, "temperature": a.temperature, "top_p": a.top_p})]
    out = []
    for item in a.plan.split(";"):
        name, _, rest = item.partition("="); parts = rest.split(",")
        spec = {"decoding": parts[0]}
        if parts[0] == "sample":
            spec["temperature"] = float(parts[1]); spec["top_p"] = float(parts[2])
        out.append((name.strip(), spec))
    return out

if __name__ == "__main__":
    main()
