#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-editor: draft translations of the BENCHMARKS (the unseen text the editor is judged on), in the evaluation's predictions.jsonl format, one file per decoding system.

    python scripts/post_editor_bench_drafts.py --model RL_DIR --bench-root /workspace/bench/test_sets --contam-dir /workspace/pe/contam --out DIR \
        --benchmarks flores_plus ntrex mafand smol_sent --per-language 250 --systems greedy beam4 mbr16 [--isos yor hau ...]

Rows: the benchmark loaders of gemma_mt.eval.benchmarks (same ids as the evaluation), a deterministic subsample of `--per-language` sentences per (benchmark, language) chosen by a hash of the id
(identical for every system and for the editor run), contamination flags from `--contam-dir` (contamination_<benchmark>.json). Decoding = the draft-generation script's: greedy; beam4
(width 4, length penalty 1.0); mbr16 (16 samples T 0.7 top-p 0.95 + greedy, chrF++ utility). Writes DIR/predictions_<system>.jsonl.
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))


def load_rows(a):
    from gemma_mt.eval import benchmarks as B
    from gemma_mt.eval import contamination as C
    from gemma_mt.languages import ALL_ISOS, PIVOT_ISO
    from gemma_mt.post_edit import stable_unit
    rows = []
    for b in ([] if a.afsci_only else a.benchmarks):
        isos = [i for i in (a.isos or B.languages_with_data(b)) if i != PIVOT_ISO]
        try:
            rs = B.load_benchmark(b, path=os.path.join(a.bench_root, b), isos=isos)
        except Exception as e:                                                   # a benchmark folder that is missing or has other languages: say it and go on
            print(f"[{b}] skipped: {type(e).__name__}: {e}", flush=True); continue
        by = {}
        for r in rs: by.setdefault(r["iso"], []).append(r)
        for iso, v in by.items():
            v = sorted(v, key=lambda r: stable_unit("pe-bench", a.seed, b, iso, r["id"]))[: a.per_language] if a.per_language else v
            rows += [dict(r, benchmark=b) for r in sorted(v, key=lambda r: str(r["id"]))]
        print(f"[{b}] {sum(len(v) for v in by.values())} rows in {len(by)} languages -> kept {sum(1 for r in rows if r['benchmark']==b)}", flush=True)
    if a.afsci_dir:                                                # AfriScience-MT TEST (sentence level, 970 sentences x 6 languages): an extra benchmark of the editor
        import glob, pandas as pd
        files = sorted(glob.glob(os.path.join(a.afsci_dir, "test-*.parquet")))
        df = pd.concat([pd.read_parquet(f) for f in files])
        for r in df.to_dict("records"):
            iso = str(r.get("lang_pair", "")).split("-")[-1]            # lang_pair = eng-amh
            if iso in ("amh", "hau", "lug", "nso", "yor", "zul") and str(r.get("source", "")).strip() and str(r.get("target", "")).strip():
                rows.append({"benchmark": "afriscience_test", "iso": iso, "id": f"{r['paper_id']}-{r['sentence_id']}", "english": r["source"].strip(), "reference": r["target"].strip()})
        print("afriscience_test rows:", sum(r["benchmark"] == "afriscience_test" for r in rows), flush=True)
    for r in rows: r["id"] = str(r["id"])
    if a.contam_dir:
        miss = C.apply_flags(rows, C.default_flag_files(a.contam_dir, a.benchmarks))
        print("rows without a contamination flag:", miss, flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--bench-root", required=True); ap.add_argument("--contam-dir", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=["flores_plus", "ntrex", "mafand", "smol_sent"]); ap.add_argument("--isos", nargs="*", default=None)
    ap.add_argument("--per-language", type=int, default=250); ap.add_argument("--systems", nargs="+", default=["greedy", "beam4", "mbr16"]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="google/gemma-3-270m-it"); ap.add_argument("--fake-tokenizer", action="store_true"); ap.add_argument("--backend", default="vllm", choices=["vllm", "hf", "hfgpu"]); ap.add_argument("--batch-tokens", type=int, default=100000)
    ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--gpu-util", type=float, default=0.8); ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=768, help="greedy cap (the evaluation's); beam/MBR use --max-new-search")
    ap.add_argument("--max-new-search", type=int, default=256); ap.add_argument("--k", type=int, default=16); ap.add_argument("--mbr-workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--afsci-dir", default=None, help="directory with the corpus/test-*.parquet of dsfsi/afriscience_mt: adds the benchmark afriscience_test"); ap.add_argument("--afsci-only", action="store_true")
    ap.add_argument("--label", default=None, help="system name written in the rows (default: the decoding)")
    a = ap.parse_args(argv)
    from gemma_mt import format as fmt
    from gemma_mt.post_edit import decode_gen, mbr_select
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    rows = load_rows(a)
    if a.limit: rows = rows[: a.limit]
    if a.fake_tokenizer:
        from gemma_mt.fake_tokenizer import FakeGemmaTokenizer; tok = FakeGemmaTokenizer()
    else:
        from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained(a.tokenizer)
    from gemma_mt.post_edit import make_backend
    be = make_backend(a.backend, a.model, tok, a.dtype, a.gpu_util, a.max_model_len, a.batch_tokens, a.seed)
    prompts = [list(fmt.build_prompt(r["english"], r["iso"], tok)) for r in rows]
    for sysname in a.systems:
        t1 = time.time()
        if sysname == "greedy":
            gens = be.greedy(prompts, a.max_new)
        elif sysname == "beam4":
            gens = be.beam(prompts, 4, a.max_new_search)
        elif sysname == "mbr16":
            greedy = be.greedy(prompts, a.max_new_search); samp = be.sample(prompts, a.k, 0.7, 0.95, a.max_new_search, a.seed)
            allc = [[g] + list(s) for g, s in zip(greedy, samp)]
            texts = [[decode_gen(tok, c, r["iso"])[0] for c in cs] for r, cs in zip(rows, allc)]
            if a.mbr_workers > 1:
                from multiprocessing import Pool
                with Pool(a.mbr_workers) as pool: pick = pool.map(mbr_select, texts, chunksize=16)
            else:
                pick = [mbr_select(t) for t in texts]
            gens = [cs[j] for cs, j in zip(allc, pick)]
        elif sysname.startswith("sample"):                                  # sample0.7
            T = float(sysname[6:] or 0.7); gens = [x[0] for x in be.sample(prompts, 1, T, 0.95 if T < 1 else 1.0, a.max_new_search, a.seed)]
        else:
            raise SystemExit(f"unknown system {sysname}")
        path = os.path.join(a.out, f"predictions_{sysname}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r, g in zip(rows, gens):
                text, fin, st = decode_gen(tok, g, r["iso"])
                f.write(json.dumps({"system": a.label or sysname, "benchmark": r["benchmark"], "iso": r["iso"], "id": r["id"], "english": r["english"], "reference": r["reference"], "hyp": text,
                                    "finished": fin, "prefix_status": st, "contaminated": r.get("contaminated"), "contaminated_sentence": r.get("contaminated_sentence"),
                                    "near_dup": r.get("near_dup"), "pair_seen": r.get("pair_seen")}, ensure_ascii=False) + "\n")
        print(f"{path}: {len(rows)} rows, {time.time()-t1:.0f}s (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
