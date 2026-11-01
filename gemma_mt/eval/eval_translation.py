# SPDX-License-Identifier: AGPL-3.0-or-later
"""Evaluate a model (or pre-computed translations) on FLORES+ / NTREX-128 / MAFAND-MT.

    python -m gemma_mt.eval.eval_translation --model runs/rl/best --name rl \\
        --bench flores_plus:/data/flores_plus --bench ntrex:/data/NTREX --bench mafand --out results/rl

Decoding is GREEDY (num_beams=1, no sampling) with the same prompt format as training (gemma_mt.format). Outputs are
cut at the first <end_of_turn>/<eos>, special tokens are removed and the leading language tag is stripped before
any metric sees the text (the tag status is reported separately: `prefix_ok_rate`). A base model that does not
know the tag convention is scored on its whole output (the tag is stripped only if it is there).

Metrics per (benchmark, language): chrF++ (sacrebleu, word_order=2), spBLEU (flores200 tokenizer), BLEU (13a),
AfriCOMET-STL-1.1 (src, mt, ref) and AfriCOMET-QE-STL-1.1 (src, mt) for languages the AfriCOMET encoder covers
(gemma_mt.languages.QE_VALIDATED: the 42 languages whose AfriCOMET validity was measured, D-32). Macro averages are unweighted means over languages.

Contamination (gemma_mt.eval.contamination): `--contamination benchmark=path.json` (or --contamination-dir) flags every row whose
English sentence occurs in the SFT training windows; every table then has the ALL rows and the CLEAN subset (contaminated rows removed;
`clean` is the headline). More subsets: `clean_sentence` (the dev filter's sentence rule), `clean_strict` (no near duplicates either),
`no_pair` (the exact reference pair was not seen). A language enters a subset's macro average only with >= --min-subset-rows rows.

`--sampler vllm` (the default since 2026-10-29; needs `--vllm-python`) generates through a vLLM worker process (greedy, continuous batching); `--sampler hf` is the HF `generate` path for CPU tests only.
`--comet-python` / `--comet-pythonpath` run AfriCOMET in a child process with its own transformers 4.x (D-33).

Writes results.json, results.csv and predictions.jsonl into --out.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from gemma_mt import format as fmt
from gemma_mt.eval import benchmarks as B
from gemma_mt.eval import metrics as M
from gemma_mt.languages import LANGUAGES, QE_VALIDATED, canonical_iso
from gemma_mt.rl.text import decode_completion

CSV_FIELDS = ["benchmark", "iso", "language", "subset", "n", "chrf", "spbleu", "bleu", "comet", "comet_qe", "prefix_ok_rate",
              "finished_rate", "empty_rate", "n_contaminated", "n_sentence_rule", "n_near", "n_pair", "exact"]

# Row subsets. A row's flags come from gemma_mt.eval.contamination.apply_flags: `contaminated` (English sentence in the training text),
# `contaminated_sentence` (the stricter-free sentence-level rule of the 2026-10-14 dev filter), `near_dup`, `pair_seen`.
SUBSETS: Dict[str, Any] = {
    "all": lambda r: True,
    "clean": lambda r: r.get("contaminated") is False,
    "clean_sentence": lambda r: r.get("contaminated_sentence") is False,
    "clean_strict": lambda r: r.get("contaminated") is False and r.get("near_dup") is False,
    "no_pair": lambda r: r.get("pair_seen") is False,
}
_SUBSET_FLAG = {"clean": "contaminated", "clean_sentence": "contaminated_sentence", "clean_strict": "near_dup", "no_pair": "pair_seen"}
FULL_METRIC_SUBSETS = ("all", "clean")           # spBLEU / BLEU (slow tokenizers) only here; the other subsets get chrF++ and AfriCOMET


class TimeLimitExceeded(RuntimeError):
    """A wall-clock limit of the pipeline (generation or scoring of one benchmark) ran out."""


def _check_deadline(deadline: Optional[float], what: str) -> None:
    if deadline is not None and time.time() > deadline:
        raise TimeLimitExceeded(f"wall-clock limit reached during {what}")


def generate_translations(model: Any, tokenizer: Any, rows: Sequence[Dict[str, str]], *, stop_ids, pad_id: int,
                          device, batch_size: int = 32, max_new_tokens: int = 256, autocast=None,
                          sampler: Any = None, deadline: Optional[float] = None, chunk: int = 8192,
                          log: Optional[Any] = None, require_vllm: bool = False) -> List[Dict[str, Any]]:
    """Greedy decoding in length-sorted batches; returns one dict per row in INPUT order.
    `sampler` (gemma_mt.rl.vllm_sampler, already synced to the model): greedy decoding of ALL rows through the vLLM engine instead
    (continuous batching, no padding; `model` / `batch_size` / `autocast` are then unused); rows go to the engine in chunks of `chunk`.
    `deadline` (time.time() value): TimeLimitExceeded between two batches / chunks once it has passed.
    `require_vllm` (2026-10-29, Amendment 4: all evaluation generation uses vLLM): raise instead of falling back to the HF `generate` path below when no `sampler` is given. The HF path stays for CPU tests only."""
    import torch
    from contextlib import nullcontext

    if require_vllm and sampler is None:
        raise RuntimeError("generate_translations(require_vllm=True) needs a vLLM sampler: the HF generate fallback is not allowed for evaluation")

    # rows of the reverse direction carry `prompt_iso` = "eng" (their `english` field holds the African source text)
    prompts = [fmt.build_prompt(r["english"], r.get("prompt_iso", r["iso"]), tokenizer) for r in rows]
    if sampler is not None:
        res: List[Dict[str, Any]] = []
        for s in range(0, len(rows), chunk):
            _check_deadline(deadline, f"generation (row {s} of {len(rows)})")
            roll = sampler.generate(prompts[s: s + chunk], group_size=1, max_new_tokens=max_new_tokens, temperature=1.0, top_p=1.0,
                                    top_k=0, stop_ids=stop_ids, pad_id=pad_id, greedy=True)
            for r, ids in zip(rows[s: s + chunk], roll.completions):
                dec = decode_completion(tokenizer, ids, stop_ids)
                strip = fmt.strip_iso_prefix(dec.text, r.get("prompt_iso", r["iso"]))
                res.append({"raw": dec.text, "hyp": strip.text, "prefix_status": strip.status, "finished": dec.finished})
            if log:
                log(f"  generated {min(s + chunk, len(rows))}/{len(rows)}")
        return res
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
    out: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    model.eval()
    for k, s in enumerate(range(0, len(order), batch_size)):
        _check_deadline(deadline, f"generation (batch {k})")
        idx = order[s: s + batch_size]
        width = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for j, i in enumerate(idx):
            ids[j, width - len(prompts[i]):] = torch.tensor(prompts[i])
            att[j, width - len(prompts[i]):] = 1
        with torch.no_grad(), (autocast if autocast is not None else nullcontext()):
            gen = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), do_sample=False,
                                 temperature=None, top_p=None, top_k=None, num_beams=1, max_new_tokens=max_new_tokens,
                                 eos_token_id=list(stop_ids), pad_token_id=pad_id, use_cache=True)
        for j, i in enumerate(idx):
            dec = decode_completion(tokenizer, gen[j, width:].tolist(), stop_ids)
            strip = fmt.strip_iso_prefix(dec.text, rows[i].get("prompt_iso", rows[i]["iso"]))
            out[i] = {"raw": dec.text, "hyp": strip.text, "prefix_status": strip.status, "finished": dec.finished}
        if log and k % 50 == 0:
            log(f"  generated {min(s + batch_size, len(order))}/{len(order)}")
    return out  # type: ignore[return-value]


def reverse_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Benchmark rows of the reverse direction (African -> English): the reference-side African text becomes the
    source (`english` field = source text of the prompt) and the English sentence the reference. `iso` stays the
    African language (grouping, names, COMET coverage); `prompt_iso` is the target tag of the prompt."""
    return [{**r, "english": r["reference"], "reference": r["english"], "prompt_iso": fmt.PIVOT_ISO, "direction": fmt.X2EN}
            for r in rows]


def _score_in_chunks(fn: Any, srcs: List[str], hyps: List[str], refs: Optional[List[str]], *, chunk: int, deadline: Optional[float],
                     log: Optional[Any], what: str) -> List[float]:
    """Scores for all items, computed in length-sorted chunks (uniform batches inside the scorer, progress, deadline check)."""
    n = len(srcs)
    order = sorted(range(n), key=lambda i: len(srcs[i]) + len(hyps[i]))
    out: List[Optional[float]] = [None] * n
    for s in range(0, n, chunk):
        _check_deadline(deadline, f"{what} scoring ({s}/{n})")
        idx = order[s: s + chunk]
        sc = fn([srcs[i] for i in idx], [hyps[i] for i in idx], [refs[i] for i in idx]) if refs is not None \
            else fn([srcs[i] for i in idx], [hyps[i] for i in idx])
        if len(sc) != len(idx):
            raise RuntimeError(f"{what} scorer returned {len(sc)} scores for {len(idx)} inputs")
        for i, v in zip(idx, sc):
            out[i] = float(v)
        if log:
            log(f"  {what}: {min(s + chunk, n)}/{n}")
    return out  # type: ignore[return-value]


def _subset_metrics(idx: Sequence[int], rows, outputs, ref_scores, qe_scores, *, full: bool) -> Dict[str, Any]:
    n = len(idx)
    rec: Dict[str, Any] = {"n": n}
    if not n:
        return rec
    hyps = [outputs[i]["hyp"] for i in idx]
    refs = [rows[i]["reference"] for i in idx]
    if all(rows[i].get("references") for i in idx):
        # word-level lexicon rows (GATITOS): several accepted translations per row; exact match + chrF++ against the best one. spBLEU / BLEU of single
        # words are meaningless and are not computed.
        multi = [list(rows[i]["references"]) for i in idx]
        rec["chrf"] = M.chrf_pp_multi(hyps, multi)
        rec["exact"] = M.exact_match(hyps, multi)
    elif full:
        rec.update(M.compute_text_metrics(hyps, refs))
    else:
        rec["chrf"] = M.chrf_pp(hyps, refs)
    rec["prefix_ok_rate"] = sum(outputs[i].get("prefix_status") == "ok" for i in idx) / n
    rec["finished_rate"] = sum(bool(outputs[i].get("finished", True)) for i in idx) / n
    rec["empty_rate"] = sum(1 for h in hyps if not h.strip()) / n
    if ref_scores is not None:
        rec["comet"] = M.mean([ref_scores[i] for i in idx])
    if qe_scores is not None:
        rec["comet_qe"] = M.mean([qe_scores[i] for i in idx])
    return rec


def evaluate_rows(rows: Sequence[Dict[str, str]], outputs: Sequence[Dict[str, Any]], *, comet_ref: Any = None,
                  comet_qe: Any = None, comet_languages: Sequence[str] = QE_VALIDATED, min_subset_rows: int = 30,
                  comet_chunk: int = 8192, deadline: Optional[float] = None, log: Optional[Any] = None) -> Dict[str, Any]:
    """Pure aggregation: metrics per benchmark / language + macro averages. `outputs[i]["hyp"]` is the translation
    of rows[i] with the tag already stripped.

    Top-level keys of a language record are the ALL-rows metrics (unchanged); when the rows carry contamination flags
    (gemma_mt.eval.contamination.apply_flags) the record also has one nested dict per subset of SUBSETS ("clean", "clean_sentence",
    "clean_strict", "no_pair"; a subset exists for a benchmark only when EVERY row of it has the flag) and `n_contaminated`,
    `n_sentence_rule`, `n_near`, `n_pair`. A benchmark has `macro` (all rows) and `macro_<subset>` (languages with >= `min_subset_rows`
    rows in the subset). AfriCOMET is computed once per row (in chunks) and averaged over whichever subset."""
    if len(rows) != len(outputs):
        raise ValueError("rows and outputs differ in length")
    groups: Dict[tuple, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["benchmark"], r["iso"])].append(i)
    # --- which subsets exist for which benchmark
    flagged: Dict[str, Dict[str, bool]] = {}
    for b in {r["benchmark"] for r in rows}:
        rb = [r for r in rows if r["benchmark"] == b]
        flagged[b] = {s: all(r.get(f) is not None for r in rb) for s, f in _SUBSET_FLAG.items()}
    # --- AfriCOMET once per row
    ref_scores: Optional[List[Optional[float]]] = None
    qe_scores: Optional[List[Optional[float]]] = None
    elig = [i for i, r in enumerate(rows) if r["iso"] in comet_languages]
    if comet_ref is not None and elig:
        sc = _score_in_chunks(comet_ref, [rows[i]["english"] for i in elig], [outputs[i]["hyp"] for i in elig],
                              [rows[i]["reference"] for i in elig], chunk=comet_chunk, deadline=deadline, log=log, what="AfriCOMET-STL")
        ref_scores = [None] * len(rows)
        for i, v in zip(elig, sc):
            ref_scores[i] = v
    if comet_qe is not None and elig:
        sc = _score_in_chunks(comet_qe, [rows[i]["english"] for i in elig], [outputs[i]["hyp"] for i in elig], None,
                              chunk=comet_chunk, deadline=deadline, log=log, what="AfriCOMET-QE")
        qe_scores = [None] * len(rows)
        for i, v in zip(elig, sc):
            qe_scores[i] = v
    bench: Dict[str, Dict[str, Any]] = {}
    for (b, iso), idx in sorted(groups.items()):
        rec: Dict[str, Any] = {"language": LANGUAGES[iso].name}
        rs = ref_scores if iso in comet_languages else None
        qs = qe_scores if iso in comet_languages else None
        rec.update(_subset_metrics(idx, rows, outputs, rs, qs, full=True))
        if flagged[b]["clean"]:
            rec["n_contaminated"] = sum(1 for i in idx if rows[i].get("contaminated"))
            rec["n_sentence_rule"] = sum(1 for i in idx if rows[i].get("contaminated_sentence"))
            rec["n_near"] = sum(1 for i in idx if rows[i].get("near_dup"))
            rec["n_pair"] = sum(1 for i in idx if rows[i].get("pair_seen"))
        for sname in SUBSETS:
            if sname == "all" or not flagged[b][sname]:
                continue
            sub = [i for i in idx if SUBSETS[sname](rows[i])]
            rec[sname] = _subset_metrics(sub, rows, outputs, rs, qs, full=sname in FULL_METRIC_SUBSETS)
        bench.setdefault(b, {"languages": {}})["languages"][iso] = rec
    for b, d in bench.items():
        d["macro"] = M.macro_average(d["languages"], M.ALL_METRICS)
        d["macro"]["n_languages"] = len(d["languages"])
        for sname in SUBSETS:
            if sname == "all" or not flagged[b][sname]:
                continue
            keep = {iso: rec[sname] for iso, rec in d["languages"].items() if rec[sname].get("n", 0) >= min_subset_rows}
            mac = M.macro_average(keep, M.ALL_METRICS)
            mac["n_languages"] = len(keep)
            mac["n_rows"] = sum(v["n"] for v in keep.values())
            d[f"macro_{sname}"] = mac
    return overall(bench, min_subset_rows)


def overall(bench: Dict[str, Dict[str, Any]], min_subset_rows: int = 30) -> Dict[str, Any]:
    """{"benchmarks", "macro_all", ["macro_all_clean"], "min_subset_rows"}: macro averages over every (benchmark, language) pair."""
    langs_all: Dict[str, Dict[str, float]] = {}
    for b, d in bench.items():
        for iso, rec in d["languages"].items():
            langs_all.setdefault(f"{b}:{iso}", rec)
    out = {"benchmarks": bench, "macro_all": M.macro_average(langs_all, M.ALL_METRICS), "min_subset_rows": min_subset_rows}
    langs_clean = {f"{b}:{iso}": rec["clean"] for b, d in bench.items() for iso, rec in d["languages"].items()
                   if "clean" in rec and rec["clean"].get("n", 0) >= min_subset_rows}
    if langs_clean:
        out["macro_all_clean"] = M.macro_average(langs_clean, M.ALL_METRICS)
    return out


def write_results(out_dir: str, name: str, meta: Dict[str, Any], results: Dict[str, Any], rows, outputs) -> None:
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(meta, name=name, spbleu_tokenizer=M.spbleu_tokenizer())
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, **results}, f, indent=2, ensure_ascii=False)
    write_csv(os.path.join(out_dir, "results.csv"), results)
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for r, o in zip(rows, outputs):
            f.write(json.dumps({**r, "hyp": o["hyp"], "raw": o.get("raw", ""), "prefix_status": o.get("prefix_status"),
                                "finished": o.get("finished", True)}, ensure_ascii=False) + "\n")


def csv_rows(results: Dict[str, Any], system: Optional[str] = None) -> List[Dict[str, Any]]:
    """Flat rows (one per benchmark x language x subset, plus MACRO rows) of a results dict."""
    out: List[Dict[str, Any]] = []
    for b, d in results["benchmarks"].items():
        for iso, rec in d["languages"].items():
            for sname in SUBSETS:
                src = rec if sname == "all" else rec.get(sname)
                if src is None:
                    continue
                row = {"benchmark": b, "iso": iso, "language": rec.get("language", ""), "subset": sname,
                       **{k: src.get(k, "") for k in CSV_FIELDS[4:13]}, "exact": src.get("exact", "")}
                if sname == "all":
                    row.update({k: rec.get(k, "") for k in ("n_contaminated", "n_sentence_rule", "n_near", "n_pair")})
                out.append(row)
        for sname in SUBSETS:
            mac = d["macro"] if sname == "all" else d.get(f"macro_{sname}")
            if mac is None:
                continue
            out.append({"benchmark": b, "iso": "MACRO", "language": "", "subset": sname, "n": mac.get("n_languages", ""),
                        **{k: mac.get(k, "") for k in ("chrf", "spbleu", "bleu", "comet", "comet_qe", "exact")}})
    if system is not None:
        for r in out:
            r["system"] = system
    return out


def write_csv(path: str, results: Dict[str, Any], systems: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """results.csv: one system (`results`) or several (`systems` = {name: results}, a `system` column is added)."""
    fields = (["system"] if systems else []) + CSV_FIELDS
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        if systems:
            for name, res in systems.items():
                for row in csv_rows(res, name):
                    w.writerow(row)
        else:
            for row in csv_rows(results):
                w.writerow(row)


def _parse_bench(specs: Sequence[str]):
    for s in specs:
        name, _, path = s.partition(":")
        yield name, (path or None)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="checkpoint dir / Hub id (omit with --predictions)")
    ap.add_argument("--revision", default=None, help="Hub branch / tag / commit of --model")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--name", required=True, help="label of this system in the report")
    ap.add_argument("--bench", action="append", default=[], help="name[:local_path]; name in flores_plus|ntrex|mafand|smol_sent|smol_doc|gatitos")
    ap.add_argument("--bench-file", action="append", default=[], help="parquet/jsonl with english,reference,iso")
    ap.add_argument("--split", default=None)
    ap.add_argument("--languages", nargs="*", default=None)
    ap.add_argument("--max-per-language", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--sampler", default="vllm", choices=["hf", "vllm"], help="vllm (default since 2026-10-29): worker process, needs --vllm-python; hf: model.generate in length-sorted batches (CPU tests / debugging only)")
    ap.add_argument("--vllm-python", default=None, help="interpreter of the environment that has vllm (--sampler vllm)")
    ap.add_argument("--vllm-gpu-util", type=float, default=0.2)
    ap.add_argument("--vllm-max-model-len", type=int, default=4096)
    ap.add_argument("--vllm-max-num-seqs", type=int, default=512)
    ap.add_argument("--predictions", default=None,
                    help="jsonl with benchmark,iso,id,hyp to score instead of generating (external baselines)")
    ap.add_argument("--direction", default="en2x", choices=list(fmt.DIRECTIONS),
                    help="en2x: English -> African (default); x2en: African -> English (the benchmark rows are swapped)")
    ap.add_argument("--contamination", action="append", default=[], help="benchmark=contamination_<benchmark>.json (repeatable)")
    ap.add_argument("--contamination-dir", default=None, help="directory with contamination_<benchmark>.json for every --bench")
    ap.add_argument("--min-subset-rows", type=int, default=30)
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--comet-batch-size", type=int, default=32)
    ap.add_argument("--comet-python", default=None, help="interpreter for the AfriCOMET child process (default: this one, in-process scorer)")
    ap.add_argument("--comet-pythonpath", default=None,
                    help="PYTHONPATH of the AfriCOMET child (a transformers 4.x directory + the comet libraries; D-33)")
    ap.add_argument("--time-limit", type=float, default=0, help="seconds; generation / scoring stop with TimeLimitExceeded after it (0 = none)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    deadline = time.time() + a.time_limit if a.time_limit else None

    rows: List[Dict[str, str]] = []
    for name, path in _parse_bench(a.bench):
        rows += B.load_benchmark(name, path=path, isos=a.languages, split=a.split, limit=a.max_per_language)
    for p in a.bench_file:
        rows += B.load_normalised_file(p)
    if a.languages:
        keep = {canonical_iso(l) for l in a.languages}
        rows = [r for r in rows if r["iso"] in keep]
    if not rows:
        print("no benchmark rows loaded", file=sys.stderr)
        return 2
    flag_files = {}
    for spec in a.contamination:
        b, _, p = spec.partition("=")
        flag_files[b] = p
    if a.contamination_dir:
        from gemma_mt.eval.contamination import default_flag_files

        flag_files = {**default_flag_files(a.contamination_dir, sorted({r["benchmark"] for r in rows})), **flag_files}
    if flag_files:
        from gemma_mt.eval.contamination import apply_flags

        print(f"[eval] contamination flags: {apply_flags(rows, flag_files)} rows without a flag", flush=True)
    if a.direction == fmt.X2EN:
        rows = reverse_rows(rows)
    print(f"[eval] {len(rows)} rows, direction={a.direction}, benchmarks={sorted({r['benchmark'] for r in rows})}", flush=True)

    t0 = time.time()
    sampler_name = a.sampler
    if a.predictions:
        from gemma_mt.data.io import iter_rows

        pred = {(p["benchmark"], canonical_iso(p["iso"]), str(p["id"])): p["hyp"] for p in iter_rows(a.predictions,
                columns=("benchmark", "iso", "id", "hyp"))}
        keep, outputs = [], []
        for r in rows:
            k = (r["benchmark"], r["iso"], r["id"])
            if k in pred:
                keep.append(r)
                outputs.append({"hyp": pred[k], "raw": pred[k], "prefix_status": "n/a", "finished": True})
        rows = keep
        sampler_name = "predictions"
    else:
        if not a.model:
            ap.error("--model is required unless --predictions is given")
        from gemma_mt.eval import runtime as RT

        ref = RT.resolve_model(a.model, revision=a.revision)
        tok, stop, pad = RT.load_tokenizer_and_stops(a.tokenizer or ref.path)
        if a.sampler == "vllm":
            if not a.vllm_python:
                ap.error("--sampler vllm needs --vllm-python")
            sampler = RT.make_vllm_sampler(ref.path, python=a.vllm_python, gpu_util=a.vllm_gpu_util, max_model_len=a.vllm_max_model_len,
                                           max_num_seqs=a.vllm_max_num_seqs, log_path=os.path.join(a.out, "vllm_worker.log"))
            try:
                outputs = generate_translations(None, tok, rows, stop_ids=stop.ids, pad_id=pad, device=None, max_new_tokens=a.max_new_tokens,
                                                sampler=sampler, deadline=deadline, log=lambda m: print(m, flush=True))
            finally:
                sampler.close()
        else:
            import torch

            from gemma_mt.modeling import load_model

            dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[a.dtype]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = load_model(ref.path, attn_implementation=a.attn, dtype=dtype, device=device)
            outputs = generate_translations(model, tok, rows, stop_ids=stop.ids, pad_id=pad, device=device, batch_size=a.batch_size,
                                            max_new_tokens=a.max_new_tokens, deadline=deadline, log=lambda m: print(m, flush=True))
            del model
    t_gen = time.time() - t0

    comet_ref = comet_qe = None
    pair = None
    if not a.no_comet:
        if a.comet_python:
            from gemma_mt.eval.runtime import CometPair

            pair = CometPair(python=a.comet_python, pythonpath=a.comet_pythonpath, batch_size=a.comet_batch_size,
                             cache_dir=os.path.join(a.out, "comet_cache"))
            comet_ref, comet_qe = pair.reference, pair.qe
        else:
            from gemma_mt.rl.comet import QE_MODEL_ID, REF_MODEL_ID, CometScorer

            comet_ref = CometScorer(REF_MODEL_ID, batch_size=a.comet_batch_size, reference_based=True)
            comet_qe = CometScorer(QE_MODEL_ID, batch_size=a.comet_batch_size)
    try:
        results = evaluate_rows(rows, outputs, comet_ref=comet_ref, comet_qe=comet_qe, min_subset_rows=a.min_subset_rows,
                                deadline=deadline, log=lambda m: print(m, flush=True))
    finally:
        if pair is not None:
            pair.close()
    meta = {"model": a.model, "revision": a.revision, "tokenizer": a.tokenizer, "decoding": "greedy", "max_new_tokens": a.max_new_tokens,
            "n_rows": len(rows), "seconds_generation": round(t_gen, 1), "comet": not a.no_comet, "sampler": sampler_name,
            "prompt_template": fmt.INSTRUCTION_TEMPLATE, "direction": a.direction, "contamination_files": flag_files}
    write_results(a.out, a.name, meta, results, rows, outputs)
    for b, d in results["benchmarks"].items():
        print(f"[eval] {b}: macro {json.dumps({k: round(v, 3) for k, v in d['macro'].items() if isinstance(v, float)})}")
        if "macro_clean" in d:
            print(f"[eval] {b}: macro CLEAN {json.dumps({k: round(v, 3) for k, v in d['macro_clean'].items() if isinstance(v, float)})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
