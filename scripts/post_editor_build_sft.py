#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-editor, step 3: assemble the editor SFT set (source + draft -> human reference) and write it as a PACKED dataset the existing SFT trainer reads unchanged.

    python scripts/post_editor_build_sft.py --pairs OUT/pairs.jsonl --drafts DRAFTS_DIR --out PACKED_DIR \
        --benchmark-predictions /workspace/eval_385m/out_m385/predictions.jsonl [--tokenizer google/gemma-3-270m-it] [--seq-len 2048]
    python scripts/post_editor_build_sft.py --init-marker-rows --model DIR --out DIR2 [--marker '<unused6150>' --init-text 'same']      # only for the KEEP-marker variant

The ANTI-COPY design (owner, 2026-11-03): the SFT model saw most of these pairs, so its own drafts are better here than on unseen text; an editor trained on them learns "copy the draft".
  1. Candidate drafts per pair come from systems of different quality (drafts_<system>.jsonl: SFT-final beam/MBR, greedy, samples at T 0.7 and 1.0, the A0 270M checkpoint, a checkpoint that
     did not see the pair, synthetic corruptions of the human reference). See post_editor_make_drafts.py.
  2. EDIT-RATE MATCHING ("quantile matching"): every pair draws a desired sentence chrF++ from the empirical distribution of the SFT model's drafts on UNSEEN benchmark text of the same language
     (--benchmark-predictions; else --default-mean/--default-sd), and gets the candidate draft whose chrF++ is closest. The training draft quality therefore follows the benchmark draft quality,
     language by language (matching_report.csv prints both distributions).
  3. LEAVE-ALONE CONTROL: --identity-share of all examples (default 7%) have draft == reference; --near-share (3%) a reference with 1-2 tiny edits. Target = the reference, so a good editor copies here.
     With --keep-variant the target of a row whose draft chrF++ >= --keep-threshold is the KEEP marker token instead of the text.
  4. Tail languages (< --multi-draft-below train pairs) get 2 (or 3 below a third of it) examples per pair with DIFFERENT drafts.
Loss: only `{tag} {reference}<end_of_turn>` (gemma_mt.post_edit.build_pe_example). Output: PACKED_DIR/{train,val}/ + meta.json (+ examples_*.jsonl, matching_report.csv, build_summary.json).
"""
import argparse, csv, json, os, random, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))

DEFAULT_SYSTEM_WEIGHTS = {"nllb": 1.0, "madlad": 1.0, "sft_best": 1.0, "sft_greedy": 1.0, "sft_t07": 1.0, "sft_t10": 1.0, "a0_greedy": 1.5, "early_greedy": 2.0, "stageA_greedy": 1.5, "corrupt": 1.5}


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def benchmark_chrf_distributions(pred_path, per_lang=300, benchmarks=None, seed=0):
    """{iso: sorted sentence chrF++ values (<= per_lang)} of the hypotheses in an evaluation predictions.jsonl (hyp vs reference): the draft quality the editor will meet at test time."""
    from gemma_mt.post_edit import sentence_chrf, stable_unit
    by = defaultdict(list)
    for r in read_jsonl(pred_path):
        if benchmarks and r["benchmark"] not in benchmarks: continue
        h = r.get("hyp", r.get("hypothesis", "")) or ""
        by[r["iso"]].append((stable_unit("bd", seed, r["benchmark"], r["id"]), h, r["reference"]))
    return {iso: sorted(round(sentence_chrf(h, ref), 1) for _, h, ref in sorted(v)[:per_lang]) for iso, v in by.items()}


def pctl(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def assemble(pairs, drafts, bench_dist, a, weights):
    """-> list of example dicts {pair_id, iso, split, english, draft, reference, kind, system, chrf, keep}. Deterministic in (pair_id, seed)."""
    from gemma_mt.post_edit import perturb_light, pick_candidate, sample_desired_chrf, sentence_chrf, stable_hash, stable_unit
    pooled = sorted(x for v in bench_dist.values() for x in v) if bench_dist else None
    n_train = defaultdict(int)
    for p in pairs:
        n_train[p["iso"]] += p["split"] == "train"
    out = []
    for p in pairs:
        pid, iso = p["pair_id"], p["iso"]
        cands = [d for d in drafts.get(pid, []) if d["draft"].strip()]
        cands = [d for d in cands if d["system"] not in set(a.exclude_systems) and not any(x in str(d.get("licence", "")).lower() for x in a.exclude_draft_licences)]
        rng = random.Random(stable_hash("match", pid, a.seed))
        dist = bench_dist.get(iso) if bench_dist and len(bench_dist.get(iso, [])) >= 30 else pooled
        k = 1 + (n_train[iso] < a.multi_draft_below) + (n_train[iso] < a.multi_draft_below // 3)
        used = set()
        if cands:
            for j in range(k):
                pool = [d for d in cands if (d["system"], d["draft"]) not in used] or cands
                if j == 0 and stable_unit("cor", pid, a.seed) < a.corrupt_share:        # a guaranteed share of corrupted-reference drafts (real edit work even where the model draft is good)
                    pool = [d for d in pool if d["system"].startswith("corrupt")] or pool
                desired = sample_desired_chrf(dist, rng, a.default_mean, a.default_sd)
                ws = [weights.get(d["system"], next((w for k, w in weights.items() if d["system"].startswith(k)), 1.0)) for d in pool]
                if p.get("sft_input") is False:                  # pairs the SFT / RL models never saw (AfriScience train, SALT train, Infopankki...): their own drafts are realistic, prefer them
                    ws = [w * (a.unseen_boost if d["system"].startswith(("rl_", "sft_")) else 1.0) for w, d in zip(ws, pool)]
                c = pool[pick_candidate([d["chrf"] for d in pool], desired, rng, ws)]
                used.add((c["system"], c["draft"]))
                out.append({"pair_id": pid, "iso": iso, "split": p["split"], "english": p["english"], "draft": c["draft"], "reference": p["reference"], "kind": "matched", "system": c["system"],
                            "how": c["how"], "chrf": c["chrf"], "desired": round(desired, 1), "draft_system": c["system"], "draft_decoding": c.get("decoding", c["how"]),
                            "draft_licence": c.get("licence", "own"), "domain": p.get("domain", ""), "sft_input": p.get("sft_input"), "target_type": "human", "target_system": p.get("source_family", ""), "target_licence": "non_commercial" if p.get("non_commercial") else p.get("licence", "")})
        # leave-alone controls (extra rows)
        u = stable_unit("ctl", pid, a.seed)
        if u < a.identity_share:
            out.append({"pair_id": pid, "iso": iso, "split": p["split"], "english": p["english"], "draft": p["reference"], "reference": p["reference"], "kind": "identity", "system": "identity", "how": "identity", "chrf": 100.0, "desired": None, "draft_system": "identity", "draft_decoding": "identity", "draft_licence": "none",
                    "target_type": "human", "target_system": p.get("source_family", ""), "target_licence": "non_commercial" if p.get("non_commercial") else p.get("licence", "")})
        elif u < a.identity_share + a.near_share:
            d = perturb_light(p["reference"], [pid, a.seed])
            out.append({"pair_id": pid, "iso": iso, "split": p["split"], "english": p["english"], "draft": d, "reference": p["reference"], "kind": "near", "system": "near", "how": "near", "chrf": round(sentence_chrf(d, p["reference"]), 3), "desired": None, "draft_system": "near", "draft_decoding": "near", "draft_licence": "none",
                    "target_type": "human", "target_system": p.get("source_family", ""), "target_licence": "non_commercial" if p.get("non_commercial") else p.get("licence", "")})
    # the controls are drawn per pair, so their share of all examples is ~ (identity_share + near_share) * pairs / examples; rescale below if the tail multiplication changed it
    for e in out:
        e["keep"] = bool(a.keep_variant and e["chrf"] >= a.keep_threshold)
    return out


def matching_report(examples, bench_dist, path):
    import numpy as np
    by = defaultdict(list)
    for e in examples:
        if e["split"] == "train": by[e["iso"]].append(e)
    rows = []
    for iso, es in sorted(by.items()):
        tr = [e["chrf"] for e in es if e["kind"] == "matched"]
        b = bench_dist.get(iso, []) if bench_dist else []
        rows.append({"iso": iso, "train_examples": len(es), "train_matched_mean": round(float(np.mean(tr)), 1) if tr else "", "train_p10": pctl(tr, .1), "train_p50": pctl(tr, .5), "train_p90": pctl(tr, .9),
                     "bench_n": len(b), "bench_mean": round(float(np.mean(b)), 1) if b else "", "bench_p10": pctl(b, .1) if b else "", "bench_p50": pctl(b, .5) if b else "", "bench_p90": pctl(b, .9) if b else "",
                     "identity_share": round(sum(e["kind"] == "identity" for e in es) / len(es), 3), "near_share": round(sum(e["kind"] == "near" for e in es) / len(es), 3),
                     "share_draft_equals_reference": round(sum(e["draft"] == e["reference"] for e in es) / len(es), 3),
                     "share_corrupt": round(sum(e["system"] == "corrupt" for e in es if e["kind"] == "matched") / max(1, len(tr)), 3)})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["iso"]); w.writeheader(); w.writerows(rows)
    return rows


def write_packed(examples, out_dir, tok, seq_len, keep_marker, seed=0):
    """Tokenise with gemma_mt.post_edit.build_pe_example and write train/ + val/ in the packed format of gemma_mt.data.prepare_sft (one language per pack)."""
    import numpy as np
    from gemma_mt.data.prepare_sft import DTYPE, FORMAT_VERSION, PrepareConfig, _SampleStore, _write_split, tokenizer_fingerprint
    from gemma_mt.languages import ALL_ISOS, iso_to_id
    from gemma_mt import format as fmt
    from gemma_mt.post_edit import PE_FORMAT_VERSION, PE_INSTRUCTION_TEMPLATE, build_pe_example
    cfg = PrepareConfig(input="post_editor", out_dir=out_dir, seq_len=seq_len, max_sample_tokens=seq_len, pack_by_language=True, seed=seed, val_fraction=0.0)
    stop = fmt.resolve_stop_tokens(tok)
    tmp = os.path.join(out_dir, "_tmp_store"); dropped = defaultdict(int); meta_splits = {}
    for split in ("train", "val"):
        store = _SampleStore(os.path.join(tmp, split))
        for e in examples:
            if e["split"] != split: continue
            try:
                ex = build_pe_example(e["english"], e["draft"], e["iso"], e["reference"], tok, keep=e["keep"], keep_marker=keep_marker)
            except ValueError:
                dropped["format_error"] += 1; continue
            if ex.n_tokens > seq_len:
                dropped["too_long"] += 1; continue
            store.add(ex, iso_to_id(e["iso"]), 0)
        meta_splits[split] = _write_split(store, os.path.join(out_dir, split), cfg, split)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    meta = {"format_version": FORMAT_VERSION, "dtype": DTYPE, "seq_len": seq_len, "instruction_template": PE_INSTRUCTION_TEMPLATE, "post_edit_format_version": PE_FORMAT_VERSION,
            "end_of_turn": fmt.END_OF_TURN, "ignore_index": fmt.IGNORE_INDEX,
            "tokenizer": {"name": getattr(tok, "name_or_path", ""), "requested": getattr(tok, "name_or_path", ""), "revision": None, "fingerprint": tokenizer_fingerprint(tok), "vocab_size": len(tok),
                          "bos_id": tok.bos_token_id, "eos_id": tok.eos_token_id, "pad_id": tok.pad_token_id, "end_of_turn_id": stop.end_of_turn_id},
            "iso_table": list(ALL_ISOS), "directions": ["en2x"], "pivot_tag": fmt.PIVOT_ISO, "dropped": dict(dropped), "splits": meta_splits, "config": {"seq_len": seq_len, "pack_by_language": True}}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    return meta


def init_marker_rows(model_dir, out_dir, tokenizer, marker, init_text):
    """KEEP-marker variant only: set the embedding row of the marker token to the mean embedding of the pieces of `init_text` (the tied output row follows)."""
    import torch
    from gemma_mt.modeling import load_model, save_model
    model = load_model(model_dir)
    mid = tokenizer.convert_tokens_to_ids(marker)
    ids = tokenizer(init_text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        w = model.get_input_embeddings().weight
        w[mid] = w[torch.tensor(ids)].mean(0)
    save_model(model, tokenizer, out_dir)
    return {"marker": marker, "id": int(mid), "init_from_ids": ids}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs"); ap.add_argument("--drafts"); ap.add_argument("--out", required=True)
    ap.add_argument("--benchmark-predictions", default=None, help="predictions.jsonl of the SFT model on the benchmarks: the target draft-quality distribution (hyp vs reference) per language")
    ap.add_argument("--benchmarks", nargs="*", default=None, help="restrict the target distribution to these benchmarks (default: all rows)")
    ap.add_argument("--bench-dist", default=None, help="cache of the target distributions (json); written when --benchmark-predictions is given")
    ap.add_argument("--default-mean", type=float, default=30.0); ap.add_argument("--default-sd", type=float, default=18.0)
    ap.add_argument("--corrupt-share", type=float, default=0.15, help="share of pairs whose first draft is a corruption of the reference (system `corrupt`)"); ap.add_argument("--identity-share", type=float, default=0.07); ap.add_argument("--near-share", type=float, default=0.03)
    ap.add_argument("--multi-draft-below", type=int, default=300); ap.add_argument("--system-weights", default=None, help="json {system: prior weight} (default in the script)")
    ap.add_argument("--exclude-systems", nargs="*", default=["identity", "near"], help="draft systems to leave out; identity / near are excluded by default because the build adds those controls itself with --identity-share / --near-share (ablation: e.g. madlad3b_greedy madlad7b_greedy for the held-out-system editor)")
    ap.add_argument("--exclude-draft-licences", nargs="*", default=[], help="drop drafts whose licence contains this text (e.g. nc  -> removes the cc-by-nc-4.0 NLLB drafts)")
    ap.add_argument("--tag-mode", default="single_token", choices=["text", "single_token"], help="language tag of the target: ONE <unusedK> token (the 385M models) or the 3 letters as text (A0 270M)")
    ap.add_argument("--unseen-boost", type=float, default=3.0, help="prior weight multiplier of our own models' drafts on pairs the SFT / RL models never saw")
    ap.add_argument("--qe-filter", action="store_true", help="leave out the pairs flagged qe_drop (input = pairs_qe.jsonl)")
    ap.add_argument("--keep-variant", action="store_true"); ap.add_argument("--keep-threshold", type=float, default=97.0)
    ap.add_argument("--tokenizer", default="google/gemma-3-270m-it"); ap.add_argument("--fake-tokenizer", action="store_true"); ap.add_argument("--seq-len", type=int, default=2048); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-marker-rows", action="store_true"); ap.add_argument("--model", default=None); ap.add_argument("--marker", default=None); ap.add_argument("--init-text", default="same")
    a = ap.parse_args(argv)
    from gemma_mt import format as fmt0
    fmt0.set_tag_mode(a.tag_mode)
    from gemma_mt.post_edit import KEEP_MARKER, assert_marker_free
    if a.fake_tokenizer:
        from gemma_mt.fake_tokenizer import FakeGemmaTokenizer; tok = FakeGemmaTokenizer()
    else:
        from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained(a.tokenizer)
    marker = a.marker or KEEP_MARKER
    if a.init_marker_rows:
        print(json.dumps(init_marker_rows(a.model, a.out, tok, marker, a.init_text))); return
    if not a.marker: assert_marker_free()
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    pairs = read_jsonl(a.pairs); drafts = defaultdict(list)
    if a.qe_filter:                                              # pairs_qe.jsonl from post_editor_qe_filter.py: the flagged TRAIN pairs are left out (validation pairs are never filtered)
        n0 = len(pairs); pairs = [p for p in pairs if not p.get("qe_drop")]; print(f"qe filter: {n0 - len(pairs)} pairs dropped")
    for fn in sorted(os.listdir(a.drafts)):
        if fn.startswith("drafts_") and fn.endswith(".jsonl"):
            for d in read_jsonl(os.path.join(a.drafts, fn)): drafts[d["pair_id"]].append(d)
    if a.bench_dist and os.path.exists(a.bench_dist):
        bench = json.load(open(a.bench_dist))
    elif a.benchmark_predictions:
        bench = benchmark_chrf_distributions(a.benchmark_predictions, benchmarks=a.benchmarks, seed=a.seed)
        if a.bench_dist: json.dump(bench, open(a.bench_dist, "w"))
    else:
        bench = {}
    weights = dict(DEFAULT_SYSTEM_WEIGHTS); weights.update(json.load(open(a.system_weights)) if a.system_weights else {})
    ex = assemble(pairs, drafts, bench, a, weights)
    for split in ("train", "val"):
        with open(os.path.join(a.out, f"examples_{split}.jsonl"), "w", encoding="utf-8") as f:
            for e in ex:
                if e["split"] == split: f.write(json.dumps(e, ensure_ascii=False) + "\n")
    rep = matching_report(ex, bench, os.path.join(a.out, "matching_report.csv"))
    meta = write_packed(ex, a.out, tok, a.seq_len, marker, a.seed)
    tr = [e for e in ex if e["split"] == "train"]
    summ = {"pairs": len(pairs), "examples_train": len(tr), "examples_val": len(ex) - len(tr), "kinds_train": {k: sum(e["kind"] == k for e in tr) for k in ("matched", "identity", "near")},
            "systems_train": {s: sum(e["system"] == s for e in tr) for s in sorted({e["system"] for e in tr})},
            "draft_licence_share_train": {l: round(sum(e.get("draft_licence") == l for e in tr) / max(1, len(tr)), 4) for l in sorted({e.get("draft_licence") for e in tr}, key=str)},
            "nc_origin_draft_share_train": round(sum("nc" in str(e.get("draft_licence", "")).lower() for e in tr) / max(1, len(tr)), 4),
            "nc_target_share_train": round(sum(e.get("target_licence") == "non_commercial" for e in tr) / max(1, len(tr)), 4),
            "target_type_train": {k: sum(e.get("target_type") == k for e in tr) for k in ("human", "qe_high")}, "share_draft_equals_reference_train": round(sum(e["draft"] == e["reference"] for e in tr) / max(1, len(tr)), 4),
            "mean_chrf_train_matched": round(sum(e["chrf"] for e in tr if e["kind"] == "matched") / max(1, sum(e["kind"] == "matched" for e in tr)), 2),
            "mean_chrf_benchmark_drafts": round(sum(sum(v) for v in bench.values()) / max(1, sum(len(v) for v in bench.values())), 2) if bench else None,
            "pairs_without_any_draft": sum(p["pair_id"] not in drafts for p in pairs), "packed_train_samples": meta["splits"]["train"]["n_samples"], "packed_train_packs": meta["splits"]["train"]["n_packs"],
            "packed_train_tokens": meta["splits"]["train"]["n_tokens"], "packed_train_target_tokens": meta["splits"]["train"]["target_tokens"], "dropped": meta["dropped"], "seconds": round(time.time() - t0)}
    json.dump(summ, open(os.path.join(a.out, "build_summary.json"), "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
