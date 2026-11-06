#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-editor: DRAFTS from other machine-translation systems (NLLB-200, MADLAD-400) so the editor sees error patterns that are not our own model's. DRAFT side only: the target stays the human reference.

    python scripts/post_editor_mt_drafts.py --pairs OUT/pairs.jsonl --out DRAFTS_DIR --plan "nllb3.3b:greedy,beam4;madlad7b:greedy;nllb600m:greedy,sample0.8" [--per-language 160] [--splits train val]
    python scripts/post_editor_mt_drafts.py --pred-file benchmark_predictions.jsonl --out DIR --plan "nllb3.3b:greedy;madlad7b:greedy"       # benchmark drafts of the same systems (same rows, evaluation format)

Systems (HF ids; bf16; HF `generate` with token-budget batches sorted by length; vLLM does not run these models):
  nllb600m facebook/nllb-200-distilled-600M    nllb1.3b facebook/nllb-200-distilled-1.3B    nllb3.3b facebook/nllb-200-3.3B        licence CC-BY-NC-4.0  (NON-COMMERCIAL: rows are tagged draft_licence=cc-by-nc-4.0)
  madlad3b google/madlad400-3b-mt             madlad7b google/madlad400-7b-mt                                                       licence apache-2.0 (model card; the data CC-BY-4.0 / ODC)
Decodings: greedy | beam4 | sampleT (e.g. sample0.8: one sample, top-p 0.95). Language support = the target-language token exists in the tokenizer (NLLB: FLORES code; MADLAD: <2xx>); recorded in
DIR/mt_coverage.csv (language x system). Writes DIR/drafts_<system>_<decoding>.jsonl with the fields of post_editor_make_drafts.py plus `licence`, `decoding`, `model`.
"""
import argparse, csv, json, os, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
MODELS = {
    "nllb600m": ("facebook/nllb-200-distilled-600M", "nllb", "cc-by-nc-4.0"), "nllb1.3b": ("facebook/nllb-200-distilled-1.3B", "nllb", "cc-by-nc-4.0"),
    "nllb3.3b": ("facebook/nllb-200-3.3B", "nllb", "cc-by-nc-4.0"), "madlad3b": ("google/madlad400-3b-mt", "madlad", "apache-2.0"), "madlad7b": ("google/madlad400-7b-mt", "madlad", "apache-2.0"),
}
ISO1 = {"afr": "af", "amh": "am", "hau": "ha", "ibo": "ig", "kin": "rw", "lin": "ln", "lug": "lg", "mlg": "mg", "nya": "ny", "orm": "om", "sna": "sn", "som": "so", "sot": "st", "ssw": "ss", "swh": "sw",
        "tir": "ti", "tsn": "tn", "tso": "ts", "twi": "tw", "wol": "wo", "xho": "xh", "yor": "yo", "zul": "zu", "ewe": "ee", "bam": "bm", "ful": "ff", "ven": "ve", "kon": "kg", "fon": "fon", "kik": "ki",
        "run": "rn", "sag": "sg", "luo": "luo", "bem": "bem", "nso": "nso", "tiv": "tiv", "lua": "lua", "tum": "tum", "umb": "umb", "kmb": "kmb", "kam": "kam", "dyu": "dyu", "kbp": "kbp", "aar": "aa"}
NLLB_EXTRA = {"twi": "twi_Latn", "orm": "gaz_Latn", "mlg": "plt_Latn", "ful": "fuv_Latn", "swh": "swh_Latn", "tzm": "tzm_Tfng", "shi": "taq_Tfng"}      # codes of the NLLB tokenizer where FLORES+ names differ


def nllb_code(iso, tok_vocab):
    from gemma_mt.languages import LANGUAGES
    cands = [NLLB_EXTRA.get(iso), getattr(LANGUAGES.get(iso), "flores", None), f"{iso}_Latn", f"{iso}_Ethi", f"{iso}_Arab"]
    for c in cands:
        if c and c in tok_vocab: return c
    return None


def madlad_code(iso, tok_vocab):
    for c in (ISO1.get(iso), iso):
        if c and f"<2{c}>" in tok_vocab: return f"<2{c}>"
    return None


def load_rows(a):
    if a.pred_file:
        rows = [json.loads(l) for l in open(a.pred_file, encoding="utf-8") if l.strip()]
        for r in rows:
            r["pair_id"] = f"{r['benchmark']}|{r['iso']}|{r['id']}"
        return rows, "pred"
    from post_editor_make_drafts import load_pairs
    from gemma_mt.post_edit import stable_unit
    rows = load_pairs(a.pairs, a.splits)
    if a.per_language:
        by = defaultdict(list)
        for r in rows: by[r["iso"]].append(r)
        keep = []
        for iso, v in by.items():
            tr = sorted([r for r in v if r["split"] == "train"], key=lambda r: stable_unit("mtpool", r["pair_id"]))[: a.per_language]
            keep += tr + [r for r in v if r["split"] == "val"]
        rows = keep
    return rows, "pairs"


def translate(model, tok, kind, texts, code, decoding, device, max_tokens_batch=6000, seed=0):
    """-> list of (text, finished) for `texts` (one target language). Batches by token budget, longest first."""
    import torch
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out = [None] * len(texts); i = 0
    gen_kw = {"greedy": dict(num_beams=1, do_sample=False), "beam4": dict(num_beams=4, do_sample=False, early_stopping=True)}
    if decoding.startswith("sample"):
        gen_kw[decoding] = dict(num_beams=1, do_sample=True, temperature=float(decoding[6:] or 0.8), top_p=0.95, top_k=0)
    torch.manual_seed(seed)
    while i < len(order):
        L = max(8, len(texts[order[i]]) // 3)
        bs = max(1, min(256, max_tokens_batch // (L * (4 if decoding == "beam4" else 1))))
        idx = order[i:i + bs]; i += bs
        batch = [texts[j] for j in idx]
        if kind == "madlad":
            batch = [f"{code} {t}" for t in batch]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        kw = dict(max_new_tokens=min(256, int(enc["input_ids"].shape[1] * 2.5) + 16), **gen_kw[decoding])
        if kind == "nllb":
            kw["forced_bos_token_id"] = tok.convert_tokens_to_ids(code)
        with torch.no_grad():
            g = model.generate(**enc, **kw)
        txt = tok.batch_decode(g, skip_special_tokens=True)
        for j, t, row in zip(idx, txt, g):
            out[j] = (t.strip(), bool((row == tok.eos_token_id).any()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs"); ap.add_argument("--pred-file"); ap.add_argument("--out", required=True); ap.add_argument("--plan", required=True); ap.add_argument("--per-language", type=int, default=0)
    ap.add_argument("--splits", nargs="*", default=None); ap.add_argument("--isos", nargs="*", default=None); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max-tokens-batch", type=int, default=6000)
    ap.add_argument("--time-budget-min", type=float, default=0, help="stop starting new (system, decoding) jobs after this many minutes (the drafts written so far stay)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from gemma_mt.post_edit import sentence_chrf
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    rows, mode = load_rows(a)
    if a.isos: rows = [r for r in rows if r["iso"] in a.isos]
    if a.limit: rows = rows[: a.limit]
    by_iso = defaultdict(list)
    for i, r in enumerate(rows): by_iso[r["iso"]].append(i)
    cov = []
    dev = "cuda"
    for item in a.plan.split(";"):
        name, _, decs = item.partition(":")
        repo, kind, lic = MODELS[name]
        if a.time_budget_min and (time.time() - t0) / 60 > a.time_budget_min:
            print("time budget reached: skipping", name, flush=True); continue
        tok = AutoTokenizer.from_pretrained(repo, src_lang="eng_Latn") if kind == "nllb" else AutoTokenizer.from_pretrained(repo)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo, dtype=torch.bfloat16).to(dev).eval()
        vocab = set(tok.get_vocab()) if kind == "madlad" else set(tok.get_vocab())
        codes = {iso: (nllb_code(iso, vocab) if kind == "nllb" else madlad_code(iso, vocab)) for iso in by_iso}
        cov += [{"system": name, "iso": iso, "code": c or "", "supported": int(bool(c)), "n_rows": len(by_iso[iso])} for iso, c in sorted(codes.items())]
        print(f"[{name}] loaded in {time.time()-t0:.0f}s; supported {sum(bool(c) for c in codes.values())}/{len(codes)} languages", flush=True)
        for dec in decs.split(","):
            if a.time_budget_min and (time.time() - t0) / 60 > a.time_budget_min: break
            t1 = time.time(); outs = [None] * len(rows)
            for iso, idx in sorted(by_iso.items()):
                if not codes[iso]: continue
                res = translate(model, tok, kind, [rows[i]["english"] for i in idx], codes[iso], dec, dev, a.max_tokens_batch, a.seed)
                for i, r_ in zip(idx, res): outs[i] = r_
            fname = f"drafts_{name}_{dec}.jsonl" if mode == "pairs" else f"predictions_{name}_{dec}.jsonl"
            with open(os.path.join(a.out, fname), "w", encoding="utf-8") as f:
                for r, o in zip(rows, outs):
                    if o is None: continue
                    if mode == "pairs":
                        f.write(json.dumps({"pair_id": r["pair_id"], "system": f"{name}_{dec}", "how": dec, "decoding": dec, "model": repo, "licence": lic, "draft": o[0], "finished": o[1], "tag_status": "ok",
                                            "chrf": round(sentence_chrf(o[0], r["reference"]), 3)}, ensure_ascii=False) + "\n")
                    else:
                        d = {k: v for k, v in r.items() if k != "pair_id"}; d.update({"system": f"{name}_{dec}", "hyp": o[0], "finished": o[1], "licence": lic}); f.write(json.dumps(d, ensure_ascii=False) + "\n")
            print(f"[{name}/{dec}] {sum(o is not None for o in outs)} drafts in {time.time()-t1:.0f}s -> {fname}", flush=True)
        del model; torch.cuda.empty_cache()
    with open(os.path.join(a.out, "mt_coverage.csv"), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "iso", "code", "supported", "n_rows"])
        if f.tell() == 0: w.writeheader()
        w.writerows(cov)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
