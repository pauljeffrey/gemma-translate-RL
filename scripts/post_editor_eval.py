#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-editor, step 5: run the editor on (English source + draft) of a benchmark predictions file, score draft vs edited, and report what the editor did.

    python scripts/post_editor_eval.py --predictions /workspace/eval_385m/out_m385/predictions.jsonl --model EDITOR_DIR --out DIR \
        [--decoding greedy|beam4] [--benchmarks flores_plus ntrex mafand smol_sent salt glot_story] [--subsets all,clean,no_pair] [--limit-per-cell 0] \
        [--copy-control 600] [--lid-model model_v3.bin] [--decode-study-csv reports/data/eval_385m_results_2026-11-03/decoding_study20_per_language.csv] [--boot 1000]
    python scripts/post_editor_eval.py --score-only --predictions P --edited DIR/edited_predictions.jsonl --out DIR2          # re-score an existing run (CPU)

Input rows: the evaluation's predictions.jsonl (benchmark, iso, id, english [= source], reference, hyp [= draft; `hypothesis` also accepted], finished?, contaminated?, pair_seen?...).
The same file format comes out of `post_editor_make_drafts.py --summarize --val-system S` (benchmark `pe_val_*`: drafts of a held-out slice of the editor's OWN training sources; run the editor on it
as a second evaluation: it shows whether the editor helps where the SFT model's drafts are "too good").

Writes (OUT): edited_predictions.jsonl (same schema, hyp = edited text, plus draft / edited_status / edit_rate), metrics_by_language.csv (draft and edited), macro.csv, bootstrap_by_language.csv,
bootstrap_macro.csv (paired sentence bootstrap of edited - draft, equal weight per language, same resample for both), diagnostics.csv (unchanged share, edit-rate bins, hard failures before/after),
wins_losses.csv, copy_control.json (draft == reference rows: the false-edit rate), baselines.md (decode-study beam-4 / MBR-16 next to the editor on the same sentences), report.md.
"""
import argparse, csv, glob, json, os, sys, time, unicodedata
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
BENCHMARKS = ["flores_plus", "ntrex", "mafand", "smol_sent", "salt", "glot_story"]
nfc = lambda s: unicodedata.normalize("NFC", s or "")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def normalise_rows(rows):
    for r in rows:
        if "hyp" not in r: r["hyp"] = r.get("hypothesis", "")
        if "english" not in r: r["english"] = r.get("source", "")
        r["id"] = str(r["id"])
    return rows


def pick_rows(rows, benchmarks, limit, decode_study=False):
    rows = [r for r in rows if (not benchmarks or r["benchmark"] in benchmarks)]
    if limit:
        seen = defaultdict(int); out = []
        for r in rows:
            k = (r["benchmark"], r["iso"])
            if seen[k] < limit: out.append(r); seen[k] += 1
        rows = out
    return rows


# ----------------------------------------------------------------------------------------------- the editor run
def get_tok_backend(a):
    if a.fake_tokenizer:
        from gemma_mt.fake_tokenizer import FakeGemmaTokenizer; tok = FakeGemmaTokenizer()
    else:
        from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained(a.tokenizer)
    from gemma_mt.post_edit import make_backend
    backend = make_backend(a.backend, a.model, tok, a.dtype, a.gpu_util, a.max_model_len, a.batch_tokens)
    return tok, backend


def run_editor(rows, a, tok, backend):
    """-> list of (edited text, finished, status) aligned with rows. The draft is cut to --max-draft-tokens exactly as in training."""
    from gemma_mt.post_edit import build_pe_prompt, decode_gen
    prompts = [build_pe_prompt(r["english"], r["hyp"], r["iso"], tok, a.max_draft_tokens) for r in rows]
    gens = backend.greedy(prompts, a.max_new) if a.decoding == "greedy" else backend.beam(prompts, 4, a.max_new)
    out = []
    for r, g in zip(rows, gens):
        text, fin, st = decode_gen(tok, g, r["iso"])
        if text.strip() == a.keep_marker:                      # KEEP-marker variant: the draft is returned unchanged
            text, st = r["hyp"], "keep"
        out.append((text, fin, st))
    return out


def copy_control(rows, a, tok, backend, n):
    """Rows with draft == reference: how often does the editor change a correct draft, and what does it do to chrF++?"""
    import random
    from gemma_mt.post_edit import edit_rate, sentence_chrf
    rng = random.Random(0); pool = [r for r in rows if r["reference"].strip()]
    sel = rng.sample(pool, min(n, len(pool)))
    ctl = [dict(r, hyp=r["reference"]) for r in sel]
    res = run_editor(ctl, a, tok, backend)
    changed = [nfc(e).strip() != nfc(r["reference"]).strip() for r, (e, _, _) in zip(ctl, res)]
    chrf = [sentence_chrf(e, r["reference"]) for r, (e, _, _) in zip(ctl, res)]
    er = [edit_rate(r["reference"], e) for r, (e, _, _) in zip(ctl, res)]
    by = defaultdict(list)
    for r, c in zip(ctl, changed): by[r["iso"]].append(c)
    return {"n": len(ctl), "false_edit_rate": sum(changed) / max(1, len(changed)), "mean_chrf_after_edit_vs_reference": sum(chrf) / max(1, len(chrf)), "mean_edit_rate": sum(er) / max(1, len(er)),
            "false_edit_rate_by_language": {k: round(sum(v) / len(v), 3) for k, v in sorted(by.items())}}


# ----------------------------------------------------------------------------------------------- scoring
def flags(rows, key):
    from gemma_mt.rl.guards import is_copy_of_source, is_repetitive, is_self_repetitive
    for r in rows:
        h = r[key]
        r["_empty_" + key] = not h.strip()
        r["_loop_" + key] = bool(h.strip()) and (is_repetitive(h) or is_self_repetitive(h))
        r["_copy_" + key] = bool(h.strip()) and len(r["english"].split()) >= 4 and is_copy_of_source(r["english"], h, 0.9)


def lid_flags(rows, key, lid_model):
    import eval_385m_score as E
    tmp = [{"hyp": r[key], "iso": r["iso"]} for r in rows]
    E._lid(tmp, lid_model)
    for r, t in zip(rows, tmp): r["_lid_" + key] = None if t["_lid"] is None else bool(t["_lid"][0])


def diagnostics(rows, lid):
    from gemma_mt.post_edit import edit_rate
    out = []
    by = defaultdict(list)
    for r in rows: by[(r["benchmark"], r["iso"])].append(r)
    for (b, iso), rs in sorted(by.items()):
        n = len(rs); er = [r["edit_rate"] for r in rs]
        d = {"benchmark": b, "iso": iso, "n": n, "unchanged": sum(nfc(r["draft"]).strip() == nfc(r["hyp"]).strip() for r in rs) / n, "edit_rate_mean": sum(er) / n,
             "edit_0": sum(x == 0 for x in er) / n, "edit_0_10": sum(0 < x <= .1 for x in er) / n, "edit_10_30": sum(.1 < x <= .3 for x in er) / n, "edit_30_60": sum(.3 < x <= .6 for x in er) / n, "edit_60p": sum(x > .6 for x in er) / n,
             "draft_finished": sum(bool(r.get("finished_draft", True)) for r in rs) / n, "edited_finished": sum(bool(r.get("finished", True)) for r in rs) / n}
        for k in ("empty", "loop", "copy"):
            d[f"{k}_before"] = sum(r[f"_{k}_draft"] for r in rs) / n; d[f"{k}_after"] = sum(r[f"_{k}_hyp"] for r in rs) / n
        if lid:
            lb = [r["_lid_draft"] for r in rs if r["_lid_draft"] is not None]; la = [r["_lid_hyp"] for r in rs if r["_lid_hyp"] is not None]
            d["wrong_lang_before"] = (1 - sum(lb) / len(lb)) if lb else ""; d["wrong_lang_after"] = (1 - sum(la) / len(la)) if la else ""
        out.append(d)
    return out


def score(rows, a):
    """metrics per cell, paired bootstrap per cell and macro (edited minus draft)."""
    import numpy as np
    from multiprocessing import Pool
    import eval_385m_score as E
    from gemma_mt.eval import metrics as M
    subs = [s for s in a.subsets.split(",") if s]
    cells = defaultdict(list)
    for r in rows:
        for s in subs:
            if E.SUBSETS[s](r): cells[(r["benchmark"], s, r["iso"])].append(r)
    metr = []
    for (b, s, iso), rs in sorted(cells.items()):
        refs = [r["reference"] for r in rs]
        for name, key in (("draft", "draft"), ("edited", "hyp")):
            metr.append({"system": name, "benchmark": b, "subset": s, "iso": iso, "n": len(rs), "chrf": M.chrf_pp([r[key] for r in rs], refs), "spbleu": M.spbleu([r[key] for r in rs], refs)})
    jobs = [(("edited:draft"), b, s, iso, [r["draft"] for r in rs], [r["hyp"] for r in rs], [r["reference"] for r in rs], a.boot, a.seed + i)
            for i, ((b, s, iso), rs) in enumerate(sorted(cells.items())) if len(rs) >= a.min_n]
    with Pool(a.processes) as p: res = p.map(E.boot_cell, jobs, chunksize=2)
    per_lang = []
    for (pr, b, s, iso, n), o in res:
        per_lang.append({"benchmark": b, "subset": s, "iso": iso, "n": n, "d_chrf": o["chrf"][0], "lo": E.ci(o["chrf"][1])[0], "hi": E.ci(o["chrf"][1])[1], "d_spbleu": o["spbleu"][0],
                         "slo": E.ci(o["spbleu"][1])[0], "shi": E.ci(o["spbleu"][1])[1]})
    agg = defaultdict(list); isos_of = defaultdict(set); strata = getattr(a, "strata_map", None) or {}
    for (pr, b, s, iso, n), o in res:
        for lab in ("all", strata.get(iso)):
            if lab: agg[(b, s, lab)].append(o); isos_of[(b, s, lab)].add(iso)
    macro = []
    for (b, s, lab), os_ in sorted(agg.items()):
        m = {k: np.mean([o[k][1] for o in os_], axis=0) for k in ("chrf", "spbleu")}
        isos = isos_of[(b, s, lab)]
        d = [x for x in metr if x["benchmark"] == b and x["subset"] == s and x["n"] >= a.min_n and x["iso"] in isos]
        dm = float(np.mean([x["chrf"] for x in d if x["system"] == "draft"])); em = float(np.mean([x["chrf"] for x in d if x["system"] == "edited"]))
        dsb = float(np.mean([x["spbleu"] for x in d if x["system"] == "draft"])); esb = float(np.mean([x["spbleu"] for x in d if x["system"] == "edited"]))
        pl = [x for x in per_lang if x["benchmark"] == b and x["subset"] == s and x["iso"] in isos]
        macro.append({"benchmark": b if lab == "all" else f"{b} [{lab}]", "subset": s, "n_languages": len(os_), "chrf_draft": dm, "chrf_edited": em, "d_chrf": em - dm, "lo": E.ci(m["chrf"])[0], "hi": E.ci(m["chrf"])[1],
                      "spbleu_draft": dsb, "spbleu_edited": esb, "d_spbleu": esb - dsb, "slo": E.ci(m["spbleu"])[0], "shi": E.ci(m["spbleu"])[1],
                      "langs_better": sum(x["lo"] > 0 for x in pl), "langs_worse": sum(x["hi"] < 0 for x in pl), "langs_ns": sum(x["lo"] <= 0 <= x["hi"] for x in pl)})
    return metr, per_lang, macro


def sentence_wins(rows):
    from gemma_mt.post_edit import sentence_chrf
    by = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        a_, b_ = sentence_chrf(r["draft"], r["reference"]), sentence_chrf(r["hyp"], r["reference"])
        k = (r["benchmark"], r["iso"]); by[k][0 if b_ > a_ + 1e-9 else 1 if b_ < a_ - 1e-9 else 2] += 1
    return [{"benchmark": b, "iso": i, "n": sum(v), "edited_better": v[0], "edited_worse": v[1], "tie": v[2]} for (b, i), v in sorted(by.items())]


def decode_study_baselines(rows, csv_path, a):
    """Editor vs the decode study on the SAME sentences: FLORES+ devtest, the study's 20 languages, every 5th sentence (200 per language). The study's numbers (256-token cap) come from its CSV."""
    import eval_385m_score as E
    from gemma_mt.eval import metrics as M
    st = defaultdict(dict)
    for r in csv.DictReader(open(csv_path)):
        st[r["system"]][r["language"]] = float(r["chrf"])
    langs = sorted(st["greedy"])
    sub = [r for r in rows if r["benchmark"] == "flores_plus" and r["iso"] in langs and int(r["id"]) % 5 == 0]
    by = defaultdict(list)
    for r in sub: by[r["iso"]].append(r)
    ok = [l for l in langs if len(by[l]) >= 100]
    if not ok: return "No FLORES+ rows of the decode-study languages in this predictions file (needs every 5th devtest sentence).\n"
    d = sum(M.chrf_pp([r["draft"] for r in by[l]], [r["reference"] for r in by[l]]) for l in ok) / len(ok)
    e = sum(M.chrf_pp([r["hyp"] for r in by[l]], [r["reference"] for r in by[l]]) for l in ok) / len(ok)
    mac = lambda s: sum(st[s][l] for l in ok) / len(ok)
    lines = [f"Decode-study sentences (FLORES+ devtest, every 5th, {len(ok)} languages with >= 100 rows), macro chrF++:", "",
             "| system | chrF++ | cost relative to one greedy pass |", "|---|---:|---|",
             f"| draft in this predictions file (greedy, 768-token cap) | {d:.2f} | 1x |", f"| editor on that draft ({a.decoding}) | {e:.2f} | 1x draft + 1 editor pass (about 1.3-1.6x greedy time: the draft is read as prompt, the output is a full sentence) |",
             f"| decode study greedy (256-token cap) | {mac('greedy'):.2f} | 1x |", f"| decode study beam-4 | {mac('beam4'):.2f} | about 2-4x |", f"| decode study MBR-16 (16 samples + greedy, chrF++ utility) | {mac('mbr16'):.2f} | about 17x |", "",
             "Caveat: the study capped generation at 256 tokens (loops are truncated and MBR suppresses them); the predictions file uses 768. Compare the editor with `draft`, and use the study rows as the bar the editor has to reach at lower cost.", ""]
    return "\n".join(lines)


def write_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def baseline_tables(rows, edit_macro, a):
    """Baselines on the SAME sentences: each `--baseline name=path` (e.g. beam-4 and MBR-16 of the model that wrote the draft) scored against the draft, next to the editor."""
    if not a.baseline:
        return ""
    import copy
    key = lambda r: (r["benchmark"], r["iso"], str(r["id"]))
    out = ["", "### Editor vs equal-compute baselines (same sentences; delta vs the draft, macro over languages, 95% paired bootstrap)", "",
           "| system | benchmark | subset | chrF++ delta [CI] | spBLEU delta [CI] | langs better / ns / worse |", "|---|---|---|---|---|---|"]
    for m in edit_macro:
        out.append(f"| editor ({a.decoding}) | {m['benchmark']} | {m['subset']} | {m['d_chrf']:+.2f} [{m['lo']:+.2f}, {m['hi']:+.2f}] | {m['d_spbleu']:+.2f} [{m['slo']:+.2f}, {m['shi']:+.2f}] | {m['langs_better']} / {m['langs_ns']} / {m['langs_worse']} |")
    for spec in a.baseline:
        name, path = spec.split("=", 1)
        alt = {key(r): r for r in normalise_rows(read_jsonl(path))}
        rr = [dict(r, hyp=alt[key(r)]["hyp"]) for r in rows if key(r) in alt]
        if not rr: continue
        for r in rr: r["finished"] = alt[key(r)].get("finished", True)
        _, _, mac = score(rr, a)
        for m in mac:
            out.append(f"| {name} | {m['benchmark']} | {m['subset']} | {m['d_chrf']:+.2f} [{m['lo']:+.2f}, {m['hi']:+.2f}] | {m['d_spbleu']:+.2f} [{m['slo']:+.2f}, {m['shi']:+.2f}] | {m['langs_better']} / {m['langs_ns']} / {m['langs_worse']} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", nargs="+", required=True, help="one or several predictions.jsonl (draft sources); with several, give --labels and the engine is loaded once"); ap.add_argument("--out", required=True); ap.add_argument("--model", default=None)
    ap.add_argument("--edited", default=None); ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--decoding", default="greedy", choices=["greedy", "beam4"]); ap.add_argument("--benchmarks", nargs="*", default=BENCHMARKS); ap.add_argument("--subsets", default="all,clean,no_pair")
    ap.add_argument("--limit-per-cell", type=int, default=0); ap.add_argument("--copy-control", type=int, default=0); ap.add_argument("--lid-model", default=None); ap.add_argument("--no-lid", action="store_true")
    ap.add_argument("--decode-study-csv", default=None); ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--min-n", type=int, default=30); ap.add_argument("--processes", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="google/gemma-3-270m-it"); ap.add_argument("--fake-tokenizer", action="store_true"); ap.add_argument("--backend", default="vllm", choices=["vllm", "hf", "hfgpu"]); ap.add_argument("--batch-tokens", type=int, default=100000)
    ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--gpu-util", type=float, default=0.25); ap.add_argument("--max-model-len", type=int, default=2048); ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--labels", nargs="*", default=None); ap.add_argument("--strata-csv", default=None, help="coverage_by_language.csv: macros are also reported for religious-only / religious-mostly (> 50%) / mixed-register languages"); ap.add_argument("--baseline", action="append", default=[], metavar="NAME=predictions.jsonl", help="other hypotheses for the SAME sentences (beam-4, MBR-16 of the same model): scored against the draft"); ap.add_argument("--max-draft-tokens", type=int, default=192); ap.add_argument("--keep-marker", default="<unused6150>")
    a = ap.parse_args(argv)
    a.strata_map = {}
    if a.strata_csv:
        for r in csv.DictReader(open(a.strata_csv)):
            if r.get("chosen") and int(r["chosen"]) > 0:
                a.strata_map[r["iso"]] = "religious-only" if r.get("religious_only") == "1" else ("religious-mostly" if r.get("religious_share") and float(r["religious_share"]) > 0.5 else "mixed-register")
    labels = a.labels or [os.path.splitext(os.path.basename(p))[0] for p in a.predictions]
    tok = backend = None
    if not a.score_only:
        tok, backend = get_tok_backend(a)
    for pred, lab in zip(a.predictions, labels):
        run_one(a, pred, os.path.join(a.out, lab) if len(a.predictions) > 1 else a.out, tok, backend)


def run_one(a, pred, out, tok, backend):
    os.makedirs(out, exist_ok=True); t0 = time.time()
    rows = pick_rows(normalise_rows(read_jsonl(pred)), a.benchmarks, a.limit_per_cell)
    for r in rows: r["draft"] = r["hyp"]; r["finished_draft"] = r.get("finished", True)
    if a.score_only:
        ed = {(r["benchmark"], r["iso"], str(r["id"])): r for r in read_jsonl(a.edited)}
        rows = [dict(r, hyp=ed[(r["benchmark"], r["iso"], r["id"])]["hyp"], finished=ed[(r["benchmark"], r["iso"], r["id"])].get("finished", True), edited_status=ed[(r["benchmark"], r["iso"], r["id"])].get("edited_status", "ok")) for r in rows if (r["benchmark"], r["iso"], r["id"]) in ed]
    else:
        res = run_editor(rows, a, tok, backend)
        for r, (e, fin, st) in zip(rows, res): r["hyp"], r["finished"], r["edited_status"] = e, fin, st
    from gemma_mt.post_edit import edit_rate
    for r in rows: r["edit_rate"] = edit_rate(r["draft"], r["hyp"])
    with open(os.path.join(out, "edited_predictions.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            o = {k: v for k, v in r.items() if not k.startswith("_")}
            o["system"] = "editor_" + a.decoding; f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"[{time.time()-t0:.0f}s] editor outputs: {len(rows)} rows", flush=True)
    flags(rows, "draft"); flags(rows, "hyp")
    lid = None
    if not a.no_lid:
        lid = a.lid_model or next(iter(sorted(glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub/models--cis-lmu--glotlid/snapshots/*/model_v3.bin")))), None)
        if lid:
            lid_flags(rows, "draft", lid); lid_flags(rows, "hyp", lid)
        else:
            print("no GlotLID model found: wrong-language rates skipped (pass --lid-model)", flush=True)
    diag = diagnostics(rows, lid)
    write_csv(os.path.join(out, "diagnostics.csv"), diag)
    metr, per_lang, macro = score(rows, a)
    write_csv(os.path.join(out, "metrics_by_language.csv"), metr); write_csv(os.path.join(out, "bootstrap_by_language.csv"), per_lang); write_csv(os.path.join(out, "bootstrap_macro.csv"), macro)
    wl = sentence_wins(rows); write_csv(os.path.join(out, "wins_losses.csv"), wl)
    cc = None
    if a.copy_control and not a.score_only:
        cc = copy_control(rows, a, tok, backend, a.copy_control); json.dump(cc, open(os.path.join(out, "copy_control.json"), "w"), indent=1)
    base = decode_study_baselines(rows, a.decode_study_csv, a) if a.decode_study_csv else "(no --decode-study-csv)\n"
    base += baseline_tables(rows, macro, a)
    open(os.path.join(out, "baselines.md"), "w").write(base)
    # ---- report
    L = [f"# Post-editor evaluation ({pred}, editor {a.model or a.edited}, {a.decoding})", "", "## Paired comparison edited - draft (macro over languages, equal weight; 95% paired sentence bootstrap)", "",
         "| benchmark | subset | languages | chrF++ draft | chrF++ edited | delta [CI] | spBLEU delta [CI] | langs better / ns / worse |", "|---|---|---:|---:|---:|---|---|---|"]
    for m in macro:
        L.append(f"| {m['benchmark']} | {m['subset']} | {m['n_languages']} | {m['chrf_draft']:.2f} | {m['chrf_edited']:.2f} | {m['d_chrf']:+.2f} [{m['lo']:+.2f}, {m['hi']:+.2f}] | {m['d_spbleu']:+.2f} [{m['slo']:+.2f}, {m['shi']:+.2f}] | {m['langs_better']} / {m['langs_ns']} / {m['langs_worse']} |")
    import numpy as np
    keys = ["unchanged", "edit_rate_mean", "empty_before", "empty_after", "loop_before", "loop_after", "copy_before", "copy_after"] + (["wrong_lang_before", "wrong_lang_after"] if lid else [])
    L += ["", "## What the editor did (mean over benchmark x language cells)", "", "| " + " | ".join(keys) + " |", "|" + "---:|" * len(keys),
          "| " + " | ".join(f"{np.mean([d[k] for d in diag if d.get(k, '') != '']):.3f}" for k in keys) + " |", ""]
    tot = [sum(w["edited_better"] for w in wl), sum(w["edited_worse"] for w in wl), sum(w["tie"] for w in wl)]
    L += [f"Sentence-level (chrF++): edited better {tot[0]}, worse {tot[1]}, tie {tot[2]}.", ""]
    if cc: L += ["## Copy control (draft == reference)", "", f"false-edit rate {cc['false_edit_rate']:.3f} on {cc['n']} rows; chrF++ of the edited text vs the reference {cc['mean_chrf_after_edit_vs_reference']:.1f}.", ""]
    L += ["## Baselines", "", base]
    open(os.path.join(out, "report.md"), "w").write("\n".join(L))
    print("\n".join(L[:30])); print(f"[{time.time()-t0:.0f}s] done -> {out}")


if __name__ == "__main__":
    main()
