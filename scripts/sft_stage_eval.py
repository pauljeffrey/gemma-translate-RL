#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Evaluation harness of the short science-SFT stage (D-58): before / after tables, forgetting check, weight-interpolation sweep.

    # one model, three suites, ONE vLLM engine (greedy, English -> X):
    python scripts/sft_stage_eval.py run --model DIR --name final_1433 --out OUT/final_1433 --vllm-python /workspace/venv_vllm/bin/python
    # WiSE-FT / model-soup sweep: theta = (1-a) theta_init + a theta_sft for every a, evaluated one after the other (the HF dir of an
    # alpha is deleted after its evaluation unless --keep), results in OUT/alpha_0.30/ ...
    python scripts/sft_stage_eval.py sweep --init /workspace/sft_build/runs/final_1433 --sft RUN/last/trainer_state.pt --alphas 0.3 0.5 0.7 1.0 --out OUT ...
    # tables (markdown + json) from finished runs:
    python scripts/sft_stage_eval.py compare --baseline OUT/final_1433 --runs OUT/alpha_0.30 OUT/alpha_0.50 ... --out OUT/report.md

Suites (all greedy, new tokens capped at 512 for a sentence and 7,168 for a whole paper):
  afs      AfriScience-MT DEV (843 sentences + 25 papers per language, amh hau lug nso yor zul): chrF, chrF++, BLEU, spBLEU, finished rate
           (`scripts/eval_afriscience.py` functions, so the numbers are comparable with agent A4's baseline). --splits dev,test adds the TEST split:
           use it ONCE, for the winning alpha, after the choice has been made on dev.
  general  `bench_dev_clean.jsonl` (the clean general dev set: 19 languages, ~10k sentences): chrF++ per language = the FORGETTING check on the
           languages the model was trained to translate well.
  weak     FLORES+ DEV, 60 fixed sentences (same ids in every language, 44 languages that have a FLORES+ dev file) + for the weak-signal languages
           without FLORES+ dev (RL consensus-only / blend groups) 60 reference sentences of the rl-vs-sft evaluation pool: chrF++ per language.
The language groups of the report: science6 (amh hau lug nso yor zul), consensus-only (23), blend (22), control (10) from the RL-vs-SFT check,
other.  Nothing here reads the AfriScience TEST split unless --splits says so.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SCI6 = ("amh", "hau", "lug", "nso", "yor", "zul")
CONSENSUS = ["ibb", "urh", "nbl", "bin", "dag", "oke", "shi", "bbj", "kea", "ach", "tsc", "vmw", "seh", "nyn", "her", "yao", "sid", "ndc", "kri", "ish", "dua", "bas", "wes"]
BLEND = ["efi", "bam", "fon", "kab", "kam", "kbp", "kmb", "mos", "nus", "umb", "dyu", "sag", "swc", "loz", "ven", "iso", "tiv", "kqn", "wal", "lun", "kua", "kon"]
CONTROL = ["yor", "hau", "swh", "kin", "wol", "zul", "ibo", "xho", "som", "sna"]
WEAK45 = CONSENSUS + BLEND
FLORES_SNAP = "/workspace/sft_build/hf/hub/datasets--openlanguagedata--flores_plus/snapshots/*"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def group_of(iso: str) -> str:
    if iso in SCI6:
        return "science6"
    if iso in CONSENSUS:
        return "weak-consensus"
    if iso in BLEND:
        return "weak-blend"
    if iso in CONTROL:
        return "control"
    return "other"


# --------------------------------------------------------------------------------------------------------------------------- eval sets
def _jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_weak_set(n: int = 60, flores_snap: str = FLORES_SNAP, pool_path: str = "/workspace/rlcheck/res/pool.jsonl", seed: int = 7) -> List[dict]:
    """rows {iso, english, reference, suite, id}: FLORES+ dev (the n ids with the smallest blake2 hash, identical for all languages) + pool rows for the
    weak-signal languages that have no FLORES+ dev file."""
    from gemma_mt.languages import ALL_ISOS, LANGUAGES

    snap = glob.glob(flores_snap)[0]
    en = {int(r["id"]): r["text"] for r in _jsonl(os.path.join(snap, "dev", "eng_Latn.jsonl"))}
    key = lambda i: hashlib.blake2b(f"flores-dev-60|{seed}|{i}".encode(), digest_size=8).hexdigest()
    ids = sorted(en, key=key)
    rows, have = [], set()
    for iso in ALL_ISOS:
        code = LANGUAGES[iso].flores
        p = os.path.join(snap, "dev", f"{code}.jsonl") if code else None
        if not p or not os.path.exists(p):
            continue
        ref = {int(r["id"]): r["text"] for r in _jsonl(p)}
        got = [i for i in ids if i in ref and ref[i].strip()][:n]
        rows += [{"iso": iso, "english": en[i], "reference": ref[i].strip(), "suite": "flores_dev", "id": f"flores-{i}"} for i in got]
        have.add(iso)
    if pool_path and os.path.exists(pool_path):
        by: Dict[str, List[dict]] = {}
        for r in _jsonl(pool_path):
            if r["iso"] in WEAK45 and r["iso"] not in have and r.get("reference"):
                by.setdefault(r["iso"], []).append(r)
        for iso, rs in sorted(by.items()):
            rs.sort(key=lambda r: hashlib.blake2b(f"pool60|{seed}|{r['bench']}|{r['id']}".encode(), digest_size=8).hexdigest())
            rows += [{"iso": iso, "english": r["english"], "reference": r["reference"], "suite": "pool_" + r["bench"], "id": f"{r['bench']}-{r['id']}"} for r in rs[:n]]
    return rows


def chrf_pp_scores(hyps: Sequence[str], refs: Sequence[str]) -> Dict[str, float]:
    from sacrebleu.metrics import CHRF
    import unicodedata

    h = [unicodedata.normalize("NFC", x) for x in hyps]
    r = [unicodedata.normalize("NFC", x) for x in refs]
    return {"chrf_pp": float(CHRF(word_order=2).corpus_score(h, [r]).score), "chrf": float(CHRF(word_order=0).corpus_score(h, [r]).score)}


def score_rows(rows: List[dict], outs: List[dict]) -> Dict[str, Any]:
    by: Dict[str, Tuple[List[str], List[str], List[bool]]] = {}
    for r, o in zip(rows, outs):
        h, rf, fin = by.setdefault(r["iso"], ([], [], []))
        h.append(o["hyp"]); rf.append(r["reference"]); fin.append(bool(o.get("finished", True)))
    per = {}
    for iso, (h, rf, fin) in by.items():
        m = chrf_pp_scores(h, rf)
        m.update({"n": len(h), "finished_rate": sum(fin) / max(len(fin), 1), "empty_rate": sum(not x.strip() for x in h) / max(len(h), 1)})
        per[iso] = m
    return per


# --------------------------------------------------------------------------------------------------------------------------- run
def make_sampler(path: str, a, log_path: str, tries: int = 3):
    """vLLM worker engine. On a SHARED card vLLM's memory profiler can trip over other jobs' allocations ("Error in memory profiling ... other processes
    release GPU memory"); a fixed KV cache (--kv-cache-gib) skips that profile, and the start-up is retried."""
    from gemma_mt.rl.vllm_engine import EngineConfig, SamplerError, WorkerEngine
    from gemma_mt.rl.vllm_sampler import VLLMSampler

    ec = EngineConfig(model=path, gpu_memory_utilization=a.vllm_gpu_util, kv_cache_gib=(a.kv_cache_gib or None), max_model_len=a.vllm_max_model_len,
                      max_num_seqs=a.vllm_max_num_seqs, seed=0)
    last: Optional[Exception] = None
    for k in range(tries):
        try:
            engine = WorkerEngine(ec, python=a.vllm_python, log_path=log_path, sync_dir="/dev/shm" if os.path.isdir("/dev/shm") else "/tmp",
                                  startup_timeout=900.0, request_timeout=3600.0)
            s = VLLMSampler(engine, seed=0)
            s.assume_loaded(0)
            return s
        except SamplerError as e:
            last = e
            log(f"vLLM start-up failed (try {k + 1}/{tries}): {str(e)[:200]}")
            time.sleep(20)
    raise last  # type: ignore[misc]


def run_model(a) -> int:
    import eval_afriscience as E
    from gemma_mt.eval import runtime as RT

    RT.add_extra_libs(a.extra_libs)
    os.makedirs(a.out, exist_ok=True)
    suites = [s for s in a.suites.split(",") if s]
    ref = RT.resolve_model(a.model)
    tok, stop, pad = RT.load_tokenizer_and_stops(ref.path)
    name = a.name or os.path.basename(os.path.normpath(a.model))
    dec = E.parse_decoding("greedy")
    RT.wait_for_gpu(a.vllm_gpu_util * 96.0 * 0.9, a.gpu_wait)
    sampler = make_sampler(ref.path, a, os.path.join(a.out, "vllm_worker.log"))
    res: Dict[str, Any] = {"name": name, "model": a.model, "decoding": "greedy", "suites": {}}
    t0 = time.time()
    try:
        gen = lambda items, mnt: E.generate(sampler, tok, stop.ids, pad, items, dec, mnt, log=RT.log)
        if "afs" in suites:
            data = {sp: E.load_corpus(a.afs_corpus, sp, E.LANGS) for sp in a.splits.split(",") if sp}
            if a.limit:                                                  # smoke tests only
                for corpus in data.values():
                    for blk in corpus.values():
                        blk["sentences"], blk["documents"] = blk["sentences"][:a.limit], blk["documents"][:2]
            res["suites"]["afs"] = {}
            for sp, corpus in data.items():
                class _A:  # noqa: N801
                    max_new_tokens_sentence = a.max_new_tokens_sentence
                    max_new_tokens_document = a.max_new_tokens_document
                r = E.run_split(corpus, ("sentence", "document"), gen, _A)
                res["suites"]["afs"][sp] = r["scores"]
                with open(os.path.join(a.out, f"afs_predictions_{sp}.json"), "w", encoding="utf-8") as f:
                    json.dump(r["predictions"], f, ensure_ascii=False)
                log(f"afs {sp} done ({time.time() - t0:.0f}s)")
        for suite, rows in (("general", None), ("weak", None), ("ibom", None)):
            if suite not in suites:
                continue
            if suite == "ibom":                      # D-59: IbomMT TEST (= FLORES+ devtest English, bench/test_sets/ibom_mt/en_<iso>.jsonl: id, src, trg) for ibb / efi; test data: use for before/after reports only
                rows = []
                for iso in ("ibb", "efi"):
                    for r in _jsonl(os.path.join(a.ibom_dir, f"en_{iso}.jsonl")):
                        rows.append({"iso": iso, "english": r["src"], "reference": r["trg"]})
            else:
                rows = _jsonl(a.general_dev) if suite == "general" else build_weak_set(a.weak_n, pool_path=a.eval_pool)
            if suite == "general" and a.general_max_per_language:
                cnt: Dict[str, int] = {}
                keep = []
                for r in rows:
                    cnt[r["iso"]] = cnt.get(r["iso"], 0) + 1
                    if cnt[r["iso"]] <= a.general_max_per_language:
                        keep.append(r)
                rows = keep
            if a.limit:
                seen: Dict[str, int] = {}
                rows = [r for r in rows if seen.__setitem__(r["iso"], seen.get(r["iso"], 0) + 1) or seen[r["iso"]] <= a.limit]
            outs = gen([(r["english"], r["iso"]) for r in rows], a.max_new_tokens_sentence)
            res["suites"][suite] = {"per_language": score_rows(rows, outs), "n_rows": len(rows)}
            with open(os.path.join(a.out, f"{suite}_predictions.jsonl"), "w", encoding="utf-8") as f:
                for r, o in zip(rows, outs):
                    f.write(json.dumps({**{k: r[k] for k in ("iso", "english", "reference")}, "hyp": o["hyp"], "finished": o["finished"]}, ensure_ascii=False) + "\n")
            log(f"{suite}: {len(rows)} rows ({time.time() - t0:.0f}s)")
    finally:
        sampler.close()
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "stage_eval.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    log(f"wrote {a.out}/stage_eval.json")
    return 0


def sweep(a) -> int:
    from interpolate_weights import interpolate_to_dir

    os.makedirs(a.out, exist_ok=True)
    for alpha in a.alphas:
        tag = f"alpha_{alpha:.2f}"
        out = os.path.join(a.out, tag)
        if os.path.exists(os.path.join(out, "stage_eval.json")) and not a.force:
            log(f"{tag}: already evaluated"); continue
        mdir = os.path.join(a.work, tag)
        interpolate_to_dir(a.init, a.sft, alpha, mdir)
        ns = argparse.Namespace(**{**vars(a), "model": mdir, "name": tag, "out": out})
        try:
            run_model(ns)
        finally:
            if not a.keep:
                shutil.rmtree(mdir, ignore_errors=True)
    return 0


# --------------------------------------------------------------------------------------------------------------------------- compare
def _boot_ci(deltas: List[float], n: int = 4000, seed: int = 0) -> Tuple[float, float]:
    if len(deltas) < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    ms = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def _load(d: str) -> Dict[str, Any]:
    return json.load(open(os.path.join(d, "stage_eval.json"), encoding="utf-8"))


def compare(a) -> int:
    base = _load(a.baseline)
    runs = [(os.path.basename(os.path.normpath(d)), _load(d)) for d in a.runs]
    L: List[str] = []
    names = [base["name"]] + [r[1]["name"] for r in runs]

    def table_lang(title: str, getter, langs: Sequence[str], fmt_="{:.2f}") -> None:
        L.append(f"### {title}\n")
        L.append("| lang | group | " + " | ".join(names) + " | " + " | ".join(f"d {n}" for n in names[1:]) + " |")
        L.append("|---|---|" + "---|" * (2 * len(names) - 1))
        for iso in langs:
            b = getter(base, iso)
            vals = [getter(r, iso) for _, r in runs]
            if b is None:
                continue
            cells = [fmt_.format(b)] + ["-" if v is None else fmt_.format(v) for v in vals]
            ds = ["-" if v is None else f"{v - b:+.2f}" for v in vals]
            L.append(f"| {iso} | {group_of(iso)} | " + " | ".join(cells) + " | " + " | ".join(ds) + " |")
        L.append("")

    def group_table(title: str, getter, langs: Sequence[str]) -> None:
        L.append(f"### {title} (macro over languages; delta CI = bootstrap over languages)\n")
        L.append("| group | n | " + " | ".join(names) + " | " + " | ".join(f"delta {n} [95% CI]" for n in names[1:]) + " |")
        L.append("|---|---|" + "---|" * (2 * len(names) - 1))
        groups = ["science6", "weak-consensus", "weak-blend", "control", "other", "ALL"]
        for g in groups:
            ls = [i for i in langs if (g == "ALL" or group_of(i) == g) and getter(base, i) is not None]
            if not ls:
                continue
            bm = sum(getter(base, i) for i in ls) / len(ls)
            row = [f"{bm:.2f}"]
            dl = []
            for _, r in runs:
                ok = [i for i in ls if getter(r, i) is not None]
                if not ok:
                    row.append("-"); dl.append("-"); continue
                m = sum(getter(r, i) for i in ok) / len(ok)
                d = [getter(r, i) - getter(base, i) for i in ok]
                lo, hi = _boot_ci(d)
                row.append(f"{m:.2f}")
                dl.append(f"{m - sum(getter(base, i) for i in ok) / len(ok):+.2f} [{lo:+.2f}, {hi:+.2f}]")
            L.append(f"| {g} | {len(ls)} | " + " | ".join(row) + " | " + " | ".join(dl) + " |")
        L.append("")

    L.append(f"# Science-SFT stage: before / after ({', '.join(names)})\n")
    for sp in ("dev", "test"):
        if sp not in base["suites"].get("afs", {}):
            continue
        for level in ("sentence", "document"):
            for metric in ("chrf", "chrf_pp", "bleu", "finished_rate") if level == "document" else ("chrf", "chrf_pp", "bleu", "spbleu"):
                g = lambda r, iso, sp=sp, level=level, metric=metric: (r["suites"].get("afs", {}).get(sp, {}).get(level, {}).get("per_language", {}).get(iso, {}).get(metric))
                v = lambda x: None if x is None else (x * 100 if metric == "finished_rate" else x)
                gg = lambda r, iso: v(g(r, iso))
                table_lang(f"AfriScience-MT {sp}, {level} level: {metric}" + (" (%)" if metric == "finished_rate" else ""), gg, SCI6)
    for suite, title in (("general", "General dev (bench_dev_clean, 19 languages): chrF++"), ("weak", "Weak-language dev (FLORES+ dev 60 sentences / pool 60): chrF++")):
        if suite not in base["suites"]:
            continue
        g = lambda r, iso, suite=suite: r["suites"].get(suite, {}).get("per_language", {}).get(iso, {}).get("chrf_pp")
        langs = sorted(base["suites"][suite]["per_language"], key=lambda i: (group_of(i), i))
        group_table(title, g, langs)
        table_lang(title + ", per language", g, langs)
        gf = lambda r, iso, suite=suite: (lambda x: None if x is None else x * 100)(r["suites"].get(suite, {}).get("per_language", {}).get(iso, {}).get("finished_rate"))
        group_table(title.split(":")[0] + ": finished rate (%)", gf, langs)
    md = "\n".join(L)
    open(a.out, "w", encoding="utf-8").write(md)
    print(md)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--suites", default="afs,general,weak")
        p.add_argument("--splits", default="dev", help="AfriScience-MT splits; the TEST split only once, for the final choice")
        p.add_argument("--afs-corpus", default="/workspace/jobs/a4/afs_raw/corpus")
        p.add_argument("--general-dev", default="/workspace/sft_build/bench/bench_dev_clean.jsonl")
        p.add_argument("--general-max-per-language", type=int, default=0)
        p.add_argument("--eval-pool", default="/workspace/rlcheck/res/pool.jsonl")
        p.add_argument("--weak-n", type=int, default=60)
        p.add_argument("--ibom-dir", default="/workspace/sft_build/bench/test_sets/ibom_mt")
        p.add_argument("--limit", type=int, default=0, help="smoke test: sentences / rows per language (and 2 papers)")
        p.add_argument("--max-new-tokens-sentence", type=int, default=512)
        p.add_argument("--max-new-tokens-document", type=int, default=7168)
        p.add_argument("--vllm-python", default="/workspace/venv_vllm/bin/python")
        p.add_argument("--vllm-gpu-util", type=float, default=0.15)
        p.add_argument("--kv-cache-gib", type=float, default=4.0, help="fixed KV cache (skips vLLM's memory profile, which fails on a shared card); 0 = whatever fits in --vllm-gpu-util")
        p.add_argument("--vllm-max-model-len", type=int, default=10240)
        p.add_argument("--vllm-max-num-seqs", type=int, default=256)
        p.add_argument("--gpu-wait", type=float, default=900.0)
        p.add_argument("--extra-libs", default="/workspace/sft_build/pylibs")

    p = sub.add_parser("run"); common(p)
    p.add_argument("--model", required=True); p.add_argument("--name", default=None); p.add_argument("--out", required=True)
    p.set_defaults(fn=run_model)
    p = sub.add_parser("sweep"); common(p)
    p.add_argument("--init", required=True); p.add_argument("--sft", required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    p.add_argument("--out", required=True); p.add_argument("--work", default="/workspace/jobs/r4/interp_tmp")
    p.add_argument("--keep", action="store_true"); p.add_argument("--force", action="store_true")
    p.set_defaults(fn=sweep)
    p = sub.add_parser("compare")
    p.add_argument("--baseline", required=True); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=compare)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
