#!/usr/bin/env python3
"""r4: ONE idempotent script that redoes the whole clean reward calibration on a fresh GPU instance. Inputs: the env var HF_TOKEN and the FLORES+ dev source.

    export HF_TOKEN=...                       # read access to openlanguagedata/flores_plus (gated) and the two private <HF_ORG_LEGACY>/sabiyarn-clean-* repos; write access only for --push
    python data_pipeline/sft_clean/r4_run_all.py [--flores-dir DIR] [--work /workspace/calib_r4] [--push] [--plan] [--force STEP ...] [--only STEP ...]

Run it from a checkout of the project repository (it finds gemma_mt/ and data_pipeline/sft_clean/ relative to itself; the repository holds NO benchmark text and the script refuses a --work
directory inside the repository). --flores-dir = a local copy of openlanguagedata/flores_plus (contains dev/<code>.jsonl); without it the dev files are downloaded with HF_TOKEN.

CONCURRENCY. The steps form a dependency graph and independent steps run at the same time (--jobs, default 8): the Hub streaming of the SFT shards (the long pole, network bound) runs while the
GPU scores the references, and the CPU refits run while both are busy. The QE scoring starts --qe-parts processes per model (default 2, so 4 processes share one big GPU such as an RTX PRO 6000
with 96 GB) with large batches (AfriCOMET 512, SSA-COMET-MTL 256). Every step writes a marker <work>/.done/<step> and is skipped when done (--force STEP redoes it); a QE part also skips every
(language, set) score file that exists. Logs: <work>/logs/<step>.log.

LICENCES (owner decisions 2026-10-29): function-word tables and bigram statistics use ALL clean SFT target text (statistics only, no licence filter); the LID proxy and the diacritic widening of
every other language use strict-licence rows only (cc-by, cc0, apache, mit, mpl, odc-by, share_alike); IbomMT (efi, ibb; CC-BY-NC) is an explicit owner exception and everything derived from it is
tagged  nc_exception: ibom-mt. AfriScience-MT train, AfriCOMET-QE and SSA-COMET are accepted.

Steps (wall minutes ESTIMATED on one RTX PRO 6000 instance; the SFT streaming dominates):
  env        venv with torch, unbabel-comet, fasttext, pyarrow, ...                                                                                  ~5 min
  inputs     SSA-MTE train, IbomMT train (efi, ibb), clean AfriScience train, GlotLID v3, the two QE checkpoints                                      ~3 min
  refs       H_cal / H_sel human references of the 45 FLORES+ languages (split v2) -> <work>/refs/{cal,sel}.jsonl (box only)                           ~1 min
  monitors   general / weak dev ids + draw weights (+ text files in <work>/monitors, box only)
  sft        one pass over the Hub shards, three samples: all licences (40,000 per language), strict licences (5,000)                                  ~30-60 min, network bound
  sftprep    function-word corpus from `all`; LID / diacritic sample = strict + efi / ibb rows of `all`
  qe_*       AfriCOMET-QE-STL-1.1 and SSA-COMET-MTL scores of H_cal references + 4 corruptions                                                         ~4 min wall (66k pairs)
  efi        IbomMT TRAIN calibration / hold-out halves and the D-43 reference statistics of efi and ibb (nc_exception)                                ~2 min
  noqe       length window, diacritics (ext-min-count 5), lexical, word ratio, omission cost FIXED at 4.0                                              ~10 min
  lid        GlotLID recall / guard per language (min n 250)                                                                                           ~5 min
  funcwords  function words + bigram table (statistics only)                                                                                           ~5 min
  science    science anchor from the clean AfriScience train set                                                                                       <1 min
  longdoc    long-document repetition limits from H_cal ONLY (FLORES-only; pass --longdoc-afriscience to add the AfriScience windows)                 ~1 min
  assemble   shaping_clean.json (AfriCOMET anchors, reliability, modes, qe2, ens, pooled priors, efi / ibb IbomMT entries, pcm pooled + LID off)       ~1 min
  anchors    constraint anchors v1 + v2                                                                                                                ~10 min
  outputs    tag nc_exception, leak scan, copy of the numbers to --out-dir (*_final.json)
  push       (with --push) upload to <HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28 calibration/
NOT done here: lexicons (build_lexicons.py), NLLB anchors + NLLB language rule (RL-ONLY: NLLB pseudo-references only during RL for the very poorly resourced languages, scripts/r14/),
and everything that needs the new policy (ensemble delta check, efi / ibb consensus weights, tail_gap, baselines of the dev monitors).
"""
import os
import argparse, concurrent.futures as cf, hashlib, json, os, shutil, subprocess, sys, threading, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CODE = os.path.join(REPO, "gemma_mt")
SPLIT = os.path.join(REPO, "reports", "data", "sft_clean_2026-10-28", "heldout_split_cal_sel_v2.json")
DEVHALF = os.path.join(REPO, "reports", "data", "sft_clean_2026-10-28", "devhalf_split.json")
WEAK = os.path.join(REPO, "reports", "data", "weak_selection_2026-10-26", "weak_list.json")
SFT_REPO = (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-clean-sft-2026-10-28")
RLP_REPO = (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-clean-rl-prompts-2026-10-28")
SPLIT_SHA = "18fe1ddad1c8e64ad93a82254342f4a713f509a55fc2cb7e0a782a514abebddd"
NC = "ibom-mt"


def build_graph(parts):
    g = {"env": [], "inputs": ["env"], "refs": ["env"], "sft": ["env"], "sftprep": ["sft"], "monitors": ["refs"], "efi": ["inputs"],
         "noqe": ["refs", "sftprep"], "lid": ["refs", "sftprep"], "funcwords": ["sftprep"], "science": ["inputs"], "longdoc": ["refs"]}
    qe = []
    for m in ("africomet", "ssa_mtl"):
        for i in range(parts):
            g[f"qe_{m}_{i}"] = ["inputs", "refs"]; qe.append(f"qe_{m}_{i}")
    g["assemble"] = qe + ["noqe", "lid", "efi"]
    g["anchors"] = ["assemble", "funcwords", "inputs", "efi"]
    g["outputs"] = ["monitors", "anchors", "science", "longdoc", "funcwords", "noqe", "efi", "assemble"]
    g["push"] = ["outputs"]
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="/workspace/calib_r4"); ap.add_argument("--flores-dir", default=None)
    ap.add_argument("--out-dir", default=os.path.join(CODE, "calibration", "clean_2026-10-28"))
    ap.add_argument("--push", action="store_true"); ap.add_argument("--plan", action="store_true"); ap.add_argument("--force", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None, help="run only these steps (and nothing else, dependencies are assumed done)")
    ap.add_argument("--python", default=None, help="use this python instead of creating <work>/venv")
    ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--qe-parts", type=int, default=2); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-africomet", type=int, default=512); ap.add_argument("--batch-ssa", type=int, default=256)
    ap.add_argument("--longdoc-afriscience", action="store_true", help="add the AfriScience windows to the long-document limits (default FLORES-only)")
    a = ap.parse_args()
    W = os.path.abspath(a.work)
    if os.path.commonpath([W, REPO]) == REPO:
        sys.exit(f"REFUSED: --work {W} is inside the repository; benchmark text must never be inside a git tree")
    G = build_graph(a.qe_parts)
    for d in ("refs", "qe_work", "clean", "monitors", "in", "hf", ".done", "logs"):
        os.makedirs(os.path.join(W, d), exist_ok=True)
    assert hashlib.sha256(open(SPLIT, "rb").read()).hexdigest() == SPLIT_SHA, "split v2 file changed"
    P = lambda *p: os.path.join(W, *p)
    done = lambda s: os.path.exists(P(".done", s)) and s not in a.force
    mark = lambda s: open(P(".done", s), "w").write(time.strftime("%F %T"))
    want = [s for s in G if (a.only is None or s in a.only or any(s.startswith(o + "_") for o in a.only)) and (s != "push" or a.push)]
    if a.plan:
        for s in G:
            print(f"{s:22} {'done' if done(s) else 'TODO' if s in want else 'skipped'}   after: {', '.join(G[s]) or '-'}")
        return
    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN is not set")
    PY = a.python or P("venv", "bin", "python")
    if "env" in want and not done("env") and not a.python:
        sh([sys.executable, "-m", "venv", P("venv")], log=P("logs", "env.log"))
        for pk in (["--upgrade", "pip"], ["numpy<2", "pyarrow<18", "huggingface_hub", "fasttext-wheel", "pandas", "scipy", "sacrebleu", "tokenizers", "unbabel-comet", "torch"]):
            sh([PY, "-m", "pip", "install", "-q"] + pk, log=P("logs", "env.log"))
        mark("env")
    if os.path.exists(PY) and os.path.realpath(sys.executable) != os.path.realpath(PY):
        print("re-executing under", PY, flush=True)
        os.execv(PY, [PY] + sys.argv)
    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False
    print("GPU:", torch.cuda.get_device_name(0) if gpu else "NONE (QE would run on CPU at ~1 pair/s: use a GPU)", flush=True)
    env = {"HF_HOME": P("hf"), "PYTHONPATH": CODE + os.pathsep + os.path.join(CODE, "scripts") + os.pathsep + HERE, "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": str(a.threads)}
    INP = lambda: json.load(open(P("in", "inputs.json")))
    strict_file, all_file = P("sft_strict.jsonl"), P("sft_all.jsonl")

    def run_step(s):
        L = P("logs", s + ".log")
        if s == "env":
            return
        if s == "inputs":
            code = (f"import os,json;from huggingface_hub import HfApi,hf_hub_download as d,snapshot_download as sd;t=os.environ['HF_TOKEN'];api=HfApi(token=t);ls=api.list_repo_files('{SFT_REPO}',repo_type='dataset');"
                    f"pick=lambda n:[f for f in ls if f.endswith(n)][0];ib=os.path.join('{P('in','ibom')}');os.makedirs(ib,exist_ok=True);"
                    f"o={{'ssa': d('{SFT_REPO}',pick('ssa_mte_train.parquet'),repo_type='dataset',token=t,local_dir='{P('in')}'),"
                    f"'ibom_efi': d('{SFT_REPO}',pick('ibom_train_efi.parquet'),repo_type='dataset',token=t,local_dir='{P('in')}'),"
                    f"'ibom_ibb': d('{SFT_REPO}',pick('ibom_train_ibb.parquet'),repo_type='dataset',token=t,local_dir='{P('in')}'),"
                    f"'afs': d('{RLP_REPO}','afriscience_clean_train/afriscience_clean_train.parquet',repo_type='dataset',token=t,local_dir='{P('in')}'),"
                    f"'glot': d('cis-lmu/glotlid','model_v3.bin',token=t)}};"
                    f"[sd(r,token=t) for r in ('masakhane/africomet-qe-stl-1.1','McGill-NLP/ssa-comet-mtl')];sd('Davlan/afro-xlmr-large-76L',token=t,allow_patterns=['*.json','*.model','*.txt']);"
                    f"assert os.path.dirname(o['ibom_efi'])==os.path.dirname(o['ibom_ibb']);o['ibom_dir']=os.path.dirname(o['ibom_efi']);json.dump(o,open('{P('in','inputs.json')}','w'));print(o)")
            sh([PY, "-c", code], env, L)
        elif s == "refs":
            sh([PY, os.path.join(HERE, "r4_refs.py"), "--split", SPLIT, "--langs-dir", CODE, "--out", P("refs"), "--tmp", P("in", "flores_dl")] + (["--flores-dir", a.flores_dir] if a.flores_dir else []), env, L)
        elif s == "monitors":
            sh([PY, os.path.join(HERE, "r4_monitors.py"), SPLIT, DEVHALF, WEAK, P("clean", "monitors_and_weights_final.json")], env, L)
            m = json.load(open(P("clean", "monitors_and_weights_final.json")))
            gen, gl, wk, wl = set(m["general_dev"]["ids"]), set(m["general_dev"]["languages"]), set(m["weak_dev"]["ids"]), set(m["weak_dev"]["languages"])
            with open(P("monitors", "general_dev_clean.jsonl"), "w") as fg, open(P("monitors", "weak_dev_clean.jsonl"), "w") as fw:
                for l in open(P("refs", "sel.jsonl")):
                    r = json.loads(l)
                    assert r["split"] == "selection"
                    if r["id"] in gen and r["iso"] in gl:
                        fg.write(l)
                    if r["id"] in wk and r["iso"] in wl:
                        fw.write(l)
        elif s == "sft":
            sh([PY, os.path.join(HERE, "r4_sft_targets.py"), "--hub-repo", SFT_REPO, "--hub-prefix", "data/", "specialist/", "--cache", P("in", "sft_dl"), "--per-file", 20000,
                "--profile", f"all:all:40000:{all_file}", "--profile", f"strict:strict:5000:{strict_file}"], env, L)
        elif s == "sftprep":
            import collections, pyarrow as pa, pyarrow.parquet as pq
            by, ibom_rows, cnt = collections.defaultdict(list), [], collections.Counter()
            os.makedirs(P("fw_corpus"), exist_ok=True)
            for l in open(all_file, encoding="utf-8"):
                r = json.loads(l)
                by[r["iso"]].append(r["text"])
                cnt[r["iso"]] += 1
                if r["iso"] in ("efi", "ibb") and cnt[r["iso"]] <= 5000:
                    ibom_rows.append(l)
            for iso, xs in by.items():
                pq.write_table(pa.table({"x": xs, "held": [False] * len(xs)}), P("fw_corpus", iso + ".parquet"))
            keep = [l for l in open(strict_file, encoding="utf-8") if json.loads(l)["iso"] not in ("efi", "ibb")]
            open(P("sft_targets.jsonl"), "w", encoding="utf-8").writelines(keep + ibom_rows)
        elif s.startswith("qe_"):
            m, i = s[3:].rsplit("_", 1)
            dev = ["--device", "cuda", "--batch", a.batch_africomet if m == "africomet" else a.batch_ssa] if gpu else ["--batch", 32]
            sh([PY, os.path.join(HERE, "r4_qe_score.py"), "--refs", P("refs", "cal.jsonl"), "--split", SPLIT, "--work", P("qe_work"), "--model", m, "--phase", "ALL", "--code", CODE,
                "--threads", a.threads, "--part", f"{i}/{a.qe_parts}"] + dev, env, L)
        elif s == "efi":
            sh([PY, os.path.join(HERE, "r4_efi.py"), "--kept", INP()["ibom_dir"], "--code", CODE, "--out", P("clean", "efi_ibb_blocks.json"), "--work", P("clean", "ibom_halves")], env, L)
        elif s == "noqe":
            sh([PY, os.path.join(HERE, "r4_noqe.py"), "--cal", P("refs", "cal.jsonl"), "--sel", P("refs", "sel.jsonl"), "--split", SPLIT, "--code", CODE, "--sft-targets", P("sft_targets.jsonl"),
                "--old", os.path.join(CODE, "calibration", "shaping_flores_dev_v8.json"), "--out", P("clean", "noqe_blocks.json"), "--report", P("clean", "noqe_report.json"), "--omission-fixed", "4.0"], env, L)
        elif s == "lid":
            sh([PY, os.path.join(CODE, "scripts", "measure_lid.py"), "--dev", P("refs", "cal.jsonl"), "--sft-targets", P("sft_targets.jsonl"), "--min-n", "250", "--out", P("clean", "lid_measure_clean.json")], env, L)
        elif s == "funcwords":
            sh([PY, os.path.join(CODE, "scripts", "build_funcwords.py"), "--corpus", P("fw_corpus"), "--out", P("clean", "funcwords_clean.json"), "--bigrams-out", P("clean", "bigrams_clean.npz"), "--cap-corpus", 40000], env, L)
        elif s == "science":
            sh([PY, os.path.join(HERE, "r4_science_anchor.py"), "--parquet", INP()["afs"], "--code", CODE, "--out", P("clean", "science_anchor_clean.json")], env, L)
        elif s == "longdoc":
            sh([PY, os.path.join(HERE, "r4_longdoc.py"), "--cal", P("refs", "cal.jsonl"), "--split", SPLIT, "--code", CODE, "--out", P("clean", "longdoc_clean.json"), "--report", P("clean", "longdoc_clean_report.json")]
               + (["--afs", INP()["afs"]] if a.longdoc_afriscience else []), env, L)
        elif s == "assemble":
            sh([PY, os.path.join(HERE, "r4_assemble.py"), "--code", CODE, "--old", os.path.join(CODE, "calibration", "shaping_flores_dev_v8.json"), "--work", W, "--split", SPLIT,
                "--out", P("clean", "shaping_clean.json"), "--status", P("clean", "status_assemble.json")], env, L)
        elif s == "anchors":
            sh([PY, os.path.join(HERE, "r4_anchors_v1.py"), "--code", CODE, "--cal", P("refs", "cal.jsonl"), "--split", SPLIT, "--kept", os.path.dirname(INP()["ssa"]), "--ibom-halves", P("clean", "ibom_halves"),
                "--afs", INP()["afs"], "--shaping", P("clean", "shaping_clean.json"), "--lid-model", INP()["glot"], "--out", P("clean", "anchors_clean_v1.json"), "--dump-human", P("clean", "human_units_clean.jsonl")], env, L)
            sh([PY, os.path.join(CODE, "scripts", "build_anchors_v2_calibration.py"), "--v1", P("clean", "anchors_clean_v1.json"), "--human", P("clean", "human_units_clean.jsonl"),
                "--funcwords", P("clean", "funcwords_clean.json"), "--bigrams", P("clean", "bigrams_clean.npz"), "--out", P("clean", "anchors_clean_v2.json")], env, L)
        elif s == "outputs":
            outputs(W, a.out_dir)
        elif s == "push":
            push(a.out_dir)

    # ---- dependency-graph executor: independent steps run at the same time
    state = {s: ("done" if done(s) else "todo") for s in G}
    for s in G:
        if s not in want and state[s] == "todo":
            state[s] = "skipped"
    lock = threading.Lock(); failed = {}
    pool = cf.ThreadPoolExecutor(max_workers=a.jobs); running = {}

    def worker(s):
        t0 = time.time()
        try:
            run_step(s); mark(s)
            with lock:
                state[s] = "done"
            print(f"[{s}] done in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:                                           # noqa
            with lock:
                state[s] = "failed"; failed[s] = repr(e)[:300]
            print(f"[{s}] FAILED: {failed[s]} (see logs/{s}.log)", flush=True)

    while True:
        with lock:
            for s in G:
                if state[s] == "todo" and all(state[d] in ("done", "skipped") for d in G[s]):
                    state[s] = "running"; print(f"[{s}] start {time.strftime('%T')}", flush=True)
                    running[s] = pool.submit(worker, s)
            pending = [s for s in G if state[s] in ("todo", "running")]
            blocked = [s for s in G if state[s] == "todo" and any(state[d] == "failed" for d in G[s])]
            for s in blocked:
                state[s] = "blocked"
            pending = [s for s in G if state[s] in ("todo", "running")]
        if not pending:
            break
        time.sleep(2)
    pool.shutdown()
    print("\nstates:", {s: v for s, v in state.items() if v not in ("done", "skipped")} or "all done")
    print("OPEN (not part of this script): lexicons; NLLB anchors + language rule (RL-only); ensemble delta check, efi / ibb consensus weights, tail_gap, baselines of the dev monitors (need the new policy).")
    if failed:
        sys.exit(f"FAILED steps: {failed}")


def sh(cmd, env=None, log=None):
    e = dict(os.environ, **(env or {}))
    cmd = [str(c) for c in cmd]
    with open(log, "a") if log else open(os.devnull, "w") as fh:
        fh.write("+ " + " ".join(cmd) + "\n"); fh.flush()
        r = subprocess.run(cmd, env=e, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        tail = open(log).read()[-600:] if log else ""
        raise RuntimeError(f"exit {r.returncode}: {cmd[1] if len(cmd) > 1 else cmd[0]} ... {tail}")


NUMBER_FILES = {"shaping_clean.json": "shaping_clean_final.json", "noqe_blocks.json": "noqe_blocks_final.json", "noqe_report.json": "noqe_report_final.json",
                "lid_measure_clean.json": "lid_measure_clean_final.json", "funcwords_clean.json": "funcwords_clean_final.json", "bigrams_clean.npz": "bigrams_clean_final.npz",
                "longdoc_clean.json": "longdoc_clean_final.json", "longdoc_clean_report.json": "longdoc_clean_report_final.json", "science_anchor_clean.json": "science_anchor_clean_final.json",
                "anchors_clean_v1.json": "anchors_clean_v1_final.json", "anchors_clean_v2.json": "anchors_clean_final.json", "monitors_and_weights_final.json": "monitors_and_weights_final.json",
                "efi_ibb_blocks.json": "efi_ibb_blocks_final.json"}


def tag_nc(W):
    """nc_exception tags on every piece derived from IbomMT, and the statistics-only metadata of the function-word table (idempotent)."""
    c = lambda f: os.path.join(W, "clean", f)
    for f, getter in (("lid_measure_clean.json", lambda o: o), ("funcwords_clean.json", lambda o: o), ("anchors_clean_v1.json", lambda o: o["languages"]), ("anchors_clean_v2.json", lambda o: o["languages"])):
        o = json.load(open(c(f)))
        for iso in ("efi", "ibb"):
            if iso in getter(o):
                getter(o)[iso]["nc_exception"] = NC
        if f.startswith("anchors"):
            o["_meta"]["nc_exception_languages"] = {"efi": NC, "ibb": NC}
        if f == "funcwords_clean.json":
            o["_statistics_only"] = {"note": "word counts / document frequencies of ALL clean SFT target text per language; licence filtering does not apply to these statistics (owner 2026-10-29)",
                                     "licence_filter": "none", "nc_exception_languages": {"efi": NC, "ibb": NC}}
        json.dump(o, open(c(f), "w"), ensure_ascii=False, **({"separators": (",", ":")} if f.startswith("funcwords") else {"indent": 1}))
    eb = json.load(open(c("efi_ibb_blocks.json"))); eb["_nc_exception"] = {"source": NC, "licence": "CC-BY-NC-4.0", "owner_exception": "2026-10-29"}
    json.dump(eb, open(c("efi_ibb_blocks.json"), "w"), indent=1, ensure_ascii=False)


def leak_scan(W, paths):
    """abort when any string of an output json contains (or equals) a benchmark sentence of H_cal / H_sel (english or reference, >= 25 chars)."""
    sents = set()
    for fn in ("cal.jsonl", "sel.jsonl"):
        for l in open(os.path.join(W, "refs", fn), encoding="utf-8"):
            r = json.loads(l)
            for k in ("english", "reference"):
                if len(r[k]) >= 25:
                    sents.add(r[k])
    bad = []

    def walk(o, f):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(k, f); walk(v, f)
        elif isinstance(o, list):
            for v in o:
                walk(v, f)
        elif isinstance(o, str) and len(o) >= 25:
            if o in sents or any(x in o for x in sents if len(x) <= len(o)):
                bad.append((f, o[:40]))
    for p in paths:
        if p.endswith(".json"):
            walk(json.load(open(p)), os.path.basename(p))
    if bad:
        raise RuntimeError(f"LEAK: benchmark text found in outputs {bad[:3]}; nothing was copied")


def outputs(W, out_dir):
    tag_nc(W)
    src = {k: os.path.join(W, "clean", k) for k in NUMBER_FILES}
    miss = [k for k, p in src.items() if not os.path.exists(p)]
    assert not miss, f"missing outputs {miss}"
    qe = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(W, "qe_work")) for f in fs if f.endswith(".json")]
    leak_scan(W, list(src.values()) + qe)
    os.makedirs(out_dir, exist_ok=True)
    st = {"date": time.strftime("%F"), "split_file_sha256": SPLIT_SHA,
          "licence_policy": "commercial use; explicit owner exception for IbomMT efi / ibb (tag nc_exception: ibom-mt, also inside bigrams_clean_final.npz); function-word / bigram tables are statistics only (all clean SFT text); "
                            "LID proxy and diacritic widening of the other languages use strict-licence rows", "files": {}}
    for k, v in NUMBER_FILES.items():
        shutil.copy(src[k], os.path.join(out_dir, v)); st["files"][v] = hashlib.sha256(open(src[k], "rb").read()).hexdigest()
    for p in qe:
        rel = os.path.relpath(p, os.path.join(W, "qe_work"))
        os.makedirs(os.path.dirname(os.path.join(out_dir, "qe_work_numbers", rel)), exist_ok=True)
        shutil.copy(p, os.path.join(out_dir, "qe_work_numbers", rel))
    json.dump(st, open(os.path.join(out_dir, "status_final.json"), "w"), indent=1)
    print("copied", len(NUMBER_FILES), "files +", len(qe), "score files to", out_dir)


def push(out_dir):
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=os.environ["HF_TOKEN"])
    ops = []
    for fn in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, fn)
        if os.path.isfile(p) and (fn.endswith(("_final.json", "_final.npz")) or fn == "status_final.json"):
            ops.append(CommitOperationAdd(path_in_repo=f"calibration/{fn}", path_or_fileobj=p))
    ops.append(CommitOperationAdd(path_in_repo="protected_index/heldout_split_cal_sel_v2.json", path_or_fileobj=SPLIT))
    r = api.create_commit(repo_id=SFT_REPO, repo_type="dataset", operations=ops, commit_message="r4: clean calibration (final, IbomMT owner exception tagged)")
    print("pushed", len(ops), "files", r.commit_url.split("/")[-1][:12])


if __name__ == "__main__":
    main()
