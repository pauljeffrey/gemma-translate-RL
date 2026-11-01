"""R14 stage 2: hold-out filter = R4's HoldOut (exact sentence + word 8-gram) built from every evaluation English side we know of on the box."""
import glob, json, os, sys
sys.path.insert(0, "/workspace/jobs/r4/gemma_mt_sft/scripts")
import build_sft_science_mix as R4    # noqa: E402  (HoldOut, build_holdout, nfc)
import pandas as pd

def build():
    a = type("A", (), dict(longdoc_dir="/workspace/jobs/a4/holdout/afrilongdoc", bench_dev="/workspace/sft_build/bench/bench_dev_clean.jsonl",
                           flores_dev_eng="/workspace/sft_build/hf/hub/datasets--openlanguagedata--flores_plus/snapshots/*/dev/eng_Latn.jsonl",
                           eval_pool="/workspace/rlcheck/res/pool.jsonl"))()
    afs = {sp: pd.read_parquet(f"/workspace/jobs/a4/afs_raw/corpus/{sp}-00000.parquet") for sp in ("dev", "test")}
    ho = R4.build_holdout(a, afs)
    # + devtest/test sides of every benchmark we evaluate on (English): FLORES+ devtest, NTREX, SmolSent, MAFAND, SALT (from the r14 dump) and bench_dev (all rows)
    seen = {}
    for l in open("/workspace/jobs/r14/out/bench_rows.jsonl"):
        r = json.loads(l); seen.setdefault(r["benchmark"], set()).add(r["english"])
    for b, s in seen.items():
        ho.add("bench_" + b, sorted(s), sorted(s)); ho.counts["bench_" + b] = len(s)
    en = sorted({json.loads(l)["english"] for l in open("/workspace/sft_build/bench/bench_dev.jsonl")})
    ho.add("bench_dev_all", en, en)
    for p in glob.glob("/workspace/sft_build/bench/test_sets/flores_plus/devtest/eng_Latn.jsonl"):
        en = [json.loads(l)["text"] for l in open(p)]; ho.add("flores_plus_devtest_english", en, en)
    print("holdout:", dict(ho.counts), len(ho.sent), "sentences", len(ho.grams), "8-grams", flush=True)
    return ho
