"""R14 follow-up: translate exactly the prompt windows of a missing-windows json ({iso: [window text, ...]}) with NLLB-3.3B (same settings as precompute_nllb_pseudorefs.py),
merge them into the per-language parquet files (same format and keys) and report the coverage of the prompt windows and of the listed (drawn) windows.
usage: pseudorefs_fill_missing.py MISSING.json PROMPTS_DIR(<iso>.parquet with `english`) STORE_DIR [STORE_DIR2 ...]   (STORE_DIRs are rewritten with the merged files; the first one also holds cache_<iso>.jsonl)"""
import json, os, shutil, sys, time, unicodedata
import pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, "/workspace/jobs/r14/code")
from gemma_mt.rl import nllb_signal as N
from gemma_mt.rl.nllb_signal import NllbCT2, key_of
from gemma_mt.rl.segments import split_sentences
import importlib.util
spec = importlib.util.spec_from_file_location("pre", "/workspace/jobs/r7/gemma_mt_run2/scripts/precompute_nllb_pseudorefs.py"); pre = importlib.util.module_from_spec(spec); spec.loader.exec_module(pre)
nfc = lambda s: unicodedata.normalize("NFC", s)
miss = json.load(open(sys.argv[1])); prompts = sys.argv[2]; stores = sys.argv[3:]
m = NllbCT2("/workspace/jobs/r14/models/nllb3p3b-ct2-int8", "/workspace/jobs/r14/models/nllb3p3b-ct2-int8", compute_type="int8_float16")
report = {}
def coverage(iso, store_path, drawn):
    t = pq.read_table(store_path, columns=["key", "translations"]).to_pydict()
    full = {k for k, tr in zip(t["key"], t["translations"]) if all(tr)}
    pw = pq.read_table(os.path.join(prompts, f"{iso}.parquet"), columns=["english"]).column(0).to_pylist()
    pk = [key_of(iso, nfc(x)) for x in pw]
    return {"prompt_windows": len(pk), "prompt_windows_in_store": sum(k in set(t["key"]) for k in pk) if False else None,
            "prompt_cov": sum(k in full for k in pk) / len(pk), "drawn_windows": len(drawn), "drawn_cov": sum(key_of(iso, nfc(x)) in full for x in drawn) / max(1, len(drawn))}
for iso, wins in miss.items():
    t0 = time.time()
    base = stores[0]; path = os.path.join(base, f"{iso}.parquet")
    old = pq.read_table(path).to_pylist(); have = {r["key"] for r in old}
    before = coverage(iso, path, wins)
    cache = {}
    cp = os.path.join(base, f"cache_{iso}.jsonl")
    if os.path.exists(cp):
        for l in open(cp):
            d = json.loads(l); cache[d["english"]] = (d["hyp"], d.get("score"))
    new = [(nfc(w), split_sentences(nfc(w))) for w in wins if key_of(iso, nfc(w)) not in have]
    todo = sorted({s for _, ss in new for s in ss if s not in cache and len(s) <= 700})
    print(f"{iso}: {len(new)} new windows, {len(todo)} sentences to translate", flush=True)
    with open(cp, "a") as f:
        for j in range(0, len(todo), 2000):
            part = todo[j:j + 2000]; hyps, sc = m.translate(part, iso, return_scores=True)
            for e, h, s in zip(part, hyps, sc):
                cache[e] = (h, s); f.write(json.dumps({"english": e, "hyp": h, "score": s}, ensure_ascii=False) + "\n")
            f.flush(); print(f"  {iso} {j + len(part)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    rows = list(old); rej = 0
    for text, ss in new:
        tr = []
        for s in ss:
            h, _ = cache.get(s, ("", None))
            if not h or pre.reject(s, h): h = ""; rej += 1
            tr.append(h)
        rows.append({"key": key_of(iso, text), "iso": iso, "source": text, "sentences": ss, "translations": tr, "seq_scores": [float(cache.get(s, ("", None))[1] or 0.0) for s in ss],
                     "n_sentences": len(ss), "source_dataset": "rl_prompts_sci_v2", "model": "nllb-200-3.3B", "decode": "beam4_no_repeat_ngram4", "licence_class": "reward_only_nc"})
    cols = {k: [r[k] for r in rows] for k in old[0].keys()}
    tbl = pa.table(cols)
    for sd in stores:
        tmp = os.path.join(sd, f".{iso}.parquet.tmp"); pq.write_table(tbl, tmp, compression="zstd"); os.replace(tmp, os.path.join(sd, f"{iso}.parquet"))
    after = coverage(iso, os.path.join(stores[0], f"{iso}.parquet"), wins)
    report[iso] = {"new_windows": len(new), "translated_sentences": len(todo), "rejected_sentences": rej, "before": before, "after": after, "seconds": round(time.time() - t0)}
    print(iso, json.dumps(report[iso]), flush=True)
json.dump(report, open("/workspace/jobs/r14/pseudorefs_v2_report.json", "w"), indent=1)
