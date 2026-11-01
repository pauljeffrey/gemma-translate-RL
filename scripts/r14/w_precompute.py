"""R14 widening job: NLLB-3.3B pseudo-references for a bin-stratified sample of the rl_prompts_sci_v2 windows, merged with the existing store, into a NEW directory.
usage: w_precompute.py OUT_DIR iso [iso ...]
Windows: all b64/b256/b512 windows, 25 % of the b1024 and 15 % of the b2048 windows (deterministic hash order); the existing store (assets2 pseudorefs) is the base.
Same settings as precompute_nllb_pseudorefs.py (beam 4, no-repeat-4-gram, rl/segments.split_sentences, same rejection rules). Resumable per language."""
import hashlib, importlib.util, json, os, shutil, sys, time, unicodedata
import pyarrow as pa, pyarrow.parquet as pq
from gemma_mt.rl.nllb_signal import NllbCT2, key_of
from gemma_mt.rl.segments import split_sentences
spec = importlib.util.spec_from_file_location("pre", "/workspace/jobs/r7/gemma_mt_run2/scripts/precompute_nllb_pseudorefs.py"); pre = importlib.util.module_from_spec(spec); spec.loader.exec_module(pre)
nfc = lambda s: unicodedata.normalize("NFC", s)
OUT = sys.argv[1]; LANGS = sys.argv[2:]
POOL = "/workspace/jobs/r7/assets2/rl_prompts_sci_v2"; BASE = "/workspace/jobs/r7/assets2/pseudorefs_reward_only_nc_v1"; R14 = "/workspace/jobs/r14"
FRAC = {"b64": 1.0, "b256": 1.0, "b512": 1.0, "b1024": 0.25, "b2048": 0.15}
os.makedirs(OUT, exist_ok=True)
m = NllbCT2(R14 + "/models/nllb3p3b-ct2-int8", R14 + "/models/nllb3p3b-ct2-int8", compute_type="int8_float16")
for iso in LANGS:
    if os.path.exists(f"{OUT}/DONE_{iso}"): continue
    t0 = time.time()
    pool = pq.read_table(f"{POOL}/{iso}.parquet", columns=["english", "length_bin", "source_dataset"]).to_pylist()
    byb = {}
    for r in pool: byb.setdefault(r["length_bin"], []).append(r)
    chosen = []
    for b, rs in byb.items():
        rs = sorted(rs, key=lambda r: hashlib.sha1(("w14|" + iso + "|" + nfc(r["english"])).encode()).hexdigest())
        chosen += rs[: int(round(len(rs) * FRAC.get(b, 0.15)))]
    base_p = f"{BASE}/{iso}.parquet"
    old = pq.read_table(base_p).to_pylist() if os.path.exists(base_p) else []
    have = {r["key"] for r in old}
    new = []
    for r in chosen:
        text = nfc(r["english"]); k = key_of(iso, text)
        if k in have: continue
        have.add(k); new.append((text, split_sentences(text), r["source_dataset"]))
    cache = {}
    for p in (f"{R14}/pseudorefs_reward_only_nc_v1/cache_{iso}.jsonl", f"{R14}/s2/trans_{iso}.jsonl", f"{OUT}/cache_{iso}.jsonl"):
        if os.path.exists(p):
            for l in open(p):
                d = json.loads(l); cache[d["english"]] = (d["hyp"], d.get("score"))
    todo = sorted({s for _, ss, _ in new for s in ss if s not in cache and len(s) <= 700})
    print(f"{iso}: pool {len(pool)} windows, chosen {len(chosen)}, base {len(old)}, new {len(new)}, to translate {len(todo)}", flush=True)
    with open(f"{OUT}/cache_{iso}.jsonl", "a") as f:
        for j in range(0, len(todo), 1500):
            part = todo[j:j + 1500]
            for attempt in range(4):
                try: hyps, sc = m.translate(part, iso, max_batch_tokens=3072, return_scores=True); break
                except RuntimeError as e:
                    print("  retry after", repr(e)[:80], flush=True); time.sleep(30)
            else: raise SystemExit("persistent failure")
            for e, h, s in zip(part, hyps, sc):
                cache[e] = (h, s); f.write(json.dumps({"english": e, "hyp": h, "score": s}, ensure_ascii=False) + "\n")
            f.flush(); print(f"  {iso} {j + len(part)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    rows = list(old); rej = 0
    for text, ss, ds in new:
        tr = []
        for s in ss:
            h, _ = cache.get(s, ("", None))
            if not h or pre.reject(s, h): h = ""; rej += 1
            tr.append(h)
        rows.append({"key": key_of(iso, text), "iso": iso, "source": text, "sentences": ss, "translations": tr, "seq_scores": [float(cache.get(s, ("", None))[1] or 0.0) for s in ss],
                     "n_sentences": len(ss), "source_dataset": ds, "model": "nllb-200-3.3B", "decode": "beam4_no_repeat_ngram4", "licence_class": "reward_only_nc"})
    names = ["key", "iso", "source", "sentences", "translations", "seq_scores", "n_sentences", "source_dataset", "model", "decode", "licence_class"]
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in names}), f"{OUT}/.{iso}.tmp", compression="zstd"); os.replace(f"{OUT}/.{iso}.tmp", f"{OUT}/{iso}.parquet")
    open(f"{OUT}/DONE_{iso}", "w").write(json.dumps({"new_windows": len(new), "translated": len(todo), "rejected": rej, "seconds": round(time.time() - t0)}))
    print(f"{iso}: done, {len(rows)} windows, {rej} rejected sentences, {time.time() - t0:.0f}s", flush=True)
json.dump({"licence_class": "reward_only_nc", "model": "nllb-200-3.3B", "note": "NLLB output is CC-BY-NC-4.0: reward scoring only, never SFT/training data"}, open(f"{OUT}/LICENCE_REWARD_ONLY_NC.json", "w"))
