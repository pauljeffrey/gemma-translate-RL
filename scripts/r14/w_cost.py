"""R14 widening: CPU cost per RL step of signal A with the widened language set: store load + lookups + chrF++ for the chunks of one step.
usage: w_cost.py STORE_DIR CALIBRATION_V2.json OUT.json"""
import json, random, sys, time, unicodedata
import pyarrow.parquet as pq
from gemma_mt.rl import nllb_signal as N
from gemma_mt.rl.segments import split_sentences
store_dir, calp, outp = sys.argv[1:4]
cal = json.load(open(calp)); langs = sorted(k for k, v in cal.items() if v["nllb"]["use"] and k != "kon")
cfg = N.NllbSignalConfig(nllb_signal=True, nllb_pseudoref_dir=store_dir, nllb_calibration_path=calp, nllb_languages=langs)
sig = N.NllbSignal(cfg)
t0 = time.time(); [sig.store.load(l) for l in langs]; load_s = time.time() - t0
rnd = random.Random(0); items = []
for l in langs:
    t = pq.read_table(f"{store_dir}/{l}.parquet", columns=["source", "sentences", "translations"]).to_pylist()
    for r in rnd.sample(t, min(60, len(t))):
        for s, x in zip(r["sentences"], r["translations"]):
            if x: items.append((l, s, x[: max(5, len(x) - 3)] + " x"))         # a near-copy of the pseudo-reference: the chrF++ is computed on realistic sentence pairs
rnd.shuffle(items)
n = 4300                                                                       # chunks of one step with 42 prompts x 16 completions, ~32 % of the draws in the widened languages, ~20 sentences per window
sel = (items * (n // len(items) + 1))[:n]
t0 = time.time(); [sig.blend_chunk(l, s, mt, 0.0, "qe") for l, s, mt in sel]; step_s = time.time() - t0
res = {"languages": len(langs), "store_load_seconds_total": round(load_s, 2), "chunks_timed": n, "seconds_for_chunks": round(step_s, 3), "ms_per_chunk": round(1000 * step_s / n, 3), "stats": sig.stats}
json.dump(res, open(outp, "w"), indent=1); print(res)
