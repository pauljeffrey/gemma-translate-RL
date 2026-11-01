"""R14: translate the benchmark English sources with NLLB-600M (CT2). Usage: nllb_run.py CONFIG OUT.jsonl bench1,bench2 [nrows_per_lang_cap]. Resumable."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nllb_langmap import EXACT
from nllb_ct2 import NLLB
cfg, out, benches = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
R = "/workspace/jobs/r14"
by = {}
for l in open(R + "/out/bench_rows.jsonl"):
    r = json.loads(l)
    if r["benchmark"] in benches and r["iso"] in EXACT:
        by.setdefault(r["iso"], {})[r["english"]] = 1
done = set()
if os.path.exists(out):
    for l in open(out):
        d = json.loads(l); done.add((d["iso"], d["english"]))
MODEL = os.environ.get("NLLB_MODEL", "600m")
m = {"600m": lambda: NLLB(R + "/models/nllb600m-ct2-fp16", R + "/models/nllb600m-hf"),
     "3b3": lambda: NLLB(R + "/models/nllb3p3b-ct2-int8", R + "/models/nllb600m-hf", compute_type="int8_float16")}[MODEL]()
t00 = time.time(); n = 0
with open(out, "a") as f:
    for iso in sorted(by):
        if os.environ.get("ISOS") and iso not in os.environ["ISOS"].split(","): continue
        todo = [e for e in by[iso] if (iso, e) not in done]
        if not todo:
            continue
        t0 = time.time()
        hyps, sc = m.translate(todo, EXACT[iso], cfg, return_scores=True)
        for e, h, s in zip(todo, hyps, sc):
            f.write(json.dumps({"iso": iso, "english": e, "hyp": h, "score": s, "config": cfg}, ensure_ascii=False) + "\n")
        f.flush(); n += len(todo)
        print(f"{cfg} {iso} {EXACT[iso]} {len(todo)} sents {time.time()-t0:.1f}s ({len(todo)/(time.time()-t0):.0f}/s)", flush=True)
print("DONE", n, "sents", round(time.time() - t00), "s", flush=True)
