"""T1: AfriCOMET (masakhane/africomet-stl-1.1, reference-based) for the benchmark rows that have no score yet (non-FLORES benchmarks, models sft / rl / nllb3b3),
restricted to `no_pair` rows and the AfriCOMET-covered languages; existing E1 scores (hash sample) are reused, not recomputed. GPU share capped (<= 3.5 GiB), nice, resumable.
Output /workspace/jobs/t1/comet_extra/afri_<bench>.jsonl : {metric, b, iso, model, id, score}"""
import sys, os, gzip, json, glob, time, unicodedata, collections
sys.path[:0] = ["/workspace/jobs/e1/code2c", "/workspace/sft_build/pylibs_tf4", "/workspace/sft_build/pylibs"]
import torch
torch.cuda.set_per_process_memory_fraction(3.5 / 95.0)
from gemma_mt.languages import LANGUAGES
from gemma_mt.rl.comet import CometScorer
nfc = lambda s: unicodedata.normalize("NFC", s or "")
OUT = "/workspace/jobs/t1/comet_extra"; os.makedirs(OUT, exist_ok=True)
BENCH = sys.argv[1:] or ["ntrex", "mafand", "smol_sent", "glot_story", "salt"]
rows = collections.defaultdict(dict)       # (b,iso) -> model -> id -> (en, hyp, ref)
ps = collections.defaultdict(dict)
for model, fn in (("sft", "sys_sft.jsonl.gz"), ("rl", "sys_rl.jsonl.gz")):
    for l in gzip.open("/workspace/jobs/e1/" + fn, "rt"):
        r = json.loads(l)
        if r["b"] not in BENCH or r["pair_seen"]: continue
        rows[(r["b"], r["iso"])].setdefault(model, {})[r["id"]] = (nfc(r["en"]), nfc(r["hyp"]), nfc(r["ref"]))
        ps[(r["b"], r["iso"])][r["id"]] = nfc(r["ref"])
nl = {}
for fn in ("nllb3b3_beam4_ntrex.jsonl", "nllb3b3_beam4_mafand+salt+smol_sent.jsonl"):
    for l in open("/workspace/jobs/r14/out/" + fn):
        r = json.loads(l); nl.setdefault((r["iso"], r["english"]), nfc(r["hyp"]))
for key, d in rows.items():
    if key[0] == "glot_story": continue
    got = {}
    for i, (en, h, rf) in d["rl"].items():
        if (key[1], d["rl"][i][0]) in nl or (key[1], en) in nl:
            got[i] = (en, nl[(key[1], en)], rf) if (key[1], en) in nl else None
    d["nllb3b3"] = {i: v for i, v in got.items() if v}
have = collections.defaultdict(lambda: collections.defaultdict(set))
for l in open("/workspace/jobs/e1/neural/afri_ref.jsonl"):
    r = json.loads(l); b, iso = r["chunk"].split("|")
    for i, d in r["scores"].items():
        for m in d: have[(b, iso)][m].add(i)
ck = glob.glob("/workspace/sft_build/hf/hub/models--masakhane--africomet-stl-1.1/snapshots/*/checkpoints/model.ckpt")[0]
sc = CometScorer(ck, batch_size=32, reference_based=True, max_batch_chars=32 * 700)
t0 = time.time(); tot = 0
for key in sorted(rows):
    b, iso = key
    lg = LANGUAGES.get(iso)
    if lg is None or not lg.qe_covered: continue
    marker = f"{OUT}/done_{b}_{iso}"
    if os.path.exists(marker): continue
    todo = []
    for m, d in rows[key].items():
        for i, t in d.items():
            if i in have[key][m]: continue
            todo.append((m, i, t))
    uniq = {}
    for m, i, t in todo: uniq.setdefault(t, None)
    ul = list(uniq)
    if ul:
        while True:
            try:
                s = sc([u[0] for u in ul], [u[1] for u in ul], [u[2] for u in ul]); break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); sc.batch_size = max(1, sc.batch_size // 2); sc.max_batch_chars = sc.batch_size * 700; print("OOM ->", sc.batch_size, flush=True); time.sleep(5)
        look = dict(zip(ul, s))
        with open(f"{OUT}/afri_{b}.jsonl", "a") as f:
            for m, i, t in todo:
                f.write(json.dumps({"metric": "afri", "b": b, "iso": iso, "model": m, "id": i, "score": float(look[t])}) + "\n")
    open(marker, "w").close(); tot += len(ul)
    print(key, len(ul), "unique pairs; total", tot, round(time.time() - t0), "s", flush=True)
print("done")
