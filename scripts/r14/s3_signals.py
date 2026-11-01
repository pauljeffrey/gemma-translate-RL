"""R14 stage 3: compute the NLLB signals for the a5 corruption rows (hack_rows), the policy samples, the lexicon stuffing attack and the dev calibration sets.
usage: s3_signals.py MODEL(3b3|600m)    -> /workspace/jobs/r14/s3/sig_<set>_<model>.json  (+ cost_<model>.json)
Signals per row: pr = chrF++(mt, NLLB translation of src)/100 ; fd = mean per-token log-prob of mt given src under NLLB (forced decoding)."""
import collections, json, os, sys, time, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gemma_mt.rl.nllb_signal import NLLB_CODE, NllbCT2, chrf01
R = "/workspace/jobs/r14"; S3 = R + "/s3"; os.makedirs(S3, exist_ok=True)
model = sys.argv[1]
M = {"3b3": lambda: NllbCT2(R + "/models/nllb3p3b-ct2-int8", R + "/models/nllb600m-hf", compute_type="int8_float16"),
     "600m": lambda: NllbCT2(R + "/models/nllb600m-ct2-fp16", R + "/models/nllb600m-hf", compute_type="float16")}[model]()
nfc = lambda s: unicodedata.normalize("NFC", s)
jl = lambda p: [json.loads(l) for l in open(p)]
hack = jl("/workspace/jobs/a5/hack_rows.jsonl"); pol = jl("/workspace/jobs/a5/policy_samples.jsonl")
FAST600 = {"kea", "amh", "tir", "nus", "kbp", "bam", "kab", "ewe", "kin", "zul", "hau", "yor", "wol", "dyu"}      # the 600M model is only needed for cost + a comparison on a language subset
if model == "600m":
    hack = [r for r in hack if r["iso"] in FAST600]; pol = [r for r in pol if r["iso"] in FAST600]
cost = {"model": model}
def refs_for(srcs_by_iso):
    out, t0, n = {}, time.time(), 0
    for iso, srcs in srcs_by_iso.items():
        if iso not in NLLB_CODE: continue
        u = sorted(set(srcs)); h = M.translate(u, iso, max_batch_tokens=2048); out[iso] = dict(zip(u, h)); n += len(u)
    cost["translate_sentences"] = n; cost["translate_seconds"] = time.time() - t0
    return out
by = collections.defaultdict(set)
for r in hack: by[r["iso"]].add(r["src"])
for r in pol: by[r["iso"]].add(r["src"])
# dev calibration sentences: FLORES+ dev ids >= 100 (the a5 sets use the first sentences; checked below), up to 300 per language
import glob
snap = glob.glob("/workspace/sft_build/hf/hub/datasets--openlanguagedata--flores_plus/snapshots/*/dev")[0]
dev_en = {int(r["id"]): r["text"] for r in jl(snap + "/eng_Latn.jsonl")}
import importlib; L = importlib.import_module("gemma_mt.languages")
dev = {}
used_src = {s for r in hack for s in [r["src"]]}
for iso in (by if model != "600m" else []):
    code = L.LANGUAGES[iso].flores
    if iso not in NLLB_CODE or not code or not os.path.exists(f"{snap}/{code}.jsonl"): continue
    ref = {int(r["id"]): nfc(r["text"]) for r in jl(f"{snap}/{code}.jsonl")}
    ids = [i for i in sorted(dev_en) if i in ref and ref[i].strip() and dev_en[i] not in used_src][:300]
    dev[iso] = [(dev_en[i], ref[i]) for i in ids]
    by[iso] |= {s for s, _ in dev[iso]}
refs = refs_for(by)
json.dump({k: v for k, v in refs.items()}, open(f"{S3}/nllb_refs_{model}.json", "w"), ensure_ascii=False)
def signals(rows, key_mt):
    pr, fd = [None] * len(rows), [None] * len(rows)
    idx = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["iso"] in refs: idx[r["iso"]].append(i)
    t0, n, calls = time.time(), 0, 0
    for iso, ii in idx.items():
        for i in ii: pr[i] = chrf01(rows[i][key_mt] or ".", refs[iso][rows[i]["src"]])
        for j in range(0, len(ii), 64):                       # 64 pairs = one RL reward call (8 prompts x G=8)
            b = ii[j:j + 64]
            lp = M.score(iso, [rows[i]["src"] for i in b], [rows[i][key_mt] or "." for i in b], 2048)
            for i, v in zip(b, lp): fd[i] = v
            n += len(b); calls += 1
    return pr, fd, time.time() - t0, n, calls
for name, rows, key in (("hack", hack, "mt"), ("policy", pol, "hyp")):
    if os.path.exists(f"{S3}/sig_{name}_{model}.json") and model == "3b3" and name == "hack":
        continue                                            # resumable (the first run was killed by a CUDA OOM of the shared GPU after the hack set)
    pr, fd, sec, n, calls = signals(rows, key)
    json.dump({"pr": pr, "fd": fd}, open(f"{S3}/sig_{name}_{model}.json", "w"))
    cost[f"forced_{name}_pairs"] = n; cost[f"forced_{name}_seconds_incl_chrf"] = sec; cost[f"forced_{name}_calls64"] = calls
# pure forced-decoding cost: time score() alone on 64-pair batches
t0 = time.time(); k = 0
for iso in list(refs)[:6]:
    rs = [r for r in hack if r["iso"] == iso][:256]
    for j in range(0, len(rs), 64):
        M.score(iso, [r["src"] for r in rs[j:j+64]], [r["mt"] for r in rs[j:j+64]], 2048); k += 1
cost["forced_pure_seconds_per_call64"] = (time.time() - t0) / max(k, 1)
# dev calibration rows: human refs and mismatched refs (sentence i+1)
cal = {}
for iso, pairs in dev.items():
    hs = [(s, r) for s, r in pairs]
    mism = [(s, pairs[(i + 7) % len(pairs)][1]) for i, (s, r) in enumerate(pairs)]
    hp = [chrf01(r, refs[iso][s]) for s, r in hs]; mp = [chrf01(r, refs[iso][s]) for s, r in mism]
    hf = M.score(iso, [s for s, _ in hs], [r for _, r in hs], 2048); mf = M.score(iso, [s for s, _ in mism], [r for _, r in mism], 2048)
    cal[iso] = {"human_pr": hp, "mism_pr": mp, "human_fd": hf, "mism_fd": mf, "n": len(hs)}
json.dump(cal, open(f"{S3}/sig_devcal_{model}.json", "w"))
# lexicon stuffing attack (R3's): constant stuffed string (+ non-candidate filler) vs the human reference, per source
att = {}
try:
    if model == "600m": raise RuntimeError("skipped for the 600M model")
    sys.path[:0] = ["/workspace/jobs/b3/code", "/workspace/jobs/b3/code/scripts"]
    from gemma_mt.rl.lexicon import load_lexicon, x_tokens
    import validate_lexicons as V
    for iso in "fon kab kam kbp kea kmb mos sag umb".split():
        if iso not in refs: continue
        lex = load_lexicon(iso, "/workspace/jobs/b3/lexicon_final")
        src, ref = V.flores_pairs(snap.replace("/dev", "/devtest") if False else "/workspace/sft_build/bench/test_sets/flores_plus/devtest", L.LANGUAGES[iso].flores)
        src, ref = src[:200], ref[:200]
        st = V.stuffed_output(lex, src, 30).split()
        cand = lex.candidates
        u = sorted(set(src)); refs[iso].update({k: v for k, v in zip(u, M.translate([s for s in u if s not in refs[iso]], iso))} if any(s not in refs[iso] for s in u) else {})
        rows = []
        for i, s in enumerate(src):
            filler = [t for t in x_tokens(ref[(i + 17) % len(ref)]) if V.x_folded_tokens(t) and V.x_folded_tokens(t)[0] not in cand]
            rows.append({"src": s, "human": ref[i], "stuffed": " ".join(st), "padded": " ".join(st + filler[:30])})
        res = {}
        for c in ("human", "stuffed", "padded"):
            res[c + "_pr"] = [chrf01(r[c], refs[iso][r["src"]]) for r in rows]
            res[c + "_fd"] = M.score(iso, [r["src"] for r in rows], [r[c] for r in rows], 2048)
        att[iso] = res
except Exception as e:
    att["_error"] = repr(e)
json.dump(att, open(f"{S3}/sig_attack_{model}.json", "w"))
json.dump(cost, open(f"{S3}/cost_{model}.json", "w"), indent=1)
print("done", cost)
