"""Independent cross-check of the 385M scanner against the filter of contam_audit_12 (rules R1 exact, R2 contiguous, R3 8-gram, R4 mark-stripped variants) on slices of real training samples.
For every sample (both sides) the set of samples flagged by contam_audit_12 (R1|R2, protected strings >= 3 tokens, as the intake filter) is compared with the samples that carry an exact hit of
the scanner (min 3 tokens applied to the scanner's hit strings). Slices: s1 0..100k, g0 2.0M..2.1M (the g0 region with the most reference hits)."""
import os, sys, json, time
os.environ["PYTHONHASHSEED"] = "0"
sys.dont_write_bytecode = True
sys.argv = ["x"]; sys.path.insert(0, "/workspace/eval_385m/contam_audit"); sys.path.insert(0, "/workspace/eval_385m/contam_audit_v2")
import contam_audit_14_385m_scan as M
import contam_audit_12_decontaminate as D
import numpy as np
class A: pass
a = A(); a.tree = "/workspace/eval_385m/gemma_mt_tree"; a.bench_root = "/workspace/eval_385m/bench/test_sets"; a.extra_root = "/workspace/eval_385m/contam_audit/extra_bench"
sets = M.load_rows(a)
from gemma_mt.eval import contamination as C
fwd, rev = M.build_indexes(sets)
prot_en, prot_ref = set(), set()
for b, rows in sets.items():
    if b in ("flores_plus_dev_trainhalf", "flores_plus_dev_other"): continue
    for r in rows: prot_en.add(C.normalise(r["english"])); prot_ref.add(C.normalise(r["reference"]))
P = D.Protected()
t0 = time.time()
for b, rows in sets.items():
    if b in ("flores_plus_dev_trainhalf", "flores_plus_dev_other"): continue
    for r in rows:
        P.add(b, r["english"]); P.add(b, r["reference"])
print("protected sentences", len(P.sent), "grams", len(P.gram), f"{time.time()-t0:.0f}s", flush=True)
M.G["tokenizer"] = "/workspace/clean_sft/tokenizers/gemma3_tags/tokenizer.json"; M.G["prefix"] = {"s1": os.environ["S3_PREFIX"] + "/shards/s1", "g0": os.environ["S3_PREFIX"] + "/g0/packed_g0"}
M.G["arr"] = {}; M.G["iso_table"] = {}
for n in ("s1", "g0"):
    d = f"/workspace/eval_385m/contam_audit/full/arrays/{n}"
    M.G["arr"][n] = {fn: np.load(f"{d}/{fn}.npy", mmap_mode="r") for fn in ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id", "direction")}
    M.G["iso_table"][n] = json.load(open(d + "/meta.json"))["iso_table"]
ntok = lambda s: len(C.normalise(s).split())
for name, lo, hi in (("s1", 0, 100000), ("g0", 2000000, 2100000)):
    en, ot, iso, tr, _ = M.decode_chunk(name, lo, hi)
    hits = []
    acc = M.scan_items(fwd, rev, list(zip(en, ot, iso)), tr, hits)
    mine = set(); mine3 = set(); strict8 = set()
    for idx, sid, kind, rule, side, trf in hits:
        s = (fwd if kind == 0 else rev).texts[sid]
        if s not in (prot_en if kind == 0 else prot_ref): continue          # the train half of the FLORES+ dev ids is exempt (D2 rule), not protected
        mine.add(idx)
        if len(s.split()) >= 3: mine3.add(idx)
    theirs = set(); r3 = set(); r3near = set(); names = {}
    for idx, (p, t) in enumerate(zip(en, ot)):
        for text in (p, t):
            h = P.check(text)
            if h.get("R1_exact") or h.get("R2_contiguous"): theirs.add(idx)
            if h.get("R3_8gram"): r3.add(idx)
            if h.get("R3_near_ge50pct"): r3near.add(idx)
    print(f"{name}[{lo}:{hi}] samples {hi-lo}: scanner exact-hit samples {len(mine)} (>=3 tok {len(mine3)}) | contam_audit_12 R1|R2 samples {len(theirs)} | R3 any 8-gram {len(r3)} | R3 near>=50% {len(r3near)}")
    print("   scanner>=3tok only:", len(mine3 - theirs), "| contam_audit_12 only:", len(theirs - mine3), "| both:", len(mine3 & theirs), flush=True)
    for idx in sorted(theirs - mine3)[:4]: print("   c12-only example:", repr(en[idx][:120]), "||", repr(ot[idx][:80]))
    for idx in sorted(mine3 - theirs)[:4]: print("   scanner-only example:", repr(en[idx][:120]), "||", repr(ot[idx][:80]))
