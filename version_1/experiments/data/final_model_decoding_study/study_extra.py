"""a1: beam + greedy fallback (when the beam output is unfinished / repetitive), pairwise paired differences between the leading configs, loops per language."""
import json, sys, unicodedata
import numpy as np
sys.path.insert(0, "/workspace/sft_build/pylibs")
from sacrebleu.metrics import CHRF, BLEU
from gemma_mt.rl import guards
D = "/workspace/jobs/a1/study"
S = json.load(open(D + "/study_set.json")); H = json.load(open(D + "/hyps.json")); R = json.load(open(D + "/results.json"))
Z = np.load(D + "/bootstrap.npz"); cfgs = [str(c) for c in Z["cfgs"]]; isos = [str(i) for i in Z["isos"]]
rows = S["rows"]; nfc = lambda s: unicodedata.normalize("NFC", s)
idx = {i: [k for k, r in enumerate(rows) if r["iso"] == i] for i in isos}
refs = [nfc(r["reference"]) for r in rows]
chrf, bleu = CHRF(word_order=2), BLEU(tokenize="flores200")
B = 1000; rng = np.random.default_rng(0)
samp = {i: rng.integers(0, len(idx[i]), size=(B, len(idx[i]))) for i in isos}
GR = R["groups_members"]
unfin = {c: None for c in H}
def loopy(h): return (not h.strip()) or guards.is_repetitive(h) or guards.is_self_repetitive(h)
out = {"loop_by_lang": {}, "hybrid": {}}
for c in ("greedy", "beam4", "beam8", "beam2"):
    out["loop_by_lang"][c] = {i: float(np.mean([loopy(nfc(H[c][k])) for k in idx[i]])) for i in isos}
def boot_of(hyps):
    res = {"chrf": np.zeros((B, len(isos))), "bleu": np.zeros((B, len(isos)))}
    for li, i in enumerate(isos):
        hs = [nfc(hyps[k]) for k in idx[i]]; rs = [refs[k] for k in idx[i]]
        sa = np.array(chrf._extract_corpus_statistics(hs, [rs]), dtype=float); sb = np.array(bleu._extract_corpus_statistics(hs, [rs]), dtype=float)
        for b in range(B):
            res["chrf"][b, li] = chrf._compute_score_from_stats(list(sa[samp[i][b]].sum(0))).score
            res["bleu"][b, li] = bleu._compute_score_from_stats(list(sb[samp[i][b]].sum(0))).score
    return res
def summ(res, base_cfg):
    o = {}
    for m in ("chrf", "bleu"):
        for g, members in (("all", isos), *((g, GR[g]) for g in ("strong", "medium", "weak"))):
            ix = [isos.index(i) for i in members]; x = res[m][:, ix].mean(1)
            bb = Z[m][:, cfgs.index(base_cfg), :][:, ix].mean(1); d = x - bb
            o[f"{m}_{g}"] = {"point": float(x.mean()), "d_vs_" + base_cfg: float(d.mean()), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
    return o
for c in ("beam4", "beam8", "beam2"):
    flag = [loopy(nfc(H[c][k])) for k in range(len(rows))]
    hyb = [H["greedy"][k] if flag[k] else H[c][k] for k in range(len(rows))]
    res = boot_of(hyb)
    out["hybrid"][c + "+greedy_fallback"] = {"n_fallback": int(sum(flag)), "vs_greedy": summ(res, "greedy"), "vs_" + c: summ(res, c)}
json.dump(out, open(D + "/extra.json", "w"))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "vs_greedy"} for k, v in out["hybrid"].items()})[:1500])
